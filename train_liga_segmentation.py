#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_liga_segmentation.py

ResNet-UNet を使って、lIGA 用 5クラスマスク
  0: 背景(皮疹なし)
  1: lIGA1（わずか）
  2: lIGA2（軽度）
  3: lIGA3（中等度）
  4: lIGA4（重度）
を学習するスクリプト。

前提フォルダ構成：
  lesion_segmentation/
    data/
      train_images/
        0001_8.JPG
        0010_4.JPG
        ...
      train_masks/
        0001_8_mask.png  (0〜4 のラベルが入っている)
        0010_4_mask.png
        ...

使い方（例）:
  cd ~/Desktop/project_root/lesion_segmentation
  python train_liga_segmentation.py --device cpu
"""

import os
from pathlib import Path
import argparse
import random
from typing import List, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from torchvision import transforms

from models_resnet_unet import ResNetUNet


# ===================== Dataset =====================

class LesionSegmentationDataset(Dataset):
    """
    画像と lIGA 多クラスマスクのペアを読み込む Dataset
    - 画像: 正規化済み RGB tensor [3,H,W]
    - マスク: int64 の [H,W] （値は 0〜4）
    """
    def __init__(
        self,
        image_paths: List[Path],
        masks_dir: Path,
        image_size: int = 256,
        train: bool = False,
    ):
        self.image_paths = image_paths
        self.masks_dir = masks_dir
        self.image_size = image_size
        self.train = train

        # 色ゆらぎ用（明るさ・コントラスト・彩度を ±10%）
        self.color_jitter = transforms.ColorJitter(
            brightness=0.1,
            contrast=0.1,
            saturation=0.1,
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        stem = img_path.stem
        mask_name = f"{stem}_liga.png"
        mask_path = self.masks_dir / mask_name

        # --- 画像 & マスク読み込み ---
        img = Image.open(img_path).convert("RGB")
        # マスク: L モードだが中身は 0〜4 の整数
        mask = Image.open(mask_path).convert("L")

        # ======== Augmentation（train のときだけ）========
        if self.train:
            # 1) 水平反転
            if random.random() < 0.5:
                img = TF.hflip(img)
                mask = TF.hflip(mask)

            # 2) 軽い回転（-15〜+15 度）
            if random.random() < 0.5:
                angle = random.uniform(-15.0, 15.0)
                img = TF.rotate(img, angle, interpolation=Image.BILINEAR)
                mask = TF.rotate(mask, angle, interpolation=Image.NEAREST)

            # 3) 色ゆらぎ（画像のみ。マスクにはかけない）
            if random.random() < 0.8:
                img = self.color_jitter(img)

        # ======== サイズを 256x256 に揃える ========
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)

        # ======== Tensor 変換 ========
        img = TF.to_tensor(img)  # [3,H,W], float32, 0~1

        # マスクを numpy → int64 Tensor (0〜4)
        mask_np = np.array(mask, dtype=np.int64)
        mask = torch.from_numpy(mask_np)  # [H,W], int64

        # 画像の正規化（ImageNet 事前学習 ResNet に合わせる）
        img = TF.normalize(
            img,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        return img, mask


# ===================== メトリクス（マルチクラス版） =====================

def multiclass_dice(pred: torch.Tensor,
                    target: torch.Tensor,
                    num_classes: int,
                    ignore_index: int = 0,
                    eps: float = 1e-6) -> torch.Tensor:
    """
    マルチクラス Dice 係数（背景クラスを除外して平均）
    pred, target: [B,H,W] (クラスインデックス)
    """
    dice_list = []

    for cls in range(num_classes):
        if cls == ignore_index:
            continue
        pred_c = (pred == cls).float()
        target_c = (target == cls).float()

        intersection = (pred_c * target_c).sum(dim=(1, 2))
        union = pred_c.sum(dim=(1, 2)) + target_c.sum(dim=(1, 2))
        dice_c = (2 * intersection + eps) / (union + eps)  # [B]
        dice_list.append(dice_c)

    if not dice_list:
        return torch.tensor(1.0)  # すべて背景だけのとき

    dice_stack = torch.stack(dice_list, dim=0)  # [C-1,B]
    return dice_stack.mean()  # 全クラス・バッチ平均


def multiclass_iou(pred: torch.Tensor,
                   target: torch.Tensor,
                   num_classes: int,
                   ignore_index: int = 0,
                   eps: float = 1e-6) -> torch.Tensor:
    """
    マルチクラス IoU（Jaccard index）
    """
    iou_list = []

    for cls in range(num_classes):
        if cls == ignore_index:
            continue
        pred_c = (pred == cls).float()
        target_c = (target == cls).float()

        intersection = (pred_c * target_c).sum(dim=(1, 2))
        union = pred_c.sum(dim=(1, 2)) + target_c.sum(dim=(1, 2)) - intersection
        iou_c = (intersection + eps) / (union + eps)
        iou_list.append(iou_c)

    if not iou_list:
        return torch.tensor(1.0)

    iou_stack = torch.stack(iou_list, dim=0)  # [C-1,B]
    return iou_stack.mean()


# ===================== 学習・検証ループ =====================

def train_one_epoch(model, loader, optimizer, criterion, device, num_classes: int):
    model.train()
    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0
    n_batches = 0

    for imgs, masks in loader:
        imgs = imgs.to(device)              # [B,3,H,W]
        masks = masks.to(device)            # [B,H,W] (int64, 0〜4)

        optimizer.zero_grad()
        logits = model(imgs)                # [B,5,H,W]
        loss = criterion(logits, masks)

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)  # [B,H,W]
            dice = multiclass_dice(preds, masks, num_classes=num_classes)
            iou = multiclass_iou(preds, masks, num_classes=num_classes)

        running_loss += loss.item()
        running_dice += float(dice.item())
        running_iou += float(iou.item())
        n_batches += 1

    return (running_loss / n_batches,
            running_dice / n_batches,
            running_iou / n_batches)


def validate_one_epoch(model, loader, criterion, device, num_classes: int):
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0
    n_batches = 0

    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            logits = model(imgs)
            loss = criterion(logits, masks)

            preds = torch.argmax(logits, dim=1)
            dice = multiclass_dice(preds, masks, num_classes=num_classes)
            iou = multiclass_iou(preds, masks, num_classes=num_classes)

            running_loss += loss.item()
            running_dice += float(dice.item())
            running_iou += float(iou.item())
            n_batches += 1

    return (running_loss / n_batches,
            running_dice / n_batches,
            running_iou / n_batches)


# ===================== ヘルパー =====================

def list_image_files(images_dir: Path) -> List[Path]:
    exts = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".JPG", ".JPEG"]
    files = []
    for p in sorted(images_dir.iterdir()):
        if p.suffix.lower() in [e.lower() for e in exts]:
            files.append(p)
    return files


def split_train_val(all_paths: List[Path], val_ratio: float = 0.2, seed: int = 42
                    ) -> Tuple[List[Path], List[Path]]:
    rng = random.Random(seed)
    paths = list(all_paths)
    rng.shuffle(paths)
    n_total = len(paths)
    n_val = max(1, int(n_total * val_ratio))
    val_paths = paths[:n_val]
    train_paths = paths[n_val:]
    return train_paths, val_paths


# ===================== メイン =====================

def parse_args():
    parser = argparse.ArgumentParser(description="lIGA segmentation ResNetUNet training")
    parser.add_argument("--data-dir", type=str, default="data",
                        help="data ディレクトリへのパス (デフォルト: ./data)")
    parser.add_argument("--img-size", type=int, default=256,
                        help="学習時の画像サイズ (正方形)")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="バッチサイズ")
    parser.add_argument("--epochs", type=int, default=50,
                        help="エポック数")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="学習率")
    parser.add_argument("--device", type=str, default="cpu",
                        help="cpu / cuda / mps など")
    parser.add_argument("--val-ratio", type=float, default=0.2,
                        help="バリデーションの割合 (0~1)")
    parser.add_argument("--out-dir", type=str, default="checkpoints",
                        help="モデル保存先ディレクトリ")
    parser.add_argument("--num-classes", type=int, default=5,
                        help="クラス数（背景+4段階）")
    return parser.parse_args()


def main():
    args = parse_args()

    # device 判定
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA が利用できないため cpu を使用します")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"[INFO] device = {device}")

    num_classes = args.num_classes  # 5 のはず

    data_dir = Path(args.data_dir)
    images_dir = data_dir / "train_images"
    masks_dir = data_dir / "train_masks"

    assert images_dir.exists(), f"画像ディレクトリが存在しません: {images_dir}"
    assert masks_dir.exists(), f"マスクディレクトリが存在しません: {masks_dir}"

    # 画像一覧
    all_image_paths = list_image_files(images_dir)
    print(f"[INFO] 画像枚数 = {len(all_image_paths)}")

    # マスクがあるものだけに絞る
    filtered_image_paths = []
    for p in all_image_paths:
        stem = p.stem
        mask_path = masks_dir / f"{stem}_liga.png"
        if mask_path.exists():
            filtered_image_paths.append(p)
        else:
            print(f"[WARN] 対応するマスクが見つかりません: {p.name}")

    print(f"[INFO] マスク付き画像枚数 = {len(filtered_image_paths)}")

    if len(filtered_image_paths) < 2:
        raise RuntimeError("マスク付き画像が2枚未満です。学習用にはもう少し用意してください。")

    train_paths, val_paths = split_train_val(
        filtered_image_paths,
        val_ratio=args.val_ratio,
        seed=42,
    )
    print(f"[INFO] train = {len(train_paths)}, val = {len(val_paths)}")

    # Dataset & DataLoader
    train_ds = LesionSegmentationDataset(
        train_paths, masks_dir, image_size=args.img_size, train=True
    )
    val_ds = LesionSegmentationDataset(
        val_paths, masks_dir, image_size=args.img_size, train=False
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    # モデル・損失・最適化
    model = ResNetUNet(n_classes=num_classes, pretrained=True).to(device)

    # 多クラス用の CrossEntropyLoss
    criterion = nn.CrossEntropyLoss()  # logits [B,C,H,W], target [B,H,W]

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # モデル保存用ディレクトリ
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_dice = -1.0
    best_model_path = out_dir / "best_model_liga.pth"

    # 学習ループ
    for epoch in range(1, args.epochs + 1):
        print(f"\n===== Epoch {epoch}/{args.epochs} =====")
        train_loss, train_dice, train_iou = train_one_epoch(
            model, train_loader, optimizer, criterion, device, num_classes=num_classes
        )
        print(f"[Train] loss={train_loss:.4f}, dice={train_dice:.4f}, iou={train_iou:.4f}")

        val_loss, val_dice, val_iou = validate_one_epoch(
            model, val_loader, criterion, device, num_classes=num_classes
        )
        print(f"[Val]   loss={val_loss:.4f}, dice={val_dice:.4f}, iou={val_iou:.4f}")

        # ベストモデル更新
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_dice": val_dice,
                    "val_iou": val_iou,
                    "img_size": args.img_size,
                    "num_classes": num_classes,
                },
                best_model_path,
            )
            print(f"[SAVE] ベストモデル更新: dice={best_dice:.4f} -> {best_model_path}")

    print(f"\n[FINISHED] 学習完了。ベストDice={best_dice:.4f} のモデルを {best_model_path} に保存しました。")


if __name__ == "__main__":
    main()
