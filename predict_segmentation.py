#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
predict_segmentation.py

train_segmentation.py で学習した U-Net モデル (checkpoints/best_model.pth) を使って、
画像から皮疹マスクを予測して保存するスクリプト。

- 2値マスク: *_pred_mask.png
- オーバーレイ: *_overlay.png
"""

import argparse
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from models_resnet_unet import ResNetUNet



# ============ U-Net (train_segmentation.py と同じ定義) ============

class DoubleConv(nn.Module):
    """(Conv => BN => ReLU) x 2"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    """Down-scaling with maxpool then double conv"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch),
        )

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    """Up-scaling then double conv"""
    def __init__(self, in_ch, out_ch, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # サイズ調整（偶奇ズレ対策）
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(
            x1,
            [diffX // 2, diffX - diffX // 2,
             diffY // 2, diffY - diffY // 2]
        )
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, bilinear=True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x,  x3)
        x = self.up3(x,  x2)
        x = self.up4(x,  x1)
        logits = self.outc(x)
        return logits


# ============ 画像リスト取得 ============

def list_image_files_from_dir(images_dir: Path) -> List[Path]:
    exts = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".JPG", ".JPEG"]
    files = []
    for p in sorted(images_dir.iterdir()):
        if p.suffix.lower() in [e.lower() for e in exts]:
            files.append(p)
    return files


# ============ 予測処理 ============
#簡易版
#def load_model(checkpoint_path: Path, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)

    img_size = ckpt.get("img_size", 256)  # train_segmentation.py で保存したやつ
    model = UNet(n_channels=3, n_classes=1, bilinear=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"[INFO] checkpoint loaded: {checkpoint_path}")
    print(f"[INFO] img_size = {img_size}")
    return model, img_size

#ResNetUNet版
def load_model(checkpoint_path: Path, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)

    img_size = ckpt.get("img_size", 256)
    model = ResNetUNet(n_classes=1, pretrained=False)  # 推論なので pretrained=False でもOK（重みはckptから読む）
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"[INFO] checkpoint loaded: {checkpoint_path}")
    print(f"[INFO] img_size = {img_size}")
    return model, img_size



def predict_one_image(model: UNet, img_path: Path, img_size: int, device: torch.device,
                      out_dir: Path, threshold: float = 0.5):
    print(f"[PREDICT] {img_path}")
    # ---- 画像読み込み & リサイズ ----
    orig_img = Image.open(img_path).convert("RGB")
    w0, h0 = orig_img.size

    img_resized = orig_img.resize((img_size, img_size), Image.BILINEAR)

    img_tensor = TF.to_tensor(img_resized)
    img_tensor = TF.normalize(
        img_tensor,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    img_tensor = img_tensor.unsqueeze(0).to(device)


    with torch.no_grad():
        logits = model(img_tensor)           # [1,1,H,W]
        probs = torch.sigmoid(logits)[0, 0]  # [H,W]
        pred_mask = (probs > threshold).float().cpu().numpy()  # 0/1

    # マスクを元のサイズに戻す
    mask_img_small = Image.fromarray((pred_mask * 255).astype(np.uint8))  # 0/255
    mask_img = mask_img_small.resize((w0, h0), Image.NEAREST)

    # ===== 保存先パス =====
    stem = img_path.stem
    pred_mask_path = out_dir / f"{stem}_pred_mask.png"
    overlay_path   = out_dir / f"{stem}_overlay.png"

    # 2値マスク保存
    mask_img.save(pred_mask_path)
    print(f"[SAVE] pred mask: {pred_mask_path}")

    # ===== オーバーレイ画像作成 =====
    # 元画像: RGB, マスク: L (0/255) → 色付きの半透明レイヤとして重ねる
    base = orig_img.convert("RGBA")
    mask_rgba = Image.new("RGBA", base.size, (0, 0, 0, 0))

    # 赤色で塗る（例: (255,0,0,120)）
    overlay_color = (0, 255, 0, 120)
    mask_array = np.array(mask_img)  # 0 or 255

    # マスクが255のところだけ overlay_color を置く
    alpha_layer = np.zeros((h0, w0, 4), dtype=np.uint8)
    alpha_layer[mask_array > 127] = overlay_color
    mask_rgba = Image.fromarray(alpha_layer, mode="RGBA")

    overlay = Image.alpha_composite(base, mask_rgba)
    overlay.save(overlay_path)
    print(f"[SAVE] overlay:   {overlay_path}")


# ============ CLI & main ============

def parse_args():
    parser = argparse.ArgumentParser(description="Lesion segmentation prediction using trained U-Net")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth",
                        help="学習済みモデル (.pth) のパス")
    parser.add_argument("--image", type=str, default=None,
                        help="単一画像のパス")
    parser.add_argument("--image-dir", type=str, default=None,
                        help="フォルダ内の全画像に対して予測を行う")
    parser.add_argument("--device", type=str, default="cpu",
                        help="cpu / cuda / mps など")
    parser.add_argument("--out-dir", type=str, default="outputs",
                        help="予測マスクとオーバーレイ画像の保存先フォルダ")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="2値化の閾値 (0~1)")
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device != "cuda" else "cpu")
    print(f"[INFO] device = {device}")

    ckpt_path = Path(args.checkpoint)
    assert ckpt_path.exists(), f"checkpoint が見つかりません: {ckpt_path}"

    model, img_size = load_model(ckpt_path, device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 予測対象の画像リストを作る
    image_paths = []

    if args.image:
        p = Path(args.image)
        assert p.exists(), f"画像ファイルが見つかりません: {p}"
        image_paths.append(p)

    if args.image_dir:
        img_dir = Path(args.image_dir)
        assert img_dir.exists(), f"画像フォルダが見つかりません: {img_dir}"
        image_paths.extend(list_image_files_from_dir(img_dir))

    # 重複を消す
    image_paths = sorted(set(image_paths))

    if not image_paths:
        raise RuntimeError("予測対象の画像が指定されていません。--image または --image-dir を指定してください。")

    print(f"[INFO] 予測対象画像枚数 = {len(image_paths)}")

    for p in image_paths:
        predict_one_image(model, p, img_size, device, out_dir, threshold=args.threshold)

    print("\n[FINISHED] 全画像の予測と保存が完了しました。")


if __name__ == "__main__":
    main()
