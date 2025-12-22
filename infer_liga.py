#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import numpy as np
from PIL import Image

import torch
from torchvision.transforms import functional as TF

from models_resnet_unet import ResNetUNet

# === lIGA 色設定（背景+4段階） ===
CLASS_COLORS = {
    0: (0, 0, 0, 0),          # 透明
    1: (0, 255, 255, 120),    # cyan（わずか）
    2: (0, 255, 0, 120),      # green（軽度）
    3: (255, 255, 0, 120),    # yellow（中等度）
    4: (255, 0, 0, 120),      # red（重度）
}

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".JPG", ".JPEG", ".PNG"}


def load_image(img_path: Path, img_size: int):
    """画像を読み込み、推論用 Tensor に変換。orig は元解像度の PIL で返す"""
    img = Image.open(img_path).convert("RGB")
    orig = img.copy()
    img_rs = img.resize((img_size, img_size), Image.BILINEAR)

    img_t = TF.to_tensor(img_rs)
    img_t = TF.normalize(
        img_t,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    return img_t.unsqueeze(0), orig


def apply_overlay(orig_img: Image.Image, pred_mask_np: np.ndarray, alpha_colors: dict):
    """元画像に多クラスマスクのオーバーレイを重ねる"""
    w, h = orig_img.size
    overlay_img = Image.new("RGBA", (w, h), (0, 0, 0, 0))

    for cls, rgba in alpha_colors.items():
        if cls == 0:
            continue
        cls_mask = (pred_mask_np == cls).astype(np.uint8) * 255
        if cls_mask.max() == 0:
            continue
        cls_mask_img = Image.fromarray(cls_mask, mode="L")
        tint = Image.new("RGBA", (w, h), rgba)
        overlay_img = Image.composite(tint, overlay_img, cls_mask_img)

    return Image.alpha_composite(orig_img.convert("RGBA"), overlay_img)


def iter_images(image_arg: str):
    """--image に file/dir が来てもOKにして、処理対象の Path を列挙する"""
    p = Path(image_arg)
    if p.is_file():
        return [p]
    if p.is_dir():
        files = []
        for x in sorted(p.iterdir()):
            if x.is_file() and x.suffix in SUPPORTED_EXTS:
                files.append(x)
        return files
    raise FileNotFoundError(f"--image のパスが見つかりません: {p}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True,
                        help="画像ファイル or 画像が入ったフォルダ")
    parser.add_argument("--model", type=str, required=True,
                        help="学習済み pth（best_model_liga.pth など）")
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out-dir", type=str, default="output",
                        help="保存先フォルダ（デフォルト: output）")
    args = parser.parse_args()

    device = torch.device(args.device)

    # === モデル読み込み ===
    checkpoint = torch.load(args.model, map_location=device)
    num_classes = int(checkpoint.get("num_classes", 5))

    model = ResNetUNet(n_classes=num_classes, pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    targets = iter_images(args.image)
    if not targets:
        print("[WARN] 対象画像がありません（拡張子を確認してね）")
        return

    print(f"[INFO] Targets = {len(targets)} images")
    print(f"[INFO] out_dir = {out_dir}")

    for img_path in targets:
        # === 画像読み込み ===
        img_t, orig = load_image(img_path, args.img_size)
        img_t = img_t.to(device)

        # === 推論 ===
        with torch.no_grad():
            logits = model(img_t)                 # [1,C,H,W]
            pred = torch.argmax(logits, dim=1)[0] # [H,W]
            pred = pred.cpu().numpy().astype(np.uint8)

        # === 元解像度へ拡大（NEARESTでラベル保持） ===
        pred_mask = Image.fromarray(pred, mode="L")
        pred_mask = pred_mask.resize(orig.size, Image.NEAREST)
        pred_np = np.array(pred_mask, dtype=np.uint8)

        # === overlay 合成 ===
        overlay = apply_overlay(orig, pred_np, CLASS_COLORS)

        stem = img_path.stem
        overlay_path = out_dir / f"{stem}_overlay.png"
        mask_path = out_dir / f"{stem}_pred_mask.png"

        overlay.save(overlay_path)
        pred_mask.save(mask_path)

        print(f"[SAVED] {overlay_path.name} / {mask_path.name}")

    print("Done!")


if __name__ == "__main__":
    main()
