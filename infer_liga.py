#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter

import torch
from torchvision.transforms import functional as TF

from models_resnet_unet import ResNetUNet


# =========================
# lIGA 可視化用カラー（緑系ヒートマップ）
# =========================
CLASS_COLORS = {
    1: (120, 255, 120, 90),   # lIGA1: bright light green
    2: (60, 220, 60, 110),    # lIGA2: vivid green
    3: (0, 180, 0, 130),      # lIGA3: strong green
    4: (0, 140, 0, 150),      # lIGA4: deep vivid green
}

# 輪郭線の色（RGBA）
BOUNDARY_COLOR = (0, 80, 0, 255)  # 濃い緑（ほぼ黒）

SUPPORTED_EXTS = {
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp",
    ".JPG", ".JPEG", ".PNG"
}


# =========================
# 画像読み込み
# =========================
def load_image(img_path: Path, img_size: int):
    img = Image.open(img_path).convert("RGB")
    orig = img.copy()

    img_rs = img.resize((img_size, img_size), Image.BILINEAR)
    img_t = TF.to_tensor(img_rs)
    img_t = TF.normalize(
        img_t,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    return img_t.unsqueeze(0), orig


# =========================
# 輪郭抽出
# =========================
def extract_boundary(mask: Image.Image) -> Image.Image:
    """
    マスク画像（L）から1px相当の輪郭を抽出
    """
    dilated = mask.filter(ImageFilter.MaxFilter(3))
    eroded = mask.filter(ImageFilter.MinFilter(3))
    boundary = np.array(dilated, dtype=np.int16) - np.array(eroded, dtype=np.int16)
    boundary = np.clip(boundary, 0, 255).astype(np.uint8)
    return Image.fromarray(boundary, mode="L")


# =========================
# overlay 作成
# =========================
def apply_overlay(orig_img: Image.Image, pred_mask_np: np.ndarray):
    w, h = orig_img.size
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))

    # ---- クラスごとの塗り ----
    for cls, rgba in CLASS_COLORS.items():
        cls_mask = (pred_mask_np == cls).astype(np.uint8) * 255
        if cls_mask.max() == 0:
            continue
        cls_mask_img = Image.fromarray(cls_mask, mode="L")
        tint = Image.new("RGBA", (w, h), rgba)
        overlay = Image.composite(tint, overlay, cls_mask_img)

    # ---- 輪郭線 ----
    boundary_total = Image.new("L", (w, h), 0)
    for cls in CLASS_COLORS.keys():
        cls_mask = (pred_mask_np == cls).astype(np.uint8) * 255
        if cls_mask.max() == 0:
            continue
        b = extract_boundary(Image.fromarray(cls_mask, mode="L"))
        boundary_total = Image.composite(
            Image.new("L", (w, h), 255),
            boundary_total,
            b,
        )

    boundary_rgba = Image.new("RGBA", (w, h), BOUNDARY_COLOR)
    overlay = Image.composite(boundary_rgba, overlay, boundary_total)

    return Image.alpha_composite(orig_img.convert("RGBA"), overlay)


# =========================
# 入力列挙
# =========================
def iter_images(image_arg: str):
    p = Path(image_arg)
    if p.is_file():
        return [p]
    if p.is_dir():
        return [x for x in sorted(p.iterdir())
                if x.is_file() and x.suffix in SUPPORTED_EXTS]
    raise FileNotFoundError(f"--image のパスが見つかりません: {p}")


# =========================
# main
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True,
                        help="画像ファイル or 画像フォルダ")
    parser.add_argument("--model", type=str, required=True,
                        help="best_model_liga.pth など")
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out-dir", type=str, default="output")
    args = parser.parse_args()

    device = torch.device(args.device)

    checkpoint = torch.load(args.model, map_location=device)
    num_classes = int(checkpoint.get("num_classes", 5))

    model = ResNetUNet(n_classes=num_classes, pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = iter_images(args.image)
    print(f"[INFO] Targets = {len(targets)} images")

    for img_path in targets:
        img_t, orig = load_image(img_path, args.img_size)
        img_t = img_t.to(device)

        with torch.no_grad():
            logits = model(img_t)
            pred = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)

        pred_mask = Image.fromarray(pred, mode="L").resize(orig.size, Image.NEAREST)
        pred_np = np.array(pred_mask)

        overlay = apply_overlay(orig, pred_np)

        stem = img_path.stem
        overlay_path = out_dir / f"{stem}_overlay.png"
        mask_path = out_dir / f"{stem}_pred_mask.png"

        overlay.save(overlay_path)
        pred_mask.save(mask_path)

        print(f"[SAVED] {overlay_path.name}, {mask_path.name}")

    print("Done!")


if __name__ == "__main__":
    main()
