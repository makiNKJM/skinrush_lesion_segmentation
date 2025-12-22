#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
models_resnet_unet.py

ResNet34 を Encoder に使った U-Net 風セグメンテーションモデル。
入力: RGB画像
出力: 1チャンネルのマスク（ロジット）
"""

from typing import Tuple

import torch
import torch.nn as nn
from torchvision import models


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


class UpBlock(nn.Module):
    """アップサンプリング + skip connection + DoubleConv"""
    def __init__(self, in_ch, skip_ch, out_ch):
        """
        in_ch: up前の特徴マップのチャンネル数
        skip_ch: skip connection 側のチャンネル数
        out_ch: 出力チャンネル数
        """
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x, skip: torch.Tensor):
        x = self.up(x)

        # サイズがズレる場合はパディングで調整
        diffY = skip.size()[2] - x.size()[2]
        diffX = skip.size()[3] - x.size()[3]
        x = nn.functional.pad(
            x,
            [diffX // 2, diffX - diffX // 2,
             diffY // 2, diffY - diffY // 2]
        )

        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        return x


class ResNetUNet(nn.Module):
    """
    ResNet34 Encoder + U-Net Decoder
    n_classes: 出力マスクチャネル数（2値なら1）
    """
    def __init__(self, n_classes: int = 1, pretrained: bool = True):
        super().__init__()

        # ---- ResNet34 をバックボーンとしてロード ----
        # torchvision のバージョンで引数が違うので try/except にしておく
        try:
            from torchvision.models import resnet34, ResNet34_Weights
            backbone = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
        except Exception:
            backbone = models.resnet34(pretrained=pretrained)

        # ResNet34 の各ステージを取り出す
        self.input_block = nn.Sequential(
            backbone.conv1,   # 64ch, stride=2
            backbone.bn1,
            backbone.relu,
        )                    # 出力: 64ch, 1/2サイズ
        self.maxpool = backbone.maxpool  # 出力: 64ch, 1/4サイズ

        self.encoder1 = backbone.layer1  # 出力: 64ch,   1/4
        self.encoder2 = backbone.layer2  # 出力: 128ch,  1/8
        self.encoder3 = backbone.layer3  # 出力: 256ch,  1/16
        self.encoder4 = backbone.layer4  # 出力: 512ch,  1/32

        # ---- Decoder ----
        # center: bottleneck 部分を少し処理しても良いが、ここでは 512 -> 512 の DoubleConv
        self.center = DoubleConv(512, 512)

        # アップサンプリングブロック
        self.up4 = UpBlock(in_ch=512, skip_ch=256, out_ch=256)  # encoder3 と結合
        self.up3 = UpBlock(in_ch=256, skip_ch=128, out_ch=128)  # encoder2 と結合
        self.up2 = UpBlock(in_ch=128, skip_ch=64,  out_ch=64)   # encoder1 と結合
        self.up1 = UpBlock(in_ch=64,  skip_ch=64,  out_ch=64)   # input_block と結合

        # 出力層
        self.out_conv = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x0 = self.input_block(x)        # [B,64,H/2,W/2]
        x1 = self.maxpool(x0)          # [B,64,H/4,W/4]
        x1 = self.encoder1(x1)         # [B,64,H/4,W/4]
        x2 = self.encoder2(x1)         # [B,128,H/8,W/8]
        x3 = self.encoder3(x2)         # [B,256,H/16,W/16]
        x4 = self.encoder4(x3)         # [B,512,H/32,W/32]

        # Center
        center = self.center(x4)       # [B,512,H/32,W/32]

        # Decoder (U-Net style skip connection)
        d4 = self.up4(center, x3)      # [B,256,H/16,W/16]
        d3 = self.up3(d4, x2)          # [B,128,H/8,W/8]
        d2 = self.up2(d3, x1)          # [B,64,H/4,W/4]
        d1 = self.up1(d2, x0)          # [B,64,H/2,W/2]

        # 必要ならさらにアップサンプリングして元サイズに戻す
        out = self.out_conv(d1)        # [B,1,H/2,W/2] になっている可能性あり

        # 入力サイズに合わせて補間
        # (ResNet の構造次第で 1/2 になっているため、元サイズにリサイズする)
        if out.size(2) != x.size(2) or out.size(3) != x.size(3):
            out = nn.functional.interpolate(out, size=x.shape[2:], mode="bilinear", align_corners=False)

        return out
