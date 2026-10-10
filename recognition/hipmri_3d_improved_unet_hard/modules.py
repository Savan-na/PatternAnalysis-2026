"""Standard 3D U-Net baseline for six-class HipMRI segmentation."""

from __future__ import annotations

import torch
from torch import nn


class _DoubleConv3D(nn.Sequential):
    """Two spatial convolutions with batch-size-independent normalisation."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
            nn.ReLU(inplace=True),
        )


class StandardUNet3D(nn.Module):
    """Four-level encoder-decoder U-Net returning unnormalised class logits."""

    def __init__(
        self, in_channels: int = 1, out_channels: int = 6, base_channels: int = 16
    ) -> None:
        super().__init__()
        if not isinstance(in_channels, int) or isinstance(in_channels, bool) or in_channels < 1:
            raise ValueError("in_channels must be a positive integer")
        if not isinstance(out_channels, int) or isinstance(out_channels, bool) or out_channels < 1:
            raise ValueError("out_channels must be a positive integer")
        if not isinstance(base_channels, int) or isinstance(base_channels, bool) or base_channels < 1:
            raise ValueError("base_channels must be a positive integer")

        widths = tuple(base_channels * 2**level for level in range(5))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.encoder = nn.ModuleList(
            _DoubleConv3D(in_channels if level == 0 else widths[level - 1], width)
            for level, width in enumerate(widths[:4])
        )
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.bottleneck = _DoubleConv3D(widths[3], widths[4])
        self.upconvs = nn.ModuleList(
            nn.ConvTranspose3d(widths[level + 1], widths[level], kernel_size=2, stride=2)
            for level in range(3, -1, -1)
        )
        self.decoder = nn.ModuleList(
            _DoubleConv3D(2 * widths[level], widths[level])
            for level in range(3, -1, -1)
        )
        self.head = nn.Conv3d(widths[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            raise TypeError("input must be a torch.Tensor")
        if x.ndim != 5:
            raise ValueError("input must have shape [B,C,D,H,W]")
        if x.shape[0] < 1:
            raise ValueError("batch dimension must be nonempty")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"expected {self.in_channels} input channels, got {x.shape[1]}")
        if x.dtype != torch.float32:
            raise TypeError("input must be torch.float32")
        if any(size < 16 or size % 16 for size in x.shape[2:]):
            raise ValueError("D, H and W must each be at least 16 and divisible by 16")

        skips = []
        for block in self.encoder:
            x = block(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for upconv, block, skip in zip(self.upconvs, self.decoder, reversed(skips)):
            x = upconv(x)
            x = block(torch.cat((skip, x), dim=1))
        return self.head(x)
