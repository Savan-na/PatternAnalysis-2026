"""Isensee-inspired improved 3D U-Net with explicit HipMRI adaptations."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .. import NUM_CLASSES


class _ContextResidual(nn.Module):
    """Pre-activation residual context module with dropout between convolutions."""

    def __init__(self, channels: int, dropout: float):
        super().__init__()
        self.norm1 = nn.InstanceNorm3d(channels, affine=True)
        self.norm2 = nn.InstanceNorm3d(channels, affine=True)
        self.conv1 = nn.Conv3d(channels, channels, 3, padding=1, bias=False)
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1, bias=False)
        self.dropout = nn.Dropout3d(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.conv1(F.leaky_relu(self.norm1(x), negative_slope=0.01))
        residual = self.dropout(residual)
        residual = self.conv2(F.leaky_relu(self.norm2(residual), negative_slope=0.01))
        return x + residual


class _Localization(nn.Module):
    """3x3 fusion followed by 1x1 channel reduction in the localization path."""

    def __init__(self, width: int):
        super().__init__()
        self.conv3 = nn.Conv3d(2 * width, 2 * width, 3, padding=1, bias=False)
        self.norm = nn.InstanceNorm3d(2 * width, affine=True)
        self.conv1 = nn.Conv3d(2 * width, width, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv1(F.leaky_relu(self.norm(self.conv3(x)), negative_slope=0.01))


class ImprovedUNet3D(nn.Module):
    """Residual 3D U-Net with additive deep supervision.

    Auxiliary segmentation logits are upsampled and summed. This is an
    independent HipMRI implementation, not a direct port of author code.
    """

    def __init__(self, base_channels: int = 8, dropout: float = 0.3):
        super().__init__()
        if base_channels < 2 or not 0 <= dropout < 1:
            raise ValueError("Invalid base_channels/dropout")
        widths = [base_channels * 2**i for i in range(5)]
        self.stem = nn.Conv3d(1, widths[0], 3, padding=1)
        self.context = nn.ModuleList([_ContextResidual(width, dropout) for width in widths])
        self.down = nn.ModuleList([nn.Conv3d(widths[i], widths[i + 1], 3, stride=2, padding=1)
                                   for i in range(4)])
        self.up_reduce = nn.ModuleList([nn.Conv3d(widths[i + 1], widths[i], 3, padding=1)
                                        for i in range(3, -1, -1)])
        self.localization = nn.ModuleList([_Localization(widths[i]) for i in range(3, -1, -1)])
        self.head = nn.Conv3d(widths[0], NUM_CLASSES, 1)
        self.aux_half = nn.Conv3d(widths[1], NUM_CLASSES, 1)
        self.aux_quarter = nn.Conv3d(widths[2], NUM_CLASSES, 1)

    def forward(self, image: torch.Tensor, *, return_components: bool = False):
        if image.ndim != 5 or image.shape[1] != 1:
            raise ValueError("Expected [B,1,D,H,W]")
        if any(size < 16 or size % 16 for size in image.shape[2:]):
            raise ValueError("Every spatial dimension must be >=16 and divisible by 16")
        x = self.stem(image)
        skips = []
        for i, block in enumerate(self.context):
            x = block(x)
            if i < 4:
                skips.append(x)
                x = self.down[i](x)
        decoder_features = {}
        for i, (reduce, localize, skip) in enumerate(zip(self.up_reduce, self.localization, reversed(skips))):
            x = F.interpolate(x, size=skip.shape[2:], mode="nearest")
            x = F.leaky_relu(reduce(x), negative_slope=0.01)
            x = localize(torch.cat((x, skip), dim=1))
            decoder_features[3 - i] = x
        main = self.head(decoder_features[0])
        half = F.interpolate(self.aux_half(decoder_features[1]), size=main.shape[2:], mode="trilinear", align_corners=False)
        quarter = F.interpolate(self.aux_quarter(decoder_features[2]), size=main.shape[2:], mode="trilinear", align_corners=False)
        combined = main + half + quarter
        return (combined, (main, half, quarter)) if return_components else combined
