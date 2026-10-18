"""Convolutional blocks shared by the two standard U-Net baselines."""

from __future__ import annotations

import torch
from torch import nn

from .. import NUM_CLASSES


class _DoubleConv(nn.Sequential):
    def __init__(self, dimensions: int, input_channels: int, output_channels: int):
        convolution = nn.Conv2d if dimensions == 2 else nn.Conv3d
        super().__init__(
            convolution(input_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(1, output_channels), nn.ReLU(inplace=True),
            convolution(output_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(1, output_channels), nn.ReLU(inplace=True),
        )


class _StandardUNet(nn.Module):
    """Four downsamples, paired convolutions, skip concatenation and 1x1 head."""

    def __init__(self, dimensions: int, base_channels: int):
        super().__init__()
        if dimensions not in (2, 3) or base_channels < 2:
            raise ValueError("dimensions must be 2/3 and base_channels >= 2")
        convolution = nn.Conv2d if dimensions == 2 else nn.Conv3d
        upconv = nn.ConvTranspose2d if dimensions == 2 else nn.ConvTranspose3d
        pool = nn.MaxPool2d if dimensions == 2 else nn.MaxPool3d
        widths = [base_channels * 2**level for level in range(5)]
        self.dimensions = dimensions
        self.encoder = nn.ModuleList(
            [_DoubleConv(dimensions, 1 if i == 0 else widths[i - 1], widths[i]) for i in range(4)]
        )
        self.pool = pool(2)
        self.bridge = _DoubleConv(dimensions, widths[3], widths[4])
        self.up = nn.ModuleList([upconv(widths[i + 1], widths[i], 2, stride=2) for i in range(3, -1, -1)])
        self.decoder = nn.ModuleList([_DoubleConv(dimensions, widths[i] * 2, widths[i]) for i in range(3, -1, -1)])
        self.head = convolution(widths[0], NUM_CLASSES, 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != self.dimensions + 2 or image.shape[1] != 1:
            raise ValueError(f"Expected [B,1,...] with {self.dimensions} spatial axes")
        if any(size < 16 or size % 16 for size in image.shape[2:]):
            raise ValueError("Every spatial dimension must be >=16 and divisible by 16")
        skips = []
        x = image
        for block in self.encoder:
            x = block(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bridge(x)
        for up, block, skip in zip(self.up, self.decoder, reversed(skips)):
            x = block(torch.cat((up(x), skip), dim=1))
        return self.head(x)
