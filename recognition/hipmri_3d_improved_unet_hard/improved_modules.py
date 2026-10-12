"""Residual 3D U-Net with optional deep supervision; baseline source is unchanged.

Default forward returns full-resolution raw logits. With deep_supervision=True,
return (main, (half_resolution, quarter_resolution)) raw logits. For 64-cubed
inputs their shapes are [B,6,64,64,64], [B,6,32,32,32], [B,6,16,16,16].
The proposed training objective averages full-grid losses with weights 1,.5,.25.
Auxiliary logits are trilinearly interpolated; integer targets are never resized.
This is an engineering design, not a verified reproduction of a specific paper.
"""

import torch
from torch import nn
from torch.nn import functional as F

from .modules import StandardUNet3D


class ResidualBlock3D(nn.Module):
    """Two padded convolutions plus identity or learned channel projection."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.branch = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels), nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels))
        self.shortcut = (nn.Identity() if in_channels == out_channels else
                         nn.Conv3d(in_channels, out_channels, 1, bias=False))
        self.activation = nn.ReLU(inplace=False)

    def forward(self, x):
        return self.activation(self.branch(x) + self.shortcut(x))


class ImprovedUNet3D(StandardUNet3D):
    """Same four-level topology as baseline, with residual blocks and two heads."""

    def __init__(self, in_channels=1, out_channels=6, base_channels=16):
        super().__init__(in_channels, out_channels, base_channels)
        widths = [base_channels * 2**level for level in range(5)]
        self.encoder = nn.ModuleList(
            ResidualBlock3D(in_channels if level == 0 else widths[level - 1], width)
            for level, width in enumerate(widths[:4]))
        self.bottleneck = ResidualBlock3D(widths[3], widths[4])
        self.decoder = nn.ModuleList(
            ResidualBlock3D(2 * widths[level], widths[level]) for level in range(3, -1, -1))
        self.auxiliary_heads = nn.ModuleList(
            nn.Conv3d(widths[level], out_channels, 1) for level in (1, 2))

    def forward(self, x, *, deep_supervision=False):
        if not isinstance(x, torch.Tensor):
            raise TypeError("input must be a torch.Tensor")
        if x.ndim != 5 or x.shape[0] < 1 or x.shape[1] != self.in_channels:
            raise ValueError(f"input must be nonempty [B,{self.in_channels},D,H,W]")
        if x.dtype != torch.float32:
            raise TypeError("input must be torch.float32")
        if any(size < 16 or size % 16 for size in x.shape[2:]):
            raise ValueError("D,H,W must be at least 16 and divisible by 16")
        skips = []
        for block in self.encoder:
            x = block(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        auxiliary = {}
        for index, (upconv, block, skip) in enumerate(zip(self.upconvs, self.decoder, reversed(skips))):
            x = block(torch.cat((skip, upconv(x)), dim=1))
            if deep_supervision and index in (1, 2):
                auxiliary[index] = self.auxiliary_heads[2 - index](x)
        main = self.head(x)
        return (main, (auxiliary[2], auxiliary[1])) if deep_supervision else main


def deep_supervision_loss(outputs, target, loss_fn):
    """Weighted mean of scalar losses: (L_main + .5 L_half + .25 L_quarter)/1.75.

Pass the approved CE/foreground-Dice scalar objective during future integration.
No loss mathematics, target labels or training policy are changed here.
"""
    main, auxiliary = outputs
    if len(auxiliary) != 2 or target.dtype != torch.long:
        raise ValueError("Expected two auxiliary heads and integer class-index targets")
    if main.ndim != 5 or target.shape != (main.shape[0], *main.shape[2:]):
        raise ValueError("Main logits and target spatial shapes must match")
    losses = []
    for logits, weight in zip((main, *auxiliary), (1., .5, .25)):
        if logits.ndim != 5 or logits.shape[:2] != main.shape[:2]:
            raise ValueError("Auxiliary batch/class dimensions must match main logits")
        if logits.shape[2:] != main.shape[2:]:
            logits = F.interpolate(logits, size=main.shape[2:], mode="trilinear", align_corners=False)
        loss = loss_fn(logits, target)
        if loss.ndim != 0 or not bool(torch.isfinite(loss)):
            raise ValueError("loss_fn must return a finite scalar tensor")
        losses.append(weight * loss)
    return sum(losses) / 1.75
