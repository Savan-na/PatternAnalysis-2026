"""Slice-wise 2D U-Net baseline for the spatial-context comparison."""

from .common import _StandardUNet


class UNet2D(_StandardUNet):
    """Operational slice-wise baseline. Output is raw six-class logits."""

    def __init__(self, base_channels: int = 16):
        super().__init__(2, base_channels)
