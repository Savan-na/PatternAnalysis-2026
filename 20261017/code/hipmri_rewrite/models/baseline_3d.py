"""Standard volumetric U-Net baseline for the spatial-context comparison."""

from .common import _StandardUNet


class UNet3D(_StandardUNet):
    """Çiçek-style volumetric baseline with documented HipMRI adaptations."""

    def __init__(self, base_channels: int = 8):
        super().__init__(3, base_channels)
