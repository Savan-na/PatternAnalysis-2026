"""v1 residual/additive-supervision network with only IN -> GN(1,C)."""
from __future__ import annotations

import hashlib
import torch
from torch import nn
from .models.improved_3d import ImprovedUNet3D


class ImprovedUNet3DV3(ImprovedUNet3D):
    def __init__(self, base_channels=8, dropout=0.3):
        super().__init__(base_channels=base_channels, dropout=dropout)
        count = 0
        # Preserve the post-v1-initialization RNG state for subsequent dropout.
        with torch.random.fork_rng(devices=[]):
            for parent in list(self.modules()):
                for name, layer in list(parent.named_children()):
                    if isinstance(layer, nn.InstanceNorm3d):
                        new = nn.GroupNorm(1, layer.num_features, eps=layer.eps,
                                           affine=layer.affine)
                        new.load_state_dict(layer.state_dict(), strict=True)
                        setattr(parent, name, new)
                        count += 1
        if count != 14:
            raise ValueError(f'Expected 14 normalization replacements, found {count}')


def state_digest(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
