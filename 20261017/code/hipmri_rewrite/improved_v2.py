"""Independent supervision experiment; inherited v1 parameters and geometry."""
from __future__ import annotations

import torch

from .models.improved_3d import ImprovedUNet3D
from .losses import ce_dice_loss


class ImprovedUNet3DV2(ImprovedUNet3D):
    """Same parameter initialization as v1; main head is the inference output."""

    def forward(self, image, *, return_components=False):
        _, components = super().forward(image, return_components=True)
        return (components[0], components) if return_components else components[0]


def supervised_loss(components, target, *, class_weights):
    if len(components) != 3:
        raise ValueError("Expected main, upsampled half and quarter logits")
    terms = [ce_dice_loss(z, target, ce_weight=0.5, class_weights=class_weights)
             for z in components]
    total = (terms[0] + 0.5 * terms[1] + 0.25 * terms[2]) / 1.75
    return total, {name: float(term.detach()) for name, term in
                   zip(("main", "half", "quarter"), terms)}


def state_digest(model):
    import hashlib
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
