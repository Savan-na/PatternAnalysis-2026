"""Explicit six-class objectives; objective choice must be fixed per run."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from . import NUM_CLASSES


def _check(logits: torch.Tensor, target: torch.Tensor) -> None:
    if logits.ndim not in (4, 5) or logits.shape[1] != NUM_CLASSES:
        raise ValueError("Expected 2D/3D six-class logits")
    if target.shape != (logits.shape[0], *logits.shape[2:]) or target.dtype != torch.long:
        raise ValueError("Expected matching integer class-index target")
    if not torch.isfinite(logits).all() or ((target < 0) | (target >= NUM_CLASSES)).any():
        raise ValueError("Non-finite logits or labels outside 0..5")


def foreground_soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-5) -> torch.Tensor:
    _check(logits, target)
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    probability = logits.softmax(dim=1)
    onehot = F.one_hot(target, NUM_CLASSES).movedim(-1, 1).to(probability.dtype)
    axes = (0, *range(2, logits.ndim))
    intersection = (probability * onehot).sum(dim=axes)
    denominator = probability.sum(dim=axes) + onehot.sum(dim=axes)
    dice = (2 * intersection + epsilon) / (denominator + epsilon)
    return 1 - dice[1:].mean()


def ce_dice_loss(logits: torch.Tensor, target: torch.Tensor, *, ce_weight: float = 0.5,
                 class_weights: torch.Tensor | None = None) -> torch.Tensor:
    """Shared architecture-control objective; paper-loss studies stay separate."""
    _check(logits, target)
    if not 0 <= ce_weight <= 1:
        raise ValueError("ce_weight must be in [0,1]")
    ce = F.cross_entropy(logits, target, weight=class_weights)
    dice = foreground_soft_dice_loss(logits, target)
    return ce_weight * ce + (1 - ce_weight) * dice
