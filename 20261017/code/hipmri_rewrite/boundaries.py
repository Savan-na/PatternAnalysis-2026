"""Geometry-aware prostate boundary diagnostics for full XYZ volumes."""

from __future__ import annotations

import numpy as np


def prostate_axial_regions(truth_xyz: np.ndarray, affine: np.ndarray,
                            end_fraction: float = 0.2) -> dict[str, tuple[int, ...]]:
    """Partition truth-positive axial slices by physical superior position.

    The superior end is called base and inferior end apex. Refuse an oblique
    grid whose in-plane axes materially change superior coordinate. This is a
    geometric proxy; selected cases still need anatomical image review.
    """
    if truth_xyz.ndim != 3 or affine.shape != (4, 4) or not 0 < end_fraction < 0.5:
        raise ValueError("Invalid volume, affine, or end fraction")
    if not np.isfinite(affine).all() or abs(affine[2, 2]) < 1e-8:
        raise ValueError("Invalid superior axis")
    if max(abs(affine[2, 0]), abs(affine[2, 1])) > 1e-3 * abs(affine[2, 2]):
        raise ValueError("Oblique grid requires anatomical reorientation review")
    indices = np.flatnonzero((truth_xyz == 5).any(axis=(0, 1)))
    if len(indices) < 5:
        raise ValueError("Too few prostate-positive slices to partition")
    superior_mm = affine[2, 2] * indices + affine[2, 3]
    order = indices[np.argsort(superior_mm)]
    end_count = max(1, int(np.ceil(len(order) * end_fraction)))
    if 2 * end_count >= len(order):
        raise ValueError("No central prostate slices remain")
    return {"apex": tuple(int(v) for v in order[:end_count]),
            "central": tuple(int(v) for v in order[end_count:-end_count]),
            "base": tuple(int(v) for v in order[-end_count:])}


def slice_dice(truth_xyz: np.ndarray, prediction_xyz: np.ndarray, label: int = 5) -> list[float | None]:
    """Per-array-slice overlap; absent/absent is null, not a perfect score."""
    if truth_xyz.shape != prediction_xyz.shape or truth_xyz.ndim != 3:
        raise ValueError("Expected matching XYZ volumes")
    values = []
    for z in range(truth_xyz.shape[2]):
        truth = truth_xyz[:, :, z] == label
        predicted = prediction_xyz[:, :, z] == label
        denominator = int(truth.sum() + predicted.sum())
        values.append(None if denominator == 0 else float(2 * (truth & predicted).sum() / denominator))
    return values


def symmetric_surface_distances(truth_xyz: np.ndarray, prediction_xyz: np.ndarray,
                                spacing_mm: tuple[float, float, float], label: int = 5) -> dict:
    """HD95 and mean symmetric surface distance in mm for a specified class.

    One-empty masks yield status `one_empty` and null distances; both-empty
    masks yield `both_empty`. They must be counted separately in reports.
    """
    from scipy.ndimage import binary_erosion, distance_transform_edt

    if truth_xyz.shape != prediction_xyz.shape or truth_xyz.ndim != 3:
        raise ValueError("Expected matching XYZ volumes")
    if len(spacing_mm) != 3 or any(not np.isfinite(s) or s <= 0 for s in spacing_mm):
        raise ValueError("Spacing must be three positive finite millimetre values")
    truth = truth_xyz == label
    prediction = prediction_xyz == label
    if not truth.any() or not prediction.any():
        return {"status": "both_empty" if not truth.any() and not prediction.any() else "one_empty",
                "hd95_mm": None, "assd_mm": None}
    truth_surface = truth & ~binary_erosion(truth)
    predicted_surface = prediction & ~binary_erosion(prediction)
    to_truth = distance_transform_edt(~truth_surface, sampling=spacing_mm)[predicted_surface]
    to_prediction = distance_transform_edt(~predicted_surface, sampling=spacing_mm)[truth_surface]
    distances = np.concatenate((to_truth, to_prediction))
    return {"status": "both_present", "hd95_mm": float(np.percentile(distances, 95)),
            "assd_mm": float(distances.mean())}


def prostate_organ_confusion(truth_xyz: np.ndarray, prediction_xyz: np.ndarray) -> dict[str, int]:
    """Count prostate predictions inside true bladder/rectum and missed prostate."""
    if truth_xyz.shape != prediction_xyz.shape or truth_xyz.ndim != 3:
        raise ValueError("Expected matching XYZ volumes")
    return {"prostate_pred_in_true_bladder": int(((prediction_xyz == 5) & (truth_xyz == 3)).sum()),
            "prostate_pred_in_true_rectum": int(((prediction_xyz == 5) & (truth_xyz == 4)).sum()),
            "true_prostate_pred_other": int(((truth_xyz == 5) & (prediction_xyz != 5)).sum())}
