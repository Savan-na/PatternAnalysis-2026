"""Integer-count overlap metrics, reusable for patient and pooled aggregation."""

from __future__ import annotations

import numpy as np

from . import NUM_CLASSES


def confusion(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    if target.shape != prediction.shape or target.ndim != 3:
        raise ValueError("Expected matching 3D target and prediction")
    if target.dtype.kind not in "iu" or prediction.dtype.kind not in "iu":
        raise ValueError("Expected integer labels")
    if np.any((target < 0) | (target >= NUM_CLASSES) | (prediction < 0) | (prediction >= NUM_CLASSES)):
        raise ValueError("Labels must be 0..5")
    encoded = NUM_CLASSES * target.astype(np.int64).ravel() + prediction.astype(np.int64).ravel()
    return np.bincount(encoded, minlength=NUM_CLASSES**2).reshape(NUM_CLASSES, NUM_CLASSES)


def overlap(matrix: np.ndarray) -> dict[str, list[float | None]]:
    """Rows are truth; columns are prediction; absent/absent metrics are null."""
    if matrix.shape != (NUM_CLASSES, NUM_CLASSES) or np.any(matrix < 0):
        raise ValueError("Expected nonnegative 6x6 confusion counts")
    truth = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    tp = np.diag(matrix)
    result: dict[str, list[float | None]] = {"dice": [], "iou": [], "precision": [], "recall": []}
    for c in range(NUM_CLASSES):
        def ratio(numerator, denominator):
            return None if denominator == 0 else float(numerator / denominator)
        result["dice"].append(ratio(2 * tp[c], truth[c] + predicted[c]))
        result["iou"].append(ratio(tp[c], truth[c] + predicted[c] - tp[c]))
        result["precision"].append(ratio(tp[c], predicted[c]))
        result["recall"].append(ratio(tp[c], truth[c]))
    return result
