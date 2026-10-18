"""Predeclared patient-balanced Validation checkpoint score for G5."""

from __future__ import annotations

import numpy as np

from .metrics import overlap


SMALL_ORGAN_IDS = (3, 4, 5)


def patient_small_organ_score(matrices: dict[str, np.ndarray]) -> tuple[float, dict]:
    """Mean available Bladder/Rectum/Prostate Dice over patients equally.

    Each input matrix pools one patient's visits before Dice is calculated.
    Both-absent organ/patient pairs are excluded; missed or extra organs score
    zero through the standard confusion-matrix Dice rule.
    """
    if not matrices:
        raise ValueError("No Validation patient confusion matrices")
    details = {}
    patient_scores = []
    for patient, matrix in sorted(matrices.items()):
        if matrix.shape != (6, 6):
            raise ValueError(f"{patient}: expected six-class confusion matrix")
        dice = overlap(matrix)["dice"]
        small = {name: dice[index] for name, index in
                 (("Bladder", 3), ("Rectum", 4), ("Prostate", 5))}
        details[patient] = small
        available = [value for value in small.values() if value is not None]
        if not available:
            raise ValueError(f"{patient}: no small-organ Validation Dice available")
        patient_scores.append(float(np.mean(available)))
    return float(np.mean(patient_scores)), details
