"""Train-only patient-balanced patch selection and measured class exposure.

Coordinates remain XYZ until ``crop_for_3d_model`` converts them to DHW.
No source MRI or mask is modified. Call the exposure recorder on the exact
mask returned with each model input, after any later spatial augmentation.
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np

from . import NUM_CLASSES
from .data import Volume
from .splits import Case


CLASS_NAMES = ("Background", "Body", "Bones", "Bladder", "Rectum", "Prostate")
TARGET_IDS = {"Bladder": 3, "Rectum": 4, "Prostate": 5}
PATCH_XYZ = (64, 64, 32)
UNIFORM_POLICY = ("uniform",) * 10
ORGAN_AWARE_POLICY = ("uniform",) * 4 + ("Bladder",) * 2 + ("Rectum",) * 2 + ("Prostate",) * 2


def _rng(*parts: object) -> np.random.Generator:
    key = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(key[:8], "little"))


def epoch_cases(cases: tuple[Case, ...], epoch: int, seed: int) -> tuple[Case, ...]:
    """Choose one Train visit per patient; cycle through shuffled visits."""
    if epoch < 0:
        raise ValueError("epoch must be nonnegative")
    grouped: dict[str, list[Case]] = defaultdict(list)
    for case in cases:
        if case.split != "train":
            raise ValueError("Sampler accepts Train cases only")
        grouped[case.patient].append(case)
    if not grouped:
        raise ValueError("No Train cases")
    selected = []
    for patient, visits in sorted(grouped.items()):
        visits.sort(key=lambda case: case.week)
        cycle, offset = divmod(epoch, len(visits))
        order = _rng(seed, patient, cycle, "visits").permutation(len(visits))
        selected.append(visits[int(order[offset])])
    order = _rng(seed, epoch, "patients").permutation(len(selected))
    return tuple(selected[int(i)] for i in order)


def crop_origin(mask: np.ndarray, patch_xyz: tuple[int, int, int],
                target: str, *, seed: int, epoch: int, case_id: str,
                draw: int) -> tuple[int, int, int]:
    """Return a valid origin; target policy guarantees one target voxel."""
    if mask.ndim != 3 or mask.dtype.kind not in "iu":
        raise ValueError("Expected integer XYZ mask")
    if len(patch_xyz) != 3 or any(size < 1 or size > n
                                  for size, n in zip(patch_xyz, mask.shape)):
        raise ValueError("Patch shape does not fit mask")
    rng = _rng(seed, epoch, case_id, draw, target)
    if target == "uniform":
        return tuple(int(rng.integers(n - size + 1))
                     for n, size in zip(mask.shape, patch_xyz))
    if target not in TARGET_IDS:
        raise ValueError(f"Unknown target: {target}")
    locations = np.argwhere(mask == TARGET_IDS[target])
    if len(locations) == 0:
        raise ValueError(f"{case_id}: target {target} absent after preprocessing")
    voxel = locations[int(rng.integers(len(locations)))]
    starts = []
    for n, size, pos in zip(mask.shape, patch_xyz, voxel):
        low = max(0, int(pos) - size + 1)
        high = min(int(pos), n - size)
        starts.append(int(rng.integers(low, high + 1)))
    return tuple(starts)


def crop_for_3d_model(volume: Volume, origin: tuple[int, int, int],
                      patch_xyz: tuple[int, int, int] = PATCH_XYZ) -> tuple[np.ndarray, np.ndarray]:
    """Crop normalized/downsampled paired arrays into C,D,H,W and D,H,W."""
    if volume.image.shape != volume.mask.shape or len(origin) != 3:
        raise ValueError("MRI/mask grid or origin mismatch")
    if any(start < 0 or size < 1 or start + size > n
           for start, size, n in zip(origin, patch_xyz, volume.image.shape)):
        raise ValueError("Crop falls outside the shared grid")
    slices = tuple(slice(start, start + size) for start, size in zip(origin, patch_xyz))
    image = np.transpose(volume.image[slices], (2, 1, 0))[None].astype(np.float32, copy=False)
    mask = np.transpose(volume.mask[slices], (2, 1, 0)).astype(np.int64, copy=False)
    if not np.isfinite(image).all() or np.any((mask < 0) | (mask >= NUM_CLASSES)):
        raise ValueError("Nonfinite MRI or illegal label in sampled patch")
    return image, mask


@dataclass
class ExposureRecorder:
    """Accumulate exact post-transform patch labels supplied to a model."""

    patches: int = 0

    def __post_init__(self) -> None:
        self.hits: Counter = Counter()
        self.voxels: Counter = Counter()
        self.by_patient: dict[str, Counter] = defaultdict(Counter)

    def add(self, mask_dhw: np.ndarray, patient: str) -> dict[str, int]:
        if mask_dhw.ndim != 3 or mask_dhw.dtype.kind not in "iu":
            raise ValueError("Expected integer DHW mask after all transforms")
        if np.any((mask_dhw < 0) | (mask_dhw >= NUM_CLASSES)):
            raise ValueError("Illegal class in actual patch")
        counts = np.bincount(mask_dhw.ravel(), minlength=NUM_CLASSES)
        self.patches += 1
        self.by_patient[patient]["patches"] += 1
        for index, name in enumerate(CLASS_NAMES):
            amount = int(counts[index])
            self.voxels[name] += amount
            self.hits[name] += int(amount > 0)
            self.by_patient[patient][f"{name}_hits"] += int(amount > 0)
        return {name: int(counts[index]) for index, name in enumerate(CLASS_NAMES)}

    def summary(self) -> dict:
        total = sum(self.voxels.values())
        return {
            "patches": self.patches,
            "per_class": {name: {"patch_hits": self.hits[name],
                                 "patch_hit_rate": self.hits[name] / self.patches if self.patches else None,
                                 "patch_voxels": self.voxels[name],
                                 "voxel_share": self.voxels[name] / total if total else None}
                          for name in CLASS_NAMES},
            "per_patient": {patient: dict(counts) for patient, counts in sorted(self.by_patient.items())},
        }
