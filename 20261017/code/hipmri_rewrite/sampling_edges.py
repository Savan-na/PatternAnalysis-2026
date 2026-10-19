"""Train-only candidate volume-corner crops; frozen G5 sampling stays intact."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt

from .data import Volume
from .sampling import _rng, crop_origin


def boundary_hits(origin: tuple[int, int, int], shape: tuple[int, int, int],
                  patch: tuple[int, int, int], width: int) -> tuple[bool, bool]:
    """Return any-face and any-corner intersection with outer voxel slabs."""
    sides = [start < width or start + size > n - width
             for start, n, size in zip(origin, shape, patch)]
    return any(sides), all(sides)


@dataclass(frozen=True)
class EdgeSelection:
    origin: tuple[int, int, int]
    accepted: bool
    attempts: int
    corner: str | None


class EdgePatchSelector:
    """Search deterministic corner crops containing textured distant negatives.

    A hard-negative voxel is GT Background, nonzero in the model MRI input,
    >= distance_mm from GT Body, and in an outer-width volume slab. The
    thresholds define a candidate sampling policy, not a quality guarantee.
    """

    def __init__(self, volume: Volume, patch: tuple[int, int, int], *, width: int,
                 distance_mm: float, min_nonzero_fraction: float,
                 min_hard_negative_voxels: int, max_attempts: int):
        if volume.case.split != "train":
            raise ValueError("Edge selection accepts Train cases only")
        if (volume.image.ndim != 3 or volume.image.shape != volume.mask.shape
                or volume.mask.dtype.kind not in "iu"
                or not np.isfinite(volume.image).all()
                or np.any((volume.mask < 0) | (volume.mask > 5))):
            raise ValueError("Expected paired finite XYZ MRI and integer labels 0..5")
        if (len(patch) != 3 or any(type(v) is not int or v < 1 for v in patch)
                or any(size > n for size, n in zip(patch, volume.mask.shape))
                or type(width) is not int or width < 1
                or not np.isfinite(distance_mm) or distance_mm < 0
                or not 0 < min_nonzero_fraction <= 1
                or type(min_hard_negative_voxels) is not int
                or not 1 <= min_hard_negative_voxels <= np.prod(patch)
                or type(max_attempts) is not int or max_attempts < 8
                or len(volume.spacing_mm) != 3
                or any(not np.isfinite(s) or s <= 0 for s in volume.spacing_mm)):
            raise ValueError("Invalid edge-sampling geometry or thresholds")
        if not np.any(volume.mask == 1):
            raise ValueError("GT Body is required for physical-distance negatives")
        self.volume = volume
        self.patch = patch
        self.width = width
        self.min_nonzero_fraction = min_nonzero_fraction
        self.min_hard_negative_voxels = min_hard_negative_voxels
        self.max_attempts = max_attempts
        self.nonzero = volume.image != 0
        shell = np.zeros(volume.mask.shape, dtype=bool)
        for axis in range(3):
            low = [slice(None)] * 3
            high = [slice(None)] * 3
            low[axis] = slice(0, width)
            high[axis] = slice(-width, None)
            shell[tuple(low)] = True
            shell[tuple(high)] = True
        distance = distance_transform_edt(volume.mask != 1, sampling=volume.spacing_mm)
        self.hard_negative = ((volume.mask == 0) & self.nonzero & shell
                              & (distance >= distance_mm))

    def measurements(self, origin: tuple[int, int, int]) -> tuple[float, int]:
        if (len(origin) != 3 or any(start < 0 or start + size > n
                                   for start, size, n in
                                   zip(origin, self.patch, self.volume.mask.shape))):
            raise ValueError("Patch origin outside volume")
        region = tuple(slice(start, start + size)
                       for start, size in zip(origin, self.patch))
        return (float(self.nonzero[region].mean()),
                int(self.hard_negative[region].sum()))

    def select(self, *, seed: int, epoch: int, draw: int) -> EdgeSelection:
        if epoch < 0 or draw < 0:
            raise ValueError("Epoch and draw must be nonnegative")
        case_id = self.volume.case.identifier
        rng = _rng(seed, epoch, case_id, draw, "edge_candidate_v1")
        corners = rng.permutation(8)
        for attempt in range(self.max_attempts):
            corner = int(corners[attempt % 8])
            origin = []
            for axis, (n, size) in enumerate(zip(self.volume.mask.shape, self.patch)):
                gap = n - size
                offset = int(rng.integers(min(self.width - 1, gap) + 1))
                origin.append(gap - offset if corner & (1 << axis) else offset)
            origin = tuple(origin)
            nonzero, negative = self.measurements(origin)
            if (nonzero >= self.min_nonzero_fraction
                    and negative >= self.min_hard_negative_voxels):
                return EdgeSelection(origin, True, attempt + 1, f"{corner:03b}")
        fallback = crop_origin(self.volume.mask, self.patch, "uniform", seed=seed,
                               epoch=epoch, case_id=case_id, draw=draw)
        return EdgeSelection(fallback, False, self.max_attempts, None)
