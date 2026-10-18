"""Deterministic Train-only axial crops for the optional 2D comparison."""

from __future__ import annotations

import hashlib

import numpy as np

from . import NUM_CLASSES
from .data import Volume
from .sampling import TARGET_IDS


PATCH_XY = (64, 64)
UNIFORM_2D = ("uniform",) * 10
ORGAN_AWARE_2D = ("uniform",) * 4 + ("Bladder",) * 2 + ("Rectum",) * 2 + ("Prostate",) * 2


def slice_crop_origin(mask_xyz: np.ndarray, target: str, *, seed: int, epoch: int,
                      case_id: str, draw: int, patch_xy: tuple[int, int] = PATCH_XY
                      ) -> tuple[int, int, int]:
    """Select an axial slice and XY crop; a target draw contains its class."""
    if (mask_xyz.ndim != 3 or mask_xyz.dtype.kind not in "iu"
            or any(size < 1 or size > n for size, n in zip(patch_xy, mask_xyz.shape[:2]))
            or mask_xyz.shape[2] < 1):
        raise ValueError("Invalid XYZ mask or 2D crop shape")
    key = hashlib.sha256(f"{seed}|{epoch}|{case_id}|{draw}|{target}|2d".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(key[:8], "little"))
    if target == "uniform":
        return (int(rng.integers(mask_xyz.shape[0] - patch_xy[0] + 1)),
                int(rng.integers(mask_xyz.shape[1] - patch_xy[1] + 1)),
                int(rng.integers(mask_xyz.shape[2])))
    if target not in TARGET_IDS:
        raise ValueError(f"Unknown target: {target}")
    locations = np.argwhere(mask_xyz == TARGET_IDS[target])
    if len(locations) == 0:
        raise ValueError(f"{case_id}: {target} absent after preprocessing")
    x, y, z = (int(v) for v in locations[int(rng.integers(len(locations)))])
    starts = []
    for n, size, pos in zip(mask_xyz.shape[:2], patch_xy, (x, y)):
        low, high = max(0, pos - size + 1), min(pos, n - size)
        starts.append(int(rng.integers(low, high + 1)))
    return starts[0], starts[1], z


def crop_for_2d_model(volume: Volume, origin: tuple[int, int, int],
                      patch_xy: tuple[int, int] = PATCH_XY
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Return image [1,H,W] and paired integer mask [H,W] from an XYZ volume."""
    if volume.image.shape != volume.mask.shape or len(origin) != 3:
        raise ValueError("MRI/mask grid or origin mismatch")
    x, y, z = origin
    sx, sy = patch_xy
    nx, ny, nz = volume.image.shape
    if x < 0 or y < 0 or z < 0 or x + sx > nx or y + sy > ny or z >= nz:
        raise ValueError("2D crop falls outside the shared grid")
    bounds = (slice(x, x + sx), slice(y, y + sy), z)
    image = np.asarray(volume.image[bounds].T[None], dtype=np.float32)
    mask = np.asarray(volume.mask[bounds].T, dtype=np.int64)
    if not np.isfinite(image).all() or np.any((mask < 0) | (mask >= NUM_CLASSES)):
        raise ValueError("Nonfinite MRI or illegal 2D label")
    return image, mask
