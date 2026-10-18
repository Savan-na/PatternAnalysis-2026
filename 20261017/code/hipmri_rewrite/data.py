"""Paired geometry-aware NIfTI handling and shared downsampling.

All arrays use NIfTI voxel order (X, Y, Z) until the model input adapter
explicitly transposes them. This module never modifies source files.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import NUM_CLASSES
from .splits import Case


@dataclass(frozen=True)
class Volume:
    case: Case
    image: np.ndarray
    mask: np.ndarray
    affine: np.ndarray
    spacing_mm: tuple[float, float, float]


def load_case(case: Case) -> Volume:
    """Decode one approved pair, refusing mismatched grids and invalid labels."""
    import nibabel as nib

    mri = nib.load(str(case.mri))
    seg = nib.load(str(case.mask))
    if mri.ndim != 3 or seg.ndim != 3 or mri.shape != seg.shape:
        raise ValueError(f"{case.identifier}: MRI/mask must share a 3D shape")
    if not np.allclose(mri.affine, seg.affine, rtol=0, atol=1e-4):
        raise ValueError(f"{case.identifier}: MRI/mask affines differ")
    if not np.allclose(mri.header.get_zooms()[:3], seg.header.get_zooms()[:3], rtol=0, atol=1e-5):
        raise ValueError(f"{case.identifier}: MRI/mask voxel spacing differs")
    image_unit = mri.header.get_xyzt_units()[0]
    mask_unit = seg.header.get_xyzt_units()[0]
    if image_unit != "mm" or mask_unit not in ("mm", "unknown"):
        raise ValueError(f"{case.identifier}: unsupported spatial units {image_unit}, {mask_unit}")
    affine = np.asarray(mri.affine, dtype=np.float64)
    if not np.isfinite(affine).all() or np.linalg.matrix_rank(affine[:3, :3]) != 3:
        raise ValueError(f"{case.identifier}: invalid affine")
    image = mri.get_fdata(dtype=np.float32, caching="unchanged")
    raw_mask = np.asarray(seg.dataobj)
    if not np.isfinite(image).all() or not np.isfinite(raw_mask).all():
        raise ValueError(f"{case.identifier}: non-finite image or mask")
    if np.any(raw_mask != np.floor(raw_mask)) or np.any((raw_mask < 0) | (raw_mask >= NUM_CLASSES)):
        raise ValueError(f"{case.identifier}: mask labels must be integer 0..5")
    return Volume(case, image, raw_mask.astype(np.uint8), affine,
                  tuple(float(v) for v in mri.header.get_zooms()[:3]))


def downsample(volume: Volume, factor: tuple[int, int, int] = (2, 2, 2)) -> Volume:
    """Sample MRI/mask at identical voxel-edge-aligned physical locations.

    Output centre j maps to source centre (j + 1/2)*factor - 1/2. The affine
    is transformed by that same map. Lost source labels trigger a failure.
    """
    from scipy.ndimage import map_coordinates

    if len(factor) != 3 or any(type(v) is not int or v < 1 for v in factor):
        raise ValueError("factor must contain three positive integers")
    source_shape = np.asarray(volume.image.shape, dtype=np.int64)
    target_shape = tuple(int(np.ceil(n / f)) for n, f in zip(source_shape, factor))
    axes = [(np.arange(size, dtype=np.float64) + 0.5) * f - 0.5
            for size, f in zip(target_shape, factor)]
    grid = np.meshgrid(*axes, indexing="ij", sparse=False)
    coords = np.asarray(grid)
    image = map_coordinates(volume.image, coords, order=1, mode="nearest").astype(np.float32)
    mask = map_coordinates(volume.mask, coords, order=0, mode="grid-constant", cval=0).astype(np.uint8)
    before_counts = np.bincount(volume.mask.ravel(), minlength=NUM_CLASSES)
    after_counts = np.bincount(mask.ravel(), minlength=NUM_CLASSES)
    lost = np.flatnonzero((before_counts > 0) & (after_counts == 0))
    if lost.size:
        raise ValueError(f"{volume.case.identifier}: downsampling removed classes {lost.tolist()}")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.diag(factor)
    transform[:3, 3] = (np.asarray(factor, dtype=np.float64) - 1) / 2
    affine = volume.affine @ transform
    spacing = tuple(float(v * f) for v, f in zip(volume.spacing_mm, factor))
    return Volume(volume.case, image, mask, affine, spacing)


def zscore_nonzero(volume: Volume) -> Volume:
    """Normalize each MRI using image-only nonzero voxels; retain outside zeros."""
    chosen = volume.image != 0
    if not np.any(chosen):
        raise ValueError(f"{volume.case.identifier}: all-zero MRI")
    values = volume.image[chosen].astype(np.float64)
    mean, scale = float(values.mean()), float(values.std())
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"{volume.case.identifier}: constant or invalid MRI")
    image = np.zeros_like(volume.image, dtype=np.float32)
    image[chosen] = ((values - mean) / scale).astype(np.float32)
    return Volume(volume.case, image, volume.mask, volume.affine, volume.spacing_mm)


def one_hot(mask: np.ndarray) -> np.ndarray:
    """Return channel-first six-class float32 mask in the same spatial grid."""
    if mask.ndim not in (2, 3) or mask.dtype.kind not in "iu":
        raise ValueError("mask must be a 2D/3D integer array")
    if np.any((mask < 0) | (mask >= NUM_CLASSES)):
        raise ValueError("mask contains IDs outside 0..5")
    return np.moveaxis(np.eye(NUM_CLASSES, dtype=np.float32)[mask], -1, 0)


def to_3d_model_arrays(volume: Volume) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int]]:
    """Convert XYZ to DHW, padding only high ends to a multiple of 16.

    The original downsampled XYZ shape is returned for exact inversion after
    prediction. Padded mask voxels are background and must be excluded from
    final metrics by removing padding first.
    """
    shape_xyz = tuple(int(n) for n in volume.image.shape)
    image = np.transpose(volume.image, (2, 1, 0))
    mask = np.transpose(volume.mask, (2, 1, 0))
    padding = tuple((0, (-n) % 16) for n in image.shape)
    return (np.pad(image, padding).astype(np.float32, copy=False),
            np.pad(mask, padding).astype(np.int64, copy=False), shape_xyz)


def from_3d_model_prediction(prediction_dhw: np.ndarray, shape_xyz: tuple[int, int, int]) -> np.ndarray:
    """Remove high-end padding and return a label volume in the input XYZ grid."""
    if prediction_dhw.ndim != 3 or len(shape_xyz) != 3:
        raise ValueError("Expected DHW prediction and XYZ original shape")
    shape_dhw = shape_xyz[::-1]
    if any(size > available or size < 1 for size, available in zip(shape_dhw, prediction_dhw.shape)):
        raise ValueError("Prediction is too small for the recorded original shape")
    cropped = prediction_dhw[tuple(slice(0, size) for size in shape_dhw)]
    return np.transpose(cropped, (2, 1, 0))
