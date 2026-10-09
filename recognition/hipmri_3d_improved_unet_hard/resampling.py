"""Resample one validated HipMRI pair onto a shared physical 3D grid."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import affine_transform

from .dataset import VolumePair
from .volume_io import AFFINE_ATOL, ALLOWED_LABELS, LoadedVolumePair


GRID_ROUND_TOL = 1e-9  # Output-voxel units; removes numerical ceil overshoot.
ORTHOGONAL_ATOL = 1e-6  # Absolute tolerance on unit-direction dot products.
MEMORY_SAFETY_MULTIPLIER = 4  # Planning estimate, not a peak-RSS guarantee.


def _boundary_policy() -> tuple[str, str, int]:
    """Read the declared policy; reject unsupported edits instead of ignoring them."""

    config = json.loads(Path(__file__).with_name("resampling_config.json").read_text(encoding="utf-8"))
    mri_mode = config["mri_boundary_mode"]
    mask_mode = config["segmentation_boundary_mode"]
    value = config["boundary_value"]
    if mri_mode != "nearest" or mask_mode != "grid-constant" or value != 0:
        raise ValueError("Unsupported resampling boundary policy in resampling_config.json.")
    return mri_mode, mask_mode, value


MRI_BOUNDARY_MODE, SEGMENTATION_BOUNDARY_MODE, BOUNDARY_VALUE = _boundary_policy()


@dataclass(frozen=True)
class ResampledVolumePair:
    """One resampled pair; source arrays are deliberately not retained."""

    pair: VolumePair
    mri: np.ndarray
    segmentation: np.ndarray
    affine: np.ndarray
    target_spacing_mm: tuple[float, float, float]
    original_shape: tuple[int, int, int]
    input_labels: tuple[int, ...]
    output_labels: tuple[int, ...]
    input_label_counts: tuple[int, int, int, int, int, int]
    output_label_counts: tuple[int, int, int, int, int, int]
    input_voxel_volume_mm3: float
    output_voxel_volume_mm3: float
    input_class_volumes_mm3: tuple[float, float, float, float, float, float]
    output_class_volumes_mm3: tuple[float, float, float, float, float, float]
    spatial_unit: str
    segmentation_header_spatial_unit: str
    segmentation_unit_inferred: bool


def _spacing(value: object) -> np.ndarray:
    try:
        spacing = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("target_spacing_mm must contain three positive finite numbers.") from exc
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError("target_spacing_mm must contain three positive finite numbers.")
    return spacing


def _geometry(shape: object, affine: object, role: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        sizes = np.asarray(shape, dtype=np.int64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{role} shape must contain three positive integers.") from exc
    if sizes.shape != (3,) or np.any(sizes <= 0) or tuple(sizes) != tuple(shape):
        raise ValueError(f"{role} shape must contain three positive integers.")
    matrix = np.asarray(affine, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{role} affine must be a finite 4x4 matrix.")
    if not np.allclose(matrix[3], [0, 0, 0, 1], rtol=0, atol=1e-8):
        raise ValueError(f"{role} affine must have homogeneous bottom row [0, 0, 0, 1].")
    axes = matrix[:3, :3]
    spacing = np.linalg.norm(axes, axis=0)
    if not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError(f"{role} affine has singular or invalid spatial axes.")
    directions = axes / spacing
    if not np.allclose(directions.T @ directions, np.eye(3), rtol=0, atol=ORTHOGONAL_ATOL):
        raise ValueError(f"{role} affine has shear or non-orthogonal spatial axes.")
    if abs(np.linalg.det(directions)) < 1 - 3 * ORTHOGONAL_ATOL:
        raise ValueError(f"{role} affine has singular spatial axes.")
    return sizes, matrix, directions


def make_target_grid(
    shape: tuple[int, int, int], affine: np.ndarray, target_spacing_mm: object
) -> tuple[tuple[int, int, int], np.ndarray]:
    """Cover source voxel *edges* with a grid keeping source axis directions.

    Source centers have indices 0..n-1, so edges span [-0.5, n-0.5].
    Project the eight edge corners onto target voxel axes. If projected
    bounds are lo/hi, output center 0 is at lo+0.5 and the output length is
    ceil(hi-lo). Near-integer extents are rounded to remove floating error.
    """

    sizes, source_affine, directions = _geometry(shape, affine, "source")
    target = _spacing(target_spacing_mm)
    output_axes = directions * target
    corners = np.asarray(list(product(*[(-0.5, float(n) - 0.5) for n in sizes])))
    world = corners @ source_affine[:3, :3].T + source_affine[:3, 3]
    coordinates = np.linalg.solve(output_axes, (world - source_affine[:3, 3]).T).T
    lower = coordinates.min(axis=0)
    extent = coordinates.max(axis=0) - lower
    rounded = np.rint(extent)
    near_integer = np.abs(extent - rounded) <= GRID_ROUND_TOL
    lengths = np.where(near_integer, rounded, np.ceil(extent))
    if not np.isfinite(lengths).all() or np.any(lengths < 1) or np.any(lengths > np.iinfo(np.intp).max):
        raise ValueError("target spacing produces an invalid or oversized output grid.")
    output_shape = tuple(int(n) for n in lengths)
    output_affine = np.eye(4, dtype=np.float64)
    output_affine[:3, :3] = output_axes
    output_affine[:3, 3] = source_affine[:3, 3] + output_axes @ (lower + 0.5)
    return output_shape, output_affine


def _counts(mask: np.ndarray) -> tuple[int, int, int, int, int, int]:
    labels, counts = np.unique(mask, return_counts=True)
    if not set(int(label) for label in labels).issubset(ALLOWED_LABELS):
        raise ValueError(f"Segmentation contains labels outside 0–5: {labels.tolist()}.")
    result = np.zeros(6, dtype=np.int64)
    result[labels.astype(np.intp)] = counts
    return tuple(int(count) for count in result)


def resample_volume_pair(
    loaded: LoadedVolumePair,
    target_spacing_mm: object,
    *,
    max_estimated_bytes: int,
) -> ResampledVolumePair:
    """Resample one loaded pair on CPU with linear MRI and nearest mask sampling.

    The 4x estimate includes source and output array bytes plus a safety
    multiplier for transient allocations. It does not bound actual process RSS.
    """

    case = f"{loaded.pair.patient_id}_Week{loaded.pair.week}"
    target = _spacing(target_spacing_mm)
    if isinstance(max_estimated_bytes, bool) or not isinstance(max_estimated_bytes, int) or max_estimated_bytes <= 0:
        raise ValueError("max_estimated_bytes must be a positive integer.")
    if loaded.spatial_unit != "mm":
        raise ValueError(f"{case}: target spacing is in mm but MRI spatial unit is {loaded.spatial_unit!r}.")
    if loaded.segmentation_header_spatial_unit != "mm" and not (
        loaded.segmentation_header_spatial_unit == "unknown" and loaded.segmentation_unit_inferred
    ):
        raise ValueError(f"{case}: segmentation spatial unit is incompatible with mm.")
    if loaded.mri.ndim != 3 or loaded.segmentation.ndim != 3 or loaded.mri.shape != loaded.segmentation.shape:
        raise ValueError(f"{case}: MRI and segmentation must have the same 3D shape.")
    if loaded.mri.dtype != np.float32 or loaded.segmentation.dtype != np.uint8:
        raise ValueError(f"{case}: expected float32 MRI and uint8 segmentation.")
    if not np.isfinite(loaded.mri).all():
        raise ValueError(f"{case}: MRI contains non-finite values.")
    source_shape, source_affine, _ = _geometry(loaded.mri.shape, loaded.mri_affine, "MRI")
    _, mask_affine, _ = _geometry(loaded.segmentation.shape, loaded.segmentation_affine, "segmentation")
    if not np.allclose(source_affine, mask_affine, rtol=0, atol=AFFINE_ATOL):
        raise ValueError(f"{case}: MRI and segmentation source grids do not align.")
    input_counts = _counts(loaded.segmentation)
    output_shape, output_affine = make_target_grid(tuple(source_shape), source_affine, target)

    # Estimate before either output is allocated. The multiplier is conservative
    # planning headroom, not an empirical bound on SciPy or interpreter memory.
    output_voxels = int(np.prod(output_shape, dtype=object))
    estimated_bytes = MEMORY_SAFETY_MULTIPLIER * (
        loaded.mri.nbytes + loaded.segmentation.nbytes
        + output_voxels * (np.dtype("float32").itemsize + np.dtype("uint8").itemsize)
    )
    if estimated_bytes > max_estimated_bytes:
        raise MemoryError(
            f"{case}: estimated resampling allocation {estimated_bytes} bytes exceeds "
            f"max_estimated_bytes={max_estimated_bytes}; this is a planning guard, not peak RSS."
        )

    def transform(
        array: np.ndarray, affine: np.ndarray, order: int, dtype: np.dtype, mode: str
    ) -> np.ndarray:
        pull = np.linalg.solve(affine, output_affine)
        return affine_transform(
            array, pull[:3, :3], offset=pull[:3, 3], output_shape=output_shape,
            output=dtype, order=order, mode=mode, cval=BOUNDARY_VALUE, prefilter=False,
        )

    # MRI extends its edge intensity rather than forming an artificial dark
    # ramp between the last center and voxel edge. Nearest-neighbour mask
    # sampling instead uses grid-constant so edge labels survive inside the
    # source voxel-edge domain, with background beyond it.
    mri = transform(loaded.mri, source_affine, 1, np.float32, MRI_BOUNDARY_MODE)
    segmentation = transform(
        loaded.segmentation, mask_affine, 0, np.uint8, SEGMENTATION_BOUNDARY_MODE
    )
    if mri.shape != output_shape or mri.dtype != np.float32 or not np.isfinite(mri).all():
        raise ValueError(f"{case}: resampled MRI has invalid shape, dtype, or non-finite values.")
    if segmentation.shape != output_shape or segmentation.dtype != np.uint8:
        raise ValueError(f"{case}: resampled segmentation has invalid shape or dtype.")
    output_counts = _counts(segmentation)
    missing = [label for label in range(1, 6) if input_counts[label] and not output_counts[label]]
    if missing:
        raise ValueError(f"{case}: resampling removed existing foreground label(s) {missing}.")
    new_labels = [label for label in range(6) if output_counts[label] and not input_counts[label]]
    if any(label != 0 for label in new_labels):
        raise ValueError(f"{case}: resampling introduced unexpected label(s) {new_labels}.")

    input_voxel_volume = float(abs(np.linalg.det(source_affine[:3, :3])))
    output_voxel_volume = float(abs(np.linalg.det(output_affine[:3, :3])))
    return ResampledVolumePair(
        pair=loaded.pair, mri=mri, segmentation=segmentation, affine=output_affine,
        target_spacing_mm=tuple(float(value) for value in target),
        original_shape=tuple(int(value) for value in source_shape),
        input_labels=tuple(label for label, count in enumerate(input_counts) if count),
        output_labels=tuple(label for label, count in enumerate(output_counts) if count),
        input_label_counts=input_counts, output_label_counts=output_counts,
        input_voxel_volume_mm3=input_voxel_volume,
        output_voxel_volume_mm3=output_voxel_volume,
        input_class_volumes_mm3=tuple(count * input_voxel_volume for count in input_counts),
        output_class_volumes_mm3=tuple(count * output_voxel_volume for count in output_counts),
        spatial_unit=loaded.spatial_unit,
        segmentation_header_spatial_unit=loaded.segmentation_header_spatial_unit,
        segmentation_unit_inferred=loaded.segmentation_unit_inferred,
    )
