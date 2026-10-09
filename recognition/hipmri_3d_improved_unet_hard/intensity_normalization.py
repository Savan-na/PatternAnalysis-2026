"""Normalise one resampled HipMRI MRI without using segmentation labels."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .dataset import VolumePair
    from .resampling import ResampledVolumePair


def _percentile_settings() -> tuple[float, float]:
    """Read and verify the fixed candidate settings declared in JSON."""

    config = json.loads(Path(__file__).with_name("normalization_config.json").read_text(encoding="utf-8"))
    expected = {
        "lower_nonzero_percentile": 0.5,
        "upper_nonzero_percentile": 99.5,
        "voxel_selection": "original_mri_nonzero",
        "clipping_scope": "selected_mri_voxels_only",
        "zscore_ddof": 0,
        "original_zero_policy": "preserve_zero",
        "statistics_dtype": "float64",
        "output_dtype": "float32",
    }
    if config != expected:
        raise ValueError("normalization_config.json does not match the supported Batch 2.3 policy.")
    return config["lower_nonzero_percentile"], config["upper_nonzero_percentile"]


LOWER_PERCENTILE, UPPER_PERCENTILE = _percentile_settings()


@dataclass(frozen=True)
class NormalizationStatistics:
    """Statistics from the selected MRI voxels, before and after normalisation."""

    nonzero_voxel_count: int
    lower_percentile: float
    upper_percentile: float
    lower_intensity_bound: float
    upper_intensity_bound: float
    clipped_mean: float
    clipped_population_std: float
    original_zero_fraction: float
    output_selected_mean: float
    output_selected_population_std: float


@dataclass(frozen=True)
class NormalizedVolumePair:
    """One normalised pair, retaining spatial and label provenance without source MRI."""

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
    normalization: NormalizationStatistics


def normalize_volume_pair(resampled: ResampledVolumePair) -> NormalizedVolumePair:
    """Clip MRI nonzeros at 0.5/99.5 percentiles, then Z-score those voxels.

    Selection uses the original MRI alone. Segmentation is validated and
    returned by reference, without a full-volume copy or any modification.
    """

    case = f"{resampled.pair.patient_id}_Week{resampled.pair.week}"
    mri = resampled.mri
    segmentation = resampled.segmentation
    if not isinstance(mri, np.ndarray) or mri.ndim != 3 or mri.dtype != np.float32:
        raise ValueError(f"{case}: MRI must be a three-dimensional float32 array.")
    if not isinstance(segmentation, np.ndarray) or segmentation.ndim != 3 or segmentation.shape != mri.shape:
        raise ValueError(f"{case}: segmentation must have the same three-dimensional shape as MRI.")
    if segmentation.dtype != np.uint8 or np.any(segmentation > 5):
        raise ValueError(f"{case}: segmentation must be uint8 with labels 0–5.")
    affine = np.asarray(resampled.affine)
    if affine.shape != (4, 4) or not np.isfinite(affine).all():
        raise ValueError(f"{case}: affine must be a finite 4x4 matrix.")
    if not np.isfinite(mri).all():
        raise ValueError(f"{case}: MRI contains non-finite values.")

    selected = mri != 0
    count = int(np.count_nonzero(selected))
    if count == 0:
        raise ValueError(f"{case}: MRI is entirely zero; nonzero normalisation is undefined.")

    # Extract selected voxels into a float64 working vector. Never mutate
    # source MRI; clip and Z-score this vector before one output cast.
    values = np.asarray(mri[selected], dtype=np.float64)
    lower, upper = (float(v) for v in np.percentile(values, (LOWER_PERCENTILE, UPPER_PERCENTILE)))
    if not np.isfinite((lower, upper)).all():
        raise ValueError(f"{case}: nonzero percentile bounds are not finite.")
    np.clip(values, lower, upper, out=values)
    mean = float(np.mean(values, dtype=np.float64))
    std = float(np.std(values, dtype=np.float64, ddof=0))
    scale = max(abs(lower), abs(upper))
    tolerance = np.finfo(np.float32).eps * scale
    if not np.isfinite((mean, std)).all() or std <= tolerance:
        raise ValueError(
            f"{case}: clipped nonzero MRI population standard deviation is zero or "
            f"numerically degenerate (std={std}, tolerance={tolerance})."
        )

    values -= mean
    values /= std
    normalized = np.zeros(mri.shape, dtype=np.float32)
    normalized[selected] = values
    if not np.isfinite(normalized).all():
        raise ValueError(f"{case}: normalised MRI contains non-finite float32 values.")
    output_selected = normalized[selected]
    stats = NormalizationStatistics(
        nonzero_voxel_count=count,
        lower_percentile=LOWER_PERCENTILE,
        upper_percentile=UPPER_PERCENTILE,
        lower_intensity_bound=lower,
        upper_intensity_bound=upper,
        clipped_mean=mean,
        clipped_population_std=std,
        original_zero_fraction=1.0 - count / mri.size,
        output_selected_mean=float(np.mean(output_selected, dtype=np.float64)),
        output_selected_population_std=float(np.std(output_selected, dtype=np.float64, ddof=0)),
    )
    return NormalizedVolumePair(
        pair=resampled.pair,
        mri=normalized,
        segmentation=segmentation,
        affine=np.array(affine, copy=True),
        target_spacing_mm=resampled.target_spacing_mm,
        original_shape=resampled.original_shape,
        input_labels=resampled.input_labels,
        output_labels=resampled.output_labels,
        input_label_counts=resampled.input_label_counts,
        output_label_counts=resampled.output_label_counts,
        input_voxel_volume_mm3=resampled.input_voxel_volume_mm3,
        output_voxel_volume_mm3=resampled.output_voxel_volume_mm3,
        input_class_volumes_mm3=resampled.input_class_volumes_mm3,
        output_class_volumes_mm3=resampled.output_class_volumes_mm3,
        spatial_unit=resampled.spatial_unit,
        segmentation_header_spatial_unit=resampled.segmentation_header_spatial_unit,
        segmentation_unit_inferred=resampled.segmentation_unit_inferred,
        normalization=stats,
    )
