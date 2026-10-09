"""Load and validate one approved HipMRI MRI/segmentation NIfTI pair."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np

from .dataset import VolumePair


# Header values are stored with finite precision. These absolute tolerances
# use the MRI's explicitly declared spatial unit and allow no relative drift.
AFFINE_ATOL = 1e-4
SPACING_ATOL = 1e-5
SUPPORTED_SPATIAL_UNITS = frozenset(("meter", "mm", "micron"))
ALLOWED_LABELS = frozenset(range(6))


@dataclass(frozen=True)
class LoadedVolumePair:
    """One 3D pair with its declared and, if needed, inferred unit provenance."""

    pair: VolumePair
    mri: np.ndarray
    segmentation: np.ndarray
    mri_affine: np.ndarray
    segmentation_affine: np.ndarray
    voxel_spacing: tuple[float, float, float]
    spatial_unit: str  # Working unit explicitly declared by the MRI.
    segmentation_header_spatial_unit: str  # Actual mask header value.
    segmentation_unit_inferred: bool  # True only for validated mm/unknown pairs.
    axis_codes: tuple[str, str, str]
    mri_stored_dtype: np.dtype
    segmentation_stored_dtype: np.dtype
    labels_present: tuple[int, ...]


def _open_nifti(path: Path, role: str, case: str) -> nib.Nifti1Image | nib.Nifti2Image:
    if not path.is_file():
        raise FileNotFoundError(f"{case}: {role} NIfTI file is missing: {path}")
    try:
        image = nib.load(str(path))
    except (OSError, EOFError, ValueError, nib.filebasedimages.ImageFileError) as exc:
        raise ValueError(f"{case}: cannot read {role} NIfTI file {path}: {exc}") from exc
    if not isinstance(image, (nib.Nifti1Image, nib.Nifti2Image)):
        raise ValueError(f"{case}: {role} file is not a NIfTI image: {path}")
    return image


def load_volume_pair(pair: VolumePair) -> LoadedVolumePair:
    """Decode one pair without reorientation, resampling, or normalisation."""

    case = f"{pair.patient_id}_Week{pair.week}"
    mri_path = Path(pair.mri_path)
    segmentation_path = Path(pair.segmentation_path)
    mri_image = _open_nifti(mri_path, "MRI", case)
    segmentation_image = _open_nifti(segmentation_path, "segmentation", case)

    if mri_image.ndim != 3 or segmentation_image.ndim != 3:
        raise ValueError(
            f"{case}: expected 3D MRI and segmentation; "
            f"got {mri_image.shape} and {segmentation_image.shape}."
        )
    if any(size <= 0 for size in mri_image.shape + segmentation_image.shape):
        raise ValueError(f"{case}: MRI and segmentation dimensions must be positive.")
    if mri_image.shape != segmentation_image.shape:
        raise ValueError(
            f"{case}: MRI/segmentation shape mismatch: "
            f"{mri_image.shape} versus {segmentation_image.shape}."
        )

    mri_affine = np.asarray(mri_image.affine, dtype=np.float64)
    segmentation_affine = np.asarray(segmentation_image.affine, dtype=np.float64)
    for role, affine in (("MRI", mri_affine), ("segmentation", segmentation_affine)):
        if affine.shape != (4, 4) or not np.isfinite(affine).all():
            raise ValueError(f"{case}: {role} affine must be a finite 4x4 matrix.")
        if np.linalg.matrix_rank(affine[:3, :3]) != 3:
            raise ValueError(f"{case}: {role} affine has singular spatial axes.")

    mri_spacing = np.asarray(mri_image.header.get_zooms()[:3], dtype=np.float64)
    segmentation_spacing = np.asarray(
        segmentation_image.header.get_zooms()[:3], dtype=np.float64
    )
    for role, spacing in (("MRI", mri_spacing), ("segmentation", segmentation_spacing)):
        if spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
            raise ValueError(f"{case}: {role} voxel spacing must contain three positive finite values.")

    mri_unit = mri_image.header.get_xyzt_units()[0]
    segmentation_unit = segmentation_image.header.get_xyzt_units()[0]
    if mri_unit not in SUPPORTED_SPATIAL_UNITS:
        raise ValueError(
            f"{case}: MRI spatial unit {mri_unit!r} is not an explicitly "
            "supported physical unit; mask units cannot be inferred."
        )
    segmentation_unit_inferred = mri_unit == "mm" and segmentation_unit == "unknown"
    if segmentation_unit != mri_unit and not segmentation_unit_inferred:
        raise ValueError(
            f"{case}: spatial unit mismatch: MRI={mri_unit!r}, "
            f"segmentation={segmentation_unit!r}."
        )
    if not np.allclose(mri_spacing, segmentation_spacing, rtol=0, atol=SPACING_ATOL):
        raise ValueError(
            f"{case}: voxel spacing mismatch: MRI={tuple(mri_spacing)}, "
            f"segmentation={tuple(segmentation_spacing)} "
            f"(absolute tolerance {SPACING_ATOL})."
        )
    if not np.allclose(mri_affine, segmentation_affine, rtol=0, atol=AFFINE_ATOL):
        raise ValueError(
            f"{case}: MRI/segmentation affine mismatch "
            f"(maximum absolute difference "
            f"{np.max(np.abs(mri_affine - segmentation_affine)):.6g}; "
            f"absolute tolerance {AFFINE_ATOL})."
        )

    axis_codes = nib.aff2axcodes(mri_affine)
    segmentation_axis_codes = nib.aff2axcodes(segmentation_affine)
    if any(code is None for code in axis_codes + segmentation_axis_codes):
        raise ValueError(f"{case}: affine does not define three spatial axis directions.")
    if axis_codes != segmentation_axis_codes:
        raise ValueError(
            f"{case}: MRI/segmentation orientation mismatch: "
            f"{axis_codes} versus {segmentation_axis_codes}."
        )

    try:
        # Do not fill NiBabel's image cache; conversion to float32 is not
        # intensity normalisation. NiBabel applies any NIfTI intensity scaling.
        mri = mri_image.get_fdata(dtype=np.float32, caching="unchanged")
    except (OSError, EOFError, TypeError, ValueError) as exc:
        raise ValueError(f"{case}: cannot decode MRI data from {mri_path}: {exc}") from exc
    if not np.isfinite(mri).all():
        raise ValueError(f"{case}: MRI contains non-finite voxel values.")

    try:
        # ArrayProxy applies NIfTI scaling without caching the full mask on
        # the image object. Validate decoded values before integer conversion.
        decoded_segmentation = np.asarray(segmentation_image.dataobj)
    except (OSError, EOFError, TypeError, ValueError) as exc:
        raise ValueError(
            f"{case}: cannot decode segmentation data from {segmentation_path}: {exc}"
        ) from exc
    if decoded_segmentation.dtype.kind not in "iuf":
        raise ValueError(
            f"{case}: segmentation data must have a real numeric dtype; "
            f"got {decoded_segmentation.dtype}."
        )
    labels = np.unique(decoded_segmentation)
    if not np.isfinite(labels).all():
        raise ValueError(f"{case}: segmentation contains non-finite label values.")
    if not np.all(np.equal(labels, np.floor(labels))):
        raise ValueError(f"{case}: segmentation contains fractional label values.")
    if not set(labels.tolist()).issubset(ALLOWED_LABELS):
        raise ValueError(
            f"{case}: segmentation labels must be within 0-5; "
            f"found {labels.tolist()}."
        )

    segmentation = decoded_segmentation.astype(np.uint8, copy=False)
    return LoadedVolumePair(
        pair=pair,
        mri=mri,
        segmentation=segmentation,
        mri_affine=mri_affine.copy(),
        segmentation_affine=segmentation_affine.copy(),
        voxel_spacing=tuple(float(value) for value in mri_spacing),
        spatial_unit=mri_unit,
        segmentation_header_spatial_unit=segmentation_unit,
        segmentation_unit_inferred=segmentation_unit_inferred,
        axis_codes=tuple(axis_codes),
        mri_stored_dtype=np.dtype(mri_image.get_data_dtype()),
        segmentation_stored_dtype=np.dtype(segmentation_image.get_data_dtype()),
        labels_present=tuple(int(value) for value in labels),
    )
