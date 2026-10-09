"""Discover HipMRI volumes and apply the approved patient-level split."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


DEFAULT_DATASET_ROOT = Path("/home/groups/comp3710/HipMRI_Study_open")
SPLIT_SEED_PROVENANCE = 3710

# These assignments are frozen; the seed records their provenance, not a runtime draw.
APPROVED_PATIENTS = {
    "train": (
        "B006", "B038", "C032", "D031", "G021", "H007", "H017",
        "J005", "K019", "K042", "M004", "M013", "M023", "M036",
        "N010", "O025", "R016", "R024", "R039", "S028", "S033",
        "S035", "T009", "T014", "V027", "W012",
    ),
    "validation": ("B040", "J026", "K008", "K018", "M030", "S022"),
    "test": ("B037", "L011", "M015", "M020", "W029", "W041"),
}
EXPECTED_VOLUME_COUNTS = {"train": 143, "validation": 38, "test": 30}
EXPECTED_PATIENT_COUNTS = {"train": 26, "validation": 6, "test": 6}
EXPECTED_PATIENT_COUNT = 38
EXPECTED_VOLUME_COUNT = 211

_FILENAME_PATTERN = re.compile(
    r"^(?P<patient>[A-Z][0-9]{3})_Week(?P<week>[0-9]+)_(?P<kind>LFOV|SEMANTIC)\.nii\.gz$"
)


@dataclass(frozen=True)
class VolumePair:
    """Paths for one patient's MRI and semantic mask at one Week."""

    patient_id: str
    week: int
    mri_path: Path
    segmentation_path: Path


def parse_volume_filename(path: str | Path, expected_kind: str) -> tuple[str, int]:
    """Return the patient ID and Week number from a HipMRI NIfTI filename."""

    filename = Path(path).name
    match = _FILENAME_PATTERN.fullmatch(filename)
    if match is None or match.group("kind") != expected_kind:
        raise ValueError(
            f"Invalid {expected_kind} filename {filename!r}; expected "
            f"<PatientID>_Week<number>_{expected_kind}.nii.gz "
            f"(for example, K019_Week1_{expected_kind}.nii.gz)."
        )
    return match.group("patient"), int(match.group("week"))


def validate_approved_split() -> dict[str, str]:
    """Check the frozen assignments and return patient-to-split lookup."""

    actual_counts = {split: len(ids) for split, ids in APPROVED_PATIENTS.items()}
    if actual_counts != EXPECTED_PATIENT_COUNTS:
        raise ValueError(
            f"Approved split patient counts are {actual_counts}; "
            f"expected {EXPECTED_PATIENT_COUNTS}."
        )

    patient_to_split: dict[str, str] = {}
    for split, patient_ids in APPROVED_PATIENTS.items():
        for patient_id in patient_ids:
            if patient_id in patient_to_split:
                raise ValueError(
                    f"Patient leakage in approved split: {patient_id} appears in "
                    f"both {patient_to_split[patient_id]} and {split}."
                )
            patient_to_split[patient_id] = split

    if len(patient_to_split) != EXPECTED_PATIENT_COUNT:
        raise ValueError(
            f"Approved split contains {len(patient_to_split)} unique patients; "
            f"expected {EXPECTED_PATIENT_COUNT}."
        )
    if patient_to_split.get("K019") != "train":
        raise ValueError("Approved split must keep K019 (including Week1) in train.")
    return patient_to_split


def _index_files(directory: Path, kind: str) -> dict[tuple[str, int], Path]:
    """Index only the specified HipMRI file type, rejecting duplicate Weeks."""

    suffix = "LFOV" if kind == "MRI" else "SEMANTIC"
    indexed: dict[tuple[str, int], Path] = {}
    for path in sorted(directory.glob(f"*_{suffix}.nii.gz")):
        if not path.is_file():
            continue
        key = parse_volume_filename(path, suffix)
        if key in indexed:
            raise ValueError(
                f"Duplicate {kind} for {key[0]}_Week{key[1]}: "
                f"{indexed[key]} and {path}."
            )
        indexed[key] = path
    return indexed


def discover_pairs(dataset_root: str | Path = DEFAULT_DATASET_ROOT) -> tuple[VolumePair, ...]:
    """Find MRI/mask pairs without reading or changing volume contents."""

    root = Path(dataset_root)
    if not root.is_dir():
        raise FileNotFoundError(f"HipMRI dataset root is missing or not a directory: {root}")

    mri_dir = root / "semantic_MRs"
    mask_dir = root / "semantic_labels_only"
    for directory in (mri_dir, mask_dir):
        if not directory.is_dir():
            raise FileNotFoundError(f"Required HipMRI directory is missing: {directory}")

    mris = _index_files(mri_dir, "MRI")
    masks = _index_files(mask_dir, "segmentation")
    if not mris and not masks:
        raise FileNotFoundError(
            f"No *_LFOV.nii.gz MRI or *_SEMANTIC.nii.gz segmentation files "
            f"were found under {root}."
        )

    without_mask = sorted(mris.keys() - masks.keys())
    without_mri = sorted(masks.keys() - mris.keys())
    if without_mask or without_mri:
        missing_masks = [f"{patient}_Week{week}_SEMANTIC.nii.gz" for patient, week in without_mask]
        missing_mris = [f"{patient}_Week{week}_LFOV.nii.gz" for patient, week in without_mri]
        raise FileNotFoundError(
            "HipMRI MRI/segmentation pairing mismatch. "
            f"Missing segmentation files in {mask_dir}: {missing_masks}; "
            f"missing MRI files in {mri_dir}: {missing_mris}."
        )

    return tuple(
        VolumePair(patient, week, mris[(patient, week)], masks[(patient, week)])
        for patient, week in sorted(mris)
    )


def assign_approved_split(
    pairs: tuple[VolumePair, ...],
) -> dict[str, tuple[VolumePair, ...]]:
    """Validate full dataset coverage and group every Week by approved patient."""

    patient_to_split = validate_approved_split()
    observed_patients = {pair.patient_id for pair in pairs}
    approved_patients = set(patient_to_split)
    unexpected = sorted(observed_patients - approved_patients)
    missing = sorted(approved_patients - observed_patients)
    if unexpected or missing:
        raise ValueError(
            "HipMRI patient coverage mismatch: "
            f"unexpected patients={unexpected}; missing approved patients={missing}."
        )

    keys = [(pair.patient_id, pair.week) for pair in pairs]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate patient/Week pairs were supplied.")
    if ("K019", 1) not in set(keys):
        raise ValueError("Expected K019_Week1 MRI/segmentation pair is missing.")

    grouped: dict[str, list[VolumePair]] = {split: [] for split in APPROVED_PATIENTS}
    for pair in pairs:
        grouped[patient_to_split[pair.patient_id]].append(pair)

    actual_counts = {split: len(grouped[split]) for split in APPROVED_PATIENTS}
    if actual_counts != EXPECTED_VOLUME_COUNTS or len(pairs) != EXPECTED_VOLUME_COUNT:
        raise ValueError(
            "HipMRI volume coverage mismatch: "
            f"found {actual_counts} (total {len(pairs)}); "
            f"expected {EXPECTED_VOLUME_COUNTS} (total {EXPECTED_VOLUME_COUNT})."
        )
    return {
        split: tuple(sorted(grouped[split], key=lambda pair: (pair.patient_id, pair.week)))
        for split in APPROVED_PATIENTS
    }


def discover_dataset(
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
) -> dict[str, tuple[VolumePair, ...]]:
    """Discover, pair, and validate the complete approved HipMRI dataset."""

    return assign_approved_split(discover_pairs(dataset_root))
