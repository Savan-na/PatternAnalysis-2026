"""Frozen patient-level discovery. Listing files does not decode any NIfTI arrays."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


_NAME = re.compile(r"^(?P<patient>[A-Z][0-9]{3})_Week(?P<week>[0-9]+)_(?P<kind>LFOV|SEMANTIC)\.nii\.gz$")
DEFAULT_SPLIT = Path(__file__).resolve().parents[1] / "config" / "patient_split.json"


@dataclass(frozen=True)
class Case:
    patient: str
    week: int
    split: str
    mri: Path
    mask: Path

    @property
    def identifier(self) -> str:
        return f"{self.patient}_Week{self.week}"


def read_split(path: str | Path = DEFAULT_SPLIT) -> dict:
    """Reject duplicate patients and silently changed approved cohort sizes."""
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    groups = ("train", "validation", "test")
    expected = (26, 6, 6)
    seen: set[str] = set()
    for group, count in zip(groups, expected):
        names = record[group]
        if len(names) != count or len(set(names)) != count:
            raise ValueError(f"{group} must contain {count} distinct patients")
        if any(re.fullmatch(r"[A-Z][0-9]{3}", name) is None for name in names):
            raise ValueError(f"Invalid patient ID in {group}")
        if seen.intersection(names):
            raise ValueError(f"Patient appears in multiple splits: {seen.intersection(names)}")
        seen.update(names)
    if len(seen) != 38 or record["expected_volumes"] != {
        "train": 143, "validation": 38, "test": 30
    }:
        raise ValueError("Approved cohort counts changed")
    return record


def discover(root: str | Path, split_path: str | Path = DEFAULT_SPLIT) -> dict[str, tuple[Case, ...]]:
    """Match all 211 source files by patient and week; preserve patient grouping."""
    root = Path(root)
    record = read_split(split_path)
    reverse = {name: group for group in ("train", "validation", "test") for name in record[group]}
    indexed: dict[str, dict[tuple[str, int], Path]] = {}
    for role, subdir, suffix in (
        ("mri", "semantic_MRs", "LFOV"),
        ("mask", "semantic_labels_only", "SEMANTIC"),
    ):
        folder = root / subdir
        if not folder.is_dir():
            raise FileNotFoundError(folder)
        matches: dict[tuple[str, int], Path] = {}
        for file in folder.glob("*.nii.gz"):
            match = _NAME.fullmatch(file.name)
            if match is None or match["kind"] != suffix:
                raise ValueError(f"Unexpected {role} file: {file.name}")
            key = (match["patient"], int(match["week"]))
            if key in matches:
                raise ValueError(f"Duplicate {role} case: {key}")
            matches[key] = file
        indexed[role] = matches
    if indexed["mri"].keys() != indexed["mask"].keys():
        raise ValueError("MRI/mask case sets differ")
    cases: dict[str, list[Case]] = {group: [] for group in ("train", "validation", "test")}
    for patient, week in sorted(indexed["mri"]):
        if patient not in reverse:
            raise ValueError(f"Patient is outside approved split: {patient}")
        group = reverse[patient]
        cases[group].append(Case(patient, week, group, indexed["mri"][(patient, week)], indexed["mask"][(patient, week)]))
    for group, expected_count in record["expected_volumes"].items():
        if len(cases[group]) != expected_count:
            raise ValueError(f"{group} has {len(cases[group])}, expected {expected_count} volumes")
    return {group: tuple(items) for group, items in cases.items()}
