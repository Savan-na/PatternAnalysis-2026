"""Read-only Train-array quality audit; other splits are inventoried only.

Run from an environment with the project's numpy, scipy and nibabel dependencies:
    python -B tools/audit_train_raw.py --dataset-root PATH --output ../logs/audits/train_full.json

The input dataset is never modified. The output must be inside this 20261017 project.
A mechanical pass does not certify anatomical annotation quality.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from time import perf_counter
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
CODE = PROJECT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

LABELS = ("Background", "Body", "Bones", "Bladder", "Rectum", "Prostate")
PATTERN = re.compile(r"^(?P<patient>[A-Z][0-9]{3})_Week(?P<week>[0-9]+)_(?P<role>LFOV|SEMANTIC)\.nii\.gz$")


def train_inventory(root: Path, patients: set[str]) -> tuple[list[tuple[str, Path, Path]], list[str]]:
    folders = {"mri": root / "semantic_MRs", "mask": root / "semantic_labels_only"}
    for folder in folders.values():
        if not folder.is_dir():
            raise FileNotFoundError(folder)
    found: dict[str, dict[str, Path]] = {"mri": {}, "mask": {}}
    problems: list[str] = []
    for role, suffix in (("mri", "LFOV"), ("mask", "SEMANTIC")):
        for patient in sorted(patients):
            for path in folders[role].glob(f"{patient}_Week*_{suffix}.nii.gz"):
                match = PATTERN.fullmatch(path.name)
                if match is None or match["patient"] != patient or match["role"] != suffix:
                    problems.append(f"Unexpected Train {role} filename: {path.name}")
                    continue
                case = f"{patient}_Week{int(match['week'])}"
                if case in found[role]:
                    problems.append(f"Duplicate Train {role} case: {case}")
                found[role][case] = path
    pairs = []
    for case in sorted(found["mri"].keys() | found["mask"].keys()):
        if case not in found["mri"] or case not in found["mask"]:
            problems.append(f"Missing paired Train file: {case}")
        else:
            pairs.append((case, found["mri"][case], found["mask"][case]))
    return pairs, problems


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_case(case: str, mri_path: Path, mask_path: Path) -> dict:
    import nibabel as nib
    import numpy as np
    from scipy import ndimage
    row: dict = {"case": case, "errors": [], "warnings": []}
    try:
        row["mri_file_bytes"] = mri_path.stat().st_size
        row["mask_file_bytes"] = mask_path.stat().st_size
        mri, seg = nib.load(str(mri_path)), nib.load(str(mask_path))
        row["mri_shape"] = list(mri.shape)
        row["mask_shape"] = list(seg.shape)
        row["mri_dtype"] = str(mri.get_data_dtype())
        row["mask_dtype"] = str(seg.get_data_dtype())
        row["mri_spacing_mm"] = list(map(float, mri.header.get_zooms()[:3]))
        row["mask_spacing_mm"] = list(map(float, seg.header.get_zooms()[:3]))
        row["mri_unit"] = mri.header.get_xyzt_units()[0]
        row["mask_unit"] = seg.header.get_xyzt_units()[0]
        row["mri_orientation"] = list(nib.aff2axcodes(mri.affine))
        row["mask_orientation"] = list(nib.aff2axcodes(seg.affine))
        row["mri_sform_code"] = int(mri.header["sform_code"])
        row["mask_sform_code"] = int(seg.header["sform_code"])
        row["mri_qform_code"] = int(mri.header["qform_code"])
        row["mask_qform_code"] = int(seg.header["qform_code"])
        row["affine_max_abs_difference"] = float(np.max(np.abs(mri.affine - seg.affine)))
        row["mri_qform_sform_max_abs_difference"] = None
        row["mask_qform_sform_max_abs_difference"] = None
        for name, image in (("mri", mri), ("mask", seg)):
            sform, sform_code = image.get_sform(coded=True)
            qform, qform_code = image.get_qform(coded=True)
            if sform_code and qform_code:
                difference = float(np.max(np.abs(sform - qform)))
                row[f"{name}_qform_sform_max_abs_difference"] = difference
                if difference > 1e-4:
                    row["warnings"].append(f"{name.upper()} qform and sform disagree")
        if mri.ndim != 3 or seg.ndim != 3 or mri.shape != seg.shape:
            row["errors"].append("MRI/mask shape mismatch or non-3D input")
        if not np.isfinite(mri.affine).all() or not np.isfinite(seg.affine).all():
            row["errors"].append("Non-finite affine")
        elif not np.allclose(mri.affine, seg.affine, rtol=0, atol=1e-4):
            row["errors"].append("MRI/mask affine mismatch")
        elif np.linalg.matrix_rank(mri.affine[:3, :3]) != 3:
            row["errors"].append("Singular spatial affine")
        if any(not np.isfinite(value) or value <= 0 for value in row["mri_spacing_mm"]):
            row["errors"].append("MRI spacing must be positive and finite")
        if any(not np.isfinite(value) or value <= 0 for value in row["mask_spacing_mm"]):
            row["errors"].append("Mask spacing must be positive and finite")
        if not np.allclose(row["mri_spacing_mm"], row["mask_spacing_mm"], rtol=0, atol=1e-5):
            row["errors"].append("MRI/mask spacing mismatch")
        if row["mask_unit"] == "unknown":
            row["warnings"].append("Mask spatial unit unknown; verify physical alignment")
        elif row["mask_unit"] != "mm":
            row["errors"].append("Mask spatial unit is not millimetres")
        if row["mri_unit"] != "mm":
            row["errors"].append("MRI spatial unit is not millimetres")

        image = mri.get_fdata(dtype=np.float32, caching="unchanged")
        mask = np.asanyarray(seg.dataobj)
        finite_image = np.isfinite(image)
        finite_mask = np.isfinite(mask)
        row["image_nonfinite_voxels"] = int(image.size - finite_image.sum())
        row["mask_nonfinite_voxels"] = int(mask.size - finite_mask.sum())
        row["image_zero_fraction"] = float(np.count_nonzero(image == 0) / image.size)
        if row["image_nonfinite_voxels"]:
            row["errors"].append("Non-finite MRI values")
        if row["mask_nonfinite_voxels"]:
            row["errors"].append("Non-finite mask values")
        row["image_finite_voxels"] = int(finite_image.sum())
        row["mask_finite_voxels"] = int(finite_mask.sum())
        finite_values = image[finite_image]
        if finite_values.size:
            row["image_finite_min"] = float(finite_values.min())
            row["image_finite_max"] = float(finite_values.max())
            row["image_finite_median"] = float(np.median(finite_values))
            if row["image_finite_min"] == row["image_finite_max"]:
                row["errors"].append("Constant finite MRI")
        else:
            row["errors"].append("MRI has no finite voxels")
        nz = image[(image != 0) & finite_image]
        if nz.size == 0:
            row["errors"].append("Blank or all-zero finite MRI")
        else:
            row["image_min"] = float(nz.min())
            row["image_max"] = float(nz.max())
            row["image_nonzero_mean"] = float(nz.mean(dtype=np.float64))
            row["image_nonzero_std"] = float(nz.std(dtype=np.float64))
            row["image_nonzero_percentiles_0_5_1_50_99_99_5"] = [
                float(v) for v in np.percentile(nz, [0.5, 1, 50, 99, 99.5])
            ]
            if 0 < row["image_nonzero_std"] < 1e-8:
                row["warnings"].append("MRI has near-zero nonzero-voxel variance")

        legal = bool(
            finite_mask.all()
            and not np.any(mask != np.floor(mask))
            and not np.any((mask < 0) | (mask >= len(LABELS)))
        )
        row["mask_legal_ids_only"] = legal
        if not legal:
            row["errors"].append("Mask contains non-finite, fractional or out-of-range IDs")
        if legal and mri.shape == seg.shape and mri.ndim == 3:
            mask = mask.astype(np.uint8, copy=False)
            voxel_mm3 = float(abs(np.linalg.det(seg.affine[:3, :3])))
            row["mask_voxel_volume_mm3"] = voxel_mm3
            class_counts = np.bincount(mask.ravel(), minlength=len(LABELS))
            row["class_counts"] = {
                name: int(class_counts[i]) for i, name in enumerate(LABELS)
            }
            row["missing_class_ids"] = [i for i in range(len(LABELS))
                                        if row["class_counts"][LABELS[i]] == 0]
            if row["missing_class_ids"]:
                row["warnings"].append(
                    f"Class IDs absent from this volume: {row['missing_class_ids']}"
                )
            row["class_volume_ml"] = {name: row["class_counts"][name] * voxel_mm3 / 1000
                                      for name in LABELS[1:]}
            for i in (3, 4, 5):
                name = LABELS[i]
                binary = mask == i
                if not binary.any():
                    continue
                points = np.nonzero(binary)
                item = {
                    "bounding_box_min_xyz": [int(axis.min()) for axis in points],
                    "bounding_box_max_xyz": [int(axis.max()) for axis in points],
                    "occupied_slices_xyz": [int(np.unique(axis).size) for axis in points],
                }
                components, count = ndimage.label(binary)
                item["connected_components_6_neighbour"] = int(count)
                if count:
                    sizes = np.bincount(components.ravel())[1:]
                    item["largest_component_fraction"] = float(sizes.max() / sizes.sum())
                for axis, occupied in enumerate(points):
                    indices = np.unique(occupied)
                    item.setdefault("slice_gaps_xyz", {})[str(axis)] = [
                        int(right - left - 1)
                        for left, right in zip(indices[:-1], indices[1:])
                        if right - left > 1
                    ]
                row.setdefault("small_organ_geometry", {})[name] = item

        row["mri_sha256"] = sha256_file(mri_path)
        row["mask_sha256"] = sha256_file(mask_path)
    except Exception as exc:
        row["errors"].append(f"Case audit failed: {type(exc).__name__}: {exc}")
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, help="Audit only the first N Train cases; never a full-audit pass")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    output = args.output.resolve()
    if not output.is_relative_to(PROJECT):
        parser.error("Output must be within the 20261017 project")
    if output.exists():
        parser.error("Refusing to overwrite an existing audit result")
    from hipmri_rewrite.splits import discover, read_split
    split = read_split()
    inventory_errors: list[str] = []
    try:
        inventory_counts = {
            name: len(cases) for name, cases in discover(args.dataset_root).items()
        }
    except (FileNotFoundError, ValueError) as exc:
        inventory_counts = None
        inventory_errors.append(f"Dataset inventory failed: {type(exc).__name__}: {exc}")
    try:
        pairs, train_inventory_errors = train_inventory(args.dataset_root, set(split["train"]))
        inventory_errors.extend(train_inventory_errors)
    except FileNotFoundError as exc:
        pairs = []
        inventory_errors.append(f"Train inventory failed: {type(exc).__name__}: {exc}")
    if len(pairs) != split["expected_volumes"]["train"]:
        inventory_errors.append(f"Found {len(pairs)} paired Train cases; expected 143")
    if args.limit is not None:
        pairs = pairs[:args.limit]
    rows = []
    total = len(pairs)
    for index, pair in enumerate(pairs, start=1):
        started = perf_counter()
        print(f"[{index}/{total}] Auditing {pair[0]}...", flush=True)
        row = audit_case(*pair)
        rows.append(row)
        elapsed = perf_counter() - started
        print(
            f"[{index}/{total}] {pair[0]}: "
            f"errors={len(row['errors'])}, warnings={len(row['warnings'])}, "
            f"elapsed={elapsed:.1f}s",
            flush=True,
        )
    complete = args.limit is None and len(rows) == split["expected_volumes"]["train"]
    duplicate_file_hashes: dict[str, list[str]] = {}
    for row in rows:
        for role in ("mri", "mask"):
            digest = row.get(f"{role}_sha256")
            if digest:
                duplicate_file_hashes.setdefault(f"{role}:{digest}", []).append(row["case"])
    exact_duplicates = {
        key: cases for key, cases in duplicate_file_hashes.items() if len(cases) > 1
    }
    result = {
        "scope": "G1 raw Train arrays decoded; all split filenames inventoried only; no preprocessing",
        "dataset_root": str(args.dataset_root.resolve()),
        "split_file_sha256": sha256_file(Path(__file__).resolve().parents[1] / "config" / "patient_split.json"),
        "audit_script_sha256": sha256_file(Path(__file__).resolve()),
        "inventory_counts": inventory_counts,
        "expected_train_cases": 143,
        "paired_train_cases": len(rows),
        "complete_train_audit": complete,
        "inventory_errors": inventory_errors,
        "case_error_count": sum(bool(row["errors"]) for row in rows),
        "warning_count": sum(len(row["warnings"]) for row in rows),
        "exact_duplicate_file_contents": exact_duplicates,
        "mechanical_pass": complete and not inventory_errors and all(not row["errors"] for row in rows),
        "anatomical_review_complete": False,
        "cases": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(f"Saved {len(rows)} Train case results to {output}; mechanical_pass={result['mechanical_pass']}")


if __name__ == "__main__":
    main()
