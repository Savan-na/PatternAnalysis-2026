"""Read-only Train audit of the existing downsample/normalize/model-array path."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from time import perf_counter


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "code"))
from hipmri_rewrite.data import downsample, load_case, to_3d_model_arrays, zscore_nonzero  # noqa: E402
from hipmri_rewrite.splits import discover  # noqa: E402


CLASSES = ("Background", "Body", "Bones", "Bladder", "Rectum", "Prostate")
AUDIT_SHA = "6a4c674564a0364f5f56730cbc31e67b3ff9a8ffeae7d9d3dd18b198bbd0e1a1"
SPLIT_SHA = "81531ec7cb84ef58c41393ff23e1105e7cf03aa97828fb12853ce8624fcb8900"
DATA_SHA = "b1aa857d8da735fc4d467d520f8a8ebbf8cc6a15614e225a76a4bf0a47c06a85"
SPLITS_CODE_SHA = "a172311556c4bda0364cc7a84def06cf4271e36dd95e6392ce8ef27d39a07a47"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def class_counts(mask) -> dict[str, int]:
    import numpy as np

    counts = np.bincount(mask.ravel(), minlength=len(CLASSES))
    return {name: int(counts[k]) for k, name in enumerate(CLASSES)}


def occupied_z_slices(mask, label: int) -> int:
    import nibabel as nib
    import numpy as np
    import scipy

    return int(np.count_nonzero(np.any(mask == label, axis=(0, 1))))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--audit", type=Path, default=PROJECT / "logs/audits/train_full_20261018_job671208.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    import numpy as np

    output = args.output.resolve()
    if not output.is_relative_to(PROJECT) or output.exists():
        parser.error("Output must be a new file inside the 20261017 project")
    if args.limit is not None and not 1 <= args.limit <= 143:
        parser.error("--limit must be 1..143")
    if (sha256_file(PROJECT / "code/hipmri_rewrite/data.py") != DATA_SHA
            or sha256_file(PROJECT / "code/hipmri_rewrite/splits.py") != SPLITS_CODE_SHA
            or sha256_file(PROJECT / "code/config/patient_split.json") != SPLIT_SHA):
        raise ValueError("Transferred preprocessing or split source differs from reviewed version")
    if sha256_file(args.audit) != AUDIT_SHA:
        raise ValueError("Complete G1 audit SHA-256 mismatch")
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    if not (audit["complete_train_audit"] and audit["mechanical_pass"]
            and audit["split_file_sha256"] == SPLIT_SHA and len(audit["cases"]) == 143):
        raise ValueError("G1 audit is not the approved complete Train run")
    expected = {row["case"]: row for row in audit["cases"]}
    cohort = discover(args.dataset_root)["train"]
    if len(cohort) != 143 or {case.identifier for case in cohort} != set(expected):
        raise ValueError("Dataset Train cases differ from G1")
    selected = cohort[:args.limit] if args.limit else cohort

    rows = []
    for index, case in enumerate(selected, 1):
        started = perf_counter()
        prior = expected[case.identifier]
        if sha256_file(case.mri) != prior["mri_sha256"] or sha256_file(case.mask) != prior["mask_sha256"]:
            raise ValueError(f"Source file changed since G1: {case.identifier}")
        row = {"case": case.identifier, "patient": case.patient, "errors": []}
        try:
            raw = load_case(case)
            raw_counts = class_counts(raw.mask)
            if raw_counts != prior["class_counts"]:
                raise ValueError("Decoded class counts differ from G1")
            small = downsample(raw)
            normalized = zscore_nonzero(small)
            model_image, model_mask, shape_xyz = to_3d_model_arrays(normalized)
            if tuple(shape_xyz) != small.image.shape:
                raise ValueError("Model adapter changed unpadded shape")
            if not np.all(np.isfinite(model_image)) or not np.isin(model_mask, range(6)).all():
                raise ValueError("Invalid model input or labels")
            small_counts = class_counts(small.mask)
            model_counts = class_counts(model_mask)
            padding_voxels = int(model_mask.size - small.mask.size)
            for name in CLASSES:
                expected_model_count = small_counts[name] + (padding_voxels if name == "Background" else 0)
                if model_counts[name] != expected_model_count:
                    raise ValueError(f"Model adapter count mismatch for {name}")
            expected_mask_dhw = np.transpose(small.mask, (2, 1, 0))
            if not np.array_equal(model_mask[:expected_mask_dhw.shape[0],
                                             :expected_mask_dhw.shape[1],
                                             :expected_mask_dhw.shape[2]], expected_mask_dhw):
                raise ValueError("Model adapter changed labels")
            positive = small.image != 0
            mean = float(normalized.image[positive].mean(dtype=np.float64))
            std = float(normalized.image[positive].std(dtype=np.float64))
            if abs(mean) > 1e-4 or abs(std - 1) > 1e-4:
                raise ValueError(f"Unexpected normalized mean/std: {mean}, {std}")
            raw_voxel_mm3 = float(abs(np.linalg.det(raw.affine[:3, :3])))
            small_voxel_mm3 = float(abs(np.linalg.det(small.affine[:3, :3])))
            row.update({
                "raw_shape_xyz": list(raw.image.shape),
                "downsampled_shape_xyz": list(small.image.shape),
                "model_shape_dhw": list(model_image.shape),
                "raw_spacing_mm": list(raw.spacing_mm),
                "downsampled_spacing_mm": list(small.spacing_mm),
                "raw_counts": raw_counts,
                "downsampled_counts": small_counts,
                "model_counts_including_padding": model_counts,
                "raw_nonzero_voxels": int(np.count_nonzero(raw.image)),
                "downsampled_nonzero_voxels": int(np.count_nonzero(small.image)),
                "normalized_nonzero_mean": mean,
                "normalized_nonzero_std": std,
                "padded_background_voxels": padding_voxels,
                "raw_voxel_volume_mm3": raw_voxel_mm3,
                "downsampled_voxel_volume_mm3": small_voxel_mm3,
                "raw_occupied_z_slices": {name: occupied_z_slices(raw.mask, k)
                                          for k, name in enumerate(CLASSES)},
                "downsampled_occupied_z_slices": {name: occupied_z_slices(small.mask, k)
                                                  for k, name in enumerate(CLASSES)},
                "physical_volume_ratio": {
                    name: (small_counts[name] * small_voxel_mm3) / (raw_counts[name] * raw_voxel_mm3)
                    for name in CLASSES if raw_counts[name] > 0
                },
            })
        except (ValueError, RuntimeError, MemoryError) as exc:
            row["errors"].append(f"{type(exc).__name__}: {exc}")
        row["elapsed_seconds"] = round(perf_counter() - started, 3)
        rows.append(row)
        print(f"[{index}/{len(selected)}] {case.identifier}: errors={len(row['errors'])}, elapsed={row['elapsed_seconds']}s", flush=True)

    result = {
        "scope": "Existing 20261017 Train preprocessing only; no crop, augmentation, sampler or training",
        "source_g1_sha256": AUDIT_SHA,
        "split_file_sha256": SPLIT_SHA,
        "preprocessing_code_sha256": DATA_SHA,
        "split_code_sha256": SPLITS_CODE_SHA,
        "software": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__, "nibabel": nib.__version__},
        "complete_train_audit": args.limit is None,
        "audited_cases": len(rows),
        "case_error_count": sum(bool(row["errors"]) for row in rows),
        "pipeline_pass": args.limit is None and not any(row["errors"] for row in rows),
        "cases": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Saved {len(rows)} case results to {output}; pipeline_pass={result['pipeline_pass']}")


if __name__ == "__main__":
    main()
