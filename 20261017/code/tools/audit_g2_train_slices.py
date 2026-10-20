"""Compute Train-only axial class coverage and nonzero MRI histograms on Rangpur CPU."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "code"))
from hipmri_rewrite.splits import discover  # noqa: E402


CLASSES = ("Background", "Body", "Bones", "Bladder", "Rectum", "Prostate")
AUDIT_SHA = "6a4c674564a0364f5f56730cbc31e67b3ff9a8ffeae7d9d3dd18b198bbd0e1a1"
SPLIT_SHA = "81531ec7cb84ef58c41393ff23e1105e7cf03aa97828fb12853ce8624fcb8900"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--audit", type=Path, default=PROJECT / "logs/audits/train_full_20261018_job671208.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "results/data_analysis/training_distribution")
    parser.add_argument("--limit", type=int, default=None, help="Smoke run only; never use as the full G2 result")
    args = parser.parse_args()

    import nibabel as nib
    import numpy as np

    if hashlib.sha256(args.audit.read_bytes()).hexdigest() != AUDIT_SHA:
        raise ValueError("Complete G1 audit hash mismatch")
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    if not (audit["complete_train_audit"] and audit["mechanical_pass"]
            and audit["split_file_sha256"] == SPLIT_SHA and len(audit["cases"]) == 143):
        raise ValueError("G1 Train audit is incomplete")
    checked = {row["case"]: row for row in audit["cases"]}
    cohort = discover(args.dataset_root)["train"]
    if len(cohort) != 143 or {case.identifier for case in cohort} != set(checked):
        raise ValueError("Current Train cohort differs from the G1 audit")
    if args.limit is not None and not 1 <= args.limit <= 143:
        parser.error("--limit must be between 1 and 143")
    subset = cohort[:args.limit] if args.limit else cohort

    edges = np.arange(0, 1201, 10, dtype=np.float64)
    slice_rows: list[dict] = []
    histogram_rows: list[dict] = []
    for index, case in enumerate(subset, 1):
        ident = case.identifier
        prior = checked[ident]
        if (sha256_file(case.mri) != prior["mri_sha256"]
                or sha256_file(case.mask) != prior["mask_sha256"]):
            raise ValueError(f"Source file changed since G1: {ident}")
        mri = nib.load(str(case.mri))
        mask = nib.load(str(case.mask))
        if (list(mri.shape) != prior["mri_shape"] or list(mask.shape) != prior["mask_shape"]
                or not np.array_equal(mri.affine, mask.affine)):
            raise ValueError(f"Geometry changed: {ident}")
        image = np.asarray(mri.dataobj, dtype=np.float32)
        labels = np.asarray(mask.dataobj, dtype=np.int16)
        if not np.all(np.isfinite(image)) or not np.isin(labels, range(6)).all():
            raise ValueError(f"Invalid decoded values: {ident}")
        global_counts = np.bincount(labels.ravel(), minlength=6)
        if any(int(global_counts[k]) != prior["class_counts"][name]
               for k, name in enumerate(CLASSES)):
            raise ValueError(f"Class counts changed: {ident}")

        for z in range(labels.shape[2]):
            slice_counts = np.bincount(labels[:, :, z].ravel(), minlength=6)
            row = {"case": ident, "patient": case.patient, "axial_slice_index_z": z,
                   "slice_center_superior_mm": float((mri.affine @ [
                       (labels.shape[0] - 1) / 2, (labels.shape[1] - 1) / 2, z, 1
                   ])[2]),
                   "slice_voxels": int(slice_counts.sum())}
            for k, name in enumerate(CLASSES):
                row[f"{name.lower()}_voxels"] = int(slice_counts[k])
                row[f"{name.lower()}_present"] = int(slice_counts[k] > 0)
            slice_rows.append(row)

        positive = image[image > 0]
        bins, _ = np.histogram(positive, bins=edges)
        if int(bins.sum()) != len(positive):
            raise ValueError(f"MRI intensity outside fixed 0–1200 histogram range: {ident}")
        for j, count in enumerate(bins):
            histogram_rows.append({"case": ident, "patient": case.patient,
                                   "lower_inclusive": int(edges[j]),
                                   "upper_exclusive": int(edges[j + 1]),
                                   "positive_voxels": int(count)})
        print(f"[{index}/{len(subset)}] {ident}: {labels.shape[2]} slices", flush=True)

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "per_slice_class_coverage.csv", slice_rows, list(slice_rows[0]))
    write_csv(output / "per_volume_intensity_histogram.csv", histogram_rows, list(histogram_rows[0]))
    print(f"Saved {len(slice_rows)} Train axial slices and {len(histogram_rows)} intensity bins to {output}")
    print("Full G2 detail:", args.limit is None)


if __name__ == "__main__":
    main()
