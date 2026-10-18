"""Audit optional 2D Train slice/crop exposure on a Rangpur CPU node."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import socket
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "code"))
G1_SHA = "6a4c674564a0364f5f56730cbc31e67b3ff9a8ffeae7d9d3dd18b198bbd0e1a1"
G3_SHA = "319973a8759265521870f831e7b17eb7249f4130b6a9e9634fe993f22bf687e9"
CLASS_NAMES = ("Background", "Body", "Bones", "Bladder", "Rectum", "Prostate")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--limit-patients", type=int, default=None)
    parser.add_argument("--g1", type=Path,
                        default=PROJECT / "logs/audits/train_full_20261018_job671208.json")
    parser.add_argument("--g3", type=Path,
                        default=PROJECT / "logs/audits/g3_existing_pipeline_full_20261018_job671892.json")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or socket.gethostname().startswith("login"):
        parser.error("Run inside an allocated Rangpur CPU compute job")
    output = args.output_dir.resolve()
    if not output.is_relative_to(PROJECT) or output.exists():
        parser.error("Output must be a new directory within 20261017")
    if args.epoch < 0 or (args.limit_patients is not None and not 1 <= args.limit_patients <= 26):
        parser.error("Invalid epoch or patient limit")
    if sha256_file(args.g1) != G1_SHA or sha256_file(args.g3) != G3_SHA:
        raise ValueError("Verified G1/G3 Train audit changed")

    import numpy as np

    from hipmri_rewrite.data import downsample, load_case, zscore_nonzero
    from hipmri_rewrite.sampling import ExposureRecorder, epoch_cases
    from hipmri_rewrite.sampling_2d import (ORGAN_AWARE_2D, PATCH_XY, UNIFORM_2D,
                                           crop_for_2d_model, slice_crop_origin)
    from hipmri_rewrite.splits import discover

    g1 = json.loads(args.g1.read_text(encoding="utf-8"))
    g3 = json.loads(args.g3.read_text(encoding="utf-8"))
    if not (g1["complete_train_audit"] and g1["mechanical_pass"]
            and g3["complete_train_audit"] and g3["pipeline_pass"]):
        raise ValueError("Source Train audits did not pass")
    raw = {row["case"]: row for row in g1["cases"]}
    processed = {row["case"]: row for row in g3["cases"]}
    cases = discover(args.dataset_root)["train"]
    if len(cases) != 143 or {c.identifier for c in cases} != set(raw) or set(raw) != set(processed):
        raise ValueError("Train cohort differs from G1/G3")
    selected = epoch_cases(cases, args.epoch, seed=371017)
    if args.limit_patients:
        selected = selected[:args.limit_patients]
    policies = {"patient_uniform_2d": UNIFORM_2D,
                "patient_organ_aware_2d": ORGAN_AWARE_2D}
    exposure = {name: ExposureRecorder() for name in policies}
    rows = []
    for index, case in enumerate(selected, 1):
        if (sha256_file(case.mri) != raw[case.identifier]["mri_sha256"]
                or sha256_file(case.mask) != raw[case.identifier]["mask_sha256"]):
            raise ValueError(f"Source pair changed: {case.identifier}")
        volume = zscore_nonzero(downsample(load_case(case)))
        counts = np.bincount(volume.mask.ravel(), minlength=6)
        if any(int(counts[i]) != processed[case.identifier]["downsampled_counts"][name]
               for i, name in enumerate(CLASS_NAMES)):
            raise ValueError(f"Processed mask differs from G3: {case.identifier}")
        for policy, schedule in policies.items():
            for draw, target in enumerate(schedule):
                origin = slice_crop_origin(volume.mask, target, seed=371017,
                                           epoch=args.epoch, case_id=case.identifier, draw=draw)
                image, mask = crop_for_2d_model(volume, origin)
                if image.shape != (1, 64, 64) or mask.shape != (64, 64):
                    raise ValueError("Unexpected model-ready 2D crop")
                delivered = exposure[policy].add(mask[None], case.patient)
                if target != "uniform" and delivered[target] == 0:
                    raise ValueError(f"2D target not exposed: {case.identifier} {target}")
                rows.append({"epoch": args.epoch, "policy": policy, "patient": case.patient,
                             "case": case.identifier, "draw": draw, "target": target,
                             "origin_x": origin[0], "origin_y": origin[1], "slice_z": origin[2],
                             **{f"{name.lower()}_voxels": n for name, n in delivered.items()}})
        print(f"[{index}/{len(selected)}] {case.identifier}: 20 paired 2D crops", flush=True)
    output.mkdir(parents=True)
    with (output / "patches.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = {"scope": "optional 2D Train-only CPU slice exposure audit; no training",
                "complete_patient_epoch": args.limit_patients is None,
                "epoch": args.epoch, "seed": 371017, "patients": len(selected),
                "patches_per_policy": len(selected) * 10,
                "selected_cases": [case.identifier for case in selected],
                "patch_xy": PATCH_XY, "g1_sha256": G1_SHA, "g3_sha256": G3_SHA,
                "policies": policies,
                "script_sha256": sha256_file(Path(__file__)),
                "sampling_2d_sha256": sha256_file(PROJECT / "code/hipmri_rewrite/sampling_2d.py"),
                "data_sha256": sha256_file(PROJECT / "code/hipmri_rewrite/data.py"),
                "summary": {name: recorder.summary() for name, recorder in exposure.items()}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                                          encoding="utf-8")
    print(f"Saved {len(rows)} 2D crop records to {output}; no model was trained")


if __name__ == "__main__":
    main()
