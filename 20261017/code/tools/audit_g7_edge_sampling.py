"""Replay G5 Train schedules to audit a 9-uniform/1-corner candidate on CPU.

Reads Train MRI/masks only, verifies frozen G1/G3 and baseline exposure, and
records exact paired model-input crops. It never trains or changes weights.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import socket
import sys
from collections import Counter, defaultdict
from pathlib import Path

import nibabel
import numpy as np
import scipy

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "code"))
from hipmri_rewrite.data import downsample, load_case, zscore_nonzero  # noqa: E402
from hipmri_rewrite.sampling import (CLASS_NAMES, ExposureRecorder,  # noqa: E402
                                     crop_for_3d_model, crop_origin, epoch_cases)
from hipmri_rewrite.sampling_edges import EdgePatchSelector, boundary_hits  # noqa: E402
from hipmri_rewrite.splits import DEFAULT_SPLIT, discover  # noqa: E402

G1 = PROJECT / "logs/audits/train_full_20261018_job671208.json"
G3 = PROJECT / "logs/audits/g3_existing_pipeline_full_20261018_job671892.json"
G1_SHA = "6a4c674564a0364f5f56730cbc31e67b3ff9a8ffeae7d9d3dd18b198bbd0e1a1"
G3_SHA = "319973a8759265521870f831e7b17eb7249f4130b6a9e9634fe993f22bf687e9"
POLICIES = ("original_uniform", "nine_uniform_one_edge")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path,
                        default=PROJECT / "code/config/g7_edge_sampling_candidate.json")
    parser.add_argument("--reference-run", type=Path,
                        default=PROJECT / "logs/training/g5_standard_20261018")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--limit-patients", type=int)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or socket.gethostname().startswith("login"):
        parser.error("Run on an allocated Rangpur CPU node")
    output = args.output_dir.resolve()
    if not output.is_relative_to(PROJECT / "logs/audits") or output.exists():
        parser.error("Use a new output directory inside logs/audits")
    if (not 1 <= args.epochs <= 50
            or (args.limit_patients is not None and not 1 <= args.limit_patients <= 26)):
        parser.error("Invalid epoch or patient limit")
    candidate = json.loads(args.config.read_text(encoding="utf-8"))
    frozen_path = PROJECT / "code/config/g5_controlled_training_frozen.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    seed = candidate["seed"]
    patch = tuple(candidate["patch_xyz"])
    if (seed != frozen["seed"] or patch != tuple(frozen["preprocessing"]["train_patch_xyz"])
            or candidate["preserved_uniform_draws"] != 9
            or candidate["patches_per_patient_epoch"] != 10
            or candidate["epochs"] != 50
            or frozen["sampling"]["patch_rule"] != "uniform"
            or tuple(frozen["preprocessing"]["downsample_factor_xyz"]) != (2, 2, 2)
            or candidate["edge"]["width"] != 8):
        raise ValueError("Candidate must implement the declared 9+1 G5 comparison")
    if sha(G1) != G1_SHA or sha(G3) != G3_SHA:
        raise ValueError("G1/G3 source audit differs from the verified full result")
    g1 = json.loads(G1.read_text(encoding="utf-8"))
    g3 = json.loads(G3.read_text(encoding="utf-8"))
    if not (g1["mechanical_pass"] and g1["complete_train_audit"]
            and g3["pipeline_pass"] and g3["complete_train_audit"]):
        raise ValueError("Full G1/G3 audits are required")
    raw = {row["case"]: row for row in g1["cases"]}
    processed = {row["case"]: row for row in g3["cases"]}
    cases = discover(args.dataset_root)["train"]
    if (len(cases) != 143 or len({c.patient for c in cases}) != 26
            or {c.identifier for c in cases} != set(raw) or set(raw) != set(processed)):
        raise ValueError("Train case inventory differs from G1/G3")
    manifest_path = args.reference_run / "manifest.json"
    reference_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (reference_manifest["model"] != "standard_3d"
            or reference_manifest["max_epochs"] != 50
            or reference_manifest["config_sha256"] != sha(frozen_path)
            or reference_manifest["split_sha256"] != sha(DEFAULT_SPLIT)
            or reference_manifest["dataset_root"] != str(args.dataset_root.resolve())):
        raise ValueError("Reference is not the frozen completed G5 Standard run")
    for key, name in (("sampling_sha256", "hipmri_rewrite/sampling.py"),
                      ("data_sha256", "hipmri_rewrite/data.py"),
                      ("splits_sha256", "hipmri_rewrite/splits.py")):
        if reference_manifest[key] != sha(PROJECT / "code" / name):
            raise ValueError(f"Frozen source changed: {name}")
    chosen_patients = sorted({case.patient for case in cases})
    if args.limit_patients:
        chosen_patients = chosen_patients[:args.limit_patients]
    selected_patients = set(chosen_patients)
    plan = defaultdict(list)
    reference = {}
    reference_hashes = {}
    for epoch in range(args.epochs):
        path = args.reference_run / f"epoch_{epoch + 1:03d}.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        selected = epoch_cases(cases, epoch, seed)
        if record["epoch"] != epoch + 1 or record["selected_cases"] != [c.identifier for c in selected]:
            raise ValueError(f"Epoch {epoch + 1}: patient schedule differs from G5")
        reference[epoch] = record["train"]["exposure"]
        reference_hashes[path.name] = sha(path)
        for case in selected:
            if case.patient in selected_patients:
                plan[case.identifier].append(epoch)
    recorders = {policy: ExposureRecorder() for policy in POLICIES}
    by_epoch = {epoch: {policy: ExposureRecorder() for policy in POLICIES}
                for epoch in range(args.epochs)}
    geometry = {policy: Counter() for policy in POLICIES}
    rows = []
    case_receipts = []
    selections = Counter()
    corners = Counter()
    for index, case in enumerate(sorted((c for c in cases if c.identifier in plan),
                                        key=lambda c: c.identifier), 1):
        if (sha(case.mri) != raw[case.identifier]["mri_sha256"]
                or sha(case.mask) != raw[case.identifier]["mask_sha256"]):
            raise ValueError(f"Raw Train source changed: {case.identifier}")
        volume = zscore_nonzero(downsample(load_case(case)))
        counts = np.bincount(volume.mask.ravel(), minlength=6)
        if any(int(counts[i]) != processed[case.identifier]["downsampled_counts"][name]
               for i, name in enumerate(CLASS_NAMES)):
            raise ValueError(f"{case.identifier}: mask differs from G3 preprocessing")
        selector = EdgePatchSelector(volume, patch, **candidate["edge"])
        case_receipts.append({"case": case.identifier, "patient": case.patient,
                              "shape_xyz": list(volume.mask.shape),
                              "spacing_mm": volume.spacing_mm,
                              "scheduled_epochs": [e + 1 for e in plan[case.identifier]],
                              "eligible_negative_voxels": int(selector.hard_negative.sum())})

        def add(policy, epoch, draw, origin, image, label, rule, attempts=0, corner=None):
            if image.shape != (1, *patch[::-1]) or label.shape != patch[::-1]:
                raise ValueError("Unexpected model-ready tensor shape")
            counts = recorders[policy].add(label, case.patient)
            by_epoch[epoch][policy].add(label, case.patient)
            nonzero, hard_negative = selector.measurements(origin)
            flags = {}
            geometry[policy]["patches"] += 1
            for width in (2, 8):
                face, corner_hit = boundary_hits(origin, volume.mask.shape, patch, width)
                flags[f"outer_{width}_any_face"] = int(face)
                flags[f"outer_{width}_any_corner"] = int(corner_hit)
                geometry[policy].update({key: value for key, value in flags.items()
                                         if key.startswith(f"outer_{width}_")})
            rows.append({"policy": policy, "epoch": epoch + 1, "patient": case.patient,
                         "case": case.identifier, "draw": draw, "rule": rule,
                         "origin_x": origin[0], "origin_y": origin[1], "origin_z": origin[2],
                         "attempts": attempts, "corner_bits": corner,
                         "nonzero_fraction": nonzero,
                         "outer_shell_distant_nonzero_background_voxels": hard_negative,
                         "image_min": float(image.min()), "image_max": float(image.max()),
                         **flags, **{f"{name.lower()}_voxels": amount
                                    for name, amount in counts.items()}})

        for epoch in plan[case.identifier]:
            for draw in range(10):
                origin = crop_origin(volume.mask, patch, "uniform", seed=seed,
                                     epoch=epoch, case_id=case.identifier, draw=draw)
                image, label = crop_for_3d_model(volume, origin, patch)
                add(POLICIES[0], epoch, draw, origin, image, label, "uniform")
                if draw < 9:
                    add(POLICIES[1], epoch, draw, origin, image, label, "shared_uniform")
                else:
                    selected = selector.select(seed=seed, epoch=epoch, draw=draw)
                    rule = "accepted_edge" if selected.accepted else "uniform_fallback"
                    selections[rule] += 1
                    if selected.accepted:
                        corners[selected.corner] += 1
                        if not boundary_hits(selected.origin, volume.mask.shape, patch, 8)[1]:
                            raise ValueError("Accepted corner crop missed the outer 8 slabs")
                    edge_image, edge_label = crop_for_3d_model(volume, selected.origin, patch)
                    add(POLICIES[1], epoch, draw, selected.origin, edge_image, edge_label,
                        rule, selected.attempts, selected.corner)
        print(f"[{index}/{len(plan)}] {case.identifier}: "
              f"{len(plan[case.identifier])} paired patient visits", flush=True)
    for epoch in range(args.epochs):
        actual = by_epoch[epoch][POLICIES[0]].summary()
        expected = reference[epoch]
        if args.limit_patients is None:
            if actual != expected:
                raise ValueError(f"Epoch {epoch + 1}: replay does not reproduce actual G5 exposure")
        elif any(actual["per_patient"][p] != expected["per_patient"][p]
                 for p in chosen_patients):
            raise ValueError("Smoke replay differs from original patient exposure")
    full = (args.limit_patients is None and args.epochs == 50 and len(plan) == 143)
    expected_patches = len(chosen_patients) * args.epochs * 10
    summary = {}
    for policy in POLICIES:
        exposure = recorders[policy].summary()
        if (exposure["patches"] != expected_patches
                or any(v["patches"] != args.epochs * 10 for v in exposure["per_patient"].values())):
            raise ValueError("Patient patch counts are not balanced")
        nonzero = np.asarray([r["nonzero_fraction"] for r in rows if r["policy"] == policy])
        summary[policy] = {"exposure": exposure, "geometry_counts": dict(geometry[policy]),
                           "geometry_rates": {k: geometry[policy][k] / expected_patches
                                              for k in candidate["baseline_geometry_counts"]
                                              if k != "patches"},
                           "nonzero_fraction": {"min": float(nonzero.min()),
                                                "median": float(np.median(nonzero)),
                                                "p05": float(np.quantile(nonzero, .05))}}
    if full and any(geometry[POLICIES[0]][k] != v
                    for k, v in candidate["baseline_geometry_counts"].items()):
        raise ValueError("Full original geometry differs from verified G5 reconstruction")
    old = summary[POLICIES[0]]["exposure"]["per_class"]
    new = summary[POLICIES[1]]["exposure"]["per_class"]
    ratios = {organ: {"hit_rate_ratio": new[organ]["patch_hit_rate"] / old[organ]["patch_hit_rate"],
                      "voxel_share_ratio": new[organ]["voxel_share"] / old[organ]["voxel_share"]}
              for organ in ("Bladder", "Rectum", "Prostate")}
    thresholds = candidate["training_gate"]
    gates = {"outer_8_corner_rate": summary[POLICIES[1]]["geometry_rates"]["outer_8_any_corner"]
             >= thresholds["min_outer_8_corner_rate"],
             "small_organ_hit_rates": all(v["hit_rate_ratio"] >= thresholds["min_small_organ_hit_ratio"]
                                          for v in ratios.values()),
             "small_organ_voxel_shares": all(v["voxel_share_ratio"] >= thresholds["min_small_organ_voxel_share_ratio"]
                                            for v in ratios.values())}
    output.mkdir(parents=True)
    with (output / "patches.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {"kind": "G7_Train_only_edge_sampling_CPU_dry_run", "job_id": os.environ["SLURM_JOB_ID"],
              "scope": "Train MRI/Mask only; no training, Validation/Test arrays or weights read",
              "epochs": args.epochs, "patients": len(chosen_patients), "unique_train_cases": len(plan),
              "full_train": full, "patches_per_policy": expected_patches, "smoke_pass": True,
              "eligible_for_training": full and all(gates.values()), "gates": gates,
              "small_organ_ratios": ratios, "edge_selections": dict(selections),
              "accepted_corner_counts": dict(corners), "summary": summary,
              "case_receipts": case_receipts,
              "epoch_exposure": {str(e + 1): {p: v.summary() for p, v in group.items()}
                                 for e, group in by_epoch.items()},
              "sources": {"candidate_config": candidate, "candidate_config_sha256": sha(args.config),
                          "frozen_config_sha256": sha(frozen_path), "g1_sha256": G1_SHA, "g3_sha256": G3_SHA,
                          "split_sha256": sha(DEFAULT_SPLIT), "reference_manifest_sha256": sha(manifest_path),
                          "reference_epoch_sha256": reference_hashes,
                          "script_sha256": sha(Path(__file__)),
                          "edge_sampling_sha256": sha(PROJECT / "code/hipmri_rewrite/sampling_edges.py"),
                          "frozen_sampling_sha256": sha(PROJECT / "code/hipmri_rewrite/sampling.py"),
                          "data_sha256": sha(PROJECT / "code/hipmri_rewrite/data.py")},
              "environment": {"python": sys.version, "numpy": np.__version__,
                              "scipy": scipy.__version__, "nibabel": nibabel.__version__},
              "patches_csv_sha256": sha(output / "patches.csv")}
    (output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(rows)} paired patch records; full_train={full}; "
          f"eligible_for_training={result['eligible_for_training']}")
    print("gates", gates, "edge_selections", dict(selections))
    print(output)


if __name__ == "__main__":
    main()
