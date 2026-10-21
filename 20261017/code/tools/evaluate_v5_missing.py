"""Evaluate frozen v5 C/L/S best checkpoints on the previously inspected Test cohort.

This is descriptive assessment only. Validation selects the checkpoints; Test
must not be used to change weights, epochs, or the evaluation protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "code/tools"))

EPOCHS = {"C": 50, "L": 45, "S": 45}
RECORD_HASHES = {
    "C": "72f8dae5a36f00779d01f58e224c02a2e7e01a0d1ae304e405bf6a96728bd791",
    "L": "91cdfd9e0fe2998948b513c1db14ac4ad782f659d3ab880d6571f98e974ed56c",
    "S": "c6895035ab4dc0c89a345fa9527e4d6bd903e32da9350d2b6aeccdd61449583b",
}
HISTORICAL_TEST_SHA256 = "21ce5e2eb53e3bf758478fd7c9fd95b887d82031b7a66ed77c7ef3168adc3e2f"
NAMES = ("Background", "Body", "Bones", "Bladder", "Rectum", "Prostate")
SOURCE_MAP = {
    "data_sha256": "hipmri_rewrite/data.py",
    "inference_sha256": "hipmri_rewrite/inference.py",
    "metrics_sha256": "hipmri_rewrite/metrics.py",
    "boundaries_sha256": "hipmri_rewrite/boundaries.py",
    "model_source_sha256": "hipmri_rewrite/models/improved_3d.py",
    "v3_source_sha256": "hipmri_rewrite/improved_v3.py",
    "model_common_sha256": "hipmri_rewrite/models/common.py",
    "model_package_sha256": "hipmri_rewrite/models/__init__.py",
    "splits_sha256": "hipmri_rewrite/splits.py",
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def verify_run(arm: str, torch):
    from hipmri_rewrite.improved_v3 import ImprovedUNet3DV3
    from hipmri_rewrite.splits import DEFAULT_SPLIT

    run = ROOT / f"logs/training/improved_3d_v5_{arm.lower()}_20261021"
    files = sorted(run.glob("*.json"))
    if len(files) != 52:
        raise ValueError(f"{arm}: expected 50 epoch JSON files plus config and manifest")
    aggregate = hashlib.sha256("".join(
        f"{p.name} {sha(p)}\n" for p in files).encode()).hexdigest()
    if aggregate != RECORD_HASHES[arm]:
        raise ValueError(f"{arm}: training record hash differs from verified Validation analysis")
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    records = [json.loads((run / f"epoch_{e:03d}.json").read_text(encoding="utf-8"))
               for e in range(1, 51)]
    selected = max((r for r in records if r["validation"] is not None),
                   key=lambda r: (r["validation"]["selection_score"], -r["epoch"]))
    epoch = EPOCHS[arm]
    if (manifest["kind"] != "v5_v3_single_factor_Train_Validation_run"
            or manifest["model"] != f"improved_3d_v5_{arm.lower()}"
            or manifest["split_sha256"] != sha(DEFAULT_SPLIT)
            or manifest["torch"] != str(torch.__version__)
            or records[-1]["best_epoch"] != epoch
            or selected["epoch"] != epoch):
        raise ValueError(f"{arm}: run identity, runtime, or best Validation epoch changed")
    for field, relative in SOURCE_MAP.items():
        if manifest[field] != sha(ROOT / "code" / relative):
            raise ValueError(f"{arm}: frozen source changed: {relative}")
    if sha(run / "config_frozen.json") != manifest["config_sha256"]:
        raise ValueError(f"{arm}: frozen configuration changed")
    immutable = run / f"epoch_{epoch:03d}.pt"
    if sha(run / "best.pt") != sha(immutable):
        raise ValueError(f"{arm}: best.pt differs from selected immutable checkpoint")
    checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    score = selected["validation"]["selection_score"]
    if (checkpoint["model"] != manifest["model"]
            or checkpoint["epoch"] != epoch
            or checkpoint["best_epoch"] != epoch
            or checkpoint["config_sha256"] != manifest["config_sha256"]
            or abs(checkpoint["best_score"] - score) > 1e-12):
        raise ValueError(f"{arm}: selected checkpoint metadata changed")
    model = ImprovedUNet3DV3()
    model.load_state_dict(checkpoint["model_state"], strict=True)
    del checkpoint
    receipt = {"epoch": epoch, "validation_score": score,
               "checkpoint_sha256": sha(immutable), "record_sha256": aggregate,
               "parameters": sum(p.numel() for p in model.parameters())}
    return model, receipt


def summarize(rows, np):
    from hipmri_rewrite.metrics import overlap

    patients = defaultdict(lambda: np.zeros((6, 6), dtype=np.int64))
    for row in rows:
        patients[row["patient"]] += np.asarray(row["confusion"], dtype=np.int64)
    patient_scores = {p: overlap(matrix) for p, matrix in patients.items()}
    classes = {}
    for k, name in enumerate(NAMES):
        classes[name] = {}
        for metric in ("dice", "iou", "precision", "recall"):
            valid = [v[metric][k] for v in patient_scores.values() if v[metric][k] is not None]
            classes[name][metric] = float(np.mean(valid)) if valid else None
            classes[name][metric + "_patient_count"] = len(valid)
    regions = {}
    for key in ("apex", "central", "base"):
        means = []
        for patient in patients:
            values = [r["prostate_regions"][key] for r in rows
                      if r["patient"] == patient and r["prostate_regions"] is not None]
            if values:
                means.append(float(np.mean(values)))
        regions[key] = float(np.mean(means)) if means else None
        regions[key + "_patient_count"] = len(means)
    surfaces = {}
    for name in NAMES[1:]:
        values = [r["surfaces"][name] for r in rows]
        good = [v["hd95_mm"] for v in values if v["status"] == "both_present"]
        surfaces[name] = {
            "median_hd95_mm": float(median(good)) if good else None,
            "max_hd95_mm": float(max(good)) if good else None,
            "status_counts": {status: sum(v["status"] == status for v in values)
                              for status in ("both_present", "one_empty", "both_empty")},
        }
    predicted_prostate = sum(int(np.asarray(r["confusion"]).sum(axis=0)[5]) for r in rows)
    oar = sum(sum(r["prostate_in_true_oar"].values()) for r in rows)
    small_organ = {p: float(np.mean([value for value in score["dice"][3:]
                                     if value is not None]))
                   for p, score in patient_scores.items()}
    return {
        "patient_equal_classes": classes,
        "patient_equal_small_organ_dice": float(np.mean(list(small_organ.values()))),
        "per_patient_small_organ_dice": small_organ,
        "patient_equal_prostate_regions": regions,
        "volume_surface_hd95": surfaces,
        "distant_background_to_body_voxels": sum(r["far_background_to_body_voxels"] for r in rows),
        "prostate_prediction_in_true_bladder_rectum_fraction": oar / predicted_prostate,
        "volumes_with_multiple_prostate_components": sum(
            r["geometry"]["Prostate"]["prediction"]["components_6"] > 1 for r in rows),
        "median_single_pass_inference_seconds": float(median(
            r["inference_seconds_single_pass"] for r in rows)),
        "max_single_pass_allocated_bytes": max(r["inference_peak_allocated_bytes"] for r in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    import numpy as np
    import torch
    from audit_all_models import diagnostics
    from hipmri_rewrite.data import downsample, load_case, zscore_nonzero
    from hipmri_rewrite.inference import predict_volume
    from hipmri_rewrite.metrics import confusion
    from hipmri_rewrite.splits import discover

    runs = {arm: verify_run(arm, torch) for arm in EPOCHS}
    if args.preflight_only:
        print(json.dumps({a: receipt for a, (_, receipt) in runs.items()}, indent=2))
        return
    if not os.environ.get("SLURM_JOB_ID") or socket.gethostname().startswith("login"):
        parser.error("Full evaluation requires an allocated Rangpur GPU compute job")
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA device allocated")
    output = args.output.resolve()
    allowed = (ROOT / "results/experiments/improved_3d_v5").resolve()
    if not output.is_relative_to(allowed) or output.exists():
        parser.error("Choose a new output directory within improved_3d_v5 results")
    cases = discover(args.dataset_root)
    if (len(cases["validation"]) != 38 or len(cases["test"]) != 30
            or len({c.patient for c in cases["test"]}) != 6):
        raise ValueError("Frozen Validation/Test cohort count changed")
    test = cases["test"]
    historical_path = ROOT / "results/tables/comparisons/g6_test_3d.json"
    if sha(historical_path) != HISTORICAL_TEST_SHA256:
        raise ValueError("Historical Test receipt differs from the frozen seven-model audit")
    historical = json.loads(historical_path.read_text(encoding="utf-8"))
    expected_hashes = {row["case"]: row for row in historical["arms"]["standard_3d"]["cases"]}
    case_hashes = {}
    for case in test:
        actual = {"mri_sha256": sha(case.mri), "mask_sha256": sha(case.mask)}
        if any(actual[key] != expected_hashes[case.identifier][key] for key in actual):
            raise ValueError(f"{case.identifier}: Test MRI/mask differs from historical evaluation")
        case_hashes[case.identifier] = actual
    reference_case = next(c for c in cases["validation"] if c.identifier == "K008_Week0")
    reference = zscore_nonzero(downsample(load_case(reference_case)))
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    output.mkdir(parents=True)
    result = {"kind": "v5_frozen_best_descriptive_Test_completion",
              "test_scope": "Previously inspected Test cohort; no checkpoint selection or tuning",
              "case_count": 30, "patient_count": 6,
              "gpu": torch.cuda.get_device_name(0), "job_id": os.environ["SLURM_JOB_ID"],
              "script_sha256": sha(Path(__file__)), "arms": {}, "complete": False}
    for arm in EPOCHS:
        model, receipt = runs.pop(arm)
        model = model.cuda().eval()
        validation_record = json.loads((ROOT / f"logs/training/improved_3d_v5_{arm.lower()}_20261021"
                                        / f"epoch_{EPOCHS[arm]:03d}.json").read_text(encoding="utf-8"))
        expected = next(row["confusion"] for row in validation_record["validation"]["cases"]
                        if row["case"] == "K008_Week0")
        replay = predict_volume(model, reference, mode="3d", device="cuda")
        if confusion(reference.mask, replay).tolist() != expected:
            raise ValueError(f"{arm}: Validation K008_Week0 confusion failed to replay")
        receipt["validation_replay_case"] = "K008_Week0"
        for _ in range(3):
            predict_volume(model, reference, mode="3d", device="cuda")
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        benchmark = []
        for _ in range(5):
            torch.cuda.synchronize()
            start = perf_counter()
            predict_volume(model, reference, mode="3d", device="cuda")
            torch.cuda.synchronize()
            benchmark.append(perf_counter() - start)
        bench_peak = int(torch.cuda.max_memory_allocated())
        rows = []
        for index, case in enumerate(test, 1):
            volume = zscore_nonzero(downsample(load_case(case)))
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = perf_counter()
            prediction = predict_volume(model, volume, mode="3d", device="cuda")
            torch.cuda.synchronize()
            seconds = perf_counter() - start
            rows.append({"case": case.identifier, "patient": case.patient,
                         **case_hashes[case.identifier],
                         "confusion": confusion(volume.mask, prediction).tolist(),
                         "inference_seconds_single_pass": seconds,
                         "inference_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                         **diagnostics(volume, prediction)})
            print(f"{arm} Test {index}/30: {case.identifier}", flush=True)
        result["arms"][arm] = {
            "selected": receipt,
            "benchmark": {"case": "K008_Week0", "warmups": 3, "repeats": 5,
                          "seconds": benchmark, "median_seconds": float(median(benchmark)),
                          "peak_allocated_bytes": bench_peak,
                          "definition": "Prepared whole-volume prediction; excludes disk/preprocessing/metrics"},
            "summary": summarize(rows, np), "cases": rows,
        }
        write_json(output / "metrics_partial.json", result)
        summary = result["arms"][arm]["summary"]
        print(f"{arm} Test summary: small-organ Dice={summary['patient_equal_small_organ_dice']:.6f}, "
              f"Prostate Dice={summary['patient_equal_classes']['Prostate']['dice']:.6f}, "
              f"far Body FP={summary['distant_background_to_body_voxels']}, "
              f"median inference={summary['median_single_pass_inference_seconds']:.4f}s",
              flush=True)
        del model
        torch.cuda.empty_cache()
    result["complete"] = True
    write_json(output / "metrics.json", result)
    print(f"Saved complete v5 Test assessment: {output / 'metrics.json'}", flush=True)
    print(f"metrics.json SHA-256: {sha(output / 'metrics.json')}", flush=True)


if __name__ == "__main__":
    main()
