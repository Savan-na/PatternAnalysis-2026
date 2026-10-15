"""Protected V3 E12 inference. Preflight reads metadata only, never MRI arrays.

Real prediction requires the original A/B/C run tree, selected checkpoint and
selection receipt. Validation is the default; Test requires two explicit flags.
All metrics and panels remain on the shared resampled XYZ grid.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import sys
from time import perf_counter

import torch

from . import predict, validate as evaluation
from . import run_improved_v3_split as split
from .dataset import DEFAULT_DATASET_ROOT, discover_dataset
from .improved_modules import ImprovedUNet3D
from .train import configure_reproducibility


CHECKPOINT_SHA256 = "bb0d03c7a047ed187fadb5e1bf6ac1776448b351621764e9b6f0ff1fb67660d4"
RECEIPT_SHA256 = "fffe08a6c6502989e1ad033a0b78ea5117377dbad2de762791a3a895b2e189da"


def inspect_candidate(run_root, checkpoint, receipt_path):
    """Authenticate the selected E12 boundary and complete Validation lineage."""
    run_root = Path(run_root).expanduser().resolve(strict=True)
    checkpoint = Path(checkpoint).expanduser().resolve(strict=True)
    receipt_path = Path(receipt_path).expanduser().resolve(strict=True)
    expected_checkpoint = split.checkpoint_path(run_root, "B").resolve(strict=True)
    expected_receipt = (run_root / "C" / "selection.json").resolve(strict=True)
    if checkpoint != expected_checkpoint or receipt_path != expected_receipt:
        raise ValueError("V3 inputs must be the original B checkpoint and C receipt in the audited run tree")
    if split.v3.sha256(checkpoint) != CHECKPOINT_SHA256 or split.v3.sha256(receipt_path) != RECEIPT_SHA256:
        raise ValueError("V3 E12 checkpoint or selection receipt digest differs")
    audit = split.audit_complete(run_root, "C")
    if audit["status"] != "PASS":
        raise ValueError("V3 A/B/C lineage audit failed")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    cfg = payload.get("config", {})
    split.validate_config(cfg)
    result = payload.get("epoch_result", {})
    if (receipt != audit["reports"]["C"]["selection"]
            or receipt.get("epoch") != 12 or receipt.get("decision", {}).get("selected") is not True
            or receipt.get("checkpoint_sha256") != CHECKPOINT_SHA256
            or Path(receipt.get("checkpoint", "")).resolve() != checkpoint
            or receipt.get("config_sha256") != cfg.get("config_sha256")
            or receipt.get("validation_completed") is not True or receipt.get("test_evaluated") is not False
            or payload.get("schema_version") != 1 or payload.get("completed_epochs") != 12
            or payload.get("total_steps") != 1716 or result.get("epoch_completed") is not True
            or result.get("validation_pending") is not True or result.get("processed_steps") != 143
            or result.get("expected_steps") != 143
            or not payload.get("optimizer", {}).get("state")
            or not {"python", "numpy", "torch_cpu", "torch_cuda", "loader"} <= payload.get("rng", {}).keys()):
        raise ValueError("V3 E12 selection or checkpoint payload is incompatible")
    model = ImprovedUNet3D()
    model.load_state_dict(payload["model"], strict=True)
    if any(not bool(torch.isfinite(tensor).all()) for tensor in model.state_dict().values()):
        raise ValueError("V3 E12 model contains non-finite weights or buffers")
    metadata = {"experiment_id": split.IDENTITY, "completed_epochs": 12, "total_steps": 1716,
                "checkpoint_sha256": CHECKPOINT_SHA256, "selection_sha256": RECEIPT_SHA256,
                "config_sha256": cfg["config_sha256"], "lineage_audit": "PASS",
                "training_versions": cfg["protocol"]["parent_config"].get("versions")}
    if split.v3.sha256(checkpoint) != CHECKPOINT_SHA256 or split.v3.sha256(receipt_path) != RECEIPT_SHA256:
        raise ValueError("V3 inputs changed during inspection")
    return model.eval(), metadata


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--selection-receipt", type=Path, required=True)
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    p.add_argument("--split", choices=("validation", "test"), default="validation")
    p.add_argument("--allow-test", action="store_true")
    p.add_argument("--freeze-receipt", type=Path,
                   help="independently reviewed final-model and Validation-equivalence receipt for Test")
    scope = p.add_mutually_exclusive_group()
    scope.add_argument("--cases", nargs="+", type=evaluation._case, metavar="PATIENT_WeekN")
    scope.add_argument("--all-split", action="store_true")
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = p.parse_args(argv)
    if args.preflight_only:
        if (args.cases or args.all_split or args.output_dir or args.split != "validation"
                or args.allow_test or args.freeze_receipt):
            p.error("Preflight accepts only V3 evidence paths and optional dataset-root/device")
    elif not (args.cases or args.all_split) or args.output_dir is None:
        p.error("Prediction requires cases or --all-split and a new --output-dir")
    if (args.split == "test") != args.allow_test:
        p.error("Test requires --split test --allow-test; opt-in is invalid for Validation")
    if (args.split == "test") != bool(args.freeze_receipt):
        p.error("Test additionally requires --freeze-receipt; invalid for Validation")
    if args.cases and len(set(args.cases)) != len(args.cases):
        p.error("Duplicate case selection")
    return args


def inspect_freeze_receipt(path, candidate):
    """Enforce the documented final freeze boundary before any Test discovery."""
    receipt = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    required = {"candidate_checkpoint_sha256": candidate["checkpoint_sha256"],
                "selection_sha256": candidate["selection_sha256"],
                "config_sha256": candidate["config_sha256"],
                "adapter_sha256": split.v3.sha256(__file__),
                "validation_equivalence": "PASS", "formal_freeze": True,
                "test_evaluated_before_freeze": False}
    if any(receipt.get(key) != value for key, value in required.items()):
        raise ValueError("Final freeze receipt is missing, stale or incompatible")
    return receipt


def compare_full_validation(run_root, records):
    """Compare every case's confusion matrix with the authenticated V3 C run."""
    journals = list((Path(run_root) / "C").glob("events_*.jsonl"))
    if len(journals) != 1:
        raise ValueError("Expected one V3 C Validation journal")
    historical = [r for r in split.v3.audit_journal(journals[0]) if r.get("event") == "volume"]
    if len(historical) != 38 or len(records) != 38:
        return False
    def by_case(rows):
        result = {}
        for row in rows:
            key = (row.get("patient_id"), row.get("week"))
            if key in result or row.get("status") != "success":
                return None
            result[key] = row.get("confusion_counts")
        return result
    observed, expected = by_case(records), by_case(historical)
    return observed is not None and expected is not None and observed == expected


def main(argv=None):
    args = parse_args(argv)
    model, metadata = inspect_candidate(args.run_root, args.checkpoint, args.selection_receipt)
    if args.preflight_only:
        evaluation._emit({"event": "preflight", "status": "PASS", "candidate": metadata,
                          "arrays_decoded": 0, "inference_performed": False, "test_evaluated": False})
        return 0
    if args.split == "test":
        inspect_freeze_receipt(args.freeze_receipt, metadata)
    device = torch.device(args.device)
    if device.type == "cuda" and (not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available()):
        raise RuntimeError("CUDA prediction requires an allocated Slurm GPU session")
    output = predict.output_path(args.output_dir, args.dataset_root)
    configure_reproducibility(3710, device)
    pairs = predict.select_pairs(discover_dataset(args.dataset_root), args.split, args.cases,
                                 allow_test=args.allow_test)
    policies = evaluation._local_policies()
    model = model.to(device).eval()
    setup = {"event": "setup", "model": "ImprovedUNet3D V3 E12", "candidate": metadata,
             "split": args.split, "test_opt_in": args.allow_test, "selected_volumes": len(pairs),
             "all_split": args.all_split, "dataset_root": str(args.dataset_root.resolve()),
             "patch_shape_dhw": list(evaluation.PATCH_SHAPE), "stride": 32,
             "aggregation": "uniform_mean_softmax_probabilities_then_argmax", "policies": policies,
             "metric_grid": "shared resampled XYZ; no inverse resampling",
             "device": str(device), "output_dir": str(output)}
    output.mkdir()
    evaluation._emit(setup)
    records = []
    start = perf_counter()
    for pair in pairs:
        try:
            record = predict.process_pair(pair, model, device, policies, 32, output, args.split)
        except Exception as exc:
            record = {"event": "volume", "status": "failed", "case": f"{pair.patient_id}_Week{pair.week}",
                      "patient_id": pair.patient_id, "week": pair.week, "split": args.split,
                      "exception_type": type(exc).__name__, "exception_message": str(exc)}
        records.append(record)
        evaluation._emit(record)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    successes = [r for r in records if r["status"] == "success"]
    failures = [r for r in records if r["status"] != "success"]
    equivalence = None
    if args.split == "validation" and args.all_split and not failures:
        equivalence = compare_full_validation(args.run_root, successes)
    passed = not failures and len(successes) == len(pairs) and equivalence is not False
    summary = {"event": "summary", "status": "PASS" if passed else "FAIL", "split": args.split,
               "successful": len(successes), "failed": len(failures), "selected_volumes": len(pairs),
               "full_split_completed": bool(args.all_split and passed), "failed_cases": failures,
               "historical_v3_c_confusion_equivalence": equivalence,
               "session_seconds": perf_counter() - start, **evaluation.summarize(successes),
               "metric_scope": "successful cases only; both-absent classes excluded from macro averages"}
    with (output / "results.json").open("x", encoding="utf-8") as handle:
        json.dump({"schema_version": 1, "setup": setup, "volumes": records, "summary": summary},
                  handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    evaluation._emit(summary)
    return 0 if passed else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)
