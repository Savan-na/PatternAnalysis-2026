"""One-time held-out Test evaluation of the two selected G5 3D checkpoints.

Requires both bounded continuations to have completed epoch 50. Checkpoint
selection uses the frozen Validation score; Test labels never select a model.
Only derived JSON leaves this process, not medical-image arrays.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import socket
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median
from time import perf_counter

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "code"))

CONFIG_SHA = "e5bd98a99213f4fe6cf6e3e8c6db3b95d5ef334a143a91444426d8ea019923ce"
LABELS = ("Background", "Body", "Bones", "Bladder", "Rectum", "Prostate")
ARMS = (("standard_3d", "g5_standard_20261018", "baseline_3d.py", "672954"),
        ("improved_3d", "g5_improved_20261018", "improved_3d.py", "672955"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selected_model(root: Path, identity: tuple[str, str, str, str], dataset_root: Path, torch):
    from hipmri_rewrite.models import ImprovedUNet3D, UNet3D
    from hipmri_rewrite.splits import DEFAULT_SPLIT

    model_name, folder, source, initial_job = identity
    run = root / folder
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    final_path = run / "epoch_050.json"
    final = json.loads(final_path.read_text(encoding="utf-8"))
    if (manifest["kind"] != "G5_frozen_Train_Validation_cohort_run"
            or manifest["model"] != model_name
            or manifest["job_id_initial"] != initial_job
            or manifest["max_epochs"] != 50
            or manifest["config_sha256"] != CONFIG_SHA
            or manifest["dataset_root"] != str(dataset_root.resolve())
            or sha256(PROJECT / "code/config/g5_controlled_training_frozen.json") != CONFIG_SHA
            or manifest["split_sha256"] != sha256(DEFAULT_SPLIT)
            or final["epoch"] != 50 or final["validation"] is None):
        raise ValueError(f"{run}: frozen 50-epoch run is incomplete or changed")
    source_files = {
        "trainer_sha256": "tools/train_g5_cohort.py",
        "sampling_sha256": "hipmri_rewrite/sampling.py",
        "data_sha256": "hipmri_rewrite/data.py",
        "inference_sha256": "hipmri_rewrite/inference.py",
        "loss_sha256": "hipmri_rewrite/losses.py",
        "metrics_sha256": "hipmri_rewrite/metrics.py",
        "selection_sha256": "hipmri_rewrite/training.py",
        "splits_sha256": "hipmri_rewrite/splits.py",
        "package_sha256": "hipmri_rewrite/__init__.py",
        "model_source_sha256": f"hipmri_rewrite/models/{source}",
        "model_common_sha256": "hipmri_rewrite/models/common.py",
        "model_package_sha256": "hipmri_rewrite/models/__init__.py",
    }
    for field, relative in source_files.items():
        if manifest[field] != sha256(PROJECT / "code" / relative):
            raise ValueError(f"{run}: {relative} changed after training")
    checkpoint_path = run / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    epoch = checkpoint["epoch"]
    chosen_path = run / f"epoch_{epoch:03d}.json"
    chosen = json.loads(chosen_path.read_text(encoding="utf-8"))
    if sha256(checkpoint_path) != sha256(run / f"epoch_{epoch:03d}.pt"):
        raise ValueError(f"{run}: best.pt differs from immutable epoch checkpoint")
    if (checkpoint["model"] != model_name
            or checkpoint["config_sha256"] != CONFIG_SHA
            or checkpoint["best_epoch"] != epoch or epoch % 5 or not 5 <= epoch <= 50
            or final["best_epoch"] != epoch
            or chosen["best_epoch"] != epoch
            or chosen["validation"] is None
            or not math.isclose(checkpoint["best_score"], final["best_score"], abs_tol=1e-12)
            or not math.isclose(checkpoint["best_score"], chosen["validation"]["selection_score"], abs_tol=1e-12)):
        raise ValueError(f"{run}: best checkpoint and final Validation records disagree")
    constructor = UNet3D if model_name == "standard_3d" else ImprovedUNet3D
    model = constructor().to("cuda").eval()
    model.load_state_dict(checkpoint["model_state"], strict=True)
    receipt = {"model": model_name, "initial_job": initial_job,
               "selected_epoch": epoch, "validation_selection_score": checkpoint["best_score"],
               "checkpoint_sha256": sha256(checkpoint_path),
               "checkpoint_bytes": checkpoint_path.stat().st_size,
               "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
               "manifest_sha256": sha256(manifest_path),
               "final_record_sha256": sha256(final_path),
               "selected_record_sha256": sha256(chosen_path)}
    return model, receipt


def boundary_summary(rows: list[dict]) -> dict:
    result = {}
    for label in LABELS[1:]:
        values = [row["boundary"][label]["hd95_mm"] for row in rows
                  if row["boundary"][label]["status"] == "both_present"]
        status = {state: sum(row["boundary"][label]["status"] == state for row in rows)
                  for state in ("both_present", "one_empty", "both_empty")}
        result[label] = {"median_hd95_mm": median(values) if values else None,
                         "status_counts": status}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or socket.gethostname().startswith("login"):
        parser.error("Use an allocated Rangpur GPU compute job")
    output = args.output.resolve()
    if not output.is_relative_to(PROJECT / "results/tables/comparisons"):
        parser.error("Output must be under results/tables/comparisons")
    if output.exists():
        parser.error("Held-out Test output already exists; do not rerun or overwrite it")

    import numpy as np
    import torch

    from hipmri_rewrite.boundaries import (prostate_axial_regions, slice_dice,
                                           symmetric_surface_distances)
    from hipmri_rewrite.data import downsample, load_case, zscore_nonzero
    from hipmri_rewrite.inference import predict_volume
    from hipmri_rewrite.metrics import confusion, overlap
    from hipmri_rewrite.splits import discover
    from hipmri_rewrite.training import patient_small_organ_score

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA device allocated")
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    cases = discover(args.dataset_root)["test"]
    if len(cases) != 30 or len({case.patient for case in cases}) != 6:
        raise ValueError("Frozen held-out Test cohort differs")
    run_root = PROJECT / "logs/training"
    selected = {identity[0]: selected_model(run_root, identity, args.dataset_root, torch)
                for identity in ARMS}
    result = {"kind": "G6_one_time_held_out_Test", "split": "test",
              "case_count": 30, "patient_count": 6, "classes": LABELS,
              "grid": "factor-two preprocessed XYZ; padded model edges removed",
              "surface_distance": "symmetric 3D voxel surfaces, physical spacing mm; one-empty null",
              "checkpoint_selection": "frozen Validation patient small-organ score, strict earlier-epoch tie",
              "gpu": torch.cuda.get_device_name(0), "job_id": os.environ["SLURM_JOB_ID"],
              "evaluation_script_sha256": sha256(Path(__file__)), "arms": {}}
    for model_name, (model, receipt) in selected.items():
        matrices = defaultdict(lambda: np.zeros((6, 6), dtype=np.int64))
        rows = []
        for index, case in enumerate(cases, 1):
            preparation_started = perf_counter()
            volume = zscore_nonzero(downsample(load_case(case)))
            preparation_seconds = perf_counter() - preparation_started
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = perf_counter()
            prediction = predict_volume(model, volume, mode="3d", device="cuda")
            torch.cuda.synchronize()
            inference_seconds = perf_counter() - started
            peak_bytes = int(torch.cuda.max_memory_allocated())
            matrix = confusion(volume.mask, prediction)
            matrices[case.patient] += matrix
            boundaries = {label: symmetric_surface_distances(
                volume.mask, prediction, volume.spacing_mm, label=class_id)
                for class_id, label in enumerate(LABELS) if class_id > 0}
            try:
                regions = prostate_axial_regions(volume.mask, volume.affine)
                per_slice = slice_dice(volume.mask, prediction)
                apex_base = {name: float(np.mean([per_slice[z] for z in indices]))
                             for name, indices in regions.items()}
            except ValueError:
                apex_base = None
            rows.append({"case": case.identifier, "patient": case.patient,
                         "shape_xyz": volume.mask.shape, "spacing_mm": volume.spacing_mm,
                         "mri_sha256": sha256(case.mri), "mask_sha256": sha256(case.mask),
                         "confusion": matrix.tolist(), "overlap": overlap(matrix),
                         "boundary": boundaries, "prostate_axial_region_dice": apex_base,
                         "preprocessing_seconds": preparation_seconds,
                         "inference_seconds": inference_seconds,
                         "peak_allocated_bytes": peak_bytes})
            print(f"{model_name} Test {index}/30: {case.identifier}", flush=True)
        pooled = sum(matrices.values(), np.zeros((6, 6), dtype=np.int64))
        patient_score, patient_dice = patient_small_organ_score(dict(matrices))
        result["arms"][model_name] = {
            "checkpoint": receipt, "cases": rows,
            "pooled_confusion": pooled.tolist(), "pooled_overlap": overlap(pooled),
            "patient_small_organ_score": patient_score,
            "per_patient_small_organ_dice": patient_dice,
            "boundary_summary": boundary_summary(rows),
            "median_inference_seconds": median(row["inference_seconds"] for row in rows),
            "max_peak_allocated_bytes": max(row["peak_allocated_bytes"] for row in rows),
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(f"Saved one-time held-out Test result to {output}", flush=True)


if __name__ == "__main__":
    main()
