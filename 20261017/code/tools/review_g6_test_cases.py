"""Render three descriptive held-out Test cases from fixed 3D checkpoints.

Case review explains the one-time Test result; it does not select checkpoints,
change training, or rerun the cohort evaluation. PNGs contain MRI-derived data.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "code"))

from hipmri_rewrite.data import downsample, load_case, zscore_nonzero  # noqa: E402
from hipmri_rewrite.inference import predict_volume  # noqa: E402
from hipmri_rewrite.metrics import confusion, overlap  # noqa: E402
from hipmri_rewrite.splits import discover  # noqa: E402
from tools.evaluate_g6_test import ARMS, selected_model, sha256  # noqa: E402

CASES = (
    ("M020_Week1", 5, "Prostate", "high_overlap"),
    ("M015_Week7", 3, "Bladder", "low_overlap"),
    ("W029_Week6", 5, "Prostate", "low_overlap"),
)
NAMES = {"standard_3d": "Standard 3D", "improved_3d": "Improved 3D"}


def axial_slice(truth: np.ndarray, predictions: dict[str, np.ndarray], label: int,
                reason: str) -> int:
    gt = truth == label
    candidates = np.flatnonzero(gt.any(axis=(0, 1)))
    if not len(candidates):
        raise ValueError("Selected Test organ is absent")
    if reason == "high_overlap":
        quantity = gt.sum(axis=(0, 1))
    else:
        quantity = (gt & (predictions["standard_3d"] != label)).sum(axis=(0, 1))
        if not quantity[candidates].any():
            quantity = gt.sum(axis=(0, 1))
    return int(candidates[np.argmax(quantity[candidates])])


def render(volume, predictions: dict[str, np.ndarray], case_id: str,
           organ: str, label: int, z: int, path: Path) -> None:
    plane = volume.image[:, :, z]
    truth = volume.mask[:, :, z] == label
    occupied = truth.copy()
    for prediction in predictions.values():
        occupied |= prediction[:, :, z] == label
    coordinates = np.argwhere(occupied)
    if not len(coordinates):
        raise ValueError("Selected Test slice has no organ voxels")
    lower = np.maximum(coordinates.min(axis=0) - 15, 0)
    upper = np.minimum(coordinates.max(axis=0) + 16, np.asarray(plane.shape))
    sx, sy = (slice(int(lower[index]), int(upper[index])) for index in range(2))
    nonzero = volume.image[volume.image != 0]
    low, high = np.percentile(nonzero, (1, 99))
    if high <= low:
        high = low + 1
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), layout="constrained")
    for ax, (name, prediction) in zip(axes, predictions.items()):
        ax.imshow(plane[sx, sy].T, cmap="gray", vmin=low, vmax=high, origin="lower")
        for mask, color in ((truth[sx, sy], "#00d5e9"),
                            (prediction[sx, sy, z] == label, "#ff9d24")):
            if mask.any() and not mask.all():
                ax.contour(mask.T.astype(float), levels=[0.5], colors=[color], linewidths=1.4)
        ax.set_title(NAMES[name])
        ax.set_xlabel("X (downsampled voxels)")
        ax.set_ylabel("Y (downsampled voxels)")
        ax.set_aspect(volume.spacing_mm[1] / volume.spacing_mm[0])
    fig.suptitle(f"Held-out Test {case_id} · {organ} · axial z={z}")
    fig.legend([Line2D([0], [0], color="#00d5e9", lw=2),
                Line2D([0], [0], color="#ff9d24", lw=2)],
               ["Ground truth", "Prediction"], loc="lower center", ncol=2)
    fig.savefig(path, dpi=150,
                metadata={"Description": "Held-out Test MRI and frozen 3D predictions"})
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if (not output.is_relative_to(PROJECT / "results/case_studies/comparisons")
            or not os.environ.get("SLURM_JOB_ID")
            or socket.gethostname().startswith("login")):
        parser.error("Use the existing case-study folder on a Slurm GPU node")
    if any(output.glob("g6_test_*")):
        parser.error("Test case-study output already exists")
    if not torch.cuda.is_available():
        raise RuntimeError("No allocated CUDA device")
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False

    test_result_path = PROJECT / "results/tables/comparisons/g6_test_3d.json"
    test_result = json.loads(test_result_path.read_text(encoding="utf-8"))
    if (test_result["kind"] != "G6_one_time_held_out_Test"
            or test_result["job_id"] != "674125"
            or test_result["evaluation_script_sha256"] !=
            sha256(PROJECT / "code/tools/evaluate_g6_test.py")):
        raise ValueError("Unexpected one-time Test result")
    selected = {identity[0]: selected_model(PROJECT / "logs/training", identity,
                                           args.dataset_root, torch)
                for identity in ARMS}
    for name, (_, receipt) in selected.items():
        if receipt != test_result["arms"][name]["checkpoint"]:
            raise ValueError(f"{name}: selected checkpoint differs from Test result")
    cases = {case.identifier: case for case in discover(args.dataset_root)["test"]}
    if any(case_id not in cases for case_id, *_ in CASES):
        raise ValueError("Selected case is outside frozen Test cohort")

    result = {"kind": "G6_descriptive_held_out_Test_case_review", "split": "test",
              "test_result_sha256": sha256(test_result_path), "cases": []}
    rendered = []
    for case_id, label, organ, reason in CASES:
        case = cases[case_id]
        volume = zscore_nonzero(downsample(load_case(case)))
        predictions = {}
        row = {"case": case_id, "organ": organ, "descriptive_reason": reason,
               "models": {}, "spacing_mm": volume.spacing_mm}
        for name, (model, _) in selected.items():
            prediction = predict_volume(model, volume, mode="3d", device="cuda")
            saved = next(item for item in test_result["arms"][name]["cases"]
                         if item["case"] == case_id)
            if (sha256(case.mri) != saved["mri_sha256"]
                    or sha256(case.mask) != saved["mask_sha256"]):
                raise ValueError(f"{case_id}: Test source differs")
            matrix = confusion(volume.mask, prediction)
            if matrix.tolist() != saved["confusion"]:
                raise ValueError(f"{name} {case_id}: Test prediction differs")
            predictions[name] = prediction
            measure = overlap(matrix)
            row["models"][name] = {
                "dice": measure["dice"][label],
                "precision": measure["precision"][label],
                "recall": measure["recall"][label],
                "truth_voxels": int((volume.mask == label).sum()),
                "predicted_voxels": int((prediction == label).sum()),
            }
        z = axial_slice(volume.mask, predictions, label, reason)
        row["axial_slice"] = z
        row["figure"] = f"g6_test_{case_id}_{organ.lower()}.png"
        result["cases"].append(row)
        rendered.append((volume, predictions, case_id, organ, label, z, row["figure"]))
        print(f"{case_id}: reproduced both one-time Test confusion matrices", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    for volume, predictions, case_id, organ, label, z, filename in rendered:
        render(volume, predictions, case_id, organ, label, z, output / filename)
        print(output / filename, flush=True)
    receipt = output / "g6_test_selected_cases.json"
    receipt.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(receipt, flush=True)


if __name__ == "__main__":
    main()
