"""Reproduce selected fixed Test predictions and inspect distant Body errors.

This is a descriptive review of selected Test checkpoints. It does not select
weights, change training, or update the one-time Test evaluation.
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
from scipy.ndimage import distance_transform_edt

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "code"))
from hipmri_rewrite.data import downsample, load_case, zscore_nonzero  # noqa: E402
from hipmri_rewrite.inference import predict_volume  # noqa: E402
from hipmri_rewrite.metrics import confusion  # noqa: E402
from hipmri_rewrite.splits import discover  # noqa: E402
from tools.evaluate_g6_test import ARMS, selected_model, sha256  # noqa: E402

CASES = ("M015_Week3", "L011_Week5", "W041_Week0")
MODELS = ("standard_3d", "improved_3d")


def planes(volume, pred: np.ndarray, center: tuple[int, int, int]):
    x, y, z = center
    return (("Axial", volume.image[:, :, z].T, (volume.mask[:, :, z] == 1).T,
             (pred[:, :, z] == 1).T),
            ("Coronal", volume.image[:, y, :].T, (volume.mask[:, y, :] == 1).T,
             (pred[:, y, :] == 1).T),
            ("Sagittal", volume.image[x, :, :].T, (volume.mask[x, :, :] == 1).T,
             (pred[x, :, :] == 1).T))


def draw(volume, predictions: dict[str, np.ndarray], center: tuple[int, int, int],
         case_id: str, path: Path) -> None:
    values = volume.image[volume.image != 0]
    low, high = np.percentile(values, (1, 99))
    if high <= low:
        high = low + 1
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.8), layout="constrained")
    for row, arm in enumerate(MODELS):
        for col, (orientation, image, truth, predicted) in enumerate(
                planes(volume, predictions[arm], center)):
            ax = axes[row, col]
            ax.imshow(image, cmap="gray", origin="lower", vmin=low, vmax=high)
            if truth.any() and not truth.all():
                ax.contour(truth.astype(float), levels=[0.5], colors=["#00d5e9"], linewidths=1)
            if predicted.any() and not predicted.all():
                ax.contour(predicted.astype(float), levels=[0.5], colors=["#ff9d24"], linewidths=1)
            extra = predicted & ~truth
            if extra.any():
                ax.imshow(np.ma.masked_where(~extra, extra), cmap="Reds", alpha=0.25,
                          origin="lower", interpolation="none")
            ax.set_title(f"{arm.replace('_', ' ').title()} · {orientation}")
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(f"Held-out Test {case_id} · Body · slices through a distant false positive")
    fig.legend((Line2D([0], [0], color="#00d5e9"),
                Line2D([0], [0], color="#ff9d24")),
               ("Ground truth", "Prediction"), loc="lower center", ncol=2)
    fig.savefig(path, dpi=160)
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
        parser.error("Run on an allocated GPU node in the existing comparison folder")
    if any(output.glob("g6_body_*")):
        parser.error("Body review output already exists")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    torch.set_num_threads(2)
    source = PROJECT / "results/tables/comparisons/g6_test_3d.json"
    result = json.loads(source.read_text(encoding="utf-8"))
    if (result["kind"] != "G6_one_time_held_out_Test"
            or result["job_id"] != "674125"
            or result["evaluation_script_sha256"] !=
            sha256(PROJECT / "code/tools/evaluate_g6_test.py")):
        raise ValueError("Unexpected one-time Test source")
    selected = {identity[0]: selected_model(PROJECT / "logs/training", identity,
                                           args.dataset_root, torch)
                for identity in ARMS}
    if any(selected[arm][1] != result["arms"][arm]["checkpoint"] for arm in MODELS):
        raise ValueError("Selected weights differ from one-time Test")
    cases = {case.identifier: case for case in discover(args.dataset_root)["test"]}
    review = {"kind": "G6_descriptive_Body_boundary_review", "split": "test",
              "test_result_sha256": sha256(source), "cases": []}
    rendered = []
    for case_id in CASES:
        case = cases[case_id]
        volume = zscore_nonzero(downsample(load_case(case)))
        predictions = {}
        row = {"case": case_id, "models": {}}
        for arm in MODELS:
            prediction = predict_volume(selected[arm][0], volume, mode="3d", device="cuda")
            saved = next(item for item in result["arms"][arm]["cases"]
                         if item["case"] == case_id)
            if (sha256(case.mri) != saved["mri_sha256"]
                    or sha256(case.mask) != saved["mask_sha256"]
                    or confusion(volume.mask, prediction).tolist() != saved["confusion"]):
                raise ValueError(f"{case_id}: {arm} does not reproduce the one-time Test")
            predictions[arm] = prediction
            extra = (prediction == 1) & (volume.mask != 1)
            row["models"][arm] = {
                "body_hd95_mm": saved["boundary"]["Body"]["hd95_mm"],
                "body_dice": saved["overlap"]["dice"][1],
                "body_false_positive_voxels": int(extra.sum()),
                "background_to_body_voxels": int(((volume.mask == 0) & extra).sum()),
            }
        improved_extra = (predictions["improved_3d"] == 1) & (volume.mask != 1)
        if not improved_extra.any():
            raise ValueError(f"{case_id}: no Improved Body false positives")
        distance = distance_transform_edt(volume.mask != 1, sampling=volume.spacing_mm)
        index = np.argmax(np.where(improved_extra, distance, -1))
        center = tuple(int(value) for value in np.unravel_index(index, distance.shape))
        row["slice_center_xyz"] = center
        row["center_distance_to_gt_body_mm"] = float(distance[center])
        row["figure"] = f"g6_body_{case_id}.png"
        review["cases"].append(row)
        rendered.append((volume, predictions, center, case_id, row["figure"]))
        print(f"{case_id}: both predictions reproduce the fixed Test matrices", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    for volume, predictions, center, case_id, name in rendered:
        draw(volume, predictions, center, case_id, output / name)
        print(output / name, flush=True)
    path = output / "g6_body_selected_cases.json"
    path.write_text(json.dumps(review, indent=2) + "\n", encoding="utf-8")
    print(path, flush=True)


if __name__ == "__main__":
    main()
