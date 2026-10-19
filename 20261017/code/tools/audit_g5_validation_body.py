"""Measure Body boundaries on the fixed G5 Validation cohort and draw outliers.

This reads the two existing Validation-selected 3D checkpoints. It never reads
the Test split or alters weights. Full-volume confusion must reproduce the
selected epoch-50 Validation records before boundary metrics are accepted.
"""

from __future__ import annotations

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
from hipmri_rewrite.boundaries import symmetric_surface_distances  # noqa: E402
from hipmri_rewrite.data import downsample, load_case, zscore_nonzero  # noqa: E402
from hipmri_rewrite.inference import predict_volume  # noqa: E402
from hipmri_rewrite.metrics import confusion, overlap  # noqa: E402
from hipmri_rewrite.splits import discover  # noqa: E402
from tools.evaluate_g6_test import ARMS, selected_model, sha256  # noqa: E402

MODELS = ("standard_3d", "improved_3d")


def draw(volume, predictions: dict, center: tuple[int, int, int], case_id: str,
         output: Path) -> None:
    x, y, z = center
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.8), layout="constrained")
    nonzero = volume.image[volume.image != 0]
    low, high = np.percentile(nonzero, (1, 99))
    if high <= low:
        high = low + 1
    for row, model in enumerate(MODELS):
        pred = predictions[model]
        slices = (("Axial", volume.image[:, :, z].T,
                   (volume.mask[:, :, z] == 1).T, (pred[:, :, z] == 1).T),
                  ("Coronal", volume.image[:, y, :].T,
                   (volume.mask[:, y, :] == 1).T, (pred[:, y, :] == 1).T),
                  ("Sagittal", volume.image[x, :, :].T,
                   (volume.mask[x, :, :] == 1).T, (pred[x, :, :] == 1).T))
        for col, (plane, image, truth, predicted) in enumerate(slices):
            ax = axes[row, col]
            ax.imshow(image, cmap="gray", origin="lower", vmin=low, vmax=high)
            if truth.any() and not truth.all():
                ax.contour(truth.astype(float), levels=[0.5], colors=["#00d5e9"])
            if predicted.any() and not predicted.all():
                ax.contour(predicted.astype(float), levels=[0.5], colors=["#ff9d24"])
            extra = predicted & ~truth
            if extra.any():
                ax.imshow(np.ma.masked_where(~extra, extra), cmap="Reds", alpha=0.25,
                          origin="lower", interpolation="none")
            ax.set_title(f"{model.replace('_', ' ').title()} · {plane}")
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(f"Validation {case_id} · Body · slices through distant false positive")
    fig.legend((Line2D([0], [0], color="#00d5e9"),
                Line2D([0], [0], color="#ff9d24")),
               ("Ground truth", "Prediction"), loc="lower center", ncol=2)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    args = parser.parse_args()
    if (not os.environ.get("SLURM_JOB_ID")
            or socket.gethostname().startswith("login")
            or not torch.cuda.is_available()):
        parser.error("Run on an allocated GPU compute node")
    torch.set_num_threads(2)
    output_json = PROJECT / "results/tables/comparisons/g5_validation_body_audit.json"
    figure_dir = PROJECT / "results/case_studies/comparisons"
    if output_json.exists() or any(figure_dir.glob("g5_body_validation_*")):
        parser.error("Validation Body audit outputs already exist")
    model_info = {identity[0]: selected_model(PROJECT / "logs/training", identity,
                                              args.dataset_root, torch)
                  for identity in ARMS}
    saved = {}
    for model, folder in (("standard_3d", "g5_standard_20261018"),
                          ("improved_3d", "g5_improved_20261018")):
        source = PROJECT / "logs/training" / folder / "epoch_050.json"
        record = json.loads(source.read_text(encoding="utf-8"))
        if record["best_epoch"] != 50 or record["validation"] is None:
            raise ValueError(f"{model}: selected Validation record missing")
        saved[model] = {row["case"]: row for row in record["validation"]["cases"]}
    cases = {case.identifier: case for case in discover(args.dataset_root)["validation"]}
    if (len(cases) != 38 or any(set(saved[model]) != set(cases) for model in MODELS)):
        raise ValueError("Validation case inventory differs from selected records")
    result = {
        "kind": "G5_selected_3D_Validation_Body_boundary_audit",
        "split": "validation", "selection": "fixed G5 epoch-50 best checkpoints",
        "cases": [],
        "sources": {model: model_info[model][1] for model in MODELS},
    }
    selected_images = {}
    for index, case_id in enumerate(sorted(cases), 1):
        volume = zscore_nonzero(downsample(load_case(cases[case_id])))
        distance = distance_transform_edt(volume.mask != 1,
                                          sampling=volume.spacing_mm)
        predictions = {}
        row = {"case": case_id, "patient": cases[case_id].patient, "models": {}}
        for model in MODELS:
            prediction = predict_volume(model_info[model][0], volume,
                                        mode="3d", device="cuda")
            matrix = confusion(volume.mask, prediction)
            if matrix.tolist() != saved[model][case_id]["confusion"]:
                raise ValueError(f"{case_id}: {model} does not reproduce G5 Validation")
            predictions[model] = prediction
            extra = (prediction == 1) & (volume.mask != 1)
            background_extra = extra & (volume.mask == 0)
            scores = overlap(matrix)
            row["models"][model] = {
                "body_dice": scores["dice"][1],
                "body_precision": scores["precision"][1],
                "body_recall": scores["recall"][1],
                "body_gt_voxels": int((volume.mask == 1).sum()),
                "body_predicted_voxels": int((prediction == 1).sum()),
                "body_false_positive_voxels": int(extra.sum()),
                "background_to_body_voxels": int(background_extra.sum()),
                "background_to_body_at_least_20mm": int(
                    (background_extra & (distance >= 20)).sum()),
                "max_false_positive_distance_to_gt_body_mm": (
                    float(distance[extra].max()) if extra.any() else None),
                "boundary": symmetric_surface_distances(
                    volume.mask, prediction, volume.spacing_mm, label=1),
            }
        improved_extra = ((predictions["improved_3d"] == 1)
                          & (volume.mask != 1))
        if improved_extra.any():
            farthest = np.argmax(np.where(improved_extra, distance, -1))
            center = tuple(int(v) for v in np.unravel_index(farthest, distance.shape))
        else:
            center = tuple(int(v) for v in np.argwhere(volume.mask == 1).mean(axis=0))
        selected_images[case_id] = (volume, predictions, center)
        result["cases"].append(row)
        print(f"[{index}/38] {case_id}: both fixed Validation matrices reproduced",
              flush=True)
    # Three highest Improved Body HD95 cases from different patients.
    ordered = sorted(result["cases"],
                     key=lambda row: (row["models"]["improved_3d"]["boundary"]["hd95_mm"]
                                      or -1), reverse=True)
    chosen_patients = set()
    result["pictured_cases"] = []
    for row in ordered:
        if row["patient"] in chosen_patients:
            continue
        chosen_patients.add(row["patient"])
        case_id = row["case"]
        image_path = figure_dir / f"g5_body_validation_{case_id}.png"
        draw(*selected_images[case_id], case_id, image_path)
        result["pictured_cases"].append({"case": case_id,
                                          "figure": image_path.name,
                                          "selection": "highest Improved Body HD95 from a distinct patient"})
        print(image_path, flush=True)
        if len(result["pictured_cases"]) == 3:
            break
    if len(result["pictured_cases"]) != 3:
        raise ValueError("Could not select three different patients")
    output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(output_json, flush=True)


if __name__ == "__main__":
    main()
