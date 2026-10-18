"""Reproduce selected Validation failures from frozen G5 3D checkpoints.

The PNGs contain medical-image-derived content. Check HipMRI data-use terms
before exporting them from Rangpur. No Test image or label is read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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

from hipmri_rewrite.boundaries import symmetric_surface_distances  # noqa: E402
from hipmri_rewrite.data import downsample, load_case, zscore_nonzero  # noqa: E402
from hipmri_rewrite.inference import predict_volume  # noqa: E402
from hipmri_rewrite.metrics import confusion, overlap  # noqa: E402
from hipmri_rewrite.models import ImprovedUNet3D, UNet3D  # noqa: E402
from hipmri_rewrite.splits import DEFAULT_SPLIT, discover  # noqa: E402

CONFIG_SHA = "e5bd98a99213f4fe6cf6e3e8c6db3b95d5ef334a143a91444426d8ea019923ce"
LABELS = {
    "J026_Week1": (4, "Rectum", "high_overlap"),
    "K018_Week3": (4, "Rectum", "overprediction"),
    "B040_Week1": (5, "Prostate", "high_overlap"),
    "S022_Week6": (5, "Prostate", "missed_tissue"),
}
MODELS = {
    "Standard 3D": ("standard_3d", UNet3D),
    "Improved 3D": ("improved_3d", ImprovedUNet3D),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_selected(run_dir: Path, model_name: str, constructor):
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest["model"] != model_name or manifest["config_sha256"] != CONFIG_SHA:
        raise ValueError(f"{run_dir}: wrong frozen run identity")
    if manifest["split_sha256"] != sha256(DEFAULT_SPLIT):
        raise ValueError(f"{run_dir}: split changed")
    sources = {
        "data_sha256": "hipmri_rewrite/data.py",
        "inference_sha256": "hipmri_rewrite/inference.py",
        "metrics_sha256": "hipmri_rewrite/metrics.py",
        "model_package_sha256": "hipmri_rewrite/models/__init__.py",
        "model_source_sha256": ("hipmri_rewrite/models/baseline_3d.py" if model_name == "standard_3d"
                                else "hipmri_rewrite/models/improved_3d.py"),
        "model_common_sha256": "hipmri_rewrite/models/common.py",
    }
    for key, relative in sources.items():
        if manifest[key] != sha256(PROJECT / "code" / relative):
            raise ValueError(f"{run_dir}: {relative} changed since training")
    checkpoint_path = run_dir / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    selected_epoch = checkpoint["epoch"]
    if (checkpoint["model"] != model_name or checkpoint["config_sha256"] != CONFIG_SHA
            or checkpoint["best_epoch"] != selected_epoch or selected_epoch < 5
            or selected_epoch % 5):
        raise ValueError(f"{checkpoint_path}: not a selected Validation checkpoint")
    record_path = run_dir / f"epoch_{selected_epoch:03d}.json"
    epoch_record = json.loads(record_path.read_text(encoding="utf-8"))
    if (epoch_record["best_epoch"] != selected_epoch or epoch_record["validation"] is None
            or not np.isclose(epoch_record["best_score"], checkpoint["best_score"])):
        raise ValueError(f"{run_dir}: selected Validation record differs from checkpoint")
    model = constructor().to("cuda").eval()
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model, epoch_record, {"selected_epoch": selected_epoch,
                                 "manifest_sha256": sha256(run_dir / "manifest.json"),
                                 "checkpoint_sha256": sha256(checkpoint_path),
                                 "epoch_record_sha256": sha256(record_path)}


def selected_slice(truth: np.ndarray, predictions: dict[str, np.ndarray], label: int,
                   reason: str) -> int:
    candidates = np.flatnonzero((truth == label).any(axis=(0, 1)))
    if not len(candidates):
        raise ValueError("Selected Validation case has no target slice")
    if reason == "high_overlap":
        area = (truth == label).sum(axis=(0, 1))
        return int(candidates[np.argmax(area[candidates])])
    if reason == "overprediction":
        fp = (predictions["Improved 3D"] == label) & (truth != label)
        return int(candidates[np.argmax(fp.sum(axis=(0, 1))[candidates])])
    fn = (truth == label) & (predictions["Standard 3D"] != label)
    return int(candidates[np.argmax(fn.sum(axis=(0, 1))[candidates])])


def render(case_id: str, volume, predictions: dict[str, np.ndarray], label: int,
           organ: str, path: Path, z: int) -> None:
    plane = volume.image[:, :, z]
    gt = volume.mask[:, :, z] == label
    nonzero = volume.image[volume.image != 0]
    low, high = np.percentile(nonzero, (1, 99))
    if high <= low:
        high = low + 1
    occupied = gt.copy()
    for prediction in predictions.values():
        occupied |= prediction[:, :, z] == label
    coords = np.argwhere(occupied)
    if not len(coords):
        raise ValueError("Selected slice contains no target pixels")
    lower = np.maximum(coords.min(axis=0) - 15, 0)
    upper = np.minimum(coords.max(axis=0) + 16, np.asarray(plane.shape))
    sx, sy = (slice(int(lower[i]), int(upper[i])) for i in range(2))
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), layout="constrained")
    for ax, (name, prediction) in zip(axes, predictions.items()):
        ax.imshow(plane[sx, sy].T, cmap="gray", vmin=low, vmax=high, origin="lower")
        for mask, color in ((gt[sx, sy], "#00d5e9"),
                            (prediction[sx, sy, z] == label, "#ff9d24")):
            if mask.any() and not mask.all():
                ax.contour(mask.T.astype(float), levels=[0.5], colors=[color], linewidths=1.4)
        ax.set_title(name)
        ax.set_xlabel("X (downsampled voxels)")
        ax.set_ylabel("Y (downsampled voxels)")
        ax.set_aspect(volume.spacing_mm[1] / volume.spacing_mm[0])
    fig.suptitle(f"Validation {case_id} · {organ} · axial z={z} · selected checkpoints")
    fig.legend([Line2D([0], [0], color="#00d5e9", lw=2),
                Line2D([0], [0], color="#ff9d24", lw=2)],
               ["Ground truth", "Prediction"], loc="lower center", ncol=2)
    fig.savefig(path, dpi=150, metadata={"Description": "Validation MRI and selected G5 3D predictions"})
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--standard-run-dir", type=Path, required=True)
    parser.add_argument("--improved-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if not output.is_relative_to(PROJECT / "results/case_studies"):
        parser.error("Output must be inside results/case_studies")
    if output.exists() and any(output.glob("g5_validation_*")):
        parser.error("Selected output files already exist")
    runs = {"Standard 3D": args.standard_run_dir, "Improved 3D": args.improved_run_dir}
    loaded = {name: load_selected(runs[name], *MODELS[name]) for name in MODELS}
    validation = {case.identifier: case for case in discover(args.dataset_root)["validation"]}
    if any(case_id not in validation for case_id in LABELS):
        raise ValueError("Selected case is outside frozen Validation split")
    if not torch.cuda.is_available():
        raise RuntimeError("Run this selected Validation inference on a Slurm GPU node")
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(2)
    results = {"kind": "G5_selected_Validation_case_review", "split": "validation",
               "cases": [],
               "sources": {name: loaded[name][2] for name in MODELS}}
    rendered = []
    for case_id, (label, organ, reason) in LABELS.items():
        volume = zscore_nonzero(downsample(load_case(validation[case_id])))
        predictions = {}
        row = {"case": case_id, "organ": organ, "selection_reason": reason,
               "spacing_mm": volume.spacing_mm,
               "models": {}}
        for name, (model, record, _) in loaded.items():
            prediction = predict_volume(model, volume, mode="3d", device="cuda")
            expected = next(item for item in record["validation"]["cases"]
                            if item["case"] == case_id)["confusion"]
            matrix = confusion(volume.mask, prediction)
            if matrix.tolist() != expected:
                raise ValueError(f"{name} {case_id}: prediction differs from saved Validation")
            predictions[name] = prediction
            metrics = overlap(matrix)
            row["models"][name] = {
                "dice": metrics["dice"][label], "precision": metrics["precision"][label],
                "recall": metrics["recall"][label],
                "truth_voxels": int((volume.mask == label).sum()),
                "predicted_voxels": int((prediction == label).sum()),
                "boundary": symmetric_surface_distances(volume.mask, prediction,
                                                        volume.spacing_mm, label=label),
            }
        z = selected_slice(volume.mask, predictions, label, reason)
        row["axial_slice"] = z
        row["figure"] = f"g5_validation_{case_id}_{organ.lower()}.png"
        results["cases"].append(row)
        rendered.append((case_id, volume, predictions, label, organ, z, row["figure"]))
        print(f"{case_id}: reproduced both frozen Validation confusion matrices", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    for case_id, volume, predictions, label, organ, z, filename in rendered:
        render(case_id, volume, predictions, label, organ, output / filename, z)
        print(output / filename, flush=True)
    result_path = output / "g5_validation_selected_cases.json"
    result_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(result_path, flush=True)


if __name__ == "__main__":
    main()
