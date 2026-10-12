"""Protected checkpoint inference, full-volume metrics and aligned HipMRI panels.

Validation is the default split. Test processing requires --split test and
--allow-test. Select cases explicitly or request --all-split. Outputs use a new
external directory; original NIfTIs and checkpoints are never written.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
from pathlib import Path
import os
import sys
from time import perf_counter
import json

import numpy as np
import torch

from .dataset import (
    APPROVED_PATIENTS, DEFAULT_DATASET_ROOT, EXPECTED_VOLUME_COUNTS,
    discover_dataset,
)
from .train import configure_reproducibility
from . import validate as evaluation


# Class index, name and colour remain fixed even when a class is absent.
CLASS_COLORS = ("#000000", "#e6b800", "#bdbdbd", "#377eb8", "#4daf4a", "#e7298a")
ERROR_COLORS = ("#000000", "#ff3030", "#00bfff", "#ffcc33")
MODEL_NAMES = {"standard": "StandardUNet3D", "improved": "ImprovedUNet3D"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_NAMES, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--allow-test", action="store_true", help="explicit held-out Test opt-in")
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument("--cases", "--case", nargs="+", type=evaluation._case,
                       metavar="PATIENT_WeekN", help="one or more cases in the selected split")
    scope.add_argument("--all-split", action="store_true", help="all volumes in the explicitly selected split")
    parser.add_argument("--stride", type=int, default=32, help="shared 64-cube sliding-window stride, 1..64")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--output-dir", type=Path, required=True, help="new directory outside Git and source data")
    args = parser.parse_args(argv)
    if not 1 <= args.stride <= 64:
        parser.error("stride must be in 1..64 for complete 64-cube coverage")
    if (args.split == "test") != args.allow_test:
        parser.error("Test requires --split test --allow-test; --allow-test is invalid for Validation")
    if args.cases and len(set(args.cases)) != len(args.cases):
        parser.error("case selection contains duplicate patient/Week identifiers")
    return args


def select_pairs(splits, split="validation", cases=None, *, allow_test=False):
    """Select frozen evaluation pairs without decoding other partitions."""
    if split not in ("validation", "test") or (split == "test") != allow_test:
        raise ValueError("Select Validation, or explicitly opt into Test; Train is forbidden.")
    pairs = tuple(splits[split])
    keys = [(pair.patient_id, pair.week) for pair in pairs]
    if (len(pairs) != EXPECTED_VOLUME_COUNTS[split] or len(set(keys)) != len(keys)
            or {pair.patient_id for pair in pairs} != set(APPROVED_PATIENTS[split])):
        raise ValueError(f"{split}: discovered coverage differs from the frozen patient/volume assignments.")
    if cases is None:
        return pairs
    if not cases or len(set(cases)) != len(cases):
        raise ValueError("Cases must be a nonempty list of unique patient/Week identifiers.")
    lookup = {(pair.patient_id, pair.week): pair for pair in pairs}
    missing = [f"{patient}_Week{week}" for patient, week in cases if (patient, week) not in lookup]
    if missing:
        raise ValueError(f"Cases are not in approved {split}: {missing}; no other split is decoded.")
    return tuple(lookup[key] for key in cases)


def load_model(model_type, checkpoint, device):
    """Delegate all checkpoint/schema/policy/hash/state checks to approved loaders."""
    if model_type == "standard":
        return evaluation.load_checkpoint_model(checkpoint, device)
    if model_type == "improved":
        from .validate_improved import load_checkpoint_model
        return load_checkpoint_model(checkpoint, device)
    raise ValueError("model must be standard or improved")


def prepare_volume(pair, policies):
    """Run the unchanged production pipeline for exactly one selected pair."""
    from .volume_io import load_volume_pair
    from .resampling import resample_volume_pair
    from .intensity_normalization import normalize_volume_pair
    loaded = load_volume_pair(pair)
    original_shape = list(loaded.mri.shape)
    settings = policies["resampling"]
    resampled = resample_volume_pair(
        loaded, settings["target_spacing_mm"], max_estimated_bytes=settings["max_estimated_bytes"])
    del loaded
    normalized = normalize_volume_pair(resampled)
    del resampled
    return normalized, original_shape


def aligned_slice_panels(mri, truth, prediction, z):
    """Same XYZ slice/transposition for all panels; error codes target Prostate (5).

    Error 1: prostate FP, 2: prostate FN, 3: other incorrect class, 0: correct.
    No anatomical apex/base designation is inferred from these array indices.
    """
    if (mri.ndim != 3 or truth.shape != mri.shape or prediction.shape != mri.shape
            or mri.dtype != np.float32 or truth.dtype != np.uint8 or prediction.dtype != np.uint8
            or type(z) is not int or not 0 <= z < mri.shape[2]):
        raise ValueError("Panels require aligned float32 MRI/uint8 XYZ masks and a valid Z index.")
    image, reference, predicted = (array[:, :, z].T for array in (mri, truth, prediction))
    if (not np.isfinite(image).all() or np.any(reference > 5) or np.any(predicted > 5)):
        raise ValueError("Panel slice contains non-finite MRI or invalid class IDs.")
    errors = np.zeros(reference.shape, np.uint8)
    errors[predicted != reference] = 3
    errors[(predicted == 5) & (reference != 5)] = 1
    errors[(reference == 5) & (predicted != 5)] = 2
    return image, reference, predicted, errors


def save_visualization(path, mri, truth, prediction, affine, case):
    """Save three source-axis Z rows on the shared resampled grid using Agg."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.figure import Figure
    from matplotlib.patches import Patch
    if (mri.ndim != 3 or any(size < 1 for size in mri.shape)
            or truth.shape != mri.shape or prediction.shape != mri.shape):
        raise ValueError("Visualisation requires matching nonempty XYZ grids.")
    affine = np.asarray(affine)
    if affine.shape != (4, 4) or not np.isfinite(affine).all() or np.linalg.matrix_rank(affine[:3, :3]) != 3:
        raise ValueError("Visualisation requires a finite nonsingular shared affine.")
    occupied = np.flatnonzero(np.any(truth == 5, axis=(0, 1)))
    if occupied.size:
        slices = sorted(set((int(occupied[0]), int(occupied[len(occupied) // 2]), int(occupied[-1]))))
        selection = "ground-truth prostate extent: low / middle / high Z"
    else:
        slices = sorted(set(int(fraction * (mri.shape[2] - 1)) for fraction in (.25, .5, .75)))
        selection = "geometric Z samples (no ground-truth prostate)"
    labels = ListedColormap(CLASS_COLORS)
    errors = ListedColormap(ERROR_COLORS)
    label_norm, error_norm = BoundaryNorm(np.arange(7) - .5, 6), BoundaryNorm(np.arange(5) - .5, 4)
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    figure = Figure(figsize=(12, 3 * len(slices) + 1.6))
    FigureCanvasAgg(figure)
    axes = figure.subplots(len(slices), 4, squeeze=False)
    try:
        for row, z in enumerate(slices):
            image, reference, predicted, difference = aligned_slice_panels(mri, truth, prediction, z)
            for column, (title, overlay) in enumerate((
                    ("Normalised MRI", None), ("Ground truth", reference),
                    ("Prediction", predicted), ("Prostate FP/FN + other class errors", difference))):
                axis = axes[row, column]
                axis.imshow(image, cmap="gray", origin="lower", interpolation="nearest",
                            aspect=spacing[1] / spacing[0])
                if overlay is not None:
                    axis.imshow(np.ma.masked_equal(overlay, 0), cmap=errors if column == 3 else labels,
                                norm=error_norm if column == 3 else label_norm, alpha=.65,
                                origin="lower", interpolation="nearest", aspect=spacing[1] / spacing[0])
                axis.set_title(f"{title}\nZ={z}", fontsize=9)
                axis.set_axis_off()
        legend = [Patch(facecolor=color, label=f"{index}: {name}")
                  for index, (name, color) in enumerate(zip(evaluation.CLASS_NAMES, CLASS_COLORS))]
        legend += [Patch(facecolor=ERROR_COLORS[index], label=name) for index, name in
                   ((1, "Prostate FP"), (2, "Prostate FN"), (3, "Other wrong class"))]
        figure.legend(handles=legend, loc="lower center", ncol=5, fontsize=8)
        figure.suptitle(f"{case}: shared resampled XYZ grid\n{selection}; source-axis display, no reorientation", fontsize=11)
        figure.subplots_adjust(top=.85, bottom=.15, hspace=.35, wspace=.08)
        figure.savefig(path, dpi=120)
    finally:
        figure.clear()
    return {"path": str(path), "z_indices": slices, "slice_selection": selection,
            "display": "all arrays use [:,:,z].T, origin=lower; physical XY aspect ratio",
            "error_codes": {"0": "correct", "1": "prostate_false_positive",
                            "2": "prostate_false_negative", "3": "other_class_mismatch"}}


def process_pair(pair, model, device, policies, stride, output_dir, split):
    """Infer and report one pair; no full-volume arrays are retained in results."""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = perf_counter()
    normalized, original_shape = prepare_volume(pair, policies)
    preprocessing_seconds = perf_counter() - start
    model.eval()
    with torch.inference_mode():
        prediction, evidence = evaluation.sliding_window_predict(
            model, normalized.mri, device, patch_shape=evaluation.PATCH_SHAPE, stride=stride)
    if prediction.dtype != np.uint8:
        raise ValueError("Full-volume prediction must preserve uint8 class IDs.")
    start = perf_counter()
    counts = evaluation.confusion_matrix(prediction, normalized.segmentation)
    metrics_seconds = perf_counter() - start
    case = f"{pair.patient_id}_Week{pair.week}"
    start = perf_counter()
    panels = save_visualization(output_dir / f"{case}_panels.png", normalized.mri,
                                normalized.segmentation, prediction, normalized.affine, case)
    visualization_seconds = perf_counter() - start
    evidence = dict(evidence)
    evidence["inference_input_shape_xyz"] = evidence.pop("original_shape_xyz")
    return {"event": "volume", "status": "success", "case": case,
            "patient_id": pair.patient_id, "week": pair.week, "split": split,
            "original_shape_xyz": original_shape, "resampled_shape_xyz": list(normalized.mri.shape),
            "prediction_shape_xyz": list(prediction.shape), "prediction_dtype": str(prediction.dtype),
            "mri_dtype": str(normalized.mri.dtype), "mask_dtype": str(normalized.segmentation.dtype),
            "affine": normalized.affine.tolist(), "target_spacing_mm": list(normalized.target_spacing_mm),
            "mask_header_unit": normalized.segmentation_header_spatial_unit,
            "mask_unit_inferred": normalized.segmentation_unit_inferred,
            "predicted_classes": [int(value) for value in np.unique(prediction)],
            "metrics": evaluation.metrics_from_confusion(counts), "confusion_counts": counts.tolist(),
            "preprocessing_seconds": preprocessing_seconds, "metrics_seconds": metrics_seconds,
            "visualization_seconds": visualization_seconds, "panels": panels,
            "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
            "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
            **evidence}


def output_path(value, dataset_root):
    """Require a new external run directory, never overwriting prior outputs."""
    path = Path(value).expanduser().resolve()
    source = Path(dataset_root).expanduser().resolve()
    workspace = Path(__file__).resolve().parents[2]
    if (path.is_relative_to(source) or source.is_relative_to(path) or path.is_relative_to(workspace)
            or any((parent / ".git").exists() for parent in (path, *path.parents))):
        raise ValueError("Output directory must be outside the source dataset and Git/source workspace.")
    if path.exists() or not path.parent.is_dir():
        raise ValueError("Output directory must be new and have an existing parent; existing outputs are never overwritten.")
    return path


def main(argv=None):
    args = parse_args(argv)
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint file is missing: {checkpoint}")
    output = output_path(args.output_dir, args.dataset_root)
    device = torch.device(args.device)
    if device.type == "cuda" and (not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available()):
        raise RuntimeError("CUDA prediction requires an allocated Slurm GPU session; use --device cpu for CPU checks.")
    configure_reproducibility(3710, device)
    pairs = select_pairs(discover_dataset(args.dataset_root), args.split, args.cases, allow_test=args.allow_test)
    # Incompatible checkpoints fail before any output directory is created.
    model, checkpoint_metadata = load_model(args.model, checkpoint, device)
    policies = evaluation._local_policies()
    checkpoint_digest = hashlib.sha256()
    with checkpoint.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            checkpoint_digest.update(chunk)
    setup = {"event": "setup", "model": MODEL_NAMES[args.model], "checkpoint": str(checkpoint),
             "checkpoint_sha256": checkpoint_digest.hexdigest(), "checkpoint_metadata": checkpoint_metadata,
             "dataset_root": str(args.dataset_root.expanduser().resolve()), "split": args.split,
             "test_opt_in": args.allow_test, "selected_volumes": len(pairs), "all_split": args.all_split,
             "patch_shape_dhw": list(evaluation.PATCH_SHAPE), "stride": args.stride,
             "aggregation": "uniform_mean_softmax_probabilities_then_argmax", "policies": policies,
             "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
             "torch_version": str(torch.__version__), "cuda_build": torch.version.cuda,
             "reproducibility": "best_effort_warn_only", "output_dir": str(output),
             "metric_grid": "shared resampled XYZ; no inverse resampling"}
    output.mkdir()
    evaluation._emit(setup)
    records, successes, failures = [], [], []
    start = perf_counter()
    for pair in pairs:
        try:
            record = process_pair(pair, model, device, policies, args.stride, output, args.split)
        except Exception as exc:
            record = {"event": "volume", "status": "failed", "case": f"{pair.patient_id}_Week{pair.week}",
                      "patient_id": pair.patient_id, "week": pair.week, "split": args.split,
                      "exception_type": type(exc).__name__, "exception_message": str(exc)}
        records.append(record)
        (successes if record["status"] == "success" else failures).append(record)
        evaluation._emit(record)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    passed = not failures and len(successes) == len(pairs)
    summary = {"event": "summary", "status": "PASS" if passed else "FAIL", "split": args.split,
               "successful": len(successes), "failed": len(failures), "selected_volumes": len(pairs),
               "full_split_completed": bool(args.all_split and passed), "failed_cases": failures,
               "session_seconds": perf_counter() - start, **evaluation.summarize(successes),
               "metric_scope": "successful cases only; both-absent classes excluded from macro averages"}
    result_path = output / "results.json"
    with result_path.open("x", encoding="utf-8") as handle:
        json.dump({"schema_version": 1, "setup": setup, "volumes": records, "summary": summary},
                  handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    evaluation._emit(summary)
    evaluation._emit({"event": "result", "status": "PASS" if passed else "FAIL", "results": str(result_path)})
    return 0 if passed else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)
