"""Read-only full-volume Validation inference; no training or Test-array access.

Average overlapping softmax probabilities uniformly before full-volume argmax.
Both-absent class metrics are None (JSON null), excluded from macro averages.
Patient metrics pool confusion counts across Weeks; report both pooled overall
metrics and equally weighted volume/patient macro metrics on the resampled grid.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
from itertools import product
import json
import os
from pathlib import Path
import sys
from time import perf_counter

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from .dataset import APPROVED_PATIENTS, DEFAULT_DATASET_ROOT, discover_dataset, parse_volume_filename


CLASS_NAMES = ("Background", "Body", "Bones", "Bladder", "Rectum", "Prostate")
PATCH_SHAPE = (64, 64, 64)


def window_starts(length, patch, stride):
    """Append the last legal start, guaranteeing high-edge coverage without gaps."""
    if any(type(v) is not int or v < 1 for v in (length, patch, stride)) or stride > patch:
        raise ValueError("Length/patch/stride must be positive integers; stride cannot exceed patch.")
    length = max(length, patch)  # Small axes receive high-side inference-only zero padding.
    starts = list(range(0, length - patch + 1, stride))
    if starts[-1] != length - patch:
        starts.append(length - patch)
    return tuple(starts)


def sliding_window_predict(model, mri_xyz, device, *, patch_shape=PATCH_SHAPE, stride=32,
                           max_accumulator_bytes=536870912):
    """One GPU patch at a time; return uint8 XYZ prediction and coverage evidence.

    High-side zero padding is cropped off before restoring XYZ. The original
    affine/origin is unchanged. The guard bounds score/count array bytes only,
    not total process RAM or GPU peak memory. CPU accumulation avoids keeping
    full-volume logits on the GPU.
    """
    if (not isinstance(mri_xyz, np.ndarray) or mri_xyz.ndim != 3
            or any(size < 1 for size in mri_xyz.shape) or mri_xyz.dtype != np.float32
            or not np.isfinite(mri_xyz).all()):
        raise ValueError("Expected a finite nonempty float32 MRI in XYZ order.")
    if (len(patch_shape) != 3 or any(type(v) is not int or v < 16 or v % 16 for v in patch_shape)
            or type(max_accumulator_bytes) is not int or max_accumulator_bytes < 1):
        raise ValueError("Patch dimensions must be positive multiples of 16; accumulator cap must be positive.")
    original_dhw = mri_xyz.shape[::-1]
    working_shape = tuple(max(size, patch) for size, patch in zip(original_dhw, patch_shape))
    starts = [window_starts(size, patch, stride) for size, patch in zip(working_shape, patch_shape)]
    estimate = int(np.prod(working_shape, dtype=object)) * 28  # Six float32 scores + float32 counts.
    if estimate > max_accumulator_bytes:
        raise MemoryError(f"Score/count allocation {estimate} exceeds guard {max_accumulator_bytes} bytes.")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = perf_counter()
    mri_dhw = mri_xyz.transpose(2, 1, 0)
    padding = tuple((0, working - size) for size, working in zip(original_dhw, working_shape))
    if any(high for _, high in padding):
        mri_dhw = np.pad(mri_dhw, padding, mode="constant", constant_values=0)
    scores = np.zeros((6, *working_shape), dtype=np.float32)
    counts = np.zeros(working_shape, dtype=np.float32)
    model.eval()
    windows = 0
    with torch.inference_mode():
        for origin in product(*starts):
            crop = tuple(slice(pos, pos + width) for pos, width in zip(origin, patch_shape))
            image = torch.from_numpy(np.ascontiguousarray(mri_dhw[crop])).unsqueeze(0).unsqueeze(0).to(device)
            logits = model(image)
            if (tuple(logits.shape) != (1, 6, *patch_shape) or logits.dtype != torch.float32
                    or not bool(torch.isfinite(logits).all())):
                raise ValueError("Model must return finite float32 [1,6,D,H,W] logits.")
            probabilities = logits.softmax(dim=1)[0].cpu().numpy()
            scores[(slice(None), *crop)] += probabilities
            counts[crop] += 1
            windows += 1
            del image, logits, probabilities
    if not np.all(counts > 0) or not np.isfinite(scores).all():
        raise RuntimeError("Incomplete or non-finite sliding-window accumulation.")
    np.divide(scores, counts[None], out=scores)
    # Slice-wise argmax avoids a full-volume int64 prediction temporary.
    prediction = np.empty(original_dhw, dtype=np.uint8)
    for depth in range(original_dhw[0]):
        prediction[depth] = scores[:, depth, :original_dhw[1], :original_dhw[2]].argmax(axis=0)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return prediction.transpose(2, 1, 0), {
        "window_count": windows, "coverage_min": int(counts.min()), "coverage_max": int(counts.max()),
        "working_shape_dhw": list(working_shape), "original_shape_xyz": list(mri_xyz.shape),
        "accumulator_array_bytes": estimate, "inference_seconds": perf_counter() - start,
        "aggregation": "uniform_mean_softmax_probabilities_then_argmax",
    }


def confusion_matrix(prediction, truth):
    """Rows are ground truth, columns prediction; count all aligned XYZ voxels."""
    if (not isinstance(prediction, np.ndarray) or not isinstance(truth, np.ndarray)
            or truth.ndim != 3 or prediction.shape != truth.shape or truth.size == 0
            or prediction.dtype.kind not in "iu" or truth.dtype.kind not in "iu"):
        raise ValueError("Prediction/truth must be matching nonempty 3D integer arrays.")
    if any(int(a.min()) < 0 or int(a.max()) > 5 for a in (prediction, truth)):
        raise ValueError("Prediction/truth labels must lie within 0-5.")
    counts = np.zeros((6, 6), dtype=np.int64)
    for z in range(truth.shape[2]):
        codes = truth[:, :, z].astype(np.int64) * 6 + prediction[:, :, z]
        counts += np.bincount(codes.ravel(), minlength=36).reshape(6, 6)
    return counts


def metrics_from_confusion(counts):
    counts = np.asarray(counts)
    if counts.shape != (6, 6) or counts.dtype.kind not in "iu" or np.any(counts < 0):
        raise ValueError("Expected nonnegative integer 6x6 confusion counts.")
    rows = []
    for label, name in enumerate(CLASS_NAMES):
        tp = int(counts[label, label])
        true_count = int(counts[label].sum())
        predicted_count = int(counts[:, label].sum())
        denominator = true_count + predicted_count
        union = denominator - tp
        rows.append({"class_id": label, "structure": name, "true_voxels": true_count,
                     "predicted_voxels": predicted_count, "intersection": tp,
                     "dice": None if denominator == 0 else 2 * tp / denominator,
                     "iou": None if union == 0 else tp / union})
    return rows


def macro_metrics(metric_lists):
    """Equal weight per evaluable volume or patient; no both-absent inflation."""
    result = []
    for label, name in enumerate(CLASS_NAMES):
        values = [rows[label] for rows in metric_lists if rows[label]["dice"] is not None]
        result.append({"class_id": label, "structure": name, "evaluable_units": len(values),
                       "dice": None if not values else sum(v["dice"] for v in values) / len(values),
                       "iou": None if not values else sum(v["iou"] for v in values) / len(values)})
    return result


def summarize(records):
    """Records contain small confusion matrices, never volume arrays."""
    patients = {}
    pooled = np.zeros((6, 6), dtype=np.int64)
    for record in records:
        counts = np.asarray(record["confusion_counts"], dtype=np.int64)
        pooled += counts
        patients.setdefault(record["patient_id"], []).append(counts)
    patient_rows = [{"patient_id": patient, "volumes": len(matrices),
                     "metrics": metrics_from_confusion(np.sum(matrices, axis=0))}
                    for patient, matrices in sorted(patients.items())]
    return {"patients": patient_rows, "overall_pooled": metrics_from_confusion(pooled),
            "overall_patient_macro": macro_metrics([r["metrics"] for r in patient_rows]),
            "overall_volume_macro": macro_metrics([r["metrics"] for r in records]),
            "absent_class_policy": "both_absent_null_and_excluded; one_sided_absence_zero"}


def _local_policies():
    directory = Path(__file__).parent
    return {name: json.loads((directory / f"{name}_config.json").read_text(encoding="utf-8"))
            for name in ("resampling", "normalization", "patch_dataset")}


def load_checkpoint_model(path, device):
    """Read the completed baseline checkpoint; never restore optimizer or RNG."""
    from .modules import StandardUNet3D
    checkpoint = torch.load(Path(path).expanduser(), map_location="cpu", weights_only=True)
    config = checkpoint.get("config", {})
    hp = config.get("hyperparameters", {})
    expected = {"model": "StandardUNet3D", "in_channels": 1, "out_channels": 6, "base_channels": 16}
    epoch = checkpoint.get("completed_epochs")
    if (checkpoint.get("schema_version") != 1 or type(epoch) is not int or epoch < 1
            or config.get("steps_per_epoch") != 143 or checkpoint.get("total_steps") != epoch * 143
            or checkpoint.get("epoch_result", {}).get("epoch_completed") is not True
            or checkpoint["epoch_result"].get("processed_steps") != 143
            or any(hp.get(key) != value for key, value in expected.items())):
        raise ValueError("Checkpoint must contain the default StandardUNet3D at a completed 143-step epoch boundary.")
    if config.get("policies") != _local_policies():
        raise ValueError("Checkpoint preprocessing policies differ from current approved JSON policies.")
    directory = Path(__file__).parent
    for name in ("modules.py", "dataset.py", "volume_io.py", "resampling.py", "intensity_normalization.py"):
        actual = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        if config.get("source_sha256", {}).get(name) != actual:
            raise ValueError(f"Checkpoint production source hash differs: {name}.")
    model = StandardUNet3D()
    model.load_state_dict(checkpoint["model"], strict=True)
    if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise ValueError("Checkpoint contains non-finite model parameters.")
    metadata = {"completed_epochs": epoch, "total_steps": checkpoint["total_steps"],
                "training_versions": config.get("versions"), "training_hyperparameters": hp}
    del checkpoint
    return model.to(device).eval(), metadata


def _case(value):
    try:
        return parse_volume_filename(value + "_LFOV.nii.gz", "LFOV")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def select_validation(splits, case=None):
    pairs = splits["validation"]
    if len(pairs) != 38 or {p.patient_id for p in pairs} != set(APPROVED_PATIENTS["validation"]):
        raise ValueError("Validation must contain exactly 38 volumes from the six approved patients.")
    if case is None:
        return pairs
    matches = tuple(p for p in pairs if (p.patient_id, p.week) == case)
    if len(matches) != 1:
        raise ValueError("Selected case is not exactly one approved Validation pair; Train/Test processing is forbidden.")
    return matches


def _process_pair(pair, model, device, spacing, memory_cap, stride):
    record = {"event": "volume", "case": f"{pair.patient_id}_Week{pair.week}",
              "patient_id": pair.patient_id, "week": pair.week, "split": "validation"}
    try:
        from .volume_io import load_volume_pair
        from .resampling import resample_volume_pair
        from .intensity_normalization import normalize_volume_pair
        torch.cuda.reset_peak_memory_stats(device)
        start = perf_counter()
        loaded = load_volume_pair(pair)
        original_shape = list(loaded.mri.shape)
        resampled = resample_volume_pair(loaded, spacing, max_estimated_bytes=memory_cap)
        del loaded
        normalized = normalize_volume_pair(resampled)
        del resampled
        record["preprocessing_seconds"] = perf_counter() - start
        prediction, evidence = sliding_window_predict(model, normalized.mri, device, stride=stride)
        # Distinguish the inference input grid from the original source NIfTI grid.
        evidence = dict(evidence)
        evidence["inference_input_shape_xyz"] = evidence.pop("original_shape_xyz")
        start = perf_counter()
        counts = confusion_matrix(prediction, normalized.segmentation)
        record.update(status="success", original_shape_xyz=original_shape,
                      resampled_shape_xyz=list(normalized.mri.shape), affine=normalized.affine.tolist(),
                      target_spacing_mm=list(normalized.target_spacing_mm),
                      mask_header_unit=normalized.segmentation_header_spatial_unit,
                      mask_unit_inferred=normalized.segmentation_unit_inferred,
                      metrics=metrics_from_confusion(counts), confusion_counts=counts.tolist(), **evidence)
        record["metrics_seconds"] = perf_counter() - start
        record["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        record["peak_gpu_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
        del prediction, normalized
    except Exception as exc:
        record.update(status="failed", exception_type=type(exc).__name__, exception_message=str(exc))
    return record


def _emit(record):
    print(json.dumps(record, sort_keys=True, allow_nan=False), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--stride", type=int, default=32)
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument("--case", type=_case, metavar="PATIENT_WeekN")
    scope.add_argument("--all-validation", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.stride <= 64:
        parser.error("stride must be in 1..64 for complete 64-cube coverage")
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("Real validation requires an allocated CUDA Slurm compute session; no CPU fallback.")
    device = torch.device("cuda:0")
    torch.manual_seed(3710)
    torch.cuda.manual_seed_all(3710)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    pairs = select_validation(discover_dataset(args.dataset_root), args.case)
    model, checkpoint_metadata = load_checkpoint_model(args.checkpoint, device)
    policies = _local_policies()
    import nibabel
    import scipy
    _emit({"event": "setup", "checkpoint": str(args.checkpoint.expanduser().resolve()),
           "checkpoint_metadata": checkpoint_metadata, "selected_volumes": len(pairs),
           "patch_shape_dhw": list(PATCH_SHAPE), "stride": args.stride,
           "gpu": torch.cuda.get_device_name(device), "reproducibility": "best_effort_warn_only",
           "versions": {"torch": str(torch.__version__), "numpy": np.__version__,
                        "nibabel": nibabel.__version__, "scipy": scipy.__version__, "cuda_build": torch.version.cuda},
           "metric_grid": "resampled XYZ, original physical axis directions; no inverse resampling"})
    records, failures = [], []
    start = perf_counter()
    for pair in pairs:
        record = _process_pair(pair, model, device, policies["resampling"]["target_spacing_mm"],
                               policies["resampling"]["max_estimated_bytes"], args.stride)
        _emit(record)
        (records if record["status"] == "success" else failures).append(record)
        gc.collect()
        torch.cuda.empty_cache()
    _emit({"event": "summary", "successful": len(records), "failed": len(failures),
           "failed_cases": failures, "full_38_volume_validation_completed": bool(args.all_validation and not failures and len(records) == 38),
           "validation_loop_seconds": perf_counter() - start, **summarize(records),
           "metric_scope": "successful cases only; macro averages exclude both-absent classes"})
    if failures or len(records) != len(pairs):
        return 1
    _emit({"event": "result", "status": "PASS", "scope": "all 38 Validation volumes" if args.all_validation else "selected Validation case"})
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)
