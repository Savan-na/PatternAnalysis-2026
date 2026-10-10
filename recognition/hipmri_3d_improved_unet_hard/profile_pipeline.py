"""Read-only, CPU-only timing and coverage audit of approved preprocessing.

Run as a package module. JSON Lines go to stdout; no files or caches are
written. Peak RSS is cumulative for this process, never a stage allocation.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import statistics
import sys
import time

from .dataset import (
    DEFAULT_DATASET_ROOT, EXPECTED_PATIENT_COUNTS, EXPECTED_VOLUME_COUNTS,
    discover_dataset, parse_volume_filename,
)


DEFAULT_CASES = ("K019_Week1", "S035_Week0", "B040_Week0")
STAGES = ("loading", "resampling", "normalization")


def _case_argument(value):
    try:
        return parse_volume_filename(value + "_LFOV.nii.gz", "LFOV")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--cases", nargs="+", type=_case_argument,
                           metavar="PATIENT_WeekN", help="default: three small acceptance cases")
    selection.add_argument("--full-coverage", action="store_true",
                           help="explicitly process all 143 train and 38 validation volumes")
    args = parser.parse_args(argv)
    if not args.full_coverage:
        args.cases = args.cases or [_case_argument(case) for case in DEFAULT_CASES]
        if len(set(args.cases)) != len(args.cases):
            parser.error("Duplicate case selections are not allowed.")
    return args


def _emit(record):
    print(json.dumps(record, allow_nan=False, sort_keys=True), flush=True)


def _peak_memory():
    try:
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform.startswith("linux"):
            value *= 1024
        elif sys.platform != "darwin":
            return {"bytes": None, "reason": "ru_maxrss units unsupported on this platform"}
        return {"bytes": int(value), "scope": "process-lifetime cumulative peak RSS",
                "source": "getrusage(RUSAGE_SELF).ru_maxrss"}
    except ImportError:
        return {"bytes": None, "reason": "resource unavailable; RSS not measured",
                "scope": "process-lifetime cumulative peak RSS"}


def _select(splits, args):
    allowed = [(split, pair) for split in ("train", "validation") for pair in splits[split]]
    if args.full_coverage:
        for split in ("train", "validation"):
            pairs = splits[split]
            if (len(pairs) != EXPECTED_VOLUME_COUNTS[split]
                    or len({p.patient_id for p in pairs}) != EXPECTED_PATIENT_COUNTS[split]):
                raise ValueError(f"Frozen {split} coverage does not match approved counts.")
        return allowed
    lookup = {(p.patient_id, p.week): (split, p) for split, p in allowed}
    test_ids = {p.patient_id for p in splits["test"]}
    selected = []
    for key in args.cases:
        if key[0] in test_ids:
            raise ValueError(f"{key[0]}_Week{key[1]}: test-patient processing is forbidden.")
        if key not in lookup:
            raise ValueError(f"{key[0]}_Week{key[1]}: case not found in train/validation.")
        selected.append(lookup[key])
    return selected


def _profile_pair(split, pair, config, api):
    """Retain only this case's arrays; return scalar metadata even on failure."""
    np, load, resample, normalize, affine_atol = api
    record = {"event": "case", "case": f"{pair.patient_id}_Week{pair.week}",
              "patient_id": pair.patient_id, "week": pair.week, "split": split,
              "stage_seconds": {name: None for name in STAGES}}
    loaded = resampled = normalized = None
    start = time.perf_counter()
    stage = "loading"
    stage_start = start
    try:
        loaded = load(pair)
        record["stage_seconds"][stage] = time.perf_counter() - stage_start
        record["original_shape_xyz"] = list(loaded.mri.shape)
        record["original_dtypes"] = {"mri": str(loaded.mri.dtype),
                                      "mask": str(loaded.segmentation.dtype)}
        if loaded.mri.shape != loaded.segmentation.shape or not np.allclose(
                loaded.mri_affine, loaded.segmentation_affine, rtol=0, atol=affine_atol):
            raise ValueError("Loaded MRI/mask spatial grids differ.")
        stage = "resampling"
        stage_start = time.perf_counter()
        resampled = resample(loaded, config["target_spacing_mm"],
                             max_estimated_bytes=config["max_estimated_bytes"])
        record["stage_seconds"][stage] = time.perf_counter() - stage_start
        loaded = None  # Do not retain decoded source arrays during normalisation.
        record["resampled_shape_xyz"] = list(resampled.mri.shape)
        stage = "normalization"
        stage_start = time.perf_counter()
        normalized = normalize(resampled)
        record["stage_seconds"][stage] = time.perf_counter() - stage_start
        stage = "integrity_checks"
        if (normalized.mri.shape != normalized.segmentation.shape
                or normalized.mri.shape != resampled.mri.shape):
            raise ValueError("Output MRI/mask shapes differ or normalization changed shape.")
        if not np.array_equal(normalized.affine, resampled.affine):
            raise ValueError("Normalization changed the shared affine.")
        if not np.allclose(np.linalg.norm(normalized.affine[:3, :3], axis=0),
                           config["target_spacing_mm"], rtol=0, atol=1e-5):
            raise ValueError("Output affine spacing differs from approved target.")
        if normalized.mri.dtype != np.float32 or not np.isfinite(normalized.mri).all():
            raise ValueError("Normalized MRI must be finite float32.")
        if normalized.segmentation.dtype != np.uint8:
            raise ValueError("Segmentation must remain uint8.")
        if not np.array_equal(normalized.segmentation, resampled.segmentation):
            raise ValueError("Normalization changed segmentation values.")
        labels = tuple(int(v) for v in np.unique(normalized.segmentation))
        if not set(labels).issubset(range(6)):
            raise ValueError(f"Invalid segmentation IDs: {labels}.")
        if labels != normalized.output_labels:
            raise ValueError("Label metadata does not match output array.")
        lost = set(normalized.input_labels) - set(labels) - {0}
        if lost:
            raise ValueError(f"Foreground classes disappeared: {sorted(lost)}.")
        # Avoid bincount's full-volume uint8-to-int64 temporary conversion.
        counts = tuple(int(np.count_nonzero(normalized.segmentation == label))
                       for label in range(6))
        if counts != normalized.output_label_counts:
            raise ValueError("Output label counts do not match recorded counts.")
        record.update(status="success", mri_dtype=str(normalized.mri.dtype),
                      mask_dtype=str(normalized.segmentation.dtype), labels_present=list(labels),
                      prostate_present=5 in labels, input_label_counts=list(normalized.input_label_counts),
                      output_label_counts=list(counts), geometry_integrity="PASS",
                      label_integrity="PASS", finite_mri=True,
                      shared_affine=normalized.affine.tolist(),
                      spatial_unit=normalized.spatial_unit,
                      mask_header_unit=normalized.segmentation_header_spatial_unit,
                      mask_unit_inferred=normalized.segmentation_unit_inferred)
    except Exception as exc:
        if stage in STAGES and record["stage_seconds"][stage] is None:
            record["stage_seconds"][stage] = time.perf_counter() - stage_start
        record.update(status="failed", failed_stage=stage,
                      exception_type=type(exc).__name__, exception_message=str(exc))
    finally:
        record["total_processing_seconds"] = time.perf_counter() - start
        loaded = resampled = normalized = None
    # Exceptions/tracebacks and arrays are not returned or retained between cases.
    return record


def _timing_summary(records):
    completed = [r for r in records if r["status"] == "success"]
    result = {}
    for name in (*STAGES, "total_processing"):
        values = [r["total_processing_seconds"] if name == "total_processing"
                  else r["stage_seconds"][name] for r in completed]
        result[name] = ({"count": len(values), "sum_seconds": sum(values),
                         "mean_seconds": statistics.mean(values),
                         "median_seconds": statistics.median(values),
                         "min_seconds": min(values), "max_seconds": max(values)}
                        if values else {"count": 0})
    return result


def main(argv=None):
    args = _arguments(argv)
    session_start = time.perf_counter()
    try:
        # Lazy imports allow help/argument checks without scientific dependencies.
        import numpy as np
        import nibabel
        import scipy
        from .volume_io import load_volume_pair, AFFINE_ATOL
        from .resampling import resample_volume_pair
        from .intensity_normalization import normalize_volume_pair

        directory = Path(__file__).parent
        configs = {name: json.loads((directory / f"{name}_config.json").read_text(encoding="utf-8"))
                   for name in ("resampling", "normalization", "patch_dataset")}
        splits = discover_dataset(args.dataset_root)
        selected = _select(splits, args)
        _emit({"event": "setup", "mode": "full_coverage" if args.full_coverage else "targeted",
               "dataset_root": str(args.dataset_root), "selected_count": len(selected),
               "test_volumes_processed": 0, "configs": configs,
               "versions": {"numpy": np.__version__, "nibabel": nibabel.__version__,
                            "scipy": scipy.__version__},
               "timing_scope": "CPU production calls; total includes integrity checks, excludes cleanup",
               "discovery_scope": "all frozen filenames checked; only train/validation arrays decoded"})
        records = []
        for split, pair in selected:
            record = _profile_pair(split, pair, configs["resampling"],
                                   (np, load_volume_pair, resample_volume_pair, normalize_volume_pair, AFFINE_ATOL))
            gc.collect()
            record["peak_resident_memory"] = _peak_memory()
            records.append(record)
            _emit(record)
        failed = [r for r in records if r["status"] == "failed"]
        by_split = {}
        for split in ("train", "validation"):
            subset = [r for r in records if r["split"] == split]
            successes = [r for r in subset if r["status"] == "success"]
            by_split[split] = {"attempted": len(subset), "successful": len(successes),
                               "failed": len(subset) - len(successes),
                               "successful_patients": len({r["patient_id"] for r in successes}),
                               "timings": _timing_summary(subset)}
        coverage_ok = args.full_coverage and not failed and all(
            by_split[s]["successful"] == EXPECTED_VOLUME_COUNTS[s]
            and by_split[s]["successful_patients"] == EXPECTED_PATIENT_COUNTS[s]
            for s in ("train", "validation"))
        _emit({"event": "summary", "successful": len(records) - len(failed), "failed": len(failed),
               "failed_cases": [{k: r[k] for k in ("case", "split", "failed_stage", "exception_type", "exception_message")}
                                for r in failed],
               "by_split": by_split, "completed_case_timings": _timing_summary(records),
               "full_181_volume_coverage_verified": bool(coverage_ok),
               "session_wall_seconds": time.perf_counter() - session_start,
               "peak_resident_memory": _peak_memory(),
               "runtime_projection": "Not computed; targeted cases need not represent all volumes.",
               "interpretation": "Preprocessing executability only; no clinical or model performance validation."})
        if failed or (args.full_coverage and not coverage_ok):
            return 1
        _emit({"event": "result", "status": "PASS",
               "scope": "181 train/validation volumes" if coverage_ok else "selected cases only"})
        return 0
    except Exception as exc:
        _emit({"event": "setup_failure", "exception_type": type(exc).__name__,
               "exception_message": str(exc), "full_181_volume_coverage_verified": False})
        return 1


if __name__ == "__main__":
    sys.exit(main())
