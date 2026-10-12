"""Optional, single-case CPU benchmark of production preprocessing versus cache read.

Run from the repository root as
``python -B -m recognition.hipmri_3d_improved_unet_hard.benchmark_k019_preprocessing_cache``.
The only write is one temporary, uncompressed NumPy archive below --cache-parent.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import sys
import tempfile
from time import perf_counter

# The production functions below use NumPy/SciPy on CPU. Hide any GPU before
# importing scientific libraries, including when this file is imported for a test.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from .dataset import DEFAULT_DATASET_ROOT, discover_dataset


CASE = ("K019", 1)
REPETITIONS = 5


def _cache_parent(value: Path, dataset_root: Path) -> Path:
    """Require a pre-existing personal directory away from source data."""

    parent = value.expanduser().resolve()
    home = Path.home().resolve()
    source = dataset_root.expanduser().resolve()
    if not parent.is_dir() or parent == home or not parent.is_relative_to(home):
        raise ValueError("--cache-parent must be an existing subdirectory of the user's home.")
    if parent.is_relative_to(source) or source.is_relative_to(parent):
        raise ValueError("--cache-parent must be separate from the HipMRI dataset root.")
    repository = Path(__file__).resolve().parents[2]
    if parent.is_relative_to(repository) or repository.is_relative_to(parent):
        raise ValueError("--cache-parent must be separate from the project repository.")
    if any((ancestor / ".git").exists() for ancestor in (parent, *parent.parents)):
        raise ValueError("--cache-parent must not be inside a Git worktree.")
    return parent


def _select_train_case(splits):
    matches = [pair for pair in splits["train"] if (pair.patient_id, pair.week) == CASE]
    if len(matches) != 1:
        raise ValueError("Frozen Train must contain exactly one K019_Week1 pair.")
    return matches[0]


def _canonical_json(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))


def _prepare(pair, config, *, np, load, resample, normalize, policy_metadata):
    """Use production stages, capturing small source metadata before release."""
    stages = {}
    total_start = perf_counter()
    start = perf_counter()
    loaded = load(pair)
    stages["load"] = perf_counter() - start
    source = {
        "mri_affine": loaded.mri_affine.tolist(),
        "segmentation_affine": loaded.segmentation_affine.tolist(),
        "voxel_spacing": list(loaded.voxel_spacing),
        "axis_codes": list(loaded.axis_codes),
        "mri_stored_dtype": str(loaded.mri_stored_dtype),
        "segmentation_stored_dtype": str(loaded.segmentation_stored_dtype),
    }
    start = perf_counter()
    resampled = resample(loaded, config["target_spacing_mm"],
                         max_estimated_bytes=config["max_estimated_bytes"])
    stages["resample"] = perf_counter() - start
    del loaded
    start = perf_counter()
    normalized = normalize(resampled)
    stages["normalize"] = perf_counter() - start
    del resampled
    if (normalized.pair.patient_id, normalized.pair.week) != CASE:
        raise ValueError("Production pipeline changed the selected case identity.")
    # Never asdict() a volume object: that would copy its full-sized arrays.
    fields = (
        "target_spacing_mm", "original_shape", "input_labels", "output_labels",
        "input_label_counts", "output_label_counts", "input_voxel_volume_mm3",
        "output_voxel_volume_mm3", "input_class_volumes_mm3",
        "output_class_volumes_mm3", "spatial_unit", "segmentation_header_spatial_unit",
        "segmentation_unit_inferred",
    )
    metadata = {field: getattr(normalized, field) for field in fields}
    metadata.update(schema_version=1, patient_id=pair.patient_id, week=pair.week,
                    split="train", array_convention="XYZ", source=source,
                    source_paths={"mri": str(pair.mri_path), "mask": str(pair.segmentation_path)},
                    normalization=asdict(normalized.normalization),
                    policy=policy_metadata or {"resampling": config})
    return normalized, metadata, stages, perf_counter() - total_start


def _array_digest(array, np):
    # Preserve logical XYZ values irrespective of C/Fortran storage layout.
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest()


def _fingerprint(mri, mask, affine, metadata, np):
    if (mri.ndim != 3 or mask.shape != mri.shape or mri.dtype != np.float32
            or mask.dtype != np.uint8 or affine.shape != (4, 4)
            or affine.dtype != np.float64 or not np.isfinite(mri).all()
            or not np.isfinite(affine).all()):
        raise ValueError("Invalid full-volume shapes, dtypes or finite values.")
    labels = tuple(int(v) for v in np.unique(mask))
    if not set(labels).issubset(range(6)) or 5 not in labels:
        raise ValueError("Mask must contain prostate label 5 and only IDs 0-5.")
    counts = tuple(int(np.count_nonzero(mask == label)) for label in range(6))
    if labels != tuple(metadata["output_labels"]) or counts != tuple(metadata["output_label_counts"]):
        raise ValueError("Mask labels/counts disagree with spatial label metadata.")
    if (metadata["patient_id"], metadata["week"], metadata["split"], metadata["array_convention"]) != ("K019", 1, "train", "XYZ"):
        raise ValueError("Cache identity, split or axis convention is incompatible.")
    if not np.allclose(np.linalg.norm(affine[:3, :3], axis=0),
                       metadata["target_spacing_mm"], rtol=0, atol=1e-5):
        raise ValueError("Shared affine spacing differs from metadata.")
    return {
        "shape_xyz": list(mri.shape), "mri_dtype": str(mri.dtype),
        "mask_dtype": str(mask.dtype), "affine_dtype": str(affine.dtype),
        "labels": list(labels), "mri_sha256": _array_digest(mri, np),
        "mask_sha256": _array_digest(mask, np), "affine_sha256": _array_digest(affine, np),
        "metadata_json": _canonical_json(metadata),
    }


def _read_cache(archive, np):
    # Materialise all arrays AND parse JSON inside the read timer.
    with np.load(archive, allow_pickle=False) as cached:
        if set(cached.files) != {"mri", "mask", "affine", "metadata_json"}:
            raise ValueError("Cache archive has unexpected or missing entries.")
        mri, mask, affine = cached["mri"], cached["mask"], cached["affine"]
        encoded = cached["metadata_json"]
        if encoded.shape != () or encoded.dtype.kind != "U":
            raise ValueError("Cache metadata must be a Unicode JSON scalar, not pickle.")
        metadata = json.loads(str(encoded.item()))
    return mri, mask, affine, metadata


def _summary(values):
    if len(values) != REPETITIONS or any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError("Expected five successful finite positive timing measurements.")
    return {"count": len(values), "median_seconds": statistics.median(values),
            "min_seconds": min(values), "max_seconds": max(values)}


def _peak_rss():
    try:
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform.startswith("linux"):
            value *= 1024
        elif sys.platform != "darwin":
            return {"bytes": None, "reason": "Unsupported ru_maxrss units"}
        return {"bytes": int(value), "scope": "cumulative whole-process peak RSS, not GPU VRAM"}
    except ImportError:
        return {"bytes": None, "reason": "resource unavailable on this platform",
                "scope": "cumulative whole-process peak RSS, not GPU VRAM"}


def _benchmark_pair(pair, cache_parent, config, *, np, load, resample, normalize,
                    policy_metadata=None):
    """One initial preparation, one archive, five alternating trials per arm."""
    if (pair.patient_id, pair.week) != CASE:
        raise ValueError("Benchmark is restricted to K019_Week1.")
    kwargs = dict(np=np, load=load, resample=resample, normalize=normalize,
                  policy_metadata=policy_metadata)
    normalized, metadata, initial_stages, initial_total = _prepare(pair, config, **kwargs)
    start = perf_counter()
    reference = _fingerprint(normalized.mri, normalized.segmentation, normalized.affine, metadata, np)
    initial_integrity = perf_counter() - start
    expected_bytes = normalized.mri.nbytes + normalized.segmentation.nbytes + normalized.affine.nbytes
    if shutil.disk_usage(cache_parent).free < 2 * expected_bytes + 1048576:
        raise OSError("Insufficient free disk space for one archive plus planning headroom; quota may differ.")
    trials = []
    temporary_dir = None
    try:
        with tempfile.TemporaryDirectory(prefix="hipmri_k019_cache_", dir=cache_parent) as name:
            temporary_dir = Path(name)
            archive = temporary_dir / "K019_Week1.npz"
            start = perf_counter()
            np.savez(archive, mri=normalized.mri, mask=normalized.segmentation,
                     affine=normalized.affine, metadata_json=np.asarray(_canonical_json(metadata)))
            write_seconds = perf_counter() - start
            cache_bytes = archive.stat().st_size
            blocks = getattr(archive.stat(), "st_blocks", None)
            disk_allocated_bytes = None if blocks is None else blocks * 512
            start = perf_counter()
            cached_mri, cached_mask, cached_affine, cached_metadata = _read_cache(archive, np)
            first_read_seconds = perf_counter() - start
            start = perf_counter()
            checks = {
                "mri_exact": np.array_equal(cached_mri, normalized.mri),
                "mask_exact": np.array_equal(cached_mask, normalized.segmentation),
                "affine_exact": np.array_equal(cached_affine, normalized.affine),
                "metadata_exact": _canonical_json(cached_metadata) == _canonical_json(metadata),
            }
            if not all(checks.values()) or _fingerprint(cached_mri, cached_mask, cached_affine, cached_metadata, np) != reference:
                raise ValueError(f"Initial cache round-trip integrity failure: {checks}")
            first_integrity_seconds = perf_counter() - start
            del normalized, metadata, cached_mri, cached_mask, cached_affine, cached_metadata
            gc.collect()
            # Initial preparation/read are the warm-ups, reported separately.
            for repeat in range(REPETITIONS):
                order = ("preprocessing", "cache_read") if repeat % 2 == 0 else ("cache_read", "preprocessing")
                for arm in order:
                    if arm == "preprocessing":
                        current, current_metadata, stages, seconds = _prepare(pair, config, **kwargs)
                        arrays = (current.mri, current.segmentation, current.affine)
                    else:
                        start = perf_counter()
                        mri, mask, affine, current_metadata = _read_cache(archive, np)
                        seconds = perf_counter() - start
                        arrays = (mri, mask, affine)
                        stages = None
                    start = perf_counter()
                    if _fingerprint(*arrays, current_metadata, np) != reference:
                        raise ValueError(f"Integrity mismatch: {arm}, repetition {repeat + 1}.")
                    integrity_seconds = perf_counter() - start
                    trials.append({"repetition": repeat + 1, "arm": arm,
                                   "seconds": seconds, "stage_seconds": stages,
                                   "integrity_seconds": integrity_seconds,
                                   "preparation_plus_integrity_seconds": seconds + integrity_seconds})
                    del arrays, current_metadata
                    if arm == "preprocessing":
                        del current
                    else:
                        del mri, mask, affine
                    gc.collect()
    except Exception as exc:
        if temporary_dir is not None and temporary_dir.exists():
            raise RuntimeError(f"Benchmark failed and temporary cleanup failed: {temporary_dir}") from exc
        raise RuntimeError(f"{type(exc).__name__}: {exc}; temporary_cache_removed=True") from exc
    if temporary_dir.exists():
        raise RuntimeError(f"Temporary cache cleanup failed: {temporary_dir}")
    summaries = {}
    for arm in ("preprocessing", "cache_read"):
        arm_trials = [trial for trial in trials if trial["arm"] == arm]
        summaries[arm] = _summary([trial["seconds"] for trial in arm_trials])
        summaries[arm]["integrity"] = _summary([trial["integrity_seconds"] for trial in arm_trials])
        summaries[arm]["preparation_plus_integrity"] = _summary(
            [trial["preparation_plus_integrity_seconds"] for trial in arm_trials])
    for value in (initial_total, initial_integrity, write_seconds, first_read_seconds,
                  first_integrity_seconds, *initial_stages.values()):
        if not math.isfinite(value) or value <= 0:
            raise ValueError("Initial/write/read/integrity timings must be finite and positive.")
    return {
        "event": "benchmark", "status": "PASS", "case": "K019_Week1", "split": "train",
        "initial_preprocessing_seconds": initial_total, "initial_stage_seconds": initial_stages,
        "initial_integrity_seconds": initial_integrity,
        "cache_write_seconds": write_seconds, "cache_write_policy": "buffered write/close; no fsync; not durable-storage timing",
        "initial_cache_read_seconds": first_read_seconds,
        "initial_reload_integrity_seconds": first_integrity_seconds,
        "trials": trials, "summaries": summaries,
        "observed_read_over_preprocessing_median_ratio": summaries["cache_read"]["median_seconds"] / summaries["preprocessing"]["median_seconds"],
        "cache_bytes": cache_bytes, "disk_allocated_bytes": disk_allocated_bytes,
        "array_payload_bytes": expected_bytes, "shape_xyz": reference["shape_xyz"],
        "mri_dtype": reference["mri_dtype"], "mask_dtype": reference["mask_dtype"],
        "labels": reference["labels"], "metadata": json.loads(reference["metadata_json"]),
        "equivalence": checks, "repeated_integrity": "PASS", "temporary_cache_removed": True,
        "peak_resident_memory": _peak_rss(),
        "timing_scope": "Full-volume CPU preparation versus materialised npz reads; common integrity checks timed separately; no patch or GPU work",
        "limitations": "Initial preparation/read serve as warm-ups; OS filesystem cache uncontrolled, no verified cold-disk measurement or actual training-speed claim; RSS cumulative",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--cache-parent", type=Path, required=True,
                        help="existing personal subdirectory for one auto-removed cache")
    args = parser.parse_args(argv)
    try:
        parent = _cache_parent(args.cache_parent, args.dataset_root)
        import numpy as np
        import nibabel
        import scipy
        from .volume_io import load_volume_pair
        from .resampling import resample_volume_pair
        from .intensity_normalization import normalize_volume_pair

        directory = Path(__file__).parent
        configs = {name: json.loads((directory / f"{name}_config.json").read_text(encoding="utf-8"))
                   for name in ("resampling", "normalization", "patch_dataset")}
        policy = {
            "configs": configs,
            "versions": {"numpy": np.__version__, "nibabel": nibabel.__version__,
                         "scipy": scipy.__version__},
            "source_sha256": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                              for name in ("dataset.py", "volume_io.py", "resampling.py",
                                           "intensity_normalization.py")},
        }
        pair = _select_train_case(discover_dataset(args.dataset_root))
        result = _benchmark_pair(
            pair, parent, configs["resampling"], np=np, load=load_volume_pair,
            resample=resample_volume_pair, normalize=normalize_volume_pair,
            policy_metadata=policy,
        )
        print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
        return 0
    except Exception as exc:
        print(json.dumps({"event": "benchmark", "status": "FAIL",
                          "case": "K019_Week1", "exception_type": type(exc).__name__,
                          "exception_message": str(exc)}, sort_keys=True), flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
