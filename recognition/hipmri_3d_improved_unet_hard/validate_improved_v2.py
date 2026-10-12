"""Strict V2 checkpoint loading; reuse unchanged Validation inference and metrics."""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
from time import perf_counter
import numpy as np
import torch
from .dataset import DEFAULT_DATASET_ROOT, discover_dataset
from .improved_modules import ImprovedUNet3D
from .train_improved_v2 import HYPERPARAMETERS, source_hashes, validate_config, EXPERIMENT_ID
from .validate import (_case, _emit, _local_policies, _process_pair, select_validation,
                       summarize, PATCH_SHAPE)

def load_checkpoint_model(path, device):
    """Read the fixed V2 Epoch 10 endpoint; never restore optimizer or RNG."""
    checkpoint = torch.load(Path(path).expanduser(), map_location="cpu", weights_only=True)
    config = checkpoint.get("config", {})
    validate_config(config)
    hp = config.get("hyperparameters", {})
    expected = {"model": "ImprovedUNet3D", "in_channels": 1, "out_channels": 6, "base_channels": 16}
    epoch = checkpoint.get("completed_epochs")
    if (checkpoint.get("schema_version") != 1 or type(epoch) is not int or epoch != 10
            or config.get("steps_per_epoch") != 143 or checkpoint.get("total_steps") != epoch * 143
            or checkpoint.get("epoch_result", {}).get("epoch_completed") is not True
            or checkpoint["epoch_result"].get("processed_steps") != 143
            or checkpoint["epoch_result"].get("expected_steps") != 143
            or any(hp.get(key) != value for key, value in expected.items())):
        raise ValueError("V2 validation requires default ImprovedUNet3D at Epoch 10 / 1430 updates.")
    if hp != HYPERPARAMETERS:
        raise ValueError("Improved training hyperparameters/deep supervision policy differ")
    if config.get("policies") != _local_policies():
        raise ValueError("Checkpoint preprocessing policies differ from current approved JSON policies.")
    for name, actual in source_hashes().items():
        if config.get("source_sha256", {}).get(name) != actual:
            raise ValueError(f"Checkpoint production source hash differs: {name}.")
    model = ImprovedUNet3D()
    model.load_state_dict(checkpoint["model"], strict=True)
    if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise ValueError("Checkpoint contains non-finite model parameters.")
    metadata = {"experiment_id": EXPERIMENT_ID, "config_sha256": config["config_sha256"], "completed_epochs": epoch, "total_steps": checkpoint["total_steps"],
                "training_versions": config.get("versions"), "training_hyperparameters": hp}
    del checkpoint
    return model.to(device).eval(), metadata


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
    _emit({"event": "setup", "experiment_id": EXPERIMENT_ID, "checkpoint": str(args.checkpoint.expanduser().resolve()),
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
