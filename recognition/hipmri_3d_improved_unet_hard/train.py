"""Minimal float32 training; no validation, cache or best-model selection.

Loss = 0.5 CE + 0.5 (1 - mean foreground soft Dice), labels 1..5.
For each class c, Dice = (2 sum(p_c * t_c) + 1e-5) /
(sum(p_c) + sum(t_c) + 1e-5), summed over batch and spatial axes.
All five classes participate, including target-absent classes: false-positive
probability is penalised; an exactly empty prediction/target has Dice 1.
These soft training terms are not held-out Dice/IoU evaluation metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import tempfile
from time import perf_counter

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .dataset import APPROVED_PATIENTS, DEFAULT_DATASET_ROOT


HYPERPARAMETERS = {
    "seed": 3710, "batch_size": 1, "num_workers": 0, "shuffle": True,
    "dtype": "float32", "optimizer": "AdamW", "learning_rate": 0.001,
    "weight_decay": 0.00001, "betas": [0.9, 0.999], "adam_epsilon": 1e-8,
    "ce_weight": 0.5, "dice_weight": 0.5, "dice_smoothing": 1e-5,
    "dice_classes": [1, 2, 3, 4, 5], "absent_classes": "include_and_penalize_false_positives",
    "dice_reduction": "sum_batch_and_spatial_then_mean_classes",
    "model": "StandardUNet3D", "in_channels": 1, "out_channels": 6, "base_channels": 16,
    "deterministic_algorithms": True, "deterministic_warn_only": True,
    "reproducibility_policy": "best_effort_warn_only", "tf32": False,
}


def combined_loss(logits, target):
    """Return differentiable total, mean CE, and foreground soft Dice loss."""
    if (logits.ndim != 5 or logits.shape[0] < 1 or logits.shape[1] != 6
            or target.shape != (logits.shape[0], *logits.shape[2:])
            or any(size < 1 for size in logits.shape[2:])):
        raise ValueError("Expected logits [B,6,D,H,W] and target [B,D,H,W].")
    if logits.dtype != torch.float32 or target.dtype != torch.long:
        raise ValueError("Loss requires float32 logits and torch.long targets.")
    if logits.device != target.device:
        raise ValueError("Logits and target must share a device.")
    if not bool(torch.isfinite(logits).all()) or bool(((target < 0) | (target > 5)).any()):
        raise ValueError("Non-finite logits or target labels outside 0-5.")
    # Avoid the CUDA NLL spatial reduction; average all voxel losses explicitly.
    ce = nn.functional.cross_entropy(logits, target, reduction="none").mean()
    probabilities = logits.softmax(dim=1)
    scores = []
    for label in range(1, 6):
        prediction = probabilities[:, label]
        truth = (target == label).to(torch.float32)
        scores.append((2 * (prediction * truth).sum() + 1e-5) /
                      (prediction.sum() + truth.sum() + 1e-5))
    dice_loss = 1 - torch.stack(scores).mean()
    total = 0.5 * ce + 0.5 * dice_loss
    if not all(bool(torch.isfinite(v)) for v in (total, ce, dice_loss)):
        raise ValueError("Combined training loss is non-finite.")
    return total, ce, dice_loss


def configure_reproducibility(seed, device):
    """Seed all generators and prefer deterministic kernels; warn when unavailable.

    CUDA MaxPool3D backward may remain nondeterministic. This shared training
    policy does not guarantee bitwise reproducibility on CUDA.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def run_epoch(model, loader, optimizer, device, *, diagnostic=False,
              expected_patch_shape=None, progress=None):
    """Train all loader steps, or exactly one diagnostic step without completion."""
    if len(loader) < 1:
        raise ValueError("Training loader must not be empty.")
    model.train()
    start = perf_counter()
    sums = [0.0, 0.0, 0.0]
    steps = 0
    for batch in loader:
        image, target = batch["mri"], batch["mask"]
        if (image.device.type != "cpu" or target.device.type != "cpu"
                or image.dtype != torch.float32 or target.dtype != torch.long
                or image.ndim != 5 or image.shape[0] != 1 or image.shape[1] != 1
                or target.shape != (1, *image.shape[2:])
                or not bool(torch.isfinite(image).all())):
            raise ValueError("Expected finite CPU float32 [1,1,D,H,W] and long [1,D,H,W].")
        if expected_patch_shape is not None and tuple(image.shape[2:]) != tuple(expected_patch_shape):
            raise ValueError("Training patch shape differs from the approved configuration.")
        image, target = image.to(device), target.to(device)
        optimizer.zero_grad(set_to_none=True)
        losses = combined_loss(model(image), target)
        losses[0].backward()
        for name, parameter in model.named_parameters():
            if parameter.requires_grad and (parameter.grad is None or not bool(torch.isfinite(parameter.grad).all())):
                raise RuntimeError(f"Missing or non-finite gradient: {name}")
        optimizer.step()
        if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
            raise RuntimeError("Optimizer produced a non-finite parameter.")
        values = [float(loss.detach().item()) for loss in losses]
        sums = [a + b for a, b in zip(sums, values)]
        steps += 1
        if progress:
            progress({"event": "step", "step": steps, "expected_steps": len(loader),
                      "loss": values[0], "ce": values[1], "soft_dice_loss": values[2]})
        if diagnostic:
            break
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    completed = not diagnostic and steps == len(loader)
    if not diagnostic and not completed:
        raise RuntimeError(f"Incomplete epoch: processed {steps}/{len(loader)} steps.")
    return {"processed_steps": steps, "expected_steps": len(loader), "epoch_completed": completed,
            "mean_loss": sums[0] / steps, "mean_ce": sums[1] / steps,
            "mean_soft_dice_loss": sums[2] / steps, "runtime_seconds": perf_counter() - start}


def _rng_state(generator, device):
    state = np.random.get_state()
    return {"python": random.getstate(), "numpy": [state[0], state[1].tolist(),
            state[2], state[3], state[4]], "torch_cpu": torch.get_rng_state(),
            "loader": generator.get_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else []}


def atomic_checkpoint(path, model, optimizer, *, completed_epochs, total_steps,
                      config, generator, device, epoch_result):
    """Replace last checkpoint only after an entire epoch; temp file is same-directory."""
    if (not epoch_result["epoch_completed"] or type(completed_epochs) is not int
            or completed_epochs < 1 or total_steps != completed_epochs * config["steps_per_epoch"]
            or epoch_result["processed_steps"] != config["steps_per_epoch"]
            or epoch_result["expected_steps"] != config["steps_per_epoch"]):
        raise ValueError("Cannot checkpoint a partial/diagnostic epoch.")
    payload = {"schema_version": 1, "completed_epochs": completed_epochs,
               "total_steps": total_steps, "config": config, "model": model.state_dict(),
               "optimizer": optimizer.state_dict(), "rng": _rng_state(generator, device),
               "epoch_result": epoch_result}
    path = Path(path)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def resume_checkpoint(path, model, optimizer, *, config, generator, device):
    """Restore an epoch boundary; reject changed experiment/software before applying state."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("schema_version") != 1 or checkpoint.get("config") != config:
        raise ValueError("Checkpoint configuration/schema is incompatible with this run.")
    epoch = checkpoint["completed_epochs"]
    steps = checkpoint["total_steps"]
    if (type(epoch) is not int or epoch < 1 or type(steps) is not int
            or steps != epoch * config["steps_per_epoch"]
            or not checkpoint["epoch_result"]["epoch_completed"]
            or checkpoint["epoch_result"]["processed_steps"] != config["steps_per_epoch"]):
        raise ValueError("Checkpoint is not a valid complete-epoch boundary.")
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    rng = checkpoint["rng"]
    random.setstate(rng["python"])
    state = rng["numpy"]
    np.random.set_state((state[0], np.asarray(state[1], dtype=np.uint32), *state[2:]))
    torch.set_rng_state(rng["torch_cpu"])
    generator.set_state(rng["loader"])
    if device.type == "cuda":
        if len(rng["torch_cuda"]) != torch.cuda.device_count():
            raise ValueError("Checkpoint CUDA RNG device count differs from allocation.")
        torch.cuda.set_rng_state_all(rng["torch_cuda"])
    return epoch, steps


def checkpoint_path(value, dataset_root):
    path = Path(value).expanduser().resolve()
    source = Path(dataset_root).expanduser().resolve()
    repository = Path(__file__).resolve().parents[2]
    if (path.is_relative_to(source) or path.is_relative_to(repository)
            or any((p / ".git").exists() for p in (path.parent, *path.parent.parents))):
        raise ValueError("Checkpoint must be outside the source dataset and Git repository.")
    if not path.parent.is_dir() or path.is_dir():
        raise ValueError("Checkpoint requires an existing output directory and a file path.")
    return path


def _emit(record):
    print(json.dumps(record, sort_keys=True, allow_nan=False), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--checkpoint", type=Path, required=True, help="explicit external last-checkpoint file")
    parser.add_argument("--resume", type=Path, help="completed-epoch checkpoint only")
    parser.add_argument("--epochs", type=int, default=1, help="total desired completed epochs, not additional epochs")
    parser.add_argument("--diagnostic-one-step", action="store_true", help="one update, no checkpoint or completed epoch")
    args = parser.parse_args(argv)
    if args.epochs < 1 or (args.diagnostic_one_step and args.resume):
        parser.error("epochs must be positive; diagnostic mode cannot resume.")
    output = checkpoint_path(args.checkpoint, args.dataset_root)
    resume = checkpoint_path(args.resume, args.dataset_root) if args.resume else None
    if output.exists() and (resume != output or args.diagnostic_one_step):
        raise ValueError("Checkpoint already exists; explicit --resume is required to continue it.")
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("Real training requires an allocated Slurm CUDA compute session; no CPU fallback.")
    from .modules import StandardUNet3D
    from .patch_dataset import HipMRIPatchDataset
    import nibabel
    import scipy

    device = torch.device("cuda:0")
    configure_reproducibility(3710, device)
    dataset = HipMRIPatchDataset("train", args.dataset_root)
    if (len(dataset.pairs) != 143 or len(dataset) != 143
            or {p.patient_id for p in dataset.pairs} != set(APPROVED_PATIENTS["train"])
            or dataset.patch_shape_dhw != (64, 64, 64)
            or dataset.train_patches_per_volume != 1 or dataset.seed != 3710
            or dataset.train_foreground_probability != 0.5):
        raise ValueError("Training Dataset differs from the approved 26-patient/143-volume patch policy.")
    directory = Path(__file__).parent
    policies = {name: json.loads((directory / f"{name}_config.json").read_text(encoding="utf-8"))
                for name in ("resampling", "normalization", "patch_dataset")}
    config = {"hyperparameters": HYPERPARAMETERS, "policies": policies,
              "steps_per_epoch": 143, "dataset_root": str(args.dataset_root.resolve()),
              "train_pairs": [[p.patient_id, p.week, str(p.mri_path), str(p.segmentation_path)] for p in dataset.pairs],
              "source_sha256": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                                for name in ("train.py", "modules.py", "dataset.py", "patch_dataset.py",
                                             "volume_io.py", "resampling.py", "intensity_normalization.py")},
              "versions": {"torch": str(torch.__version__), "numpy": np.__version__,
                           "nibabel": nibabel.__version__, "scipy": scipy.__version__,
                           "cuda_build": torch.version.cuda},
              "gpu": torch.cuda.get_device_name(device), "cuda_device_count": torch.cuda.device_count()}
    generator = torch.Generator().manual_seed(3710)
    loader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=True,
                        generator=generator, drop_last=False)
    model = StandardUNet3D().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.00001,
                                 betas=(0.9, 0.999), eps=1e-8, foreach=False, fused=False)
    completed_epochs, total_steps = 0, 0
    if resume:
        completed_epochs, total_steps = resume_checkpoint(
            resume, model, optimizer, config=config, generator=generator, device=device)
    if not args.diagnostic_one_step and args.epochs <= completed_epochs:
        raise ValueError("--epochs must exceed the checkpoint's completed epoch count.")
    _emit({"event": "setup", "config": config, "checkpoint": str(output),
           "resume_completed_epochs": completed_epochs, "diagnostic": args.diagnostic_one_step})
    torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(completed_epochs, args.epochs):
        dataset.set_epoch(epoch)
        result = run_epoch(model, loader, optimizer, device,
                           diagnostic=args.diagnostic_one_step, expected_patch_shape=(64, 64, 64), progress=_emit)
        total_steps += result["processed_steps"]
        if args.diagnostic_one_step:
            _emit({"event": "diagnostic", **result, "checkpoint_saved": False})
            break
        completed_epochs = epoch + 1
        atomic_checkpoint(output, model, optimizer, completed_epochs=completed_epochs,
                          total_steps=total_steps, config=config, generator=generator,
                          device=device, epoch_result=result)
        _emit({"event": "epoch", "completed_epochs": completed_epochs, "total_steps": total_steps,
               **result, "checkpoint": str(output)})
    _emit({"event": "result", "status": "PASS", "scope": "one-step diagnostic" if args.diagnostic_one_step else "completed training epochs",
           "completed_epochs": completed_epochs, "total_steps": total_steps,
           "checkpoint": None if args.diagnostic_one_step else str(output),
           "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(device),
           "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(device),
           "interpretation": "Training loss/executability only; no validation Dice/IoU or convergence claim."})


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)
