"""Separate Improved U-Net training: same data/optimizer, weighted deep supervision."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from time import perf_counter
import numpy as np
import torch
from torch.utils.data import DataLoader
from .dataset import APPROVED_PATIENTS, DEFAULT_DATASET_ROOT
from . import train as baseline
from .train import configure_reproducibility, atomic_checkpoint, resume_checkpoint, checkpoint_path, _emit
from .improved_modules import ImprovedUNet3D

HYPERPARAMETERS = dict(baseline.HYPERPARAMETERS, model="ImprovedUNet3D",
    deep_supervision_weights=[1.0, 0.5, 0.25], deep_supervision_divisor=1.75,
    auxiliary_interpolation="trilinear_align_corners_false")
SOURCE_FILES = ("train_improved.py", "train.py", "improved_modules.py", "modules.py", "dataset.py",
                "patch_dataset.py", "volume_io.py", "resampling.py", "intensity_normalization.py")


def source_hashes():
    directory = Path(__file__).parent
    return {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in SOURCE_FILES}


def combined_loss(outputs, target):
    """Weighted total/CE/Dice; resize logits only, retaining full-grid integer labels."""
    main, auxiliary = outputs
    spatial = main.shape[2:]
    if len(auxiliary) != 2 or any(tuple(a.shape) != (main.shape[0], 6, *(s // scale for s in spatial))
                                  for a, scale in zip(auxiliary, (2, 4))):
        raise ValueError("Expected half- and quarter-resolution six-class auxiliary logits")
    terms = [baseline.combined_loss(main, target)]
    for logits in auxiliary:
        logits = torch.nn.functional.interpolate(logits, size=spatial, mode="trilinear", align_corners=False)
        terms.append(baseline.combined_loss(logits, target))
    return tuple(sum(weight * term[index] for weight, term in zip((1., .5, .25), terms)) / 1.75
                 for index in range(3))


def improved_checkpoint_path(value, dataset_root):
    path = checkpoint_path(value, dataset_root)
    if not path.name.startswith("improved_unet_"):
        raise ValueError("Improved checkpoint filename must start with improved_unet_; Standard paths are forbidden")
    if path.exists():
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("config", {}).get("hyperparameters", {}).get("model") != "ImprovedUNet3D":
            raise ValueError("Refusing a non-Improved checkpoint path")
    return path

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
        losses = combined_loss(model(image, deep_supervision=True), target)
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
    output = improved_checkpoint_path(args.checkpoint, args.dataset_root)
    resume = improved_checkpoint_path(args.resume, args.dataset_root) if args.resume else None
    if output.exists() and (resume != output or args.diagnostic_one_step):
        raise ValueError("Checkpoint already exists; explicit --resume is required to continue it.")
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("Real training requires an allocated Slurm CUDA compute session; no CPU fallback.")
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
              "source_sha256": source_hashes(),
              "versions": {"torch": str(torch.__version__), "numpy": np.__version__,
                           "nibabel": nibabel.__version__, "scipy": scipy.__version__,
                           "cuda_build": torch.version.cuda},
              "gpu": torch.cuda.get_device_name(device), "cuda_device_count": torch.cuda.device_count()}
    generator = torch.Generator().manual_seed(3710)
    loader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=True,
                        generator=generator, drop_last=False)
    model = ImprovedUNet3D().to(device)
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
