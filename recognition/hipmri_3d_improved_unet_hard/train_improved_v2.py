"""Isolated V2: frequency-weighted CE, unchanged Dice/model/sampling/optimizer.

The orchestration mirrors train_improved.py deliberately to protect its source
hashes. V2 starts fresh; only V2 completed-epoch checkpoints can resume.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from time import perf_counter
import numpy as np
import torch
from torch.utils.data import DataLoader
from .dataset import APPROVED_PATIENTS, DEFAULT_DATASET_ROOT
from . import train as baseline
from .train import configure_reproducibility, atomic_checkpoint, checkpoint_path, _emit
from . import train_improved as original
from .improved_modules import ImprovedUNet3D

EXPERIMENT_ID = "improved_v2_wce_sqrt_v1"
CONFIG_PATH = Path(__file__).with_name("improved_v2_config.json")


def experiment_settings():
    """Frozen, evidence-derived weights; editing the JSON is not a tuning API."""
    settings = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    counts = [69380520, 246261463, 34110362, 15115893, 4618511, 5379171]
    frequencies = [n / sum(counts) for n in counts]
    denominator = sum(math.sqrt(f) for f in frequencies)
    weights = [f ** -0.5 / denominator for f in frequencies]
    expected = {
        "experiment_id": EXPERIMENT_ID, "epochs": 10,
        "class_names": ["Background", "Body", "Bones", "Bladder", "Rectum", "Prostate"],
        "exposure_file": "patch_full_650736.jsonl",
        "exposure_sha256": "028a495c41e14bec4855bad5fd55c1dcea51472c890285d7c499f5f4d93b22a1",
        "training_patch_count": 1430, "voxel_counts": counts,
        "frequency_exponent": -0.5, "weight_normalization": "sum(frequency * weight) = 1",
        "ce_reduction": "sum(weight[target] * unweighted_voxel_ce) / voxel_count",
        "ce_weight": 0.5, "dice_weight": 0.5,
    }
    if set(settings) != set(expected) | {"class_weights"} or any(settings[k] != v for k, v in expected.items()):
        raise ValueError("V2 configuration differs from the approved frozen experiment")
    actual = settings["class_weights"]
    if (len(actual) != 6 or any(not isinstance(w, (int, float)) or not math.isfinite(w)
                              or not math.isclose(w, e, rel_tol=1e-14) for w, e in zip(actual, weights))):
        raise ValueError("V2 class weights disagree with authenticated Train voxel counts")
    return settings


SETTINGS = experiment_settings()
CLASS_WEIGHTS = tuple(SETTINGS["class_weights"])
HYPERPARAMETERS = dict(original.HYPERPARAMETERS, experiment_id=EXPERIMENT_ID,
                      weighted_ce=SETTINGS, planned_epochs=10)
SOURCE_FILES = (*original.SOURCE_FILES, "train_improved_v2.py", "validate_improved_v2.py",
                "validate.py", "validate_improved.py", "improved_v2_config.json",
                "resampling_config.json", "normalization_config.json", "patch_dataset_config.json")


def source_hashes():
    directory = Path(__file__).parent
    return {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in SOURCE_FILES}


def config_hash(config):
    content = {key: value for key, value in config.items() if key != "config_sha256"}
    return hashlib.sha256(json.dumps(content, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def validate_config(config):
    """Authenticate identity, complete configuration digest and all protected sources."""
    experiment_settings()
    if config.get("experiment_id") != EXPERIMENT_ID or config.get("hyperparameters") != HYPERPARAMETERS:
        raise ValueError("Incompatible V2 experiment identity/hyperparameters")
    if config.get("config_sha256") != config_hash(config):
        raise ValueError("V2 configuration hash differs")
    if config.get("source_sha256") != source_hashes():
        raise ValueError("V2 source/configuration file hashes differ")
    directory = Path(__file__).parent
    policies = {name: json.loads((directory / f"{name}_config.json").read_text(encoding="utf-8"))
                for name in ("resampling", "normalization", "patch_dataset")}
    if config.get("policies") != policies or config.get("steps_per_epoch") != 143:
        raise ValueError("V2 preprocessing or epoch policy differs")
    pairs = config.get("train_pairs", [])
    if (len(pairs) != 143 or any(not isinstance(p, list) or len(p) != 4 for p in pairs)
            or len({(p[0], p[1]) for p in pairs}) != 143
            or {p[0] for p in pairs} != set(APPROVED_PATIENTS["train"])):
        raise ValueError("V2 frozen Training manifest differs")


def resume_checkpoint(path, model, optimizer, *, config, generator, device):
    validate_config(config)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    epoch = payload.get("completed_epochs")
    if (type(epoch) is not int or not 1 <= epoch <= 10
            or payload.get("epoch_result", {}).get("expected_steps") != 143):
        raise ValueError("V2 resume requires a completed epoch within the ten-epoch budget")
    del payload
    # Original helper compares the ENTIRE config before restoring state/RNG.
    return baseline.resume_checkpoint(path, model, optimizer, config=config,
                                      generator=generator, device=device)


def head_loss(logits, target, *, class_weights=CLASS_WEIGHTS):
    """Return total, weighted CE, ORIGINAL Dice and ordinary CE.

    WCE = sum_i w[target_i] * -log(p_i[target_i]) / N (fixed voxel count).
    Reuse baseline validation/Dice verbatim, including absent-class policy.
    The ordinary CE is logged for context, not used again in the objective.
    """
    _, ordinary_ce, dice = baseline.combined_loss(logits, target)
    weights = torch.as_tensor(class_weights, dtype=logits.dtype, device=logits.device)
    if weights.shape != (6,) or not bool(torch.isfinite(weights).all()) or bool((weights <= 0).any()):
        raise ValueError("Expected six finite positive class weights")
    voxel_ce = torch.nn.functional.cross_entropy(logits, target, reduction="none")
    weighted_ce = (voxel_ce * weights[target]).mean()
    total = 0.5 * weighted_ce + 0.5 * dice
    if not bool(torch.isfinite(total)):
        raise ValueError("V2 combined loss is non-finite")
    return total, weighted_ce, dice, ordinary_ce


def combined_loss(outputs, target, *, class_weights=CLASS_WEIGHTS):
    """Same three heads, full-grid targets, interpolation and 1:.5:.25 weights."""
    main, auxiliary = outputs
    spatial = main.shape[2:]
    if len(auxiliary) != 2 or any(tuple(a.shape) != (main.shape[0], 6, *(s // scale for s in spatial))
                                  for a, scale in zip(auxiliary, (2, 4))):
        raise ValueError("Expected half- and quarter-resolution six-class auxiliary logits")
    terms = [head_loss(main, target, class_weights=class_weights)]
    for logits in auxiliary:
        logits = torch.nn.functional.interpolate(logits, size=spatial, mode="trilinear", align_corners=False)
        terms.append(head_loss(logits, target, class_weights=class_weights))
    return tuple(sum(weight * term[index] for weight, term in zip((1., .5, .25), terms)) / 1.75
                 for index in range(4))


def v2_checkpoint_path(value, dataset_root):
    path = checkpoint_path(value, dataset_root)
    if not path.name.startswith("improved_unet_v2_"):
        raise ValueError("V2 checkpoint filename must start with improved_unet_v2_")
    if path.exists():
        payload = torch.load(path, map_location="cpu", weights_only=True)
        validate_config(payload.get("config", {}))
    return path



def run_epoch(model, loader, optimizer, device, *, diagnostic=False,
              expected_patch_shape=None, progress=None):
    """Train all loader steps, or exactly one diagnostic step without completion."""
    if len(loader) < 1:
        raise ValueError("Training loader must not be empty.")
    model.train()
    start = perf_counter()
    sums = [0.0, 0.0, 0.0, 0.0]
    steps = 0
    diagnostic_evidence = {}
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
        before = [p.detach().clone() for p in model.parameters()] if diagnostic else None
        optimizer.zero_grad(set_to_none=True)
        losses = combined_loss(model(image, deep_supervision=True), target)
        losses[0].backward()
        for name, parameter in model.named_parameters():
            if parameter.requires_grad and (parameter.grad is None or not bool(torch.isfinite(parameter.grad).all())):
                raise RuntimeError(f"Missing or non-finite gradient: {name}")
        optimizer.step()
        if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
            raise RuntimeError("Optimizer produced a non-finite parameter.")
        if diagnostic:
            changed = sum(not torch.equal(old, new.detach())
                          for old, new in zip(before, model.parameters()))
            if changed == 0:
                raise RuntimeError("Diagnostic optimizer step changed no parameters")
            diagnostic_evidence = {
                "finite_gradient_parameter_tensors": sum(p.requires_grad for p in model.parameters()),
                "changed_parameter_tensors": changed,
            }
            del before
        values = [float(loss.detach().item()) for loss in losses]
        sums = [a + b for a, b in zip(sums, values)]
        steps += 1
        if progress:
            progress({"event": "step", "step": steps, "expected_steps": len(loader),
                      "loss": values[0], "weighted_ce": values[1], "soft_dice_loss": values[2], "ordinary_ce": values[3]})
        if diagnostic:
            break
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    completed = not diagnostic and steps == len(loader)
    if not diagnostic and not completed:
        raise RuntimeError(f"Incomplete epoch: processed {steps}/{len(loader)} steps.")
    return {"processed_steps": steps, "expected_steps": len(loader), "epoch_completed": completed,
            "mean_loss": sums[0] / steps, "mean_weighted_ce": sums[1] / steps, "mean_ordinary_ce": sums[3] / steps,
            "mean_soft_dice_loss": sums[2] / steps, "runtime_seconds": perf_counter() - start,
            **diagnostic_evidence}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--checkpoint", type=Path, required=True, help="explicit external last-checkpoint file")
    parser.add_argument("--resume", type=Path, help="completed-epoch checkpoint only")
    parser.add_argument("--epochs", type=int, default=10, help="fixed controlled experiment budget: 10 completed epochs")
    parser.add_argument("--diagnostic-one-step", action="store_true", help="one update, no checkpoint or completed epoch")
    args = parser.parse_args(argv)
    if args.epochs != 10 or (args.diagnostic_one_step and args.resume):
        parser.error("V2 fixes --epochs at 10; diagnostic mode cannot resume.")
    output = v2_checkpoint_path(args.checkpoint, args.dataset_root)
    resume = v2_checkpoint_path(args.resume, args.dataset_root) if args.resume else None
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
              "gpu": torch.cuda.get_device_name(device), "cuda_device_count": torch.cuda.device_count(),
              "experiment_id": EXPERIMENT_ID}
    config["config_sha256"] = config_hash(config)
    validate_config(config)
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
    _emit({"event": "setup", "experiment_id": EXPERIMENT_ID, "config": config, "checkpoint": str(output),
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
