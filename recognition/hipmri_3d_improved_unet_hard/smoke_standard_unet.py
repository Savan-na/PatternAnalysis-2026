"""One real float32 GPU optimisation step; run only on an allocated compute node."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root", type=Path,
        default=Path("/home/groups/comp3710/HipMRI_Study_open"),
        help="Read-only HipMRI dataset root.",
    )
    args = parser.parse_args()

    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Subset

    print(f"PyTorch: {torch.__version__}", flush=True)
    print(f"CUDA build: {torch.version.cuda}", flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Run inside an allocated Rangpur GPU compute session; "
            "check GPU allocation, driver compatibility and CUDA_VISIBLE_DEVICES "
            f"(current value: {os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')!r}). "
            "This script has no CPU training fallback."
        )

    from .modules import StandardUNet3D
    from .patch_dataset import HipMRIPatchDataset

    torch.manual_seed(3710)
    torch.cuda.manual_seed_all(3710)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda:0")
    print(f"GPU: {torch.cuda.get_device_name(device)}", flush=True)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    print(f"GPU memory before step: free={free_bytes} bytes, total={total_bytes} bytes", flush=True)

    # Discovery checks the frozen split; only the selected pair's arrays are decoded.
    dataset = HipMRIPatchDataset(
        "train", args.dataset_root, train_foreground_probability=1.0, seed=3710,
    )
    if dataset.patch_shape_dhw != (64, 64, 64):
        raise ValueError(f"Expected configured patch (64,64,64), got {dataset.patch_shape_dhw}")
    matches = [index for index, pair in enumerate(dataset.pairs)
               if pair.patient_id == "K019" and pair.week == 1]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one K019_Week1 training pair, found {len(matches)}")
    sample_index = matches[0] * dataset.train_patches_per_volume
    loader = DataLoader(Subset(dataset, [sample_index]), batch_size=1, num_workers=0, shuffle=False)
    batch = next(iter(loader))
    image, target = batch["mri"], batch["mask"]
    if image.device.type != "cpu" or target.device.type != "cpu":
        raise ValueError("Dataset preprocessing must produce CPU tensors")
    if tuple(image.shape) != (1, 1, 64, 64, 64) or image.dtype != torch.float32:
        raise ValueError(f"Invalid MRI batch: shape={tuple(image.shape)}, dtype={image.dtype}")
    if tuple(target.shape) != (1, 64, 64, 64) or target.dtype != torch.long:
        raise ValueError(f"Invalid target batch: shape={tuple(target.shape)}, dtype={target.dtype}")
    if not bool(torch.isfinite(image).all()):
        raise ValueError("MRI patch contains non-finite values")
    labels = torch.unique(target).tolist()
    if not set(labels).issubset(range(6)) or 5 not in labels:
        raise ValueError(f"Expected valid labels 0–5 with prostate present, got {labels}")
    if batch["sampling_mode"] != ["prostate"]:
        raise ValueError(f"Expected prostate sampling, got {batch['sampling_mode']}")
    print(
        f"Case: {batch['patient_id'][0]}_Week{batch['week'][0].item()}; "
        f"crop origin DHW: {batch['crop_origin_dhw'][0].tolist()}; labels: {labels}", flush=True,
    )
    print(f"CPU patch: MRI={tuple(image.shape)}, target={tuple(target.shape)}", flush=True)

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    model = StandardUNet3D().to(device)
    model.train()
    image, target = image.to(device), target.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    # Host snapshots verify an update without adding a second model-sized GPU copy.
    trainable = [(name, parameter) for name, parameter in model.named_parameters()
                 if parameter.requires_grad]
    before = {name: parameter.detach().cpu().clone() for name, parameter in trainable}

    logits = model(image)  # Exactly one forward pass, raw logits; no AMP or softmax.
    if tuple(logits.shape) != (1, 6, 64, 64, 64) or logits.dtype != torch.float32:
        raise RuntimeError(f"Invalid logits: shape={tuple(logits.shape)}, dtype={logits.dtype}")
    if not bool(torch.isfinite(logits).all()):
        raise RuntimeError("Logits contain non-finite values")
    loss = nn.CrossEntropyLoss()(logits, target)
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("Cross-entropy loss is not finite")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()  # Exactly one backward pass.
    for name, parameter in trainable:
        if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
            raise RuntimeError(f"Missing or non-finite gradient: {name}")
    print(f"Gradients: all {len(trainable)} trainable parameter tensors present and finite", flush=True)
    optimizer.step()  # Exactly one SGD update.
    if not all(bool(torch.isfinite(parameter).all()) for _, parameter in trainable):
        raise RuntimeError("An updated parameter is non-finite")
    changed = [name for name, parameter in trainable
               if not torch.equal(before[name], parameter.detach().cpu())]
    if not changed:
        raise RuntimeError("SGD step did not change any trainable parameter")

    torch.cuda.synchronize(device)
    print(f"Loss: {loss.item():.8f}", flush=True)
    print(f"Parameter update: {len(changed)}/{len(trainable)} tensors changed", flush=True)
    print(f"GPU allocated: {torch.cuda.memory_allocated(device)} bytes", flush=True)
    print(f"GPU reserved: {torch.cuda.memory_reserved(device)} bytes", flush=True)
    print(f"Peak GPU allocated: {torch.cuda.max_memory_allocated(device)} bytes", flush=True)
    print(f"Peak GPU reserved: {torch.cuda.max_memory_reserved(device)} bytes", flush=True)
    print("PASS: ONE REAL STANDARD 3D U-NET GPU OPTIMISATION STEP", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"FAIL: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        sys.exit(1)
