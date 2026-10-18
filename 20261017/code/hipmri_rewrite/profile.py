"""One Train-volume A100 capacity probe, no optimizer step or checkpoint.

Requires an allocated Slurm CUDA session. This is a memory feasibility check,
not training and not a Validation/Test evaluation.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .data import downsample, load_case, to_3d_model_arrays, zscore_nonzero
from .losses import ce_dice_loss
from .models import ImprovedUNet3D, UNet2D, UNet3D
from .splits import discover


MODELS = {"2d": UNet2D, "3d": UNet3D, "improved3d": ImprovedUNet3D}


def profile(root: str | Path, model_name: str, case_index: int = 0) -> dict:
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("Capacity probe requires an allocated Slurm CUDA session")
    if model_name not in MODELS or case_index < 0:
        raise ValueError("Unknown model or negative case index")
    cases = discover(root)["train"]
    if case_index >= len(cases):
        raise ValueError("Train case index outside discovered set")
    volume = zscore_nonzero(downsample(load_case(cases[case_index])))
    image, target, shape_xyz = to_3d_model_arrays(volume)
    model = MODELS[model_name]().to("cuda").train()
    if model_name == "2d":
        z = shape_xyz[2] // 2
        image_tensor = torch.from_numpy(np.ascontiguousarray(image[z:z + 1, None])).to("cuda")
        target_tensor = torch.from_numpy(np.ascontiguousarray(target[z:z + 1, :shape_xyz[1], :shape_xyz[0]])).to("cuda")
    else:
        image_tensor = torch.from_numpy(np.ascontiguousarray(image[None, None])).to("cuda")
        target_tensor = torch.from_numpy(np.ascontiguousarray(target[None, :shape_xyz[2], :shape_xyz[1], :shape_xyz[0]])).to("cuda")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = perf_counter()
    logits = model(image_tensor)
    if model_name == "2d":
        logits = logits[..., :shape_xyz[1], :shape_xyz[0]]
    else:
        logits = logits[..., :shape_xyz[2], :shape_xyz[1], :shape_xyz[0]]
    loss = ce_dice_loss(logits, target_tensor)
    loss.backward()
    torch.cuda.synchronize()
    return {
        "kind": "capacity_probe_no_optimizer_step",
        "case": cases[case_index].identifier,
        "model": model_name,
        "input_shape": list(image_tensor.shape),
        "parameters": sum(p.numel() for p in model.parameters()),
        "finite_loss": bool(torch.isfinite(loss).item()),
        "finite_gradients": all(p.grad is not None and bool(torch.isfinite(p.grad).all().item()) for p in model.parameters()),
        "seconds_forward_backward": perf_counter() - start,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model", choices=tuple(MODELS), required=True)
    parser.add_argument("--case-index", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = profile(args.dataset_root, args.model, args.case_index)
    serialized = json.dumps(result, indent=2, sort_keys=True)
    if args.output is None:
        print(serialized)
    else:
        project = Path(__file__).resolve().parents[2]
        output = args.output.resolve()
        if not output.is_relative_to(project):
            parser.error("Output must be inside the 20261017 project")
        if output.exists():
            parser.error("Refusing to overwrite a capacity result")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized + "\n", encoding="utf-8")
        print(output)


if __name__ == "__main__":
    main()
