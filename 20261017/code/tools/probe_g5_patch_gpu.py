"""Probe both 3D models on real Train patches; no optimizer step or checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
from pathlib import Path
from time import perf_counter


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "code"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--case-id", default="B006_Week0")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or socket.gethostname().startswith("login"):
        parser.error("Run inside an allocated Slurm GPU compute job")
    output = args.output.resolve()
    if not output.is_relative_to(PROJECT) or output.exists():
        parser.error("Output must be a new JSON file inside 20261017")
    if not 1 <= args.batch_size <= 4:
        parser.error("Batch size must be 1..4 for this probe")

    import numpy as np
    import torch

    from hipmri_rewrite.data import downsample, load_case, zscore_nonzero
    from hipmri_rewrite.losses import ce_dice_loss
    from hipmri_rewrite.models import ImprovedUNet3D, UNet3D
    from hipmri_rewrite.sampling import (PATCH_XYZ, ExposureRecorder,
                                        crop_for_3d_model, crop_origin)
    from hipmri_rewrite.splits import discover

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA device allocated")
    cases = {case.identifier: case for case in discover(args.dataset_root)["train"]}
    if args.case_id not in cases:
        parser.error("Case ID must belong to the frozen Train split")
    case = cases[args.case_id]
    volume = zscore_nonzero(downsample(load_case(case)))
    images, masks = [], []
    recorder = ExposureRecorder()
    origins = []
    for draw in range(args.batch_size):
        origin = crop_origin(volume.mask, PATCH_XYZ, "uniform", seed=371017,
                             epoch=0, case_id=case.identifier, draw=draw)
        image, mask = crop_for_3d_model(volume, origin)
        images.append(image)
        masks.append(mask)
        origins.append(origin)
        recorder.add(mask, case.patient)
    image_array = np.stack(images)
    mask_array = np.stack(masks)
    results = []
    for name, constructor in (("standard_3d", UNet3D),
                              ("improved_3d", ImprovedUNet3D)):
        torch.manual_seed(371017)
        torch.cuda.manual_seed_all(371017)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model = constructor().to("cuda").train()
        x = torch.from_numpy(np.ascontiguousarray(image_array)).to("cuda")
        y = torch.from_numpy(np.ascontiguousarray(mask_array)).to("cuda")
        torch.cuda.synchronize()
        start = perf_counter()
        logits = model(x)
        loss = ce_dice_loss(logits, y, ce_weight=0.5)
        loss.backward()
        torch.cuda.synchronize()
        elapsed = perf_counter() - start
        gradients = {param_name: float(param.grad.norm().item()) if param.grad is not None else None
                     for param_name, param in model.named_parameters()}
        finite_gradients = all(value is not None and np.isfinite(value)
                               for value in gradients.values())
        result = {"model": name, "parameters": sum(p.numel() for p in model.parameters()),
                  "input_shape": list(x.shape), "output_shape": list(logits.shape),
                  "loss": float(loss.item()), "finite_loss": bool(torch.isfinite(loss).item()),
                  "finite_gradients": finite_gradients,
                  "missing_gradient_parameters": [key for key, value in gradients.items()
                                                  if value is None],
                  "zero_gradient_parameters": [key for key, value in gradients.items()
                                               if value == 0.0],
                  "gradient_norm_l2": float(sum(value * value for value in gradients.values()
                                                if value is not None) ** 0.5),
                  "forward_backward_seconds": elapsed,
                  "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                  "peak_reserved_bytes": int(torch.cuda.max_memory_reserved())}
        results.append(result)
        print(f"{name}: loss={result['loss']:.5f}, gradients_finite={finite_gradients}, "
              f"peak_allocated_GiB={result['peak_allocated_bytes'] / 2**30:.3f}, "
              f"seconds={elapsed:.2f}", flush=True)
        del model, x, y, logits, loss
    record = {"kind": "G5_Train_patch_capacity_probe_no_optimizer_step",
              "job_id": os.environ["SLURM_JOB_ID"], "case": case.identifier,
              "mri_sha256": sha256_file(case.mri), "mask_sha256": sha256_file(case.mask),
              "batch_size": args.batch_size, "patch_shape_xyz": PATCH_XYZ,
              "origins_xyz": origins, "patch_exposure": recorder.summary(),
              "objective": "0.5*unweighted_CE + 0.5*foreground_soft_Dice (probe only)",
              "gpu": torch.cuda.get_device_name(0),
              "gpu_total_bytes": int(torch.cuda.get_device_properties(0).total_memory),
              "python": sys.version, "torch": str(torch.__version__), "cuda_build": torch.version.cuda,
              "numpy": np.__version__,
              "script_sha256": sha256_file(Path(__file__)),
              "sampling_code_sha256": sha256_file(PROJECT / "code/hipmri_rewrite/sampling.py"),
              "data_code_sha256": sha256_file(PROJECT / "code/hipmri_rewrite/data.py"),
              "loss_code_sha256": sha256_file(PROJECT / "code/hipmri_rewrite/losses.py"),
              "results": results}
    if not all(item["finite_loss"] and item["finite_gradients"]
               and not item["zero_gradient_parameters"] for item in results):
        raise RuntimeError("Probe found nonfinite, missing or zero gradients")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Saved {output}; no optimizer step or checkpoint was produced", flush=True)


if __name__ == "__main__":
    main()
