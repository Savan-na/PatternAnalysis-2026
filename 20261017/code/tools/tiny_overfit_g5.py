"""Train-only two-patch learning check for the 3D arms; no checkpoint saved."""

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
G1_SHA = "6a4c674564a0364f5f56730cbc31e67b3ff9a8ffeae7d9d3dd18b198bbd0e1a1"
CLASS_NAMES = ("Background", "Body", "Bones", "Bladder", "Rectum", "Prostate")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def measure(model, image, target, loss_fn, torch) -> dict:
    """Measure fixed-batch predictions in evaluation mode at one step."""
    model.eval()
    with torch.no_grad():
        logits = model(image)
        loss = loss_fn(logits, target)
        prediction = logits.argmax(dim=1)
        rows = {}
        for label, name in enumerate(CLASS_NAMES):
            truth = target == label
            found = prediction == label
            true_voxels = int(truth.sum().item())
            predicted_voxels = int(found.sum().item())
            overlap = int((truth & found).sum().item())
            rows[name] = {"true_voxels": true_voxels, "predicted_voxels": predicted_voxels,
                          "true_positive_voxels": overlap,
                          "dice": 2 * overlap / (true_voxels + predicted_voxels)
                          if true_voxels + predicted_voxels else None,
                          "recall": overlap / true_voxels if true_voxels else None}
    model.train()
    return {"loss": float(loss.item()), "per_class": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--case-id", default="B006_Week0")
    parser.add_argument("--model", choices=("standard_3d", "improved_3d", "both"),
                        default="both")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--g1", type=Path, default=PROJECT / "logs/audits/train_full_20261018_job671208.json")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or socket.gethostname().startswith("login"):
        parser.error("Run inside an allocated Slurm GPU compute job")
    output = args.output.resolve()
    if not output.is_relative_to(PROJECT) or output.exists():
        parser.error("Output must be a new JSON file inside 20261017")
    if not 1 <= args.steps <= 1000:
        parser.error("Tiny overfit step count must be 1..1000")
    if sha256_file(args.g1) != G1_SHA:
        raise ValueError("G1 Train audit differs from verified source")

    import numpy as np
    import torch

    from hipmri_rewrite.data import downsample, load_case, zscore_nonzero
    from hipmri_rewrite.losses import ce_dice_loss
    from hipmri_rewrite.models import ImprovedUNet3D, UNet3D
    from hipmri_rewrite.sampling import (PATCH_XYZ, crop_for_3d_model, crop_origin)
    from hipmri_rewrite.splits import discover

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA device allocated")
    cases = {case.identifier: case for case in discover(args.dataset_root)["train"]}
    if args.case_id not in cases:
        parser.error("Case ID must belong to frozen Train split")
    case = cases[args.case_id]
    g1 = json.loads(args.g1.read_text(encoding="utf-8"))
    audited = {row["case"]: row for row in g1["cases"]}[case.identifier]
    if (sha256_file(case.mri) != audited["mri_sha256"]
            or sha256_file(case.mask) != audited["mask_sha256"]):
        raise ValueError("Source MRI/Mask changed since G1")
    volume = zscore_nonzero(downsample(load_case(case)))
    images, masks, origins = [], [], []
    for draw, target in enumerate(("Rectum", "Prostate")):
        origin = crop_origin(volume.mask, PATCH_XYZ, target, seed=371017,
                             epoch=0, case_id=case.identifier, draw=draw)
        image, mask = crop_for_3d_model(volume, origin)
        images.append(image)
        masks.append(mask)
        origins.append(origin)
    image_array = np.stack(images)
    mask_array = np.stack(masks)
    true_counts = np.bincount(mask_array.ravel(), minlength=6)
    if true_counts[4] == 0 or true_counts[5] == 0:
        raise ValueError("Tiny overfit batch must contain Rectum and Prostate")
    image = torch.from_numpy(np.ascontiguousarray(image_array)).to("cuda")
    target = torch.from_numpy(np.ascontiguousarray(mask_array)).to("cuda")
    objective = lambda logits, truth: ce_dice_loss(logits, truth, ce_weight=0.5)
    constructors = {"standard_3d": UNet3D, "improved_3d": ImprovedUNet3D}
    selected_models = tuple(constructors) if args.model == "both" else (args.model,)
    records = []
    for name in selected_models:
        constructor = constructors[name]
        torch.manual_seed(371017)
        torch.cuda.manual_seed_all(371017)
        model = constructor().to("cuda").train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)
        snapshots = [{"step": 0, **measure(model, image, target, objective, torch)}]
        started = perf_counter()
        for step in range(1, args.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            logits = model(image)
            loss = objective(logits, target)
            if not bool(torch.isfinite(loss).item()):
                raise ValueError(f"{name} nonfinite loss at step {step}")
            loss.backward()
            if any(param.grad is None or not bool(torch.isfinite(param.grad).all().item())
                   for param in model.parameters()):
                raise ValueError(f"{name} missing or nonfinite gradient at step {step}")
            optimizer.step()
            if step % 10 == 0 or step == args.steps:
                snapshot = {"step": step, **measure(model, image, target, objective, torch)}
                snapshots.append(snapshot)
                print(f"{name} step {step}: fixed-batch loss={snapshot['loss']:.5f}, "
                      f"Rectum Dice={snapshot['per_class']['Rectum']['dice']:.4f}, "
                      f"Prostate Dice={snapshot['per_class']['Prostate']['dice']:.4f}", flush=True)
        torch.cuda.synchronize()
        records.append({"model": name, "parameters": sum(p.numel() for p in model.parameters()),
                        "steps": args.steps, "seconds": perf_counter() - started,
                        "snapshots": snapshots})
        del model, optimizer, logits, loss
        torch.cuda.empty_cache()
    result = {"kind": "G5_two_patch_Train_only_learning_check_no_checkpoint",
              "job_id": os.environ["SLURM_JOB_ID"], "case": case.identifier,
              "mri_sha256": audited["mri_sha256"], "mask_sha256": audited["mask_sha256"],
              "g1_sha256": G1_SHA, "seed": 371017, "patch_shape_xyz": PATCH_XYZ,
              "requested_models": list(selected_models),
              "input_shape": list(image.shape), "origins_xyz": origins,
              "target_order": ["Rectum", "Prostate"],
              "target_voxels": {name: int(true_counts[index])
                                for index, name in enumerate(CLASS_NAMES)},
              "objective": "0.5*unweighted_CE + 0.5*foreground_soft_Dice; check only",
              "optimizer": "Adam(lr=1e-3,weight_decay=0); check only",
              "gpu": torch.cuda.get_device_name(0), "torch": str(torch.__version__),
              "cuda_build": torch.version.cuda, "numpy": np.__version__,
              "script_sha256": sha256_file(Path(__file__)),
              "sampling_code_sha256": sha256_file(PROJECT / "code/hipmri_rewrite/sampling.py"),
              "data_code_sha256": sha256_file(PROJECT / "code/hipmri_rewrite/data.py"),
              "loss_code_sha256": sha256_file(PROJECT / "code/hipmri_rewrite/losses.py"),
              "results": records}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Saved {output}; no checkpoint was produced", flush=True)


if __name__ == "__main__":
    main()
