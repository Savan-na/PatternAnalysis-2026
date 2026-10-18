"""Bounded two-patient Train optimizer smoke for the paired G5 candidate.

Two batches per model test patient selection, uniform crops, weighted loss,
gradient updates and exact delivered-mask exposure. No checkpoint is saved.
"""

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


def check_config(config: dict, weight_source: dict) -> None:
    """Refuse a changed candidate or a mismatched Train-derived weight source."""
    if (config["status"] != "candidate_pending_training_smoke_and_runtime_check"
            or config["models"] != ["standard_3d", "improved_3d"]
            or config["train_patients"] != 26
            or config["sampling"]["visit_rule"]
            != "one Train visit per patient per epoch, cycling through shuffled visits"
            or config["sampling"]["patch_rule"] != "uniform"
            or config["sampling"]["batch_size"] != 2
            or config["preprocessing"]["downsample_factor_xyz"] != [2, 2, 2]
            or config["preprocessing"]["train_patch_xyz"] != [64, 64, 32]
            or config["objective"]["ce_fraction"] != 0.5
            or config["objective"]["class_order"] != list(CLASS_NAMES)
            or config["optimizer"] != {"name": "Adam", "learning_rate": 0.001,
                                      "weight_decay": 0.0}):
        raise ValueError("Unexpected G5 candidate configuration")
    weights = [weight_source["class_weights"][name] for name in CLASS_NAMES]
    if config["objective"]["ce_class_weights"] != weights:
        raise ValueError("Candidate weights differ from Train-derived source")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path,
                        default=PROJECT / "code/config/g5_controlled_training_candidate.json")
    parser.add_argument("--g1", type=Path,
                        default=PROJECT / "logs/audits/train_full_20261018_job671208.json")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or socket.gethostname().startswith("login"):
        parser.error("Run inside an allocated Slurm GPU compute job")
    output = args.output.resolve()
    if not output.is_relative_to(PROJECT) or output.exists():
        parser.error("Output must be a new JSON file inside 20261017")
    if sha256_file(args.g1) != G1_SHA:
        raise ValueError("G1 Train audit differs from verified source")

    import numpy as np
    import torch

    from hipmri_rewrite.data import downsample, load_case, zscore_nonzero
    from hipmri_rewrite.losses import ce_dice_loss
    from hipmri_rewrite.models import ImprovedUNet3D, UNet3D
    from hipmri_rewrite.sampling import (PATCH_XYZ, ExposureRecorder,
                                        crop_for_3d_model, crop_origin, epoch_cases)
    from hipmri_rewrite.splits import DEFAULT_SPLIT, discover

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA device allocated")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    weight_path = PROJECT / "code/config/g5_weighted_ce_diagnostic.json"
    if (sha256_file(DEFAULT_SPLIT) != config["split_sha256"]
            or sha256_file(weight_path) != config["objective"]["weight_source_config_sha256"]):
        raise ValueError("Candidate split or Train weight source changed")
    check_config(config, json.loads(weight_path.read_text(encoding="utf-8")))
    cases = discover(args.dataset_root)
    if (len(cases["train"]) != config["train_cases"]
            or len(cases["validation"]) != config["validation_cases"]):
        raise ValueError("Frozen cohort size changed")
    selected = epoch_cases(cases["train"], epoch=0, seed=config["seed"])[:2]
    if len({case.patient for case in selected}) != 2:
        raise ValueError("Smoke must select two distinct Train patients")
    g1_cases = {row["case"]: row for row in json.loads(args.g1.read_text(encoding="utf-8"))["cases"]}
    batch_arrays = []
    records = []
    for case in selected:
        audited = g1_cases[case.identifier]
        if (sha256_file(case.mri) != audited["mri_sha256"]
                or sha256_file(case.mask) != audited["mask_sha256"]):
            raise ValueError(f"{case.identifier}: source changed since G1")
        volume = zscore_nonzero(downsample(load_case(case)))
        for draw in range(2):
            origin = crop_origin(volume.mask, PATCH_XYZ, "uniform", seed=config["seed"],
                                 epoch=0, case_id=case.identifier, draw=draw)
            image, mask = crop_for_3d_model(volume, origin)
            batch_arrays.append((image, mask, case.patient))
            records.append({"case": case.identifier, "patient": case.patient,
                            "draw": draw, "origin_xyz": origin,
                            "mask_counts": np.bincount(mask.ravel(), minlength=6).tolist()})
    # Each smoke batch contains one patch from each selected patient.
    mixed_order = (0, 2, 1, 3)
    batch_arrays = [batch_arrays[index] for index in mixed_order]
    records = [records[index] for index in mixed_order]
    weights = torch.tensor(config["objective"]["ce_class_weights"],
                           dtype=torch.float32, device="cuda")
    outcomes = []
    constructors = (("standard_3d", UNet3D), ("improved_3d", ImprovedUNet3D))
    for name, constructor in constructors:
        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed_all(config["seed"])
        model = constructor().to("cuda").train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0)
        recorder = ExposureRecorder()
        torch.cuda.reset_peak_memory_stats()
        steps = []
        started = perf_counter()
        for batch_start in (0, 2):
            current = batch_arrays[batch_start:batch_start + 2]
            x = torch.from_numpy(np.ascontiguousarray(np.stack([item[0] for item in current]))).to("cuda")
            y = torch.from_numpy(np.ascontiguousarray(np.stack([item[1] for item in current]))).to("cuda")
            for _, mask, patient in current:
                recorder.add(mask, patient)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = ce_dice_loss(logits, y, ce_weight=0.5, class_weights=weights)
            if not bool(torch.isfinite(loss).item()):
                raise ValueError(f"{name}: nonfinite loss")
            loss.backward()
            gradient_norms = [param.grad.norm() for param in model.parameters()
                              if param.grad is not None]
            if (len(gradient_norms) != sum(1 for _ in model.parameters())
                    or any(not bool(torch.isfinite(norm).item()) for norm in gradient_norms)):
                raise ValueError(f"{name}: missing or nonfinite gradient")
            gradient_l2 = float(torch.linalg.vector_norm(torch.stack(gradient_norms)).item())
            optimizer.step()
            with torch.no_grad():
                prediction_counts = torch.bincount(logits.argmax(dim=1).ravel(), minlength=6)
            steps.append({"step": len(steps) + 1, "loss": float(loss.item()),
                          "gradient_norm_l2": gradient_l2,
                          "prediction_counts": prediction_counts.cpu().tolist()})
        torch.cuda.synchronize()
        result = {"model": name, "steps": steps, "exposure": recorder.summary(),
                  "elapsed_seconds": perf_counter() - started,
                  "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                  "peak_reserved_bytes": int(torch.cuda.max_memory_reserved())}
        if result["exposure"]["patches"] != 4:
            raise ValueError("Smoke exposure recorder missed model inputs")
        outcomes.append(result)
        print(f"{name}: 2 batches, final_loss={steps[-1]['loss']:.5f}, "
              f"peak_allocated_GiB={result['peak_allocated_bytes'] / 2**30:.3f}", flush=True)
        del model, optimizer, x, y, logits, loss
        torch.cuda.empty_cache()

    if outcomes[0]["exposure"] != outcomes[1]["exposure"]:
        raise ValueError("Paired models saw different delivered-mask exposure")
    result = {"kind": "G5_two_patient_paired_training_smoke_no_checkpoint",
              "job_id": os.environ["SLURM_JOB_ID"], "seed": config["seed"],
              "cases": [case.identifier for case in selected], "patches": records,
              "config_sha256": sha256_file(args.config), "g1_sha256": G1_SHA,
              "split_sha256": sha256_file(DEFAULT_SPLIT),
              "weight_source_sha256": sha256_file(weight_path),
              "script_sha256": sha256_file(Path(__file__)),
              "sampling_code_sha256": sha256_file(PROJECT / "code/hipmri_rewrite/sampling.py"),
              "data_code_sha256": sha256_file(PROJECT / "code/hipmri_rewrite/data.py"),
              "loss_code_sha256": sha256_file(PROJECT / "code/hipmri_rewrite/losses.py"),
              "gpu": torch.cuda.get_device_name(0), "torch": str(torch.__version__),
              "numpy": np.__version__, "results": outcomes}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Saved {output}; no checkpoint was produced", flush=True)


if __name__ == "__main__":
    main()
