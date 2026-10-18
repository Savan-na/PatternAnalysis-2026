"""Measure full-volume 3D inference capacity on one approved Train MRI.

Models have random weights. This checks geometry, memory and execution only;
it does not evaluate segmentation quality or save a checkpoint.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import socket
import sys
from pathlib import Path
from statistics import median
from time import perf_counter


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "code"))
G1_SHA = "6a4c674564a0364f5f56730cbc31e67b3ff9a8ffeae7d9d3dd18b198bbd0e1a1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--case-id", default="K019_Week1")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--g1", type=Path,
                        default=PROJECT / "logs/audits/train_full_20261018_job671208.json")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or socket.gethostname().startswith("login"):
        parser.error("Run inside an allocated Slurm GPU compute job")
    output = args.output.resolve()
    if not output.is_relative_to(PROJECT) or output.exists():
        parser.error("Output must be a new JSON file inside 20261017")
    if not 1 <= args.repeats <= 5:
        parser.error("Repeats must be 1..5")
    if sha256_file(args.g1) != G1_SHA:
        raise ValueError("G1 Train audit differs from verified source")

    import numpy as np
    import torch

    from hipmri_rewrite.data import downsample, load_case, to_3d_model_arrays, zscore_nonzero
    from hipmri_rewrite.inference import predict_volume
    from hipmri_rewrite.models import ImprovedUNet3D, UNet3D
    from hipmri_rewrite.splits import discover

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA device allocated")
    cases = {case.identifier: case for case in discover(args.dataset_root)["train"]}
    if args.case_id not in cases:
        parser.error("Case ID must belong to frozen Train split")
    case = cases[args.case_id]
    audited = {row["case"]: row for row in json.loads(args.g1.read_text(encoding="utf-8"))["cases"]}
    if (sha256_file(case.mri) != audited[case.identifier]["mri_sha256"]
            or sha256_file(case.mask) != audited[case.identifier]["mask_sha256"]):
        raise ValueError("Source MRI/Mask changed since G1")
    volume = zscore_nonzero(downsample(load_case(case)))
    padded_image, _, original_shape = to_3d_model_arrays(volume)
    results = []
    for name, constructor in (("standard_3d", UNet3D),
                              ("improved_3d", ImprovedUNet3D)):
        torch.manual_seed(371017)
        torch.cuda.manual_seed_all(371017)
        model = constructor().to("cuda").eval()
        with torch.inference_mode():
            warmup = predict_volume(model, volume, mode="3d", device="cuda")
            if warmup.shape != volume.image.shape or np.any(warmup > 5):
                raise ValueError(f"{name}: wrong XYZ shape or illegal predicted class")
            del warmup
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            seconds = []
            for _ in range(args.repeats):
                start = perf_counter()
                prediction = predict_volume(model, volume, mode="3d", device="cuda")
                torch.cuda.synchronize()
                seconds.append(perf_counter() - start)
                if prediction.shape != volume.image.shape or np.any(prediction > 5):
                    raise ValueError(f"{name}: wrong XYZ shape or illegal predicted class")
                del prediction
        row = {"model": name, "parameters": sum(p.numel() for p in model.parameters()),
               "inference_seconds": seconds, "median_inference_seconds": median(seconds),
               "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
               "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
               "output_shape_xyz": list(volume.image.shape), "legal_class_ids_only": True}
        results.append(row)
        print(f"{name}: shape={row['output_shape_xyz']}, "
              f"median_seconds={row['median_inference_seconds']:.3f}, "
              f"peak_allocated_GiB={row['peak_allocated_bytes'] / 2**30:.3f}", flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()

    record = {"kind": "G5_random_weight_full_Train_volume_inference_capacity_only",
              "job_id": os.environ["SLURM_JOB_ID"], "case": case.identifier,
              "mri_sha256": audited[case.identifier]["mri_sha256"],
              "mask_sha256": audited[case.identifier]["mask_sha256"],
              "g1_sha256": G1_SHA, "seed": 371017, "repeats": args.repeats,
              "preprocessed_shape_xyz": list(original_shape),
              "padded_input_shape_dhw": list(padded_image.shape),
              "gpu": torch.cuda.get_device_name(0),
              "gpu_total_bytes": int(torch.cuda.get_device_properties(0).total_memory),
              "torch": str(torch.__version__), "numpy": np.__version__,
              "script_sha256": sha256_file(Path(__file__)),
              "data_code_sha256": sha256_file(PROJECT / "code/hipmri_rewrite/data.py"),
              "inference_code_sha256": sha256_file(PROJECT / "code/hipmri_rewrite/inference.py"),
              "results": results}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Saved {output}; random weights, no checkpoint or quality metrics", flush=True)


if __name__ == "__main__":
    main()
