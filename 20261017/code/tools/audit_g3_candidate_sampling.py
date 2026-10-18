"""Train-only simulation of two fixed patch policies on the verified G3 grid.

This audits candidate crop exposure. It is not an active model-training sampler.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import socket
from collections import Counter
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(PROJECT / "code"))


CLASSES = ("Background", "Body", "Bones", "Bladder", "Rectum", "Prostate")
ORGANS = ("Bladder", "Rectum", "Prostate")
LABELS = {name: index for index, name in enumerate(CLASSES)}
G1_SHA = "6a4c674564a0364f5f56730cbc31e67b3ff9a8ffeae7d9d3dd18b198bbd0e1a1"
G3_SHA = "319973a8759265521870f831e7b17eb7249f4130b6a9e9634fe993f22bf687e9"
DATA_SHA = "b1aa857d8da735fc4d467d520f8a8ebbf8cc6a15614e225a76a4bf0a47c06a85"
SPLITS_SHA = "a172311556c4bda0364cc7a84def06cf4271e36dd95e6392ce8ef27d39a07a47"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fixed_rng(seed: int, case: str, draw: int):
    import numpy as np

    key = hashlib.sha256(f"{seed}|{case}|{draw}".encode("ascii")).digest()
    return np.random.default_rng(int.from_bytes(key[:8], "little"))


def uniform_origin(shape: tuple[int, int, int], patch: tuple[int, int, int], rng) -> tuple[int, int, int]:
    return tuple(int(rng.integers(0, n - size + 1)) for n, size in zip(shape, patch))


def target_origin(shape: tuple[int, int, int], patch: tuple[int, int, int],
                  voxel: tuple[int, int, int], rng) -> tuple[int, int, int]:
    """Choose a valid crop origin that contains the selected target voxel."""
    origin = []
    for n, size, pos in zip(shape, patch, voxel):
        low = max(0, pos - size + 1)
        high = min(pos, n - size)
        if low > high:
            raise ValueError("Target voxel cannot fit inside crop")
        origin.append(int(rng.integers(low, high + 1)))
    return tuple(origin)


def patch_counts(mask, origin: tuple[int, int, int], patch: tuple[int, int, int]):
    import numpy as np

    crop = mask[tuple(slice(start, start + size) for start, size in zip(origin, patch))]
    if crop.shape != patch:
        raise ValueError("Crop shape differs from frozen patch size")
    return np.bincount(crop.ravel(), minlength=len(CLASSES))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=PROJECT / "code/config/g3_patch_audit.json")
    parser.add_argument("--g1", type=Path, default=PROJECT / "logs/audits/train_full_20261018_job671208.json")
    parser.add_argument("--g3", type=Path, default=PROJECT / "logs/audits/g3_existing_pipeline_full_20261018_job671892.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if not os.environ.get("SLURM_JOB_ID") or socket.gethostname().startswith("login"):
        parser.error("Run this MRI audit inside an allocated Slurm CPU job, never on login0")

    import numpy as np

    from hipmri_rewrite.data import downsample, load_case
    from hipmri_rewrite.splits import discover

    output = args.output_dir.resolve()
    if not output.is_relative_to(PROJECT) or output.exists():
        parser.error("Output must be a new directory inside the 20261017 project")
    if args.limit is not None and not 1 <= args.limit <= 143:
        parser.error("--limit must be 1..143")
    if sha256_file(PROJECT / "code/hipmri_rewrite/data.py") != DATA_SHA:
        raise ValueError("Data transform code changed since G3")
    if sha256_file(PROJECT / "code/hipmri_rewrite/splits.py") != SPLITS_SHA:
        raise ValueError("Patient discovery code changed since G3")
    if sha256_file(args.g1) != G1_SHA or sha256_file(args.g3) != G3_SHA:
        raise ValueError("G1/G3 audit hashes differ from verified full runs")
    config_hash = sha256_file(args.config)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    patch = tuple(config["patch_shape_xyz"])
    draws = config["draws_per_case_per_arm"]
    schedules = config["arm_schedules"]
    if (config["scope"] != "Train-only candidate crop exposure audit; not an active training sampler"
            or config["seed"] != 371017 or patch != (64, 64, 32) or draws != 10
            or schedules != {
                "uniform": ["uniform"] * 10,
                "organ_aware_candidate": ["uniform"] * 4 + ["Bladder"] * 2
                + ["Rectum"] * 2 + ["Prostate"] * 2,
            }):
        raise ValueError("Candidate audit policy differs from reviewed design")
    g1 = json.loads(args.g1.read_text(encoding="utf-8"))
    g3 = json.loads(args.g3.read_text(encoding="utf-8"))
    if not (g1["complete_train_audit"] and g1["mechanical_pass"]
            and g3["complete_train_audit"] and g3["pipeline_pass"]
            and g3["source_g1_sha256"] == G1_SHA
            and g3["split_file_sha256"] == g1["split_file_sha256"]):
        raise ValueError("Inputs are not matching complete Train audits")
    old = {row["case"]: row for row in g1["cases"]}
    processed = {row["case"]: row for row in g3["cases"]}
    cohort = discover(args.dataset_root)["train"]
    if (len(cohort) != 143 or len(old) != 143 or len(processed) != 143
            or {case.identifier for case in cohort} != set(old) or set(old) != set(processed)):
        raise ValueError("Train case inventory differs from verified audits")

    selected = cohort[:args.limit] if args.limit else cohort
    records: list[dict] = []
    totals = {arm: {"patches": 0, "voxel_counts": Counter(), "hits": Counter(),
                    "complete_organ_patches": Counter()} for arm in schedules}
    for index, case in enumerate(selected, 1):
        earlier = old[case.identifier]
        prior = processed[case.identifier]
        if (sha256_file(case.mri) != earlier["mri_sha256"]
                or sha256_file(case.mask) != earlier["mask_sha256"]):
            raise ValueError(f"Source file changed since G1: {case.identifier}")
        volume = downsample(load_case(case))
        mask = volume.mask
        shape = tuple(mask.shape)
        if any(n < size for n, size in zip(shape, patch)):
            raise ValueError(f"Patch larger than case grid: {case.identifier}")
        full_counts = np.bincount(mask.ravel(), minlength=len(CLASSES))
        if any(int(full_counts[k]) != prior["downsampled_counts"][name]
               for k, name in enumerate(CLASSES)):
            raise ValueError(f"Transformed labels differ from G3: {case.identifier}")
        coordinates = {name: np.argwhere(mask == LABELS[name]) for name in ORGANS}
        if any(len(coords) == 0 for coords in coordinates.values()):
            raise ValueError(f"Missing target organ: {case.identifier}")
        for draw in range(draws):
            rng = fixed_rng(config["seed"], case.identifier, draw)
            random_origin = uniform_origin(shape, patch, rng)
            for arm, schedule in schedules.items():
                target = schedule[draw]
                if target == "uniform":
                    origin = random_origin
                else:
                    coords = coordinates[target]
                    voxel = tuple(int(v) for v in coords[int(rng.integers(len(coords)))])
                    origin = target_origin(shape, patch, voxel, rng)
                counts = patch_counts(mask, origin, patch)
                if target != "uniform" and counts[LABELS[target]] == 0:
                    raise ValueError(f"Targeted patch missed {target}: {case.identifier}")
                row = {"case": case.identifier, "patient": case.patient,
                       "arm": arm, "draw": draw, "target": target,
                       "origin_x": origin[0], "origin_y": origin[1], "origin_z": origin[2]}
                for k, name in enumerate(CLASSES):
                    n = int(counts[k])
                    row[f"{name.lower()}_voxels"] = n
                    totals[arm]["voxel_counts"][name] += n
                    totals[arm]["hits"][name] += int(n > 0)
                    if name in ORGANS:
                        row[f"{name.lower()}_complete"] = int(n == full_counts[k])
                        totals[arm]["complete_organ_patches"][name] += row[f"{name.lower()}_complete"]
                if sum(int(counts[k]) for k in range(len(CLASSES))) != int(np.prod(patch)):
                    raise ValueError(f"Patch label count mismatch: {case.identifier}")
                records.append(row)
                totals[arm]["patches"] += 1
        print(f"[{index}/{len(selected)}] {case.identifier}: {draws * len(schedules)} patches", flush=True)

    summary = {}
    for arm, count in totals.items():
        patches = count["patches"]
        voxels = patches * int(np.prod(patch))
        if sum(count["voxel_counts"].values()) != voxels:
            raise ValueError(f"Aggregated patch count mismatch: {arm}")
        summary[arm] = {
            "patches": patches,
            "voxel_total": voxels,
            "per_class": {name: {
                "patch_hits": count["hits"][name],
                "patch_hit_rate": count["hits"][name] / patches,
                "patch_voxels": count["voxel_counts"][name],
                "voxel_share": count["voxel_counts"][name] / voxels,
                **({"complete_organ_patches": count["complete_organ_patches"][name]}
                   if name in ORGANS else {}),
            } for name in CLASSES},
        }
    manifest = {
        "scope": config["scope"],
        "full_train": args.limit is None,
        "train_cases": len(selected),
        "patients": len({case.patient for case in selected}),
        "g1_sha256": G1_SHA,
        "g3_sha256": G3_SHA,
        "audit_script_sha256": sha256_file(Path(__file__)),
        "data_code_sha256": DATA_SHA,
        "split_code_sha256": SPLITS_SHA,
        "config_sha256": config_hash,
        "config": config,
        "summary": summary,
    }
    output.mkdir(parents=True)
    with (output / "patches.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                                          encoding="utf-8")
    print(f"Saved {len(records)} candidate patches to {output}; full_train={manifest['full_train']}")


if __name__ == "__main__":
    main()
