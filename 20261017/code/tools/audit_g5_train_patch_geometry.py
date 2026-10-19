"""Reconstruct frozen 3D Train patch positions without reading MRI voxels.

The uniform crop generator depends only on preprocessed shape, case ID, epoch,
draw and seed. G1 raw shapes and the completed G5 selected-case schedules are
therefore sufficient to audit how often volume faces and corners were seen.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "code"))
from hipmri_rewrite.sampling import PATCH_XYZ, crop_origin  # noqa: E402

G1 = PROJECT / "logs/audits/train_full_20261018_job671208.json"
RUNS = PROJECT / "logs/training"
OUTPUT = PROJECT / "results/data_analysis/g4_patient_sampling/g5_train_patch_geometry.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def touches(origin: tuple[int, int, int], shape: tuple[int, int, int],
            width: int) -> tuple[bool, bool, bool, bool, bool, bool]:
    low = [start < width for start in origin]
    high = [start + size > n - width
            for start, size, n in zip(origin, PATCH_XYZ, shape)]
    return low[0], high[0], low[1], high[1], low[2], high[2]


def main() -> None:
    source = json.loads(G1.read_text(encoding="utf-8"))
    if (not source["complete_train_audit"] or not source["mechanical_pass"]
            or len(source["cases"]) != 143):
        raise ValueError("G1 source is not the complete audited Train cohort")
    shapes = {row["case"]: tuple(int(np.ceil(n / 2)) for n in row["mri_shape"])
              for row in source["cases"]}
    config_path = PROJECT / "code/config/g5_controlled_training_frozen.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if (config["sampling"]["patch_rule"] != "uniform"
            or tuple(config["preprocessing"]["downsample_factor_xyz"]) != (2, 2, 2)
            or tuple(config["preprocessing"]["train_patch_xyz"]) != PATCH_XYZ
            or config["sampling"]["patches_per_patient_epoch"] != 10):
        raise ValueError("The frozen uniform patch protocol changed")
    folders = ("baseline_3d", "improved_3d")
    records = []
    for folder in folders:
        run = RUNS / folder
        manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
        if manifest["max_epochs"] != 50 or manifest["config_sha256"] != digest(config_path):
            raise ValueError(f"{folder}: unexpected frozen run")
        paths = sorted(run.glob("epoch_*.json"))
        if [p.name for p in paths] != [f"epoch_{i:03d}.json" for i in range(1, 51)]:
            raise ValueError(f"{folder}: incomplete records")
        records.append([json.loads(p.read_text(encoding="utf-8")) for p in paths])
    counters = Counter()
    by_patient = defaultdict(Counter)
    for epoch, (standard, improved) in enumerate(zip(*records)):
        if (standard["selected_cases"] != improved["selected_cases"]
                or standard["train"]["exposure"] != improved["train"]["exposure"]
                or len(standard["selected_cases"]) != 26):
            raise ValueError(f"Epoch {epoch + 1}: paired Train exposure differs")
        for case_id in standard["selected_cases"]:
            shape = shapes[case_id]
            mask = np.empty(shape, dtype=np.uint8)
            patient = case_id.split("_Week")[0]
            for draw in range(10):
                origin = crop_origin(mask, PATCH_XYZ, "uniform",
                                     seed=config["seed"], epoch=epoch,
                                     case_id=case_id, draw=draw)
                counters["patches"] += 1
                by_patient[patient]["patches"] += 1
                for width in (2, 8):
                    sides = touches(origin, shape, width)
                    for axis, index in zip("xyz", (0, 2, 4)):
                        for side, flag in (("low", sides[index]),
                                           ("high", sides[index + 1])):
                            if flag:
                                counters[f"outer_{width}_{axis}_{side}"] += 1
                    if any(sides):
                        counters[f"outer_{width}_any_face"] += 1
                    if any(sides[0:2]) and any(sides[2:4]) and any(sides[4:6]):
                        counters[f"outer_{width}_any_corner"] += 1
                        by_patient[patient][f"outer_{width}_any_corner"] += 1
    if counters["patches"] != 50 * 26 * 10 or len(by_patient) != 26:
        raise ValueError("Unexpected Train patch count or patient inventory")
    result = {
        "kind": "G5_uniform_3D_Train_patch_spatial_coverage",
        "scope": "Train only; deterministic origin reconstruction, not label exposure",
        "g1_sha256": digest(G1), "config_sha256": digest(config_path),
        "run_manifest_sha256": {folder: digest(RUNS / folder / "manifest.json")
                                for folder in folders},
        "definition": "A patch hits an outer-width face if any sampled voxel lies in that face's width-voxel slab; a corner requires all three axes to hit an outer slab simultaneously.",
        "patch_xyz": PATCH_XYZ, "epochs": 50, "counts": dict(counters),
        "rates": {key: value / counters["patches"] for key, value in counters.items()
                  if key != "patches"},
        "by_patient": {patient: dict(counts) for patient, counts in sorted(by_patient.items())},
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("patches", counters["patches"])
    for width in (2, 8):
        for location in ("any_face", "any_corner"):
            key = f"outer_{width}_{location}"
            print(key, counters[key], f"{100 * result['rates'][key]:.3f}%")
    print(OUTPUT)


if __name__ == "__main__":
    main()
