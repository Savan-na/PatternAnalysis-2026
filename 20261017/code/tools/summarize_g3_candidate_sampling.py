"""Verify full Train candidate-patch records and compare class exposure.

Usage: python -B code/tools/summarize_g3_candidate_sampling.py
         --manifest-sha256 HASH --patches-sha256 HASH
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
SOURCE = PROJECT / "logs/audits/g3_patch_candidate_full_20261018"
OUTPUT = PROJECT / "results/data_analysis/g3_pipeline"
G3 = PROJECT / "logs/audits/g3_existing_pipeline_full_20261018_job671892.json"
CLASSES = ("Background", "Body", "Bones", "Bladder", "Rectum", "Prostate")
ORGANS = ("Bladder", "Rectum", "Prostate")
ARMS = ("uniform", "organ_aware_candidate")
G1_SHA = "6a4c674564a0364f5f56730cbc31e67b3ff9a8ffeae7d9d3dd18b198bbd0e1a1"
G3_SHA = "319973a8759265521870f831e7b17eb7249f4130b6a9e9634fe993f22bf687e9"
CONFIG_SHA = "9516941e3058509fe1ead1bbab461189cef8b495b3ac07520097b1a588d7bf05"
SCRIPT_SHA = "4c284b68821b7ec65f1ae1402399e8fe4c380c30e6f2cd71df5cd98a9e12b3d8"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--patches-sha256", required=True)
    args = parser.parse_args()

    if sha256_file(SOURCE / "manifest.json") != args.manifest_sha256.lower():
        raise ValueError("Transferred manifest SHA-256 differs from Rangpur")
    if sha256_file(SOURCE / "patches.csv") != args.patches_sha256.lower():
        raise ValueError("Transferred patch CSV SHA-256 differs from Rangpur")
    if sha256_file(G3) != G3_SHA:
        raise ValueError("G3 source SHA-256 changed")
    manifest = json.loads((SOURCE / "manifest.json").read_text(encoding="utf-8"))
    g3 = json.loads(G3.read_text(encoding="utf-8"))
    config = manifest["config"]
    if not (manifest["scope"] == "Train-only candidate crop exposure audit; not an active training sampler"
            and manifest["full_train"] and manifest["train_cases"] == 143
            and manifest["patients"] == 26 and manifest["g1_sha256"] == G1_SHA
            and manifest["g3_sha256"] == G3_SHA
            and manifest["config_sha256"] == CONFIG_SHA
            and manifest["audit_script_sha256"] == SCRIPT_SHA
            and config["patch_shape_xyz"] == [64, 64, 32]
            and config["draws_per_case_per_arm"] == 10
            and set(config["arm_schedules"]) == set(ARMS)):
        raise ValueError("Manifest differs from frozen full Train candidate audit")
    cases = {case["case"]: case for case in g3["cases"]}
    if len(cases) != 143:
        raise ValueError("G3 case set is incomplete")
    with (SOURCE / "patches.csv").open(encoding="utf-8-sig", newline="") as stream:
        patches = list(csv.DictReader(stream))
    if len(patches) != 2860:
        raise ValueError(f"Expected 2860 patch records, found {len(patches)}")

    seen = set()
    by_arm = {arm: {"voxels": Counter(), "hits": Counter(), "complete": Counter(),
                    "patches": 0} for arm in ARMS}
    by_patient = defaultdict(lambda: {"patches": 0, "cases": set(), "hits": Counter()})
    origins = {}
    for row in patches:
        case, arm, patient = row["case"], row["arm"], row["patient"]
        draw = int(row["draw"])
        if case not in cases or arm not in ARMS or patient != cases[case]["patient"]:
            raise ValueError("Unknown case, arm or patient in patch CSV")
        if draw not in range(10) or row["target"] != config["arm_schedules"][arm][draw]:
            raise ValueError(f"Unexpected draw target: {case} {arm} {draw}")
        key = case, arm, draw
        if key in seen:
            raise ValueError(f"Duplicate patch key: {key}")
        seen.add(key)
        origin = tuple(int(row[f"origin_{axis}"]) for axis in "xyz")
        shape = cases[case]["downsampled_shape_xyz"]
        if any(not 0 <= start <= n - size
               for start, n, size in zip(origin, shape, (64, 64, 32))):
            raise ValueError(f"Crop origin outside case grid: {key}")
        origins[key] = origin
        counts = {name: int(row[f"{name.lower()}_voxels"]) for name in CLASSES}
        if any(n < 0 for n in counts.values()) or sum(counts.values()) != 64 * 64 * 32:
            raise ValueError(f"Invalid patch class counts: {key}")
        if row["target"] != "uniform" and counts[row["target"]] == 0:
            raise ValueError(f"Target organ missing from anchored patch: {key}")
        by_arm[arm]["patches"] += 1
        group = by_patient[(patient, arm)]
        group["patches"] += 1
        group["cases"].add(case)
        for name, n in counts.items():
            by_arm[arm]["voxels"][name] += n
            by_arm[arm]["hits"][name] += int(n > 0)
            group["hits"][name] += int(n > 0)
        for name in ORGANS:
            complete = int(row[f"{name.lower()}_complete"])
            if complete != int(counts[name] == cases[case]["downsampled_counts"][name]):
                raise ValueError(f"Complete-organ flag differs from G3: {key} {name}")
            by_arm[arm]["complete"][name] += complete

    expected_keys = {(case, arm, draw) for case in cases for arm in ARMS for draw in range(10)}
    if seen != expected_keys:
        raise ValueError("Patch case/arm/draw coverage is incomplete")
    for case in cases:
        for draw in range(4):
            if origins[(case, "uniform", draw)] != origins[(case, "organ_aware_candidate", draw)]:
                raise ValueError(f"Paired uniform crop differs: {case} {draw}")

    summary_rows = []
    for arm in ARMS:
        computed = by_arm[arm]
        recorded = manifest["summary"][arm]
        if computed["patches"] != 1430 or recorded["patches"] != 1430:
            raise ValueError(f"Patch count differs: {arm}")
        total_voxels = 1430 * 64 * 64 * 32
        if recorded["voxel_total"] != total_voxels:
            raise ValueError(f"Manifest voxel total differs: {arm}")
        for name in CLASSES:
            item = recorded["per_class"][name]
            hits = computed["hits"][name]
            voxels = computed["voxels"][name]
            if (item["patch_hits"] != hits or item["patch_voxels"] != voxels
                    or abs(item["patch_hit_rate"] - hits / 1430) > 1e-12
                    or abs(item["voxel_share"] - voxels / total_voxels) > 1e-12):
                raise ValueError(f"Manifest class totals differ: {arm} {name}")
            if name in ORGANS and item["complete_organ_patches"] != computed["complete"][name]:
                raise ValueError(f"Complete-organ manifest total differs: {arm} {name}")
            summary_rows.append({
                "arm": arm, "class": name, "patches": 1430,
                "patch_hits": hits, "patch_hit_pct": 100 * hits / 1430,
                "patch_voxels": voxels, "voxel_share_pct": 100 * voxels / total_voxels,
                "complete_organ_patches": computed["complete"][name] if name in ORGANS else "",
            })
    patient_rows = []
    for (patient, arm), group in sorted(by_patient.items()):
        patient_rows.append({
            "patient": patient, "arm": arm, "volumes": len(group["cases"]),
            "patches": group["patches"],
            **{f"{name.lower()}_hit_pct": 100 * group["hits"][name] / group["patches"]
               for name in ORGANS},
        })
    if len(patient_rows) != 52:
        raise ValueError("Expected 26 patient rows per arm")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT / "candidate_patch_class_exposure.csv", summary_rows)
    write_csv(OUTPUT / "candidate_patch_patient_exposure.csv", patient_rows)
    x = np.arange(len(ORGANS))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for index, arm in enumerate(ARMS):
        organ_rows = {row["class"]: row for row in summary_rows if row["arm"] == arm}
        for ax, field in zip(axes, ("patch_hit_pct", "voxel_share_pct")):
            ax.bar(x + (index - 0.5) * width, [organ_rows[name][field] for name in ORGANS],
                   width, label=arm.replace("_", " "))
    for ax, title, ylabel in zip(axes,
                                 ("Patches containing organ", "Organ voxels among sampled voxels"),
                                 ("Patch hit rate (%)", "Sampled voxel share (%)")):
        ax.set_xticks(x, ORGANS)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Train-only candidate crops (1430 patches per policy)")
    fig.tight_layout()
    fig.savefig(OUTPUT / "candidate_patch_exposure.svg")
    plt.close(fig)
    print("Verified 2860 Train candidate patches, 143 cases, 26 patients; wrote two CSVs and one SVG")
    for row in summary_rows:
        if row["class"] in ORGANS:
            print(row["arm"], row["class"], f"hit={row['patch_hit_pct']:.2f}%",
                  f"voxels={row['voxel_share_pct']:.3f}%",
                  f"complete={row['complete_organ_patches']}")


if __name__ == "__main__":
    main()
