"""Verify downloaded G7 paired crop records and plot Train exposure only."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from hipmri_rewrite.sampling import CLASS_NAMES  # noqa: E402
from hipmri_rewrite.sampling_edges import boundary_hits  # noqa: E402

SOURCE = ROOT / "logs/audits/g7_edge_sampling_full_20261019"
OUTPUT = ROOT / "results/data_analysis/boundary_sampling_audit"
EXPECTED = {"manifest.json": "50d26679437bd876182156431b578ca36789657944435268bbf13f67187b1a48",
            "patches.csv": "86a7e72227ac7101432b62336f9d70d6d418d0fad152ab67746ce833b36ad4de"}
POLICIES = ("original_uniform", "nine_uniform_one_edge")
NAMES = ("Original uniform", "9 uniform + 1 edge")
COLORS = ("#246391", "#c36b29")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(fig, stem):
    fig.savefig(OUTPUT / f"{stem}.png", dpi=180)
    svg = OUTPUT / f"{stem}.svg"
    fig.savefig(svg, metadata={"Title": stem})
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines())
                   + "\n", encoding="utf-8")
    plt.close(fig)


def main() -> None:
    for name, expected in EXPECTED.items():
        if sha(SOURCE / name) != expected:
            raise ValueError(f"Download SHA-256 differs from Rangpur: {name}")
    manifest = json.loads((SOURCE / "manifest.json").read_text(encoding="utf-8"))
    if not (manifest["full_train"] and manifest["epochs"] == 50
            and manifest["patients"] == 26 and manifest["unique_train_cases"] == 143
            and manifest["patches_per_policy"] == 13000 and manifest["job_id"] == "676249"):
        raise ValueError("Not the completed 143-volume/50-epoch G7 audit")
    sources = manifest["sources"]
    for key, path in (("script_sha256", "code/tools/audit_g7_edge_sampling.py"),
                      ("edge_sampling_sha256", "code/hipmri_rewrite/sampling_edges.py"),
                      ("candidate_config_sha256", "code/config/g7_edge_sampling_candidate.json"),
                      ("frozen_config_sha256", "code/config/g5_controlled_training_frozen.json"),
                      ("frozen_sampling_sha256", "code/hipmri_rewrite/sampling.py"),
                      ("data_sha256", "code/hipmri_rewrite/data.py"),
                      ("g1_sha256", "logs/audits/train_full_20261018_job671208.json"),
                      ("g3_sha256", "logs/audits/g3_existing_pipeline_full_20261018_job671892.json"),
                      ("split_sha256", "code/config/patient_split.json")):
        if sha(ROOT / path) != sources[key]:
            raise ValueError(f"Source differs from CPU audit: {path}")
    candidate = sources["candidate_config"]
    if candidate != json.loads((ROOT / "code/config/g7_edge_sampling_candidate.json").read_text()):
        raise ValueError("Embedded candidate configuration differs")
    for name, expected in sources["reference_epoch_sha256"].items():
        if sha(ROOT / "logs/training/baseline_3d" / name) != expected:
            raise ValueError(f"Baseline epoch differs: {name}")
    if sha(ROOT / "logs/training/baseline_3d/manifest.json") != sources["reference_manifest_sha256"]:
        raise ValueError("Baseline manifest differs")
    with (SOURCE / "patches.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 26000 or manifest["patches_csv_sha256"] != EXPECTED["patches.csv"]:
        raise ValueError("Incomplete crop CSV")
    shapes = {r["case"]: tuple(r["shape_xyz"]) for r in manifest["case_receipts"]}
    patch = tuple(candidate["patch_xyz"])
    sums = defaultdict(Counter)
    patient_sums = defaultdict(Counter)
    epoch_sums = defaultdict(Counter)
    geometry = defaultdict(Counter)
    paired = {}
    edge = []
    for row in rows:
        policy, patient, case = row["policy"], row["patient"], row["case"]
        epoch, draw = int(row["epoch"]), int(row["draw"])
        key = (policy, epoch, patient, case, draw)
        if policy not in POLICIES or key in paired or not 1 <= epoch <= 50 or not 0 <= draw < 10:
            raise ValueError("Duplicate or invalid patch key")
        paired[key] = row
        origin = tuple(int(row[f"origin_{axis}"]) for axis in "xyz")
        if any(start < 0 or start + size > n for start, size, n in zip(origin, patch, shapes[case])):
            raise ValueError("Crop outside its audited shape")
        counts = {name: int(row[f"{name.lower()}_voxels"]) for name in CLASS_NAMES}
        if min(counts.values()) < 0 or sum(counts.values()) != int(np.prod(patch)):
            raise ValueError("Class counts do not cover the model-input crop")
        for counter in (sums[policy], patient_sums[policy, patient], epoch_sums[policy, epoch]):
            counter["patches"] += 1
            for name, amount in counts.items():
                counter[name] += amount
                counter[f"{name}_hits"] += int(amount > 0)
        for width in (2, 8):
            for suffix, flag in zip(("face", "corner"), boundary_hits(origin, shapes[case], patch, width)):
                name = f"outer_{width}_any_{suffix}"
                if int(row[name]) != int(flag):
                    raise ValueError("CSV geometry flag differs from crop origin")
                geometry[policy][name] += int(flag)
        if not (0 <= float(row["nonzero_fraction"]) <= 1
                and np.isfinite(float(row["image_min"]))
                and np.isfinite(float(row["image_max"]))):
            raise ValueError("Invalid MRI summary")
        if policy == POLICIES[1] and draw == 9:
            if (row["rule"] != "accepted_edge" or not int(row["outer_8_any_corner"])
                    or float(row["nonzero_fraction"]) < candidate["edge"]["min_nonzero_fraction"]
                    or int(row["outer_shell_distant_nonzero_background_voxels"])
                    < candidate["edge"]["min_hard_negative_voxels"]):
                raise ValueError("Accepted edge crop violates declared content filters")
            edge.append(row)
    same_fields = [name for name in rows[0] if name not in ("policy", "rule", "attempts", "corner_bits")]
    for key, row in paired.items():
        policy, epoch, patient, case, draw = key
        if policy == POLICIES[0] and draw < 9:
            other = paired[POLICIES[1], epoch, patient, case, draw]
            if any(row[field] != other[field] for field in same_fields):
                raise ValueError("The preserved first nine draws differ")
    class_rows, patient_rows = [], []
    for policy in POLICIES:
        total = 13000 * int(np.prod(patch))
        if sums[policy]["patches"] != 13000:
            raise ValueError("Incorrect policy patch total")
        for name in CLASS_NAMES:
            measured = {"patch_hits": sums[policy][f"{name}_hits"],
                        "patch_hit_rate": sums[policy][f"{name}_hits"] / 13000,
                        "patch_voxels": sums[policy][name], "voxel_share": sums[policy][name] / total}
            if measured != manifest["summary"][policy]["exposure"]["per_class"][name]:
                raise ValueError("CSV class totals differ from manifest")
            class_rows.append({"policy": policy, "class": name, **measured})
        for name, amount in geometry[policy].items():
            if amount != manifest["summary"][policy]["geometry_counts"][name]:
                raise ValueError("CSV geometry totals differ from manifest")
        for patient in sorted({r["patient"] for r in rows}):
            counts = patient_sums[policy, patient]
            if counts["patches"] != 500:
                raise ValueError("Patient exposure is not balanced")
            for name in CLASS_NAMES:
                patient_rows.append({"policy": policy, "patient": patient, "class": name,
                                     "patches": 500, "patch_hits": counts[f"{name}_hits"],
                                     "patch_voxels": counts[name],
                                     "voxel_share": counts[name] / (500 * int(np.prod(patch)))})
        for epoch in range(1, 51):
            expected = manifest["epoch_exposure"][str(epoch)][policy]
            counts = epoch_sums[policy, epoch]
            if counts["patches"] != expected["patches"]:
                raise ValueError("Epoch patch total differs")
            for name in CLASS_NAMES:
                if (counts[name] != expected["per_class"][name]["patch_voxels"]
                        or counts[f"{name}_hits"] != expected["per_class"][name]["patch_hits"]):
                    raise ValueError("Epoch class exposure differs")
            if policy == POLICIES[0]:
                record = json.loads((ROOT / "logs/training/baseline_3d" / f"epoch_{epoch:03d}.json").read_text())
                if expected != record["train"]["exposure"]:
                    raise ValueError("Replayed uniform exposure differs from training")
    if len(edge) != 1300 or not all(manifest["gates"].values()):
        raise ValueError("Unexpected accepted edge count or failed provisional gate")
    old, new = (manifest["summary"][p]["exposure"]["per_class"] for p in POLICIES)
    ratios = {name: {"hit_rate_ratio": new[name]["patch_hit_rate"] / old[name]["patch_hit_rate"],
                     "voxel_share_ratio": new[name]["voxel_share"] / old[name]["voxel_share"]}
              for name in ("Bladder", "Rectum", "Prostate")}
    gate = candidate["training_gate"]
    gates = {"outer_8_corner_rate": geometry[POLICIES[1]]["outer_8_any_corner"] / 13000
             >= gate["min_outer_8_corner_rate"],
             "small_organ_hit_rates": all(v["hit_rate_ratio"] >= gate["min_small_organ_hit_ratio"]
                                          for v in ratios.values()),
             "small_organ_voxel_shares": all(v["voxel_share_ratio"] >= gate["min_small_organ_voxel_share_ratio"]
                                            for v in ratios.values())}
    if ratios != manifest["small_organ_ratios"] or gates != manifest["gates"]:
        raise ValueError("Recomputed engineering gates differ from audit")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name, records in (("class_exposure.csv", class_rows), ("patient_class_exposure.csv", patient_rows)):
        with (OUTPUT / name).open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.8), layout="constrained")
    y = np.arange(6)
    for policy, label, color, offset in zip(POLICIES, NAMES, COLORS, (-.18, .18)):
        classes = manifest["summary"][policy]["exposure"]["per_class"]
        for ax, field in zip(axes[:2], ("voxel_share", "patch_hit_rate")):
            values = [100 * classes[name][field] for name in CLASS_NAMES]
            ax.barh(y + offset, values, height=.34, color=color, label=label)
        values = [100 * geometry[policy][f"outer_{w}_any_{part}"] / 13000
                  for w, part in ((2, "face"), (2, "corner"), (8, "face"), (8, "corner"))]
        axes[2].barh(np.arange(4) + offset, values, height=.34, color=color, label=label)
    for ax in axes[:2]:
        ax.set_yticks(y, CLASS_NAMES)
    axes[0].set(title="Six-class delivered voxel share", xlabel="Share (%) · logarithmic scale", xscale="log")
    axes[1].set(title="Six-class patch hit rate", xlabel="Patches containing class (%)", xlim=(0, 105))
    axes[2].set(title="Original-volume margin coverage", xlabel="Patch fraction (%) · logarithmic scale", xscale="log")
    axes[2].set_yticks(range(4), ["Outer 2: any face", "Outer 2: corner", "Outer 8: any face", "Outer 8: corner"])
    for ax in axes:
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=.18)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="outside lower center", ncol=2)
    fig.suptitle("Boundary sampling audit · 26 patients · 50 epochs · 13,000 crops per policy")
    save(fig, "sampling_exposure_and_geometry")
    fig, axes = plt.subplots(2, 3, figsize=(11, 7), layout="constrained")
    for row_i, selected in enumerate(([r for r in rows if r["policy"] == POLICIES[0]], edge)):
        points = np.array([[int(r[f"origin_{axis}"]) / (shapes[r["case"]][i] - patch[i])
                            for i, axis in enumerate("xyz")] for r in selected])
        for ax, (a, b) in zip(axes[row_i], ((0, 1), (0, 2), (1, 2))):
            ax.scatter(points[:, a], points[:, b], s=3, alpha=.08 if row_i == 0 else .15,
                       color=COLORS[row_i], linewidths=0, rasterized=True)
            ax.set(xlim=(-.03, 1.03), ylim=(-.03, 1.03),
                   xlabel=f"{'XYZ'[a]} origin / allowable extent", ylabel=f"{'XYZ'[b]} origin / allowable extent")
            ax.set_title("Original: all 13,000" if row_i == 0 else "Treatment: 1,300 edge draws only")
    fig.suptitle("Crop-start positions · normalized to each volume's allowable crop origins")
    save(fig, "crop_origin_distribution")
    receipt = {"kind": "G7_local_CSV_verification", "source_hashes": EXPECTED,
               "csv_rows": len(rows), "preserved_draw_pairs": 11700,
               "baseline_epochs_reproduced": 50, "accepted_edge_draws": len(edge),
               "eligible_under_provisional_gates": manifest["eligible_for_training"],
               "inference_or_model_quality_established": False}
    (OUTPUT / "verification.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print("Verified G7 hashes, 26,000 rows, 11,700 preserved draws and 50 baseline epochs")
    print("Corner 8 rate:", manifest["summary"][POLICIES[0]]["geometry_rates"]["outer_8_any_corner"],
          "->", manifest["summary"][POLICIES[1]]["geometry_rates"]["outer_8_any_corner"])
    print(OUTPUT)


if __name__ == "__main__":
    main()
