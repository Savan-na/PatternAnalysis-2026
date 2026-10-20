"""Verify the full Train G3 audit and summarize organ preservation after downsampling."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
G1 = ROOT / "logs/audits/train_full_20261018_job671208.json"
G3 = ROOT / "logs/audits/g3_existing_pipeline_full_20261018_job671892.json"
OUTPUT = ROOT / "results/data_analysis/preprocessing_validation"
ORGANS = ("Bladder", "Rectum", "Prostate")
G1_SHA = "6a4c674564a0364f5f56730cbc31e67b3ff9a8ffeae7d9d3dd18b198bbd0e1a1"
G3_SHA = "319973a8759265521870f831e7b17eb7249f4130b6a9e9634fe993f22bf687e9"


def verified_json(path: Path, digest: str) -> dict:
    data = path.read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if actual != digest:
        raise ValueError(f"Unexpected SHA-256 for {path.name}: {actual}")
    return json.loads(data)


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g1 = verified_json(G1, G1_SHA)
    g3 = verified_json(G3, G3_SHA)
    if not (g1["complete_train_audit"] and g1["mechanical_pass"]
            and g3["complete_train_audit"] and g3["pipeline_pass"]
            and g3["case_error_count"] == 0 and g3["audited_cases"] == 143
            and g3["source_g1_sha256"] == G1_SHA
            and g3["split_file_sha256"] == g1["split_file_sha256"]):
        raise ValueError("Inputs are not matching complete Train audits")
    earlier = {row["case"]: row for row in g1["cases"]}
    later = {row["case"]: row for row in g3["cases"]}
    if len(earlier) != 143 or len(later) != 143 or set(earlier) != set(later):
        raise ValueError("Missing or duplicated Train cases")

    rows = []
    ratios = {name: [] for name in ORGANS}
    for case in sorted(later):
        before, after = earlier[case], later[case]
        if before["class_counts"] != after["raw_counts"] or after["errors"]:
            raise ValueError(f"G1/G3 raw counts or case status differ: {case}")
        raw_total = sum(after["raw_counts"].values())
        reduced_total = sum(after["downsampled_counts"].values())
        if raw_total != 8 * reduced_total or reduced_total != int(
                after["downsampled_shape_xyz"][0]
                * after["downsampled_shape_xyz"][1]
                * after["downsampled_shape_xyz"][2]):
            raise ValueError(f"Unexpected downsampled voxel totals: {case}")
        if sum(after["model_counts_including_padding"].values()) != (
                reduced_total + after["padded_background_voxels"]):
            raise ValueError(f"Unexpected model-array voxel total: {case}")
        for name in ORGANS:
            raw_count = after["raw_counts"][name]
            reduced_count = after["downsampled_counts"][name]
            ratio = (reduced_count * after["downsampled_voxel_volume_mm3"]
                     / (raw_count * after["raw_voxel_volume_mm3"]))
            if raw_count <= 0 or reduced_count <= 0 or abs(
                    ratio - after["physical_volume_ratio"][name]) > 1e-10:
                raise ValueError(f"Organ vanished or volume ratio differs: {case} {name}")
            rows.append({
                "case": case,
                "patient": after["patient"],
                "organ": name,
                "raw_voxels": raw_count,
                "downsampled_voxels": reduced_count,
                "raw_axial_slices": after["raw_occupied_z_slices"][name],
                "downsampled_axial_slices": after["downsampled_occupied_z_slices"][name],
                "raw_physical_volume_mm3": raw_count * after["raw_voxel_volume_mm3"],
                "downsampled_physical_volume_mm3": reduced_count * after["downsampled_voxel_volume_mm3"],
                "physical_volume_ratio": ratio,
            })
            ratios[name].append(ratio)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    table = OUTPUT / "organ_preservation_by_volume.csv"
    with table.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.boxplot([ratios[name] for name in ORGANS], tick_labels=ORGANS,
               showfliers=True, widths=0.5)
    ax.axhline(1, color="0.35", linestyle="--", linewidth=1,
               label="Same physical volume")
    ax.set_ylabel("Downsampled / raw labelled physical volume")
    ax.set_title("Train organ volume after factor-two downsampling (n=143)")
    ax.legend(loc="lower left", frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    plot = OUTPUT / "organ_physical_volume_ratio.svg"
    fig.savefig(plot)
    plt.close(fig)
    print(f"Verified {len(later)} Train volumes; wrote {len(rows)} rows to {table} and {plot}")


if __name__ == "__main__":
    main()
