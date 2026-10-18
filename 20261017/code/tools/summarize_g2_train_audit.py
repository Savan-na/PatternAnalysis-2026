"""Summarize the complete G1 Train audit into reproducible G2 tables and plots.

Run from any directory with the existing Matplotlib environment:
python -B code/tools/summarize_g2_train_audit.py --audit logs/audits/train_full_20261018_job671208.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CLASSES = ("Background", "Body", "Bones", "Bladder", "Rectum", "Prostate")
ORGANS = ("Bladder", "Rectum", "Prostate")
EXPECTED_AUDIT_SHA256 = "6a4c674564a0364f5f56730cbc31e67b3ff9a8ffeae7d9d3dd18b198bbd0e1a1"
EXPECTED_SPLIT_SHA256 = "81531ec7cb84ef58c41393ff23e1105e7cf03aa97828fb12853ce8624fcb8900"


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=ROOT / "logs/audits/train_full_20261018_job671208.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/data_analysis/g2_train_summary")
    args = parser.parse_args()

    source = args.audit.resolve()
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != EXPECTED_AUDIT_SHA256:
        raise ValueError(f"Unexpected audit SHA-256: {digest}")
    audit = json.loads(source.read_text(encoding="utf-8"))
    cases = audit["cases"]
    if not (audit["scope"] == "G1 raw Train arrays decoded; all split filenames inventoried only; no preprocessing"
            and audit["complete_train_audit"]
            and audit["mechanical_pass"] and len(cases) == 143
            and audit["paired_train_cases"] == 143
            and audit["expected_train_cases"] == 143
            and audit["split_file_sha256"] == EXPECTED_SPLIT_SHA256):
        raise ValueError("Input is not the complete, approved Train audit")
    if len({r["case"] for r in cases}) != 143 or any(r["errors"] for r in cases):
        raise ValueError("Duplicate cases or case errors in source")

    counts = Counter()
    patient_counts: dict[str, Counter] = defaultdict(Counter)
    patient_exams = Counter()
    volume_rows = []
    intensity_rows = []
    spatial_rows = []
    for r in sorted(cases, key=lambda x: x["case"]):
        case = r["case"]
        patient, visit = case.split("_", 1)
        if not visit.startswith("Week") or set(r["class_counts"]) != set(CLASSES):
            raise ValueError(f"Unexpected identity or class set: {case}")
        n = sum(r["class_counts"].values())
        if n != r["image_finite_voxels"] or n != r["mask_finite_voxels"]:
            raise ValueError(f"Voxel count mismatch: {case}")
        counts.update(r["class_counts"])
        patient_counts[patient].update(r["class_counts"])
        patient_exams[patient] += 1
        vr = {"case": case, "patient": patient, "visit": visit, "total_voxels": n}
        for c in CLASSES:
            vr[f"{c.lower()}_voxels"] = r["class_counts"][c]
            vr[f"{c.lower()}_share_pct"] = 100 * r["class_counts"][c] / n
        volume_rows.append(vr)
        p = r["image_nonzero_percentiles_0_5_1_50_99_99_5"]
        intensity_rows.append({
            "case": case, "patient": patient, "finite_min": r["image_finite_min"],
            "finite_median": r["image_finite_median"], "finite_max": r["image_finite_max"],
            "zero_fraction": r["image_zero_fraction"],
            "nonzero_mean": r["image_nonzero_mean"], "nonzero_std": r["image_nonzero_std"],
            "nonzero_p0_5": p[0], "nonzero_p1": p[1], "nonzero_p50": p[2],
            "nonzero_p99": p[3], "nonzero_p99_5": p[4],
        })
        sr = {"case": case, "patient": patient,
              "shape_x": r["mri_shape"][0], "shape_y": r["mri_shape"][1],
              "shape_z": r["mri_shape"][2],
              "spacing_x_mm": r["mri_spacing_mm"][0],
              "spacing_y_mm": r["mri_spacing_mm"][1],
              "spacing_z_mm": r["mri_spacing_mm"][2],
              "voxel_volume_mm3": r["mask_voxel_volume_mm3"],
              "orientation": "".join(r["mri_orientation"])}
        for organ in ORGANS:
            geom = r["small_organ_geometry"][organ]
            sr[f"{organ.lower()}_volume_ml"] = r["class_volume_ml"][organ]
            for axis, name in enumerate("xyz"):
                minimum = geom["bounding_box_min_xyz"][axis]
                maximum = geom["bounding_box_max_xyz"][axis]
                sr[f"{organ.lower()}_bbox_min_{name}_voxel"] = minimum
                sr[f"{organ.lower()}_bbox_max_{name}_voxel"] = maximum
                sr[f"{organ.lower()}_bbox_center_{name}_fraction"] = (
                    (minimum + maximum) / 2 / (r["mri_shape"][axis] - 1)
                )
                sr[f"{organ.lower()}_bbox_{name}_mm"] = (
                    maximum - minimum + 1
                ) * r["mri_spacing_mm"][axis]
                sr[f"{organ.lower()}_occupied_{name}_slices"] = geom["occupied_slices_xyz"][axis]
        spatial_rows.append(sr)

    total = sum(counts.values())
    class_rows = [{"class": c, "voxel_count": counts[c],
                   "share_pct": 100 * counts[c] / total,
                   "denominator_voxels": total, "train_volumes": len(cases)} for c in CLASSES]
    patient_rows = []
    for patient in sorted(patient_counts):
        pc = patient_counts[patient]
        pn = sum(pc.values())
        row = {"patient": patient, "volume_count": patient_exams[patient], "total_voxels": pn}
        for c in CLASSES:
            row[f"{c.lower()}_voxels"] = pc[c]
            row[f"{c.lower()}_share_pct"] = 100 * pc[c] / pn
        patient_rows.append(row)
    if sum(r["volume_count"] for r in patient_rows) != 143 or len(patient_rows) != 26:
        raise ValueError("Patient totals do not match split")

    spatial_by_patient: dict[str, list[dict]] = defaultdict(list)
    for row in spatial_rows:
        spatial_by_patient[row["patient"]].append(row)
    longitudinal_rows = []
    for patient in sorted(spatial_by_patient):
        visits = spatial_by_patient[patient]
        row = {"patient": patient, "volume_count": len(visits)}
        for organ in ORGANS:
            volumes = [entry[f"{organ.lower()}_volume_ml"] for entry in visits]
            row[f"{organ.lower()}_min_ml"] = min(volumes)
            row[f"{organ.lower()}_max_ml"] = max(volumes)
            row[f"{organ.lower()}_max_min_ratio"] = max(volumes) / min(volumes) if len(visits) > 1 else ""
        longitudinal_rows.append(row)

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("class_distribution", class_rows), ("per_volume_distribution", volume_rows),
                       ("per_patient_distribution", patient_rows), ("intensity_statistics", intensity_rows),
                       ("spatial_statistics", spatial_rows),
                       ("longitudinal_organ_variation", longitudinal_rows)):
        write_csv(out / f"{name}.csv", rows, list(rows[0]))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none",
                         "svg.hashsalt": "hipmri-g2-train"})
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    bars = ax.bar(CLASSES, [counts[c] for c in CLASSES], color="#286f8e")
    ax.set_yscale("log")
    ax.set_ylabel("Train voxels (log scale)")
    ax.set_title("Raw class frequency, 143 Train volumes")
    ax.bar_label(bars, labels=[f"{100 * counts[c] / total:.3f}%" for c in CLASSES], padding=3, fontsize=8)
    ax.set_ylim(1e5, 5e9)
    fig.savefig(out / "class_frequency_log.svg", metadata={"Date": None})
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5), layout="constrained")
    positions = range(len(patient_rows))
    for organ, color in (("Bladder", "#2d7d69"), ("Rectum", "#cc7843"), ("Prostate", "#8e61a9")):
        ax.plot(positions, [r[f"{organ.lower()}_share_pct"] for r in patient_rows],
                marker="o", markersize=3, linewidth=1, label=organ, color=color)
    ax.set_xticks(list(positions), [r["patient"] for r in patient_rows], rotation=60)
    ax.set_ylabel("Share of raw voxels per patient (%)")
    ax.set_title("Rare organ frequency by Train patient (visits pooled)")
    ax.legend(ncols=3)
    fig.savefig(out / "patient_rare_organ_share.svg", metadata={"Date": None})
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    axes[0].hist([r["nonzero_p50"] for r in intensity_rows], bins=15, color="#477b9b")
    axes[0].set(xlabel="Per-volume nonzero median intensity", ylabel="Train volumes",
                title="MRI intensity scale")
    axes[1].hist([100 * r["zero_fraction"] for r in intensity_rows], bins=15, color="#a07175")
    axes[1].set(xlabel="Zero voxels per volume (%)", ylabel="Train volumes",
                title="Zero intensity frequency")
    fig.savefig(out / "intensity_case_summary.svg", metadata={"Date": None})
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    ax.scatter([r["spacing_x_mm"] for r in spatial_rows],
               [r["spacing_z_mm"] for r in spatial_rows], alpha=0.55, s=24, color="#397b80")
    ax.set(xlabel="In-plane voxel spacing x (mm)", ylabel="Through-plane spacing z (mm)",
           title="Raw Train MRI voxel spacing (143 volumes)")
    fig.savefig(out / "voxel_spacing.svg", metadata={"Date": None})
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained")
    for organ, color in (("Bladder", "#2d7d69"), ("Rectum", "#cc7843"), ("Prostate", "#8e61a9")):
        key = organ.lower()
        axes[0].scatter([r[f"{key}_bbox_center_x_fraction"] for r in spatial_rows],
                        [r[f"{key}_bbox_center_y_fraction"] for r in spatial_rows],
                        s=14, alpha=0.55, label=organ, color=color)
        axes[1].hist([r[f"{key}_bbox_center_z_fraction"] for r in spatial_rows],
                     bins=14, alpha=0.5, label=organ, color=color)
    axes[0].set(xlabel="Raw grid x fraction", ylabel="Raw grid y fraction",
                title="Organ box centers in-plane", xlim=(0, 1), ylim=(0, 1))
    axes[1].set(xlabel="Raw grid z fraction", ylabel="Train volumes",
                title="Organ box centers through-plane")
    axes[0].legend()
    fig.savefig(out / "organ_bbox_centers.svg", metadata={"Date": None})
    plt.close(fig)

    # Matplotlib writes spaces after many SVG path line breaks. Keep generated
    # figures stable under Git's whitespace check without changing path data.
    for svg in out.glob("*.svg"):
        cleaned = "\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n"
        svg.write_text(cleaned, encoding="utf-8")

    print(f"Source SHA-256: {digest}")
    print(f"Train cases: {len(cases)}, patients: {len(patient_rows)}, voxels: {total}")
    for row in class_rows:
        print(f"{row['class']}: {row['voxel_count']:,} ({row['share_pct']:.6f}%)")
    print(f"Saved six CSV tables and five SVG figures to {out}")


if __name__ == "__main__":
    main()
