"""Reconcile Rangpur G2 Train slice/histogram CSVs and plot their distributions."""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
OUTPUT = PROJECT / "results/data_analysis/g2_train_summary"
AUDIT = PROJECT / "logs/audits/train_full_20261018_job671208.json"
CLASSES = ("Background", "Body", "Bones", "Bladder", "Rectum", "Prostate")
ORGANS = ("Bladder", "Rectum", "Prostate")
EXPECTED_HASHES = {
    AUDIT.name: "6a4c674564a0364f5f56730cbc31e67b3ff9a8ffeae7d9d3dd18b198bbd0e1a1",
    "per_slice_class_coverage.csv": "d8b0966e9965797991f2313858d8088c1e2c32a62bcc483090be1a0dede17e87",
    "per_volume_intensity_histogram.csv": "c1b39892873b1b61dca3f03f8abcb0ec69a48154609a017a420f0502523d9930",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def check_hash(path: Path) -> None:
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != EXPECTED_HASHES[path.name]:
        raise ValueError(f"Unexpected SHA-256 for {path.name}: {actual}")


def main() -> None:
    for path in (AUDIT, OUTPUT / "per_slice_class_coverage.csv",
                 OUTPUT / "per_volume_intensity_histogram.csv"):
        check_hash(path)
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    expected = {case["case"]: case for case in audit["cases"]}
    slices = read_csv(OUTPUT / "per_slice_class_coverage.csv")
    histogram = read_csv(OUTPUT / "per_volume_intensity_histogram.csv")
    if len(expected) != 143 or len(slices) != 18320 or len(histogram) != 17160:
        raise ValueError("G2 row counts do not match the full Train cohort")

    by_case: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in slices:
        by_case[row["case"]].append(row)
    if set(by_case) != set(expected):
        raise ValueError("Slice CSV case set differs from G1")
    case_summary = []
    for case, rows in sorted(by_case.items()):
        prior = expected[case]
        nz = prior["mri_shape"][2]
        if len(rows) != nz or {int(r["axial_slice_index_z"]) for r in rows} != set(range(nz)):
            raise ValueError(f"Missing or duplicate axial slice: {case}")
        if any(int(r["slice_voxels"]) != prior["mri_shape"][0] * prior["mri_shape"][1]
               for r in rows):
            raise ValueError(f"Unexpected slice size: {case}")
        summary = {"case": case, "patient": case.split("_", 1)[0], "axial_slices": nz}
        for class_name in CLASSES:
            key = class_name.lower()
            values = [int(r[f"{key}_voxels"]) for r in rows]
            hits = sum(value > 0 for value in values)
            if sum(values) != prior["class_counts"][class_name]:
                raise ValueError(f"Slice totals differ from G1: {case} {class_name}")
            if any(int(r[f"{key}_present"]) != int(value > 0)
                   for r, value in zip(rows, values)):
                raise ValueError(f"Slice presence flag differs from count: {case} {class_name}")
            summary[f"{key}_present_slices"] = hits
            summary[f"{key}_slice_hit_pct"] = 100 * hits / nz
        case_summary.append(summary)

    by_hist_case: dict[str, list[dict[str, str]]] = defaultdict(list)
    pooled = Counter()
    for row in histogram:
        by_hist_case[row["case"]].append(row)
        low, high, n = int(row["lower_inclusive"]), int(row["upper_exclusive"]), int(row["positive_voxels"])
        if high - low != 10 or not 0 <= low < 1200 or n < 0:
            raise ValueError("Invalid intensity-bin definition or count")
        pooled[(low, high)] += n
    if set(by_hist_case) != set(expected):
        raise ValueError("Intensity histogram case set differs from G1")
    for case, rows in by_hist_case.items():
        if len(rows) != 120 or {int(r["lower_inclusive"]) for r in rows} != set(range(0, 1200, 10)):
            raise ValueError(f"Missing or duplicate intensity bins: {case}")
        prior = expected[case]
        positive_expected = prior["image_finite_voxels"] - round(
            prior["image_zero_fraction"] * prior["image_finite_voxels"]
        )
        if sum(int(r["positive_voxels"]) for r in rows) != positive_expected:
            raise ValueError(f"Positive intensity total differs from G1: {case}")

    class_rows = []
    for class_name in CLASSES:
        key = class_name.lower()
        hits = [r[f"{key}_present_slices"] for r in case_summary]
        class_rows.append({"class": class_name, "present_slices": sum(hits),
                           "total_slices": len(slices), "slice_hit_pct": 100 * sum(hits) / len(slices),
                           "median_present_slices_per_volume": statistics.median(hits),
                           "min_present_slices_per_volume": min(hits),
                           "max_present_slices_per_volume": max(hits)})
    intensity_rows = [{"lower_inclusive": low, "upper_exclusive": high,
                       "positive_voxels": pooled[(low, high)]}
                      for low, high in zip(range(0, 1200, 10), range(10, 1210, 10))]
    if sum(r["positive_voxels"] for r in intensity_rows) != sum(
            case["image_finite_voxels"] - round(case["image_zero_fraction"] * case["image_finite_voxels"])
            for case in expected.values()):
        raise ValueError("Pooled intensity total mismatch")
    write_csv(OUTPUT / "slice_presence_summary.csv", class_rows, list(class_rows[0]))
    write_csv(OUTPUT / "per_volume_slice_coverage.csv", case_summary, list(case_summary[0]))
    write_csv(OUTPUT / "pooled_positive_intensity_histogram.csv", intensity_rows, list(intensity_rows[0]))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none",
                         "svg.hashsalt": "hipmri-g2-train"})
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    bars = ax.bar([r["class"] for r in class_rows],
                  [r["slice_hit_pct"] for r in class_rows], color="#377d93")
    ax.bar_label(bars, fmt="%.1f%%", padding=3)
    ax.set(ylabel="Axial slices containing class (%)", ylim=(0, 110),
           title="Raw Train slice coverage (18,320 axial slices)")
    fig.savefig(OUTPUT / "raw_axial_slice_hit_rate.svg", metadata={"Date": None})
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    for organ, color in (("Bladder", "#2d7d69"), ("Rectum", "#cc7843"), ("Prostate", "#8e61a9")):
        ax.hist([r[f"{organ.lower()}_present_slices"] for r in case_summary],
                bins=18, alpha=0.55, label=organ, color=color)
    ax.set(xlabel="Axial slices containing organ per volume", ylabel="Train volumes",
           title="Rare-organ slice coverage across 143 Train volumes")
    ax.legend()
    fig.savefig(OUTPUT / "organ_slices_per_volume.svg", metadata={"Date": None})
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    ax.step([r["lower_inclusive"] for r in intensity_rows],
            [r["positive_voxels"] for r in intensity_rows], where="post", color="#447b9a")
    ax.set(xlabel="Raw MRI intensity (nonzero, 10-unit bins)",
           ylabel="Positive-intensity Train voxels", title="Pooled raw Train MRI intensity")
    fig.savefig(OUTPUT / "pooled_positive_intensity_histogram.svg", metadata={"Date": None})
    plt.close(fig)

    for svg in OUTPUT.glob("*.svg"):
        svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n",
                       encoding="utf-8")
    for row in class_rows:
        print(f"{row['class']}: {row['present_slices']}/{row['total_slices']} slices ({row['slice_hit_pct']:.3f}%)")
    print("Saved three reconciled CSVs and three SVG plots")


if __name__ == "__main__":
    main()
