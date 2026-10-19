"""Verify frozen derived records and plot all six classes and G6 error patterns.

The held-out Test is read only for descriptive review; no model is selected.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from summarize_g5_validation import RUNS, load_checked_run
from summarize_g6_test import check as check_test, sha256
from hipmri_rewrite.metrics import overlap


ROOT = Path(__file__).resolve().parents[2]
FIGURES = ROOT / "results/figures/comparisons"
TWO_D_FIGURES = ROOT / "results/figures/baseline_2d"
TABLES = ROOT / "results/tables/comparisons"
CLASSES = ("Background", "Body", "Bones", "Bladder", "Rectum", "Prostate")
ARMS = ("standard_3d", "improved_3d")
LABELS = ("Standard 3D", "Improved 3D")
COLORS = ("#246391", "#c36b29")
THREE_ARM_COLORS = ("#5891b5", "#246391", "#c36b29")


def save(fig, path: Path) -> None:
    fig.savefig(path.with_suffix(".svg"), metadata={"Title": path.stem})
    fig.savefig(path.with_suffix(".png"), dpi=180)
    plt.close(fig)


def paired_bars(metrics: dict[str, list[float]], title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.8, 5.5), layout="constrained")
    y = np.arange(6)
    for offset, label, color in zip((-0.19, 0.19), LABELS, COLORS):
        values = metrics[label]
        ax.barh(y + offset, values, height=0.36, label=label, color=color)
        for yy, value in zip(y + offset, values):
            ax.text(min(value + 0.009, 1.01), yy, f"{value:.3f}", va="center", fontsize=8)
    ax.set_yticks(y, CLASSES)
    ax.invert_yaxis()
    ax.set(xlim=(0, 1.085), xlabel="Pooled Dice", title=title)
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, path)


def validation_figures() -> None:
    runs = {label: load_checked_run(*identity) for label, identity in RUNS.items()}
    if runs[LABELS[0]]["case_ids"] != runs[LABELS[1]]["case_ids"]:
        raise ValueError("3D Validation cohorts differ")
    metrics = {}
    for label in LABELS:
        record = runs[label]["best"]
        metrics[label] = record["validation"]["pooled_overlap"]["dice"]
    paired_bars(metrics, "Validation · all six classes · selected 3D epochs",
                FIGURES / "g5_3d_validation_all_classes")

    case_rows = []
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), layout="constrained")
    for ax, label in zip(axes, LABELS):
        cases = runs[label]["best"]["validation"]["cases"]
        patient_matrices = {}
        for case in cases:
            matrix = np.asarray(case["confusion"], dtype=np.int64)
            patient_matrices.setdefault(case["patient"], np.zeros((6, 6), dtype=np.int64))
            patient_matrices[case["patient"]] += matrix
            scores = overlap(matrix)
            for i, name in enumerate(CLASSES):
                case_rows.append({
                    "split": "validation", "model": label, "patient": case["patient"],
                    "case": case["case"], "class": name,
                    "gt_voxels": int(matrix[i].sum()),
                    "predicted_voxels": int(matrix[:, i].sum()),
                    "tp": int(matrix[i, i]),
                    "fp": int(matrix[:, i].sum() - matrix[i, i]),
                    "fn": int(matrix[i].sum() - matrix[i, i]),
                    **{key: scores[key][i]
                       for key in ("dice", "iou", "precision", "recall")},
                })
        patients = sorted(patient_matrices)
        grid = np.array([overlap(patient_matrices[patient])["dice"]
                         for patient in patients], dtype=float)
        ax.imshow(grid, vmin=0, vmax=1, cmap="Blues", aspect="auto")
        ax.set_xticks(range(6), CLASSES, rotation=45, ha="right")
        ax.set_yticks(range(6), patients)
        ax.set_title(f"{label} · patient Dice")
        for y in range(6):
            for x in range(6):
                value = grid[y, x]
                ax.text(x, y, "NA" if np.isnan(value) else f"{value:.2f}",
                        ha="center", va="center", fontsize=8,
                        color="white" if value > 0.65 else "black")
    fig.suptitle("Validation · all classes · visits pooled within each patient")
    save(fig, FIGURES / "g5_validation_patient_class_dice")
    if len(case_rows) != 2 * 38 * 6:
        raise ValueError("Incomplete Validation case-class table")
    with (TABLES / "g5_validation_case_class_metrics.csv").open(
            "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(case_rows[0]))
        writer.writeheader()
        writer.writerows(case_rows)

    run = ROOT / "logs/training/baseline_2d"
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    if (manifest["kind"] != "G5_frozen_2D_Train_Validation_cohort_run"
            or manifest["config_sha256"] != sha256(
                ROOT / "code/config/g5_2d_training_frozen.json")):
        raise ValueError("Unexpected 2D training receipt")
    epoch = manifest["max_epochs"]
    files = sorted(run.glob("epoch_*.json"))
    if [p.name for p in files] != [f"epoch_{i:03d}.json" for i in range(1, epoch + 1)]:
        raise ValueError("2D epoch records are incomplete")
    final = json.loads(files[-1].read_text(encoding="utf-8"))
    chosen = json.loads((run / f"epoch_{final['best_epoch']:03d}.json").read_text(
        encoding="utf-8"))
    if chosen["validation"] is None or chosen["best_score"] != final["best_score"]:
        raise ValueError("2D selected Validation receipt differs")
    values = chosen["validation"]["pooled_overlap"]["dice"]
    fig, ax = plt.subplots(figsize=(9.2, 5.4), layout="constrained")
    y = np.arange(6)
    ax.barh(y, values, color="#4a779d", height=0.65)
    for yy, value in zip(y, values):
        ax.text(min(value + 0.01, 1.01), yy, f"{value:.3f}", va="center")
    ax.set_yticks(y, CLASSES)
    ax.invert_yaxis()
    ax.set(xlim=(0, 1.085), xlabel="Pooled Dice",
           title=f"Supplementary 2D Validation · all six classes · epoch {final['best_epoch']}")
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, TWO_D_FIGURES / "g5_2d_validation_all_classes")


def test_figures_and_table() -> None:
    data = check_test()
    if tuple(data["classes"]) != CLASSES:
        raise ValueError("Class order differs")
    results = data["arms"]
    paired_bars({label: results[arm]["pooled_overlap"]["dice"]
                 for arm, label in zip(ARMS, LABELS)},
                "Held-out Test · all six classes · pooled Dice",
                FIGURES / "g6_test_all_classes")

    fig, ax = plt.subplots(figsize=(9.8, 5.1), layout="constrained")
    x = np.arange(1, 6)
    for offset, arm, label, color in zip((-0.19, 0.19), ARMS, LABELS, COLORS):
        values = [results[arm]["boundary_summary"][name]["median_hd95_mm"]
                  for name in CLASSES[1:]]
        ax.bar(x + offset, values, width=0.36, label=label, color=color)
        for xx, value in zip(x + offset, values):
            ax.text(xx, value + 1, f"{value:.1f}", ha="center", fontsize=8)
    ax.set_xticks(x, CLASSES[1:])
    ax.set(ylabel="Median per-volume HD95 (mm)",
           title="Held-out Test · boundary distance by foreground class")
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, FIGURES / "g6_test_hd95_all_foreground")

    rows = []
    for arm in ARMS:
        for case in results[arm]["cases"]:
            matrix = np.asarray(case["confusion"], dtype=np.int64)
            for i, name in enumerate(CLASSES):
                rows.append({
                    "split": "test", "model": arm, "patient": case["patient"],
                    "case": case["case"], "class": name,
                    "gt_voxels": int(matrix[i].sum()),
                    "predicted_voxels": int(matrix[:, i].sum()),
                    "tp": int(matrix[i, i]),
                    "fp": int(matrix[:, i].sum() - matrix[i, i]),
                    "fn": int(matrix[i].sum() - matrix[i, i]),
                    **{key: case["overlap"][key][i]
                       for key in ("dice", "iou", "precision", "recall")},
                    "hd95_mm": (case["boundary"][name]["hd95_mm"]
                                  if i else None),
                    "boundary_status": (case["boundary"][name]["status"]
                                        if i else "not_applicable"),
                })
    path = TABLES / "g6_test_case_class_metrics.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # These panels explain fixed Test errors; they do not define a new model rule.
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.7), layout="constrained")
    for arm, label, color in zip(ARMS, LABELS, COLORS):
        cases = {row["case"]: row for row in results[arm]["cases"]}
        m015 = sorted((row for row in cases.values() if row["patient"] == "M015"),
                      key=lambda row: int(row["case"].split("Week")[1]))
        axes[0].plot(range(len(m015)), [row["overlap"]["dice"][3] for row in m015],
                     marker="o", label=label, color=color)
        patients = sorted({row["patient"] for row in cases.values()})
        positions = {patient: i for i, patient in enumerate(patients)}
        for row in cases.values():
            axes[1].scatter(positions[row["patient"]] + (-0.10 if arm == ARMS[0] else 0.10),
                            row["boundary"]["Body"]["hd95_mm"],
                            color=color, s=20, alpha=0.65)
    axes[0].set(title="M015 · Bladder Dice across visits", xlabel="Week", ylabel="Dice",
                xticks=range(8), ylim=(0, 1))
    axes[1].set(title="Body HD95 · every Test volume", ylabel="HD95 (mm)",
                xticks=range(len(patients)), xticklabels=patients)
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False)
    fig.suptitle("Held-out Test descriptive failure review · no checkpoint tuning")
    save(fig, FIGURES / "g6_test_failure_patterns")
    print("Verified Test SHA-256", sha256(ROOT / "results/tables/comparisons/g6_test_3d.json"))
    print("Saved", len(rows), "case-class rows and all-class/failure figures")


def three_arm_validation_figures() -> None:
    """Compare the current selected Validation records without mixing in Test."""
    names = ("Supplementary 2D", "Standard 3D", "Improved 3D")
    run_names = ("baseline_2d", "baseline_3d", "improved_3d")
    records = []
    selected = []
    for folder in run_names:
        run = ROOT / "logs/training" / folder
        manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
        if manifest["max_epochs"] != 50:
            raise ValueError(f"{folder}: expected completed 50-epoch run")
        paths = sorted(run.glob("epoch_*.json"))
        if [p.name for p in paths] != [f"epoch_{i:03d}.json" for i in range(1, 51)]:
            raise ValueError(f"{folder}: incomplete epoch records")
        history = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
        final = history[-1]
        if final["epoch"] != 50:
            raise ValueError(f"{folder}: wrong final epoch")
        chosen = history[final["best_epoch"] - 1]
        if (chosen["validation"] is None
                or chosen["validation"]["selection_score"] != final["best_score"]):
            raise ValueError(f"{folder}: selected record does not match final receipt")
        records.append(history)
        selected.append(chosen["validation"])
    case_sets = [{row["case"] for row in val["cases"]} for val in selected]
    if any(cases != case_sets[0] or len(cases) != 38 for cases in case_sets):
        raise ValueError("The three Validation cohorts differ")

    fig, ax = plt.subplots(figsize=(10.6, 5.8), layout="constrained")
    y = np.arange(6)
    for offset, name, color, val in zip((-0.25, 0, 0.25), names,
                                        THREE_ARM_COLORS, selected):
        ax.barh(y + offset, val["pooled_overlap"]["dice"], height=0.24,
                label=name, color=color)
    ax.set_yticks(y, CLASSES)
    ax.invert_yaxis()
    ax.set(xlim=(0, 1.02), xlabel="Pooled Dice",
           title="Validation · all six classes · selected epoch 50")
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, FIGURES / "g5_three_arm_validation_all_classes")

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), layout="constrained")
    for name, color, history in zip(names, THREE_ARM_COLORS, records):
        axes[0].plot(range(1, 51), [r["train"]["mean_loss"] for r in history],
                     label=name, color=color)
        points = [(r["epoch"], r["validation"]["selection_score"])
                  for r in history if r["validation"] is not None]
        axes[1].plot(*zip(*points), marker="o", label=name, color=color)
    axes[0].set(title="Train objective · compare trends within each arm",
                xlabel="Epoch", ylabel="Mean loss")
    axes[1].set(title="Patient-balanced small-organ Validation score",
                xlabel="Epoch", ylabel="Mean Dice", ylim=(0, 1))
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].legend(frameon=False)
    save(fig, FIGURES / "g5_three_arm_training_trajectories")


def main() -> None:
    validation_figures()
    three_arm_validation_figures()
    test_figures_and_table()


if __name__ == "__main__":
    main()
