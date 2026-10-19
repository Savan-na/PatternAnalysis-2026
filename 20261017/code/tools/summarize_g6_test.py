"""Verify the one-time held-out Test JSON and draw a descriptive comparison.

This reads derived metrics only. Test results do not select or retrain a model.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "code"))
from hipmri_rewrite.metrics import overlap  # noqa: E402
from hipmri_rewrite.training import patient_small_organ_score  # noqa: E402

RESULT = PROJECT / "results/tables/comparisons/g6_test_3d.json"
ARMS = {"standard_3d": "baseline_3d", "improved_3d": "improved_3d"}
ORGANS = ("Bladder", "Rectum", "Prostate")
COLORS = {"standard_3d": "#1f5a9d", "improved_3d": "#d57228"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def same(left, right) -> bool:
    if left is None or right is None:
        return left is right
    return math.isclose(left, right, rel_tol=0, abs_tol=1e-12)


def check() -> dict:
    data = json.loads(RESULT.read_text(encoding="utf-8"))
    split = json.loads((PROJECT / "code/config/patient_split.json").read_text(encoding="utf-8"))
    if (data["kind"] != "G6_one_time_held_out_Test" or data["split"] != "test"
            or data["case_count"] != 30 or data["patient_count"] != 6
            or data["job_id"] != "674125"
            or data["evaluation_script_sha256"] != sha256(
                PROJECT / "code/tools/evaluate_g6_test.py")
            or set(data["arms"]) != set(ARMS)
            or len(split["test"]) != 6):
        raise ValueError("Unexpected Test result identity")
    cohort = None
    for arm, folder in ARMS.items():
        result = data["arms"][arm]
        receipt = result["checkpoint"]
        run = PROJECT / "logs/training" / folder
        if (receipt["model"] != arm or receipt["selected_epoch"] != 50
                or receipt["manifest_sha256"] != sha256(run / "manifest.json")
                or receipt["final_record_sha256"] != sha256(run / "epoch_050.json")
                or receipt["selected_record_sha256"] != sha256(run / "epoch_050.json")):
            raise ValueError(f"{arm}: Test checkpoint receipt differs from frozen run")
        final = json.loads((run / "epoch_050.json").read_text(encoding="utf-8"))
        if (final["best_epoch"] != 50 or not same(final["best_score"],
                                                 receipt["validation_selection_score"])):
            raise ValueError(f"{arm}: final Validation selection differs")
        cases = result["cases"]
        ids = tuple(sorted(row["case"] for row in cases))
        if (len(cases) != 30 or len(set(ids)) != 30
                or {row["patient"] for row in cases} != set(split["test"])
                or any(row["case"].split("_Week")[0] != row["patient"] for row in cases)
                or (cohort is not None and ids != cohort)):
            raise ValueError(f"{arm}: wrong Test cohort")
        cohort = ids
        matrices = defaultdict(lambda: np.zeros((6, 6), dtype=np.int64))
        for row in cases:
            matrix = np.asarray(row["confusion"], dtype=np.int64)
            if matrix.shape != (6, 6) or (matrix < 0).any():
                raise ValueError(f"{arm}: invalid confusion matrix")
            matrices[row["patient"]] += matrix
            calculated = overlap(matrix)
            if any(not same(calculated[metric][index], row["overlap"][metric][index])
                   for metric in calculated for index in range(6)):
                raise ValueError(f"{arm}: saved case overlap differs")
        pooled = sum(matrices.values(), np.zeros((6, 6), dtype=np.int64))
        if pooled.tolist() != result["pooled_confusion"]:
            raise ValueError(f"{arm}: pooled confusion differs")
        calculated = overlap(pooled)
        if any(not same(calculated[metric][index], result["pooled_overlap"][metric][index])
               for metric in calculated for index in range(6)):
            raise ValueError(f"{arm}: pooled overlap differs")
        score, patients = patient_small_organ_score(dict(matrices))
        if (not same(score, result["patient_small_organ_score"])
                or patients != result["per_patient_small_organ_dice"]):
            raise ValueError(f"{arm}: patient score differs")
        for label in data["classes"][1:]:
            values = [row["boundary"][label]["hd95_mm"] for row in cases
                      if row["boundary"][label]["status"] == "both_present"]
            boundary = result["boundary_summary"][label]
            if not same(median(values) if values else None, boundary["median_hd95_mm"]):
                raise ValueError(f"{arm}: HD95 summary differs")
        if (not same(median(row["inference_seconds"] for row in cases),
                     result["median_inference_seconds"])
                or max(row["peak_allocated_bytes"] for row in cases) !=
                result["max_peak_allocated_bytes"]):
            raise ValueError(f"{arm}: resource summary differs")
    return data


def main() -> None:
    data = check()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.7), layout="constrained")
    positions = np.arange(len(ORGANS))
    for offset, arm in ((-0.19, "standard_3d"), (0.19, "improved_3d")):
        result = data["arms"][arm]
        dice = result["pooled_overlap"]["dice"]
        values = [dice[data["classes"].index(name)] for name in ORGANS]
        axes[0].bar(positions + offset, values, width=0.36, color=COLORS[arm],
                    label=arm.replace("_", " ").title())
    axes[0].set(title="Held-out Test pooled Dice", ylabel="Dice", ylim=(0, 1))
    axes[0].set_xticks(positions, ORGANS)
    axes[0].legend(frameon=False)
    patients = sorted(data["arms"]["standard_3d"]["per_patient_small_organ_dice"])
    x = np.arange(len(patients))
    for offset, arm in ((-0.19, "standard_3d"), (0.19, "improved_3d")):
        details = data["arms"][arm]["per_patient_small_organ_dice"]
        values = [np.mean([value for value in details[patient].values()
                           if value is not None]) for patient in patients]
        axes[1].bar(x + offset, values, width=0.36, color=COLORS[arm])
    axes[1].set(title="Held-out Test patient small-organ score", ylabel="Mean Dice",
                ylim=(0, 1))
    axes[1].set_xticks(x, patients)
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    output = PROJECT / "results/figures/comparisons/g6_test_3d.svg"
    fig.savefig(output, metadata={"Title": "HipMRI held-out Test 3D comparison"})
    fig.savefig(output.with_suffix(".png"), dpi=180)
    plt.close(fig)
    print("Test SHA-256", sha256(RESULT))
    for arm in ARMS:
        result = data["arms"][arm]
        print(arm, "Validation epoch", result["checkpoint"]["selected_epoch"],
              "Test patient score", round(result["patient_small_organ_score"], 6))
        for name in ORGANS:
            index = data["classes"].index(name)
            print(" ", name, "Dice", round(result["pooled_overlap"]["dice"][index], 6),
                  "HD95 median mm", round(result["boundary_summary"][name]["median_hd95_mm"], 3))
    print("Saved", output)


if __name__ == "__main__":
    main()
