"""Check both frozen 3D run records and draw Train/Validation curves.

Reads local derived JSON only. Held-out Test data and raw MRI are not used.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "code"))
from hipmri_rewrite.metrics import overlap  # noqa: E402
from hipmri_rewrite.training import patient_small_organ_score  # noqa: E402


RUNS = {"Standard 3D": ("baseline_3d", "standard_3d", "672954", "674102"),
        "Improved 3D": ("improved_3d", "improved_3d", "672955", "674103")}
CONFIG_SHA = "e5bd98a99213f4fe6cf6e3e8c6db3b95d5ef334a143a91444426d8ea019923ce"
CLASS_NAMES = ("Background", "Body", "Bones", "Bladder", "Rectum", "Prostate")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def same_metric(left: float | None, right: float | None) -> bool:
    if left is None or right is None:
        return left is right
    return math.isclose(left, right, abs_tol=1e-12)


def load_checked_run(folder: str, model: str, job_id: str,
                     extension_job_id: str) -> dict:
    run = PROJECT / "logs/training" / folder
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    max_epochs = manifest["max_epochs"]
    if (manifest["model"] != model or manifest["job_id_initial"] != job_id
            or manifest["config_sha256"] != CONFIG_SHA or max_epochs not in (40, 50)):
        raise ValueError(f"{folder}: unexpected run identity")
    paths = sorted(run.glob("epoch_*.json"))
    if [p.name for p in paths] != [f"epoch_{epoch:03d}.json" for epoch in range(1, max_epochs + 1)]:
        raise ValueError(f"{folder}: incomplete or extra epoch records")
    records = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    validation = []
    case_ids = None
    running_best, running_epoch = -1.0, None
    for epoch, record in enumerate(records, 1):
        train = record["train"]
        exposure = train["exposure"]
        expected_job = job_id if epoch <= 40 else extension_job_id
        if (record["epoch"] != epoch or record["job_id"] != expected_job
                or len(record["selected_cases"]) != 26
                or len(train["batch_history"]) != 130
                or exposure["patches"] != 260
                or len(exposure["per_patient"]) != 26
                or any(item["patches"] != 10 for item in exposure["per_patient"].values())
                or not math.isfinite(train["mean_loss"])):
            raise ValueError(f"{folder}: invalid Train record at epoch {epoch}")
        val = record["validation"]
        if (epoch % 5 == 0) != (val is not None):
            raise ValueError(f"{folder}: unexpected Validation interval at epoch {epoch}")
        if val is not None:
            cases = val["cases"]
            ids = tuple(sorted(item["case"] for item in cases))
            if len(cases) != 38 or len(set(ids)) != 38 or (case_ids and case_ids != ids):
                raise ValueError(f"{folder}: Validation cohort changed at epoch {epoch}")
            case_ids = ids
            matrices: dict[str, np.ndarray] = {}
            for item in cases:
                matrix = np.asarray(item["confusion"], dtype=np.int64)
                if matrix.shape != (6, 6) or (matrix < 0).any():
                    raise ValueError(f"{folder}: invalid case confusion")
                matrices.setdefault(item["patient"], np.zeros((6, 6), dtype=np.int64))
                matrices[item["patient"]] += matrix
            if len(matrices) != 6:
                raise ValueError(f"{folder}: expected six Validation patients")
            pooled = sum(matrices.values(), np.zeros((6, 6), dtype=np.int64))
            if pooled.tolist() != val["pooled_confusion"]:
                raise ValueError(f"{folder}: pooled confusion differs from cases")
            recomputed = overlap(pooled)
            if any(not same_metric(recomputed[metric][i], val["pooled_overlap"][metric][i])
                   for metric in recomputed for i in range(6)):
                raise ValueError(f"{folder}: saved pooled metric differs from confusion")
            score, patients = patient_small_organ_score(matrices)
            if (not math.isclose(score, val["selection_score"], abs_tol=1e-12)
                    or patients != val["patient_small_organ_dice"]):
                raise ValueError(f"{folder}: patient selection score differs")
            if score > running_best:
                running_best, running_epoch = score, epoch
            validation.append((epoch, val))
        if record["best_epoch"] != running_epoch or not math.isclose(
                record["best_score"], running_best, abs_tol=1e-12):
            raise ValueError(f"{folder}: saved best-score history differs")
    if running_epoch is None:
        raise ValueError(f"{folder}: no selected Validation checkpoint")
    best = records[running_epoch - 1]
    return {"records": records, "validation": validation, "best": best,
            "max_epochs": max_epochs,
            "case_ids": case_ids, "manifest_sha256": sha256(run / "manifest.json"),
            "best_record_sha256": sha256(run / f"epoch_{running_epoch:03d}.json")}


def main() -> None:
    data = {label: load_checked_run(*identity) for label, identity in RUNS.items()}
    if data["Standard 3D"]["case_ids"] != data["Improved 3D"]["case_ids"]:
        raise ValueError("3D models used different Validation cases")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
    colors = {"Standard 3D": "#1f5a9d", "Improved 3D": "#d57228"}
    for label, result in data.items():
        color = colors[label]
        records = result["records"]
        val_epochs = [epoch for epoch, _ in result["validation"]]
        val_scores = [row["selection_score"] for _, row in result["validation"]]
        axes[0].plot(range(1, result["max_epochs"] + 1),
                     [row["train"]["mean_loss"] for row in records],
                     label=label, color=color, linewidth=2)
        axes[1].plot(val_epochs, val_scores, marker="o", label=label,
                     color=color, linewidth=2)
        best = result["best"]
        print(label, "best_epoch", best["best_epoch"],
              "selection_score", round(best["best_score"], 6),
              "manifest_sha256", result["manifest_sha256"],
              "best_record_sha256", result["best_record_sha256"])
        for name in ("Bladder", "Rectum", "Prostate"):
            index = CLASS_NAMES.index(name)
            print(" ", name, {metric: round(best["validation"]["pooled_overlap"][metric][index], 5)
                                for metric in ("dice", "precision", "recall")})
    names = ("Bladder", "Rectum", "Prostate")
    x = np.arange(len(names))
    for offset, (label, result) in zip((-0.18, 0.18), data.items()):
        dice = result["best"]["validation"]["pooled_overlap"]["dice"]
        axes[2].bar(x + offset, [dice[CLASS_NAMES.index(name)] for name in names],
                    width=0.35, color=colors[label], label=label)
    axes[0].set(title="Train patch loss", xlabel="Epoch", ylabel="Mean objective")
    axes[1].set(title="Validation patient score", xlabel="Epoch", ylabel="Mean small-organ Dice")
    axes[2].set(title="Validation pooled Dice at each selected epoch", ylabel="Dice")
    axes[2].set_xticks(x, names)
    axes[2].set_ylim(0, 1)
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].legend(loc="lower right", frameon=False)
    output = PROJECT / "results/figures/comparisons/g5_3d_validation.svg"
    fig.savefig(output, metadata={"Title": "HipMRI G5 3D Train and Validation"})
    plt.close(fig)
    print("Saved", output)


if __name__ == "__main__":
    main()
