"""Check the supplementary 2D G5 records and draw its Validation trajectory."""

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

RUN = PROJECT / "logs/training/baseline_2d"
CONFIG_SHA = "3ae7717569b6434d250b6b906c60e62fac23670e400341153b837aaf322f0ec2"
ORGANS = (("Bladder", 3), ("Rectum", 4), ("Prostate", 5))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    manifest = json.loads((RUN / "manifest.json").read_text(encoding="utf-8"))
    if (manifest["kind"] != "G5_frozen_2D_Train_Validation_cohort_run"
            or manifest["model"] != "baseline_2d"
            or manifest["job_id_initial"] != "673908"
            or manifest["config_sha256"] != CONFIG_SHA
            or manifest["max_epochs"] not in (40, 50)
            or sha256(PROJECT / "code/config/g5_2d_training_frozen.json") != CONFIG_SHA):
        raise ValueError("Unexpected 2D run or frozen configuration")
    max_epochs = manifest["max_epochs"]
    paths = sorted(RUN.glob("epoch_*.json"))
    if [path.name for path in paths] != [f"epoch_{epoch:03d}.json" for epoch in range(1, max_epochs + 1)]:
        raise ValueError("Incomplete 2D epoch records")
    records = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    best_score, best_epoch = -1.0, None
    validation = []
    case_set = None
    for epoch, record in enumerate(records, 1):
        train = record["train"]
        exposure = train["exposure"]
        expected_job = "673908" if epoch <= 40 else manifest["extensions"][-1]["job_id"]
        if (record["epoch"] != epoch or record["job_id"] != expected_job
                or len(record["selected_cases"]) != 26
                or len(train["batch_history"]) != 130
                or exposure["patches"] != 260
                or len(exposure["per_patient"]) != 26
                or not math.isfinite(train["mean_loss"])):
            raise ValueError(f"Invalid 2D Train record at epoch {epoch}")
        val = record["validation"]
        if (epoch % 5 == 0) != (val is not None):
            raise ValueError(f"Unexpected 2D Validation interval at epoch {epoch}")
        if val is not None:
            cases = val["cases"]
            ids = tuple(sorted(row["case"] for row in cases))
            if len(cases) != 38 or len(set(ids)) != 38 or (case_set and ids != case_set):
                raise ValueError("2D Validation cohort changed")
            case_set = ids
            matrices = {}
            for row in cases:
                matrix = np.asarray(row["confusion"], dtype=np.int64)
                if matrix.shape != (6, 6) or (matrix < 0).any():
                    raise ValueError("Invalid 2D case confusion")
                matrices.setdefault(row["patient"], np.zeros((6, 6), dtype=np.int64))
                matrices[row["patient"]] += matrix
            if len(matrices) != 6:
                raise ValueError("Expected six Validation patients")
            pooled = sum(matrices.values(), np.zeros((6, 6), dtype=np.int64))
            if pooled.tolist() != val["pooled_confusion"]:
                raise ValueError("2D pooled confusion differs from cases")
            recomputed = overlap(pooled)
            if any(not math.isclose(recomputed[metric][index],
                                    val["pooled_overlap"][metric][index], abs_tol=1e-12)
                   for metric in recomputed for index in range(6)
                   if recomputed[metric][index] is not None):
                raise ValueError("2D overlap differs from confusion")
            score, patients = patient_small_organ_score(matrices)
            if (not math.isclose(score, val["selection_score"], abs_tol=1e-12)
                    or patients != val["patient_small_organ_dice"]):
                raise ValueError("2D selection score differs from case matrices")
            if score > best_score:
                best_score, best_epoch = score, epoch
            validation.append((epoch, val))
        if (record["best_epoch"] != best_epoch
                or not math.isclose(record["best_score"], best_score, abs_tol=1e-12)):
            raise ValueError("2D best-score history differs")
    if best_epoch is None:
        raise ValueError("No selected 2D Validation epoch")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
    epochs = [epoch for epoch, _ in validation]
    scores = [row["selection_score"] for _, row in validation]
    axes[0].plot(range(1, max_epochs + 1), [row["train"]["mean_loss"] for row in records],
                 color="#456996", linewidth=2)
    axes[1].plot(epochs, scores, marker="o", color="#c36837", linewidth=2)
    axes[1].scatter([best_epoch], [best_score], color="#7a2a2a", s=80, zorder=3)
    final = records[best_epoch - 1]["validation"]["pooled_overlap"]
    x = np.arange(len(ORGANS))
    for offset, (metric, color) in zip((-0.25, 0, 0.25),
                                       (("dice", "#3c78a8"), ("precision", "#d48b3a"),
                                        ("recall", "#5c9a63"))):
        axes[2].bar(x + offset, [final[metric][index] for _, index in ORGANS],
                    width=0.24, color=color, label=metric.title())
    axes[0].set(title="2D Train patch loss", xlabel="Epoch", ylabel="Mean objective")
    axes[1].set(title="2D Validation patient score", xlabel="Epoch", ylabel="Mean small-organ Dice")
    axes[2].set(title=f"2D Validation pooled metrics, epoch {best_epoch}", ylabel="Score")
    axes[2].set_xticks(x, [name for name, _ in ORGANS])
    axes[2].set_ylim(0, 1)
    axes[2].legend(loc="lower left", frameon=False)
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    output = PROJECT / "results/figures/baseline_2d/g5_2d_validation.svg"
    fig.savefig(output, metadata={"Title": "HipMRI G5 supplementary 2D Validation"})
    plt.close(fig)
    print("manifest_sha256", sha256(RUN / "manifest.json"))
    print("best_record_sha256", sha256(RUN / f"epoch_{best_epoch:03d}.json"))
    print("best_epoch", best_epoch, "selection_score", round(best_score, 6))
    for name, index in ORGANS:
        print(name, {metric: round(final[metric][index], 5)
                     for metric in ("dice", "precision", "recall")})
    print("Saved", output)


if __name__ == "__main__":
    main()
