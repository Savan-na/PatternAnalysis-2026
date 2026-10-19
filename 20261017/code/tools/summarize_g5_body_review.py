"""Verify the fixed Validation Body audit and draw a patient-level comparison."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/tables/comparisons/g5_validation_body_audit.json"
OUTPUT = ROOT / "results/figures/comparisons/g5_validation_body_boundaries"
MODELS = ("standard_3d", "improved_3d")
COLORS = ("#246391", "#c36b29")


def main() -> None:
    audit = json.loads(SOURCE.read_text(encoding="utf-8"))
    rows = audit["cases"]
    patients = sorted({row["patient"] for row in rows})
    if (audit["kind"] != "G5_selected_3D_Validation_Body_boundary_audit"
            or len(rows) != 38 or len(patients) != 6
            or len({row["case"] for row in rows}) != 38):
        raise ValueError("Expected complete six-patient Validation Body audit")
    for model, folder in (("standard_3d", "baseline_3d"),
                          ("improved_3d", "improved_3d")):
        import hashlib

        record = ROOT / "logs/training" / folder / "epoch_050.json"
        if hashlib.sha256(record.read_bytes()).hexdigest() != audit["sources"][model]["selected_record_sha256"]:
            raise ValueError(f"{model}: local selected record differs from GPU audit")
    stats = {model: {patient: [row["models"][model]
                              for row in rows if row["patient"] == patient]
                     for patient in patients}
             for model in MODELS}
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.6), layout="constrained")
    x = np.arange(len(patients))
    for model, color, offset, label in zip(MODELS, COLORS, (-0.19, 0.19),
                                            ("Standard 3D", "Improved 3D")):
        hd95 = [median(row["boundary"]["hd95_mm"] for row in stats[model][patient])
                for patient in patients]
        far = [sum(row["background_to_body_at_least_20mm"]
                   for row in stats[model][patient]) for patient in patients]
        axes[0].bar(x + offset, hd95, width=0.36, color=color, label=label)
        axes[1].bar(x + offset, np.asarray(far) / 1000, width=0.36,
                    color=color, label=label)
    axes[0].set(title="Validation · patient median Body HD95",
                ylabel="HD95 (mm)")
    axes[1].set(title="Validation · distant Background→Body errors",
                ylabel="False-positive voxels, thousands (≥20 mm from GT Body)")
    for ax in axes:
        ax.set_xticks(x, patients)
        ax.grid(axis="y", alpha=0.18)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False)
    fig.suptitle("Selected epoch-50 checkpoints · all 38 Validation visits")
    fig.savefig(OUTPUT.with_suffix(".svg"), metadata={"Title": OUTPUT.stem})
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=180)
    plt.close(fig)
    print(OUTPUT.with_suffix(".png"))


if __name__ == "__main__":
    main()
