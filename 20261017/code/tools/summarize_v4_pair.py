"""Summarize downloaded v4 records; no inference or Test selection."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results/experiments/improved_3d_v4"
OUT = BASE / "paired_analysis"
CLASSES = ["Background", "Body", "Bones", "Bladder", "Rectum", "Prostate"]


def write_csv(name, rows):
    with (OUT / name).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    OUT.mkdir(exist_ok=True)
    curves, patients, metrics, sources, summaries = [], [], [], {}, {}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for arm in ("a0", "a1"):
        directory = BASE / arm / "training_records"
        records = [json.loads((directory / f"epoch_{e:03}.json").read_text()) for e in range(1, 51)]
        assert [r["epoch"] for r in records] == list(range(1, 51))
        for p in directory.glob("*.json"):
            sources[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
        candidates = [r for r in records if r.get("validation")]
        best = max(candidates, key=lambda r: (r["validation"]["selection_score"], -r["epoch"]))
        assert best["epoch"] == records[-1]["best_epoch"]
        v = best["validation"]
        assert len(v["cases"]) == 38
        grouped = {}
        for case in v["cases"]:
            c = np.asarray(case["confusion"], dtype=float)
            tp = np.diag(c)
            true, pred = c.sum(1), c.sum(0)
            values = {"dice": 2*tp/(true+pred), "iou": tp/(true+pred-tp),
                      "precision": tp/pred, "recall": tp/true}
            grouped.setdefault(case["patient"], []).append(c)
        assert len(grouped) == 6
        for patient, visits in grouped.items():
            row = {"arm": arm, "patient": patient}
            c = np.sum(visits, axis=0)
            tp = np.diag(c)
            true, pred = c.sum(1), c.sum(0)
            pooled = {"dice": 2*tp/(true+pred), "iou": tp/(true+pred-tp), "precision": tp/pred, "recall": tp/true}
            for k, a in pooled.items():
                for i, label in enumerate(CLASSES):
                    row[f"{label}_{k}"] = float(a[i])
            row["small_organ_dice"] = float(np.mean([row[f"{c}_dice"] for c in CLASSES[3:]]))
            patients.append(row)
        score = np.mean([r["small_organ_dice"] for r in patients if r["arm"] == arm])
        assert abs(score-v["selection_score"]) < 1e-10
        for i, label in enumerate(CLASSES):
            row = {"arm": arm, "class": label}
            for k in values:
                row[k] = float(np.mean([r[f"{label}_{k}"] for r in patients if r["arm"] == arm]))
            row["volume_median_hd95_mm"] = (float(np.median([c["surfaces"][label]["hd95_mm"] for c in v["cases"]])) if i else "")
            metrics.append(row)
        summaries[arm] = {"best_epoch": best["epoch"], "selection_score": score,
                          "far_body_fp_voxels": sum(c["far_background_to_body_voxels"] for c in v["cases"]),
                          "body_hd95_volume_median_mm": next(r["volume_median_hd95_mm"] for r in metrics if r["arm"] == arm and r["class"] == "Body")}
        for r in records:
            curves.append({"arm": arm, "epoch": r["epoch"], "train_loss": r["train"]["mean_loss"],
                           "validation_score": r["validation"]["selection_score"] if r.get("validation") else ""})
        axes[0].plot(range(1, 51), [r["train"]["mean_loss"] for r in records], label=arm.upper())
        axes[1].plot([r["epoch"] for r in candidates], [r["validation"]["selection_score"] for r in candidates], "o-", label=arm.upper())
    for ax, title in zip(axes, ("Training objective (augmented input differs)", "Patient-equal small-organ Validation Dice")):
        ax.set(title=title, xlabel="Epoch")
        ax.legend(); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(OUT / "learning_curves.png", dpi=180); fig.savefig(OUT / "learning_curves.svg"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for j, arm in enumerate(("a0", "a1")):
        ax.bar(np.arange(6)+(j-.5)*.36, [r["dice"] for r in metrics if r["arm"] == arm], .36, label=arm.upper())
    ax.set(xticks=np.arange(6), xticklabels=CLASSES, ylabel="Patient-equal Dice", ylim=(0, 1), title="Best measured Validation checkpoints (both epoch 45)")
    ax.legend(); fig.tight_layout(); fig.savefig(OUT / "six_class_dice.png", dpi=180); fig.savefig(OUT / "six_class_dice.svg"); plt.close(fig)
    write_csv("learning_curves.csv", curves); write_csv("patient_metrics.csv", patients); write_csv("six_class_metrics.csv", metrics)
    (OUT / "record_checksums.json").write_text(json.dumps(sources, indent=2), encoding="utf-8")
    (OUT / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
