"""Verify the G4 patient-balanced CPU dry run and summarize patch exposure."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "code"))

SOURCE = PROJECT / "logs/audits/g4_patient_full_20261018"
OUTPUT = PROJECT / "results/data_analysis/patient_balanced_sampling"
G3 = PROJECT / "logs/audits/g3_existing_pipeline_full_20261018_job671892.json"
MANIFEST_SHA = "cafd3593efa5880a4a61b6281bac5d177f58cd713e734cf8dad422d191f0b90c"
PATCHES_SHA = "04a6822f15940b658f0c3eb062528aa4fa17721552030dc5dc6d45ce5c441968"
G1_SHA = "6a4c674564a0364f5f56730cbc31e67b3ff9a8ffeae7d9d3dd18b198bbd0e1a1"
G3_SHA = "319973a8759265521870f831e7b17eb7249f4130b6a9e9634fe993f22bf687e9"
ARMS = ("patient_uniform", "patient_organ_aware")
ORGANS = ("Bladder", "Rectum", "Prostate")
CLASSES = ("Background", "Body", "Bones", *ORGANS)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    from hipmri_rewrite.sampling import (ORGAN_AWARE_POLICY, UNIFORM_POLICY,
                                        epoch_cases)
    from hipmri_rewrite.splits import Case

    required = {SOURCE / "manifest.json": MANIFEST_SHA,
                SOURCE / "patches.csv": PATCHES_SHA, G3: G3_SHA}
    for path, expected in required.items():
        if sha256_file(path) != expected:
            raise ValueError(f"Source SHA-256 differs from Rangpur: {path}")
    manifest = json.loads((SOURCE / "manifest.json").read_text(encoding="utf-8"))
    policies = {"patient_uniform": UNIFORM_POLICY,
                "patient_organ_aware": ORGAN_AWARE_POLICY}
    if not (manifest["scope"] == "G4 Train-only patient-balanced CPU tensor dry run, no training"
            and manifest["complete_patient_epoch"] and manifest["epoch"] == 0
            and manifest["seed"] == 371017 and manifest["patients"] == 26
            and manifest["patches_per_policy"] == 260
            and manifest["patch_shape_xyz"] == [64, 64, 32]
            and manifest["g1_sha256"] == G1_SHA and manifest["g3_sha256"] == G3_SHA
            and manifest["policies"] == {key: list(value) for key, value in policies.items()}
            and manifest["script_sha256"] == sha256_file(PROJECT / "code/tools/audit_g4_patient_sampling.py")
            and manifest["sampling_code_sha256"] == sha256_file(PROJECT / "code/hipmri_rewrite/sampling.py")
            and manifest["data_code_sha256"] == sha256_file(PROJECT / "code/hipmri_rewrite/data.py")):
        raise ValueError("Manifest does not match the G4 source and frozen design")
    g3 = json.loads(G3.read_text(encoding="utf-8"))
    cases = {row["case"]: row for row in g3["cases"]}
    if len(cases) != 143 or any(row["errors"] for row in cases.values()):
        raise ValueError("G3 Train case evidence incomplete")
    dummy = tuple(Case(row["patient"], int(row["case"].split("Week")[1]),
                       "train", Path("unused"), Path("unused"))
                  for row in cases.values())
    selected = epoch_cases(dummy, 0, seed=371017)
    if manifest["selected_cases"] != [case.identifier for case in selected]:
        raise ValueError("Patient/visit selection differs from deterministic sampler")
    selected_by_patient = {case.patient: case.identifier for case in selected}

    with (SOURCE / "patches.csv").open(encoding="utf-8-sig", newline="") as stream:
        patches = list(csv.DictReader(stream))
    if len(patches) != 520:
        raise ValueError(f"Expected 520 patch rows, found {len(patches)}")
    observed = set()
    origins = {}
    aggregates = {arm: {"voxels": Counter(), "hits": Counter(), "patches": 0}
                  for arm in ARMS}
    per_patient = defaultdict(lambda: {"voxels": Counter(), "hits": Counter(), "patches": 0})
    for row in patches:
        arm, patient, case = row["policy"], row["patient"], row["case"]
        draw = int(row["draw"])
        if (arm not in ARMS or patient not in selected_by_patient
                or case != selected_by_patient[patient] or int(row["epoch"]) != 0
                or draw not in range(10) or row["target"] != policies[arm][draw]):
            raise ValueError("Unknown patient, case, arm or draw")
        key = patient, arm, draw
        if key in observed:
            raise ValueError(f"Duplicate patch row: {key}")
        observed.add(key)
        origin = tuple(int(row[f"origin_{axis}"]) for axis in "xyz")
        shape = cases[case]["downsampled_shape_xyz"]
        if any(start < 0 or start + size > n
               for start, size, n in zip(origin, (64, 64, 32), shape)):
            raise ValueError(f"Patch origin exceeds case grid: {key}")
        origins[key] = origin
        counts = {name: int(row[f"{name.lower()}_voxels"]) for name in CLASSES}
        if any(value < 0 for value in counts.values()) or sum(counts.values()) != 64 * 64 * 32:
            raise ValueError(f"Patch voxel count invalid: {key}")
        if row["target"] != "uniform" and counts[row["target"]] == 0:
            raise ValueError(f"Anchored patch missed target: {key}")
        aggregates[arm]["patches"] += 1
        per_patient[patient, arm]["patches"] += 1
        for name, count in counts.items():
            hit = int(count > 0)
            aggregates[arm]["voxels"][name] += count
            aggregates[arm]["hits"][name] += hit
            per_patient[patient, arm]["voxels"][name] += count
            per_patient[patient, arm]["hits"][name] += hit
    expected = {(patient, arm, draw) for patient in selected_by_patient
                for arm in ARMS for draw in range(10)}
    if observed != expected:
        raise ValueError("Patch coverage incomplete")
    for patient in selected_by_patient:
        for draw in range(4):
            if origins[patient, ARMS[0], draw] != origins[patient, ARMS[1], draw]:
                raise ValueError(f"Shared uniform origins differ: {patient}, draw {draw}")
    for arm in ARMS:
        calculated = aggregates[arm]
        saved = manifest["summary"][arm]
        if calculated["patches"] != 260 or saved["patches"] != 260:
            raise ValueError(f"Policy patch total mismatch: {arm}")
        for name in CLASSES:
            item = saved["per_class"][name]
            if (calculated["hits"][name] != item["patch_hits"]
                    or calculated["voxels"][name] != item["patch_voxels"]):
                raise ValueError(f"Class exposure mismatch: {arm} {name}")
        for patient in selected_by_patient:
            observed_patient = per_patient[patient, arm]
            saved_patient = saved["per_patient"][patient]
            if observed_patient["patches"] != 10 or saved_patient["patches"] != 10:
                raise ValueError(f"Patient patch count mismatch: {patient} {arm}")
            for name in CLASSES:
                if observed_patient["hits"][name] != saved_patient[f"{name}_hits"]:
                    raise ValueError(f"Patient hit mismatch: {patient} {arm} {name}")

    class_rows = []
    patient_rows = []
    for arm in ARMS:
        for name in CLASSES:
            counts = aggregates[arm]
            class_rows.append({"policy": arm, "class": name, "patches": 260,
                               "patch_hits": counts["hits"][name],
                               "patch_hit_rate": counts["hits"][name] / 260,
                               "patch_voxels": counts["voxels"][name],
                               "voxel_share": counts["voxels"][name] / (260 * 64 * 64 * 32)})
        for patient, case in sorted(selected_by_patient.items()):
            counts = per_patient[patient, arm]
            patient_rows.append({"policy": arm, "patient": patient, "case": case,
                                 "patches": counts["patches"],
                                 **{f"{name.lower()}_hits": counts["hits"][name] for name in ORGANS},
                                 **{f"{name.lower()}_voxel_share": counts["voxels"][name]
                                    / (10 * 64 * 64 * 32) for name in ORGANS}})

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.arange(len(ORGANS))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for index, arm in enumerate(ARMS):
        offset = (index - 0.5) * 0.35
        items = {row["class"]: row for row in class_rows if row["policy"] == arm}
        label = "Patient-uniform" if index == 0 else "Patient-organ-aware"
        axes[0].bar(x + offset, [items[name]["patch_hit_rate"] * 100 for name in ORGANS],
                    width=0.35, label=label)
        axes[1].bar(x + offset, [items[name]["voxel_share"] * 100 for name in ORGANS],
                    width=0.35, label=label)
    for axis, title, ylabel in zip(axes, ("Patch hit rate", "Sampled voxel share"),
                                   ("Patches containing organ (%)", "All patch voxels (%)")):
        axis.set_xticks(x, ORGANS)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
        axis.legend(fontsize=8)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT / "class_exposure.csv", class_rows)
    write_csv(OUTPUT / "patient_exposure.csv", patient_rows)
    figure_path = OUTPUT / "patient_balanced_exposure.svg"
    fig.savefig(figure_path)
    plt.close(fig)
    svg_lines = figure_path.read_text(encoding="utf-8").splitlines()
    figure_path.write_text("\n".join(line.rstrip() for line in svg_lines) + "\n",
                           encoding="utf-8")
    (OUTPUT / "README.md").write_text(
        "# G4 patient-balanced patch exposure\n\n"
        "The [class table](class_exposure.csv), [patient table](patient_exposure.csv), "
        "and [figure](patient_balanced_exposure.svg) describe one CPU dry-run epoch: "
        "one Train scan per patient, ten model-ready patches per policy and patient. "
        "They show whether patient balancing and crop placement change small-organ "
        "exposure before training. The uniform and organ-aware policies share their "
        "first four crops per patient.\n\n"
        "The source files are `logs/audits/g4_patient_full_20261018/manifest.json` "
        f"(SHA-256 `{MANIFEST_SHA}`) and `patches.csv` (SHA-256 `{PATCHES_SHA}`). "
        "The verification script checked each patch against the frozen G3 case grids "
        "and reconciled every class and patient total with the manifest. This is one "
        "deterministic epoch, without spatial augmentation or model optimization; "
        "Validation performance and actual training exposure remain unmeasured. "
        "Interpretation is in the [research report](../../reports/HipMRI_Research_Report.md).\n",
        encoding="utf-8")
    print(f"Verified {len(patches)} patches across {len(selected_by_patient)} patients")
    for name in ORGANS:
        print(name, {arm: {"hits": aggregates[arm]["hits"][name],
                           "voxel_share": aggregates[arm]["voxels"][name]
                           / (260 * 64 * 64 * 32)} for arm in ARMS})
    print(f"Saved G4 tables and figure to {OUTPUT}")


if __name__ == "__main__":
    main()
