"""Render one two-slice contact sheet per Train patient for G1 visual screening.

Every Train MRI/Mask pair contributes an axial Prostate and Bladder section.
This is a gross-alignment screen, not an expert review of every boundary.
Source NIfTI files are read only; outputs stay inside the project.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
CODE = PROJECT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

COLORS = {2: "#38bdf8", 3: "#2563eb", 4: "#f59e0b", 5: "#ef4444"}


def render_patient(patient: str, cases: list, output: Path) -> list[dict]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import nibabel as nib

    fig, axes = plt.subplots(len(cases), 2, figsize=(9, 2.6 * len(cases)),
                             squeeze=False)
    rows = []
    try:
        for row_index, case in enumerate(cases):
            mri, seg = nib.load(str(case.mri)), nib.load(str(case.mask))
            if mri.ndim != 3 or mri.shape != seg.shape or not np.allclose(
                mri.affine, seg.affine, rtol=0, atol=1e-4
            ):
                raise ValueError(f"{case.identifier}: MRI/Mask grid mismatch")
            image = mri.get_fdata(dtype=np.float32, caching="unchanged")
            mask = np.asanyarray(seg.dataobj)
            if not np.isfinite(image).all() or not np.isfinite(mask).all():
                raise ValueError(f"{case.identifier}: non-finite MRI/Mask values")
            slice_indices = {}
            for name, label in (("Prostate", 5), ("Bladder", 3)):
                occupied_z = np.where(np.any(mask == label, axis=(0, 1)))[0]
                if not occupied_z.size:
                    raise ValueError(f"{case.identifier}: {name} absent")
                slice_indices[name] = int(occupied_z[len(occupied_z) // 2])
            sample = image[::4, ::4, ::2]
            sample = sample[sample != 0]
            low, high = np.percentile(sample, (1, 99)) if sample.size else (0, 1)
            if high <= low:
                high = low + 1
            spacing = mri.header.get_zooms()[:2]
            extent = (0, image.shape[0] * spacing[0], 0, image.shape[1] * spacing[1])
            for column, (name, z) in enumerate(slice_indices.items()):
                ax = axes[row_index, column]
                ax.imshow(image[:, :, z].T, cmap="gray", vmin=low, vmax=high,
                          origin="lower", extent=extent, interpolation="nearest")
                plane = mask[:, :, z]
                for label, color in COLORS.items():
                    binary = (plane == label).T
                    if binary.any() and not binary.all():
                        ax.contour(binary.astype(float), levels=[0.5],
                                   colors=[color], linewidths=0.7,
                                   origin="lower", extent=extent)
                ax.set_title(f"{case.identifier} · {name} z={z}", fontsize=9)
                ax.set_aspect("equal")
                ax.set_xticks([])
                ax.set_yticks([])
            rows.append({"case": case.identifier, "prostate_z": slice_indices["Prostate"],
                         "bladder_z": slice_indices["Bladder"]})
            del image, mask, mri, seg
        fig.suptitle(f"G1 raw Train alignment screen · {patient} · no preprocessing")
        fig.text(0.5, 0.01, "Bones cyan · Bladder blue · Rectum orange · Prostate red",
                 ha="center", fontsize=9)
        fig.tight_layout(rect=(0, 0.025, 1, 0.97))
        fig.savefig(output, dpi=130,
                    metadata={"Description": "G1 raw Train patient contact sheet"})
    finally:
        plt.close(fig)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    json_output = args.json_output.resolve()
    if not output_dir.is_dir() or not output_dir.is_relative_to(PROJECT):
        parser.error("Output directory must already exist inside the project")
    if not json_output.parent.is_dir() or not json_output.is_relative_to(PROJECT):
        parser.error("JSON parent directory must already exist inside the project")
    from hipmri_rewrite.splits import discover
    train = discover(args.dataset_root)["train"]
    by_patient = defaultdict(list)
    for case in train:
        by_patient[case.identifier.split("_", 1)[0]].append(case)
    if len(train) != 143 or len(by_patient) != 26:
        parser.error(f"Expected 143 Train volumes in 26 patients; got {len(train)}, {len(by_patient)}")
    if json_output.exists() or any(
        (output_dir / f"g1_train_patient_{patient}.png").exists()
        for patient in by_patient
    ):
        parser.error("Refusing to overwrite an existing review artifact")
    rows = []
    for index, patient in enumerate(sorted(by_patient), start=1):
        output = output_dir / f"g1_train_patient_{patient}.png"
        cases = sorted(by_patient[patient], key=lambda item: int(item.identifier.split("Week")[1]))
        case_rows = render_patient(patient, cases, output)
        rows.append({"patient": patient, "figure": str(output.relative_to(PROJECT)),
                     "cases": case_rows})
        print(f"[{index}/{len(by_patient)}] {patient}: {len(case_rows)} Train volumes -> {output.name}",
              flush=True)
    json_output.write_text(json.dumps({"scope": "raw Train pairs only",
                                       "screen": "one Prostate and one Bladder axial section per volume",
                                       "patients": rows}, indent=2) + "\n",
                           encoding="utf-8")
    print(f"Saved {len(rows)} patient sheets covering {len(train)} Train volumes to {output_dir}")
    print(f"Saved {json_output}")


if __name__ == "__main__":
    main()
