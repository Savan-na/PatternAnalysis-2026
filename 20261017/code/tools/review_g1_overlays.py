"""Render raw Train MRI/GT overlays for G1 anatomical review, without transforms.

The output is a set of medical-image-derived PNGs. Review data-use conditions
before exporting them from Rangpur. This tool never writes to the dataset.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
CODE = PROJECT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

DEFAULT_CASES = (
    "B038_Week0",  # historical small Prostate label
    "K019_Week1",  # unusual source shape
    "N010_Week3",  # tiny second Bladder component
    "D031_Week7",  # tiny second Bladder component
    "S033_Week1",  # highest case MRI maximum in current audit
    "H017_Week7",  # highest case zero-voxel fraction in current audit
)
COLORS = {2: "#38bdf8", 3: "#2563eb", 4: "#f59e0b", 5: "#ef4444"}
NAMES = {2: "Bones", 3: "Bladder", 4: "Rectum", 5: "Prostate"}


def render_case(case, output_dir: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import nibabel as nib

    mri = nib.load(str(case.mri))
    seg = nib.load(str(case.mask))
    if mri.ndim != 3 or seg.ndim != 3 or mri.shape != seg.shape:
        raise ValueError(f"{case.identifier}: MRI/Mask shapes differ")
    if not np.allclose(mri.affine, seg.affine, rtol=0, atol=1e-4):
        raise ValueError(f"{case.identifier}: MRI/Mask affines differ")
    image = mri.get_fdata(dtype=np.float32, caching="unchanged")
    mask = np.asanyarray(seg.dataobj)
    if not np.isfinite(image).all() or not np.isfinite(mask).all():
        raise ValueError(f"{case.identifier}: non-finite data")
    prostate = np.argwhere(mask == 5)
    bladder = np.argwhere(mask == 3)
    if not len(prostate) or not len(bladder):
        raise ValueError(f"{case.identifier}: required organ is absent")
    x, y, z = (int(np.median(prostate[:, i])) for i in range(3))
    bladder_z = int(np.median(bladder[:, 2]))
    spacing = tuple(float(v) for v in mri.header.get_zooms()[:3])
    nonzero = image[image != 0]
    low, high = np.percentile(nonzero, (1, 99)) if len(nonzero) else (0, 1)
    if high <= low:
        high = low + 1

    panels = (
        (f"Axial · Prostate z={z}", image[:, :, z], mask[:, :, z], (0, 1)),
        (f"Axial · Bladder z={bladder_z}", image[:, :, bladder_z], mask[:, :, bladder_z], (0, 1)),
        (f"Coronal · Prostate y={y}", image[:, y, :], mask[:, y, :], (0, 2)),
        (f"Sagittal · Prostate x={x}", image[x, :, :], mask[x, :, :], (1, 2)),
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, (title, plane_image, plane_mask, dimensions) in zip(axes.flat, panels):
        extent = (0, plane_image.shape[0] * spacing[dimensions[0]],
                  0, plane_image.shape[1] * spacing[dimensions[1]])
        ax.imshow(plane_image.T, cmap="gray", vmin=low, vmax=high,
                  origin="lower", extent=extent, interpolation="nearest")
        for label, color in COLORS.items():
            occupied = (plane_mask == label).T
            if occupied.any() and not occupied.all():
                ax.contour(occupied.astype(float), levels=[0.5], colors=[color],
                           linewidths=0.9, origin="lower", extent=extent)
        ax.set_title(title)
        ax.set_xlabel(f"voxel axis {dimensions[0]} (mm)")
        ax.set_ylabel(f"voxel axis {dimensions[1]} (mm)")
        ax.set_aspect("equal")
    handles = [Line2D([0], [0], color=color, lw=2, label=NAMES[label])
               for label, color in COLORS.items()]
    fig.suptitle(f"{case.identifier} · raw MRI and ground truth · no preprocessing")
    fig.legend(handles=handles, loc="lower center", ncol=4)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    output = output_dir / f"g1_overlay_{case.identifier}.png"
    if output.exists():
        raise FileExistsError(output)
    fig.savefig(output, dpi=140, metadata={"Description": "G1 raw Train MRI and GT overlay"})
    plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", default=DEFAULT_CASES)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if not output_dir.is_dir() or not output_dir.is_relative_to(PROJECT):
        parser.error("Output directory must already exist inside the 20261017 project")
    if len(set(args.cases)) != len(args.cases) or any(
        re.fullmatch(r"[A-Z][0-9]{3}_Week[0-9]+", name) is None for name in args.cases
    ):
        parser.error("Use distinct patient-week case identifiers")
    from hipmri_rewrite.splits import discover
    train = {case.identifier: case for case in discover(args.dataset_root)["train"]}
    for name in args.cases:
        if name not in train:
            parser.error(f"Case is not in the frozen Train split: {name}")
        output = render_case(train[name], output_dir)
        print(output, flush=True)


if __name__ == "__main__":
    main()
