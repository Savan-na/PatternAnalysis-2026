"""Locate and plot the two Train bladder components flagged by the G1 audit.

Read-only source MRI/Mask inputs. Writes a small JSON and targeted PNGs inside
the existing project; never changes the HipMRI dataset or split.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
CODE = PROJECT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

CASES = ("N010_Week3", "D031_Week7")


def inspect_case(case, output_dir: Path) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import ndimage
    import nibabel as nib

    mri = nib.load(str(case.mri))
    seg = nib.load(str(case.mask))
    if mri.shape != seg.shape or mri.ndim != 3 or not np.allclose(
        mri.affine, seg.affine, rtol=0, atol=1e-4
    ):
        raise ValueError(f"{case.identifier}: MRI/Mask grid mismatch")
    image = mri.get_fdata(dtype=np.float32, caching="unchanged")
    mask = np.asanyarray(seg.dataobj)
    if not np.isfinite(image).all() or not np.isfinite(mask).all():
        raise ValueError(f"{case.identifier}: non-finite MRI/Mask values")
    bladder = mask == 3
    components, count = ndimage.label(bladder)  # Same 6-neighbour rule as G1.
    if count != 2:
        raise ValueError(f"{case.identifier}: expected 2 Bladder components, found {count}")
    sizes = np.bincount(components.ravel(), minlength=count + 1)
    primary = int(np.argmax(sizes[1:]) + 1)
    secondary = 3 - primary
    points = np.argwhere(components == secondary)
    center = np.rint(points.mean(axis=0)).astype(int)
    low_xyz, high_xyz = points.min(axis=0), points.max(axis=0)
    spacing = np.asarray(mri.header.get_zooms()[:3], dtype=float)
    voxel_mm3 = float(abs(np.linalg.det(seg.affine[:3, :3])))
    world_center = nib.affines.apply_affine(seg.affine, points.mean(axis=0))
    diagonal_contact = bool(np.any(
        ndimage.binary_dilation(components == secondary, structure=np.ones((3, 3, 3)))
        & (components == primary)
    ))
    finite_nonzero = image[image != 0]
    display_low, display_high = np.percentile(finite_nonzero, (1, 99))
    if display_high <= display_low:
        display_high = display_low + 1

    views = (
        ("Axial context", image[:, :, center[2]], components[:, :, center[2]],
         (0, 1), (center[0], center[1]), None),
        ("Axial island detail", image[:, :, center[2]], components[:, :, center[2]],
         (0, 1), (center[0], center[1]), 35),
        ("Coronal through island", image[:, center[1], :], components[:, center[1], :],
         (0, 2), (center[0], center[2]), 60),
        ("Sagittal through island", image[center[0], :, :], components[center[0], :, :],
         (1, 2), (center[1], center[2]), 60),
    )
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    for ax, (title, plane_image, plane_components, dims, location, radius) in zip(axes.flat, views):
        extent = (0, plane_image.shape[0] * spacing[dims[0]],
                  0, plane_image.shape[1] * spacing[dims[1]])
        ax.imshow(plane_image.T, cmap="gray", vmin=display_low, vmax=display_high,
                  origin="lower", extent=extent, interpolation="nearest")
        for component, color in ((primary, "#2563eb"), (secondary, "#ef4444")):
            binary = (plane_components == component).T
            if binary.any() and not binary.all():
                ax.contour(binary.astype(float), levels=[0.5], colors=[color],
                           linewidths=1.0 if component == primary else 1.6,
                           origin="lower", extent=extent)
        horizontal = (location[0] + 0.5) * spacing[dims[0]]
        vertical = (location[1] + 0.5) * spacing[dims[1]]
        ax.plot(horizontal, vertical, marker="+", color="#ffea00", markersize=11,
                markeredgewidth=1.5)
        if radius is not None:
            ax.set_xlim(max(0, horizontal - radius), min(extent[1], horizontal + radius))
            ax.set_ylim(max(0, vertical - radius), min(extent[3], vertical + radius))
        ax.set_title(title)
        ax.set_xlabel(f"voxel axis {dims[0]} (mm)")
        ax.set_ylabel(f"voxel axis {dims[1]} (mm)")
        ax.set_aspect("equal")
    fig.suptitle(f"{case.identifier} · Bladder island · red=small, blue=main, yellow=center")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    png = output_dir / f"g1_bladder_island_{case.identifier}.png"
    fig.savefig(png, dpi=150, metadata={"Description": "G1 raw Train Bladder component review"})
    plt.close(fig)

    return {
        "case": case.identifier,
        "components_6_neighbour": int(count),
        "primary_voxels": int(sizes[primary]),
        "secondary_voxels": int(sizes[secondary]),
        "secondary_volume_mm3": float(sizes[secondary] * voxel_mm3),
        "secondary_bbox_min_xyz": low_xyz.tolist(),
        "secondary_bbox_max_xyz": high_xyz.tolist(),
        "secondary_center_voxel_xyz": center.tolist(),
        "secondary_center_world_mm": [float(value) for value in world_center],
        "secondary_diagonal_contact_with_primary": diagonal_contact,
        "secondary_mri_intensity_min_max": [float(image[tuple(points.T)].min()),
                                            float(image[tuple(points.T)].max())],
        "mask_spatial_unit": seg.header.get_xyzt_units()[0],
        "figure": str(png.relative_to(PROJECT)),
    }


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
    planned = [output_dir / f"g1_bladder_island_{name}.png" for name in CASES]
    if any(path.exists() for path in (*planned, json_output)):
        parser.error("Refusing to overwrite an existing review artifact")
    from hipmri_rewrite.splits import discover
    train = {case.identifier: case for case in discover(args.dataset_root)["train"]}
    rows = [inspect_case(train[name], output_dir) for name in CASES]
    json_output.write_text(json.dumps({"scope": "two flagged raw Train Bladder masks",
                                       "connectivity": "6-neighbour", "cases": rows}, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps(rows, indent=2), flush=True)
    print(f"Saved {json_output}", flush=True)


if __name__ == "__main__":
    main()
