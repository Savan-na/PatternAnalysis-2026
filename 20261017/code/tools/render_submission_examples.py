"""Replay locked predictions for documentation figures; no training or selection."""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from dataset import discover, load_case, downsample, zscore_nonzero
from utils import load_model, predict_volume, confusion, overlap, sha256_file


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def render_panel(volume, predictions, target, views, output):
    """Separate MRI/GT/prediction views and GT-bounded local error views.

    All arrays are actual whole-volume predictions. Crops are for display only;
    the full-field row retains distant predictions outside the local GT crop.
    """
    nz = volume.image[volume.image != 0]
    low, high = np.percentile(nz, [1, 99])
    fig, axes = plt.subplots(2 * len(views), 4, figsize=(11, 6 * len(views)),
                             squeeze=False, layout="constrained")
    for view_index, view in enumerate(views):
        axis, index = view["axis"], view["index"]
        image = np.take(volume.image, index, axis=axis).T
        gt = np.take(volume.mask == target, index, axis=axis).T
        found = {name: np.take(prediction == target, index, axis=axis).T
                 for name, prediction in predictions.items()}
        other = [a for a in range(3) if a != axis]
        aspect = volume.spacing_mm[other[1]] / volume.spacing_mm[other[0]]
        points = np.argwhere(gt)
        if points.size:
            y0, x0 = np.maximum(points.min(0) - 10, 0)
            y1, x1 = np.minimum(points.max(0) + 11, gt.shape)
        else:
            y0, x0, y1, x1 = 0, 0, *gt.shape
        roi = np.s_[y0:y1, x0:x1]
        upper, lower = axes[2 * view_index], axes[2 * view_index + 1]
        upper[0].imshow(image, cmap="gray", origin="lower", vmin=low, vmax=high, aspect=aspect)
        upper[1].imshow(gt, cmap=ListedColormap(["black", "cyan"]), origin="lower",
                        vmin=0, vmax=1, aspect=aspect)
        for col, name in enumerate(("Standard", "v3"), 2):
            upper[col].imshow(found[name], cmap=ListedColormap(["black", "orange"]),
                              origin="lower", vmin=0, vmax=1, aspect=aspect)
        for ax, title in zip(upper, ("MRI: full field", "GT mask", "Standard prediction", "v3 prediction")):
            ax.set_title(title + f"\naxis {axis}, index {index}", fontsize=12)
        lower[0].imshow(image[roi], cmap="gray", origin="lower", vmin=low, vmax=high, aspect=aspect)
        lower[1].imshow(gt[roi], cmap=ListedColormap(["black", "cyan"]), origin="lower",
                        vmin=0, vmax=1, aspect=aspect)
        lower[0].set_title("MRI: display ROI", fontsize=12)
        lower[1].set_title("GT: same display ROI", fontsize=12)
        for col, name in enumerate(("Standard", "v3"), 2):
            rgba = np.zeros((*gt.shape, 4))
            rgba[found[name] & ~gt] = [1, 0, 0, .65]
            rgba[gt & ~found[name]] = [0, .2, 1, .65]
            lower[col].imshow(image[roi], cmap="gray", origin="lower", vmin=low, vmax=high, aspect=aspect)
            lower[col].imshow(rgba[roi], origin="lower", aspect=aspect)
            lower[col].set_title(name + ": red FP / blue FN", fontsize=12)
    for ax in axes.flat:
        ax.set_axis_off()
    fig.suptitle(volume.case.identifier + " | " + view["organ"] +
                 "\nFull-field masks above; local error ROI below (GT box + 10 voxels)."
                 " Display cropping does not change inference or metrics.", fontsize=13)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--standard-checkpoint", type=Path, required=True)
    parser.add_argument("--v3-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("Use an allocated GPU job")
    if args.output.exists():
        raise FileExistsError("Use a new output directory")
    plan = json.loads((ROOT / "reference_cases.json").read_text())
    checks = json.loads((ROOT / "transfer_checksums.json").read_text())
    for name, digest in checks.items():
        assert sha256_file(ROOT / name) == digest, name
    checkpoints = {"Standard": args.standard_checkpoint, "v3": args.v3_checkpoint}
    for name, path in checkpoints.items():
        assert sha256_file(path) == plan["models"][name]["checkpoint_sha256"], name
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    args.output.mkdir(parents=True)
    environment = {"python": platform.python_version(), "torch": str(torch.__version__),
                   "cuda_build": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
                   "packages": {name: importlib.metadata.version(name) for name in
                                ("numpy", "scipy", "nibabel", "matplotlib", "Pillow")}}
    write(args.output / "environment.json", environment)
    # Execute the submitted predict.py CLI itself on one fixed Validation case.
    command = [sys.executable, "-B", str(ROOT / "predict.py"), "--model", "v3",
               "--checkpoint", str(args.v3_checkpoint), "--dataset-root", str(args.dataset_root),
               "--split", "validation", "--case", "K008_Week0", "--device", "cuda",
               "--output-dir", str(args.output / "cli_example_K008_v3")]
    run = subprocess.run(command, check=True, capture_output=True, text=True)
    (args.output / "cli_stdout.txt").write_text(run.stdout + run.stderr, encoding="utf-8")
    cli = json.loads((args.output / "cli_example_K008_v3/metrics.json").read_text())
    assert cli["confusion"] == plan["models"]["v3"]["cases"]["K008_Week0"]["confusion"]
    groups = discover(args.dataset_root)
    case_map = {case.identifier: case for rows in groups.values() for case in rows}
    models = {name: load_model("standard" if name == "Standard" else "v3", path, "cuda")
              for name, path in checkpoints.items()}
    receipt = {"kind": "fixed_prediction_documentation_replay", "job_id": os.environ["SLURM_JOB_ID"],
               "checkpoint_sha256": {n: sha256_file(p) for n, p in checkpoints.items()},
               "sources": checks, "environment": environment, "cli_command": command,
               "cli_returncode": run.returncode, "cli_reference_matrix_reproduced": True,
               "cases": {}, "limitations": "Fixed cases, slices and checkpoints; no training, tuning or new Test selection"}
    for case_id, spec in plan["panels"].items():
        case = case_map[case_id]
        volume = zscore_nonzero(downsample(load_case(case)))
        predicted = {}
        receipt["cases"][case_id] = {}
        for name, model in models.items():
            expected = plan["models"][name]["cases"][case_id]
            assert sha256_file(case.mri) == expected["mri_sha256"]
            assert sha256_file(case.mask) == expected["mask_sha256"]
            predicted[name] = predict_volume(model, volume, mode="3d", device="cuda")
            matrix = confusion(volume.mask, predicted[name]).tolist()
            assert matrix == expected["confusion"], (name, case_id, "matrix changed")
            receipt["cases"][case_id][name] = {"confusion": matrix, "reproduced": True,
                                               "overlap": overlap(np.asarray(matrix))}
        render_panel(volume, predicted, spec["label"], spec["views"], args.output / spec["filename"])
        for extra in spec.get("additional_panels", []):
            render_panel(volume, predicted, spec["label"], extra["views"], args.output / extra["filename"])
        print(case_id + ": both fixed matrices reproduced; separate input/GT/prediction/error views saved", flush=True)
    write(args.output / "replay_receipt.json", receipt)
    write(args.output / "checksums.json", {p.relative_to(args.output).as_posix(): sha256_file(p)
                                           for p in args.output.rglob("*") if p.is_file()})
    publication = [p for p in args.output.rglob("*") if p.is_file() and p.suffix in (".png", ".json", ".txt")]
    archive = args.output / "documentation_outputs.zip"
    with ZipFile(archive, "w", ZIP_DEFLATED) as bundle:
        for path in publication:
            bundle.write(path, path.relative_to(args.output).as_posix())
    print("PASS: actual predict.py CLI and ten fixed model/case matrices; " + str(args.output), flush=True)
    print("Download archive SHA-256: " + sha256_file(archive), flush=True)


if __name__ == "__main__":
    main()
