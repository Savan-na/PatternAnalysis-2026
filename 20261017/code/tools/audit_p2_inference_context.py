"""Four-case, fixed-weight Validation inference-context diagnostic; no training.

Reproduce original G5 confusion first. Compare with one predefined sliding
protocol and describe same-voxel Body-vs-Background head margins. Test arrays
and Test results are never loaded. Images are derived medical data.
"""
from __future__ import annotations
import argparse
import json
import os
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import ndimage as ndi

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "code"))
from hipmri_rewrite.boundaries import symmetric_surface_distances
from hipmri_rewrite.data import downsample, load_case, to_3d_model_arrays, zscore_nonzero
from hipmri_rewrite.inference_context import infer_logits
from hipmri_rewrite.metrics import confusion, overlap
from hipmri_rewrite.splits import DEFAULT_SPLIT, discover
# Reuse the strict checkpoint/source validator only; do not invoke Test evaluation.
from tools.evaluate_g6_test import ARMS, LABELS, selected_model, sha256

CONFIG = PROJECT / "code/config/p2_inference_context.json"
CASE_IDS = ("K018_Week4", "B040_Week4", "J026_Week4", "K008_Week0")


def stats(values):
    values = np.asarray(values)
    if not values.size:
        return {"n": 0, "min": None, "p05": None, "median": None, "p95": None, "max": None}
    q = np.percentile(values, (0, 5, 50, 95, 100))
    return dict(zip(("n", "min", "p05", "median", "p95", "max"),
                    (int(values.size), *map(float, q))))


def xyz_logits(outputs, shape):
    x, y, z = shape
    return {name: value[:, :z, :y, :x].transpose(0, 3, 2, 1)
            for name, value in outputs.items()}


def border_distance(shape, spacing):
    distance = np.full(shape, np.inf, dtype=np.float32)
    for axis, (n, scale) in enumerate(zip(shape, spacing)):
        v = np.minimum(np.arange(n), np.arange(n)[::-1]) * scale
        dims = [1, 1, 1]
        dims[axis] = n
        np.minimum(distance, v.reshape(dims), out=distance)
    return distance


def components(prediction, truth, spacing):
    ids, count = ndi.label(prediction == 1)  # explicit six-neighbour connectivity
    sizes = np.bincount(ids.ravel())[1:]
    touching = np.unique(ids[truth == 1])
    separate = ~np.isin(np.arange(1, count+1), touching)
    return {"connectivity": 6, "count": int(count),
            "largest_voxels": int(sizes.max()) if count else 0,
            "components_without_gt_body_overlap": int(separate.sum()),
            "voxels_in_components_without_gt_body_overlap": int(sizes[separate].sum()),
            "largest_sizes_voxels": sorted(map(int, sizes), reverse=True)[:10],
            "voxel_volume_mm3": float(np.prod(spacing))}


def measure(truth, prediction, distance, border, spacing):
    matrix = confusion(truth, prediction)
    bg_error = (truth == 0) & (prediction == 1)
    all_error = (truth != 1) & (prediction == 1)
    return {"confusion": matrix.tolist(), "overlap": overlap(matrix),
            "body_surface": symmetric_surface_distances(truth, prediction, spacing, label=1),
            "background_to_body_voxels": int(bg_error.sum()),
            "background_to_body_at_least_20mm": int((bg_error & (distance >= 20)).sum()),
            "all_body_false_positive_voxels": int(all_error.sum()),
            "false_body_distance_to_gt_body_mm": stats(distance[all_error]),
            "false_body_distance_to_volume_edge_mm": stats(border[all_error]),
            "background_false_body_distance_to_gt_mm": stats(distance[bg_error]),
            "body_components": components(prediction, truth, spacing)}


def margin_comparison(truth, full, sliding, full_pred, slide_pred, distance, reference_regions=None):
    fp = (truth == 0) & (full_pred == 1)
    groups = {"full_background_to_body": fp,
              "full_distant_background_to_body": fp & (distance >= 20),
              "full_false_body_removed_by_sliding": fp & (slide_pred != 1),
              "new_sliding_background_to_body": (truth == 0) & (full_pred != 1) & (slide_pred == 1),
              "gt_background_correct_in_both": (truth == 0) & (full_pred == 0) & (slide_pred == 0),
              "gt_body": truth == 1}
    if reference_regions is not None:
        groups.update(reference_regions)
    result = {}
    for group, region in groups.items():
        entry = {"voxels": int(region.sum()), "protocols": {}}
        for name, outputs in (("full", full), ("sliding", sliding)):
            margins = {head: stats((logit[1]-logit[0])[region]) for head, logit in outputs.items()}
            combined = outputs["combined"]
            competition = combined[1] - np.max(combined[[0, 2, 3, 4, 5]], axis=0)
            entry["protocols"][name] = {"body_minus_background": margins,
                                            "body_minus_best_other_class": stats(competition[region])}
            if "main" in outputs:
                summed = sum(outputs[h][1]-outputs[h][0] for h in ("main", "half", "quarter"))
                residual = summed - (combined[1]-combined[0])
                entry["protocols"][name]["max_head_margin_sum_residual"] = (
                    float(np.abs(residual[region]).max()) if region.any() else None)
        result[group] = entry
    return result


def draw(volume, maps, center, arm, output):
    x, y, z = center
    slices = ((slice(None), slice(None), z), (slice(None), y, slice(None)),
              (x, slice(None), slice(None)))
    aspects = (volume.spacing_mm[1]/volume.spacing_mm[0],
               volume.spacing_mm[2]/volume.spacing_mm[0], volume.spacing_mm[2]/volume.spacing_mm[1])
    nz = volume.image[volume.image != 0]
    low, high = np.percentile(nz, (1, 99))
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), layout="constrained")
    for row, (name, logits) in enumerate(maps.items()):
        prediction = logits["combined"].argmax(0)
        for col, (region, aspect) in enumerate(zip(slices, aspects)):
            ax = axes[row, col]
            ax.imshow(volume.image[region].T, cmap="gray", origin="lower", vmin=low, vmax=high, aspect=aspect)
            for binary, color in (((volume.mask == 1)[region].T, "cyan"),
                                  ((prediction == 1)[region].T, "orange")):
                if binary.any() and not binary.all():
                    ax.contour(binary, levels=[0.5], colors=[color], linewidths=0.9)
            ax.set_title(f"{name}: {('axial', 'coronal', 'sagittal')[col]}")
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{volume.case.identifier} | {arm} | Body: GT cyan, prediction orange\n"
                 f"Same evaluation-selected XYZ voxel {tuple(center)}; no GT-based inference ROI")
    fig.savefig(output / f"{volume.case.identifier}_{arm}_body.png", dpi=160)
    plt.close(fig)
    if arm != "improved_3d":
        return
    # All head maps share one symmetric scale for this case.
    limit = max(float(np.percentile(np.abs((maps[p][h][1]-maps[p][h][0])[:, :, z]), 99))
                for p in maps for h in ("combined", "main", "half", "quarter"))
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), layout="constrained")
    for row, (protocol, outputs) in enumerate(maps.items()):
        for col, head in enumerate(("main", "half", "quarter", "combined")):
            ax = axes[row, col]
            margin = outputs[head][1] - outputs[head][0]
            im = ax.imshow(margin[:, :, z].T, origin="lower", cmap="coolwarm",
                           vmin=-max(limit, 1e-6), vmax=max(limit, 1e-6), aspect=aspects[0])
            ax.set_title(f"{protocol}: {head}")
            ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axes.ravel().tolist(), label="Body minus Background raw logit; not probability")
    fig.suptitle(f"{volume.case.identifier} | fixed Improved heads, same axial z={z}")
    fig.savefig(output / f"{volume.case.identifier}_head_margins.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    args = parser.parse_args()
    if (not os.environ.get("SLURM_JOB_ID") or socket.gethostname().startswith("login")
            or not torch.cuda.is_available()):
        parser.error("Use an allocated GPU compute node")
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    if (tuple(cfg["cases"]) != CASE_IDS or cfg["split"] != "validation"
            or cfg["window_xyz"] != [64, 64, 32] or cfg["overlap_fraction"] != 0.5
            or cfg["far_error_distance_mm"] != 20):
        parser.error("Predeclared four-case protocol changed; create a new experiment instead")
    raw = PROJECT / "logs/validation/inference_context" / cfg["experiment_id"]
    figures = PROJECT / "results/diagnostics/inference_context" / cfg["experiment_id"]
    if raw.exists() or figures.exists():
        parser.error("Output already exists; inspect it, do not overwrite")
    cases = {c.identifier: c for c in discover(args.dataset_root)["validation"]}
    if len(cases) != 38 or not set(CASE_IDS).issubset(cases):
        raise ValueError("Fixed Validation inventory changed")
    raw.mkdir(parents=True); figures.mkdir(parents=True)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    result = {"kind": "P2_fixed_weights_four_case_inference_context_diagnostic", "complete": False,
              "split": "validation", "config": cfg, "config_sha256": sha256(CONFIG),
              "split_sha256": sha256(DEFAULT_SPLIT), "classes": LABELS,
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "job_id": os.environ["SLURM_JOB_ID"], "hostname": socket.gethostname(),
              "environment": {"python": sys.version, "torch": torch.__version__,
                              "numpy": np.__version__, "scipy": scipy.__version__,
                              "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name()},
              "sources": {}, "models": {}, "cases": [],
              "limitations": ["Four preselected cases: exploratory, not a model selection cohort",
                              "Context, padding, receptive field and normalization change together",
                              "Head margins are co-adapted contributions, not independently trained heads",
                              "Body edge distance is to outermost voxel centres; not the physical voxel face",
                              "Latency is diagnostic timing, not a warmed fair resource benchmark"]}
    for rel in ("code/tools/audit_p2_inference_context.py", "code/hipmri_rewrite/inference_context.py",
                "code/tools/evaluate_g6_test.py", "code/hipmri_rewrite/boundaries.py"):
        result["sources"][rel] = sha256(PROJECT / rel)
    # Evaluate Improved first to define shared post-hoc error regions and slice
    # centres. These references never affect either network's inference.
    references = {}
    for identity in reversed(ARMS):
        arm, folder, _, _ = identity
        model, receipt = selected_model(PROJECT / "logs/training", identity, args.dataset_root, torch)
        if receipt["selected_epoch"] != 50:
            raise ValueError("P2 requires the reviewed epoch-50 checkpoint")
        model.requires_grad_(False)
        result["models"][arm] = receipt
        saved = json.loads((PROJECT / "logs/training" / folder / "epoch_050.json").read_text())
        expected = {r["case"]: r["confusion"] for r in saved["validation"]["cases"]}
        for case_id in CASE_IDS:
            volume = zscore_nonzero(downsample(load_case(cases[case_id])))
            if not (volume.mask == 1).any():
                raise ValueError("Body missing; distance diagnostic undefined")
            image, _, shape = to_3d_model_arrays(volume)
            outputs, receipts = {}, {}
            for name, window in (("full", None), ("sliding", (32, 64, 64))):
                torch.cuda.synchronize(); started = perf_counter()
                logits, detail = infer_logits(model, image, device="cuda", window_dhw=window,
                                               components=(arm == "improved_3d"))
                torch.cuda.synchronize()
                detail["seconds_inference_and_cpu_logit_aggregation"] = perf_counter()-started
                outputs[name] = xyz_logits(logits, shape)
                receipts[name] = detail
                if name == "full":
                    matrix = confusion(volume.mask, outputs[name]["combined"].argmax(0))
                    if matrix.tolist() != expected[case_id]:
                        raise ValueError(f"{arm}/{case_id}: original G5 confusion not reproduced")
                del logits
            distance = ndi.distance_transform_edt(volume.mask != 1, sampling=volume.spacing_mm)
            border = border_distance(shape, volume.spacing_mm)
            pred = {p: h["combined"].argmax(0).astype(np.uint8) for p, h in outputs.items()}
            row = {"case": case_id, "patient": cases[case_id].patient, "model": arm,
                   "shape_xyz": shape, "spacing_mm": volume.spacing_mm, "affine": volume.affine.tolist(),
                   "input_sha256": {"mri": sha256(cases[case_id].mri), "mask": sha256(cases[case_id].mask)},
                   "full_confusion_reproduced": True, "protocols": {},
                   "changed_prediction_voxels": int((pred["full"] != pred["sliding"]).sum())}
            for name in outputs:
                row["protocols"][name] = {**measure(volume.mask, pred[name], distance, border, volume.spacing_mm),
                                           "inference": receipts[name]}
            if arm == "improved_3d":
                fp = (pred["full"] == 1) & (volume.mask == 0)
                center = (np.unravel_index(np.argmax(np.where(fp, distance, -1)), shape) if fp.any()
                          else tuple(np.argwhere(volume.mask == 1).mean(0).astype(int)))
                references[case_id] = (center, {
                    "reference_improved_full_background_to_body": fp,
                    "reference_improved_full_distant_background_to_body": fp & (distance >= 20)})
            center, shared_regions = references[case_id]
            row["same_voxel_margins"] = margin_comparison(volume.mask, outputs["full"], outputs["sliding"],
                                                           pred["full"], pred["sliding"], distance, shared_regions)
            row["shared_reference"] = "Improved full-volume errors; same voxel regions and picture centres for both models"
            row["illustration_center_xyz"] = list(map(int, center))
            draw(volume, outputs, center, arm, figures)
            (raw / f"{case_id}_{arm}.json").write_text(json.dumps(row, indent=2, allow_nan=False)+"\n")
            result["cases"].append(row)
            print(f"{arm}/{case_id}: G5 reproduced; full/sliding far false Body "
                  f"{row['protocols']['full']['background_to_body_at_least_20mm']}/"
                  f"{row['protocols']['sliding']['background_to_body_at_least_20mm']}", flush=True)
            del outputs, pred, volume, image
        del model
        torch.cuda.empty_cache()
    result["complete"] = len(result["cases"]) == 8
    result["finished_utc"] = datetime.now(timezone.utc).isoformat()
    (raw / "manifest.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    (figures / "README.md").write_text(
        "# P2 fixed-weight Validation diagnostic\n\nFour predefined cases, two unchanged epoch-50 models. "
        "Body panels compare full-volume and fixed 64x64x32 XYZ sliding-window inference; "
        "cyan is ground truth, orange prediction. Head panels show raw Body-minus-Background logits "
        "on the same evaluation-selected slice; these are not probabilities. "
        "All full-volume confusion matrices reproduced G5 before comparison.\n\n"
        f"Raw measurements: logs/validation/inference_context/{cfg['experiment_id']}/manifest.json. "
        "Interpret all six classes and per-case changes before proposing a larger experiment. "
        "No training or Test evaluation was performed.\n")
    for root in (raw, figures):
        files = {p.name: sha256(p) for p in sorted(root.iterdir()) if p.is_file()}
        (root / "checksums.json").write_text(json.dumps(files, indent=2)+"\n")
    print(f"Completed {len(result['cases'])} paired model/case comparisons: {raw}", flush=True)
    print(figures, flush=True)


if __name__ == "__main__":
    main()
