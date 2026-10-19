"""P2B: fixed-weight normalization replay, six-way confidence and geometry.

No retraining or Test access. Reproduce P2 full/native-window confusion first.
One predefined conservative connected-component filter is reported separately.
"""
from __future__ import annotations
import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
from itertools import product
import json
import os
from pathlib import Path
import socket
import sys
import torch
import numpy as np
from scipy import ndimage as ndi

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "code"))
from hipmri_rewrite.data import load_case, downsample, zscore_nonzero, to_3d_model_arrays
from hipmri_rewrite.inference_context import infer_logits
from hipmri_rewrite.prediction_diagnostics import (capture_instance_stats, replay_instance_stats,
    model_digest, confidence, geometry, conservative_component_filter)
from hipmri_rewrite.metrics import confusion
from hipmri_rewrite.splits import discover
from tools.audit_p2_inference_context import xyz_logits, measure, border_distance, draw
from tools.evaluate_g6_test import ARMS, LABELS, selected_model, sha256

CONFIG = PROJECT / "code/config/p2b_prediction_mechanisms.json"
PREVIOUS = PROJECT / "logs/validation/inference_context/p2_four_cases_20261020/manifest.json"
PREVIOUS_SHA = "a21a38f6bd2c312768c923138da25190ce4346b202dea9a1a61563c6d7acad68"
CASES = ("K018_Week4", "B040_Week4", "J026_Week4", "K008_Week0")


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def filter_figure(volume, native, filtered, center, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    z = int(center[2])
    truth = volume.mask[:, :, z].T == 1
    low, high = np.percentile(volume.image[volume.image != 0], (1, 99))
    fig, axes = plt.subplots(1, 3, figsize=(12, 5), layout="constrained")
    for ax, prediction, title in zip(axes, (None, native, filtered),
                                    ("MRI / GT Body", "Unfiltered", "Fixed tiny-component filter")):
        ax.imshow(volume.image[:, :, z].T, origin="lower", cmap="gray", vmin=low, vmax=high,
                  aspect=volume.spacing_mm[1]/volume.spacing_mm[0])
        if truth.any() and not truth.all():
            ax.contour(truth, levels=[.5], colors=["cyan"], linewidths=.8)
        if prediction is not None:
            p = prediction[:, :, z].T == 1
            rgba = np.zeros((*p.shape, 4))
            rgba[p & ~truth] = (1, .1, .1, .65)
            rgba[truth & ~p] = (.2, .4, 1, .65)
            ax.imshow(rgba, origin="lower", aspect=volume.spacing_mm[1]/volume.spacing_mm[0])
        ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{volume.case.identifier}: GT cyan, FP red, FN blue | axial z={z}\n"
                 "GT is evaluation-only; filtering uses predicted components and logits")
    fig.savefig(path, dpi=160); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    args = parser.parse_args()
    if (not os.environ.get("SLURM_JOB_ID") or socket.gethostname().startswith("login")
            or not torch.cuda.is_available()):
        parser.error("Use an allocated GPU compute node")
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    if tuple(cfg["cases"]) != CASES or cfg["split"] != "validation":
        parser.error("Fixed four-case Validation protocol changed")
    if sha256(PREVIOUS) != PREVIOUS_SHA:
        parser.error("Previous P2 record changed or is missing")
    previous = json.loads(PREVIOUS.read_text(encoding="utf-8"))
    for name, digest in previous["sources"].items():
        if sha256(PROJECT / name) != digest:
            raise ValueError(f"Previously measured P2 source changed: {name}")
    expected = {(r["model"], r["case"]): r for r in previous["cases"]}
    if not previous["complete"] or len(expected) != 8:
        parser.error("Complete P2 prerequisite required")
    raw = PROJECT / "logs/validation/inference_context" / cfg["experiment_id"]
    figures = PROJECT / "results/diagnostics/inference_context" / cfg["experiment_id"]
    if raw.exists() or figures.exists():
        parser.error("Output exists; inspect rather than overwrite or resubmit")
    cases = {c.identifier: c for c in discover(args.dataset_root)["validation"]}
    if len(cases) != 38 or not set(CASES).issubset(cases):
        raise ValueError("Validation inventory differs")
    raw.mkdir(parents=True); figures.mkdir(parents=True)
    torch.set_num_threads(2); torch.backends.cudnn.benchmark = False
    result = {"kind": "P2B_prediction_mechanisms", "complete": False, "split": "validation",
              "config": cfg, "config_sha256": sha256(CONFIG), "classes": LABELS,
              "previous_manifest_sha256": PREVIOUS_SHA, "models": {}, "cases": [],
              "job_id": os.environ["SLURM_JOB_ID"], "hostname": socket.gethostname(),
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "environment": {"torch": torch.__version__, "numpy": np.__version__,
                              "gpu": torch.cuda.get_device_name()},
              "sources": {name: sha256(PROJECT / name) for name in (
                  "code/tools/audit_p2b_prediction_mechanisms.py",
                  "code/hipmri_rewrite/prediction_diagnostics.py")},
              "limitations": ["Four selected Validation cases, not a model selection cohort",
                              "Softmax confidence is uncalibrated; descriptive bins are not tuned thresholds",
                              "Replayed full-volume statistics on windows can induce distribution shift",
                              "Connectivity filtering may remove genuine anatomy; not a default prediction",
                              "Six-neighbour Body labels are not a whole-anatomy binary union"]}
    references = {}
    for identity in reversed(ARMS):
        arm = identity[0]
        model, receipt = selected_model(PROJECT / "logs/training", identity, args.dataset_root, torch)
        if receipt["checkpoint_sha256"] != previous["models"][arm]["checkpoint_sha256"]:
            raise ValueError("Selected weights differ from P2")
        model.requires_grad_(False)
        original_digest = model_digest(model)
        result["models"][arm] = receipt
        for case_id in CASES:
            native_volume = load_case(cases[case_id])
            # The even-sized source FOV is exactly preserved by factor-two edge alignment.
            if any(n % 2 for n in native_volume.image.shape):
                raise ValueError("Odd source extent needs explicit original-FOV face handling")
            volume = zscore_nonzero(downsample(native_volume))
            image, _, shape = to_3d_model_arrays(volume)
            old = expected[(arm, case_id)]
            if old["input_sha256"] != {"mri": sha256(cases[case_id].mri), "mask": sha256(cases[case_id].mask)}:
                raise ValueError("Input MRI or mask changed since P2")
            bank = {}
            with (capture_instance_stats(model) if arm == "improved_3d" else nullcontext(bank)) as bank:
                full, full_grid = infer_logits(model, image, device="cuda", components=(arm == "improved_3d"))
            outputs = {"full": xyz_logits(full, shape)}
            self_check = None
            if arm == "improved_3d":
                if len(bank) != 14:
                    raise ValueError("Expected exactly 14 InstanceNorm modules")
                with replay_instance_stats(model, bank):
                    replayed_full, _ = infer_logits(model, image, device="cuda", components=True)
                maximum = max(float(np.abs(full[h]-replayed_full[h]).max()) for h in full)
                disagreement = float(np.mean(full["combined"].argmax(0) != replayed_full["combined"].argmax(0)))
                self_check = {"max_abs_logit_error_all_heads": maximum,
                              "padded_grid_argmax_disagreement_fraction": disagreement}
                if (maximum > cfg["self_replay_max_abs_logit_error"]
                        or disagreement > cfg["self_replay_max_argmax_disagreement_fraction"]):
                    dump(raw / f"{case_id}_self_replay_failure.json", self_check)
                    raise ValueError("Native-full self-replay numerical control failed; stop interpretation")
                del replayed_full
            del full
            native, grid = infer_logits(model, image, device="cuda", window_dhw=(32, 64, 64),
                                         components=(arm == "improved_3d"))
            outputs["native_sliding"] = xyz_logits(native, shape); del native
            for name, old_name in (("full", "full"), ("native_sliding", "sliding")):
                matrix = confusion(volume.mask, outputs[name]["combined"].argmax(0))
                if matrix.tolist() != old["protocols"][old_name]["confusion"]:
                    raise ValueError(f"{arm}/{case_id}/{name}: original P2 confusion not reproduced")
            if arm == "improved_3d":
                with replay_instance_stats(model, bank):
                    replay, replay_grid = infer_logits(model, image, device="cuda", window_dhw=(32, 64, 64), components=True)
                if replay_grid != grid:
                    raise ValueError("Counterfactual window grid or fusion differs from native")
                outputs["replay_sliding"] = xyz_logits(replay, shape); del replay
            distance = ndi.distance_transform_edt(volume.mask != 1, sampling=volume.spacing_mm)
            border = border_distance(shape, volume.spacing_mm)
            preds = {n: d["combined"].argmax(0).astype(np.uint8) for n, d in outputs.items()}
            full_fp = (volume.mask == 0) & (preds["full"] == 1)
            if arm == "improved_3d":
                references[case_id] = full_fp & (distance >= 20)
            regions = {"reference_improved_full_distant_fp": references[case_id],
                       "same_background_correct_full_and_native": (volume.mask == 0) & (preds["full"] == 0) & (preds["native_sliding"] == 0),
                       "own_full_distant_fp": full_fp & (distance >= 20), "gt_body": volume.mask == 1}
            origins = [tuple(o[::-1]) for o in product(*grid["axis_starts_dhw"])]
            row = {"case": case_id, "patient": cases[case_id].patient, "model": arm,
                   "shape_xyz": shape, "spacing_mm": volume.spacing_mm, "affine": volume.affine.tolist(),
                   "original_source_shape_xyz": native_volume.image.shape,
                   "full_and_native_confusion_reproduced": True, "self_replay_check": self_check,
                   "window_grid": grid, "protocols": {}, "normalization_statistics": {}}
            # Evaluation denominators distinguish error concentration from the
            # background's own position/coverage distribution.
            row["available_background_geometry"] = geometry(volume.mask == 0, distance,
                volume.spacing_mm, volume.affine, origins)
            shell = np.zeros(shape, dtype=bool)
            for axis in range(3):
                lo, hi = [slice(None)]*3, [slice(None)]*3
                lo[axis], hi[axis] = slice(0, 8), slice(max(0, shape[axis]-8), shape[axis])
                shell[tuple(lo)] = True; shell[tuple(hi)] = True
            candidate_like = (volume.mask == 0) & (volume.image != 0) & (distance >= 20) & shell
            row["candidate_like_validation_background_available_voxels"] = int(candidate_like.sum())
            for name, record in bank.items():
                row["normalization_statistics"][name] = {"mean": record["mean"].cpu().tolist(),
                    "biased_variance": record["variance"].cpu().tolist(), "eps": record["eps"],
                    "full_input_shape_dhw": record["shape_dhw"]}
            for name, maps in outputs.items():
                fp = (volume.mask == 0) & (preds[name] == 1)
                local_regions = {**regions, "own_current_distant_fp": fp & (distance >= 20)}
                row["protocols"][name] = {
                    **measure(volume.mask, preds[name], distance, border, volume.spacing_mm),
                    "distant_background_fp_in_outer_8_nonzero_shell": int((fp & candidate_like).sum()),
                    "confidence": confidence(maps["combined"], local_regions),
                    "head_body_minus_background": {h: {n: {"n": int(region.sum()),
                        "median": float(np.median((v[1]-v[0])[region])) if region.any() else None,
                        "mean": float(np.mean((v[1]-v[0])[region], dtype=np.float64)) if region.any() else None}
                        for n, region in local_regions.items()} for h, v in maps.items()},
                    "background_fp_geometry": geometry(fp, distance, volume.spacing_mm, volume.affine,
                                                         None if name == "full" else origins),
                    "distant_background_fp_geometry": geometry(fp & (distance >= 20), distance,
                        volume.spacing_mm, volume.affine, None if name == "full" else origins)}
            filtered, filter_receipt = conservative_component_filter(outputs["full"]["combined"], volume.spacing_mm)
            row["postprocessing_diagnostic"] = {"parent": "native_full", "receipt": filter_receipt,
                "metrics": measure(volume.mask, filtered, distance, border, volume.spacing_mm),
                "removed_true_body_voxels": int(((preds["full"] == 1) & (filtered != 1) & (volume.mask == 1)).sum()),
                "removed_false_body_voxels": int(((preds["full"] == 1) & (filtered != 1) & (volume.mask != 1)).sum())}
            if arm == "improved_3d":
                f, n, r = (row["protocols"][p]["background_to_body_at_least_20mm"]
                           for p in ("full", "native_sliding", "replay_sliding"))
                if f <= n:
                    raise ValueError("Predeclared positive mechanism-gate denominator no longer holds")
                ratio = (r-n)/(f-n)
                margins = {p: row["protocols"][p]["confidence"]["reference_improved_full_distant_fp"]
                              ["body_minus_background"]["median"] for p in outputs}
                closer = abs(margins["replay_sliding"]-margins["full"]) < abs(margins["native_sliding"]-margins["full"])
                row["mechanism_gate"] = {"restored_far_fp_fraction_R": ratio,
                    "reference_median_margin_closer_to_full": bool(closer), "case_support": bool(ratio >= .5 and closer)}
                draw(volume, {p: outputs[p] for p in ("native_sliding", "replay_sliding")},
                     old["illustration_center_xyz"], arm, figures)
            filter_figure(volume, preds["full"], filtered, old["illustration_center_xyz"],
                           figures / f"{case_id}_{arm}_filter.png")
            dump(raw / f"{case_id}_{arm}.json", row); result["cases"].append(row)
            print(f"{arm}/{case_id}: P2 reproduced; self-replay={self_check}; "
                  f"gate={row.get('mechanism_gate')}; filter removed={filter_receipt['removed_voxels']}", flush=True)
            del outputs, preds, volume, native_volume, bank
        if model_digest(model) != original_digest:
            raise ValueError("Parameters or buffers changed during diagnostic")
        result["models"][arm]["state_dict_unchanged"] = True
        del model; torch.cuda.empty_cache()
    support = sum(r.get("mechanism_gate", {}).get("case_support", False) for r in result["cases"])
    result["mechanism_gate"] = {"supporting_cases": support, "required_cases": 3,
                                "supports_statistic_source_contribution": support >= 3}
    result["complete"] = len(result["cases"]) == 8
    result["finished_utc"] = datetime.now(timezone.utc).isoformat()
    dump(raw / "manifest.json", result)
    (figures / "README.md").write_text(
        "# P2B prediction-mechanism diagnostic\n\nUnchanged epoch-50 models; four fixed Validation cases. "
        "Body/head panels compare native sliding windows against the identical windows with full-volume "
        "InstanceNorm statistics replayed. Cyan is GT, orange Body prediction. Raw-logit maps are not probabilities. "
        "Filter panels separately show one GT-free tiny-component rule: red FP, blue FN. Neither replay nor filtering "
        "is a default inference protocol. Full/native predictions reproduced P2, numerical replay controls passed, "
        "and model parameter/buffer hashes were unchanged.\n", encoding="utf-8")
    for root in (raw, figures):
        dump(root / "checksums.json", {p.name: sha256(p) for p in sorted(root.iterdir()) if p.is_file()})
    print("Completed P2B", result["mechanism_gate"], raw, figures, sep="\n", flush=True)


if __name__ == "__main__":
    main()
