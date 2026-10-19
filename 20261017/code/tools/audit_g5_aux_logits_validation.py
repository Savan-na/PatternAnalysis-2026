"""Diagnose the selected Improved 3D auxiliary-logit sum on Validation only.

The main head was trained as part of the summed output. Its stand-alone
prediction is a diagnostic counterfactual, not a trained model comparison.
No weights, losses, thresholds or Test data are changed.
"""

from __future__ import annotations

import json
import os
import socket
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "code"))
from hipmri_rewrite.boundaries import symmetric_surface_distances  # noqa: E402
from hipmri_rewrite.data import (downsample, from_3d_model_prediction,  # noqa: E402
                                 load_case, to_3d_model_arrays, zscore_nonzero)
from hipmri_rewrite.metrics import confusion, overlap  # noqa: E402
from hipmri_rewrite.splits import discover  # noqa: E402
from hipmri_rewrite.training import patient_small_organ_score  # noqa: E402
from tools.evaluate_g6_test import ARMS, selected_model, sha256  # noqa: E402

OUTPUT = PROJECT / "results/tables/comparisons/g5_validation_aux_logits.json"
BODY_AUDIT = PROJECT / "results/tables/comparisons/g5_validation_body_audit.json"
SAVED = PROJECT / "logs/training/g5_improved_20261018/epoch_050.json"


@torch.inference_mode()
def predictions(model, volume) -> dict[str, np.ndarray]:
    image, _, shape = to_3d_model_arrays(volume)
    tensor = torch.from_numpy(np.ascontiguousarray(image))[None, None].to("cuda")
    combined, components = model(tensor, return_components=True)
    main, half, quarter = components
    if not torch.allclose(combined, main + half + quarter, atol=1e-6):
        raise ValueError("Auxiliary-logit sum has changed")
    if combined.shape != (1, 6, *image.shape):
        raise ValueError("Unexpected logit dimensions")
    return {name: from_3d_model_prediction(logits.argmax(1)[0].cpu().numpy().astype(np.uint8),
                                           shape)
            for name, logits in (("combined", combined), ("main_only", main))}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    args = parser.parse_args()
    if (not os.environ.get("SLURM_JOB_ID") or socket.gethostname().startswith("login")
            or not torch.cuda.is_available()):
        parser.error("Run on an allocated GPU compute node")
    if OUTPUT.exists():
        parser.error("Diagnostic output already exists")
    if not BODY_AUDIT.exists():
        parser.error("Completed Validation Body audit is required")
    torch.set_num_threads(2)
    body = json.loads(BODY_AUDIT.read_text(encoding="utf-8"))
    if len(body["cases"]) != 38:
        raise ValueError("Incomplete Validation Body audit")
    identity = next(identity for identity in ARMS if identity[0] == "improved_3d")
    model, receipt = selected_model(PROJECT / "logs/training", identity,
                                    args.dataset_root, torch)
    if receipt["checkpoint_sha256"] != body["sources"]["improved_3d"]["checkpoint_sha256"]:
        raise ValueError("Improved checkpoint differs from Body audit")
    saved = json.loads(SAVED.read_text(encoding="utf-8"))
    expected = {row["case"]: row["confusion"] for row in saved["validation"]["cases"]}
    cases = {case.identifier: case for case in discover(args.dataset_root)["validation"]}
    if len(cases) != 38 or set(cases) != set(expected):
        raise ValueError("Validation cohort differs from selected record")
    result = {"kind": "G5_Improved_auxiliary_sum_Validation_diagnostic",
              "split": "validation", "interpretation": "main_only is an untrained stand-alone head",
              "checkpoint": receipt, "body_audit_sha256": sha256(BODY_AUDIT),
              "cases": [], "summary": {}}
    pooled = {name: np.zeros((6, 6), dtype=np.int64) for name in ("combined", "main_only")}
    patients = {name: defaultdict(lambda: np.zeros((6, 6), dtype=np.int64))
                for name in pooled}
    for index, case_id in enumerate(sorted(cases), 1):
        volume = zscore_nonzero(downsample(load_case(cases[case_id])))
        outputs = predictions(model, volume)
        row = {"case": case_id, "patient": cases[case_id].patient, "outputs": {}}
        distance = distance_transform_edt(volume.mask != 1, sampling=volume.spacing_mm)
        for name, prediction in outputs.items():
            matrix = confusion(volume.mask, prediction)
            if name == "combined" and matrix.tolist() != expected[case_id]:
                raise ValueError(f"{case_id}: combined output does not reproduce G5")
            pooled[name] += matrix
            patients[name][row["patient"]] += matrix
            false_body = (prediction == 1) & (volume.mask == 0)
            row["outputs"][name] = {
                "confusion": matrix.tolist(),
                "background_to_body_voxels": int(false_body.sum()),
                "background_to_body_at_least_20mm": int((false_body & (distance >= 20)).sum()),
                "body_hd95_mm": symmetric_surface_distances(
                    volume.mask, prediction, volume.spacing_mm, label=1)["hd95_mm"],
            }
        result["cases"].append(row)
        print(f"[{index}/38] {case_id}: combined output reproduced", flush=True)
    for name in pooled:
        score, detail = patient_small_organ_score(patients[name])
        result["summary"][name] = {
            "pooled_confusion": pooled[name].tolist(),
            "pooled_overlap": overlap(pooled[name]),
            "patient_small_organ_score": score,
            "patient_small_organ_detail": detail,
            "median_body_hd95_mm": median(row["outputs"][name]["body_hd95_mm"]
                                        for row in result["cases"]),
            "background_to_body_voxels": sum(row["outputs"][name]["background_to_body_voxels"]
                                              for row in result["cases"]),
            "background_to_body_at_least_20mm": sum(
                row["outputs"][name]["background_to_body_at_least_20mm"]
                for row in result["cases"]),
        }
    if abs(result["summary"]["combined"]["patient_small_organ_score"]
           - saved["validation"]["selection_score"]) > 1e-10:
        raise ValueError("Combined Validation selection score differs")
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for name, summary in result["summary"].items():
        print(name, "selection", round(summary["patient_small_organ_score"], 4),
              "Body_HD95", round(summary["median_body_hd95_mm"], 2),
              "Background_to_Body", summary["background_to_body_voxels"])
    print(OUTPUT)


if __name__ == "__main__":
    main()
