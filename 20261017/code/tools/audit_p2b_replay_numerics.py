"""Locate P2B numerical-control failure; no mechanism/segmentation intervention.

One unchanged Improved checkpoint, K018_Week4, full-volume inference only.
Compare native repetition, historical manual replay, and two predeclared
backend-statistic implementations. Historical P2B thresholds are unchanged.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import socket
import sys
import torch
import numpy as np
from torch.nn import functional as F

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "code"))
from hipmri_rewrite.data import load_case, downsample, zscore_nonzero, to_3d_model_arrays
from hipmri_rewrite.inference_context import infer_logits
from hipmri_rewrite.metrics import confusion
from hipmri_rewrite.prediction_diagnostics import model_digest, replay_instance_stats
from hipmri_rewrite.splits import discover
from tools.audit_p2_inference_context import xyz_logits
from tools.evaluate_g6_test import ARMS, selected_model, sha256


def replay_layer(module, x, record, method):
    if method == "backend_elemt":
        return torch.ops.aten.batch_norm_elemt(x.contiguous(), module.weight, module.bias,
            record["backend_mean"], record["backend_invstd"], module.eps)
    if method != "backend_fused_affine":
        raise ValueError("Unknown numerical implementation")
    scale = record["backend_invstd"]
    if module.affine:
        scale = scale * module.weight
        offset = module.bias-record["backend_mean"]*scale
    else:
        offset = -record["backend_mean"]*scale
    return torch.addcmul(offset[None, :, None, None, None], x,
                        scale[None, :, None, None, None])


def inspect_layer(module, x, native):
    if x.shape[0] != 1 or module.training or module.track_running_stats:
        raise ValueError("Numerical comparison requires B=1, eval, no running statistics")
    var, mean = torch.var_mean(x.detach(), dim=(0, 2, 3, 4), correction=0)
    # With B=1, InstanceNorm's reshaped batch-normalization domain is identical.
    # The training flag here selects input statistics; no model is trained and
    # no running buffers are supplied or updated.
    reference, saved_mean, saved_invstd, _, backend = torch.ops.aten._batch_norm_impl_index(
        x.contiguous(), module.weight, module.bias, None, None, True,
        module.momentum if module.momentum is not None else 0., module.eps,
        torch.backends.cudnn.enabled)
    bank = {"mean": mean, "variance": var, "eps": module.eps,
            "shape_dhw": list(x.shape[2:]), "backend_mean": saved_mean,
            "backend_invstd": saved_invstd}
    report = {"shape": list(x.shape), "dtype": str(x.dtype), "eps": module.eps,
              "backend_index": int(backend),
              "input_abs_max": float(x.abs().max()), "input_variance_min": float(var.min()),
              "native_vs_batch_backend_max_abs": float((native-reference).abs().max()),
              "manual_vs_backend_mean_max_abs": float((mean-saved_mean).abs().max()),
              "manual_vs_backend_invstd_max_abs": float((torch.rsqrt(var+module.eps)-saved_invstd).abs().max()),
              "local_output_errors": {}}
    manual = (x-mean[None, :, None, None, None]) * torch.rsqrt(var+module.eps)[None, :, None, None, None]
    if module.affine:
        manual = manual*module.weight[None, :, None, None, None]+module.bias[None, :, None, None, None]
    repeated = F.instance_norm(x, None, None, module.weight, module.bias, True,
                              module.momentum if module.momentum is not None else 0., module.eps)
    for method, value in (("native_repeat", repeated), ("historical_manual", manual)):
        report["local_output_errors"][method] = {"available": True,
            "max_abs": float((value-native).abs().max()), "mean_abs": float((value-native).abs().mean())}
    for method in ("backend_fused_affine", "backend_elemt"):
        try:
            value = replay_layer(module, x, bank, method)
            report["local_output_errors"][method] = {"available": True,
                "max_abs": float((value-native).abs().max()), "mean_abs": float((value-native).abs().mean())}
        except (RuntimeError, NotImplementedError) as error:
            report["local_output_errors"][method] = {"available": False, "error": str(error)[:300]}
    return bank, report


@contextmanager
def capture_layers(model, bank, reports):
    handles = []
    def hook(name):
        def inspect(module, args, output):
            if name in bank:
                raise ValueError("Expected exactly one reference call per normalization")
            bank[name], reports[name] = inspect_layer(module, args[0], output)
        return inspect
    try:
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.InstanceNorm3d):
                handles.append(module.register_forward_hook(hook(name)))
        yield
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def replay_layers(model, bank, method):
    handles = []
    def hook(name):
        return lambda module, args, output: replay_layer(module, args[0], bank[name], method)
    try:
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.InstanceNorm3d):
                handles.append(module.register_forward_hook(hook(name)))
        yield
    finally:
        for handle in handles:
            handle.remove()


def differences(reference, output, cfg):
    per_head = {h: {"max_abs": float(np.abs(reference[h]-output[h]).max()),
                    "mean_abs": float(np.abs(reference[h]-output[h]).mean())} for h in reference}
    labels = reference["combined"].argmax(0) != output["combined"].argmax(0)
    maximum = max(row["max_abs"] for row in per_head.values())
    fraction = float(labels.mean())
    return {"heads": per_head, "changed_padded_voxels": int(labels.sum()),
            "padded_argmax_disagreement_fraction": fraction,
            "passes_original_control": bool(maximum <= cfg["self_replay_max_abs_logit_error"]
                and fraction <= cfg["self_replay_max_argmax_disagreement_fraction"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--self-test", action="store_true")
    group.add_argument("--dataset-root", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.self_test:
        torch.manual_seed(73)
        m = torch.nn.InstanceNorm3d(3, affine=True).eval().requires_grad_(False)
        with torch.inference_mode():
            x = torch.randn(1, 3, 16, 16, 16)
            native = m(x); bank, report = inspect_layer(m, x, native)
            torch.testing.assert_close(native, replay_layer(m, x, bank, "backend_fused_affine"), atol=1e-4, rtol=1e-4)
        print("CPU native-statistics/fused-affine self-test passed; CUDA equivalence is not established")
        return
    if (not os.environ.get("SLURM_JOB_ID") or socket.gethostname().startswith("login")
            or not torch.cuda.is_available()):
        parser.error("Run the real audit on an allocated GPU node")
    out = PROJECT / "logs/validation/inference_context/p2b_numerical_audit_20261020"
    if out.exists():
        parser.error("Numerical-audit output exists; do not overwrite or resubmit")
    cfg_path = PROJECT / "code/config/p2b_prediction_mechanisms.json"
    cfg = json.loads(cfg_path.read_text())
    if (cfg["self_replay_max_abs_logit_error"] != .001
            or cfg["self_replay_max_argmax_disagreement_fraction"] != .00001):
        raise ValueError("Historical numerical thresholds must remain unchanged")
    previous = PROJECT / "logs/validation/inference_context/p2_four_cases_20261020/manifest.json"
    if sha256(previous) != "a21a38f6bd2c312768c923138da25190ce4346b202dea9a1a61563c6d7acad68":
        raise ValueError("Fixed P2 prerequisite changed")
    expected = next(r for r in json.loads(previous.read_text())["cases"]
                    if r["case"] == "K018_Week4" and r["model"] == "improved_3d")
    case = next(c for c in discover(args.dataset_root)["validation"] if c.identifier == "K018_Week4")
    inputs = {"mri": sha256(case.mri), "mask": sha256(case.mask)}
    if inputs != expected["input_sha256"]:
        raise ValueError("K018 inputs changed since P2")
    model, receipt = selected_model(PROJECT / "logs/training", ARMS[1], args.dataset_root, torch)
    model.requires_grad_(False); before = model_digest(model)
    volume = zscore_nonzero(downsample(load_case(case)))
    image, _, shape = to_3d_model_arrays(volume)
    truth = volume.mask
    torch.backends.cudnn.benchmark = False
    out.mkdir(parents=True)
    bank, local = {}, {}
    result = {"kind": "P2B_numerical_control_audit_only", "complete": False, "case": case.identifier,
              "split": "validation", "job_id": os.environ["SLURM_JOB_ID"], "model": receipt,
              "torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
              "cudnn_version": torch.backends.cudnn.version(), "config_sha256": sha256(cfg_path),
              "input_sha256": inputs,
              "historical_replay_module_sha256": sha256(PROJECT / "code/hipmri_rewrite/prediction_diagnostics.py"),
              "original_control_thresholds": {k: cfg[k] for k in (
                  "self_replay_max_abs_logit_error", "self_replay_max_argmax_disagreement_fraction")},
              "script_sha256": sha256(Path(__file__)), "layer_local_errors": local, "full_volume_errors": {},
              "interpretation": "Only numerical implementations compared. No sliding-window mechanism, filtering or new segmentation performance evaluation."}
    with capture_layers(model, bank, local):
        reference, _ = infer_logits(model, image, device="cuda", components=True)
    if len(bank) != 14:
        raise ValueError("Expected 14 normalization modules")
    prediction = xyz_logits(reference, shape)["combined"].argmax(0)
    if confusion(truth, prediction).tolist() != expected["protocols"]["full"]["confusion"]:
        raise ValueError("Native full volume no longer reproduces P2")
    for method in ("native_repeat", "historical_manual", "backend_fused_affine", "backend_elemt"):
        if method in ("backend_fused_affine", "backend_elemt") and not all(
                r["local_output_errors"][method]["available"] for r in local.values()):
            result["full_volume_errors"][method] = {"available": False}
            continue
        context = (replay_instance_stats(model, bank) if method == "historical_manual" else
                   replay_layers(model, bank, method) if method != "native_repeat" else null_context())
        with context:
            output, _ = infer_logits(model, image, device="cuda", components=True)
        result["full_volume_errors"][method] = {"available": True, **differences(reference, output, cfg)}
        del output
        (out / "audit.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
        print(method, result["full_volume_errors"][method], flush=True)
    if model_digest(model) != before:
        raise ValueError("Model state changed")
    result["state_dict_unchanged"] = True; result["complete"] = True
    (out / "audit.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(out / "audit.json", flush=True)


@contextmanager
def null_context():
    yield


if __name__ == "__main__":
    main()
