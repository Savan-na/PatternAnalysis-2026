"""Isolated V2 continuation, full Validation and conservative checkpoint selection.

Only training duration changes. Reuses V2 loss/run_epoch and protected V2 loader.
Default batch stops at E12; extending to E14..20 needs a separate compute decision.
Each epoch checkpoint is immutable. Validation precedes the scheduled epoch save;
an interruption before that save requires replay from the previous saved boundary.
"""
import argparse
from contextlib import contextmanager
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
from time import perf_counter
import uuid

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import train as base
from . import train_improved_v2 as v2
from . import validate as evaluation
from . import validate_improved_v2 as v2_evaluation
from .dataset import APPROVED_PATIENTS, DEFAULT_DATASET_ROOT, discover_dataset

EXPERIMENT_ID = "improved_v3_v2_continuation_v1"
PARENT_SHA256 = "3229bdb38afa7b83af04e79cd7f882059a3510963eff29957554a5438d490e9f"
VALIDATION_SHA256 = "d77e0c53c13a53b8a0af0dd22cf45f6f64798c82a077f597ba9aaa29a80a215e"
SETTINGS_PATH = Path(__file__).with_name("improved_v3_config.json")


def settings():
    value = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    if (value["experiment_id"] != EXPERIMENT_ID or value["parent_epoch"] != 10
            or value["maximum_epoch"] != 20 or value["validation_epochs"] != [12, 14, 16, 18, 20]
            or value["stride"] != 32):
        raise ValueError("V3 schedule/inference policy differs from the declared experiment")
    return value


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


class Journal:
    """Exclusive new JSONL, append-only writes, fsync, size/inode and hash-chain checks.

    This prevents log reuse/truncation by this program and detects common external
    changes. It cannot guarantee storage hardware health or explain V2's NUL block.
    """
    def __init__(self, path):
        self.path = Path(path)
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_APPEND, 0o600)
        self.handle = os.fdopen(fd, "ab", buffering=0)
        self.identity = (os.fstat(fd).st_dev, os.fstat(fd).st_ino)
        self.size, self.sequence, self.previous = 0, 0, "0" * 64

    def emit(self, record):
        stat = self.path.stat()
        if (stat.st_dev, stat.st_ino) != self.identity or stat.st_size != self.size:
            raise RuntimeError("Journal was replaced, truncated or externally modified")
        if {"sequence", "previous_sha256", "record_sha256"} & record.keys():
            raise ValueError("Reserved journal fields")
        item = dict(record, sequence=self.sequence, previous_sha256=self.previous)
        digest = hashlib.sha256(canonical(item)).hexdigest()
        encoded = canonical(dict(item, record_sha256=digest)) + b"\n"
        if self.handle.write(encoded) != len(encoded):
            raise OSError("Short journal write")
        os.fsync(self.handle.fileno())
        self.size += len(encoded)
        self.sequence += 1
        self.previous = digest

    def close(self):
        self.handle.close()


def audit_journal(path):
    data = Path(path).read_bytes()
    if not data or b"\0" in data or not data.endswith(b"\n"):
        raise ValueError("Empty, NUL-corrupted or partial journal")
    previous = "0" * 64
    rows = []
    for sequence, line in enumerate(data.splitlines()):
        row = json.loads(line)
        digest = row.pop("record_sha256")
        if (row["sequence"] != sequence or row["previous_sha256"] != previous
                or hashlib.sha256(canonical(row)).hexdigest() != digest):
            raise ValueError("Journal sequence/hash chain mismatch")
        previous = digest
        rows.append(row)
    return rows


@contextmanager
def preserve_rng(generator, device, model):
    state = base._rng_state(generator, device)
    was_training = model.training
    try:
        yield
    finally:
        random.setstate(state["python"])
        n = state["numpy"]
        np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), *n[2:]))
        torch.set_rng_state(state["torch_cpu"])
        generator.set_state(state["loader"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(state["torch_cuda"])
        model.train(was_training)


def read_parent(path):
    if sha256(path) != PARENT_SHA256:
        raise ValueError("Parent bytes differ from the authenticated V2 E10 checkpoint")
    # Existing protected loader checks V2 identity, sources, config, E10 and strict weights.
    model, metadata = v2_evaluation.load_checkpoint_model(path, torch.device("cpu"))
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if metadata["total_steps"] != 1430 or not payload.get("optimizer", {}).get("state"):
        raise ValueError("Parent is missing the trained optimiser state")
    if not {"python", "numpy", "torch_cpu", "torch_cuda", "loader"} <= payload.get("rng", {}).keys():
        raise ValueError("Parent is missing continuation RNG state")
    if sha256(path) != PARENT_SHA256:
        raise ValueError("Parent changed during authentication")
    return model, payload["config"]


def validate_records(records, pairs):
    expected = {(p.patient_id, p.week) for p in pairs}
    actual = [(r.get("patient_id"), r.get("week")) for r in records]
    if (len(expected) != 38 or len(actual) != 38 or len(set(actual)) != 38
            or set(actual) != expected
            or {p.patient_id for p in pairs} != set(APPROVED_PATIENTS["validation"])):
        raise ValueError("Validation requires exactly the 38 distinct approved identities")
    for record in records:
        if record.get("split") != "validation" or record.get("status") != "success":
            raise ValueError("Failed or non-Validation record; selection forbidden")
        counts = np.asarray(record["confusion_counts"])
        if int(counts.sum()) <= 0 or evaluation.metrics_from_confusion(counts) != record["metrics"]:
            raise ValueError("Validation metric/confusion mismatch")


def authenticate_validation(path, pairs, parent_config):
    if sha256(path) != VALIDATION_SHA256:
        raise ValueError("V2 Validation log SHA-256 differs from authenticated evidence")
    data = Path(path).read_bytes()
    if b"\0" in data:
        raise ValueError("NUL-corrupted V2 Validation log")
    rows = [json.loads(line) for line in data.splitlines()]
    events = [r.get("event") for r in rows]
    if events != ["setup"] + ["volume"] * 38 + ["summary", "result"]:
        raise ValueError("Unexpected V2 Validation event sequence")
    setup, summary, result = rows[0], rows[-2], rows[-1]
    if (setup.get("experiment_id") != v2.EXPERIMENT_ID or setup.get("stride") != 32
            or setup.get("patch_shape_dhw") != [64, 64, 64]
            or setup.get("checkpoint_metadata", {}).get("config_sha256") != parent_config["config_sha256"]
            or setup["checkpoint_metadata"].get("completed_epochs") != 10
            or setup["checkpoint_metadata"].get("total_steps") != 1430
            or summary.get("successful") != 38 or summary.get("failed") != 0
            or summary.get("full_38_volume_validation_completed") is not True
            or result.get("status") != "PASS"):
        raise ValueError("V2 baseline checkpoint/inference/completion identity differs")
    records = rows[1:-2]
    validate_records(records, pairs)
    recomputed = evaluation.summarize(records)
    for key, value in recomputed.items():
        if summary.get(key) != value:
            raise ValueError(f"V2 baseline summary differs from volume records: {key}")
    for key, configured in (("overall_pooled", "v2_pooled_dice"),
                            ("overall_patient_macro", "v2_patient_macro_dice")):
        if any(not math.isclose(r["dice"], d, rel_tol=0, abs_tol=1e-12)
               for r, d in zip(recomputed[key], settings()[configured])):
            raise ValueError("V2 reference metrics differ from predeclared selection policy")
    return enrich_summary(records)


def enrich_summary(records):
    summary = evaluation.summarize(records)
    for row in summary["overall_pooled"]:
        tp, pred, true = row["intersection"], row["predicted_voxels"], row["true_voxels"]
        label = row["class_id"]
        row.update(precision=tp / pred if pred else None, recall=tp / true if true else None,
                   false_positives=pred - tp, false_negatives=true - tp,
                   predicted_volumes=sum(r["metrics"][label]["predicted_voxels"] > 0 for r in records),
                   tp_positive_volumes=sum(r["metrics"][label]["intersection"] > 0 for r in records))
    foreground = [r["dice"] for r in summary["overall_pooled"][1:] if r["dice"] is not None]
    summary.update(foreground_mean_pooled_dice=sum(foreground) / len(foreground) if foreground else None,
                   minimum_foreground_pooled_dice=min(foreground) if foreground else None,
                   successful=len(records), failed=0, full_38_volume_validation_completed=len(records) == 38)
    return summary


def selection(summary, previous, epoch):
    policy = settings()
    rule = policy["selection"]
    pooled = [r["dice"] for r in summary["overall_pooled"]]
    patient = [r["dice"] for r in summary["overall_patient_macro"]]
    if (summary.get("successful") != 38 or summary.get("failed") != 0
            or summary.get("full_38_volume_validation_completed") is not True
            or len(pooled) != 6 or len(patient) != 6
            or any(v is None or not math.isfinite(v) or not 0 <= v <= 1 for v in pooled + patient)):
        raise ValueError("Incomplete or non-finite Validation cannot select a checkpoint")
    score = sum(patient[c] for c in (2, 4, 5)) / 3
    reasons = [f"class_{c}_regression" for c, (v, ref, tol) in enumerate(zip(
        pooled, policy["v2_pooled_dice"], rule["maximum_pooled_dice_drop_from_v2"])) if v < ref - tol]
    reasons += [f"class_{c}_coverage" for c in rule["require_tp_in_all_38_volumes_for_classes"]
                if summary["overall_pooled"][c]["tp_positive_volumes"] != 38]
    choose = not reasons and score > previous["score"] + rule["strict_score_improvement_epsilon"]
    ref_score = sum(policy["v2_patient_macro_dice"][c] for c in (2, 4, 5)) / 3
    meaningful = (not reasons and score >= ref_score + rule["meaningful_score_gain"]
                  and min(pooled[1:]) >= min(policy["v2_pooled_dice"][1:]) + rule["meaningful_minimum_foreground_gain"])
    return ({"epoch": epoch, "score": score} if choose else dict(previous)), {
        "candidate_score": score, "selected": choose, "rejection_reasons": reasons,
        "meaningful_improvement": meaningful, "all_class_pooled_dice_at_least_070": min(pooled) >= .70}


def source_hashes():
    return {**v2.source_hashes(), **{name: sha256(Path(__file__).with_name(name)) for name in
            ("train_improved_v3.py", "improved_v3_config.json")}}


def make_config(parent_config, validation_pairs):
    result = {"experiment_id": EXPERIMENT_ID, "settings": settings(), "steps_per_epoch": 143,
              "parent_sha256": PARENT_SHA256, "baseline_validation_sha256": VALIDATION_SHA256,
              "parent_config": parent_config, "source_sha256": source_hashes(),
              "validation_pairs": [[p.patient_id, p.week, str(p.mri_path), str(p.segmentation_path)] for p in validation_pairs]}
    result["config_sha256"] = v2.config_hash(result)
    return result


def validate_config(config):
    v2.validate_config(config["parent_config"])
    if (config.get("experiment_id") != EXPERIMENT_ID or config.get("settings") != settings()
            or config.get("parent_sha256") != PARENT_SHA256
            or config.get("baseline_validation_sha256") != VALIDATION_SHA256
            or config.get("source_sha256") != source_hashes()
            or config.get("steps_per_epoch") != 143
            or config.get("config_sha256") != v2.config_hash(config)):
        raise ValueError("V3 identity/configuration/source provenance mismatch")


def resume_checkpoint(path, model, optimizer, *, config, generator, device):
    validate_config(config)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    epoch = payload.get("completed_epochs")
    result = payload.get("epoch_result", {})
    if (type(epoch) is not int or not 11 <= epoch <= 20 or result.get("expected_steps") != 143
            or "selection" not in result or (epoch in settings()["validation_epochs"] and "validation" not in result)):
        raise ValueError("V3 resume requires a saved training/Validation boundary")
    selected = result["selection"]
    if (selected.get("epoch") not in [10] + [e for e in settings()["validation_epochs"] if e <= epoch]
            or not isinstance(selected.get("score"), (int, float)) or not math.isfinite(selected["score"])):
        raise ValueError("Invalid selection state")
    completed, steps = base.resume_checkpoint(path, model, optimizer, config=config, generator=generator, device=device)
    return completed, steps, selected


def full_validation(model, pairs, device, generator, emit, epoch):
    records = []
    start = perf_counter()
    policies = evaluation._local_policies()["resampling"]
    with preserve_rng(generator, device, model), torch.inference_mode():
        model.eval()
        for pair in pairs:
            record = evaluation._process_pair(pair, model, device, policies["target_spacing_mm"],
                                               policies["max_estimated_bytes"], settings()["stride"])
            emit(dict(record, completed_epochs=epoch))
            if record.get("status") != "success":
                raise RuntimeError(f"Validation failed for {record['case']}: {record.get('exception_message')}")
            records.append(record)
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    validate_records(records, pairs)
    summary = enrich_summary(records)
    summary.update(validation_loop_seconds=perf_counter() - start,
                   peak_gpu_allocated_bytes=max(r["peak_gpu_allocated_bytes"] for r in records),
                   peak_gpu_reserved_bytes=max(r["peak_gpu_reserved_bytes"] for r in records))
    emit(dict(summary, event="validation_summary", completed_epochs=epoch))
    return summary


def atomic_json(path, record):
    fd, name = tempfile.mkstemp(prefix=".selection_", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(canonical(record) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def checkpoint_file(run_dir, epoch):
    return run_dir / f"improved_unet_v3_epoch{epoch:02d}.pt"


def execute(model, optimizer, loader, device, generator, config, run_dir, parent_path,
            completed, steps, selected, pairs, stop_epoch, diagnostic, emit):
    """Run an authorised bounded batch; tests inject synthetic loader/Validation."""
    for epoch in range(completed, stop_epoch):
        validate_config(config)
        path = checkpoint_file(run_dir, epoch + 1)
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}; resume the saved boundary")
        loader.dataset.set_epoch(epoch)
        start = perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        result = v2.run_epoch(model, loader, optimizer, device, diagnostic=diagnostic,
                              expected_patch_shape=(64, 64, 64),
                              progress=lambda r: emit(dict(r, completed_epochs=epoch, human_epoch=epoch + 1,
                                                           global_step=steps + r["step"])))
        if device.type == "cuda":
            result.update(peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(device),
                          peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(device))
        if diagnostic:
            emit(dict(result, event="diagnostic", checkpoint_saved=False, starting_total_steps=steps))
            return completed, steps, selected
        steps += result["processed_steps"]
        completed = epoch + 1
        if completed in settings()["validation_epochs"]:
            summary = full_validation(model, pairs, device, generator, emit, completed)
            selected, decision = selection(summary, selected, completed)
            result.update(validation=summary, selection_decision=decision)
            emit(dict(decision, event="selection", completed_epochs=completed, retained=selected))
        result["selection"] = dict(selected)
        validate_config(config)
        base.atomic_checkpoint(path, model, optimizer, completed_epochs=completed, total_steps=steps,
                               config=config, generator=generator, device=device, epoch_result=result)
        chosen = parent_path if selected["epoch"] == 10 else checkpoint_file(run_dir, selected["epoch"])
        atomic_json(run_dir / "selection.json", {**selected, "checkpoint": str(chosen),
                    "checkpoint_sha256": sha256(chosen), "latest_epoch": completed,
                    "config_sha256": config["config_sha256"], "test_evaluated": False})
        emit(dict(result, event="epoch", completed_epochs=completed, total_steps=steps,
                  checkpoint=str(path), checkpoint_sha256=sha256(path),
                  epoch_including_validation_and_save_seconds=perf_counter() - start))
    return completed, steps, selected


def runtime_contract(parent_config, dataset, root, device=None):
    import nibabel
    import scipy
    versions = {"torch": str(torch.__version__), "numpy": np.__version__, "nibabel": nibabel.__version__,
                "scipy": scipy.__version__, "cuda_build": torch.version.cuda}
    if (parent_config.get("versions") != versions
            or parent_config.get("dataset_root") != str(root.resolve())
            or parent_config.get("train_pairs") != [[p.patient_id, p.week, str(p.mri_path), str(p.segmentation_path)] for p in dataset.pairs]
            or len(dataset) != 143 or dataset.patch_shape_dhw != (64, 64, 64)
            or dataset.train_patches_per_volume != 1 or dataset.seed != 3710
            or dataset.train_foreground_probability != .5):
        raise ValueError("V2 runtime, dataset manifest or sampling policy differs; exact continuation rejected")
    if device is not None and (parent_config.get("gpu") != torch.cuda.get_device_name(device)
                              or parent_config.get("cuda_device_count") != torch.cuda.device_count()):
        raise ValueError("V2 GPU identity/device count differs")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-checkpoint", required=True, type=Path)
    parser.add_argument("--v2-validation-log", required=True, type=Path)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after-epoch", type=int, choices=(12, 14, 16, 18, 20), default=12)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--preflight-only", action="store_true")
    modes.add_argument("--diagnostic-one-step", action="store_true")
    args = parser.parse_args(argv)
    if (not args.preflight_only and args.run_dir is None) or (args.resume and (args.preflight_only or args.diagnostic_one_step)):
        parser.error("run-dir is required for compute; preflight/diagnostic cannot resume V3")
    parent = args.parent_checkpoint.expanduser().resolve(strict=True)
    model, parent_config = read_parent(parent)
    from .patch_dataset import HipMRIPatchDataset
    dataset = HipMRIPatchDataset("train", args.dataset_root)
    runtime_contract(parent_config, dataset, args.dataset_root)
    pairs = evaluation.select_validation(discover_dataset(args.dataset_root))
    reference = authenticate_validation(args.v2_validation_log, pairs, parent_config)
    config = make_config(parent_config, pairs)
    validate_config(config)
    if args.preflight_only:
        print(json.dumps({"status": "PASS", "scope": "CPU provenance preflight; no arrays decoded",
                          "parent_sha256": PARENT_SHA256, "config": config}, allow_nan=False), flush=True)
        return
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("An allocated Slurm CUDA compute session is required")
    run_dir = args.run_dir.expanduser().resolve(strict=True)
    base.checkpoint_path(run_dir / "improved_unet_v3_guard.pt", args.dataset_root)
    if run_dir.is_relative_to(parent.parent):
        raise ValueError("Use a separate V3 directory outside the protected V2 directory")
    device = torch.device("cuda:0")
    runtime_contract(parent_config, dataset, args.dataset_root, device)
    base.configure_reproducibility(3710, device)
    generator = torch.Generator().manual_seed(3710)
    loader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=True, generator=generator, drop_last=False)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.00001,
                                 betas=(.9, .999), eps=1e-8, foreach=False, fused=False)
    if args.resume:
        resume = args.resume.expanduser().resolve(strict=True)
        if resume.parent != run_dir:
            raise ValueError("Resume must be an existing checkpoint in this V3 run directory")
        completed, steps, selected = resume_checkpoint(resume, model, optimizer, config=config, generator=generator, device=device)
        if resume != checkpoint_file(run_dir, completed):
            raise ValueError("Resume filename does not match saved epoch")
    else:
        completed, steps = v2.resume_checkpoint(parent, model, optimizer, config=parent_config, generator=generator, device=device)
        selected = {"epoch": 10, "score": sum(reference["overall_patient_macro"][c]["dice"] for c in (2, 4, 5)) / 3}
    if completed >= args.stop_after_epoch:
        raise ValueError("Requested stopping boundary has already been reached")
    chosen = parent if selected["epoch"] == 10 else checkpoint_file(run_dir, selected["epoch"])
    if not chosen.is_file():
        raise ValueError("Previously selected checkpoint is missing")
    lock = run_dir / "active.lock"
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.write(fd, f"Slurm {os.environ['SLURM_JOB_ID']}; PID {os.getpid()}\n".encode())
    os.close(fd)
    journal = None
    try:
        log = run_dir / f"events_{os.environ['SLURM_JOB_ID']}_{uuid.uuid4().hex}.jsonl"
        journal = Journal(log)
        journal.emit({"event": "setup", "config": config, "parent_checkpoint": str(parent),
                      "resume": str(args.resume) if args.resume else None, "starting_completed_epochs": completed,
                      "starting_total_steps": steps, "stop_after_epoch": args.stop_after_epoch,
                      "diagnostic": args.diagnostic_one_step, "gpu": torch.cuda.get_device_name(device)})
        print(f"Evidence journal: {log}", flush=True)
        completed, steps, selected = execute(model, optimizer, loader, device, generator, config, run_dir,
            parent, completed, steps, selected, pairs, args.stop_after_epoch, args.diagnostic_one_step, journal.emit)
        if sha256(parent) != PARENT_SHA256:
            raise RuntimeError("Protected parent changed during execution")
        journal.emit({"event": "result", "status": "PASS", "completed_epochs": completed, "total_steps": steps,
                      "diagnostic_updates": 1 if args.diagnostic_one_step else 0, "selected": selected,
                      "checkpoint_saved": not args.diagnostic_one_step, "parent_unchanged": True, "test_arrays_decoded": 0})
        audit_journal(log)
        print(f"PASS: bounded V3 batch; journal SHA-256 {sha256(log)}", flush=True)
    except Exception as exc:
        if journal is not None:
            try:
                journal.emit({"event": "error", "status": "FAIL", "type": type(exc).__name__, "message": str(exc)})
            except Exception:
                pass  # Preserve the original error; never repair a damaged journal.
        raise
    finally:
        if journal is not None:
            journal.close()
        lock.unlink()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)
