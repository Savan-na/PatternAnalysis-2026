"""Isolated A/B/C execution of the frozen V3 E10->E12 experiment.

A/B save training-only boundaries; these are NOT validated V3 checkpoints.
C alone validates E12 and selects. Original V3 loaders remain unchanged.
No retry, scheduling, Test decoding, or checkpoint conversion is performed.
"""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import uuid

import torch
from torch.utils.data import DataLoader

from . import train_improved_v3 as v3

IDENTITY = "improved_v3_split_e12_v1"


def config_for(parent_config, pairs):
    cfg = {
        "experiment_id": IDENTITY,
        "steps_per_epoch": 143,
        "protocol": v3.make_config(parent_config, pairs),
        "schedule": {"A": 11, "B": 12, "C": "validation12"},
        "source_sha256": {
            name: v3.sha256(Path(__file__).with_name(name))
            for name in ("run_improved_v3_split.py", "test_improved_v3_split.py")
        },
    }
    cfg["config_sha256"] = v3.v2.config_hash(cfg)
    return cfg


def validate_config(cfg):
    v3.validate_config(cfg["protocol"])
    if (cfg.get("experiment_id") != IDENTITY or cfg.get("steps_per_epoch") != 143
            or cfg.get("schedule") != {"A": 11, "B": 12, "C": "validation12"}
            or cfg.get("config_sha256") != v3.v2.config_hash(cfg)
            or cfg.get("source_sha256") != {
                n: v3.sha256(Path(__file__).with_name(n))
                for n in ("run_improved_v3_split.py", "test_improved_v3_split.py")}):
        raise ValueError("Split identity/configuration/source mismatch")


def restore_boundary(path, expected_epoch, model, optimizer, cfg, generator, device):
    """Strict training boundary loader; deliberately separate from V3 validated resume."""
    validate_config(cfg)
    payload = torch.load(path, weights_only=True, map_location="cpu")
    if (expected_epoch not in (11, 12) or payload.get("config") != cfg
            or payload.get("completed_epochs") != expected_epoch
            or payload.get("epoch_result", {}).get("validation_pending") is not True
            or payload.get("epoch_result", {}).get("expected_steps") != 143
            or not payload.get("optimizer", {}).get("state")
            or not {"python", "numpy", "torch_cpu", "torch_cuda", "loader"}
                <= payload.get("rng", {}).keys()):
        raise ValueError("Invalid split training-only checkpoint boundary")
    del payload
    return v3.base.resume_checkpoint(path, model, optimizer, config=cfg,
                                    generator=generator, device=device)


def checkpoint_path(root, stage):
    return root / stage / f"training_epoch{11 if stage == 'A' else 12}_pending_validation.pt"


def audit_stage(root, stage, cfg):
    """Authenticate a successful stage and its immutable output before consuming it."""
    logs = list((root / stage).glob("events_*.jsonl"))
    if len(logs) != 1:
        raise ValueError("Expected exactly one stage journal; do not merge attempts")
    rows = v3.audit_journal(logs[0])
    if (rows[0].get("event") != "setup" or rows[0].get("stage") != stage
            or rows[0].get("config") != cfg or rows[-1].get("event") != "result"
            or rows[-1].get("status") != "PASS"
            or rows[-1].get("parent_unchanged") is not True
            or rows[-1].get("test_arrays_decoded") != 0):
        raise ValueError("Stage identity/completion differs")
    expected_start = {"A": (10, 1430), "B": (11, 1573), "C": (12, 1716)}[stage]
    if (rows[0].get("starting_completed_epochs"), rows[0].get("starting_total_steps")) != expected_start:
        raise ValueError("Incorrect stage starting boundary")
    if stage in ("A", "B"):
        steps = [r for r in rows if r["event"] == "step"]
        start = 1430 if stage == "A" else 1573
        epoch = 11 if stage == "A" else 12
        if ([r["global_step"] for r in steps] != list(range(start+1, start+144))
                or any(r["human_epoch"] != epoch for r in steps)
                or [r["event"] for r in rows] != ["setup"] + ["step"]*143 + ["checkpoint", "result"]):
            raise ValueError("Incomplete or unexpected training step sequence")
        if any(not math.isfinite(r[k]) for r in steps
               for k in ("loss", "weighted_ce", "ordinary_ce", "soft_dice_loss")):
            raise ValueError("Non-finite training loss")
        record = rows[-2]
        if (Path(record["path"]) != checkpoint_path(root, stage)
                or v3.sha256(record["path"]) != record["sha256"]):
            raise ValueError("Boundary checksum differs")
    return rows, v3.sha256(logs[0])


def audit_complete(root, stage):
    """CPU-only evidence audit. No MRI decoding, model forward or CUDA call."""
    first = list((root / "A").glob("events_*.jsonl"))
    if len(first) != 1:
        raise ValueError("Missing/ambiguous A journal")
    cfg = v3.audit_journal(first[0])[0]["config"]
    validate_config(cfg)
    reports = {}
    for name in ("A", "B", "C")[:("A", "B", "C").index(stage)+1]:
        rows, digest = audit_stage(root, name, cfg)
        expected_inputs = {k: reports[k]["journal_sha256"] for k in reports}
        if rows[0].get("input_journals") != expected_inputs:
            raise ValueError("Stage lineage differs")
        reports[name] = {"journal_sha256": digest}
        if name in ("A", "B"):
            payload = torch.load(checkpoint_path(root, name), weights_only=True, map_location="cpu")
            epoch = 11 if name == "A" else 12
            if (payload["config"] != cfg or payload["completed_epochs"] != epoch
                    or payload["total_steps"] != epoch*143
                    or payload["epoch_result"].get("validation_pending") is not True
                    or payload["epoch_result"].get("epoch_completed") is not True
                    or payload["epoch_result"].get("processed_steps") != 143
                    or not payload["optimizer"]["state"]
                    or not {"python", "numpy", "torch_cpu", "torch_cuda", "loader"} <= payload["rng"].keys()):
                raise ValueError("Invalid saved training boundary")
            reports[name]["training"] = rows[-2]
            del payload
        else:
            if [r["event"] for r in rows] != ["setup"] + ["volume"]*38 + ["validation_summary", "selection", "result"]:
                raise ValueError("Unexpected Validation event sequence")
            from .dataset import VolumePair
            pairs = [VolumePair(p,w,Path(m),Path(s)) for p,w,m,s in cfg["protocol"]["validation_pairs"]]
            volumes = rows[1:39]
            if any(r["completed_epochs"] != 12 for r in volumes):
                raise ValueError("Wrong Validation epoch")
            v3.validate_records(volumes, pairs)
            summary = v3.enrich_summary(volumes)
            for key, value in summary.items():
                if rows[39].get(key) != value:
                    raise ValueError(f"Summary mismatch: {key}")
            policy = v3.settings()
            baseline = dict(epoch=10, score=sum(policy["v2_patient_macro_dice"][c] for c in (2,4,5))/3)
            best, decision = v3.selection(summary, baseline, 12)
            receipt = json.loads((root / "C" / "selection.json").read_text())
            expected = dict(receipt, event="selection")
            logged = {k:v for k,v in rows[40].items() if k not in ("sequence", "previous_sha256")}
            if (logged != expected or receipt["epoch"] != best["epoch"]
                    or receipt["score"] != best["score"] or receipt["decision"] != decision
                    or receipt["config_sha256"] != cfg["config_sha256"]
                    or receipt["validation_completed"] is not True or receipt["test_evaluated"] is not False
                    or v3.sha256(receipt["checkpoint"]) != receipt["checkpoint_sha256"]):
                raise ValueError("Selection receipt differs from frozen recomputed decision")
            if best["epoch"] == 12 and Path(receipt["checkpoint"]) != checkpoint_path(root, "B"):
                raise ValueError("Wrong selected E12 checkpoint")
            if best["epoch"] == 10 and receipt["checkpoint_sha256"] != v3.PARENT_SHA256:
                raise ValueError("Wrong selected parent")
            reports[name].update(validation=rows[39], selection=receipt)
    return dict(status="PASS", stage=stage, reports=reports)


def train_one(model, optimizer, loader, device, generator, cfg, path, epoch, emit):
    validate_config(cfg)
    if path.exists():
        raise FileExistsError("Immutable training boundary already exists")
    loader.dataset.set_epoch(epoch-1)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    result = v3.v2.run_epoch(model, loader, optimizer, device,
        expected_patch_shape=(64, 64, 64),
        progress=lambda r: emit(dict(r, human_epoch=epoch,
                                     global_step=(epoch-1)*143+r["step"])))
    result["validation_pending"] = True
    if device.type == "cuda":
        result.update(peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(device),
                      peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(device))
    validate_config(cfg)
    v3.base.atomic_checkpoint(path, model, optimizer, completed_epochs=epoch,
        total_steps=epoch*143, config=cfg, generator=generator, device=device,
        epoch_result=result)
    emit(dict(event="checkpoint", path=str(path), sha256=v3.sha256(path),
              completed_epochs=epoch, total_steps=epoch*143, **result))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("A", "B", "C"), required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--parent-checkpoint", type=Path, required=True)
    p.add_argument("--v2-validation-log", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, default=v3.DEFAULT_DATASET_ROOT)
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--audit-only", action="store_true")
    args = p.parse_args(argv)
    parent = args.parent_checkpoint.expanduser().resolve(strict=True)
    root = args.root.expanduser().resolve(strict=True)
    if args.audit_only:
        if args.preflight_only:
            p.error("Choose only one read-only mode")
        if v3.sha256(parent) != v3.PARENT_SHA256 or v3.sha256(args.v2_validation_log) != v3.VALIDATION_SHA256:
            raise ValueError("Protected evidence hash mismatch")
        print(json.dumps(audit_complete(root, args.stage), allow_nan=False))
        return
    if root.is_relative_to(parent.parent):
        raise ValueError("Split outputs must be outside V2")
    v3.base.checkpoint_path(root / "guard.pt", args.dataset_root)
    model, parent_cfg = v3.read_parent(parent)
    from .patch_dataset import HipMRIPatchDataset
    dataset = HipMRIPatchDataset("train", args.dataset_root)
    v3.runtime_contract(parent_cfg, dataset, args.dataset_root)
    pairs = v3.evaluation.select_validation(v3.discover_dataset(args.dataset_root))
    reference = v3.authenticate_validation(args.v2_validation_log, pairs, parent_cfg)
    cfg = config_for(parent_cfg, pairs)
    validate_config(cfg)
    if args.preflight_only:
        print(json.dumps(dict(status="PASS", config=cfg, arrays_decoded=0)))
        return
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("Allocated Slurm CUDA node required")
    device = torch.device("cuda:0")
    v3.runtime_contract(parent_cfg, dataset, args.dataset_root, device)
    stage_dir = root / args.stage
    # Never reuse a failed/partial stage directory.
    stage_dir.mkdir()
    lock = root / "active.lock"
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(fd)
    journal = None
    try:
        inputs = {}
        if args.stage in ("B", "C"):
            for previous in (["A"] if args.stage == "B" else ["A", "B"]):
                _, digest = audit_stage(root, previous, cfg)
                inputs[previous] = digest
        v3.base.configure_reproducibility(3710, device)
        generator = torch.Generator().manual_seed(3710)
        loader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=True,
                            generator=generator, drop_last=False)
        model = model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.00001,
                                      betas=(.9,.999), eps=1e-8, foreach=False, fused=False)
        if args.stage == "A":
            completed, steps = v3.v2.resume_checkpoint(parent, model, optimizer,
                config=parent_cfg, generator=generator, device=device)
        else:
            previous = "A" if args.stage == "B" else "B"
            completed, steps = restore_boundary(checkpoint_path(root, previous),
                11 if previous == "A" else 12, model, optimizer, cfg, generator, device)
        log = stage_dir / f"events_{os.environ['SLURM_JOB_ID']}_{uuid.uuid4().hex}.jsonl"
        journal = v3.Journal(log)
        journal.emit(dict(event="setup", stage=args.stage, config=cfg,
                          starting_completed_epochs=completed, starting_total_steps=steps,
                          input_journals=inputs, gpu=torch.cuda.get_device_name(device)))
        if args.stage in ("A", "B"):
            train_one(model, optimizer, loader, device, generator, cfg,
                      checkpoint_path(root, args.stage), completed+1, journal.emit)
        else:
            summary = v3.full_validation(model, pairs, device, generator, journal.emit, 12)
            baseline = {"epoch": 10, "score": sum(
                reference["overall_patient_macro"][c]["dice"] for c in (2,4,5))/3}
            selected, decision = v3.selection(summary, baseline, 12)
            chosen = parent if selected["epoch"] == 10 else checkpoint_path(root, "B")
            # Separate evidence receipt; never rewrite the pre-validation checkpoint.
            receipt = dict(selected, checkpoint=str(chosen), checkpoint_sha256=v3.sha256(chosen),
                decision=decision, config_sha256=cfg["config_sha256"],
                validation_completed=True, test_evaluated=False)
            v3.atomic_json(stage_dir / "selection.json", receipt)
            journal.emit(dict(receipt, event="selection"))
        validate_config(cfg)
        if v3.sha256(parent) != v3.PARENT_SHA256:
            raise RuntimeError("V2 parent changed")
        for previous, digest in inputs.items():
            if audit_stage(root, previous, cfg)[1] != digest:
                raise RuntimeError("Input journal/checkpoint changed")
        journal.emit(dict(event="result", status="PASS", stage=args.stage,
                          parent_unchanged=True, test_arrays_decoded=0))
        v3.audit_journal(log)
        print(f"PASS: stage {args.stage}; journal={log}; sha256={v3.sha256(log)}")
    except Exception as exc:
        if journal:
            journal.emit(dict(event="error", status="FAIL", message=str(exc)))
        raise
    finally:
        if journal:
            journal.close()
        lock.unlink()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)
