"""One-factor paired V3-E12 continuation: Prostate FP-weighted Tversky.

No historical files are modified. Manifest-only dataset construction avoids even
Test filename discovery. Identical production __getitem__/set_epoch are inherited.
Control uses the original production loss directly. Treatment changes class 5 only.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from time import perf_counter
import uuid

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import run_improved_v3_split as split
from . import train_improved_v3 as v3
from .dataset import APPROVED_PATIENTS, VolumePair, parse_volume_filename
from .patch_dataset import HipMRIPatchDataset, _config as patch_config
from .improved_modules import ImprovedUNet3D

ID = "v4_prostate_fp_tversky_paired_e20_v1"
PARENT_SHA = "bb0d03c7a047ed187fadb5e1bf6ac1776448b351621764e9b6f0ff1fb67660d4"
RECEIPT_SHA = "fffe08a6c6502989e1ad033a0b78ea5117377dbad2de762791a3a895b2e189da"
LOCAL_FILES = ("experiment_prostate_fp_e20.py", "prostate_fp_e20_config.json", "test_prostate_fp_e20.py")


def settings():
    cfg = json.loads(Path(__file__).with_name("prostate_fp_e20_config.json").read_text())
    expected = dict(experiment_id=ID, parent_sha256=PARENT_SHA, receipt_sha256=RECEIPT_SHA,
        parent_epoch=12, end_epoch=20, steps_per_epoch=143, alpha_control=.5,
        alpha_treatment=.7, changed_class=5, epsilon=1e-5)
    if any(cfg.get(k) != x for k,x in expected.items()):
        raise ValueError("Not the approved one-factor experiment")
    if cfg.get("selection") != dict(prostate_patient_macro_gain_vs_control=.03,
            prostate_fp_reduction_vs_control=.15,maximum_prostate_recall_drop_vs_control=.05,
            maximum_other_class_patient_macro_drop=.02,minimum_nonregressing_prostate_patients=4,
            require_tp_all_volumes_classes=[2,3,4,5],require_prostate_patient_macro_above_v3=True,
            require_minority_patient_macro_not_below_v3=True):
        raise ValueError("Frozen candidate-selection rules changed")
    if (cfg.get("validation_epochs") != [14, 16, 18, 20]
            or cfg.get("primary_comparison_epoch") != 20
            or cfg.get("max_epochs_per_job") != 1):
        raise ValueError("Frozen E20 execution/validation schedule changed")
    return cfg


def authenticate_parent(root):
    root = Path(root).resolve(strict=True)
    receipt_path = root / "C" / "selection.json"
    parent = split.checkpoint_path(root, "B")
    if v3.sha256(parent) != PARENT_SHA or v3.sha256(receipt_path) != RECEIPT_SHA:
        raise ValueError("Selected V3 checkpoint/receipt SHA-256 mismatch")
    receipt = json.loads(receipt_path.read_text())
    if (receipt.get("epoch") != 12 or Path(receipt["checkpoint"]).resolve() != parent
            or receipt.get("checkpoint_sha256") != PARENT_SHA):
        raise ValueError("Receipt does not select this V3 E12 checkpoint")
    # Existing source, checkpoint lineage, confusion and selection protections.
    audited = split.audit_complete(root, "C")
    payload = torch.load(parent, map_location="cpu", weights_only=True)
    split.validate_config(payload["config"])
    if payload["completed_epochs"] != 12 or payload["total_steps"] != 1716:
        raise ValueError("Expected E12 / 1716")
    if v3.sha256(parent) != PARENT_SHA or v3.sha256(receipt_path) != RECEIPT_SHA:
        raise ValueError("Protected input changed while reading")
    return parent, payload["config"], audited["reports"]["C"]["validation"]


def manifest_pairs(rows, partition, root, *, check_files=True):
    """Validate identities BEFORE resolving/stat'ing paths; never scan directories."""
    if partition not in ("train", "validation"):
        raise ValueError("Only Train/Validation manifests are permitted")
    allowed = set(APPROVED_PATIENTS[partition])
    expected = 143 if partition == "train" else 38
    if (len(rows) != expected or any(len(r) != 4 or r[0] not in allowed for r in rows)
            or len({(r[0],r[1]) for r in rows}) != expected
            or {r[0] for r in rows} != allowed):
        raise ValueError("Frozen partition identity/count mismatch")
    result = []
    root = Path(root).resolve()
    for patient, week, mri, mask in rows:
        if type(week) is not int or week < 0:
            raise ValueError("Invalid Week")
        for value, folder, kind in ((mri,"semantic_MRs","LFOV"),(mask,"semantic_labels_only","SEMANTIC")):
            path = Path(value)
            if (parse_volume_filename(path, kind) != (patient,week)
                    or path.parent != root/folder or path.resolve().parent != root/folder):
                raise ValueError("Manifest filename/path identity mismatch")
            if check_files and not path.is_file():
                raise FileNotFoundError(path)
        result.append(VolumePair(patient,week,Path(mri),Path(mask)))
    return tuple(result)


class ManifestPatchDataset(HipMRIPatchDataset):
    """Only initialization differs: use protected Train manifest, never discovery."""
    def __init__(self, parent_cfg, root, *, check_files=True):
        settings = patch_config()
        if settings != parent_cfg["policies"]["patch_dataset"]:
            raise ValueError("Patch policy changed")
        self.pairs = manifest_pairs(parent_cfg["train_pairs"], "train", root, check_files=check_files)
        self.patch_shape_dhw = tuple(settings["patch_shape_dhw"])
        self.train_patches_per_volume = settings["train_patches_per_volume"]
        self.train_foreground_probability = settings["train_foreground_probability"]
        self.seed, self.epoch, self.split = settings["seed"], 0, "train"
        spatial = parent_cfg["policies"]["resampling"]
        self.target_spacing_mm = tuple(spatial["target_spacing_mm"])
        self.max_estimated_bytes = spatial["max_estimated_bytes"]


def head_loss(logits, target, alpha):
    total, ce, dice, ordinary = v3.v2.head_loss(logits, target)
    if alpha == .5:
        return total, ce, dice, ordinary, dice
    if alpha != .7:
        raise ValueError("Only predeclared alpha=.5 or .7 permitted")
    p = logits.softmax(1)[:,5]
    y = (target == 5).to(logits.dtype)
    tp = (p*y).sum()
    fp = (p*(1-y)).sum()
    fn = ((1-p)*y).sum()
    eps = 1e-5
    original = (2*tp+eps)/(p.sum()+y.sum()+eps)
    tversky = (tp+eps/2)/(tp+alpha*fp+(1-alpha)*fn+eps/2)
    overlap = dice + (original-tversky)/5
    total = .5*ce + .5*overlap
    if not bool(torch.isfinite(total)):
        raise ValueError("Non-finite experimental loss")
    return total, ce, overlap, ordinary, dice


def loss(outputs, target, arm):
    if arm not in ("control", "treatment"):
        raise ValueError("Unknown arm")
    if arm == "control":
        values = v3.v2.combined_loss(outputs,target)
        return (*values, values[2])
    main, aux = outputs
    if len(aux) != 2 or any(tuple(a.shape) != (main.shape[0],6,*(s//scale for s in main.shape[2:]))
                            for a,scale in zip(aux,(2,4))):
        raise ValueError("Expected half/quarter auxiliary logits")
    terms = [head_loss(main,target,.7)]
    for a in aux:
        terms.append(head_loss(torch.nn.functional.interpolate(a,size=main.shape[2:],
                            mode="trilinear",align_corners=False),target,.7))
    return tuple(sum(w*t[i] for w,t in zip((1.,.5,.25),terms))/1.75 for i in range(5))


def local_hashes():
    return {n:v3.sha256(Path(__file__).with_name(n)) for n in LOCAL_FILES}


def configuration(parent_cfg, arm):
    cfg = dict(experiment_id=ID, arm=arm, parent_config=parent_cfg, settings=settings(),
               source_sha256=local_hashes(), steps_per_epoch=143)
    cfg["config_sha256"] = v3.v2.config_hash(cfg)
    return cfg


def validate_config(cfg):
    split.validate_config(cfg["parent_config"])
    if cfg != configuration(cfg["parent_config"],cfg["arm"]) or cfg["arm"] not in ("control","treatment"):
        raise ValueError("Experiment/config/source/arm mismatch")


def restore(path, model, opt, cfg, generator, device, *, parent=False):
    validate_config(cfg)
    if parent:
        if v3.sha256(path) != PARENT_SHA:
            raise ValueError("Parent checksum mismatch")
        return split.restore_boundary(path,12,model,opt,cfg["parent_config"],generator,device)
    payload = torch.load(path,weights_only=True,map_location="cpu")
    if (payload.get("config") != cfg or payload.get("completed_epochs") not in range(13,20)
            or payload.get("epoch_result",{}).get("validation_pending") is not True
            or payload.get("epoch_result",{}).get("expected_steps") != 143
            or payload.get("epoch_result",{}).get("epoch_completed") is not True
            or payload.get("epoch_result",{}).get("processed_steps") != 143
            or payload.get("total_steps") != payload.get("completed_epochs",-1)*143
            or not payload.get("optimizer",{}).get("state")
            or not {"python","numpy","torch_cpu","torch_cuda","loader"} <= payload.get("rng",{}).keys()):
        raise ValueError("Invalid experimental resume boundary")
    return v3.base.resume_checkpoint(path,model,opt,config=cfg,generator=generator,device=device)


def run_epoch(model, loader, opt, device, arm, epoch, emit, *, diagnostic=False, patch_shape=(64,64,64)):
    model.train(); started = perf_counter(); sums = np.zeros(5); count=0
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
    for batch in loader:
        image,target = batch["mri"],batch["mask"]
        if (image.device.type != "cpu" or target.device.type != "cpu" or image.dtype != torch.float32
                or target.dtype != torch.long or tuple(image.shape) != (1,1,*patch_shape)
                or tuple(target.shape) != (1,*patch_shape) or not bool(torch.isfinite(image).all())):
            raise ValueError("Expected aligned finite float32/long CPU patches")
        before = [p.detach().clone() for p in model.parameters()] if diagnostic else None
        opt.zero_grad(set_to_none=True)
        terms = loss(model(image.to(device),deep_supervision=True),target.to(device),arm)
        terms[0].backward()
        for p in model.parameters():
            if p.requires_grad and (p.grad is None or not bool(torch.isfinite(p.grad).all())):
                raise RuntimeError("Missing/non-finite gradient")
        opt.step()
        if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
            raise RuntimeError("Non-finite parameter")
        vals = [x.detach().item() for x in terms]; sums += vals; count+=1
        emit(dict(event="step",human_epoch=epoch,step=count,global_step=(epoch-1)*143+count,
            experiment_id=ID,arm=arm,
            learning_rate=float(opt.param_groups[0]["lr"]),
            weight_decay=float(opt.param_groups[0]["weight_decay"]),
            adamw_betas=list(opt.param_groups[0]["betas"]),
            tversky_alpha=(.5 if arm=="control" else .7),
            loss=vals[0],weighted_ce=vals[1],overlap_loss=vals[2],ordinary_ce=vals[3],original_dice_loss=vals[4],
            patient_id=batch["patient_id"][0],week=int(batch["week"][0]),
            crop_origin_dhw=batch["crop_origin_dhw"][0].tolist(),sampling_mode=batch["sampling_mode"][0]))
        if diagnostic:
            changed = sum(not torch.equal(a,b.detach()) for a,b in zip(before,model.parameters()))
            if not changed: raise RuntimeError("No parameter changed")
            break
    if not count or (not diagnostic and count != len(loader)):
        raise RuntimeError("Incomplete training")
    if device.type == "cuda": torch.cuda.synchronize(device)
    out = dict(processed_steps=count,expected_steps=len(loader),epoch_completed=not diagnostic,
        mean_loss=float(sums[0]/count),mean_weighted_ce=float(sums[1]/count),
        mean_overlap_loss=float(sums[2]/count),mean_original_dice_loss=float(sums[4]/count),
        runtime_seconds=perf_counter()-started,validation_pending=True)
    if diagnostic: out.update(changed_parameter_tensors=changed,finite_gradient_parameter_tensors=sum(p.requires_grad for p in model.parameters()))
    if device.type == "cuda": out.update(peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(device),
        peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(device))
    return out


def decision(control, treatment, baseline):
    rule = settings()["selection"]; reasons=[]
    pm=lambda s,c:s["overall_patient_macro"][c]["dice"]
    for s in (control,treatment,baseline):
        if s["successful"] != 38 or s["failed"] or any(pm(s,c) is None or not math.isfinite(pm(s,c)) for c in range(6)):
            raise ValueError("Incomplete evaluation")
    if pm(treatment,5)-pm(control,5) < rule["prostate_patient_macro_gain_vs_control"]: reasons.append("prostate_gain")
    cp,tp=control["overall_pooled"][5],treatment["overall_pooled"][5]
    if cp["false_positives"] <= 0 or tp["false_positives"] > .85*cp["false_positives"]: reasons.append("fp_reduction")
    if tp["recall"] < cp["recall"]-.05: reasons.append("recall_regression")
    for c in range(5):
        if pm(treatment,c)<pm(control,c)-.02 or pm(treatment,c)<pm(baseline,c)-.02:
            reasons.append(f"class_{c}_regression")
    for c in (2,3,4,5):
        if treatment["overall_pooled"][c]["tp_positive_volumes"] != 38: reasons.append(f"class_{c}_coverage")
    cm={r["patient_id"]:r["metrics"][5]["dice"] for r in control["patients"]}
    tm={r["patient_id"]:r["metrics"][5]["dice"] for r in treatment["patients"]}
    if set(cm)!=set(APPROVED_PATIENTS["validation"]) or cm.keys()!=tm.keys(): raise ValueError("Wrong patients")
    if sum(tm[p]>=cm[p] for p in cm)<4: reasons.append("patient_regressions")
    if pm(treatment,5)<=pm(baseline,5): reasons.append("not_above_v3_prostate")
    if sum(pm(treatment,c) for c in (2,4,5))<sum(pm(baseline,c) for c in (2,4,5)):
        reasons.append("minority_regression_vs_v3")
    return dict(accepted=not reasons,reasons=reasons,selected="treatment" if not reasons else "protected_v3_e12",
                interpretation="exploratory development selection; not independent confirmation")


def checkpoint_evidence(directory, path, cfg):
    found=[]
    for log in directory.glob("events_*.jsonl"):
        rows=v3.audit_journal(log)
        if rows[0].get("config") != cfg: raise ValueError("Journal configuration mismatch")
        for r in rows:
            if r.get("event")=="checkpoint" and Path(r["path"])==path:
                if v3.sha256(path)!=r["sha256"]: raise ValueError("Checkpoint bytes changed")
                found.append((r,rows))
    if len(found)!=1: raise ValueError("Missing/duplicate checkpoint provenance")
    r,rows=found[0]
    steps=[x for x in rows if x.get("event")=="step" and x["human_epoch"]==r["completed_epochs"]]
    e=r["completed_epochs"]
    if [x["global_step"] for x in steps]!=list(range((e-1)*143+1,e*143+1)):
        raise ValueError("Incomplete checkpoint training evidence")
    identities={(p,w) for p,w,_,_ in cfg["parent_config"]["protocol"]["parent_config"]["train_pairs"]}
    if {(x["patient_id"],x["week"]) for x in steps}!=identities:
        raise ValueError("Training epoch manifest coverage mismatch")
    if any(not math.isfinite(x[k]) for x in steps for k in ("loss","weighted_ce","overlap_loss")):
        raise ValueError("Non-finite recorded loss")
    expected_alpha = .5 if cfg["arm"] == "control" else .7
    for record in steps:
        if (record.get("experiment_id") != ID
                or record.get("arm") != cfg["arm"]
                or record.get("tversky_alpha") != expected_alpha
                or record.get("weight_decay") != .00001
                or record.get("adamw_betas") != [.9, .999]
                or not isinstance(record.get("learning_rate"), (int, float))
                or not math.isfinite(record["learning_rate"])
                or record["learning_rate"] <= 0):
            raise ValueError("Training-step hyperparameter evidence mismatch")

    return steps


def audit_arm(directory, cfg, pairs):
    """Audit all E13-E20 checkpoints and the four scheduled validations."""
    validate_config(cfg)

    expected_epochs = tuple(range(13, 21))
    validation_epochs = tuple(cfg["settings"]["validation_epochs"])

    if validation_epochs != (14, 16, 18, 20):
        raise ValueError("Unexpected validation schedule")

    completed_logs = {}

    for log in directory.glob("events_*.jsonl"):
        rows = v3.audit_journal(log)

        if not rows or rows[0].get("config") != cfg:
            raise ValueError("Journal configuration mismatch")

        final = rows[-1]

        if (final.get("event") == "result"
                and final.get("status") == "PASS"
                and final.get("checkpoint_saved") is True):
            epoch = final.get("completed_epochs")

            if epoch not in expected_epochs:
                raise ValueError("Unexpected completed epoch")

            completed_logs.setdefault(epoch, []).append(rows)

    if set(completed_logs) != set(expected_epochs):
        raise ValueError("Missing completed-epoch journals")

    if any(len(completed_logs[e]) != 1 for e in expected_epochs):
        raise ValueError("Duplicate successful epoch journals")

    all_steps = []
    validation_summaries = {}

    for epoch in expected_epochs:
        checkpoint = directory / f"epoch{epoch}_training.pt"

        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)

        all_steps.extend(checkpoint_evidence(directory, checkpoint, cfg))

        payload = torch.load(
            checkpoint, weights_only=True, map_location="cpu"
        )

        result = payload.get("epoch_result", {})

        if (payload.get("config") != cfg
                or payload.get("completed_epochs") != epoch
                or payload.get("total_steps") != epoch * 143
                or result.get("epoch_completed") is not True
                or result.get("processed_steps") != 143
                or result.get("expected_steps") != 143
                or result.get("validation_pending") is not True
                or not payload.get("optimizer", {}).get("state")
                or not {"python", "numpy", "torch_cpu",
                        "torch_cuda", "loader"} <=
                       payload.get("rng", {}).keys()):
            raise ValueError(f"E{epoch} checkpoint boundary mismatch")

        del payload

        rows = completed_logs[epoch][0]
        setup = rows[0]
        final = rows[-1]

        if (setup.get("starting_epoch") != epoch - 1
                or setup.get("starting_steps") != (epoch - 1) * 143
                or final.get("total_steps") != epoch * 143
                or final.get("parent_unchanged") is not True
                or final.get("test_paths_accessed") is not False):
            raise ValueError(f"E{epoch} journal boundary mismatch")

        volumes = [r for r in rows if r.get("event") == "volume"]
        logged = [
            r for r in rows
            if r.get("event") == "validation_summary"
        ]

        if epoch in validation_epochs:
            if any(r.get("completed_epochs") != epoch for r in volumes):
                raise ValueError("Validation epoch label mismatch")

            v3.validate_records(volumes, pairs)
            summary = v3.enrich_summary(volumes)

            if (len(logged) != 1
                    or any(logged[0].get(k) != value
                           for k, value in summary.items())):
                raise ValueError(
                    f"E{epoch} validation arithmetic mismatch"
                )

            validation_summaries[epoch] = summary

        elif volumes or logged:
            raise ValueError(
                f"Unexpected validation during E{epoch}"
            )

    if set(validation_summaries) != set(validation_epochs):
        raise ValueError("Incomplete scheduled validations")

    if len(all_steps) != 8 * 143:
        raise ValueError("Incomplete E13-E20 step evidence")

    return validation_summaries[20], all_steps


def compare(root, parent_cfg, pairs, baseline):
    """Compare matched E20 Validation results with complete paired evidence."""

    control, control_steps = audit_arm(
        root / "control",
        configuration(parent_cfg, "control"),
        pairs
    )
    treatment, treatment_steps = audit_arm(
        root / "treatment",
        configuration(parent_cfg, "treatment"),
        pairs
    )

    expected_epochs = [
        epoch
        for epoch in range(13, 21)
        for _ in range(143)
    ]
    expected_global_steps = list(range(1717, 2861))

    for arm, steps in (
        ("control", control_steps),
        ("treatment", treatment_steps)
    ):
        if len(steps) != 1144:
            raise ValueError(f"{arm}: expected exactly 1144 steps")

        if [r["human_epoch"] for r in steps] != expected_epochs:
            raise ValueError(f"{arm}: incorrect epoch sequence")

        if [r["global_step"] for r in steps] != expected_global_steps:
            raise ValueError(f"{arm}: incorrect global-step sequence")

        if [r["step"] for r in steps] != list(range(1, 144)) * 8:
            raise ValueError(f"{arm}: incorrect within-epoch steps")

    pairing_keys = (
        "human_epoch",
        "global_step",
        "patient_id",
        "week",
        "crop_origin_dhw",
        "sampling_mode"
    )

    for control_step, treatment_step in zip(
        control_steps, treatment_steps
    ):
        if any(
            control_step[key] != treatment_step[key]
            for key in pairing_keys
        ):
            raise ValueError(
                "Control/Treatment data-order or crop mismatch at "
                f"global step {control_step['global_step']}"
            )

    outcome = decision(control, treatment, baseline)

    return dict(
        event="paired_comparison",
        experiment_id=ID,
        primary_comparison_epoch=20,
        parent_epoch=12,
        paired_optimizer_updates_per_arm=1144,
        verified_global_step_range=[1717, 2860],
        crop_order_parity=True,
        baseline=baseline,
        control=control,
        treatment=treatment,
        decision=outcome,
        held_out_test_accessed=False,
        evidence_scope=(
            "exploratory Validation comparison; "
            "not independent confirmation"
        )
    )


def validate_failed_parent_retry(directory, cfg):
    """Permit explicit E13 replay only after intact, failed attempts."""

    directory = Path(directory)

    if not directory.is_dir() or directory.is_symlink():
        raise ValueError("Missing or unsafe failed-E13 directory")

    contents = list(directory.iterdir())

    logs = [
        p for p in contents
        if p.name.startswith("events_")
        and p.name.endswith(".jsonl")
        and p.is_file()
        and not p.is_symlink()
    ]

    if not logs or len(logs) != len(contents):
        raise ValueError(
            "Failed-E13 directory contains missing or unknown evidence"
        )

    for previous in logs:
        rows = v3.audit_journal(previous)

        if len(rows) < 2:
            raise ValueError("Incomplete failed-E13 journal")

        setup = rows[0]
        final = rows[-1]

        if (setup.get("event") != "setup"
                or setup.get("config") != cfg
                or setup.get("mode") != "run"
                or setup.get("starting_epoch") != 12
                or setup.get("starting_steps") != 1716
                or setup.get("parent_sha256") != PARENT_SHA
                or setup.get("resume_sha256") != PARENT_SHA
                or setup.get("test_paths_accessed") is not False):
            raise ValueError("Invalid failed-E13 journal identity")

        if (final.get("event") != "error"
                or final.get("status") != "FAIL"):
            raise ValueError(
                "Interrupted or ambiguous journal requires manual audit"
            )

        steps = rows[1:-1]

        if len(steps) > 143:
            raise ValueError("Too many failed-E13 steps")

        for number, record in enumerate(steps, start=1):
            if (record.get("event") != "step"
                    or record.get("human_epoch") != 13
                    or record.get("step") != number
                    or record.get("global_step") != 1716 + number):
                raise ValueError("Invalid failed-E13 step sequence")

    return len(logs)




def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",choices=("preflight","diagnostic","run","compare"),required=True)
    parser.add_argument("--arm",choices=("control","treatment"))
    parser.add_argument("--v3-root",type=Path,required=True)
    parser.add_argument("--output-root",type=Path,required=True)
    parser.add_argument("--resume",type=Path)
    parser.add_argument("--retry-failed-parent",action="store_true")
    args=parser.parse_args(argv)
    if args.mode in ("diagnostic","run") and args.arm is None: parser.error("Compute requires --arm")
    if args.resume and args.mode!="run": parser.error("Resume requires run mode")
    if args.retry_failed_parent and (args.mode!="run" or args.resume):
        parser.error("Failed-parent retry requires run mode without --resume")
    parent,parent_cfg,baseline=authenticate_parent(args.v3_root)
    original=parent_cfg["protocol"]["parent_config"]
    dataset_root=Path(original["dataset_root"])
    ds=ManifestPatchDataset(original,dataset_root)
    pairs=manifest_pairs(parent_cfg["protocol"]["validation_pairs"],"validation",dataset_root)
    v3.runtime_contract(original,ds,dataset_root)
    output=args.output_root.expanduser().resolve(strict=True)
    protected=args.v3_root.expanduser().resolve(strict=True)
    if output.is_relative_to(protected) or protected.is_relative_to(output):
        raise ValueError("Output root must be independent of historical V3 root")
    v3.base.checkpoint_path(output/"guard.pt",dataset_root)
    if args.mode=="preflight":
        print(json.dumps(dict(status="PASS",parent_sha256=PARENT_SHA,receipt_sha256=RECEIPT_SHA,
            configs={arm:configuration(parent_cfg,arm) for arm in ("control","treatment")},
            training_volumes=len(ds),validation_volumes=len(pairs),test_paths_accessed=False)))
        return
    if args.mode=="compare":
        print(json.dumps(compare(output,parent_cfg,pairs,baseline),allow_nan=False)); return
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("Allocated CUDA node required")
    device=torch.device("cuda:0"); v3.runtime_contract(original,ds,dataset_root,device)
    cfg=configuration(parent_cfg,args.arm); validate_config(cfg)
    directory=output/("diagnostic_"+args.arm if args.mode=="diagnostic" else args.arm)
    lock=output/"active.lock"
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    lock_identity=(os.fstat(fd).st_dev,os.fstat(fd).st_ino)
    os.close(fd)
    journal=None
    try:
        if args.resume:
            directory=directory.resolve(strict=True)
            resume=args.resume.resolve(strict=True)
            if resume.parent!=directory or resume.name not in tuple(f"epoch{e}_training.pt" for e in range(13,20)):
                raise ValueError("Resume must be this arm's own immutable boundary")
            checkpoint_evidence(directory,resume,cfg)
            resume_epoch = int(resume.name.split("_")[0][5:])
            if any((directory/f"epoch{e}_training.pt").exists()
                   for e in range(resume_epoch+1, 21)):
                raise ValueError("A newer checkpoint already exists")
            successful_boundaries = 0
            for previous in directory.glob("events_*.jsonl"):
                final = v3.audit_journal(previous)[-1]
                if final.get("event") == "result" and final.get("status") == "PASS":
                    if final.get("completed_epochs") == 20:
                        raise ValueError("Arm already completed E20")
                    if final.get("completed_epochs") == resume_epoch:
                        successful_boundaries += 1
            if successful_boundaries != 1:
                raise ValueError("Resume requires one successful completed-epoch journal")
        else:
            if args.retry_failed_parent:
                validate_failed_parent_retry(directory, cfg)
            else:
                directory.mkdir()
            resume=parent

        v3.base.configure_reproducibility(3710,device)
        generator=torch.Generator().manual_seed(3710)
        loader=DataLoader(ds,batch_size=1,num_workers=0,shuffle=True,generator=generator,drop_last=False)
        model=ImprovedUNet3D().to(device)
        opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.00001,
                             betas=(.9,.999),eps=1e-8,foreach=False,fused=False)
        completed,total=restore(resume,model,opt,cfg,generator,device,parent=not bool(args.resume))
        log=directory/f"events_{os.environ['SLURM_JOB_ID']}_{uuid.uuid4().hex}.jsonl"
        journal=v3.Journal(log)
        journal.emit(dict(event="setup",config=cfg,starting_epoch=completed,starting_steps=total,
            parent_sha256=PARENT_SHA,resume_sha256=v3.sha256(resume),gpu=torch.cuda.get_device_name(device),
            mode=args.mode,test_paths_accessed=False))
        for epoch in range(completed+1, min(completed+1+settings()["max_epochs_per_job"], settings()["end_epoch"]+1)):
            validate_config(cfg)
            target=directory/f"epoch{epoch}_training.pt"
            if target.exists(): raise FileExistsError("Immutable epoch already exists")
            ds.set_epoch(epoch-1)
            result=run_epoch(model,loader,opt,device,args.arm,epoch,journal.emit,
                             diagnostic=args.mode=="diagnostic")
            if args.mode=="diagnostic":
                journal.emit(dict(result,event="diagnostic",checkpoint_saved=False)); break
            if args.mode=="run" and epoch in settings()["validation_epochs"]:
                v3.full_validation(model,pairs,device,generator,journal.emit,epoch)
            validate_config(cfg)
            v3.base.atomic_checkpoint(target,model,opt,completed_epochs=epoch,total_steps=epoch*143,
                config=cfg,generator=generator,device=device,epoch_result=result)
            journal.emit(dict(result,event="checkpoint",completed_epochs=epoch,total_steps=epoch*143,
                              path=str(target),sha256=v3.sha256(target)))
            completed,total=epoch,epoch*143
        validate_config(cfg)
        if v3.sha256(parent)!=PARENT_SHA or v3.sha256(protected/"C"/"selection.json")!=RECEIPT_SHA:
            raise RuntimeError("Protected evidence changed")
        journal.emit(dict(event="result",status="PASS",completed_epochs=completed,total_steps=total,
            parent_unchanged=True,checkpoint_saved=args.mode=="run",test_paths_accessed=False,
            diagnostic_updates=1 if args.mode=="diagnostic" else 0))
        v3.audit_journal(log)
        print(f"PASS: {args.mode} {args.arm}; journal={log}; SHA256={v3.sha256(log)}")
    except Exception as exc:
        if journal:
            try: journal.emit(dict(event="error",status="FAIL",message=str(exc)))
            except Exception: pass
        raise
    finally:
        if journal: journal.close()
        try:
            lock_stat=os.stat(lock,follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            if (lock_stat.st_dev,lock_stat.st_ino)==lock_identity:
                lock.unlink()


if __name__=="__main__":
    try: main()
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}",file=sys.stderr,flush=True)
        sys.exit(1)
