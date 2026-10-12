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

ID = "v4_prostate_fp_tversky_paired_v1"
PARENT_SHA = "bb0d03c7a047ed187fadb5e1bf6ac1776448b351621764e9b6f0ff1fb67660d4"
RECEIPT_SHA = "fffe08a6c6502989e1ad033a0b78ea5117377dbad2de762791a3a895b2e189da"
LOCAL_FILES = ("experiment_prostate_fp.py", "prostate_fp_config.json", "test_prostate_fp.py")


def settings():
    cfg = json.loads(Path(__file__).with_name("prostate_fp_config.json").read_text())
    expected = dict(experiment_id=ID, parent_sha256=PARENT_SHA, receipt_sha256=RECEIPT_SHA,
        parent_epoch=12, end_epoch=14, steps_per_epoch=143, alpha_control=.5,
        alpha_treatment=.7, changed_class=5, epsilon=1e-5)
    if any(cfg.get(k) != x for k,x in expected.items()):
        raise ValueError("Not the approved one-factor experiment")
    if cfg.get("selection") != dict(prostate_patient_macro_gain_vs_control=.03,
            prostate_fp_reduction_vs_control=.15,maximum_prostate_recall_drop_vs_control=.05,
            maximum_other_class_patient_macro_drop=.02,minimum_nonregressing_prostate_patients=4,
            require_tp_all_volumes_classes=[2,3,4,5],require_prostate_patient_macro_above_v3=True,
            require_minority_patient_macro_not_below_v3=True):
        raise ValueError("Frozen candidate-selection rules changed")
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
    if (payload.get("config") != cfg or payload.get("completed_epochs") not in (13,14)
            or payload.get("epoch_result",{}).get("validation_pending") is not True
            or payload.get("epoch_result",{}).get("expected_steps") != 143
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
    return steps


def audit_arm(directory, cfg, pairs):
    validate_config(cfg)
    steps=[]
    for epoch in (13,14):
        path=directory/f"epoch{epoch}_training.pt"
        steps+=checkpoint_evidence(directory,path,cfg)
        payload=torch.load(path,weights_only=True,map_location="cpu")
        if (payload["config"]!=cfg or payload["completed_epochs"]!=epoch or payload["total_steps"]!=epoch*143
                or not payload["epoch_result"]["epoch_completed"] or payload["epoch_result"]["processed_steps"]!=143
                or not payload["optimizer"]["state"] or not {"python","numpy","torch_cpu","torch_cuda","loader"}<=payload["rng"].keys()):
            raise ValueError("Checkpoint boundary mismatch")
        del payload
    successful=[]
    for log in directory.glob("events_*.jsonl"):
        rows=v3.audit_journal(log)
        if rows[0].get("config")!=cfg: raise ValueError("Wrong arm/source")
        if rows[-1].get("event")=="result" and rows[-1].get("status")=="PASS":
            if (rows[-1].get("completed_epochs")!=14 or rows[-1].get("total_steps")!=2002
                    or rows[-1].get("parent_unchanged") is not True): raise ValueError("Wrong final state")
            successful.append(rows)
    if len(successful)!=1: raise ValueError("Expected one successful full Validation")
    rows=successful[0]
    volumes=[r for r in rows if r["event"]=="volume"]
    v3.validate_records(volumes,pairs)
    summary=v3.enrich_summary(volumes)
    logged=[r for r in rows if r["event"]=="validation_summary"]
    if len(logged)!=1 or any(logged[0].get(k)!=v for k,v in summary.items()):
        raise ValueError("Validation arithmetic mismatch")
    return summary,steps


def compare(root, parent_cfg, pairs, baseline):
    control,cs=audit_arm(root/"control",configuration(parent_cfg,"control"),pairs)
    treatment,ts=audit_arm(root/"treatment",configuration(parent_cfg,"treatment"),pairs)
    keys=("human_epoch","global_step","patient_id","week","crop_origin_dhw","sampling_mode")
    if [[r[k] for k in keys] for r in cs]!=[[r[k] for k in keys] for r in ts]:
        raise ValueError("Paired crop/data-order mismatch; scientific comparison rejected")
    return dict(event="paired_comparison",baseline=baseline,control=control,treatment=treatment,
                decision=decision(control,treatment,baseline),crop_order_parity=True,
                held_out_test_accessed=False)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",choices=("preflight","diagnostic","run","compare"),required=True)
    parser.add_argument("--arm",choices=("control","treatment"))
    parser.add_argument("--v3-root",type=Path,required=True)
    parser.add_argument("--output-root",type=Path,required=True)
    parser.add_argument("--resume",type=Path)
    args=parser.parse_args(argv)
    if args.mode in ("diagnostic","run") and args.arm is None: parser.error("Compute requires --arm")
    if args.resume and args.mode!="run": parser.error("Resume requires run mode")
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
    if args.resume:
        directory=directory.resolve(strict=True)
        resume=args.resume.resolve(strict=True)
        if resume.parent!=directory or resume.name not in ("epoch13_training.pt","epoch14_training.pt"):
            raise ValueError("Resume must be this arm's own immutable boundary")
        checkpoint_evidence(directory,resume,cfg)
        for previous in directory.glob("events_*.jsonl"):
            final=v3.audit_journal(previous)[-1]
            if final.get("event")=="result" and final.get("status")=="PASS":
                raise ValueError("Arm already completed; repeat Validation forbidden")
    else:
        directory.mkdir()  # Refuse any previous attempt, including failed diagnostics.
        resume=parent
    lock=output/"active.lock"
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600); os.close(fd)
    journal=None
    try:
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
        for epoch in range(completed+1,15):
            validate_config(cfg)
            target=directory/f"epoch{epoch}_training.pt"
            if target.exists(): raise FileExistsError("Immutable epoch already exists")
            ds.set_epoch(epoch-1)
            result=run_epoch(model,loader,opt,device,args.arm,epoch,journal.emit,
                             diagnostic=args.mode=="diagnostic")
            if args.mode=="diagnostic":
                journal.emit(dict(result,event="diagnostic",checkpoint_saved=False)); break
            validate_config(cfg)
            v3.base.atomic_checkpoint(target,model,opt,completed_epochs=epoch,total_steps=epoch*143,
                config=cfg,generator=generator,device=device,epoch_result=result)
            journal.emit(dict(result,event="checkpoint",completed_epochs=epoch,total_steps=epoch*143,
                              path=str(target),sha256=v3.sha256(target)))
            completed,total=epoch,epoch*143
        if args.mode=="run":
            v3.full_validation(model,pairs,device,generator,journal.emit,14)
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
        lock.unlink()


if __name__=="__main__":
    try: main()
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}",file=sys.stderr,flush=True)
        sys.exit(1)
