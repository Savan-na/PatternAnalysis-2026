"""Train the supplementary frozen G5 2D arm with batch recovery.

Every completed optimizer batch atomically updates the resumable state. Each
completed epoch keeps its own weights and record; Test files are not decoded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import sys
from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "code"))
G1_SHA = "6a4c674564a0364f5f56730cbc31e67b3ff9a8ffeae7d9d3dd18b198bbd0e1a1"
G3_SHA = "319973a8759265521870f831e7b17eb7249f4130b6a9e9634fe993f22bf687e9"
CLASS_NAMES = ("Background", "Body", "Bones", "Bladder", "Rectum", "Prostate")
MODEL_ID = "baseline_2d"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_checkpoint(path: Path, state: dict) -> None:
    """Replace only after a complete checkpoint has been written."""
    import torch

    temporary = path.with_name(path.name + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def load_durable_state(run_dir: Path, map_location: str):
    """Prefer the newest completed epoch if the latest pointer was interrupted."""
    import torch

    latest = run_dir / "latest.pt"
    snapshots = sorted((path for path in run_dir.glob("epoch_*.pt")
                        if path.stem[6:].isdigit()),
                       key=lambda path: int(path.stem[6:]))
    if not latest.exists() and not snapshots:
        raise ValueError("No durable checkpoint exists for resume")
    checkpoint = (torch.load(latest, map_location=map_location, weights_only=False)
                  if latest.exists() else None)
    completed = (torch.load(snapshots[-1], map_location=map_location, weights_only=False)
                 if snapshots else None)
    if completed is not None and (checkpoint is None or completed["epoch"] > checkpoint["epoch"]):
        checkpoint = completed
    return checkpoint, completed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--check-config-only", action="store_true",
                        help="Validate frozen 2D sources on a CPU node without creating a run")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-epoch", type=int,
                        help="Stop after this completed epoch; resume later with the same config")
    parser.add_argument("--extend-to-epochs", type=int,
                        help="On resume only: increase the epoch horizon without changing training settings")
    parser.add_argument("--g1", type=Path,
                        default=PROJECT / "logs/audits/train_full_20261018_job671208.json")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or socket.gethostname().startswith("login"):
        parser.error("Run inside an allocated Slurm GPU compute job")
    run_dir = args.run_dir.resolve()
    if not run_dir.is_relative_to(PROJECT / "logs/training"):
        parser.error("Run directory must be inside 20261017/logs/training")
    if not args.check_config_only and args.resume != run_dir.exists():
        parser.error("New run directory must be absent; resume directory must exist")
    if sha256_file(args.g1) != G1_SHA:
        raise ValueError("G1 Train audit differs from verified source")

    import numpy as np
    import torch

    from hipmri_rewrite.data import downsample, load_case, zscore_nonzero
    from hipmri_rewrite.inference import predict_volume
    from hipmri_rewrite.losses import ce_dice_loss
    from hipmri_rewrite.metrics import confusion, overlap
    from hipmri_rewrite.models import UNet2D
    from hipmri_rewrite.sampling import (CLASS_NAMES as SAMPLING_NAMES,
                                        ExposureRecorder, epoch_cases)
    from hipmri_rewrite.sampling_2d import (ORGAN_AWARE_2D, PATCH_XY,
                                           crop_for_2d_model, slice_crop_origin)
    from hipmri_rewrite.splits import DEFAULT_SPLIT, discover
    from hipmri_rewrite.training import patient_small_organ_score

    config = json.loads(args.config.read_text(encoding="utf-8"))
    config_sha = sha256_file(args.config)
    audit_path = PROJECT / "logs/audits/g5_2d_sampling_full_20261018/manifest.json"
    audit_patches = audit_path.with_name("patches.csv")
    if sha256_file(audit_path) != config["sampling"]["audit_manifest_sha256"]:
        raise ValueError("2D Train-only exposure source changed")
    if sha256_file(audit_patches) != config["sampling"]["audit_patches_sha256"]:
        raise ValueError("2D Train-only patch records changed")
    exposure_audit = json.loads(audit_path.read_text(encoding="utf-8"))
    policy = config["sampling"]["patch_rule"]
    if policy != "patient_organ_aware_2d":
        raise ValueError("Unexpected 2D sampling policy")
    shares = exposure_audit["summary"][policy]["per_class"]
    body_share = shares["Body"]["voxel_share"]
    expected_weights = [round(min(10.0, float(np.sqrt(body_share / shares[name]["voxel_share"]))), 6)
                        for name in CLASS_NAMES]
    if (config["status"] != "frozen_for_G5_supplementary_2d_training"
            or sha256_file(DEFAULT_SPLIT) != config["split_sha256"]
            or not exposure_audit["complete_patient_epoch"]
            or exposure_audit["patients"] != 26
            or exposure_audit["patches_per_policy"] != 260
            or exposure_audit["seed"] != config["seed"]
            or exposure_audit["g1_sha256"] != G1_SHA
            or exposure_audit["g3_sha256"] != G3_SHA
            or exposure_audit["script_sha256"]
            != sha256_file(PROJECT / "code/tools/audit_g5_2d_sampling.py")
            or exposure_audit["sampling_2d_sha256"]
            != sha256_file(PROJECT / "code/hipmri_rewrite/sampling_2d.py")
            or exposure_audit["data_sha256"]
            != sha256_file(PROJECT / "code/hipmri_rewrite/data.py")
            or exposure_audit["policies"][policy] != list(ORGAN_AWARE_2D)
            or config["objective"]["class_order"] != list(CLASS_NAMES)
            or tuple(SAMPLING_NAMES) != CLASS_NAMES
            or config["objective"]["ce_class_weights"] != expected_weights
            or config["model"] != MODEL_ID
            or config["model_defaults"] != {"base_channels": 16}
            or config["sampling"]["visit_rule"]
            != "one Train visit per patient per epoch, cycling through shuffled visits"
            or config["sampling"]["spatial_augmentation"] != "none"
            or config["sampling"]["intensity_augmentation"] != "none"
            or config["preprocessing"]["downsample_factor_xyz"] != [2, 2, 2]
            or config["preprocessing"]["normalization"]
            != "per-volume MRI nonzero Z-score; retain zero outside"
            or config["preprocessing"]["mask_interpolation"] != "nearest-neighbour"
            or config["preprocessing"]["mri_interpolation"] != "linear"
            or config["preprocessing"]["train_patch_xy"] != list(PATCH_XY)
            or config["objective"]["ce_fraction"] != 0.5
            or config["sampling"]["batch_size"] != 2
            or config["optimizer"] != {"name": "Adam", "learning_rate": 0.001,
                                      "weight_decay": 0.0}
            or config["validation"]["grid"]
            != "preprocessed XYZ grid with model padding removed"
            or config["validation"]["patient_aggregation"]
            != "sum confusion matrices across visits within each patient"
            or config["validation"]["checkpoint_score"]
            != "mean of patient Dice for Bladder, Rectum and Prostate across six patients; absent/absent excluded"
            or config["validation"]["tie_break"] != "earlier epoch"
            or config["validation"]["test_access"]
            != "none before checkpoint selection and protocol freeze"):
        raise ValueError("Frozen 2D configuration or Train-only provenance mismatch")
    if (config["budget"]["updates_per_epoch"]
            != config["train_patients"] * config["sampling"]["patches_per_patient_epoch"] // 2
            or config["budget"]["total_planned_updates"]
            != config["budget"]["epochs"] * config["budget"]["updates_per_epoch"]):
        raise ValueError("G5 frozen update budget inconsistent")
    if config["budget"].get("checkpoint_every_batches") != 1:
        raise ValueError("G5 requires a durable checkpoint after every completed batch")
    if (config["validation"]["frequency_epochs"] < 1
            or config["budget"]["epochs"] < config["validation"]["frequency_epochs"]
            or any(not np.isfinite(weight) or weight <= 0
                   for weight in config["objective"]["ce_class_weights"])):
        raise ValueError("Invalid Validation interval, epoch budget or CE weight")
    if args.stop_after_epoch is not None and args.stop_after_epoch < 1:
        parser.error("Stop epoch must be positive")
    if args.extend_to_epochs is not None and (
            not args.resume or args.extend_to_epochs < config["budget"]["epochs"]):
        parser.error("Epoch extension requires resume and cannot reduce the frozen budget")
    if args.check_config_only:
        if args.resume or args.stop_after_epoch is not None or args.extend_to_epochs is not None:
            parser.error("Config check cannot resume or change an epoch limit")
        print("Frozen 2D config, split and Train-only audit provenance passed", flush=True)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA device allocated")

    groups = discover(args.dataset_root)
    train_cases, val_cases = groups["train"], groups["validation"]
    if (len(train_cases) != config["train_cases"]
            or len(val_cases) != config["validation_cases"]
            or len({case.patient for case in val_cases}) != config["validation_patients"]):
        raise ValueError("Frozen cohort case/patient counts changed")
    audit = {row["case"]: row for row in json.loads(args.g1.read_text(encoding="utf-8"))["cases"]}
    cache = {}
    def train_volume(case):
        if case.identifier not in cache:
            row = audit[case.identifier]
            if (sha256_file(case.mri) != row["mri_sha256"]
                    or sha256_file(case.mask) != row["mask_sha256"]):
                raise ValueError(f"{case.identifier}: source changed since G1")
            cache[case.identifier] = zscore_nonzero(downsample(load_case(case)))
        return cache[case.identifier]

    seed = config["seed"]
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    model_source = PROJECT / "code/hipmri_rewrite/models/baseline_2d.py"
    common_source = PROJECT / "code/hipmri_rewrite/models/common.py"
    model = UNet2D().to("cuda")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0)
    weights = torch.tensor(config["objective"]["ce_class_weights"],
                           dtype=torch.float32, device="cuda")

    source_files = {"trainer_sha256": Path(__file__),
                    "sampling_sha256": PROJECT / "code/hipmri_rewrite/sampling.py",
                    "sampling_2d_sha256": PROJECT / "code/hipmri_rewrite/sampling_2d.py",
                    "data_sha256": PROJECT / "code/hipmri_rewrite/data.py",
                    "loss_sha256": PROJECT / "code/hipmri_rewrite/losses.py",
                    "inference_sha256": PROJECT / "code/hipmri_rewrite/inference.py",
                    "selection_sha256": PROJECT / "code/hipmri_rewrite/training.py",
                    "metrics_sha256": PROJECT / "code/hipmri_rewrite/metrics.py",
                    "splits_sha256": PROJECT / "code/hipmri_rewrite/splits.py",
                    "package_sha256": PROJECT / "code/hipmri_rewrite/__init__.py",
                    "model_package_sha256": PROJECT / "code/hipmri_rewrite/models/__init__.py",
                    "model_source_sha256": model_source,
                    "model_common_sha256": common_source}

    def state_at(epoch: int, next_batch: int, accumulation: dict | None,
                 summary: dict | None = None) -> dict:
        return {"epoch": epoch, "next_batch": next_batch,
                "accumulation": accumulation, "summary": summary,
                "model": MODEL_ID, "config_sha256": config_sha,
                "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_states": torch.cuda.get_rng_state_all(),
                "numpy_rng_state": np.random.get_state(),
                "best_score": best_score, "best_epoch": best_epoch}

    if args.resume:
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        if (manifest["config_sha256"] != config_sha or manifest["model"] != MODEL_ID
                or manifest["dataset_root"] != str(args.dataset_root.resolve())):
            raise ValueError("Resume configuration/model differs from original run")
        if any(manifest[key] != sha256_file(path) for key, path in source_files.items()):
            raise ValueError("Resume code differs from original run")
        if (manifest["torch"] != str(torch.__version__)
                or manifest["numpy"] != np.__version__
                or manifest["python"] != sys.version):
            raise ValueError("Resume Python/PyTorch/NumPy version differs from original run")
        checkpoint, completed = load_durable_state(run_dir, "cuda")
        if checkpoint["config_sha256"] != config_sha or checkpoint["model"] != MODEL_ID:
            raise ValueError("Latest checkpoint configuration/model mismatch")
        if not 0 <= checkpoint["next_batch"] <= config["budget"]["updates_per_epoch"]:
            raise ValueError("Resume batch position is invalid")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng_states"]])
        np.random.set_state(checkpoint["numpy_rng_state"])
        start_epoch = checkpoint["epoch"]
        start_batch = checkpoint["next_batch"]
        resume_accumulation = checkpoint["accumulation"]
        best_score = checkpoint["best_score"]
        best_epoch = checkpoint["best_epoch"]
        planned_epochs = manifest["max_epochs"]
        if args.extend_to_epochs is not None and args.extend_to_epochs > planned_epochs:
            manifest["max_epochs"] = args.extend_to_epochs
            manifest.setdefault("extensions", []).append({
                "job_id": os.environ["SLURM_JOB_ID"], "from_epoch": start_epoch,
                "previous_max_epochs": planned_epochs,
                "new_max_epochs": args.extend_to_epochs,
                "reason": "explicit continuation with unchanged frozen training parameters"})
            atomic_json(run_dir / "manifest.json", manifest)
            planned_epochs = args.extend_to_epochs
        if completed is not None:
            summary_path = run_dir / f"epoch_{completed['epoch']:03d}.json"
            if not summary_path.exists():
                if completed["summary"] is None:
                    raise ValueError("Completed epoch snapshot has no summary")
                atomic_json(summary_path, completed["summary"])
        best_path = run_dir / "best.pt"
        best_file_epoch = (torch.load(best_path, map_location="cpu", weights_only=False)["epoch"]
                           if best_path.exists() else None)
        if best_epoch is not None and best_file_epoch != best_epoch:
            source = run_dir / f"epoch_{best_epoch:03d}.pt"
            if not source.exists():
                raise ValueError("Best Validation epoch snapshot is missing")
            shutil.copyfile(source, run_dir / "best.pt.tmp")
            os.replace(run_dir / "best.pt.tmp", best_path)
    else:
        run_dir.mkdir(parents=True)
        start_epoch, start_batch, resume_accumulation = 0, 0, None
        best_score, best_epoch = -1.0, None
        planned_epochs = config["budget"]["epochs"]
        atomic_json(run_dir / "manifest.json", {
            "kind": "G5_frozen_2D_Train_Validation_cohort_run", "model": MODEL_ID,
            "max_epochs": planned_epochs,
            "dataset_root": str(args.dataset_root.resolve()),
            "config_sha256": config_sha, "split_sha256": sha256_file(DEFAULT_SPLIT),
            "g1_sha256": G1_SHA, "seed": seed, "job_id_initial": os.environ["SLURM_JOB_ID"],
            "gpu": torch.cuda.get_device_name(0), "torch": str(torch.__version__),
            "numpy": np.__version__, "python": sys.version,
            **{key: sha256_file(path) for key, path in source_files.items()}})

    def evaluate() -> dict:
        model.eval()
        patient_matrices = defaultdict(lambda: np.zeros((6, 6), dtype=np.int64))
        case_rows = []
        start = perf_counter()
        torch.cuda.reset_peak_memory_stats()
        for case in val_cases:
            volume = zscore_nonzero(downsample(load_case(case)))
            prediction = predict_volume(model, volume, mode="2d", device="cuda")
            matrix = confusion(volume.mask, prediction)
            patient_matrices[case.patient] += matrix
            case_rows.append({"case": case.identifier, "patient": case.patient,
                              "confusion": matrix.tolist()})
        score, patient_dice = patient_small_organ_score(dict(patient_matrices))
        pooled = sum(patient_matrices.values(), np.zeros((6, 6), dtype=np.int64))
        result = {"selection_score": score, "patient_small_organ_dice": patient_dice,
                  "pooled_confusion": pooled.tolist(), "pooled_overlap": overlap(pooled),
                  "cases": case_rows, "seconds": perf_counter() - start,
                  "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                  "peak_reserved_bytes": int(torch.cuda.max_memory_reserved())}
        model.train()
        return result

    final_epoch = args.stop_after_epoch or planned_epochs
    if final_epoch > planned_epochs:
        parser.error("Stop epoch exceeds the recorded training horizon")
    if final_epoch <= start_epoch:
        parser.error("Stop epoch must exceed the last completed epoch")
    for epoch in range(start_epoch, final_epoch):
        model.train()
        chosen = epoch_cases(train_cases, epoch=epoch, seed=seed)
        descriptors = [(case, draw, target) for case in chosen
                       for draw, target in enumerate(ORGAN_AWARE_2D)]
        order_seed = int.from_bytes(hashlib.sha256(f"{seed}|{epoch}|batches".encode()).digest()[:8],
                                    "little")
        order = np.random.default_rng(order_seed).permutation(len(descriptors))
        descriptors = [descriptors[int(index)] for index in order]
        if len(descriptors) != config["budget"]["updates_per_epoch"] * 2:
            raise ValueError("Epoch patch count differs from frozen budget")
        first_batch = start_batch if epoch == start_epoch else 0
        prior = resume_accumulation if epoch == start_epoch else None
        recorder = ExposureRecorder()
        if prior is not None:
            if len(prior["batch_history"]) != first_batch:
                raise ValueError("Saved batch history does not match resume position")
            previous_exposure = prior["exposure"]
            recorder.patches = previous_exposure["patches"]
            recorder.hits = Counter({name: row["patch_hits"]
                                     for name, row in previous_exposure["per_class"].items()})
            recorder.voxels = Counter({name: row["patch_voxels"]
                                       for name, row in previous_exposure["per_class"].items()})
            recorder.by_patient = defaultdict(Counter, {
                patient: Counter(row) for patient, row in previous_exposure["per_patient"].items()})
        train_matrix = (np.asarray(prior["train_matrix"], dtype=np.int64)
                        if prior is not None else np.zeros((6, 6), dtype=np.int64))
        batch_history = list(prior["batch_history"]) if prior is not None else []
        source_load_seconds = prior["source_load_seconds"] if prior is not None else 0.0
        train_seconds = prior["train_seconds"] if prior is not None else 0.0
        checkpoint_io_seconds = prior["checkpoint_io_seconds"] if prior is not None else 0.0
        train_peak_allocated = prior["peak_allocated_bytes"] if prior is not None else 0
        train_peak_reserved = prior["peak_reserved_bytes"] if prior is not None else 0
        torch.cuda.reset_peak_memory_stats()
        for batch_index in range(first_batch, len(descriptors) // 2):
            started_batch = perf_counter()
            start = batch_index * 2
            image_arrays, mask_arrays = [], []
            delivered_counts = Counter()
            for case, draw, target in descriptors[start:start + 2]:
                load_start = perf_counter()
                volume = train_volume(case)
                source_load_seconds += perf_counter() - load_start
                origin = slice_crop_origin(volume.mask, target, seed=seed,
                                           epoch=epoch, case_id=case.identifier, draw=draw)
                image, mask = crop_for_2d_model(volume, origin)
                delivered = recorder.add(mask[None], case.patient)
                if target != "uniform" and delivered[target] == 0:
                    raise ValueError(f"2D target missing from delivered crop: {case.identifier}")
                delivered_counts.update(delivered)
                image_arrays.append(image)
                mask_arrays.append(mask)
            x = torch.from_numpy(np.ascontiguousarray(np.stack(image_arrays))).to("cuda")
            y = torch.from_numpy(np.ascontiguousarray(np.stack(mask_arrays))).to("cuda")
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = ce_dice_loss(logits, y, ce_weight=0.5, class_weights=weights)
            if not bool(torch.isfinite(loss).item()):
                raise ValueError(f"Nonfinite Train loss at epoch {epoch}, batch {start // 2}")
            loss.backward()
            gradients = [param.grad for param in model.parameters()]
            if any(grad is None or not bool(torch.isfinite(grad).all().item())
                   for grad in gradients):
                raise ValueError(f"Missing/nonfinite Train gradient at epoch {epoch}")
            norm = torch.linalg.vector_norm(torch.stack([grad.norm() for grad in gradients]))
            optimizer.step()
            with torch.no_grad():
                encoded = (6 * y + logits.argmax(dim=1)).reshape(-1)
                matrix = torch.bincount(encoded, minlength=36).reshape(6, 6)
            train_matrix += matrix.cpu().numpy()
            loss_value, norm_value = float(loss.item()), float(norm.item())
            torch.cuda.synchronize()
            train_seconds += perf_counter() - started_batch
            train_peak_allocated = max(train_peak_allocated, int(torch.cuda.max_memory_allocated()))
            train_peak_reserved = max(train_peak_reserved, int(torch.cuda.max_memory_reserved()))
            batch_history.append({"batch": batch_index + 1, "loss": loss_value,
                                  "gradient_norm_l2": norm_value,
                                  "delivered_voxels": dict(delivered_counts)})
            accumulation = {"batch_history": batch_history,
                            "train_matrix": train_matrix.tolist(),
                            "exposure": recorder.summary(),
                            "source_load_seconds": source_load_seconds,
                            "train_seconds": train_seconds,
                            "checkpoint_io_seconds": checkpoint_io_seconds,
                            "peak_allocated_bytes": train_peak_allocated,
                            "peak_reserved_bytes": train_peak_reserved}
            save_started = perf_counter()
            atomic_checkpoint(run_dir / "latest.pt",
                              state_at(epoch, batch_index + 1, accumulation))
            checkpoint_io_seconds += perf_counter() - save_started
            accumulation["checkpoint_io_seconds"] = checkpoint_io_seconds
            if (batch_index + 1) % 10 == 0 or batch_index + 1 == len(descriptors) // 2:
                print(f"{MODEL_ID} epoch {epoch + 1}: saved batch {batch_index + 1}/"
                      f"{len(descriptors) // 2}, loss={loss_value:.5f}", flush=True)
            del x, y, logits, loss
        optimizer.zero_grad(set_to_none=True)
        if len(batch_history) != config["budget"]["updates_per_epoch"]:
            raise ValueError("Epoch does not contain all frozen optimizer batches")
        losses = [row["loss"] for row in batch_history]
        gradient_norms = [row["gradient_norm_l2"] for row in batch_history]
        exposure = recorder.summary()
        expected_patients = config["train_patients"]
        expected_patches = config["sampling"]["patches_per_patient_epoch"]
        if (exposure["patches"] != expected_patients * expected_patches
                or any(row["patches"] != expected_patches
                       for row in exposure["per_patient"].values())
                or len(exposure["per_patient"]) != expected_patients):
            raise ValueError("Actual delivered Train exposure violates frozen patient policy")
        train_overlap = overlap(train_matrix)
        validation = None
        if (epoch + 1) % config["validation"]["frequency_epochs"] == 0:
            validation = evaluate()
            if validation["selection_score"] > best_score:
                best_score, best_epoch = validation["selection_score"], epoch + 1
                improved = True
            else:
                improved = False
        else:
            improved = False
        summary = {"epoch": epoch + 1, "job_id": os.environ["SLURM_JOB_ID"],
                   "selected_cases": [case.identifier for case in chosen],
                   "train": {"mean_loss": float(np.mean(losses)),
                             "loss_last": losses[-1],
                             "batch_history": batch_history,
                             "mean_gradient_norm_l2": float(np.mean(gradient_norms)),
                             "max_gradient_norm_l2": max(gradient_norms),
                             "confusion": train_matrix.tolist(),
                             "overlap": train_overlap,
                             "exposure": exposure,
                             "learning_rate": optimizer.param_groups[0]["lr"],
                             "source_load_seconds": source_load_seconds,
                             "train_seconds": train_seconds,
                             "checkpoint_io_seconds": checkpoint_io_seconds,
                             "epoch_seconds": train_seconds + checkpoint_io_seconds,
                             "peak_allocated_bytes": train_peak_allocated,
                             "peak_reserved_bytes": train_peak_reserved},
                   "validation": validation, "best_score": best_score,
                   "best_epoch": best_epoch}
        epoch_path = run_dir / f"epoch_{epoch + 1:03d}.pt"
        if epoch_path.exists():
            raise ValueError(f"Immutable epoch snapshot already exists: {epoch_path}")
        atomic_checkpoint(epoch_path, state_at(epoch + 1, 0, None, summary))
        atomic_json(run_dir / f"epoch_{epoch + 1:03d}.json", summary)
        if improved:
            temporary = run_dir / "best.pt.tmp"
            shutil.copyfile(epoch_path, temporary)
            os.replace(temporary, run_dir / "best.pt")
        temporary = run_dir / "latest.pt.tmp"
        shutil.copyfile(epoch_path, temporary)
        os.replace(temporary, run_dir / "latest.pt")
        prostate_dice = train_overlap["dice"][5]
        prostate_text = "NA" if prostate_dice is None else f"{prostate_dice:.4f}"
        message = (f"{MODEL_ID} epoch {epoch + 1}/{planned_epochs}: "
                   f"Train loss={summary['train']['mean_loss']:.5f}, "
                   f"Prostate patch Dice={prostate_text}, "
                   f"Train + checkpoint seconds={train_seconds + checkpoint_io_seconds:.1f}")
        if validation is not None:
            message += f", Validation selection={validation['selection_score']:.4f}"
        print(message, flush=True)
    if best_epoch is None:
        print(f"Stopped {MODEL_ID} after epoch {final_epoch}; latest checkpoint in {run_dir}; "
              "Validation has not run yet", flush=True)
    else:
        print(f"Stopped {MODEL_ID} after epoch {final_epoch}; "
              f"best Validation score={best_score:.4f} at epoch {best_epoch}; "
              f"checkpoints in {run_dir}", flush=True)


if __name__ == "__main__":
    main()
