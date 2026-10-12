"""Isolated E10 probability probe: three fixed patches, six main-head forwards.

Preflight is CPU-only and decodes no MRI arrays. Real inference requires an
explicit --run-six-forwards and an allocated CUDA Slurm session. JSONL goes to
stdout; no images, logits, probabilities, checkpoints or files are written.
"""

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
from time import perf_counter
from unittest.mock import patch

import numpy as np
import torch

from ..dataset import APPROVED_PATIENTS, DEFAULT_DATASET_ROOT
from . import class_collapse as exposure


EXPOSURE_SHA256 = '028a495c41e14bec4855bad5fd55c1dcea51472c890285d7c499f5f4d93b22a1'
CHECKPOINT_SHA256 = {
    'standard': 'f2a87f9c73e068719e2f1d878291116d26320be7c5176827f78537fc4fb4f9cf',
    'improved': '4861c5d65a780af7c8ab67e59715ec37f70ca81dee314f3b80a6a26ad031c0d6',
}
MODEL_NAMES = {'standard': 'StandardUNet3D', 'improved': 'ImprovedUNet3D'}
CLASS_NAMES = ('Background', 'Body', 'Bones', 'Bladder', 'Rectum', 'Prostate')
FIXED_PATCHES = (
    dict(case='K019_Week1', patient_id='K019', week=1, split='train',
         dataset_index=42, completed_epoch=4, sampler_epoch=3,
         crop_origin_dhw=[39, 92, 93], sampling_mode='prostate',
         voxel_counts=[0, 210974, 22665, 5207, 6149, 17149],
         mask_sha256='103f068d01591b1f66498210b7f35663fce150880edbf5de49ec4f48e913df92'),
    dict(case='S035_Week0', patient_id='S035', week=0, split='train',
         dataset_index=117, completed_epoch=6, sampler_epoch=5,
         crop_origin_dhw=[43, 122, 107], sampling_mode='prostate',
         voxel_counts=[0, 213642, 22864, 9266, 11924, 4448],
         mask_sha256='6ec4bea2d93d6f9b508c238cfd4fd57237477d7769b9fad0f5e2c71ff86e5b63'),
    dict(case='B040_Week0', patient_id='B040', week=0, split='validation',
         dataset_index=0, completed_epoch=None, sampler_epoch=0,
         crop_origin_dhw=[32, 87, 87], sampling_mode='center'),
)


def file_sha256(path):
    return exposure._file_hash(path)


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(',', ':')).encode()).hexdigest()


def read_exposure(path):
    """Pin the authenticated original log, then independently validate its scope."""
    if file_sha256(path) != EXPOSURE_SHA256:
        raise ValueError('Exposure JSONL SHA-256 differs from authenticated job 650736.')
    rows = [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()]
    if (len(rows) != 1575 or rows[0].get('event') != 'setup'
            or rows[-1].get('event') != 'summary'
            or sum(r.get('event') == 'setup' for r in rows) != 1
            or sum(r.get('event') == 'summary' for r in rows) != 1):
        raise ValueError('Exposure requires one setup, 143 volumes, 1430 patches, one summary.')
    computed = exposure.summarize_exposure(rows[1:-1], expected_volumes=143)
    if (computed['status'] != 'PASS' or computed['unique_patients'] != 26
            or rows[-1].get('full_train_completed') is not True
            or any(rows[-1].get(k) != v for k, v in computed.items())):
        raise ValueError('Exposure summary is incomplete or differs from recomputed records.')
    full = {r['dataset_index']: r for r in rows[1:-1] if r['event'] == 'full_volume'}
    if (set(full) != set(range(143))
            or {r['patient_id'] for r in full.values()} != set(APPROVED_PATIENTS['train'])):
        raise ValueError('Exposure full-volume identities differ from frozen Training split.')
    patches = {(r['dataset_index'], r['completed_epoch']): r for r in rows[1:-1]
               if r['event'] == 'patch'}
    for selected in FIXED_PATCHES[:2]:
        record = patches[(selected['dataset_index'], selected['completed_epoch'])]
        if any(record.get(k) != v for k, v in selected.items() if k != 'split'):
            raise ValueError(f"{selected['case']}: fixed selection differs from exposure record.")
    setup = rows[0]
    if (setup.get('model') != MODEL_NAMES['standard']
            or setup.get('checkpoint_sha256') != CHECKPOINT_SHA256['standard']
            or setup.get('checkpoint_metadata', {}).get('completed_epochs') != 10
            or setup.get('checkpoint_metadata', {}).get('total_steps') != 1430):
        raise ValueError('Exposure setup is not the authenticated Standard E10 snapshot.')
    return {'setup': setup, 'full': full, 'patches': patches,
            'sha256': EXPOSURE_SHA256, 'record_counts': [1, 143, 1430, 1]}


def validate_dataset_selection(train_dataset, validation_dataset):
    """Only filename discovery here; no Test arrays or preprocessing in preflight."""
    for dataset, split, volumes, patients in (
            (train_dataset, 'train', 143, 26), (validation_dataset, 'validation', 38, 6)):
        if (dataset.split != split or len(dataset) != volumes or len(dataset.pairs) != volumes
                or {p.patient_id for p in dataset.pairs} != set(APPROVED_PATIENTS[split])
                or len({p.patient_id for p in dataset.pairs}) != patients
                or dataset.patch_shape_dhw != (64, 64, 64) or dataset.seed != 3710
                or dataset.train_patches_per_volume != 1
                or dataset.train_foreground_probability != .5):
            raise ValueError(f'{split}: Dataset differs from frozen patch policy/coverage.')
    for selected in FIXED_PATCHES:
        dataset = train_dataset if selected['split'] == 'train' else validation_dataset
        pair = dataset.pairs[selected['dataset_index']]
        if (pair.patient_id, pair.week) != (selected['patient_id'], selected['week']):
            raise ValueError(f"{selected['case']}: actual Dataset index differs from fixed selection.")


def validate_checkpoint_config(payload, model_type, dataset):
    """Add full training-source/order/version protections to protected eval loaders."""
    if model_type not in MODEL_NAMES:
        raise ValueError('Model must be standard or improved.')
    config = payload.get('config', {})
    if (payload.get('completed_epochs') != 10 or payload.get('total_steps') != 1430
            or config.get('hyperparameters', {}).get('model') != MODEL_NAMES[model_type]):
        raise ValueError('Probability probe requires the requested architecture at E10 / 1430 updates.')
    exposure.validate_replay_provenance(config, dataset)
    import nibabel
    import scipy
    actual = {'numpy': np.__version__, 'nibabel': nibabel.__version__,
              'scipy': scipy.__version__, 'torch': str(torch.__version__),
              'cuda_build': torch.version.cuda}
    if config.get('versions') != actual:
        raise ValueError('Checkpoint runtime versions differ; exact preprocessing/model diagnosis not authenticated.')
    return config


def protected_load(path, model_type, dataset, *, expected_sha256=None):
    """CPU weights-only read, complete provenance, then unchanged strict eval loader."""
    digest = file_sha256(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f'{model_type}: checkpoint SHA-256 differs from authenticated snapshot.')
    payload = torch.load(Path(path).expanduser(), map_location='cpu', weights_only=True)
    config = validate_checkpoint_config(payload, model_type, dataset)
    del payload
    if model_type == 'standard':
        from ..validate import load_checkpoint_model
    else:
        from ..validate_improved import load_checkpoint_model
    model, metadata = load_checkpoint_model(path, torch.device('cpu'))
    if (metadata['completed_epochs'] != 10 or metadata['total_steps'] != 1430
            or type(model).__name__ != MODEL_NAMES[model_type]):
        raise ValueError('Protected loader returned unexpected model or checkpoint identity.')
    return model, {'path': str(Path(path).expanduser().resolve()), 'sha256': digest,
                   'metadata': metadata, 'config': config,
                   'config_sha256': canonical_sha256(config)}


def compare_standard_checkpoints(snapshot_model, validation_model, snapshot_info, validation_info):
    """Byte differences are reported; any unequal state tensor blocks the probe."""
    left, right = snapshot_model.state_dict(), validation_model.state_dict()
    same = (left.keys() == right.keys() and all(
        left[k].dtype == right[k].dtype and left[k].shape == right[k].shape
        and torch.equal(left[k], right[k]) for k in left))
    result = {'snapshot_path': snapshot_info['path'], 'snapshot_sha256': snapshot_info['sha256'],
              'full_validation_path': validation_info['path'],
              'full_validation_current_sha256': validation_info['sha256'],
              'file_bytes_identical': snapshot_info['sha256'] == validation_info['sha256'],
              'state_tensors_identical': same,
              'historical_validation_file_hash_available': False,
              'interpretation': 'Current files compared; a missing historical file hash prevents proving its past bytes.'}
    if not same:
        raise ValueError('Standard Validation checkpoint and probe snapshot have different weights; stop comparison. '
                         + json.dumps(result, sort_keys=True))
    return result


def validate_fixed_sample(sample, selected):
    if (sample['patient_id'] != selected['patient_id'] or sample['week'] != selected['week']
            or sample['sampling_mode'] != selected['sampling_mode']
            or sample['crop_origin_dhw'].tolist() != selected['crop_origin_dhw']):
        raise ValueError(f"{selected['case']}: sample identity/origin/sampling mode mismatch.")
    mri, mask = sample['mri'], sample['mask']
    if (mri.device.type != 'cpu' or mask.device.type != 'cpu'
            or mri.dtype != torch.float32 or mask.dtype != torch.long
            or tuple(mri.shape) != (1, 64, 64, 64) or tuple(mask.shape) != (64, 64, 64)
            or not bool(torch.isfinite(mri).all()) or bool(((mask < 0) | (mask > 5)).any())):
        raise ValueError('Fixed sample requires finite CPU float32 [1,64,64,64] MRI and long [64,64,64] labels 0-5.')
    counts = exposure.class_counts(mask.numpy())
    digest = hashlib.sha256(mask.numpy().tobytes()).hexdigest()
    if selected['split'] == 'train':
        if counts != selected['voxel_counts'] or digest != selected['mask_sha256']:
            raise ValueError(f"{selected['case']}: reconstructed mask counts/hash differ from authentic exposure.")
    elif [c for c, n in enumerate(counts) if n] != [1, 2, 3, 4, 5]:
        raise ValueError('B040 center-patch labels differ from recorded acceptance; do not substitute another crop.')
    return {'voxel_counts': counts, 'mask_sha256': digest}


def reconstruct_patch(dataset, selected, evidence):
    """One production preprocessing run, two identical sampler calls, no new crop algorithm."""
    from .. import patch_dataset as production
    index = selected['dataset_index']
    pair = dataset.pairs[index]
    normalized, source = exposure.production_preprocess(pair, dataset)
    if selected['split'] == 'train':
        full = evidence['full'][index]
        if (any(source[k] != full[k] for k in source)
                or list(normalized.segmentation.shape) != full['resampled_shape_xyz']
                or exposure.class_counts(normalized.segmentation) != full['resampled_voxel_counts']
                or not np.allclose(normalized.affine, full['affine'], atol=1e-8, rtol=0)
                or normalized.spatial_unit != full['spatial_unit']
                or normalized.segmentation_header_spatial_unit != full['mask_header_unit']
                or normalized.segmentation_unit_inferred != full['mask_unit_inferred']):
            raise ValueError(f"{selected['case']}: full preprocessed geometry/labels differ from exposure.")
    elif normalized.segmentation.shape != (239, 239, 128):
        raise ValueError('B040 processed shape differs from recorded acceptance.')
    if (not np.isfinite(normalized.affine).all()
            or normalized.affine.shape != (4, 4)):
        raise ValueError('Invalid processed affine metadata.')
    dataset.set_epoch(selected['sampler_epoch'])
    # Same scoped interception as approved exposure replay: production crop logic
    # runs intact; reuse this ONE validated processed pair, never a disk cache.
    with patch.object(production, 'load_volume_pair', return_value=None), \
            patch.object(production, 'resample_volume_pair', return_value=None), \
            patch.object(production, 'normalize_volume_pair', return_value=normalized):
        sample = dataset[index]
        validation = validate_fixed_sample(sample, selected)
        repeated = dataset[index]
        validate_fixed_sample(repeated, selected)
        if not torch.equal(sample['mri'], repeated['mri']) or not torch.equal(sample['mask'], repeated['mask']):
            raise ValueError('Repeated fixed sampler call is not bitwise identical.')
        del repeated
    offset = np.eye(4)
    offset[:3, 3] = selected['crop_origin_dhw'][::-1]
    metadata = dict(selected, **validation, source_shape_xyz=source['source_shape_xyz'],
                    resampled_shape_xyz=list(normalized.segmentation.shape),
                    affine=normalized.affine.tolist(), affine_scope='full resampled XYZ grid',
                    patch_affine_xyz=(normalized.affine @ offset).tolist(),
                    crop_origin_xyz=selected['crop_origin_dhw'][::-1], tensor_axes='DHW = ZYX',
                    mri_patch_sha256=hashlib.sha256(sample['mri'].numpy().tobytes()).hexdigest(),
                    target_spacing_mm=list(normalized.target_spacing_mm),
                    spatial_unit=normalized.spatial_unit,
                    mask_header_unit=normalized.segmentation_header_spatial_unit,
                    mask_unit_inferred=normalized.segmentation_unit_inferred,
                    source_mri_sha256=file_sha256(pair.mri_path),
                    source_mask_sha256=file_sha256(pair.segmentation_path),
                    repeated_sampler_bitwise_equal=True)
    del normalized
    gc.collect()
    return sample, metadata


def _distribution(values):
    if not bool(torch.isfinite(values).all()):
        raise ValueError('Non-finite probability, logit margin or NLL statistic.')
    values = values.detach().cpu().double().numpy()
    return {'mean': float(values.mean()), **dict(zip(
        ('min', 'p10', 'median', 'p90', 'p99', 'max'),
        np.percentile(values, [0, 10, 50, 90, 99, 100]).tolist()))}


def probability_statistics(logits, target):
    """CPU float32 softmax; GT-conditional statistics, all-patch Dice/CE terms.

    Rank ties favour the lowest channel ID, matching torch.argmax. Signed
    target-minus-best-other margins are separate from winner-minus-target loss.
    Absent GT classes have null conditional metrics, not successful predictions.
    """
    from ..train import combined_loss
    from ..validate import confusion_matrix, metrics_from_confusion
    if (logits.device.type != 'cpu' or target.device.type != 'cpu'
            or logits.dtype != torch.float32 or target.dtype != torch.long
            or logits.ndim != 5 or logits.shape[0] != 1 or logits.shape[1] != 6
            or any(s < 1 for s in logits.shape[2:])
            or target.shape != (1, *logits.shape[2:])
            or not bool(torch.isfinite(logits).all()) or bool(((target < 0) | (target > 5)).any())):
        raise ValueError('Expected finite CPU float32 [1,6,D,H,W] logits and long [1,D,H,W] labels 0-5.')
    with torch.inference_mode():
        probabilities = logits.softmax(1)[0].reshape(6, -1)
        log_probabilities = logits.log_softmax(1)[0].reshape(6, -1)
        sums = probabilities.sum(0)
        if (not bool(torch.isfinite(probabilities).all())
                or not bool(torch.isfinite(log_probabilities).all()) or not torch.allclose(
                sums, torch.ones_like(sums), atol=1e-6, rtol=0)):
            raise ValueError('Six-channel softmax probabilities are non-finite or do not sum to one.')
        truth = target.reshape(-1)
        prediction = probabilities.argmax(0)
        counts = torch.bincount(prediction, minlength=6).tolist()
        cm = confusion_matrix(prediction.reshape(target.shape[1:]).numpy(), target[0].numpy())
        hard = metrics_from_confusion(cm)
        classes = []
        for label in range(6):
            select = truth == label
            n = int(select.sum())
            p = probabilities[label]
            score = float((2 * p[select].sum() + 1e-5) / (p.sum() + n + 1e-5))
            row = {'label': label, 'name': CLASS_NAMES[label], 'gt_voxels': n,
                   'predicted_voxels_all_patch': counts[label], 'evaluable_on_gt': bool(n),
                   'probability_mass_all_patch': float(p.sum()),
                   'soft_dice': score if n else None,
                   'objective_soft_dice_score': score, 'included_in_foreground_loss': label != 0,
                   'hard_dice_iou_all_patch': hard[label],
                   'conditional': None, 'ce_contribution_all_patch': 0.}
            if n:
                selected_p = probabilities[:, select]
                pc = selected_p[label]
                winners = prediction[select]
                channel_ids = torch.arange(6).unsqueeze(1)
                ranks = 1 + ((selected_p > pc) | ((selected_p == pc) & (channel_ids < label))).sum(0)
                rank_counts = torch.bincount(ranks, minlength=7)[1:].tolist()
                winning_counts = torch.bincount(winners, minlength=6).tolist()
                competing = winning_counts.copy(); competing[label] = 0
                competitor = int(np.argmax(competing)) if sum(competing) else None
                other = [c for c in range(6) if c != label]
                signed_p = pc - selected_p[other].max(0).values
                selected_z = logits[0].reshape(6, -1)[:, select]
                signed_z = selected_z[label] - selected_z[other].max(0).values
                nll = float(-log_probabilities[label, select].mean())
                row['conditional'] = {
                    'denominator': n, 'argmax_recall': winning_counts[label] / n,
                    'true_class_probability': _distribution(pc), 'true_class_rank_counts_1_to_6': rank_counts,
                    'rank_first_fraction': rank_counts[0] / n, 'rank_second_fraction': rank_counts[1] / n,
                    'rank_below_second_fraction': sum(rank_counts[2:]) / n,
                    'winning_class_counts': winning_counts, 'competing_winner_counts': competing,
                    'most_frequent_competing_winner': competitor,
                    'winner_minus_true_probability': _distribution(selected_p.max(0).values - pc),
                    'true_minus_best_other_probability': _distribution(signed_p),
                    'true_minus_best_other_logit': _distribution(signed_z),
                    'strictly_positive_probability_margin_fraction': float((signed_p > 0).double().mean()),
                    'top_probability_tie_voxels': int(((selected_p == selected_p.max(0).values).sum(0) > 1).sum()),
                    'mean_negative_log_likelihood': nll,
                }
                row['ce_contribution_all_patch'] = n / truth.numel() * nll
            classes.append(row)
        total, ce, dice = combined_loss(logits, target)
        if not np.isclose(sum(r['ce_contribution_all_patch'] for r in classes), float(ce), atol=1e-6, rtol=1e-6):
            raise ValueError('GT-conditional NLL contributions do not reproduce production CE.')
    return {'patch_voxels': truth.numel(), 'predicted_class_counts': counts,
            'predicted_classes': [c for c, n in enumerate(counts) if n],
            'confusion_counts_truth_rows_prediction_columns': cm.tolist(), 'classes': classes,
            'softmax_sum_max_abs_error': float((sums - 1).abs().max()),
            'rank_tie_policy': 'descending_probability_then_lowest_channel_id',
            'smoothing': 1e-5, 'main_head_loss': {'ce': float(ce), 'foreground_soft_dice_loss': float(dice),
                                               'total': float(total)},
            'scope': 'one fixed patch; CPU float32 softmax; not overlapping full-volume inference'}


def infer_once(model, sample, device):
    """Exactly one main-head call; no optimizer, backward or auxiliary call."""
    model.to(device).eval()
    image = sample['mri'].unsqueeze(0).to(device)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = perf_counter()
    with torch.inference_mode():
        logits = model(image)  # Improved default returns main head only.
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    seconds = perf_counter() - start
    if (not isinstance(logits, torch.Tensor) or logits.dtype != torch.float32
            or tuple(logits.shape) != (1, 6, *sample['mask'].shape) or not bool(torch.isfinite(logits).all())):
        raise ValueError('Model did not return finite full-resolution six-class float32 main logits.')
    memory = None if device.type != 'cuda' else {
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
        'peak_reserved_bytes': torch.cuda.max_memory_reserved(device)}
    cpu_logits = logits.detach().cpu()
    del logits, image
    model.to('cpu')
    stats = probability_statistics(cpu_logits, sample['mask'].unsqueeze(0))
    if stats != probability_statistics(cpu_logits, sample['mask'].unsqueeze(0)):
        raise ValueError('Repeated statistics on identical logits differ.')
    return {'forward_seconds': seconds, 'gpu_memory': memory,
            'repeated_statistics_equal': True, 'statistics': stats}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--exposure-log', type=Path, required=True)
    result.add_argument('--standard-checkpoint', type=Path, required=True)
    result.add_argument('--standard-validation-checkpoint', type=Path, required=True)
    result.add_argument('--improved-checkpoint', type=Path, required=True)
    result.add_argument('--dataset-root', type=Path, default=DEFAULT_DATASET_ROOT)
    action = result.add_mutually_exclusive_group(required=True)
    action.add_argument('--preflight-only', action='store_true')
    action.add_argument('--run-six-forwards', action='store_true')
    return result


def _emit(row):
    print(json.dumps(row, allow_nan=False), flush=True)


def main(argv=None):
    args = parser().parse_args(argv)
    for key in ('exposure_log', 'standard_checkpoint', 'standard_validation_checkpoint', 'improved_checkpoint'):
        path = getattr(args, key).expanduser().resolve(strict=True)
        if not path.is_file():
            raise ValueError(f'{key} must be an existing file.')
        setattr(args, key, path)
    if args.run_six_forwards and (not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available()):
        raise RuntimeError('Six-forward real probe requires an allocated CUDA Slurm compute session; no CPU fallback.')
    evidence = read_exposure(args.exposure_log)
    from ..patch_dataset import HipMRIPatchDataset
    train_dataset = HipMRIPatchDataset('train', args.dataset_root)
    validation_dataset = HipMRIPatchDataset('validation', args.dataset_root)
    validate_dataset_selection(train_dataset, validation_dataset)
    models, provenance = {}, {}
    for kind in ('standard', 'improved'):
        models[kind], provenance[kind] = protected_load(
            getattr(args, kind + '_checkpoint'), kind, train_dataset, expected_sha256=CHECKPOINT_SHA256[kind])
    if provenance['standard']['config'] != evidence['setup']['config']:
        raise ValueError('Standard snapshot config differs from authenticated exposure setup.')
    validation_model, validation_info = protected_load(args.standard_validation_checkpoint, 'standard', train_dataset)
    comparison = compare_standard_checkpoints(models['standard'], validation_model,
                                               provenance['standard'], validation_info)
    del validation_model
    _emit({'event': 'preflight', 'status': 'PASS', 'exposure_sha256': evidence['sha256'],
           'exposure_record_counts': evidence['record_counts'], 'checkpoints': provenance,
           'standard_snapshot_vs_validation': comparison, 'fixed_patches': FIXED_PATCHES,
           'diagnostic_source_sha256': {
               'checkpoint_probabilities.py': file_sha256(__file__),
               'class_collapse.py': file_sha256(exposure.__file__)},
           'interpretation': 'File/config/index checks only; fixed MRI/mask replay still pending until inference mode.'})
    if args.preflight_only:
        _emit({'event': 'summary', 'status': 'PASS', 'scope': 'CPU preflight only',
               'main_head_forward_passes': 0, 'real_arrays_decoded': 0})
        return 0
    from ..train import configure_reproducibility
    device = torch.device('cuda:0')
    configure_reproducibility(3710, device)
    _emit({'event': 'setup', 'slurm_job_id': os.environ['SLURM_JOB_ID'], 'hostname': socket.gethostname(),
           'device': torch.cuda.get_device_name(device), 'expected_main_head_forward_passes': 6,
           'reproducibility': 'production best_effort_warn_only; repeated sampler/statistics checked, no repeat forwards',
           'scope': 'three selected patches; no full-volume inference, Test arrays or backward'})
    forwards = 0
    for selected in FIXED_PATCHES:
        dataset = train_dataset if selected['split'] == 'train' else validation_dataset
        start = perf_counter()
        sample, metadata = reconstruct_patch(dataset, selected, evidence)
        _emit({'event': 'patch', 'status': 'PASS', 'metadata': metadata,
               'preprocessing_replay_hash_seconds': perf_counter() - start})
        for kind in ('standard', 'improved'):
            result = infer_once(models[kind], sample, device)
            forwards += 1
            _emit({'event': 'probability', 'status': 'PASS', 'model': MODEL_NAMES[kind],
                   'case': selected['case'], 'split': selected['split'],
                   'checkpoint_sha256': provenance[kind]['sha256'], **result})
            gc.collect()
            torch.cuda.empty_cache()
        del sample
        gc.collect()
    if forwards != 6:
        raise ValueError('Incomplete six-forward scope.')
    _emit({'event': 'summary', 'status': 'PASS', 'patches': 3, 'main_head_forward_passes': forwards,
           'optimizer_updates': 0, 'backward_passes': 0, 'test_arrays_decoded': 0,
           'scope': 'fixed-patch probabilities; not clinical accuracy, generalisation or causal proof'})
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        _emit({'event': 'error', 'status': 'FAIL', 'exception': type(exc).__name__, 'detail': str(exc)})
        print(f'FAIL: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        sys.exit(1)
