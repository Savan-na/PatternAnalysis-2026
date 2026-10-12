"""Isolated CPU Train exposure reconstruction and historical-loss log audit.

Real-data execution needs separate authorisation. No training, CUDA, remote
commands, automatic file writes or production changes occur here. The actual
Dataset sampler runs intact; scoped mocks reuse one preprocessed pair across
ten epoch/index calls, without implementing a second crop algorithm or cache.
"""

import argparse
from collections import Counter
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import torch

from ..dataset import APPROVED_PATIENTS, parse_volume_filename


def class_counts(mask):
    """Six integer counts, including zero-count IDs; no one-hot array allocation."""
    if (not isinstance(mask, np.ndarray) or mask.ndim != 3 or not mask.size
            or mask.dtype.kind not in 'iu' or mask.min() < 0 or mask.max() > 5):
        raise ValueError('Expected a nonempty 3D integer mask with IDs 0-5.')
    return np.bincount(mask.ravel(), minlength=6).astype(np.int64).tolist()


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_replay_provenance(config, dataset):
    """Require the saved training source, policies, seed and exact pair ordering."""
    from .. import train, train_improved
    directory = Path(__file__).resolve().parents[1]
    hp = config.get('hyperparameters', {})
    model = hp.get('model')
    expected = {'StandardUNet3D': train.HYPERPARAMETERS,
                'ImprovedUNet3D': train_improved.HYPERPARAMETERS}
    if model not in expected or hp != expected[model]:
        raise ValueError('Checkpoint model/hyperparameters differ from approved training.')
    sources = (train_improved.SOURCE_FILES if model == 'ImprovedUNet3D' else
               ('train.py', 'modules.py', 'dataset.py', 'patch_dataset.py',
                'volume_io.py', 'resampling.py', 'intensity_normalization.py'))
    hashes = {name: _file_hash(directory / name) for name in sources}
    if config.get('source_sha256') != hashes:
        raise ValueError('Checkpoint training source hashes differ; replay is not authenticated.')
    policies = {name: json.loads((directory / (name + '_config.json')).read_text(encoding='utf-8'))
                for name in ('resampling', 'normalization', 'patch_dataset')}
    if config.get('policies') != policies:
        raise ValueError('Checkpoint preprocessing/sampling policies differ.')
    if config.get('versions', {}).get('numpy') != np.__version__:
        raise ValueError('Checkpoint NumPy version differs; exact sampler replay is not authenticated.')
    pairs = [[p.patient_id, p.week, str(p.mri_path), str(p.segmentation_path)] for p in dataset.pairs]
    if config.get('train_pairs') != pairs:
        raise ValueError('Checkpoint training pair order/paths differ from discovery.')
    if (dataset.split != 'train' or len(dataset) != 143 or len(dataset.pairs) != 143
            or {p.patient_id for p in dataset.pairs} != set(APPROVED_PATIENTS['train'])
            or dataset.patch_shape_dhw != (64, 64, 64) or dataset.seed != 3710
            or dataset.train_patches_per_volume != 1
            or dataset.train_foreground_probability != 0.5
            or config.get('steps_per_epoch') != 143):
        raise ValueError('Replay requires the approved 143-volume, 26-patient patch policy.')


def load_replay_checkpoint(path, dataset):
    """CPU weights-only load plus unchanged protected strict checkpoint loader."""
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload.get('completed_epochs') != 10 or payload.get('total_steps') != 1430:
        raise ValueError('Exposure diagnosis requires the completed Epoch 10 checkpoint.')
    config = payload.get('config', {})
    validate_replay_provenance(config, dataset)
    import nibabel
    import scipy
    if any(config.get('versions', {}).get(name) != module.__version__
           for name, module in (('nibabel', nibabel), ('scipy', scipy))):
        raise ValueError('Checkpoint NiBabel/SciPy versions differ; exact preprocessing replay is not authenticated.')
    model_name = config['hyperparameters']['model']
    del payload
    if model_name == 'StandardUNet3D':
        from ..validate import load_checkpoint_model
    else:
        from ..validate_improved import load_checkpoint_model
    model, metadata = load_checkpoint_model(path, torch.device('cpu'))
    del model  # No forward pass, optimizer restore or CUDA allocation.
    return {'model': model_name, 'checkpoint': str(Path(path).resolve()),
            'checkpoint_sha256': _file_hash(path), 'checkpoint_metadata': metadata,
            'config': config, 'cpu_numpy_version': np.__version__,
            'cpu_torch_version': str(torch.__version__)}


def select_indices(dataset, *, cases=None, max_cases=None, full_train=False):
    """Explicit bounded scope; retain original Dataset indices for seed replay."""
    if sum((cases is not None, max_cases is not None, full_train)) != 1:
        raise ValueError('Select exactly one of cases, max_cases or full_train.')
    if dataset.split != 'train':
        raise ValueError('Exposure scope is Train only.')
    if full_train:
        return tuple(range(len(dataset.pairs)))
    if max_cases is not None:
        if type(max_cases) is not int or not 1 <= max_cases <= len(dataset.pairs):
            raise ValueError('max_cases must be in 1..number of training volumes.')
        return tuple(range(max_cases))
    keys = [parse_volume_filename(case + '_LFOV.nii.gz', 'LFOV') for case in cases]
    if not keys or len(keys) != len(set(keys)):
        raise ValueError('Cases must be nonempty and unique.')
    lookup = {(p.patient_id, p.week): i for i, p in enumerate(dataset.pairs)}
    if any(key not in lookup for key in keys):
        raise ValueError('Requested case is not in the frozen Training split.')
    return tuple(lookup[key] for key in keys)


def production_preprocess(pair, dataset):
    """One production decode/resample/normalise run; retain one processed pair."""
    from ..volume_io import load_volume_pair
    from ..resampling import resample_volume_pair
    from ..intensity_normalization import normalize_volume_pair
    loaded = load_volume_pair(pair)
    source = {'source_shape_xyz': list(loaded.segmentation.shape),
              'source_voxel_counts': class_counts(loaded.segmentation),
              'source_voxel_volume_mm3': float(abs(np.linalg.det(loaded.mri_affine[:3, :3])))}
    resampled = resample_volume_pair(loaded, dataset.target_spacing_mm,
                                   max_estimated_bytes=dataset.max_estimated_bytes)
    del loaded
    normalized = normalize_volume_pair(resampled)
    del resampled
    return normalized, source


def exposure_records(dataset, preprocess=production_preprocess, indices=None):
    """Stream exact production crops for sampler epochs 0..9 (human epochs 1..10).

    Index iteration changes visit order, not seed/epoch/index-keyed crop choice.
    This reconstructs targets, not DataLoader ordering or optimisation history.
    Synthetic tests supply a smaller train-only Dataset and mocked preprocessing.
    """
    from .. import patch_dataset as production
    if dataset.split != 'train' or dataset.train_patches_per_volume != 1:
        raise ValueError('Exposure replay is train-only, one patch per volume.')
    if any(p.patient_id not in APPROVED_PATIENTS['train'] for p in dataset.pairs):
        raise ValueError('Replay includes a non-training patient.')
    indices = tuple(range(len(dataset.pairs))) if indices is None else tuple(indices)
    if (len(set(indices)) != len(indices) or not indices
            or any(type(i) is not int or not 0 <= i < len(dataset.pairs) for i in indices)):
        raise ValueError('Replay indices must be unique valid original Dataset indices.')
    for index in indices:
        pair = dataset.pairs[index]
        normalized, source = preprocess(pair, dataset)
        full = dict(event='full_volume', dataset_index=index, patient_id=pair.patient_id,
                    week=pair.week, case=f'{pair.patient_id}_Week{pair.week}', **source,
                    resampled_shape_xyz=list(normalized.segmentation.shape),
                    resampled_voxel_counts=class_counts(normalized.segmentation),
                    affine=normalized.affine.tolist(),
                    resampled_voxel_volume_mm3=float(abs(np.linalg.det(normalized.affine[:3, :3]))),
                    spatial_unit=normalized.spatial_unit,
                    mask_header_unit=normalized.segmentation_header_spatial_unit,
                    mask_unit_inferred=normalized.segmentation_unit_inferred)
        yield full
        # Only preprocessing return values are intercepted, within this scope.
        # Actual production transpose, RNG, prostate selection and crop run intact.
        with patch.object(production, 'load_volume_pair', return_value=None), \
                patch.object(production, 'resample_volume_pair', return_value=None), \
                patch.object(production, 'normalize_volume_pair', return_value=normalized):
            for epoch in range(10):
                dataset.set_epoch(epoch)
                sample = dataset[index]
                counts = class_counts(sample['mask'].numpy())
                if sample['patient_id'] != pair.patient_id or sample['week'] != pair.week:
                    raise ValueError('Production sample identity differs from replay pair.')
                if sample['sampling_mode'] == 'prostate' and counts[5] == 0:
                    raise ValueError('Prostate-aware sample has no Prostate voxels.')
                yield {'event': 'patch', 'case': full['case'], 'patient_id': pair.patient_id,
                       'week': pair.week, 'dataset_index': index, 'completed_epoch': epoch + 1,
                       'sampler_epoch': epoch, 'crop_origin_dhw': sample['crop_origin_dhw'].tolist(),
                       'sampling_mode': sample['sampling_mode'], 'patch_shape_dhw': list(sample['mask'].shape),
                       'voxel_counts': counts,
                       'mask_sha256': hashlib.sha256(sample['mask'].numpy().tobytes()).hexdigest()}
                del sample
        del normalized
        gc.collect()


def _validate_counts(counts, shape):
    if (len(counts) != 6 or any(type(v) is not int or v < 0 for v in counts)
            or len(shape) != 3 or any(type(v) is not int or v < 1 for v in shape)
            or sum(counts) != math.prod(shape)):
        raise ValueError('Six nonnegative class counts must sum to the 3D voxel count.')


def _distribution(values):
    """Min/Q25/median/Q75/max; empty samples are null, never invented zeros."""
    return np.percentile(values, [0, 25, 50, 75, 100]).tolist() if values else None


def _volume_summary(rows, prefix):
    field = prefix + '_voxel_counts'
    counts = [sum(r[field][c] for r in rows) for c in range(6)]
    denominator = sum(counts)
    return {'volumes': len(rows), 'voxel_denominator': denominator, 'voxel_counts': counts,
            'voxel_fractions': [v / denominator if denominator else None for v in counts],
            'volumes_present': [sum(r[field][c] > 0 for r in rows) for c in range(6)],
            'patients_present': [len({r['patient_id'] for r in rows if r[field][c] > 0}) for c in range(6)],
            'per_volume_fraction_min_q25_median_q75_max': [_distribution([
                r[field][c] / sum(r[field]) for r in rows]) for c in range(6)]}


def _patch_summary(rows):
    counts = [sum(r['voxel_counts'][c] for r in rows) for c in range(6)]
    denominator, n = sum(counts), len(rows)
    present = [sum(r['voxel_counts'][c] > 0 for r in rows) for c in range(6)]
    return {'patches': n, 'voxel_denominator': denominator, 'voxel_counts': counts,
            'voxel_fractions': [v / denominator if denominator else None for v in counts],
            'patches_present': present,
            'positive_patch_fractions': [v / n if n else None for v in present],
            'positive_voxel_count_min_q25_median_q75_max': [_distribution([
                r['voxel_counts'][c] for r in rows if r['voxel_counts'][c] > 0]) for c in range(6)],
            'positive_voxel_fraction_min_q25_median_q75_max': [_distribution([
                r['voxel_counts'][c] / sum(r['voxel_counts']) for r in rows
                if r['voxel_counts'][c] > 0]) for c in range(6)],
            'largest_class_patch_counts': [sum(int(np.argmax(r['voxel_counts'])) == c for r in rows)
                                           for c in range(6)],
            'background_or_body_over_half_patches': sum(r['voxel_counts'][0] + r['voxel_counts'][1]
                > sum(r['voxel_counts']) / 2 for r in rows),
            'sampling_modes': dict(Counter(r['sampling_mode'] for r in rows))}


def summarize_exposure(records, expected_volumes=143):
    """Keep raw/full/patch denominators separate; PASS describes chosen scope only."""
    if type(expected_volumes) is not int or expected_volumes < 1:
        raise ValueError('Expected volume count must be a positive integer.')
    full, patches, seen_full, seen_patch = [], [], set(), set()
    for record in records:
        if record['event'] == 'full_volume':
            key = record['dataset_index']
            if key in seen_full:
                raise ValueError('Duplicate full-volume record.')
            for prefix in ('source', 'resampled'):
                _validate_counts(record[prefix + '_voxel_counts'], record[prefix + '_shape_xyz'])
            seen_full.add(key); full.append(record)
        elif record['event'] == 'patch':
            key = (record['dataset_index'], record['completed_epoch'])
            if key in seen_patch or not 1 <= record['completed_epoch'] <= 10:
                raise ValueError('Duplicate or invalid patch epoch record.')
            if record['sampler_epoch'] != record['completed_epoch'] - 1:
                raise ValueError('Sampler epoch must be completed epoch minus one.')
            _validate_counts(record['voxel_counts'], record['patch_shape_dhw'])
            seen_patch.add(key); patches.append(record)
        else:
            raise ValueError('Unknown exposure event.')
    identities = {r['dataset_index']: (r['patient_id'], r['week']) for r in full}
    if any(identities.get(r['dataset_index']) != (r['patient_id'], r['week']) for r in patches):
        raise ValueError('Patch identity has no matching full-volume record.')
    complete = (len(full) == expected_volumes and len(patches) == 10 * expected_volumes
                and seen_patch == {(i, e) for i in seen_full for e in range(1, 11)})
    patients = sorted({r['patient_id'] for r in full})
    return {'event': 'summary', 'status': 'PASS' if complete else 'INCOMPLETE',
            'expected_volumes': expected_volumes, 'full_volumes': len(full),
            'unique_patients': len(patients), 'source_full': _volume_summary(full, 'source'),
            'resampled_full': _volume_summary(full, 'resampled'), 'all_patches': _patch_summary(patches),
            'epochs': [{'completed_epoch': e, **_patch_summary([r for r in patches if r['completed_epoch'] == e])}
                       for e in range(1, 11)],
            'patients': [{'patient_id': p, **_patch_summary([r for r in patches if r['patient_id'] == p]),
                          'resampled_full': _volume_summary([r for r in full if r['patient_id'] == p], 'resampled')}
                         for p in patients],
            'branches': [{'sampling_mode': mode, **_patch_summary([
                r for r in patches if r['sampling_mode'] == mode])} for mode in sorted(
                    {r['sampling_mode'] for r in patches})],
            'interpretation': 'Deterministic target reconstruction, not historical optimisation or clinical accuracy.'}


def summarize_loss_logs(paths):
    """Audit saved stdout JSONL; never manufacture missing history or splice policies."""
    epochs, identities, non_json, experiment = {}, set(), 0, None
    for path in paths:
        model, steps = None, []
        for number, line in enumerate(Path(path).read_text(encoding='utf-8').splitlines(), 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                if line.lstrip().startswith('{'):
                    raise ValueError(f'{path}:{number}: malformed JSON record.')
                non_json += bool(line.strip()); continue
            if not isinstance(record, dict):
                raise ValueError(f'{path}:{number}: expected a JSON object.')
            if record.get('event') == 'setup':
                if steps:
                    raise ValueError('A new setup cannot discard an unfinished step sequence.')
                model = record.get('config', {}).get('hyperparameters', {}).get('model')
                if model not in ('StandardUNet3D', 'ImprovedUNet3D'):
                    raise ValueError('Training setup model is missing/invalid.')
                identity = {k: record['config'].get(k) for k in
                            ('hyperparameters', 'policies', 'train_pairs', 'source_sha256', 'steps_per_epoch')}
                if experiment is not None and identity != experiment:
                    raise ValueError('Log configurations differ; do not splice incompatible experiments.')
                experiment = identity; identities.add(model)
            elif record.get('event') == 'step':
                if model is None or record.get('step') != len(steps) + 1:
                    raise ValueError('Step sequence has no setup or is incomplete/out of order.')
                steps.append(record)
            elif record.get('event') == 'epoch':
                e = record.get('completed_epochs')
                if (model is None or type(e) is not int or not 1 <= e <= 10
                        or record.get('epoch_completed') is not True
                        or record.get('processed_steps') != 143 or record.get('expected_steps') != 143
                        or record.get('total_steps') != e * 143 or e in epochs):
                    raise ValueError('Invalid, duplicate or incompatible epoch summary.')
                values = [record.get(k) for k in ('mean_loss', 'mean_ce', 'mean_soft_dice_loss')]
                if any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
                    raise ValueError('Epoch loss is missing/non-finite.')
                if not math.isclose(values[0], .5 * (values[1] + values[2]), abs_tol=1e-6):
                    raise ValueError('Epoch components do not match the approved objective.')
                if steps:
                    if len(steps) != 143:
                        raise ValueError('Partial step records cannot authenticate an epoch.')
                    for key, mean in zip(('loss', 'ce', 'soft_dice_loss'), values):
                        if any(type(s.get(key)) not in (int, float) or not math.isfinite(s[key]) for s in steps):
                            raise ValueError('Missing/non-finite step loss.')
                        if not math.isclose(sum(s[key] for s in steps) / 143, mean, abs_tol=1e-6):
                            raise ValueError('Step means disagree with epoch summary.')
                epochs[e] = {'completed_epoch': e, 'mean_loss': values[0], 'mean_ce': values[1],
                             'mean_soft_dice_loss': values[2], 'steps_verified': len(steps),
                             'source': str(path), 'line': number}
                steps = []
        if steps:
            raise ValueError('Unfinished/diagnostic steps are not a completed training epoch.')
    if len(identities) > 1:
        raise ValueError('Audit Standard and Improved trajectories separately.')
    missing = sorted(set(range(1, 11)) - epochs.keys())
    return {'event': 'loss_history', 'status': 'PASS' if not missing else 'INCOMPLETE',
            'model': next(iter(identities), None), 'epochs': [epochs[e] for e in sorted(epochs)],
            'missing_epochs': missing, 'non_json_lines': non_json,
            'interpretation': 'Training objective history only; no per-class generalisation or convergence proof.'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    exposure = sub.add_parser('patch-exposure', help='CPU Train-only replay; separate real-data approval required')
    exposure.add_argument('--checkpoint', type=Path, required=True)
    exposure.add_argument('--dataset-root', type=Path, required=True)
    scope = exposure.add_mutually_exclusive_group(required=True)
    scope.add_argument('--cases', nargs='+', metavar='PATIENT_WeekN')
    scope.add_argument('--max-cases', type=int)
    scope.add_argument('--full-train', action='store_true')
    history = sub.add_parser('loss-history', help='Inspect supplied existing logs only')
    history.add_argument('--logs', type=Path, nargs='+', required=True)
    args = parser.parse_args(argv)
    if args.mode == 'patch-exposure' and args.max_cases is not None and not 1 <= args.max_cases <= 143:
        parser.error('max-cases must be in 1..143; no data were loaded.')
    if args.mode == 'loss-history':
        summary = summarize_loss_logs(args.logs)
    else:
        from ..patch_dataset import HipMRIPatchDataset
        dataset = HipMRIPatchDataset('train', args.dataset_root)
        indices = select_indices(dataset, cases=args.cases, max_cases=args.max_cases, full_train=args.full_train)
        setup = load_replay_checkpoint(args.checkpoint, dataset)
        print(json.dumps({'event': 'setup', 'selected_indices': list(indices),
                          'full_train_requested': args.full_train, **setup}, allow_nan=False), flush=True)
        def stream():
            completed = 0
            for record in exposure_records(dataset, indices=indices):
                print(json.dumps(record, allow_nan=False), flush=True)
                yield record
                if record['event'] == 'patch' and record['completed_epoch'] == 10:
                    completed += 1
                    print(f"Progress: {completed}/{len(indices)} cases, {record['case']}", file=sys.stderr, flush=True)
        summary = summarize_exposure(stream(), expected_volumes=len(indices))
        summary['scope'] = 'all 143 training volumes' if args.full_train else 'selected training subset'
        summary['full_train_completed'] = bool(args.full_train and summary['status'] == 'PASS'
                                                and summary['full_volumes'] == 143
                                                and summary['unique_patients'] == 26)
        if args.full_train and not summary['full_train_completed']:
            summary['status'] = 'INCOMPLETE'
    print(json.dumps(summary, allow_nan=False), flush=True)
    return 0 if summary['status'] == 'PASS' else 1


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        print(f'FAIL: {type(exc).__name__}: {exc}', file=sys.stderr)
        sys.exit(1)
