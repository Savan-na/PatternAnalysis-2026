"""CPU synthetic probability tests; no real images, checkpoints or remote calls.

If NiBabel is absent locally, an explicitly synthetic import shim permits
mocked preprocessing/loader tests. No NIfTI decode is claimed by these tests.
"""

import copy
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from torch import nn

from ..dataset import APPROVED_PATIENTS, VolumePair
from . import checkpoint_probabilities as probe


_nibabel_shim = None


def setUpModule():
    global _nibabel_shim
    if importlib.util.find_spec('nibabel') is None:
        shim = ModuleType('nibabel'); shim.__version__ = 'synthetic-import-shim'
        _nibabel_shim = patch.dict(sys.modules, {'nibabel': shim})
        _nibabel_shim.start()
        print('TEST LIMITATION: NiBabel import shim; preprocessing is mocked.', file=sys.stderr)
    torch.set_num_threads(1)


def tearDownModule():
    if _nibabel_shim is not None:
        _nibabel_shim.stop()


def sample(selected=None):
    selected = probe.FIXED_PATCHES[2] if selected is None else selected
    truth = (torch.arange(64**3).reshape(64, 64, 64) % 5 + 1).long()
    return {'mri': truth.float().unsqueeze(0), 'mask': truth,
            'patient_id': selected['patient_id'], 'week': selected['week'],
            'crop_origin_dhw': torch.tensor(selected['crop_origin_dhw']),
            'sampling_mode': selected['sampling_mode']}


class KnownModel(nn.Module):
    def __init__(self):
        super().__init__(); self.weight = nn.Parameter(torch.tensor(1.)); self.calls = 0

    def forward(self, image):
        assert not self.training and torch.is_inference_mode_enabled()
        assert not torch.is_grad_enabled()
        self.calls += 1
        labels = image[:, 0].round().long() % 6
        logits = torch.full((1, 6, *image.shape[2:]), -3.)
        return logits.scatter_(1, labels.unsqueeze(1), 3.) * self.weight


def fake_datasets():
    result = []
    for split, n in (('train', 143), ('validation', 38)):
        patients = APPROVED_PATIENTS[split]
        pairs = [VolumePair(patients[i % len(patients)], i, Path(f'mri{i}'), Path(f'mask{i}'))
                 for i in range(n)]
        for s in probe.FIXED_PATCHES:
            if s['split'] == split:
                pairs[s['dataset_index']] = VolumePair(s['patient_id'], s['week'], Path('mri'), Path('mask'))
        class Dataset:
            def __len__(self): return len(self.pairs)
        result.append(Dataset())
        result[-1].__dict__.update(split=split, pairs=tuple(pairs), patch_shape_dhw=(64, 64, 64),
                                  seed=3710, train_patches_per_volume=1, train_foreground_probability=.5)
    return result


def payload(dataset, kind='standard'):
    from .. import train, train_improved
    from ..validate import _local_policies
    import nibabel
    import scipy
    directory = Path(train.__file__).parent
    sources = train_improved.SOURCE_FILES if kind == 'improved' else (
        'train.py', 'modules.py', 'dataset.py', 'patch_dataset.py', 'volume_io.py',
        'resampling.py', 'intensity_normalization.py')
    return {'schema_version': 1, 'completed_epochs': 10, 'total_steps': 1430,
            'epoch_result': {'epoch_completed': True, 'processed_steps': 143},
            'config': {'hyperparameters': copy.deepcopy(train.HYPERPARAMETERS if kind == 'standard'
                                                       else train_improved.HYPERPARAMETERS),
                       'source_sha256': {n: probe.file_sha256(directory / n) for n in sources},
                       'policies': _local_policies(), 'steps_per_epoch': 143,
                       'train_pairs': [[p.patient_id, p.week, str(p.mri_path), str(p.segmentation_path)]
                                       for p in dataset.pairs],
                       'versions': {'numpy': np.__version__, 'torch': str(torch.__version__),
                                    'nibabel': nibabel.__version__, 'scipy': scipy.__version__,
                                    'cuda_build': torch.version.cuda}}}


class ProbabilityTests(unittest.TestCase):
    def stats(self, probabilities, truth):
        p = torch.tensor(probabilities, dtype=torch.float32).reshape(1, 6, 1, 1, -1)
        target = torch.tensor(truth).reshape(1, 1, 1, -1).long()
        return probe.probability_statistics(p.log(), target)

    def test_known_channel_probability_rank_margin_nll(self):
        result = self.stats([[.1], [.4], [.2], [.1], [.1], [.1]], [2])
        row = result['classes'][2]; conditional = row['conditional']
        self.assertEqual(row['gt_voxels'], 1)
        self.assertEqual(conditional['argmax_recall'], 0)
        self.assertAlmostEqual(conditional['true_class_probability']['mean'], .2, places=6)
        self.assertEqual(conditional['true_class_rank_counts_1_to_6'], [0, 1, 0, 0, 0, 0])
        self.assertEqual(conditional['most_frequent_competing_winner'], 1)
        self.assertAlmostEqual(conditional['winner_minus_true_probability']['mean'], .2, places=6)
        self.assertAlmostEqual(conditional['true_minus_best_other_probability']['mean'], -.2, places=6)
        self.assertAlmostEqual(conditional['mean_negative_log_likelihood'], -np.log(.2), places=6)
        self.assertAlmostEqual(row['soft_dice'], (.4 + 1e-5) / (1.2 + 1e-5), places=6)

    def test_each_integer_class_indexes_its_own_channel(self):
        truth = torch.arange(6).reshape(1, 1, 1, 6)
        logits = torch.full((1, 6, 1, 1, 6), -5.).scatter_(1, truth.unsqueeze(1), 5.)
        result = probe.probability_statistics(logits, truth)
        self.assertEqual(result['predicted_class_counts'], [1] * 6)
        for row in result['classes']:
            self.assertEqual(row['conditional']['argmax_recall'], 1.)
            self.assertGreater(row['conditional']['true_class_probability']['mean'], .99)

    def test_absent_class_is_null_not_success(self):
        result = self.stats([[.1], [.5], [.1], [.1], [.1], [.1]], [1])
        row = result['classes'][5]
        self.assertFalse(row['evaluable_on_gt']); self.assertIsNone(row['conditional'])
        self.assertIsNone(row['soft_dice']); self.assertEqual(row['ce_contribution_all_patch'], 0.)
        self.assertLess(row['objective_soft_dice_score'], .001)  # Loss term still exists.

    def test_zero_argmax_can_have_nonzero_probability(self):
        row = self.stats([[.1], [.3], [.1], [.1], [.1], [.3]], [5])['classes'][5]
        self.assertEqual(row['predicted_voxels_all_patch'], 0)
        self.assertAlmostEqual(row['conditional']['true_class_probability']['median'], .3, places=6)
        self.assertEqual(row['conditional']['top_probability_tie_voxels'], 1)
        self.assertEqual(row['conditional']['rank_second_fraction'], 1.)

    def test_all_equal_ties_match_lowest_id_argmax(self):
        result = probe.probability_statistics(torch.zeros(1, 6, 1, 1, 6), torch.arange(6).reshape(1, 1, 1, 6))
        self.assertEqual(result['predicted_class_counts'], [6, 0, 0, 0, 0, 0])
        for label, row in enumerate(result['classes']):
            self.assertEqual(row['conditional']['true_class_rank_counts_1_to_6'][label], 1)

    def test_conditional_denominator_and_weighted_nll_reproduce_ce(self):
        logits = torch.randn(1, 6, 2, 3, 4, generator=torch.Generator().manual_seed(3))
        truth = torch.arange(24).reshape(1, 2, 3, 4) % 6
        result = probe.probability_statistics(logits, truth)
        self.assertEqual(result['patch_voxels'], 24)
        self.assertEqual([r['conditional']['denominator'] for r in result['classes']], [4] * 6)
        self.assertAlmostEqual(sum(r['ce_contribution_all_patch'] for r in result['classes']),
                               result['main_head_loss']['ce'], places=6)
        from ..train import combined_loss
        self.assertAlmostEqual(result['main_head_loss']['total'], float(combined_loss(logits, truth)[0]), places=6)

    def test_probability_sum_and_distribution_percentiles(self):
        logits = torch.zeros(1, 6, 1, 1, 5); logits[:, 5] = torch.arange(5)
        truth = torch.full((1, 1, 1, 5), 5, dtype=torch.long)
        result = probe.probability_statistics(logits, truth)
        self.assertLessEqual(result['softmax_sum_max_abs_error'], 1e-6)
        got = result['classes'][5]['conditional']['true_class_probability']
        expected = torch.softmax(logits, 1)[0, 5].flatten().double().numpy()
        np.testing.assert_allclose([got[k] for k in ('min', 'p10', 'median', 'p90', 'max')],
                                   np.percentile(expected, [0, 10, 50, 90, 100]), atol=1e-7)

    def test_stable_log_nll_even_when_softmax_underflows(self):
        logits = torch.zeros(1, 6, 1, 1, 1); logits[:, 5] = -1000
        result = probe.probability_statistics(logits, torch.full((1, 1, 1, 1), 5, dtype=torch.long))
        self.assertTrue(np.isfinite(result['classes'][5]['conditional']['mean_negative_log_likelihood']))
        json.dumps(result, allow_nan=False)

    def test_stats_deterministic_and_inputs_unchanged(self):
        logits = torch.randn(1, 6, 2, 3, 4); truth = torch.arange(24).reshape(1, 2, 3, 4) % 6
        before = logits.clone(), truth.clone()
        self.assertEqual(probe.probability_statistics(logits, truth), probe.probability_statistics(logits, truth))
        torch.testing.assert_close(logits, before[0], rtol=0, atol=0)
        torch.testing.assert_close(truth, before[1], rtol=0, atol=0)

    def test_invalid_shape_dtype_labels_or_nonfinite_rejected(self):
        logits = torch.zeros(1, 6, 2, 3, 4); truth = torch.zeros(1, 2, 3, 4, dtype=torch.long)
        cases = [(logits[:, :5], truth), (logits.double(), truth), (logits, truth.float()),
                 (logits, truth + 6), (logits, truth - 1), (logits + float('nan'), truth),
                 (logits[:, :, :0], truth[:, :0]), (logits, truth[:, :, :, :3])]
        for z, t in cases:
            with self.subTest(shape=z.shape), self.assertRaises(ValueError): probe.probability_statistics(z, t)


class SelectionTests(unittest.TestCase):
    def test_unrecognized_exposure_file_rejected_before_record_use(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'exposure.jsonl'; path.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'Exposure JSONL SHA-256'):
                probe.read_exposure(path)

    def test_fixed_dataset_indices_coverage_and_policy(self):
        train, validation = fake_datasets()
        probe.validate_dataset_selection(train, validation)
        train.pairs = train.pairs[::-1]
        with self.assertRaisesRegex(ValueError, 'index'): probe.validate_dataset_selection(train, validation)

    def test_test_split_and_changed_sampling_rejected(self):
        for field, value in [('split', 'test'), ('seed', 4), ('train_foreground_probability', 1.),
                             ('patch_shape_dhw', (32, 32, 32))]:
            train, validation = fake_datasets(); setattr(train, field, value)
            with self.subTest(field=field), self.assertRaises(ValueError):
                probe.validate_dataset_selection(train, validation)

    def test_validation_center_origin_and_labels(self):
        selected = probe.FIXED_PATCHES[2]; s = sample()
        result = probe.validate_fixed_sample(s, selected)
        self.assertEqual(result['voxel_counts'][0], 0)
        self.assertEqual(sum(result['voxel_counts']), 64**3)
        s['crop_origin_dhw'][0] += 1
        with self.assertRaisesRegex(ValueError, 'origin'): probe.validate_fixed_sample(s, selected)

    def test_train_mask_counts_and_hash_both_required(self):
        selected = dict(probe.FIXED_PATCHES[0]); s = sample(selected)
        selected['voxel_counts'] = probe.exposure.class_counts(s['mask'].numpy())
        selected['mask_sha256'] = hashlib.sha256(s['mask'].numpy().tobytes()).hexdigest()
        probe.validate_fixed_sample(s, selected)
        selected['mask_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'counts/hash'): probe.validate_fixed_sample(s, selected)

    def test_identity_sampling_shape_dtype_and_labels_rejected(self):
        for change in ({'patient_id': 'B037'}, {'sampling_mode': 'random'},
                       {'mri': torch.zeros(1, 32, 64, 64)}, {'mask': torch.zeros(64, 64, 64)},
                       {'mask': torch.full((64, 64, 64), 6)},
                       {'mri': torch.full((1, 64, 64, 64), float('inf'))}):
            s = sample(); s.update(change)
            with self.subTest(change=list(change)), self.assertRaises(ValueError):
                probe.validate_fixed_sample(s, probe.FIXED_PATCHES[2])

    def test_reconstruction_uses_actual_dataset_crop_and_one_preprocess(self):
        from ..patch_dataset import HipMRIPatchDataset
        from .. import patch_dataset as production
        selected = probe.FIXED_PATCHES[2]
        pair = VolumePair('B040', 0, Path('synthetic_mri'), Path('synthetic_mask'))
        x, y, z = np.indices((239, 239, 128))
        normalized = SimpleNamespace(mri=(x + y + z).astype(np.float32),
            segmentation=((x + y + z) % 5 + 1).astype(np.uint8), affine=np.eye(4),
            target_spacing_mm=(1., 1., 1.), spatial_unit='mm',
            segmentation_header_spatial_unit='unknown', segmentation_unit_inferred=True)
        with patch.object(production, 'discover_dataset', return_value={'validation': (pair,)}):
            dataset = HipMRIPatchDataset('validation')
        with patch.object(probe.exposure, 'production_preprocess', return_value=(normalized,
                          {'source_shape_xyz': [239, 239, 128]})) as preprocessing, \
                patch.object(probe, 'file_sha256', return_value='synthetic-source-hash'):
            got, metadata = probe.reconstruct_patch(dataset, selected, {})
        self.assertEqual(preprocessing.call_count, 1)
        self.assertEqual(got['crop_origin_dhw'].tolist(), [32, 87, 87])
        self.assertTrue(metadata['repeated_sampler_bitwise_equal'])
        np.testing.assert_array_equal(got['mask'].numpy(),
                                     normalized.segmentation.transpose(2, 1, 0)[32:96, 87:151, 87:151])


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.dataset = fake_datasets()[0]

    def test_complete_current_training_provenance_accepted(self):
        for kind in ('standard', 'improved'):
            self.assertEqual(probe.validate_checkpoint_config(payload(self.dataset, kind), kind, self.dataset)
                             ['hyperparameters']['model'], probe.MODEL_NAMES[kind])

    def test_wrong_model_epoch_or_update_count_rejected(self):
        for kind, changes in [('improved', {}), ('standard', {'completed_epochs': 9}),
                              ('standard', {'total_steps': 1429})]:
            p = payload(self.dataset); p.update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                probe.validate_checkpoint_config(p, kind, self.dataset)

    def test_all_source_hashes_including_train_and_sampler_required(self):
        for name in ('train.py', 'patch_dataset.py', 'modules.py', 'resampling.py'):
            p = payload(self.dataset); p['config']['source_sha256'][name] = '0' * 64
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'source hashes'):
                probe.validate_checkpoint_config(p, 'standard', self.dataset)

    def test_versions_policies_pair_order_and_hyperparameters_rejected(self):
        for category in ('versions', 'policies', 'train_pairs', 'hyperparameters'):
            p = payload(self.dataset)
            if category == 'versions': p['config'][category]['torch'] = 'other'
            elif category == 'policies': p['config'][category]['patch_dataset']['seed'] = 4
            elif category == 'train_pairs': p['config'][category].reverse()
            else: p['config'][category]['learning_rate'] = 4
            with self.subTest(category=category), self.assertRaises(ValueError):
                probe.validate_checkpoint_config(p, 'standard', self.dataset)

    def test_snapshot_hash_checked_before_deserialization(self):
        with patch.object(probe, 'file_sha256', return_value='wrong'), patch.object(torch, 'load') as loader:
            with self.assertRaisesRegex(ValueError, 'SHA-256'):
                probe.protected_load('unused', 'standard', self.dataset, expected_sha256='expected')
            loader.assert_not_called()

    def test_protected_load_delegates_cpu_weights_only_and_strict_loader(self):
        from .. import validate
        model = type('StandardUNet3D', (KnownModel,), {})()
        metadata = {'completed_epochs': 10, 'total_steps': 1430}
        with patch.object(torch, 'load', return_value=payload(self.dataset)) as read, \
                patch.object(probe, 'file_sha256', return_value='a' * 64), \
                patch.object(validate, 'load_checkpoint_model', return_value=(model, metadata)) as strict:
            result, _ = probe.protected_load('synthetic.pt', 'standard', self.dataset)
        self.assertIs(result, model)
        self.assertEqual(read.call_args.kwargs, {'map_location': 'cpu', 'weights_only': True})
        self.assertEqual(strict.call_args.args[1], torch.device('cpu'))

    def test_protected_loader_state_dict_missing_key_is_not_bypassed(self):
        from .. import validate
        p = payload(self.dataset); p['model'] = {}
        with patch.object(torch, 'load', return_value=p), patch.object(probe, 'file_sha256', return_value='a' * 64):
            with self.assertRaisesRegex(RuntimeError, 'Missing key'):
                probe.protected_load('synthetic.pt', 'standard', self.dataset)

    def test_both_actual_protected_loaders_accept_compatible_synthetic_state(self):
        from ..modules import StandardUNet3D
        from ..improved_modules import ImprovedUNet3D
        for kind, architecture in (('standard', StandardUNet3D), ('improved', ImprovedUNet3D)):
            original = architecture()
            fixture = payload(self.dataset, kind); fixture['model'] = original.state_dict()
            with patch.object(torch, 'load', return_value=fixture), \
                    patch.object(probe, 'file_sha256', return_value='a' * 64):
                actual, metadata = probe.protected_load('synthetic.pt', kind, self.dataset)
            self.assertIs(type(actual), architecture); self.assertFalse(actual.training)
            self.assertEqual(metadata['metadata']['total_steps'], 1430)
            self.assertTrue(all(torch.equal(v, actual.state_dict()[k]) for k, v in original.state_dict().items()))

    def test_checkpoint_comparison_reports_bytes_and_requires_equal_tensors(self):
        left, right = KnownModel(), KnownModel()
        info = {'path': 'snapshot', 'sha256': 'a' * 64}
        other = {'path': 'last', 'sha256': 'b' * 64}
        result = probe.compare_standard_checkpoints(left, right, info, other)
        self.assertFalse(result['file_bytes_identical']); self.assertTrue(result['state_tensors_identical'])
        with torch.no_grad(): right.weight.add_(1)
        with self.assertRaisesRegex(ValueError, 'different weights'):
            probe.compare_standard_checkpoints(left, right, info, other)


class ExecutionTests(unittest.TestCase):
    def mocked_main(self, run):
        from .. import patch_dataset as production
        from .. import train
        datasets = fake_datasets()
        standard, improved, last = KnownModel(), KnownModel(), KnownModel()
        config = payload(datasets[0])['config']
        infos = [dict(path='snapshot', sha256='a' * 64, config=config),
                 dict(path='improved', sha256='b' * 64, config={'model': 'synthetic'}),
                 dict(path='last', sha256='a' * 64, config=config)]
        stdout = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fixture'; path.write_text('synthetic input')
            flags = ['--exposure-log', str(path), '--standard-checkpoint', str(path),
                     '--standard-validation-checkpoint', str(path), '--improved-checkpoint', str(path),
                     '--run-six-forwards' if run else '--preflight-only']
            with patch.dict('os.environ', {'SLURM_JOB_ID': 'synthetic'}), \
                    patch.object(torch.cuda, 'is_available', return_value=True), \
                    patch.object(torch.cuda, 'get_device_name', return_value='synthetic GPU'), \
                    patch.object(torch.cuda, 'empty_cache'), \
                    patch.object(train, 'configure_reproducibility'), \
                    patch.object(probe, 'read_exposure', return_value={
                        'sha256': 'synthetic', 'record_counts': [1, 143, 1430, 1], 'setup': {'config': config}}), \
                    patch.object(production, 'HipMRIPatchDataset', side_effect=datasets), \
                    patch.object(probe, 'protected_load', side_effect=zip((standard, improved, last), infos)), \
                    patch.object(probe, 'reconstruct_patch', side_effect=lambda ds, s, ev: (sample(s), {'case': s['case']})) as reconstruct, \
                    patch.object(probe, 'infer_once', return_value={'statistics': {'synthetic': True}}) as infer, \
                    patch.object(torch, 'save', side_effect=AssertionError('write prohibited')), \
                    contextlib.redirect_stdout(stdout):
                self.assertEqual(probe.main(flags), 0)
            rows = [json.loads(line) for line in stdout.getvalue().splitlines()]
            return rows, reconstruct.call_count, infer.call_count

    def test_main_flow_exactly_three_patches_six_model_calls(self):
        rows, reconstructions, calls = self.mocked_main(True)
        self.assertEqual((reconstructions, calls), (3, 6))
        probabilities = [r for r in rows if r['event'] == 'probability']
        self.assertEqual([(r['case'], r['model']) for r in probabilities], [
            (s['case'], m) for s in probe.FIXED_PATCHES for m in probe.MODEL_NAMES.values()])
        self.assertEqual(rows[-1]['main_head_forward_passes'], 6)
        self.assertEqual(rows[-1]['optimizer_updates'], 0)
        self.assertEqual(rows[-1]['backward_passes'], 0)

    def test_main_preflight_has_no_array_reconstruction_or_forward(self):
        rows, reconstructions, calls = self.mocked_main(False)
        self.assertEqual((reconstructions, calls), (0, 0))
        self.assertEqual(rows[-1]['real_arrays_decoded'], 0)

    def test_one_forward_no_backward_optimizer_save_or_input_change(self):
        model = KnownModel(); s = sample(); before = model.weight.detach().clone(), s['mri'].clone(), s['mask'].clone()
        with patch.object(torch.Tensor, 'backward', side_effect=AssertionError('backward prohibited')), \
                patch.object(torch.optim, 'AdamW', side_effect=AssertionError('optimizer prohibited')), \
                patch.object(torch.optim, 'SGD', side_effect=AssertionError('optimizer prohibited')), \
                patch.object(torch, 'save', side_effect=AssertionError('save prohibited')):
            result = probe.infer_once(model, s, torch.device('cpu'))
        self.assertEqual(model.calls, 1); self.assertIsNone(model.weight.grad)
        self.assertTrue(result['repeated_statistics_equal']); self.assertIsNone(result['gpu_memory'])
        for actual, expected in zip((model.weight, s['mri'], s['mask']), before):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_improved_called_without_auxiliary_flag(self):
        from ..improved_modules import ImprovedUNet3D
        model = ImprovedUNet3D(base_channels=1)
        s = {'mri': torch.randn(1, 16, 16, 16), 'mask': torch.zeros(16, 16, 16, dtype=torch.long)}
        with patch.object(model, 'forward', wraps=model.forward) as forward:
            result = probe.infer_once(model, s, torch.device('cpu'))
        forward.assert_called_once(); self.assertEqual(forward.call_args.kwargs, {})
        self.assertEqual(result['statistics']['patch_voxels'], 16**3)

    def test_invalid_model_output_rejected(self):
        model = Mock(); model.to.return_value = model; model.eval.return_value = model
        for output in (torch.zeros(1, 5, 64, 64, 64), (torch.zeros(1, 6, 64, 64, 64), ())):
            model.return_value = output
            with self.assertRaisesRegex(ValueError, 'main logits'): probe.infer_once(model, sample(), torch.device('cpu'))

    def test_cli_requires_explicit_scope_and_has_no_case_or_test_flags(self):
        p = probe.parser()
        flags = ['--exposure-log', 'e', '--standard-checkpoint', 's', '--standard-validation-checkpoint', 'l',
                 '--improved-checkpoint', 'i']
        with contextlib.redirect_stderr(io.StringIO()):
            for extra in ([], ['--preflight-only', '--run-six-forwards'], ['--preflight-only', '--case', 'B037_Week0']):
                with self.subTest(extra=extra), self.assertRaises(SystemExit): p.parse_args(flags + extra)
        self.assertTrue(p.parse_args(flags + ['--preflight-only']).preflight_only)

    def test_missing_cli_input_fails_before_discovery(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = str(Path(directory) / 'missing')
            with self.assertRaises(FileNotFoundError):
                probe.main(['--exposure-log', missing, '--standard-checkpoint', missing,
                            '--standard-validation-checkpoint', missing, '--improved-checkpoint', missing,
                            '--preflight-only'])

    def test_real_mode_refuses_unallocated_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'synthetic'; path.write_text('fixture')
            with patch.dict('os.environ', {}, clear=True), patch.object(torch.cuda, 'is_available', return_value=False):
                with self.assertRaisesRegex(RuntimeError, 'allocated CUDA'):
                    probe.main(['--exposure-log', str(path), '--standard-checkpoint', str(path),
                                '--standard-validation-checkpoint', str(path), '--improved-checkpoint', str(path),
                                '--run-six-forwards'])


if __name__ == '__main__':
    unittest.main()
