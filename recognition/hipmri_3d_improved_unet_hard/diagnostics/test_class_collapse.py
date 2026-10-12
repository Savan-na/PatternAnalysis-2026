"""CPU synthetic diagnosis tests; no real MRI, optimisation or CUDA execution.

Sampler tests mock discovery/preprocessing and call the actual Dataset. A
local runner may stub a missing NiBabel import; the stub cannot decode MRI.
"""

import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader

from .. import train, train_improved
from ..dataset import APPROVED_PATIENTS, VolumePair
from ..validate import CLASS_NAMES, confusion_matrix, metrics_from_confusion
from . import class_collapse as audit


def independent_loss(logits, target):
    """Independent log-softmax CE and vectorised one-hot Dice reference."""
    log_probability = logits.log_softmax(1)
    ce = -log_probability.gather(1, target.unsqueeze(1)).mean()
    truth = torch.nn.functional.one_hot(target, 6).movedim(-1, 1).float()[:, 1:]
    p = log_probability.exp()[:, 1:]
    axes = (0, 2, 3, 4)
    dice = (2 * (p * truth).sum(axes) + 1e-5) / (p.sum(axes) + truth.sum(axes) + 1e-5)
    dice_loss = 1 - dice.mean()
    return .5 * ce + .5 * dice_loss, ce, dice_loss


class LossAndChannelTests(unittest.TestCase):
    def test_manual_values_and_gradients_match_production(self):
        target = torch.arange(120).reshape(2, 3, 4, 5) % 6
        a = torch.randn(2, 6, 3, 4, 5, generator=torch.Generator().manual_seed(3710), requires_grad=True)
        b = a.detach().clone().requires_grad_()
        actual, expected = train.combined_loss(a, target), independent_loss(b, target)
        for x, y in zip(actual, expected):
            torch.testing.assert_close(x, y, rtol=1e-6, atol=1e-7)
        actual[0].backward(); expected[0].backward()
        torch.testing.assert_close(a.grad, b.grad, rtol=2e-5, atol=2e-8)

    def test_all_six_correct_channels_beat_shifted_channels(self):
        target = torch.arange(6).reshape(1, 1, 2, 3)
        correct = torch.full((1, 6, 1, 2, 3), -7.)
        correct.scatter_(1, target.unsqueeze(1), 7.)
        self.assertLess(train.combined_loss(correct, target)[0].item(), .0001)
        self.assertGreater(train.combined_loss(correct.roll(1, 1), target)[0].item(), 5.)
        self.assertTrue(torch.equal(correct.argmax(1), target))
        self.assertTrue(bool(torch.all(torch.nn.functional.cross_entropy(correct, target,
                                                                         reduction='none') < .0001)))

    def test_present_bones_rectum_prostate_have_corrective_gradients(self):
        for label in (2, 4, 5):
            with self.subTest(label=label):
                target = torch.full((1, 2, 3, 4), label, dtype=torch.long)
                logits = torch.zeros(1, 6, 2, 3, 4); logits[:, 1] = 3.
                logits.requires_grad_(); train.combined_loss(logits, target)[0].backward()
                self.assertTrue(bool(torch.isfinite(logits.grad).all()))
                self.assertLess(logits.grad[:, label].mean().item(), 0.)
                self.assertGreater(logits.grad[:, 1].mean().item(), 0.)

    def test_absent_class_dice_floor_and_weak_probability_gradient(self):
        target = torch.ones(1, 16, 16, 16, dtype=torch.long)
        logits = torch.zeros(1, 6, 16, 16, 16, requires_grad=True)
        mass = 4096 / 6
        absent_score = 1e-5 / (mass + 1e-5)
        present_score = (2 * mass + 1e-5) / (mass + 4096 + 1e-5)
        self.assertAlmostEqual(train.combined_loss(logits, target)[2].item(),
                               1 - (present_score + 4 * absent_score) / 5, places=6)
        p = torch.full((4096,), 1 / 6, dtype=torch.float64, requires_grad=True)
        contribution = (1 - 1e-5 / (p.sum() + 1e-5)) / 5
        contribution.backward()
        self.assertAlmostEqual(contribution.item(), .2, places=7)
        self.assertAlmostEqual(p.grad[0].item(), 1e-5 / (5 * (mass + 1e-5) ** 2), delta=1e-20)
        self.assertLess(p.grad.max().item(), 1e-10)
        train.combined_loss(logits, target)[1].backward()
        self.assertGreater(logits.grad[:, 5].mean().item(), 1e-6)  # CE still penalises false positives.

    def test_auxiliary_weighting_and_gradients_with_unchanged_integer_target(self):
        generator = torch.Generator().manual_seed(7)
        target = torch.arange(4 * 8 * 12).reshape(1, 4, 8, 12) % 6
        arrays = [torch.randn(1, 6, 4 // s, 8 // s, 12 // s, generator=generator, requires_grad=True)
                  for s in (1, 2, 4)]
        before = target.clone()
        actual = train_improved.combined_loss((arrays[0], tuple(arrays[1:])), target)
        terms = [independent_loss(arrays[0], target)] + [independent_loss(
            torch.nn.functional.interpolate(a, size=target.shape[1:], mode='trilinear', align_corners=False),
            target) for a in arrays[1:]]
        for index, value in enumerate(actual):
            expected = sum(w * t[index] for w, t in zip((1., .5, .25), terms)) / 1.75
            torch.testing.assert_close(value, expected, rtol=1e-6, atol=1e-7)
        actual[0].backward()
        self.assertTrue(all(bool(torch.isfinite(a.grad).all()) and a.grad.abs().sum() > 0 for a in arrays))
        self.assertTrue(torch.equal(before, target))

    def test_metric_rows_and_confusion_follow_same_class_indices(self):
        truth = np.arange(6, dtype=np.uint8).reshape(1, 2, 3)
        rows = metrics_from_confusion(confusion_matrix(truth.copy(), truth))
        self.assertEqual(tuple(r['structure'] for r in rows), CLASS_NAMES)
        for i, row in enumerate(rows):
            self.assertEqual(row['class_id'], i); self.assertEqual(row['dice'], 1.)
        prediction = truth.copy(); prediction[prediction == 5] = 1
        counts = confusion_matrix(prediction, truth)
        self.assertEqual(counts[5, 1], 1)
        self.assertEqual(metrics_from_confusion(counts)[5]['dice'], 0.)

    def test_invalid_labels_rejected_instead_of_remapped(self):
        for label in (-1, 6):
            with self.subTest(label=label), self.assertRaisesRegex(ValueError, 'outside 0-5'):
                train.combined_loss(torch.zeros(1, 6, 2, 2, 2),
                                    torch.full((1, 2, 2, 2), label, dtype=torch.long))


class ExposureTests(unittest.TestCase):
    def setUp(self):
        from .. import patch_dataset
        self.production = patch_dataset
        self.pairs = tuple(VolumePair(p, w, Path(f'{p}_Week{w}_LFOV.nii.gz'),
                                     Path(f'{p}_Week{w}_SEMANTIC.nii.gz'))
                           for p, w in (('K019', 1), ('S035', 0), ('B006', 0)))
        self.discovery = patch.object(patch_dataset, 'discover_dataset', return_value={'train': self.pairs})
        self.discovery.start(); self.addCleanup(self.discovery.stop)
        xyz = np.indices((24, 28, 32))
        self.mask = ((xyz[0] + xyz[1] + xyz[2]) % 5).astype(np.uint8)
        self.mask[9:11, 12:14, 15:18] = 5
        self.normalized = SimpleNamespace(mri=(xyz[0] + xyz[1] + xyz[2]).astype(np.float32),
            segmentation=self.mask, affine=np.diag([1.6, 1.6, 1.5, 1.]), spatial_unit='mm',
            segmentation_header_spatial_unit='unknown', segmentation_unit_inferred=True)
        self.calls = []

    def dataset(self, **kwargs):
        return self.production.HipMRIPatchDataset('train', Path('synthetic_root'),
                                                 patch_shape_dhw=(16, 16, 16), **kwargs)

    def preprocess(self, pair, dataset):
        self.calls.append((pair.patient_id, pair.week))
        return self.normalized, {'source_shape_xyz': list(self.mask.shape),
            'source_voxel_counts': audit.class_counts(self.mask), 'source_voxel_volume_mm3': 3.84}

    def records(self, **kwargs):
        return list(audit.exposure_records(self.dataset(**kwargs), self.preprocess))

    def test_exact_repetition_and_epoch_zero_through_nine(self):
        before = self.mask.copy(); a, b = self.records(), self.records()
        self.assertEqual(a, b); self.assertEqual(len(self.calls), 6)
        patches = [r for r in a if r['event'] == 'patch']
        self.assertEqual({r['sampler_epoch'] for r in patches}, set(range(10)))
        self.assertEqual({r['completed_epoch'] for r in patches}, set(range(1, 11)))
        np.testing.assert_array_equal(before, self.mask)

    def test_reconstruction_matches_shuffled_actual_dataloader(self):
        dataset = self.dataset()
        replay = {(r['dataset_index'], r['sampler_epoch']): r for r in self.records() if r['event'] == 'patch'}
        with patch.object(self.production, 'load_volume_pair', return_value=None), \
                patch.object(self.production, 'resample_volume_pair', return_value=None), \
                patch.object(self.production, 'normalize_volume_pair', return_value=self.normalized):
            loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0,
                                generator=torch.Generator().manual_seed(3710))
            for epoch in range(10):
                dataset.set_epoch(epoch)
                for batch in loader:
                    index = [p.patient_id for p in self.pairs].index(batch['patient_id'][0])
                    expected = replay[index, epoch]
                    self.assertEqual(batch['crop_origin_dhw'][0].tolist(), expected['crop_origin_dhw'])
                    self.assertEqual(audit.class_counts(batch['mask'][0].numpy()), expected['voxel_counts'])
                    self.assertEqual(hashlib.sha256(batch['mask'][0].numpy().tobytes()).hexdigest(),
                                     expected['mask_sha256'])

    def test_changed_seed_changes_crops(self):
        a, b = self.records(seed=3710), self.records(seed=3711)
        self.assertNotEqual([r.get('crop_origin_dhw') for r in a], [r.get('crop_origin_dhw') for r in b])

    def test_foreground_presence_is_not_whole_structure_coverage(self):
        self.mask.fill(1); self.mask[0, 0, 0] = 5; self.mask[-1, -1, -1] = 5
        patches = [r for r in self.records(train_foreground_probability=1.) if r['event'] == 'patch']
        self.assertTrue(all(r['sampling_mode'] == 'prostate' and r['voxel_counts'][5] == 1 for r in patches))

    def test_absent_prostate_random_fallback(self):
        self.mask[self.mask == 5] = 1
        patches = [r for r in self.records(train_foreground_probability=1.) if r['event'] == 'patch']
        self.assertTrue(all(r['sampling_mode'] == 'random_fallback' and r['voxel_counts'][5] == 0 for r in patches))

    def test_raw_full_and_patch_denominators_are_separate(self):
        summary = audit.summarize_exposure(self.records(), 3)
        self.assertEqual(summary['status'], 'PASS')
        self.assertEqual(summary['source_full']['voxel_denominator'], 3 * self.mask.size)
        self.assertEqual(summary['resampled_full']['voxel_denominator'], 3 * self.mask.size)
        self.assertEqual(summary['all_patches']['voxel_denominator'], 30 * 16 ** 3)
        self.assertAlmostEqual(sum(summary['all_patches']['voxel_fractions']), 1.)
        self.assertTrue(all(r['patches'] == 3 for r in summary['epochs']))
        self.assertEqual(len(summary['patients']), 3)

    def test_class_presence_patient_counts_and_fraction_distributions(self):
        summary = audit.summarize_exposure(self.records(train_foreground_probability=1.), 3)
        self.assertEqual(summary['resampled_full']['volumes_present'], [3] * 6)
        self.assertEqual(summary['resampled_full']['patients_present'], [3] * 6)
        patch = summary['all_patches']
        self.assertEqual(patch['positive_patch_fractions'][5], 1.)
        self.assertEqual(patch['positive_voxel_fraction_min_q25_median_q75_max'][5][-1],
                         max(r['voxel_counts'][5] / 16 ** 3 for r in self.records(
                             train_foreground_probability=1.) if r['event'] == 'patch'))
        self.assertEqual(sum(patch['largest_class_patch_counts']), 30)

    def test_background_body_dominance_and_branch_counts(self):
        self.mask.fill(1); self.mask[9, 12, 15] = 5
        summary = audit.summarize_exposure(self.records(train_foreground_probability=1.), 3)
        self.assertEqual(summary['all_patches']['background_or_body_over_half_patches'], 30)
        self.assertEqual(summary['all_patches']['largest_class_patch_counts'][1], 30)
        self.assertEqual(summary['branches'][0]['sampling_mode'], 'prostate')
        self.assertEqual(summary['branches'][0]['patches_present'][5], 30)
        self.assertIsNone(summary['branches'][0]['positive_voxel_fraction_min_q25_median_q75_max'][2])

    def test_duplicate_missing_and_corrupt_patch_records(self):
        records = self.records()
        self.assertEqual(audit.summarize_exposure(records[:-1], 3)['status'], 'INCOMPLETE')
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            audit.summarize_exposure(records + [records[1]], 3)
        for field, value, message in (('voxel_counts', [1] * 6, 'class counts'),
                                     ('patient_id', 'B040', 'identity'), ('sampler_epoch', 1, 'minus one')):
            changed = copy.deepcopy(records); changed[1][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, message):
                audit.summarize_exposure(changed, 3)

    def test_train_only_and_no_test_patient(self):
        dataset = self.dataset(); dataset.split = 'validation'
        with self.assertRaisesRegex(ValueError, 'train-only'):
            list(audit.exposure_records(dataset, self.preprocess))
        dataset.split = 'train'; dataset.pairs = (VolumePair('B037', 0, Path('x'), Path('y')),)
        with self.assertRaisesRegex(ValueError, 'non-training'):
            list(audit.exposure_records(dataset, self.preprocess))
        self.assertEqual(self.calls, [])

    def test_selected_subset_preserves_original_indices_and_is_not_full_coverage(self):
        dataset = self.dataset()
        indices = audit.select_indices(dataset, cases=['S035_Week0'])
        self.assertEqual(indices, (1,))
        records = list(audit.exposure_records(dataset, self.preprocess, indices=indices))
        self.assertTrue(all(r['dataset_index'] == 1 for r in records))
        self.assertEqual(audit.summarize_exposure(records, 1)['status'], 'PASS')
        self.assertEqual(audit.summarize_exposure(records, 143)['status'], 'INCOMPLETE')

    def test_scope_is_explicit_and_rejects_other_splits_or_duplicate_cases(self):
        dataset = self.dataset()
        self.assertEqual(audit.select_indices(dataset, max_cases=2), (0, 1))
        self.assertEqual(audit.select_indices(dataset, full_train=True), (0, 1, 2))
        for kwargs in ({}, {'max_cases': 0}, {'cases': ['B040_Week0']},
                       {'cases': ['K019_Week1', 'K019_Week1']}, {'full_train': True, 'max_cases': 1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                audit.select_indices(dataset, **kwargs)

    def test_production_preprocess_delegates_and_records_source_counts(self):
        from .. import volume_io, resampling, intensity_normalization
        dataset = self.dataset()
        loaded = SimpleNamespace(segmentation=self.mask, mri_affine=self.normalized.affine)
        intermediate = object()
        with patch.object(volume_io, 'load_volume_pair', return_value=loaded) as load, \
                patch.object(resampling, 'resample_volume_pair', return_value=intermediate) as resample, \
                patch.object(intensity_normalization, 'normalize_volume_pair', return_value=self.normalized) as norm:
            result, source = audit.production_preprocess(self.pairs[0], dataset)
            self.assertIs(result, self.normalized)
            self.assertEqual(source['source_voxel_counts'], audit.class_counts(self.mask))
            self.assertAlmostEqual(source['source_voxel_volume_mm3'], 3.84)
            load.assert_called_once_with(self.pairs[0])
            resample.assert_called_once_with(loaded, dataset.target_spacing_mm,
                                            max_estimated_bytes=dataset.max_estimated_bytes)
            norm.assert_called_once_with(intermediate)

    def test_cli_invalid_bound_fails_before_any_discovery(self):
        with patch.object(self.production, 'HipMRIPatchDataset') as constructor, \
                patch('sys.stderr', new_callable=io.StringIO):
            with self.assertRaises(SystemExit) as caught:
                audit.main(['patch-exposure', '--checkpoint', 'synthetic.pt', '--dataset-root', 'synthetic',
                            '--max-cases', '0'])
            self.assertEqual(caught.exception.code, 2)
            constructor.assert_not_called()


class ProvenanceTests(unittest.TestCase):
    def fixture(self):
        directory = Path(train.__file__).parent
        names = ('train.py', 'modules.py', 'dataset.py', 'patch_dataset.py', 'volume_io.py',
                 'resampling.py', 'intensity_normalization.py')
        patients = APPROVED_PATIENTS['train']
        pairs = tuple(VolumePair(patients[i % 26], i // 26, Path(f'mri/{i}'), Path(f'mask/{i}'))
                      for i in range(143))
        cls = type('Synthetic143Dataset', (), {'__len__': lambda self: len(self.pairs)})
        dataset = cls()
        dataset.__dict__.update(split='train', pairs=pairs, patch_shape_dhw=(64, 64, 64),
            seed=3710, train_patches_per_volume=1, train_foreground_probability=.5)
        config = {'hyperparameters': copy.deepcopy(train.HYPERPARAMETERS), 'steps_per_epoch': 143,
            'source_sha256': {n: audit._file_hash(directory / n) for n in names},
            'policies': {n: json.loads((directory / (n + '_config.json')).read_text())
                         for n in ('resampling', 'normalization', 'patch_dataset')},
            'versions': {'numpy': np.__version__},
            'train_pairs': [[p.patient_id, p.week, str(p.mri_path), str(p.segmentation_path)] for p in pairs]}
        return config, dataset

    def test_exact_configuration_passes(self):
        config, dataset = self.fixture(); audit.validate_replay_provenance(config, dataset)

    def test_hash_pair_order_policy_and_numpy_mismatch_rejected(self):
        for field, message in (('source_sha256', 'source hashes'), ('train_pairs', 'pair order'),
                               ('policies', 'policies'), ('versions', 'NumPy version')):
            config, dataset = self.fixture()
            if field == 'train_pairs': config[field].reverse()
            else: config[field] = {}
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, message):
                audit.validate_replay_provenance(config, dataset)

    def test_no_silent_hyperparameter_change(self):
        config, dataset = self.fixture(); config['hyperparameters']['dice_smoothing'] = 1.
        with self.assertRaisesRegex(ValueError, 'hyperparameters'):
            audit.validate_replay_provenance(config, dataset)

    def test_cpu_weights_only_protected_loader_delegation_for_both_models(self):
        import nibabel, scipy
        from .. import validate, validate_improved
        for name, protected in (('StandardUNet3D', validate), ('ImprovedUNet3D', validate_improved)):
            config = {'hyperparameters': {'model': name},
                      'versions': {'nibabel': 'synthetic-import-only', 'scipy': scipy.__version__}}
            payload = {'completed_epochs': 10, 'total_steps': 1430, 'config': config}
            with self.subTest(model=name), \
                    patch.object(nibabel, '__version__', 'synthetic-import-only', create=True), \
                    patch.object(audit.torch, 'load', return_value=payload) as reader, \
                    patch.object(audit, 'validate_replay_provenance') as provenance, \
                    patch.object(protected, 'load_checkpoint_model', return_value=(object(), {})) as loader, \
                    patch.object(audit, '_file_hash', return_value='synthetic-checkpoint-hash'):
                self.assertEqual(audit.load_replay_checkpoint(Path('synthetic.pt'), object())['model'], name)
                reader.assert_called_once_with(Path('synthetic.pt'), map_location='cpu', weights_only=True)
                provenance.assert_called_once()
                loader.assert_called_once_with(Path('synthetic.pt'), torch.device('cpu'))

    def test_partial_checkpoint_rejected_before_loader(self):
        with patch.object(audit.torch, 'load', return_value={'completed_epochs': 9, 'total_steps': 1287}), \
                patch.object(audit, 'validate_replay_provenance') as validator:
            with self.assertRaisesRegex(ValueError, 'Epoch 10 checkpoint'):
                audit.load_replay_checkpoint(Path('synthetic.pt'), object())
            validator.assert_not_called()


class LossHistoryTests(unittest.TestCase):
    def setup_record(self, model='StandardUNet3D'):
        return {'event': 'setup', 'config': {'hyperparameters': {'model': model}, 'steps_per_epoch': 143}}

    def epoch_record(self, e):
        return {'event': 'epoch', 'completed_epochs': e, 'total_steps': e * 143, 'epoch_completed': True,
                'processed_steps': 143, 'expected_steps': 143,
                'mean_loss': .75, 'mean_ce': .7, 'mean_soft_dice_loss': .8}

    def parse(self, records):
        with tempfile.TemporaryDirectory(prefix='collapse-synthetic-log-') as directory:
            path = Path(directory) / 'synthetic.jsonl'
            path.write_text('\n'.join(json.dumps(r) for r in records))
            return audit.summarize_loss_logs([path])

    def test_complete_history_and_step_means_verified(self):
        steps = [{'event': 'step', 'step': s, 'loss': .75, 'ce': .7, 'soft_dice_loss': .8}
                 for s in range(1, 144)]
        result = self.parse([self.setup_record(), *steps, *[self.epoch_record(e) for e in range(1, 11)]])
        self.assertEqual(result['status'], 'PASS'); self.assertEqual(result['epochs'][0]['steps_verified'], 143)
        self.assertEqual(result['epochs'][1]['steps_verified'], 0)  # Summary only, not invented step evidence.

    def test_missing_history_remains_incomplete(self):
        result = self.parse([self.setup_record(), self.epoch_record(1)])
        self.assertEqual(result['status'], 'INCOMPLETE'); self.assertEqual(result['missing_epochs'], list(range(2, 11)))

    def test_invalid_loss_partial_duplicate_or_out_of_order_records_rejected(self):
        epoch = self.epoch_record(1)
        step = {'event': 'step', 'step': 1, 'loss': .75, 'ce': .7, 'soft_dice_loss': .8}
        cases = ([self.setup_record(), epoch, epoch], [self.setup_record(), step, epoch],
                 [self.setup_record(), step, self.setup_record()],
                 [self.setup_record(), dict(epoch, mean_loss=float('nan'))],
                 [self.setup_record(), dict(epoch, mean_loss=.1)], [self.setup_record(), dict(step, step=2)])
        for records in cases:
            with self.subTest(records=records), self.assertRaises(ValueError): self.parse(records)

    def test_models_or_changed_configs_cannot_be_spliced(self):
        for setup in (self.setup_record('ImprovedUNet3D'),
                      {'event': 'setup', 'config': {'hyperparameters': {'model': 'StandardUNet3D', 'seed': 19}}}):
            with self.subTest(setup=setup), self.assertRaisesRegex(ValueError, 'configurations differ'):
                self.parse([self.setup_record(), self.epoch_record(1), setup, self.epoch_record(2)])

    def test_malformed_json_is_not_silently_ignored(self):
        with tempfile.TemporaryDirectory(prefix='collapse-synthetic-log-') as directory:
            path = Path(directory) / 'synthetic.jsonl'; path.write_text('{broken')
            with self.assertRaisesRegex(ValueError, 'malformed JSON'): audit.summarize_loss_logs([path])

    def test_invalid_class_count_input_rejected(self):
        for mask in (np.zeros((2, 2, 2), dtype=np.float32), np.full((2, 2, 2), 6, dtype=np.uint8)):
            with self.subTest(dtype=mask.dtype), self.assertRaisesRegex(ValueError, 'integer mask'):
                audit.class_counts(mask)


if __name__ == '__main__':
    unittest.main()
