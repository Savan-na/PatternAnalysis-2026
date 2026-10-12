"""CPU-only V2 objective, initialization, lifecycle and compatibility regression."""
import copy
import json
import random
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from . import train as baseline
from . import train_improved as original
from . import train_improved_v2 as v2
from . import validate as evaluator
from . import validate_improved as original_validation
from . import validate_improved_v2 as validation
from .dataset import APPROVED_PATIENTS, VolumePair
from .improved_modules import ImprovedUNet3D


def manifest():
    patients = sorted(APPROVED_PATIENTS['train'])
    return [[patients[i % 26], i // 26, f'/synthetic/{i}_LFOV.nii.gz',
             f'/synthetic/{i}_SEMANTIC.nii.gz'] for i in range(143)]


def config():
    result = {'experiment_id': v2.EXPERIMENT_ID,
              'hyperparameters': copy.deepcopy(v2.HYPERPARAMETERS),
              'policies': evaluator._local_policies(), 'steps_per_epoch': 143,
              'source_sha256': v2.source_hashes(), 'train_pairs': manifest()}
    result['config_sha256'] = v2.config_hash(result)
    return result


class V2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        v2.configure_reproducibility(3710, torch.device('cpu'))

    def logits_target(self):
        logits = torch.randn(2, 6, 4, 4, 4, requires_grad=True)
        target = torch.arange(128).reshape(2, 4, 4, 4).long() % 6
        return logits, target

    def batch(self):
        return {'mri': torch.randn(1, 1, 16, 16, 16),
                'mask': torch.arange(4096).reshape(1, 16, 16, 16).long() % 6}

    def test_weights_from_counts_and_empirical_normalization(self):
        settings = v2.experiment_settings()
        f = np.array(settings['voxel_counts'], dtype=np.float64)
        self.assertEqual(int(f.sum()), 1430 * 64**3)
        f /= f.sum()
        expected = f**(-.5) / np.sum(f * f**(-.5))
        np.testing.assert_allclose(v2.CLASS_WEIGHTS, expected, rtol=1e-14)
        self.assertAlmostEqual(float(np.dot(f, v2.CLASS_WEIGHTS)), 1., places=14)
        bad = copy.deepcopy(settings)
        bad['class_weights'][5] *= 2
        with patch.object(Path, 'read_text', return_value=json.dumps(bad)):
            with self.assertRaisesRegex(ValueError, 'weights'):
                v2.experiment_settings()

    def test_weighted_ce_value_and_analytic_gradient_voxel_denominator(self):
        logits, target = self.logits_target()
        # Make the class histogram deliberately nonuniform.
        target[:, :2] = 1
        weights = torch.tensor(v2.CLASS_WEIGHTS, dtype=torch.float32)
        actual = v2.head_loss(logits, target)[1]
        expected = -(logits.log_softmax(1).gather(1, target[:, None]).squeeze(1)
                     * weights[target]).sum() / target.numel()
        torch.testing.assert_close(actual, expected)
        gradient = torch.autograd.grad(actual, logits)[0]
        truth = F.one_hot(target, 6).movedim(-1, 1)
        analytic = (logits.detach().softmax(1) - truth) * weights[target][:, None] / target.numel()
        torch.testing.assert_close(gradient, analytic, rtol=2e-5, atol=1e-8)
        self.assertFalse(torch.isclose(actual, F.cross_entropy(logits, target, weight=weights)))

    def test_all_ones_complete_loss_and_gradient_equivalence(self):
        logits, target = self.logits_target()
        result = v2.head_loss(logits, target, class_weights=(1.,) * 6)
        expected = baseline.combined_loss(logits, target)
        for actual, reference in zip(result, expected):
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        ga = torch.autograd.grad(result[0], logits, retain_graph=True)[0]
        gb = torch.autograd.grad(expected[0], logits)[0]
        torch.testing.assert_close(ga, gb, rtol=0, atol=0)

    def test_original_dice_value_and_gradient_including_absent_classes(self):
        logits, target = self.logits_target()
        target[:] = 1
        actual = v2.head_loss(logits, target)[2]
        expected = baseline.combined_loss(logits, target)[2]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        ga = torch.autograd.grad(actual, logits, retain_graph=True)[0]
        gb = torch.autograd.grad(expected, logits)[0]
        torch.testing.assert_close(ga, gb, rtol=0, atol=0)

    def test_complete_deep_supervision_loss_and_all_head_gradients(self):
        heads = [torch.randn(1, 6, s, s, s, requires_grad=True) for s in (8, 4, 2)]
        target = torch.arange(512).reshape(1, 8, 8, 8).long() % 6
        result = v2.combined_loss((heads[0], heads[1:]), target)
        weights = torch.tensor(v2.CLASS_WEIGHTS)
        independent = []
        for index, head in enumerate(heads):
            logits = head if index == 0 else F.interpolate(head, size=(8, 8, 8), mode='trilinear', align_corners=False)
            ce = (-logits.log_softmax(1).gather(1, target[:, None]).squeeze(1) * weights[target]).mean()
            dice = baseline.combined_loss(logits, target)[2]
            independent.append(.5 * ce + .5 * dice)
        reference = sum(w * loss for w, loss in zip((1., .5, .25), independent)) / 1.75
        torch.testing.assert_close(result[0], reference)
        actual_grad = torch.autograd.grad(result[0], heads, retain_graph=True)
        ref_grad = torch.autograd.grad(reference, heads)
        for actual, expected in zip(actual_grad, ref_grad):
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=1e-8)
            self.assertTrue(torch.isfinite(actual).all())
            self.assertGreater(actual.abs().sum().item(), 0)

    def test_all_ones_deep_supervision_matches_original(self):
        model = ImprovedUNet3D(base_channels=1)
        batch = self.batch()
        output = model(batch['mri'], deep_supervision=True)
        actual = v2.combined_loss(output, batch['mask'], class_weights=(1.,) * 6)
        expected = original.combined_loss(output, batch['mask'])
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        ga = torch.autograd.grad(actual[0], tuple(model.parameters()), retain_graph=True)
        gb = torch.autograd.grad(expected[0], tuple(model.parameters()))
        for a, b in zip(ga, gb):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_invalid_loss_inputs_and_weights(self):
        logits, target = self.logits_target()
        for weights in ((1.,) * 5, (1, 1, 1, 1, 1, 0), (1, 1, 1, 1, 1, float('nan'))):
            with self.assertRaises(ValueError):
                v2.head_loss(logits, target, class_weights=weights)
        for bad_logits, bad_target in ((logits.double(), target), (logits, target.float()),
                                       (logits, target + 6), (logits[:, :5], target),
                                       (logits * float('nan'), target)):
            with self.assertRaises(ValueError):
                v2.head_loss(bad_logits, bad_target)
        with self.assertRaises(ValueError):
            v2.combined_loss((logits, (logits, logits)), target)

    def test_epoch_update_all_parameter_gradients_and_diagnostic_accounting(self):
        model = ImprovedUNet3D(base_channels=1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.00001,
                                     foreach=False, fused=False)
        batch = self.batch()
        before = {k: v.clone() for k, v in model.state_dict().items()}
        result = v2.run_epoch(model, [batch, batch], optimizer, torch.device('cpu'))
        self.assertTrue(result['epoch_completed'])
        self.assertEqual(result['processed_steps'], 2)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertTrue(any(not torch.equal(before[k], v) for k, v in model.state_dict().items()))
        for key in ('mean_loss', 'mean_weighted_ce', 'mean_ordinary_ce', 'mean_soft_dice_loss'):
            self.assertTrue(np.isfinite(result[key]))
        partial = v2.run_epoch(model, [batch, batch], optimizer, torch.device('cpu'), diagnostic=True)
        self.assertFalse(partial['epoch_completed'])
        self.assertEqual(partial['processed_steps'], 1)
        self.assertGreater(partial['changed_parameter_tensors'], 0)
        self.assertEqual(partial['finite_gradient_parameter_tensors'], len(list(model.parameters())))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                v2.atomic_checkpoint(Path(directory)/'partial.pt', model, optimizer,
                    completed_epochs=1, total_steps=1, config={'steps_per_epoch': 2},
                    generator=torch.Generator(), device=torch.device('cpu'), epoch_result=partial)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_actual_main_initialization_rng_and_loader_order_parity(self):
        """Execute BOTH production mains up to optimizer construction, mocking I/O/CUDA.

        Real default architecture constructors consume RNG, including the parent
        constructor's discarded layers. No substitute initialization is used.
        """
        class FakeDataset:
            def __init__(self, split, root):
                assert split == 'train'
                self.pairs = [VolumePair(p, w, Path(m), Path(s)) for p, w, m, s in manifest()]
                self.patch_shape_dhw = (64, 64, 64)
                self.train_patches_per_volume = 1
                self.seed = 3710
                self.train_foreground_probability = .5
            def __len__(self):
                return 143
            def __getitem__(self, index):
                return {'index': index}

        class StopAtOptimizer(Exception):
            pass

        def capture(module):
            state = {}
            real_loader = torch.utils.data.DataLoader
            def loader(*args, **kwargs):
                result = real_loader(*args, **kwargs)
                state['loader'] = result
                return result
            def optimizer(parameters, **kwargs):
                state['weights'] = [p.detach().clone() for p in parameters]
                state['optimizer'] = kwargs
                state['torch_rng'] = torch.get_rng_state().clone()
                state['python_next'] = random.random()
                state['numpy_next'] = np.random.random()
                state['torch_next'] = torch.rand(5)
                state['orders'] = [[int(batch['index'].item()) for batch in state['loader']]
                                   for _ in range(10)]
                raise StopAtOptimizer
            name = __package__ + '.patch_dataset'
            replacements = {name: types.SimpleNamespace(HipMRIPatchDataset=FakeDataset),
                            'nibabel': types.SimpleNamespace(__version__='synthetic'),
                            'scipy': types.SimpleNamespace(__version__='synthetic')}
            with tempfile.TemporaryDirectory() as directory, patch.dict('sys.modules', replacements), \
                    patch.dict('os.environ', {'SLURM_JOB_ID': 'synthetic'}), \
                    patch.object(torch.cuda, 'is_available', return_value=True), \
                    patch.object(torch.cuda, 'get_device_name', return_value='synthetic'), \
                    patch.object(torch.cuda, 'device_count', return_value=1), \
                    patch.object(torch.nn.Module, 'to', lambda self, *a, **kw: self), \
                    patch.object(module, 'DataLoader', side_effect=loader), \
                    patch.object(torch.optim, 'AdamW', side_effect=optimizer):
                prefix = 'improved_unet_v2_' if module is v2 else 'improved_unet_'
                with self.assertRaises(StopAtOptimizer):
                    module.main(['--checkpoint', str(Path(directory)/(prefix+'last.pt'))])
            return state

        old, new = capture(original), capture(v2)
        self.assertEqual(len(old['weights']), len(new['weights']))
        self.assertEqual(sum(p.numel() for p in new['weights']), 5734114)
        for a, b in zip(old['weights'], new['weights']):
            self.assertTrue(torch.equal(a, b))
        for key in ('torch_rng', 'torch_next'):
            self.assertTrue(torch.equal(old[key], new[key]))
        for key in ('python_next', 'numpy_next', 'optimizer', 'orders'):
            self.assertEqual(old[key], new[key])

    def test_checkpoint_roundtrip_resume_rng_optimizer_and_next_update(self):
        model = ImprovedUNet3D(base_channels=1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.00001,
                                     foreach=False, fused=False)
        batch = self.batch()
        v2.run_epoch(model, [batch], optimizer, torch.device('cpu'))
        generator = torch.Generator().manual_seed(3710)
        cfg = config()
        # Synthetic completed-boundary metadata; this is NOT 143 real updates.
        boundary = {'epoch_completed': True, 'processed_steps': 143, 'expected_steps': 143}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'improved_unet_v2_last.pt'
            v2.atomic_checkpoint(path, model, optimizer, completed_epochs=1, total_steps=143,
                config=cfg, generator=generator, device=torch.device('cpu'), epoch_result=boundary)
            self.assertEqual(list(Path(directory).iterdir()), [path])
            expected_rng = (random.random(), np.random.random(), torch.rand(4), torch.rand(4, generator=generator))
            expected_result = v2.run_epoch(model, [batch], optimizer, torch.device('cpu'))
            restored = ImprovedUNet3D(base_channels=1)
            opt = torch.optim.AdamW(restored.parameters())
            self.assertEqual(v2.resume_checkpoint(path, restored, opt, config=cfg,
                generator=generator, device=torch.device('cpu')), (1, 143))
            actual_rng = (random.random(), np.random.random(), torch.rand(4), torch.rand(4, generator=generator))
            self.assertEqual(expected_rng[:2], actual_rng[:2])
            for a, b in zip(expected_rng[2:], actual_rng[2:]):
                self.assertTrue(torch.equal(a, b))
            result = v2.run_epoch(restored, [batch], opt, torch.device('cpu'))
            self.assertEqual(expected_result['mean_loss'], result['mean_loss'])
            for a, b in zip(model.parameters(), restored.parameters()):
                self.assertTrue(torch.equal(a, b))
            self.assertEqual(opt.param_groups[0]['lr'], .001)
            self.assertTrue(opt.state)

    def test_configuration_identity_policy_manifest_and_hash_rejection(self):
        good = config()
        v2.validate_config(good)
        changes = [lambda x: x.update(experiment_id='original'),
                   lambda x: x['hyperparameters'].update(model='StandardUNet3D'),
                   lambda x: x['policies']['patch_dataset'].update(seed=42),
                   lambda x: x['source_sha256'].update({'train.py': 'bad'}),
                   lambda x: x.update(train_pairs=x['train_pairs'][:-1]),
                   lambda x: x.update(steps_per_epoch=1)]
        for change in changes:
            bad = copy.deepcopy(good)
            change(bad)
            bad['config_sha256'] = v2.config_hash(bad)
            with self.assertRaises(ValueError):
                v2.validate_config(bad)
        good['config_sha256'] = 'bad'
        with self.assertRaisesRegex(ValueError, 'configuration hash'):
            v2.validate_config(good)

    def test_real_serialized_default_model_strict_validation_and_old_loader_rejection(self):
        cfg = config()
        model = ImprovedUNet3D()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        boundary = {'epoch_completed': True, 'processed_steps': 143, 'expected_steps': 143}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'improved_unet_v2_last.pt'
            v2.atomic_checkpoint(path, model, optimizer, completed_epochs=10, total_steps=1430,
                config=cfg, generator=torch.Generator(), device=torch.device('cpu'), epoch_result=boundary)
            restored, meta = validation.load_checkpoint_model(path, torch.device('cpu'))
            self.assertFalse(restored.training)
            self.assertEqual(meta['total_steps'], 1430)
            for a, b in zip(model.parameters(), restored.parameters()):
                self.assertTrue(torch.equal(a, b))
            with self.assertRaises(ValueError):
                original_validation.load_checkpoint_model(path, torch.device('cpu'))
            payload = torch.load(path, weights_only=True)
            for epoch, steps in ((1, 143), (10, 1429), (11, 1573)):
                invalid = dict(payload, completed_epochs=epoch, total_steps=steps)
                with patch.object(torch, 'load', return_value=invalid):
                    with self.assertRaisesRegex(ValueError, 'Epoch 10'):
                        validation.load_checkpoint_model(path, torch.device('cpu'))
            payload['model'].pop('head.weight')
            torch.save(payload, path)
            with self.assertRaises(RuntimeError):
                validation.load_checkpoint_model(path, torch.device('cpu'))

    def test_original_checkpoint_cannot_resume_or_validate_as_v2(self):
        payload = {'schema_version': 1, 'completed_epochs': 10,
                   'epoch_result': {'expected_steps': 143},
                   'config': {'hyperparameters': original.HYPERPARAMETERS}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'improved_unet_v2_last.pt'
            torch.save(payload, path)
            with self.assertRaises(ValueError):
                validation.load_checkpoint_model(path, torch.device('cpu'))
            with self.assertRaises(ValueError):
                v2.v2_checkpoint_path(path, '/synthetic-dataset')
            with self.assertRaises(ValueError):
                v2.resume_checkpoint(path, None, None, config=config(),
                                     generator=torch.Generator(), device=torch.device('cpu'))

    def test_output_paths_and_cli_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ('standard_unet_last.pt', 'improved_unet_last.pt'):
                with self.assertRaises(ValueError):
                    v2.v2_checkpoint_path(Path(directory)/name, '/synthetic-dataset')
        with self.assertRaises(ValueError):
            v2.v2_checkpoint_path(Path(__file__).with_name('improved_unet_v2_bad.pt'), '/synthetic-dataset')
        for args in ([], ['--checkpoint', 'x', '--epochs', '11'],
                     ['--checkpoint', 'x', '--diagnostic-one-step', '--resume', 'x']):
            with self.assertRaises(SystemExit):
                v2.main(args)
        with self.assertRaises(SystemExit):
            validation.main(['--checkpoint', 'x', '--all-test'])

    def test_shared_validation_math_and_main_head_inference(self):
        self.assertIs(validation._process_pair, evaluator._process_pair)
        self.assertIs(validation.summarize, evaluator.summarize)
        self.assertIs(validation.select_validation, evaluator.select_validation)
        model = ImprovedUNet3D(base_channels=1)
        prediction, evidence = evaluator.sliding_window_predict(
            model, np.ones((17, 16, 18), np.float32), torch.device('cpu'),
            patch_shape=(16, 16, 16), stride=8)
        self.assertEqual(prediction.shape, (17, 16, 18))
        self.assertGreaterEqual(evidence['coverage_min'], 1)
        self.assertTrue(np.isin(prediction, np.arange(6)).all())


if __name__ == '__main__':
    unittest.main()
