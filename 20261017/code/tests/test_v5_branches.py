"""Numerical control and stochastic checkpoint recovery for the C/L/S screen."""
import copy
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import importlib.util
import sys
import tempfile
from types import SimpleNamespace
import numpy as np
import torch
from hipmri_rewrite.v5_branches import (
    expected_config, validate_config, class_weights, learning_rate,
    scheduler_state, make_optimizer, loss_components, restore_state)
from hipmri_rewrite.losses import ce_dice_loss
from hipmri_rewrite.improved_v3 import ImprovedUNet3DV3


class BranchTests(unittest.TestCase):
    def test_submission_space_gate_and_array_dependency(self):
        path = Path(__file__).resolve().parents[1]/'tools/submit_v5_screen.py'
        spec = importlib.util.spec_from_file_location('v5_submit_test', path)
        module = importlib.util.module_from_spec(spec)
        fake_fcntl = SimpleNamespace(LOCK_EX=2, LOCK_NB=4, flock=lambda *args: None)
        with patch.dict(sys.modules, {'fcntl': fake_fcntl}):
            spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            module.ROOT = Path(directory)
            with patch.object(module, 'verify'), patch.object(sys, 'argv', ['submit', 'new']), \
                 patch.dict('os.environ', {'USER': 'test'}), \
                 patch.object(module.subprocess, 'check_output', return_value=''), \
                 patch.object(module.subprocess, 'run'), \
                 patch.object(module.shutil, 'disk_usage', return_value=SimpleNamespace(free=5*1024**3)), \
                 patch.object(module, 'submit', side_effect=['123', '124']) as submit:
                with self.assertRaisesRegex(ValueError, 'Free space'):
                    module.main()
                submit.assert_not_called()
                with patch.object(module.shutil, 'disk_usage', return_value=SimpleNamespace(free=7*1024**3)):
                    module.main()
                self.assertIn('--array=0-2%3', submit.call_args_list[1].args[0])
                self.assertIn('--dependency=afterok:123', submit.call_args_list[1].args[0])
                self.assertTrue((module.ROOT/'logs/profiles/v5_submission_124.json').is_file())

    def test_configs_and_single_factor_changes(self):
        root = Path(__file__).resolve().parents[1]
        for arm in 'CLS':
            config = json.loads((root/f'config/v5_{arm.lower()}_frozen.json').read_text())
            validate_config(config)
            broken = copy.deepcopy(config)
            broken['seed'] += 1
            with self.assertRaises(ValueError):
                validate_config(broken)
        c, l, s = [expected_config(a) for a in 'CLS']
        self.assertEqual(c['optimizer'], l['optimizer'])
        self.assertEqual(c['objective'], s['objective'])
        self.assertEqual(class_weights('L')[:3], class_weights('C')[:3])
        self.assertTrue(all(x < y for x, y in zip(class_weights('L')[3:], class_weights('C')[3:])))
        for key in ('sampling', 'preprocessing', 'network_normalization', 'supervision', 'budget'):
            self.assertEqual(c[key], l[key])
            self.assertEqual(c[key], s[key])

    def test_original_loss_and_actual_network_update_identity(self):
        torch.manual_seed(371017)
        model = ImprovedUNet3DV3(base_channels=2)
        other = copy.deepcopy(model)
        x = torch.randn(2, 1, 16, 16, 16)
        y = torch.randint(0, 6, (2, 16, 16, 16))
        weights = torch.tensor(class_weights('C'))
        optimizer = make_optimizer(model, 'C')
        reference = torch.optim.Adam(other.parameters(), lr=.001)
        rng = torch.get_rng_state()
        new_loss, _, _ = loss_components(model(x), y, weights)
        new_loss.backward()
        optimizer.step()
        torch.set_rng_state(rng)
        old_loss = ce_dice_loss(other(x), y, class_weights=weights)
        old_loss.backward()
        reference.step()
        self.assertTrue(torch.equal(new_loss, old_loss))
        for p, q in zip(model.parameters(), other.parameters()):
            self.assertTrue(torch.equal(p, q))

    def test_schedule_endpoints_and_monotonicity(self):
        values = [learning_rate('S', n) for n in range(6501)]
        self.assertAlmostEqual(values[0], .001)
        self.assertAlmostEqual(values[-1], .00001)
        self.assertTrue(all(a >= b for a, b in zip(values, values[1:])))
        self.assertEqual(learning_rate('L', 4550), .001)
        with self.assertRaises(ValueError):
            learning_rate('S', 6501)

    def test_stochastic_adam_resume_matches_uninterrupted_all_branches(self):
        for arm in 'CLS':
            torch.manual_seed(17)
            np.random.seed(17)
            model = torch.nn.Sequential(torch.nn.Linear(4, 6), torch.nn.Dropout(.3))
            optimizer = make_optimizer(model, arm)
            def step(net, opt, update):
                x = torch.randn(2, 4) + float(np.random.rand())
                opt.zero_grad()
                net(x).square().mean().backward()
                opt.step()
                for group in opt.param_groups:
                    group['lr'] = learning_rate(arm, update)
            step(model, optimizer, 1)
            checkpoint = dict(config_sha256='test', model=arm, epoch=0, next_batch=1,
                              completed_updates=1, scheduler_state=scheduler_state(arm, 1),
                              model_state=model.state_dict(), optimizer_state=optimizer.state_dict(),
                              torch_rng_state=torch.get_rng_state(), numpy_rng_state=np.random.get_state(),
                              cuda_rng_states=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
            stream = io.BytesIO()
            torch.save(checkpoint, stream)
            stream.seek(0)
            saved = torch.load(stream, weights_only=False)
            step(model, optimizer, 2)
            recovered = torch.nn.Sequential(torch.nn.Linear(4, 6), torch.nn.Dropout(.3))
            opt = make_optimizer(recovered, arm)
            restore_state(recovered, opt, saved, 'test', arm, arm)
            step(recovered, opt, 2)
            for a, b in zip(model.parameters(), recovered.parameters()):
                self.assertTrue(torch.equal(a, b), arm)
            self.assertEqual(opt.param_groups[0]['lr'], optimizer.param_groups[0]['lr'])
            damaged = copy.deepcopy(saved)
            damaged['scheduler_state']['completed_updates'] = 2
            with self.assertRaises(ValueError):
                restore_state(recovered, opt, damaged, 'test', arm, arm)
            damaged = copy.deepcopy(saved)
            damaged['optimizer_state']['param_groups'][0]['lr'] = .9
            with self.assertRaises(ValueError):
                restore_state(recovered, opt, damaged, 'test', arm, arm)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
