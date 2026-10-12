"""CPU synthetic training/checkpoint tests; no HipMRI arrays or CUDA work."""

from copy import deepcopy
import math
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from . import train


class _SyntheticDataset(Dataset):
    def __init__(self, count=3):
        self.count = count
        self.epoch = 0
        self.visited = []

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        self.visited.append(index)
        # Exercise all restored global RNG states in the resume comparison.
        image = torch.randn(1, 4, 4, 4) + random.random() + float(np.random.random())
        target = (torch.arange(64).reshape(4, 4, 4) + index + self.epoch) % 6
        return {"mri": image, "mask": target.long()}


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setup_run(self, count=3):
        device = torch.device("cpu")
        train.configure_reproducibility(3710, device)
        model = nn.Sequential(nn.Conv3d(1, 6, 1), nn.Dropout3d(0.1))
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.00001,
                                     foreach=False, fused=False)
        generator = torch.Generator().manual_seed(3710)
        dataset = _SyntheticDataset(count)
        loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, generator=generator)
        return model, optimizer, generator, dataset, loader, device

    def save(self, path, state, result, epoch=1):
        model, optimizer, generator, dataset, loader, device = state
        config = {"steps_per_epoch": len(loader), "hyperparameters": deepcopy(train.HYPERPARAMETERS)}
        train.atomic_checkpoint(path, model, optimizer, completed_epochs=epoch,
                                total_steps=epoch * len(loader), config=config,
                                generator=generator, device=device, epoch_result=result)
        return config

    def test_reproducibility_uses_warn_only_policy(self):
        enabled = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        try:
            train.configure_reproducibility(3710, torch.device("cpu"))
            self.assertTrue(torch.are_deterministic_algorithms_enabled())
            self.assertTrue(torch.is_deterministic_algorithms_warn_only_enabled())
            self.assertTrue(train.HYPERPARAMETERS["deterministic_algorithms"])
            self.assertTrue(train.HYPERPARAMETERS["deterministic_warn_only"])
            self.assertEqual(train.HYPERPARAMETERS["reproducibility_policy"], "best_effort_warn_only")
            self.assertFalse(torch.backends.cudnn.benchmark)
            self.assertTrue(torch.backends.cudnn.deterministic)
            self.assertFalse(torch.backends.cudnn.allow_tf32)
            self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
        finally:
            torch.use_deterministic_algorithms(enabled, warn_only=warn_only)

    def test_loss_formula_and_absent_classes(self):
        logits = torch.zeros(1, 6, 2, 2, 2, requires_grad=True)
        target = torch.ones(1, 2, 2, 2, dtype=torch.long)
        total, ce, dice = train.combined_loss(logits, target)
        smooth, n = 1e-5, 8
        expected_dice = 1 - ((2 * n / 6 + smooth) / (n / 6 + n + smooth)
                             + 4 * smooth / (n / 6 + smooth)) / 5
        self.assertAlmostEqual(ce.item(), math.log(6), places=6)
        self.assertAlmostEqual(dice.item(), expected_dice, places=6)
        self.assertAlmostEqual(total.item(), .5 * math.log(6) + .5 * expected_dice, places=6)
        total.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_background_only_patch_is_finite(self):
        logits = torch.zeros(1, 6, 2, 2, 2, requires_grad=True)
        target = torch.zeros(1, 2, 2, 2, dtype=torch.long)
        total, _, dice = train.combined_loss(logits, target)
        self.assertAlmostEqual(dice.item(), 1 - 1e-5 / (8 / 6 + 1e-5), places=6)
        total.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_unreduced_ce_matches_original_value_and_logit_gradients(self):
        generator = torch.Generator().manual_seed(3710)
        shape = (2, 6, 3, 4, 5)
        masks = {
            "background": torch.zeros(2, 3, 4, 5, dtype=torch.long),
            "foreground": torch.full((2, 3, 4, 5), 5, dtype=torch.long),
            "mixed_six_classes": (torch.arange(120).reshape(2, 3, 4, 5) % 6).long(),
        }
        for name, target in masks.items():
            with self.subTest(mask=name):
                original_logits = torch.randn(shape, generator=generator, requires_grad=True)
                modified_logits = original_logits.detach().clone().requires_grad_(True)
                original = nn.functional.cross_entropy(original_logits, target)
                # Exercise the modified production CE branch, not a duplicate formula.
                _, modified, _ = train.combined_loss(modified_logits, target)
                original.backward()
                modified.backward()
                torch.testing.assert_close(modified, original, rtol=1e-6, atol=1e-7)
                torch.testing.assert_close(modified_logits.grad, original_logits.grad,
                                           rtol=1e-6, atol=1e-7)
                self.assertTrue(torch.isfinite(modified_logits.grad).all())

    def test_nonfinite_loss_from_finite_extreme_logits_rejected(self):
        logits = torch.full((1, 6, 2, 2, 2), -3e38)
        logits[:, 0] = 3e38
        target = torch.ones(1, 2, 2, 2, dtype=torch.long)
        with self.assertRaisesRegex(ValueError, "loss is non-finite"):
            train.combined_loss(logits, target)

    def test_actual_standard_unet_synthetic_optimizer_step(self):
        from .modules import StandardUNet3D
        train.configure_reproducibility(3710, torch.device("cpu"))
        model = StandardUNet3D(base_channels=1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.00001,
                                     foreach=False, fused=False)
        batch = {"mri": torch.randn(1, 1, 16, 16, 16),
                 "mask": (torch.arange(4096).reshape(1, 16, 16, 16) % 6).long()}
        before = [p.detach().clone() for p in model.parameters()]
        result = train.run_epoch(model, [batch], optimizer, torch.device("cpu"),
                                 expected_patch_shape=(16, 16, 16))
        self.assertTrue(result["epoch_completed"])
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, model.parameters())))

    def test_incompatible_batch_dimensions_rejected(self):
        state = self.setup_run()
        with self.assertRaisesRegex(ValueError, "patch shape"):
            train.run_epoch(state[0], state[4], state[1], state[5], expected_patch_shape=(64, 64, 64))
        bad = {"mri": torch.zeros(1, 2, 4, 4, 4),
               "mask": torch.zeros(1, 4, 4, 4, dtype=torch.long)}
        with self.assertRaises(ValueError):
            train.run_epoch(state[0], [bad], state[1], state[5])

    def test_invalid_loss_inputs(self):
        logits = torch.zeros(1, 6, 2, 2, 2)
        target = torch.zeros(1, 2, 2, 2, dtype=torch.long)
        for bad_logits, bad_target in (
            (logits[:, :5], target), (logits, target[:, :, :, :1]),
            (logits.double(), target), (logits, target.float()),
            (logits, target + 6), (logits, target - 1),
            (logits + float("nan"), target), (logits + float("inf"), target),
        ):
            with self.subTest(shape=bad_logits.shape, dtype=bad_target.dtype), self.assertRaises(ValueError):
                train.combined_loss(bad_logits, bad_target)

    def test_full_epoch_accounting_and_parameter_update(self):
        state = self.setup_run()
        model, optimizer, _, dataset, loader, device = state
        before = [p.detach().clone() for p in model.parameters()]
        events = []
        result = train.run_epoch(model, loader, optimizer, device, progress=events.append)
        self.assertTrue(result["epoch_completed"])
        self.assertEqual(result["processed_steps"], 3)
        self.assertEqual(sorted(dataset.visited), [0, 1, 2])
        self.assertEqual(len(events), 3)
        self.assertAlmostEqual(result["mean_loss"], sum(e["loss"] for e in events) / 3)
        self.assertGreater(result["runtime_seconds"], 0)
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, model.parameters())))

    def test_diagnostic_never_completes_or_saves(self):
        for count in (1, 3):
            state = self.setup_run(count)
            model, optimizer, _, dataset, loader, device = state
            result = train.run_epoch(model, loader, optimizer, device, diagnostic=True)
            self.assertFalse(result["epoch_completed"])
            self.assertEqual(result["processed_steps"], 1)
            self.assertEqual(len(dataset.visited), 1)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "last.pt"
                with self.assertRaises(ValueError):
                    self.save(path, state, result)
                self.assertFalse(path.exists())

    def test_atomic_checkpoint_contents(self):
        state = self.setup_run()
        result = train.run_epoch(state[0], state[4], state[1], state[5])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pt"
            config = self.save(path, state, result)
            loaded = torch.load(path, weights_only=True, map_location="cpu")
            self.assertEqual(loaded["config"], config)
            self.assertEqual(loaded["completed_epochs"], 1)
            self.assertEqual(loaded["total_steps"], 3)
            self.assertTrue(loaded["optimizer"]["state"])
            self.assertIn("loader", loaded["rng"])
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_failed_atomic_write_preserves_previous_checkpoint(self):
        state = self.setup_run()
        result = train.run_epoch(state[0], state[4], state[1], state[5])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pt"
            self.save(path, state, result)
            before = path.read_bytes()
            with patch.object(train.torch, "save", side_effect=OSError("synthetic disk failure")):
                with self.assertRaises(OSError):
                    self.save(path, state, result)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_resume_matches_uninterrupted_two_epochs(self):
        full = self.setup_run()
        for epoch in range(2):
            full[3].set_epoch(epoch)
            train.run_epoch(full[0], full[4], full[1], full[5])
        expected = deepcopy(full[0].state_dict())
        expected_optimizer = deepcopy(full[1].state_dict())
        expected_draws = (random.random(), np.random.random(), torch.rand(3))
        split = self.setup_run()
        first = train.run_epoch(split[0], split[4], split[1], split[5])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pt"
            config = self.save(path, split, first)
            restored = self.setup_run()
            epoch, steps = train.resume_checkpoint(path, restored[0], restored[1], config=config,
                                                   generator=restored[2], device=restored[5])
            self.assertEqual((epoch, steps), (1, 3))
            restored[3].set_epoch(epoch)
            train.run_epoch(restored[0], restored[4], restored[1], restored[5])
            for name, value in restored[0].state_dict().items():
                self.assertTrue(torch.equal(value, expected[name]), name)
            actual_optimizer = restored[1].state_dict()
            for key, values in expected_optimizer["state"].items():
                for name, value in values.items():
                    self.assertTrue(torch.equal(value, actual_optimizer["state"][key][name]))
            self.assertEqual(random.random(), expected_draws[0])
            self.assertEqual(np.random.random(), expected_draws[1])
            self.assertTrue(torch.equal(torch.rand(3), expected_draws[2]))

    def test_resume_rejects_changed_config_before_model_mutation(self):
        state = self.setup_run()
        result = train.run_epoch(state[0], state[4], state[1], state[5])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pt"
            config = self.save(path, state, result)
            config["hyperparameters"]["learning_rate"] = 0.002
            restored = self.setup_run()
            before = deepcopy(restored[0].state_dict())
            with self.assertRaisesRegex(ValueError, "incompatible"):
                train.resume_checkpoint(path, restored[0], restored[1], config=config,
                                        generator=restored[2], device=restored[5])
            for name, value in restored[0].state_dict().items():
                self.assertTrue(torch.equal(value, before[name]))

    def test_resume_rejects_partial_step_count(self):
        state = self.setup_run()
        result = train.run_epoch(state[0], state[4], state[1], state[5])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pt"
            config = self.save(path, state, result)
            payload = torch.load(path, weights_only=True)
            payload["total_steps"] = 1
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "epoch boundary"):
                train.resume_checkpoint(path, state[0], state[1], config=config,
                                        generator=state[2], device=state[5])

    def test_fresh_runs_are_deterministic(self):
        expected = None
        for _ in range(2):
            state = self.setup_run()
            train.run_epoch(state[0], state[4], state[1], state[5])
            actual = deepcopy(state[0].state_dict())
            if expected is not None:
                self.assertTrue(all(torch.equal(actual[k], expected[k]) for k in actual))
            expected = actual

    def test_nonfinite_image_and_gradient_rejection(self):
        state = self.setup_run()
        bad = {"mri": torch.full((1, 1, 4, 4, 4), float("nan")),
               "mask": torch.zeros(1, 4, 4, 4, dtype=torch.long)}
        with self.assertRaises(ValueError):
            train.run_epoch(state[0], [bad], state[1], state[5])
        handle = next(state[0].parameters()).register_hook(lambda grad: grad * float("nan"))
        with self.assertRaisesRegex(RuntimeError, "gradient"):
            train.run_epoch(state[0], state[4], state[1], state[5])
        handle.remove()

    def test_incomplete_loader_rejected(self):
        state = self.setup_run(1)
        class ShortLoader:
            def __len__(self):
                return 2
            def __iter__(self):
                return iter(state[4])
        with self.assertRaisesRegex(RuntimeError, "Incomplete epoch"):
            train.run_epoch(state[0], ShortLoader(), state[1], state[5])

    def test_output_path_safety(self):
        with self.assertRaises(ValueError):
            train.checkpoint_path(Path(__file__).parent / "last.pt", "/unused-source")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pt"
            self.assertEqual(train.checkpoint_path(path, "/unused-source"), path.resolve())
            with self.assertRaises(ValueError):
                train.checkpoint_path(path, directory)
            (Path(directory) / ".git").mkdir()
            with self.assertRaises(ValueError):
                train.checkpoint_path(path, "/unused-source")


if __name__ == "__main__":
    unittest.main()
