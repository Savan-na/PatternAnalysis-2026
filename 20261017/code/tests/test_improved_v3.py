import tempfile
import unittest
from pathlib import Path

import torch
from hipmri_rewrite.improved_v3 import ImprovedUNet3DV3, state_digest
from hipmri_rewrite.models.improved_3d import ImprovedUNet3D
from hipmri_rewrite.losses import ce_dice_loss
from tools.train_v3_cohort import atomic_checkpoint, load_durable_state


class V3Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_same_initial_parameters_rng_additive_inference_and_all_head_gradients(self):
        torch.manual_seed(371017)
        original = ImprovedUNet3D(base_channels=2)
        original_rng = torch.get_rng_state().clone()
        torch.manual_seed(371017)
        model = ImprovedUNet3DV3(base_channels=2)
        self.assertEqual(state_digest(original), state_digest(model))
        self.assertTrue(torch.equal(torch.get_rng_state(), original_rng))
        self.assertEqual(sum(isinstance(m, torch.nn.GroupNorm) for m in model.modules()),14)
        self.assertEqual(sum(isinstance(m, torch.nn.InstanceNorm3d) for m in model.modules()),0)
        self.assertEqual(sum(p.numel() for p in model.parameters()), sum(p.numel() for p in original.parameters()))
        for layer in model.modules():
            if isinstance(layer, torch.nn.GroupNorm):
                self.assertEqual(layer.num_groups, 1)
                self.assertEqual(layer.eps, 1e-5)
                self.assertTrue(layer.affine)
        x = torch.randn(2, 1, 32, 32, 32)
        y = torch.randint(0, 6, (2, 32, 32, 32))
        model.eval()
        combined, heads = model(x, return_components=True)
        torch.testing.assert_close(model(x), combined, rtol=0, atol=0)
        torch.testing.assert_close(combined, heads[0]+heads[1]+heads[2], rtol=0, atol=0)
        self.assertEqual(combined.shape, (2, 6, 32, 32, 32))
        weights = torch.tensor([1.453871, 1., 2.617199, 4.498586, 8.893215, 9.69995])
        loss = ce_dice_loss(combined, y, class_weights=weights)
        loss.backward()
        for name, param in model.named_parameters():
            self.assertIsNotNone(param.grad, name)
            self.assertTrue(torch.isfinite(param.grad).all(), name)
        for head in (model.head, model.aux_half, model.aux_quarter):
            self.assertGreater(float(head.weight.grad.abs().sum()), 0)

    def test_v3_optimizer_and_rng_resume_matches_uninterrupted_update(self):
        torch.manual_seed(31)
        model = ImprovedUNet3DV3(base_channels=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        x = torch.randn(2, 1, 32, 32, 32)
        y = torch.randint(0, 6, (2, 32, 32, 32))
        def step(m, opt):
            m.train()
            opt.zero_grad(set_to_none=True)
            loss = ce_dice_loss(m(x), y, class_weights=torch.ones(6))
            loss.backward()
            opt.step()
        step(model, optimizer)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            atomic_checkpoint(root / 'latest.pt', {
                'epoch': 0, 'next_batch': 1, 'model': model.state_dict(),
                'optimizer': optimizer.state_dict(), 'rng': torch.get_rng_state()})
            step(model, optimizer)
            expected = state_digest(model)
            restored = ImprovedUNet3DV3(base_channels=2)
            opt = torch.optim.Adam(restored.parameters(), lr=.001)
            state, _ = load_durable_state(root, 'cpu')
            restored.load_state_dict(state['model'])
            opt.load_state_dict(state['optimizer'])
            torch.set_rng_state(state['rng'])
            step(restored, opt)
            self.assertEqual(state_digest(restored), expected)


if __name__ == '__main__':
    unittest.main()
