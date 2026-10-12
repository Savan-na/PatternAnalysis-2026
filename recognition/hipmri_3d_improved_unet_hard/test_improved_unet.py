"""Synthetic CPU tests; no HipMRI images, training pipeline or GPU required."""

import unittest
import torch
from torch import nn
from .improved_modules import ImprovedUNet3D, ResidualBlock3D, deep_supervision_loss


class ImprovedUNetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        torch.manual_seed(3710)
        self.model = ImprovedUNet3D(base_channels=2)

    def test_planned_patch_and_auxiliary_shapes(self):
        with torch.no_grad():
            main, aux = self.model(torch.zeros(1, 1, 64, 64, 64), deep_supervision=True)
        self.assertEqual(main.shape, (1, 6, 64, 64, 64))
        self.assertEqual([tuple(a.shape) for a in aux], [(1, 6, 32, 32, 32), (1, 6, 16, 16, 16)])
        for output in (main, *aux):
            self.assertEqual(output.dtype, torch.float32)
            self.assertTrue(bool(torch.isfinite(output).all()))

    def test_default_forward_and_non_cubic_input(self):
        self.model.eval()
        image = torch.randn(1, 1, 16, 32, 48)
        with torch.no_grad():
            main = self.model(image)
            supervised, _ = self.model(image, deep_supervision=True)
        self.assertEqual(main.shape, (1, 6, 16, 32, 48))
        torch.testing.assert_close(main, supervised, rtol=0, atol=0)

    def test_identity_residual_path(self):
        block = ResidualBlock3D(2, 2)
        for parameter in block.branch.parameters():
            nn.init.zeros_(parameter)
        image = torch.randn(1, 2, 4, 5, 6, requires_grad=True)
        output = block(image)
        torch.testing.assert_close(output, image.relu())
        output.sum().backward()
        torch.testing.assert_close(image.grad, (image > 0).float())

    def test_projected_residual_path_and_all_blocks(self):
        block = ResidualBlock3D(1, 2)
        for parameter in block.branch.parameters():
            nn.init.zeros_(parameter)
        nn.init.ones_(block.shortcut.weight)
        image = torch.ones(1, 1, 4, 4, 4)
        torch.testing.assert_close(block(image), torch.ones(1, 2, 4, 4, 4))
        self.assertTrue(all(isinstance(b, ResidualBlock3D) for b in
                            (*self.model.encoder, self.model.bottleneck, *self.model.decoder)))

    def test_loss_backward_all_heads_and_finite_gradients(self):
        outputs = self.model(torch.randn(1, 1, 16, 16, 16), deep_supervision=True)
        target = torch.arange(16**3).reshape(1, 16, 16, 16).long() % 6
        # Same scalar loss intended for both model arms; no production training executed.
        from .train import combined_loss
        loss = deep_supervision_loss(outputs, target, lambda logits, labels: combined_loss(logits, labels)[0])
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()
        self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                            for p in self.model.parameters()))

    def test_loss_weighting_and_target_preservation(self):
        main = torch.randn(1, 6, 16, 16, 16, requires_grad=True)
        aux = (torch.randn(1, 6, 8, 8, 8), torch.randn(1, 6, 4, 4, 4))
        target = torch.randint(6, (1, 16, 16, 16))
        before = target.clone()
        ce = lambda logits, labels: nn.functional.cross_entropy(logits, labels, reduction="none").mean()
        expected = ce(main, target)
        for output, weight in zip(aux, (.5, .25)):
            expected = expected + weight * ce(nn.functional.interpolate(
                output, size=(16, 16, 16), mode="trilinear", align_corners=False), target)
        torch.testing.assert_close(deep_supervision_loss((main, aux), target, ce), expected / 1.75)
        self.assertTrue(torch.equal(before, target))

    def test_invalid_inputs(self):
        for image in (torch.zeros(1, 2, 16, 16, 16), torch.zeros(1, 1, 17, 16, 16),
                      torch.zeros(1, 16, 16, 16)):
            with self.assertRaises(ValueError):
                self.model(image)
        with self.assertRaises(TypeError):
            self.model(torch.zeros(1, 1, 16, 16, 16, dtype=torch.float64))


if __name__ == "__main__":
    unittest.main()
