"""Focused CPU tests for the conventional Standard 3D U-Net baseline."""

from __future__ import annotations

import unittest

import torch
from torch import nn

from .modules import StandardUNet3D


class StandardUNet3DTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(3710)
        self.model = StandardUNet3D(base_channels=2).cpu()

    def test_default_channels_and_feature_widths(self) -> None:
        model = StandardUNet3D()
        self.assertEqual((model.in_channels, model.out_channels), (1, 6))
        self.assertEqual([block[0].out_channels for block in model.encoder], [16, 32, 64, 128])
        self.assertEqual(model.bottleneck[0].out_channels, 256)
        self.assertEqual(model.head.out_channels, 6)

    def test_batch_one_shape_dtype_and_finite_logits(self) -> None:
        with torch.no_grad():
            logits = self.model(torch.randn(1, 1, 16, 16, 16))
        self.assertEqual(tuple(logits.shape), (1, 6, 16, 16, 16))
        self.assertEqual(logits.dtype, torch.float32)
        self.assertTrue(bool(torch.isfinite(logits).all()))

    def test_non_cubic_spatial_shape_is_preserved(self) -> None:
        with torch.no_grad():
            logits = self.model(torch.randn(1, 1, 16, 32, 48))
        self.assertEqual(tuple(logits.shape), (1, 6, 16, 32, 48))

    def test_invalid_inputs_are_rejected(self) -> None:
        cases = (
            (torch.randn(1, 16, 16, 16), ValueError, "[B,C,D,H,W]"),
            (torch.randn(1, 2, 16, 16, 16), ValueError, "input channels"),
            (torch.randn(1, 1, 8, 16, 16), ValueError, "divisible by 16"),
            (torch.randn(1, 1, 16, 24, 16), ValueError, "divisible by 16"),
            (torch.randn(1, 1, 16, 16, 16, dtype=torch.float64), TypeError, "float32"),
            (torch.empty(0, 1, 16, 16, 16), ValueError, "nonempty"),
        )
        for tensor, error_type, message in cases:
            with self.subTest(shape=tuple(tensor.shape), dtype=tensor.dtype):
                with self.assertRaisesRegex(error_type, message):
                    self.model(tensor)

    def test_invalid_constructor_channels_are_rejected(self) -> None:
        for kwargs in ({"in_channels": 0}, {"out_channels": 0}, {"base_channels": 0},
                       {"base_channels": True}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    StandardUNet3D(**kwargs)

    def test_conventional_blocks_downsampling_and_skip_concatenation(self) -> None:
        self.assertEqual((len(self.model.encoder), len(self.model.upconvs), len(self.model.decoder)),
                         (4, 4, 4))
        self.assertIsInstance(self.model.pool, nn.MaxPool3d)
        self.assertEqual(self.model.pool.kernel_size, 2)
        self.assertEqual(self.model.pool.stride, 2)
        for block in (*self.model.encoder, self.model.bottleneck, *self.model.decoder):
            convolutions = [layer for layer in block if isinstance(layer, nn.Conv3d)]
            norms = [layer for layer in block if isinstance(layer, nn.GroupNorm)]
            self.assertEqual(len(convolutions), 2)
            self.assertEqual(len(norms), 2)
            self.assertTrue(all(conv.kernel_size == (3, 3, 3) and conv.padding == (1, 1, 1)
                                for conv in convolutions))
            self.assertTrue(all(norm.num_groups == 1 for norm in norms))
        self.assertTrue(all(isinstance(layer, nn.ConvTranspose3d) and
                            layer.kernel_size == (2, 2, 2) and layer.stride == (2, 2, 2)
                            for layer in self.model.upconvs))
        self.assertEqual(self.model.head.kernel_size, (1, 1, 1))

        decoder_inputs: list[tuple[int, ...]] = []
        handles = [block.register_forward_pre_hook(
            lambda _module, inputs: decoder_inputs.append(tuple(inputs[0].shape))
        ) for block in self.model.decoder]
        try:
            with torch.no_grad():
                self.model(torch.randn(1, 1, 16, 16, 16))
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(decoder_inputs, [
            (1, 32, 2, 2, 2), (1, 16, 4, 4, 4),
            (1, 8, 8, 8, 8), (1, 4, 16, 16, 16),
        ])

    def test_cross_entropy_backward_has_finite_gradients(self) -> None:
        logits = self.model(torch.randn(1, 1, 16, 16, 16))
        labels = torch.randint(0, 6, (1, 16, 16, 16), dtype=torch.long)
        loss = nn.CrossEntropyLoss()(logits, labels)
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()
        gradients = [parameter.grad for parameter in self.model.parameters() if parameter.requires_grad]
        self.assertTrue(gradients)
        self.assertTrue(all(gradient is not None and bool(torch.isfinite(gradient).all())
                            for gradient in gradients))

    def test_optimizer_step_changes_a_parameter(self) -> None:
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        before = self.model.head.weight.detach().clone()
        optimizer.zero_grad(set_to_none=True)
        logits = self.model(torch.randn(1, 1, 16, 16, 16))
        labels = torch.randint(0, 6, (1, 16, 16, 16), dtype=torch.long)
        nn.CrossEntropyLoss()(logits, labels).backward()
        optimizer.step()
        self.assertFalse(torch.equal(before, self.model.head.weight.detach()))

    def test_initialisation_is_deterministic_with_fixed_seed(self) -> None:
        torch.manual_seed(19)
        first = StandardUNet3D(base_channels=2)
        torch.manual_seed(19)
        second = StandardUNet3D(base_channels=2)
        self.assertTrue(all(torch.equal(first.state_dict()[name], second.state_dict()[name])
                            for name in first.state_dict()))

    def test_planned_64_cubed_patch_shape_without_gradients(self) -> None:
        with torch.no_grad():
            logits = self.model(torch.zeros(1, 1, 64, 64, 64))
        self.assertEqual(tuple(logits.shape), (1, 6, 64, 64, 64))
        self.assertTrue(bool(torch.isfinite(logits).all()))


if __name__ == "__main__":
    unittest.main()
