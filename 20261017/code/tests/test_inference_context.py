"""Check P2 coverage, grid restoration and head averaging on synthetic images."""
import unittest
from pathlib import Path
import numpy as np
import torch
from hipmri_rewrite.inference_context import axis_starts, infer_logits


class Pointwise(torch.nn.Module):
    def forward(self, image, *, return_components=False):
        logits = torch.cat([image + c for c in range(6)], dim=1)
        heads = (logits * 0.25, logits * 0.25, logits * 0.5)
        return (sum(heads), heads) if return_components else logits


class ContextTests(unittest.TestCase):
    def test_full_path_reproduces_frozen_adapter_for_both_real_networks(self):
        from hipmri_rewrite.data import Volume, to_3d_model_arrays
        from hipmri_rewrite.inference import predict_volume
        from hipmri_rewrite.models import UNet3D, ImprovedUNet3D
        from hipmri_rewrite.splits import Case
        torch.set_num_threads(2)
        torch.manual_seed(42)
        shape = (35, 33, 29)
        image = np.random.default_rng(7).normal(size=shape).astype(np.float32)
        volume = Volume(Case("A001", 0, "validation", Path("unused"), Path("unused")),
                        image, np.zeros(shape, dtype=np.uint8), np.eye(4), (1., 1., 2.))
        padded, _, _ = to_3d_model_arrays(volume)
        for cls in (UNet3D, ImprovedUNet3D):
            model = cls(base_channels=2).eval()
            expected = predict_volume(model, volume, mode="3d", device="cpu")
            outputs, _ = infer_logits(model, padded, device="cpu", components=(cls is ImprovedUNet3D))
            x, y, z = shape
            actual = outputs["combined"][:, :z, :y, :x].argmax(0).transpose(2, 1, 0)
            np.testing.assert_array_equal(expected, actual)

    def test_context_dependent_model_actually_changes_with_window_scope(self):
        class SpatialMean(torch.nn.Module):
            def forward(self, x):
                centered = x - x.mean(dim=(2, 3, 4), keepdim=True)
                return torch.cat([centered + c for c in range(6)], dim=1)
        image = np.broadcast_to(np.arange(80, dtype=np.float32), (32, 32, 80)).copy()
        full, _ = infer_logits(SpatialMean(), image, device="cpu")
        slide, _ = infer_logits(SpatialMean(), image, device="cpu", window_dhw=(32, 32, 32))
        self.assertGreater(float(np.abs(full["combined"]-slide["combined"]).max()), 1)

    def test_anchored_grid_covers_all_voxels(self):
        self.assertEqual(axis_starts(80, 32), (0, 16, 32, 48))
        self.assertEqual(axis_starts(80, 64), (0, 16))
        with self.assertRaises(ValueError):
            axis_starts(16, 32)

    def test_overlap_and_extra_padding_preserve_image_and_head_sum(self):
        rng = np.random.default_rng(23)
        image = rng.normal(size=(16, 48, 80)).astype(np.float32)
        model = Pointwise()
        full, _ = infer_logits(model, image, device="cpu", components=True)
        sliding, receipt = infer_logits(model, image, device="cpu", components=True,
                                        window_dhw=(32, 32, 64))
        self.assertGreater(receipt["coverage_max"], 1)
        self.assertGreaterEqual(receipt["coverage_min"], 1)
        for name in full:
            np.testing.assert_allclose(sliding[name], full[name], atol=1e-6)
        np.testing.assert_allclose(sliding["combined"],
                                   sliding["main"]+sliding["half"]+sliding["quarter"], atol=1e-6)

    def test_invalid_or_nonfinite_inputs_are_rejected(self):
        for image in (np.zeros((17, 32, 32), dtype=np.float32),
                      np.full((16, 32, 32), np.nan, dtype=np.float32)):
            with self.assertRaises(ValueError):
                infer_logits(Pointwise(), image, device="cpu")


if __name__ == "__main__":
    unittest.main()
