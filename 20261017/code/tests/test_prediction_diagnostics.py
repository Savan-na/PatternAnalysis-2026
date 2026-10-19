"""Fixed-weight replay identity, geometry and GT-free filter safeguards."""
import unittest
import numpy as np
import torch
from hipmri_rewrite.models import ImprovedUNet3D
from hipmri_rewrite.prediction_diagnostics import (capture_instance_stats, replay_instance_stats,
    model_digest, confidence, geometry, conservative_component_filter)


class PredictionDiagnosticTests(unittest.TestCase):
    def test_same_window_replay_isolates_statistics_in_pointwise_network(self):
        torch.set_num_threads(2); torch.manual_seed(19)
        model = torch.nn.Sequential(torch.nn.Conv3d(1, 2, 1),
                                    torch.nn.InstanceNorm3d(2, affine=True),
                                    torch.nn.Conv3d(2, 6, 1)).eval().requires_grad_(False)
        image = torch.arange(64, dtype=torch.float32)[None, None, None, None, :].expand(1, 1, 16, 16, 64)
        patch = image[..., :32]
        with torch.inference_mode():
            with capture_instance_stats(model) as bank:
                full = model(image)
            native_patch = model(patch)
            with replay_instance_stats(model, bank):
                replay_patch = model(patch)
        torch.testing.assert_close(replay_patch, full[..., :32], atol=1e-5, rtol=1e-5)
        self.assertGreater(float((native_patch-replay_patch).abs().max()), .1)

    def test_real_network_replay_identity_and_hook_cleanup(self):
        torch.set_num_threads(2); torch.manual_seed(12)
        model = ImprovedUNet3D(base_channels=2).eval().requires_grad_(False)
        image = torch.randn(1, 1, 32, 48, 48)
        digest = model_digest(model)
        with torch.inference_mode():
            with capture_instance_stats(model) as bank:
                reference = model(image)
            self.assertEqual(len(bank), 14)
            with replay_instance_stats(model, bank):
                replayed = model(image)
            torch.testing.assert_close(reference, replayed, atol=1e-4, rtol=1e-4)
            try:
                with replay_instance_stats(model, bank):
                    raise RuntimeError("synthetic interruption")
            except RuntimeError:
                pass
            torch.testing.assert_close(reference, model(image), atol=0, rtol=0)
        self.assertEqual(digest, model_digest(model))
        self.assertFalse(any(m._forward_hooks or m._forward_pre_hooks for m in model.modules()))

    def test_softmax_confidence_uses_all_six_classes(self):
        logits = np.zeros((6, 2, 2, 2), np.float32); logits[1] = 2
        output = confidence(logits, {"all": np.ones((2, 2, 2), bool)})["all"]
        self.assertAlmostEqual(output["body_probability"]["median"], np.exp(2)/(np.exp(2)+5), places=6)
        self.assertEqual(output["body_probability_ge_0p9"], 0)
        logits[0] = -100  # pairwise margin rises; other classes still compete
        output = confidence(logits, {"all": np.ones((2, 2, 2), bool)})["all"]
        self.assertLess(output["body_probability"]["median"], .9)

    def test_physical_faces_extents_and_sheared_grid_rejection(self):
        region = np.zeros((16, 16, 16), bool); region[0, 0, 0] = True
        affine = np.diag([2., 3., 4., 1.])
        row = geometry(region, np.zeros(region.shape), (2., 3., 4.), affine)
        self.assertEqual(row["distance_to_original_fov_faces_mm"]["median"], 1.)
        self.assertEqual(row["volume_mm3"], 24.)
        self.assertEqual(row["largest_10_components"][0]["bbox_extent_mm"], [2., 3., 4.])
        affine[0, 1] = .5
        with self.assertRaises(ValueError):
            geometry(region, np.zeros(region.shape), (2., 3., 4.), affine)

    def test_filter_uses_both_thresholds_and_best_nonbody_class(self):
        logits = np.zeros((6, 40, 40, 40), np.float32); logits[2] = .5
        logits[1, 10:30, 10:30, 10:30] = 2
        logits[1, 0, 0, 0] = 2
        logits[1, :3, 35:38, 35:38] = 2  # 216 mm3: retain despite relative size
        filtered, record = conservative_component_filter(logits, (2., 2., 2.))
        self.assertEqual(record["removed_voxels"], 1)
        self.assertEqual(filtered[0, 0, 0], 2)
        self.assertEqual(filtered[0, 35, 35], 1)
        self.assertEqual(filtered[15, 15, 15], 1)
        before = logits.argmax(0)
        np.testing.assert_array_equal(filtered[before != 1], before[before != 1])


if __name__ == "__main__":
    unittest.main()
