"""Check the candidate crop audit's geometry and repeatability contract."""

import importlib.util
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "tools/audit_g3_candidate_sampling.py"
SPEC = importlib.util.spec_from_file_location("candidate_sampling_audit", SCRIPT)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class CandidateCropTests(unittest.TestCase):
    def test_target_crop_contains_edge_voxel_and_exact_patch_shape(self):
        mask = np.zeros((9, 10, 8), dtype=np.uint8)
        mask[8, 0, 7] = 5
        patch = (4, 5, 3)
        origin = AUDIT.target_origin(mask.shape, patch, (8, 0, 7),
                                     AUDIT.fixed_rng(7, "synthetic", 0))
        counts = AUDIT.patch_counts(mask, origin, patch)
        self.assertEqual(int(counts[5]), 1)
        self.assertEqual(int(sum(counts)), 60)
        self.assertTrue(all(0 <= start <= n - size
                            for start, n, size in zip(origin, mask.shape, patch)))

    def test_uniform_draw_is_reproducible_and_in_bounds(self):
        shape, patch = (128, 128, 64), (64, 64, 32)
        first = AUDIT.uniform_origin(shape, patch, AUDIT.fixed_rng(371017, "B006_Week0", 4))
        second = AUDIT.uniform_origin(shape, patch, AUDIT.fixed_rng(371017, "B006_Week0", 4))
        self.assertEqual(first, second)
        self.assertTrue(all(0 <= start <= n - size
                            for start, n, size in zip(first, shape, patch)))


if __name__ == "__main__":
    unittest.main()
