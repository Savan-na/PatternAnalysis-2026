"""Check physical negative selection, deterministic geometry and safe fallback."""

import unittest
from pathlib import Path

import numpy as np

from hipmri_rewrite.data import Volume
from hipmri_rewrite.sampling import crop_for_3d_model, crop_origin
from hipmri_rewrite.sampling_edges import EdgePatchSelector, boundary_hits
from hipmri_rewrite.splits import Case


def volume(image=None, spacing=(1.0, 1.0, 1.0), split="train"):
    mask = np.zeros((32, 32, 32), dtype=np.uint8)
    mask[12:20, 12:20, 12:20] = 1
    if image is None:
        image = np.arange(mask.size, dtype=np.float32).reshape(mask.shape) + 1
    case = Case("A001", 0, split, Path("unused"), Path("unused"))
    return Volume(case, image, mask, np.eye(4), spacing)


def selector(v, distance=20.0):
    return EdgePatchSelector(v, (16, 16, 16), width=2, distance_mm=distance,
                             min_nonzero_fraction=0.05,
                             min_hard_negative_voxels=8, max_attempts=16)


class EdgeSamplingTests(unittest.TestCase):
    def test_physical_distance_and_corner_crop_preserve_pair(self):
        v = volume()
        first = selector(v)
        # At (0,12,12), the nearest Body is 12 voxels away: <20 mm.
        self.assertFalse(first.hard_negative[0, 12, 12])
        anisotropic = selector(volume(spacing=(2.0, 1.0, 1.0)))
        self.assertTrue(anisotropic.hard_negative[0, 12, 12])
        chosen = anisotropic.select(seed=17, epoch=0, draw=9)
        self.assertTrue(chosen.accepted)
        self.assertEqual(chosen, anisotropic.select(seed=17, epoch=0, draw=9))
        self.assertEqual(boundary_hits(chosen.origin, v.mask.shape, (16, 16, 16), 2),
                         (True, True))
        image, mask = crop_for_3d_model(anisotropic.volume, chosen.origin, (16, 16, 16))
        x, y, z = chosen.origin
        np.testing.assert_array_equal(image[0].transpose(2, 1, 0),
                                      v.image[x:x+16, y:y+16, z:z+16])
        np.testing.assert_array_equal(mask.transpose(2, 1, 0),
                                      v.mask[x:x+16, y:y+16, z:z+16])

    def test_zero_mri_cannot_supply_textured_negatives_and_falls_back(self):
        v = volume(image=np.zeros((32, 32, 32), dtype=np.float32))
        chosen = selector(v).select(seed=17, epoch=3, draw=9)
        self.assertFalse(chosen.accepted)
        self.assertIsNone(chosen.corner)
        self.assertEqual(chosen.origin, crop_origin(v.mask, (16, 16, 16), "uniform",
                         seed=17, epoch=3, case_id=v.case.identifier, draw=9))

    def test_rejects_non_train_data_and_missing_body(self):
        with self.assertRaises(ValueError):
            selector(volume(split="validation"))
        v = volume()
        v.mask[:] = 0
        with self.assertRaises(ValueError):
            selector(v)


if __name__ == "__main__":
    unittest.main()
