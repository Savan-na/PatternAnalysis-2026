"""Check the optional 2D crop's label exposure and paired orientation."""

import unittest

import numpy as np

from hipmri_rewrite.data import Volume
from hipmri_rewrite.sampling_2d import crop_for_2d_model, slice_crop_origin


class TwoDSamplingTests(unittest.TestCase):
    def test_target_draw_contains_prostate_and_preserves_alignment(self):
        mask = np.zeros((128, 128, 64), dtype=np.uint8)
        mask[126, 2, 37] = 5
        image = np.zeros(mask.shape, dtype=np.float32)
        image[126, 2, 37] = 13.0
        volume = Volume(None, image, mask, np.eye(4), (1.0, 1.0, 1.0))
        origin = slice_crop_origin(mask, "Prostate", seed=371017, epoch=0,
                                   case_id="synthetic", draw=4)
        self.assertEqual(origin, slice_crop_origin(mask, "Prostate", seed=371017,
                                                    epoch=0, case_id="synthetic", draw=4))
        x, label = crop_for_2d_model(volume, origin)
        self.assertEqual(x.shape, (1, 64, 64))
        self.assertEqual(label.shape, (64, 64))
        yy, xx = np.argwhere(label == 5)[0]
        self.assertEqual(float(x[0, yy, xx]), 13.0)

    def test_uniform_draw_has_valid_bounds(self):
        mask = np.zeros((128, 128, 72), dtype=np.uint8)
        origin = slice_crop_origin(mask, "uniform", seed=1, epoch=2,
                                   case_id="synthetic", draw=3)
        self.assertTrue(0 <= origin[0] <= 64)
        self.assertTrue(0 <= origin[1] <= 64)
        self.assertTrue(0 <= origin[2] < 72)


if __name__ == "__main__":
    unittest.main()
