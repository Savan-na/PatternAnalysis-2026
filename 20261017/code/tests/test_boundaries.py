import unittest

import numpy as np

from hipmri_rewrite.boundaries import prostate_axial_regions, prostate_organ_confusion, slice_dice


class BoundaryTests(unittest.TestCase):
    def test_base_apex_follow_physical_superior_direction(self):
        truth = np.zeros((8, 8, 10), dtype=np.uint8)
        truth[2:6, 2:6, 1:9] = 5
        superior_affine = np.diag([1., 1., 2., 1.])
        inferior_affine = np.diag([1., 1., -2., 1.])
        self.assertEqual(prostate_axial_regions(truth, superior_affine)["apex"], (1, 2))
        self.assertEqual(prostate_axial_regions(truth, inferior_affine)["apex"], (8, 7))

    def test_slice_failure_and_organ_leakage(self):
        truth = np.zeros((4, 4, 3), dtype=np.uint8)
        prediction = np.zeros_like(truth)
        truth[0, 0, 1] = 5
        truth[1, 1, 1] = 3
        truth[2, 2, 1] = 4
        prediction[1, 1, 1] = 5
        prediction[2, 2, 1] = 5
        self.assertEqual(slice_dice(truth, prediction), [None, 0.0, None])
        self.assertEqual(prostate_organ_confusion(truth, prediction), {
            "prostate_pred_in_true_bladder": 1,
            "prostate_pred_in_true_rectum": 1,
            "true_prostate_pred_other": 1,
        })


try:
    import scipy  # noqa: F401
except ImportError:
    scipy = None


@unittest.skipIf(scipy is None, "SciPy is not installed")
class SurfaceTests(unittest.TestCase):
    def test_identical_and_empty_masks(self):
        from hipmri_rewrite.boundaries import symmetric_surface_distances

        truth = np.zeros((10, 10, 10), dtype=np.uint8)
        truth[2:5, 2:5, 2:5] = 5
        identical = symmetric_surface_distances(truth, truth, (1., 2., 3.))
        self.assertEqual(identical["status"], "both_present")
        self.assertEqual(identical["hd95_mm"], 0)
        self.assertEqual(identical["assd_mm"], 0)
        self.assertEqual(symmetric_surface_distances(truth, np.zeros_like(truth), (1., 2., 3.))["status"], "one_empty")


if __name__ == "__main__":
    unittest.main()
