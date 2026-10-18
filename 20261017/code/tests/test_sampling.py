"""Check patient balance, paired patch geometry and measured exposure."""

import unittest
from pathlib import Path

import numpy as np

from hipmri_rewrite.data import Volume
from hipmri_rewrite.sampling import (ExposureRecorder, crop_for_3d_model,
                                     crop_origin, epoch_cases)
from hipmri_rewrite.splits import Case


def case(patient, week, split="train"):
    return Case(patient, week, split, Path("unused_mri"), Path("unused_mask"))


class PatientBalancedSamplingTests(unittest.TestCase):
    def test_one_case_per_patient_and_cycle_visits(self):
        cases = (case("B002", 2), case("A001", 0), case("B002", 0),
                 case("B002", 1), case("A001", 1))
        for epoch in range(6):
            selected = epoch_cases(cases, epoch, 17)
            self.assertEqual({c.patient for c in selected}, {"A001", "B002"})
            self.assertEqual(len(selected), 2)
            self.assertEqual(selected, epoch_cases(cases, epoch, 17))
        self.assertEqual({next(c.week for c in epoch_cases(cases, e, 17)
                               if c.patient == "B002") for e in range(3)}, {0, 1, 2})
        with self.assertRaises(ValueError):
            epoch_cases((case("A001", 0, "validation"),), 0, 17)

    def test_target_patch_reaches_edge_and_image_mask_stay_paired(self):
        mask = np.zeros((9, 10, 8), dtype=np.uint8)
        mask[8, 0, 7] = 5
        image = mask.astype(np.float32) * 7
        origin = crop_origin(mask, (4, 5, 3), "Prostate", seed=17,
                             epoch=0, case_id="A001_Week0", draw=0)
        volume = Volume(case("A001", 0), image, mask, np.eye(4), (1, 1, 1))
        patch_image, patch_mask = crop_for_3d_model(volume, origin, (4, 5, 3))
        self.assertEqual(patch_image.shape, (1, 3, 5, 4))
        self.assertEqual(patch_mask.shape, (3, 5, 4))
        self.assertEqual(int((patch_mask == 5).sum()), 1)
        np.testing.assert_array_equal(patch_image[0], patch_mask * 7)

    def test_exposure_measures_supplied_mask_and_rejects_invalid_label(self):
        recorder = ExposureRecorder()
        mask = np.zeros((2, 3, 4), dtype=np.int64)
        mask[0, 0, 0] = 5
        counts = recorder.add(mask, "A001")
        self.assertEqual(counts["Prostate"], 1)
        self.assertEqual(recorder.summary()["per_class"]["Prostate"]["patch_hit_rate"], 1)
        self.assertEqual(recorder.summary()["per_class"]["Prostate"]["voxel_share"], 1 / 24)
        self.assertEqual(recorder.summary()["per_patient"]["A001"]["patches"], 1)
        mask[0, 0, 0] = 6
        with self.assertRaises(ValueError):
            recorder.add(mask, "A001")


if __name__ == "__main__":
    unittest.main()
