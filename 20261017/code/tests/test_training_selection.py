"""Selection must weight patients equally after pooling each one's visits."""

import unittest

import numpy as np

from hipmri_rewrite.training import patient_small_organ_score


class PatientSelectionTests(unittest.TestCase):
    def test_per_patient_score_and_missed_small_organ(self):
        good = np.zeros((6, 6), dtype=np.int64)
        poor = np.zeros((6, 6), dtype=np.int64)
        for label in (3, 4, 5):
            good[label, label] = 100
            poor[label, 1] = 100
        score, detail = patient_small_organ_score({"many_visits": good * 8,
                                                   "one_visit": poor})
        self.assertEqual(score, 0.5)
        self.assertEqual(detail["many_visits"]["Prostate"], 1.0)
        self.assertEqual(detail["one_visit"]["Prostate"], 0.0)

    def test_absent_absent_is_excluded(self):
        matrix = np.zeros((6, 6), dtype=np.int64)
        matrix[3, 3] = 10
        score, detail = patient_small_organ_score({"patient": matrix})
        self.assertEqual(score, 1.0)
        self.assertIsNone(detail["patient"]["Rectum"])


if __name__ == "__main__":
    unittest.main()
