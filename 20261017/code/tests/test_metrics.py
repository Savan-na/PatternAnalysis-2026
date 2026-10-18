import unittest

import numpy as np

from hipmri_rewrite.metrics import confusion, overlap


class OverlapTests(unittest.TestCase):
    def test_false_positive_and_negative_counting(self):
        truth = np.array([[[5, 5], [4, 0]]], dtype=np.uint8)
        prediction = np.array([[[5, 4], [5, 0]]], dtype=np.uint8)
        matrix = confusion(truth, prediction)
        self.assertEqual(matrix[5, 5], 1)
        self.assertEqual(matrix[5, 4], 1)
        self.assertEqual(matrix[4, 5], 1)
        scores = overlap(matrix)
        self.assertAlmostEqual(scores["dice"][5], 0.5)
        self.assertAlmostEqual(scores["iou"][5], 1 / 3)
        self.assertIsNone(scores["dice"][3])


if __name__ == "__main__":
    unittest.main()
