import unittest
import tempfile
from pathlib import Path

import numpy as np

from hipmri_rewrite.data import Volume, from_3d_model_prediction, one_hot, to_3d_model_arrays, zscore_nonzero
from hipmri_rewrite.splits import Case


class DataContractTests(unittest.TestCase):
    def setUp(self):
        self.case = Case("K019", 1, "train", Path("image.nii.gz"), Path("mask.nii.gz"))

    def test_one_hot_class_axis_and_invalid_label(self):
        labels = np.array([[0, 4], [5, 1]], dtype=np.uint8)
        encoded = one_hot(labels)
        self.assertEqual(encoded.shape, (6, 2, 2))
        np.testing.assert_array_equal(encoded.sum(axis=0), 1)
        self.assertEqual(encoded[5, 1, 0], 1)
        with self.assertRaisesRegex(ValueError, "outside"):
            one_hot(np.array([[6]], dtype=np.uint8))

    def test_normalization_keeps_mask_and_outside_zeros(self):
        image = np.zeros((2, 2, 2), dtype=np.float32)
        image[0, 0, 0] = 2
        image[1, 1, 1] = 4
        mask = np.zeros((2, 2, 2), dtype=np.uint8)
        mask[1, 1, 1] = 5
        source = Volume(self.case, image, mask, np.eye(4), (1., 1., 1.))
        result = zscore_nonzero(source)
        self.assertEqual(result.image[0, 0, 0], -1)
        self.assertEqual(result.image[1, 1, 1], 1)
        self.assertEqual(result.image[0, 0, 1], 0)
        np.testing.assert_array_equal(result.mask, mask)

    def test_model_axis_and_padding_roundtrip(self):
        image = np.zeros((17, 20, 18), dtype=np.float32)
        mask = np.zeros_like(image, dtype=np.uint8)
        mask[16, 19, 17] = 5
        source = Volume(self.case, image, mask, np.eye(4), (1., 1., 1.))
        model_image, model_mask, original_shape = to_3d_model_arrays(source)
        self.assertEqual(model_image.shape, (32, 32, 32))
        self.assertEqual(model_mask[17, 19, 16], 5)
        np.testing.assert_array_equal(from_3d_model_prediction(model_mask, original_shape), mask)


try:
    import scipy  # noqa: F401
except ImportError:
    scipy = None


@unittest.skipIf(scipy is None, "SciPy is not installed")
class DownsamplingGeometryTests(unittest.TestCase):
    def test_common_grid_and_affine_preserve_physical_sample_location(self):
        from hipmri_rewrite.data import downsample

        case = Case("K019", 1, "train", Path("image.nii.gz"), Path("mask.nii.gz"))
        source_affine = np.diag([1.5, 2., 3., 1.])
        image = np.arange(8, dtype=np.float32)[:, None, None] * np.ones((1, 8, 8), dtype=np.float32)
        mask = np.zeros((8, 8, 8), dtype=np.uint8)
        mask[2:6, 2:6, 2:6] = 5
        result = downsample(Volume(case, image, mask, source_affine, (1.5, 2., 3.)))
        self.assertEqual(result.image.shape, (4, 4, 4))
        self.assertEqual(result.mask.shape, (4, 4, 4))
        self.assertEqual(set(np.unique(result.mask)), {0, 5})
        output_world = result.affine @ np.array([1., 1., 1., 1.])
        input_world = source_affine @ np.array([2.5, 2.5, 2.5, 1.])
        np.testing.assert_allclose(output_world, input_world)
        self.assertAlmostEqual(float(result.image[1, 1, 1]), 2.5)

    def test_lost_small_organ_stops_pipeline(self):
        from hipmri_rewrite.data import downsample

        case = Case("K019", 1, "train", Path("image.nii.gz"), Path("mask.nii.gz"))
        image = np.ones((8, 8, 8), dtype=np.float32)
        mask = np.zeros_like(image, dtype=np.uint8)
        mask[0, 0, 0] = 5
        with self.assertRaisesRegex(ValueError, "removed classes"):
            downsample(Volume(case, image, mask, np.eye(4), (1., 1., 1.)))


try:
    import nibabel as nib
except ImportError:
    nib = None


@unittest.skipIf(nib is None, "NiBabel is not installed")
class NiftiPairTests(unittest.TestCase):
    def test_real_nifti_decoding_and_grid_rejection(self):
        from hipmri_rewrite.data import load_case

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            mri_path = root / "K019_Week1_LFOV.nii.gz"
            mask_path = root / "K019_Week1_SEMANTIC.nii.gz"
            image = np.ones((16, 16, 16), dtype=np.float32)
            mask = np.zeros_like(image, dtype=np.uint8)
            mask[5:7, 5:7, 5:7] = 5
            affine = np.diag([1.5, 1.5, 1.5, 1.])
            mri = nib.Nifti1Image(image, affine)
            seg = nib.Nifti1Image(mask, affine)
            mri.header.set_xyzt_units("mm")
            seg.header.set_xyzt_units("mm")
            nib.save(mri, str(mri_path))
            nib.save(seg, str(mask_path))
            case = Case("K019", 1, "train", mri_path, mask_path)
            loaded = load_case(case)
            self.assertEqual(set(np.unique(loaded.mask)), {0, 5})
            self.assertEqual(loaded.spacing_mm, (1.5, 1.5, 1.5))
            shifted = nib.Nifti1Image(mask, affine @ np.array([
                [1, 0, 0, 2], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]
            ]))
            shifted.header.set_xyzt_units("mm")
            nib.save(shifted, str(mask_path))
            with self.assertRaisesRegex(ValueError, "affines differ"):
                load_case(case)


if __name__ == "__main__":
    unittest.main()
