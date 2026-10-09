"""CPU-only tests of real NIfTI decoding and one-pair spatial validation."""

import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from recognition.hipmri_3d_improved_unet_hard.dataset import VolumePair
from recognition.hipmri_3d_improved_unet_hard.volume_io import (
    AFFINE_ATOL,
    LoadedVolumePair,
    load_volume_pair,
)


class VolumeIOTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        self.pair = VolumePair(
            "K019",
            1,
            root / "K019_Week1_LFOV.nii.gz",
            root / "K019_Week1_SEMANTIC.nii.gz",
        )

    def write_pair(
        self,
        *,
        mri: np.ndarray | None = None,
        segmentation: np.ndarray | None = None,
        mri_affine: np.ndarray | None = None,
        segmentation_affine: np.ndarray | None = None,
        mri_unit: str = "mm",
        segmentation_unit: str = "mm",
        mri_scaling: tuple[float, float] | None = None,
        segmentation_scaling: tuple[float, float] | None = None,
    ) -> None:
        if mri is None:
            mri = np.arange(24, dtype=np.int16).reshape(2, 3, 4)
        if segmentation is None:
            segmentation = np.zeros((2, 3, 4), dtype=np.int16)
            segmentation[0, 0, 0] = 3
            segmentation[0, 0, 1] = 5
        if mri_affine is None:
            mri_affine = np.diag([1.0, 1.5, 2.0, 1.0])
        if segmentation_affine is None:
            segmentation_affine = mri_affine.copy()

        mri_image = nib.Nifti1Image(mri, mri_affine)
        segmentation_image = nib.Nifti1Image(segmentation, segmentation_affine)
        mri_image.header.set_xyzt_units(mri_unit)
        segmentation_image.header.set_xyzt_units(segmentation_unit)
        if mri_scaling is not None:
            mri_image.header.set_slope_inter(*mri_scaling)
        if segmentation_scaling is not None:
            segmentation_image.header.set_slope_inter(*segmentation_scaling)
        nib.save(mri_image, str(self.pair.mri_path))
        nib.save(segmentation_image, str(self.pair.segmentation_path))

    def test_valid_pair_preserves_shape_dtype_and_metadata(self) -> None:
        self.write_pair()
        loaded = load_volume_pair(self.pair)
        self.assertIsInstance(loaded, LoadedVolumePair)
        self.assertEqual(loaded.pair, self.pair)
        self.assertEqual(loaded.mri.shape, (2, 3, 4))
        self.assertEqual(loaded.segmentation.shape, (2, 3, 4))
        self.assertEqual(loaded.mri.dtype, np.dtype("float32"))
        self.assertEqual(loaded.segmentation.dtype, np.dtype("uint8"))
        self.assertEqual(loaded.mri_stored_dtype, np.dtype("int16"))
        self.assertEqual(loaded.segmentation_stored_dtype, np.dtype("int16"))
        self.assertEqual(loaded.voxel_spacing, (1.0, 1.5, 2.0))
        self.assertEqual(loaded.spatial_unit, "mm")
        self.assertEqual(loaded.segmentation_header_spatial_unit, "mm")
        self.assertFalse(loaded.segmentation_unit_inferred)
        self.assertEqual(loaded.axis_codes, ("R", "A", "S"))
        self.assertEqual(loaded.labels_present, (0, 3, 5))
        np.testing.assert_array_equal(loaded.mri, np.arange(24).reshape(2, 3, 4))
        np.testing.assert_array_equal(loaded.mri_affine, loaded.segmentation_affine)

    def test_depth_144_is_preserved(self) -> None:
        self.write_pair(
            mri=np.zeros((2, 3, 144), dtype=np.float32),
            segmentation=np.zeros((2, 3, 144), dtype=np.uint8),
        )
        loaded = load_volume_pair(self.pair)
        self.assertEqual(loaded.mri.shape, (2, 3, 144))
        self.assertEqual(loaded.labels_present, (0,))

    def test_scaled_values_are_decoded_before_label_conversion(self) -> None:
        mask = np.zeros((2, 3, 4), dtype=np.int16)
        mask[0, 0, 0] = 1
        mask[0, 0, 1] = 2
        self.write_pair(
            segmentation=mask,
            mri_scaling=(2.0, 10.0),
            segmentation_scaling=(2.0, 0.0),
        )
        loaded = load_volume_pair(self.pair)
        self.assertEqual(float(loaded.mri[0, 0, 0]), 10.0)
        self.assertEqual(float(loaded.mri[0, 0, 1]), 12.0)
        self.assertEqual(loaded.labels_present, (0, 2, 4))
        self.assertEqual(loaded.segmentation.dtype, np.dtype("uint8"))
        self.assertEqual(loaded.segmentation_stored_dtype, np.dtype("int16"))

    def test_missing_file_identifies_case_and_role(self) -> None:
        self.write_pair()
        self.pair.segmentation_path.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "K019_Week1: segmentation NIfTI file is missing"):
            load_volume_pair(self.pair)

    def test_corrupt_file_identifies_case_and_role(self) -> None:
        self.write_pair()
        self.pair.segmentation_path.write_bytes(b"not a NIfTI file")
        with self.assertRaisesRegex(ValueError, "K019_Week1: cannot read segmentation NIfTI"):
            load_volume_pair(self.pair)

    def test_four_dimensional_mri_is_rejected(self) -> None:
        self.write_pair(mri=np.zeros((2, 3, 4, 1), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "K019_Week1: expected 3D MRI and segmentation"):
            load_volume_pair(self.pair)

    def test_shape_mismatch_is_rejected(self) -> None:
        self.write_pair(segmentation=np.zeros((2, 3, 5), dtype=np.uint8))
        with self.assertRaisesRegex(ValueError, "K019_Week1: MRI/segmentation shape mismatch"):
            load_volume_pair(self.pair)

    def test_affine_mismatch_with_same_shape_and_spacing_is_rejected(self) -> None:
        mask_affine = np.diag([1.0, 1.5, 2.0, 1.0])
        mask_affine[0, 3] = 0.1
        self.write_pair(segmentation_affine=mask_affine)
        with self.assertRaisesRegex(ValueError, "K019_Week1: MRI/segmentation affine mismatch"):
            load_volume_pair(self.pair)

    def test_tiny_affine_rounding_difference_is_allowed(self) -> None:
        mask_affine = np.diag([1.0, 1.5, 2.0, 1.0])
        mask_affine[0, 3] = AFFINE_ATOL / 2
        self.write_pair(segmentation_affine=mask_affine)
        loaded = load_volume_pair(self.pair)
        self.assertEqual(loaded.labels_present, (0, 3, 5))

    def test_spacing_mismatch_is_rejected(self) -> None:
        mask_affine = np.diag([1.0, 1.5, 2.1, 1.0])
        self.write_pair(segmentation_affine=mask_affine)
        with self.assertRaisesRegex(ValueError, "K019_Week1: voxel spacing mismatch"):
            load_volume_pair(self.pair)

    def test_spatial_unit_mismatch_is_rejected(self) -> None:
        self.write_pair(segmentation_unit="meter")
        with self.assertRaisesRegex(ValueError, "K019_Week1: spatial unit mismatch"):
            load_volume_pair(self.pair)

    def test_matching_explicit_meter_units_are_accepted(self) -> None:
        self.write_pair(mri_unit="meter", segmentation_unit="meter")
        loaded = load_volume_pair(self.pair)
        self.assertEqual(loaded.spatial_unit, "meter")
        self.assertEqual(loaded.segmentation_header_spatial_unit, "meter")
        self.assertFalse(loaded.segmentation_unit_inferred)

    def test_mm_mri_and_unknown_mask_unit_with_matching_grid_are_accepted(self) -> None:
        self.write_pair(segmentation_unit="unknown")
        loaded = load_volume_pair(self.pair)
        self.assertEqual(loaded.spatial_unit, "mm")
        self.assertEqual(loaded.segmentation_header_spatial_unit, "unknown")
        self.assertTrue(loaded.segmentation_unit_inferred)
        self.assertEqual(loaded.axis_codes, ("R", "A", "S"))
        np.testing.assert_array_equal(loaded.mri_affine, loaded.segmentation_affine)

    def test_unknown_mask_unit_does_not_excuse_affine_mismatch(self) -> None:
        mask_affine = np.diag([1.0, 1.5, 2.0, 1.0])
        mask_affine[0, 3] = 0.1
        self.write_pair(segmentation_affine=mask_affine, segmentation_unit="unknown")
        with self.assertRaisesRegex(ValueError, "K019_Week1: MRI/segmentation affine mismatch"):
            load_volume_pair(self.pair)

    def test_unknown_mask_unit_does_not_excuse_spacing_mismatch(self) -> None:
        mask_affine = np.diag([1.0, 1.5, 2.1, 1.0])
        self.write_pair(segmentation_affine=mask_affine, segmentation_unit="unknown")
        with self.assertRaisesRegex(ValueError, "K019_Week1: voxel spacing mismatch"):
            load_volume_pair(self.pair)

    def test_unknown_mri_unit_cannot_be_inferred(self) -> None:
        self.write_pair(mri_unit="unknown", segmentation_unit="unknown")
        with self.assertRaisesRegex(ValueError, "K019_Week1: MRI spatial unit 'unknown'"):
            load_volume_pair(self.pair)

    def test_meter_mri_and_unknown_mask_unit_are_rejected(self) -> None:
        self.write_pair(mri_unit="meter", segmentation_unit="unknown")
        with self.assertRaisesRegex(ValueError, "K019_Week1: spatial unit mismatch"):
            load_volume_pair(self.pair)

    def test_non_finite_mri_is_rejected(self) -> None:
        for value in (np.nan, np.inf):
            with self.subTest(value=value):
                mri = np.zeros((2, 3, 4), dtype=np.float32)
                mri[0, 0, 0] = value
                self.write_pair(mri=mri)
                with self.assertRaisesRegex(ValueError, "K019_Week1: MRI contains non-finite"):
                    load_volume_pair(self.pair)

    def test_fractional_segmentation_is_rejected(self) -> None:
        segmentation = np.zeros((2, 3, 4), dtype=np.float32)
        segmentation[0, 0, 0] = 1.5
        self.write_pair(segmentation=segmentation)
        with self.assertRaisesRegex(ValueError, "K019_Week1: segmentation contains fractional"):
            load_volume_pair(self.pair)

    def test_fractional_label_created_by_nifti_scaling_is_rejected(self) -> None:
        segmentation = np.zeros((2, 3, 4), dtype=np.int16)
        segmentation[0, 0, 0] = 1
        self.write_pair(segmentation=segmentation, segmentation_scaling=(0.5, 0.0))
        with self.assertRaisesRegex(ValueError, "K019_Week1: segmentation contains fractional"):
            load_volume_pair(self.pair)

    def test_non_finite_segmentation_is_rejected(self) -> None:
        for value in (np.nan, np.inf):
            with self.subTest(value=value):
                segmentation = np.zeros((2, 3, 4), dtype=np.float32)
                segmentation[0, 0, 0] = value
                self.write_pair(segmentation=segmentation)
                with self.assertRaisesRegex(ValueError, "K019_Week1: segmentation contains non-finite"):
                    load_volume_pair(self.pair)

    def test_out_of_range_segmentation_is_rejected(self) -> None:
        for value in (-1, 6):
            with self.subTest(value=value):
                segmentation = np.zeros((2, 3, 4), dtype=np.int16)
                segmentation[0, 0, 0] = value
                self.write_pair(segmentation=segmentation)
                with self.assertRaisesRegex(ValueError, "K019_Week1: segmentation labels must be within 0-5"):
                    load_volume_pair(self.pair)


if __name__ == "__main__":
    unittest.main()
