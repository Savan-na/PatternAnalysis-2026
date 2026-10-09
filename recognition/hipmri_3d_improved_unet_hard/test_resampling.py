"""Synthetic CPU regression tests for physical-grid HipMRI resampling."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from .dataset import VolumePair
from .resampling import (
    BOUNDARY_VALUE,
    MRI_BOUNDARY_MODE,
    SEGMENTATION_BOUNDARY_MODE,
    make_target_grid,
    resample_volume_pair,
)
from .volume_io import LoadedVolumePair


CAP = 536870912


def loaded_pair(
    shape=(4, 5, 6), affine=None, mask=None, mask_affine=None, inferred=True
):
    if affine is None:
        affine = np.eye(4, dtype=np.float64)
    if mask is None:
        mask = np.zeros(shape, dtype=np.uint8)
    mri = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    axes = np.asarray(affine, dtype=np.float64)[:3, :3]
    world_codes = (("L", "R"), ("P", "A"), ("I", "S"))
    axis_codes = tuple(
        world_codes[axis][int(axes[axis, column] > 0)]
        for column in range(3)
        for axis in (int(np.argmax(np.abs(axes[:, column]))),)
    )
    return LoadedVolumePair(
        pair=VolumePair("K019", 1, Path("MRI.nii.gz"), Path("mask.nii.gz")),
        mri=mri,
        segmentation=mask,
        mri_affine=np.asarray(affine, dtype=np.float64),
        segmentation_affine=np.asarray(mask_affine if mask_affine is not None else affine, dtype=np.float64),
        voxel_spacing=tuple(float(v) for v in np.linalg.norm(np.asarray(affine)[:3, :3], axis=0)),
        spatial_unit="mm",
        segmentation_header_spatial_unit="unknown" if inferred else "mm",
        segmentation_unit_inferred=inferred,
        axis_codes=axis_codes,
        mri_stored_dtype=np.dtype("float32"),
        segmentation_stored_dtype=np.dtype("uint8"),
        labels_present=tuple(int(v) for v in np.unique(mask)),
    )


class ResamplingTests(unittest.TestCase):
    def test_boundary_policy_matches_configuration(self):
        self.assertEqual(MRI_BOUNDARY_MODE, "nearest")
        self.assertEqual(SEGMENTATION_BOUNDARY_MODE, "grid-constant")
        self.assertEqual(BOUNDARY_VALUE, 0)

    def test_synthetic_axis_codes_follow_affine(self):
        self.assertEqual(loaded_pair().axis_codes, ("R", "A", "S"))
        rotated = np.eye(4)
        rotated[:3, :3] = ((0, -2, 0), (1, 0, 0), (0, 0, 3))
        self.assertEqual(loaded_pair(affine=rotated).axis_codes, ("A", "L", "S"))

    def test_six_face_labels_survive_within_voxel_edge_bounds(self):
        shape = (16, 16, 4)
        mask = np.zeros(shape, dtype=np.uint8)
        mask[[0, -1], :, :] = 5
        mask[:, [0, -1], :] = 5
        mask[:, :, [0, -1]] = 5
        affine = np.diag([1.875, 1.875, 1.875, 1.0])
        loaded = loaded_pair(shape=shape, affine=affine, mask=mask)
        loaded.mri.fill(100)
        result = resample_volume_pair(
            loaded, (1.6796875, 1.6796875, 1.875), max_estimated_bytes=CAP
        )
        self.assertEqual(result.segmentation.shape, (18, 18, 4))
        pull = np.linalg.solve(affine, result.affine)
        first = pull[0, 3]
        last = first + (result.segmentation.shape[0] - 1) * pull[0, 0]
        self.assertTrue(-0.5 <= first < 0)
        self.assertTrue(15 < last <= 15.5)
        for index in ((0, 4, 1), (-1, 4, 1), (4, 0, 1), (4, -1, 1), (4, 4, 0), (4, 4, -1)):
            with self.subTest(index=index):
                self.assertEqual(result.segmentation[index], 5)
        # MRI edge intensity is extended rather than attenuated toward zero.
        self.assertAlmostEqual(float(result.mri[0, 4, 1]), 100)

    def test_mask_outside_voxel_edge_domain_is_background(self):
        shape = (4, 4, 4)
        mask = np.full(shape, 5, dtype=np.uint8)
        affine = np.diag([1.875, 1.875, 1.875, 1.0])
        result = resample_volume_pair(
            loaded_pair(shape=shape, affine=affine, mask=mask),
            (1.6796875, 1.6796875, 1.875), max_estimated_bytes=CAP,
        )
        self.assertEqual(result.segmentation.shape, (5, 5, 4))
        pull = np.linalg.solve(affine, result.affine)
        last = pull[0, 3] + 4 * pull[0, 0]
        self.assertGreater(last, 3.5)
        self.assertEqual(result.segmentation[0, 2, 1], 5)
        self.assertEqual(result.segmentation[-1, 2, 1], 0)

    def test_identity_preserves_arrays_shape_affine(self):
        mask = np.zeros((4, 5, 6), dtype=np.uint8)
        mask[1, 2, 3] = 5
        loaded = loaded_pair(mask=mask)
        result = resample_volume_pair(loaded, (1, 1, 1), max_estimated_bytes=CAP)
        np.testing.assert_array_equal(result.mri, loaded.mri)
        np.testing.assert_array_equal(result.segmentation, loaded.segmentation)
        np.testing.assert_array_equal(result.affine, loaded.mri_affine)
        self.assertEqual(result.mri.shape, (4, 5, 6))
        self.assertEqual(result.mri.dtype, np.float32)
        self.assertEqual(result.segmentation.dtype, np.uint8)

    def test_anisotropic_edges_covered_and_directions_retained(self):
        affine = np.diag([2., 3., 4., 1.])
        shape, output = make_target_grid((4, 5, 6), affine, (1, 2, 3))
        self.assertEqual(shape, (8, 8, 8))
        np.testing.assert_allclose(np.linalg.norm(output[:3, :3], axis=0), (1, 2, 3))
        source_low = affine[:3, 3] - np.diag(affine[:3, :3]) / 2
        source_high = affine[:3, 3] + np.diag(affine[:3, :3]) * (np.array((4, 5, 6)) - .5)
        out_low = output[:3, 3] - np.diag(output[:3, :3]) / 2
        out_high = output[:3, 3] + np.diag(output[:3, :3]) * (np.array(shape) - .5)
        self.assertTrue(np.all(out_low <= source_low + 1e-8))
        self.assertTrue(np.all(out_high >= source_high - 1e-8))

    def test_translated_affine(self):
        affine = np.eye(4)
        affine[:3, 3] = (13, -9, 21)
        shape, output = make_target_grid((4, 5, 6), affine, (1, 1, 1))
        self.assertEqual(shape, (4, 5, 6))
        np.testing.assert_allclose(output, affine)
        result = resample_volume_pair(loaded_pair(affine=affine), (1, 1, 1), max_estimated_bytes=CAP)
        np.testing.assert_allclose(result.affine, affine)

    def test_rotated_affine(self):
        affine = np.eye(4)
        affine[:3, :3] = ((0, -2, 0), (1, 0, 0), (0, 0, 3))
        affine[:3, 3] = (10, 20, 30)
        shape, output = make_target_grid((4, 5, 6), affine, (1, 1, 1.5))
        self.assertEqual(shape, (4, 10, 12))
        np.testing.assert_allclose(output[:3, :3], ((0, -1, 0), (1, 0, 0), (0, 0, 1.5)))
        result = resample_volume_pair(loaded_pair(affine=affine), (1, 1, 1.5), max_estimated_bytes=CAP)
        np.testing.assert_allclose(result.affine, output)

    def test_shared_grid_with_tolerated_source_affine_difference(self):
        mask = np.zeros((4, 5, 6), dtype=np.uint8)
        mask[1:3, 1:4, 1:5] = 5
        shifted = np.eye(4)
        shifted[0, 3] = 1e-5
        result = resample_volume_pair(
            loaded_pair(mask=mask, mask_affine=shifted), (.5, .5, .5), max_estimated_bytes=CAP
        )
        self.assertEqual(result.mri.shape, result.segmentation.shape)
        self.assertEqual(result.affine.shape, (4, 4))
        self.assertIn(5, result.output_labels)

    def test_nearest_preserves_all_six_category_ids(self):
        mask = np.zeros((4, 5, 6), dtype=np.uint8)
        mask[0, :, :] = 1
        mask[1, :, :] = 2
        mask[2, :, :] = 3
        mask[3, 0:2, :] = 4
        mask[3, 2:4, :] = 5
        result = resample_volume_pair(loaded_pair(mask=mask), (.5, .5, .5), max_estimated_bytes=CAP)
        self.assertEqual(result.output_labels, (0, 1, 2, 3, 4, 5))
        self.assertEqual(set(np.unique(result.segmentation)), set(range(6)))

    def test_single_voxel_foreground_survives_upsampling(self):
        mask = np.zeros((4, 5, 6), dtype=np.uint8)
        mask[2, 2, 2] = 5
        result = resample_volume_pair(loaded_pair(mask=mask), (.5, .5, .5), max_estimated_bytes=CAP)
        self.assertGreater(result.output_label_counts[5], 0)

    def test_lost_foreground_is_rejected(self):
        mask = np.zeros((4, 4, 4), dtype=np.uint8)
        mask[0, 0, 0] = 5
        with self.assertRaisesRegex(ValueError, "removed existing foreground label.*5"):
            resample_volume_pair(loaded_pair(shape=mask.shape, mask=mask), (4, 4, 4), max_estimated_bytes=CAP)

    def test_invalid_target_spacing(self):
        for spacing in ((0, 1, 1), (-1, 1, 1), (float("nan"), 1, 1), (float("inf"), 1, 1), (1, 1)):
            with self.subTest(spacing=spacing), self.assertRaisesRegex(ValueError, "target_spacing_mm"):
                make_target_grid((4, 5, 6), np.eye(4), spacing)

    def test_singular_and_sheared_affines_rejected(self):
        singular = np.diag([1., 0., 1., 1.])
        shear = np.eye(4)
        shear[0, 1] = .2
        for affine, message in ((singular, "singular"), (shear, "shear")):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                make_target_grid((4, 5, 6), affine, (1, 1, 1))

    def test_incompatible_mri_mask_geometry_rejected(self):
        shifted = np.eye(4)
        shifted[0, 3] = .01
        with self.assertRaisesRegex(ValueError, "source grids do not align"):
            resample_volume_pair(loaded_pair(mask_affine=shifted), (1, 1, 1), max_estimated_bytes=CAP)

    def test_144_slice_volume(self):
        result = resample_volume_pair(loaded_pair(shape=(2, 2, 144)), (1, 1, 1), max_estimated_bytes=CAP)
        self.assertEqual(result.original_shape, (2, 2, 144))
        self.assertEqual(result.mri.shape, (2, 2, 144))

    def test_memory_cap_rejects_before_transform(self):
        with patch("recognition.hipmri_3d_improved_unet_hard.resampling.affine_transform") as transform:
            with self.assertRaisesRegex(MemoryError, "estimated resampling allocation"):
                resample_volume_pair(loaded_pair(), (1, 1, 1), max_estimated_bytes=1)
            transform.assert_not_called()

    def test_counts_and_physical_volumes(self):
        affine = np.diag([1., 2., 3., 1.])
        mask = np.zeros((2, 2, 2), dtype=np.uint8)
        mask[0, 0, 0] = 5
        result = resample_volume_pair(loaded_pair(shape=mask.shape, mask=mask, affine=affine), (1, 2, 3), max_estimated_bytes=CAP)
        self.assertEqual(result.input_label_counts, (7, 0, 0, 0, 0, 1))
        self.assertEqual(result.output_label_counts, (7, 0, 0, 0, 0, 1))
        self.assertAlmostEqual(result.input_voxel_volume_mm3, 6)
        self.assertAlmostEqual(result.output_voxel_volume_mm3, 6)
        self.assertAlmostEqual(result.input_class_volumes_mm3[5], 6)
        self.assertAlmostEqual(result.output_class_volumes_mm3[0], 42)

    def test_spatial_unit_provenance_preserved(self):
        result = resample_volume_pair(loaded_pair(), (1, 1, 1), max_estimated_bytes=CAP)
        self.assertEqual(result.spatial_unit, "mm")
        self.assertEqual(result.segmentation_header_spatial_unit, "unknown")
        self.assertTrue(result.segmentation_unit_inferred)

    def test_nonfinite_mri_and_invalid_mask_rejected(self):
        loaded = loaded_pair()
        loaded.mri[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            resample_volume_pair(loaded, (1, 1, 1), max_estimated_bytes=CAP)
        mask = np.zeros((4, 5, 6), dtype=np.uint8)
        mask[1, 1, 1] = 6
        with self.assertRaisesRegex(ValueError, "outside 0–5"):
            resample_volume_pair(loaded_pair(mask=mask), (1, 1, 1), max_estimated_bytes=CAP)


if __name__ == "__main__":
    unittest.main()
