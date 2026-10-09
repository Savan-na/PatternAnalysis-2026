"""Synthetic CPU tests for one-pair nonzero MRI intensity normalisation."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import unittest

import numpy as np

from .dataset import VolumePair
from .intensity_normalization import (
    LOWER_PERCENTILE,
    UPPER_PERCENTILE,
    normalize_volume_pair,
)
from .resampling import ResampledVolumePair


def resampled_pair(mri=None, segmentation=None) -> ResampledVolumePair:
    """Build synthetic validated-stage metadata without invoking resampling."""

    if mri is None:
        mri = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1000, 0], dtype=np.float32).reshape(2, 2, 3)
    if segmentation is None:
        segmentation = np.arange(mri.size, dtype=np.uint8).reshape(mri.shape) % 6
    counts = tuple(int(np.count_nonzero(segmentation == label)) for label in range(6))
    affine = np.diag([1.5, 2.0, 2.5, 1.0])
    voxel_volume = 7.5
    return ResampledVolumePair(
        pair=VolumePair("K019", 1, Path("MRI.nii.gz"), Path("mask.nii.gz")),
        mri=mri,
        segmentation=segmentation,
        affine=affine,
        target_spacing_mm=(1.5, 2.0, 2.5),
        original_shape=mri.shape,
        input_labels=tuple(label for label, count in enumerate(counts) if count),
        output_labels=tuple(label for label, count in enumerate(counts) if count),
        input_label_counts=counts,
        output_label_counts=counts,
        input_voxel_volume_mm3=voxel_volume,
        output_voxel_volume_mm3=voxel_volume,
        input_class_volumes_mm3=tuple(count * voxel_volume for count in counts),
        output_class_volumes_mm3=tuple(count * voxel_volume for count in counts),
        spatial_unit="mm",
        segmentation_header_spatial_unit="unknown",
        segmentation_unit_inferred=True,
    )


class IntensityNormalizationTests(unittest.TestCase):
    def test_configured_percentiles(self):
        self.assertEqual((LOWER_PERCENTILE, UPPER_PERCENTILE), (0.5, 99.5))

    def test_percentile_clipping_and_zscore_values(self):
        source = resampled_pair()
        result = normalize_volume_pair(source)
        nonzero = source.mri[source.mri != 0].astype(np.float64)
        lower, upper = np.percentile(nonzero, (0.5, 99.5))
        clipped = np.clip(nonzero, lower, upper)
        mean = np.mean(clipped, dtype=np.float64)
        std = np.std(clipped, dtype=np.float64, ddof=0)
        expected = ((clipped - mean) / std).astype(np.float32)
        np.testing.assert_allclose(result.mri[source.mri != 0], expected, rtol=0, atol=1e-6)
        self.assertAlmostEqual(result.normalization.lower_intensity_bound, lower)
        self.assertAlmostEqual(result.normalization.upper_intensity_bound, upper)
        self.assertAlmostEqual(result.normalization.clipped_mean, mean)
        self.assertAlmostEqual(result.normalization.clipped_population_std, std)
        self.assertAlmostEqual(float(result.mri[1, 1, 1]), float(expected[-1]), places=6)

    def test_original_zeros_remain_zero_and_excluded(self):
        source = resampled_pair()
        result = normalize_volume_pair(source)
        self.assertTrue(np.all(result.mri[source.mri == 0] == 0))
        self.assertEqual(result.normalization.nonzero_voxel_count, 10)
        self.assertAlmostEqual(result.normalization.original_zero_fraction, 2 / 12)
        self.assertGreater(result.normalization.lower_intensity_bound, 0)

    def test_output_selected_mean_and_population_std(self):
        result = normalize_volume_pair(resampled_pair())
        self.assertAlmostEqual(result.normalization.output_selected_mean, 0, delta=1e-6)
        self.assertAlmostEqual(result.normalization.output_selected_population_std, 1, delta=1e-6)

    def test_segmentation_bitwise_unchanged(self):
        source = resampled_pair()
        before = source.segmentation.copy()
        result = normalize_volume_pair(source)
        np.testing.assert_array_equal(result.segmentation, before)
        np.testing.assert_array_equal(source.segmentation, before)
        self.assertIs(result.segmentation, source.segmentation)
        self.assertEqual(result.segmentation.dtype, np.uint8)
        self.assertEqual(set(np.unique(result.segmentation)), set(range(6)))

    def test_geometry_identity_and_label_metadata_preserved(self):
        source = resampled_pair()
        result = normalize_volume_pair(source)
        self.assertEqual(result.pair, source.pair)
        self.assertEqual(result.mri.shape, source.mri.shape)
        self.assertEqual(result.segmentation.shape, source.segmentation.shape)
        np.testing.assert_array_equal(result.affine, source.affine)
        self.assertEqual(result.target_spacing_mm, source.target_spacing_mm)
        self.assertEqual(result.original_shape, source.original_shape)
        for field in (
            "input_labels", "output_labels", "input_label_counts", "output_label_counts",
            "input_voxel_volume_mm3", "output_voxel_volume_mm3",
            "input_class_volumes_mm3", "output_class_volumes_mm3",
            "spatial_unit", "segmentation_header_spatial_unit", "segmentation_unit_inferred",
        ):
            self.assertEqual(getattr(result, field), getattr(source, field))

    def test_input_mri_is_unchanged_and_output_separate(self):
        source = resampled_pair()
        before = source.mri.copy()
        result = normalize_volume_pair(source)
        np.testing.assert_array_equal(source.mri, before)
        self.assertFalse(np.shares_memory(result.mri, source.mri))

    def test_deterministic(self):
        source = resampled_pair()
        first = normalize_volume_pair(source)
        second = normalize_volume_pair(source)
        np.testing.assert_array_equal(first.mri, second.mri)
        self.assertEqual(first.normalization, second.normalization)

    def test_all_zero_rejected(self):
        source = resampled_pair(mri=np.zeros((2, 2, 3), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "entirely zero"):
            normalize_volume_pair(source)

    def test_constant_nonzero_rejected(self):
        mri = np.zeros((2, 2, 3), dtype=np.float32)
        mri[0] = 7
        with self.assertRaisesRegex(ValueError, "standard deviation is zero or numerically degenerate"):
            normalize_volume_pair(resampled_pair(mri=mri))

    def test_numerically_degenerate_nonzero_rejected(self):
        mri = np.full((2, 2, 3), 1_000_000, dtype=np.float32)
        mri[0, 0, 0] = np.nextafter(np.float32(1_000_000), np.float32(np.inf))
        with self.assertRaisesRegex(ValueError, "numerically degenerate"):
            normalize_volume_pair(resampled_pair(mri=mri))

    def test_nonfinite_rejected(self):
        for invalid in (np.nan, np.inf, -np.inf):
            with self.subTest(invalid=invalid):
                source = resampled_pair()
                source.mri[0, 0, 1] = invalid
                with self.assertRaisesRegex(ValueError, "non-finite"):
                    normalize_volume_pair(source)

    def test_invalid_mri_shape_and_dtype_rejected(self):
        for mri in (np.ones((2, 3), dtype=np.float32), np.ones((2, 2, 3), dtype=np.float64)):
            with self.subTest(shape=mri.shape, dtype=mri.dtype):
                with self.assertRaisesRegex(ValueError, "three-dimensional float32"):
                    normalize_volume_pair(replace(resampled_pair(), mri=mri))

    def test_invalid_segmentation_shape_dtype_and_labels_rejected(self):
        source = resampled_pair()
        cases = (
            np.zeros((2, 3), dtype=np.uint8),
            np.zeros(source.mri.shape, dtype=np.float32),
            np.full(source.mri.shape, 6, dtype=np.uint8),
        )
        for segmentation in cases:
            with self.subTest(shape=segmentation.shape, dtype=segmentation.dtype):
                with self.assertRaisesRegex(ValueError, "segmentation"):
                    normalize_volume_pair(replace(source, segmentation=segmentation))

    def test_invalid_affine_rejected(self):
        source = resampled_pair()
        affine = source.affine.copy()
        affine[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "affine"):
            normalize_volume_pair(replace(source, affine=affine))

    def test_output_float32_finite(self):
        result = normalize_volume_pair(resampled_pair())
        self.assertEqual(result.mri.dtype, np.float32)
        self.assertTrue(np.isfinite(result.mri).all())

    def test_segmentation_cannot_change_statistics(self):
        source = resampled_pair()
        changed_mask = np.full(source.segmentation.shape, 5, dtype=np.uint8)
        other = replace(source, segmentation=changed_mask)
        first = normalize_volume_pair(source)
        second = normalize_volume_pair(other)
        np.testing.assert_array_equal(first.mri, second.mri)
        self.assertEqual(first.normalization, second.normalization)

    def test_small_nonzero_scale_uses_relative_degeneracy_check(self):
        values = np.linspace(1e-8, 2e-8, 12, dtype=np.float32).reshape(2, 2, 3)
        result = normalize_volume_pair(resampled_pair(mri=values))
        self.assertTrue(np.isfinite(result.mri).all())
        self.assertAlmostEqual(result.normalization.output_selected_population_std, 1, delta=1e-6)


if __name__ == "__main__":
    unittest.main()
