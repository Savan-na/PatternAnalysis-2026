"""Focused synthetic tests for on-demand aligned HipMRI 3D patches."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import VolumePair
from .patch_dataset import HipMRIPatchDataset


def coordinate_arrays(shape_xyz=(20, 24, 32)) -> tuple[np.ndarray, np.ndarray]:
    x, y, z = np.indices(shape_xyz)
    mri = (10000 * x + 100 * y + z).astype(np.float32)
    mask = ((x + 2 * y + 3 * z) % 6).astype(np.uint8)
    return mri, mask


class PatchDatasetTests(unittest.TestCase):
    def setUp(self):
        self.train_pair = VolumePair("K019", 1, Path("train_mri.nii.gz"), Path("train_mask.nii.gz"))
        self.validation_pair = VolumePair("B040", 0, Path("val_mri.nii.gz"), Path("val_mask.nii.gz"))
        self.test_pair = VolumePair("B037", 0, Path("test_mri.nii.gz"), Path("test_mask.nii.gz"))
        self.discovered = {
            "train": (self.train_pair,),
            "validation": (self.validation_pair,),
            "test": (self.test_pair,),
        }
        self.mri_xyz, self.mask_xyz = coordinate_arrays()
        self.normalized = SimpleNamespace(mri=self.mri_xyz, segmentation=self.mask_xyz)
        discovery_patcher = patch(
            "recognition.hipmri_3d_improved_unet_hard.patch_dataset.discover_dataset",
            return_value=self.discovered,
        )
        load_patcher = patch(
            "recognition.hipmri_3d_improved_unet_hard.patch_dataset.load_volume_pair",
            return_value="loaded pair",
        )
        resample_patcher = patch(
            "recognition.hipmri_3d_improved_unet_hard.patch_dataset.resample_volume_pair",
            return_value="resampled pair",
        )
        normalize_patcher = patch(
            "recognition.hipmri_3d_improved_unet_hard.patch_dataset.normalize_volume_pair",
            return_value=self.normalized,
        )
        patchers = (discovery_patcher, load_patcher, resample_patcher, normalize_patcher)
        self.discovery, self.load, self.resample, self.normalize = (item.start() for item in patchers)
        for item in patchers:
            self.addCleanup(item.stop)

    def dataset(self, split="train", **kwargs):
        kwargs.setdefault("patch_shape_dhw", (16, 16, 16))
        return HipMRIPatchDataset(split, Path("synthetic_root"), **kwargs)

    def test_tensor_shapes_dtypes_and_mask_ids(self):
        sample = self.dataset()[0]
        self.assertEqual(sample["mri"].shape, (1, 16, 16, 16))
        self.assertEqual(sample["mask"].shape, (16, 16, 16))
        self.assertEqual(sample["mri"].dtype, torch.float32)
        self.assertEqual(sample["mask"].dtype, torch.long)
        self.assertTrue(set(torch.unique(sample["mask"]).tolist()).issubset(set(range(6))))

    def test_xyz_to_dhw_asymmetric_coordinate_mapping(self):
        sample = self.dataset("validation")[0]
        origin = tuple(sample["crop_origin_dhw"].tolist())
        self.assertEqual(origin, (8, 4, 2))
        for local in ((0, 0, 0), (3, 5, 7), (15, 15, 15)):
            d, h, w = (origin[axis] + local[axis] for axis in range(3))
            self.assertEqual(sample["mri"][(0, *local)].item(), self.mri_xyz[w, h, d])
            self.assertEqual(sample["mask"][local].item(), self.mask_xyz[w, h, d])

    def test_mri_mask_crops_use_identical_origin(self):
        sample = self.dataset(seed=19, train_foreground_probability=0)[0]
        d, h, w = sample["crop_origin_dhw"].tolist()
        expected_mri = self.mri_xyz[w:w + 16, h:h + 16, d:d + 16].transpose(2, 1, 0)
        expected_mask = self.mask_xyz[w:w + 16, h:h + 16, d:d + 16].transpose(2, 1, 0)
        np.testing.assert_array_equal(sample["mri"].numpy()[0], expected_mri)
        np.testing.assert_array_equal(sample["mask"].numpy(), expected_mask)

    def test_prostate_sampling_contains_class_five(self):
        self.mask_xyz.fill(0)
        self.mask_xyz[10, 12, 20] = 5
        sample = self.dataset(train_foreground_probability=1)[0]
        self.assertEqual(sample["sampling_mode"], "prostate")
        self.assertTrue(bool(torch.any(sample["mask"] == 5)))

    def test_absent_prostate_falls_back_safely(self):
        self.mask_xyz.fill(0)
        sample = self.dataset(train_foreground_probability=1)[0]
        self.assertEqual(sample["sampling_mode"], "random_fallback")
        self.assertEqual(sample["mask"].shape, (16, 16, 16))

    def test_fixed_seed_and_epoch_are_reproducible(self):
        first = self.dataset(seed=23, train_foreground_probability=0)
        second = self.dataset(seed=23, train_foreground_probability=0)
        first.set_epoch(4)
        second.set_epoch(4)
        a, b = first[0], second[0]
        torch.testing.assert_close(a["crop_origin_dhw"], b["crop_origin_dhw"])
        torch.testing.assert_close(a["mri"], b["mri"])
        torch.testing.assert_close(a["mask"], b["mask"])
        torch.testing.assert_close(a["crop_origin_dhw"], first[0]["crop_origin_dhw"])

    def test_validation_is_centered_and_seed_independent(self):
        first = self.dataset("validation", seed=1, train_foreground_probability=1)
        second = self.dataset("validation", seed=999, train_foreground_probability=0)
        first.set_epoch(7)
        a, b = first[0], second[0]
        self.assertEqual(a["sampling_mode"], "center")
        torch.testing.assert_close(a["crop_origin_dhw"], b["crop_origin_dhw"])
        torch.testing.assert_close(a["mri"], b["mri"])

    def test_foreground_boundary_crops(self):
        self.mask_xyz.fill(0)
        dataset = self.dataset(train_foreground_probability=1)
        self.mask_xyz[0, 0, 0] = 5
        at_start = dataset[0]
        self.assertEqual(tuple(at_start["crop_origin_dhw"].tolist()), (0, 0, 0))
        self.assertEqual(at_start["mask"][0, 0, 0].item(), 5)
        self.mask_xyz.fill(0)
        self.mask_xyz[-1, -1, -1] = 5
        at_end = dataset[0]
        self.assertEqual(tuple(at_end["crop_origin_dhw"].tolist()), (16, 8, 4))
        self.assertEqual(at_end["mask"][-1, -1, -1].item(), 5)

    def test_one_batch_dataloader_shapes(self):
        dataset = self.dataset(train_patches_per_volume=2, seed=7)
        loader = DataLoader(dataset, batch_size=2, num_workers=0, shuffle=False)
        batch = next(iter(loader))
        self.assertEqual(batch["mri"].shape, (2, 1, 16, 16, 16))
        self.assertEqual(batch["mask"].shape, (2, 16, 16, 16))
        self.assertEqual(batch["crop_origin_dhw"].shape, (2, 3))
        self.assertEqual(batch["patient_id"], ["K019", "K019"])

    def test_input_arrays_are_not_modified(self):
        before_mri, before_mask = self.mri_xyz.copy(), self.mask_xyz.copy()
        self.dataset(train_foreground_probability=1)[0]
        np.testing.assert_array_equal(self.mri_xyz, before_mri)
        np.testing.assert_array_equal(self.mask_xyz, before_mask)

    def test_production_pipeline_is_called_on_demand_without_cache(self):
        dataset = self.dataset()
        self.discovery.assert_called_once_with(Path("synthetic_root"))
        self.load.assert_not_called()
        self.resample.assert_not_called()
        self.normalize.assert_not_called()
        dataset[0]
        self.load.assert_called_once_with(self.train_pair)
        self.resample.assert_called_once_with(
            "loaded pair", dataset.target_spacing_mm,
            max_estimated_bytes=dataset.max_estimated_bytes,
        )
        self.normalize.assert_called_once_with("resampled pair")
        dataset[0]
        self.assertEqual(self.load.call_count, 2)
        self.assertFalse(any(isinstance(value, np.ndarray) for value in vars(dataset).values()))

    def test_frozen_split_and_no_test_exposure(self):
        train = self.dataset("train")
        validation = self.dataset("validation")
        self.assertEqual([pair.patient_id for pair in train.pairs], ["K019"])
        self.assertEqual([pair.patient_id for pair in validation.pairs], ["B040"])
        self.assertEqual(validation[0]["patient_id"], "B040")
        with self.assertRaisesRegex(ValueError, "test patients are not exposed"):
            self.dataset("test")
        self.discovery.return_value = {"train": (self.test_pair,), "validation": (), "test": ()}
        with self.assertRaisesRegex(ValueError, "outside the approved split"):
            self.dataset("train")

    def test_invalid_patch_and_sampling_settings_rejected(self):
        for shape in ((16, 16), (16, 16, 15), (0, 16, 16), (16.0, 16, 16)):
            with self.subTest(shape=shape), self.assertRaisesRegex(ValueError, "patch_shape_dhw"):
                self.dataset(patch_shape_dhw=shape)
        for probability in (-0.1, 1.1, float("nan")):
            with self.subTest(probability=probability), self.assertRaisesRegex(ValueError, "probability"):
                self.dataset(train_foreground_probability=probability)

    def test_volume_smaller_than_patch_rejected_without_padding(self):
        self.normalized.mri, self.normalized.segmentation = coordinate_arrays((8, 24, 32))
        with self.assertRaisesRegex(ValueError, "smaller than patch"):
            self.dataset()[0]

    def test_invalid_normalized_arrays_rejected(self):
        self.normalized.mri = self.mri_xyz.astype(np.float64)
        with self.assertRaisesRegex(ValueError, "invalid normalised"):
            self.dataset()[0]
        self.normalized.mri = self.mri_xyz
        self.normalized.segmentation = np.full(self.mask_xyz.shape, 6, dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "invalid normalised"):
            self.dataset()[0]

    def test_metadata_identifies_patient_week_and_crop(self):
        sample = self.dataset("validation")[0]
        self.assertEqual(sample["patient_id"], "B040")
        self.assertEqual(sample["week"], 0)
        self.assertEqual(sample["crop_origin_dhw"].dtype, torch.long)
        self.assertEqual(tuple(sample["crop_origin_dhw"].tolist()), (8, 4, 2))


if __name__ == "__main__":
    unittest.main()
