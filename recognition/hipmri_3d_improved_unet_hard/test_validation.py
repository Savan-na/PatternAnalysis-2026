"""CPU synthetic full-volume inference and metric tests, without HipMRI access."""

import argparse
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from . import validate as v
from .dataset import APPROVED_PATIENTS, VolumePair


class _VoxelClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, image):
        assert not self.training and torch.is_inference_mode_enabled()
        self.calls += 1
        labels = image.round().long() % 6
        logits = torch.full((1, 6, *image.shape[2:]), -8., dtype=torch.float32)
        return logits.scatter_(1, labels, 8.)


class ValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def predict(self, mri, model=None, **kwargs):
        return v.sliding_window_predict(model or _VoxelClassifier(), mri, torch.device("cpu"),
                                        patch_shape=(16, 16, 16), stride=8, **kwargs)

    def test_starts_cover_regular_irregular_and_144_depth(self):
        for length, patch_size, stride in ((64, 64, 32), (144, 64, 32),
                                          (256, 64, 32), (286, 64, 32), (19, 16, 8)):
            starts = v.window_starts(length, patch_size, stride)
            coverage = np.zeros(length, dtype=np.int32)
            for start in starts:
                coverage[start:start + patch_size] += 1
            self.assertTrue(np.all(coverage > 0))
            self.assertEqual(starts[0], 0)
            self.assertEqual(starts[-1], length - patch_size)
            self.assertEqual(len(starts), len(set(starts)))
        self.assertEqual(v.window_starts(144, 64, 32), (0, 32, 64, 80))

    def test_invalid_grid_parameters(self):
        for args in ((0, 64, 32), (64, 0, 32), (64, 64, 0), (64, 64, 65), (64, 64, 1.5)):
            with self.assertRaises(ValueError):
                v.window_starts(*args)

    def test_xyz_dhw_full_volume_alignment_and_eval_mode(self):
        x, y, z = np.indices((17, 19, 21))
        truth = ((100 * x + 10 * y + z) % 6).astype(np.uint8)
        mri = truth.astype(np.float32)
        before = mri.copy()
        model = _VoxelClassifier().train()
        prediction, evidence = self.predict(mri, model)
        np.testing.assert_array_equal(prediction, truth)
        np.testing.assert_array_equal(mri, before)
        self.assertEqual(prediction.dtype, np.uint8)
        self.assertEqual(evidence["window_count"], 8)
        self.assertEqual(model.calls, 8)
        self.assertFalse(model.training)
        self.assertGreaterEqual(evidence["coverage_min"], 1)
        self.assertEqual(int(v.confusion_matrix(prediction, truth).trace()), truth.size)

    def test_small_volume_padding_is_removed_and_alignment_preserved(self):
        truth = (np.arange(8 * 9 * 10).reshape(8, 9, 10) % 6).astype(np.uint8)
        prediction, evidence = self.predict(truth.astype(np.float32))
        np.testing.assert_array_equal(prediction, truth)
        self.assertEqual(evidence["working_shape_dhw"], [16, 16, 16])
        self.assertEqual(evidence["window_count"], 1)

    def test_overlap_averages_probabilities_not_logits(self):
        class PatchDependent(nn.Module):
            def forward(self, image):
                logits = torch.full((1, 6, *image.shape[2:]), -20.)
                logits[:, 1:3] = 0
                if image[0, 0, 0, 0, 0] < 8:
                    logits[:, 1] = 5
                else:
                    logits[:, 2] = 2
                return logits
        mri = np.broadcast_to(np.arange(25, dtype=np.float32), (16, 16, 25)).copy()
        prediction, evidence = self.predict(mri, PatchDependent())
        self.assertTrue(np.all(prediction[:, :, :9] == 1))
        self.assertTrue(np.all(prediction[:, :, 9:] == 2))
        self.assertEqual(evidence["coverage_max"], 3)
        self.assertEqual(evidence["window_count"], 3)
        # At triple overlap, mean logits would incorrectly select class 1 (5 > 2+2).

    def test_memory_guard_precedes_model_execution(self):
        model = _VoxelClassifier()
        with self.assertRaises(MemoryError):
            self.predict(np.zeros((16, 16, 16), np.float32), model, max_accumulator_bytes=1)
        self.assertEqual(model.calls, 0)

    def test_invalid_mri_and_model_outputs(self):
        for mri in (np.zeros((16, 16), np.float32), np.zeros((16, 16, 16), np.float64),
                    np.full((16, 16, 16), np.nan, np.float32), np.zeros((0, 16, 16), np.float32)):
            with self.assertRaises(ValueError):
                self.predict(mri)
        class InvalidModel(nn.Module):
            def forward(self, image):
                return torch.full((1, 6, 16, 16, 16), float("nan"))
        with self.assertRaises(ValueError):
            self.predict(np.zeros((16, 16, 16), np.float32), InvalidModel())
        with self.assertRaises(ValueError):
            v.sliding_window_predict(_VoxelClassifier(), np.zeros((16, 16, 16), np.float32),
                                     torch.device("cpu"), patch_shape=(15, 16, 16))

    def test_dice_iou_and_confusion_arithmetic(self):
        truth = np.array([0, 1, 1, 5], np.uint8).reshape(2, 2, 1)
        prediction = np.array([0, 1, 5, 0], np.uint8).reshape(2, 2, 1)
        counts = v.confusion_matrix(prediction, truth)
        rows = v.metrics_from_confusion(counts)
        self.assertEqual(int(counts.sum()), 4)
        self.assertEqual(counts[1, 5], 1)
        self.assertAlmostEqual(rows[1]["dice"], 2 / 3)
        self.assertEqual(rows[1]["iou"], .5)
        self.assertEqual(rows[5]["dice"], 0)
        self.assertIsNone(rows[2]["dice"])

    def test_absent_classes_excluded_and_false_positives_zero(self):
        truth = np.zeros((2, 2, 2), np.uint8)
        perfect = v.metrics_from_confusion(v.confusion_matrix(truth, truth))
        wrong = v.metrics_from_confusion(v.confusion_matrix(np.full_like(truth, 5), truth))
        self.assertIsNone(perfect[5]["dice"])
        self.assertEqual(wrong[5]["dice"], 0)
        self.assertEqual(wrong[5]["iou"], 0)
        macro = v.macro_metrics([perfect, wrong])
        self.assertEqual(macro[5]["evaluable_units"], 1)
        self.assertEqual(macro[5]["dice"], 0)

    def test_patient_pooling_and_distinct_overall_aggregations(self):
        truth = np.full((2, 2, 2), 5, np.uint8)
        records = []
        for patient, prediction in (("A", truth), ("A", np.zeros_like(truth)), ("B", truth)):
            counts = v.confusion_matrix(prediction, truth)
            records.append({"patient_id": patient, "confusion_counts": counts.tolist(),
                            "metrics": v.metrics_from_confusion(counts)})
        result = v.summarize(records)
        self.assertAlmostEqual(result["patients"][0]["metrics"][5]["dice"], 2 / 3)
        self.assertAlmostEqual(result["overall_patient_macro"][5]["dice"], 5 / 6)
        self.assertAlmostEqual(result["overall_volume_macro"][5]["dice"], 2 / 3)
        self.assertAlmostEqual(result["overall_pooled"][5]["dice"], .8)

    def test_invalid_metric_inputs(self):
        good = np.zeros((2, 2, 2), np.uint8)
        for prediction in (good[:, :, :1], good.astype(np.float32), np.full_like(good, 6)):
            with self.assertRaises(ValueError):
                v.confusion_matrix(prediction, good)
        with self.assertRaises(ValueError):
            v.metrics_from_confusion(np.full((6, 6), -1, np.int64))

    def test_validation_selection_rejects_training_or_test_cases(self):
        pairs = [VolumePair(patient, 0, Path("x"), Path("y")) for patient in APPROVED_PATIENTS["validation"]]
        pairs += [VolumePair("B040", week, Path("x"), Path("y")) for week in range(1, 33)]
        splits = {"validation": tuple(pairs)}
        self.assertEqual(len(v.select_validation(splits)), 38)
        self.assertEqual(len(v.select_validation(splits, ("B040", 0))), 1)
        for case in (("K019", 1), ("B037", 0), ("B040", 100)):
            with self.assertRaises(ValueError):
                v.select_validation(splits, case)
        with self.assertRaises(ValueError):
            v.select_validation({"validation": pairs[:-1]})

    def test_checkpoint_requires_complete_epoch_and_safe_load(self):
        with patch.object(v.torch, "load", return_value={}) as load:
            with self.assertRaises(ValueError):
                v.load_checkpoint_model("synthetic.pt", torch.device("cpu"))
            load.assert_called_once_with(Path("synthetic.pt"), map_location="cpu", weights_only=True)

    def test_actual_standard_unet_inference_creates_no_gradients(self):
        from .modules import StandardUNet3D
        model = StandardUNet3D(base_channels=1)
        prediction, evidence = self.predict(np.ones((16, 16, 16), np.float32), model)
        self.assertEqual(prediction.shape, (16, 16, 16))
        self.assertTrue(set(np.unique(prediction)).issubset(range(6)))
        self.assertEqual(evidence["window_count"], 1)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertFalse(model.training)

    def test_valid_baseline_checkpoint_load_and_policy_rejection(self):
        from .modules import StandardUNet3D
        directory = Path(v.__file__).parent
        config = {"hyperparameters": {"model": "StandardUNet3D", "in_channels": 1,
                                      "out_channels": 6, "base_channels": 16},
                  "steps_per_epoch": 143, "policies": v._local_policies(),
                  "source_sha256": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                                    for name in ("modules.py", "dataset.py", "volume_io.py",
                                                 "resampling.py", "intensity_normalization.py")}}
        payload = {"schema_version": 1, "completed_epochs": 1, "total_steps": 143,
                   "epoch_result": {"epoch_completed": True, "processed_steps": 143},
                   "config": config, "model": StandardUNet3D().state_dict()}
        with patch.object(v.torch, "load", return_value=payload):
            model, metadata = v.load_checkpoint_model("synthetic.pt", torch.device("cpu"))
            self.assertFalse(model.training)
            self.assertEqual(metadata["completed_epochs"], 1)
            self.assertTrue(torch.equal(model.head.weight, payload["model"]["head.weight"]))
            config["policies"]["resampling"]["target_spacing_mm"][0] = 99
            with self.assertRaisesRegex(ValueError, "policies differ"):
                v.load_checkpoint_model("synthetic.pt", torch.device("cpu"))

    def test_case_parser(self):
        self.assertEqual(v._case("B040_Week0"), ("B040", 0))
        with self.assertRaises(argparse.ArgumentTypeError):
            v._case("bad")

    def test_process_pair_preserves_source_and_inference_shape_metadata(self):
        pair = VolumePair("B040", 0, Path("mri"), Path("mask"))
        loaded = SimpleNamespace(mri=np.zeros((18, 20, 22), np.float32))
        truth = np.full((16, 17, 18), 5, np.uint8)
        normalized = SimpleNamespace(
            mri=truth.astype(np.float32), segmentation=truth, affine=np.eye(4),
            target_spacing_mm=(1.5, 1.5, 1.5),
            segmentation_header_spatial_unit="unknown", segmentation_unit_inferred=True)
        # Mock preprocessing and GPU counters; exercise the actual reporting path on CPU.
        prefix = v.__package__
        modules = {
            prefix + ".volume_io": SimpleNamespace(load_volume_pair=lambda pair: loaded),
            prefix + ".resampling": SimpleNamespace(
                resample_volume_pair=lambda loaded, spacing, **kwargs: normalized),
            prefix + ".intensity_normalization": SimpleNamespace(
                normalize_volume_pair=lambda resampled: resampled),
        }
        evidence = {"original_shape_xyz": list(truth.shape), "window_count": 1,
                    "coverage_min": 1, "inference_seconds": 0.01}
        with patch.dict(sys.modules, modules), \
                patch.object(v, "sliding_window_predict", return_value=(truth.copy(), evidence)), \
                patch.object(torch.cuda, "reset_peak_memory_stats"), \
                patch.object(torch.cuda, "max_memory_allocated", return_value=123), \
                patch.object(torch.cuda, "max_memory_reserved", return_value=456):
            record = v._process_pair(pair, None, torch.device("cpu"), (1.5,) * 3, 512, 32)
        self.assertEqual(record["status"], "success", record)
        self.assertEqual(record["original_shape_xyz"], [18, 20, 22])
        self.assertEqual(record["resampled_shape_xyz"], [16, 17, 18])
        self.assertEqual(record["inference_input_shape_xyz"], [16, 17, 18])
        self.assertEqual(record["window_count"], 1)
        self.assertEqual(record["metrics"][5]["dice"], 1)
        self.assertEqual(record["mask_header_unit"], "unknown")
        self.assertTrue(record["mask_unit_inferred"])
        np.testing.assert_array_equal(record["affine"], normalized.affine)
        self.assertEqual(evidence["original_shape_xyz"], [16, 17, 18])


if __name__ == "__main__":
    unittest.main()
