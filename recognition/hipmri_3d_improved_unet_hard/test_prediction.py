"""CPU prediction-driver tests: synthetic arrays/checkpoints, no HipMRI access."""

import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from torch import nn

from . import predict as p
from . import validate as v
from .dataset import APPROVED_PATIENTS, VolumePair


class _VoxelClassifier(nn.Module):
    """Known voxel labels expose any XYZ/DHW or evaluation-mode mistake."""

    def forward(self, image):
        assert not self.training and torch.is_inference_mode_enabled()
        labels = image.round().long() % 6
        logits = torch.full((1, 6, *image.shape[2:]), -8., dtype=torch.float32)
        return logits.scatter_(1, labels, 8.)


def _splits():
    result = {}
    for split, extra in (("validation", 32), ("test", 24)):
        patients = APPROVED_PATIENTS[split]
        keys = [(patient, 0) for patient in patients] + [(patients[0], week) for week in range(1, extra + 1)]
        result[split] = tuple(VolumePair(patient, week, Path(f"{patient}_Week{week}_LFOV.nii.gz"),
                                       Path(f"{patient}_Week{week}_SEMANTIC.nii.gz")) for patient, week in keys)
    return result


def _normalized(pair=None):
    x, y, z = np.indices((17, 19, 21))
    truth = ((100 * x + 10 * y + z) % 6).astype(np.uint8)
    return SimpleNamespace(pair=pair, mri=truth.astype(np.float32), segmentation=truth,
                           affine=np.diag([-1.5, -2., 1.5, 1.]), target_spacing_mm=(1.5, 2., 1.5),
                           segmentation_header_spatial_unit="unknown", segmentation_unit_inferred=True)


def _payload(model_type):
    from .modules import StandardUNet3D
    from .improved_modules import ImprovedUNet3D
    from .train import HYPERPARAMETERS
    from .train_improved import HYPERPARAMETERS as IMPROVED_HP, source_hashes
    if model_type == "standard":
        model = StandardUNet3D()
        hp = dict(HYPERPARAMETERS)
        directory = Path(v.__file__).parent
        hashes = {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                  for name in ("modules.py", "dataset.py", "volume_io.py", "resampling.py", "intensity_normalization.py")}
    else:
        model, hp, hashes = ImprovedUNet3D(), dict(IMPROVED_HP), source_hashes()
    return {"schema_version": 1, "completed_epochs": 1, "total_steps": 143,
            "epoch_result": {"epoch_completed": True, "processed_steps": 143},
            "config": {"hyperparameters": hp, "steps_per_epoch": 143,
                       "policies": v._local_policies(), "source_sha256": hashes}, "model": model.state_dict()}


class PredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def arguments(self, directory):
        return ["--model", "standard", "--checkpoint", str(directory / "synthetic.pt"),
                "--cases", "B040_Week0", "--output-dir", str(directory / "outputs"),
                "--dataset-root", str(directory / "source"), "--device", "cpu"]

    def test_cli_defaults_are_validation_safe(self):
        args = p.parse_args(self.arguments(Path("/tmp")))
        self.assertEqual(args.split, "validation")
        self.assertFalse(args.allow_test)
        self.assertEqual(args.cases, [("B040", 0)])
        self.assertEqual(args.stride, 32)
        args = p.parse_args(["--model", "improved", "--checkpoint", "x", "--case", "B040_Week0",
                             "--output-dir", "y"])
        self.assertEqual(args.model, "improved")
        self.assertEqual(args.device, "cuda")

    def test_cli_rejects_invalid_or_ambiguous_inputs(self):
        base = self.arguments(Path("/tmp"))
        variants = [[], base + ["--stride", "0"], base + ["--stride", "65"],
                    base + ["--model", "invalid"], base + ["--split", "train"],
                    base + ["--all-split"], base + ["--allow-test"],
                    base + ["--cases", "bad"], base + ["--cases", "B040_Week0", "B040_Week0"]]
        for argv in variants:
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                p.parse_args(argv)

    def test_cli_test_requires_both_explicit_selections(self):
        base = self.arguments(Path("/tmp"))
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            p.parse_args(base + ["--split", "test"])
        args = p.parse_args(base + ["--split", "test", "--allow-test", "--cases", "B037_Week0"])
        self.assertTrue(args.allow_test)
        self.assertEqual(args.split, "test")

    def test_select_pairs_uses_frozen_validation_and_requested_order(self):
        splits = _splits()
        self.assertEqual(len(p.select_pairs(splits)), 38)
        pairs = p.select_pairs(splits, cases=[("B040", 1), ("J026", 0)])
        self.assertEqual([(q.patient_id, q.week) for q in pairs], [("B040", 1), ("J026", 0)])

    def test_select_pairs_blocks_cross_split_or_implicit_test_access(self):
        splits = _splits()
        for kwargs in ({"split": "test"}, {"split": "train"}, {"allow_test": True},
                       {"cases": [("K019", 1)]}, {"cases": [("B037", 0)]},
                       {"cases": []}, {"cases": [("B040", 0), ("B040", 0)]}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                p.select_pairs(splits, **kwargs)
        self.assertEqual(len(p.select_pairs(splits, "test", allow_test=True)), 30)
        pair = p.select_pairs(splits, "test", [("B037", 0)], allow_test=True)[0]
        self.assertEqual(pair.patient_id, "B037")

    def test_select_pairs_checks_coverage_and_unexpected_patients(self):
        splits = _splits()
        for bad in (splits["validation"][:-1], (splits["test"][0], *splits["validation"][1:]),
                    (*splits["validation"][:-1], splits["validation"][0])):
            with self.assertRaisesRegex(ValueError, "coverage"):
                p.select_pairs({"validation": bad})

    def test_model_selection_delegates_to_existing_loaders(self):
        from . import validate_improved as vi
        device, result = torch.device("cpu"), (object(), {})
        with patch.object(v, "load_checkpoint_model", return_value=result) as standard, \
                patch.object(vi, "load_checkpoint_model", return_value=result) as improved:
            self.assertIs(p.load_model("standard", "x.pt", device), result)
            self.assertIs(p.load_model("improved", "y.pt", device), result)
            standard.assert_called_once_with("x.pt", device)
            improved.assert_called_once_with("y.pt", device)
        with self.assertRaises(ValueError):
            p.load_model("invalid", "x.pt", device)

    def test_actual_protected_serialized_checkpoints_load_both_models(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "fixture.pt"
            for kind in ("standard", "improved"):
                payload = _payload(kind)
                torch.save(payload, checkpoint)
                before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                model, metadata = p.load_model(kind, checkpoint, torch.device("cpu"))
                self.assertEqual(type(model).__name__, p.MODEL_NAMES[kind])
                self.assertFalse(model.training)
                self.assertEqual(metadata["completed_epochs"], 1)
                self.assertTrue(torch.equal(model.head.weight, payload["model"]["head.weight"]))
                self.assertEqual(hashlib.sha256(checkpoint.read_bytes()).hexdigest(), before)

    def test_protected_loaders_reject_model_policy_hash_and_state_mismatches(self):
        for kind in ("standard", "improved"):
            payload = _payload(kind)
            with patch.object(torch, "load", return_value=payload) as loading:
                payload["config"]["hyperparameters"]["model"] = "WrongModel"
                with self.assertRaises(ValueError):
                    p.load_model(kind, Path("x.pt"), torch.device("cpu"))
                self.assertTrue(loading.call_args.kwargs["weights_only"])
                payload["config"]["hyperparameters"]["model"] = p.MODEL_NAMES[kind]
                payload["config"]["policies"]["normalization"]["zscore_ddof"] = 1
                with self.assertRaisesRegex(ValueError, "policies"):
                    p.load_model(kind, Path("x.pt"), torch.device("cpu"))
                payload["config"]["policies"] = v._local_policies()
                original = payload["config"]["source_sha256"]["modules.py"]
                payload["config"]["source_sha256"]["modules.py"] = "invalid"
                with self.assertRaisesRegex(ValueError, "hash"):
                    p.load_model(kind, Path("x.pt"), torch.device("cpu"))
                payload["config"]["source_sha256"]["modules.py"] = original
                del payload["model"]["head.weight"]
                with self.assertRaises(RuntimeError):
                    p.load_model(kind, Path("x.pt"), torch.device("cpu"))

    def test_prepare_volume_calls_only_approved_pipeline(self):
        pair = _splits()["validation"][0]
        loaded = SimpleNamespace(mri=np.zeros((18, 20, 22), np.float32))
        resampled, normalized = object(), _normalized(pair)
        loading, resampling, normalizing = Mock(return_value=loaded), Mock(return_value=resampled), Mock(return_value=normalized)
        prefix = p.__package__
        with patch.dict(sys.modules, {
                prefix + ".volume_io": SimpleNamespace(load_volume_pair=loading),
                prefix + ".resampling": SimpleNamespace(resample_volume_pair=resampling),
                prefix + ".intensity_normalization": SimpleNamespace(normalize_volume_pair=normalizing)}):
            result, shape = p.prepare_volume(pair, v._local_policies())
        self.assertIs(result, normalized)
        self.assertEqual(shape, [18, 20, 22])
        loading.assert_called_once_with(pair)
        settings = v._local_policies()["resampling"]
        resampling.assert_called_once_with(loaded, settings["target_spacing_mm"],
                                          max_estimated_bytes=settings["max_estimated_bytes"])
        normalizing.assert_called_once_with(resampled)

    def test_aligned_panels_preserve_asymmetric_xyz_and_error_semantics(self):
        x, y, z = np.indices((4, 3, 5))
        mri = (100 * x + 10 * y + z).astype(np.float32)
        truth = np.zeros(mri.shape, np.uint8)
        truth[1, 2, 3], truth[3, 0, 3] = 5, 1
        predicted = truth.copy()
        predicted[1, 2, 3], predicted[0, 1, 3], predicted[3, 0, 3] = 0, 5, 2
        before = [a.copy() for a in (mri, truth, predicted)]
        image, reference, prediction, errors = p.aligned_slice_panels(mri, truth, predicted, 3)
        self.assertEqual(image.shape, (3, 4))
        self.assertEqual(image[2, 1], 123)
        self.assertEqual(reference[2, 1], 5)
        self.assertEqual(prediction[2, 1], 0)
        self.assertEqual(errors[2, 1], 2)
        self.assertEqual(errors[1, 0], 1)
        self.assertEqual(errors[0, 3], 3)
        for old, current in zip(before, (mri, truth, predicted)):
            np.testing.assert_array_equal(old, current)

    def test_visualization_writes_png_and_records_exact_slice_indices(self):
        volume = _normalized()
        volume.segmentation.fill(0)
        volume.segmentation[2:4, 3:5, 3:8] = 5
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "panels.png"
            record = p.save_visualization(path, volume.mri, volume.segmentation,
                                          volume.segmentation.copy(), volume.affine, "Synthetic")
            self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            self.assertEqual(record["z_indices"], [3, 5, 7])
            from matplotlib.image import imread
            image = imread(path)
            self.assertGreaterEqual(image.shape[0], 900)
            self.assertGreaterEqual(image.shape[1], 1200)
            self.assertEqual(len(p.CLASS_COLORS), 6)

    def test_no_prostate_visualization_uses_geometric_slices(self):
        volume = _normalized()
        volume.segmentation.fill(0)
        with tempfile.TemporaryDirectory() as directory:
            result = p.save_visualization(Path(directory) / "panels.png", volume.mri,
                                         volume.segmentation, volume.segmentation, volume.affine, "Synthetic")
        self.assertEqual(result["z_indices"], [5, 10, 15])
        self.assertIn("no ground-truth prostate", result["slice_selection"])

    def test_panel_invalid_shapes_dtypes_labels_and_geometry_fail(self):
        volume = _normalized()
        for truth, prediction, z in ((volume.segmentation[:, :, :1], volume.segmentation, 1),
                                     (volume.segmentation, volume.segmentation.astype(np.float32), 1),
                                     (volume.segmentation, np.full_like(volume.segmentation, 6), 1),
                                     (volume.segmentation, volume.segmentation, 100)):
            with self.assertRaises(ValueError):
                p.aligned_slice_panels(volume.mri, truth, prediction, z)
        with self.assertRaises(ValueError):
            p.save_visualization(Path("unused.png"), volume.mri, volume.segmentation,
                                 volume.segmentation, np.zeros((4, 4)), "Synthetic")

    def test_cpu_process_uses_real_sliding_windows_and_identical_metrics(self):
        pair, model = _splits()["validation"][0], _VoxelClassifier().train()
        volume = _normalized(pair)
        before = volume.mri.copy(), volume.segmentation.copy()
        with tempfile.TemporaryDirectory() as directory, patch.object(p, "prepare_volume", return_value=(volume, [18, 20, 22])), \
                patch.object(torch.cuda, "reset_peak_memory_stats") as gpu_reset:
            record = p.process_pair(pair, model, torch.device("cpu"), v._local_policies(), 32, Path(directory), "validation")
        expected = v.metrics_from_confusion(v.confusion_matrix(volume.segmentation, volume.segmentation))
        self.assertEqual(record["metrics"], expected)
        self.assertEqual(record["prediction_shape_xyz"], [17, 19, 21])
        self.assertEqual(record["inference_input_shape_xyz"], [17, 19, 21])
        self.assertEqual(record["original_shape_xyz"], [18, 20, 22])
        self.assertEqual(record["predicted_classes"], list(range(6)))
        self.assertEqual(record["prediction_dtype"], "uint8")
        self.assertIsNone(record["peak_gpu_allocated_bytes"])
        self.assertIsNone(record["peak_gpu_reserved_bytes"])
        self.assertGreater(record["inference_seconds"], 0)
        self.assertGreaterEqual(record["coverage_min"], 1)
        self.assertFalse(model.training)
        gpu_reset.assert_not_called()
        np.testing.assert_array_equal(volume.mri, before[0])
        np.testing.assert_array_equal(volume.segmentation, before[1])

    def test_process_rejects_invalid_prediction_shape_or_dtype(self):
        pair, volume = _splits()["validation"][0], _normalized()
        with tempfile.TemporaryDirectory() as directory, patch.object(p, "prepare_volume", return_value=(volume, list(volume.mri.shape))):
            for invalid in (np.zeros((16, 16, 16), np.uint8), volume.segmentation.astype(np.float32)):
                with patch.object(v, "sliding_window_predict", return_value=(invalid, {})), self.assertRaises(ValueError):
                    p.process_pair(pair, _VoxelClassifier(), torch.device("cpu"), {}, 32, Path(directory), "validation")

    def test_output_path_rejects_dataset_git_and_existing_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            existing = root / "existing"
            existing.mkdir()
            git = root / "other_repo"
            git.mkdir()
            (git / ".git").mkdir()
            for path in (source / "outputs", existing, root / "missing_parent" / "outputs", git / "outputs",
                         Path(p.__file__).parent / "outputs"):
                with self.assertRaises(ValueError):
                    p.output_path(path, source)
            self.assertEqual(p.output_path(root / "new", source), (root / "new").resolve())

    def test_main_synthetic_validation_writes_results_and_panels_without_test_access(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "synthetic.pt"
            checkpoint.write_bytes(b"mock checkpoint")
            selected = _splits()["validation"][0]
            with patch.object(p, "discover_dataset", return_value=_splits()), \
                    patch.object(p, "load_model", return_value=(_VoxelClassifier(), {"completed_epochs": 1})), \
                    patch.object(p, "prepare_volume", return_value=(_normalized(selected), [17, 19, 21])) as preprocess, \
                    contextlib.redirect_stdout(io.StringIO()) as terminal:
                code = p.main(self.arguments(root))
            self.assertEqual(code, 0)
            preprocess.assert_called_once_with(selected, v._local_policies())
            results = json.loads((root / "outputs" / "results.json").read_text())
            self.assertEqual(results["setup"]["split"], "validation")
            self.assertFalse(results["setup"]["test_opt_in"])
            self.assertEqual(results["summary"]["status"], "PASS")
            self.assertFalse(results["summary"]["full_split_completed"])
            self.assertEqual(results["summary"]["successful"], 1)
            self.assertEqual(results["summary"]["overall_pooled"][5]["dice"], 1)
            self.assertTrue((root / "outputs" / "B040_Week0_panels.png").is_file())
            self.assertEqual(checkpoint.read_bytes(), b"mock checkpoint")
            messages = [json.loads(line) for line in terminal.getvalue().splitlines()]
            self.assertEqual(messages[-1]["status"], "PASS")

    def test_main_preserves_explicit_test_split_metadata_without_real_test_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "synthetic.pt").write_bytes(b"mock")
            counts = np.zeros((6, 6), np.int64)
            fake = {"status": "success", "patient_id": "B037", "confusion_counts": counts.tolist(),
                    "metrics": v.metrics_from_confusion(counts)}
            with patch.object(p, "discover_dataset", return_value=_splits()), \
                    patch.object(p, "load_model", return_value=(_VoxelClassifier(), {})), \
                    patch.object(p, "process_pair", return_value=fake) as processing, contextlib.redirect_stdout(io.StringIO()):
                code = p.main(self.arguments(root) + ["--split", "test", "--allow-test", "--cases", "B037_Week0"])
            self.assertEqual(code, 0)
            self.assertEqual(processing.call_args.args[-1], "test")
            self.assertEqual(processing.call_args.args[0].patient_id, "B037")
            results = json.loads((root / "outputs" / "results.json").read_text())
            self.assertEqual(results["setup"]["split"], "test")
            self.assertTrue(results["setup"]["test_opt_in"])

    def test_main_records_failure_exits_nonzero_and_never_prints_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "synthetic.pt").write_bytes(b"mock")
            with patch.object(p, "discover_dataset", return_value=_splits()), \
                    patch.object(p, "load_model", return_value=(_VoxelClassifier(), {})), \
                    patch.object(p, "prepare_volume", side_effect=ValueError("synthetic geometry failure")), \
                    contextlib.redirect_stdout(io.StringIO()) as terminal:
                code = p.main(self.arguments(root))
            self.assertEqual(code, 1)
            results = json.loads((root / "outputs" / "results.json").read_text())
            self.assertEqual(results["summary"]["failed"], 1)
            self.assertEqual(results["volumes"][0]["exception_message"], "synthetic geometry failure")
            self.assertNotIn('"status": "PASS"', terminal.getvalue())

    def test_missing_or_incompatible_checkpoint_fails_before_output_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FileNotFoundError):
                p.main(self.arguments(root))
            (root / "synthetic.pt").write_bytes(b"mock")
            with patch.object(p, "discover_dataset", return_value=_splits()), \
                    patch.object(p, "load_model", side_effect=ValueError("source hash differs")), \
                    self.assertRaisesRegex(ValueError, "source hash"):
                p.main(self.arguments(root))
            self.assertFalse((root / "outputs").exists())

    def test_cuda_without_allocation_fails_before_discovery(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "synthetic.pt").write_bytes(b"mock")
            with patch.dict(p.os.environ, {}, clear=True), patch.object(p, "discover_dataset") as discovery, \
                    self.assertRaisesRegex(RuntimeError, "allocated Slurm"):
                p.main(self.arguments(root) + ["--device", "cuda"])
            discovery.assert_not_called()


if __name__ == "__main__":
    unittest.main()
