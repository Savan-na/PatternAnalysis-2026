"""CPU integration checks with synthetic data and real checkpoint serialization."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from . import train_improved as t
from . import validate_improved as v
from . import validate as standard_validation
from .improved_modules import ImprovedUNet3D


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        torch.manual_seed(3710)
        self.model = ImprovedUNet3D(base_channels=1)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=.001, weight_decay=.00001)
        self.batch = {"mri": torch.randn(1, 1, 16, 16, 16),
                      "mask": torch.arange(16**3).reshape(1, 16, 16, 16).long() % 6}

    def test_epoch_update_and_diagnostic(self):
        before = self.model.head.weight.detach().clone()
        result = t.run_epoch(self.model, [self.batch], self.optimizer, torch.device("cpu"))
        self.assertTrue(result["epoch_completed"])
        self.assertEqual(result["processed_steps"], 1)
        self.assertFalse(torch.equal(before, self.model.head.weight))
        self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                            for p in self.model.parameters()))
        result = t.run_epoch(self.model, [self.batch, self.batch], self.optimizer,
                             torch.device("cpu"), diagnostic=True)
        self.assertFalse(result["epoch_completed"])
        self.assertEqual(result["processed_steps"], 1)

    def test_weighted_loss_matches_documented_helper(self):
        from .improved_modules import deep_supervision_loss
        outputs = self.model(self.batch["mri"], deep_supervision=True)
        expected = deep_supervision_loss(outputs, self.batch["mask"],
                                        lambda logits, target: t.baseline.combined_loss(logits, target)[0])
        torch.testing.assert_close(t.combined_loss(outputs, self.batch["mask"])[0], expected)

    def test_serialization_resume_and_config_rejection(self):
        result = t.run_epoch(self.model, [self.batch], self.optimizer, torch.device("cpu"))
        generator = torch.Generator().manual_seed(3710)
        config = {"steps_per_epoch": 1, "hyperparameters": t.HYPERPARAMETERS,
                  "source_sha256": t.source_hashes()}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "improved_unet_last.pt"
            t.atomic_checkpoint(path, self.model, self.optimizer, completed_epochs=1,
                                total_steps=1, config=config, generator=generator,
                                device=torch.device("cpu"), epoch_result=result)
            model = ImprovedUNet3D(base_channels=1)
            optimizer = torch.optim.AdamW(model.parameters())
            epoch, steps = t.resume_checkpoint(path, model, optimizer, config=config,
                                              generator=generator, device=torch.device("cpu"))
            self.assertEqual((epoch, steps), (1, 1))
            self.assertTrue(all(torch.equal(p, model.state_dict()[name])
                                for name, p in self.model.state_dict().items()))
            self.assertTrue(optimizer.state)
            changed = copy.deepcopy(config)
            changed["source_sha256"]["improved_modules.py"] = "changed"
            with self.assertRaisesRegex(ValueError, "incompatible"):
                t.resume_checkpoint(path, model, optimizer, config=changed,
                                    generator=generator, device=torch.device("cpu"))

    def test_sliding_window_and_metric_parity(self):
        mri = np.ones((17, 16, 18), np.float32)
        prediction, evidence = standard_validation.sliding_window_predict(
            self.model, mri, torch.device("cpu"), patch_shape=(16, 16, 16), stride=8)
        self.assertEqual(prediction.shape, mri.shape)
        self.assertGreaterEqual(evidence["coverage_min"], 1)
        counts = standard_validation.confusion_matrix(prediction, np.zeros_like(prediction))
        records = [{"patient_id": "B040", "confusion_counts": counts.tolist(),
                    "metrics": standard_validation.metrics_from_confusion(counts)}]
        self.assertIs(v.summarize, standard_validation.summarize)
        self.assertEqual(v.summarize(records), standard_validation.summarize(records))

    def test_production_model_load_and_mismatch_rejection(self):
        config = {"hyperparameters": copy.deepcopy(t.HYPERPARAMETERS), "steps_per_epoch": 143,
                  "policies": v._local_policies(), "source_sha256": t.source_hashes()}
        payload = {"schema_version": 1, "completed_epochs": 1, "total_steps": 143,
                   "epoch_result": {"epoch_completed": True, "processed_steps": 143},
                   "config": config, "model": ImprovedUNet3D().state_dict()}
        with patch.object(torch, "load", return_value=payload):
            model, metadata = v.load_checkpoint_model("synthetic.pt", torch.device("cpu"))
            self.assertIsInstance(model, ImprovedUNet3D)
            self.assertFalse(model.training)
            self.assertEqual(metadata["total_steps"], 143)
            config["source_sha256"]["improved_modules.py"] = "bad"
            with self.assertRaisesRegex(ValueError, "hash differs"):
                v.load_checkpoint_model("synthetic.pt", torch.device("cpu"))
            config["source_sha256"] = t.source_hashes()
            config["hyperparameters"]["model"] = "StandardUNet3D"
            with self.assertRaises(ValueError):
                v.load_checkpoint_model("synthetic.pt", torch.device("cpu"))

    def test_standard_path_and_existing_standard_payload_rejected(self):
        with patch.object(t, "checkpoint_path", return_value=Path("standard_unet_last.pt")):
            with self.assertRaises(ValueError):
                t.improved_checkpoint_path("ignored", "/dataset")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "improved_unet_last.pt"
            torch.save({"config": {"hyperparameters": {"model": "StandardUNet3D"}}}, path)
            with patch.object(t, "checkpoint_path", return_value=path):
                with self.assertRaisesRegex(ValueError, "non-Improved"):
                    t.improved_checkpoint_path(path, "/dataset")


if __name__ == "__main__":
    unittest.main()
