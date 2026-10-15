"""Synthetic V3 final-inference admission tests; no HipMRI or Test access."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from . import predict_v3 as p
from .improved_modules import ImprovedUNet3D


class V3PredictionTests(unittest.TestCase):
    def args(self, root):
        return ["--run-root", str(root), "--checkpoint", str(root / "B" /
                "training_epoch12_pending_validation.pt"), "--selection-receipt",
                str(root / "C" / "selection.json")]

    def test_preflight_and_test_selection_are_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertTrue(p.parse_args(self.args(root) + ["--preflight-only"]).preflight_only)
            base = self.args(root) + ["--all-split", "--output-dir", str(root / "out")]
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                p.parse_args(base + ["--split", "test"])
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                p.parse_args(base + ["--split", "test", "--allow-test"])
            self.assertTrue(p.parse_args(base + ["--split", "test", "--allow-test",
                                           "--freeze-receipt", str(root / "freeze.json")]).allow_test)
            for extra in (["--preflight-only", "--all-split"],
                          ["--preflight-only", "--split", "test", "--allow-test"],
                          ["--cases", "B040_Week0", "B040_Week0", "--output-dir", str(root / "out")]):
                with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    p.parse_args(self.args(root) + extra)

    def test_preflight_never_discovers_dataset_or_runs_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(p, "inspect_candidate", return_value=(ImprovedUNet3D(), {"lineage_audit": "PASS"})), \
                    patch.object(p, "discover_dataset", side_effect=AssertionError("dataset opened")), \
                    patch.object(p.predict, "process_pair", side_effect=AssertionError("inference run")), \
                    patch.object(p.evaluation, "_emit") as emit:
                self.assertEqual(p.main(self.args(root) + ["--preflight-only"]), 0)
                self.assertEqual(emit.call_args.args[0]["arrays_decoded"], 0)
                self.assertFalse(emit.call_args.args[0]["test_evaluated"])

    def test_test_requires_matching_formal_freeze_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "freeze.json"
            candidate = {"checkpoint_sha256": "a", "selection_sha256": "b", "config_sha256": "c"}
            receipt = {"candidate_checkpoint_sha256": "a", "selection_sha256": "b",
                       "config_sha256": "c", "adapter_sha256": p.split.v3.sha256(p.__file__),
                       "validation_equivalence": "PASS", "formal_freeze": True,
                       "test_evaluated_before_freeze": False}
            path.write_text(json.dumps(receipt), encoding="utf-8")
            self.assertEqual(p.inspect_freeze_receipt(path, candidate), receipt)
            receipt["validation_equivalence"] = "UNKNOWN"
            path.write_text(json.dumps(receipt), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "freeze receipt"):
                p.inspect_freeze_receipt(path, candidate)

    def test_full_validation_equivalence_requires_all_38_identical_cases(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "C").mkdir()
            journal = root / "C" / "events_synthetic.jsonl"
            journal.write_text("synthetic", encoding="utf-8")
            rows = [{"event": "volume", "status": "success", "patient_id": f"P{i:02}",
                     "week": 0, "confusion_counts": [[i]]} for i in range(38)]
            with patch.object(p.split.v3, "audit_journal", return_value=rows):
                self.assertTrue(p.compare_full_validation(root, rows))
                changed = [dict(row) for row in rows]
                changed[0]["confusion_counts"] = [[999]]
                self.assertFalse(p.compare_full_validation(root, changed))
                self.assertFalse(p.compare_full_validation(root, rows[:-1]))

    def test_candidate_requires_strict_state_and_epoch_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "B").mkdir()
            (root / "C").mkdir()
            checkpoint = root / "B" / "training_epoch12_pending_validation.pt"
            receipt_path = root / "C" / "selection.json"
            cfg = {"config_sha256": "synthetic-config", "protocol": {"parent_config": {}}}
            payload = {"schema_version": 1, "completed_epochs": 12, "total_steps": 1716,
                       "config": cfg, "epoch_result": {"epoch_completed": True,
                       "validation_pending": True, "processed_steps": 143, "expected_steps": 143},
                       "optimizer": {"state": {1: {"step": 1}}},
                       "rng": dict.fromkeys(("python", "numpy", "torch_cpu", "torch_cuda", "loader"), 1),
                       "model": ImprovedUNet3D().state_dict()}
            receipt = {"epoch": 12, "decision": {"selected": True},
                       "checkpoint": str(checkpoint), "checkpoint_sha256": "synthetic-checkpoint",
                       "config_sha256": cfg["config_sha256"], "validation_completed": True,
                       "test_evaluated": False}

            def write():
                torch.save(payload, checkpoint)
                receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
                return p.split.v3.sha256(checkpoint), p.split.v3.sha256(receipt_path)

            with patch.object(p.split, "audit_complete", return_value={"status": "PASS", "reports": {
                    "C": {"selection": receipt}}}), patch.object(p.split, "validate_config"):
                digest, receipt_digest = write()
                receipt["checkpoint_sha256"] = digest
                digest, receipt_digest = write()
                with patch.object(p, "CHECKPOINT_SHA256", digest), patch.object(p, "RECEIPT_SHA256", receipt_digest):
                    model, metadata = p.inspect_candidate(root, checkpoint, receipt_path)
                    self.assertFalse(model.training)
                    self.assertEqual(metadata["completed_epochs"], 12)
                    self.assertEqual(metadata["lineage_audit"], "PASS")
                payload["epoch_result"]["validation_pending"] = False
                digest, receipt_digest = write()
                receipt["checkpoint_sha256"] = digest
                digest, receipt_digest = write()
                with patch.object(p, "CHECKPOINT_SHA256", digest), patch.object(p, "RECEIPT_SHA256", receipt_digest):
                    with self.assertRaisesRegex(ValueError, "incompatible"):
                        p.inspect_candidate(root, checkpoint, receipt_path)


if __name__ == "__main__":
    unittest.main()
