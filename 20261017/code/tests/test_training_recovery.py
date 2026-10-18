"""Check recovery picks a completed batch or epoch despite interrupted writes."""

import tempfile
import unittest
from pathlib import Path

from tools.train_g5_cohort import atomic_checkpoint, load_durable_state


class RecoveryTests(unittest.TestCase):
    def test_batch_and_completed_epoch_recovery(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is not installed")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            atomic_checkpoint(root / "latest.pt", {"epoch": 0, "next_batch": 1,
                                                    "weight": torch.tensor([1.0])})
            (root / "latest.pt.tmp").write_bytes(b"interrupted write")
            state, completed = load_durable_state(root, "cpu")
            self.assertEqual((state["epoch"], state["next_batch"]), (0, 1))
            self.assertIsNone(completed)

            atomic_checkpoint(root / "epoch_001.pt", {"epoch": 1, "next_batch": 0,
                                                       "weight": torch.tensor([2.0])})
            state, completed = load_durable_state(root, "cpu")
            self.assertEqual(float(state["weight"][0]), 2.0)
            self.assertEqual(completed["epoch"], 1)

            atomic_checkpoint(root / "latest.pt", {"epoch": 1, "next_batch": 7,
                                                    "weight": torch.tensor([3.0])})
            state, _ = load_durable_state(root, "cpu")
            self.assertEqual((state["epoch"], state["next_batch"]), (1, 7))


if __name__ == "__main__":
    unittest.main()
