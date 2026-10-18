import json
import tempfile
import unittest
from pathlib import Path

from hipmri_rewrite.splits import discover, read_split


class PatientSplitTests(unittest.TestCase):
    def test_approved_patients_are_disjoint(self):
        split = read_split()
        groups = [set(split[key]) for key in ("train", "validation", "test")]
        self.assertEqual([len(group) for group in groups], [26, 6, 6])
        self.assertEqual(len(set.union(*groups)), 38)

    def test_duplicate_patient_is_rejected(self):
        split = read_split()
        split["test"][0] = split["train"][0]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "split.json"
            path.write_text(json.dumps(split), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "multiple splits"):
                read_split(path)

    def test_discovery_rejects_missing_week_mask(self):
        split = read_split()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "semantic_MRs").mkdir()
            (root / "semantic_labels_only").mkdir()
            (root / "semantic_MRs" / f"{split['train'][0]}_Week0_LFOV.nii.gz").touch()
            with self.assertRaisesRegex(ValueError, "case sets differ"):
                discover(root)


if __name__ == "__main__":
    unittest.main()
