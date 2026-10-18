import hashlib
import tempfile
import unittest
from pathlib import Path

from tools.audit_train_raw import sha256_file, train_inventory


class RawAuditInventoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "semantic_MRs").mkdir()
        (self.root / "semantic_labels_only").mkdir()

    def test_pairs_train_files_by_case_identity(self):
        mri = self.root / "semantic_MRs" / "A001_Week0_LFOV.nii.gz"
        mask = self.root / "semantic_labels_only" / "A001_Week0_SEMANTIC.nii.gz"
        mri.touch()
        mask.touch()

        pairs, errors = train_inventory(self.root, {"A001"})

        self.assertEqual(pairs, [("A001_Week0", mri, mask)])
        self.assertEqual(errors, [])

    def test_reports_missing_train_pair(self):
        (self.root / "semantic_MRs" / "A001_Week0_LFOV.nii.gz").touch()

        pairs, errors = train_inventory(self.root, {"A001"})

        self.assertEqual(pairs, [])
        self.assertEqual(errors, ["Missing paired Train file: A001_Week0"])

    def test_hashes_file_bytes(self):
        source = self.root / "source.bin"
        source.write_bytes(b"immutable source bytes")

        self.assertEqual(
            sha256_file(source),
            hashlib.sha256(b"immutable source bytes").hexdigest(),
        )


if __name__ == "__main__":
    unittest.main()
