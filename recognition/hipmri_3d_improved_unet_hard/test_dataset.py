"""Focused filename, pairing, and approved-split tests using empty files."""

import tempfile
import unittest
from pathlib import Path

from recognition.hipmri_3d_improved_unet_hard import dataset


class HipMRIDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.mri_dir = self.root / "semantic_MRs"
        self.mask_dir = self.root / "semantic_labels_only"
        self.mri_dir.mkdir()
        self.mask_dir.mkdir()

    def add_pair(self, patient_id: str, week: int) -> None:
        (self.mri_dir / f"{patient_id}_Week{week}_LFOV.nii.gz").touch()
        (self.mask_dir / f"{patient_id}_Week{week}_SEMANTIC.nii.gz").touch()

    def add_complete_synthetic_dataset(self) -> None:
        for split, patient_ids in dataset.APPROVED_PATIENTS.items():
            weeks = {patient_id: 0 for patient_id in patient_ids}
            for index in range(dataset.EXPECTED_VOLUME_COUNTS[split]):
                patient_id = patient_ids[index % len(patient_ids)]
                weeks[patient_id] += 1
                self.add_pair(patient_id, weeks[patient_id])

    def test_filename_parses_patient_and_week(self) -> None:
        self.assertEqual(
            dataset.parse_volume_filename("K019_Week12_LFOV.nii.gz", "LFOV"),
            ("K019", 12),
        )
        self.assertEqual(
            dataset.parse_volume_filename("K019_Week1_SEMANTIC.nii.gz", "SEMANTIC"),
            ("K019", 1),
        )
        with self.assertRaisesRegex(ValueError, "Invalid LFOV filename"):
            dataset.parse_volume_filename("K019_WeekX_LFOV.nii.gz", "LFOV")
        with self.assertRaisesRegex(ValueError, "Invalid LFOV filename"):
            dataset.parse_volume_filename("K019_Week1_SEMANTIC.nii.gz", "LFOV")

    def test_approved_patient_sets_have_no_intersections(self) -> None:
        lookup = dataset.validate_approved_split()
        self.assertEqual(len(lookup), 38)
        self.assertEqual(lookup["K019"], "train")
        train, validation, test = map(set, dataset.APPROVED_PATIENTS.values())
        self.assertFalse(train & validation)
        self.assertFalse(train & test)
        self.assertFalse(validation & test)
        self.assertEqual(tuple(map(len, dataset.APPROVED_PATIENTS.values())), (26, 6, 6))
        self.assertEqual(dataset.SPLIT_SEED_PROVENANCE, 3710)

    def test_complete_discovery_is_reproducible_and_leakage_free(self) -> None:
        self.add_complete_synthetic_dataset()
        first = dataset.discover_dataset(self.root)
        self.assertEqual(first, dataset.discover_dataset(self.root))
        self.assertEqual(
            {split: len(pairs) for split, pairs in first.items()},
            dataset.EXPECTED_VOLUME_COUNTS,
        )
        self.assertEqual(sum(map(len, first.values())), 211)
        self.assertEqual(
            {pair.patient_id for pairs in first.values() for pair in pairs},
            set(dataset.validate_approved_split()),
        )
        self.assertIn(("K019", 1), {(pair.patient_id, pair.week) for pair in first["train"]})
        for split, pairs in first.items():
            self.assertEqual(
                {pair.patient_id for pair in pairs},
                set(dataset.APPROVED_PATIENTS[split]),
            )
            self.assertTrue(all(pair.mri_path.is_file() and pair.segmentation_path.is_file() for pair in pairs))

    def test_missing_segmentation_names_the_expected_file(self) -> None:
        (self.mri_dir / "K019_Week1_LFOV.nii.gz").touch()
        with self.assertRaisesRegex(FileNotFoundError, "K019_Week1_SEMANTIC.nii.gz"):
            dataset.discover_pairs(self.root)

    def test_missing_mri_names_the_expected_file(self) -> None:
        (self.mask_dir / "K019_Week1_SEMANTIC.nii.gz").touch()
        with self.assertRaisesRegex(FileNotFoundError, "K019_Week1_LFOV.nii.gz"):
            dataset.discover_pairs(self.root)

    def test_missing_dataset_root_is_explicit(self) -> None:
        with self.assertRaisesRegex(FileNotFoundError, "HipMRI dataset root is missing"):
            dataset.discover_pairs(self.root / "absent")

    def test_duplicate_logical_week_is_rejected(self) -> None:
        self.add_pair("K019", 1)
        (self.mri_dir / "K019_Week01_LFOV.nii.gz").touch()
        with self.assertRaisesRegex(ValueError, "Duplicate MRI for K019_Week1"):
            dataset.discover_pairs(self.root)

    def test_unapproved_patient_is_rejected(self) -> None:
        self.add_pair("Z999", 1)
        with self.assertRaisesRegex(ValueError, "unexpected patients=\\['Z999'\\]"):
            dataset.discover_dataset(self.root)

    def test_missing_approved_patients_are_reported(self) -> None:
        self.add_pair("K019", 1)
        with self.assertRaisesRegex(ValueError, "missing approved patients="):
            dataset.discover_dataset(self.root)

    def test_incomplete_volume_count_is_rejected(self) -> None:
        self.add_complete_synthetic_dataset()
        (self.mri_dir / "B006_Week1_LFOV.nii.gz").unlink()
        (self.mask_dir / "B006_Week1_SEMANTIC.nii.gz").unlink()
        with self.assertRaisesRegex(ValueError, "HipMRI volume coverage mismatch"):
            dataset.discover_dataset(self.root)

    def test_k019_week1_is_required(self) -> None:
        self.add_complete_synthetic_dataset()
        (self.mri_dir / "K019_Week1_LFOV.nii.gz").unlink()
        (self.mask_dir / "K019_Week1_SEMANTIC.nii.gz").unlink()
        with self.assertRaisesRegex(ValueError, "Expected K019_Week1"):
            dataset.discover_dataset(self.root)


if __name__ == "__main__":
    unittest.main()
