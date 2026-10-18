"""Guard the Train-only provenance and paired budget of the G5 candidate."""

import json
import unittest
from pathlib import Path

from tools.probe_g5_training_smoke import check_config, sha256_file


CODE = Path(__file__).resolve().parents[1]


class G5CandidateTests(unittest.TestCase):
    def test_candidate_matches_frozen_train_sources_and_budget(self):
        config = json.loads((CODE / "config/g5_controlled_training_candidate.json").read_text())
        source_path = CODE / "config/g5_weighted_ce_diagnostic.json"
        source = json.loads(source_path.read_text())
        self.assertEqual(config["split_sha256"], sha256_file(CODE / "config/patient_split.json"))
        self.assertEqual(config["objective"]["weight_source_config_sha256"], sha256_file(source_path))
        self.assertEqual(source["source_policy"], "patient_uniform")
        self.assertEqual(config["train_patients"], 26)
        self.assertEqual(config["budget"]["updates_per_epoch"],
                         26 * config["sampling"]["patches_per_patient_epoch"]
                         // config["sampling"]["batch_size"])
        self.assertEqual(config["budget"]["total_planned_updates"],
                         config["budget"]["epochs"] * config["budget"]["updates_per_epoch"])
        check_config(config, source)

    def test_smoke_refuses_unpaired_class_weights(self):
        config = json.loads((CODE / "config/g5_controlled_training_candidate.json").read_text())
        source = json.loads((CODE / "config/g5_weighted_ce_diagnostic.json").read_text())
        config["objective"]["ce_class_weights"][5] = 1.0
        with self.assertRaisesRegex(ValueError, "weights differ"):
            check_config(config, source)


if __name__ == "__main__":
    unittest.main()
