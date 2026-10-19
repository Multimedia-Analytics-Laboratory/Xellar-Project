from __future__ import annotations

import unittest

from kgml_pipeline.config import validate_config


def _base_config():
    return {
        "paths": {
            "intermediate_dir": "/tmp/intermediate",
            "processed_dir": "/tmp/processed",
            "reports_dir": "/tmp/reports",
            "primekg_csv": "/tmp/primekg.csv",
            "combined_metadata_csv": "/tmp/metadata.csv",
            "repurposing_hub_tsv": "/tmp/hub.tsv",
        },
        "benchmark": {
            "drug_name_column": "drug",
            "repurposing_hub_name_column": "name",
            "repurposing_hub_moa_column": "moa",
            "repurposing_hub_target_column": "target",
            "minimum_moa_count": 3,
        },
        "training": {
            "split_ratios": [0.8, 0.1, 0.1],
            "models": ["TransE"],
        },
        "views": {"full": ["target"]},
        "evaluation": {
            "protocol": "repeated_stratified_cv",
            "label_group": "moa",
            "allow_label_group_in_views": False,
        },
    }


class LeakageConfigTest(unittest.TestCase):
    def test_strict_view_without_moa_is_valid(self):
        report = validate_config(_base_config(), check_inputs=False)
        self.assertEqual([], report["errors"])

    def test_strict_view_with_moa_is_rejected(self):
        config = _base_config()
        config["views"]["full"].append("moa")
        report = validate_config(config, check_inputs=False)
        self.assertTrue(any("leaks downstream labels" in error for error in report["errors"]))

    def test_legacy_override_is_explicit_and_warns(self):
        config = _base_config()
        config["views"]["full"].append("moa")
        config["evaluation"]["allow_label_group_in_views"] = True
        report = validate_config(config, check_inputs=False)
        self.assertEqual([], report["errors"])
        self.assertTrue(any("Legacy label leakage" in warning for warning in report["warnings"]))


if __name__ == "__main__":
    unittest.main()
