import unittest

import numpy as np
import pandas as pd

from kgml_pipeline.pair_evaluation import _make_pair_matrix
from kgml_pipeline.relation_routing import (
    _build_aligned_matrices,
    _common_pair_frame,
    _routing_feature_config,
)


class MatchedRoutingFeatureTest(unittest.TestCase):
    def setUp(self):
        self.source = {
            "view": "one",
            "entity_to_id": {"drug": 0, "t1": 1, "t2": 2},
            "features": np.array([[1., 2.], [3., 5.], [8., 13.]]),
        }
        self.pairs = pd.DataFrame([
            {"drug_name": "drug", "target_name": "t1", "action": "agonist"},
            {"drug_name": "drug", "target_name": "t2", "action": "agonist"},
            {"drug_name": "drug", "target_name": "missing", "action": "agonist"},
        ])

    def test_default_pair_is_identical_to_existing_builder(self):
        matrices, labels, drugs, targets = _build_aligned_matrices(
            [self.source], self.pairs, {},
        )
        original, old_labels, old_drugs, old_targets = _make_pair_matrix(
            self.source, self.pairs, {},
        )
        np.testing.assert_array_equal(matrices["one"], original)
        self.assertEqual((labels, drugs, targets), (old_labels, old_drugs, old_targets))
        self.assertEqual(original.shape, (2, 8))
        self.assertFalse(np.array_equal(original[0], original[1]))

    def test_drug_only_removes_all_target_components_without_changing_rows(self):
        eligible = _common_pair_frame(self.pairs, [self.source])
        original_cfg = {"feature_mode": "drug_only", "include_target_embedding": True}
        matrices, labels, drugs, targets = _build_aligned_matrices(
            [self.source], eligible, original_cfg,
        )
        np.testing.assert_array_equal(matrices["one"], [[1., 2.], [1., 2.]])
        self.assertEqual(targets, ["t1", "t2"])
        self.assertEqual(labels, ["agonist", "agonist"])
        self.assertEqual(drugs, ["drug", "drug"])
        self.assertTrue(original_cfg["include_target_embedding"])

    def test_invalid_feature_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            _routing_feature_config({"feature_mode": "unsupported"})


if __name__ == "__main__":
    unittest.main()
