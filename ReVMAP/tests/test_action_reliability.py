import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from kgml_pipeline.action_reliability import (
    DEFAULT_CLASSES,
    _balanced_blind_sample,
    _evidence_status,
    _hierarchical_metrics,
    _probability_metrics,
    _rerank_kag_paths,
    _temperature_scale,
    evaluate_target_context_sensitivity,
    fit_temperature,
)


class ActionReliabilityTest(unittest.TestCase):
    def test_temperature_scaling_preserves_argmax_and_can_reduce_nll(self):
        probabilities = np.asarray([
            [0.55, 0.20, 0.10, 0.15],
            [0.15, 0.55, 0.10, 0.20],
            [0.10, 0.15, 0.55, 0.20],
            [0.15, 0.15, 0.15, 0.55],
        ] * 20)
        labels = list(DEFAULT_CLASSES) * 20
        temperature = fit_temperature(probabilities, labels, DEFAULT_CLASSES)
        scaled = _temperature_scale(probabilities, temperature)
        self.assertTrue(np.array_equal(probabilities.argmax(1), scaled.argmax(1)))
        before = _probability_metrics(probabilities, labels, DEFAULT_CLASSES)["nll"]
        after = _probability_metrics(scaled, labels, DEFAULT_CLASSES)["nll"]
        self.assertLess(after, before)

    def test_hierarchy_separates_direction_from_inhibitory_subtype_error(self):
        metrics = _hierarchical_metrics(
            ["agonist", "blocker", "antagonist", "inhibitor"],
            ["inhibitor", "inhibitor", "antagonist", "blocker"],
        )
        self.assertAlmostEqual(metrics["four_class_accuracy"], 0.25)
        self.assertAlmostEqual(metrics["severe_direction_error_rate"], 0.25)
        self.assertAlmostEqual(metrics["inhibitory_subtype_error_rate_all"], 0.5)
        self.assertAlmostEqual(metrics["hierarchical_credit"], 0.5)

    def test_evidence_status_uses_exact_class_and_polarity(self):
        self.assertEqual(_evidence_status("blocker", []), "insufficient")
        self.assertEqual(_evidence_status("blocker", ["blocker"]), "supported")
        self.assertEqual(
            _evidence_status("blocker", ["inhibitor"]), "ontology_disagreement"
        )
        self.assertEqual(_evidence_status("agonist", ["inhibitor"]), "contradicted")
        self.assertEqual(
            _evidence_status("blocker", ["blocker", "inhibitor"]),
            "partially_supported",
        )

    def test_target_context_audit_detects_drug_level_label_expansion(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for repeat in (0, 1):
                for drug, target, label in (
                    ("d1", "t1", "agonist"),
                    ("d1", "t2", "agonist"),
                    ("d2", "t1", "inhibitor"),
                ):
                    rows.append({
                        "strategy": "pair_conditioned_gate",
                        "split": "random_pair",
                        "repeat": repeat,
                        "fold": 0,
                        "drug_name": drug,
                        "target_name": target,
                        "true_label": label,
                    })
            prediction_path = root / "predictions.csv"
            pd.DataFrame(rows).to_csv(prediction_path, index=False)
            audit, sensitivity = evaluate_target_context_sensitivity(
                prediction_path,
                output_dir=root,
            )
            self.assertEqual(int(audit.loc[0, "pair_count"]), 3)
            self.assertEqual(int(audit.loc[0, "multi_target_drug_count"]), 1)
            self.assertEqual(int(audit.loc[0, "multi_action_drug_count"]), 0)
            self.assertEqual(float(audit.loc[0, "single_action_drug_fraction"]), 1.0)
            self.assertTrue(sensitivity.empty)

    def test_kag_reranking_uses_only_available_relation_views(self):
        attribution = pd.DataFrame([
            {"view": "targets_pathway", "deletion_delta": 0.4, "gate_weight": 0.2},
            {"view": "targets_anatomy", "deletion_delta": 0.1, "gate_weight": 0.2},
            {"view": "full_label_free", "deletion_delta": 9.0, "gate_weight": 0.9},
        ])
        paths = [
            {"path_id": "p1", "path_type": "shared_pathway", "score": 0.8},
            {"path_id": "p2", "path_type": "shared_anatomy", "score": 0.8},
        ]
        reranked = _rerank_kag_paths(paths, attribution)
        self.assertEqual(reranked[0]["path_id"], "p1")
        self.assertAlmostEqual(reranked[0]["normalised_view_importance"], 1.0)
        self.assertEqual(
            reranked[0]["evidence_tier"], "relation_specific_hypothesis"
        )
        anatomy = next(row for row in reranked if row["path_id"] == "p2")
        self.assertEqual(anatomy["evidence_tier"], "broad_context_only")
        self.assertLess(anatomy["path_type_prior"], reranked[0]["path_type_prior"])

    def test_blind_sample_is_unique_and_keeps_insufficient_cases(self):
        rows = []
        graph_lookup = {}
        for split in ("drug_disjoint", "random_pair"):
            for index in range(8):
                pair_key = "d%s||t%s" % (index, index)
                rows.append({
                    "split": split,
                    "pair_key": pair_key,
                    "drug_name": "d%s" % index,
                    "target_name": "t%s" % index,
                    "true_label": "agonist",
                    "predicted_label": "agonist" if index % 2 else "inhibitor",
                    "evidence_status": "insufficient" if index % 3 == 0 else "supported",
                })
                graph_lookup[(split, pair_key)] = {
                    "attribution_guided_context_paths": (
                        []
                        if index % 3 == 0
                        else [{"path_type": "shared_pathway"}]
                    )
                }
        sample = _balanced_blind_sample(pd.DataFrame(rows), graph_lookup, 8)
        self.assertEqual(len(sample), 8)
        self.assertEqual(sample["pair_key"].nunique(), 8)
        self.assertIn("insufficient", set(sample["evidence_status"]))


if __name__ == "__main__":
    unittest.main()
