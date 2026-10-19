import csv
import json
import logging
import tempfile
import unittest
from pathlib import Path

import numpy as np

from kgml_pipeline.relation_routing import (
    SoftmaxRoutingGate,
    _component_deletion_deltas,
    _guide_kag_evidence,
    evaluate_relation_routing,
)


def _write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class RelationRoutingTest(unittest.TestCase):
    def test_gate_learns_to_route_to_the_informative_expert(self):
        labels = np.asarray([index % 2 for index in range(160)], dtype=int)
        probabilities = np.zeros((len(labels), 3, 2), dtype=float)
        probabilities[:, 0, :] = np.where(
            labels[:, None] == np.asarray([0, 1])[None, :],
            0.95,
            0.05,
        )
        probabilities[:, 1, :] = 0.5
        probabilities[:, 2, :] = probabilities[:, 0, ::-1]

        gate = SoftmaxRoutingGate(
            learning_rate=0.05,
            max_iter=500,
            l2=1.0e-4,
            patience=40,
            random_seed=7,
        ).fit(probabilities, labels)
        weights = gate.predict_weights(probabilities)
        mixture = gate.predict_proba(probabilities)

        self.assertGreater(weights[:, 0].mean(), 0.8)
        self.assertGreater(
            float(np.mean(mixture.argmax(axis=1) == labels)),
            0.99,
        )
        deltas, _, _ = _component_deletion_deltas(probabilities, weights)
        self.assertGreater(deltas[:, 0].mean(), deltas[:, 1].mean())

    def test_kag_ranking_uses_positive_component_deletion_signal(self):
        bundles = [{
            "query_id": "q1",
            "drug_name": "D",
            "target_name": "T",
            "question": "Why?",
            "context_edges": [],
            "paths": [
                {
                    "path_id": "anatomy",
                    "path_type": "shared_anatomy",
                    "score": 3.0,
                    "citations": [],
                },
                {
                    "path_id": "pathway",
                    "path_type": "shared_pathway",
                    "score": 2.0,
                    "citations": [],
                },
            ],
        }]
        routing_cases = [{
            "drug_name": "D",
            "target_name": "T",
            "split": "double_disjoint",
            "repeat": 0,
            "fold": 0,
            "predicted_label": "blocker",
            "prediction_scores": {"blocker": 0.8, "agonist": 0.2},
            "relation_attribution": [
                {
                    "relation": "pathway",
                    "gate_weight": 0.7,
                    "positive_deletion_delta": 0.8,
                    "signed_deletion_delta": 0.8,
                    "source_views": ["targets_pathway"],
                },
                {
                    "relation": "anatomy",
                    "gate_weight": 0.3,
                    "positive_deletion_delta": 0.1,
                    "signed_deletion_delta": 0.1,
                    "source_views": ["targets_anatomy"],
                },
            ],
        }]
        guided, audit = _guide_kag_evidence(
            bundles,
            routing_cases,
            score_floor=0.05,
        )
        self.assertEqual("pathway", guided[0]["paths"][0]["path_id"])
        self.assertTrue(audit[0]["top_path_changed"])
        self.assertEqual(
            "positive_view_component_deletion_delta",
            guided[0]["guidance_basis"],
        )

    def test_end_to_end_writes_nested_oof_and_faithfulness_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            processed = root / "processed"
            reports = root / "reports"
            pairs_path = root / "pairs.csv"
            drugs = ["d%02d" % index for index in range(12)]
            targets = ["t%02d" % index for index in range(3)]
            rows = []
            for drug_index, drug in enumerate(drugs):
                for target_index, target in enumerate(targets):
                    rows.append({
                        "drug_name": drug,
                        "target_name": target,
                        "action": (
                            "agonist"
                            if (drug_index + target_index) % 2 == 0
                            else "blocker"
                        ),
                    })
            _write_csv(
                pairs_path,
                ["drug_name", "target_name", "action"],
                rows,
            )

            entities = drugs + targets
            mapping = {entity: index for index, entity in enumerate(entities)}
            rng = np.random.RandomState(11)
            for view_index, view in enumerate(
                ["targets_only", "targets_pathway", "full"]
            ):
                values = rng.normal(
                    loc=float(view_index) * 0.1,
                    scale=1.0,
                    size=(len(entities), 6),
                ).astype("float32")
                model_root = processed / "embeddings" / view / "RGCN"
                model_root.mkdir(parents=True)
                np.save(model_root / "entity_embeddings.npy", values)
                (model_root / "entity_to_id.json").write_text(
                    json.dumps(mapping),
                    encoding="utf-8",
                )

            config = {
                "random_seed": 42,
                "paths": {
                    "intermediate_dir": str(root / "intermediate"),
                    "processed_dir": str(processed),
                    "reports_dir": str(reports),
                },
                "views": {
                    "targets_only": ["target"],
                    "targets_pathway": ["target", "pathway"],
                    "full": ["target", "pathway", "anatomy"],
                },
                "pair_evaluation": {
                    "pair_table_path": str(pairs_path),
                },
                "relation_routing": {
                    "cohort_name": "synthetic",
                    "pair_table_path": str(pairs_path),
                    "output_dir": str(reports / "routing"),
                    "embedding_model": "RGCN",
                    "expert_classifier": "Random Forest",
                    "allowed_views": [
                        "targets_only",
                        "targets_pathway",
                        "full",
                    ],
                    "splits": ["random_pair"],
                    "cv_folds": 3,
                    "cv_repeats": 1,
                    "gate_inner_folds": 2,
                    "rf_estimators": 10,
                    "gate_max_iter": 80,
                    "gate_patience": 8,
                    "label_tasks": {
                        "core2": {
                            "label_column": "action",
                            "min_class_count": 2,
                            "allowed_classes": ["agonist", "blocker"],
                        }
                    },
                },
            }
            summary = evaluate_relation_routing(
                config,
                logging.getLogger("relation-routing-test"),
            )
            output = reports / "routing"
            self.assertFalse(summary.empty)
            self.assertIn(
                "pair_conditioned_gate",
                set(summary["strategy"]),
            )
            self.assertTrue((output / "routing_oof_predictions.csv").exists())
            self.assertTrue((output / "routing_view_attribution.csv").exists())
            self.assertTrue((output / "routing_faithfulness.csv").exists())
            self.assertTrue((output / "routing_cases.jsonl").exists())
            manifest = json.loads(
                (output / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertTrue(manifest["nested_oof_gate_training"])
            self.assertFalse(manifest["outer_test_used_for_gate_training"])
            self.assertFalse(manifest["rgcn_edge_level_attribution"])


if __name__ == "__main__":
    unittest.main()
