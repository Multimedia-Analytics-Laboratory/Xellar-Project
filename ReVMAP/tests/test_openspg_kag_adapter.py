import unittest

from kgml_pipeline.openspg_kag_adapter import (
    _degree_matched_random_controls,
    _double_disjoint_training_pairs,
    _mask_evaluated_target_edges,
    _node_id,
    _retrieve_pair_bundle,
    _signed_analogy_evidence,
)
from kgml_pipeline.pair_evaluation import _iter_splits


def _edge(group, source_type, source, target_type, target, label):
    return {
        "id": "%s-%s-%s" % (group, source, target),
        "from": _node_id(source_type, source),
        "fromType": source_type,
        "fromName": source,
        "to": _node_id(target_type, target),
        "toType": target_type,
        "toName": target,
        "label": label,
        "source_group": group,
        "source_relation": label,
        "source_database": "synthetic",
    }


class OpenSPGKAGAdapterTest(unittest.TestCase):
    def setUp(self):
        self.edges = [
            _edge("target", "Drug", "D", "Target", "T", "targets"),
            _edge("target", "Drug", "D", "Target", "T2", "targets"),
            _edge("pathway", "Target", "T", "Pathway", "P", "participatesIn"),
            _edge("pathway", "Target", "T2", "Pathway", "P", "participatesIn"),
            _edge("anatomy", "Target", "T", "Anatomy", "A", "expressedIn"),
            _edge("anatomy", "Target", "T2", "Anatomy", "A", "expressedIn"),
        ]
        self.pairs = [{"drug_name": "D", "target_name": "T", "action": "inhibitor"}]

    def test_direct_pair_edge_is_removed(self):
        retained, removed = _mask_evaluated_target_edges(self.edges, self.pairs)
        self.assertEqual(1, len(removed))
        self.assertFalse(
            any(
                edge["from"] == _node_id("Drug", "D")
                and edge["to"] == _node_id("Target", "T")
                for edge in retained
            )
        )

    def test_indirect_shared_context_paths_are_retrieved_without_action_claim(self):
        retained, _ = _mask_evaluated_target_edges(self.edges, self.pairs)
        bundle = _retrieve_pair_bundle(retained, "D", "T", "inhibitor")
        self.assertTrue(bundle["direct_edge_masked"])
        self.assertEqual(2, bundle["indirect_path_count"])
        self.assertEqual(
            {"shared_pathway", "shared_anatomy"},
            {path["path_type"] for path in bundle["paths"]},
        )
        self.assertIn(
            "insufficient relation-specific path evidence", bundle["prompt"]["system"]
        )
        self.assertNotIn("inhibitor", bundle["question"])
        self.assertEqual("targets", bundle["held_out_relation"])
        retrieved_ids = {
            citation["edge_id"]
            for path in bundle["paths"]
            for citation in path["citations"]
        }
        self.assertNotIn("target-D-T", retrieved_ids)

    def test_shared_indication_cotarget_path_is_retrieved(self):
        edges = list(self.edges)
        edges.extend([
            _edge("indication", "Drug", "D", "Indication", "I", "indicatedFor"),
            _edge("indication", "Drug", "D3", "Indication", "I", "indicatedFor"),
            _edge("target", "Drug", "D3", "Target", "T", "targets"),
        ])
        retained, _ = _mask_evaluated_target_edges(edges, self.pairs)
        bundle = _retrieve_pair_bundle(retained, "D", "T", "inhibitor")
        indication_paths = [
            path
            for path in bundle["paths"]
            if path["path_type"] == "shared_indication_cotarget"
        ]
        self.assertEqual(1, len(indication_paths))
        self.assertEqual(["D", "I", "D3", "T"], indication_paths[0]["nodes"])

    def test_degree_matched_control_uses_a_different_unknown_target(self):
        retained, _ = _mask_evaluated_target_edges(self.edges, self.pairs)
        retained.extend([
            _edge("pathway", "Target", "T3", "Pathway", "P2", "participatesIn"),
            _edge("pathway", "Target", "T2", "Pathway", "P2", "participatesIn"),
        ])
        bundle = _retrieve_pair_bundle(retained, "D", "T", "inhibitor")
        controls = _degree_matched_random_controls(
            retained, self.pairs, [bundle], max_paths=20, max_context=20
        )
        self.assertEqual(1, len(controls))
        self.assertNotEqual("T", controls[0]["control_target_name"])
        self.assertNotEqual("T2", controls[0]["control_target_name"])

    def test_signed_analogies_use_exact_double_disjoint_training_fold(self):
        import numpy as np

        drugs = ["D%d" % index for index in range(4)]
        targets = ["T%d" % index for index in range(4)]
        actions = ["agonist", "antagonist", "blocker", "inhibitor"]
        pairs = []
        edges = []
        for drug_index, drug in enumerate(drugs):
            edges.append(
                _edge("indication", "Drug", drug, "Indication", "I", "indicatedFor")
            )
            for target_index, target in enumerate(targets):
                pairs.append({
                    "drug_name": drug,
                    "target_name": target,
                    "action": actions[(drug_index + target_index) % len(actions)],
                })
        for target in targets:
            edges.append(
                _edge("pathway", "Target", target, "Pathway", "P", "participatesIn")
            )
        pairs.sort(key=lambda row: (row["drug_name"], row["target_name"], row["action"]))
        label_ids = {label: index for index, label in enumerate(sorted(actions))}
        encoded = np.asarray([label_ids[row["action"]] for row in pairs])
        drug_array = np.asarray([row["drug_name"] for row in pairs], dtype=object)
        target_array = np.asarray([row["target_name"] for row in pairs], dtype=object)
        predictions = []
        for fold, (_, test_indices) in enumerate(
            _iter_splits(
                "double_disjoint",
                2,
                42,
                np.zeros((len(pairs), 1)),
                encoded,
                drug_array,
                target_array,
            )
        ):
            for index in test_indices:
                pair = pairs[int(index)]
                predictions.append({
                    "drug_name": pair["drug_name"],
                    "target_name": pair["target_name"],
                    "true_label": pair["action"],
                    "predicted_label": pair["action"],
                    "fold": fold,
                })

        training_by_fold, _ = _double_disjoint_training_pairs(
            pairs, predictions, entity_partitions=2, seed=42
        )
        bundles, summary = _signed_analogy_evidence(
            edges,
            pairs,
            predictions,
            top_k=20,
            entity_partitions=2,
            seed=42,
        )
        self.assertEqual(len(pairs), summary["pair_count"])
        self.assertEqual(1.0, summary["coverage"])
        for bundle in bundles:
            allowed = {
                (row["drug_name"], row["target_name"])
                for row in training_by_fold[bundle["fold"]]
            }
            cited = {
                (row["training_drug"], row["training_target"])
                for row in bundle["analogies"]
            }
            self.assertTrue(cited)
            self.assertTrue(cited <= allowed)
            self.assertFalse(
                bundle["leakage_audit"]["training_contains_test_drug"]
            )
            self.assertFalse(
                bundle["leakage_audit"]["training_contains_test_target"]
            )


if __name__ == "__main__":
    unittest.main()
