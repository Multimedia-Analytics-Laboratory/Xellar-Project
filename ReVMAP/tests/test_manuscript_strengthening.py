import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from kgml_pipeline.manuscript_strengthening import (
    diagnose_distmult,
    evaluate_graph_context_sparsity,
)


class ManuscriptStrengtheningTest(unittest.TestCase):
    def test_distmult_diagnostic_separates_near_random_views(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "training.csv"
            pd.DataFrame([
                {
                    "view": "full",
                    "model": "DistMult",
                    "mrr": 0.002,
                    "entity_count": 5000,
                    "relation_count": 4,
                    "status": "success",
                },
                {
                    "view": "full",
                    "model": "RGCN",
                    "mrr": 0.60,
                    "entity_count": 5000,
                    "relation_count": 4,
                    "status": "success",
                },
                {
                    "view": "anatomy",
                    "model": "DistMult",
                    "mrr": 0.30,
                    "entity_count": 3000,
                    "relation_count": 2,
                    "status": "success",
                },
                {
                    "view": "anatomy",
                    "model": "RGCN",
                    "mrr": 0.50,
                    "entity_count": 3000,
                    "relation_count": 2,
                    "status": "success",
                },
            ]).to_csv(source, index=False)
            audit = diagnose_distmult(source, output_dir=root)
            flags = audit.set_index("view")["near_random_flag"].to_dict()
            self.assertTrue(flags["full"])
            self.assertFalse(flags["anatomy"])

    def test_context_sparsity_uses_masked_graph_degrees(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            predictions = root / "predictions.csv"
            rows = []
            for repeat in (0, 1):
                for index in range(8):
                    rows.append({
                        "strategy": "pair_conditioned_gate",
                        "split": "random_pair",
                        "repeat": repeat,
                        "drug_name": "d%s" % index,
                        "target_name": "t%s" % index,
                        "true_label": "agonist" if index % 2 else "inhibitor",
                        "predicted_label": "agonist" if index % 2 else "inhibitor",
                    })
            pd.DataFrame(rows).to_csv(predictions, index=False)
            edges = []
            for index in range(8):
                for degree in range(index + 1):
                    edges.append({
                        "from": "Drug::d%s" % index,
                        "to": "Indication::i%s_%s" % (index, degree),
                    })
                    edges.append({
                        "from": "Target::t%s" % index,
                        "to": "Pathway::p%s_%s" % (index, degree),
                    })
            edge_path = root / "edges.json"
            edge_path.write_text(json.dumps(edges), encoding="utf-8")
            summary, contrast = evaluate_graph_context_sparsity(
                predictions, edge_path, output_dir=root
            )
            self.assertFalse(summary.empty)
            self.assertFalse(contrast.empty)
            pairs = pd.read_csv(root / "graph_context_degree_by_pair.csv")
            self.assertEqual(int(pairs["drug_context_degree"].max()), 8)


if __name__ == "__main__":
    unittest.main()
