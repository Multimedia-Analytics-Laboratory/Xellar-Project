import csv
import json
import logging
import tempfile
import unittest
from pathlib import Path

import numpy as np

from kgml_pipeline.embedding_interpretation import (
    _candidate_drug_pairs,
    _load_relation_profiles,
    _view_attribution_rows,
    run_embedding_interpretation,
)


def _write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class EmbeddingInterpretationTest(unittest.TestCase):
    def test_candidate_pairs_are_same_action_and_explicit_cases_are_retained(self):
        pair_rows = [
            {"drug_name": "d1", "target_name": "t1", "action": "agonist"},
            {"drug_name": "d2", "target_name": "t1", "action": "agonist"},
            {"drug_name": "d3", "target_name": "t2", "action": "agonist"},
            {"drug_name": "d4", "target_name": "t3", "action": "blocker"},
        ]
        rows = _candidate_drug_pairs(
            pair_rows,
            {"d1", "d2", "d3", "d4"},
            {
                "max_pairs_per_action": 100,
                "explicit_case_pairs": [["d1", "d4"]],
                "random_seed": 42,
            },
        )
        by_pair = {(row["drug_a"], row["drug_b"]): row for row in rows}
        self.assertEqual(4, len(rows))
        self.assertEqual(1, by_pair[("d1", "d2")]["shared_target_count"])
        self.assertTrue(by_pair[("d1", "d4")]["explicit_case"])
        self.assertEqual("explicit_case", by_pair[("d1", "d4")]["primary_action"])

    def test_drug_contexts_are_derived_through_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_csv(
                root / "target.csv",
                ["head", "relation", "tail"],
                [
                    {"head": "d1", "relation": "target", "tail": "t1"},
                    {"head": "d2", "relation": "target", "tail": "t2"},
                ],
            )
            _write_csv(
                root / "indication.csv",
                ["head", "relation", "tail"],
                [{"head": "d1", "relation": "indication", "tail": "i1"}],
            )
            _write_csv(
                root / "pathway.csv",
                ["head", "relation", "tail"],
                [
                    {"head": "t1", "relation": "pathway", "tail": "p1"},
                    {"head": "t2", "relation": "pathway", "tail": "p1"},
                ],
            )
            _write_csv(
                root / "anatomy.csv",
                ["head", "relation", "tail"],
                [{"head": "t1", "relation": "anatomy", "tail": "a1"}],
            )
            profiles, support = _load_relation_profiles(root, ["d1", "d2"])
            self.assertEqual({"p1"}, profiles["d1"]["pathway"])
            self.assertEqual({"p1"}, profiles["d2"]["pathway"])
            self.assertEqual({"t1"}, support["d1"]["pathway"]["p1"])
            self.assertEqual({"i1"}, profiles["d1"]["indication"])

    def test_view_attribution_has_auditable_sign_convention(self):
        views = {
            "targets_only": ["target"],
            "targets_pathway": ["target", "pathway"],
            "full": ["target", "pathway", "indication"],
            "full_minus_pathway": ["target", "indication"],
        }
        rows = []
        for view, z_value in (
            ("targets_only", 0.1),
            ("targets_pathway", 0.7),
            ("full", 1.0),
            ("full_minus_pathway", 0.4),
        ):
            rows.append({
                "pair_id": "p1",
                "drug_a": "d1",
                "drug_b": "d2",
                "view": view,
                "cosine_distance_z": z_value,
                "cosine_distance_percentile": z_value / 2.0,
            })
        attribution = _view_attribution_rows(rows, views, "full", "targets_only")
        by_type = {
            (row["contrast_type"], row["relation"]): row
            for row in attribution
        }
        self.assertAlmostEqual(
            0.6,
            by_type[("add_one_relation", "pathway")]["cosine_distance_z_delta"],
        )
        self.assertAlmostEqual(
            0.6,
            by_type[("leave_one_relation_out", "pathway")][
                "cosine_distance_z_delta"
            ],
        )

    def test_end_to_end_writes_distances_attribution_and_grounded_cases(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            intermediate = root / "intermediate"
            processed = root / "processed"
            reports = root / "reports"
            group_root = intermediate / "triples" / "groups"
            pair_path = root / "pairs.csv"
            pair_rows = [
                {"drug_name": "d1", "target_name": "t1", "action": "agonist"},
                {"drug_name": "d2", "target_name": "t1", "action": "agonist"},
                {"drug_name": "d3", "target_name": "t2", "action": "agonist"},
                {"drug_name": "d4", "target_name": "t3", "action": "agonist"},
            ]
            _write_csv(pair_path, ["drug_name", "target_name", "action"], pair_rows)
            _write_csv(
                group_root / "target.csv",
                ["head", "relation", "tail"],
                [
                    {"head": row["drug_name"], "relation": "target", "tail": row["target_name"]}
                    for row in pair_rows
                ],
            )
            _write_csv(
                group_root / "indication.csv",
                ["head", "relation", "tail"],
                [
                    {"head": "d1", "relation": "indication", "tail": "i1"},
                    {"head": "d2", "relation": "indication", "tail": "i1"},
                    {"head": "d3", "relation": "indication", "tail": "i2"},
                    {"head": "d4", "relation": "indication", "tail": "i3"},
                ],
            )
            _write_csv(
                group_root / "pathway.csv",
                ["head", "relation", "tail"],
                [
                    {"head": "t1", "relation": "pathway", "tail": "p1"},
                    {"head": "t2", "relation": "pathway", "tail": "p2"},
                    {"head": "t3", "relation": "pathway", "tail": "p3"},
                ],
            )
            _write_csv(
                group_root / "anatomy.csv",
                ["head", "relation", "tail"],
                [
                    {"head": "t1", "relation": "anatomy", "tail": "a1"},
                    {"head": "t2", "relation": "anatomy", "tail": "a2"},
                    {"head": "t3", "relation": "anatomy", "tail": "a3"},
                ],
            )

            mapping = {"d1": 0, "d2": 1, "d3": 2, "d4": 3}
            embeddings = {
                "targets_only": np.asarray(
                    [[1.0, 0.0], [0.9, 0.1], [0.8, 0.2], [0.7, 0.3]]
                ),
                "targets_pathway": np.asarray(
                    [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [-0.2, 0.8]]
                ),
                "full": np.asarray(
                    [[1.0, 0.0], [0.7, 0.3], [0.0, 1.0], [-1.0, 0.0]]
                ),
                "full_minus_pathway": np.asarray(
                    [[1.0, 0.0], [0.8, 0.2], [0.4, 0.6], [0.0, 1.0]]
                ),
            }
            for view, values in embeddings.items():
                model_root = processed / "embeddings" / view / "RGCN"
                model_root.mkdir(parents=True)
                np.save(model_root / "entity_embeddings.npy", values)
                (model_root / "entity_to_id.json").write_text(
                    json.dumps(mapping), encoding="utf-8"
                )

            config = {
                "random_seed": 42,
                "paths": {
                    "intermediate_dir": str(intermediate),
                    "processed_dir": str(processed),
                    "reports_dir": str(reports),
                },
                "views": {
                    "targets_only": ["target"],
                    "targets_pathway": ["target", "pathway"],
                    "full": ["target", "indication", "anatomy", "pathway"],
                    "full_minus_pathway": ["target", "indication", "anatomy"],
                },
                "embedding_interpretation": {
                    "cohort_name": "synthetic",
                    "pair_table_path": str(pair_path),
                    "output_dir": str(reports / "interpretation"),
                    "embedding_model": "RGCN",
                    "reference_view": "full",
                    "baseline_view": "targets_only",
                    "resampling_iterations": 5,
                    "distant_quantile": 0.5,
                    "close_quantile": 0.5,
                    "case_count": 2,
                    "explicit_case_pairs": [["d1", "d4"]],
                },
            }
            manifests = run_embedding_interpretation(
                config, logging.getLogger("embedding-interpretation-test")
            )
            output = reports / "interpretation"
            self.assertIn("synthetic", manifests)
            self.assertTrue((output / "embedding_distances_by_view.csv").exists())
            self.assertTrue((output / "view_level_attribution.csv").exists())
            self.assertTrue((output / "grounded_embedding_cases.jsonl").exists())
            self.assertTrue((output / "case_report_index.csv").exists())
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertFalse(manifest["uses_tsne_for_distance"])
            self.assertEqual(
                "separately_trained_graph_view_contrast",
                manifest["attribution_level"],
            )
            case = json.loads(
                (output / "grounded_embedding_cases.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()[0]
            )
            self.assertIn("not a causal drug mechanism", case["prompt"]["system"])
            self.assertIn("coverage_status", case["relation_evidence"][0])
            self.assertIn("geometry_role", case["relation_evidence"][0])
            case_reports = list((output / "case_reports").glob("*.md"))
            self.assertTrue(case_reports)


if __name__ == "__main__":
    unittest.main()
