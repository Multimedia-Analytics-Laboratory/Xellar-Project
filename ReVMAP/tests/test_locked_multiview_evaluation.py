import unittest

import pandas as pd

from kgml_pipeline.locked_multiview_evaluation import select_locked_strategy


class LockedMultiviewSelectionTest(unittest.TestCase):
    def test_pareto_minimax_rule_selects_one_strategy_across_splits(self):
        scores = {
            "expert::full_label_free": [0.95, 0.70, 0.80],
            "uniform_average": [0.94, 0.75, 0.82],
            "ordinary_stacking": [0.97, 0.74, 0.81],
            "pair_conditioned_gate": [0.96, 0.76, 0.79],
        }
        splits = ["random_pair", "drug_disjoint", "target_disjoint"]
        rows = []
        for strategy, values in scores.items():
            for split, value in zip(splits, values):
                rows.append({
                    "task": "core4_action",
                    "split": split,
                    "strategy": strategy,
                    "cv_repeats": 3,
                    "f1_macro_mean": value,
                })
        summary = pd.DataFrame(rows)
        selection = {
            "task": "core4_action",
            "required_repeats": 3,
            "selection_splits": splits,
            "candidates": list(scores),
            "deterministic_tie_break_order": list(scores),
        }

        selected, audit = select_locked_strategy(summary, selection)

        self.assertEqual("ordinary_stacking", selected)
        self.assertEqual(1, int(audit["selected"].sum()))
        self.assertTrue(
            bool(
                audit.loc[
                    audit["strategy"].eq(selected),
                    "pareto_optimal",
                ].iloc[0]
            )
        )

    def test_selection_rejects_fewer_than_required_repeats(self):
        summary = pd.DataFrame([{
            "task": "core4_action",
            "split": "random_pair",
            "strategy": "uniform_average",
            "cv_repeats": 1,
            "f1_macro_mean": 0.8,
        }])
        selection = {
            "task": "core4_action",
            "required_repeats": 3,
            "selection_splits": ["random_pair"],
            "candidates": ["uniform_average"],
        }
        with self.assertRaisesRegex(ValueError, "at least 3 repeats"):
            select_locked_strategy(summary, selection)


if __name__ == "__main__":
    unittest.main()
