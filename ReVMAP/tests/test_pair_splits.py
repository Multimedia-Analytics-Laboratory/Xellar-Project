import unittest

import numpy as np

from kgml_pipeline.pair_evaluation import _effective_fold_count, _iter_splits


class DoubleDisjointSplitTest(unittest.TestCase):
    def setUp(self):
        # A dense-enough bipartite cohort with four action classes. Repeated
        # entities ensure that ordinary pair-level splitting would leak both
        # drugs and targets across train and test.
        self.drugs = []
        self.targets = []
        self.labels = []
        for drug_index in range(12):
            for target_index in range(9):
                self.drugs.append("drug_%02d" % drug_index)
                self.targets.append("target_%02d" % target_index)
                self.labels.append((drug_index + target_index) % 4)
        self.labels = np.asarray(self.labels, dtype=int)
        self.matrix = np.zeros((len(self.labels), 1), dtype=np.float32)

    def test_double_disjoint_has_zero_entity_overlap_and_full_coverage(self):
        splits = list(
            _iter_splits(
                "double_disjoint",
                3,
                42,
                self.matrix,
                self.labels,
                self.drugs,
                self.targets,
            )
        )
        self.assertEqual(9, len(splits))
        self.assertEqual(9, _effective_fold_count("double_disjoint", 3))

        test_counts = np.zeros(len(self.labels), dtype=int)
        for train_indices, test_indices in splits:
            train_drugs = {self.drugs[index] for index in train_indices}
            test_drugs = {self.drugs[index] for index in test_indices}
            train_targets = {self.targets[index] for index in train_indices}
            test_targets = {self.targets[index] for index in test_indices}
            self.assertFalse(train_drugs & test_drugs)
            self.assertFalse(train_targets & test_targets)
            test_counts[test_indices] += 1

        np.testing.assert_array_equal(test_counts, np.ones(len(self.labels), dtype=int))

    def test_double_disjoint_is_deterministic_for_a_seed(self):
        first = list(
            _iter_splits(
                "double_disjoint",
                3,
                17,
                self.matrix,
                self.labels,
                self.drugs,
                self.targets,
            )
        )
        second = list(
            _iter_splits(
                "double_disjoint",
                3,
                17,
                self.matrix,
                self.labels,
                self.drugs,
                self.targets,
            )
        )
        self.assertEqual(len(first), len(second))
        for (train_a, test_a), (train_b, test_b) in zip(first, second):
            np.testing.assert_array_equal(train_a, train_b)
            np.testing.assert_array_equal(test_a, test_b)


if __name__ == "__main__":
    unittest.main()
