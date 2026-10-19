"""Pair-conditioned routing across heterogeneous KG relation views.

The routing model treats each relation view as a predictive expert.  Base
experts are trained inside every outer fold.  A softmax gate is then trained
from inner out-of-fold expert probabilities, so the gate never learns from
in-sample predictions and the outer test fold remains untouched.

The same gate is used for two auditable interpretation products:

* view-component deletion: remove one routed expert, renormalize the remaining
  weights, and measure the predicted-class probability change; and
* attribution-guided KAG ranking: map view weights/deletion effects to typed
  relations and use them to rerank already grounded KAG evidence paths.

This is faithfulness to the routed mixture, not edge-level causal attribution
inside an individual RGCN expert.
"""

from __future__ import annotations

import gzip
import json
import math
from collections import defaultdict
from pathlib import Path

import pandas as pd

from .io_utils import ensure_dir, write_json
from .pair_evaluation import (
    _build_pair_classifiers,
    _embedding_sources,
    _filter_pairs_for_task,
    _fold_count,
    _iter_splits,
    _load_pair_table,
    _make_pair_matrix,
    _pair_label_tasks,
)
from .strict_evaluation import METRIC_NAMES, _metric_values, _summary_stats


def _resolve_section_path(config, value):
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    project_root = Path(config.get("_meta", {}).get("project_root", "."))
    return (project_root / path).resolve()


def _softmax_rows(values):
    import numpy as np

    values = np.asarray(values, dtype=float)
    shifted = values - np.max(values, axis=1, keepdims=True)
    exponentiated = np.exp(np.clip(shifted, -60.0, 60.0))
    denominator = exponentiated.sum(axis=1, keepdims=True)
    return exponentiated / np.clip(denominator, 1.0e-12, None)


def _normalise_probabilities(probabilities):
    import numpy as np

    probabilities = np.asarray(probabilities, dtype=float)
    probabilities = np.nan_to_num(
        probabilities,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
    probabilities = np.clip(probabilities, 0.0, None)
    row_sums = probabilities.sum(axis=-1, keepdims=True)
    empty = row_sums <= 1.0e-12
    if np.any(empty):
        probabilities = probabilities.copy()
        probabilities[empty.repeat(probabilities.shape[-1], axis=-1)] = 1.0
        row_sums = probabilities.sum(axis=-1, keepdims=True)
    return probabilities / np.clip(row_sums, 1.0e-12, None)


def _gate_feature_matrix(expert_probabilities):
    """Build pair-specific gate features from the experts' probability profiles."""

    import numpy as np

    probabilities = _normalise_probabilities(expert_probabilities)
    sample_count, view_count, class_count = probabilities.shape
    flat = probabilities.reshape(sample_count, view_count * class_count)
    entropy = -np.sum(
        probabilities * np.log(np.clip(probabilities, 1.0e-12, 1.0)),
        axis=2,
    )
    if class_count > 1:
        ordered = np.sort(probabilities, axis=2)
        margin = ordered[:, :, -1] - ordered[:, :, -2]
    else:
        margin = probabilities[:, :, 0]
    return np.concatenate([flat, entropy, margin], axis=1)


class SoftmaxRoutingGate:
    """A small, explicit mixture-of-experts gate trained with NumPy Adam."""

    def __init__(
        self,
        *,
        learning_rate=0.03,
        max_iter=1500,
        l2=1.0e-3,
        entropy_penalty=0.0,
        temperature=1.0,
        tolerance=1.0e-8,
        patience=80,
        balanced_loss=True,
        random_seed=42,
    ):
        self.learning_rate = float(learning_rate)
        self.max_iter = int(max_iter)
        self.l2 = float(l2)
        self.entropy_penalty = float(entropy_penalty)
        self.temperature = float(temperature)
        self.tolerance = float(tolerance)
        self.patience = int(patience)
        self.balanced_loss = bool(balanced_loss)
        self.random_seed = int(random_seed)
        self.feature_mean_ = None
        self.feature_scale_ = None
        self.coefficients_ = None
        self.loss_history_ = []
        self.n_iter_ = 0
        self.best_loss_ = math.nan

    def _prepare_features(self, expert_probabilities, fit):
        import numpy as np

        features = _gate_feature_matrix(expert_probabilities)
        if fit:
            self.feature_mean_ = features.mean(axis=0)
            self.feature_scale_ = features.std(axis=0)
            self.feature_scale_[self.feature_scale_ < 1.0e-8] = 1.0
        if self.feature_mean_ is None or self.feature_scale_ is None:
            raise RuntimeError("Routing gate has not been fitted")
        standardised = (features - self.feature_mean_) / self.feature_scale_
        return np.column_stack([np.ones(len(standardised)), standardised])

    def _weights_from_design(self, design, coefficients):
        logits = design @ coefficients.T
        return _softmax_rows(logits / max(self.temperature, 1.0e-6))

    def _loss_and_gradient(
        self,
        coefficients,
        design,
        expert_probabilities,
        labels,
        sample_weights,
    ):
        import numpy as np

        gate_weights = self._weights_from_design(design, coefficients)
        target_expert_probabilities = expert_probabilities[
            np.arange(len(labels)), :, labels
        ]
        target_mixture_probability = np.sum(
            gate_weights * target_expert_probabilities,
            axis=1,
        )
        target_mixture_probability = np.clip(
            target_mixture_probability,
            1.0e-12,
            1.0,
        )
        weight_sum = float(sample_weights.sum())
        loss = float(
            -np.sum(sample_weights * np.log(target_mixture_probability)) / weight_sum
        )

        derivative_logits = (
            gate_weights
            * (
                1.0
                - target_expert_probabilities
                / target_mixture_probability[:, None]
            )
            * sample_weights[:, None]
            / weight_sum
            / max(self.temperature, 1.0e-6)
        )

        if self.entropy_penalty:
            entropy = -np.sum(
                gate_weights
                * np.log(np.clip(gate_weights, 1.0e-12, 1.0)),
                axis=1,
            )
            loss += self.entropy_penalty * float(entropy.mean())
            derivative_logits += (
                self.entropy_penalty
                * (
                    -gate_weights
                    * (
                        np.log(np.clip(gate_weights, 1.0e-12, 1.0))
                        + entropy[:, None]
                    )
                )
                / float(len(labels))
                / max(self.temperature, 1.0e-6)
            )

        gradient = derivative_logits.T @ design
        if self.l2:
            loss += 0.5 * self.l2 * float(
                np.sum(coefficients[:, 1:] ** 2)
            )
            gradient[:, 1:] += self.l2 * coefficients[:, 1:]
        return loss, gradient

    def fit(self, expert_probabilities, labels):
        import numpy as np

        probabilities = _normalise_probabilities(expert_probabilities)
        labels = np.asarray(labels, dtype=int)
        if probabilities.ndim != 3:
            raise ValueError("Expected expert probabilities with shape N x V x C")
        if len(labels) != probabilities.shape[0]:
            raise ValueError("Routing labels and probability rows do not match")
        if probabilities.shape[1] < 2:
            raise ValueError("Relation routing requires at least two expert views")

        design = self._prepare_features(probabilities, fit=True)
        view_count = probabilities.shape[1]
        rng = np.random.RandomState(self.random_seed)
        coefficients = rng.normal(
            loc=0.0,
            scale=1.0e-4,
            size=(view_count, design.shape[1]),
        )

        if self.balanced_loss:
            class_counts = np.bincount(
                labels,
                minlength=probabilities.shape[2],
            ).astype(float)
            class_weights = np.zeros_like(class_counts)
            present = class_counts > 0
            class_weights[present] = len(labels) / (
                float(present.sum()) * class_counts[present]
            )
            sample_weights = class_weights[labels]
        else:
            sample_weights = np.ones(len(labels), dtype=float)

        first_moment = np.zeros_like(coefficients)
        second_moment = np.zeros_like(coefficients)
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1.0e-8
        best_loss = math.inf
        best_coefficients = coefficients.copy()
        stale = 0
        previous_loss = math.inf
        self.loss_history_ = []

        for iteration in range(1, self.max_iter + 1):
            loss, gradient = self._loss_and_gradient(
                coefficients,
                design,
                probabilities,
                labels,
                sample_weights,
            )
            self.loss_history_.append(float(loss))
            if loss < best_loss:
                best_loss = float(loss)
                best_coefficients = coefficients.copy()
            if abs(previous_loss - loss) <= self.tolerance:
                stale += 1
            else:
                stale = 0
            previous_loss = float(loss)

            first_moment = beta1 * first_moment + (1.0 - beta1) * gradient
            second_moment = (
                beta2 * second_moment + (1.0 - beta2) * (gradient ** 2)
            )
            corrected_first = first_moment / (1.0 - beta1 ** iteration)
            corrected_second = second_moment / (1.0 - beta2 ** iteration)
            coefficients -= (
                self.learning_rate
                * corrected_first
                / (np.sqrt(corrected_second) + epsilon)
            )
            if stale >= self.patience:
                break

        self.coefficients_ = best_coefficients
        self.n_iter_ = int(iteration)
        self.best_loss_ = float(best_loss)
        return self

    def predict_weights(self, expert_probabilities):
        if self.coefficients_ is None:
            raise RuntimeError("Routing gate has not been fitted")
        probabilities = _normalise_probabilities(expert_probabilities)
        design = self._prepare_features(probabilities, fit=False)
        return self._weights_from_design(design, self.coefficients_)

    def predict_proba(self, expert_probabilities):
        import numpy as np

        probabilities = _normalise_probabilities(expert_probabilities)
        weights = self.predict_weights(probabilities)
        mixture = np.einsum("nv,nvc->nc", weights, probabilities)
        return _normalise_probabilities(mixture)

    def diagnostics(self):
        return {
            "iterations": int(self.n_iter_),
            "best_training_loss": float(self.best_loss_),
            "initial_training_loss": (
                float(self.loss_history_[0]) if self.loss_history_ else math.nan
            ),
            "final_training_loss": (
                float(self.loss_history_[-1]) if self.loss_history_ else math.nan
            ),
        }


def _aligned_estimator_probabilities(estimator, matrix, class_count):
    import numpy as np

    if hasattr(estimator, "predict_proba"):
        raw = estimator.predict_proba(matrix)
    elif hasattr(estimator, "decision_function"):
        raw_scores = estimator.decision_function(matrix)
        if raw_scores.ndim == 1:
            raw_scores = np.column_stack([-raw_scores, raw_scores])
        raw = _softmax_rows(raw_scores)
    else:
        predictions = estimator.predict(matrix)
        raw = np.zeros((len(predictions), len(estimator.classes_)), dtype=float)
        class_to_column = {
            int(class_index): column
            for column, class_index in enumerate(estimator.classes_)
        }
        for row_index, predicted in enumerate(predictions):
            raw[row_index, class_to_column[int(predicted)]] = 1.0

    aligned = np.zeros((len(matrix), int(class_count)), dtype=float)
    for local_column, class_index in enumerate(estimator.classes_):
        aligned[:, int(class_index)] = raw[:, local_column]
    return _normalise_probabilities(aligned)


def _fit_estimator(template, matrix, labels):
    import numpy as np
    from sklearn.base import clone
    from sklearn.dummy import DummyClassifier

    unique = np.unique(labels)
    if len(unique) < 2:
        estimator = DummyClassifier(strategy="prior")
    else:
        estimator = clone(template)
    estimator.fit(matrix, labels)
    return estimator


def _expert_oof_probabilities(
    matrices,
    labels,
    drugs,
    targets,
    *,
    split_name,
    requested_folds,
    seed,
    classifier_template,
    class_count,
):
    import numpy as np

    view_names = list(matrices)
    labels = np.asarray(labels, dtype=int)
    folds = _fold_count(
        split_name,
        labels,
        drugs,
        targets,
        requested_folds,
    )
    if folds < 2:
        raise ValueError(
            "Inner routing cross-validation cannot support two folds for %s"
            % split_name
        )
    probabilities = np.full(
        (len(labels), len(view_names), class_count),
        np.nan,
        dtype=float,
    )
    coverage = np.zeros(len(labels), dtype=int)
    first_matrix = matrices[view_names[0]]
    for train_indices, test_indices in _iter_splits(
        split_name,
        folds,
        seed,
        first_matrix,
        labels,
        drugs,
        targets,
    ):
        coverage[test_indices] += 1
        for view_index, view_name in enumerate(view_names):
            estimator = _fit_estimator(
                classifier_template,
                matrices[view_name][train_indices],
                labels[train_indices],
            )
            probabilities[test_indices, view_index, :] = (
                _aligned_estimator_probabilities(
                    estimator,
                    matrices[view_name][test_indices],
                    class_count,
                )
            )
    if not np.all(coverage == 1):
        raise RuntimeError(
            "Inner OOF expert predictions must cover every sample exactly once"
        )
    if np.isnan(probabilities).any():
        raise RuntimeError("Inner OOF expert probability matrix is incomplete")
    return probabilities, int(folds)


def _expert_test_probabilities(
    matrices,
    labels,
    train_indices,
    test_indices,
    *,
    classifier_template,
    class_count,
):
    import numpy as np

    view_names = list(matrices)
    probabilities = np.zeros(
        (len(test_indices), len(view_names), class_count),
        dtype=float,
    )
    for view_index, view_name in enumerate(view_names):
        estimator = _fit_estimator(
            classifier_template,
            matrices[view_name][train_indices],
            labels[train_indices],
        )
        probabilities[:, view_index, :] = _aligned_estimator_probabilities(
            estimator,
            matrices[view_name][test_indices],
            class_count,
        )
    return probabilities


def _fit_stacking_classifier(expert_probabilities, labels, random_seed):
    from sklearn.dummy import DummyClassifier
    from sklearn.linear_model import LogisticRegression

    features = expert_probabilities.reshape(len(expert_probabilities), -1)
    if len(set(int(value) for value in labels)) < 2:
        estimator = DummyClassifier(strategy="prior")
    else:
        estimator = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=3000,
            random_state=int(random_seed),
        )
    estimator.fit(features, labels)
    return estimator


def _component_deletion_probabilities(expert_probabilities, gate_weights):
    """Return mixture probabilities after deleting each routed view component."""

    import numpy as np

    expert_probabilities = _normalise_probabilities(expert_probabilities)
    gate_weights = np.asarray(gate_weights, dtype=float)
    mixture = np.einsum("nv,nvc->nc", gate_weights, expert_probabilities)
    sample_count, view_count, class_count = expert_probabilities.shape
    deleted = np.zeros(
        (sample_count, view_count, class_count),
        dtype=float,
    )
    for view_index in range(view_count):
        denominator = 1.0 - gate_weights[:, view_index]
        numerator = (
            mixture
            - gate_weights[:, view_index, None]
            * expert_probabilities[:, view_index, :]
        )
        safe = denominator > 1.0e-8
        deleted[safe, view_index, :] = (
            numerator[safe] / denominator[safe, None]
        )
        if np.any(~safe):
            other_indices = [
                index for index in range(view_count) if index != view_index
            ]
            deleted[~safe, view_index, :] = expert_probabilities[
                ~safe
            ][:, other_indices, :].mean(axis=1)
    return _normalise_probabilities(deleted)


def _component_deletion_deltas(
    expert_probabilities,
    gate_weights,
    predicted_classes=None,
):
    import numpy as np

    mixture = np.einsum(
        "nv,nvc->nc",
        gate_weights,
        _normalise_probabilities(expert_probabilities),
    )
    mixture = _normalise_probabilities(mixture)
    if predicted_classes is None:
        predicted_classes = mixture.argmax(axis=1)
    predicted_classes = np.asarray(predicted_classes, dtype=int)
    deleted = _component_deletion_probabilities(
        expert_probabilities,
        gate_weights,
    )
    rows = np.arange(len(mixture))
    original = mixture[rows, predicted_classes]
    deleted_for_prediction = deleted[
        rows[:, None],
        np.arange(deleted.shape[1])[None, :],
        predicted_classes[:, None],
    ]
    return original[:, None] - deleted_for_prediction, deleted, mixture


def _rank_correlation(first, second):
    import numpy as np

    first_rank = pd.Series(first).rank(method="average").to_numpy(dtype=float)
    second_rank = pd.Series(second).rank(method="average").to_numpy(dtype=float)
    if len(first_rank) < 2:
        return math.nan
    if np.std(first_rank) <= 1.0e-12 or np.std(second_rank) <= 1.0e-12:
        return math.nan
    return float(np.corrcoef(first_rank, second_rank)[0, 1])


def _default_view_relation_map(config, view_names):
    mapping = {}
    configured_views = config.get("views", {})
    for view_name in view_names:
        groups = [
            str(group)
            for group in configured_views.get(view_name, [])
            if str(group) != "moa"
        ]
        non_target = [group for group in groups if group != "target"]
        mapping[view_name] = non_target or ["target"]
    return mapping


def _view_relation_map(config, routing_cfg, view_names):
    inferred = _default_view_relation_map(config, view_names)
    configured = routing_cfg.get("view_relation_map", {})
    for view_name, relations in configured.items():
        if view_name not in inferred:
            continue
        if isinstance(relations, str):
            relations = [relations]
        cleaned = [str(relation) for relation in relations if str(relation)]
        if cleaned:
            inferred[view_name] = cleaned
    return inferred


def _relation_attribution_for_case(
    view_names,
    view_relation_map,
    gate_weights,
    deletion_deltas,
):
    relation_rows = defaultdict(
        lambda: {
            "gate_weight": 0.0,
            "positive_deletion_delta": 0.0,
            "signed_deletion_delta": 0.0,
            "source_views": [],
        }
    )
    for view_index, view_name in enumerate(view_names):
        relations = list(view_relation_map.get(view_name, [view_name]))
        share = 1.0 / float(len(relations))
        for relation in relations:
            row = relation_rows[relation]
            row["gate_weight"] += float(gate_weights[view_index]) * share
            row["positive_deletion_delta"] += max(
                0.0,
                float(deletion_deltas[view_index]),
            ) * share
            row["signed_deletion_delta"] += (
                float(deletion_deltas[view_index]) * share
            )
            row["source_views"].append(view_name)
    return [
        {"relation": relation, **values}
        for relation, values in sorted(relation_rows.items())
    ]


def _common_pair_frame(pair_frame, sources):
    allowed = None
    for source in sources:
        names = set(source["entity_to_id"])
        source_pairs = {
            (str(row.drug_name), str(row.target_name), str(row.action))
            for row in pair_frame.itertuples(index=False)
            if str(row.drug_name) in names and str(row.target_name) in names
        }
        allowed = source_pairs if allowed is None else allowed & source_pairs
    if not allowed:
        return pair_frame.iloc[0:0].copy()
    mask = [
        (str(row.drug_name), str(row.target_name), str(row.action)) in allowed
        for row in pair_frame.itertuples(index=False)
    ]
    return pair_frame.loc[mask].reset_index(drop=True)


def _routing_feature_config(routing_cfg):
    """Resolve an explicit drug-only control without changing pair defaults.

    Row eligibility still requires both entity identifiers in every view so
    removing target-containing features cannot change the evaluated cohort.
    """

    feature_cfg = dict(routing_cfg)
    mode = str(feature_cfg.get("feature_mode", "pair"))
    if mode not in {"pair", "drug_only"}:
        raise ValueError("Unsupported relation-routing feature_mode: %s" % mode)
    if mode == "drug_only":
        feature_cfg.update({
            "include_drug_embedding": True,
            "include_target_embedding": False,
            "include_abs_difference": False,
            "include_hadamard_product": False,
        })
    return feature_cfg


def _build_aligned_matrices(sources, pair_frame, routing_cfg):
    feature_cfg = _routing_feature_config(routing_cfg)
    matrices = {}
    reference = None
    for source in sources:
        built = _make_pair_matrix(source, pair_frame, feature_cfg)
        if built is None:
            raise ValueError(
                "No common pair features for routing view %s" % source["view"]
            )
        matrix, labels, drugs, targets = built
        identity = (list(labels), list(drugs), list(targets))
        if reference is None:
            reference = identity
        elif identity != reference:
            raise RuntimeError(
                "Relation-view pair matrices are not aligned for %s"
                % source["view"]
            )
        matrices[source["view"]] = matrix
    if reference is None:
        raise ValueError("No relation-view feature matrices were built")
    return matrices, reference[0], reference[1], reference[2]


def _read_jsonl(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(str(path), "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            )


def _path_relation(path):
    path_type = str(path.get("path_type", "")).lower()
    for relation in ("pathway", "anatomy", "indication", "target"):
        if relation in path_type:
            return relation
    citations = path.get("citations", [])
    groups = {
        str(citation.get("source_group", "")).lower()
        for citation in citations
    }
    for relation in ("pathway", "anatomy", "indication", "target"):
        if relation in groups:
            return relation
    return "other"


def _guide_kag_evidence(
    bundles,
    routing_cases,
    *,
    score_floor=0.05,
):
    routing_by_pair = {
        (row["drug_name"], row["target_name"]): row
        for row in routing_cases
    }
    guided = []
    audit_rows = []
    for bundle in bundles:
        key = (str(bundle.get("drug_name")), str(bundle.get("target_name")))
        routing = routing_by_pair.get(key)
        if routing is None:
            continue
        relation_rows = routing["relation_attribution"]
        positive_total = sum(
            max(0.0, float(row["positive_deletion_delta"]))
            for row in relation_rows
        )
        if positive_total > 1.0e-12:
            relation_weights = {
                row["relation"]: max(
                    0.0,
                    float(row["positive_deletion_delta"]),
                )
                / positive_total
                for row in relation_rows
            }
            guidance_basis = "positive_view_component_deletion_delta"
        else:
            gate_total = sum(
                max(0.0, float(row["gate_weight"]))
                for row in relation_rows
            )
            relation_weights = {
                row["relation"]: (
                    max(0.0, float(row["gate_weight"])) / gate_total
                    if gate_total > 1.0e-12 else 0.0
                )
                for row in relation_rows
            }
            guidance_basis = "gate_weight_fallback"

        paths = [dict(path) for path in bundle.get("paths", [])]
        max_base_score = max(
            [float(path.get("score", 0.0)) for path in paths] or [1.0]
        )
        original_top = paths[0].get("path_id") if paths else None
        for path in paths:
            relation = _path_relation(path)
            base_score = float(path.get("score", 0.0))
            base_normalised = (
                base_score / max_base_score if max_base_score > 0.0 else 0.0
            )
            attribution_weight = float(relation_weights.get(relation, 0.0))
            path["base_retrieval_score"] = base_score
            path["relation_family"] = relation
            path["relation_attribution_weight"] = attribution_weight
            path["guided_score"] = base_normalised * (
                float(score_floor) + attribution_weight
            )
        paths.sort(
            key=lambda path: (
                -float(path["guided_score"]),
                -float(path["base_retrieval_score"]),
                str(path.get("path_id", "")),
            )
        )
        guided_top = paths[0].get("path_id") if paths else None
        prompt = {
            "system": (
                "Use only the supplied, source-cited graph evidence. The paths "
                "are ordered using relation sensitivity from a locked "
                "pair-conditioned routing model. Explain supporting, "
                "countervailing, and coverage-limited context separately. Cite "
                "path IDs for every biological statement. The deletion scores "
                "measure faithfulness to the routed mixture, not a causal drug "
                "mechanism. Do not infer inhibitor, agonist, antagonist, or "
                "blocker direction from unsigned paths."
            ),
            "question": bundle.get("question"),
            "predicted_action": routing["predicted_label"],
            "relation_attribution": relation_rows,
            "paths": paths,
            "context_edges": bundle.get("context_edges", []),
        }
        guided_bundle = {
            **bundle,
            "routing_split": routing["split"],
            "routing_repeat": routing["repeat"],
            "routing_fold": routing["fold"],
            "routing_predicted_action": routing["predicted_label"],
            "routing_prediction_scores": routing["prediction_scores"],
            "relation_attribution": relation_rows,
            "guidance_basis": guidance_basis,
            "paths": paths,
            "prompt": prompt,
            "claim_scope": (
                "Attribution-guided contextual evidence for the routed mixture; "
                "not a causal mechanism or proof of action direction."
            ),
        }
        guided.append(guided_bundle)
        audit_rows.append({
            "query_id": bundle.get("query_id"),
            "drug_name": key[0],
            "target_name": key[1],
            "path_count": len(paths),
            "guidance_basis": guidance_basis,
            "original_top_path_id": original_top,
            "guided_top_path_id": guided_top,
            "top_path_changed": original_top != guided_top,
            "top_relation": (
                paths[0]["relation_family"] if paths else None
            ),
            "top_guided_score": (
                paths[0]["guided_score"] if paths else 0.0
            ),
        })
    return guided, audit_rows


def _markdown_summary(summary_df, faithfulness_df, manifest):
    lines = [
        "# Pair-conditioned relation-routing results",
        "",
        "This run uses nested out-of-fold base-expert predictions to train the "
        "gate. Outer test pairs are not used to fit either the experts or the gate.",
        "",
        "## Cohort",
        "",
        "- Common pairs: %d" % manifest["common_pair_count"],
        "- Views: %s" % ", ".join(manifest["views"]),
        "- Embedding model: %s" % manifest["embedding_model"],
        "- Expert classifier: %s" % manifest["expert_classifier"],
        "- Feature mode: %s" % manifest.get("feature_mode", "pair"),
        "",
        "## Macro-F1",
        "",
        "| Task | Split | Strategy | Mean | 95% CI half-width |",
        "|---|---|---|---:|---:|",
    ]
    if not summary_df.empty:
        ordered = summary_df.sort_values(
            ["task", "split", "f1_macro_mean"],
            ascending=[True, True, False],
        )
        for row in ordered.itertuples(index=False):
            lines.append(
                "| %s | %s | %s | %.4f | %.4f |"
                % (
                    row.task,
                    row.split,
                    row.strategy,
                    row.f1_macro_mean,
                    row.f1_macro_ci95,
                )
            )
    lines.extend([
        "",
        "## Routing-component faithfulness",
        "",
        "Deletion removes one view expert from the mixture and renormalizes the "
        "remaining gate weights. It does not delete RGCN edges.",
        "",
        "| Task | Split | Top-view delta | Random-view delta | Gate/delta rank rho |",
        "|---|---|---:|---:|---:|",
    ])
    if not faithfulness_df.empty:
        for row in faithfulness_df.sort_values(
            ["task", "split", "repeat"]
        ).itertuples(index=False):
            lines.append(
                "| %s | %s | %.4f | %.4f | %.4f |"
                % (
                    row.task,
                    row.split,
                    row.mean_top_weight_deletion_delta,
                    row.mean_random_non_top_deletion_delta,
                    row.gate_weight_deletion_delta_spearman,
                )
            )
    lines.extend([
        "",
        "## Claim boundary",
        "",
        "Gate weights and component-deletion deltas explain the routed "
        "mixture-of-experts model. Attribution-guided KAG records retrieve "
        "typed contextual evidence; they do not prove a causal drug mechanism.",
        "",
    ])
    return "\n".join(lines)


def evaluate_relation_routing(config, logger):
    """Train and evaluate pair-conditioned relation-view routing."""

    import numpy as np
    from sklearn.preprocessing import LabelEncoder

    routing_cfg = dict(config.get("relation_routing", {}))
    if not routing_cfg:
        raise ValueError("Missing relation_routing config section")

    embedding_model = str(routing_cfg.get("embedding_model", "RGCN"))
    allowed_views = [
        str(view)
        for view in routing_cfg.get(
            "allowed_views",
            list(config.get("views", {})),
        )
    ]
    if len(allowed_views) < 2:
        raise ValueError("relation_routing.allowed_views requires at least two views")

    source_cfg = dict(routing_cfg)
    source_cfg["allowed_views"] = allowed_views
    source_cfg["allowed_models"] = [embedding_model]
    sources_by_view = {
        source["view"]: source
        for source in _embedding_sources(config, source_cfg, logger)
        if source["model"] == embedding_model
    }
    missing_views = [
        view for view in allowed_views if view not in sources_by_view
    ]
    if missing_views:
        raise FileNotFoundError(
            "Missing %s embeddings for routing views: %s"
            % (embedding_model, ", ".join(missing_views))
        )
    sources = [sources_by_view[view] for view in allowed_views]

    pair_cfg = dict(config.get("pair_evaluation", {}))
    pair_cfg.update(routing_cfg)
    pair_frame = _load_pair_table(config, pair_cfg)
    if pair_frame.empty:
        raise ValueError("No drug-target action pairs could be constructed")

    output_dir = ensure_dir(
        _resolve_section_path(
            config,
            routing_cfg.get(
                "output_dir",
                Path(config["paths"]["reports_dir"]) / "relation_routing",
            ),
        )
    )
    view_relation_map = _view_relation_map(
        config,
        routing_cfg,
        allowed_views,
    )
    splits = [
        str(split)
        for split in routing_cfg.get(
            "splits",
            ["random_pair", "drug_disjoint", "target_disjoint", "double_disjoint"],
        )
    ]
    repeats = int(routing_cfg.get("cv_repeats", 1))
    requested_folds = int(routing_cfg.get("cv_folds", 5))
    inner_requested_folds = int(routing_cfg.get("gate_inner_folds", 3))
    expert_classifier_name = str(
        routing_cfg.get("expert_classifier", "Random Forest")
    )
    classifier_cfg = dict(routing_cfg)
    classifier_cfg["classifiers"] = [expert_classifier_name]
    classifier_templates = _build_pair_classifiers(
        classifier_cfg,
        int(config.get("random_seed", 42)),
    )
    if expert_classifier_name not in classifier_templates:
        raise ValueError(
            "Unsupported relation-routing expert classifier: %s"
            % expert_classifier_name
        )
    classifier_template = classifier_templates[expert_classifier_name]

    repeat_rows = []
    prediction_rows = []
    attribution_rows = []
    faithfulness_rows = []
    gate_diagnostic_rows = []
    routing_cases = []
    common_counts = []

    for task_name, task_cfg in _pair_label_tasks(pair_cfg).items():
        task_pairs = _filter_pairs_for_task(pair_frame, task_cfg)
        common_pairs = _common_pair_frame(task_pairs, sources)
        if common_pairs.empty:
            logger.warning(
                "Relation-routing task %s has no common pairs across views",
                task_name,
            )
            continue
        matrices, labels, drugs, targets = _build_aligned_matrices(
            sources,
            common_pairs,
            pair_cfg,
        )
        common_counts.append(len(labels))
        label_encoder = LabelEncoder()
        encoded = label_encoder.fit_transform(labels)
        class_count = len(label_encoder.classes_)
        if class_count < 2:
            logger.warning(
                "Relation-routing task %s has fewer than two classes",
                task_name,
            )
            continue
        logger.info(
            "Relation routing task=%s pairs=%d views=%d actions=%d",
            task_name,
            len(labels),
            len(allowed_views),
            class_count,
        )

        for split_name in splits:
            outer_folds = _fold_count(
                split_name,
                labels,
                drugs,
                targets,
                requested_folds,
            )
            if outer_folds < 2:
                logger.warning(
                    "Skipping routing task=%s split=%s because fewer than two folds are possible",
                    task_name,
                    split_name,
                )
                continue
            for repeat in range(repeats):
                seed = int(config.get("random_seed", 42)) + repeat
                strategy_scores = {
                    "uniform_average": np.full(
                        (len(encoded), class_count),
                        np.nan,
                    ),
                    "best_inner_view": np.full(
                        (len(encoded), class_count),
                        np.nan,
                    ),
                    "ordinary_stacking": np.full(
                        (len(encoded), class_count),
                        np.nan,
                    ),
                    "pair_conditioned_gate": np.full(
                        (len(encoded), class_count),
                        np.nan,
                    ),
                }
                for view_name in allowed_views:
                    strategy_scores["expert::%s" % view_name] = np.full(
                        (len(encoded), class_count),
                        np.nan,
                    )
                fold_ids = np.full(len(encoded), -1, dtype=int)
                selected_views = np.full(len(encoded), "", dtype=object)
                all_gate_weights = np.full(
                    (len(encoded), len(allowed_views)),
                    np.nan,
                )
                all_expert_probabilities = np.full(
                    (len(encoded), len(allowed_views), class_count),
                    np.nan,
                )
                all_deletion_deltas = np.full(
                    (len(encoded), len(allowed_views)),
                    np.nan,
                )

                split_iterator = _iter_splits(
                    split_name,
                    outer_folds,
                    seed,
                    matrices[allowed_views[0]],
                    encoded,
                    drugs,
                    targets,
                )
                for fold, (train_indices, test_indices) in enumerate(split_iterator):
                    logger.info(
                        "Routing outer fold task=%s split=%s repeat=%d fold=%d train=%d test=%d",
                        task_name,
                        split_name,
                        repeat,
                        fold,
                        len(train_indices),
                        len(test_indices),
                    )
                    train_matrices = {
                        view: matrix[train_indices]
                        for view, matrix in matrices.items()
                    }
                    inner_probabilities, inner_folds = _expert_oof_probabilities(
                        train_matrices,
                        encoded[train_indices],
                        np.asarray(drugs, dtype=object)[train_indices],
                        np.asarray(targets, dtype=object)[train_indices],
                        split_name=split_name,
                        requested_folds=inner_requested_folds,
                        seed=seed + 1009 + fold,
                        classifier_template=classifier_template,
                        class_count=class_count,
                    )
                    test_probabilities = _expert_test_probabilities(
                        matrices,
                        encoded,
                        train_indices,
                        test_indices,
                        classifier_template=classifier_template,
                        class_count=class_count,
                    )

                    gate = SoftmaxRoutingGate(
                        learning_rate=float(
                            routing_cfg.get("gate_learning_rate", 0.03)
                        ),
                        max_iter=int(routing_cfg.get("gate_max_iter", 1500)),
                        l2=float(routing_cfg.get("gate_l2", 1.0e-3)),
                        entropy_penalty=float(
                            routing_cfg.get("gate_entropy_penalty", 0.0)
                        ),
                        temperature=float(
                            routing_cfg.get("gate_temperature", 1.0)
                        ),
                        tolerance=float(
                            routing_cfg.get("gate_tolerance", 1.0e-8)
                        ),
                        patience=int(routing_cfg.get("gate_patience", 80)),
                        balanced_loss=bool(
                            routing_cfg.get("gate_balanced_loss", True)
                        ),
                        random_seed=seed + fold,
                    ).fit(inner_probabilities, encoded[train_indices])
                    gate_weights = gate.predict_weights(test_probabilities)
                    gate_probabilities = gate.predict_proba(test_probabilities)
                    deletion_deltas, _, _ = _component_deletion_deltas(
                        test_probabilities,
                        gate_weights,
                    )

                    inner_view_metrics = []
                    for view_index, view_name in enumerate(allowed_views):
                        inner_prediction = inner_probabilities[
                            :, view_index, :
                        ].argmax(axis=1)
                        metric = _metric_values(
                            encoded[train_indices],
                            inner_prediction,
                            inner_probabilities[:, view_index, :],
                            class_count,
                        )["f1_macro"]
                        inner_view_metrics.append(
                            (float(metric), view_name, view_index)
                        )
                    _, best_view_name, best_view_index = sorted(
                        inner_view_metrics,
                        key=lambda row: (-row[0], row[1]),
                    )[0]

                    stacker = _fit_stacking_classifier(
                        inner_probabilities,
                        encoded[train_indices],
                        seed + fold,
                    )
                    stacking_probabilities = _aligned_estimator_probabilities(
                        stacker,
                        test_probabilities.reshape(len(test_indices), -1),
                        class_count,
                    )

                    strategy_scores["uniform_average"][test_indices] = (
                        test_probabilities.mean(axis=1)
                    )
                    strategy_scores["best_inner_view"][test_indices] = (
                        test_probabilities[:, best_view_index, :]
                    )
                    strategy_scores["ordinary_stacking"][test_indices] = (
                        stacking_probabilities
                    )
                    strategy_scores["pair_conditioned_gate"][test_indices] = (
                        gate_probabilities
                    )
                    for view_index, view_name in enumerate(allowed_views):
                        strategy_scores["expert::%s" % view_name][
                            test_indices
                        ] = test_probabilities[:, view_index, :]
                    fold_ids[test_indices] = int(fold)
                    selected_views[test_indices] = best_view_name
                    all_gate_weights[test_indices] = gate_weights
                    all_expert_probabilities[test_indices] = test_probabilities
                    all_deletion_deltas[test_indices] = deletion_deltas
                    gate_diagnostic_rows.append({
                        "task": task_name,
                        "split": split_name,
                        "repeat": repeat,
                        "seed": seed,
                        "fold": fold,
                        "train_count": len(train_indices),
                        "test_count": len(test_indices),
                        "inner_entity_partitions": inner_folds,
                        "selected_best_inner_view": best_view_name,
                        **gate.diagnostics(),
                    })

                if (fold_ids < 0).any():
                    raise RuntimeError(
                        "Outer routing predictions did not cover every pair"
                    )
                for strategy, scores in strategy_scores.items():
                    if np.isnan(scores).any():
                        raise RuntimeError(
                            "Incomplete routing scores for strategy %s"
                            % strategy
                        )
                    scores = _normalise_probabilities(scores)
                    predicted = scores.argmax(axis=1)
                    metrics = _metric_values(
                        encoded,
                        predicted,
                        scores,
                        class_count,
                    )
                    repeat_rows.append({
                        "task": task_name,
                        "cohort": str(
                            routing_cfg.get("cohort_name", "common_view_pairs")
                        ),
                        "split": split_name,
                        "strategy": strategy,
                        "embedding_model": embedding_model,
                        "expert_classifier": expert_classifier_name,
                        "repeat": repeat,
                        "seed": seed,
                        "pair_count": len(encoded),
                        "drug_count": len(set(drugs)),
                        "target_count": len(set(targets)),
                        "action_count": class_count,
                        "entity_partitions": outer_folds,
                        "outer_fold_count": (
                            outer_folds ** 2
                            if split_name == "double_disjoint"
                            else outer_folds
                        ),
                        **metrics,
                    })
                    for sample_index in range(len(encoded)):
                        row = {
                            "task": task_name,
                            "cohort": str(
                                routing_cfg.get(
                                    "cohort_name",
                                    "common_view_pairs",
                                )
                            ),
                            "split": split_name,
                            "strategy": strategy,
                            "repeat": repeat,
                            "seed": seed,
                            "fold": int(fold_ids[sample_index]),
                            "drug_name": drugs[sample_index],
                            "target_name": targets[sample_index],
                            "true_label": labels[sample_index],
                            "predicted_label": str(
                                label_encoder.classes_[
                                    int(predicted[sample_index])
                                ]
                            ),
                        }
                        for class_index, class_name in enumerate(
                            label_encoder.classes_
                        ):
                            row["score__%s" % class_name] = float(
                                scores[sample_index, class_index]
                            )
                        if strategy == "best_inner_view":
                            row["selected_view"] = str(
                                selected_views[sample_index]
                            )
                        prediction_rows.append(row)

                routed_scores = _normalise_probabilities(
                    strategy_scores["pair_conditioned_gate"]
                )
                routed_predictions = routed_scores.argmax(axis=1)
                top_indices = all_gate_weights.argmax(axis=1)
                low_indices = all_gate_weights.argmin(axis=1)
                random_indices = np.zeros(len(encoded), dtype=int)
                rng = np.random.RandomState(seed + 8081)
                for sample_index, top_index in enumerate(top_indices):
                    candidates = [
                        index
                        for index in range(len(allowed_views))
                        if index != int(top_index)
                    ]
                    random_indices[sample_index] = int(rng.choice(candidates))
                row_indices = np.arange(len(encoded))
                top_deltas = all_deletion_deltas[row_indices, top_indices]
                low_deltas = all_deletion_deltas[row_indices, low_indices]
                random_deltas = all_deletion_deltas[
                    row_indices,
                    random_indices,
                ]
                faithfulness_rows.append({
                    "task": task_name,
                    "split": split_name,
                    "repeat": repeat,
                    "seed": seed,
                    "pair_count": len(encoded),
                    "mean_top_weight_deletion_delta": float(top_deltas.mean()),
                    "median_top_weight_deletion_delta": float(
                        np.median(top_deltas)
                    ),
                    "top_weight_positive_delta_rate": float(
                        np.mean(top_deltas > 0.0)
                    ),
                    "mean_random_non_top_deletion_delta": float(
                        random_deltas.mean()
                    ),
                    "mean_lowest_weight_deletion_delta": float(
                        low_deltas.mean()
                    ),
                    "top_minus_random_delta": float(
                        top_deltas.mean() - random_deltas.mean()
                    ),
                    "top_minus_lowest_delta": float(
                        top_deltas.mean() - low_deltas.mean()
                    ),
                    "gate_weight_deletion_delta_spearman": _rank_correlation(
                        all_gate_weights.reshape(-1),
                        all_deletion_deltas.reshape(-1),
                    ),
                    "claim_scope": (
                        "View-component deletion in the routed mixture; "
                        "not RGCN edge-level attribution."
                    ),
                })

                for sample_index in range(len(encoded)):
                    relation_attribution = _relation_attribution_for_case(
                        allowed_views,
                        view_relation_map,
                        all_gate_weights[sample_index],
                        all_deletion_deltas[sample_index],
                    )
                    case_views = []
                    for view_index, view_name in enumerate(allowed_views):
                        case_views.append({
                            "view": view_name,
                            "relation_families": view_relation_map[view_name],
                            "gate_weight": float(
                                all_gate_weights[sample_index, view_index]
                            ),
                            "view_component_deletion_delta": float(
                                all_deletion_deltas[
                                    sample_index,
                                    view_index,
                                ]
                            ),
                            "expert_prediction_scores": {
                                str(class_name): float(
                                    all_expert_probabilities[
                                        sample_index,
                                        view_index,
                                        class_index,
                                    ]
                                )
                                for class_index, class_name in enumerate(
                                    label_encoder.classes_
                                )
                            },
                        })
                        attribution_rows.append({
                            "task": task_name,
                            "split": split_name,
                            "repeat": repeat,
                            "seed": seed,
                            "fold": int(fold_ids[sample_index]),
                            "drug_name": drugs[sample_index],
                            "target_name": targets[sample_index],
                            "true_label": labels[sample_index],
                            "predicted_label": str(
                                label_encoder.classes_[
                                    int(routed_predictions[sample_index])
                                ]
                            ),
                            "view": view_name,
                            "relation_families": "|".join(
                                view_relation_map[view_name]
                            ),
                            "gate_weight": float(
                                all_gate_weights[sample_index, view_index]
                            ),
                            "view_component_deletion_delta": float(
                                all_deletion_deltas[
                                    sample_index,
                                    view_index,
                                ]
                            ),
                        })
                    routing_cases.append({
                        "task": task_name,
                        "split": split_name,
                        "repeat": repeat,
                        "seed": seed,
                        "fold": int(fold_ids[sample_index]),
                        "drug_name": drugs[sample_index],
                        "target_name": targets[sample_index],
                        "true_label": labels[sample_index],
                        "predicted_label": str(
                            label_encoder.classes_[
                                int(routed_predictions[sample_index])
                            ]
                        ),
                        "prediction_scores": {
                            str(class_name): float(
                                routed_scores[
                                    sample_index,
                                    class_index,
                                ]
                            )
                            for class_index, class_name in enumerate(
                                label_encoder.classes_
                            )
                        },
                        "views": case_views,
                        "relation_attribution": relation_attribution,
                        "claim_scope": (
                            "Faithfulness to the pair-conditioned routed mixture; "
                            "not a causal drug mechanism."
                        ),
                    })

    if not repeat_rows:
        raise ValueError("Relation routing produced no evaluation results")

    repeat_df = pd.DataFrame(repeat_rows)
    summary_rows = []
    group_columns = [
        "task",
        "cohort",
        "split",
        "strategy",
        "embedding_model",
        "expert_classifier",
    ]
    for keys, group in repeat_df.groupby(group_columns, sort=True):
        row = dict(zip(group_columns, keys))
        row.update({
            "cv_repeats": int(group["repeat"].nunique()),
            "pair_count": int(group["pair_count"].iloc[0]),
            "drug_count": int(group["drug_count"].iloc[0]),
            "target_count": int(group["target_count"].iloc[0]),
            "action_count": int(group["action_count"].iloc[0]),
        })
        for metric_name in METRIC_NAMES:
            mean, std, ci95 = _summary_stats(group[metric_name])
            row["%s_mean" % metric_name] = mean
            row["%s_std" % metric_name] = std
            row["%s_ci95" % metric_name] = ci95
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    prediction_df = pd.DataFrame(prediction_rows)
    attribution_df = pd.DataFrame(attribution_rows)
    faithfulness_df = pd.DataFrame(faithfulness_rows)
    gate_diagnostics_df = pd.DataFrame(gate_diagnostic_rows)

    summary_df.to_csv(output_dir / "routing_summary.csv", index=False)
    repeat_df.to_csv(output_dir / "routing_repeat_metrics.csv", index=False)
    prediction_df.to_csv(output_dir / "routing_oof_predictions.csv", index=False)
    attribution_df.to_csv(
        output_dir / "routing_view_attribution.csv",
        index=False,
    )
    faithfulness_df.to_csv(
        output_dir / "routing_faithfulness.csv",
        index=False,
    )
    gate_diagnostics_df.to_csv(
        output_dir / "routing_gate_diagnostics.csv",
        index=False,
    )
    _write_jsonl(output_dir / "routing_cases.jsonl", routing_cases)

    kag_summary = None
    kag_questions_path = routing_cfg.get("kag_questions_path")
    if kag_questions_path:
        kag_path = _resolve_section_path(config, kag_questions_path)
        if not kag_path.exists():
            raise FileNotFoundError(
                "KAG questions for routing guidance not found: %s" % kag_path
            )
        selected_split = str(
            routing_cfg.get("kag_guidance_split", "double_disjoint")
        )
        selected_repeat = int(routing_cfg.get("kag_guidance_repeat", 0))
        selected_cases = [
            row
            for row in routing_cases
            if row["split"] == selected_split
            and int(row["repeat"]) == selected_repeat
        ]
        guided_bundles, kag_audit_rows = _guide_kag_evidence(
            list(_read_jsonl(kag_path)),
            selected_cases,
            score_floor=float(
                routing_cfg.get("kag_relation_score_floor", 0.05)
            ),
        )
        kag_dir = ensure_dir(output_dir / "attribution_guided_kag")
        _write_jsonl(
            kag_dir / "guided_questions.jsonl",
            guided_bundles,
        )
        _write_jsonl(
            kag_dir / "guided_prompt_records.jsonl",
            (
                {"query_id": row["query_id"], **row["prompt"]}
                for row in guided_bundles
            ),
        )
        kag_audit_df = pd.DataFrame(kag_audit_rows)
        kag_audit_df.to_csv(
            kag_dir / "guidance_audit.csv",
            index=False,
        )
        kag_summary = {
            "requested_split": selected_split,
            "requested_repeat": selected_repeat,
            "routing_case_count": len(selected_cases),
            "guided_query_count": len(guided_bundles),
            "query_coverage": (
                len(guided_bundles) / float(len(selected_cases))
                if selected_cases else 0.0
            ),
            "queries_with_paths": sum(
                bool(row.get("paths")) for row in guided_bundles
            ),
            "top_path_change_rate": (
                float(kag_audit_df["top_path_changed"].mean())
                if not kag_audit_df.empty else 0.0
            ),
            "guidance_basis_counts": (
                kag_audit_df["guidance_basis"].value_counts().to_dict()
                if not kag_audit_df.empty else {}
            ),
            "claim_scope": (
                "Attribution-guided contextual evidence; the LLM is not used "
                "to determine routing weights or evidence scores."
            ),
        }
        write_json(kag_dir / "guidance_summary.json", kag_summary)

    manifest = {
        "method": "pair_conditioned_relation_routing",
        "cohort": str(routing_cfg.get("cohort_name", "common_view_pairs")),
        "common_pair_count": int(min(common_counts) if common_counts else 0),
        "views": allowed_views,
        "view_relation_map": view_relation_map,
        "embedding_model": embedding_model,
        "expert_classifier": expert_classifier_name,
        "feature_mode": str(routing_cfg.get("feature_mode", "pair")),
        "feature_components": {
            name: bool(_routing_feature_config(pair_cfg).get(name, True))
            for name in (
                "include_drug_embedding", "include_target_embedding",
                "include_abs_difference", "include_hadamard_product",
            )
        },
        "splits": splits,
        "cv_folds": requested_folds,
        "cv_repeats": repeats,
        "gate_inner_folds": inner_requested_folds,
        "nested_oof_gate_training": True,
        "outer_test_used_for_gate_training": False,
        "view_component_deletion": True,
        "rgcn_edge_level_attribution": False,
        "kag_guidance": kag_summary,
        "claim_scope": (
            "Pair-specific routing and view-component faithfulness. KAG evidence "
            "is typed and grounded but is not causal pharmacological proof."
        ),
        "outputs": {
            "summary": "routing_summary.csv",
            "repeat_metrics": "routing_repeat_metrics.csv",
            "oof_predictions": "routing_oof_predictions.csv",
            "view_attribution": "routing_view_attribution.csv",
            "faithfulness": "routing_faithfulness.csv",
            "gate_diagnostics": "routing_gate_diagnostics.csv",
            "cases": "routing_cases.jsonl",
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    (output_dir / "routing_summary.md").write_text(
        _markdown_summary(summary_df, faithfulness_df, manifest),
        encoding="utf-8",
    )
    logger.info(
        "Relation routing wrote %d summary rows, %d OOF predictions, and %d attribution rows to %s",
        len(summary_df),
        len(prediction_df),
        len(attribution_df),
        output_dir,
    )
    return summary_df
