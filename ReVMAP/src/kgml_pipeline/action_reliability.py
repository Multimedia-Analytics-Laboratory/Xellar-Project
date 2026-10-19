"""Reliability and evidence auditing for the locked four-class action model.

The analysis in this module is deliberately post-lock.  Temperature scaling
and conformal thresholds are learned from out-of-fold predictions on the
cpg0004-excluded development cohort and then applied unchanged to cpg0004.
The predicted class is therefore not altered by calibration. Source-isolated
DrugBank and Open Targets action records are used only for
post-prediction evidence auditing, never as model features or training labels.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)

from .io_utils import ensure_dir, write_json


DEFAULT_CLASSES = ("agonist", "antagonist", "blocker", "inhibitor")
INHIBITORY_ACTIONS = frozenset(("antagonist", "blocker", "inhibitor"))
INDEPENDENT_SOURCES = frozenset(("DrugBank", "Open Targets"))
EXPERT_PREFIX = "expert::"

# These priors encode the evidential scope of the three prespecified path
# templates.  A shared pathway or indication/cotarget route can motivate a
# relation-specific hypothesis.  Shared expression in an anatomical site is
# retained, but is deliberately down-weighted and labelled as broad context:
# co-expression alone is usually too nonspecific to explain a drug--target
# relation.  The priors affect KAG ordering only and never the classifier.
KAG_PATH_TYPE_PRIOR = {
    "shared_pathway": 1.00,
    "shared_indication_cotarget": 0.95,
    "shared_anatomy": 0.55,
}
KAG_PATH_EVIDENCE_TIER = {
    "shared_pathway": "relation_specific_hypothesis",
    "shared_indication_cotarget": "relation_specific_hypothesis",
    "shared_anatomy": "broad_context_only",
}


def _normalise_name(value):
    return " ".join(str(value).strip().lower().split())


def _pair_key(drug_name, target_name):
    return "%s||%s" % (_normalise_name(drug_name), _normalise_name(target_name))


def _stable_fraction(key):
    digest = hashlib.sha256(str(key).encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12 - 1)


def _file_fingerprint(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path),
        "size_bytes": int(path.stat().st_size),
        "sha256": digest.hexdigest(),
    }


def _configured_input_fingerprints(section):
    candidates = {
        "development_oof_predictions": section.get("development_oof_predictions"),
        "development_double_oof_predictions": section.get(
            "development_double_oof_predictions"
        ),
        "external_oof_predictions": section.get("external_oof_predictions"),
        "external_view_attribution": section.get("external_view_attribution"),
        "external_routing_cases": section.get("external_routing_cases"),
        "pair_action_evidence": section.get("pair_action_evidence"),
        "kag_question_records": section.get("kag_question_records"),
    }
    for group_name in ("routing_repeat_metrics", "drug_only_repeat_metrics"):
        for cohort, path in (section.get(group_name) or {}).items():
            candidates["%s__%s" % (group_name, cohort)] = path
    return {
        name: _file_fingerprint(path)
        for name, path in sorted(candidates.items())
        if path and Path(path).exists()
    }


def _normalise_probabilities(values):
    values = np.asarray(values, dtype=float)
    values = np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=0.0)
    values = np.clip(values, 1.0e-12, None)
    return values / np.clip(values.sum(axis=1, keepdims=True), 1.0e-12, None)


def _temperature_scale(probabilities, temperature):
    probabilities = _normalise_probabilities(probabilities)
    logits = np.log(np.clip(probabilities, 1.0e-12, 1.0))
    logits = logits / max(float(temperature), 1.0e-6)
    logits -= logits.max(axis=1, keepdims=True)
    scaled = np.exp(np.clip(logits, -60.0, 60.0))
    return scaled / np.clip(scaled.sum(axis=1, keepdims=True), 1.0e-12, None)


def fit_temperature(probabilities, labels, class_names=DEFAULT_CLASSES):
    """Fit a scalar temperature by deterministic one-dimensional search."""

    probabilities = _normalise_probabilities(probabilities)
    label_to_index = {label: i for i, label in enumerate(class_names)}
    indices = np.asarray([label_to_index[str(label)] for label in labels], dtype=int)

    def objective(log_temperature):
        scaled = _temperature_scale(probabilities, math.exp(log_temperature))
        target = np.clip(scaled[np.arange(len(indices)), indices], 1.0e-12, 1.0)
        return float(-np.mean(np.log(target)))

    low = math.log(0.05)
    high = math.log(8.0)
    grid = np.linspace(low, high, 161)
    losses = np.asarray([objective(value) for value in grid])
    best = int(np.argmin(losses))
    left = grid[max(0, best - 1)]
    right = grid[min(len(grid) - 1, best + 1)]
    golden = (math.sqrt(5.0) - 1.0) / 2.0
    x1 = right - golden * (right - left)
    x2 = left + golden * (right - left)
    f1 = objective(x1)
    f2 = objective(x2)
    for _ in range(80):
        if f1 <= f2:
            right, x2, f2 = x2, x1, f1
            x1 = right - golden * (right - left)
            f1 = objective(x1)
        else:
            left, x1, f1 = x1, x2, f2
            x2 = left + golden * (right - left)
            f2 = objective(x2)
    return float(math.exp((left + right) / 2.0))


def _expected_calibration_error(probabilities, labels, class_names, bins=10):
    probabilities = _normalise_probabilities(probabilities)
    predicted = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    label_to_index = {label: i for i, label in enumerate(class_names)}
    truth = np.asarray([label_to_index[str(label)] for label in labels], dtype=int)
    correct = (predicted == truth).astype(float)
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    ece = 0.0
    for index in range(len(edges) - 1):
        if index == len(edges) - 2:
            mask = (confidence >= edges[index]) & (confidence <= edges[index + 1])
        else:
            mask = (confidence >= edges[index]) & (confidence < edges[index + 1])
        if mask.any():
            ece += float(mask.mean()) * abs(
                float(correct[mask].mean()) - float(confidence[mask].mean())
            )
    return float(ece)


def _classwise_ece(probabilities, labels, class_names, bins=10):
    probabilities = _normalise_probabilities(probabilities)
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    values = []
    for class_index, class_name in enumerate(class_names):
        truth = np.asarray([str(label) == class_name for label in labels], dtype=float)
        confidence = probabilities[:, class_index]
        ece = 0.0
        for index in range(len(edges) - 1):
            if index == len(edges) - 2:
                mask = (confidence >= edges[index]) & (confidence <= edges[index + 1])
            else:
                mask = (confidence >= edges[index]) & (confidence < edges[index + 1])
            if mask.any():
                ece += float(mask.mean()) * abs(
                    float(truth[mask].mean()) - float(confidence[mask].mean())
                )
        values.append(float(ece))
    return values


def _probability_metrics(probabilities, labels, class_names, bins=10):
    probabilities = _normalise_probabilities(probabilities)
    label_to_index = {label: i for i, label in enumerate(class_names)}
    truth = np.asarray([label_to_index[str(label)] for label in labels], dtype=int)
    one_hot = np.eye(len(class_names), dtype=float)[truth]
    classwise = _classwise_ece(probabilities, labels, class_names, bins=bins)
    return {
        "nll": float(log_loss(truth, probabilities, labels=list(range(len(class_names))))),
        "multiclass_brier": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "ece": _expected_calibration_error(probabilities, labels, class_names, bins=bins),
        "classwise_ece_mean": float(np.mean(classwise)),
        "classwise_ece_max": float(np.max(classwise)),
    }


def _polarity(label):
    return "activating" if str(label) == "agonist" else "inhibitory"


def _hierarchical_metrics(true_labels, predicted_labels):
    true_labels = np.asarray(true_labels, dtype=str)
    predicted_labels = np.asarray(predicted_labels, dtype=str)
    true_polarity = np.asarray([_polarity(value) for value in true_labels])
    predicted_polarity = np.asarray([_polarity(value) for value in predicted_labels])
    exact = true_labels == predicted_labels
    same_polarity = true_polarity == predicted_polarity
    inhibitory_truth = np.isin(true_labels, list(INHIBITORY_ACTIONS))
    inhibitory_subtype_error = inhibitory_truth & same_polarity & (~exact)
    severe = ~same_polarity
    credit = np.where(exact, 1.0, np.where(same_polarity, 0.5, 0.0))
    return {
        "four_class_accuracy": float(accuracy_score(true_labels, predicted_labels)),
        "four_class_balanced_accuracy": float(
            balanced_accuracy_score(true_labels, predicted_labels)
        ),
        "four_class_macro_f1": float(
            f1_score(true_labels, predicted_labels, average="macro", zero_division=0)
        ),
        "polarity_accuracy": float(accuracy_score(true_polarity, predicted_polarity)),
        "polarity_macro_f1": float(
            f1_score(true_polarity, predicted_polarity, average="macro", zero_division=0)
        ),
        "severe_direction_error_rate": float(severe.mean()),
        "inhibitory_subtype_error_rate_all": float(inhibitory_subtype_error.mean()),
        "inhibitory_subtype_error_rate_within_inhibitory_truth": float(
            inhibitory_subtype_error[inhibitory_truth].mean()
            if inhibitory_truth.any()
            else math.nan
        ),
        "hierarchical_credit": float(credit.mean()),
    }


def _risk_coverage_rows(probabilities, labels, class_names, coverages):
    probabilities = _normalise_probabilities(probabilities)
    predicted_indices = probabilities.argmax(axis=1)
    predicted = np.asarray([class_names[index] for index in predicted_indices])
    truth = np.asarray(labels, dtype=str)
    confidence = probabilities.max(axis=1)
    uncertainty = 1.0 - confidence
    order = np.argsort(-confidence, kind="mergesort")
    ordered_errors = (predicted[order] != truth[order]).astype(float)
    cumulative_risk = np.cumsum(ordered_errors) / np.arange(1, len(order) + 1)
    coverage_grid = np.arange(1, len(order) + 1, dtype=float) / len(order)
    aurc = float(np.trapz(cumulative_risk, coverage_grid))
    try:
        error_auroc = float(roc_auc_score((predicted != truth).astype(int), uncertainty))
    except ValueError:
        error_auroc = math.nan

    rows = []
    for coverage in coverages:
        keep_count = max(1, int(math.ceil(float(coverage) * len(order))))
        keep = order[:keep_count]
        reject = order[keep_count:]
        retained_error = float((predicted[keep] != truth[keep]).mean())
        rejected_error = (
            float((predicted[reject] != truth[reject]).mean()) if len(reject) else math.nan
        )
        rows.append({
            "coverage": float(keep_count / len(order)),
            "retained_count": int(keep_count),
            "macro_f1": float(
                f1_score(truth[keep], predicted[keep], average="macro", zero_division=0)
            ),
            "accuracy": float(accuracy_score(truth[keep], predicted[keep])),
            "retained_error_rate": retained_error,
            "rejected_error_rate": rejected_error,
            "rejected_to_retained_error_ratio": (
                rejected_error / retained_error
                if len(reject) and retained_error > 0
                else math.nan
            ),
            "aurc": aurc,
            "error_detection_auroc": error_auroc,
        })
    return rows


def _conformal_quantile(probabilities, labels, class_names, alpha):
    label_to_index = {label: i for i, label in enumerate(class_names)}
    truth = np.asarray([label_to_index[str(label)] for label in labels], dtype=int)
    scores = 1.0 - probabilities[np.arange(len(truth)), truth]
    rank = int(math.ceil((len(scores) + 1) * (1.0 - float(alpha))))
    rank = min(max(rank, 1), len(scores))
    return float(np.sort(scores)[rank - 1])


def _evaluate_prediction_sets(probabilities, labels, class_names, threshold):
    truth_indices = np.asarray(
        [{label: i for i, label in enumerate(class_names)}[str(value)] for value in labels],
        dtype=int,
    )
    included = (1.0 - probabilities) <= float(threshold)
    sizes = included.sum(axis=1)
    covered = included[np.arange(len(labels)), truth_indices]
    return {
        "empirical_coverage": float(covered.mean()),
        "mean_set_size": float(sizes.mean()),
        "singleton_rate": float((sizes == 1).mean()),
        "empty_set_rate": float((sizes == 0).mean()),
    }


def _score_columns(class_names):
    return ["score__%s" % name for name in class_names]


def _locked_predictions(frame, strategy):
    result = frame.loc[frame["strategy"].astype(str) == str(strategy)].copy()
    result["pair_key"] = [
        _pair_key(drug, target)
        for drug, target in zip(result["drug_name"], result["target_name"])
    ]
    return result


def _calibration_source(development, split, repeat):
    if split in set(development["split"].astype(str)):
        return development.loc[
            (development["split"].astype(str) == str(split))
            & (development["repeat"].astype(int) == int(repeat))
        ].copy(), "split_matched"
    disjoint = development.loc[
        development["split"].astype(str).isin(("drug_disjoint", "target_disjoint"))
        & (development["repeat"].astype(int) == int(repeat))
    ].copy()
    return disjoint, "pooled_drug_and_target_disjoint"


def _expert_uncertainty(oof_frame, class_names):
    key_columns = [
        "split", "repeat", "fold", "drug_name", "target_name", "true_label"
    ]
    experts = oof_frame.loc[
        oof_frame["strategy"].astype(str).str.startswith(EXPERT_PREFIX)
    ].copy()
    rows = []
    score_columns = _score_columns(class_names)
    for keys, group in experts.groupby(key_columns, sort=False, dropna=False):
        scores = group[score_columns].to_numpy(dtype=float)
        scores = _normalise_probabilities(scores)
        votes = scores.argmax(axis=1)
        vote_count = Counter(votes)
        disagreement = 1.0 - max(vote_count.values()) / float(len(votes))
        row = dict(zip(key_columns, keys))
        row.update({
            "expert_probability_variance": float(np.mean(np.var(scores, axis=0))),
            "expert_vote_disagreement": float(disagreement),
            "expert_count": int(len(scores)),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_reliability(
    development_oof,
    external_oof,
    *,
    output_dir,
    development_double_oof=None,
    strategy="pair_conditioned_gate",
    class_names=DEFAULT_CLASSES,
    calibration_fraction=0.5,
    conformal_alphas=(0.1, 0.2),
    selective_coverages=(1.0, 0.9, 0.8),
    ece_bins=10,
):
    output_dir = ensure_dir(Path(output_dir))
    development_all = pd.read_csv(development_oof)
    if development_double_oof and Path(development_double_oof).exists():
        double_frame = pd.read_csv(development_double_oof)
        development_all = pd.concat(
            [
                development_all.loc[
                    development_all["split"].astype(str) != "double_disjoint"
                ],
                double_frame,
            ],
            ignore_index=True,
        )
    external_all = pd.read_csv(external_oof)
    development = _locked_predictions(development_all, strategy)
    external = _locked_predictions(external_all, strategy)
    expert_uncertainty = _expert_uncertainty(external_all, class_names)
    merge_keys = ["split", "repeat", "fold", "drug_name", "target_name", "true_label"]

    calibrated_rows = []
    calibration_rows = []
    hierarchy_rows = []
    selective_rows = []
    conformal_rows = []
    classwise_rows = []
    score_columns = _score_columns(class_names)

    for (split, repeat), test in external.groupby(["split", "repeat"], sort=True):
        calibration_pool, calibration_scope = _calibration_source(
            development, str(split), int(repeat)
        )
        calibration_pool = calibration_pool.copy()
        calibration_pool["partition_fraction"] = [
            _stable_fraction("%s|%s|%s" % (split, repeat, key))
            for key in calibration_pool["pair_key"]
        ]
        temperature_fit = calibration_pool.loc[
            calibration_pool["partition_fraction"] < float(calibration_fraction)
        ].copy()
        conformal_fit = calibration_pool.loc[
            calibration_pool["partition_fraction"] >= float(calibration_fraction)
        ].copy()
        if min(len(temperature_fit), len(conformal_fit)) < 20:
            raise ValueError("Development calibration partition is unexpectedly small")
        raw_fit = temperature_fit[score_columns].to_numpy(dtype=float)
        temperature = fit_temperature(raw_fit, temperature_fit["true_label"], class_names)
        raw_test = test[score_columns].to_numpy(dtype=float)
        calibrated_test = _temperature_scale(raw_test, temperature)
        raw_test = _normalise_probabilities(raw_test)
        calibrated_fit = _temperature_scale(
            conformal_fit[score_columns].to_numpy(dtype=float), temperature
        )

        for state, probabilities in (
            ("uncalibrated", raw_test),
            ("temperature_scaled", calibrated_test),
        ):
            metrics = _probability_metrics(
                probabilities, test["true_label"], class_names, bins=ece_bins
            )
            calibration_rows.append({
                "split": split,
                "repeat": int(repeat),
                "state": state,
                "calibration_scope": calibration_scope,
                "temperature": temperature,
                "temperature_fit_count": int(len(temperature_fit)),
                "conformal_fit_count": int(len(conformal_fit)),
                "external_count": int(len(test)),
                **metrics,
            })

        predicted = np.asarray(
            [class_names[index] for index in calibrated_test.argmax(axis=1)]
        )
        hierarchy_rows.append({
            "split": split,
            "repeat": int(repeat),
            "pair_count": int(len(test)),
            **_hierarchical_metrics(test["true_label"], predicted),
        })
        precision, recall, f1, support = precision_recall_fscore_support(
            test["true_label"],
            predicted,
            labels=list(class_names),
            zero_division=0,
        )
        for index, class_name in enumerate(class_names):
            classwise_rows.append({
                "split": split,
                "repeat": int(repeat),
                "class": class_name,
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            })

        for row in _risk_coverage_rows(
            calibrated_test,
            test["true_label"],
            class_names,
            selective_coverages,
        ):
            selective_rows.append({"split": split, "repeat": int(repeat), **row})

        for alpha in conformal_alphas:
            threshold = _conformal_quantile(
                calibrated_fit,
                conformal_fit["true_label"],
                class_names,
                alpha,
            )
            conformal_rows.append({
                "split": split,
                "repeat": int(repeat),
                "alpha": float(alpha),
                "nominal_coverage": float(1.0 - alpha),
                "calibration_scope": calibration_scope,
                "threshold": threshold,
                **_evaluate_prediction_sets(
                    calibrated_test,
                    test["true_label"],
                    class_names,
                    threshold,
                ),
            })

        uncertainty = expert_uncertainty.loc[
            (expert_uncertainty["split"].astype(str) == str(split))
            & (expert_uncertainty["repeat"].astype(int) == int(repeat))
        ]
        test_output = test.copy().reset_index(drop=True)
        for index, class_name in enumerate(class_names):
            test_output["raw_score__%s" % class_name] = raw_test[:, index]
            test_output["calibrated_score__%s" % class_name] = calibrated_test[:, index]
        test_output["calibrated_predicted_label"] = predicted
        test_output["confidence"] = calibrated_test.max(axis=1)
        test_output["entropy"] = -np.sum(
            calibrated_test * np.log(np.clip(calibrated_test, 1.0e-12, 1.0)), axis=1
        ) / math.log(len(class_names))
        ordered = np.sort(calibrated_test, axis=1)
        test_output["top_two_margin"] = ordered[:, -1] - ordered[:, -2]
        test_output["temperature"] = temperature
        test_output["calibration_scope"] = calibration_scope
        test_output = test_output.merge(uncertainty, on=merge_keys, how="left")
        calibrated_rows.append(test_output)

    calibrated = pd.concat(calibrated_rows, ignore_index=True)
    calibration = pd.DataFrame(calibration_rows)
    hierarchy = pd.DataFrame(hierarchy_rows)
    selective = pd.DataFrame(selective_rows)
    conformal = pd.DataFrame(conformal_rows)
    classwise = pd.DataFrame(classwise_rows)

    calibrated.to_csv(output_dir / "external_calibrated_predictions.csv", index=False)
    calibration.to_csv(output_dir / "calibration_repeat_metrics.csv", index=False)
    hierarchy.to_csv(output_dir / "hierarchical_repeat_metrics.csv", index=False)
    selective.to_csv(output_dir / "selective_prediction_repeat_metrics.csv", index=False)
    conformal.to_csv(output_dir / "conformal_repeat_metrics.csv", index=False)
    classwise.to_csv(output_dir / "classwise_repeat_metrics.csv", index=False)

    summaries = {
        "calibration": _numeric_summary(calibration, ["split", "state"]),
        "hierarchical": _numeric_summary(hierarchy, ["split"]),
        "selective": _numeric_summary(selective, ["split", "coverage"]),
        "conformal": _numeric_summary(conformal, ["split", "alpha"]),
        "classwise": _numeric_summary(classwise, ["split", "class"]),
    }
    for name, frame in summaries.items():
        frame.to_csv(output_dir / ("%s_summary.csv" % name), index=False)
    return calibrated, summaries


def _numeric_summary(frame, group_columns):
    numeric = [
        column
        for column in frame.select_dtypes(include=[np.number]).columns
        if column not in set(group_columns) | {"repeat", "seed", "fold"}
    ]
    if not numeric:
        return frame[list(group_columns)].drop_duplicates().reset_index(drop=True)
    grouped = frame.groupby(group_columns, dropna=False)[numeric]
    mean = grouped.mean().add_suffix("_mean")
    standard = grouped.std(ddof=1).fillna(0.0).add_suffix("_sd")
    return mean.join(standard).reset_index()


def evaluate_target_context_sensitivity(
    external_oof_predictions,
    *,
    output_dir,
    strategy="pair_conditioned_gate",
    routing_repeat_metrics=None,
    drug_only_repeat_metrics=None,
):
    """Audit label granularity and compare with a drug-embedding-only control.

    The historical four-class cohort assigns one action annotation to each drug
    and expands it across the drug's annotated targets. This function makes
    that structure explicit and tests whether target-containing pair features
    improve over a full-KG RGCN drug embedding alone under identical outer
    split definitions. It does not relabel the cohort or introduce a DTI task.
    """

    output_dir = ensure_dir(Path(output_dir))
    external_all = pd.read_csv(external_oof_predictions)
    locked = _locked_predictions(external_all, strategy)
    pair_frame = locked[["drug_name", "target_name", "true_label"]].drop_duplicates()
    per_drug = pair_frame.groupby("drug_name", dropna=False).agg(
        target_count=("target_name", "nunique"),
        action_count=("true_label", "nunique"),
    )
    audit = pd.DataFrame([{
        "pair_count": int(len(pair_frame)),
        "drug_count": int(pair_frame["drug_name"].nunique()),
        "target_count": int(pair_frame["target_name"].nunique()),
        "multi_target_drug_count": int((per_drug["target_count"] > 1).sum()),
        "single_action_drug_count": int((per_drug["action_count"] == 1).sum()),
        "multi_action_drug_count": int((per_drug["action_count"] > 1).sum()),
        "single_action_drug_fraction": float((per_drug["action_count"] == 1).mean()),
        "median_targets_per_drug": float(per_drug["target_count"].median()),
        "maximum_targets_per_drug": int(per_drug["target_count"].max()),
        "label_scope": "one drug-level four-class action expanded across annotated targets",
    }])
    audit.to_csv(output_dir / "cohort_label_structure_audit.csv", index=False)

    routing_repeat_metrics = routing_repeat_metrics or {}
    drug_only_repeat_metrics = drug_only_repeat_metrics or {}
    rows = []
    for cohort in sorted(set(routing_repeat_metrics) & set(drug_only_repeat_metrics)):
        routing_path = Path(routing_repeat_metrics[cohort])
        control_path = Path(drug_only_repeat_metrics[cohort])
        if not routing_path.exists() or not control_path.exists():
            continue
        routing = pd.read_csv(routing_path)
        routing = routing.loc[
            routing["strategy"].astype(str) == str(strategy)
        ].copy()
        control = pd.read_csv(control_path)
        if "view" in control.columns:
            control = control.loc[control["view"].astype(str) == "full_label_free"]
        if "model" in control.columns:
            control = control.loc[control["model"].astype(str) == "RGCN"]
        if "classifier" in control.columns:
            control = control.loc[
                control["classifier"].astype(str) == "Random Forest"
            ]
        common_splits = sorted(
            set(routing["split"].astype(str)) & set(control["split"].astype(str))
        )
        for split in common_splits:
            routed_values = routing.loc[
                routing["split"].astype(str) == split, "f1_macro"
            ].astype(float)
            control_values = control.loc[
                control["split"].astype(str) == split, "f1_macro"
            ].astype(float)
            if routed_values.empty or control_values.empty:
                continue
            rows.append({
                "cohort": cohort,
                "split": split,
                "routed_pair_feature_macro_f1_mean": float(routed_values.mean()),
                "routed_pair_feature_macro_f1_sd": float(routed_values.std(ddof=1)),
                "drug_embedding_only_macro_f1_mean": float(control_values.mean()),
                "drug_embedding_only_macro_f1_sd": float(control_values.std(ddof=1)),
                "routed_minus_drug_only_macro_f1": float(
                    routed_values.mean() - control_values.mean()
                ),
                "repeat_count": int(min(len(routed_values), len(control_values))),
            })
    sensitivity_columns = [
        "cohort", "split", "routed_pair_feature_macro_f1_mean",
        "routed_pair_feature_macro_f1_sd", "drug_embedding_only_macro_f1_mean",
        "drug_embedding_only_macro_f1_sd", "routed_minus_drug_only_macro_f1",
        "repeat_count",
    ]
    sensitivity = pd.DataFrame(rows, columns=sensitivity_columns)
    sensitivity.to_csv(output_dir / "target_context_sensitivity_summary.csv", index=False)
    return audit, sensitivity


def evaluate_relation_attribution(attribution_csv, cases_jsonl, *, output_dir):
    """Summarise gate weights and exact component-deletion behaviour."""

    output_dir = ensure_dir(Path(output_dir))
    attribution = pd.read_csv(attribution_csv)
    view_repeat = (
        attribution.groupby(["split", "repeat", "view"], dropna=False)
        .agg(
            pair_count=("drug_name", "size"),
            mean_gate_weight=("gate_weight", "mean"),
            mean_deletion_delta=("view_component_deletion_delta", "mean"),
            median_deletion_delta=("view_component_deletion_delta", "median"),
            positive_deletion_rate=(
                "view_component_deletion_delta", lambda values: float((values > 0).mean())
            ),
        )
        .reset_index()
    )
    view_repeat.to_csv(output_dir / "relation_view_repeat_summary.csv", index=False)
    _numeric_summary(view_repeat, ["split", "view"]).to_csv(
        output_dir / "relation_view_summary.csv", index=False
    )

    pair_columns = ["split", "repeat", "drug_name", "target_name"]
    pair_differences = []
    top_view_rows = []
    for keys, group in attribution.groupby(pair_columns, sort=False):
        group = group.sort_values("gate_weight", ascending=False, kind="mergesort")
        top = group.iloc[0]
        others = group.iloc[1:]
        pair_differences.append({
            **dict(zip(pair_columns, keys)),
            "top_view": top["view"],
            "top_weight": float(top["gate_weight"]),
            "top_deletion_delta": float(top["view_component_deletion_delta"]),
            "mean_non_top_deletion_delta": float(
                others["view_component_deletion_delta"].mean()
            ),
            "top_minus_non_top_delta": float(
                top["view_component_deletion_delta"]
                - others["view_component_deletion_delta"].mean()
            ),
        })
        top_view_rows.append({
            "split": keys[0],
            "repeat": int(keys[1]),
            "pair_key": _pair_key(keys[2], keys[3]),
            "top_view": top["view"],
        })
    pair_differences = pd.DataFrame(pair_differences)
    pair_differences.to_csv(output_dir / "relation_top_view_pair_effects.csv", index=False)

    faithfulness_rows = []
    rng = np.random.default_rng(20260725)
    for (split, repeat), group in pair_differences.groupby(["split", "repeat"]):
        values = group["top_minus_non_top_delta"].to_numpy(dtype=float)
        bootstrap = np.asarray([
            rng.choice(values, size=len(values), replace=True).mean() for _ in range(2000)
        ])
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=(2000, len(values)))
        null = np.mean(signs * values[None, :], axis=1)
        observed = float(values.mean())
        faithfulness_rows.append({
            "split": split,
            "repeat": int(repeat),
            "pair_count": int(len(values)),
            "mean_top_minus_non_top_delta": observed,
            "bootstrap_ci_low": float(np.quantile(bootstrap, 0.025)),
            "bootstrap_ci_high": float(np.quantile(bootstrap, 0.975)),
            "sign_flip_p_value": float(
                (int((np.abs(null) >= abs(observed)).sum()) + 1) / (len(null) + 1)
            ),
        })
    faithfulness = pd.DataFrame(faithfulness_rows)
    faithfulness.to_csv(output_dir / "relation_faithfulness_inference.csv", index=False)

    stability_rows = []
    top_views = pd.DataFrame(top_view_rows)
    for (split, pair_key), group in top_views.groupby(["split", "pair_key"]):
        counts = group["top_view"].value_counts()
        stability_rows.append({
            "split": split,
            "pair_key": pair_key,
            "repeat_count": int(len(group)),
            "modal_top_view": str(counts.index[0]),
            "top_view_stability": float(counts.iloc[0] / len(group)),
        })
    stability = pd.DataFrame(stability_rows)
    stability.to_csv(output_dir / "relation_top_view_stability.csv", index=False)
    _numeric_summary(stability, ["split"]).to_csv(
        output_dir / "relation_top_view_stability_summary.csv", index=False
    )

    flip_rows = []
    with Path(cases_jsonl).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            class_names = sorted(record["prediction_scores"])
            mixture = np.asarray(
                [record["prediction_scores"][name] for name in class_names], dtype=float
            )
            original_index = int(np.argmax(mixture))
            for view in record["views"]:
                weight = float(view["gate_weight"])
                expert = np.asarray(
                    [view["expert_prediction_scores"][name] for name in class_names],
                    dtype=float,
                )
                if weight >= 1.0 - 1.0e-12:
                    deleted = np.full_like(mixture, 1.0 / len(mixture))
                else:
                    deleted = np.clip((mixture - weight * expert) / (1.0 - weight), 0.0, None)
                    deleted = deleted / np.clip(deleted.sum(), 1.0e-12, None)
                deleted_label = class_names[int(np.argmax(deleted))]
                flip_rows.append({
                    "split": record["split"],
                    "repeat": int(record["repeat"]),
                    "drug_name": record["drug_name"],
                    "target_name": record["target_name"],
                    "true_label": record["true_label"],
                    "original_label": class_names[original_index],
                    "deleted_label": deleted_label,
                    "view": view["view"],
                    "gate_weight": weight,
                    "deletion_delta": float(view["view_component_deletion_delta"]),
                    "prediction_flip": bool(deleted_label != class_names[original_index]),
                    "correct_to_incorrect": bool(
                        class_names[original_index] == record["true_label"]
                        and deleted_label != record["true_label"]
                    ),
                })
    flips = pd.DataFrame(flip_rows)
    flips.to_csv(output_dir / "relation_deletion_pair_outcomes.csv", index=False)
    flip_summary = (
        flips.groupby(["split", "repeat", "view"])
        .agg(
            pair_count=("drug_name", "size"),
            prediction_flip_rate=("prediction_flip", "mean"),
            correct_to_incorrect_rate=("correct_to_incorrect", "mean"),
            mean_deletion_delta=("deletion_delta", "mean"),
        )
        .reset_index()
    )
    flip_summary.to_csv(output_dir / "relation_deletion_flip_repeat_summary.csv", index=False)
    _numeric_summary(flip_summary, ["split", "view"]).to_csv(
        output_dir / "relation_deletion_flip_summary.csv", index=False
    )
    return {
        "view_summary": view_repeat,
        "faithfulness": faithfulness,
        "stability": stability,
        "flip_summary": flip_summary,
    }


def _evidence_record_id(row):
    value = "|".join(
        str(row.get(column, ""))
        for column in (
            "cohort", "pair_id", "source", "evidence_type", "source_drug_id",
            "source_target_id", "action_raw", "core4_action"
        )
    )
    return "ev_" + hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _evidence_status(prediction, actions):
    actions = {str(action) for action in actions if str(action) in DEFAULT_CLASSES}
    if not actions:
        return "insufficient"
    prediction_polarity = _polarity(prediction)
    matching = prediction in actions
    same_polarity = {action for action in actions if _polarity(action) == prediction_polarity}
    opposite = actions - same_polarity
    if matching and len(actions) == 1:
        return "supported"
    if matching:
        return "partially_supported" if not opposite else "mixed"
    if same_polarity and not opposite:
        return "ontology_disagreement"
    if opposite and not same_polarity:
        return "contradicted"
    return "mixed"


def _load_kag_questions(path):
    if not path or not Path(path).exists():
        return {}
    records = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            drug = record.get("drug_name")
            target = record.get("target_name")
            if drug is None or target is None:
                continue
            records[_pair_key(drug, target)] = record
    return records


def _rerank_kag_paths(paths, pair_attribution, limit=5):
    """Rerank grounded paths with the model's quantitative view attribution.

    Path scores remain retrieval-specificity scores.  The multiplicative
    attribution factor changes only ordering and is never treated as signed
    pharmacological evidence.
    """

    path_view = {
        "shared_pathway": "targets_pathway",
        "shared_anatomy": "targets_anatomy",
        "shared_indication_cotarget": "targets_indication",
    }
    importance = {}
    for row in pair_attribution.itertuples(index=False):
        importance[str(row.view)] = max(0.0, float(row.deletion_delta)) + 0.10 * max(
            0.0, float(row.gate_weight)
        )
    available_views = {
        path_view.get(str(path.get("path_type", "unknown"))) for path in (paths or [])
    }
    available_values = [
        importance[view]
        for view in available_views
        if view is not None and view in importance
    ]
    maximum = max(available_values) if available_values else 0.0
    reranked = []
    for path in paths or []:
        path_type = str(path.get("path_type", "unknown"))
        view = path_view.get(path_type)
        raw_importance = importance.get(view, 0.0)
        normalised_importance = raw_importance / maximum if maximum > 0 else 0.0
        retrieval_score = float(path.get("score", 0.0))
        path_type_prior = float(KAG_PATH_TYPE_PRIOR.get(path_type, 0.50))
        evidence_tier = KAG_PATH_EVIDENCE_TIER.get(
            path_type, "unclassified_context"
        )
        guided_score = (
            retrieval_score * (0.5 + normalised_importance) * path_type_prior
        )
        reranked.append({
            **path,
            "attributed_view": view,
            "view_importance": float(raw_importance),
            "normalised_view_importance": float(normalised_importance),
            "retrieval_score": retrieval_score,
            "path_type_prior": path_type_prior,
            "evidence_tier": evidence_tier,
            "attribution_guided_score": float(guided_score),
        })
    reranked.sort(
        key=lambda row: (
            row["attribution_guided_score"],
            row["retrieval_score"],
            str(row.get("path_id", "")),
        ),
        reverse=True,
    )
    return reranked[: int(limit)]


def _aggregate_external_predictions(calibrated):
    score_columns = [column for column in calibrated if column.startswith("calibrated_score__")]
    rows = []
    for keys, group in calibrated.groupby(
        ["split", "drug_name", "target_name", "true_label"], sort=True
    ):
        scores = group[score_columns].mean().to_numpy(dtype=float)
        class_names = [column.replace("calibrated_score__", "") for column in score_columns]
        predicted = class_names[int(np.argmax(scores))]
        rows.append({
            "split": keys[0],
            "drug_name": keys[1],
            "target_name": keys[2],
            "true_label": keys[3],
            "pair_key": _pair_key(keys[1], keys[2]),
            "predicted_label": predicted,
            "mean_confidence": float(np.max(scores)),
            "prediction_repeat_stability": float(
                (group["calibrated_predicted_label"].astype(str) == predicted).mean()
            ),
            "mean_expert_vote_disagreement": float(
                group["expert_vote_disagreement"].mean()
            ),
            **{
                "mean_score__%s" % name: float(scores[index])
                for index, name in enumerate(class_names)
            },
        })
    return pd.DataFrame(rows)


def evaluate_independent_evidence(
    calibrated_predictions,
    evidence_csv,
    attribution_csv,
    *,
    output_dir,
    kag_question_records=None,
):
    """Audit predictions using source-isolated pair-action evidence records."""

    output_dir = ensure_dir(Path(output_dir))
    calibrated = (
        calibrated_predictions.copy()
        if isinstance(calibrated_predictions, pd.DataFrame)
        else pd.read_csv(calibrated_predictions)
    )
    predictions = _aggregate_external_predictions(calibrated)
    evidence = pd.read_csv(evidence_csv)
    evidence = evidence.loc[
        (evidence["cohort"].astype(str) == "cpg0004")
        & evidence["source"].astype(str).isin(INDEPENDENT_SOURCES)
        & evidence["direct_pair_action"].fillna(False).astype(bool)
        & evidence["core4_action"].astype(str).isin(DEFAULT_CLASSES)
    ].copy()
    evidence["pair_key"] = [
        _pair_key(drug, target)
        for drug, target in zip(evidence["drug_name"], evidence["target_name"])
    ]
    evidence["evidence_record_id"] = [_evidence_record_id(row) for _, row in evidence.iterrows()]

    attribution = pd.read_csv(attribution_csv)
    attribution["pair_key"] = [
        _pair_key(drug, target)
        for drug, target in zip(attribution["drug_name"], attribution["target_name"])
    ]
    attribution_mean = (
        attribution.groupby(["split", "pair_key", "view"], as_index=False)
        .agg(
            gate_weight=("gate_weight", "mean"),
            deletion_delta=("view_component_deletion_delta", "mean"),
        )
    )
    kag_records = _load_kag_questions(kag_question_records)

    audit_rows = []
    graph_records = []
    kag_rows = []
    for _, prediction in predictions.iterrows():
        pair_evidence = evidence.loc[evidence["pair_key"] == prediction["pair_key"]]
        actions = sorted(set(pair_evidence["core4_action"].dropna().astype(str)))
        status = _evidence_status(str(prediction["predicted_label"]), actions)
        source_actions = {
            source: sorted(set(group["core4_action"].dropna().astype(str)))
            for source, group in pair_evidence.groupby("source")
        }
        pair_attribution = attribution_mean.loc[
            (attribution_mean["split"].astype(str) == str(prediction["split"]))
            & (attribution_mean["pair_key"] == prediction["pair_key"])
        ].sort_values(["deletion_delta", "gate_weight"], ascending=False)
        top_view = str(pair_attribution.iloc[0]["view"]) if len(pair_attribution) else "unavailable"
        top_delta = (
            float(pair_attribution.iloc[0]["deletion_delta"])
            if len(pair_attribution)
            else math.nan
        )
        kag_record = kag_records.get(prediction["pair_key"], {})
        raw_paths = kag_record.get("paths", [])
        guided_paths = _rerank_kag_paths(raw_paths, pair_attribution, limit=5)
        available_views = {
            {
                "shared_pathway": "targets_pathway",
                "shared_anatomy": "targets_anatomy",
                "shared_indication_cotarget": "targets_indication",
            }.get(str(path.get("path_type")))
            for path in raw_paths
        }
        available_views.discard(None)
        if available_views and len(pair_attribution):
            available_importance = pair_attribution.loc[
                pair_attribution["view"].astype(str).isin(available_views)
            ].copy()
            if len(available_importance):
                available_importance["importance"] = (
                    available_importance["deletion_delta"].clip(lower=0.0)
                    + 0.10 * available_importance["gate_weight"].clip(lower=0.0)
                )
                highest_available_view = str(
                    available_importance.sort_values(
                        ["importance", "view"], ascending=[False, True]
                    ).iloc[0]["view"]
                )
            else:
                highest_available_view = "unavailable"
        else:
            highest_available_view = "unavailable"
        top_path_view = (
            str(guided_paths[0].get("attributed_view")) if guided_paths else "unavailable"
        )
        kag_rows.append({
            "split": prediction["split"],
            "drug_name": prediction["drug_name"],
            "target_name": prediction["target_name"],
            "pair_key": prediction["pair_key"],
            "has_masked_edge_context_path": bool(len(raw_paths)),
            "retrieved_path_count": int(len(raw_paths)),
            "returned_guided_path_count": int(len(guided_paths)),
            "top_path_type": guided_paths[0].get("path_type") if guided_paths else "",
            "top_path_view": top_path_view,
            "highest_attributed_available_view": highest_available_view,
            "top_path_matches_highest_attributed_available_view": (
                float(top_path_view == highest_available_view)
                if guided_paths
                else math.nan
            ),
            "top_attribution_guided_score": (
                float(guided_paths[0]["attribution_guided_score"])
                if guided_paths
                else math.nan
            ),
            "has_independent_signed_action_evidence": bool(len(actions)),
            "joint_context_and_signed_evidence": bool(len(guided_paths) and len(actions)),
        })
        record_ids = sorted(set(pair_evidence["evidence_record_id"].astype(str)))
        audit_rows.append({
            **prediction.to_dict(),
            "evidence_status": status,
            "independent_action_labels": "|".join(actions),
            "drugbank_actions": "|".join(source_actions.get("DrugBank", [])),
            "open_targets_actions": "|".join(source_actions.get("Open Targets", [])),
            "independent_source_count": int(len(source_actions)),
            "independent_record_count": int(len(pair_evidence)),
            "evidence_record_ids": "|".join(record_ids),
            "top_attributed_view": top_view,
            "top_view_deletion_delta": top_delta,
            "masked_edge_context_path_count": int(len(raw_paths)),
            "top_context_path_id": (
                str(guided_paths[0].get("path_id", "")) if guided_paths else ""
            ),
            "top_context_path_type": (
                str(guided_paths[0].get("path_type", "")) if guided_paths else ""
            ),
        })
        evidence_nodes = []
        evidence_edges = []
        for _, item in pair_evidence.iterrows():
            evidence_nodes.append({
                "id": item["evidence_record_id"],
                "type": "source_action_record",
                "source": item["source"],
                "source_drug_id": None if pd.isna(item["source_drug_id"]) else str(item["source_drug_id"]),
                "source_target_id": None if pd.isna(item["source_target_id"]) else str(item["source_target_id"]),
                "action_raw": None if pd.isna(item["action_raw"]) else str(item["action_raw"]),
                "mapped_action": str(item["core4_action"]),
                "evidence_text": None if pd.isna(item["evidence_text"]) else str(item["evidence_text"]),
            })
            evidence_edges.append({
                "source": item["evidence_record_id"],
                "relation": (
                    "supports_exact" if str(item["core4_action"]) == prediction["predicted_label"]
                    else "supports_same_polarity"
                    if _polarity(item["core4_action"]) == _polarity(prediction["predicted_label"])
                    else "contradicts_direction"
                ),
                "target": "prediction",
            })
        if guided_paths and guided_paths[0].get("evidence_tier") == "broad_context_only":
            context_scope = (
                "This anatomy-only route is broad co-localisation context, not "
                "relation-specific or directional evidence. "
            )
        else:
            context_scope = (
                "This unsigned path motivates a relation hypothesis but does not "
                "establish action direction. "
            )
        context_sentence = (
            "The highest-ranked masked-edge context was %s [%s], attributed to %s. %s"
            % (
                str(guided_paths[0].get("statement", "typed context path")),
                str(guided_paths[0].get("path_id", "unavailable")),
                top_path_view,
                context_scope,
            )
            if guided_paths
            else "No typed indirect context path was retrieved after masking the direct edge. "
        )
        explanation = (
            "KGML predicted %s for %s--%s (mean confidence %.3f; repeat stability %.3f). "
            "The source-isolated evidence audit was %s: DrugBank=%s and Open Targets=%s. "
            "The largest positive routed-view deletion effect was %s (delta %.3f). "
            "%sThe view effect is faithful to the locked mixture but is not an edge-level causal mechanism."
            % (
                prediction["predicted_label"],
                prediction["drug_name"],
                prediction["target_name"],
                prediction["mean_confidence"],
                prediction["prediction_repeat_stability"],
                status.replace("_", " "),
                ",".join(source_actions.get("DrugBank", [])) or "no direct action record",
                ",".join(source_actions.get("Open Targets", [])) or "no direct action record",
                top_view,
                top_delta,
                context_sentence,
            )
        )
        graph_records.append({
            "query": {
                "drug": prediction["drug_name"],
                "target": prediction["target_name"],
                "split": prediction["split"],
                "question": (
                    "Which typed, sourced context supports, contradicts, or fails to verify "
                    "the predicted drug--target action?"
                ),
            },
            "prediction": {
                "id": "prediction",
                "action": prediction["predicted_label"],
                "mean_confidence": prediction["mean_confidence"],
                "repeat_stability": prediction["prediction_repeat_stability"],
            },
            "evidence_status": status,
            "nodes": evidence_nodes,
            "edges": evidence_edges,
            "model_attribution": pair_attribution.to_dict(orient="records"),
            "attribution_guided_context_paths": guided_paths,
            "grounded_explanation": explanation,
            "claim_scope": (
                "Source-isolated evidence verification plus routed-view attribution; "
                "not an independent outcome benchmark and not proof of a molecular "
                "mechanism or edge-level RGCN causality."
            ),
        })

    audit = pd.DataFrame(audit_rows)
    audit.to_csv(output_dir / "independent_evidence_audit.csv", index=False)
    with (output_dir / "evidence_graphs.jsonl").open("w", encoding="utf-8") as handle:
        for record in graph_records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    kag_audit = pd.DataFrame(kag_rows)
    kag_audit.to_csv(output_dir / "attribution_guided_kag_audit.csv", index=False)
    if len(kag_audit):
        kag_summary = (
            kag_audit.groupby("split")
            .agg(
                pair_count=("pair_key", "size"),
                masked_edge_context_coverage=("has_masked_edge_context_path", "mean"),
                mean_retrieved_path_count=("retrieved_path_count", "mean"),
                attribution_alignment_rate=(
                    "top_path_matches_highest_attributed_available_view", "mean"
                ),
                signed_action_evidence_coverage=(
                    "has_independent_signed_action_evidence", "mean"
                ),
                joint_context_and_signed_evidence_coverage=(
                    "joint_context_and_signed_evidence", "mean"
                ),
            )
            .reset_index()
        )
        kag_summary.to_csv(output_dir / "attribution_guided_kag_summary.csv", index=False)

    coverage = (
        audit.groupby(["split", "evidence_status"]).size().rename("pair_count").reset_index()
    )
    totals = audit.groupby("split").size().rename("total_pairs")
    coverage = coverage.merge(totals, on="split")
    coverage["fraction"] = coverage["pair_count"] / coverage["total_pairs"]
    coverage.to_csv(output_dir / "evidence_status_summary.csv", index=False)

    source_metrics = _source_disjoint_metrics(calibrated, evidence)
    source_metrics.to_csv(output_dir / "source_isolated_repeat_metrics.csv", index=False)
    _numeric_summary(source_metrics, ["split", "source"]).to_csv(
        output_dir / "source_isolated_summary.csv", index=False
    )

    groundedness_rows = []
    evidence_lookup = set(evidence["evidence_record_id"].astype(str))
    for record in graph_records:
        node_ids = {node["id"] for node in record["nodes"]}
        edge_ids = {edge["source"] for edge in record["edges"]}
        context_paths = record.get("attribution_guided_context_paths", [])
        context_citations = [
            citation
            for path in context_paths
            for citation in path.get("citations", [])
        ]
        context_paths_cited = all(
            path.get("path_id")
            and path.get("statement")
            and len(path.get("citations", [])) > 0
            for path in context_paths
        )
        context_edges_identified = all(
            citation.get("edge_id") and citation.get("source_database")
            for citation in context_citations
        )
        action_records_grounded = edge_ids <= node_ids and edge_ids <= evidence_lookup
        groundedness_rows.append({
            "split": record["query"]["split"],
            "drug_name": record["query"]["drug"],
            "target_name": record["query"]["target"],
            "claim_record_count": int(len(edge_ids)),
            "all_claim_records_in_retrieval": bool(edge_ids <= node_ids),
            "all_claim_records_in_source_table": bool(edge_ids <= evidence_lookup),
            "context_path_count": int(len(context_paths)),
            "context_citation_count": int(len(context_citations)),
            "all_context_paths_have_citations": bool(context_paths_cited),
            "all_context_citations_have_edge_and_source_ids": bool(
                context_edges_identified
            ),
            "groundedness_pass": bool(
                action_records_grounded and context_paths_cited and context_edges_identified
            ),
        })
    groundedness = pd.DataFrame(groundedness_rows)
    groundedness.to_csv(output_dir / "evidence_groundedness_checks.csv", index=False)
    groundedness.groupby("split")["groundedness_pass"].agg(["count", "mean"]).reset_index().to_csv(
        output_dir / "evidence_groundedness_summary.csv", index=False
    )

    _write_expert_review_forms(
        audit,
        graph_records,
        kag_records,
        output_dir=output_dir,
        sample_count=48,
    )
    return audit, source_metrics


def _source_disjoint_metrics(calibrated, evidence):
    rows = []
    predictions = calibrated.copy()
    predictions["pair_key"] = [
        _pair_key(drug, target)
        for drug, target in zip(predictions["drug_name"], predictions["target_name"])
    ]
    for source, source_frame in evidence.groupby("source"):
        unambiguous = (
            source_frame.groupby("pair_key")["core4_action"]
            .agg(lambda values: sorted(set(values.astype(str))))
        )
        unambiguous = unambiguous.loc[unambiguous.map(len) == 1].map(lambda values: values[0])
        for (split, repeat), group in predictions.groupby(["split", "repeat"]):
            merged = group.merge(
                unambiguous.rename("source_action"), left_on="pair_key", right_index=True
            )
            if not len(merged):
                continue
            predicted = merged["calibrated_predicted_label"].astype(str)
            truth = merged["source_action"].astype(str)
            rows.append({
                "split": split,
                "repeat": int(repeat),
                "source": source,
                "audited_pair_count": int(len(merged)),
                "exact_accuracy": float(accuracy_score(truth, predicted)),
                "exact_macro_f1": float(
                    f1_score(truth, predicted, average="macro", zero_division=0)
                ),
                "polarity_accuracy": float(
                    accuracy_score(truth.map(_polarity), predicted.map(_polarity))
                ),
                "polarity_macro_f1": float(
                    f1_score(
                        truth.map(_polarity),
                        predicted.map(_polarity),
                        average="macro",
                        zero_division=0,
                    )
                ),
                "reference_legacy_exact_agreement": float(
                    accuracy_score(truth, merged["true_label"].astype(str))
                ),
                "reference_legacy_polarity_agreement": float(
                    accuracy_score(
                        truth.map(_polarity), merged["true_label"].astype(str).map(_polarity)
                    )
                ),
            })
    return pd.DataFrame(rows)


def _review_path_text(path):
    if not path:
        return {
            "path_present": False,
            "path_statement": "No typed indirect path was retrieved after masking the direct edge.",
            "path_type": "no_path",
            "path_id": "",
            "path_evidence_tier": "insufficient_path_evidence",
            "path_citations": "",
        }
    citations = " || ".join(
        "%s --%s--> %s [%s; edge %s]"
        % (
            citation.get("subject", ""),
            citation.get("predicate", ""),
            citation.get("object", ""),
            citation.get("source_database", ""),
            citation.get("edge_id", ""),
        )
        for citation in path.get("citations", [])
    )
    return {
        "path_present": True,
        "path_statement": str(path.get("statement", "")),
        "path_type": str(path.get("path_type", "unknown")),
        "path_id": str(path.get("path_id", "")),
        "path_evidence_tier": str(
            path.get(
                "evidence_tier",
                KAG_PATH_EVIDENCE_TIER.get(
                    str(path.get("path_type", "unknown")), "unclassified_context"
                ),
            )
        ),
        "path_citations": citations,
    }


def _balanced_blind_sample(audit, graph_lookup, sample_count):
    """Deterministically sample globally unique pairs with broad stratum coverage."""

    available = audit.copy()
    if not len(available):
        return available
    available["path_stratum"] = [
        (
            graph_lookup.get(
                (str(row["split"]), str(row["pair_key"])), {}
            ).get("attribution_guided_context_paths")
            or [{"path_type": "no_path"}]
        )[0].get("path_type", "no_path")
        for _, row in available.iterrows()
    ]
    available["sample_rank"] = [
        hashlib.sha256(
            ("expert-review-v2|%s|%s" % (row["split"], row["pair_key"])).encode(
                "utf-8"
            )
        ).hexdigest()
        for _, row in available.iterrows()
    ]
    split_names = sorted(available["split"].astype(str).unique())
    base_quota, remainder = divmod(int(sample_count), max(1, len(split_names)))
    split_quota = {
        split: base_quota + int(index < remainder)
        for index, split in enumerate(split_names)
    }
    selected = []
    used_pairs = set()
    counts = {
        "path": Counter(),
        "status": Counter(),
        "label": Counter(),
    }
    for split in split_names:
        candidates = available.loc[available["split"].astype(str) == split].copy()
        for _ in range(split_quota[split]):
            candidates = candidates.loc[~candidates["pair_key"].isin(used_pairs)]
            if not len(candidates):
                break
            ranked = []
            for index, row in candidates.iterrows():
                balance_score = (
                    counts["path"][str(row["path_stratum"])]
                    + counts["status"][str(row["evidence_status"])]
                    + counts["label"][str(row["predicted_label"])]
                )
                ranked.append((balance_score, str(row["sample_rank"]), index))
            _, _, chosen_index = min(ranked)
            chosen = candidates.loc[chosen_index]
            selected.append(chosen_index)
            used_pairs.add(chosen["pair_key"])
            counts["path"][str(chosen["path_stratum"])] += 1
            counts["status"][str(chosen["evidence_status"])] += 1
            counts["label"][str(chosen["predicted_label"])] += 1
            candidates = candidates.drop(index=chosen_index)
    return available.loc[selected].copy().reset_index(drop=True)


def _write_expert_review_forms(
    audit,
    graph_records,
    kag_records,
    *,
    output_dir,
    sample_count=48,
):
    """Write a genuinely blinded relevance sheet, key, and A/B path comparison."""

    output_dir = Path(output_dir)
    graph_lookup = {
        (
            str(record["query"]["split"]),
            _pair_key(record["query"]["drug"], record["query"]["target"]),
        ): record
        for record in graph_records
    }
    sampled = _balanced_blind_sample(audit, graph_lookup, sample_count)
    blind_rows = []
    key_rows = []
    ab_rows = []
    ab_key_rows = []
    for _, row in sampled.iterrows():
        case_id = "ER%s" % hashlib.sha256(
            ("blind-case|%s|%s" % (row["split"], row["pair_key"])).encode("utf-8")
        ).hexdigest()[:10].upper()
        graph = graph_lookup.get((str(row["split"]), str(row["pair_key"])), {})
        guided_paths = graph.get("attribution_guided_context_paths", [])
        guided = guided_paths[0] if guided_paths else None
        path_fields = _review_path_text(guided)
        blind_rows.append({
            "case_id": case_id,
            "drug_name": row["drug_name"],
            "target_name": row["target_name"],
            **path_fields,
            "reviewer_path_relevance_1_to_5": "",
            "reviewer_biological_specificity_1_to_5": "",
            "reviewer_traceability_1_to_5": "",
            "reviewer_no_path_handling_appropriate_yes_no": "",
            "reviewer_comments": "",
        })
        key_rows.append({
            "case_id": case_id,
            "split": row["split"],
            "pair_key": row["pair_key"],
            "true_label": row["true_label"],
            "predicted_label": row["predicted_label"],
            "mean_confidence": row["mean_confidence"],
            "evidence_status": row["evidence_status"],
            "drugbank_actions": row["drugbank_actions"],
            "open_targets_actions": row["open_targets_actions"],
            "top_attributed_view": row["top_attributed_view"],
            "top_view_deletion_delta": row["top_view_deletion_delta"],
            "guided_path_id": path_fields["path_id"],
        })

        raw_paths = kag_records.get(row["pair_key"], {}).get("paths", [])
        unguided = max(
            raw_paths,
            key=lambda path: (float(path.get("score", 0.0)), str(path.get("path_id", ""))),
            default=None,
        )
        if not guided or not unguided or guided.get("path_id") == unguided.get("path_id"):
            continue
        candidate_paths = [("attribution_guided", guided), ("unguided", unguided)]
        if _stable_fraction("ab-order|%s" % case_id) < 0.5:
            candidate_paths.reverse()
        rendered = []
        mapping = {}
        for label, (identity, path) in zip(("A", "B"), candidate_paths):
            rendered.append((label, _review_path_text(path)))
            mapping[label] = identity
        ab_rows.append({
            "comparison_id": "AB%s" % case_id[2:],
            "drug_name": row["drug_name"],
            "target_name": row["target_name"],
            "candidate_A_statement": rendered[0][1]["path_statement"],
            "candidate_A_type": rendered[0][1]["path_type"],
            "candidate_A_citations": rendered[0][1]["path_citations"],
            "candidate_B_statement": rendered[1][1]["path_statement"],
            "candidate_B_type": rendered[1][1]["path_type"],
            "candidate_B_citations": rendered[1][1]["path_citations"],
            "reviewer_preference_A_B_tie": "",
            "reviewer_confidence_high_medium_low": "",
            "reviewer_comments": "",
        })
        ab_key_rows.append({
            "comparison_id": "AB%s" % case_id[2:],
            "candidate_A_identity": mapping["A"],
            "candidate_B_identity": mapping["B"],
            "guided_path_id": guided.get("path_id", ""),
            "unguided_path_id": unguided.get("path_id", ""),
        })

    pd.DataFrame(blind_rows).to_csv(
        output_dir / "expert_relevance_review_blinded.csv", index=False
    )
    pd.DataFrame(key_rows).to_csv(
        output_dir / "expert_relevance_review_key.csv", index=False
    )
    pd.DataFrame(ab_rows).to_csv(
        output_dir / "expert_relevance_ab_blinded.csv", index=False
    )
    pd.DataFrame(ab_key_rows).to_csv(
        output_dir / "expert_relevance_ab_key.csv", index=False
    )
    # Retain the historical filename as an explicitly deprecated pointer so
    # downstream archives do not silently confuse it with the new blind form.
    pd.DataFrame(blind_rows).to_csv(
        output_dir / "expert_relevance_review_sample.csv", index=False
    )
    protocol = """# Blinded path-relevance review protocol

The blinded sheet contains globally unique drug--target pairs and hides the
split, predicted action, confidence, relation-view attribution, signed-evidence
status, and source action labels. Reviewers score only the direct-edge-masked,
typed path and its cited graph edges. Cases without a path are retained so that
the system's abstention wording can be evaluated.

Scores use 1 (irrelevant/generic) to 5 (specific and biologically informative).
Anatomy-only paths must not be interpreted as action-direction evidence. The
separate A/B sheet compares attribution-guided and unguided retrieval only when
their top paths differ; candidate order is deterministically randomized. The
key files must remain hidden until all reviews are frozen.

AI-assisted pilot ratings may be used to debug the protocol, but they must not
be described as human domain-expert validation.
"""
    (output_dir / "expert_relevance_review_protocol.md").write_text(
        protocol, encoding="utf-8"
    )


def evaluate_blocker_challenge(calibrated_predictions, evidence_audit, *, output_dir):
    output_dir = ensure_dir(Path(output_dir))
    calibrated = (
        calibrated_predictions.copy()
        if isinstance(calibrated_predictions, pd.DataFrame)
        else pd.read_csv(calibrated_predictions)
    )
    audit = (
        evidence_audit.copy()
        if isinstance(evidence_audit, pd.DataFrame)
        else pd.read_csv(evidence_audit)
    )
    rows = []
    score_column = "calibrated_score__blocker"
    for (split, repeat), group in calibrated.groupby(["split", "repeat"]):
        truth = (group["true_label"].astype(str) == "blocker").astype(int)
        predicted = (group["calibrated_predicted_label"].astype(str) == "blocker").astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            truth, predicted, average="binary", zero_division=0
        )
        try:
            auroc = float(roc_auc_score(truth, group[score_column]))
            auprc = float(average_precision_score(truth, group[score_column]))
        except ValueError:
            auroc = math.nan
            auprc = math.nan
        false_positive_rate = float(((predicted == 1) & (truth == 0)).sum() / max(1, (truth == 0).sum()))
        inhibitory = group.loc[group["true_label"].astype(str).isin(INHIBITORY_ACTIONS)]
        rows.append({
            "split": split,
            "repeat": int(repeat),
            "pair_count": int(len(group)),
            "blocker_count": int(truth.sum()),
            "blocker_precision": float(precision),
            "blocker_recall": float(recall),
            "blocker_f1": float(f1),
            "blocker_auroc": auroc,
            "blocker_auprc": auprc,
            "blocker_false_positive_rate": false_positive_rate,
            "inhibitory_subtype_macro_f1": float(
                f1_score(
                    inhibitory["true_label"],
                    inhibitory["calibrated_predicted_label"],
                    labels=sorted(INHIBITORY_ACTIONS),
                    average="macro",
                    zero_division=0,
                )
            ),
        })
    repeat_metrics = pd.DataFrame(rows)
    repeat_metrics.to_csv(output_dir / "blocker_challenge_repeat_metrics.csv", index=False)
    summary = _numeric_summary(repeat_metrics, ["split"])
    summary.to_csv(output_dir / "blocker_challenge_summary.csv", index=False)

    independent_blocker = audit.loc[
        audit["independent_action_labels"].fillna("").astype(str).str.split("|").map(
            lambda values: "blocker" in values
        )
    ].copy()
    independent_blocker["predicted_blocker"] = (
        independent_blocker["predicted_label"].astype(str) == "blocker"
    )
    independent_blocker.to_csv(
        output_dir / "independent_blocker_evidence_cases.csv", index=False
    )
    if len(independent_blocker):
        independent_blocker.groupby("split").agg(
            independent_blocker_pairs=("pair_key", "size"),
            exact_blocker_prediction_rate=("predicted_blocker", "mean"),
        ).reset_index().to_csv(
            output_dir / "independent_blocker_evidence_summary.csv", index=False
        )
    return repeat_metrics, summary


def _write_markdown_summary(output_dir):
    output_dir = Path(output_dir)
    hierarchy = pd.read_csv(output_dir / "hierarchical_summary.csv")
    calibration = pd.read_csv(output_dir / "calibration_summary.csv")
    selective = pd.read_csv(output_dir / "selective_summary.csv")
    conformal = pd.read_csv(output_dir / "conformal_summary.csv")
    evidence = pd.read_csv(output_dir / "evidence_status_summary.csv")
    source = pd.read_csv(output_dir / "source_isolated_summary.csv")
    blocker = pd.read_csv(output_dir / "blocker_challenge_summary.csv")
    label_structure_path = output_dir / "cohort_label_structure_audit.csv"
    sensitivity_path = output_dir / "target_context_sensitivity_summary.csv"
    lines = [
        "# Locked four-class action reliability and evidence audit",
        "",
        "All calibration parameters and conformal thresholds were learned from the "
        "cpg0004-excluded development cohort. Source-isolated DrugBank and Open "
        "Targets records were used only after prediction for evidence verification.",
        "",
        "## Hierarchical action metrics",
        "",
        hierarchy.to_markdown(index=False),
        "",
        "## Probability calibration",
        "",
        calibration.to_markdown(index=False),
        "",
        "## Selective prediction",
        "",
        selective.to_markdown(index=False),
        "",
        "## Transferred conformal prediction sets",
        "",
        conformal.to_markdown(index=False),
        "",
        "## Source-isolated evidence status",
        "",
        evidence.to_markdown(index=False),
        "",
        "## Source-isolated agreement",
        "",
        source.to_markdown(index=False),
        "",
        "## Blocker challenge",
        "",
        blocker.to_markdown(index=False),
        "",
        "The evidence groundedness check verifies that every generated claim points "
        "to a retrieved record and to an immutable record ID in the source table. "
        "The provided expert-review sample is intentionally unscored until domain "
        "experts complete it; no synthetic expert judgement is reported.",
    ]
    if label_structure_path.exists():
        lines.extend([
            "",
            "## Four-class cohort label structure",
            "",
            pd.read_csv(label_structure_path).to_markdown(index=False),
        ])
    if sensitivity_path.exists() and sensitivity_path.stat().st_size:
        sensitivity = pd.read_csv(sensitivity_path)
        if len(sensitivity):
            lines.extend([
                "",
                "## Target-context sensitivity",
                "",
                sensitivity.to_markdown(index=False),
            ])
    (output_dir / "reliability_evidence_audit.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _write_result_figures(output_dir):
    """Create compact manuscript-ready reliability and evidence figures."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    predictions = pd.read_csv(output_dir / "external_calibrated_predictions.csv")
    evidence = pd.read_csv(output_dir / "evidence_status_summary.csv")
    class_names = [
        column.replace("calibrated_score__", "")
        for column in predictions.columns
        if column.startswith("calibrated_score__")
    ]
    score_columns = ["calibrated_score__%s" % name for name in class_names]

    split_order = [
        value
        for value in ("random_pair", "drug_disjoint", "target_disjoint", "double_disjoint")
        if value in set(predictions["split"].astype(str))
    ]
    labels = {
        "random_pair": "Random pair",
        "drug_disjoint": "Drug-disjoint",
        "target_disjoint": "Target-disjoint",
        "double_disjoint": "Double-disjoint",
    }
    colours = {
        "random_pair": "#2E86AB",
        "drug_disjoint": "#F18F01",
        "target_disjoint": "#4C956C",
        "double_disjoint": "#C73E1D",
    }

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    fine_coverages = np.linspace(0.5, 1.0, 51)
    for split in split_order:
        curves = []
        for _, group in predictions.loc[predictions["split"] == split].groupby("repeat"):
            curves.append(
                _risk_coverage_rows(
                    group[score_columns].to_numpy(float),
                    group["true_label"],
                    class_names,
                    fine_coverages,
                )
            )
        macro = np.mean([[row["macro_f1"] for row in curve] for curve in curves], axis=0)
        risk = np.mean(
            [[row["retained_error_rate"] for row in curve] for curve in curves], axis=0
        )
        effective_coverage = [row["coverage"] for row in curves[0]]
        axes[0].plot(
            effective_coverage,
            macro,
            label=labels[split],
            color=colours[split],
            linewidth=2.0,
        )
        axes[1].plot(
            effective_coverage,
            risk,
            label=labels[split],
            color=colours[split],
            linewidth=2.0,
        )
    axes[0].set(xlabel="Retained coverage", ylabel="Macro-F1", xlim=(0.5, 1.0))
    axes[1].set(xlabel="Retained coverage", ylabel="Error rate", xlim=(0.5, 1.0))
    axes[0].set_title("Selective performance")
    axes[1].set_title("Risk-coverage")
    for axis in axes:
        axis.grid(alpha=0.25, linewidth=0.6)
    axes[0].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / "selective_prediction_curves.pdf", bbox_inches="tight")
    figure.savefig(output_dir / "selective_prediction_curves.png", dpi=220, bbox_inches="tight")
    plt.close(figure)

    statuses = [
        "supported", "partially_supported", "ontology_disagreement", "mixed",
        "contradicted", "insufficient"
    ]
    status_labels = {
        "supported": "Supported",
        "partially_supported": "Partially supported",
        "ontology_disagreement": "Ontology disagreement",
        "mixed": "Mixed",
        "contradicted": "Contradicted",
        "insufficient": "Insufficient",
    }
    status_colours = {
        "supported": "#2A9D8F",
        "partially_supported": "#8AB17D",
        "ontology_disagreement": "#E9C46A",
        "mixed": "#F4A261",
        "contradicted": "#E76F51",
        "insufficient": "#9A9A9A",
    }
    pivot = evidence.pivot(index="split", columns="evidence_status", values="fraction").fillna(0.0)
    figure, axis = plt.subplots(figsize=(8.2, 3.8))
    left = np.zeros(len(split_order))
    for status in statuses:
        values = np.asarray(
            [pivot.loc[split, status] if status in pivot.columns else 0.0 for split in split_order]
        )
        axis.barh(
            [labels[split] for split in split_order],
            values,
            left=left,
            label=status_labels[status],
            color=status_colours[status],
            edgecolor="white",
            linewidth=0.4,
        )
        left += values
    axis.set(xlabel="Fraction of cpg0004 pairs", xlim=(0.0, 1.0))
    axis.set_title("Source-isolated DrugBank/Open Targets action agreement")
    axis.grid(axis="x", alpha=0.2, linewidth=0.6)
    axis.legend(
        frameon=False,
        fontsize=7.5,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
    )
    figure.tight_layout()
    figure.savefig(output_dir / "independent_evidence_status.pdf", bbox_inches="tight")
    figure.savefig(output_dir / "independent_evidence_status.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def run_action_reliability(config, logger):
    section = config.get("action_reliability", {})
    paths = config.get("paths", {})
    output_dir = ensure_dir(Path(section.get("output_dir", paths.get("reports_dir", "reports"))))
    class_names = tuple(section.get("class_names", DEFAULT_CLASSES))
    logger.info("Running post-lock action reliability analysis")
    calibrated, _ = evaluate_reliability(
        section["development_oof_predictions"],
        section["external_oof_predictions"],
        output_dir=output_dir,
        development_double_oof=section.get("development_double_oof_predictions"),
        strategy=section.get("strategy", "pair_conditioned_gate"),
        class_names=class_names,
        calibration_fraction=float(section.get("calibration_fraction", 0.5)),
        conformal_alphas=tuple(section.get("conformal_alphas", (0.1, 0.2))),
        selective_coverages=tuple(section.get("selective_coverages", (1.0, 0.9, 0.8))),
        ece_bins=int(section.get("ece_bins", 10)),
    )
    relation_results = evaluate_relation_attribution(
        section["external_view_attribution"],
        section["external_routing_cases"],
        output_dir=output_dir,
    )
    audit, source_metrics = evaluate_independent_evidence(
        calibrated,
        section["pair_action_evidence"],
        section["external_view_attribution"],
        output_dir=output_dir,
        kag_question_records=section.get("kag_question_records"),
    )
    evaluate_blocker_challenge(calibrated, audit, output_dir=output_dir)
    label_structure, sensitivity = evaluate_target_context_sensitivity(
        section["external_oof_predictions"],
        output_dir=output_dir,
        strategy=section.get("strategy", "pair_conditioned_gate"),
        routing_repeat_metrics=section.get("routing_repeat_metrics"),
        drug_only_repeat_metrics=section.get("drug_only_repeat_metrics"),
    )
    _write_markdown_summary(output_dir)
    _write_result_figures(output_dir)
    output_artifacts = {
        path.name: _file_fingerprint(path)
        for path in sorted(output_dir.iterdir())
        if path.is_file() and path.name != "action_reliability_manifest.json"
    }
    input_files = _configured_input_fingerprints(section)
    config_path = config.get("_meta", {}).get("config_path")
    if config_path and Path(config_path).exists():
        input_files["analysis_config"] = _file_fingerprint(config_path)
    manifest = {
        "analysis": "locked_four_class_action_reliability_and_evidence_audit",
        "strategy": section.get("strategy", "pair_conditioned_gate"),
        "class_names": list(class_names),
        "development_oof_predictions": str(section["development_oof_predictions"]),
        "development_double_oof_predictions": section.get(
            "development_double_oof_predictions"
        ),
        "external_oof_predictions": str(section["external_oof_predictions"]),
        "source_isolated_evidence_sources": sorted(INDEPENDENT_SOURCES),
        "kag_question_records": section.get("kag_question_records"),
        "calibration_policy": (
            "Temperature and conformal partitions use only development OOF predictions. "
            "A split-matched development double-disjoint file is used when present; "
            "otherwise the prespecified pooled drug- and target-disjoint fallback is used."
        ),
        "attribution_claim_scope": (
            "Exact component deletion in the locked routed mixture, not edge-level "
            "causal attribution inside RGCN."
        ),
        "evidence_claim_scope": (
            "Post-prediction source-isolated verification; evidence records are not "
            "model inputs, are not an independent outcome cohort, and do not prove a "
            "molecular mechanism."
        ),
        "human_expert_relevance_status": (
            "Protocol and blinded sample generated; ratings intentionally pending."
        ),
        "input_files": input_files,
        "output_files": sorted(output_artifacts),
        "output_artifacts": output_artifacts,
        "relation_pair_records": int(len(relation_results["stability"])),
        "evidence_audit_records": int(len(audit)),
        "source_metric_records": int(len(source_metrics)),
        "label_structure_records": int(len(label_structure)),
        "target_context_sensitivity_records": int(len(sensitivity)),
    }
    write_json(output_dir / "action_reliability_manifest.json", manifest)
    logger.info("Action reliability reports written to %s", output_dir)
    return manifest
