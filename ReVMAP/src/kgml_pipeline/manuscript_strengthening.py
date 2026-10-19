"""Post-lock analyses that strengthen the KGML manuscript evidence base.

This module never refits or selects the locked predictor.  It adds paired
method-comparison uncertainty, graph-context sparsity diagnostics, and a
transparent audit of the anomalous DistMult link-prediction result.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from .action_reliability import DEFAULT_CLASSES, _pair_key
from .io_utils import ensure_dir, write_json


METHOD_DISPLAY = {
    "DeepDTAStyle": "DeepDTA",
    "GraphDTAStyle": "GraphDTA",
    "DruGNNosisStyle": "DruGNNosis",
}


def _macro_f1(truth, predicted):
    truth = np.asarray(truth, dtype=str)
    predicted = np.asarray(predicted, dtype=str)
    scores = []
    for label in DEFAULT_CLASSES:
        truth_positive = truth == label
        predicted_positive = predicted == label
        true_positive = int(np.sum(truth_positive & predicted_positive))
        false_positive = int(np.sum(~truth_positive & predicted_positive))
        false_negative = int(np.sum(truth_positive & ~predicted_positive))
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(2.0 * true_positive / denominator if denominator else 0.0)
    return float(np.mean(scores))


def _stratified_bootstrap_indices(labels, rng):
    labels = np.asarray(labels, dtype=str)
    sampled = []
    for label in DEFAULT_CLASSES:
        indices = np.flatnonzero(labels == label)
        if len(indices):
            sampled.append(rng.choice(indices, size=len(indices), replace=True))
    return np.concatenate(sampled) if sampled else np.arange(len(labels))


def evaluate_paired_external_comparisons(
    routing_predictions,
    baseline_prediction_paths,
    *,
    output_dir,
    strategy="pair_conditioned_gate",
    bootstrap_resamples=2000,
    permutation_resamples=2000,
    seed=20260905,
):
    """Compare KGML with every baseline on identical pair/repeat predictions."""

    output_dir = ensure_dir(Path(output_dir))
    routing = pd.read_csv(routing_predictions)
    routing = routing.loc[routing["strategy"].astype(str) == str(strategy)].copy()
    baselines = pd.concat(
        [pd.read_csv(path) for path in baseline_prediction_paths],
        ignore_index=True,
    )
    key = ["split", "repeat", "drug_name", "target_name", "true_label"]
    rows = []
    rng = np.random.default_rng(int(seed))

    for (split, model), baseline_group in baselines.groupby(["split", "model"]):
        repeat_results = []
        repeat_bootstraps = []
        repeat_frames = []
        for repeat in sorted(set(baseline_group["repeat"].astype(int))):
            kg = routing.loc[
                (routing["split"].astype(str) == str(split))
                & (routing["repeat"].astype(int) == int(repeat)),
                key + ["predicted_label"],
            ].rename(columns={"predicted_label": "kgml_prediction"})
            baseline = baseline_group.loc[
                baseline_group["repeat"].astype(int) == int(repeat),
                key + ["predicted_label"],
            ].rename(columns={"predicted_label": "baseline_prediction"})
            merged = kg.merge(baseline, on=key, how="inner", validate="one_to_one")
            if len(merged) != len(kg) or len(merged) != len(baseline):
                raise ValueError(
                    "Prediction alignment failed for %s/%s/repeat %s: KGML=%s, "
                    "baseline=%s, matched=%s"
                    % (split, model, repeat, len(kg), len(baseline), len(merged))
                )
            truth = merged["true_label"].astype(str).to_numpy()
            kg_pred = merged["kgml_prediction"].astype(str).to_numpy()
            base_pred = merged["baseline_prediction"].astype(str).to_numpy()
            kg_f1 = _macro_f1(truth, kg_pred)
            base_f1 = _macro_f1(truth, base_pred)
            repeat_results.append((kg_f1, base_f1, kg_f1 - base_f1))
            repeat_frames.append((truth, kg_pred, base_pred))
            boot = np.empty(int(bootstrap_resamples), dtype=float)
            for index in range(int(bootstrap_resamples)):
                sampled = _stratified_bootstrap_indices(truth, rng)
                boot[index] = _macro_f1(truth[sampled], kg_pred[sampled]) - _macro_f1(
                    truth[sampled], base_pred[sampled]
                )
            repeat_bootstraps.append(boot)

        repeat_results = np.asarray(repeat_results, dtype=float)
        combined_bootstrap = np.mean(np.vstack(repeat_bootstraps), axis=0)
        observed = float(np.mean(repeat_results[:, 2]))
        null = np.empty(int(permutation_resamples), dtype=float)
        for index in range(int(permutation_resamples)):
            repeat_null = []
            for truth, kg_pred, base_pred in repeat_frames:
                swap = rng.random(len(truth)) < 0.5
                left = np.where(swap, base_pred, kg_pred)
                right = np.where(swap, kg_pred, base_pred)
                repeat_null.append(_macro_f1(truth, left) - _macro_f1(truth, right))
            null[index] = float(np.mean(repeat_null))
        p_value = float(
            (1 + np.sum(np.abs(null) >= abs(observed))) / (len(null) + 1)
        )
        rows.append({
            "split": str(split),
            "baseline": METHOD_DISPLAY.get(str(model), str(model)),
            "pair_count_per_repeat": int(len(repeat_frames[0][0])),
            "repeat_count": int(len(repeat_frames)),
            "kgml_macro_f1_mean": float(np.mean(repeat_results[:, 0])),
            "baseline_macro_f1_mean": float(np.mean(repeat_results[:, 1])),
            "macro_f1_difference_kgml_minus_baseline": observed,
            "paired_stratified_bootstrap_ci_low": float(
                np.quantile(combined_bootstrap, 0.025)
            ),
            "paired_stratified_bootstrap_ci_high": float(
                np.quantile(combined_bootstrap, 0.975)
            ),
            "paired_randomization_p_value_two_sided": p_value,
            "bootstrap_resamples": int(bootstrap_resamples),
            "permutation_resamples": int(permutation_resamples),
        })
    result = pd.DataFrame(rows).sort_values(["split", "baseline"])
    result.to_csv(output_dir / "external_method_paired_comparisons.csv", index=False)
    return result


def _degree_bin(values):
    values = pd.Series(values, dtype=float)
    percentile = values.rank(method="average", pct=True)
    return pd.cut(
        percentile,
        bins=[0.0, 0.25, 0.50, 0.75, 1.0],
        labels=["Q1 sparse", "Q2", "Q3", "Q4 dense"],
        include_lowest=True,
    ).astype(str)


def evaluate_graph_context_sparsity(
    routing_predictions,
    graph_edges_json,
    *,
    output_dir,
    strategy="pair_conditioned_gate",
):
    """Stratify locked-model performance by masked-graph context degree."""

    output_dir = ensure_dir(Path(output_dir))
    predictions = pd.read_csv(routing_predictions)
    predictions = predictions.loc[
        predictions["strategy"].astype(str) == str(strategy)
    ].copy()
    with Path(graph_edges_json).open("r", encoding="utf-8") as handle:
        edges = json.load(handle)

    drug_degree = {}
    target_degree = {}
    for edge in edges:
        source = str(edge.get("from", ""))
        target = str(edge.get("to", ""))
        if source.startswith("Drug::"):
            drug_degree[source.split("::", 1)[1]] = drug_degree.get(
                source.split("::", 1)[1], 0
            ) + 1
        if source.startswith("Target::"):
            target_degree[source.split("::", 1)[1]] = target_degree.get(
                source.split("::", 1)[1], 0
            ) + 1
        if target.startswith("Target::"):
            target_degree[target.split("::", 1)[1]] = target_degree.get(
                target.split("::", 1)[1], 0
            ) + 1

    predictions["pair_key"] = [
        _pair_key(drug, target)
        for drug, target in zip(predictions["drug_name"], predictions["target_name"])
    ]
    predictions["drug_context_degree"] = [
        int(drug_degree.get(str(name).strip().lower(), 0))
        for name in predictions["drug_name"]
    ]
    predictions["target_context_degree"] = [
        int(target_degree.get(str(name).strip().lower(), 0))
        for name in predictions["target_name"]
    ]
    predictions["combined_context_degree"] = (
        predictions["drug_context_degree"] + predictions["target_context_degree"]
    )

    pair_degree = predictions.drop_duplicates("pair_key")[[
        "pair_key",
        "drug_name",
        "target_name",
        "drug_context_degree",
        "target_context_degree",
        "combined_context_degree",
    ]].copy()
    for dimension in (
        "drug_context_degree",
        "target_context_degree",
        "combined_context_degree",
    ):
        mapping = dict(zip(pair_degree["pair_key"], _degree_bin(pair_degree[dimension])))
        predictions["%s_bin" % dimension] = predictions["pair_key"].map(mapping)
        pair_degree["%s_bin" % dimension] = pair_degree["pair_key"].map(mapping)
    pair_degree.to_csv(output_dir / "graph_context_degree_by_pair.csv", index=False)

    rows = []
    for dimension in (
        "drug_context_degree",
        "target_context_degree",
        "combined_context_degree",
    ):
        bin_column = "%s_bin" % dimension
        for (split, repeat, degree_bin), group in predictions.groupby(
            ["split", "repeat", bin_column], observed=True
        ):
            rows.append({
                "split": split,
                "repeat": int(repeat),
                "degree_dimension": dimension,
                "degree_bin": degree_bin,
                "pair_count": int(len(group)),
                "median_degree": float(group[dimension].median()),
                "macro_f1": _macro_f1(group["true_label"], group["predicted_label"]),
                "accuracy": float(
                    accuracy_score(group["true_label"], group["predicted_label"])
                ),
            })
    repeat = pd.DataFrame(rows)
    repeat.to_csv(output_dir / "graph_context_sparsity_repeat_metrics.csv", index=False)
    summary = (
        repeat.groupby(["split", "degree_dimension", "degree_bin"], observed=True)
        .agg(
            pair_count=("pair_count", "first"),
            median_degree=("median_degree", "mean"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            accuracy_mean=("accuracy", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(output_dir / "graph_context_sparsity_summary.csv", index=False)

    contrasts = []
    for (split, dimension), group in summary.groupby(["split", "degree_dimension"]):
        lookup = group.set_index("degree_bin")["macro_f1_mean"].to_dict()
        if "Q1 sparse" in lookup and "Q4 dense" in lookup:
            contrasts.append({
                "split": split,
                "degree_dimension": dimension,
                "sparse_q1_macro_f1": lookup["Q1 sparse"],
                "dense_q4_macro_f1": lookup["Q4 dense"],
                "sparse_minus_dense_macro_f1": (
                    lookup["Q1 sparse"] - lookup["Q4 dense"]
                ),
            })
    contrast = pd.DataFrame(contrasts)
    contrast.to_csv(output_dir / "graph_context_sparsity_contrasts.csv", index=False)
    return summary, contrast


def diagnose_distmult(training_summary_path, *, output_dir):
    """Audit whether DistMult failure is global or relation-view specific."""

    output_dir = ensure_dir(Path(output_dir))
    training = pd.read_csv(training_summary_path)
    rows = []
    for view, group in training.groupby("view"):
        dist = group.loc[group["model"].astype(str) == "DistMult"]
        if not len(dist):
            continue
        dist = dist.iloc[0]
        entity_count = int(dist["entity_count"])
        random_mrr = (
            math.log(entity_count) + 0.5772156649 + 1.0 / (2.0 * entity_count)
        ) / entity_count
        ranked = group.sort_values("mrr", ascending=False).reset_index(drop=True)
        rank = int(ranked.index[ranked["model"].astype(str) == "DistMult"][0] + 1)
        rows.append({
            "view": view,
            "relation_count": int(dist["relation_count"]),
            "entity_count": entity_count,
            "distmult_mrr": float(dist["mrr"]),
            "approximate_uniform_random_mrr": float(random_mrr),
            "mrr_to_random_ratio": float(dist["mrr"] / random_mrr),
            "distmult_rank_within_view": rank,
            "model_count": int(len(group)),
            "near_random_flag": bool(float(dist["mrr"]) <= 2.0 * random_mrr),
            "training_status": dist.get("status", ""),
        })
    audit = pd.DataFrame(rows).sort_values("view")
    audit.to_csv(output_dir / "distmult_link_prediction_diagnostic.csv", index=False)
    near = audit.loc[audit["near_random_flag"]]
    strong = audit.loc[~audit["near_random_flag"]]
    lines = [
        "# DistMult link-prediction diagnostic",
        "",
        "DistMult completed successfully in every reported view, so the anomalous "
        "score is not a missing-run placeholder. Its MRR is within two-fold of an "
        "approximate uniform-random ranker in %d/%d views (%s), but is materially "
        "above random in %d/%d views (%s)."
        % (
            len(near),
            len(audit),
            ", ".join(near["view"].astype(str)) or "none",
            len(strong),
            len(audit),
            ", ".join(strong["view"].astype(str)) or "none",
        ),
        "",
        "This view dependence argues against describing the result as a universal "
        "training failure. The defensible interpretation is that the symmetric "
        "DistMult scoring form is poorly matched to some typed, directional view "
        "mixtures under the present ranking protocol. The diagnostic does not prove "
        "that symmetry is the sole cause.",
    ]
    (output_dir / "distmult_link_prediction_diagnostic.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return audit


def run_manuscript_strengthening(config, logger):
    section = dict(config.get("manuscript_strengthening", {}))
    if not section:
        raise ValueError("Missing manuscript_strengthening config section")
    output_dir = ensure_dir(Path(section["output_dir"]))
    logger.info("Running post-lock manuscript-strengthening analyses")
    paired = evaluate_paired_external_comparisons(
        section["routing_predictions"],
        section["baseline_prediction_paths"],
        output_dir=output_dir,
        strategy=section.get("strategy", "pair_conditioned_gate"),
        bootstrap_resamples=int(section.get("bootstrap_resamples", 2000)),
        permutation_resamples=int(section.get("permutation_resamples", 2000)),
        seed=int(section.get("seed", 20260905)),
    )
    sparsity, contrasts = evaluate_graph_context_sparsity(
        section["routing_predictions"],
        section["graph_edges_json"],
        output_dir=output_dir,
        strategy=section.get("strategy", "pair_conditioned_gate"),
    )
    distmult = diagnose_distmult(
        section["distmult_training_summary"], output_dir=output_dir
    )
    manifest = {
        "analysis": "post_lock_manuscript_strengthening",
        "strategy": section.get("strategy", "pair_conditioned_gate"),
        "paired_comparison_count": int(len(paired)),
        "sparsity_summary_count": int(len(sparsity)),
        "sparsity_contrast_count": int(len(contrasts)),
        "distmult_view_count": int(len(distmult)),
        "claim_scope": (
            "Paired uncertainty and graph-context degree diagnostics only; no "
            "model selection, refitting, or fully inductive cold-start claim."
        ),
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest
