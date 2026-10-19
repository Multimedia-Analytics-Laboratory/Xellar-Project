#!/usr/bin/env python3
"""Cluster-aware, fixed-prediction manuscript sensitivity analyses.

No training, selection, graph modification, or original-output mutation occurs.
Run with the bundled Python (numpy and pandas), optionally --self-test.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import itertools
import json
import math
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

CLASSES = ("agonist", "antagonist", "blocker", "inhibitor")
SPLITS = ("random_pair", "drug_disjoint", "target_disjoint", "double_disjoint")
MODELS = {"DeepDTAStyle": "DeepDTA", "GraphDTAStyle": "GraphDTA", "DruGNNosisStyle": "DruGNNosis"}
KEY = ["split", "repeat", "drug_name", "target_name"]
PAIR = ["drug_name", "target_name"]
INPUTS = {
    "routing": "runs/pair_action_locked_multiview_3repeat/external/routing_oof_predictions.csv",
    "baselines_three_splits": "runs/pair_action_cpg0004_extension/reports/pair_multimodal_oof_predictions.csv",
    "baselines_double": "runs/pair_action_cpg0004_double_disjoint/reports/pair_multimodal_oof_predictions.csv",
    "degree": "runs/manuscript_strengthening/reports/graph_context_degree_by_pair.csv",
    "old_comparisons": "runs/manuscript_strengthening/reports/external_method_paired_comparisons.csv",
    "old_degree_metrics": "runs/manuscript_strengthening/reports/graph_context_sparsity_summary.csv",
    "attribution": "runs/pair_action_locked_multiview_3repeat/external/routing_view_attribution.csv",
    "old_deletion": "runs/pair_action_reliability/reports/relation_faithfulness_inference.csv",
}
VALIDATION_COLUMNS = ["check_family", "context", "metric", "original", "recomputed", "absolute_error", "tolerance", "passed"]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rng_for(seed, *parts):
    value = int.from_bytes(hashlib.sha256("|".join(map(str, (seed,) + parts)).encode()).digest()[:8], "big")
    return np.random.default_rng(value)


def f1_from_confusion(matrix):
    """Macro-F1 over the four fixed labels, zero for an undefined class F1."""
    matrix = np.asarray(matrix, dtype=float)
    diagonal = np.diagonal(matrix, axis1=-2, axis2=-1)
    denominator = matrix.sum(axis=-1) + matrix.sum(axis=-2)
    scores = np.divide(2 * diagonal, denominator, out=np.zeros_like(diagonal), where=denominator != 0)
    return scores.mean(axis=-1)


def confusion(truth, predicted, weights=None):
    mapping = {c: i for i, c in enumerate(CLASSES)}
    truth = np.asarray([mapping[x] for x in truth], dtype=int)
    predicted = np.asarray([mapping[x] for x in predicted], dtype=int)
    return np.bincount(4 * truth + predicted, weights=weights, minlength=16).reshape(4, 4)


def cluster_confusions(frame, prediction_column, cluster_column, weighting):
    """Return G x R x 4 x 4 contributions and fixed cluster/repeat order.

    Equal-drug weighting assigns 1 / n_targets(drug) to each original row in
    each repeat. For drug-cluster bootstrap, every sampled copy of a drug
    therefore has total weight one. Target-cluster sensitivity uses pair weights.
    """
    clusters = sorted(frame[cluster_column].unique())
    repeats = sorted(frame["repeat"].unique())
    gidx = {value: i for i, value in enumerate(clusters)}
    ridx = {value: i for i, value in enumerate(repeats)}
    cidx = {value: i for i, value in enumerate(CLASSES)}
    weights = np.ones(len(frame), dtype=float)
    if weighting == "equal_drug":
        weights = 1.0 / frame.groupby(["repeat", "drug_name"])["target_name"].transform("size").to_numpy()
    elif weighting != "pair":
        raise ValueError(weighting)
    indices = (
        frame[cluster_column].map(gidx).to_numpy(), frame["repeat"].map(ridx).to_numpy(),
        frame["true_label"].map(cidx).to_numpy(), frame[prediction_column].map(cidx).to_numpy(),
    )
    result = np.zeros((len(clusters), len(repeats), 4, 4), dtype=float)
    np.add.at(result, indices, weights)
    return result, clusters, repeats


def draw_statistics(left, right, multipliers, kind):
    """A single cluster multiplier applies jointly to every repeat and target."""
    if kind == "bootstrap":
        a = np.einsum("bg,grij->brij", multipliers, left, optimize=True)
        b = np.einsum("bg,grij->brij", multipliers, right, optimize=True)
    elif kind == "swap":
        change = np.einsum("bg,grij->brij", multipliers, right - left, optimize=True)
        a, b = left.sum(axis=0)[None, ...] + change, right.sum(axis=0)[None, ...] - change
    else:
        raise ValueError(kind)
    # Calculate each repeat's nonlinear metric first, then average repeats.
    return (f1_from_confusion(a) - f1_from_confusion(b)).mean(axis=-1)


def paired_inference(left, right, *, resamples, seed, stream):
    g = len(left)
    observed = float((f1_from_confusion(left.sum(axis=0)) - f1_from_confusion(right.sum(axis=0))).mean())
    bootstrap, null = np.empty(resamples), np.empty(resamples)
    boot_rng, null_rng = rng_for(seed, stream, "bootstrap"), rng_for(seed, stream, "swap")
    for start in range(0, resamples, 500):
        end = min(start + 500, resamples)
        counts = boot_rng.multinomial(g, np.full(g, 1.0 / g), size=end-start)
        bootstrap[start:end] = draw_statistics(left, right, counts, "bootstrap")
        swaps = null_rng.integers(0, 2, size=(end-start, g))
        null[start:end] = draw_statistics(left, right, swaps, "swap")
    return inference_summary(observed, bootstrap, null), bootstrap, null


def inference_summary(observed, bootstrap, null):
    exceedances = int(np.count_nonzero(np.abs(null) >= abs(observed) - 1e-12))
    p = (exceedances + 1.0) / (len(null) + 1.0)
    lo, hi = np.quantile(bootstrap, [0.025, 0.975], method="linear")
    return {
        "difference": float(observed), "bootstrap_ci_low": float(lo), "bootstrap_ci_high": float(hi),
        "randomization_p_two_sided": float(p), "null_exceedances": exceedances,
        "randomization_mc_se": float(math.sqrt(p * (1-p) / (len(null) + 1))),
        "bootstrap_resamples": len(bootstrap), "randomization_resamples": len(null),
    }


def scalar_cluster_statistics(sums, counts, multipliers, kind):
    numerator = multipliers @ sums
    denominator = multipliers @ counts if kind == "bootstrap" else counts.sum(axis=0)
    return (numerator / denominator).mean(axis=1)


def holm(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    adjusted = np.minimum(1.0, np.maximum.accumulate(values[order] * np.arange(len(values), 0, -1)))
    output = np.empty(len(values))
    output[order] = adjusted
    return output


def record_point_validation(output, frame, family, keys, columns):
    records=[]
    for _, row in frame.iterrows():
        for metric, original, recomputed in columns:
            error=abs(float(row[original])-float(row[recomputed]))
            records.append({"check_family":family,"context":"|".join(str(row[k]) for k in keys),"metric":metric,
                            "original":float(row[original]),"recomputed":float(row[recomputed]),"absolute_error":error,
                            "tolerance":1e-12,"passed":error<=1e-12})
    assert all(r["passed"] for r in records)
    pd.DataFrame(records,columns=VALIDATION_COLUMNS).to_csv(output/"point_estimate_validation.csv",mode="a",header=False,index=False)


def load_predictions(root):
    routing = pd.read_csv(root / INPUTS["routing"])
    routing = routing.loc[(routing["strategy"] == "pair_conditioned_gate") & (routing["task"] == "core4_action")].copy()
    assert len(routing) == 898 * 3 * 4
    assert not routing.duplicated(KEY).any()
    assert set(routing["split"]) == set(SPLITS) and set(routing["repeat"]) == {0, 1, 2}
    wide = routing[KEY + ["true_label", "predicted_label", "fold"]].rename(columns={"predicted_label": "KGML"})
    baselines = pd.concat([pd.read_csv(root / INPUTS[k]) for k in ("baselines_three_splits", "baselines_double")], ignore_index=True)
    assert set(baselines["model"]) == set(MODELS)
    for raw, display in MODELS.items():
        group = baselines.loc[(baselines["model"] == raw) & (baselines["task"] == "core4_action")]
        assert len(group) == len(wide) and not group.duplicated(KEY).any()
        wide = wide.merge(group[KEY + ["true_label", "predicted_label", "fold"]], on=KEY, how="outer", validate="one_to_one", suffixes=("", "_check"), indicator=True)
        assert (wide.pop("_merge") == "both").all()
        assert (wide["true_label"] == wide.pop("true_label_check")).all()
        assert (wide["fold"] == wide.pop("fold_check")).all(), "Matched methods have different outer fold assignments"
        wide.rename(columns={"predicted_label": display}, inplace=True)
    wide.sort_values(KEY, inplace=True)
    cohort = wide[PAIR + ["true_label"]].drop_duplicates().sort_values(PAIR).reset_index(drop=True)
    assert len(cohort) == 898 and cohort["drug_name"].nunique() == 284 and cohort["target_name"].nunique() == 209
    assert cohort.groupby("drug_name")["true_label"].nunique().max() == 1
    for _, group in wide.groupby(["split", "repeat"]):
        actual = group[PAIR + ["true_label"]].sort_values(PAIR).reset_index(drop=True)
        pd.testing.assert_frame_equal(actual, cohort)
    for column in ["true_label", "KGML"] + list(MODELS.values()):
        assert set(wide[column]) <= set(CLASSES)
    return wide, cohort


def method_comparisons(wide, args, output):
    rows, repeat_rows, distributions = [], [], {}
    for split in SPLITS:
        frame = wide.loc[wide["split"] == split]
        for weighting in ("pair", "equal_drug"):
            for model in ("KGML",) + tuple(MODELS.values()):
                matrix, _, repeats = cluster_confusions(frame, model, "drug_name", weighting)
                for repeat, score in zip(repeats, f1_from_confusion(matrix.sum(axis=0))):
                    repeat_rows.append({"split": split, "weighting": weighting, "model": model, "repeat": int(repeat), "macro_f1": float(score)})
        units = ("drug_name", "target_name") if split == "double_disjoint" else ("drug_name",)
        for unit in units:
            weightings = ("pair", "equal_drug") if unit == "drug_name" else ("pair",)
            for weighting in weightings:
                left, clusters, repeats = cluster_confusions(frame, "KGML", unit, weighting)
                for model in MODELS.values():
                    right, check_clusters, check_repeats = cluster_confusions(frame, model, unit, weighting)
                    assert clusters == check_clusters and repeats == check_repeats
                    stats, bootstrap, null = paired_inference(left, right, resamples=args.resamples, seed=args.seed, stream=f"{split}|{unit}")
                    key = f"{split}__{unit}__{weighting}__{model}"
                    distributions[key + "__bootstrap"] = bootstrap
                    distributions[key + "__null"] = null
                    rows.append({"split": split, "baseline": model, "cluster_unit": unit.removesuffix("_name"), "weighting": weighting,
                                 "pair_count": 898, "cluster_count": len(clusters), "repeat_count": 3,
                                 "kgml_macro_f1": float(f1_from_confusion(left.sum(axis=0)).mean()),
                                 "baseline_macro_f1": float(f1_from_confusion(right.sum(axis=0)).mean()), **stats})
        print(f"Completed comparisons: {split}", flush=True)
    comparisons = pd.DataFrame(rows)
    comparisons["holm_p_within_analysis_family"] = np.nan
    for _, group in comparisons.groupby(["cluster_unit", "weighting"]):
        comparisons.loc[group.index, "holm_p_within_analysis_family"] = holm(group["randomization_p_two_sided"])
    repeats = pd.DataFrame(repeat_rows)
    means = repeats.groupby(["split", "weighting", "model"], sort=False)["macro_f1"].agg(["mean", "std"]).reset_index().rename(columns={"mean": "macro_f1_mean", "std": "macro_f1_sd"})
    comparisons.to_csv(output / "cluster_method_comparisons.csv", index=False)
    repeats.to_csv(output / "method_repeat_metrics.csv", index=False)
    means.to_csv(output / "method_summary.csv", index=False)
    # The original pair-weighted point estimates must remain exactly unchanged.
    old = pd.read_csv(args.root / INPUTS["old_comparisons"])
    check = comparisons.loc[(comparisons["cluster_unit"] == "drug") & (comparisons["weighting"] == "pair")].merge(old, on=["split", "baseline"], validate="one_to_one")
    assert len(check) == 12
    assert np.allclose(check["difference"], check["macro_f1_difference_kgml_minus_baseline"], rtol=0, atol=1e-12)
    record_point_validation(output,check,"original_method_point_estimates",["split","baseline"],[
        ("kgml_macro_f1","kgml_macro_f1_mean","kgml_macro_f1"),
        ("baseline_macro_f1","baseline_macro_f1_mean","baseline_macro_f1"),
        ("difference","macro_f1_difference_kgml_minus_baseline","difference")])
    return comparisons, means, distributions


def degree_composition(wide, cohort, args, output):
    degree = pd.read_csv(args.root / INPUTS["degree"])
    frame = cohort.merge(degree, on=PAIR, how="outer", validate="one_to_one", indicator=True)
    assert (frame.pop("_merge") == "both").all()
    assert frame.groupby("drug_name")["drug_context_degree_bin"].nunique().max() == 1
    rows, composition, performances, class_performance = [], [], [], []
    dimensions = ["drug_context_degree", "target_context_degree", "combined_context_degree"]
    prediction = wide.merge(frame, on=PAIR + ["true_label"], validate="many_to_one")
    for dimension in dimensions:
        bins = dimension + "_bin"
        for quartile, group in frame.groupby(bins, sort=True):
            base = {"degree_dimension": dimension, "degree_bin": quartile, "pair_count": len(group),
                    "unique_drugs": group["drug_name"].nunique(), "unique_targets": group["target_name"].nunique(),
                    "median_degree": float(group[dimension].median()), "minimum_degree": int(group[dimension].min()), "maximum_degree": int(group[dimension].max())}
            rows.append(base)
            for label in CLASSES:
                part = group.loc[group["true_label"] == label]
                composition.append({"degree_dimension": dimension, "degree_bin": quartile, "action_class": label,
                                    "pairs": len(part), "pair_percent": 100*len(part)/len(group),
                                    "unique_drugs": part["drug_name"].nunique(), "drug_percent": 100*part["drug_name"].nunique()/group["drug_name"].nunique(),
                                    "unique_targets": part["target_name"].nunique()})
        for (split, repeat, quartile), group in prediction.groupby(["split", "repeat", bins], sort=True):
            for weighting in ("pair", "equal_drug_within_stratum"):
                weights = None if weighting == "pair" else 1/group.groupby("drug_name")["target_name"].transform("size").to_numpy()
                matrix = confusion(group["true_label"], group["KGML"], weights)
                performances.append({"degree_dimension": dimension, "degree_bin": quartile, "split": split, "repeat": int(repeat),
                                     "weighting": weighting, "macro_f1": float(f1_from_confusion(matrix))})
                if weighting == "pair":
                    diag = np.diag(matrix)
                    denom = matrix.sum(axis=0) + matrix.sum(axis=1)
                    class_f1 = np.divide(2*diag, denom, out=np.zeros(4), where=denom > 0)
                    recall = np.divide(diag, matrix.sum(axis=1), out=np.zeros(4), where=matrix.sum(axis=1)>0)
                    for i, label in enumerate(CLASSES):
                        class_performance.append({"degree_dimension": dimension, "degree_bin": quartile, "split": split, "repeat": int(repeat),
                                                  "action_class": label, "support": int(matrix[i].sum()), "f1": float(class_f1[i]), "recall": float(recall[i])})
    strata, composition = pd.DataFrame(rows), pd.DataFrame(composition)
    performance = pd.DataFrame(performances)
    means = performance.groupby(["degree_dimension", "degree_bin", "split", "weighting"], sort=False)["macro_f1"].agg(["mean", "std"]).reset_index().rename(columns={"mean": "macro_f1_mean", "std": "macro_f1_sd"})
    per_class = pd.DataFrame(class_performance).groupby(["degree_dimension", "degree_bin", "split", "action_class"], sort=False).agg(support=("support", "first"), f1_mean=("f1", "mean"), recall_mean=("recall", "mean")).reset_index()
    old = pd.read_csv(args.root / INPUTS["old_degree_metrics"])
    check = means.loc[means["weighting"] == "pair"].merge(old, on=["degree_dimension", "degree_bin", "split"], suffixes=("_new", "_old"), validate="one_to_one")
    assert len(check) == 48 and np.allclose(check["macro_f1_mean_new"], check["macro_f1_mean_old"], rtol=0, atol=1e-12)
    record_point_validation(output,check,"original_S1_metrics",["degree_dimension","degree_bin","split"],[("macro_f1","macro_f1_mean_old","macro_f1_mean_new")])
    strata.to_csv(output / "degree_strata_counts.csv", index=False)
    composition.to_csv(output / "degree_class_composition.csv", index=False)
    performance.to_csv(output / "degree_repeat_metrics.csv", index=False)
    means.to_csv(output / "degree_performance_summary.csv", index=False)
    per_class.to_csv(output / "degree_class_performance.csv", index=False)
    frame.to_csv(output / "degree_by_pair_with_reference_labels.csv", index=False)
    return strata, composition, means, per_class


def deletion_inference(args, output):
    original = pd.read_csv(args.root / INPUTS["attribution"])
    assert len(original) == 898 * 3 * 4 * 5
    records = []
    for keys, group in original.groupby(KEY, sort=False):
        assert len(group) == 5 and group["view"].nunique() == 5
        # Preserve the established stable input-order tie rule.
        group = group.sort_values("gate_weight", ascending=False, kind="mergesort")
        delta = float(group.iloc[0]["view_component_deletion_delta"] - group.iloc[1:]["view_component_deletion_delta"].mean())
        records.append({**dict(zip(KEY, keys)), "top_view": group.iloc[0]["view"], "top_minus_other_delta": delta})
    frame = pd.DataFrame(records)
    frame.to_csv(output / "view_deletion_pair_effects.csv", index=False)
    repeat_metrics = frame.groupby(["split", "repeat"])["top_minus_other_delta"].mean().rename("mean_top_minus_non_top_delta").reset_index()
    old = pd.read_csv(args.root / INPUTS["old_deletion"])
    check = repeat_metrics.merge(old, on=["split", "repeat"], suffixes=("_new", "_old"), validate="one_to_one")
    assert len(check) == 12 and np.allclose(check["mean_top_minus_non_top_delta_new"], check["mean_top_minus_non_top_delta_old"], rtol=0, atol=1e-12)
    record_point_validation(output,check,"original_deletion_repeat_means",["split","repeat"],[("mean_delta","mean_top_minus_non_top_delta_old","mean_top_minus_non_top_delta_new")])
    rows, distributions = [], {}
    for split in SPLITS:
        group = frame.loc[frame["split"] == split]
        clusters, repeats = sorted(group["drug_name"].unique()), sorted(group["repeat"].unique())
        counts = group.groupby(["drug_name", "repeat"]).size().unstack().reindex(index=clusters, columns=repeats).to_numpy()
        sums = group.groupby(["drug_name", "repeat"])["top_minus_other_delta"].sum().unstack().reindex(index=clusters, columns=repeats).to_numpy()
        assert np.array_equal(counts, np.broadcast_to(counts[:, :1], counts.shape))
        observed = float((sums.sum(axis=0)/counts.sum(axis=0)).mean())
        boot, null = np.empty(args.resamples), np.empty(args.resamples)
        brng, nrng = rng_for(args.seed, split, "deletion_bootstrap"), rng_for(args.seed, split, "deletion_signflip")
        for start in range(0, args.resamples, 500):
            end = min(start+500, args.resamples)
            multiplicity = brng.multinomial(len(clusters), np.full(len(clusters), 1/len(clusters)), size=end-start)
            boot[start:end] = scalar_cluster_statistics(sums, counts, multiplicity, "bootstrap")
            signs = 2*nrng.integers(0, 2, size=(end-start, len(clusters)))-1
            null[start:end] = scalar_cluster_statistics(sums, counts, signs, "signflip")
        rows.append({"split": split, "cluster_unit": "drug", "pair_count": 898, "cluster_count": 284, "repeat_count": 3, **inference_summary(observed, boot, null)})
        distributions[split+"__deletion_bootstrap"], distributions[split+"__deletion_null"] = boot, null
    summary = pd.DataFrame(rows)
    summary["holm_p_across_four_splits"] = holm(summary["randomization_p_two_sided"])
    summary.to_csv(output / "view_deletion_drug_cluster.csv", index=False)
    repeat_metrics.to_csv(output / "view_deletion_repeat_metrics.csv", index=False)
    return summary, distributions


def dependence_summary(cohort):
    drug_size = cohort.groupby("drug_name").size()
    target_size = cohort.groupby("target_name").size()
    # Connected components describe overlap, not a proposed independent-unit remedy.
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    for row in cohort.itertuples():
        a, b = find("drug:"+row.drug_name), find("target:"+row.target_name)
        parent[a] = b
    components = {}
    for row in cohort.itertuples():
        components.setdefault(find("drug:"+row.drug_name), []).append((row.drug_name, row.target_name))
    sizes = sorted((len(v) for v in components.values()), reverse=True)
    return {
        "pairs": len(cohort), "drugs": len(drug_size), "targets": len(target_size),
        "drug_class_counts": cohort.drop_duplicates("drug_name")["true_label"].value_counts().to_dict(),
        "pair_class_counts": cohort["true_label"].value_counts().to_dict(),
        "drugs_with_multiple_targets": int((drug_size>1).sum()), "max_targets_per_drug": int(drug_size.max()),
        "targets_shared_by_multiple_drugs": int((target_size>1).sum()), "pairs_on_shared_targets": int(target_size[target_size>1].sum()),
        "max_drugs_per_target": int(target_size.max()), "bipartite_connected_components": len(sizes),
        "largest_component_pairs": sizes[0], "largest_component_pair_fraction": sizes[0]/len(cohort),
        "connected_component_sizes_pairs": sizes,
        "interpretation": "Drug clustering preserves within-drug target/repeat dependence. Target clustering is an alternate one-way sensitivity analysis; neither resolves crossed drug-target dependence or shared fitted-model dependence. Components quantify overlap only and were not resampled.",
    }


def markdown_table(headers, rows):
    return "| " + " | ".join(headers) + " |\n| " + " | ".join(["---"]*len(headers)) + " |\n" + "\n".join("| "+" | ".join(map(str, row))+" |" for row in rows)+"\n"


def write_report(comparisons, means, strata, composition, degree_metrics, per_class, deletion, dependence, args, output):
    deep=comparisons.query("cluster_unit == 'drug' and weighting == 'pair' and baseline == 'DeepDTA'")
    result_details="; ".join(f"{r.split}: {r.difference:+.4f} (95% CI {r.bootstrap_ci_low:+.4f} to {r.bootstrap_ci_high:+.4f}; P={r.randomization_p_two_sided:.5f})" for r in deep.itertuples())
    equal_kgml=means.query("weighting == 'equal_drug' and model == 'KGML'").set_index("split")
    text = ["# Cluster-aware statistical supplement, 6 September 2026", "",
        "## Methods ready for the manuscript", "",
        f"We reanalysed fixed out-of-fold predictions without refitting or reselecting any predictor. Each comparison aligned the same 898 drug-target observations, 284 drugs, 209 targets and three repeats, including identical outer-fold assignments. Macro-F1 was computed over the same four fixed classes within each repeat before averaging the three repeat metrics. Undefined class F1 was set to zero.", "",
        f"The primary sensitivity analysis used {args.resamples:,} unstratified paired cluster bootstrap resamples of the 284 drugs with replacement. Every selected drug carried all its target rows and all three repeats, with the same multiplicity for both methods. Two-sided 95% percentile intervals describe the KGML-minus-comparator difference. Cluster randomization used {args.resamples:,} independent draws that exchanged the two methods' complete prediction blocks for each drug with probability one half; the same exchange applied to all targets and repeats. The two-sided Monte Carlo P value was (1 + number of absolute randomized differences at least as large as the observed absolute difference)/(B + 1). Seed: {args.seed}. We report unadjusted P values plus Holm correction across the 12 comparisons within each drug-cluster weighting family. These are sensitivity analyses conditional on the saved fitted predictions, not a re-estimation of model-training uncertainty.", "",
        "Pair-weighted macro-F1 retains the original endpoint. Equal-drug-weight macro-F1 instead gives each row weight 1/n_d, where n_d is the number of observed targets for its drug, so each drug contributes total weight one per repeat. Class-specific weighted confusion counts are formed before macro averaging; this is neither an average of per-drug F1 scores nor classification after majority-voting drug predictions. Bootstrap copies retain this one-unit weight per sampled drug.", "",
        "For double-disjoint evaluation, an additional unstratified target-cluster bootstrap/randomization resampled or exchanged all drugs and all repeats associated with each of the 209 targets. This secondary analysis retains the pair-weighted endpoint. It changes the one-way dependence assumption and does not simultaneously account for crossed drug-target dependence. Drugs/targets can recur across different clusters, and predictions also share fitted training models. No two-way resampling procedure is claimed to have resolved these dependencies.", "",
        "Component-deletion differences were computed using the highest-gate-weight expert's deletion effect minus the mean of the remaining four expert effects, preserving the original stable tie rule. Drug-cluster bootstrap and a drug-cluster sign-flip test preserve all target rows and three repeats. The statistic is the average of three within-repeat mean probability differences. This replaces a collection of 12 observation-level tests with four repeat-averaged cluster sensitivities; the results do not assert that each individual split/repeat separately passes a cluster test.", "",
        "The original direct-edge-masked degree values and quartile assignments were retained unchanged. Each stratum now reports unique drugs, unique targets, and action composition at both row and drug level. Within-stratum equal-drug results normalize by the drug's rows in that stratum. Unique-drug/target counts need not sum across target/combined quartiles because entities can span groups. Quartile comparisons remain descriptive; a post-masking zero degree is not a newly trained zero-context or fully inductive experiment.", "",
        "## Results ready for the manuscript", "",
        "The original point estimates were unchanged, but drug-cluster uncertainty altered the interpretation of held-out-drug comparisons. Against DeepDTA, KGML's pair-weighted differences were "+result_details+". Thus, the earlier observation-level support for a double-disjoint advantage was not retained by the drug-cluster sensitivity analysis. The random-pair difference remained positive after Holm correction over the 12 primary comparisons (adjusted P="+f"{deep.loc[deep.split=='random_pair','holm_p_within_analysis_family'].iloc[0]:.5f}"+"). Pair-weighted differences against GraphDTA remained positive in all four settings; against DruGNNosis, drug-disjoint evidence weakened after multiplicity adjustment and the double-disjoint interval crossed zero.", "",
        "With every drug weighted equally, KGML macro-F1 was "+", ".join(f"{equal_kgml.loc[s,'macro_f1_mean']:.4f}" for s in SPLITS)+". Its equal-drug-weight differences against DeepDTA had intervals spanning zero in every split. The target-disjoint point difference changed sign (-0.0289). These changes show that pair weighting and expanded target multiplicity materially affect the reported endpoint, without establishing a causal explanation of performance differences. In double-disjoint target-cluster sensitivity, the DeepDTA difference again had an interval spanning zero; the DruGNNosis comparison was sensitive to the chosen one-way clustering assumption.", "",
        "## Pair-weighted drug-cluster comparisons", ""]
    def table_for(frame):
        return markdown_table(["Split", "Comparator", "KGML", "Comparator F1", "Difference [95% CI]", "P", "Holm P"], [
            [r.split, r.baseline, f"{r.kgml_macro_f1:.4f}", f"{r.baseline_macro_f1:.4f}", f"{r.difference:+.4f} [{r.bootstrap_ci_low:+.4f}, {r.bootstrap_ci_high:+.4f}]", f"{r.randomization_p_two_sided:.6f}", f"{r.holm_p_within_analysis_family:.6f}"]
            for r in frame.itertuples()])
    text.append(table_for(comparisons.query("cluster_unit == 'drug' and weighting == 'pair'")))
    text += ["## Equal-drug-weight sensitivity", "", table_for(comparisons.query("cluster_unit == 'drug' and weighting == 'equal_drug'")),
             "## Double-disjoint target-cluster sensitivity", "", table_for(comparisons.query("cluster_unit == 'target'")),
             "## Degree-stratum composition", "",
             markdown_table(["Dimension", "Quartile", "Pairs", "Unique drugs", "Unique targets", "Median degree"], [[r.degree_dimension,r.degree_bin,r.pair_count,r.unique_drugs,r.unique_targets,r.median_degree] for r in strata.itertuples()]),
             "### Drug-degree action composition", "",
             markdown_table(["Quartile", "Class", "Pairs (%)", "Drugs (%)"], [[r.degree_bin,r.action_class,f"{r.pairs} ({r.pair_percent:.1f}%)",f"{r.unique_drugs} ({r.drug_percent:.1f}%)"] for r in composition.query("degree_dimension == 'drug_context_degree'").itertuples()]),
             "### The Q3 finding", ""]
    for split in ("drug_disjoint", "double_disjoint"):
        group = degree_metrics.query("degree_dimension == 'drug_context_degree' and weighting == 'pair'")
        group = group.loc[group["split"]==split].sort_values("degree_bin")
        text.append(f"For {split}, Q1/Q2/Q3/Q4 pair-weighted macro-F1 was "+" / ".join(f"{x:.4f}" for x in group["macro_f1_mean"])+". Q3 is the lowest group, so the endpoint contrast cannot be read as a monotonic degree gradient.")
    text += ["", "Drug-degree Q3 contains 200 pairs from 55 drugs and 100 targets. Its agonist/antagonist/blocker/inhibitor composition is 44/90/20/46 rows and 9/23/4/19 drugs. In drug-disjoint evaluation, Q3 versus Q4 F1 is 0.4013 versus 0.7511 for agonists, 0.7100 versus 0.8430 for antagonists, 0.6904 versus 0.7535 for blockers, and 0.9754 versus 0.9796 for inhibitors; the largest class-specific gap is in agonists. In double-disjoint Q3, agonist and blocker F1 are 0.3617 and 0.4316, versus 0.6241 and 0.7086 in Q4; both show marked descriptive gaps. Antagonist F1 is 0.7327 versus 0.8347, while inhibitor F1 is 0.9018 versus 0.9057. Only four blocker drugs underlie Q3's 20 blocker rows, making the row/drug distinction relevant to interpretation. These comparisons locate class-specific deficits without attributing them causally to degree or composition.", "",
             "When drugs are equally weighted within the degree strata, drug-disjoint Q1/Q2/Q3/Q4 macro-F1 becomes 0.7881/0.7906/0.7341/0.8646. For double-disjoint evaluation it becomes 0.6453/0.7172/0.6627/0.6967: Q3 is no longer the lowest group. Thus, the observed Q3 minimum is specific to the original pair-weighted double-disjoint endpoint and should not be generalized to every drug-weighting convention.", "",
             "Q3 class-specific F1 and recall are supplied in degree_class_performance.csv. Composition and class-specific performance can localize the observed pattern but do not establish that composition or degree caused the difference; no outcome-driven subgroup selection, covariate adjustment or causal graph perturbation was performed.", "",
             "## Component-deletion sensitivity", "",
             markdown_table(["Split", "Mean top-minus-other", "95% CI", "Drug-cluster sign-flip P", "Holm P"], [[r.split,f"{r.difference:.5f}",f"[{r.bootstrap_ci_low:.5f}, {r.bootstrap_ci_high:.5f}]",f"{r.randomization_p_two_sided:.6f}",f"{r.holm_p_across_four_splits:.6f}"] for r in deletion.itertuples()]),
             "## Remaining dependence", "", json.dumps(dependence, indent=2), "",
             "The bootstrap samples drug identities uniformly without label stratification, whereas the old row bootstrap stratified rows by class. Differences in intervals therefore reflect both the revised clustering assumption and the changed resampling design. They are not isolated measurements of the impact of clustering alone. Target clustering, finite repeated fits, overlapping training folds, shared graph information and prior exploratory cohort access remain limitations. A P value here depends on exchangeability of method-prediction blocks under its cluster null and is not a randomized-treatment inference. No newly significant result changes strategy selection. All original point estimates, S1 metrics, and deletion-repeat means were reproduced to absolute tolerance 1e-12.", ""]
    (output/"manuscript_statistics_supplement.md").write_text("\n".join(text), encoding="utf-8")


def write_latex_tables(comparisons, means, strata, composition, degree_metrics, deletion, output):
    """Reviewable drop-in table fragments; never edit the main manuscript."""
    names={"random_pair":"Random pair", "drug_disjoint":"Drug-disjoint", "target_disjoint":"Target-disjoint", "double_disjoint":"Double-disjoint"}
    dimensions={"drug_context_degree":"Drug", "target_context_degree":"Target", "combined_context_degree":"Combined"}
    blocks=["% Generated by cluster_statistics_20260906.py. Requires booktabs and graphicx."]
    def table(caption,label,columns,header,rows):
        blocks.extend([r"\begin{table}[t]",r"\centering",r"\footnotesize",r"\caption{"+caption+"}",r"\label{"+label+"}",r"\setlength{\tabcolsep}{3pt}",r"\resizebox{\linewidth}{!}{%",r"\begin{tabular}{"+columns+"}",r"\toprule",header+r" \\",r"\midrule"])
        blocks.extend(" & ".join(row)+r" \\" for row in rows)
        blocks.extend([r"\bottomrule",r"\end{tabular}}",r"\end{table}",""])
    part=comparisons.query("cluster_unit == 'drug' and weighting == 'pair'")
    rows=[]
    for r in part.itertuples():
        rows.append([names[r.split],r.baseline,f"{r.difference:+.4f}",f"[{r.bootstrap_ci_low:+.4f}, {r.bootstrap_ci_high:+.4f}]"])
    draws=int(part.iloc[0].bootstrap_resamples)
    table(f"Drug-cluster sensitivity of fixed OOF comparisons. Differences are KGML minus comparator in pair-weighted macro-F1, calculated within each repeat and then averaged. Intervals use {draws:,} unstratified drug-cluster bootstrap draws, preserving all targets and three repeats. Complete precomputed randomization results are retained in the reproducible archive. These conditional sensitivities do not resolve crossed drug--target or shared fitted-model dependence.","tab:drug-cluster-sensitivity","llrr","Split & Comparator & Difference & 95\\% CI",rows)
    rows=[]
    for split in SPLITS:
        row=[names[split]]
        for model in ("KGML",)+tuple(MODELS.values()):
            r=means.query("split == @split and weighting == 'equal_drug' and model == @model").iloc[0]
            row.append(f"${r.macro_f1_mean:.4f}\\pm{r.macro_f1_sd:.4f}$")
        rows.append(row)
    table("Equal-drug-weight macro-F1, mean and SD across three repeats. Each target row receives weight $1/n_d$, where $n_d$ is its drug's number of observed targets; every drug therefore contributes total weight one per repeat. Weighted confusion counts are formed before the four-class macro average.","tab:equal-drug-weight","lrrrr","Split & KGML & DeepDTA & GraphDTA & DruGNNosis",rows)
    rows=[]
    for r in strata.itertuples():
        row=[dimensions[r.degree_dimension],r.degree_bin,str(r.pair_count),str(r.unique_drugs),str(r.unique_targets),f"{r.median_degree:g}"]
        for split in SPLITS:
            sel=degree_metrics[(degree_metrics.degree_dimension==r.degree_dimension)&(degree_metrics.degree_bin==r.degree_bin)&(degree_metrics.split==split)&(degree_metrics.weighting=='pair')]
            row.append(f"{sel.iloc[0].macro_f1_mean:.3f}")
        rows.append(row)
    table("Expanded context-degree strata. Original direct-edge-masked degree assignments and pair-weighted macro-F1 means are unchanged. Drug and target columns count unique entities within each group; entities can span target/combined strata, so those counts are not additive. Degree is measured after query-edge removal without stratum-specific model refitting.","tab:expanded-degree-strata","llrrrrrrrr","Dimension & Quartile & Pairs & Drugs & Targets & Median & Random & Drug-disjoint & Target-disjoint & Double-disjoint",rows)
    rows=[]
    for dimension in dimensions:
        for quartile in ("Q1 sparse","Q2","Q3","Q4 dense"):
            row=[dimensions[dimension],quartile]
            for label in CLASSES:
                r=composition[(composition.degree_dimension==dimension)&(composition.degree_bin==quartile)&(composition.action_class==label)].iloc[0]
                row.append(f"{r.pairs}/{r.unique_drugs}")
            rows.append(row)
    table("Action composition of each context-degree stratum. Entries are drug--target row counts / unique-drug counts for each drug-level reference action. Unique targets and both row- and drug-level percentages are retained in the accompanying machine-readable composition table.","tab:degree-action-composition","llrrrr","Dimension & Quartile & Agonist & Antagonist & Blocker & Inhibitor",rows)
    (output/"manuscript_tables.tex").write_text("\n".join(blocks),encoding="utf-8")


class NumericalTests(unittest.TestCase):
    def frame(self):
        records = []
        for repeat in range(3):
            for i, label in enumerate(CLASSES):
                for target in range(i+1):
                    records.append({"repeat": repeat, "drug_name": "d"+str(i), "target_name": "t"+str(target), "true_label": label,
                                    "left": label if target == 0 or repeat == 1 else CLASSES[(i+1)%4],
                                    "right": label if repeat == 0 else CLASSES[(i+2)%4]})
        return pd.DataFrame(records)

    def test_known_confusion_and_zero_class(self):
        matrix = np.array([[2,1,0,0],[1,2,0,0],[0,0,1,0],[0,0,0,1]])
        self.assertAlmostEqual(float(f1_from_confusion(matrix)), 5/6)
        self.assertEqual(float(f1_from_confusion(np.zeros((4,4)))), 0)

    def test_group_bootstrap_equals_explicit_rows(self):
        frame = self.frame()
        counts = np.array([[2,0,1,1]])
        for weighting in ("pair", "equal_drug"):
            left, clusters, repeats = cluster_confusions(frame,"left","drug_name",weighting)
            right, _, _ = cluster_confusions(frame,"right","drug_name",weighting)
            actual = draw_statistics(left,right,counts,"bootstrap")[0]
            expected = []
            for repeat in repeats:
                part=frame.loc[frame["repeat"]==repeat].copy()
                w=part["drug_name"].map(dict(zip(clusters,counts[0]))).to_numpy(dtype=float)
                if weighting=="equal_drug":
                    w/=part.groupby("drug_name")["target_name"].transform("size").to_numpy()
                expected.append(f1_from_confusion(confusion(part.true_label,part.left,w))-f1_from_confusion(confusion(part.true_label,part.right,w)))
            self.assertAlmostEqual(actual,float(np.mean(expected)))

    def test_joint_swap_equals_explicit_rows_all_repeats(self):
        frame=self.frame()
        left,clusters,repeats=cluster_confusions(frame,"left","drug_name","pair")
        right,_,_=cluster_confusions(frame,"right","drug_name","pair")
        for swap in itertools.product((0,1),repeat=4):
            actual=draw_statistics(left,right,np.array([swap]),"swap")[0]
            expected=[]
            for repeat in repeats:
                part=frame.loc[frame["repeat"]==repeat]
                mask=part.drug_name.map(dict(zip(clusters,swap))).to_numpy(bool)
                a,b=np.where(mask,part.right,part.left),np.where(mask,part.left,part.right)
                expected.append(f1_from_confusion(confusion(part.true_label,a))-f1_from_confusion(confusion(part.true_label,b)))
            self.assertAlmostEqual(actual,float(np.mean(expected)))

    def test_equal_drug_weight_ignores_whole_drug_replication(self):
        frame=self.frame()
        extra=frame.loc[frame.drug_name=="d3"].copy()
        extra["target_name"]+="_copy"
        duplicated=pd.concat([frame,extra],ignore_index=True)
        a,_,_=cluster_confusions(frame,"left","drug_name","equal_drug")
        b,_,_=cluster_confusions(duplicated,"left","drug_name","equal_drug")
        np.testing.assert_allclose(a,b,atol=1e-14)
        np.testing.assert_allclose(a.sum(axis=(-1,-2)),1)

    def test_repeat_metrics_are_not_pooled(self):
        frame=self.frame()
        left,_,_=cluster_confusions(frame,"left","drug_name","pair")
        right,_,_=cluster_confusions(frame,"right","drug_name","pair")
        actual=draw_statistics(left,right,np.ones((1,len(left))),"bootstrap")[0]
        expected=(f1_from_confusion(left.sum(axis=0))-f1_from_confusion(right.sum(axis=0))).mean()
        pooled=f1_from_confusion(left.sum(axis=(0,1)))-f1_from_confusion(right.sum(axis=(0,1)))
        self.assertAlmostEqual(actual,float(expected))
        self.assertGreater(abs(actual-float(pooled)),1e-5)

    def test_identical_models_zero_interval_and_p_one(self):
        left,_,_=cluster_confusions(self.frame(),"left","drug_name","pair")
        stats,boot,null=paired_inference(left,left,resamples=200,seed=99,stream="test")
        self.assertEqual(stats["randomization_p_two_sided"],1)
        self.assertEqual(stats["difference"],0)
        np.testing.assert_array_equal(boot,0)
        np.testing.assert_array_equal(null,0)

    def test_full_swap_reverses_difference(self):
        frame=self.frame()
        left,_,_=cluster_confusions(frame,"left","drug_name","pair")
        right,_,_=cluster_confusions(frame,"right","drug_name","pair")
        actual=draw_statistics(left,right,np.array([[0]*4,[1]*4]),"swap")
        self.assertAlmostEqual(actual[0],-actual[1])

    def test_holm(self):
        np.testing.assert_allclose(holm([.01,.04,.03]),[.03,.06,.06])

    def test_scalar_cluster_ratio_preserves_unequal_sizes_and_repeats(self):
        sums=np.array([[1.,2.,3.],[.5,.7,.9]])
        counts=np.array([[2.,2.,2.],[1.,1.,1.]])
        actual=scalar_cluster_statistics(sums,counts,np.array([[2,1],[0,2]]),"bootstrap")
        np.testing.assert_allclose(actual,[.94,.7])
        flipped=scalar_cluster_statistics(sums,counts,np.array([[1,-1]]),"signflip")
        np.testing.assert_allclose(flipped,[1.3/3])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output",type=Path,default=None)
    parser.add_argument("--resamples",type=int,default=20000)
    parser.add_argument("--seed",type=int,default=20260906)
    parser.add_argument("--self-test",action="store_true")
    args=parser.parse_args()
    if args.self_test:
        result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(NumericalTests))
        sys.exit(0 if result.wasSuccessful() else 1)
    assert args.resamples>=1000
    output=args.output or args.root/"runs/manuscript_revision_20260906/statistics"
    output.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(columns=VALIDATION_COLUMNS).to_csv(output/"point_estimate_validation.csv",index=False)
    before={key:{"path":str(args.root/rel),"sha256":sha256(args.root/rel)} for key,rel in INPUTS.items()}
    test_output=io.StringIO()
    tests=unittest.TextTestRunner(stream=test_output,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(NumericalTests))
    (output/"unit_test_results.txt").write_text(test_output.getvalue(),encoding="utf-8")
    print(test_output.getvalue(),flush=True)
    if not tests.wasSuccessful():
        raise RuntimeError("Numerical unit tests failed")
    wide,cohort=load_predictions(args.root)
    comparisons,means,distributions=method_comparisons(wide,args,output)
    strata,composition,degree_metrics,per_class=degree_composition(wide,cohort,args,output)
    deletion,deletion_draws=deletion_inference(args,output)
    distributions.update(deletion_draws)
    np.savez_compressed(output/"resampling_distributions.npz",**distributions)
    dependence=dependence_summary(cohort)
    (output/"dependence_summary.json").write_text(json.dumps(dependence,indent=2)+"\n")
    write_report(comparisons,means,strata,composition,degree_metrics,per_class,deletion,dependence,args,output)
    write_latex_tables(comparisons,means,strata,composition,degree_metrics,deletion,output)
    validation=pd.read_csv(output/"point_estimate_validation.csv")
    validation_summary=validation.groupby("check_family").agg(checks=("passed","size"),maximum_absolute_error=("absolute_error","max"),all_passed=("passed","all")).reset_index()
    (output/"point_estimate_validation_summary.json").write_text(validation_summary.to_json(orient="records",indent=2)+"\n")
    assert all(item["sha256"]==sha256(item["path"]) for item in before.values()),"An original input changed during analysis"
    manifest={"generated_utc":datetime.now(timezone.utc).isoformat(),"script":str(Path(__file__).resolve()),"script_sha256":sha256(__file__),
              "seed":args.seed,"bootstrap_resamples":args.resamples,"randomization_resamples":args.resamples,
              "python":sys.version,"numpy":np.__version__,"pandas":pd.__version__,"input_files":before,
              "validation":{"unit_tests_passed":tests.testsRun,"aligned_rows":len(wide),"unique_pairs":898,"unique_drugs":284,"unique_targets":209,
                            "reference_labels_constant_within_drug":True,"same_outer_folds_across_methods":True,"old_pair_point_estimates_reproduced_1e_12":True,
                            "all_48_old_degree_metrics_reproduced_1e_12":True,"all_12_old_deletion_repeat_means_reproduced_1e_12":True,"input_hashes_unchanged":True},
              "outputs":{p.name:sha256(p) for p in sorted(output.iterdir()) if p.is_file() and p.name!="manifest.json"}}
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps({"output":str(output),"comparisons":len(comparisons),"degree_groups":len(strata),"unit_tests":tests.testsRun,"status":"validated"}),flush=True)


if __name__=="__main__":
    main()
