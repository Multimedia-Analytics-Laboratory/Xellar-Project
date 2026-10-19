"""Post-hoc error analysis and paired comparisons for strict OOF predictions."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from .io_utils import ensure_dir


def _predictor_id(frame):
    view = frame.get("view", pd.Series("", index=frame.index)).fillna("")
    classifier = frame.get(
        "classifier",
        pd.Series("", index=frame.index),
    ).fillna("")
    return (
        view.astype(str)
        + "::"
        + frame["model"].astype(str)
        + "::"
        + classifier.astype(str)
    )


def _load_predictions(reports_dir):
    frames = []
    strict_path = reports_dir / "strict_oof_predictions.csv"
    if strict_path.exists():
        strict = pd.read_csv(strict_path)
        strict["source_type"] = "knowledge_graph"
        strict["predictor"] = _predictor_id(strict)
        frames.append(strict)
    multimodal_path = reports_dir / "multimodal_oof_predictions.csv"
    if multimodal_path.exists():
        multimodal = pd.read_csv(multimodal_path)
        multimodal["view"] = "multimodal_baseline"
        multimodal["classifier"] = multimodal["model"]
        multimodal["source_type"] = "multimodal"
        multimodal["predictor"] = _predictor_id(multimodal)
        frames.append(multimodal)
    if not frames:
        raise FileNotFoundError("No strict or multimodal OOF predictions were found")
    common = [
        "task",
        "cohort",
        "view",
        "model",
        "classifier",
        "source_type",
        "predictor",
        "repeat",
        "fold",
        "drug_name",
        "true_label",
        "predicted_label",
    ]
    return pd.concat([frame[common] for frame in frames], ignore_index=True)


def _metric_row(y_true, y_pred):
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        precision_score,
        recall_score,
    )

    return {
        "sample_count": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
    }


def _per_class_metrics(predictions):
    from sklearn.metrics import precision_recall_fscore_support

    rows = []
    keys = [
        "task",
        "cohort",
        "source_type",
        "view",
        "model",
        "classifier",
        "predictor",
        "repeat",
    ]
    for values, frame in predictions.groupby(keys, dropna=False):
        labels = sorted(
            set(frame["true_label"].astype(str))
            | set(frame["predicted_label"].astype(str))
        )
        precision, recall, f1, support = precision_recall_fscore_support(
            frame["true_label"].astype(str),
            frame["predicted_label"].astype(str),
            labels=labels,
            zero_division=0,
        )
        base = dict(zip(keys, values))
        for index, label in enumerate(labels):
            rows.append({
                **base,
                "label": label,
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            })
    raw = pd.DataFrame(rows)
    aggregate_keys = keys[:-1] + ["label"]
    aggregate = raw.groupby(aggregate_keys, dropna=False).agg(
        precision_mean=("precision", "mean"),
        precision_std=("precision", "std"),
        recall_mean=("recall", "mean"),
        recall_std=("recall", "std"),
        f1_mean=("f1", "mean"),
        f1_std=("f1", "std"),
        support=("support", "max"),
        repeats=("repeat", "nunique"),
    ).reset_index()
    for metric in ("precision", "recall", "f1"):
        aggregate["%s_ci95" % metric] = (
            1.96
            * aggregate["%s_std" % metric].fillna(0.0)
            / aggregate["repeats"].pow(0.5)
        )
    return raw, aggregate


def _confusion_matrices(predictions):
    from sklearn.metrics import confusion_matrix

    rows = []
    keys = [
        "task",
        "cohort",
        "source_type",
        "view",
        "model",
        "classifier",
        "predictor",
    ]
    for values, frame in predictions.groupby(keys, dropna=False):
        labels = sorted(set(frame["true_label"].astype(str)))
        matrix = confusion_matrix(
            frame["true_label"].astype(str),
            frame["predicted_label"].astype(str),
            labels=labels,
        )
        base = dict(zip(keys, values))
        for true_index, true_label in enumerate(labels):
            row_total = int(matrix[true_index].sum())
            for predicted_index, predicted_label in enumerate(labels):
                count = int(matrix[true_index, predicted_index])
                rows.append({
                    **base,
                    "true_label": true_label,
                    "predicted_label": predicted_label,
                    "count": count,
                    "row_normalized": (
                        float(count / row_total) if row_total else math.nan
                    ),
                })
    return pd.DataFrame(rows)


def _drug_features(config):
    triple_root = Path(config["paths"]["intermediate_dir"]) / "triples"
    moa = pd.read_csv(triple_root / "groups" / "moa.csv")
    drugs = sorted(moa["head"].astype(str).unique())
    target = pd.read_csv(triple_root / "groups" / "target.csv")
    target_count = (
        target.groupby(target["head"].astype(str))["tail"].nunique().to_dict()
    )
    full_view = next(
        (
            name for name in config.get("views", {})
            if name.startswith("full_label_free")
        ),
        next(iter(config.get("views", {}))),
    )
    graph = pd.read_csv(triple_root / "views" / ("%s.csv" % full_view))
    head_counts = graph["head"].astype(str).value_counts()
    tail_counts = graph["tail"].astype(str).value_counts()
    rows = []
    for drug in drugs:
        rows.append({
            "drug_name": drug,
            "target_count": int(target_count.get(drug, 0)),
            "graph_degree": int(head_counts.get(drug, 0) + tail_counts.get(drug, 0)),
        })
    frame = pd.DataFrame(rows)
    for column in ("target_count", "graph_degree"):
        positive = frame[column] > 0
        frame["%s_stratum" % column] = "zero"
        if positive.sum() >= 3:
            ranked = frame.loc[positive, column].rank(method="first")
            frame.loc[positive, "%s_stratum" % column] = pd.qcut(
                ranked,
                q=3,
                labels=["low", "medium", "high"],
            ).astype(str)
    return frame


def _stratified_metrics(predictions, features):
    merged = predictions.merge(features, on="drug_name", how="left")
    rows = []
    base_keys = [
        "task",
        "cohort",
        "source_type",
        "view",
        "model",
        "classifier",
        "predictor",
        "repeat",
    ]
    for variable in ("target_count", "graph_degree"):
        stratum_column = "%s_stratum" % variable
        for values, frame in merged.groupby(
            base_keys + [stratum_column],
            dropna=False,
        ):
            base = dict(zip(base_keys + [stratum_column], values))
            rows.append({
                **base,
                "stratification_variable": variable,
                "stratum": base.pop(stratum_column),
                **_metric_row(frame["true_label"], frame["predicted_label"]),
            })
    return pd.DataFrame(rows)


def _select_comparisons(predictions):
    from sklearn.metrics import f1_score

    repeat_rows = []
    keys = ["task", "cohort", "predictor", "repeat"]
    for values, frame in predictions.groupby(keys):
        repeat_rows.append({
            **dict(zip(keys, values)),
            "f1_macro": float(
                f1_score(
                    frame["true_label"],
                    frame["predicted_label"],
                    average="macro",
                    zero_division=0,
                )
            ),
        })
    scores = pd.DataFrame(repeat_rows)
    mean_scores = scores.groupby(
        ["task", "cohort", "predictor"]
    )["f1_macro"].mean().reset_index()
    comparisons = []
    for (task, cohort), group in mean_scores.groupby(["task", "cohort"]):
        available = predictions[
            (predictions["task"] == task)
            & (predictions["cohort"] == cohort)
        ]
        full = group[group["predictor"].str.startswith("full_label_free::")]
        if full.empty:
            continue
        reference = full.sort_values("f1_macro", ascending=False).iloc[0]["predictor"]
        candidate_sets = {
            "targets_only": group[
                group["predictor"].str.startswith("targets_only_label_free::")
            ],
            "target_multihot": group[
                group["predictor"].str.startswith("baseline::TargetMultiHot::")
            ],
            "multimodal": group[
                group["predictor"].str.startswith("multimodal_baseline::")
            ],
        }
        for comparator_type, candidates in candidate_sets.items():
            if candidates.empty:
                continue
            comparator = candidates.sort_values(
                "f1_macro",
                ascending=False,
            ).iloc[0]["predictor"]
            if (
                available[available["predictor"] == reference]["drug_name"].nunique()
                == available[available["predictor"] == comparator]["drug_name"].nunique()
            ):
                comparisons.append({
                    "task": task,
                    "cohort": cohort,
                    "reference": reference,
                    "comparator": comparator,
                    "comparator_type": comparator_type,
                    "selection_note": (
                        "Best mean macro-F1 within each source family; "
                        "post-hoc and therefore exploratory."
                    ),
                })
    return comparisons


def _paired_bootstrap(predictions, comparisons, seed, iterations=2000):
    import numpy as np
    from sklearn.metrics import f1_score

    rng = np.random.RandomState(seed)
    rows = []
    for comparison in comparisons:
        subset = predictions[
            (predictions["task"] == comparison["task"])
            & (predictions["cohort"] == comparison["cohort"])
            & predictions["predictor"].isin(
                [comparison["reference"], comparison["comparator"]]
            )
        ]
        repeat_differences = []
        bootstrap_differences = []
        shared_repeats = sorted(
            set(subset[subset["predictor"] == comparison["reference"]]["repeat"])
            & set(subset[subset["predictor"] == comparison["comparator"]]["repeat"])
        )
        aligned_by_repeat = []
        for repeat in shared_repeats:
            left = subset[
                (subset["repeat"] == repeat)
                & (subset["predictor"] == comparison["reference"])
            ][["drug_name", "true_label", "predicted_label"]].rename(
                columns={"predicted_label": "reference_prediction"}
            )
            right = subset[
                (subset["repeat"] == repeat)
                & (subset["predictor"] == comparison["comparator"])
            ][["drug_name", "true_label", "predicted_label"]].rename(
                columns={"predicted_label": "comparator_prediction"}
            )
            aligned = left.merge(
                right,
                on=["drug_name", "true_label"],
                how="inner",
            ).sort_values("drug_name")
            if aligned.empty:
                continue
            reference_score = f1_score(
                aligned["true_label"],
                aligned["reference_prediction"],
                average="macro",
                zero_division=0,
            )
            comparator_score = f1_score(
                aligned["true_label"],
                aligned["comparator_prediction"],
                average="macro",
                zero_division=0,
            )
            repeat_differences.append(reference_score - comparator_score)
            aligned_by_repeat.append(aligned.reset_index(drop=True))
        if not aligned_by_repeat:
            continue
        sample_count = min(len(frame) for frame in aligned_by_repeat)
        for _ in range(iterations):
            indices = rng.randint(0, sample_count, size=sample_count)
            differences = []
            for aligned in aligned_by_repeat:
                sample = aligned.iloc[indices]
                differences.append(
                    f1_score(
                        sample["true_label"],
                        sample["reference_prediction"],
                        average="macro",
                        zero_division=0,
                    )
                    - f1_score(
                        sample["true_label"],
                        sample["comparator_prediction"],
                        average="macro",
                        zero_division=0,
                    )
                )
            bootstrap_differences.append(float(np.mean(differences)))
        bootstrap = np.asarray(bootstrap_differences)
        observed = float(np.mean(repeat_differences))
        p_value = float(
            2.0
            * min(
                (np.sum(bootstrap <= 0.0) + 1) / (len(bootstrap) + 1),
                (np.sum(bootstrap >= 0.0) + 1) / (len(bootstrap) + 1),
            )
        )
        rows.append({
            **comparison,
            "shared_sample_count": int(sample_count),
            "shared_repeat_count": int(len(aligned_by_repeat)),
            "macro_f1_difference": observed,
            "bootstrap_ci95_low": float(np.percentile(bootstrap, 2.5)),
            "bootstrap_ci95_high": float(np.percentile(bootstrap, 97.5)),
            "bootstrap_two_sided_p": min(p_value, 1.0),
            "bootstrap_iterations": int(iterations),
        })
    return pd.DataFrame(rows)


def _export_graph_evidence(predictions, config):
    """Export traceable graph context for selected correct and incorrect cases."""

    from sklearn.metrics import f1_score

    triple_root = Path(config["paths"]["intermediate_dir"]) / "triples" / "groups"
    group_frames = {
        group: pd.read_csv(triple_root / ("%s.csv" % group)).astype(str)
        for group in ("target", "indication", "anatomy", "pathway")
        if (triple_root / ("%s.csv" % group)).exists()
    }
    source_by_group = {
        "target": "Repurposing Hub",
        "indication": "PrimeKG:indication",
        "anatomy": "PrimeKG:anatomy_protein_present",
        "pathway": "PrimeKG:pathway_protein",
    }

    scores = []
    full = predictions[predictions["view"] == "full_label_free"]
    for values, frame in full.groupby(["task", "cohort", "predictor", "repeat"]):
        scores.append({
            "task": values[0],
            "cohort": values[1],
            "predictor": values[2],
            "repeat": values[3],
            "f1_macro": f1_score(
                frame["true_label"],
                frame["predicted_label"],
                average="macro",
                zero_division=0,
            ),
        })
    if not scores:
        return pd.DataFrame(), pd.DataFrame()
    score_frame = pd.DataFrame(scores)
    best = (
        score_frame.groupby(["task", "cohort", "predictor"])["f1_macro"]
        .mean()
        .reset_index()
        .sort_values(
            ["task", "cohort", "f1_macro"],
            ascending=[True, True, False],
        )
        .groupby(["task", "cohort"])
        .head(1)
    )

    examples = []
    for row in best.itertuples(index=False):
        frame = full[
            (full["task"] == row.task)
            & (full["cohort"] == row.cohort)
            & (full["predictor"] == row.predictor)
            & (full["repeat"] == 0)
        ].copy()
        frame["correct"] = frame["true_label"] == frame["predicted_label"]
        for correctness, status in ((True, "correct"), (False, "incorrect")):
            selected = (
                frame[frame["correct"] == correctness]
                .sort_values(["true_label", "drug_name"])
                .groupby("true_label")
                .head(1)
                .head(3)
            )
            for selected_row in selected.itertuples(index=False):
                examples.append({
                    "example_id": "%s__%s__%s__%s" % (
                        row.task,
                        row.cohort,
                        status,
                        selected_row.drug_name.replace(" ", "_"),
                    ),
                    "task": row.task,
                    "cohort": row.cohort,
                    "predictor": row.predictor,
                    "repeat": 0,
                    "drug_name": selected_row.drug_name,
                    "true_label": selected_row.true_label,
                    "predicted_label": selected_row.predicted_label,
                    "prediction_status": status,
                    "selection_rule": (
                        "First drug alphabetically among one-per-true-class "
                        "candidates for repeat 0 of the post-hoc best full view."
                    ),
                })

    evidence_rows = []
    for example in examples:
        drug = str(example["drug_name"])
        direct_targets = set()
        for group in ("target", "indication"):
            frame = group_frames.get(group)
            if frame is None:
                continue
            direct = frame[frame["head"] == drug]
            if group == "target":
                direct_targets.update(direct["tail"].astype(str))
            for triple in direct.itertuples(index=False):
                evidence_rows.append({
                    **example,
                    "hop": 1,
                    "source_group": group,
                    "source_database": source_by_group[group],
                    "head": triple.head,
                    "relation": triple.relation,
                    "tail": triple.tail,
                    "grounded_statement": "%s --%s--> %s" % (
                        triple.head,
                        triple.relation,
                        triple.tail,
                    ),
                    "faithfulness_claim": False,
                })
        for group in ("anatomy", "pathway"):
            frame = group_frames.get(group)
            if frame is None:
                continue
            second_hop = frame[frame["head"].isin(direct_targets)]
            for triple in second_hop.itertuples(index=False):
                evidence_rows.append({
                    **example,
                    "hop": 2,
                    "source_group": group,
                    "source_database": source_by_group[group],
                    "head": triple.head,
                    "relation": triple.relation,
                    "tail": triple.tail,
                    "grounded_statement": "%s --%s--> %s" % (
                        triple.head,
                        triple.relation,
                        triple.tail,
                    ),
                    "faithfulness_claim": False,
                })
    return pd.DataFrame(examples), pd.DataFrame(evidence_rows)


def run_posthoc_analysis(config, logger):
    """Generate all post-hoc tables from saved out-of-fold predictions."""

    reports_dir = ensure_dir(Path(config["paths"]["reports_dir"]))
    predictions = _load_predictions(reports_dir)
    class_raw, class_summary = _per_class_metrics(predictions)
    confusion = _confusion_matrices(predictions)
    features = _drug_features(config)
    strata = _stratified_metrics(predictions, features)
    comparisons = _paired_bootstrap(
        predictions,
        _select_comparisons(predictions),
        seed=int(config["random_seed"]),
    )
    examples, evidence = _export_graph_evidence(predictions, config)

    class_raw.to_csv(reports_dir / "per_class_metrics_by_repeat.csv", index=False)
    class_summary.to_csv(reports_dir / "per_class_metrics_summary.csv", index=False)
    confusion.to_csv(reports_dir / "confusion_matrices.csv", index=False)
    features.to_csv(reports_dir / "drug_graph_features.csv", index=False)
    strata.to_csv(reports_dir / "stratified_error_analysis.csv", index=False)
    comparisons.to_csv(reports_dir / "paired_bootstrap_comparisons.csv", index=False)
    examples.to_csv(reports_dir / "selected_prediction_examples.csv", index=False)
    evidence.to_csv(reports_dir / "selected_example_graph_evidence.csv", index=False)
    logger.info(
        "Post-hoc analysis wrote per-class, confusion, strata, and paired-comparison reports."
    )
    return {
        "per_class": reports_dir / "per_class_metrics_summary.csv",
        "confusion": reports_dir / "confusion_matrices.csv",
        "strata": reports_dir / "stratified_error_analysis.csv",
        "comparisons": reports_dir / "paired_bootstrap_comparisons.csv",
        "examples": reports_dir / "selected_prediction_examples.csv",
        "evidence": reports_dir / "selected_example_graph_evidence.csv",
    }
