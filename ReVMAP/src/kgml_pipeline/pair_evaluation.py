"""Pair-level drug--target action relation evaluation.

This module keeps KGEM training unchanged and changes only the downstream
prediction task from drug-node classification to drug--target pair
classification.  For each labeled drug--target pair, it constructs pair
features from fixed label-free embeddings:

    [drug_emb, target_emb, abs(drug_emb - target_emb), drug_emb * target_emb]

The evaluator supports random pair, drug-disjoint, target-disjoint, and
double-disjoint cross-validation settings. Double-disjoint evaluation uses a
blocked Cartesian protocol: drugs and targets are each assigned to ``K``
entity partitions and the ``K x K`` drug-partition/target-partition blocks are
used as outer test folds. The corresponding training fold excludes every drug
and every target present in its test block.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from .io_utils import ensure_dir
from .strict_evaluation import (
    METRIC_NAMES,
    _as_real_feature_matrix,
    _metric_values,
    _summary_stats,
)


def _pair_label_tasks(pair_cfg):
    configured = pair_cfg.get("label_tasks")
    if configured:
        return configured
    return {
        "action_default": {
            "label_column": str(pair_cfg.get("label_column", "relation")),
            "min_class_count": int(pair_cfg.get("min_class_count", 5)),
        }
    }


def _build_pair_classifiers(pair_cfg, random_seed):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    hidden_layers = tuple(
        int(value) for value in pair_cfg.get("mlp_hidden_layers", [128, 64])
    )
    classifiers = {
        "Random Forest": RandomForestClassifier(
            n_estimators=int(pair_cfg.get("rf_estimators", 300)),
            max_depth=None,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=random_seed,
            n_jobs=-1,
        ),
        "Logistic Regression": Pipeline([
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=float(pair_cfg.get("logistic_c", 1.0)),
                    class_weight="balanced",
                    max_iter=int(pair_cfg.get("logistic_max_iter", 3000)),
                    random_state=random_seed,
                ),
            ),
        ]),
        "SVM (RBF)": Pipeline([
            ("scale", StandardScaler()),
            (
                "model",
                SVC(
                    C=float(pair_cfg.get("svm_c", 1.0)),
                    kernel="rbf",
                    class_weight="balanced",
                    max_iter=int(pair_cfg.get("svm_max_iter", 10000)),
                    random_state=random_seed,
                ),
            ),
        ]),
        "KNN (k=5)": Pipeline([
            ("scale", StandardScaler()),
            (
                "model",
                KNeighborsClassifier(
                    n_neighbors=int(pair_cfg.get("knn_neighbors", 5)),
                    weights="uniform",
                ),
            ),
        ]),
        "MLP": Pipeline([
            ("scale", StandardScaler()),
            (
                "model",
                MLPClassifier(
                    hidden_layer_sizes=hidden_layers,
                    activation=str(pair_cfg.get("mlp_activation", "relu")),
                    alpha=float(pair_cfg.get("mlp_alpha", 1e-4)),
                    learning_rate_init=float(pair_cfg.get("mlp_learning_rate", 1e-3)),
                    max_iter=int(pair_cfg.get("mlp_max_iter", 300)),
                    early_stopping=True,
                    n_iter_no_change=int(pair_cfg.get("mlp_patience", 15)),
                    random_state=random_seed,
                ),
            ),
        ]),
    }
    allowed = pair_cfg.get("classifiers")
    if allowed:
        allowed = {str(name) for name in allowed}
        classifiers = {
            name: estimator for name, estimator in classifiers.items() if name in allowed
        }
    return classifiers


def _embedding_sources(config, pair_cfg, logger):
    import numpy as np

    allowed_views = pair_cfg.get("allowed_views")
    if allowed_views:
        allowed_views = {str(view) for view in allowed_views}
    allowed_models = pair_cfg.get("allowed_models")
    if allowed_models:
        allowed_models = {str(model) for model in allowed_models}

    embedding_root = Path(config["paths"]["processed_dir"]) / "embeddings"
    for view_name in config.get("views", {}):
        if allowed_views and view_name not in allowed_views:
            continue
        view_dir = embedding_root / view_name
        if not view_dir.exists():
            logger.warning("Embedding directory missing for pair view %s", view_name)
            continue
        for model_dir in sorted(view_dir.iterdir()):
            if allowed_models and model_dir.name not in allowed_models:
                continue
            entity_path = model_dir / "entity_to_id.json"
            embedding_path = model_dir / "entity_embeddings.npy"
            if not entity_path.exists() or not embedding_path.exists():
                continue
            with entity_path.open("r", encoding="utf-8") as handle:
                entity_to_id = json.load(handle)
            yield {
                "view": view_name,
                "model": model_dir.name,
                "entity_to_id": {str(key): int(value) for key, value in entity_to_id.items()},
                "features": np.load(str(embedding_path)),
            }


def _load_pair_table(config, pair_cfg):
    pair_table_path = pair_cfg.get("pair_table_path")
    if pair_table_path:
        frame = pd.read_csv(Path(pair_table_path).expanduser())
        rename_map = {}
        for source, target in (
            (pair_cfg.get("pair_drug_column", "drug_name"), "drug_name"),
            (pair_cfg.get("pair_target_column", "target_name"), "target_name"),
            (pair_cfg.get("pair_action_column", "action"), "action"),
        ):
            if source != target and source in frame.columns:
                rename_map[source] = target
        if rename_map:
            frame = frame.rename(columns=rename_map)
        required = {"drug_name", "target_name", "action"}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(
                "Pair table %s is missing required columns: %s"
                % (pair_table_path, ", ".join(missing))
            )
        frame = frame[["drug_name", "target_name", "action"]].dropna()
        for column in ("drug_name", "target_name", "action"):
            frame[column] = frame[column].astype(str)
        return (
            frame.drop_duplicates()
            .sort_values(["drug_name", "target_name", "action"])
            .reset_index(drop=True)
        )

    root = Path(config["paths"]["intermediate_dir"]) / "triples" / "groups"
    label_group = str(pair_cfg.get("label_group", "moa"))
    target_group = str(pair_cfg.get("target_group", "target"))
    label_column = str(pair_cfg.get("label_column", "relation"))

    action_frame = pd.read_csv(root / ("%s.csv" % label_group)).drop_duplicates()
    target_frame = pd.read_csv(root / ("%s.csv" % target_group)).drop_duplicates()

    per_drug = (
        action_frame[["head", label_column]]
        .dropna()
        .assign(head=lambda frame: frame["head"].astype(str))
        .assign(**{label_column: lambda frame: frame[label_column].astype(str)})
        .drop_duplicates()
    )
    if bool(pair_cfg.get("drop_ambiguous_drug_actions", True)):
        action_counts = per_drug.groupby("head")[label_column].nunique()
        unambiguous = set(action_counts[action_counts == 1].index.astype(str))
        per_drug = per_drug[per_drug["head"].isin(unambiguous)]
    label_by_drug = dict(zip(per_drug["head"].astype(str), per_drug[label_column].astype(str)))

    rows = []
    for row in target_frame.itertuples(index=False):
        drug = str(row.head)
        target = str(row.tail)
        action = label_by_drug.get(drug)
        if action is None:
            continue
        rows.append({"drug_name": drug, "target_name": target, "action": action})
    pair_frame = pd.DataFrame(rows).drop_duplicates()
    if pair_frame.empty:
        return pair_frame
    return pair_frame.sort_values(["drug_name", "target_name", "action"]).reset_index(drop=True)


def _filter_pairs_for_task(pair_frame, task_cfg):
    import numpy as np

    if pair_frame.empty:
        return pair_frame
    filtered = pair_frame.copy()
    allowed_classes = task_cfg.get("allowed_classes")
    if allowed_classes:
        allowed = {str(label) for label in allowed_classes}
        filtered = filtered[filtered["action"].astype(str).isin(allowed)]
    min_class_count = int(task_cfg.get("min_class_count", 5))
    counts = filtered["action"].astype(str).value_counts()
    valid = set(counts[counts >= min_class_count].index.astype(str))
    filtered = filtered[filtered["action"].astype(str).isin(valid)]
    return filtered.reset_index(drop=True)


def _make_pair_matrix(source, pair_frame, pair_cfg):
    import numpy as np

    entity_to_id = source["entity_to_id"]
    embeddings = _as_real_feature_matrix(source["features"])
    rows = []
    labels = []
    drugs = []
    targets = []
    for row in pair_frame.itertuples(index=False):
        drug_id = entity_to_id.get(str(row.drug_name))
        target_id = entity_to_id.get(str(row.target_name))
        if drug_id is None or target_id is None:
            continue
        drug_vector = embeddings[drug_id]
        target_vector = embeddings[target_id]
        parts = []
        if bool(pair_cfg.get("include_drug_embedding", True)):
            parts.append(drug_vector)
        if bool(pair_cfg.get("include_target_embedding", True)):
            parts.append(target_vector)
        if bool(pair_cfg.get("include_abs_difference", True)):
            parts.append(np.abs(drug_vector - target_vector))
        if bool(pair_cfg.get("include_hadamard_product", True)):
            parts.append(drug_vector * target_vector)
        if not parts:
            raise ValueError("At least one pair feature component must be enabled")
        rows.append(np.concatenate(parts, axis=0))
        labels.append(str(row.action))
        drugs.append(str(row.drug_name))
        targets.append(str(row.target_name))
    if not rows:
        return None
    matrix = np.vstack(rows).astype("float32", copy=False)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    matrix = np.clip(matrix, -1.0e6, 1.0e6)
    return matrix, labels, drugs, targets


def _fold_count(split_name, labels, drugs, targets, requested_folds):
    counts = pd.Series(labels).value_counts()
    folds = min(int(requested_folds), int(counts.min()))
    if split_name == "drug_disjoint":
        folds = min(folds, len(set(drugs)))
    elif split_name == "target_disjoint":
        folds = min(folds, len(set(targets)))
    elif split_name == "double_disjoint":
        folds = min(folds, len(set(drugs)), len(set(targets)))
    return int(folds)


def _effective_fold_count(split_name, entity_folds):
    """Return the number of outer test blocks reported for a split."""

    if split_name == "double_disjoint":
        return int(entity_folds) ** 2
    return int(entity_folds)


def _splitter(split_name, folds, seed):
    from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, StratifiedKFold

    if split_name == "random_pair":
        return StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed), None
    if split_name in {"drug_disjoint", "target_disjoint"}:
        try:
            return (
                StratifiedGroupKFold(
                    n_splits=folds,
                    shuffle=True,
                    random_state=seed,
                ),
                "stratified_group",
            )
        except TypeError:
            return GroupKFold(n_splits=folds), "group"
    raise ValueError("Unsupported action-pair split: %s" % split_name)


def _entity_fold_assignments(labels, groups, folds, seed):
    """Assign every entity group to one approximately stratified partition."""

    import numpy as np
    from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

    labels = np.asarray(labels)
    groups = np.asarray(groups, dtype=object)
    dummy = np.zeros((len(labels), 1), dtype=np.float32)
    try:
        splitter = StratifiedGroupKFold(
            n_splits=folds,
            shuffle=True,
            random_state=seed,
        )
        iterator = splitter.split(dummy, labels, groups=groups)
    except TypeError:
        splitter = GroupKFold(n_splits=folds)
        iterator = splitter.split(dummy, labels, groups=groups)

    assignment = {}
    for fold, (_, test_indices) in enumerate(iterator):
        for index in test_indices:
            group = str(groups[index])
            previous = assignment.setdefault(group, int(fold))
            if previous != int(fold):
                raise RuntimeError("Entity group assigned to multiple folds: %s" % group)
    missing = set(str(group) for group in groups) - set(assignment)
    if missing:
        raise RuntimeError(
            "Entity fold assignment did not cover groups: %s"
            % ", ".join(sorted(missing)[:10])
        )
    return assignment


def _iter_splits(split_name, folds, seed, matrix, encoded, drugs, targets):
    import numpy as np

    if split_name == "double_disjoint":
        drug_assignment = _entity_fold_assignments(encoded, drugs, folds, seed)
        target_assignment = _entity_fold_assignments(
            encoded,
            targets,
            folds,
            seed + 100003,
        )
        drug_folds = np.asarray([drug_assignment[str(drug)] for drug in drugs])
        target_folds = np.asarray([target_assignment[str(target)] for target in targets])
        for drug_fold in range(folds):
            for target_fold in range(folds):
                test_mask = (drug_folds == drug_fold) & (target_folds == target_fold)
                train_mask = (drug_folds != drug_fold) & (target_folds != target_fold)
                test_indices = np.flatnonzero(test_mask)
                train_indices = np.flatnonzero(train_mask)
                if len(test_indices) == 0 or len(train_indices) == 0:
                    continue
                yield train_indices, test_indices
        return

    splitter, splitter_type = _splitter(split_name, folds, seed)
    if split_name == "random_pair":
        yield from splitter.split(matrix, encoded)
        return
    groups = drugs if split_name == "drug_disjoint" else targets
    if splitter_type == "stratified_group":
        yield from splitter.split(matrix, encoded, groups=groups)
    else:
        yield from splitter.split(matrix, encoded, groups=groups)


def _evaluate_pair_source(
    *,
    source,
    pair_frame,
    task_name,
    task_cfg,
    split_name,
    cohort_name,
    pair_cfg,
    random_seed,
    output_dir,
    logger,
):
    import numpy as np
    from sklearn.base import clone
    from sklearn.preprocessing import LabelEncoder

    built = _make_pair_matrix(source, pair_frame, pair_cfg)
    if built is None:
        logger.warning(
            "No action pairs overlap with embeddings for %s/%s",
            source["view"],
            source["model"],
        )
        return [], [], []
    matrix, labels, drugs, targets = built
    if len(set(labels)) < 2:
        logger.warning("Task %s has fewer than two action classes", task_name)
        return [], [], []

    encoder = LabelEncoder()
    encoded = encoder.fit_transform(labels)
    requested_folds = int(task_cfg.get("cv_folds", pair_cfg.get("cv_folds", 5)))
    folds = _fold_count(split_name, labels, drugs, targets, requested_folds)
    effective_folds = _effective_fold_count(split_name, folds)
    repeats = int(task_cfg.get("cv_repeats", pair_cfg.get("cv_repeats", 3)))
    if folds < 2:
        logger.warning("Task %s split %s cannot support at least two folds", task_name, split_name)
        return [], [], []

    classifiers = _build_pair_classifiers(pair_cfg, random_seed)
    repeat_rows = []
    prediction_rows = []
    summary_rows = []

    for classifier_name, classifier_template in classifiers.items():
        logger.info(
            "Action-pair evaluation classifier start: task=%s split=%s source=%s/%s classifier=%s",
            task_name,
            split_name,
            source["view"],
            source["model"],
            classifier_name,
        )
        classifier_repeat_rows = []
        for repeat in range(repeats):
            seed = int(random_seed + repeat)
            oof_pred = np.full(len(encoded), -1, dtype=int)
            oof_score = np.full((len(encoded), len(encoder.classes_)), np.nan, dtype=float)
            fold_ids = np.full(len(encoded), -1, dtype=int)
            for fold, (train_indices, test_indices) in enumerate(
                _iter_splits(split_name, folds, seed, matrix, encoded, drugs, targets)
            ):
                estimator = clone(classifier_template)
                estimator.fit(matrix[train_indices], encoded[train_indices])
                oof_pred[test_indices] = estimator.predict(matrix[test_indices])
                fold_ids[test_indices] = fold
                if hasattr(estimator, "predict_proba"):
                    probabilities = estimator.predict_proba(matrix[test_indices])
                    estimator_classes = [int(value) for value in estimator.classes_]
                    for local_column, class_index in enumerate(estimator_classes):
                        oof_score[test_indices, class_index] = probabilities[:, local_column]
                elif hasattr(estimator, "decision_function"):
                    scores = estimator.decision_function(matrix[test_indices])
                    if scores.ndim == 1 and len(estimator.classes_) == 2:
                        scores = np.column_stack([-scores, scores])
                    estimator_classes = [int(value) for value in estimator.classes_]
                    for local_column, class_index in enumerate(estimator_classes):
                        oof_score[test_indices, class_index] = scores[:, local_column]

            if (oof_pred < 0).any():
                raise RuntimeError(
                    "Out-of-fold prediction did not cover every action-pair sample"
                )
            score_matrix = None
            if not np.isnan(oof_score).all():
                score_matrix = np.nan_to_num(oof_score, nan=0.0)
            metrics = _metric_values(encoded, oof_pred, score_matrix, len(encoder.classes_))
            repeat_row = {
                "task": task_name,
                "cohort": cohort_name,
                "split": split_name,
                "view": source["view"],
                "model": source["model"],
                "classifier": classifier_name,
                "repeat": repeat,
                "seed": seed,
                "sample_count": int(len(encoded)),
                "pair_count": int(len(encoded)),
                "drug_count": int(len(set(drugs))),
                "target_count": int(len(set(targets))),
                "action_count": int(len(encoder.classes_)),
                "cv_folds": int(effective_folds),
                "entity_partitions": int(folds),
                **metrics,
            }
            repeat_rows.append(repeat_row)
            classifier_repeat_rows.append(repeat_row)

            for sample_index, drug_name in enumerate(drugs):
                row = {
                    "task": task_name,
                    "cohort": cohort_name,
                    "split": split_name,
                    "view": source["view"],
                    "model": source["model"],
                    "classifier": classifier_name,
                    "repeat": repeat,
                    "fold": int(fold_ids[sample_index]),
                    "drug_name": drug_name,
                    "target_name": targets[sample_index],
                    "true_label": str(labels[sample_index]),
                    "predicted_label": str(
                        encoder.inverse_transform([oof_pred[sample_index]])[0]
                    ),
                }
                if score_matrix is not None:
                    for class_index, class_name in enumerate(encoder.classes_):
                        row["score__%s" % class_name] = float(
                            score_matrix[sample_index, class_index]
                        )
                prediction_rows.append(row)

        summary = {
            "task": task_name,
            "cohort": cohort_name,
            "split": split_name,
            "view": source["view"],
            "model": source["model"],
            "classifier": classifier_name,
            "sample_count": int(len(encoded)),
            "pair_count": int(len(encoded)),
            "drug_count": int(len(set(drugs))),
            "target_count": int(len(set(targets))),
            "action_count": int(len(encoder.classes_)),
            "cv_folds": int(effective_folds),
            "entity_partitions": int(folds),
            "cv_repeats": int(repeats),
            "status": "success",
        }
        for metric_name in METRIC_NAMES:
            mean, std, ci95 = _summary_stats(
                row[metric_name] for row in classifier_repeat_rows
            )
            summary["%s_mean" % metric_name] = mean
            summary["%s_std" % metric_name] = std
            summary["%s_ci95" % metric_name] = ci95
        summary["best_accuracy"] = summary["accuracy_mean"]
        summary["best_f1_macro"] = summary["f1_macro_mean"]
        summary["best_f1_weighted"] = summary["f1_weighted_mean"]
        summary["best_classifier_by_f1_macro"] = classifier_name
        summary["best_classifier"] = classifier_name
        summary_rows.append(summary)
        logger.info(
            "Action-pair evaluation classifier done: task=%s split=%s source=%s/%s classifier=%s f1_macro=%.4f",
            task_name,
            split_name,
            source["view"],
            source["model"],
            classifier_name,
            float(summary["f1_macro_mean"]),
        )

    ensure_dir(output_dir)
    with (Path(output_dir) / "label_mapping.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "task": task_name,
                "cohort": cohort_name,
                "split": split_name,
                "view": source["view"],
                "model": source["model"],
                "labels": encoder.classes_.tolist(),
                "class_counts": pd.Series(labels).value_counts().to_dict(),
                "sample_count": int(len(encoded)),
                "drug_count": int(len(set(drugs))),
                "target_count": int(len(set(targets))),
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )
    pd.DataFrame(summary_rows).to_csv(Path(output_dir) / "summary.csv", index=False)
    pd.DataFrame(repeat_rows).to_csv(Path(output_dir) / "repeat_metrics.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(Path(output_dir) / "oof_predictions.csv", index=False)
    with (Path(output_dir) / "completion.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "summary_rows": len(summary_rows),
                "repeat_rows": len(repeat_rows),
                "prediction_rows": len(prediction_rows),
                "cv_folds": int(effective_folds),
                "entity_partitions": int(folds),
                "cv_repeats": int(repeats),
                "sample_count": int(len(encoded)),
                "class_count": int(len(encoder.classes_)),
            },
            handle,
            indent=2,
        )
    return summary_rows, repeat_rows, prediction_rows


def evaluate_action_pairs(config, logger):
    """Evaluate drug--target action relation classifiers on fixed embeddings."""

    pair_cfg = dict(config.get("pair_evaluation", {}))
    if not pair_cfg:
        raise ValueError("Missing pair_evaluation config section")

    pair_frame = _load_pair_table(config, pair_cfg)
    if pair_frame.empty:
        raise ValueError("No drug--target action pairs could be constructed")

    classification_root = ensure_dir(
        Path(config["paths"]["processed_dir"]) / "action_pair_classification"
    )
    cohort_name = str(pair_cfg.get("cohort_name", "all_pairs"))
    cohort_root = (
        classification_root / cohort_name
        if pair_cfg.get("cohort_name")
        else classification_root
    )
    summary_rows = []
    repeat_rows = []
    prediction_rows = []
    sources = list(_embedding_sources(config, pair_cfg, logger))
    splits = [str(split) for split in pair_cfg.get("splits", ["random_pair"])]

    for task_name, task_cfg in _pair_label_tasks(pair_cfg).items():
        task_pairs = _filter_pairs_for_task(pair_frame, task_cfg)
        if task_pairs.empty:
            logger.warning("Action-pair task %s has no samples after filtering.", task_name)
            continue
        logger.info(
            "Action-pair task %s has %d pairs, %d drugs, %d targets, %d actions.",
            task_name,
            len(task_pairs),
            task_pairs["drug_name"].nunique(),
            task_pairs["target_name"].nunique(),
            task_pairs["action"].nunique(),
        )
        for source in sources:
            for split_name in splits:
                output_dir = (
                    cohort_root
                    / task_name
                    / split_name
                    / source["view"]
                    / source["model"]
                )
                completed_files = {
                    "summary": output_dir / "summary.csv",
                    "repeats": output_dir / "repeat_metrics.csv",
                    "predictions": output_dir / "oof_predictions.csv",
                }
                completion_path = output_dir / "completion.json"
                if (
                    pair_cfg.get("reuse_completed_results", False)
                    and completion_path.exists()
                    and all(path.exists() for path in completed_files.values())
                ):
                    logger.info(
                        "Reusing completed action-pair result: task=%s split=%s source=%s/%s",
                        task_name,
                        split_name,
                        source["view"],
                        source["model"],
                    )
                    source_summary = pd.read_csv(completed_files["summary"]).to_dict("records")
                    source_repeats = pd.read_csv(completed_files["repeats"]).to_dict("records")
                    source_predictions = pd.read_csv(
                        completed_files["predictions"]
                    ).to_dict("records")
                else:
                    source_summary, source_repeats, source_predictions = _evaluate_pair_source(
                        source=source,
                        pair_frame=task_pairs,
                        task_name=task_name,
                        task_cfg=task_cfg,
                        split_name=split_name,
                        cohort_name=cohort_name,
                        pair_cfg=pair_cfg,
                        random_seed=int(config.get("random_seed", 42)),
                        output_dir=output_dir,
                        logger=logger,
                    )
                summary_rows.extend(source_summary)
                repeat_rows.extend(source_repeats)
                prediction_rows.extend(source_predictions)

    reports_dir = ensure_dir(Path(config["paths"]["reports_dir"]))
    summary_df = pd.DataFrame(summary_rows)
    repeat_df = pd.DataFrame(repeat_rows)
    prediction_df = pd.DataFrame(prediction_rows)
    summary_df.to_csv(reports_dir / "action_pair_summary.csv", index=False)
    repeat_df.to_csv(reports_dir / "action_pair_repeat_metrics.csv", index=False)
    prediction_df.to_csv(reports_dir / "action_pair_oof_predictions.csv", index=False)
    with (reports_dir / "action_pair_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "cohort": cohort_name,
                "pair_table_path": pair_cfg.get("pair_table_path"),
                "tasks": list(_pair_label_tasks(pair_cfg).keys()),
                "splits": splits,
                "views": pair_cfg.get("allowed_views"),
                "models": pair_cfg.get("allowed_models"),
                "classifiers": pair_cfg.get("classifiers"),
                "summary_rows": int(len(summary_df)),
                "repeat_rows": int(len(repeat_df)),
                "prediction_rows": int(len(prediction_df)),
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )
    logger.info(
        "Action-pair evaluation wrote %d summary rows, %d repeat rows, and %d predictions.",
        len(summary_df),
        len(repeat_df),
        len(prediction_df),
    )
    return summary_df
