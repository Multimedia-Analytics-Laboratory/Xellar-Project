"""Leakage-controlled downstream evaluation for MoA classification.

This module deliberately keeps MoA labels outside representation learning.
It evaluates fixed, label-free feature matrices with repeated out-of-fold
predictions so every reported score is computed on drugs not used to fit the
classifier that produced their prediction.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from .io_utils import ensure_dir


METRIC_NAMES = (
    "accuracy",
    "balanced_accuracy",
    "f1_macro",
    "f1_weighted",
    "precision_macro",
    "recall_macro",
    "auc_ovr_macro",
)


def _build_classifiers(random_seed):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=500,
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
                    C=1.0,
                    class_weight="balanced",
                    max_iter=3000,
                    random_state=random_seed,
                ),
            ),
        ]),
        "SVM (RBF)": Pipeline([
            ("scale", StandardScaler()),
            (
                "model",
                SVC(
                    C=1.0,
                    kernel="rbf",
                    class_weight="balanced",
                    max_iter=10000,
                    random_state=random_seed,
                ),
            ),
        ]),
        "KNN (k=5)": Pipeline([
            ("scale", StandardScaler()),
            (
                "model",
                KNeighborsClassifier(
                    n_neighbors=5,
                    weights="uniform",
                ),
            ),
        ]),
    }


def _as_real_feature_matrix(features):
    import numpy as np

    matrix = np.asarray(features)
    if np.iscomplexobj(matrix):
        matrix = np.concatenate([matrix.real, matrix.imag], axis=-1)
    if matrix.ndim < 2:
        raise ValueError("Expected at least a two-dimensional feature matrix")
    matrix = matrix.reshape(matrix.shape[0], -1).astype(np.float32, copy=False)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    # A few factorization embeddings contain extremely large but finite values.
    # Clipping prevents overflow in pairwise-distance calculations while the
    # downstream pipelines still standardize each feature.
    return np.clip(matrix, -1.0e6, 1.0e6)


def _label_tasks(evaluation_cfg):
    configured = evaluation_cfg.get("label_tasks")
    if configured:
        return configured
    return {
        "default": {
            "label_column": str(evaluation_cfg.get("label_column", "relation")),
            "min_class_count": int(evaluation_cfg.get("min_class_count", 3)),
        }
    }


def _task_allows_source(task_cfg, source, cohort_name):
    """Return whether a task should be evaluated for a source/cohort pair."""

    allowed_views = task_cfg.get("allowed_views")
    if allowed_views and source["view"] not in {str(view) for view in allowed_views}:
        return False

    allowed_models = task_cfg.get("allowed_models")
    if allowed_models and source["model"] not in {str(model) for model in allowed_models}:
        return False

    allowed_cohorts = task_cfg.get("allowed_cohorts")
    if allowed_cohorts and cohort_name not in {
        str(cohort) for cohort in allowed_cohorts
    }:
        return False

    return True


def _metric_values(y_true, y_pred, y_score, class_count):
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    values = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "auc_ovr_macro": math.nan,
    }
    if y_score is not None:
        try:
            values["auc_ovr_macro"] = float(
                roc_auc_score(
                    y_true,
                    y_score,
                    labels=list(range(class_count)),
                    multi_class="ovr",
                    average="macro",
                )
            )
        except ValueError:
            pass
    return values


def _summary_stats(values):
    import numpy as np

    array = np.asarray([value for value in values if not math.isnan(value)], dtype=float)
    if len(array) == 0:
        return math.nan, math.nan, math.nan
    mean = float(array.mean())
    if len(array) == 1:
        return mean, 0.0, 0.0
    std = float(array.std(ddof=1))
    ci95 = float(1.96 * std / math.sqrt(len(array)))
    return mean, std, ci95


def _evaluate_source(
    *,
    features,
    drug_names,
    moa_triples,
    task_name,
    task_cfg,
    cohort_name,
    allowed_drugs,
    view_name,
    model_name,
    evaluation_cfg,
    random_seed,
    output_dir,
    logger,
):
    import numpy as np
    from sklearn.base import clone
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import LabelEncoder

    label_column = str(task_cfg["label_column"])
    min_class_count = int(task_cfg.get("min_class_count", 3))
    label_by_drug = dict(
        zip(
            moa_triples["head"].astype(str),
            moa_triples[label_column].astype(str),
        )
    )

    selected_indices = []
    labels = []
    selected_drugs = []
    for index, drug_name in enumerate(drug_names):
        if allowed_drugs is not None and str(drug_name) not in allowed_drugs:
            continue
        label = label_by_drug.get(str(drug_name))
        if label is None:
            continue
        selected_indices.append(index)
        labels.append(label)
        selected_drugs.append(str(drug_name))

    if not selected_indices:
        logger.warning("No labeled drugs overlap with %s / %s", view_name, model_name)
        return [], [], []

    matrix = _as_real_feature_matrix(features)[selected_indices]
    labels = np.asarray(labels, dtype=object)
    allowed_classes = task_cfg.get("allowed_classes")
    if allowed_classes:
        allowed_classes = {str(label) for label in allowed_classes}
        allowed_keep = np.asarray(
            [str(label) in allowed_classes for label in labels],
            dtype=bool,
        )
        matrix = matrix[allowed_keep]
        labels = labels[allowed_keep]
        selected_drugs = [
            drug_name
            for drug_name, keep_value in zip(selected_drugs, allowed_keep.tolist())
            if keep_value
        ]
    counts = pd.Series(labels).value_counts()
    valid_classes = counts[counts >= min_class_count].index.tolist()
    keep = np.asarray([label in valid_classes for label in labels], dtype=bool)
    matrix = matrix[keep]
    labels = labels[keep]
    selected_drugs = [
        drug_name for drug_name, keep_value in zip(selected_drugs, keep.tolist())
        if keep_value
    ]
    # Stable drug ordering makes fold assignments identical across KG,
    # target-only, and multimodal feature sources for the same cohort/task.
    order = np.argsort(np.asarray(selected_drugs, dtype=str))
    matrix = matrix[order]
    labels = labels[order]
    selected_drugs = [selected_drugs[index] for index in order.tolist()]

    if len(valid_classes) < 2:
        logger.warning(
            "Task %s has fewer than two classes for %s / %s",
            task_name,
            view_name,
            model_name,
        )
        return [], [], []

    encoder = LabelEncoder()
    encoded = encoder.fit_transform(labels)
    requested_folds = int(task_cfg.get("cv_folds", evaluation_cfg.get("cv_folds", 5)))
    folds = min(requested_folds, int(pd.Series(encoded).value_counts().min()))
    repeats = int(task_cfg.get("cv_repeats", evaluation_cfg.get("cv_repeats", 5)))
    if folds < 2:
        logger.warning("Task %s cannot support at least two folds", task_name)
        return [], [], []

    repeat_rows = []
    prediction_rows = []
    summary_rows = []
    classifiers = _build_classifiers(random_seed)

    for classifier_name, classifier_template in classifiers.items():
        logger.info(
            "Strict evaluation classifier start: task=%s cohort=%s source=%s/%s "
            "classifier=%s",
            task_name,
            cohort_name,
            view_name,
            model_name,
            classifier_name,
        )
        classifier_repeat_rows = []
        for repeat in range(repeats):
            seed = int(random_seed + repeat)
            splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
            oof_pred = np.full(len(encoded), -1, dtype=int)
            oof_score = np.full(
                (len(encoded), len(encoder.classes_)),
                np.nan,
                dtype=float,
            )
            fold_ids = np.full(len(encoded), -1, dtype=int)

            for fold, (train_indices, test_indices) in enumerate(
                splitter.split(matrix, encoded)
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
                raise RuntimeError("Out-of-fold prediction did not cover every sample")

            score_matrix = None
            if not np.isnan(oof_score).all():
                score_matrix = np.nan_to_num(oof_score, nan=0.0)
            metrics = _metric_values(
                encoded,
                oof_pred,
                score_matrix,
                len(encoder.classes_),
            )
            repeat_row = {
                "task": task_name,
                "cohort": cohort_name,
                "view": view_name,
                "model": model_name,
                "classifier": classifier_name,
                "repeat": repeat,
                "seed": seed,
                "sample_count": int(len(encoded)),
                "class_count": int(len(encoder.classes_)),
                "cv_folds": int(folds),
                **metrics,
            }
            repeat_rows.append(repeat_row)
            classifier_repeat_rows.append(repeat_row)

            for sample_index, drug_name in enumerate(selected_drugs):
                row = {
                    "task": task_name,
                    "cohort": cohort_name,
                    "view": view_name,
                    "model": model_name,
                    "classifier": classifier_name,
                    "repeat": repeat,
                    "fold": int(fold_ids[sample_index]),
                    "drug_name": drug_name,
                    "true_label": str(labels[sample_index]),
                    "predicted_label": str(
                        encoder.inverse_transform([oof_pred[sample_index]])[0]
                    ),
                }
                if score_matrix is not None:
                    for class_index, class_name in enumerate(encoder.classes_):
                        row["probability__%s" % class_name] = float(
                            score_matrix[sample_index, class_index]
                        )
                prediction_rows.append(row)

        summary = {
            "task": task_name,
            "cohort": cohort_name,
            "view": view_name,
            "model": model_name,
            "classifier": classifier_name,
            "sample_count": int(len(encoded)),
            "moa_count": int(len(encoder.classes_)),
            "cv_folds": int(folds),
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

        # Compatibility columns keep the existing markdown summary usable.
        summary["best_accuracy"] = summary["accuracy_mean"]
        summary["best_f1_macro"] = summary["f1_macro_mean"]
        summary["best_f1_weighted"] = summary["f1_weighted_mean"]
        summary["best_classifier_by_f1_macro"] = classifier_name
        summary["best_classifier"] = classifier_name
        summary_rows.append(summary)
        logger.info(
            "Strict evaluation classifier done: task=%s cohort=%s source=%s/%s "
            "classifier=%s f1_macro=%.4f",
            task_name,
            cohort_name,
            view_name,
            model_name,
            classifier_name,
            float(summary["f1_macro_mean"]),
        )

    ensure_dir(output_dir)
    with (
        Path(output_dir)
        / ("label_mapping__%s__%s.json" % (task_name, cohort_name))
    ).open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(
            {
                "task": task_name,
                "cohort": cohort_name,
                "label_column": label_column,
                "labels": encoder.classes_.tolist(),
                "class_counts": pd.Series(labels).value_counts().to_dict(),
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )
    pd.DataFrame(summary_rows).to_csv(
        Path(output_dir) / "summary.csv",
        index=False,
    )
    pd.DataFrame(repeat_rows).to_csv(
        Path(output_dir) / "repeat_metrics.csv",
        index=False,
    )
    pd.DataFrame(prediction_rows).to_csv(
        Path(output_dir) / "oof_predictions.csv",
        index=False,
    )
    with (Path(output_dir) / "completion.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            {
                "summary_rows": len(summary_rows),
                "repeat_rows": len(repeat_rows),
                "prediction_rows": len(prediction_rows),
                "cv_folds": int(folds),
                "cv_repeats": int(repeats),
                "sample_count": int(len(encoded)),
                "class_count": int(len(encoder.classes_)),
            },
            handle,
            indent=2,
        )
    return summary_rows, repeat_rows, prediction_rows


def _embedding_sources(config, logger):
    import numpy as np

    embedding_root = Path(config["paths"]["processed_dir"]) / "embeddings"
    for view_name in config.get("views", {}):
        view_dir = embedding_root / view_name
        if not view_dir.exists():
            logger.warning("Embedding directory missing for strict view %s", view_name)
            continue
        for model_dir in sorted(view_dir.iterdir()):
            entity_path = model_dir / "entity_to_id.json"
            embedding_path = model_dir / "entity_embeddings.npy"
            if not entity_path.exists() or not embedding_path.exists():
                continue
            with entity_path.open("r", encoding="utf-8") as handle:
                entity_to_id = json.load(handle)
            ordered = sorted(entity_to_id.items(), key=lambda item: int(item[1]))
            yield {
                "view": view_name,
                "model": model_dir.name,
                "drug_names": [name for name, _ in ordered],
                "features": np.load(str(embedding_path)),
            }


def _target_multihot_source(config, moa_triples):
    import numpy as np

    target_path = (
        Path(config["paths"]["intermediate_dir"])
        / "triples"
        / "groups"
        / "target.csv"
    )
    target_frame = pd.read_csv(target_path)
    drug_names = sorted(moa_triples["head"].astype(str).unique().tolist())
    targets = sorted(target_frame["tail"].astype(str).unique().tolist())
    drug_to_index = {drug: index for index, drug in enumerate(drug_names)}
    target_to_index = {target: index for index, target in enumerate(targets)}
    matrix = np.zeros((len(drug_names), len(targets)), dtype=np.float32)
    for row in target_frame.itertuples(index=False):
        drug_index = drug_to_index.get(str(row.head))
        target_index = target_to_index.get(str(row.tail))
        if drug_index is not None and target_index is not None:
            matrix[drug_index, target_index] = 1.0
    return {
        "view": "baseline",
        "model": "TargetMultiHot",
        "drug_names": drug_names,
        "features": matrix,
    }


def _cohorts(evaluation_cfg, logger):
    cohorts = evaluation_cfg.get("cohorts")
    if not cohorts:
        return {"all": None}

    resolved = {}
    for cohort_name, cohort_cfg in cohorts.items():
        if not cohort_cfg:
            resolved[cohort_name] = None
            continue
        allowed = None
        if cohort_cfg.get("table_path"):
            table_path = Path(str(cohort_cfg["table_path"])).expanduser()
            if not table_path.exists():
                logger.warning(
                    "Cohort %s skipped because table does not exist: %s",
                    cohort_name,
                    table_path,
                )
                continue
            frame = pd.read_csv(table_path)
            drug_column = str(cohort_cfg.get("drug_column", "drug_name"))
            keep = pd.Series(True, index=frame.index)
            for column in cohort_cfg.get("require_nonempty", []):
                keep &= frame[column].fillna("").astype(str).str.len().gt(0)
            for column in cohort_cfg.get("require_positive", []):
                keep &= pd.to_numeric(
                    frame[column],
                    errors="coerce",
                ).fillna(0).gt(0)
            allowed = set(frame.loc[keep, drug_column].astype(str))
        require_in_view = cohort_cfg.get("require_in_view")
        if require_in_view:
            config = evaluation_cfg.get("_runtime_config")
            if config is None:
                raise ValueError(
                    "Cohort require_in_view needs runtime config injection"
                )
            view_path = (
                Path(config["paths"]["intermediate_dir"])
                / "triples"
                / "views"
                / ("%s.csv" % require_in_view)
            )
            view = pd.read_csv(view_path)
            view_entities = set(view["head"].astype(str)) | set(
                view["tail"].astype(str)
            )
            allowed = view_entities if allowed is None else allowed & view_entities
        resolved[cohort_name] = allowed
    return resolved


def evaluate_all_views_strict(config, logger):
    """Evaluate label-free embeddings and simple target baselines."""

    evaluation_cfg = dict(config["evaluation"])
    evaluation_cfg["_runtime_config"] = config
    label_group = str(evaluation_cfg.get("label_group", "moa"))
    leaking_views = [
        view_name
        for view_name, groups in config.get("views", {}).items()
        if label_group in groups
    ]
    if leaking_views:
        raise ValueError(
            "Strict evaluation refuses views containing label group %r: %s"
            % (label_group, ", ".join(sorted(leaking_views)))
        )

    root = Path(config["paths"]["intermediate_dir"]) / "triples" / "groups"
    moa_triples = pd.read_csv(root / ("%s.csv" % label_group)).drop_duplicates()
    classification_root = ensure_dir(
        Path(config["paths"]["processed_dir"]) / "strict_classification"
    )

    sources = list(_embedding_sources(config, logger))
    if evaluation_cfg.get("include_target_multihot_baseline", True):
        sources.append(_target_multihot_source(config, moa_triples))

    summary_rows = []
    repeat_rows = []
    prediction_rows = []
    cohort_items = list(_cohorts(evaluation_cfg, logger).items())
    for source in sources:
        for task_name, task_cfg in _label_tasks(evaluation_cfg).items():
            for cohort_name, allowed_drugs in cohort_items:
                if not _task_allows_source(task_cfg, source, cohort_name):
                    continue
                logger.info(
                    "Strict evaluation: task=%s cohort=%s source=%s/%s",
                    task_name,
                    cohort_name,
                    source["view"],
                    source["model"],
                )
                output_dir = (
                    classification_root
                    / task_name
                    / cohort_name
                    / source["view"]
                    / source["model"]
                )
                completion_path = output_dir / "completion.json"
                completed_files = {
                    "summary": output_dir / "summary.csv",
                    "repeats": output_dir / "repeat_metrics.csv",
                    "predictions": output_dir / "oof_predictions.csv",
                }
                if (
                    evaluation_cfg.get("reuse_completed_results", False)
                    and completion_path.exists()
                    and all(path.exists() for path in completed_files.values())
                ):
                    logger.info(
                        "Reusing completed strict result: task=%s cohort=%s "
                        "source=%s/%s",
                        task_name,
                        cohort_name,
                        source["view"],
                        source["model"],
                    )
                    source_summary = pd.read_csv(
                        completed_files["summary"]
                    ).to_dict("records")
                    source_repeats = pd.read_csv(
                        completed_files["repeats"]
                    ).to_dict("records")
                    source_predictions = pd.read_csv(
                        completed_files["predictions"]
                    ).to_dict("records")
                else:
                    source_summary, source_repeats, source_predictions = _evaluate_source(
                        features=source["features"],
                        drug_names=source["drug_names"],
                        moa_triples=moa_triples,
                        task_name=task_name,
                        task_cfg=task_cfg,
                        cohort_name=cohort_name,
                        allowed_drugs=allowed_drugs,
                        view_name=source["view"],
                        model_name=source["model"],
                        evaluation_cfg=evaluation_cfg,
                        random_seed=int(config["random_seed"]),
                        output_dir=output_dir,
                        logger=logger,
                    )
                summary_rows.extend(source_summary)
                repeat_rows.extend(source_repeats)
                prediction_rows.extend(source_predictions)

    reports_dir = ensure_dir(Path(config["paths"]["reports_dir"]))
    pd.DataFrame(repeat_rows).to_csv(
        reports_dir / "strict_repeat_metrics.csv",
        index=False,
    )
    pd.DataFrame(prediction_rows).to_csv(
        reports_dir / "strict_oof_predictions.csv",
        index=False,
    )
    logger.info(
        "Strict evaluation wrote %d summary rows, %d repeat rows, and %d predictions.",
        len(summary_rows),
        len(repeat_rows),
        len(prediction_rows),
    )
    return pd.DataFrame(summary_rows)
