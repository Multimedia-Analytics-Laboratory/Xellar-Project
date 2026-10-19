"""Downstream classifier evaluation for saved entity embeddings."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .io_utils import ensure_dir


def _build_classifiers(random_seed):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC

    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=random_seed,
            max_depth=10,
        ),
        "Logistic Regression": LogisticRegression(
            random_state=random_seed,
            max_iter=1000,
        ),
        "SVM (RBF)": SVC(
            kernel="rbf",
            random_state=random_seed,
            probability=True,
        ),
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
    }


def _safe_cv_folds(y_train, requested_folds):
    value_counts = pd.Series(y_train).value_counts()
    if value_counts.empty:
        return 0
    max_folds = int(value_counts.min())
    return max(0, min(int(requested_folds), max_folds))


def _export_drug_embeddings(drug_names, moa_labels, features, output_dir, export_similarity):
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    export_dir = ensure_dir(output_dir)
    data = {
        "drug_name": drug_names,
        "moa": moa_labels,
    }
    for index in range(features.shape[1]):
        data["embedding_%d" % index] = features[:, index]

    embedding_df = pd.DataFrame(data)
    embedding_df.to_csv(export_dir / "drug_embeddings.csv", index=False)

    if export_similarity and len(features) > 1:
        similarity_matrix = cosine_similarity(features)
        similarity_df = pd.DataFrame(
            similarity_matrix,
            index=drug_names,
            columns=drug_names,
        )
        similarity_df.to_csv(export_dir / "drug_similarity.csv")


def _max_present(values):
    numeric_values = [float(value) for value in values if value is not None]
    if not numeric_values:
        return 0.0
    return max(numeric_values)


def evaluate_all_views(config, logger):
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
    from sklearn.preprocessing import LabelEncoder

    paths = config["paths"]
    evaluation_cfg = config["evaluation"]
    embedding_root = Path(paths["processed_dir"]) / "embeddings"
    classification_root = ensure_dir(Path(paths["processed_dir"]) / "classification")
    moa_path = Path(paths["intermediate_dir"]) / "triples" / "groups" / "moa.csv"
    moa_triples = pd.read_csv(moa_path).drop_duplicates()

    summary_rows = []

    for view_name in config.get("views", {}).keys():
        view_embedding_dir = embedding_root / view_name
        if not view_embedding_dir.exists():
            logger.warning("Embedding directory missing for view %s, skipping.", view_name)
            continue

        for model_dir in sorted(view_embedding_dir.iterdir()):
            if not model_dir.is_dir():
                continue

            entity_path = model_dir / "entity_to_id.json"
            embedding_path = model_dir / "entity_embeddings.npy"
            if not entity_path.exists() or not embedding_path.exists():
                logger.warning("Missing embedding artifacts in %s, skipping.", model_dir)
                continue

            logger.info("Evaluating classifiers for %s / %s", view_name, model_dir.name)
            with entity_path.open("r", encoding="utf-8") as handle:
                entity_to_id = json.load(handle)
            embeddings = np.load(str(embedding_path))

            label_column = str(evaluation_cfg.get("label_column", "tail"))
            drug_to_moa = {}
            for _, row in moa_triples.iterrows():
                drug_to_moa[str(row["head"])] = str(row[label_column])

            features = []
            labels = []
            drug_names = []
            for drug_name, moa_label in drug_to_moa.items():
                entity_id = entity_to_id.get(drug_name)
                if entity_id is None:
                    continue
                features.append(embeddings[int(entity_id)])
                labels.append(moa_label)
                drug_names.append(drug_name)

            if len(features) == 0:
                logger.warning("No overlapping drugs found for %s / %s", view_name, model_dir.name)
                continue

            features = np.asarray(features)
            labels = np.asarray(labels)

            valid_classes = (
                pd.Series(labels)
                .value_counts()
                .loc[lambda counts: counts >= int(evaluation_cfg["min_class_count"])]
                .index
                .tolist()
            )
            keep_mask = np.array([label in valid_classes for label in labels])
            features = features[keep_mask]
            labels = labels[keep_mask]
            drug_names = [drug for drug, keep in zip(drug_names, keep_mask.tolist()) if keep]

            if len(set(labels.tolist())) < 2:
                logger.warning("Not enough classes for %s / %s after filtering.", view_name, model_dir.name)
                continue

            label_encoder = LabelEncoder()
            encoded_labels = label_encoder.fit_transform(labels)

            X_train, X_test, y_train, y_test = train_test_split(
                features,
                encoded_labels,
                test_size=float(evaluation_cfg["test_size"]),
                random_state=config["random_seed"],
                stratify=encoded_labels,
            )

            model_output_dir = ensure_dir(classification_root / view_name / model_dir.name)
            classifier_rows = []
            status = "success"
            classifiers = _build_classifiers(config["random_seed"])
            export_supported = features.ndim == 2 and not np.iscomplexobj(features)
            if export_supported:
                _export_drug_embeddings(
                    drug_names=drug_names,
                    moa_labels=labels,
                    features=features,
                    output_dir=model_output_dir,
                    export_similarity=bool(evaluation_cfg.get("export_similarity_matrices", True)),
                )
            else:
                status = "unsupported_embedding_format"
                logger.warning(
                    "Unsupported embedding format detected for %s / %s "
                    "(ndim=%d, complex=%s); classifier-level metrics will default to zero if all classifiers fail.",
                    view_name,
                    model_dir.name,
                    features.ndim,
                    np.iscomplexobj(features),
                )

            for classifier_name, classifier in classifiers.items():
                try:
                    n_folds = _safe_cv_folds(y_train, int(evaluation_cfg.get("cv_folds", 3)))
                    if n_folds >= 2:
                        cv = StratifiedKFold(
                            n_splits=n_folds,
                            shuffle=True,
                            random_state=config["random_seed"],
                        )
                        cv_scores = cross_val_score(
                            classifier,
                            X_train,
                            y_train,
                            cv=cv,
                            scoring=str(evaluation_cfg.get("cv_scoring", "accuracy")),
                        )
                        cv_mean = float(cv_scores.mean())
                        cv_std = float(cv_scores.std())
                    else:
                        cv_mean = 0.0
                        cv_std = 0.0

                    classifier.fit(X_train, y_train)
                    y_pred = classifier.predict(X_test)

                    auc_ovr = None
                    auc_ovo = None
                    try:
                        if hasattr(classifier, "predict_proba"):
                            y_proba = classifier.predict_proba(X_test)
                            auc_ovr = float(roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro"))
                            auc_ovo = float(roc_auc_score(y_test, y_proba, multi_class="ovo", average="macro"))
                    except Exception as exc:
                        logger.warning(
                            "AUC calculation failed for %s / %s / %s: %s",
                            view_name,
                            model_dir.name,
                            classifier_name,
                            exc,
                        )

                    row = {
                        "view": view_name,
                        "model": model_dir.name,
                        "classifier": classifier_name,
                        "sample_count": int(len(features)),
                        "moa_count": int(len(valid_classes)),
                        "cv_mean": cv_mean,
                        "cv_std": cv_std,
                        "test_accuracy": float(accuracy_score(y_test, y_pred)),
                        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
                        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted")),
                        "precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
                        "precision_weighted": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
                        "recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
                        "recall_weighted": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
                        "auc_ovr": auc_ovr,
                        "auc_ovo": auc_ovo,
                        "status": "success",
                    }
                    classifier_rows.append(row)
                except Exception as exc:
                    logger.warning(
                        "Classifier %s failed for %s / %s: %s",
                        classifier_name,
                        view_name,
                        model_dir.name,
                        exc,
                    )

            classifier_df = pd.DataFrame(classifier_rows)
            classifier_df.to_csv(model_output_dir / "classifier_metrics.csv", index=False)

            summary_row = {
                "view": view_name,
                "model": model_dir.name,
                "sample_count": int(len(features)),
                "moa_count": int(len(valid_classes)),
                "best_accuracy": 0.0,
                "best_f1_macro": 0.0,
                "best_f1_weighted": 0.0,
                "best_precision_macro": 0.0,
                "best_precision_weighted": 0.0,
                "best_recall_macro": 0.0,
                "best_recall_weighted": 0.0,
                "best_auc_ovr": 0.0,
                "best_auc_ovo": 0.0,
                "best_cv_mean": 0.0,
                "best_classifier_by_accuracy": None,
                "best_classifier_by_f1_macro": None,
                "best_classifier_by_f1_weighted": None,
                "status": status,
            }

            if classifier_rows:
                summary_row.update({
                    "best_accuracy": _max_present(row["test_accuracy"] for row in classifier_rows),
                    "best_f1_macro": _max_present(row["f1_macro"] for row in classifier_rows),
                    "best_f1_weighted": _max_present(row["f1_weighted"] for row in classifier_rows),
                    "best_precision_macro": _max_present(row["precision_macro"] for row in classifier_rows),
                    "best_precision_weighted": _max_present(row["precision_weighted"] for row in classifier_rows),
                    "best_recall_macro": _max_present(row["recall_macro"] for row in classifier_rows),
                    "best_recall_weighted": _max_present(row["recall_weighted"] for row in classifier_rows),
                    "best_auc_ovr": _max_present(row["auc_ovr"] for row in classifier_rows),
                    "best_auc_ovo": _max_present(row["auc_ovo"] for row in classifier_rows),
                    "best_cv_mean": _max_present(row["cv_mean"] for row in classifier_rows),
                    "best_classifier_by_accuracy": max(classifier_rows, key=lambda row: row["test_accuracy"])["classifier"],
                    "best_classifier_by_f1_macro": max(classifier_rows, key=lambda row: row["f1_macro"])["classifier"],
                    "best_classifier_by_f1_weighted": max(classifier_rows, key=lambda row: row["f1_weighted"])["classifier"],
                    "status": "success",
                })
            else:
                logger.warning("No classifier metrics produced for %s / %s", view_name, model_dir.name)

            summary_row["best_test_accuracy"] = summary_row["best_accuracy"]
            summary_row["best_test_f1_macro"] = summary_row["best_f1_macro"]
            summary_row["best_test_f1_weighted"] = summary_row["best_f1_weighted"]
            summary_row["best_cv_f1_weighted_mean"] = summary_row["best_cv_mean"]
            summary_row["best_classifier"] = summary_row["best_classifier_by_f1_macro"]
            summary_rows.append(summary_row)

            with (model_output_dir / "label_mapping.json").open("w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "label_column": label_column,
                        "labels": label_encoder.classes_.tolist(),
                        "class_counts": pd.Series(labels).value_counts().to_dict(),
                    },
                    handle,
                    indent=2,
                    ensure_ascii=False,
                )

            if classifier_rows:
                logger.info(
                    "Best macro F1 for %s / %s: %.4f | Best accuracy: %.4f",
                    view_name,
                    model_dir.name,
                    summary_row["best_f1_macro"],
                    summary_row["best_accuracy"],
                )

    return pd.DataFrame(summary_rows)
