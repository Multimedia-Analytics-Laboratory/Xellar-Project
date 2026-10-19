"""Quantify cross-stage label exposure between KG training and drug evaluation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .io_utils import ensure_dir, write_json


def _task_settings(evaluation_cfg):
    tasks = evaluation_cfg.get("label_tasks")
    if tasks:
        return tasks
    return {
        "legacy_default": {
            "label_column": evaluation_cfg.get("label_column", "relation"),
            "min_class_count": evaluation_cfg.get("min_class_count", 3),
        }
    }


def audit_label_exposure(config, logger):
    """Recreate KG and classifier splits and count exposed test labels."""

    import numpy as np
    from pykeen.triples import TriplesFactory
    from sklearn.model_selection import train_test_split

    evaluation_cfg = config["evaluation"]
    label_group = str(evaluation_cfg.get("label_group", "moa"))
    triple_root = Path(config["paths"]["intermediate_dir"]) / "triples"
    moa_frame = pd.read_csv(
        triple_root / "groups" / ("%s.csv" % label_group)
    )[["head", "relation", "tail"]].astype(str).drop_duplicates()

    rows = []
    details = {}
    for view_name, groups in config.get("views", {}).items():
        view_path = triple_root / "views" / ("%s.csv" % view_name)
        view_frame = pd.read_csv(view_path)[["head", "relation", "tail"]].astype(str)
        factory = TriplesFactory.from_labeled_triples(view_frame.to_numpy())
        training, testing, validation = factory.split(
            config["training"]["split_ratios"],
            random_state=config["random_seed"],
        )
        split_sets = {
            "kg_train": {
                tuple(values)
                for values in training.label_triples(training.mapped_triples)
            },
            "kg_test": {
                tuple(values)
                for values in testing.label_triples(testing.mapped_triples)
            },
            "kg_validation": {
                tuple(values)
                for values in validation.label_triples(validation.mapped_triples)
            },
        }
        location_by_drug = {}
        for row in moa_frame.itertuples(index=False):
            triple = (row.head, row.relation, row.tail)
            location = "not_in_view"
            for split_name, triples in split_sets.items():
                if triple in triples:
                    location = split_name
                    break
            location_by_drug[row.head] = location

        details[view_name] = {
            "groups": groups,
            "moa_location_counts": pd.Series(location_by_drug).value_counts().to_dict(),
        }

        for task_name, task_cfg in _task_settings(evaluation_cfg).items():
            label_column = str(task_cfg["label_column"])
            labels = moa_frame[["head", label_column]].drop_duplicates("head")
            allowed_classes = task_cfg.get("allowed_classes")
            if allowed_classes:
                allowed_classes = {str(label) for label in allowed_classes}
                labels = labels[
                    labels[label_column].astype(str).isin(allowed_classes)
                ]
            counts = labels[label_column].value_counts()
            valid = counts[
                counts >= int(task_cfg.get("min_class_count", 3))
            ].index
            labels = labels[labels[label_column].isin(valid)].reset_index(drop=True)
            indices = np.arange(len(labels))
            _, test_indices = train_test_split(
                indices,
                test_size=float(evaluation_cfg.get("test_size", 0.3)),
                random_state=config["random_seed"],
                stratify=labels[label_column],
            )
            test_drugs = labels.iloc[test_indices]["head"].astype(str).tolist()
            exposed = sum(
                location_by_drug.get(drug_name) == "kg_train"
                for drug_name in test_drugs
            )
            rows.append({
                "view": view_name,
                "task": task_name,
                "label_column": label_column,
                "classifier_test_drugs": int(len(test_drugs)),
                "test_drugs_with_moa_in_kg_train": int(exposed),
                "direct_label_exposure_rate": (
                    float(exposed / len(test_drugs)) if test_drugs else 0.0
                ),
                "label_group_in_view": bool(label_group in groups),
            })

    reports_dir = ensure_dir(Path(config["paths"]["reports_dir"]))
    output_frame = pd.DataFrame(rows)
    output_frame.to_csv(reports_dir / "label_exposure_audit.csv", index=False)
    write_json(
        reports_dir / "label_exposure_audit.json",
        {
            "label_group": label_group,
            "views": details,
            "downstream_overlap": rows,
            "interpretation": (
                "Triple splits can be disjoint while classifier test drugs still "
                "have their MoA triples in the KG training split."
            ),
        },
    )
    logger.info("Wrote label exposure audit to %s", reports_dir)
    return output_frame
