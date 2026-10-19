"""Pair-level DeepDTA/GraphDTA/DruGNNosis-style action baselines.

The drug-level multimodal baselines in :mod:`kgml_pipeline.multimodal_evaluation`
average all DrugBank targets for one drug.  That is not appropriate for the
target-conditioned action task.  This module instead constructs one row per
matched drug--target pair:

    drug SMILES + one target protein sequence -> action class

DrugBank pharmacological action fields are intentionally ignored.  DrugBank is
used only for drug structure and target sequence metadata.
"""

from __future__ import annotations

import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

from .io_utils import ensure_dir
from .multimodal_data import (
    NS,
    _clean_fasta_sequence,
    _clean_text,
    _primary_drugbank_id,
)
from .multimodal_evaluation import (
    DRUGNNOSIS_MODEL_NAMES,
    PROTEIN_ALPHABET,
    _array_dataset,
    _decode_json_strings,
    _drug_text,
    _encode_text,
    _make_model,
    _molecular_graph,
    _move_batch,
    _predict,
    _target_texts,
    _train_one_fold,
    _vocabulary,
)
from .pair_evaluation import (
    _effective_fold_count,
    _filter_pairs_for_task,
    _fold_count,
    _iter_splits,
    _load_pair_table,
    _pair_label_tasks,
)
from .strict_evaluation import METRIC_NAMES, _metric_values, _summary_stats


def _symbol_key(value):
    text = _clean_text(value).lower()
    if not text:
        return ""
    return text


def _relaxed_symbol_key(value):
    text = _symbol_key(value)
    return re.sub(r"[^a-z0-9]+", "", text)


def _direct_text(element, tag):
    child = element.find(NS + tag)
    if child is None or child.text is None:
        return ""
    return _clean_text(child.text)


def _child_texts(parent, child_tag):
    if parent is None:
        return []
    values = []
    for child in parent.findall(NS + child_tag):
        text = _clean_text(child.text)
        if text:
            values.append(text)
    return values


def _target_sequence_records(drug):
    records = []
    targets = drug.find(NS + "targets")
    if targets is None:
        return records
    for target in targets.findall(NS + "target"):
        polypeptide = target.find(NS + "polypeptide")
        if polypeptide is None:
            continue
        sequence_element = polypeptide.find(NS + "amino-acid-sequence")
        sequence = _clean_fasta_sequence(
            sequence_element.text if sequence_element is not None else ""
        )
        if not sequence:
            continue
        target_name = _direct_text(target, "name") or _direct_text(polypeptide, "name")
        gene_name = _direct_text(polypeptide, "gene-name")
        uniprot_id = _clean_text(polypeptide.attrib.get("id", ""))
        synonyms = _child_texts(polypeptide.find(NS + "synonyms"), "synonym")
        keys = set()
        for value in [gene_name, target_name, uniprot_id, *synonyms]:
            key = _symbol_key(value)
            relaxed = _relaxed_symbol_key(value)
            if key:
                keys.add(key)
            if relaxed:
                keys.add(relaxed)
        if not keys:
            continue
        records.append({
            "keys": sorted(keys),
            "target_gene_name": gene_name,
            "target_uniprot_id": uniprot_id,
            "target_drugbank_name": target_name,
            "target_sequence": sequence,
        })
    return records


def _drugbank_target_index(xml_path, drugbank_ids, logger):
    requested = {str(value) for value in drugbank_ids if str(value).strip()}
    index = {}
    ambiguous = {}
    seen_drugs = set()
    logger.info(
        "Parsing DrugBank XML target sequences for %d matched drugs: %s",
        len(requested),
        xml_path,
    )
    for _, drug in ET.iterparse(str(xml_path), events=("end",)):
        if drug.tag != NS + "drug":
            continue
        drugbank_id = _primary_drugbank_id(drug)
        if drugbank_id in requested:
            seen_drugs.add(drugbank_id)
            for record in _target_sequence_records(drug):
                for key in record["keys"]:
                    compound_key = (drugbank_id, key)
                    if compound_key in index:
                        previous = index[compound_key]
                        if previous["target_sequence"] != record["target_sequence"]:
                            ambiguous.setdefault(drugbank_id, set()).add(key)
                        continue
                    index[compound_key] = record
        drug.clear()
    if ambiguous:
        logger.warning(
            "DrugBank target sequence index had ambiguous keys for %d drugs; first occurrence was kept.",
            len(ambiguous),
        )
    logger.info(
        "DrugBank pair target index covers %d/%d requested drugs and %d drug-target keys.",
        len(seen_drugs),
        len(requested),
        len(index),
    )
    return index


def _filter_to_view_entities(config, frame, view_name, logger):
    if not view_name:
        return frame
    view_path = (
        Path(config["paths"]["intermediate_dir"])
        / "triples"
        / "views"
        / ("%s.csv" % view_name)
    )
    view = pd.read_csv(view_path)
    entities = set(view["head"].astype(str)) | set(view["tail"].astype(str))
    before = len(frame)
    filtered = frame[
        frame["drug_name"].astype(str).isin(entities)
        & frame["target_name"].astype(str).isin(entities)
    ].copy()
    logger.info(
        "Pair multimodal cohort retained %d/%d pairs present in graph view %s.",
        len(filtered),
        before,
        view_name,
    )
    return filtered


def _build_pair_multimodal_frame(config, pair_cfg, logger):
    drugbank_table_path = Path(
        pair_cfg.get(
            "drugbank_table_path",
            Path(config["paths"]["intermediate_dir"])
            / "multimodal"
            / "drugbank_multimodal_benchmark.csv",
        )
    ).expanduser()
    if not drugbank_table_path.exists():
        raise FileNotFoundError(
            "Run prepare-multimodal before pair multimodal evaluation: %s"
            % drugbank_table_path
        )

    pair_table_cfg = {
        "label_group": pair_cfg.get("label_group", "moa"),
        "target_group": pair_cfg.get("target_group", "target"),
        "label_column": pair_cfg.get("label_column", "relation"),
        "drop_ambiguous_drug_actions": pair_cfg.get(
            "drop_ambiguous_drug_actions",
            True,
        ),
        "pair_table_path": pair_cfg.get("pair_table_path"),
        "pair_drug_column": pair_cfg.get("pair_drug_column", "drug_name"),
        "pair_target_column": pair_cfg.get("pair_target_column", "target_name"),
        "pair_action_column": pair_cfg.get("pair_action_column", "action"),
    }
    pair_frame = _load_pair_table(config, pair_table_cfg)
    pair_frame = _filter_to_view_entities(
        config,
        pair_frame,
        pair_cfg.get("require_in_view"),
        logger,
    )

    drug_table = pd.read_csv(drugbank_table_path)
    drug_table = drug_table[
        drug_table["smiles"].fillna("").astype(str).str.len().gt(0)
        & drug_table["target_count"].fillna(0).astype(int).gt(0)
    ].copy()
    drug_table["drug_name"] = drug_table["drug_name"].astype(str)
    drug_by_name = {
        row["drug_name"]: row
        for row in drug_table.to_dict("records")
    }

    xml_path = Path(pair_cfg.get("drugbank_xml", config.get("multimodal", {}).get("drugbank_xml", ""))).expanduser()
    if not xml_path.exists():
        raise FileNotFoundError("DrugBank XML not found: %s" % xml_path)
    target_index = _drugbank_target_index(
        xml_path,
        drug_table["drugbank_id"].dropna().astype(str).unique().tolist(),
        logger,
    )

    rows = []
    missing_drug = 0
    missing_target = 0
    for pair in pair_frame.itertuples(index=False):
        drug_name = str(pair.drug_name)
        target_name = str(pair.target_name)
        drug_record = drug_by_name.get(drug_name)
        if drug_record is None:
            missing_drug += 1
            continue
        target_record = None
        for key in (_symbol_key(target_name), _relaxed_symbol_key(target_name)):
            if not key:
                continue
            target_record = target_index.get((str(drug_record["drugbank_id"]), key))
            if target_record is not None:
                break
        if target_record is None:
            missing_target += 1
            continue
        rows.append({
            "drug_name": drug_name,
            "target_name": target_name,
            "action": str(pair.action),
            "drugbank_id": drug_record["drugbank_id"],
            "drugbank_name": drug_record["drugbank_name"],
            "drug_type": drug_record.get("drug_type", ""),
            "match_type": drug_record.get("match_type", ""),
            "matched_alias": drug_record.get("matched_alias", ""),
            "smiles": drug_record["smiles"],
            "target_uniprot_id": target_record["target_uniprot_id"],
            "target_gene_name": target_record["target_gene_name"],
            "target_drugbank_name": target_record["target_drugbank_name"],
            "target_sequence": target_record["target_sequence"],
        })
    frame = pd.DataFrame(rows).drop_duplicates()
    if not frame.empty:
        frame = frame.sort_values(["drug_name", "target_name", "action"]).reset_index(drop=True)
    logger.info(
        "Pair multimodal matched cohort has %d pairs from %d drugs and %d targets; missing DrugBank drug=%d, missing target sequence match=%d.",
        len(frame),
        frame["drug_name"].nunique() if not frame.empty else 0,
        frame["target_name"].nunique() if not frame.empty else 0,
        missing_drug,
        missing_target,
    )
    return frame


def _drug_text_from_pair(row):
    parts = [
        row.get("drug_name", ""),
        row.get("drugbank_name", ""),
        row.get("drugbank_id", ""),
        row.get("drug_type", ""),
    ]
    return " ".join(str(part) for part in parts if str(part).strip())


def _target_text_from_pair(row):
    parts = [
        row.get("target_name", ""),
        row.get("target_gene_name", ""),
        row.get("target_uniprot_id", ""),
        row.get("target_drugbank_name", ""),
    ]
    return " ".join(str(part) for part in parts if str(part).strip())


def _prepare_pair_arrays(frame, config, logger):
    import numpy as np

    pair_cfg = config["multimodal"]
    max_smiles_length = int(pair_cfg.get("max_smiles_length", 256))
    max_protein_length = int(pair_cfg.get("max_protein_length", 1200))
    max_atoms = int(pair_cfg.get("max_atoms_per_drug", 160))
    max_text_length = int(pair_cfg.get("max_text_length", 256))
    max_target_text_length = int(pair_cfg.get("max_target_text_length", 192))

    smiles_vocabulary = _vocabulary(frame["smiles"].astype(str).tolist())
    protein_vocabulary = _vocabulary([], fixed_alphabet=PROTEIN_ALPHABET)
    records = frame.to_dict("records")
    text_values = []
    for row in records:
        text_values.append(_drug_text_from_pair(row))
        text_values.append(_target_text_from_pair(row))
    text_vocabulary = _vocabulary(text_values)

    graph_requested = "GraphDTAStyle" in pair_cfg.get(
        "models",
        ["DeepDTAStyle", "GraphDTAStyle", "DruGNNosisStyle"],
    )
    rdkit_available = True
    if graph_requested:
        try:
            import rdkit  # noqa: F401
        except ImportError:
            rdkit_available = False
            logger.warning(
                "RDKit is unavailable; GraphDTAStyle will be skipped."
            )

    graph_cache = {}
    skipped_graphs = []
    rows = []
    for row in records:
        smiles = str(row.get("smiles", ""))
        graph = None
        if rdkit_available:
            if smiles not in graph_cache:
                graph_cache[smiles] = _molecular_graph(smiles, max_atoms=max_atoms)
            graph = graph_cache[smiles]
            if graph is None:
                skipped_graphs.append(str(row["drug_name"]))

        target_tokens = np.zeros((1, max_protein_length), dtype=np.int64)
        target_tokens[0] = _encode_text(
            row["target_sequence"],
            protein_vocabulary,
            max_protein_length,
        )
        target_mask = np.ones(1, dtype=bool)
        target_text_tokens = np.zeros((1, max_target_text_length), dtype=np.int64)
        target_text_tokens[0] = _encode_text(
            _target_text_from_pair(row),
            text_vocabulary,
            max_target_text_length,
        )
        target_text_mask = np.ones(1, dtype=bool)

        metadata = dict(row)
        metadata["target_sequences_json"] = json.dumps([row["target_sequence"]])
        metadata["target_ids_json"] = json.dumps([row.get("target_uniprot_id", "")])
        metadata["target_names_json"] = json.dumps([row.get("target_drugbank_name", "")])
        rows.append({
            "metadata": metadata,
            "smiles_tokens": _encode_text(
                row["smiles"],
                smiles_vocabulary,
                max_smiles_length,
            ),
            "target_tokens": target_tokens,
            "target_mask": target_mask,
            "drug_text_tokens": _encode_text(
                _drug_text_from_pair(row),
                text_vocabulary,
                max_text_length,
            ),
            "target_text_tokens": target_text_tokens,
            "target_text_mask": target_text_mask,
            "graph": graph,
        })

    logger.info(
        "Prepared %d pair-level multimodal samples; %d molecular graphs were unavailable or too large.",
        len(rows),
        len(skipped_graphs),
    )
    return {
        "rows": rows,
        "smiles_vocabulary": smiles_vocabulary,
        "protein_vocabulary": protein_vocabulary,
        "text_vocabulary": text_vocabulary,
        "rdkit_available": rdkit_available,
        "skipped_graphs": sorted(set(skipped_graphs)),
        "atom_feature_dim": 8,
        "max_atoms": max_atoms,
        "drugnnosis_layers": int(pair_cfg.get("drugnnosis_layers", 2)),
    }


def _inner_validation_split(outer_train, labels, validation_fraction, seed):
    import numpy as np
    from sklearn.model_selection import train_test_split

    outer_train = np.asarray(outer_train)
    if len(outer_train) < 4:
        return outer_train, outer_train
    class_count = len(np.unique(labels))
    validation_count = max(class_count, int(round(len(outer_train) * validation_fraction)))
    validation_count = min(validation_count, max(1, len(outer_train) - 1))
    try:
        return train_test_split(
            outer_train,
            test_size=validation_count,
            random_state=seed,
            stratify=np.asarray(labels)[outer_train],
        )
    except ValueError:
        return train_test_split(
            outer_train,
            test_size=validation_count,
            random_state=seed,
            stratify=None,
        )


def _evaluate_pair_multimodal_task(
    frame,
    task_name,
    task_cfg,
    cohort_name,
    config,
    logger,
):
    import copy
    import numpy as np
    import torch
    from sklearn.preprocessing import LabelEncoder
    from torch.utils.data import DataLoader, Subset

    task_frame = _filter_pairs_for_task(frame, task_cfg)
    if task_frame.empty:
        logger.warning("Pair multimodal task %s has no matched samples.", task_name)
        return [], [], [], {
            "task": task_name,
            "sample_count": 0,
            "class_counts": {},
        }
    task_frame = task_frame.sort_values(["drug_name", "target_name", "action"]).reset_index(drop=True)
    model_config = copy.deepcopy(config)
    model_config["multimodal"] = dict(config["pair_multimodal"])
    prepared = _prepare_pair_arrays(task_frame, model_config, logger)

    labels = np.asarray([row["metadata"]["action"] for row in prepared["rows"]], dtype=object)
    encoder = LabelEncoder()
    encoded = encoder.fit_transform(labels)
    drugs = [row["metadata"]["drug_name"] for row in prepared["rows"]]
    targets = [row["metadata"]["target_name"] for row in prepared["rows"]]
    all_indices = np.arange(len(encoded))

    pair_cfg = model_config["multimodal"]
    models = list(pair_cfg.get(
        "models",
        ["DeepDTAStyle", "GraphDTAStyle", "DruGNNosisStyle"],
    ))
    if not prepared["rdkit_available"] and "GraphDTAStyle" in models:
        models.remove("GraphDTAStyle")
    requested_folds = int(task_cfg.get("cv_folds", pair_cfg.get("cv_folds", 5)))
    repeats = int(task_cfg.get("cv_repeats", pair_cfg.get("cv_repeats", 3)))
    validation_fraction = float(pair_cfg.get("validation_fraction", 0.2))
    splits = [str(split) for split in pair_cfg.get("splits", ["random_pair"])]

    summary_rows = []
    repeat_rows = []
    prediction_rows = []
    for split_name in splits:
        folds = _fold_count(split_name, labels, drugs, targets, requested_folds)
        effective_folds = _effective_fold_count(split_name, folds)
        if folds < 2:
            logger.warning(
                "Pair multimodal task %s split %s cannot support at least two folds.",
                task_name,
                split_name,
            )
            continue
        for model_name in models:
            logger.info(
                "Pair multimodal baseline start: task=%s split=%s cohort=%s model=%s",
                task_name,
                split_name,
                cohort_name,
                model_name,
            )
            model_repeat_rows = []
            for repeat in range(repeats):
                seed = int(config.get("random_seed", 42) + repeat)
                oof_pred = np.full(len(encoded), -1, dtype=int)
                oof_score = np.full((len(encoded), len(encoder.classes_)), np.nan, dtype=float)
                fold_ids = np.full(len(encoded), -1, dtype=int)
                dummy_matrix = np.zeros((len(encoded), 1), dtype=np.float32)
                for fold, (outer_train, outer_test) in enumerate(
                    _iter_splits(
                        split_name,
                        folds,
                        seed,
                        dummy_matrix,
                        encoded,
                        drugs,
                        targets,
                    )
                ):
                    inner_train, inner_validation = _inner_validation_split(
                        outer_train,
                        encoded,
                        validation_fraction,
                        seed + fold,
                    )
                    model, device, dataset = _train_one_fold(
                        model_name=model_name,
                        prepared=prepared,
                        labels=encoded,
                        train_indices=np.asarray(inner_train),
                        validation_indices=np.asarray(inner_validation),
                        config=model_config,
                        seed=seed + fold,
                        logger=logger,
                    )
                    test_loader = DataLoader(
                        Subset(dataset, outer_test.tolist()),
                        batch_size=int(pair_cfg.get("batch_size", 16)),
                        shuffle=False,
                        num_workers=0,
                    )
                    absolute, predicted, probabilities = _predict(
                        model,
                        test_loader,
                        device,
                    )
                    oof_pred[absolute] = predicted
                    oof_score[absolute] = probabilities
                    fold_ids[absolute] = fold
                    del model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                if (oof_pred < 0).any():
                    raise RuntimeError(
                        "Out-of-fold prediction did not cover every pair multimodal sample"
                    )
                metrics = _metric_values(
                    encoded,
                    oof_pred,
                    oof_score,
                    len(encoder.classes_),
                )
                repeat_row = {
                    "task": task_name,
                    "cohort": cohort_name,
                    "split": split_name,
                    "view": "multimodal_baseline",
                    "model": model_name,
                    "classifier": model_name,
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
                model_repeat_rows.append(repeat_row)

                for sample_index, row in enumerate(prepared["rows"]):
                    metadata = row["metadata"]
                    pred_label = encoder.inverse_transform([oof_pred[sample_index]])[0]
                    prediction = {
                        "task": task_name,
                        "cohort": cohort_name,
                        "split": split_name,
                        "view": "multimodal_baseline",
                        "model": model_name,
                        "classifier": model_name,
                        "repeat": repeat,
                        "fold": int(fold_ids[sample_index]),
                        "drug_name": metadata["drug_name"],
                        "target_name": metadata["target_name"],
                        "true_label": str(labels[sample_index]),
                        "predicted_label": str(pred_label),
                    }
                    for class_index, class_name in enumerate(encoder.classes_):
                        prediction["score__%s" % class_name] = float(
                            oof_score[sample_index, class_index]
                        )
                    prediction_rows.append(prediction)

            summary = {
                "task": task_name,
                "cohort": cohort_name,
                "split": split_name,
                "view": "multimodal_baseline",
                "model": model_name,
                "classifier": model_name,
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
                    row[metric_name] for row in model_repeat_rows
                )
                summary["%s_mean" % metric_name] = mean
                summary["%s_std" % metric_name] = std
                summary["%s_ci95" % metric_name] = ci95
            summary["best_accuracy"] = summary["accuracy_mean"]
            summary["best_f1_macro"] = summary["f1_macro_mean"]
            summary["best_f1_weighted"] = summary["f1_weighted_mean"]
            summary["best_classifier_by_f1_macro"] = model_name
            summary["best_classifier"] = model_name
            summary_rows.append(summary)
            logger.info(
                "Pair multimodal baseline done: task=%s split=%s cohort=%s model=%s f1_macro=%.4f",
                task_name,
                split_name,
                cohort_name,
                model_name,
                float(summary["f1_macro_mean"]),
            )

    return summary_rows, repeat_rows, prediction_rows, {
        "task": task_name,
        "labels": encoder.classes_.tolist(),
        "sample_count": int(len(encoded)),
        "pair_count": int(len(encoded)),
        "drug_count": int(len(set(drugs))),
        "target_count": int(len(set(targets))),
        "class_counts": pd.Series(labels).value_counts().to_dict(),
        "skipped_graphs": prepared["skipped_graphs"],
    }


def evaluate_pair_multimodal_baselines(config, logger):
    """Run pair-level adapted DeepDTA/GraphDTA/DruGNNosis baselines."""

    import torch

    pair_cfg = dict(config.get("pair_multimodal", {}))
    if not pair_cfg:
        raise ValueError("Missing pair_multimodal config section")
    reports_dir = ensure_dir(Path(config["paths"]["reports_dir"]))
    cohort_name = str(pair_cfg.get("cohort_name", "drugbank_matched_pairs"))

    output_paths = [
        reports_dir / "pair_multimodal_baseline_summary.csv",
        reports_dir / "pair_multimodal_repeat_metrics.csv",
        reports_dir / "pair_multimodal_oof_predictions.csv",
        reports_dir / "pair_multimodal_manifest.json",
        reports_dir / "pair_multimodal_cohort.csv",
    ]
    if pair_cfg.get("reuse_completed_results", False) and all(
        path.exists() for path in output_paths
    ):
        logger.info("Reusing completed pair multimodal reports from %s", reports_dir)
        return pd.read_csv(output_paths[0])

    pair_frame = _build_pair_multimodal_frame(config, pair_cfg, logger)
    if pair_frame.empty:
        raise ValueError("No DrugBank-matched drug--target action pairs were found")
    pair_frame.to_csv(reports_dir / "pair_multimodal_cohort.csv", index=False)

    summary_rows = []
    repeat_rows = []
    prediction_rows = []
    task_manifests = []
    for task_name, task_cfg in _pair_label_tasks(pair_cfg).items():
        task_summary, task_repeats, task_predictions, task_manifest = (
            _evaluate_pair_multimodal_task(
                pair_frame,
                task_name,
                task_cfg,
                cohort_name,
                config,
                logger,
            )
        )
        summary_rows.extend(task_summary)
        repeat_rows.extend(task_repeats)
        prediction_rows.extend(task_predictions)
        task_manifests.append(task_manifest)

    summary_frame = pd.DataFrame(summary_rows)
    summary_frame.to_csv(
        reports_dir / "pair_multimodal_baseline_summary.csv",
        index=False,
    )
    pd.DataFrame(repeat_rows).to_csv(
        reports_dir / "pair_multimodal_repeat_metrics.csv",
        index=False,
    )
    pd.DataFrame(prediction_rows).to_csv(
        reports_dir / "pair_multimodal_oof_predictions.csv",
        index=False,
    )
    with (reports_dir / "pair_multimodal_manifest.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            {
                "cohort": cohort_name,
                "cohort_path": str(reports_dir / "pair_multimodal_cohort.csv"),
                "pair_count": int(len(pair_frame)),
                "drug_count": int(pair_frame["drug_name"].nunique()),
                "target_count": int(pair_frame["target_name"].nunique()),
                "task_manifests": task_manifests,
                "models": pair_cfg.get("models"),
                "splits": pair_cfg.get("splits"),
                "device": str(
                    pair_cfg.get(
                        "device",
                        "cuda" if torch.cuda.is_available() else "cpu",
                    )
                ),
                "protocol": (
                    "One row per DrugBank-matched drug-target pair; DrugBank "
                    "SMILES and target sequence only; DrugBank action fields "
                    "are not read or exported."
                ),
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )
    logger.info("Wrote pair multimodal baseline reports to %s", reports_dir)
    return summary_frame
