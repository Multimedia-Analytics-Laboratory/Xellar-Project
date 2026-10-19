"""Evaluate task-matched multimodal baselines on the matched drug cohort."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from pathlib import Path

import pandas as pd

from .io_utils import ensure_dir
from .multimodal_models import (
    DeepDTAStyleClassifier,
    DruGNNosisStyleClassifier,
    GraphDTAStyleClassifier,
)
from .strict_evaluation import _metric_values, _summary_stats


PROTEIN_ALPHABET = "ACDEFGHIKLMNPQRSTVWYBXZJUO"
DRUGNNOSIS_MODEL_NAMES = {"DruGNNosisStyle", "DrugNNosisStyle"}


def _vocabulary(values, fixed_alphabet=None):
    characters = set(fixed_alphabet or "")
    for value in values:
        characters.update(str(value))
    mapping = {"<PAD>": 0, "<UNK>": 1}
    for character in sorted(characters):
        if character not in mapping:
            mapping[character] = len(mapping)
    return mapping


def _encode_text(value, vocabulary, max_length):
    import numpy as np

    output = np.zeros(max_length, dtype=np.int64)
    for index, character in enumerate(str(value)[:max_length]):
        output[index] = vocabulary.get(character, 1)
    return output


def _decode_targets(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    parsed = json.loads(str(value))
    return [str(sequence) for sequence in parsed if str(sequence)]


def _decode_json_strings(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    parsed = json.loads(str(value))
    return [str(item) for item in parsed if str(item)]


def _drug_text(row):
    parts = [
        row.get("drug_name", ""),
        row.get("drugbank_name", ""),
        row.get("drugbank_id", ""),
        row.get("drug_type", ""),
    ]
    return " ".join(str(part) for part in parts if str(part).strip())


def _target_texts(row, max_targets):
    target_ids = _decode_json_strings(row.get("target_ids_json"))[:max_targets]
    target_names = _decode_json_strings(row.get("target_names_json"))[:max_targets]
    count = max(len(target_ids), len(target_names))
    texts = []
    for index in range(count):
        parts = []
        if index < len(target_ids):
            parts.append(target_ids[index])
        if index < len(target_names):
            parts.append(target_names[index])
        texts.append(" ".join(part for part in parts if part.strip()))
    return texts


def _molecular_graph(smiles, max_atoms):
    import numpy as np

    from rdkit import Chem

    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None or molecule.GetNumAtoms() == 0:
        return None
    if molecule.GetNumAtoms() > max_atoms:
        return None

    feature_count = 8
    atoms = np.zeros((max_atoms, feature_count), dtype=np.float32)
    adjacency = np.zeros((max_atoms, max_atoms), dtype=np.float32)
    atom_mask = np.zeros(max_atoms, dtype=np.float32)

    for atom in molecule.GetAtoms():
        index = atom.GetIdx()
        atoms[index] = [
            atom.GetAtomicNum() / 118.0,
            min(atom.GetDegree(), 8) / 8.0,
            max(-5, min(5, atom.GetFormalCharge())) / 5.0,
            float(atom.GetIsAromatic()),
            min(atom.GetMass(), 250.0) / 250.0,
            min(atom.GetTotalNumHs(), 8) / 8.0,
            min(int(atom.GetHybridization()), 8) / 8.0,
            float(atom.IsInRing()),
        ]
        atom_mask[index] = 1.0
        adjacency[index, index] = 1.0

    for bond in molecule.GetBonds():
        left = bond.GetBeginAtomIdx()
        right = bond.GetEndAtomIdx()
        adjacency[left, right] = 1.0
        adjacency[right, left] = 1.0

    degree = adjacency.sum(axis=1)
    inverse_sqrt = np.zeros_like(degree)
    nonzero = degree > 0
    inverse_sqrt[nonzero] = np.power(degree[nonzero], -0.5)
    normalized = inverse_sqrt[:, None] * adjacency * inverse_sqrt[None, :]
    return atoms, normalized.astype(np.float32), atom_mask


def _prepare_arrays(frame, config, logger):
    import numpy as np

    multimodal_cfg = config["multimodal"]
    max_smiles_length = int(multimodal_cfg.get("max_smiles_length", 256))
    max_protein_length = int(multimodal_cfg.get("max_protein_length", 1200))
    max_targets = int(multimodal_cfg.get("max_targets_per_drug", 16))
    max_atoms = int(multimodal_cfg.get("max_atoms_per_drug", 160))
    max_text_length = int(multimodal_cfg.get("max_text_length", 256))
    max_target_text_length = int(
        multimodal_cfg.get("max_target_text_length", 192)
    )

    smiles_vocabulary = _vocabulary(frame["smiles"].astype(str).tolist())
    protein_vocabulary = _vocabulary([], fixed_alphabet=PROTEIN_ALPHABET)
    records = frame.to_dict("records")
    text_values = []
    for row in records:
        text_values.append(_drug_text(row))
        text_values.extend(_target_texts(row, max_targets=max_targets))
    text_vocabulary = _vocabulary(text_values)

    rows = []
    skipped_graphs = []
    graph_requested = "GraphDTAStyle" in multimodal_cfg.get(
        "models", ["DeepDTAStyle", "GraphDTAStyle"]
    )
    rdkit_available = True
    if graph_requested:
        try:
            import rdkit  # noqa: F401
        except ImportError:
            rdkit_available = False
            logger.warning(
                "RDKit is unavailable; GraphDTAStyle will be skipped. "
                "Install the optional 'rdkit' dependency to enable it."
            )

    for row in records:
        targets = _decode_targets(row["target_sequences_json"])[:max_targets]
        if not str(row.get("smiles", "")).strip() or not targets:
            continue

        graph = None
        if rdkit_available:
            graph = _molecular_graph(row["smiles"], max_atoms=max_atoms)
            if graph is None:
                skipped_graphs.append(str(row["drug_name"]))

        target_tokens = np.zeros(
            (max_targets, max_protein_length),
            dtype=np.int64,
        )
        target_mask = np.zeros(max_targets, dtype=bool)
        for target_index, sequence in enumerate(targets):
            target_tokens[target_index] = _encode_text(
                sequence,
                protein_vocabulary,
                max_protein_length,
            )
            target_mask[target_index] = True

        target_text_tokens = np.zeros(
            (max_targets, max_target_text_length),
            dtype=np.int64,
        )
        target_text_mask = np.zeros(max_targets, dtype=bool)
        target_texts = _target_texts(row, max_targets=max_targets)
        for target_index in range(min(len(targets), max_targets)):
            text = (
                target_texts[target_index]
                if target_index < len(target_texts)
                else "target_%d" % target_index
            )
            target_text_tokens[target_index] = _encode_text(
                text,
                text_vocabulary,
                max_target_text_length,
            )
            target_text_mask[target_index] = True

        rows.append({
            "metadata": row,
            "smiles_tokens": _encode_text(
                row["smiles"],
                smiles_vocabulary,
                max_smiles_length,
            ),
            "target_tokens": target_tokens,
            "target_mask": target_mask,
            "drug_text_tokens": _encode_text(
                _drug_text(row),
                text_vocabulary,
                max_text_length,
            ),
            "target_text_tokens": target_text_tokens,
            "target_text_mask": target_text_mask,
            "graph": graph,
        })

    logger.info(
        "Prepared %d multimodal drugs; %d molecular graphs were unavailable or too large.",
        len(rows),
        len(skipped_graphs),
    )
    return {
        "rows": rows,
        "smiles_vocabulary": smiles_vocabulary,
        "protein_vocabulary": protein_vocabulary,
        "text_vocabulary": text_vocabulary,
        "rdkit_available": rdkit_available,
        "skipped_graphs": skipped_graphs,
        "atom_feature_dim": 8,
        "max_atoms": max_atoms,
        "drugnnosis_layers": int(multimodal_cfg.get("drugnnosis_layers", 2)),
    }


def _array_dataset(prepared, labels):
    import numpy as np
    import torch
    from torch.utils.data import Dataset

    class _Dataset(Dataset):
        def __init__(self):
            self.rows = prepared["rows"]
            self.labels = np.asarray(labels, dtype=np.int64)
            max_atoms = prepared["max_atoms"]
            atom_feature_dim = prepared["atom_feature_dim"]
            self.empty_atoms = np.zeros((max_atoms, atom_feature_dim), dtype=np.float32)
            self.empty_adjacency = np.eye(max_atoms, dtype=np.float32)
            self.empty_atom_mask = np.zeros(max_atoms, dtype=np.float32)

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            row = self.rows[index]
            graph = row["graph"]
            if graph is None:
                graph = (
                    self.empty_atoms,
                    self.empty_adjacency,
                    self.empty_atom_mask,
                )
            return {
                "smiles_tokens": torch.as_tensor(row["smiles_tokens"], dtype=torch.long),
                "target_tokens": torch.as_tensor(row["target_tokens"], dtype=torch.long),
                "target_mask": torch.as_tensor(row["target_mask"], dtype=torch.bool),
                "drug_text_tokens": torch.as_tensor(
                    row["drug_text_tokens"],
                    dtype=torch.long,
                ),
                "target_text_tokens": torch.as_tensor(
                    row["target_text_tokens"],
                    dtype=torch.long,
                ),
                "target_text_mask": torch.as_tensor(
                    row["target_text_mask"],
                    dtype=torch.bool,
                ),
                "atom_features": torch.as_tensor(graph[0], dtype=torch.float32),
                "normalized_adjacency": torch.as_tensor(graph[1], dtype=torch.float32),
                "atom_mask": torch.as_tensor(graph[2], dtype=torch.float32),
                "label": torch.tensor(self.labels[index], dtype=torch.long),
                "index": torch.tensor(index, dtype=torch.long),
            }

    return _Dataset()


def _make_model(model_name, prepared, class_count, hidden_dim):
    if model_name == "DeepDTAStyle":
        return DeepDTAStyleClassifier(
            smiles_vocab_size=len(prepared["smiles_vocabulary"]),
            protein_vocab_size=len(prepared["protein_vocabulary"]),
            class_count=class_count,
            hidden_dim=hidden_dim,
        )
    if model_name == "GraphDTAStyle":
        return GraphDTAStyleClassifier(
            atom_feature_dim=prepared["atom_feature_dim"],
            protein_vocab_size=len(prepared["protein_vocabulary"]),
            class_count=class_count,
            hidden_dim=hidden_dim,
        )
    if model_name in DRUGNNOSIS_MODEL_NAMES:
        return DruGNNosisStyleClassifier(
            text_vocab_size=len(prepared["text_vocabulary"]),
            class_count=class_count,
            hidden_dim=hidden_dim,
            layer_count=int(prepared.get("drugnnosis_layers", 2)),
        )
    raise ValueError("Unsupported multimodal model: %s" % model_name)


def _move_batch(batch, device):
    return {
        key: value.to(device)
        for key, value in batch.items()
        if key != "index"
    }


def _forward(model, batch):
    return model(
        smiles_tokens=batch["smiles_tokens"],
        target_tokens=batch["target_tokens"],
        target_mask=batch["target_mask"],
        atom_features=batch["atom_features"],
        normalized_adjacency=batch["normalized_adjacency"],
        atom_mask=batch["atom_mask"],
        drug_text_tokens=batch["drug_text_tokens"],
        target_text_tokens=batch["target_text_tokens"],
        target_text_mask=batch["target_text_mask"],
    )


def _predict(model, loader, device):
    import numpy as np
    import torch

    model.eval()
    indices = []
    predictions = []
    probabilities = []
    with torch.no_grad():
        for raw_batch in loader:
            batch = _move_batch(raw_batch, device)
            logits = _forward(model, batch)
            probability = torch.softmax(logits, dim=-1)
            indices.extend(raw_batch["index"].cpu().numpy().tolist())
            predictions.extend(torch.argmax(probability, dim=-1).cpu().numpy().tolist())
            probabilities.extend(probability.cpu().numpy().tolist())
    order = np.argsort(indices)
    return (
        np.asarray(indices)[order],
        np.asarray(predictions)[order],
        np.asarray(probabilities)[order],
    )


def _train_one_fold(
    *,
    model_name,
    prepared,
    labels,
    train_indices,
    validation_indices,
    config,
    seed,
    logger,
):
    import numpy as np
    import torch
    from sklearn.metrics import f1_score
    from torch.utils.data import DataLoader, Subset

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    multimodal_cfg = config["multimodal"]
    device = torch.device(
        multimodal_cfg.get(
            "device",
            "cuda" if torch.cuda.is_available() else "cpu",
        )
    )
    dataset = _array_dataset(prepared, labels)
    batch_size = int(multimodal_cfg.get("batch_size", 16))
    train_loader = DataLoader(
        Subset(dataset, train_indices.tolist()),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    validation_loader = DataLoader(
        Subset(dataset, validation_indices.tolist()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = _make_model(
        model_name,
        prepared,
        class_count=int(len(np.unique(labels))),
        hidden_dim=int(multimodal_cfg.get("hidden_dim", 128)),
    ).to(device)

    train_counts = np.bincount(
        np.asarray(labels)[train_indices],
        minlength=int(len(np.unique(labels))),
    )
    class_weights = len(train_indices) / (
        len(train_counts) * np.maximum(train_counts, 1)
    )
    criterion = torch.nn.CrossEntropyLoss(
        weight=torch.as_tensor(class_weights, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(multimodal_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(multimodal_cfg.get("weight_decay", 1e-4)),
    )

    best_state = None
    best_score = -math.inf
    patience = int(multimodal_cfg.get("patience", 8))
    stale_epochs = 0
    max_epochs = int(multimodal_cfg.get("epochs", 60))

    for _ in range(max_epochs):
        model.train()
        for raw_batch in train_loader:
            batch = _move_batch(raw_batch, device)
            optimizer.zero_grad()
            logits = _forward(model, batch)
            loss = criterion(logits, batch["label"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        validation_absolute, validation_pred, _ = _predict(
            model,
            validation_loader,
            device,
        )
        score = f1_score(
            np.asarray(labels)[validation_absolute],
            validation_pred,
            average="macro",
            zero_division=0,
        )
        if score > best_score + 1e-6:
            best_score = float(score)
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is None:
        raise RuntimeError("No model checkpoint was produced")
    model.load_state_dict(best_state)
    logger.info(
        "Finished %s fold training with best validation macro-F1 %.4f",
        model_name,
        best_score,
    )
    return model, device, dataset


def _evaluate_task(
    frame,
    prepared,
    task_name,
    task_cfg,
    cohort_name,
    config,
    logger,
):
    import numpy as np
    from sklearn.model_selection import StratifiedKFold, train_test_split
    from sklearn.preprocessing import LabelEncoder
    from torch.utils.data import DataLoader, Subset

    label_column = (
        "superclass_4"
        if task_cfg["label_column"] == "relation"
        else "detailed_moa"
    )
    labels = np.asarray(
        [row["metadata"][label_column] for row in prepared["rows"]],
        dtype=object,
    )
    allowed_classes = task_cfg.get("allowed_classes")
    if allowed_classes:
        allowed_classes = {str(label) for label in allowed_classes}
        allowed_keep = np.asarray(
            [str(label) in allowed_classes for label in labels],
            dtype=bool,
        )
        prepared = dict(prepared)
        prepared["rows"] = [
            row for row, keep_value in zip(prepared["rows"], allowed_keep.tolist())
            if keep_value
        ]
        labels = np.asarray(
            [row["metadata"][label_column] for row in prepared["rows"]],
            dtype=object,
        )
    counts = pd.Series(labels).value_counts()
    valid_labels = counts[
        counts >= int(task_cfg.get("min_class_count", 5))
    ].index.tolist()
    keep = np.asarray([label in valid_labels for label in labels], dtype=bool)

    filtered_rows = [
        row for row, keep_value in zip(prepared["rows"], keep.tolist())
        if keep_value
    ]
    filtered_rows = sorted(
        filtered_rows,
        key=lambda row: str(row["metadata"]["drug_name"]),
    )
    task_prepared = dict(prepared)
    task_prepared["rows"] = filtered_rows
    labels = np.asarray(
        [row["metadata"][label_column] for row in filtered_rows],
        dtype=object,
    )
    encoder = LabelEncoder()
    encoded = encoder.fit_transform(labels)
    drug_names = [
        row["metadata"]["drug_name"] for row in filtered_rows
    ]

    multimodal_cfg = config["multimodal"]
    models = list(multimodal_cfg.get(
        "models",
        ["DeepDTAStyle", "GraphDTAStyle"],
    ))
    if not task_prepared["rdkit_available"] and "GraphDTAStyle" in models:
        models.remove("GraphDTAStyle")

    requested_folds = int(multimodal_cfg.get("cv_folds", 5))
    folds = min(requested_folds, int(pd.Series(encoded).value_counts().min()))
    repeats = int(multimodal_cfg.get("cv_repeats", 3))
    validation_fraction = float(multimodal_cfg.get("validation_fraction", 0.2))
    all_indices = np.arange(len(encoded))

    summary_rows = []
    repeat_rows = []
    prediction_rows = []

    for model_name in models:
        model_repeat_rows = []
        for repeat in range(repeats):
            seed = int(config["random_seed"] + repeat)
            splitter = StratifiedKFold(
                n_splits=folds,
                shuffle=True,
                random_state=seed,
            )
            oof_pred = np.full(len(encoded), -1, dtype=int)
            oof_score = np.full(
                (len(encoded), len(encoder.classes_)),
                np.nan,
                dtype=float,
            )
            fold_ids = np.full(len(encoded), -1, dtype=int)

            for fold, (outer_train, outer_test) in enumerate(
                splitter.split(all_indices, encoded)
            ):
                class_count = len(encoder.classes_)
                validation_count = max(
                    class_count,
                    int(round(len(outer_train) * validation_fraction)),
                )
                validation_count = min(
                    validation_count,
                    len(outer_train) - class_count,
                )
                inner_train, inner_validation = train_test_split(
                    outer_train,
                    test_size=validation_count,
                    random_state=seed + fold,
                    stratify=encoded[outer_train],
                )
                model, device, dataset = _train_one_fold(
                    model_name=model_name,
                    prepared=task_prepared,
                    labels=encoded,
                    train_indices=np.asarray(inner_train),
                    validation_indices=np.asarray(inner_validation),
                    config=config,
                    seed=seed + fold,
                    logger=logger,
                )
                test_loader = DataLoader(
                    Subset(dataset, outer_test.tolist()),
                    batch_size=int(multimodal_cfg.get("batch_size", 16)),
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

            metrics = _metric_values(
                encoded,
                oof_pred,
                oof_score,
                len(encoder.classes_),
            )
            repeat_row = {
                "task": task_name,
                "cohort": cohort_name,
                "model": model_name,
                "repeat": repeat,
                "seed": seed,
                "sample_count": int(len(encoded)),
                "class_count": int(len(encoder.classes_)),
                "cv_folds": int(folds),
                **metrics,
            }
            repeat_rows.append(repeat_row)
            model_repeat_rows.append(repeat_row)

            for index, drug_name in enumerate(drug_names):
                row = {
                    "task": task_name,
                    "cohort": cohort_name,
                    "model": model_name,
                    "repeat": repeat,
                    "fold": int(fold_ids[index]),
                    "drug_name": drug_name,
                    "true_label": str(labels[index]),
                    "predicted_label": str(
                        encoder.inverse_transform([oof_pred[index]])[0]
                    ),
                }
                for class_index, class_name in enumerate(encoder.classes_):
                    row["probability__%s" % class_name] = float(
                        oof_score[index, class_index]
                    )
                prediction_rows.append(row)

        summary = {
            "task": task_name,
            "cohort": cohort_name,
            "view": "multimodal_baseline",
            "model": model_name,
            "classifier": model_name,
            "sample_count": int(len(encoded)),
            "moa_count": int(len(encoder.classes_)),
            "cv_folds": int(folds),
            "cv_repeats": int(repeats),
            "status": "success",
        }
        for metric_name in (
            "accuracy",
            "balanced_accuracy",
            "f1_macro",
            "f1_weighted",
            "precision_macro",
            "recall_macro",
            "auc_ovr_macro",
        ):
            mean, std, ci95 = _summary_stats(
                row[metric_name] for row in model_repeat_rows
            )
            summary["%s_mean" % metric_name] = mean
            summary["%s_std" % metric_name] = std
            summary["%s_ci95" % metric_name] = ci95
        summary_rows.append(summary)

    return summary_rows, repeat_rows, prediction_rows, {
        "task": task_name,
        "labels": encoder.classes_.tolist(),
        "sample_count": int(len(encoded)),
        "class_counts": pd.Series(labels).value_counts().to_dict(),
    }


def evaluate_multimodal_baselines(config, logger):
    """Run group-safe drug-level multimodal baselines."""

    import torch

    reports_dir = ensure_dir(Path(config["paths"]["reports_dir"]))
    table_path = (
        Path(config["paths"]["intermediate_dir"])
        / "multimodal"
        / "drugbank_multimodal_benchmark.csv"
    )
    if not table_path.exists():
        raise FileNotFoundError(
            "Run prepare-multimodal before evaluate-multimodal: %s"
            % table_path
        )
    signature_payload = {
        "random_seed": config["random_seed"],
        "label_tasks": config["evaluation"]["label_tasks"],
        "multimodal": config["multimodal"],
        "input_size": table_path.stat().st_size,
        "input_modified_ns": table_path.stat().st_mtime_ns,
    }
    signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    completion_path = reports_dir / "multimodal_completion.json"
    output_paths = [
        reports_dir / "multimodal_baseline_summary.csv",
        reports_dir / "multimodal_repeat_metrics.csv",
        reports_dir / "multimodal_oof_predictions.csv",
        reports_dir / "multimodal_manifest.json",
    ]
    if (
        config["multimodal"].get("reuse_completed_results", False)
        and completion_path.exists()
        and all(path.exists() for path in output_paths)
    ):
        with completion_path.open("r", encoding="utf-8") as handle:
            completion = json.load(handle)
        if completion.get("config_signature") == signature:
            logger.info(
                "Reusing completed multimodal results with signature %s.",
                signature,
            )
            return pd.read_csv(output_paths[0])

    frame = pd.read_csv(table_path)
    frame = frame[
        frame["smiles"].fillna("").astype(str).str.len().gt(0)
        & frame["target_count"].gt(0)
    ].copy()
    cohort_name = str(
        config["multimodal"].get("cohort_name", "drugbank_smiles_targets")
    )
    require_in_view = config["multimodal"].get("require_in_view")
    if require_in_view:
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
        frame = frame[frame["drug_name"].astype(str).isin(view_entities)].copy()
        logger.info(
            "Matched cohort %s retains %d drugs present in graph view %s.",
            cohort_name,
            len(frame),
            require_in_view,
        )
    prepared = _prepare_arrays(frame, config, logger)

    summary_rows = []
    repeat_rows = []
    prediction_rows = []
    task_manifests = []
    selected_tasks = config["multimodal"].get("label_tasks")
    label_tasks = config["evaluation"]["label_tasks"]
    if selected_tasks:
        label_tasks = {
            task_name: task_cfg
            for task_name, task_cfg in label_tasks.items()
            if task_name in set(selected_tasks)
        }
    for task_name, task_cfg in label_tasks.items():
        task_summary, task_repeats, task_predictions, task_manifest = _evaluate_task(
            frame,
            prepared,
            task_name,
            task_cfg,
            cohort_name,
            config,
            logger,
        )
        summary_rows.extend(task_summary)
        repeat_rows.extend(task_repeats)
        prediction_rows.extend(task_predictions)
        task_manifests.append(task_manifest)

    summary_frame = pd.DataFrame(summary_rows)
    summary_frame.to_csv(
        reports_dir / "multimodal_baseline_summary.csv",
        index=False,
    )
    pd.DataFrame(repeat_rows).to_csv(
        reports_dir / "multimodal_repeat_metrics.csv",
        index=False,
    )
    pd.DataFrame(prediction_rows).to_csv(
        reports_dir / "multimodal_oof_predictions.csv",
        index=False,
    )
    with (reports_dir / "multimodal_manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(
            {
                "input_table": str(table_path),
                "device": str(
                    config["multimodal"].get(
                        "device",
                        "cuda" if torch.cuda.is_available() else "cpu",
                    )
                ),
                "task_manifests": task_manifests,
                "skipped_graphs": prepared["skipped_graphs"],
                "models": config["multimodal"].get("models"),
                "protocol": (
                    "One row per drug; outer stratified folds; inner validation "
                    "for early stopping; no DrugBank action fields."
                ),
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )
    with completion_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "config_signature": signature,
                "summary_rows": int(len(summary_frame)),
                "repeat_rows": int(len(repeat_rows)),
                "prediction_rows": int(len(prediction_rows)),
            },
            handle,
            indent=2,
        )
    logger.info("Wrote multimodal baseline reports to %s", reports_dir)
    return summary_frame
