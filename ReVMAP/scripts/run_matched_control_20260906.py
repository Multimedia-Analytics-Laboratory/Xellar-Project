#!/usr/bin/env python3
"""Run one split of a fixed five-view drug-only routing control.

The original lock and predictions are read-only references. Execution shards
separate splits, never repeats, so RF random_state remains the original 42
while split seeds remain 42, 43, and 44.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def indices_digest(indices):
    return hashlib.sha256(
        json.dumps([int(x) for x in indices], separators=(",", ":")).encode()
    ).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--code-root", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=["development", "external"], required=True)
    parser.add_argument("--split", choices=["random_pair", "drug_disjoint", "target_disjoint", "double_disjoint"], required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    code_root = (args.code_root or project_root / "src").resolve()
    sys.path.insert(0, str(code_root))

    import numpy as np
    import pandas as pd
    import sklearn
    from sklearn.preprocessing import LabelEncoder

    from kgml_pipeline.config import load_config
    from kgml_pipeline.locked_multiview_evaluation import _stage_config
    from kgml_pipeline.pair_evaluation import (
        _build_pair_classifiers, _embedding_sources, _filter_pairs_for_task,
        _fold_count, _iter_splits, _load_pair_table, _pair_label_tasks,
    )
    from kgml_pipeline.relation_routing import (
        _build_aligned_matrices, _common_pair_frame, evaluate_relation_routing,
    )

    matched = json.loads(args.config.read_text())
    assert args.stage in matched["stages"] and args.split in matched["splits"]
    assert matched["feature_mode"] == "drug_only" and matched["strategy_selection"] is False
    protocol_path = project_root / matched["protocol_config"]
    assert sha256(protocol_path) == matched["expected_protocol_sha256"], "Original protocol changed"
    protocol_config = load_config(protocol_path)
    stage_config, base_config_path = _stage_config(protocol_config, args.stage)
    original_routing = copy.deepcopy(stage_config["relation_routing"])
    assert stage_config["random_seed"] == 42
    assert original_routing["cv_repeats"] == 3
    assert original_routing["cv_folds"] == 5
    assert original_routing["gate_inner_folds"] == 3
    assert original_routing["rf_estimators"] == 300
    assert len(original_routing["allowed_views"]) == 5
    assert original_routing["embedding_model"] == "RGCN"

    out_root = Path(matched["output_dir"])
    output = out_root / args.stage / args.split
    output.mkdir(parents=True, exist_ok=True)
    if (output / "run_completed.json").exists() or (output / "routing_summary.csv").exists():
        raise FileExistsError("Completed matched-control shard will not be overwritten")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("matched-control")
    protocol_dir = Path(protocol_config["locked_multiview_evaluation"]["output_dir"])
    lock = json.loads((protocol_dir / "strategy_lock.json").read_text())
    assert lock["lock_id"] == matched["expected_strategy_lock_id"]
    assert lock["locked_strategy"] == matched["strategy"]

    routing = copy.deepcopy(original_routing)
    routing.update({
        "feature_mode": "drug_only", "output_dir": str(output), "splits": [args.split],
        "include_drug_embedding": True, "include_target_embedding": False,
        "include_abs_difference": False, "include_hadamard_product": False,
    })
    stage_config["relation_routing"] = routing
    changed_fields = {
        "feature_mode", "output_dir", "splits", "include_drug_embedding",
        "include_target_embedding", "include_abs_difference", "include_hadamard_product",
    }
    assert {k:v for k,v in routing.items() if k not in changed_fields} == {
        k:v for k,v in original_routing.items() if k not in changed_fields
    }
    pair_cfg = dict(stage_config.get("pair_evaluation", {}))
    pair_cfg.update(routing)
    sources_by_view = {s["view"]:s for s in _embedding_sources(stage_config, {
        "allowed_views": routing["allowed_views"], "allowed_models": ["RGCN"],
    }, logger)}
    sources = [sources_by_view[v] for v in routing["allowed_views"]]
    tasks = _pair_label_tasks(pair_cfg)
    assert list(tasks) == ["core4_action"]
    frame = _filter_pairs_for_task(_load_pair_table(stage_config, pair_cfg), tasks["core4_action"])
    common = _common_pair_frame(frame, sources)
    pair_features_cfg = dict(
        pair_cfg, feature_mode="pair", include_drug_embedding=True,
        include_target_embedding=True, include_abs_difference=True,
        include_hadamard_product=True,
    )
    pair_matrices, labels, drugs, targets = _build_aligned_matrices(sources, common, pair_features_cfg)
    drug_matrices, drug_labels, drug_drugs, drug_targets = _build_aligned_matrices(sources, common, pair_cfg)
    assert (labels, drugs, targets) == (drug_labels, drug_drugs, drug_targets)
    assert len(common) == len(frame) == matched["expected_pair_counts"][args.stage]
    assert len(set(drugs)) == matched["expected_entity_counts"][args.stage]["drugs"]
    assert len(set(targets)) == matched["expected_entity_counts"][args.stage]["targets"]
    feature_checks = []
    for view in routing["allowed_views"]:
        pair_matrix, drug_matrix = pair_matrices[view], drug_matrices[view]
        assert pair_matrix.shape[1] == 4 * drug_matrix.shape[1]
        np.testing.assert_array_equal(drug_matrix, pair_matrix[:, :drug_matrix.shape[1]])
        for drug in set(drugs):
            positions = np.flatnonzero(np.asarray(drugs) == drug)
            np.testing.assert_array_equal(drug_matrix[positions], np.repeat(drug_matrix[positions[:1]], len(positions), axis=0))
        feature_checks.append({"view":view,"pair_width":pair_matrix.shape[1],"drug_only_width":drug_matrix.shape[1],"drug_vectors_equal_pair_first_block":True,"same_drug_target_invariance":True})

    reference_dir = Path(original_routing["output_dir"])
    if args.stage == "development" and args.split == "double_disjoint":
        reference_dir = reference_dir.parent / "development_double_disjoint"
    reference_path = reference_dir / "routing_oof_predictions.csv"
    reference = pd.read_csv(reference_path)
    reference = reference[(reference["strategy"] == matched["strategy"]) & (reference["split"] == args.split)]
    assert len(reference) == len(common) * 3
    identity_columns = ["drug_name", "target_name", "true_label"]
    identities = list(zip(drugs, targets, labels))
    encoded = LabelEncoder().fit_transform(labels)
    drug_array, target_array = np.asarray(drugs), np.asarray(targets)
    first_view = routing["allowed_views"][0]
    audit = []
    for repeat in range(3):
        seed = 42 + repeat
        expected = reference[reference["repeat"] == repeat].set_index(identity_columns)
        assert expected.index.is_unique and set(expected.index) == set(identities)
        assert set(expected["seed"]) == {seed}
        expected = expected.reindex(pd.MultiIndex.from_tuples(identities, names=identity_columns))
        folds = _fold_count(args.split, encoded, drugs, targets, 5)
        iterator = list(_iter_splits(args.split, folds, seed, drug_matrices[first_view], encoded, drugs, targets))
        pair_iterator = list(_iter_splits(args.split, folds, seed, pair_matrices[first_view], encoded, drugs, targets))
        coverage = np.zeros(len(common), int)
        for fold, ((train, test), (pair_train, pair_test)) in enumerate(zip(iterator, pair_iterator)):
            np.testing.assert_array_equal(train, pair_train)
            np.testing.assert_array_equal(test, pair_test)
            assert (expected.iloc[test]["fold"].to_numpy() == fold).all(), "Original outer fold differs"
            coverage[test] += 1
            def overlap(a, b, entities):
                return len(set(entities[a]) & set(entities[b]))
            if args.split in {"drug_disjoint", "double_disjoint"}:
                assert overlap(train, test, drug_array) == 0
            if args.split in {"target_disjoint", "double_disjoint"}:
                assert overlap(train, test, target_array) == 0
            inner_folds = _fold_count(args.split, encoded[train], drug_array[train], target_array[train], 3)
            inner = list(_iter_splits(args.split, inner_folds, seed + 1009 + fold, drug_matrices[first_view][train], encoded[train], drug_array[train], target_array[train]))
            pair_inner = list(_iter_splits(args.split, inner_folds, seed + 1009 + fold, pair_matrices[first_view][train], encoded[train], drug_array[train], target_array[train]))
            inner_coverage = np.zeros(len(train), int)
            inner_audit = []
            for inner_fold, ((it, iv), (pt, pv)) in enumerate(zip(inner, pair_inner)):
                np.testing.assert_array_equal(it, pt)
                np.testing.assert_array_equal(iv, pv)
                inner_coverage[iv] += 1
                if args.split in {"drug_disjoint", "double_disjoint"}:
                    assert overlap(train[it], train[iv], drug_array) == 0
                if args.split in {"target_disjoint", "double_disjoint"}:
                    assert overlap(train[it], train[iv], target_array) == 0
                inner_audit.append({"inner_fold":inner_fold,"train_global_indices_sha256":indices_digest(train[it]),"validation_global_indices_sha256":indices_digest(train[iv])})
            assert (inner_coverage == 1).all()
            audit.append({"repeat":repeat,"seed":seed,"fold":fold,"train_count":len(train),"test_count":len(test),"train_indices_sha256":indices_digest(train),"test_indices_sha256":indices_digest(test),"inner_seed":seed+1009+fold,"gate_seed":seed+fold,"inner_folds":inner_audit})
        assert (coverage == 1).all()

    embeddings = []
    embedding_root = Path(stage_config["paths"]["processed_dir"]) / "embeddings"
    for view in routing["allowed_views"]:
        for name in ("entity_embeddings.npy", "entity_to_id.json"):
            path = embedding_root / view / "RGCN" / name
            embeddings.append({"path":str(path),"sha256":sha256(path)})
    rf = _build_pair_classifiers(dict(routing, classifiers=["Random Forest"]), 42)["Random Forest"]
    preflight = {
        "stage":args.stage,"split":args.split,"pair_count":len(common),"drug_count":len(set(drugs)),"target_count":len(set(targets)),
        "action_counts":dict(pd.Series(labels).value_counts().items()),
        "ordered_pair_labels_sha256":hashlib.sha256(json.dumps(identities, separators=(",", ":")).encode()).hexdigest(),
        "feature_checks":feature_checks,"outer_folds_match_original_predictions":True,"inner_folds_match_pair_features":True,
        "reference_prediction_path":str(reference_path),"reference_prediction_sha256":sha256(reference_path),
        "original_protocol_sha256":sha256(protocol_path),"strategy_lock_id":lock["lock_id"],"original_lock_sha256":sha256(protocol_dir / "strategy_lock.json"),
        "base_config_path":str(base_config_path),"base_config_sha256":sha256(base_config_path),
        "pair_table_sha256":sha256(pair_cfg["pair_table_path"]),"embedding_inputs":embeddings,
        "rf_parameters":rf.get_params(),"routing_parameters":routing,
        "python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__,"scikit_learn":sklearn.__version__,
        "source_hashes":{str(p.relative_to(code_root)):sha256(p) for p in (code_root / "kgml_pipeline").glob("*.py")},
        "cpu_affinity":sorted(os.sched_getaffinity(0)) if hasattr(os,"sched_getaffinity") else None,
        "folds":audit,
    }
    save_json(output / "matched_preflight.json", preflight)
    save_json(output / "effective_config.json", stage_config)
    logger.info("Matched preflight passed: stage=%s split=%s rows=%d outer_folds=%d", args.stage, args.split, len(common), len(audit))
    if args.preflight_only:
        return
    del sources, sources_by_view, pair_matrices, drug_matrices, reference
    start = time.monotonic()
    evaluate_relation_routing(stage_config, logger)
    assert sha256(protocol_path) == preflight["original_protocol_sha256"]
    assert sha256(reference_path) == preflight["reference_prediction_sha256"]
    assert sha256(protocol_dir / "strategy_lock.json") == preflight["original_lock_sha256"]
    save_json(output / "run_completed.json", {"stage":args.stage,"split":args.split,"elapsed_training_seconds":time.monotonic()-start,"finished_utc":datetime.now(timezone.utc).isoformat(),"original_protocol_and_reference_unchanged":True})


if __name__ == "__main__":
    main()
