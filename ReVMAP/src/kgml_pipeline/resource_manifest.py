"""Build machine-readable provenance and resource manifests for the benchmark."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .io_utils import ensure_dir


def _sha256(path, maximum_bytes=None):
    path = Path(path)
    if maximum_bytes is not None and path.stat().st_size > maximum_bytes:
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path, hash_limit_bytes=256 * 1024 * 1024):
    resolved = Path(str(path)).expanduser()
    record = {
        "path": str(resolved),
        "exists": resolved.exists(),
    }
    if not resolved.exists() or not resolved.is_file():
        return record
    stat = resolved.stat()
    record.update({
        "size_bytes": int(stat.st_size),
        "modified_utc": datetime.fromtimestamp(
            stat.st_mtime,
            tz=timezone.utc,
        ).isoformat(),
        "sha256": _sha256(resolved, maximum_bytes=hash_limit_bytes),
    })
    if record["sha256"] is None:
        record["sha256_note"] = (
            "Omitted because the source exceeds the 256 MiB manifest hash limit."
        )
    return record


def _triple_record(path, benchmark_drugs):
    path = Path(path)
    frame = pd.read_csv(path)[["head", "relation", "tail"]].astype(str)
    entities = pd.concat([frame["head"], frame["tail"]], ignore_index=True)
    covered = set(frame["head"]).intersection(benchmark_drugs)
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "triple_count": int(len(frame)),
        "entity_count": int(entities.nunique()),
        "relation_count": int(frame["relation"].nunique()),
        "head_count": int(frame["head"].nunique()),
        "tail_count": int(frame["tail"].nunique()),
        "benchmark_drug_count": int(len(covered)),
        "benchmark_drug_coverage": (
            float(len(covered) / len(benchmark_drugs)) if benchmark_drugs else 0.0
        ),
        "relation_counts": {
            str(key): int(value)
            for key, value in frame["relation"].value_counts().sort_index().items()
        },
    }


def build_resource_manifest(config, logger):
    """Describe source files, labels, graph views, and evaluation protocol."""

    triple_root = Path(config["paths"]["intermediate_dir"]) / "triples"
    group_root = triple_root / "groups"
    view_root = triple_root / "views"
    moa_path = group_root / (
        "%s.csv" % config["evaluation"].get("label_group", "moa")
    )
    moa = pd.read_csv(moa_path)[["head", "relation", "tail"]].astype(str)
    benchmark_drugs = set(moa["head"])

    input_keys = (
        "combined_metadata_csv",
        "repurposing_hub_tsv",
        "primekg_csv",
        "prebuilt_benchmark_csv",
        "prebuilt_expanded_targets_csv",
    )
    inputs = {}
    for key in input_keys:
        value = config["paths"].get(key)
        if value:
            inputs[key] = _file_record(value)
    inputs["open_targets_drug_files"] = [
        _file_record(path)
        for path in config["paths"].get("open_targets_drug_files", [])
    ]
    multimodal_xml = config.get("multimodal", {}).get("drugbank_xml")
    if multimodal_xml:
        inputs["drugbank_xml"] = _file_record(multimodal_xml)

    groups = {
        path.stem: _triple_record(path, benchmark_drugs)
        for path in sorted(group_root.glob("*.csv"))
    }
    views = {
        path.stem: {
            **_triple_record(path, benchmark_drugs),
            "groups": list(config.get("views", {}).get(path.stem, [])),
            "contains_label_group": bool(
                config["evaluation"].get("label_group", "moa")
                in config.get("views", {}).get(path.stem, [])
            ),
        }
        for path in sorted(view_root.glob("*.csv"))
    }

    label_tasks = {}
    for task_name, task_cfg in config["evaluation"]["label_tasks"].items():
        column = str(task_cfg["label_column"])
        label_values = moa[column].astype(str)
        allowed_classes = task_cfg.get("allowed_classes")
        if allowed_classes:
            allowed_classes = {str(label) for label in allowed_classes}
            label_values = label_values[label_values.isin(allowed_classes)]
        counts = label_values.value_counts().sort_index()
        threshold = int(task_cfg.get("min_class_count", 1))
        retained = counts[counts >= threshold]
        label_tasks[task_name] = {
            "label_column": column,
            "minimum_class_count": threshold,
            "raw_class_count": int(len(counts)),
            "retained_class_count": int(len(retained)),
            "retained_sample_count": int(retained.sum()),
            "class_counts": {
                str(key): int(value) for key, value in retained.items()
            },
        }

    config_path = Path(config["_meta"]["config_path"])
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "project_name": config["project_name"],
        "config": _file_record(config_path),
        "inputs": inputs,
        "benchmark": {
            "drug_count": int(len(benchmark_drugs)),
            "moa_triple_count": int(len(moa)),
            "label_tasks": label_tasks,
        },
        "triple_groups": groups,
        "graph_views": views,
        "evaluation": {
            "protocol": config["evaluation"]["protocol"],
            "cv_folds": int(config["evaluation"]["cv_folds"]),
            "cv_repeats": int(config["evaluation"]["cv_repeats"]),
            "random_seed": int(config["random_seed"]),
            "label_group": config["evaluation"]["label_group"],
            "allow_label_group_in_views": bool(
                config["evaluation"].get("allow_label_group_in_views", False)
            ),
            "cohorts": config["evaluation"].get("cohorts", {}),
        },
        "scope_note": (
            "The manifest documents the 433-drug benchmark rebuilt by this "
            "repository. It must not be used to infer unsupported cold-start "
            "or full-DrugBank coverage claims."
        ),
    }

    reports_dir = ensure_dir(Path(config["paths"]["reports_dir"]))
    json_path = reports_dir / "resource_manifest.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    rows = []
    for kind, records in (("group", groups), ("view", views)):
        for name, record in records.items():
            rows.append({
                "kind": kind,
                "name": name,
                "triple_count": record["triple_count"],
                "entity_count": record["entity_count"],
                "relation_count": record["relation_count"],
                "benchmark_drug_count": record["benchmark_drug_count"],
                "benchmark_drug_coverage": record["benchmark_drug_coverage"],
                "sha256": record["sha256"],
            })
    pd.DataFrame(rows).to_csv(
        reports_dir / "resource_manifest_table.csv",
        index=False,
    )
    logger.info("Wrote resource manifest to %s", json_path)
    return json_path
