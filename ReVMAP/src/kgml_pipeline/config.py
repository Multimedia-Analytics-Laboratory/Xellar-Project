"""Configuration loading and path resolution helpers."""

from __future__ import annotations

import copy
import json
from pathlib import Path


PATH_KEYS = (
    "combined_metadata_csv",
    "repurposing_hub_tsv",
    "primekg_csv",
    "prebuilt_benchmark_csv",
    "prebuilt_expanded_targets_csv",
    "intermediate_dir",
    "processed_dir",
    "reports_dir",
    "logs_dir",
)


def _resolve_path(base_dir, raw_value):
    path = Path(str(raw_value)).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def load_config(config_path):
    config_path = Path(config_path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    resolved = copy.deepcopy(config)
    project_root = config_path.parent.parent.resolve()
    paths = resolved.setdefault("paths", {})

    for key in PATH_KEYS:
        if key in paths and paths[key]:
            paths[key] = str(_resolve_path(project_root, paths[key]))

    raw_files = paths.get("open_targets_drug_files", [])
    paths["open_targets_drug_files"] = [
        str(_resolve_path(project_root, file_path))
        for file_path in raw_files
    ]

    prebuilt_groups = resolved.get("prebuilt_groups", {})
    resolved["prebuilt_groups"] = {
        group_name: str(_resolve_path(project_root, file_path))
        for group_name, file_path in prebuilt_groups.items()
        if file_path
    }

    resolved["_meta"] = {
        "config_path": str(config_path),
        "project_root": str(project_root),
    }
    return resolved


def validate_config(config, check_inputs=True):
    errors = []
    warnings = []

    paths = config.get("paths", {})
    benchmark = config.get("benchmark", {})
    training = config.get("training", {})
    evaluation = config.get("evaluation", {})
    views = config.get("views", {})
    prebuilt_groups = config.get("prebuilt_groups", {})

    required_path_keys = [
        "intermediate_dir",
        "processed_dir",
        "reports_dir",
    ]
    if not prebuilt_groups:
        required_path_keys.append("primekg_csv")
    for key in required_path_keys:
        if not paths.get(key):
            errors.append("Missing config.paths.%s" % key)

    required_benchmark_keys = (
        "drug_name_column",
        "repurposing_hub_name_column",
        "repurposing_hub_moa_column",
        "repurposing_hub_target_column",
        "minimum_moa_count",
    )
    for key in required_benchmark_keys:
        if key not in benchmark:
            errors.append("Missing config.benchmark.%s" % key)

    ratios = training.get("split_ratios", [])
    if ratios:
        total = float(sum(ratios))
        if abs(total - 1.0) > 1e-6:
            errors.append("training.split_ratios must sum to 1.0")
        if len(ratios) != 3:
            errors.append("training.split_ratios must contain exactly 3 values")
    else:
        errors.append("training.split_ratios is required")

    if not training.get("models"):
        errors.append("training.models must contain at least one model")

    if not views:
        errors.append("config.views must define at least one triple view")

    label_group = str(evaluation.get("label_group", "moa"))
    views_with_label_group = [
        view_name
        for view_name, group_names in views.items()
        if label_group in group_names
    ]
    allow_label_group = bool(evaluation.get("allow_label_group_in_views", False))
    if views_with_label_group and not allow_label_group:
        errors.append(
            "Label group %r is present in embedding views %s. This leaks downstream "
            "labels into learned features. Remove the group or explicitly set "
            "evaluation.allow_label_group_in_views=true for legacy reproduction only."
            % (label_group, ", ".join(sorted(views_with_label_group)))
        )
    elif views_with_label_group:
        warnings.append(
            "Legacy label leakage is enabled: group %r is present in embedding views %s."
            % (label_group, ", ".join(sorted(views_with_label_group)))
        )

    protocol = str(evaluation.get("protocol", "legacy_holdout"))
    supported_protocols = {"legacy_holdout", "repeated_stratified_cv"}
    if protocol not in supported_protocols:
        errors.append(
            "evaluation.protocol must be one of: %s"
            % ", ".join(sorted(supported_protocols))
        )

    label_tasks = evaluation.get("label_tasks", {})
    if label_tasks and not isinstance(label_tasks, dict):
        errors.append("evaluation.label_tasks must be an object mapping task names to settings")
    for task_name, task_cfg in label_tasks.items():
        if not isinstance(task_cfg, dict):
            errors.append("evaluation.label_tasks.%s must be an object" % task_name)
            continue
        if task_cfg.get("label_column") not in {"relation", "tail"}:
            errors.append(
                "evaluation.label_tasks.%s.label_column must be 'relation' or 'tail'"
                % task_name
            )

    source_scope = str(benchmark.get("source_scope", "cpg0004"))
    has_raw_benchmark_inputs = bool(
        paths.get("combined_metadata_csv") and paths.get("repurposing_hub_tsv")
    )
    has_repurposing_scope_inputs = bool(
        source_scope == "repurposing_hub" and paths.get("repurposing_hub_tsv")
    )
    has_prebuilt_benchmark_inputs = bool(
        paths.get("prebuilt_benchmark_csv") and paths.get("prebuilt_expanded_targets_csv")
    )
    if (
        not has_raw_benchmark_inputs
        and not has_repurposing_scope_inputs
        and not has_prebuilt_benchmark_inputs
    ):
        errors.append(
            "Provide either raw benchmark inputs (combined_metadata_csv + repurposing_hub_tsv) "
            "or frozen benchmark inputs (prebuilt_benchmark_csv + prebuilt_expanded_targets_csv). "
            "For benchmark.source_scope='repurposing_hub', repurposing_hub_tsv is sufficient "
            "unless metadata-drug exclusion is requested."
        )
    if (
        source_scope == "repurposing_hub"
        and benchmark.get("exclude_combined_metadata_drugs", False)
        and not paths.get("combined_metadata_csv")
    ):
        errors.append(
            "benchmark.exclude_combined_metadata_drugs=true requires "
            "config.paths.combined_metadata_csv."
        )

    if check_inputs:
        input_keys = []
        if not prebuilt_groups:
            input_keys.append("primekg_csv")
        if source_scope == "repurposing_hub":
            if paths.get("repurposing_hub_tsv"):
                input_keys.append("repurposing_hub_tsv")
            if benchmark.get("exclude_combined_metadata_drugs", False):
                input_keys.append("combined_metadata_csv")
        elif has_raw_benchmark_inputs:
            input_keys.extend(["combined_metadata_csv", "repurposing_hub_tsv"])
        if has_prebuilt_benchmark_inputs:
            input_keys.extend(["prebuilt_benchmark_csv", "prebuilt_expanded_targets_csv"])
        for key in input_keys:
            input_path = paths.get(key)
            if input_path and not Path(input_path).exists():
                errors.append("Input file not found for %s: %s" % (key, input_path))
        for file_path in paths.get("open_targets_drug_files", []):
            if file_path and not Path(file_path).exists():
                warnings.append("Open Targets parquet not found: %s" % file_path)
        for group_name, file_path in prebuilt_groups.items():
            if file_path and not Path(file_path).exists():
                errors.append("Prebuilt triple group not found for %s: %s" % (group_name, file_path))

    return {"errors": errors, "warnings": warnings}
