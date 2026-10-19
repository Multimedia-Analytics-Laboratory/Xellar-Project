"""Triple construction logic extracted from the benchmark notebooks."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .io_utils import ensure_dir, write_json


def _moa_superclass(moa_value, overrides):
    text = str(moa_value).strip()
    if text in overrides:
        return overrides[text]
    if not text:
        return "unknown"
    return text.split()[-1]


def _load_benchmark_expanded(config):
    benchmark_dir = Path(config["paths"]["intermediate_dir"]) / "benchmark"
    expanded_path = benchmark_dir / "benchmark_expanded_targets.csv"
    return pd.read_csv(expanded_path)


def _load_primekg_subset(config, logger):
    relation_map = config.get("primekg_relation_map", {})
    required_relations = sorted(set(relation_map.values()))
    logger.info("Reading PrimeKG relation subset from %s", config["paths"]["primekg_csv"])
    frame = pd.read_csv(
        config["paths"]["primekg_csv"],
        usecols=["relation", "x_name", "y_name"],
        low_memory=False,
    )
    frame = frame[frame["relation"].isin(required_relations)].copy()
    frame["x_name_lower"] = frame["x_name"].fillna("").astype(str).str.lower()
    frame["y_name_lower"] = frame["y_name"].fillna("").astype(str).str.lower()
    logger.info("PrimeKG subset contains %d rows across %d relations.", len(frame), frame["relation"].nunique())
    return frame


def _build_drug_level_relation(expanded, primekg_subset, relation_name):
    drug_frame = expanded.copy()
    drug_frame["pert_iname_lower"] = drug_frame["pert_iname"].fillna("").astype(str).str.lower()
    subset = primekg_subset[primekg_subset["relation"] == relation_name].copy()
    merged = pd.merge(
        drug_frame,
        subset,
        left_on="pert_iname_lower",
        right_on="x_name_lower",
        how="left",
    )
    merged = merged[["pert_iname", "relation", "y_name"]].drop_duplicates()
    merged = merged.dropna(subset=["y_name"])
    merged = merged.rename(columns={"pert_iname": "head", "y_name": "tail"})
    return merged[["head", "relation", "tail"]]


def _build_target_level_relation(expanded, primekg_subset, relation_name):
    subset = primekg_subset[primekg_subset["relation"] == relation_name].copy()
    merged = pd.merge(
        expanded,
        subset,
        left_on="Target",
        right_on="x_name_lower",
        how="left",
    )
    merged = merged[["Target", "relation", "y_name"]].drop_duplicates()
    merged = merged.dropna(subset=["y_name"])
    merged = merged.rename(columns={"Target": "head", "y_name": "tail"})
    return merged[["head", "relation", "tail"]]


def _build_ppi_relation(expanded, primekg_subset, relation_name):
    subset = primekg_subset[primekg_subset["relation"] == relation_name].copy()
    left_merge = pd.merge(
        expanded,
        subset,
        left_on="Target",
        right_on="x_name_lower",
        how="left",
    )
    left_merge = left_merge[["Target", "relation", "y_name_lower"]].drop_duplicates()
    left_merge = left_merge.rename(columns={"Target": "head", "y_name_lower": "tail"})

    right_merge = pd.merge(
        expanded,
        subset,
        left_on="Target",
        right_on="y_name_lower",
        how="left",
    )
    right_merge = right_merge[["Target", "relation", "x_name_lower"]].drop_duplicates()
    right_merge = right_merge.rename(columns={"Target": "head", "x_name_lower": "tail"})

    merged = pd.concat([left_merge, right_merge], ignore_index=True).drop_duplicates()
    merged = merged.dropna(subset=["tail"])
    return merged[["head", "relation", "tail"]]


def _load_prebuilt_groups(config, logger):
    group_frames = {}
    for group_name, file_path in config.get("prebuilt_groups", {}).items():
        logger.info("Loading prebuilt triple group %s from %s", group_name, file_path)
        frame = pd.read_csv(file_path)
        expected_columns = ["head", "tail", "relation"]
        missing = [column for column in expected_columns if column not in frame.columns]
        if missing:
            raise ValueError(
                "Prebuilt triple group %s is missing required columns: %s"
                % (group_name, ", ".join(missing))
            )
        frame = frame[["head", "relation", "tail"]].drop_duplicates()
        group_frames[group_name] = frame
    return group_frames


def build_triple_groups(config, logger):
    out_root = ensure_dir(Path(config["paths"]["intermediate_dir"]) / "triples")
    group_dir = ensure_dir(out_root / "groups")
    view_dir = ensure_dir(out_root / "views")

    if config.get("prebuilt_groups"):
        group_frames = _load_prebuilt_groups(config, logger)
    else:
        expanded = _load_benchmark_expanded(config)
        moa_col = config["benchmark"]["repurposing_hub_moa_column"]
        target_col = config["benchmark"]["repurposing_hub_target_column"]
        overrides = config.get("relations", {}).get("moa_superclass_overrides", {})

        expanded["pert_iname"] = expanded["pert_iname"].fillna("").astype(str).str.lower()
        expanded[target_col] = expanded[target_col].where(expanded[target_col].notna(), None)

        group_frames = {}

        moa_frame = expanded[["pert_iname", moa_col]].drop_duplicates().copy()
        moa_frame["relation"] = moa_frame[moa_col].map(lambda value: _moa_superclass(value, overrides))
        moa_frame = moa_frame.rename(columns={"pert_iname": "head", moa_col: "tail"})
        group_frames["moa"] = moa_frame[["head", "relation", "tail"]].drop_duplicates()

        target_frame = expanded[["pert_iname", target_col]].copy()
        target_frame = target_frame.dropna(subset=[target_col])
        target_frame["relation"] = "target"
        target_frame = target_frame.rename(columns={"pert_iname": "head", target_col: "tail"})
        group_frames["target"] = target_frame[["head", "relation", "tail"]].drop_duplicates()

        primekg_subset = _load_primekg_subset(config, logger)
        relation_map = config.get("primekg_relation_map", {})
        relation_flags = config.get("relations", {})

        if relation_flags.get("use_indication", True):
            group_frames["indication"] = _build_drug_level_relation(
                expanded, primekg_subset, relation_map["indication"]
            )
        if relation_flags.get("use_disease_protein", False):
            group_frames["disease_protein"] = _build_target_level_relation(
                expanded, primekg_subset, relation_map["disease_protein"]
            )
        if relation_flags.get("use_ppi", False):
            group_frames["ppi"] = _build_ppi_relation(
                expanded, primekg_subset, relation_map["ppi"]
            )
        if relation_flags.get("use_anatomy", True):
            group_frames["anatomy"] = _build_target_level_relation(
                expanded, primekg_subset, relation_map["anatomy"]
            )
        if relation_flags.get("use_pathway", True):
            group_frames["pathway"] = _build_target_level_relation(
                expanded, primekg_subset, relation_map["pathway"]
            )

    manifest = {"groups": {}, "views": {}}

    for group_name, frame in group_frames.items():
        group_path = group_dir / ("%s.csv" % group_name)
        frame.to_csv(group_path, index=False)
        manifest["groups"][group_name] = {
            "path": str(group_path),
            "rows": int(len(frame)),
            "unique_heads": int(frame["head"].nunique()) if not frame.empty else 0,
            "unique_relations": int(frame["relation"].nunique()) if not frame.empty else 0,
            "unique_tails": int(frame["tail"].nunique()) if not frame.empty else 0,
        }
        logger.info("Saved %s group with %d triples.", group_name, len(frame))

    for view_name, group_names in config.get("views", {}).items():
        missing = [group_name for group_name in group_names if group_name not in group_frames]
        if missing:
            raise ValueError("View %s references missing triple groups: %s" % (view_name, ", ".join(missing)))
        view_frame = pd.concat([group_frames[group_name] for group_name in group_names], ignore_index=True)
        view_frame = view_frame.drop_duplicates().reset_index(drop=True)
        view_path = view_dir / ("%s.csv" % view_name)
        view_frame.to_csv(view_path, index=False)
        manifest["views"][view_name] = {
            "path": str(view_path),
            "groups": group_names,
            "rows": int(len(view_frame)),
            "unique_entities": int(pd.concat([view_frame["head"], view_frame["tail"]]).nunique()) if not view_frame.empty else 0,
            "unique_relations": int(view_frame["relation"].nunique()) if not view_frame.empty else 0,
        }
        logger.info("Saved %s view with %d triples.", view_name, len(view_frame))

    write_json(out_root / "triple_manifest.json", manifest)
    return manifest
