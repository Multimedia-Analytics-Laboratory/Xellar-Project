"""Benchmark preparation extracted from the manuscript notebooks."""

from __future__ import annotations

from pathlib import Path
from shutil import copyfile

import pandas as pd

from .io_utils import ensure_dir, write_json


def _normalize_string_series(series):
    return series.fillna("").astype(str).str.strip()


def _lower_or_none(value):
    if pd.isna(value):
        return value
    text = str(value).strip()
    if not text:
        return None
    return text.lower()


def _normalize_label(value):
    if pd.isna(value):
        return None
    text = str(value).strip().lower()
    return text or None


def _tokenize_moa(value):
    tokens = []
    if pd.isna(value):
        return tokens
    for raw_token in str(value).split(","):
        token = _normalize_label(raw_token)
        if token and token not in tokens:
            tokens.append(token)
    return tokens


def _canonicalize_moa(raw_moa, metadata_moa):
    metadata_label = _normalize_label(metadata_moa)
    tokens = _tokenize_moa(raw_moa)

    # Updated Repurposing Hub exports sometimes duplicate the same MOA token
    # or append a secondary label; prefer the metadata MOA when it is still present.
    if metadata_label and (not tokens or metadata_label in tokens):
        return metadata_label
    if len(tokens) == 1:
        return tokens[0]
    if tokens:
        return ", ".join(tokens)
    return metadata_label


def _coerce_symbol_like(value):
    if value is None:
        return None

    if isinstance(value, float) and pd.isna(value):
        return None

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        return text.lower()

    if isinstance(value, dict):
        for key in ("approvedSymbol", "symbol", "value", "name"):
            if key in value:
                parsed = _coerce_symbol_like(value[key])
                if parsed:
                    return parsed
        for nested_value in value.values():
            parsed = _coerce_symbol_like(nested_value)
            if parsed:
                return parsed
        return None

    if isinstance(value, (list, tuple, set)):
        for nested_value in value:
            parsed = _coerce_symbol_like(nested_value)
            if parsed:
                return parsed
        return None

    text = str(value).strip()
    if not text:
        return None
    return text.lower()


def _combine_targets(row):
    combined = []
    for key in ("approvedSymbol", "Target"):
        value = row.get(key)
        if pd.isna(value):
            continue
        text = str(value).strip().lower()
        if text and text not in combined:
            combined.append(text)
    if not combined:
        return [None]
    return combined


def _split_targets(value, separator):
    targets = []
    if pd.isna(value):
        return [None]
    for raw_token in str(value).split(separator):
        token = _lower_or_none(raw_token)
        if token and token not in targets:
            targets.append(token)
    return targets or [None]


def load_open_targets_drug_frame(config, logger):
    import pyarrow.parquet as pq

    file_paths = config["paths"].get("open_targets_drug_files", [])
    if not file_paths:
        logger.info("No Open Targets parquet files configured; proceeding without approvedSymbol enrichment.")
        return pd.DataFrame(columns=["prefName", "approvedSymbol"])

    frames = []
    for file_path in file_paths:
        logger.info("Reading Open Targets parquet: %s", file_path)
        table = pq.read_table(file_path, columns=["prefName", "approvedSymbol"])
        frame = pd.DataFrame({
            "prefName": table.column("prefName").to_pylist(),
            "approvedSymbol": table.column("approvedSymbol").to_pylist(),
        })
        frames.append(frame)

    open_targets = pd.concat(frames, ignore_index=True)
    open_targets["prefName"] = open_targets["prefName"].map(_coerce_symbol_like)
    open_targets["approvedSymbol"] = open_targets["approvedSymbol"].map(_coerce_symbol_like)
    open_targets = open_targets.dropna(subset=["prefName"])
    open_targets = open_targets.drop_duplicates()
    logger.info("Loaded %d Open Targets drug-target rows after deduplication.", len(open_targets))
    return open_targets


def _write_benchmark_outputs(out_dir, benchmark, expanded, drug_name_col, moa_col, target_col, logger):
    benchmark_out = out_dir / "benchmark_drug_table.csv"
    expanded_out = out_dir / "benchmark_expanded_targets.csv"
    benchmark.to_csv(benchmark_out, index=False)
    expanded.to_csv(expanded_out, index=False)

    manifest = {
        "benchmark_rows": int(len(benchmark)),
        "expanded_rows": int(len(expanded)),
        "unique_drugs": int(benchmark[drug_name_col].nunique()),
        "unique_moas": int(benchmark[moa_col].nunique()),
        "unique_targets": int(expanded[target_col].dropna().nunique()),
    }
    write_json(out_dir / "benchmark_manifest.json", manifest)

    logger.info("Saved benchmark table to %s", benchmark_out)
    logger.info("Saved expanded target table to %s", expanded_out)
    logger.info("Benchmark summary: %s", manifest)

    return {
        "benchmark_path": str(benchmark_out),
        "expanded_path": str(expanded_out),
        "manifest_path": str(out_dir / "benchmark_manifest.json"),
    }


def _prepare_frozen_benchmark_tables(config, logger, out_dir):
    paths = config["paths"]
    benchmark_cfg = config["benchmark"]
    drug_name_col = benchmark_cfg["drug_name_column"]
    moa_col = benchmark_cfg["repurposing_hub_moa_column"]
    target_col = benchmark_cfg["repurposing_hub_target_column"]

    benchmark_src = Path(paths["prebuilt_benchmark_csv"])
    expanded_src = Path(paths["prebuilt_expanded_targets_csv"])
    logger.info("Using frozen benchmark inputs from %s and %s", benchmark_src, expanded_src)

    benchmark = pd.read_csv(benchmark_src)
    expanded = pd.read_csv(expanded_src)
    benchmark[drug_name_col] = _normalize_string_series(benchmark[drug_name_col]).str.lower()
    expanded[drug_name_col] = _normalize_string_series(expanded[drug_name_col]).str.lower()
    if target_col in expanded.columns:
        expanded[target_col] = expanded[target_col].map(_lower_or_none)

    result = _write_benchmark_outputs(
        out_dir=out_dir,
        benchmark=benchmark,
        expanded=expanded,
        drug_name_col=drug_name_col,
        moa_col=moa_col,
        target_col=target_col,
        logger=logger,
    )
    copyfile(benchmark_src, out_dir / "source_benchmark_snapshot.csv")
    copyfile(expanded_src, out_dir / "source_expanded_snapshot.csv")
    return result


def _prepare_repurposing_hub_benchmark_tables(config, logger, out_dir):
    paths = config["paths"]
    benchmark_cfg = config["benchmark"]
    drug_name_col = benchmark_cfg["drug_name_column"]
    repur_name_col = benchmark_cfg["repurposing_hub_name_column"]
    moa_col = benchmark_cfg["repurposing_hub_moa_column"]
    target_col = benchmark_cfg["repurposing_hub_target_column"]
    min_moa_count = int(benchmark_cfg["minimum_moa_count"])
    target_separator = benchmark_cfg.get("target_separator", ",")

    logger.info(
        "Preparing Repurposing Hub-wide benchmark from %s",
        paths["repurposing_hub_tsv"],
    )
    repurposing = pd.read_csv(paths["repurposing_hub_tsv"], sep="\t")
    repurposing = repurposing[[repur_name_col, target_col, moa_col]].drop_duplicates()
    repurposing[repur_name_col] = _normalize_string_series(
        repurposing[repur_name_col]
    ).str.lower()

    excluded_drugs = set()
    if benchmark_cfg.get("exclude_combined_metadata_drugs", False):
        metadata_path = paths.get("combined_metadata_csv")
        if metadata_path:
            metadata = pd.read_csv(metadata_path, usecols=[drug_name_col])
            excluded_drugs = set(
                _normalize_string_series(metadata[drug_name_col]).str.lower()
            )
            repurposing = repurposing[
                ~repurposing[repur_name_col].isin(excluded_drugs)
            ].copy()
            logger.info(
                "Excluded %d cpg0004/metadata drugs from Repurposing Hub-wide benchmark.",
                len(excluded_drugs),
            )

    single_label_only = bool(benchmark_cfg.get("single_label_moa_only", True))
    if single_label_only:
        repurposing["_moa_tokens"] = repurposing[moa_col].map(_tokenize_moa)
        repurposing = repurposing[
            repurposing["_moa_tokens"].map(len).eq(1)
        ].copy()
        repurposing[moa_col] = repurposing["_moa_tokens"].map(lambda tokens: tokens[0])
        repurposing = repurposing.drop(columns=["_moa_tokens"])
    else:
        repurposing[moa_col] = repurposing[moa_col].map(
            lambda value: _canonicalize_moa(value, None)
        )

    repurposing = repurposing.dropna(subset=[repur_name_col, moa_col])
    repurposing = repurposing[
        repurposing[repur_name_col].astype(str).str.len().gt(0)
        & repurposing[moa_col].astype(str).str.len().gt(0)
    ].copy()

    if benchmark_cfg.get("drop_ambiguous_drug_moa", True):
        moa_counts_by_drug = repurposing.groupby(repur_name_col)[moa_col].nunique()
        unambiguous = set(moa_counts_by_drug[moa_counts_by_drug.eq(1)].index)
        before = len(repurposing)
        repurposing = repurposing[repurposing[repur_name_col].isin(unambiguous)].copy()
        logger.info(
            "Dropped %d ambiguous Repurposing Hub rows where one drug maps to multiple MoAs.",
            before - len(repurposing),
        )

    valid_moas = repurposing[moa_col].value_counts()
    valid_moas = set(valid_moas[valid_moas >= min_moa_count].index)
    repurposing = repurposing[repurposing[moa_col].isin(valid_moas)].copy()

    expanded = repurposing.copy()
    expanded[target_col] = expanded[target_col].map(
        lambda value: _split_targets(value, target_separator)
    )
    expanded = expanded.explode(target_col, ignore_index=True)
    expanded[target_col] = expanded[target_col].map(_lower_or_none)

    open_targets = load_open_targets_drug_frame(config, logger)
    if not open_targets.empty:
        expanded = pd.merge(
            expanded,
            open_targets[["prefName", "approvedSymbol"]].drop_duplicates(),
            left_on=repur_name_col,
            right_on="prefName",
            how="left",
        )
    else:
        expanded["approvedSymbol"] = None

    expanded["combined_targets"] = expanded.apply(_combine_targets, axis=1)
    expanded = expanded.explode("combined_targets", ignore_index=True)
    expanded = expanded[[repur_name_col, "combined_targets", moa_col]].drop_duplicates()
    expanded = expanded.rename(
        columns={
            repur_name_col: drug_name_col,
            "combined_targets": target_col,
        }
    )

    has_non_null_targets = expanded.groupby(drug_name_col)[target_col].transform(
        lambda series: series.notna().any()
    )
    expanded = expanded[expanded[target_col].notna() | ~has_non_null_targets].copy()

    target_strings = (
        expanded.dropna(subset=[target_col])
        .groupby(drug_name_col)[target_col]
        .apply(lambda values: target_separator.join(sorted(set(map(str, values)))))
        .reset_index()
    )
    benchmark = (
        expanded[[drug_name_col, moa_col]]
        .drop_duplicates()
        .merge(target_strings, on=drug_name_col, how="left")
    )
    benchmark = benchmark[[drug_name_col, target_col, moa_col]]

    result = _write_benchmark_outputs(
        out_dir=out_dir,
        benchmark=benchmark,
        expanded=expanded,
        drug_name_col=drug_name_col,
        moa_col=moa_col,
        target_col=target_col,
        logger=logger,
    )
    write_json(
        out_dir / "benchmark_scope_manifest.json",
        {
            "source_scope": "repurposing_hub",
            "excluded_metadata_drug_count": int(len(excluded_drugs)),
            "single_label_moa_only": single_label_only,
            "drop_ambiguous_drug_moa": bool(
                benchmark_cfg.get("drop_ambiguous_drug_moa", True)
            ),
        },
    )
    return result


def prepare_benchmark_tables(config, logger):
    paths = config["paths"]
    benchmark_cfg = config["benchmark"]
    out_dir = ensure_dir(Path(paths["intermediate_dir"]) / "benchmark")

    drug_name_col = benchmark_cfg["drug_name_column"]
    metadata_moa_col = benchmark_cfg["metadata_moa_column"]
    repur_name_col = benchmark_cfg["repurposing_hub_name_column"]
    repur_moa_col = benchmark_cfg["repurposing_hub_moa_column"]
    repur_target_col = benchmark_cfg["repurposing_hub_target_column"]
    min_moa_count = int(benchmark_cfg["minimum_moa_count"])
    target_separator = benchmark_cfg.get("target_separator", ",")

    if paths.get("prebuilt_benchmark_csv") and paths.get("prebuilt_expanded_targets_csv"):
        return _prepare_frozen_benchmark_tables(config, logger, out_dir)

    if benchmark_cfg.get("source_scope") == "repurposing_hub":
        return _prepare_repurposing_hub_benchmark_tables(config, logger, out_dir)

    logger.info("Reading benchmark metadata from %s", paths["combined_metadata_csv"])
    metadata = pd.read_csv(paths["combined_metadata_csv"])
    metadata = metadata[[drug_name_col, metadata_moa_col]].drop_duplicates()
    metadata[drug_name_col] = _normalize_string_series(metadata[drug_name_col]).str.lower()

    logger.info("Reading Repurposing Hub data from %s", paths["repurposing_hub_tsv"])
    repurposing = pd.read_csv(paths["repurposing_hub_tsv"], sep="\t")
    repurposing = repurposing[[repur_name_col, repur_target_col, repur_moa_col]].drop_duplicates()
    repurposing[repur_name_col] = _normalize_string_series(repurposing[repur_name_col]).str.lower()

    benchmark = pd.merge(
        metadata,
        repurposing,
        left_on=drug_name_col,
        right_on=repur_name_col,
        how="left",
    ).drop_duplicates()

    benchmark = benchmark[[drug_name_col, metadata_moa_col, repur_target_col, repur_moa_col]]
    benchmark = benchmark.dropna(subset=[drug_name_col])
    benchmark[repur_moa_col] = benchmark.apply(
        lambda row: _canonicalize_moa(row[repur_moa_col], row[metadata_moa_col]),
        axis=1,
    )
    benchmark = benchmark[benchmark[repur_moa_col].isin(
        benchmark[repur_moa_col].value_counts()[lambda counts: counts >= min_moa_count].index
    )].copy()

    benchmark[repur_target_col] = benchmark[repur_target_col].fillna("").astype(str).str.split(target_separator)
    expanded = benchmark.explode(repur_target_col)
    expanded[repur_target_col] = expanded[repur_target_col].map(_lower_or_none)
    expanded[drug_name_col] = _normalize_string_series(expanded[drug_name_col]).str.lower()

    open_targets = load_open_targets_drug_frame(config, logger)
    if not open_targets.empty:
        expanded = pd.merge(
            expanded,
            open_targets[["prefName", "approvedSymbol"]].drop_duplicates(),
            left_on=drug_name_col,
            right_on="prefName",
            how="left",
        )
    else:
        expanded["approvedSymbol"] = None

    expanded["combined_targets"] = expanded.apply(_combine_targets, axis=1)
    expanded = expanded.explode("combined_targets", ignore_index=True)
    expanded = expanded[[drug_name_col, "combined_targets", repur_moa_col]].drop_duplicates()
    expanded = expanded.rename(columns={"combined_targets": repur_target_col})

    has_non_null_targets = expanded.groupby(drug_name_col)[repur_target_col].transform(lambda series: series.notna().any())
    expanded = expanded[expanded[repur_target_col].notna() | ~has_non_null_targets].copy()

    benchmark_out = out_dir / "benchmark_drug_table.csv"
    benchmark = benchmark[[drug_name_col, repur_target_col, repur_moa_col]]
    return _write_benchmark_outputs(
        out_dir=out_dir,
        benchmark=benchmark,
        expanded=expanded,
        drug_name_col=drug_name_col,
        moa_col=repur_moa_col,
        target_col=repur_target_col,
        logger=logger,
    )
