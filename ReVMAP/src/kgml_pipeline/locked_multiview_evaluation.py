"""Development-locked multi-view selection and external evaluation.

The protocol deliberately uses two commands. Development selection writes a
content-addressed lock before any external-cohort evaluation is run. The
external command verifies that lock, evaluates the already selected strategy,
and compares it with the incumbent full-KG expert on identical folds.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import load_config
from .io_utils import ensure_dir, read_json, write_json
from .relation_routing import evaluate_relation_routing


LOCK_SCHEMA = "kgml.locked_multiview_strategy.v1"


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _sha256_path(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_digest(payload):
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _resolve_protocol_path(config, value):
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    project_root = Path(config["_meta"]["project_root"])
    return (project_root / path).resolve()


def _protocol(config):
    protocol = config.get("locked_multiview_evaluation")
    if not isinstance(protocol, dict):
        raise ValueError("Missing locked_multiview_evaluation config section")
    return protocol


def _output_dir(config):
    protocol = _protocol(config)
    raw = protocol.get(
        "output_dir",
        Path(config["paths"]["reports_dir"]) / "locked_multiview",
    )
    return ensure_dir(_resolve_protocol_path(config, raw))


def _stage_config(config, stage_name):
    protocol = _protocol(config)
    stage = protocol.get(stage_name)
    if not isinstance(stage, dict):
        raise ValueError(
            "locked_multiview_evaluation.%s must be configured" % stage_name
        )
    base_path = _resolve_protocol_path(config, stage["base_config_path"])
    stage_config = load_config(base_path)
    routing = copy.deepcopy(protocol.get("relation_routing", {}))
    routing.update(copy.deepcopy(stage.get("relation_routing", {})))
    routing["cohort_name"] = str(stage["cohort_name"])
    routing["output_dir"] = str(_resolve_protocol_path(config, stage["output_dir"]))
    routing.setdefault(
        "pair_table_path",
        stage_config.get("pair_evaluation", {}).get("pair_table_path"),
    )
    stage_config["relation_routing"] = routing
    return stage_config, base_path


def _pareto_flags(score_matrix):
    flags = {}
    for strategy in score_matrix.index:
        current = score_matrix.loc[strategy]
        dominated = False
        for challenger in score_matrix.index:
            if challenger == strategy:
                continue
            other = score_matrix.loc[challenger]
            if bool((other >= current).all() and (other > current).any()):
                dominated = True
                break
        flags[strategy] = not dominated
    return flags


def select_locked_strategy(summary_df, selection_cfg):
    """Select one strategy using prespecified Pareto/minimax-rank rules."""

    task = str(selection_cfg.get("task", "core4_action"))
    splits = [str(value) for value in selection_cfg["selection_splits"]]
    candidates = [str(value) for value in selection_cfg["candidates"]]
    required_repeats = int(selection_cfg.get("required_repeats", 3))
    priority = [
        str(value)
        for value in selection_cfg.get("deterministic_tie_break_order", candidates)
    ]
    priority_index = {
        strategy: index for index, strategy in enumerate(priority)
    }

    selected_rows = summary_df[
        summary_df["task"].astype(str).eq(task)
        & summary_df["split"].astype(str).isin(splits)
        & summary_df["strategy"].astype(str).isin(candidates)
    ].copy()
    if selected_rows.empty:
        raise ValueError("No development rows matched the selection protocol")

    duplicate_keys = selected_rows.duplicated(["strategy", "split"], keep=False)
    if duplicate_keys.any():
        raise ValueError("Development summary has duplicate strategy/split rows")
    available = set(
        zip(
            selected_rows["strategy"].astype(str),
            selected_rows["split"].astype(str),
        )
    )
    missing = [
        (strategy, split)
        for strategy in candidates
        for split in splits
        if (strategy, split) not in available
    ]
    if missing:
        raise ValueError("Missing development strategy/split rows: %s" % missing)
    if (selected_rows["cv_repeats"].astype(int) < required_repeats).any():
        raise ValueError(
            "Every development result must contain at least %d repeats"
            % required_repeats
        )

    score_matrix = selected_rows.pivot(
        index="strategy",
        columns="split",
        values="f1_macro_mean",
    ).loc[candidates, splits]
    ranks = score_matrix.rank(axis=0, ascending=False, method="min")
    pareto = _pareto_flags(score_matrix)

    audit_rows = []
    for strategy in candidates:
        row = {
            "strategy": strategy,
            "pareto_optimal": bool(pareto[strategy]),
            "worst_rank": float(ranks.loc[strategy].max()),
            "rank_sum": float(ranks.loc[strategy].sum()),
            "minimum_split_macro_f1": float(score_matrix.loc[strategy].min()),
            "mean_split_macro_f1": float(score_matrix.loc[strategy].mean()),
            "tie_break_priority": int(
                priority_index.get(strategy, len(priority) + candidates.index(strategy))
            ),
        }
        for split in splits:
            row["macro_f1__%s" % split] = float(score_matrix.loc[strategy, split])
            row["rank__%s" % split] = float(ranks.loc[strategy, split])
        audit_rows.append(row)
    audit_df = pd.DataFrame(audit_rows)
    eligible = audit_df[audit_df["pareto_optimal"]].copy()
    eligible = eligible.sort_values(
        [
            "worst_rank",
            "rank_sum",
            "minimum_split_macro_f1",
            "mean_split_macro_f1",
            "tie_break_priority",
        ],
        ascending=[True, True, False, False, True],
        kind="mergesort",
    )
    locked_strategy = str(eligible.iloc[0]["strategy"])
    audit_df["selected"] = audit_df["strategy"].eq(locked_strategy)
    audit_df = audit_df.sort_values(
        ["selected", "pareto_optimal", "worst_rank", "rank_sum"],
        ascending=[False, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    return locked_strategy, audit_df


def _selection_rule_text():
    return (
        "Restrict to prespecified candidates and primary splits; retain the "
        "Pareto-optimal set; minimize worst split-specific rank; then minimize "
        "rank sum; then maximize the minimum split macro-F1; then maximize mean "
        "macro-F1; finally apply the prespecified deterministic complexity tie-break."
    )


def run_development_multiview_selection(config, logger):
    """Evaluate development candidates and write the external-blind lock."""

    protocol = _protocol(config)
    selection_cfg = protocol["selection"]
    output_dir = _output_dir(config)
    lock_path = output_dir / "strategy_lock.json"
    if lock_path.exists() and not bool(protocol.get("overwrite_lock", False)):
        raise FileExistsError(
            "Strategy lock already exists; refusing to overwrite: %s" % lock_path
        )

    development_config, base_config_path = _stage_config(config, "development")
    summary_df = evaluate_relation_routing(development_config, logger)
    locked_strategy, audit_df = select_locked_strategy(summary_df, selection_cfg)
    audit_path = output_dir / "development_selection_audit.csv"
    audit_df.to_csv(audit_path, index=False)

    development_output = Path(
        development_config["relation_routing"]["output_dir"]
    )
    summary_path = development_output / "routing_summary.csv"
    repeat_path = development_output / "routing_repeat_metrics.csv"
    lock_material = {
        "schema": LOCK_SCHEMA,
        "created_at_utc": _utc_now(),
        "external_results_accessed": False,
        "task": str(selection_cfg.get("task", "core4_action")),
        "selection_splits": [
            str(value) for value in selection_cfg["selection_splits"]
        ],
        "candidates": [str(value) for value in selection_cfg["candidates"]],
        "required_repeats": int(selection_cfg.get("required_repeats", 3)),
        "selection_rule": _selection_rule_text(),
        "locked_strategy": locked_strategy,
        "incumbent_strategy": str(
            selection_cfg.get(
                "incumbent_strategy",
                "expert::full_label_free",
            )
        ),
        "development_cohort": str(
            development_config["relation_routing"]["cohort_name"]
        ),
        "development_base_config": str(base_config_path),
        "development_base_config_sha256": _sha256_path(base_config_path),
        "development_summary": str(summary_path),
        "development_summary_sha256": _sha256_path(summary_path),
        "development_repeat_metrics": str(repeat_path),
        "development_repeat_metrics_sha256": _sha256_path(repeat_path),
        "selection_audit": str(audit_path),
        "selection_audit_sha256": _sha256_path(audit_path),
        "selected_row": audit_df[
            audit_df["strategy"].eq(locked_strategy)
        ].iloc[0].to_dict(),
    }
    lock_material["lock_id"] = _canonical_digest(lock_material)
    write_json(lock_path, lock_material)
    (output_dir / "strategy_lock.sha256").write_text(
        _sha256_path(lock_path) + "\n",
        encoding="utf-8",
    )
    logger.info(
        "Locked development-selected strategy %s in %s",
        locked_strategy,
        lock_path,
    )
    return lock_material


def _repeat_comparison(repeat_df, *, task, selected, incumbent, splits):
    subset = repeat_df[
        repeat_df["task"].astype(str).eq(task)
        & repeat_df["split"].astype(str).isin(splits)
        & repeat_df["strategy"].astype(str).isin([selected, incumbent])
    ].copy()
    wide = subset.pivot_table(
        index=["split", "repeat", "seed"],
        columns="strategy",
        values="f1_macro",
        aggfunc="first",
    ).reset_index()
    if selected not in wide.columns or incumbent not in wide.columns:
        raise ValueError("External repeat metrics lack selected/incumbent rows")
    wide["selected_strategy"] = selected
    wide["incumbent_strategy"] = incumbent
    wide["selected_f1_macro"] = wide[selected].astype(float)
    wide["incumbent_f1_macro"] = wide[incumbent].astype(float)
    wide["delta_selected_minus_incumbent"] = (
        wide["selected_f1_macro"] - wide["incumbent_f1_macro"]
    )
    wide["selected_wins"] = wide["delta_selected_minus_incumbent"] > 0.0
    return wide[[
        "split",
        "repeat",
        "seed",
        "selected_strategy",
        "incumbent_strategy",
        "selected_f1_macro",
        "incumbent_f1_macro",
        "delta_selected_minus_incumbent",
        "selected_wins",
    ]]


def _external_summary(
    summary_df,
    repeat_comparison,
    *,
    task,
    selected,
    incumbent,
    splits,
    manuscript_reference,
):
    subset = summary_df[
        summary_df["task"].astype(str).eq(task)
        & summary_df["split"].astype(str).isin(splits)
        & summary_df["strategy"].astype(str).isin([selected, incumbent])
    ].copy()
    lookup = {
        (str(row.split), str(row.strategy)): row
        for row in subset.itertuples(index=False)
    }
    rows = []
    for split in splits:
        selected_row = lookup.get((split, selected))
        incumbent_row = lookup.get((split, incumbent))
        if selected_row is None or incumbent_row is None:
            raise ValueError("External summary lacks %s selected/incumbent rows" % split)
        repeats = repeat_comparison[repeat_comparison["split"].eq(split)]
        rows.append({
            "split": split,
            "selected_strategy": selected,
            "selected_f1_macro_mean": float(selected_row.f1_macro_mean),
            "selected_f1_macro_std": float(selected_row.f1_macro_std),
            "incumbent_strategy": incumbent,
            "incumbent_f1_macro_mean": float(incumbent_row.f1_macro_mean),
            "incumbent_f1_macro_std": float(incumbent_row.f1_macro_std),
            "delta_selected_minus_incumbent": float(
                selected_row.f1_macro_mean - incumbent_row.f1_macro_mean
            ),
            "selected_repeat_wins": int(repeats["selected_wins"].sum()),
            "repeat_count": int(len(repeats)),
            "manuscript_locked_fullkg_f1_macro": (
                float(manuscript_reference[split])
                if split in manuscript_reference else math.nan
            ),
            "delta_selected_minus_manuscript": (
                float(selected_row.f1_macro_mean)
                - float(manuscript_reference[split])
                if split in manuscript_reference else math.nan
            ),
        })
    return pd.DataFrame(rows)


def _promotion_assessment(comparison_df, protocol, selected, incumbent):
    criteria = protocol.get("promotion_criteria", {})
    primary_splits = [
        str(value)
        for value in criteria.get(
            "primary_splits",
            protocol["selection"]["selection_splits"],
        )
    ]
    strict_splits = [
        str(value)
        for value in criteria.get(
            "strict_splits",
            ["drug_disjoint", "target_disjoint"],
        )
    ]
    noninferiority_margin = float(criteria.get("noninferiority_margin", 0.005))
    minimum_strict_gain = float(criteria.get("minimum_strict_gain", 0.005))
    minimum_repeat_wins = int(criteria.get("minimum_repeat_wins", 2))
    primary = comparison_df[comparison_df["split"].isin(primary_splits)]
    strict = primary[primary["split"].isin(strict_splits)]
    no_material_loss = bool(
        (
            primary["delta_selected_minus_incumbent"]
            >= -noninferiority_margin
        ).all()
    )
    qualifying = strict[
        (strict["delta_selected_minus_incumbent"] >= minimum_strict_gain)
        & (strict["selected_repeat_wins"] >= minimum_repeat_wins)
    ]
    strict_gain = not qualifying.empty
    promoted = bool(selected != incumbent and no_material_loss and strict_gain)
    return {
        "selected_is_new_strategy": selected != incumbent,
        "no_primary_split_loss_beyond_margin": no_material_loss,
        "strict_split_gain_with_repeat_majority": strict_gain,
        "qualifying_strict_splits": qualifying["split"].astype(str).tolist(),
        "noninferiority_margin": noninferiority_margin,
        "minimum_strict_gain": minimum_strict_gain,
        "minimum_repeat_wins": minimum_repeat_wins,
        "promote_over_manuscript_incumbent": promoted,
        "interpretation": (
            "Promotion is a prespecified practical decision rule, not a formal "
            "null-hypothesis significance test."
        ),
    }


def _format_mean_std(mean, std):
    return "%.4f ± %.4f" % (float(mean), float(std))


def _write_external_report(path, lock, comparison_df, assessment):
    lines = [
        "# Development-locked multi-view external evaluation",
        "",
        "- Locked strategy: `%s`" % lock["locked_strategy"],
        "- Incumbent: `%s`" % lock["incumbent_strategy"],
        "- Lock ID: `%s`" % lock["lock_id"],
        "- External cohort was not used for strategy selection.",
        "",
        "## External macro-F1",
        "",
        "| Split | Locked strategy | Full-KG incumbent | Delta | Repeat wins | Manuscript reference |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in comparison_df.itertuples(index=False):
        reference = (
            "%.4f" % row.manuscript_locked_fullkg_f1_macro
            if not math.isnan(row.manuscript_locked_fullkg_f1_macro)
            else "NA"
        )
        lines.append(
            "| %s | %s | %s | %+.4f | %d/%d | %s |"
            % (
                row.split,
                _format_mean_std(
                    row.selected_f1_macro_mean,
                    row.selected_f1_macro_std,
                ),
                _format_mean_std(
                    row.incumbent_f1_macro_mean,
                    row.incumbent_f1_macro_std,
                ),
                row.delta_selected_minus_incumbent,
                row.selected_repeat_wins,
                row.repeat_count,
                reference,
            )
        )
    lines.extend([
        "",
        "## Prespecified promotion assessment",
        "",
        "- No primary-split material loss: `%s`"
        % assessment["no_primary_split_loss_beyond_margin"],
        "- Strict-split gain with repeat majority: `%s`"
        % assessment["strict_split_gain_with_repeat_majority"],
        "- Promote over manuscript incumbent: `%s`"
        % assessment["promote_over_manuscript_incumbent"],
        "- This is a practical decision rule rather than a formal significance test.",
        "",
    ])
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def run_external_locked_multiview_evaluation(config, logger):
    """Verify the development lock, then evaluate the external cohort."""

    protocol = _protocol(config)
    output_dir = _output_dir(config)
    lock_path = output_dir / "strategy_lock.json"
    lock_sha_path = output_dir / "strategy_lock.sha256"
    if not lock_path.exists() or not lock_sha_path.exists():
        raise FileNotFoundError(
            "Run select-locked-multiview before external evaluation"
        )
    expected_lock_sha = lock_sha_path.read_text(encoding="utf-8").strip()
    actual_lock_sha = _sha256_path(lock_path)
    if actual_lock_sha != expected_lock_sha:
        raise RuntimeError("Strategy lock hash verification failed")
    lock = read_json(lock_path)
    if lock.get("schema") != LOCK_SCHEMA:
        raise ValueError("Unsupported strategy lock schema")
    if _sha256_path(lock["development_summary"]) != lock[
        "development_summary_sha256"
    ]:
        raise RuntimeError("Development summary changed after strategy locking")
    if _sha256_path(lock["selection_audit"]) != lock["selection_audit_sha256"]:
        raise RuntimeError("Development selection audit changed after locking")

    external_config, base_config_path = _stage_config(config, "external")
    external_manifest = {
        "started_at_utc": _utc_now(),
        "lock_id": lock["lock_id"],
        "strategy_lock_sha256": actual_lock_sha,
        "locked_strategy": lock["locked_strategy"],
        "incumbent_strategy": lock["incumbent_strategy"],
        "external_cohort": external_config["relation_routing"]["cohort_name"],
        "external_base_config": str(base_config_path),
        "external_base_config_sha256": _sha256_path(base_config_path),
    }
    write_json(output_dir / "external_run_manifest.json", external_manifest)

    summary_df = evaluate_relation_routing(external_config, logger)
    external_output = Path(external_config["relation_routing"]["output_dir"])
    repeat_df = pd.read_csv(external_output / "routing_repeat_metrics.csv")
    task = str(lock["task"])
    evaluation_splits = [
        str(value)
        for value in external_config["relation_routing"]["splits"]
    ]
    repeat_comparison = _repeat_comparison(
        repeat_df,
        task=task,
        selected=str(lock["locked_strategy"]),
        incumbent=str(lock["incumbent_strategy"]),
        splits=evaluation_splits,
    )
    repeat_comparison.to_csv(
        output_dir / "external_repeat_comparison.csv",
        index=False,
    )
    manuscript_reference = {
        str(key): float(value)
        for key, value in protocol.get(
            "manuscript_locked_fullkg_reference",
            {},
        ).items()
    }
    comparison_df = _external_summary(
        summary_df,
        repeat_comparison,
        task=task,
        selected=str(lock["locked_strategy"]),
        incumbent=str(lock["incumbent_strategy"]),
        splits=evaluation_splits,
        manuscript_reference=manuscript_reference,
    )
    comparison_df.to_csv(
        output_dir / "external_locked_comparison.csv",
        index=False,
    )
    assessment = _promotion_assessment(
        comparison_df,
        protocol,
        str(lock["locked_strategy"]),
        str(lock["incumbent_strategy"]),
    )
    write_json(output_dir / "promotion_assessment.json", assessment)
    _write_external_report(
        output_dir / "external_locked_evaluation.md",
        lock,
        comparison_df,
        assessment,
    )
    logger.info(
        "External locked evaluation completed for %s; promotion=%s",
        lock["locked_strategy"],
        assessment["promote_over_manuscript_incumbent"],
    )
    return comparison_df

