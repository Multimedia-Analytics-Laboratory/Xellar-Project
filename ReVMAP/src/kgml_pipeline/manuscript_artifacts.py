"""Generate reproducible LaTeX tables and a numeric digest for manuscript editing."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from .io_utils import ensure_dir


def _escape(value):
    return (
        str(value)
        .replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("&", r"\&")
        .replace("%", r"\%")
    )


def _score(row, metric):
    mean = float(row["%s_mean" % metric])
    if math.isnan(mean):
        return "--"
    return "$%.3f$" % mean


def _best_rows(frame, grouping):
    return (
        frame.sort_values(
            grouping + ["f1_macro_mean"],
            ascending=[True] * len(grouping) + [False],
        )
        .groupby(grouping, dropna=False)
        .head(1)
    )


def _table(header, align, rows, caption, label):
    lines = [
        r"\begin{table*}[ht]",
        r"\centering",
        r"\caption{%s}" % caption,
        r"\label{%s}" % label,
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{%s}" % align,
        r"\toprule",
        " & ".join(header) + r" \\",
        r"\midrule",
    ]
    lines.extend(" & ".join(row) + r" \\" for row in rows)
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table*}",
    ])
    return "\n".join(lines)


def build_manuscript_tables(config, logger):
    """Create tables only after all formal result files are present."""

    reports_dir = Path(config["paths"]["reports_dir"])
    required = [
        reports_dir / "resource_manifest.json",
        reports_dir / "model_training_summary.csv",
        reports_dir / "classification_summary.csv",
        reports_dir / "multimodal_baseline_summary.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Manuscript tables require completed formal reports: %s"
            % ", ".join(missing)
        )

    with required[0].open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    training = pd.read_csv(required[1])
    classification = pd.read_csv(required[2])
    multimodal = pd.read_csv(required[3])
    combined = pd.concat([classification, multimodal], ignore_index=True)

    tables = []
    label_rows = []
    for task, record in manifest["benchmark"]["label_tasks"].items():
        label_rows.append([
            _escape(task),
            _escape(record["label_column"]),
            str(record["retained_sample_count"]),
            str(record["retained_class_count"]),
            str(record["minimum_class_count"]),
        ])
    tables.append(_table(
        ["Task", "Label field", "Drugs", "Classes", "Minimum class count"],
        "lcccc",
        label_rows,
        "Drug-level single-label prediction tasks in the complete benchmark.",
        "tab:task-definitions",
    ))

    view_rows = []
    for view, record in sorted(manifest["graph_views"].items()):
        view_rows.append([
            _escape(view),
            _escape(", ".join(record["groups"])),
            str(record["triple_count"]),
            str(record["entity_count"]),
            str(record["relation_count"]),
            str(record["benchmark_drug_count"]),
        ])
    tables.append(_table(
        ["View", "Relation groups", "Triples", "Entities", "Relations", "Drugs"],
        "llrrrr",
        view_rows,
        "Label-free graph views. The MoA outcome relation is absent from every view.",
        "tab:graph-views",
    ))

    full_training = training[
        (training["view"] == "full_label_free")
        & (training["status"] == "success")
    ].sort_values("mrr", ascending=False)
    training_rows = [
        [
            _escape(row.model),
            "%.3f" % row.mrr,
            "%.3f" % row.hits_at_1,
            "%.3f" % row.hits_at_3,
            "%.3f" % row.hits_at_10,
        ]
        for row in full_training.itertuples(index=False)
    ]
    tables.append(_table(
        ["Embedding model", "MRR", "Hits@1", "Hits@3", "Hits@10"],
        "lrrrr",
        training_rows,
        "Filtered link-prediction performance on the full label-free graph view.",
        "tab:link-prediction",
    ))

    primary = classification[
        (classification["task"] == "detailed_moa_29")
        & (classification["cohort"] == "graph_shared_380")
        & (classification["view"] == "full_label_free")
        & (classification["status"] == "success")
    ]
    primary_best = _best_rows(primary, ["model"])
    primary_rows = []
    for row in primary_best.itertuples(index=False):
        primary_rows.append([
            _escape(row.model),
            _escape(row.classifier),
            str(int(row.sample_count)),
            str(int(row.moa_count)),
            _score(row._asdict(), "balanced_accuracy"),
            _score(row._asdict(), "f1_macro"),
            _score(row._asdict(), "auc_ovr_macro"),
        ])
    tables.append(_table(
        ["Embedding", "Classifier", "$n$", "Classes", "Balanced accuracy",
         "Macro-F1", "Macro-AUC"],
        "llrrrrr",
        primary_rows,
        "Primary detailed-MoA task on the graph-shared cohort (24 classes "
        "retained after coverage and frequency filtering). Values are mean "
        "metrics across five repeated out-of-fold evaluations; the best "
        "classifier per embedding is shown.",
        "tab:primary-kg-results",
    ))

    ablation = classification[
        (classification["task"] == "detailed_moa_29")
        & (classification["cohort"] == "graph_shared_380")
        & (classification["status"] == "success")
        & (classification["view"] != "baseline")
    ]
    ablation_best = _best_rows(ablation, ["view"])
    ablation_rows = []
    for row in ablation_best.itertuples(index=False):
        ablation_rows.append([
            _escape(row.view),
            _escape(row.model),
            _escape(row.classifier),
            str(int(row.sample_count)),
            str(int(row.moa_count)),
            _score(row._asdict(), "f1_macro"),
        ])
    tables.append(_table(
        ["Graph view", "Embedding", "Classifier", "$n$", "Classes", "Macro-F1"],
        "lllrrr",
        ablation_rows,
        "Relation-source ablation on the graph-shared detailed-MoA task "
        "(24 retained classes). The best post-hoc combination within each "
        "view is shown.",
        "tab:relation-ablation",
    ))

    matched = combined[
        (combined["cohort"] == "shared_kg_drugbank")
        & (combined["status"] == "success")
    ].copy()
    selected_rows = []
    for task, task_frame in matched.groupby("task"):
        families = [
            ("Full label-free KG", task_frame[task_frame["view"] == "full_label_free"]),
            ("Target multi-hot", task_frame[task_frame["model"] == "TargetMultiHot"]),
            ("DruGNNosis (adapted)", task_frame[task_frame["model"].isin([
                "DruGNNosisStyle",
                "DrugNNosisStyle",
            ])]),
            ("GraphDTA (adapted)", task_frame[task_frame["model"] == "GraphDTAStyle"]),
            ("DeepDTA (adapted)", task_frame[task_frame["model"] == "DeepDTAStyle"]),
        ]
        for family, candidates in families:
            if candidates.empty:
                continue
            row = candidates.sort_values("f1_macro_mean", ascending=False).iloc[0]
            selected_rows.append([
                _escape(task),
                _escape(family),
                _escape(row["model"]),
                _escape(row["classifier"]),
                str(int(row["sample_count"])),
                str(int(row["moa_count"])),
                _score(row, "f1_macro"),
                _score(row, "auc_ovr_macro"),
            ])
    tables.append(_table(
        ["Task", "Input family", "Model", "Classifier", "$n$", "Classes",
         "Macro-F1", "Macro-AUC"],
        "llllrrrr",
        selected_rows,
        "Task-matched comparison on drugs shared by the graph and DrugBank "
        "structure/sequence inputs. Every row within a task uses identical drugs, "
        "labels, outer folds, and seeds. DeepDTA, GraphDTA, and DruGNNosis "
        "baselines were adapted to drug-level MoA prediction.",
        "tab:matched-comparison",
    ))

    output_dir = ensure_dir(reports_dir / "manuscript")
    output_path = output_dir / "formal_tables.tex"
    output_path.write_text("\n\n".join(tables) + "\n", encoding="utf-8")

    digest = {
        "primary_task": "detailed_moa_29",
        "secondary_task": "superclass_4",
        "tables": [
            "task-definitions",
            "graph-views",
            "link-prediction",
            "primary-kg-results",
            "relation-ablation",
            "matched-comparison",
        ],
        "selection_warning": (
            "Best-within-family rows are post-hoc descriptive selections. "
            "Paired bootstrap reports, when generated, are retained as "
            "internal sensitivity analyses."
        ),
    }
    with (output_dir / "result_digest.json").open("w", encoding="utf-8") as handle:
        json.dump(digest, handle, indent=2)
    logger.info("Wrote manuscript tables to %s", output_path)
    return output_path
