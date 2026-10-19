"""Assemble manuscript-ready task-matched result tables from two formal runs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from .config import load_config
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


def _best_rows(frame, grouping, metric="f1_macro_mean"):
    if frame.empty:
        return frame
    return (
        frame.sort_values(
            grouping + [metric],
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


def _read_csv(path):
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def _read_json(path):
    path = Path(path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _reports(config):
    return Path(config["paths"]["reports_dir"])


def _table_link_prediction(training, label, caption):
    rows = []
    successful = training[training["status"] == "success"].sort_values(
        "mrr",
        ascending=False,
    )
    for row in successful.itertuples(index=False):
        rows.append([
            _escape(row.model),
            "%.3f" % row.mrr,
            "%.3f" % row.hits_at_1,
            "%.3f" % row.hits_at_3,
            "%.3f" % row.hits_at_10,
        ])
    return _table(
        ["Embedding model", "MRR", "Hits@1", "Hits@3", "Hits@10"],
        "lrrrr",
        rows,
        caption,
        label,
    )


def _table_best_classifier(classification, label, caption, task, cohort, view):
    frame = classification[
        (classification["task"] == task)
        & (classification["cohort"] == cohort)
        & (classification["view"] == view)
        & (classification["status"] == "success")
    ]
    best = _best_rows(frame, ["model"])
    rows = []
    for row in best.itertuples(index=False):
        data = row._asdict()
        rows.append([
            _escape(row.model),
            _escape(row.classifier),
            str(int(row.sample_count)),
            str(int(row.moa_count)),
            _score(data, "balanced_accuracy"),
            _score(data, "f1_macro"),
            _score(data, "auc_ovr_macro"),
        ])
    return _table(
        ["Embedding", "Classifier", "$n$", "Classes", "Balanced accuracy", "Macro-F1", "Macro-AUC"],
        "llrrrrr",
        rows,
        caption,
        label,
    )


def _table_ablation(classification):
    frame = classification[
        (classification["task"] == "superclass_4")
        & (classification["cohort"] == "cpg0004_graph_shared")
        & (classification["status"] == "success")
        & (classification["view"] != "baseline")
    ]
    best = _best_rows(frame, ["view"])
    rows = []
    for row in best.itertuples(index=False):
        data = row._asdict()
        rows.append([
            _escape(row.view),
            _escape(row.model),
            _escape(row.classifier),
            str(int(row.sample_count)),
            _score(data, "f1_macro"),
            _score(data, "auc_ovr_macro"),
        ])
    return _table(
        ["Graph view", "Embedding", "Classifier", "$n$", "Macro-F1", "Macro-AUC"],
        "lllrrr",
        rows,
        "Superclass-only relation-source ablation on cpg0004 graph-shared drugs. "
        "The best post-hoc combination within each graph view is shown.",
        "tab:drugmap-style-ablation",
    )


def _table_cpg_comparison(classification, multimodal):
    combined = pd.concat([classification, multimodal], ignore_index=True)
    frame = combined[
        (combined["task"] == "superclass_4")
        & (combined["cohort"] == "cpg0004_drugbank_matched")
        & (combined["status"] == "success")
    ].copy()
    families = [
        ("Full label-free KG", frame[frame["view"] == "full_label_free"]),
        ("DruGNNosis (adapted)", frame[frame["model"].isin([
            "DruGNNosisStyle",
            "DrugNNosisStyle",
        ])]),
        ("GraphDTA (adapted)", frame[frame["model"] == "GraphDTAStyle"]),
        ("DeepDTA (adapted)", frame[frame["model"] == "DeepDTAStyle"]),
    ]
    rows = []
    for family, candidates in families:
        if candidates.empty:
            continue
        row = candidates.sort_values("f1_macro_mean", ascending=False).iloc[0]
        rows.append([
            _escape(family),
            _escape(row["model"]),
            _escape(row["classifier"]),
            str(int(row["sample_count"])),
            str(int(row["moa_count"])),
            _score(row, "f1_macro"),
            _score(row, "auc_ovr_macro"),
        ])
    return _table(
        ["Input family", "Model", "Classifier", "$n$", "Classes", "Macro-F1", "Macro-AUC"],
        "lllrrrr",
        rows,
        "Task-matched cpg0004 superclass comparison on drugs shared by the graph "
        "and DrugBank structure/sequence/text inputs. DeepDTA, GraphDTA, "
        "and DruGNNosis baselines were adapted to drug-level MoA prediction "
        "and trained on the same outer folds.",
        "tab:drugmap-style-cpg-comparison",
    )


def _markdown_table(rows, header):
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    return lines


def _best_summary_rows(frame, filters, family_column="model"):
    if frame.empty:
        return []
    selected = frame.copy()
    for key, value in filters.items():
        selected = selected[selected[key] == value]
    selected = selected[selected["status"] == "success"]
    if selected.empty:
        return []
    best = _best_rows(selected, [family_column])
    rows = []
    for row in best.itertuples(index=False):
        rows.append([
            getattr(row, family_column),
            row.classifier,
            int(row.sample_count),
            int(row.moa_count),
            "%.3f" % row.f1_macro_mean,
            "%.3f" % row.auc_ovr_macro_mean,
        ])
    return rows


def build_drugmap_style_artifacts(big_config, cpg_config, output_dir):
    big_reports = _reports(big_config)
    cpg_reports = _reports(cpg_config)

    big_training = _read_csv(big_reports / "model_training_summary.csv")
    big_classification = _read_csv(big_reports / "classification_summary.csv")
    big_manifest = _read_json(big_reports / "resource_manifest.json")

    cpg_training = _read_csv(cpg_reports / "model_training_summary.csv")
    cpg_classification = _read_csv(cpg_reports / "classification_summary.csv")
    cpg_multimodal = _read_csv(cpg_reports / "multimodal_baseline_summary.csv")
    cpg_pairs = _read_csv(cpg_reports / "paired_bootstrap_comparisons.csv")
    cpg_manifest = _read_json(cpg_reports / "resource_manifest.json")

    output_dir = ensure_dir(Path(output_dir))

    tables = [
        _table_link_prediction(
            big_training,
            "tab:drugmap-style-bigkg-link",
            "Filtered link-prediction performance on the constructed label-free "
            "Repurposing Hub/PrimeKG graph after excluding all cpg0004 drugs.",
        ),
        _table_best_classifier(
            big_classification,
            "tab:drugmap-style-bigkg-classifier",
            "MoA-suffix/action-class prediction on the constructed label-free big-KG cohort "
            "after excluding cpg0004 drugs. The best classifier per embedding is shown.",
            "moa_suffix_25",
            "bigkg_graph_shared",
            "full_label_free",
        ),
        _table_link_prediction(
            cpg_training[cpg_training["view"] == "full_label_free"],
            "tab:drugmap-style-cpg-link",
            "Filtered link-prediction performance on the cpg0004 full label-free graph.",
        ),
        _table_best_classifier(
            cpg_classification,
            "tab:drugmap-style-cpg-superclass",
            "cpg0004 superclass prediction on graph-shared drugs. The best classifier "
            "per embedding is shown.",
            "superclass_4",
            "cpg0004_graph_shared",
            "full_label_free",
        ),
        _table_ablation(cpg_classification),
        _table_cpg_comparison(cpg_classification, cpg_multimodal),
        _table_best_classifier(
            cpg_classification,
            "tab:drugmap-style-cpg-detailed",
            "cpg0004 detailed-MoA prediction on graph-shared drugs. This table is "
            "reported as a fine-grained analysis rather than the main method comparison.",
            "detailed_moa_29",
            "cpg0004_graph_shared",
            "full_label_free",
        ),
    ]
    (output_dir / "drugmap_style_tables.tex").write_text(
        "\n\n".join(tables) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Task-matched KGML experiment summary",
        "",
        "## Cohorts",
        "",
        "- Big KG run: `%s`" % big_config["_meta"]["config_path"],
        "- cpg0004 run: `%s`" % cpg_config["_meta"]["config_path"],
        "- TargetMultiHot is intentionally excluded from the formal comparison.",
        "",
    ]
    if big_manifest:
        scope = big_manifest.get("benchmark", {})
        lines.extend([
            "### Big KG benchmark",
            "",
            "- Label tasks: `%s`" % ", ".join(scope.get("label_tasks", {}).keys()),
            "- Benchmark drugs: `%s`" % scope.get("drug_count", "NA"),
            "",
        ])
    if cpg_manifest:
        scope = cpg_manifest.get("benchmark", {})
        lines.extend([
            "### cpg0004 benchmark",
            "",
            "- Label tasks: `%s`" % ", ".join(scope.get("label_tasks", {}).keys()),
            "- Benchmark drugs: `%s`" % scope.get("drug_count", "NA"),
            "",
        ])

    lines.extend([
        "## Big KG: superclass KGEM + classifier",
        "",
    ])
    lines.extend(_markdown_table(
        _best_summary_rows(
            big_classification,
            {
                "task": "moa_suffix_25",
                "cohort": "bigkg_graph_shared",
                "view": "full_label_free",
            },
        ),
        ["Embedding", "Classifier", "n", "Classes", "Macro-F1", "Macro-AUC"],
    ))
    lines.extend([
        "",
        "## cpg0004: superclass KGEM + classifier",
        "",
    ])
    lines.extend(_markdown_table(
        _best_summary_rows(
            cpg_classification,
            {
                "task": "superclass_4",
                "cohort": "cpg0004_graph_shared",
                "view": "full_label_free",
            },
        ),
        ["Embedding", "Classifier", "n", "Classes", "Macro-F1", "Macro-AUC"],
    ))
    lines.extend([
        "",
        "## cpg0004: matched superclass comparison",
        "",
    ])
    comparison_rows = []
    combined = pd.concat([cpg_classification, cpg_multimodal], ignore_index=True)
    matched = combined[
        (combined["task"] == "superclass_4")
        & (combined["cohort"] == "cpg0004_drugbank_matched")
        & (combined["status"] == "success")
    ]
    for family, candidates in [
        ("Full label-free KG", matched[matched["view"] == "full_label_free"]),
        ("DruGNNosis (adapted)", matched[matched["model"].isin([
            "DruGNNosisStyle",
            "DrugNNosisStyle",
        ])]),
        ("GraphDTA (adapted)", matched[matched["model"] == "GraphDTAStyle"]),
        ("DeepDTA (adapted)", matched[matched["model"] == "DeepDTAStyle"]),
    ]:
        if candidates.empty:
            continue
        row = candidates.sort_values("f1_macro_mean", ascending=False).iloc[0]
        comparison_rows.append([
            family,
            row["model"],
            row["classifier"],
            int(row["sample_count"]),
            "%.3f" % row["f1_macro_mean"],
            "%.3f" % row["auc_ovr_macro_mean"],
        ])
    lines.extend(_markdown_table(
        comparison_rows,
        ["Family", "Model", "Classifier", "n", "Macro-F1", "Macro-AUC"],
    ))
    lines.extend([
        "",
        "## Interpretation guardrails",
        "",
        "- The cpg0004 main external-method comparison is the four-superclass task.",
        "- Detailed MoA results are retained for fine-grained analysis and KAG discussion.",
        "- Ablation is interpreted on superclass only.",
        "- Because TargetMultiHot is excluded from the formal table, claims should not state "
        "that heterogeneous relations outperform all target-only information; ablation "
        "should be used to qualify that point.",
        "",
    ])

    (output_dir / "drugmap_style_summary.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    digest = {
        "big_reports": str(big_reports),
        "cpg_reports": str(cpg_reports),
        "outputs": {
            "markdown": str(output_dir / "drugmap_style_summary.md"),
            "latex": str(output_dir / "drugmap_style_tables.tex"),
        },
    }
    with (output_dir / "drugmap_style_digest.json").open("w", encoding="utf-8") as handle:
        json.dump(digest, handle, indent=2)
    return digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--big-config", required=True)
    parser.add_argument("--cpg-config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    digest = build_drugmap_style_artifacts(
        load_config(args.big_config),
        load_config(args.cpg_config),
        args.output_dir,
    )
    print(json.dumps(digest, indent=2))


if __name__ == "__main__":
    main()
