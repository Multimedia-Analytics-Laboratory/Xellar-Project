"""Summary writers for training and classifier outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_training_summary(training_df, config, logger):
    reports_dir = Path(config["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    output_path = reports_dir / "model_training_summary.csv"
    training_df.to_csv(output_path, index=False)
    logger.info("Wrote training summary to %s", output_path)
    return output_path


def write_classification_summary(classification_df, config, logger):
    reports_dir = Path(config["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    output_path = reports_dir / "classification_summary.csv"
    classification_df.to_csv(output_path, index=False)
    logger.info("Wrote classification summary to %s", output_path)
    return output_path


def generate_pipeline_summary(config, logger):
    reports_dir = Path(config["paths"]["reports_dir"])
    training_path = reports_dir / "model_training_summary.csv"
    classification_path = reports_dir / "classification_summary.csv"
    summary_path = reports_dir / "pipeline_summary.md"

    lines = [
        "# KGML Pipeline Summary",
        "",
        "## Inputs",
        "",
        "- Config: `%s`" % config["_meta"]["config_path"],
        "- Benchmark metadata: `%s`" % config["paths"]["combined_metadata_csv"],
        "- Repurposing Hub: `%s`" % config["paths"]["repurposing_hub_tsv"],
        "- PrimeKG: `%s`" % config["paths"]["primekg_csv"],
        "",
    ]

    if training_path.exists() and training_path.stat().st_size > 0:
        try:
            training_df = pd.read_csv(training_path)
        except Exception:
            training_df = pd.DataFrame()
        lines.extend([
            "## Training Summary",
            "",
            "Rows: `%d`" % len(training_df),
            "",
        ])
        if not training_df.empty:
            successful = training_df[training_df["status"] == "success"].copy()
            if not successful.empty:
                best_by_view = successful.sort_values(["view", "mrr"], ascending=[True, False]).groupby("view").head(1)
                lines.append("| View | Best model by MRR | MRR | Hits@10 |")
                lines.append("| --- | --- | ---: | ---: |")
                for _, row in best_by_view.iterrows():
                    lines.append(
                        "| %s | %s | %.4f | %.4f |"
                        % (row["view"], row["model"], row["mrr"], row["hits_at_10"])
                    )
                lines.append("")

    if classification_path.exists() and classification_path.stat().st_size > 0:
        try:
            classification_df = pd.read_csv(classification_path)
        except Exception:
            classification_df = pd.DataFrame()
        lines.extend([
            "## Classification Summary",
            "",
            "Rows: `%d`" % len(classification_df),
            "",
        ])
        if not classification_df.empty:
            score_column = "best_f1_macro"
            classifier_column = "best_classifier_by_f1_macro"
            if score_column not in classification_df.columns:
                score_column = "best_test_f1_weighted"
                classifier_column = "best_classifier"
            grouping_columns = ["view"]
            sort_columns = ["view", score_column]
            ascending = [True, False]
            has_tasks = "task" in classification_df.columns
            has_cohorts = "cohort" in classification_df.columns
            if has_tasks:
                grouping_columns = ["task", "view"]
                sort_columns = ["task", "view", score_column]
                ascending = [True, True, False]
                if has_cohorts:
                    grouping_columns = ["task", "cohort", "view"]
                    sort_columns = ["task", "cohort", "view", score_column]
                    ascending = [True, True, True, False]
            best_by_view = classification_df.sort_values(
                sort_columns, ascending=ascending
            ).groupby(grouping_columns).head(1)
            if has_tasks and has_cohorts:
                lines.append("| Task | Cohort | View | Best model | Classifier | F1 | Accuracy |")
                lines.append("| --- | --- | --- | --- | --- | ---: | ---: |")
            elif has_tasks:
                lines.append("| Task | View | Best model | Classifier | F1 | Accuracy |")
                lines.append("| --- | --- | --- | --- | ---: | ---: |")
            else:
                lines.append("| View | Best model | Classifier | F1 | Accuracy |")
                lines.append("| --- | --- | --- | ---: | ---: |")
            for _, row in best_by_view.iterrows():
                values = (
                    row["view"],
                    row["model"],
                    row.get(classifier_column),
                    row[score_column],
                    row["best_accuracy"] if "best_accuracy" in row else row["best_test_accuracy"],
                )
                if has_tasks and has_cohorts:
                    lines.append(
                        "| %s | %s | %s | %s | %s | %.4f | %.4f |"
                        % ((row["task"], row["cohort"]) + values)
                    )
                elif has_tasks:
                    lines.append(
                        "| %s | %s | %s | %s | %.4f | %.4f |"
                        % ((row["task"],) + values)
                    )
                else:
                    lines.append("| %s | %s | %s | %.4f | %.4f |" % values)
            lines.append("")

    summary_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote markdown pipeline summary to %s", summary_path)
    return summary_path
