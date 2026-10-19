import json

import pandas as pd

from kgml_pipeline.manuscript_artifacts import build_manuscript_tables


class _Logger:
    def info(self, *args, **kwargs):
        return None


def _result_row(**overrides):
    row = {
        "task": "detailed_moa_29",
        "cohort": "graph_shared_380",
        "view": "full_label_free",
        "model": "RGCN",
        "classifier": "Random Forest",
        "sample_count": 10,
        "moa_count": 2,
        "status": "success",
    }
    for metric in (
        "accuracy",
        "balanced_accuracy",
        "f1_macro",
        "f1_weighted",
        "precision_macro",
        "recall_macro",
        "auc_ovr_macro",
    ):
        row["%s_mean" % metric] = 0.8
        row["%s_std" % metric] = 0.02
        row["%s_ci95" % metric] = 0.01
    row.update(overrides)
    return row


def test_build_manuscript_tables(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    manifest = {
        "benchmark": {
            "label_tasks": {
                "detailed_moa_29": {
                    "label_column": "tail",
                    "retained_sample_count": 433,
                    "retained_class_count": 29,
                    "minimum_class_count": 8,
                }
            }
        },
        "graph_views": {
            "full_label_free": {
                "groups": ["target", "indication"],
                "triple_count": 20,
                "entity_count": 15,
                "relation_count": 2,
                "benchmark_drug_count": 10,
            }
        },
    }
    (reports / "resource_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    pd.DataFrame([
        {
            "view": "full_label_free",
            "model": "RGCN",
            "status": "success",
            "mrr": 0.2,
            "hits_at_1": 0.1,
            "hits_at_3": 0.3,
            "hits_at_10": 0.5,
        }
    ]).to_csv(reports / "model_training_summary.csv", index=False)
    pd.DataFrame([
        _result_row(),
        _result_row(
            view="targets_only_label_free",
            model="TransE",
            f1_macro_mean=0.7,
        ),
        _result_row(
            cohort="shared_kg_drugbank",
            sample_count=8,
        ),
        _result_row(
            cohort="shared_kg_drugbank",
            view="baseline",
            model="TargetMultiHot",
            classifier="Logistic Regression",
            sample_count=8,
            f1_macro_mean=0.6,
        ),
    ]).to_csv(reports / "classification_summary.csv", index=False)
    pd.DataFrame([
        _result_row(
            cohort="shared_kg_drugbank",
            view="multimodal_baseline",
            model="DeepDTAStyle",
            classifier="DeepDTAStyle",
            sample_count=8,
            f1_macro_mean=0.5,
        )
    ]).to_csv(reports / "multimodal_baseline_summary.csv", index=False)

    path = build_manuscript_tables(
        {"paths": {"reports_dir": str(reports)}},
        _Logger(),
    )
    text = path.read_text(encoding="utf-8")
    assert "tab:task-definitions" in text
    assert "tab:matched-comparison" in text
    assert "RGCN" in text
