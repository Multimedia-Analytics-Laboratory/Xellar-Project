import pandas as pd

from kgml_pipeline.posthoc_analysis import (
    _confusion_matrices,
    _paired_bootstrap,
    _per_class_metrics,
    _select_comparisons,
)


def _predictions():
    rows = []
    labels = ["a", "a", "a", "b", "b", "b"]
    for repeat in (0, 1):
        for predictor, predicted in (
            ("full_label_free::RGCN::SVM (RBF)", labels),
            (
                "baseline::TargetMultiHot::Logistic Regression",
                ["a", "b", "a", "b", "a", "b"],
            ),
        ):
            view, model, classifier = predictor.split("::")
            for index, (true_label, predicted_label) in enumerate(
                zip(labels, predicted)
            ):
                rows.append({
                    "task": "detailed_moa_29",
                    "cohort": "shared_kg_drugbank",
                    "source_type": "knowledge_graph",
                    "view": view,
                    "model": model,
                    "classifier": classifier,
                    "predictor": predictor,
                    "repeat": repeat,
                    "fold": index % 2,
                    "drug_name": "drug_%d" % index,
                    "true_label": true_label,
                    "predicted_label": predicted_label,
                })
    return pd.DataFrame(rows)


def test_posthoc_metrics_and_paired_bootstrap():
    predictions = _predictions()
    per_repeat, summary = _per_class_metrics(predictions)
    confusion = _confusion_matrices(predictions)
    comparisons = _select_comparisons(predictions)
    paired = _paired_bootstrap(
        predictions,
        comparisons,
        seed=42,
        iterations=50,
    )

    assert not per_repeat.empty
    assert not summary.empty
    assert confusion["count"].sum() == len(predictions)
    assert len(comparisons) == 1
    assert paired.iloc[0]["macro_f1_difference"] > 0
    assert paired.iloc[0]["shared_sample_count"] == 6
