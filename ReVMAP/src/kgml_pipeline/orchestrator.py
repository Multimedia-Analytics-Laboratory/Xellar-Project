"""Stage orchestrator for the KGML pipeline."""

from __future__ import annotations

from .config import validate_config


def run_validate(config, logger, check_inputs=True):
    report = validate_config(config, check_inputs=check_inputs)
    for warning in report["warnings"]:
        logger.warning(warning)
    if report["errors"]:
        for error in report["errors"]:
            logger.error(error)
        raise SystemExit(1)
    logger.info("Configuration validated successfully.")
    return report


def run_prepare_benchmark(config, logger):
    from .data_prep import prepare_benchmark_tables

    return prepare_benchmark_tables(config, logger)


def run_build_triples(config, logger):
    from .triples import build_triple_groups

    return build_triple_groups(config, logger)


def run_prepare_multimodal(config, logger):
    from .multimodal_data import build_drugbank_multimodal_table

    return build_drugbank_multimodal_table(config, logger)


def run_evaluate_multimodal(config, logger):
    from .multimodal_evaluation import evaluate_multimodal_baselines

    return evaluate_multimodal_baselines(config, logger)


def run_evaluate_action_pairs(config, logger):
    from .pair_evaluation import evaluate_action_pairs

    return evaluate_action_pairs(config, logger)


def run_evaluate_relation_routing(config, logger):
    from .relation_routing import evaluate_relation_routing

    return evaluate_relation_routing(config, logger)


def run_select_locked_multiview(config, logger):
    from .locked_multiview_evaluation import (
        run_development_multiview_selection,
    )

    return run_development_multiview_selection(config, logger)


def run_evaluate_locked_multiview_external(config, logger):
    from .locked_multiview_evaluation import (
        run_external_locked_multiview_evaluation,
    )

    return run_external_locked_multiview_evaluation(config, logger)


def run_evaluate_pair_multimodal(config, logger):
    from .pair_multimodal_evaluation import evaluate_pair_multimodal_baselines

    return evaluate_pair_multimodal_baselines(config, logger)


def run_evaluate_action_reliability(config, logger):
    from .action_reliability import run_action_reliability

    return run_action_reliability(config, logger)


def run_strengthen_manuscript(config, logger):
    from .manuscript_strengthening import run_manuscript_strengthening

    return run_manuscript_strengthening(config, logger)


def run_build_kag_evidence(config, logger):
    from .openspg_kag_adapter import build_openspg_kag_evidence

    return build_openspg_kag_evidence(config, logger)


def run_build_pair_action_ground_truth(config, logger):
    from .pair_action_ground_truth import build_pair_action_ground_truth

    return build_pair_action_ground_truth(config, logger)


def run_interpret_embeddings(config, logger):
    from .embedding_interpretation import run_embedding_interpretation

    return run_embedding_interpretation(config, logger)


def run_audit_leakage(config, logger):
    from .leakage_audit import audit_label_exposure

    return audit_label_exposure(config, logger)


def run_train_embeddings(config, logger):
    from .pykeen_pipeline import train_all_views
    from .reporting import write_training_summary

    training_df = train_all_views(config, logger)
    return write_training_summary(training_df, config, logger)


def run_evaluate_classifiers(config, logger):
    from .reporting import write_classification_summary

    protocol = str(config.get("evaluation", {}).get("protocol", "legacy_holdout"))
    if protocol == "repeated_stratified_cv":
        from .strict_evaluation import evaluate_all_views_strict

        classification_df = evaluate_all_views_strict(config, logger)
    else:
        from .evaluation import evaluate_all_views

        classification_df = evaluate_all_views(config, logger)
    return write_classification_summary(classification_df, config, logger)


def run_summarize_results(config, logger):
    from .reporting import generate_pipeline_summary

    return generate_pipeline_summary(config, logger)


def run_build_resource_manifest(config, logger):
    from .resource_manifest import build_resource_manifest

    return build_resource_manifest(config, logger)


def run_posthoc_analysis(config, logger):
    from .posthoc_analysis import run_posthoc_analysis as analyze

    return analyze(config, logger)


def run_build_manuscript_tables(config, logger):
    from .manuscript_artifacts import build_manuscript_tables

    return build_manuscript_tables(config, logger)


def run_all(config, logger):
    run_validate(config, logger, check_inputs=True)
    run_prepare_benchmark(config, logger)
    run_build_triples(config, logger)
    run_build_resource_manifest(config, logger)
    run_train_embeddings(config, logger)
    if config.get("multimodal"):
        run_prepare_multimodal(config, logger)
    run_evaluate_classifiers(config, logger)
    if config.get("multimodal"):
        run_evaluate_multimodal(config, logger)
    run_posthoc_analysis(config, logger)
    run_build_manuscript_tables(config, logger)
    return run_summarize_results(config, logger)
