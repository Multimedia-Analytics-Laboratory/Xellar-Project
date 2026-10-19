"""Console entrypoint for the KGML pipeline package."""

from __future__ import annotations

import argparse

from .config import load_config
from .logging_utils import configure_logging
from .orchestrator import (
    run_all,
    run_audit_leakage,
    run_build_triples,
    run_build_resource_manifest,
    run_build_manuscript_tables,
    run_build_kag_evidence,
    run_build_pair_action_ground_truth,
    run_evaluate_classifiers,
    run_evaluate_action_pairs,
    run_evaluate_action_reliability,
    run_evaluate_locked_multiview_external,
    run_evaluate_relation_routing,
    run_evaluate_pair_multimodal,
    run_evaluate_multimodal,
    run_prepare_benchmark,
    run_prepare_multimodal,
    run_interpret_embeddings,
    run_posthoc_analysis,
    run_select_locked_multiview,
    run_summarize_results,
    run_train_embeddings,
    run_validate,
    run_strengthen_manuscript,
)


def build_parser():
    parser = argparse.ArgumentParser(description="Run the KGML manuscript pipeline.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a JSON config file.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate", help="Validate config and input paths.")
    validate_parser.add_argument(
        "--skip-input-checks",
        action="store_true",
        help="Validate config structure without checking whether input files exist.",
    )

    subparsers.add_parser("prepare-benchmark", help="Build cleaned benchmark tables.")
    subparsers.add_parser("build-triples", help="Build grouped and view-specific triple tables.")
    subparsers.add_parser(
        "build-resource-manifest",
        help="Record input provenance, hashes, labels, and graph-view statistics.",
    )
    subparsers.add_parser(
        "prepare-multimodal",
        help="Map benchmark drugs to DrugBank SMILES and target sequences without action labels.",
    )
    subparsers.add_parser(
        "evaluate-multimodal",
        help=(
            "Run leakage-controlled adapted DeepDTA/GraphDTA/DruGNNosis "
            "drug-level baselines."
        ),
    )
    subparsers.add_parser(
        "audit-leakage",
        help="Measure downstream test drugs whose MoA labels entered KG training.",
    )
    subparsers.add_parser("train-embeddings", help="Train PyKEEN models and save embeddings.")
    subparsers.add_parser("evaluate-classifiers", help="Run sklearn classifiers on saved embeddings.")
    subparsers.add_parser(
        "evaluate-action-pairs",
        help="Run drug-target action relation classifiers on saved embeddings.",
    )
    subparsers.add_parser(
        "evaluate-relation-routing",
        help=(
            "Run nested-OOF pair-conditioned routing across relation-view "
            "experts and optionally rerank grounded KAG evidence."
        ),
    )
    subparsers.add_parser(
        "select-locked-multiview",
        help=(
            "Evaluate multi-view strategies only on the development cohort "
            "and write a content-addressed strategy lock."
        ),
    )
    subparsers.add_parser(
        "evaluate-locked-multiview-external",
        help=(
            "Verify the development strategy lock, then evaluate the unchanged "
            "strategy on the external cohort."
        ),
    )
    subparsers.add_parser(
        "evaluate-pair-multimodal",
        help=(
            "Run pair-level adapted DeepDTA/GraphDTA/DruGNNosis "
            "drug-target action baselines."
        ),
    )
    subparsers.add_parser(
        "evaluate-action-reliability",
        help=(
            "Evaluate hierarchical action errors, development-fitted calibration, "
            "selective prediction, relation-view faithfulness, and independent evidence."
        ),
    )
    subparsers.add_parser(
        "strengthen-manuscript",
        help=(
            "Run post-lock paired method comparisons, graph-context sparsity "
            "analysis, and DistMult diagnostics."
        ),
    )
    subparsers.add_parser(
        "build-kag-evidence",
        help=(
            "Export a direct-edge-masked OpenSPG KAG graph, pair questions, "
            "typed evidence paths, and auditable prompt records."
        ),
    )
    subparsers.add_parser(
        "build-pair-action-ground-truth",
        help=(
            "Rebuild pair-specific action labels from DrugBank and Open Targets, "
            "with Repurposing Hub/PrimeKG support and explicit conflict auditing."
        ),
    )
    subparsers.add_parser(
        "interpret-embeddings",
        help=(
            "Quantify original-space drug embedding distances, relation-context "
            "differences, graph-view attribution, and grounded KAG case records."
        ),
    )
    subparsers.add_parser(
        "posthoc-analysis",
        help="Build per-class, confusion, stratified-error, and paired-bootstrap reports.",
    )
    subparsers.add_parser(
        "build-manuscript-tables",
        help="Generate LaTeX tables from completed formal reports.",
    )
    subparsers.add_parser("summarize-results", help="Generate final CSV and markdown summaries.")
    subparsers.add_parser("run-all", help="Run the full pipeline.")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    config = load_config(args.config)
    logger = configure_logging(config["paths"].get("logs_dir"), args.command)
    logger.info("Loaded config from %s", config["_meta"]["config_path"])

    if args.command == "validate":
        run_validate(config, logger, check_inputs=not args.skip_input_checks)
    elif args.command == "prepare-benchmark":
        run_prepare_benchmark(config, logger)
    elif args.command == "build-triples":
        run_build_triples(config, logger)
    elif args.command == "build-resource-manifest":
        run_build_resource_manifest(config, logger)
    elif args.command == "prepare-multimodal":
        run_prepare_multimodal(config, logger)
    elif args.command == "evaluate-multimodal":
        run_evaluate_multimodal(config, logger)
    elif args.command == "audit-leakage":
        run_audit_leakage(config, logger)
    elif args.command == "train-embeddings":
        run_train_embeddings(config, logger)
    elif args.command == "evaluate-classifiers":
        run_evaluate_classifiers(config, logger)
    elif args.command == "evaluate-action-pairs":
        run_evaluate_action_pairs(config, logger)
    elif args.command == "evaluate-relation-routing":
        run_evaluate_relation_routing(config, logger)
    elif args.command == "select-locked-multiview":
        run_select_locked_multiview(config, logger)
    elif args.command == "evaluate-locked-multiview-external":
        run_evaluate_locked_multiview_external(config, logger)
    elif args.command == "evaluate-pair-multimodal":
        run_evaluate_pair_multimodal(config, logger)
    elif args.command == "evaluate-action-reliability":
        run_evaluate_action_reliability(config, logger)
    elif args.command == "strengthen-manuscript":
        run_strengthen_manuscript(config, logger)
    elif args.command == "build-kag-evidence":
        run_build_kag_evidence(config, logger)
    elif args.command == "build-pair-action-ground-truth":
        run_build_pair_action_ground_truth(config, logger)
    elif args.command == "interpret-embeddings":
        run_interpret_embeddings(config, logger)
    elif args.command == "posthoc-analysis":
        run_posthoc_analysis(config, logger)
    elif args.command == "build-manuscript-tables":
        run_build_manuscript_tables(config, logger)
    elif args.command == "summarize-results":
        run_summarize_results(config, logger)
    elif args.command == "run-all":
        run_all(config, logger)
    else:
        parser.error("Unsupported command: %s" % args.command)


if __name__ == "__main__":
    main()
