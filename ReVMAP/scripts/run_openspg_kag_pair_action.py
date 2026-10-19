#!/usr/bin/env python3
"""Run exported KGML relation-path questions through the official KAG solver.

This script intentionally imports OpenSPG KAG only at runtime.  Install KAG
0.8.x in its own Python 3.10 environment and build either an OpenSPG-backed or
official memory-graph project from the masked ``nodes.json``/``edges.json``
assets produced by ``kgml-pipeline ... build-kag-evidence`` before invoking
this runner.
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import os
import time
import traceback
from pathlib import Path


def _read_jsonl(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else Path.open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _apply_kag080_memory_igraph_compat():
    """Restore the vertex attribute access used by KAG 0.8's memory loader."""
    import igraph

    if not hasattr(igraph.Vertex, "id"):
        igraph.Vertex.id = property(lambda vertex: vertex["id"])
    if not hasattr(igraph.Vertex, "label"):
        igraph.Vertex.label = property(lambda vertex: vertex["label"])


def _apply_kag080_memory_schema_compat(schema_spec):
    """Supply SchemaUtils from local config instead of an OpenSPG service."""
    if not schema_spec:
        return

    from kag.interface.solver.model.schema_utils import SchemaUtils

    namespace = schema_spec["namespace"]
    node_types = list(schema_spec["node_types"])
    relations = list(schema_spec["relations"])

    def get_schema_from_memory_spec(self):
        default_attributes = [
            '{"name": "name", "nameZh": "name"}',
            '{"name": "id", "nameZh": "id"}',
            '{"name": "description", "nameZh": "description"}',
        ]
        for node_type in node_types:
            full_name = "%s.%s" % (namespace, node_type)
            self.nodes.add(node_type)
            self.node_zh_en[node_type] = node_type
            self.node_en_zh[node_type] = node_type
            self.node_zh_2_full_name[node_type] = full_name
            self.node_en_2_full_name[node_type] = full_name
            self.full_name_2_node_en[full_name] = node_type
            self.node_attr[node_type] = self.get_attr(
                node_type, list(default_attributes)
            )

        for relation in relations:
            subject_type = relation["from_type"]
            predicate = relation["predicate"]
            object_type = relation["to_type"]
            triple_name = "%s_%s_%s" % (
                subject_type,
                predicate,
                object_type,
            )
            self.spo.add(triple_name)
            self.spo_en.add(triple_name)
            self.spo_zh_en[triple_name] = triple_name
            self.spo_en_zh[triple_name] = triple_name
            self.so_p[(subject_type, object_type)].add(predicate)
            self.sp_o[(subject_type, predicate)].add(object_type)
            self.op_s[(object_type, predicate)].add(subject_type)
            self.so_p_en[(subject_type, object_type)].add(predicate)
            self.sp_o_en[(subject_type, predicate)].add(object_type)
            self.op_s_en[(object_type, predicate)].add(subject_type)
            self.s_po_en[subject_type].add((predicate, object_type))
            self.o_ps_en[object_type].add((predicate, subject_type))
            self.edges.add(predicate)
            self.edge_zh_en[predicate] = predicate
            self.edge_en_zh[predicate] = predicate
            self.node_edge.setdefault(subject_type, set()).add(predicate)
            self.node_edge.setdefault(object_type, set()).add(predicate)
            self.edge_attr[predicate] = []

    SchemaUtils.get_schema = get_schema_from_memory_spec


def _apply_embedding_batch_limit(batch_limit):
    """Chunk compatible-API embedding calls to the endpoint's accepted size."""
    if not batch_limit:
        return
    from kag.common.vectorize_model.openai_model import OpenAIVectorizeModel

    if getattr(OpenAIVectorizeModel, "_kgml_batch_limit_patch", False):
        return
    original_vectorize = OpenAIVectorizeModel.vectorize

    def vectorize_with_limit(self, texts):
        if isinstance(texts, (list, tuple)) and len(texts) > batch_limit:
            result = []
            for start in range(0, len(texts), batch_limit):
                batch_result = original_vectorize(
                    self, list(texts[start : start + batch_limit])
                )
                if batch_result is None:
                    return None
                result.extend(batch_result)
            return result
        return original_vectorize(self, texts)

    OpenAIVectorizeModel.vectorize = vectorize_with_limit
    OpenAIVectorizeModel._kgml_batch_limit_patch = True


async def _run_one(pipeline, reporter_type, record):
    reporter = reporter_type()
    predicted_action = record.get("kgml_predicted_action")
    supplied_evidence = ""
    if "analogies" in record:
        constraints = (
            "Use only cited fold-safe analogies. Distinguish association from action "
            "direction, report 'insufficient signed mechanism evidence' when needed, "
            "and do not treat an analogy as causal proof."
        )
        # The true action is intentionally omitted from the model-visible payload.
        evidence_payload = {
            "kgml_predicted_action": predicted_action,
            "action_scores": record.get("action_scores", {}),
            "fold_safe_signed_analogies": record.get("analogies", []),
            "claim_scope": record.get("claim_scope"),
        }
        supplied_evidence = "\n\nSUPPLIED_EVIDENCE:\n%s" % json.dumps(
            evidence_payload, ensure_ascii=False, sort_keys=True
        )
    else:
        constraints = (
            "The direct drug-target edge has been withheld. Use the retrieved typed "
            "heterogeneous paths to explain why the relation is biologically plausible. "
            "State the node types and relation sequence for each path, preserve source "
            "edge identifiers or source databases when they are available in retrieved "
            "evidence, describe every interpretation as a hypothesis rather than causal "
            "proof, and do not infer inhibitor/agonist/antagonist/blocker direction."
        )
    constrained_query = "%s\n\n%s%s" % (
        record["question"],
        constraints,
        supplied_evidence,
    )
    answer = await pipeline.ainvoke(constrained_query, reporter=reporter)
    trace, status = reporter.generate_report_data()
    return {
        "query_id": record["query_id"],
        "drug_name": record["drug_name"],
        "target_name": record["target_name"],
        "kgml_predicted_action": predicted_action,
        "evidence_predicted_action": record.get("evidence_predicted_action"),
        "held_out_relation": record.get("held_out_relation"),
        "fold": record.get("fold"),
        "question": record["question"],
        "direct_edge_masked": record.get("direct_edge_masked"),
        "expected_indirect_path_count": record.get("indirect_path_count"),
        "expected_path_ids": [path.get("path_id") for path in record.get("paths", [])],
        "answer": answer,
        "status": status,
        "trace": trace.to_dict(),
    }


async def _main_async(args):
    import kag
    from kag.common.conf import KAG_CONFIG
    from kag.common.registry import import_modules_from_path
    from kag.interface import LLMClient, SolverPipelineABC
    from kag.solver.reporter.trace_log_reporter import TraceLogReporter

    # KAG 0.8 reads these two serialized igraph attributes with dot access;
    # python-igraph 0.11 exposes them through mapping access only.
    _apply_kag080_memory_igraph_compat()
    _apply_kag080_memory_schema_compat(
        KAG_CONFIG.all_config.get("memory_schema")
    )
    _apply_embedding_batch_limit(
        KAG_CONFIG.all_config.get("embedding_batch_limit")
    )

    if args.project_path:
        import_modules_from_path(str(Path(args.project_path).resolve()))
    pipeline = SolverPipelineABC.from_config(KAG_CONFIG.all_config["kag_solver_pipeline"])
    records = list(_read_jsonl(args.questions))
    if args.offset:
        records = records[args.offset :]
    if args.limit is not None:
        records = records[: args.limit]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed_query_ids = set()
    mode = "w"
    if args.resume and output_path.exists():
        completed_query_ids = {
            row.get("query_id") for row in _read_jsonl(output_path) if row.get("query_id")
        }
        mode = "a"

    started_at = time.time()
    attempted = 0
    succeeded = 0
    failed = 0
    with output_path.open(mode, encoding="utf-8") as handle:
        for record in records:
            if record.get("query_id") in completed_query_ids:
                continue
            attempted += 1
            try:
                result = await _run_one(pipeline, TraceLogReporter, record)
                result["token_usage_cumulative"] = LLMClient.get_token_meter().to_dict()
                succeeded += 1
            except Exception as exc:
                failed += 1
                result = {
                    "query_id": record.get("query_id"),
                    "drug_name": record.get("drug_name"),
                    "target_name": record.get("target_name"),
                    "question": record.get("question"),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "token_usage_cumulative": LLMClient.get_token_meter().to_dict(),
                }
            handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            if failed and not args.continue_on_error:
                raise RuntimeError("KAG question failed; see output record")

    summary = {
        "upstream": "OpenSPG/KAG",
        "kag_version": getattr(kag, "__version__", None),
        "backend": "official_memory_graph",
        "attempted": attempted,
        "succeeded": succeeded,
        "failed": failed,
        "skipped_existing": len(completed_query_ids),
        "elapsed_seconds": time.time() - started_at,
        "token_usage": LLMClient.get_token_meter().to_dict(),
        "output": str(output_path),
    }
    summary_path = Path(args.summary) if args.summary else output_path.with_suffix(
        output_path.suffix + ".summary.json"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        help="Path to kag_config.yaml. Its directory becomes the working directory "
        "before KAG is imported so relative memory-graph paths remain portable.",
    )
    parser.add_argument("--questions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary")
    parser.add_argument("--project-path")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    # KAG initializes its global configuration while importing ``kag``. Resolve
    # user paths first, then enter the project directory before that import.
    args.questions = str(Path(args.questions).resolve())
    args.output = str(Path(args.output).resolve())
    if args.summary:
        args.summary = str(Path(args.summary).resolve())
    if args.project_path:
        args.project_path = str(Path(args.project_path).resolve())
    if args.config:
        config_path = Path(args.config).resolve()
        if config_path.name != "kag_config.yaml":
            raise ValueError("--config must point to a file named kag_config.yaml")
        if not args.project_path:
            args.project_path = str(config_path.parent)
        os.chdir(str(config_path.parent))
    asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()
