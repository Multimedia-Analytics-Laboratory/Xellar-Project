"""KAG retriever for leakage-controlled, typed three-edge KGML paths."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict

from kag.common.tools.graph_api.graph_api_abc import GraphApiABC
from kag.interface import ChunkData, PromptABC, RetrieverABC, RetrieverOutput


def _stable_id(*parts):
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:20]


@PromptABC.register("kgml_grounded_path_summary")
class KGMLGroundedPathSummaryPrompt(PromptABC):
    template_en = """Summarize the retrieved KGML typed-path evidence for the current question.

STRICT RULES:
1. Use only the JSON records in Docs. Never add an entity, relation, edge ID, database, or biological fact from internal knowledge.
2. If Docs reports insufficient relation-specific path evidence, return that exact conclusion.
3. For every path, retain its path_id, node_sequence, type_sequence, relation_sequence, every citation edge_id, and source_database.
4. Interpret shared pathways, shared anatomy, or shared indications only as biological hypotheses. They are not causal proof and do not establish inhibitor, agonist, antagonist, or blocker direction.
5. Begin with "Answer:" and keep the response concise.

Docs:
$docs

Prior questions and answers:
$questions

Current question:
$cur_question
"""
    template_zh = template_en

    @property
    def template_variables(self):
        return ["docs", "questions", "cur_question"]

    def parse_response(self, response, **kwargs):
        return response


@RetrieverABC.register("kgml_typed_path_retriever")
class KGMLTypedPathRetriever(RetrieverABC):
    """Retrieve the three predeclared path templates from KAG's MemoryGraph."""

    def __init__(self, graph_api: GraphApiABC, top_k=20, **kwargs):
        super().__init__(top_k=top_k, **kwargs)
        self.graph_api = (
            graph_api
            if isinstance(graph_api, GraphApiABC)
            else GraphApiABC.from_config(graph_api)
        )
        self.graph = self.graph_api.graph._backend_graph
        self.vertices = {
            vertex["id"]: vertex.attributes() for vertex in self.graph.vs
        }
        self.outgoing = defaultdict(list)
        self.incoming = defaultdict(list)
        for edge in self.graph.es:
            attributes = edge.attributes()
            row = {
                "id": attributes["id"],
                "from": attributes["from_id"],
                "to": attributes["to_id"],
                "label": attributes["label"],
                "from_type": attributes["from_type"],
                "to_type": attributes["to_type"],
                "source_group": attributes.get("sourceGroup", ""),
                "source_relation": attributes.get("sourceRelation", ""),
                "source_database": attributes.get("sourceDatabase", ""),
            }
            self.outgoing[row["from"]].append(row)
            self.incoming[row["to"]].append(row)
        self.context_degree = Counter()
        for rows in self.outgoing.values():
            for edge in rows:
                if edge["source_group"] in {"pathway", "anatomy", "indication"}:
                    self.context_degree[edge["to"]] += 1
        self.names_by_type = defaultdict(dict)
        for node_id, attributes in self.vertices.items():
            short_type = attributes["label"].split(".")[-1]
            self.names_by_type[short_type][attributes["name"]] = node_id

    def _endpoint(self, query, mention, node_type):
        candidates = self.names_by_type[node_type]
        if mention in candidates:
            return mention, candidates[mention]
        query_lower = query.lower()
        for name in sorted(candidates, key=lambda value: (-len(value), value)):
            if name.lower() in query_lower:
                return name, candidates[name]
        return None, None

    def _citation(self, edge, reverse=False):
        from_name = self.vertices[edge["from"]]["name"]
        to_name = self.vertices[edge["to"]]["name"]
        return {
            "edge_id": edge["id"],
            "subject": to_name if reverse else from_name,
            "predicate": edge["label"],
            "object": from_name if reverse else to_name,
            "direction": "reverse" if reverse else "forward",
            "source_group": edge["source_group"],
            "source_relation": edge["source_relation"],
            "source_database": edge["source_database"],
        }

    def _retrieve(self, drug, drug_id, target, target_id):
        direct_edges = [
            edge
            for edge in self.outgoing.get(drug_id, [])
            if edge["to"] == target_id and edge["source_group"] == "target"
        ]
        if direct_edges:
            raise RuntimeError(
                "masked direct edge is present in the KAG memory graph: %s/%s"
                % (drug, target)
            )

        paths = []
        other_target_edges = [
            edge
            for edge in self.outgoing.get(drug_id, [])
            if edge["source_group"] == "target"
        ]
        target_context = {
            group: [
                edge
                for edge in self.outgoing.get(target_id, [])
                if edge["source_group"] == group
            ]
            for group in ("pathway", "anatomy")
        }
        for drug_target_edge in other_target_edges:
            other_target_id = drug_target_edge["to"]
            other_target = self.vertices[other_target_id]["name"]
            for group in ("pathway", "anatomy"):
                other_context = {
                    edge["to"]: edge
                    for edge in self.outgoing.get(other_target_id, [])
                    if edge["source_group"] == group
                }
                for target_edge in target_context[group]:
                    shared_id = target_edge["to"]
                    if shared_id not in other_context:
                        continue
                    shared_edge = other_context[shared_id]
                    shared_name = self.vertices[shared_id]["name"]
                    specificity = 1.0 / math.log2(
                        2.0 + self.context_degree[shared_id]
                    )
                    paths.append({
                        "path_id": "P%s" % _stable_id(
                            drug, target, group, other_target_id, shared_id
                        ),
                        "path_type": "shared_%s" % group,
                        "score": (2.0 if group == "pathway" else 1.0)
                        + specificity,
                        "node_sequence": [drug, other_target, shared_name, target],
                        "type_sequence": ["Drug", "Target", group.title(), "Target"],
                        "relation_sequence": [
                            "targets",
                            shared_edge["label"],
                            "%s (reverse traversal)" % target_edge["label"],
                        ],
                        "citations": [
                            self._citation(drug_target_edge),
                            self._citation(shared_edge),
                            self._citation(target_edge, reverse=True),
                        ],
                    })

        query_indications = {
            edge["to"]: edge
            for edge in self.outgoing.get(drug_id, [])
            if edge["source_group"] == "indication"
        }
        for other_drug_target_edge in self.incoming.get(target_id, []):
            if other_drug_target_edge["source_group"] != "target":
                continue
            other_drug_id = other_drug_target_edge["from"]
            if other_drug_id == drug_id:
                continue
            other_drug = self.vertices[other_drug_id]["name"]
            other_indications = {
                edge["to"]: edge
                for edge in self.outgoing.get(other_drug_id, [])
                if edge["source_group"] == "indication"
            }
            for shared_id in sorted(set(query_indications) & set(other_indications)):
                query_edge = query_indications[shared_id]
                other_edge = other_indications[shared_id]
                indication = self.vertices[shared_id]["name"]
                specificity = 1.0 / math.log2(
                    2.0 + self.context_degree[shared_id]
                )
                paths.append({
                    "path_id": "P%s" % _stable_id(
                        drug,
                        target,
                        "shared_indication_cotarget",
                        other_drug_id,
                        shared_id,
                    ),
                    "path_type": "shared_indication_cotarget",
                    "score": 1.5 + specificity,
                    "node_sequence": [drug, indication, other_drug, target],
                    "type_sequence": ["Drug", "Indication", "Drug", "Target"],
                    "relation_sequence": [
                        "indicatedFor",
                        "%s (reverse traversal)" % other_edge["label"],
                        other_drug_target_edge["label"],
                    ],
                    "citations": [
                        self._citation(query_edge),
                        self._citation(other_edge, reverse=True),
                        self._citation(other_drug_target_edge),
                    ],
                })
        paths.sort(key=lambda path: (-path["score"], path["path_id"]))
        return paths[: self.top_k]

    def invoke(self, task, **kwargs):
        query = task.arguments.get("rewrite_query", task.arguments.get("query", ""))
        logic_node = task.arguments.get("logic_form_node")
        subject_mention = ""
        object_mention = ""
        if logic_node is not None:
            if hasattr(logic_node, "s"):
                subject_mention = logic_node.s.get_mention_name() or ""
            if hasattr(logic_node, "o"):
                object_mention = logic_node.o.get_mention_name() or ""
        drug, drug_id = self._endpoint(query, subject_mention, "Drug")
        target, target_id = self._endpoint(query, object_mention, "Target")
        if not drug_id or not target_id:
            return RetrieverOutput(
                retriever_method=self.name,
                err_msg="Could not identify one Drug endpoint and one Target endpoint",
            )

        paths = self._retrieve(drug, drug_id, target, target_id)
        if paths:
            chunks = [ChunkData(
                content=json.dumps(
                    {
                        "held_out_relation": "targets",
                        "direct_edge_masked": True,
                        "interpretation_scope": (
                            "biological hypothesis only; not causal proof and not "
                            "evidence of action direction"
                        ),
                        "retrieved_path_count": len(paths),
                        "paths": paths,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                title="kgml_typed_paths",
                chunk_id="PATH_BUNDLE_%s" % _stable_id(drug, target),
                score=max(path["score"] for path in paths),
            )]
        else:
            chunks = [
                ChunkData(
                    content=json.dumps(
                        {
                            "held_out_relation": "targets",
                            "direct_edge_masked": True,
                            "status": "insufficient relation-specific path evidence",
                            "drug": drug,
                            "target": target,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    title="no_typed_path",
                    chunk_id="NO_PATH_%s" % _stable_id(drug, target),
                    score=1.0,
                )
            ]
        return RetrieverOutput(
            chunks=chunks,
            retriever_method=self.name,
            summary="",
        )

    def schema(self):
        return {
            "name": "kgml_typed_path_retriever",
            "description": (
                "Retrieve leakage-controlled typed heterogeneous paths between a "
                "Drug and a Target from the official KAG memory graph."
            ),
        }
