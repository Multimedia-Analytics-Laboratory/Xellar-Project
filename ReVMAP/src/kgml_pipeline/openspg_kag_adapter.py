"""OpenSPG KAG export and leakage-controlled pair-evidence construction.

The OpenSPG KAG runtime is an optional external dependency.  This module keeps
the KGML-side preparation dependency-light: it exports the official
``nodes.json``/``edges.json`` injection format, masks every evaluated direct
drug--target edge, retrieves typed indirect paths, and writes auditable prompt
records.  The generated project assets can then be consumed by OpenSPG KAG
0.8.x and its ``TraceLogReporter`` without making KAG a hard dependency of the
prediction pipeline.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from .io_utils import ensure_dir, write_json


UPSTREAM_KAG = {
    "repository": "https://github.com/OpenSPG/KAG",
    "version": "0.8.0",
    "tree_sha": "fdab15b3929d2ee40dfcdd388f90233096a6afc9",
}

GROUP_SPECS = {
    "target": ("Drug", "Target", "targets"),
    "indication": ("Drug", "Indication", "indicatedFor"),
    "anatomy": ("Target", "Anatomy", "expressedIn"),
    "pathway": ("Target", "Pathway", "participatesIn"),
}

SOURCE_DATABASES = {
    "target": "Repurposing Hub / harmonized drug-target table",
    "indication": "PrimeKG indication",
    "anatomy": "PrimeKG anatomy_protein_present",
    "pathway": "PrimeKG pathway_protein",
}


def _node_id(node_type, raw_id):
    return "%s::%s" % (node_type, str(raw_id))


def _stable_id(*parts):
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:20]


def _read_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_pairs(path):
    rows = []
    for row in _read_csv(path):
        drug = str(row.get("drug_name", "")).strip()
        target = str(row.get("target_name", "")).strip()
        action = str(row.get("action", row.get("relation", ""))).strip()
        if drug and target and action:
            rows.append({"drug_name": drug, "target_name": target, "action": action})
    rows.sort(key=lambda row: (row["drug_name"], row["target_name"], row["action"]))
    return rows


def _load_group_edges(config, groups):
    group_root = Path(config["paths"]["intermediate_dir"]) / "triples" / "groups"
    edges = []
    for group in groups:
        if group not in GROUP_SPECS:
            raise ValueError("Unsupported KAG triple group: %s" % group)
        path = group_root / ("%s.csv" % group)
        if not path.exists():
            raise FileNotFoundError("KAG triple group not found: %s" % path)
        source_type, target_type, relation_label = GROUP_SPECS[group]
        for row in _read_csv(path):
            head = str(row.get("head", "")).strip()
            tail = str(row.get("tail", "")).strip()
            if not head or not tail:
                continue
            edges.append({
                "id": _stable_id(group, head, tail, row.get("relation", "")),
                "from": _node_id(source_type, head),
                "fromType": source_type,
                "fromName": head,
                "to": _node_id(target_type, tail),
                "toType": target_type,
                "toName": tail,
                "label": relation_label,
                "source_group": group,
                "source_relation": str(row.get("relation", relation_label)),
                "source_database": SOURCE_DATABASES[group],
            })
    unique = {}
    for edge in edges:
        unique[(edge["from"], edge["label"], edge["to"])] = edge
    return sorted(unique.values(), key=lambda edge: (edge["from"], edge["label"], edge["to"]))


def _mask_evaluated_target_edges(edges, pairs):
    masked_pairs = {
        (_node_id("Drug", row["drug_name"]), _node_id("Target", row["target_name"]))
        for row in pairs
    }
    retained = []
    removed = []
    for edge in edges:
        if edge["source_group"] == "target" and (edge["from"], edge["to"]) in masked_pairs:
            removed.append(edge)
        else:
            retained.append(edge)
    return retained, removed


def _index_edges(edges):
    outgoing = defaultdict(list)
    incoming = defaultdict(list)
    for edge in edges:
        outgoing[edge["from"]].append(edge)
        incoming[edge["to"]].append(edge)
    return outgoing, incoming


def _citation(edge, direction="forward"):
    if direction == "forward":
        subject, obj = edge["fromName"], edge["toName"]
    else:
        subject, obj = edge["toName"], edge["fromName"]
    return {
        "edge_id": edge["id"],
        "subject": subject,
        "predicate": edge["label"],
        "object": obj,
        "direction": direction,
        "source_group": edge["source_group"],
        "source_relation": edge["source_relation"],
        "source_database": edge["source_database"],
    }


def _retrieve_pair_bundle(edges, drug, target, action, max_paths=20, max_context=20):
    outgoing, incoming = _index_edges(edges)
    drug_id = _node_id("Drug", drug)
    target_id = _node_id("Target", target)

    other_target_edges = [
        edge for edge in outgoing.get(drug_id, []) if edge["source_group"] == "target"
    ]
    target_context = {
        group: [
            edge
            for edge in outgoing.get(target_id, [])
            if edge["source_group"] == group
        ]
        for group in ("pathway", "anatomy")
    }

    context_degree = Counter()
    for edge in edges:
        if edge["source_group"] in {"pathway", "anatomy", "indication"}:
            context_degree[edge["to"]] += 1

    candidates = []
    for drug_target_edge in other_target_edges:
        other_target_id = drug_target_edge["to"]
        for group in ("pathway", "anatomy"):
            other_context = {
                edge["to"]: edge
                for edge in outgoing.get(other_target_id, [])
                if edge["source_group"] == group
            }
            for target_edge in target_context[group]:
                shared_id = target_edge["to"]
                if shared_id not in other_context:
                    continue
                shared_edge = other_context[shared_id]
                specificity = 1.0 / math.log2(2.0 + context_degree[shared_id])
                group_weight = 2.0 if group == "pathway" else 1.0
                score = group_weight + specificity
                citations = [
                    _citation(drug_target_edge),
                    _citation(shared_edge),
                    _citation(target_edge, direction="reverse"),
                ]
                candidates.append({
                    "path_id": "P%s" % _stable_id(
                        drug, target, group, other_target_id, shared_id
                    ),
                    "path_type": "shared_%s" % group,
                    "score": score,
                    "graph_specificity": specificity,
                    "context_node_degree": int(context_degree[shared_id]),
                    "evidence_scope": (
                        "relation_specific_hypothesis"
                        if group == "pathway"
                        else "broad_context_only"
                    ),
                    "nodes": [
                        drug,
                        drug_target_edge["toName"],
                        target_edge["toName"],
                        target,
                    ],
                    "citations": citations,
                    "statement": (
                        "%s targets %s; both %s and %s connect to %s %s."
                        % (
                            drug,
                            drug_target_edge["toName"],
                            drug_target_edge["toName"],
                            target,
                            group,
                            target_edge["toName"],
                        )
                    ),
                })

    query_indications = {
        edge["to"]: edge
        for edge in outgoing.get(drug_id, [])
        if edge["source_group"] == "indication"
    }
    for other_drug_target_edge in incoming.get(target_id, []):
        if other_drug_target_edge["source_group"] != "target":
            continue
        other_drug_id = other_drug_target_edge["from"]
        if other_drug_id == drug_id:
            continue
        other_indications = {
            edge["to"]: edge
            for edge in outgoing.get(other_drug_id, [])
            if edge["source_group"] == "indication"
        }
        for shared_id in sorted(set(query_indications) & set(other_indications)):
            query_edge = query_indications[shared_id]
            other_edge = other_indications[shared_id]
            specificity = 1.0 / math.log2(2.0 + context_degree[shared_id])
            candidates.append({
                "path_id": "P%s" % _stable_id(
                    drug, target, "shared_indication_cotarget", other_drug_id, shared_id
                ),
                "path_type": "shared_indication_cotarget",
                "score": 1.5 + specificity,
                "graph_specificity": specificity,
                "context_node_degree": int(context_degree[shared_id]),
                "evidence_scope": "relation_specific_hypothesis",
                "nodes": [
                    drug,
                    query_edge["toName"],
                    other_drug_target_edge["fromName"],
                    target,
                ],
                "citations": [
                    _citation(query_edge),
                    _citation(other_edge, direction="reverse"),
                    _citation(other_drug_target_edge),
                ],
                "statement": (
                    "%s and %s share indication %s; %s targets %s."
                    % (
                        drug,
                        other_drug_target_edge["fromName"],
                        query_edge["toName"],
                        other_drug_target_edge["fromName"],
                        target,
                    )
                ),
            })
    candidates.sort(key=lambda path: (-path["score"], path["path_id"]))

    context_edges = []
    for edge in outgoing.get(drug_id, []):
        if edge["source_group"] in {"target", "indication"}:
            context_edges.append(_citation(edge))
    for group in ("pathway", "anatomy"):
        for edge in target_context[group]:
            context_edges.append(_citation(edge))
    context_edges = context_edges[: int(max_context)]

    paths = candidates[: int(max_paths)]
    question = (
        "With the direct drug-target edge between %s and %s withheld, which typed "
        "heterogeneous graph paths still connect them, and what biological "
        "hypotheses do those paths suggest for the relation?"
        % (drug, target)
    )
    prompt = {
        "system": (
            "Use only the supplied cited graph paths. Explain how each typed path can "
            "make the held-out drug-target relation biologically plausible, but label "
            "it as a hypothesis rather than causal proof. Cite path IDs for every "
            "claim. Do not infer inhibitor, agonist, antagonist, or blocker direction "
            "from unsigned paths. A shared pathway or indication/cotarget route may "
            "motivate a relation hypothesis. A shared anatomy/expression route is only "
            "broad co-localisation context and must not be described as relation-specific "
            "support. If anatomy routes are the only paths, explicitly state "
            "'insufficient relation-specific path evidence' before listing them as "
            "background context. Use the same statement when no informative path is "
            "available."
        ),
        "question": question,
        "paths": paths,
        "context_edges": context_edges,
    }
    return {
        "query_id": "Q%s" % _stable_id(drug, target, action),
        "drug_name": drug,
        "target_name": target,
        "held_out_relation": "targets",
        "evaluation_action_label": action,
        "direct_edge_masked": True,
        "question": question,
        "indirect_path_count": len(paths),
        "has_indirect_support": bool(paths),
        "paths": paths,
        "context_edges": context_edges,
        "prompt": prompt,
    }


def _openspg_nodes(edges, pairs):
    nodes = {}
    for edge in edges:
        for prefix in ("from", "to"):
            node_id = edge[prefix]
            nodes[node_id] = {
                "id": node_id,
                "name": edge["%sName" % prefix],
                "label": edge["%sType" % prefix],
                "properties": {"rawId": edge["%sName" % prefix]},
            }
    for row in pairs:
        for node_type, value in (("Drug", row["drug_name"]), ("Target", row["target_name"])):
            node_id = _node_id(node_type, value)
            nodes.setdefault(node_id, {
                "id": node_id,
                "name": value,
                "label": node_type,
                "properties": {"rawId": value},
            })
    return sorted(nodes.values(), key=lambda node: (node["label"], node["id"]))


def _openspg_edges(edges):
    return [
        {
            "id": edge["id"],
            "from": edge["from"],
            "fromType": edge["fromType"],
            "to": edge["to"],
            "toType": edge["toType"],
            "label": edge["label"],
            "properties": {
                "sourceGroup": edge["source_group"],
                "sourceRelation": edge["source_relation"],
                "sourceDatabase": edge["source_database"],
            },
        }
        for edge in edges
    ]


def _degree_matched_random_controls(edges, pairs, bundles, max_paths, max_context):
    """Build deterministic false-target controls matched on local context degree."""

    outgoing, _ = _index_edges(edges)
    target_ids = sorted(
        {
            node_id
            for edge in edges
            for node_id, node_type in (
                (edge["from"], edge["fromType"]),
                (edge["to"], edge["toType"]),
            )
            if node_type == "Target"
        }
    )
    target_names = {node_id: node_id.split("::", 1)[1] for node_id in target_ids}

    def context_degree(target_id):
        return sum(
            edge["source_group"] in {"pathway", "anatomy"}
            for edge in outgoing.get(target_id, [])
        )

    degrees = {target_id: context_degree(target_id) for target_id in target_ids}
    known_by_drug = defaultdict(set)
    for edge in edges:
        if edge["source_group"] == "target":
            known_by_drug[edge["from"]].add(edge["to"])
    for row in pairs:
        known_by_drug[_node_id("Drug", row["drug_name"])].add(
            _node_id("Target", row["target_name"])
        )

    controls = []
    for row, bundle in zip(pairs, bundles):
        drug_id = _node_id("Drug", row["drug_name"])
        true_target_id = _node_id("Target", row["target_name"])
        true_degree = degrees.get(true_target_id, 0)
        candidates = [
            target_id
            for target_id in target_ids
            if target_id != true_target_id and target_id not in known_by_drug[drug_id]
        ]
        if not candidates:
            continue
        candidates.sort(
            key=lambda target_id: (
                abs(degrees.get(target_id, 0) - true_degree),
                _stable_id(bundle["query_id"], target_id),
            )
        )
        control_target_id = candidates[0]
        control_target = target_names[control_target_id]
        control_edges = [
            edge
            for edge in edges
            if not (
                edge["source_group"] == "target"
                and edge["from"] == drug_id
                and edge["to"] == control_target_id
            )
        ]
        control_bundle = _retrieve_pair_bundle(
            control_edges,
            row["drug_name"],
            control_target,
            row["action"],
            max_paths=max_paths,
            max_context=max_context,
        )
        controls.append({
            "query_id": bundle["query_id"],
            "drug_name": row["drug_name"],
            "true_target_name": row["target_name"],
            "control_target_name": control_target,
            "true_target_context_degree": true_degree,
            "control_target_context_degree": degrees.get(control_target_id, 0),
            "true_indirect_path_count": bundle["indirect_path_count"],
            "control_indirect_path_count": control_bundle["indirect_path_count"],
            "true_top_path_score": (
                bundle["paths"][0]["score"] if bundle["paths"] else 0.0
            ),
            "control_top_path_score": (
                control_bundle["paths"][0]["score"]
                if control_bundle["paths"]
                else 0.0
            ),
            "true_top_path": bundle["paths"][0] if bundle["paths"] else None,
            "control_top_path": (
                control_bundle["paths"][0] if control_bundle["paths"] else None
            ),
            "true_has_indirect_support": bundle["has_indirect_support"],
            "control_has_indirect_support": control_bundle["has_indirect_support"],
        })
    return controls


def _macro_f1(true_labels, predicted_labels):
    classes = sorted(set(true_labels) | set(predicted_labels))
    values = []
    for label in classes:
        tp = sum(t == label and p == label for t, p in zip(true_labels, predicted_labels))
        fp = sum(t != label and p == label for t, p in zip(true_labels, predicted_labels))
        fn = sum(t == label and p != label for t, p in zip(true_labels, predicted_labels))
        precision = tp / float(tp + fp) if tp + fp else 0.0
        recall = tp / float(tp + fn) if tp + fn else 0.0
        values.append(
            2.0 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )
    return sum(values) / float(len(values)) if values else 0.0


def _classification_diagnostics(true_labels, predicted_labels):
    classes = sorted(set(true_labels) | set(predicted_labels))
    per_class = {}
    for label in classes:
        tp = sum(t == label and p == label for t, p in zip(true_labels, predicted_labels))
        fp = sum(t != label and p == label for t, p in zip(true_labels, predicted_labels))
        fn = sum(t == label and p != label for t, p in zip(true_labels, predicted_labels))
        precision = tp / float(tp + fp) if tp + fp else 0.0
        recall = tp / float(tp + fn) if tp + fn else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )
        per_class[label] = {
            "support": sum(t == label for t in true_labels),
            "predicted_count": sum(p == label for p in predicted_labels),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return {
        "accuracy": (
            sum(t == p for t, p in zip(true_labels, predicted_labels))
            / float(len(true_labels)) if true_labels else 0.0
        ),
        "macro_f1": _macro_f1(true_labels, predicted_labels),
        "per_class": per_class,
    }


def _wilson_interval(successes, total, z=1.96):
    if not total:
        return [0.0, 0.0]
    proportion = successes / float(total)
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half_width = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return [center - half_width, center + half_width]


def _load_double_disjoint_predictions(path, repeat=0, model="RGCN", classifier="Random Forest"):
    selected = []
    for row in _read_csv(path):
        if str(row.get("split", "")) != "double_disjoint":
            continue
        if str(row.get("model", "")) != str(model):
            continue
        if str(row.get("classifier", "")) != str(classifier):
            continue
        if int(row.get("repeat", -1)) != int(repeat):
            continue
        selected.append({
            "drug_name": str(row["drug_name"]),
            "target_name": str(row["target_name"]),
            "true_label": str(row["true_label"]),
            "predicted_label": str(row["predicted_label"]),
            "fold": int(row["fold"]),
        })
    selected.sort(key=lambda row: (row["fold"], row["drug_name"], row["target_name"]))
    return selected


def _double_disjoint_training_pairs(pairs, predictions, entity_partitions, seed):
    """Reconstruct the exact training pairs used by every OOF fold."""

    import numpy as np

    from .pair_evaluation import _iter_splits

    labels = [row["action"] for row in pairs]
    label_to_id = {label: index for index, label in enumerate(sorted(set(labels)))}
    encoded = np.asarray([label_to_id[label] for label in labels], dtype=int)
    drugs = np.asarray([row["drug_name"] for row in pairs], dtype=object)
    targets = np.asarray([row["target_name"] for row in pairs], dtype=object)
    dummy = np.zeros((len(pairs), 1), dtype=np.float32)
    pair_keys = [
        (row["drug_name"], row["target_name"])
        for row in pairs
    ]
    predictions_by_fold = defaultdict(list)
    for row in predictions:
        predictions_by_fold[row["fold"]].append(row)

    training_by_fold = {}
    expected_test_by_fold = {}
    for fold, (train_indices, test_indices) in enumerate(
        _iter_splits(
            "double_disjoint",
            int(entity_partitions),
            int(seed),
            dummy,
            encoded,
            drugs,
            targets,
        )
    ):
        training_by_fold[fold] = [pairs[int(index)] for index in train_indices]
        expected_test_by_fold[fold] = {pair_keys[int(index)] for index in test_indices}

    observed_folds = set(predictions_by_fold)
    if observed_folds != set(expected_test_by_fold):
        raise ValueError(
            "Double-disjoint OOF folds do not match reconstructed folds: "
            "observed=%s expected=%s"
            % (sorted(observed_folds), sorted(expected_test_by_fold))
        )
    for fold, rows in predictions_by_fold.items():
        observed = {(row["drug_name"], row["target_name"]) for row in rows}
        expected = expected_test_by_fold[fold]
        if observed != expected:
            raise ValueError(
                "Double-disjoint OOF membership mismatch in fold %d: "
                "observed=%d expected=%d" % (fold, len(observed), len(expected))
            )
    return training_by_fold, expected_test_by_fold


def _signed_analogy_evidence(
    edges,
    pairs,
    predictions,
    top_k=20,
    entity_partitions=5,
    seed=42,
):
    """Retrieve fold-safe signed analogies for action-direction explanations.

    The queried pair action is never used.  For each 5x5 double-disjoint test
    block, candidate signed action edges are restricted to pairs containing
    neither a test drug nor a test target.  Candidates are ranked by shared
    pathway/anatomy context between targets and shared indication context
    between drugs.
    """

    outgoing, _ = _index_edges(edges)
    context_degree = Counter()
    for edge in edges:
        if edge["source_group"] in {"pathway", "anatomy", "indication"}:
            context_degree[edge["to"]] += 1

    def edge_map(node_type, name, groups):
        node_id = _node_id(node_type, name)
        result = defaultdict(dict)
        for edge in outgoing.get(node_id, []):
            if edge["source_group"] in groups:
                result[edge["source_group"]][edge["to"]] = edge
        return result

    pair_by_key = {
        (row["drug_name"], row["target_name"]): row for row in pairs
    }
    predictions_by_fold = defaultdict(list)
    for row in predictions:
        predictions_by_fold[row["fold"]].append(row)
    training_by_fold, expected_test_by_fold = _double_disjoint_training_pairs(
        pairs,
        predictions,
        entity_partitions=entity_partitions,
        seed=seed,
    )

    bundles = []
    for fold, fold_predictions in sorted(predictions_by_fold.items()):
        training_pairs = training_by_fold[fold]
        test_drugs = {
            drug for drug, _ in expected_test_by_fold[fold]
        }
        test_targets = {
            target for _, target in expected_test_by_fold[fold]
        }
        training_context = []
        for row in training_pairs:
            training_context.append((
                row,
                edge_map("Drug", row["drug_name"], {"indication"}),
                edge_map("Target", row["target_name"], {"pathway", "anatomy"}),
            ))

        for prediction in fold_predictions:
            key = (prediction["drug_name"], prediction["target_name"])
            true_pair = pair_by_key.get(key)
            if true_pair is None:
                continue
            query_drug_context = edge_map(
                "Drug", prediction["drug_name"], {"indication"}
            )
            query_target_context = edge_map(
                "Target", prediction["target_name"], {"pathway", "anatomy"}
            )
            analogies = []
            for candidate, candidate_drug_context, candidate_target_context in training_context:
                shared = []
                score = 0.0
                for group, weight in (("pathway", 2.0), ("anatomy", 1.0)):
                    shared_ids = set(query_target_context[group]) & set(
                        candidate_target_context[group]
                    )
                    for shared_id in sorted(shared_ids):
                        score += weight / math.log2(2.0 + context_degree[shared_id])
                        shared.append({
                            "context_type": group,
                            "context_name": query_target_context[group][shared_id]["toName"],
                            "candidate_citation": _citation(
                                candidate_target_context[group][shared_id]
                            ),
                            "query_citation": _citation(
                                query_target_context[group][shared_id]
                            ),
                        })
                shared_indications = set(query_drug_context["indication"]) & set(
                    candidate_drug_context["indication"]
                )
                for shared_id in sorted(shared_indications):
                    score += 0.5 / math.log2(2.0 + context_degree[shared_id])
                    shared.append({
                        "context_type": "indication",
                        "context_name": query_drug_context["indication"][shared_id]["toName"],
                        "candidate_citation": _citation(
                            candidate_drug_context["indication"][shared_id]
                        ),
                        "query_citation": _citation(
                            query_drug_context["indication"][shared_id]
                        ),
                    })
                if score <= 0.0:
                    continue
                signed_citation = {
                    "edge_id": _stable_id(
                        "training_action",
                        fold,
                        candidate["drug_name"],
                        candidate["target_name"],
                        candidate["action"],
                    ),
                    "subject": candidate["drug_name"],
                    "predicate": candidate["action"],
                    "object": candidate["target_name"],
                    "source_group": "training_action",
                    "source_database": "double-disjoint training-fold action labels",
                    "fold": fold,
                }
                analogies.append({
                    "analogy_id": "A%s" % _stable_id(
                        key, candidate["drug_name"], candidate["target_name"], fold
                    ),
                    "action": candidate["action"],
                    "score": score,
                    "training_drug": candidate["drug_name"],
                    "training_target": candidate["target_name"],
                    "signed_action_citation": signed_citation,
                    "shared_context": shared[:10],
                })
            analogies.sort(key=lambda row: (-row["score"], row["analogy_id"]))
            analogies = analogies[: int(top_k)]
            action_scores = defaultdict(float)
            for analogy in analogies:
                action_scores[analogy["action"]] += analogy["score"]
            evidence_action = (
                sorted(action_scores, key=lambda action: (-action_scores[action], action))[0]
                if action_scores else None
            )
            query_id = "Q%s" % _stable_id(
                prediction["drug_name"], prediction["target_name"], true_pair["action"]
            )
            question = (
                "Given that %s is associated with target %s, what indirect, "
                "double-disjoint training-fold evidence supports or contradicts "
                "KGML's predicted %s action, while withholding this pair's action label?"
                % (
                    prediction["drug_name"],
                    prediction["target_name"],
                    prediction["predicted_label"],
                )
            )
            bundles.append({
                "query_id": query_id,
                "fold": fold,
                "drug_name": prediction["drug_name"],
                "target_name": prediction["target_name"],
                "true_action": true_pair["action"],
                "kgml_predicted_action": prediction["predicted_label"],
                "evidence_predicted_action": evidence_action,
                "covered": evidence_action is not None,
                "training_pair_count": len(training_pairs),
                "action_scores": dict(sorted(action_scores.items())),
                "analogies": analogies,
                "question": question,
                "claim_scope": (
                    "Fold-safe signed analogy evidence, not causal pharmacological proof."
                ),
                "leakage_audit": {
                    "training_contains_test_drug": any(
                        row["drug_name"] in test_drugs for row in training_pairs
                    ),
                    "training_contains_test_target": any(
                        row["target_name"] in test_targets for row in training_pairs
                    ),
                    "query_action_withheld_from_prompt": True,
                },
            })
    bundles.sort(key=lambda row: (row["fold"], row["drug_name"], row["target_name"]))
    covered = [row for row in bundles if row["covered"]]
    evidence_diagnostics = _classification_diagnostics(
        [row["true_action"] for row in covered],
        [row["evidence_predicted_action"] for row in covered],
    )
    kgml_diagnostics = _classification_diagnostics(
        [row["true_action"] for row in bundles],
        [row["kgml_predicted_action"] for row in bundles],
    )
    summary = {
        "pair_count": len(bundles),
        "covered_pair_count": len(covered),
        "coverage": len(covered) / float(len(bundles)) if bundles else 0.0,
        "evidence_action_accuracy_on_covered": evidence_diagnostics["accuracy"],
        "evidence_action_macro_f1_on_covered": evidence_diagnostics["macro_f1"],
        "evidence_kgml_agreement_on_covered": (
            sum(
                row["evidence_predicted_action"] == row["kgml_predicted_action"]
                for row in covered
            ) / float(len(covered)) if covered else 0.0
        ),
        "zero_test_drug_overlap_by_construction": True,
        "zero_test_target_overlap_by_construction": True,
        "query_action_withheld_from_prompt": True,
        "entity_partitions": int(entity_partitions),
        "seed": int(seed),
        "evidence_action_diagnostics": evidence_diagnostics,
        "kgml_prediction_diagnostics": kgml_diagnostics,
    }
    return bundles, summary


def _write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_jsonl_gzip(path, rows):
    with gzip.open(str(path), "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _schema_text(namespace):
    return """namespace {namespace}

Drug: EntityType
    properties:
        rawId: Text
    relations:
        targets: Target
        indicatedFor: Indication

Target: EntityType
    properties:
        rawId: Text
    relations:
        expressedIn: Anatomy
        participatesIn: Pathway

Indication: EntityType
    properties:
        rawId: Text

Anatomy: EntityType
    properties:
        rawId: Text

Pathway: EntityType
    properties:
        rawId: Text
""".format(namespace=namespace)


def build_openspg_kag_evidence(config, logger):
    """Export a masked OpenSPG graph and pair-level KAG evidence bundles."""

    kag_cfg = dict(config.get("kag_evaluation", {}))
    if not kag_cfg:
        raise ValueError("Missing kag_evaluation config section")
    pair_path = kag_cfg.get("pair_table_path") or config.get("pair_evaluation", {}).get(
        "pair_table_path"
    )
    if not pair_path:
        raise ValueError("KAG evaluation requires pair_table_path")
    groups = kag_cfg.get("groups", ["target", "indication", "anatomy", "pathway"])
    pairs = _load_pairs(Path(pair_path).expanduser())
    edges = _load_group_edges(config, groups)
    retained_edges, removed_edges = _mask_evaluated_target_edges(edges, pairs)

    max_paths = int(kag_cfg.get("max_paths", 20))
    max_context = int(kag_cfg.get("max_context_edges", 20))
    bundles = [
        _retrieve_pair_bundle(
            retained_edges,
            row["drug_name"],
            row["target_name"],
            row["action"],
            max_paths=max_paths,
            max_context=max_context,
        )
        for row in pairs
    ]
    random_controls = _degree_matched_random_controls(
        retained_edges,
        pairs,
        bundles,
        max_paths=max_paths,
        max_context=max_context,
    )
    signed_analogy_bundles = []
    signed_analogy_summary = None
    prediction_path = kag_cfg.get("double_disjoint_prediction_path")
    if prediction_path:
        predictions = _load_double_disjoint_predictions(
            Path(prediction_path).expanduser(),
            repeat=int(kag_cfg.get("prediction_repeat", 0)),
            model=str(kag_cfg.get("prediction_model", "RGCN")),
            classifier=str(kag_cfg.get("prediction_classifier", "Random Forest")),
        )
        if not predictions:
            raise ValueError(
                "No matching double-disjoint predictions found in %s" % prediction_path
            )
        signed_analogy_bundles, signed_analogy_summary = _signed_analogy_evidence(
            retained_edges,
            pairs,
            predictions,
            top_k=int(kag_cfg.get("signed_analogy_top_k", 20)),
            entity_partitions=int(kag_cfg.get("entity_partitions", 5)),
            seed=int(kag_cfg.get("prediction_seed", config.get("random_seed", 42))),
        )

    output_dir = ensure_dir(
        Path(kag_cfg.get("output_dir", Path(config["paths"]["reports_dir"]) / "kag"))
    )
    namespace = str(kag_cfg.get("namespace", "KGMLPairAction"))
    write_json(output_dir / "nodes.json", _openspg_nodes(retained_edges, pairs))
    write_json(output_dir / "edges.json", _openspg_edges(retained_edges))
    _write_jsonl(output_dir / "questions.jsonl", bundles)
    _write_jsonl_gzip(output_dir / "questions.jsonl.gz", bundles)
    _write_jsonl(
        output_dir / "prompt_records.jsonl",
        ({"query_id": bundle["query_id"], **bundle["prompt"]} for bundle in bundles),
    )
    _write_jsonl(output_dir / "degree_matched_random_controls.jsonl", random_controls)
    if signed_analogy_bundles:
        _write_jsonl(output_dir / "signed_analogy_evidence.jsonl", signed_analogy_bundles)
        _write_jsonl_gzip(
            output_dir / "signed_analogy_evidence.jsonl.gz",
            signed_analogy_bundles,
        )
    (output_dir / ("%s.schema" % namespace)).write_text(
        _schema_text(namespace), encoding="utf-8"
    )

    action_summary = defaultdict(lambda: {"pairs": 0, "with_indirect_path": 0})
    for bundle in bundles:
        stats = action_summary[bundle["evaluation_action_label"]]
        stats["pairs"] += 1
        stats["with_indirect_path"] += int(bundle["has_indirect_support"])
    for stats in action_summary.values():
        stats["path_coverage"] = (
            stats["with_indirect_path"] / float(stats["pairs"]) if stats["pairs"] else 0.0
        )

    retrieved_edge_ids = {
        citation["edge_id"]
        for bundle in bundles
        for path in bundle["paths"]
        for citation in path["citations"]
    }
    removed_edge_ids = {edge["id"] for edge in removed_edges}
    path_type_summary = defaultdict(lambda: {"path_count": 0, "pair_count": 0})
    for bundle in bundles:
        present_types = set()
        for path in bundle["paths"]:
            path_type = path["path_type"]
            path_type_summary[path_type]["path_count"] += 1
            present_types.add(path_type)
        for path_type in present_types:
            path_type_summary[path_type]["pair_count"] += 1

    true_top_scores = [row["true_top_path_score"] for row in random_controls]
    control_top_scores = [row["control_top_path_score"] for row in random_controls]
    top_score_wins = sum(
        true > control for true, control in zip(true_top_scores, control_top_scores)
    )
    top_score_ties = sum(
        true == control for true, control in zip(true_top_scores, control_top_scores)
    )
    top_score_losses = len(random_controls) - top_score_wins - top_score_ties
    non_tied_count = top_score_wins + top_score_losses
    manifest = {
        "upstream_kag": UPSTREAM_KAG,
        "namespace": namespace,
        "pair_count": len(pairs),
        "input_edge_count": len(edges),
        "retained_edge_count": len(retained_edges),
        "masked_direct_edge_count": len(removed_edges),
        "direct_edge_leakage_count": len(retrieved_edge_ids & removed_edge_ids),
        "indirect_path_pair_count": sum(bundle["has_indirect_support"] for bundle in bundles),
        "indirect_path_coverage": (
            sum(bundle["has_indirect_support"] for bundle in bundles) / float(len(bundles))
            if bundles else 0.0
        ),
        "evaluation_action_stratified_path_coverage": dict(
            sorted(action_summary.items())
        ),
        "path_type_summary": dict(sorted(path_type_summary.items())),
        "path_scope_policy": {
            "shared_pathway": "relation_specific_hypothesis",
            "shared_indication_cotarget": "relation_specific_hypothesis",
            "shared_anatomy": "broad_context_only",
            "action_direction_from_unsigned_path": False,
        },
        "degree_matched_random_control": {
            "pair_count": len(random_controls),
            "true_path_coverage": (
                sum(row["true_has_indirect_support"] for row in random_controls)
                / float(len(random_controls))
                if random_controls else 0.0
            ),
            "control_path_coverage": (
                sum(row["control_has_indirect_support"] for row in random_controls)
                / float(len(random_controls))
                if random_controls else 0.0
            ),
            "mean_true_path_count": (
                sum(row["true_indirect_path_count"] for row in random_controls)
                / float(len(random_controls))
                if random_controls else 0.0
            ),
            "mean_control_path_count": (
                sum(row["control_indirect_path_count"] for row in random_controls)
                / float(len(random_controls))
                if random_controls else 0.0
            ),
            "mean_true_top_path_score": (
                sum(true_top_scores) / float(len(true_top_scores))
                if true_top_scores else 0.0
            ),
            "mean_control_top_path_score": (
                sum(control_top_scores) / float(len(control_top_scores))
                if control_top_scores else 0.0
            ),
            "true_top_score_win_rate": (
                top_score_wins / float(len(random_controls))
                if random_controls else 0.0
            ),
            "top_score_tie_rate": (
                top_score_ties / float(len(random_controls))
                if random_controls else 0.0
            ),
            "top_score_loss_rate": (
                top_score_losses / float(len(random_controls))
                if random_controls else 0.0
            ),
            "non_tied_true_win_rate": (
                top_score_wins / float(non_tied_count) if non_tied_count else 0.0
            ),
            "non_tied_true_win_rate_wilson_95ci": _wilson_interval(
                top_score_wins, non_tied_count
            ),
            "matched_pair_ranking_accuracy": (
                (top_score_wins + 0.5 * top_score_ties) / float(len(random_controls))
                if random_controls else 0.0
            ),
        },
        "double_disjoint_signed_analogy": signed_analogy_summary,
        "claim_scope": (
            "Held-out heterogeneous relation-path recovery and biological hypothesis "
            "generation only. Retrieved paths are not causal proof and do not establish "
            "action direction or local model faithfulness."
        ),
    }
    write_json(output_dir / "manifest.json", manifest)
    logger.info(
        "KAG export wrote %d pairs and %d retained edges; masked %d direct edges; indirect path coverage=%.3f",
        len(pairs),
        len(retained_edges),
        len(removed_edges),
        manifest["indirect_path_coverage"],
    )
    return manifest
