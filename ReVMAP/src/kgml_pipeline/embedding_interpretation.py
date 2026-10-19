"""Quantitative interpretation of heterogeneous KG embedding geometry.

The original manuscript used a t-SNE plot and a single generated answer to
explain why two drugs with the same nominal pharmacology were far apart.  This
module replaces that anecdotal workflow with an auditable analysis in the
*original* embedding space:

1. construct pharmacologically related drug pairs from a drug--target action
   table;
2. measure cosine and normalized Euclidean distances in every available KG
   view;
3. quantify relation-specific neighbourhood overlap;
4. contrast target-only, add-one-relation, full, and leave-one-relation-out
   embeddings; and
5. export grounded case records for optional KAG/LLM verbalisation.

The view contrasts are deliberately described as ``view-level attribution``.
They compare separately trained KG views and are not equivalent to an
edge-level causal attribution inside a fitted RGCN.  The output manifest and
report preserve this boundary explicitly.
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path

from .io_utils import ensure_dir, write_json


DEFAULT_RELATIONS = ("target", "indication", "anatomy", "pathway")


def _stable_id(*parts):
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:20]


def _read_csv(path):
    with Path(path).expanduser().open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_pair_table(path):
    rows = []
    for row in _read_csv(path):
        drug = str(row.get("drug_name", row.get("drug", ""))).strip()
        target = str(row.get("target_name", row.get("target", ""))).strip()
        action = str(row.get("action", row.get("relation", ""))).strip()
        if drug and target and action:
            rows.append({
                "drug_name": drug,
                "target_name": target,
                "action": action,
            })
    unique = {
        (row["drug_name"], row["target_name"], row["action"]): row
        for row in rows
    }
    return [unique[key] for key in sorted(unique)]


def _sample_index_pairs(size, limit, seed):
    """Return deterministic unique index pairs without materialising O(n^2)."""

    import numpy as np

    total = size * (size - 1) // 2
    if total <= limit:
        return list(itertools.combinations(range(size), 2))
    rng = np.random.default_rng(int(seed))
    selected = set()
    while len(selected) < int(limit):
        draws = rng.integers(0, size, size=(max(64, limit - len(selected)), 2))
        for left, right in draws:
            if left == right:
                continue
            selected.add((int(min(left, right)), int(max(left, right))))
            if len(selected) >= int(limit):
                break
    return sorted(selected)


def _candidate_drug_pairs(pair_rows, allowed_drugs, cfg):
    """Construct same-action drug pairs plus explicitly requested case pairs."""

    action_drugs = defaultdict(set)
    drug_actions = defaultdict(set)
    drug_targets = defaultdict(set)
    drug_action_targets = defaultdict(lambda: defaultdict(set))
    allowed = set(str(value) for value in allowed_drugs)
    allowed_actions = cfg.get("allowed_actions")
    if allowed_actions:
        allowed_actions = {str(value) for value in allowed_actions}

    for row in pair_rows:
        drug = row["drug_name"]
        if drug not in allowed:
            continue
        action = row["action"]
        if allowed_actions and action not in allowed_actions:
            continue
        target = row["target_name"]
        action_drugs[action].add(drug)
        drug_actions[drug].add(action)
        drug_targets[drug].add(target)
        drug_action_targets[drug][action].add(target)

    max_per_action = int(cfg.get("max_pairs_per_action", 20000))
    seed = int(cfg.get("random_seed", 42))
    selected = set()
    for action_index, action in enumerate(sorted(action_drugs)):
        drugs = sorted(action_drugs[action])
        for left, right in _sample_index_pairs(
            len(drugs), max_per_action, seed + action_index * 1009
        ):
            selected.add((drugs[left], drugs[right]))

    explicit = set()
    for raw in cfg.get("explicit_case_pairs", []):
        if isinstance(raw, dict):
            drug_a = str(raw.get("drug_a", "")).strip()
            drug_b = str(raw.get("drug_b", "")).strip()
        else:
            values = list(raw)
            drug_a = str(values[0]).strip() if values else ""
            drug_b = str(values[1]).strip() if len(values) > 1 else ""
        if not drug_a or not drug_b or drug_a == drug_b:
            continue
        key = tuple(sorted((drug_a, drug_b)))
        if key[0] in allowed and key[1] in allowed:
            explicit.add(key)
            selected.add(key)

    rows = []
    for drug_a, drug_b in sorted(selected):
        shared_actions = sorted(drug_actions[drug_a] & drug_actions[drug_b])
        shared_targets = drug_targets[drug_a] & drug_targets[drug_b]
        target_union = drug_targets[drug_a] | drug_targets[drug_b]
        shared_target_actions = 0
        for action in shared_actions:
            shared_target_actions += len(
                drug_action_targets[drug_a][action]
                & drug_action_targets[drug_b][action]
            )
        rows.append({
            "pair_id": "DD%s" % _stable_id(drug_a, drug_b),
            "drug_a": drug_a,
            "drug_b": drug_b,
            "shared_actions": "|".join(shared_actions),
            "primary_action": shared_actions[0] if shared_actions else "explicit_case",
            "shared_action_count": len(shared_actions),
            "shared_target_count": len(shared_targets),
            "shared_target_action_count": int(shared_target_actions),
            "target_jaccard": (
                float(len(shared_targets) / len(target_union)) if target_union else 0.0
            ),
            "explicit_case": (drug_a, drug_b) in explicit,
        })
    return rows


def _load_embedding_view(embedding_root, view, model):
    import numpy as np

    model_root = Path(embedding_root) / str(view) / str(model)
    embedding_path = model_root / "entity_embeddings.npy"
    mapping_path = model_root / "entity_to_id.json"
    if not embedding_path.exists() or not mapping_path.exists():
        return None
    with mapping_path.open("r", encoding="utf-8") as handle:
        entity_to_id = {str(key): int(value) for key, value in json.load(handle).items()}
    values = np.load(str(embedding_path))
    if np.iscomplexobj(values):
        values = np.concatenate([values.real, values.imag], axis=1)
    values = np.asarray(values, dtype=np.float64)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    normalized = values / np.where(norms > 0.0, norms, 1.0)

    lower_to_id = {}
    ambiguous = set()
    for name, index in entity_to_id.items():
        lowered = name.lower()
        if lowered in lower_to_id and lower_to_id[lowered] != index:
            ambiguous.add(lowered)
        else:
            lower_to_id[lowered] = index
    for lowered in ambiguous:
        lower_to_id.pop(lowered, None)
    return {
        "view": str(view),
        "model": str(model),
        "entity_to_id": entity_to_id,
        "lower_to_id": lower_to_id,
        "values": values,
        "normalized": normalized,
        "embedding_dim": int(values.shape[1]),
    }


def _entity_index(source, name):
    index = source["entity_to_id"].get(str(name))
    if index is not None:
        return int(index)
    return source["lower_to_id"].get(str(name).lower())


def _distance_rows(candidate_rows, sources):
    """Measure original-space distance and within-view standardized distance."""

    import numpy as np
    import pandas as pd

    rows = []
    for view, source in sources.items():
        local = []
        for pair in candidate_rows:
            index_a = _entity_index(source, pair["drug_a"])
            index_b = _entity_index(source, pair["drug_b"])
            if index_a is None or index_b is None:
                continue
            raw_a = source["values"][index_a]
            raw_b = source["values"][index_b]
            unit_a = source["normalized"][index_a]
            unit_b = source["normalized"][index_b]
            cosine = float(np.clip(1.0 - np.dot(unit_a, unit_b), 0.0, 2.0))
            local.append({
                "pair_id": pair["pair_id"],
                "drug_a": pair["drug_a"],
                "drug_b": pair["drug_b"],
                "view": view,
                "model": source["model"],
                "embedding_dim": source["embedding_dim"],
                "cosine_distance": cosine,
                "normalized_euclidean_distance": float(np.linalg.norm(unit_a - unit_b)),
                "raw_euclidean_distance": float(np.linalg.norm(raw_a - raw_b)),
            })
        if not local:
            continue
        frame = pd.DataFrame(local)
        values = frame["cosine_distance"].astype(float)
        std = float(values.std(ddof=0))
        frame["cosine_distance_z"] = (
            (values - float(values.mean())) / std if std > 0.0 else 0.0
        )
        frame["cosine_distance_percentile"] = values.rank(
            method="average", pct=True
        )
        rows.extend(frame.to_dict("records"))
    return rows


def _load_relation_profiles(group_root, drugs, relation_names=DEFAULT_RELATIONS):
    """Build drug-centred contexts and provenance paths from triple groups."""

    drug_set = set(str(value) for value in drugs)
    profiles = {
        drug: {relation: set() for relation in relation_names}
        for drug in drug_set
    }
    support = {
        drug: {relation: defaultdict(set) for relation in relation_names}
        for drug in drug_set
    }
    group_root = Path(group_root)

    target_rows = _read_csv(group_root / "target.csv")
    drug_targets = defaultdict(set)
    for row in target_rows:
        drug = str(row.get("head", "")).strip()
        target = str(row.get("tail", "")).strip()
        if drug in drug_set and target:
            drug_targets[drug].add(target)
            if "target" in profiles[drug]:
                profiles[drug]["target"].add(target)
                support[drug]["target"][target].add(target)

    indication_path = group_root / "indication.csv"
    if indication_path.exists() and "indication" in relation_names:
        for row in _read_csv(indication_path):
            drug = str(row.get("head", "")).strip()
            value = str(row.get("tail", "")).strip()
            if drug in drug_set and value:
                profiles[drug]["indication"].add(value)
                support[drug]["indication"][value].add("direct")

    for relation in ("anatomy", "pathway"):
        path = group_root / ("%s.csv" % relation)
        if relation not in relation_names or not path.exists():
            continue
        target_contexts = defaultdict(set)
        for row in _read_csv(path):
            target = str(row.get("head", "")).strip()
            context = str(row.get("tail", "")).strip()
            if target and context:
                target_contexts[target].add(context)
        for drug in drug_set:
            for target in drug_targets.get(drug, set()):
                for context in target_contexts.get(target, set()):
                    profiles[drug][relation].add(context)
                    support[drug][relation][context].add(target)
    return profiles, support


def _idf_weights(profiles, drugs, relations=DEFAULT_RELATIONS):
    weights = {relation: {} for relation in relations}
    drug_list = list(drugs)
    total = max(1, len(drug_list))
    for relation in relations:
        counts = defaultdict(int)
        for drug in drug_list:
            for value in profiles.get(drug, {}).get(relation, set()):
                counts[value] += 1
        for value, count in counts.items():
            weights[relation][value] = math.log((total + 1.0) / (count + 1.0)) + 1.0
    return weights


def _weighted_jaccard(left, right, weights):
    union = set(left) | set(right)
    if not union:
        return 1.0
    intersection = set(left) & set(right)
    denominator = sum(float(weights.get(value, 1.0)) for value in union)
    if denominator <= 0.0:
        return 0.0
    return sum(float(weights.get(value, 1.0)) for value in intersection) / denominator


def _relation_overlap_rows(candidate_rows, profiles, weights, relations):
    rows = []
    for pair in candidate_rows:
        for relation in relations:
            left = profiles.get(pair["drug_a"], {}).get(relation, set())
            right = profiles.get(pair["drug_b"], {}).get(relation, set())
            union = left | right
            intersection = left & right
            jaccard = float(len(intersection) / len(union)) if union else 1.0
            weighted = _weighted_jaccard(left, right, weights.get(relation, {}))
            rows.append({
                "pair_id": pair["pair_id"],
                "drug_a": pair["drug_a"],
                "drug_b": pair["drug_b"],
                "relation": relation,
                "drug_a_context_count": len(left),
                "drug_b_context_count": len(right),
                "shared_context_count": len(intersection),
                "union_context_count": len(union),
                "jaccard_similarity": jaccard,
                "jaccard_distance": 1.0 - jaccard,
                "weighted_jaccard_similarity": weighted,
                "weighted_jaccard_distance": 1.0 - weighted,
            })
    return rows


def _view_attribution_rows(distance_rows, views, reference_view, baseline_view):
    """Build add-one and leave-one-out standardized geometry contrasts."""

    import pandas as pd

    frame = pd.DataFrame(distance_rows)
    if frame.empty:
        return []
    metric_columns = ["cosine_distance_z", "cosine_distance_percentile"]
    wide = frame.pivot_table(index="pair_id", columns="view", values=metric_columns)
    full_groups = set(views.get(reference_view, []))
    base_groups = set(views.get(baseline_view, []))
    pair_names = frame.drop_duplicates("pair_id").set_index("pair_id")[["drug_a", "drug_b"]]
    rows = []

    def available(metric, view, pair_id):
        key = (metric, view)
        return key in wide.columns and pair_id in wide.index and not pd.isna(wide.loc[pair_id, key])

    for view, groups_raw in views.items():
        if view in {reference_view, baseline_view}:
            continue
        groups = set(groups_raw)
        contrast_specs = []
        if groups >= base_groups and len(groups - base_groups) == 1:
            contrast_specs.append((
                "add_one_relation",
                next(iter(groups - base_groups)),
                view,
                baseline_view,
            ))
        if full_groups >= groups and len(full_groups - groups) == 1:
            contrast_specs.append((
                "leave_one_relation_out",
                next(iter(full_groups - groups)),
                reference_view,
                view,
            ))
        if not contrast_specs:
            continue
        for contrast_type, relation, positive_view, negative_view in contrast_specs:
            for pair_id in wide.index:
                if not all(
                    available(metric, positive_view, pair_id)
                    and available(metric, negative_view, pair_id)
                    for metric in metric_columns
                ):
                    continue
                names = pair_names.loc[pair_id]
                rows.append({
                    "pair_id": pair_id,
                    "drug_a": names["drug_a"],
                    "drug_b": names["drug_b"],
                    "relation": relation,
                    "contrast_type": contrast_type,
                    "positive_view": positive_view,
                    "negative_view": negative_view,
                    "cosine_distance_z_delta": float(
                        wide.loc[pair_id, ("cosine_distance_z", positive_view)]
                        - wide.loc[pair_id, ("cosine_distance_z", negative_view)]
                    ),
                    "cosine_distance_percentile_delta": float(
                        wide.loc[pair_id, ("cosine_distance_percentile", positive_view)]
                        - wide.loc[pair_id, ("cosine_distance_percentile", negative_view)]
                    ),
                    "interpretation": (
                        "positive values indicate that the relation-containing view "
                        "places this pair farther apart relative to other pairs"
                    ),
                })
    return rows


def _rank_correlation(left, right):
    import numpy as np
    import pandas as pd

    if len(left) < 3:
        return math.nan
    left_rank = pd.Series(left).rank(method="average").to_numpy(dtype=float)
    right_rank = pd.Series(right).rank(method="average").to_numpy(dtype=float)
    if float(np.std(left_rank)) == 0.0 or float(np.std(right_rank)) == 0.0:
        return math.nan
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _correlation_summary(
    overlap_rows,
    distance_rows,
    attribution_rows,
    reference_view,
    iterations,
    seed,
    max_pairs=None,
):
    import numpy as np
    import pandas as pd

    overlap = pd.DataFrame(overlap_rows)
    distances = pd.DataFrame(distance_rows)
    attribution = pd.DataFrame(attribution_rows)
    if overlap.empty or distances.empty:
        return []
    reference = distances[distances["view"] == reference_view][
        ["pair_id", "cosine_distance_z"]
    ]
    rng = np.random.default_rng(int(seed))
    rows = []

    analyses = []

    def add_analysis(relation, outcome, merged, outcome_column):
        scoped_frames = [
            ("all_pairs", merged),
            (
                "both_drugs_have_context",
                merged[
                    (merged["drug_a_context_count"] > 0)
                    & (merged["drug_b_context_count"] > 0)
                ],
            ),
        ]
        for sample_scope, scoped in scoped_frames:
            if scoped.empty:
                continue
            analyses.append((
                relation,
                outcome,
                sample_scope,
                scoped["weighted_jaccard_distance"].to_numpy(dtype=float),
                scoped[outcome_column].to_numpy(dtype=float),
            ))

    for relation, local in overlap.groupby("relation"):
        merged = local.merge(reference, on="pair_id", how="inner")
        add_analysis(
            relation,
            "full_view_distance",
            merged,
            "cosine_distance_z",
        )
        if not attribution.empty:
            relation_attr = attribution[attribution["relation"] == relation]
            for contrast_type, contrast in relation_attr.groupby("contrast_type"):
                joined = local.merge(
                    contrast[["pair_id", "cosine_distance_z_delta"]],
                    on="pair_id",
                    how="inner",
                )
                add_analysis(
                    relation,
                    contrast_type,
                    joined,
                    "cosine_distance_z_delta",
                )

    for relation, outcome, sample_scope, left, right in analyses:
        if max_pairs and len(left) > int(max_pairs):
            selected = rng.choice(len(left), size=int(max_pairs), replace=False)
            left = left[selected]
            right = right[selected]
        observed = _rank_correlation(left, right)
        valid = len(left) >= 3 and not math.isnan(observed)
        bootstrap = []
        exceed = 0
        if valid and iterations > 0:
            for _ in range(int(iterations)):
                indices = rng.integers(0, len(left), size=len(left))
                value = _rank_correlation(left[indices], right[indices])
                if not math.isnan(value):
                    bootstrap.append(value)
                permuted = rng.permutation(right)
                null = _rank_correlation(left, permuted)
                if not math.isnan(null) and abs(null) >= abs(observed):
                    exceed += 1
        rows.append({
            "relation": relation,
            "outcome": outcome,
            "sample_scope": sample_scope,
            "pair_count": int(len(left)),
            "spearman_rho": observed,
            "bootstrap_ci95_low": (
                float(np.quantile(bootstrap, 0.025)) if bootstrap else math.nan
            ),
            "bootstrap_ci95_high": (
                float(np.quantile(bootstrap, 0.975)) if bootstrap else math.nan
            ),
            "permutation_p_two_sided": (
                float((exceed + 1) / (iterations + 1))
                if valid and iterations > 0 else math.nan
            ),
            "iterations": int(iterations),
        })
    valid = sorted(
        (
            (index, float(row["permutation_p_two_sided"]))
            for index, row in enumerate(rows)
            if not math.isnan(float(row["permutation_p_two_sided"]))
        ),
        key=lambda item: item[1],
    )
    adjusted = {}
    running = 1.0
    total = len(valid)
    for reverse_rank, (index, p_value) in enumerate(reversed(valid), start=1):
        rank = total - reverse_rank + 1
        running = min(running, p_value * total / rank)
        adjusted[index] = min(1.0, running)
    for index, row in enumerate(rows):
        row["permutation_q_bh"] = adjusted.get(index, math.nan)
    return rows


def _matched_control_rows(candidate_rows, distance_rows, profiles, cfg, reference_view):
    """Match distant same-action pairs to close pairs with similar KG support."""

    import numpy as np
    import pandas as pd

    candidates = pd.DataFrame(candidate_rows)
    distances = pd.DataFrame(distance_rows)
    if candidates.empty or distances.empty:
        return [], []
    reference = distances[distances["view"] == reference_view][
        ["pair_id", "cosine_distance", "cosine_distance_z", "cosine_distance_percentile"]
    ]
    frame = candidates.merge(reference, on="pair_id", how="inner")
    if frame.empty:
        return [], []

    frame["action_distance_percentile"] = frame.groupby("primary_action")[
        "cosine_distance"
    ].rank(method="average", pct=True)
    relations = sorted(next(iter(profiles.values())).keys()) if profiles else []
    degree = {
        drug: sum(len(values.get(relation, set())) for relation in relations)
        for drug, values in profiles.items()
    }
    frame["context_degree_sum"] = [
        degree.get(row.drug_a, 0) + degree.get(row.drug_b, 0)
        for row in frame.itertuples(index=False)
    ]
    distant_quantile = float(cfg.get("distant_quantile", 0.90))
    close_quantile = float(cfg.get("close_quantile", 0.10))
    case_count = int(cfg.get("case_count", 25))
    min_shared_target = int(cfg.get("min_shared_target_count", 0))
    distant = frame[
        (frame["action_distance_percentile"] >= distant_quantile)
        & (frame["shared_target_count"] >= min_shared_target)
    ].sort_values(
        ["explicit_case", "action_distance_percentile", "cosine_distance"],
        ascending=[False, False, False],
    )
    explicit = frame[frame["explicit_case"]]
    cases = pd.concat([explicit, distant], ignore_index=True).drop_duplicates("pair_id")
    cases = cases.head(case_count)

    rows = []
    used_controls = set()
    for case in cases.itertuples(index=False):
        pool = frame[
            (frame["primary_action"] == case.primary_action)
            & (frame["action_distance_percentile"] <= close_quantile)
            & (~frame["pair_id"].isin(used_controls))
        ].copy()
        case_drugs = {case.drug_a, case.drug_b}
        pool = pool[
            [not bool(case_drugs & {row.drug_a, row.drug_b}) for row in pool.itertuples()]
        ]
        if pool.empty:
            continue
        degree_scale = max(1.0, float(frame["context_degree_sum"].std(ddof=0)))
        pool["match_cost"] = (
            (pool["target_jaccard"] - float(case.target_jaccard)).abs()
            + 0.5 * (
                pool["shared_target_count"] - int(case.shared_target_count)
            ).abs()
            + 0.25 * (
                pool["context_degree_sum"] - float(case.context_degree_sum)
            ).abs() / degree_scale
        )
        control = pool.sort_values(
            ["match_cost", "cosine_distance", "pair_id"]
        ).iloc[0]
        used_controls.add(control["pair_id"])
        rows.append({
            "case_pair_id": case.pair_id,
            "case_drug_a": case.drug_a,
            "case_drug_b": case.drug_b,
            "control_pair_id": control["pair_id"],
            "control_drug_a": control["drug_a"],
            "control_drug_b": control["drug_b"],
            "primary_action": case.primary_action,
            "case_cosine_distance": float(case.cosine_distance),
            "control_cosine_distance": float(control["cosine_distance"]),
            "case_action_percentile": float(case.action_distance_percentile),
            "control_action_percentile": float(control["action_distance_percentile"]),
            "case_target_jaccard": float(case.target_jaccard),
            "control_target_jaccard": float(control["target_jaccard"]),
            "case_context_degree_sum": int(case.context_degree_sum),
            "control_context_degree_sum": int(control["context_degree_sum"]),
            "match_cost": float(control["match_cost"]),
        })
    return cases.to_dict("records"), rows


def _matched_relation_comparison(
    matched_controls,
    overlap_rows,
    iterations,
    seed,
):
    """Compare relation disparity in distant cases and matched close controls."""

    import numpy as np
    import pandas as pd

    controls = pd.DataFrame(matched_controls)
    overlap = pd.DataFrame(overlap_rows)
    if controls.empty or overlap.empty:
        return [], []
    lookup = overlap.set_index(["pair_id", "relation"])[
        "weighted_jaccard_distance"
    ].to_dict()
    detail = []
    relations = sorted(overlap["relation"].astype(str).unique())
    for row in controls.itertuples(index=False):
        for relation in relations:
            case_value = lookup.get((row.case_pair_id, relation))
            control_value = lookup.get((row.control_pair_id, relation))
            if case_value is None or control_value is None:
                continue
            detail.append({
                "case_pair_id": row.case_pair_id,
                "control_pair_id": row.control_pair_id,
                "primary_action": row.primary_action,
                "relation": relation,
                "case_weighted_jaccard_distance": float(case_value),
                "control_weighted_jaccard_distance": float(control_value),
                "paired_difference": float(case_value - control_value),
            })

    rng = np.random.default_rng(int(seed) + 7919)
    summary = []
    detail_frame = pd.DataFrame(detail)
    if detail_frame.empty:
        return detail, summary
    for relation, frame in detail_frame.groupby("relation"):
        differences = frame["paired_difference"].to_numpy(dtype=float)
        observed = float(np.mean(differences))
        bootstrap = []
        exceed = 0
        for _ in range(int(iterations)):
            indices = rng.integers(0, len(differences), size=len(differences))
            bootstrap.append(float(np.mean(differences[indices])))
            signs = rng.choice(np.asarray([-1.0, 1.0]), size=len(differences))
            null = float(np.mean(differences * signs))
            if abs(null) >= abs(observed):
                exceed += 1
        summary.append({
            "relation": relation,
            "matched_pair_count": int(len(differences)),
            "case_disparity_mean": float(
                frame["case_weighted_jaccard_distance"].mean()
            ),
            "control_disparity_mean": float(
                frame["control_weighted_jaccard_distance"].mean()
            ),
            "paired_difference_mean": observed,
            "paired_difference_median": float(np.median(differences)),
            "bootstrap_ci95_low": (
                float(np.quantile(bootstrap, 0.025)) if bootstrap else math.nan
            ),
            "bootstrap_ci95_high": (
                float(np.quantile(bootstrap, 0.975)) if bootstrap else math.nan
            ),
            "sign_flip_p_two_sided": (
                float((exceed + 1) / (iterations + 1))
                if iterations > 0 else math.nan
            ),
            "iterations": int(iterations),
        })
    return detail, summary


def _top_contexts(values, weights, limit):
    return sorted(
        values,
        key=lambda value: (-float(weights.get(value, 1.0)), str(value)),
    )[: int(limit)]


def _case_records(
    cases,
    distance_rows,
    overlap_rows,
    attribution_rows,
    profiles,
    support,
    weights,
    cfg,
):
    import pandas as pd

    distances = pd.DataFrame(distance_rows)
    overlaps = pd.DataFrame(overlap_rows)
    attributions = pd.DataFrame(attribution_rows)
    context_limit = int(cfg.get("case_context_limit", 8))
    records = []
    for case in cases:
        pair_id = case["pair_id"]
        drug_a, drug_b = case["drug_a"], case["drug_b"]
        pair_distances = distances[distances["pair_id"] == pair_id]
        pair_overlaps = overlaps[overlaps["pair_id"] == pair_id]
        pair_attr = (
            attributions[attributions["pair_id"] == pair_id]
            if not attributions.empty else attributions
        )
        relation_evidence = []
        for overlap in pair_overlaps.itertuples(index=False):
            relation = overlap.relation
            relation_attr = (
                pair_attr[pair_attr["relation"] == relation]
                if not pair_attr.empty else pair_attr
            )
            deltas = (
                relation_attr["cosine_distance_z_delta"].astype(float).tolist()
                if not relation_attr.empty else []
            )
            left = profiles[drug_a][relation]
            right = profiles[drug_b][relation]
            positive_delta = max([value for value in deltas if value > 0.0] or [0.0])
            negative_delta = min([value for value in deltas if value < 0.0] or [0.0])
            if positive_delta > abs(negative_delta):
                geometry_role = "separation_associated"
            elif negative_delta < 0.0:
                geometry_role = "convergence_associated"
            else:
                geometry_role = "context_disparity_only"
            if left and right:
                coverage_status = "both_drugs_have_context"
            elif left:
                coverage_status = "only_drug_a_has_context"
            elif right:
                coverage_status = "only_drug_b_has_context"
            else:
                coverage_status = "neither_drug_has_context"
            unique_a = _top_contexts(
                left - right, weights.get(relation, {}), context_limit
            )
            unique_b = _top_contexts(
                right - left, weights.get(relation, {}), context_limit
            )
            shared = _top_contexts(
                left & right, weights.get(relation, {}), context_limit
            )

            def cite(drug, values):
                return [
                    {
                        "context": value,
                        "support_targets": sorted(support[drug][relation].get(value, set())),
                        "idf_weight": float(weights.get(relation, {}).get(value, 1.0)),
                    }
                    for value in values
                ]

            relation_evidence.append({
                "relation": relation,
                "coverage_status": coverage_status,
                "geometry_role": geometry_role,
                "weighted_jaccard_distance": float(overlap.weighted_jaccard_distance),
                "view_contrasts": relation_attr.to_dict("records") if not relation_attr.empty else [],
                "maximum_positive_separation_delta": float(positive_delta),
                "minimum_negative_convergence_delta": float(negative_delta),
                "ranking_score": float(
                    overlap.weighted_jaccard_distance * (1.0 + positive_delta)
                ),
                "unique_to_drug_a": cite(drug_a, unique_a),
                "unique_to_drug_b": cite(drug_b, unique_b),
                "shared": cite(drug_a, shared),
            })
        relation_evidence.sort(
            key=lambda row: (-row["ranking_score"], row["relation"])
        )
        record = {
            "case_id": "EC%s" % _stable_id(pair_id, "embedding_context"),
            "pair_id": pair_id,
            "drug_a": drug_a,
            "drug_b": drug_b,
            "shared_actions": case.get("shared_actions", ""),
            "shared_target_count": int(case.get("shared_target_count", 0)),
            "selection": {
                "explicit_case": bool(case.get("explicit_case", False)),
                "action_distance_percentile": float(
                    case.get("action_distance_percentile", math.nan)
                ),
            },
            "embedding_distances": pair_distances.to_dict("records"),
            "relation_evidence": relation_evidence,
            "question": (
                "Why are %s and %s separated in the learned drug-embedding space "
                "despite sharing pharmacological action labels? Which typed KG "
                "contexts are associated with this separation?" % (drug_a, drug_b)
            ),
            "prompt": {
                "system": (
                    "Use only the supplied embedding distances, view contrasts, and "
                    "cited KG contexts. Describe association with embedding geometry, "
                    "not a causal drug mechanism. Do not use t-SNE distance as evidence. "
                    "A positive view contrast means only that the relation-containing "
                    "separately trained view placed the pair farther apart relative to "
                    "other pairs. A negative contrast is countervailing evidence. If only "
                    "one drug has a relation context, describe incomplete KG coverage "
                    "rather than claiming that the other drug lacks that biology. Cite "
                    "relation names and context nodes for every claim."
                ),
                "evidence": relation_evidence,
            },
        }
        records.append(record)
    return records


def _write_csv(path, rows):
    import pandas as pd

    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)
    return frame


def _write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=True) + "\n")


def _write_case_reports(output_root, records):
    """Write deterministic readable reports before optional LLM wording."""

    case_root = ensure_dir(Path(output_root) / "case_reports")
    index_rows = []
    for record in records:
        selection = record.get("selection", {})
        lines = [
            "# %s versus %s" % (record["drug_a"], record["drug_b"]),
            "",
            "- Shared action labels: `%s`" % record.get("shared_actions", ""),
            "- Within-action embedding-distance percentile: %.3f"
            % float(selection.get("action_distance_percentile", math.nan)),
            "- Shared targets in the evaluation table: %d"
            % int(record.get("shared_target_count", 0)),
            "",
            "## Relation-context evidence",
            "",
        ]
        for evidence in record.get("relation_evidence", []):
            lines.extend([
                "### %s" % evidence["relation"],
                "",
                "- Geometry role: `%s`" % evidence["geometry_role"],
                "- KG coverage: `%s`" % evidence["coverage_status"],
                "- Weighted context disparity: %.3f"
                % float(evidence["weighted_jaccard_distance"]),
                "- Maximum separation contrast: %.3f"
                % float(evidence["maximum_positive_separation_delta"]),
                "- Minimum convergence contrast: %.3f"
                % float(evidence["minimum_negative_convergence_delta"]),
                "",
            ])
            for label, key in (
                ("Unique contexts for %s" % record["drug_a"], "unique_to_drug_a"),
                ("Unique contexts for %s" % record["drug_b"], "unique_to_drug_b"),
                ("Shared contexts", "shared"),
            ):
                values = evidence.get(key, [])
                if not values:
                    continue
                rendered = []
                for value in values:
                    support_targets = value.get("support_targets", [])
                    suffix = (
                        " via %s"
                        % ", ".join("`%s`" % item for item in support_targets)
                        if support_targets else ""
                    )
                    rendered.append("`%s`%s" % (value["context"], suffix))
                lines.append("- %s: %s" % (label, "; ".join(rendered)))
            lines.append("")
        lines.extend([
            "## Interpretation boundary",
            "",
            "This report identifies typed KG contexts associated with embedding",
            "geometry. It does not establish a causal drug mechanism or action",
            "direction. One-sided context coverage may reflect missing KG data.",
            "",
        ])
        filename = "%s.md" % record["case_id"]
        (case_root / filename).write_text("\n".join(lines), encoding="utf-8")
        index_rows.append({
            "case_id": record["case_id"],
            "drug_a": record["drug_a"],
            "drug_b": record["drug_b"],
            "shared_actions": record.get("shared_actions", ""),
            "report_path": "case_reports/%s" % filename,
        })
    return index_rows


def _write_summary(
    path,
    cohort_name,
    model,
    reference_view,
    baseline_view,
    candidates,
    distance_rows,
    correlations,
    cases,
    matched_controls,
    matched_relation_summary,
    available_views,
):
    import pandas as pd

    distances = pd.DataFrame(distance_rows)
    lines = [
        "# Embedding-context interpretation summary",
        "",
        "## Scope",
        "",
        "- Cohort: `%s`" % cohort_name,
        "- Embedding model: `%s`" % model,
        "- Reference view: `%s`" % reference_view,
        "- Baseline view: `%s`" % baseline_view,
        "- Candidate same-action/explicit drug pairs: %d" % len(candidates),
        "- Selected distant or explicit cases: %d" % len(cases),
        "- Matched close controls: %d" % len(matched_controls),
        "- Available views: %s" % ", ".join("`%s`" % view for view in available_views),
        "",
        "Distances are calculated in the original embedding space. t-SNE is not used",
        "for pair selection or quantitative evidence.",
        "",
        "## Scientific interpretation boundary",
        "",
        "The relation contrasts compare separately trained KG views. They test whether",
        "a relation family is associated with a change in embedding geometry, but they",
        "are not an edge-level causal attribution inside a fixed RGCN. The KAG records",
        "therefore explain heterogeneous graph context, not drug-target mechanism or",
        "pharmacological action direction.",
        "",
        "## View coverage",
        "",
        "| View | Pairs with both embeddings | Mean cosine distance |",
        "|---|---:|---:|",
    ]
    if not distances.empty:
        for view, frame in distances.groupby("view"):
            lines.append(
                "| %s | %d | %.4f |"
                % (view, len(frame), float(frame["cosine_distance"].mean()))
            )
    lines.extend([
        "",
        "## Relation-context correlations",
        "",
        "| Relation | Outcome | Scope | N | Spearman rho | 95% bootstrap CI | Permutation p | BH q |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ])
    for row in correlations:
        lines.append(
            "| %s | %s | %s | %d | %.3f | [%.3f, %.3f] | %.4f | %.4f |"
            % (
                row["relation"],
                row["outcome"],
                row["sample_scope"],
                row["pair_count"],
                row["spearman_rho"],
                row["bootstrap_ci95_low"],
                row["bootstrap_ci95_high"],
                row["permutation_p_two_sided"],
                row["permutation_q_bh"],
            )
        )
    lines.extend([
        "",
        "## Distant cases versus matched close controls",
        "",
        "Positive differences mean that distant cases have more different relation",
        "contexts than same-action close controls matched on target overlap and degree.",
        "",
        "| Relation | Matched pairs | Case disparity | Control disparity | Paired difference | 95% bootstrap CI | Sign-flip p |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in matched_relation_summary:
        lines.append(
            "| %s | %d | %.3f | %.3f | %.3f | [%.3f, %.3f] | %.4f |"
            % (
                row["relation"],
                row["matched_pair_count"],
                row["case_disparity_mean"],
                row["control_disparity_mean"],
                row["paired_difference_mean"],
                row["bootstrap_ci95_low"],
                row["bootstrap_ci95_high"],
                row["sign_flip_p_two_sided"],
            )
        )
    lines.extend([
        "",
        "## Selected embedding-distance cases",
        "",
        "| Drug A | Drug B | Shared action(s) | Within-action percentile |",
        "|---|---|---|---:|",
    ])
    for row in cases:
        lines.append(
            "| %s | %s | %s | %.3f |"
            % (
                row["drug_a"],
                row["drug_b"],
                row.get("shared_actions", ""),
                float(row.get("action_distance_percentile", math.nan)),
            )
        )
    lines.extend([
        "",
        "The machine-readable case records contain the relation-specific contexts,",
        "supporting target nodes, view contrasts, and constrained KAG prompts.",
        "",
    ])
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def _run_cohort(config, global_cfg, cohort_name, cohort_cfg, logger):
    merged = dict(global_cfg)
    merged.pop("cohorts", None)
    merged.update(cohort_cfg)
    merged.setdefault("random_seed", int(config.get("random_seed", 42)))

    pair_path = merged.get("pair_table_path")
    if not pair_path:
        raise ValueError("embedding_interpretation.%s is missing pair_table_path" % cohort_name)
    model = str(merged.get("embedding_model", "RGCN"))
    reference_view = str(merged.get("reference_view", "full_label_free"))
    baseline_view = str(merged.get("baseline_view", "targets_only_label_free"))
    output_root = Path(
        merged.get(
            "output_dir",
            Path(config["paths"]["reports_dir"]) / "embedding_interpretation",
        )
    ).expanduser()
    if global_cfg.get("cohorts"):
        output_root = output_root / cohort_name
    output_root = ensure_dir(output_root)

    configured_views = config.get("views", {})
    signature_config = {
        key: value
        for key, value in merged.items()
        if key not in {"output_dir", "reuse_completed_results"}
    }
    signature_payload = json.dumps(
        {
            "analysis_version": "embedding-context-v3",
            "settings": signature_config,
            "view_definitions": configured_views,
        },
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )
    analysis_signature = hashlib.sha256(
        signature_payload.encode("utf-8")
    ).hexdigest()

    manifest_path = output_root / "manifest.json"
    if bool(merged.get("reuse_completed_results", False)) and manifest_path.exists():
        required_outputs = (
            "candidate_drug_pairs.csv",
            "embedding_distances_by_view.csv",
            "relation_context_overlap.csv",
            "view_level_attribution.csv",
            "relation_distance_correlations.csv",
            "matched_close_controls.csv",
            "matched_relation_disparity_pairs.csv",
            "matched_relation_disparity_summary.csv",
            "grounded_embedding_cases.jsonl",
            "case_report_index.csv",
            "embedding_interpretation_summary.md",
        )
        with manifest_path.open("r", encoding="utf-8") as handle:
            previous_manifest = json.load(handle)
        if (
            previous_manifest.get("analysis_signature") == analysis_signature
            and all((output_root / name).exists() for name in required_outputs)
        ):
            logger.info(
                "Reusing completed embedding interpretation cohort %s from %s",
                cohort_name,
                output_root,
            )
            return previous_manifest

    embedding_root = Path(
        merged.get("processed_dir", config["paths"]["processed_dir"])
    ).expanduser() / "embeddings"
    requested_views = merged.get("views") or list(configured_views)
    sources = {}
    for view in requested_views:
        source = _load_embedding_view(embedding_root, view, model)
        if source is None:
            logger.warning("Embedding interpretation skipped missing %s/%s", view, model)
            continue
        sources[str(view)] = source
    if reference_view not in sources:
        raise FileNotFoundError(
            "Reference embeddings not found: %s/%s" % (reference_view, model)
        )
    if baseline_view not in sources:
        raise FileNotFoundError(
            "Baseline embeddings not found: %s/%s" % (baseline_view, model)
        )

    pair_rows = _load_pair_table(pair_path)
    reference_entities = set(sources[reference_view]["entity_to_id"])
    candidates = _candidate_drug_pairs(pair_rows, reference_entities, merged)
    if not candidates:
        raise ValueError("No related drug pairs overlap the reference embeddings")
    distance_rows = _distance_rows(candidates, sources)

    relation_names = tuple(merged.get("relations", DEFAULT_RELATIONS))
    group_root = Path(
        merged.get("intermediate_dir", config["paths"]["intermediate_dir"])
    ).expanduser() / "triples" / "groups"
    drugs = sorted(
        {row["drug_a"] for row in candidates} | {row["drug_b"] for row in candidates}
    )
    profiles, support = _load_relation_profiles(group_root, drugs, relation_names)
    weights = _idf_weights(profiles, drugs, relation_names)
    overlap_rows = _relation_overlap_rows(
        candidates, profiles, weights, relation_names
    )
    available_view_defs = {
        view: configured_views.get(view, []) for view in sources
    }
    attribution_rows = _view_attribution_rows(
        distance_rows,
        available_view_defs,
        reference_view,
        baseline_view,
    )
    iterations = int(merged.get("resampling_iterations", 1000))
    correlations = _correlation_summary(
        overlap_rows,
        distance_rows,
        attribution_rows,
        reference_view,
        iterations,
        int(merged["random_seed"]),
        int(merged.get("correlation_max_pairs", 10000)),
    )
    cases, matched_controls = _matched_control_rows(
        candidates, distance_rows, profiles, merged, reference_view
    )
    matched_relation_detail, matched_relation_summary = _matched_relation_comparison(
        matched_controls,
        overlap_rows,
        iterations,
        int(merged["random_seed"]),
    )
    case_records = _case_records(
        cases,
        distance_rows,
        overlap_rows,
        attribution_rows,
        profiles,
        support,
        weights,
        merged,
    )

    _write_csv(output_root / "candidate_drug_pairs.csv", candidates)
    _write_csv(output_root / "embedding_distances_by_view.csv", distance_rows)
    _write_csv(output_root / "relation_context_overlap.csv", overlap_rows)
    _write_csv(output_root / "view_level_attribution.csv", attribution_rows)
    _write_csv(output_root / "relation_distance_correlations.csv", correlations)
    _write_csv(output_root / "matched_close_controls.csv", matched_controls)
    _write_csv(
        output_root / "matched_relation_disparity_pairs.csv",
        matched_relation_detail,
    )
    _write_csv(
        output_root / "matched_relation_disparity_summary.csv",
        matched_relation_summary,
    )
    _write_jsonl(output_root / "grounded_embedding_cases.jsonl", case_records)
    case_report_index = _write_case_reports(output_root, case_records)
    _write_csv(output_root / "case_report_index.csv", case_report_index)
    _write_summary(
        output_root / "embedding_interpretation_summary.md",
        cohort_name,
        model,
        reference_view,
        baseline_view,
        candidates,
        distance_rows,
        correlations,
        cases,
        matched_controls,
        matched_relation_summary,
        sorted(sources),
    )
    manifest = {
        "cohort": cohort_name,
        "analysis_signature": analysis_signature,
        "analysis_version": "embedding-context-v3",
        "pair_table_path": str(Path(pair_path).expanduser()),
        "embedding_root": str(embedding_root),
        "embedding_model": model,
        "reference_view": reference_view,
        "baseline_view": baseline_view,
        "available_views": sorted(sources),
        "relation_names": list(relation_names),
        "candidate_pair_count": len(candidates),
        "distance_row_count": len(distance_rows),
        "attribution_row_count": len(attribution_rows),
        "selected_case_count": len(cases),
        "matched_control_count": len(matched_controls),
        "allowed_actions": merged.get("allowed_actions"),
        "resampling_iterations": iterations,
        "uses_tsne_for_distance": False,
        "attribution_level": "separately_trained_graph_view_contrast",
        "supports_edge_level_rgcn_message_attribution": False,
        "claim_boundary": (
            "Explains heterogeneous graph context associated with embedding geometry; "
            "does not establish causal drug mechanism or action direction."
        ),
        "outputs": {
            "candidate_pairs": "candidate_drug_pairs.csv",
            "distances": "embedding_distances_by_view.csv",
            "relation_overlap": "relation_context_overlap.csv",
            "view_attribution": "view_level_attribution.csv",
            "correlations": "relation_distance_correlations.csv",
            "matched_controls": "matched_close_controls.csv",
            "matched_relation_pairs": "matched_relation_disparity_pairs.csv",
            "matched_relation_summary": "matched_relation_disparity_summary.csv",
            "grounded_cases": "grounded_embedding_cases.jsonl",
            "case_report_index": "case_report_index.csv",
            "case_reports": "case_reports/",
            "summary": "embedding_interpretation_summary.md",
        },
    }
    write_json(manifest_path, manifest)
    logger.info(
        "Embedding interpretation cohort %s wrote %d candidate pairs and %d cases to %s",
        cohort_name,
        len(candidates),
        len(cases),
        output_root,
    )
    return manifest


def run_embedding_interpretation(config, logger):
    """Run all configured embedding-geometry interpretation cohorts."""

    global_cfg = dict(config.get("embedding_interpretation", {}))
    if not global_cfg:
        raise ValueError("Missing embedding_interpretation config section")
    cohorts = global_cfg.get("cohorts")
    if cohorts:
        manifests = {
            str(name): _run_cohort(
                config, global_cfg, str(name), dict(cohort_cfg), logger
            )
            for name, cohort_cfg in cohorts.items()
        }
    else:
        cohort_name = str(global_cfg.get("cohort_name", "default"))
        manifests = {
            cohort_name: _run_cohort(
                config, global_cfg, cohort_name, {}, logger
            )
        }
    return manifests
