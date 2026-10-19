"""Rebuild drug-target pair-specific pharmacological-action ground truth.

The original pair-action cohort inherited a drug-level MoA label for every
target of the same drug.  This module replaces that broadcast label with an
evidence ledger and a conservative consensus label:

* DrugBank target ``actions`` with ``known-action=yes`` and Open Targets
  target-level mechanisms are direct, high-tier evidence.
* Other DrugBank target actions and Repurposing Hub rows are supporting
  evidence and can only produce a silver label.
* PrimeKG drug-protein edges corroborate the pair, but never provide an action
  label.
* Conflicting high-tier labels are retained for audit and excluded from gold.

Raw source data are not copied.  The outputs contain only pair-scoped derived
evidence, consensus tables, coverage summaries, and source provenance.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .io_utils import ensure_dir, write_json, write_text


DRUGBANK_NS = "{http://www.drugbank.ca}"
CORE4_ACTIONS = {"agonist", "antagonist", "blocker", "inhibitor"}
DRUGMAP7_ACTIONS = {
    "agonist",
    "allosteric_modulator",
    "antagonist",
    "inducer",
    "inhibitor",
    "ligand",
    "substrate",
}
DIRECT_HIGH = "direct_high"
DIRECT_LOW = "direct_low"
SUPPORTING = "supporting"
ASSOCIATION_ONLY = "association_only"


def _clean(value):
    if value is None:
        return ""
    return unicodedata.normalize("NFKC", str(value)).strip()


def _name_key(value):
    return re.sub(r"\s+", " ", _clean(value)).casefold()


def _relaxed_key(value):
    return re.sub(r"[^a-z0-9]+", "", _name_key(value))


def _target_key(value):
    return re.sub(r"[^a-z0-9]+", "", _name_key(value))


def _direct_text(element, tag):
    child = element.find(DRUGBANK_NS + tag)
    return _clean(child.text if child is not None else "")


def _primary_drugbank_id(drug):
    for element in drug.findall(DRUGBANK_NS + "drugbank-id"):
        if element.attrib.get("primary") == "true":
            return _clean(element.text)
    first = drug.find(DRUGBANK_NS + "drugbank-id")
    return _clean(first.text if first is not None else "")


def _has_explicit_primary_drugbank_id(drug):
    """Distinguish top-level DrugBank records from nested interaction stubs."""

    return any(
        element.attrib.get("primary") == "true"
        for element in drug.findall(DRUGBANK_NS + "drugbank-id")
    )


def _iter_values(value):
    """Yield scalar strings from parquet list/array fields."""

    if value is None:
        return
    if isinstance(value, str):
        if value.strip():
            yield value
        return
    if isinstance(value, (list, tuple, set)) or getattr(value, "ndim", 0):
        for item in value:
            if item is not None and _clean(item):
                yield _clean(item)
        return
    try:
        if pd.isna(value):
            return
    except (TypeError, ValueError):
        pass
    try:
        for item in value:
            if item is not None and _clean(item):
                yield _clean(item)
    except TypeError:
        if _clean(value):
            yield _clean(value)


def normalize_action_terms(raw_value):
    """Return detailed normalized terms and unambiguous core-four labels.

    Specific phrases are matched before generic ones.  Mechanisms outside the
    manuscript's four-class task remain in ``detailed`` and are not coerced.
    """

    text = _name_key(raw_value)
    if not text:
        return [], []

    detailed = set()
    residual = text
    phrase_rules = (
        (r"\bpositive allosteric modulat(?:or|ion)\b", "positive_allosteric_modulator"),
        (r"\bnegative allosteric modulat(?:or|ion)\b", "negative_allosteric_modulator"),
        (r"\binverse agonis(?:t|m)\b", "inverse_agonist"),
        (r"\bpartial agonis(?:t|m)\b", "partial_agonist"),
        (r"\bco[- ]?agonis(?:t|m)\b", "co_agonist"),
    )
    for pattern, label in phrase_rules:
        if re.search(pattern, residual):
            detailed.add(label)
            residual = re.sub(pattern, " ", residual)

    generic_rules = (
        (r"\bantagonis(?:t|m|tic)\b", "antagonist"),
        (r"\bblock(?:er|ade|ing|s|ed)\b", "blocker"),
        (r"\binhibit(?:or|ion|ory|ing|s|ed)\b", "inhibitor"),
        (r"\bagonis(?:t|m|tic)\b", "agonist"),
        (r"\bactivat(?:or|ion|ing|es|ed)\b", "activator"),
        (r"\bmodulat(?:or|ion|ing|es|ed)\b", "modulator"),
        (r"\bbind(?:er|ing|s)\b", "binder"),
        (r"\bsubstrate\b", "substrate"),
        (r"\binduc(?:er|tion|ing|es|ed)\b", "inducer"),
        (r"\bsuppress(?:or|ion|ing|es|ed)\b", "suppressor"),
        (r"\bpotentiat(?:or|ion|ing|es|ed)\b", "potentiator"),
        (r"\bchelating agent\b|\bchelator\b", "chelator"),
    )
    for pattern, label in generic_rules:
        if re.search(pattern, residual):
            detailed.add(label)

    core = detailed.intersection(CORE4_ACTIONS)
    if "partial_agonist" in detailed or "co_agonist" in detailed:
        core.add("agonist")
    # Inverse agonism and allosteric modulation are deliberately not collapsed
    # into agonist/antagonist because their pharmacology is distinct.
    return sorted(detailed), sorted(core)


def drugmap7_action(action_normalized, action_raw=""):
    """Map detailed evidence into the seven classes used by DrugMAP.

    This follows the authors' public preprocessing code except that this
    pipeline never randomly selects a class when a pair has multiple
    incompatible actions.
    """

    normalized = _clean(action_normalized)
    mapping = {
        "agonist": "agonist",
        "partial_agonist": "agonist",
        "inverse_agonist": "agonist",
        "co_agonist": "agonist",
        "antagonist": "antagonist",
        "blocker": "inhibitor",
        "inhibitor": "inhibitor",
        "suppressor": "inhibitor",
        "binder": "ligand",
        "substrate": "substrate",
        "modulator": "allosteric_modulator",
        "positive_allosteric_modulator": "allosteric_modulator",
        "negative_allosteric_modulator": "allosteric_modulator",
        "activator": "inducer",
        "inducer": "inducer",
        "potentiator": "inducer",
    }
    if normalized in mapping:
        return mapping[normalized]

    raw = _name_key(action_raw)
    raw_exact = {
        "binder": "ligand",
        "binding": "ligand",
        "ligand": "ligand",
        "substrate": "substrate",
        "blocker": "inhibitor",
        "downregulator": "inhibitor",
        "inactivator": "inhibitor",
        "translocation inhibitor": "inhibitor",
        "weak inhibitor": "inhibitor",
        "stimulator": "inducer",
        "partial antagonist": "antagonist",
        "inhibitory allosteric modulator": "allosteric_modulator",
        "negative modulator": "allosteric_modulator",
        "positive allosteric modulator": "allosteric_modulator",
    }
    return raw_exact.get(raw, "")


def _resolve_drugmap7_classes(classes):
    """Resolve low-information ligand/substrate labels conservatively."""

    labels = set(filter(None, map(_clean, classes)))
    if len(labels) <= 1:
        return labels, "single_class"
    specific = labels.difference({"ligand", "substrate"})
    if len(specific) == 1:
        return specific, "discarded_ligand_or_substrate_in_favor_of_specific_action"
    return labels, "unresolved_multiclass"


def _stable_pair_id(cohort, drugbank_id, drug_name, target_id, target_name):
    raw = "|".join(
        [
            _clean(cohort),
            _clean(drugbank_id) or _name_key(drug_name),
            _clean(target_id) or _target_key(target_name),
        ]
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _load_seed_cohorts(cohort_paths):
    frames = []
    for cohort_name, raw_path in cohort_paths.items():
        path = Path(raw_path).expanduser()
        if not path.exists():
            raise FileNotFoundError("Seed pair cohort not found: %s" % path)
        frame = pd.read_csv(path, dtype=str).fillna("")
        required = {"drug_name", "target_name", "action", "drugbank_id"}
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(
                "Seed cohort %s lacks required columns: %s"
                % (cohort_name, ", ".join(missing))
            )
        frame = frame.copy()
        frame["cohort"] = str(cohort_name)
        frame["legacy_action"] = frame["action"].map(_name_key)
        if "target_uniprot_id" not in frame.columns:
            frame["target_uniprot_id"] = ""
        if "target_gene_name" not in frame.columns:
            frame["target_gene_name"] = frame["target_name"]
        frame["pair_id"] = frame.apply(
            lambda row: _stable_pair_id(
                cohort_name,
                row["drugbank_id"],
                row["drug_name"],
                row["target_uniprot_id"],
                row["target_gene_name"] or row["target_name"],
            ),
            axis=1,
        )
        duplicate_ids = frame["pair_id"].duplicated(keep=False)
        if duplicate_ids.any():
            # The legacy cohort can contain two benchmark target spellings
            # mapped to the same DrugBank/UniProt target.  This is a duplicate
            # physical pair, not two observations. Prefer the spelling that
            # agrees with DrugBank's gene symbol and retain all aliases for audit.
            group_sizes = frame.groupby("pair_id")["pair_id"].transform("size")
            alias_map = (
                frame.groupby("pair_id")["target_name"]
                .agg(lambda values: "|".join(sorted(set(map(_clean, values)))))
                .to_dict()
            )
            frame["seed_rows_collapsed"] = group_sizes.astype(int)
            frame["seed_target_aliases"] = frame["pair_id"].map(alias_map)
            frame["_target_gene_exact"] = frame.apply(
                lambda row: int(
                    bool(_target_key(row.get("target_gene_name")))
                    and _target_key(row.get("target_name"))
                    == _target_key(row.get("target_gene_name"))
                ),
                axis=1,
            )
            frame = (
                frame.sort_values(
                    ["pair_id", "_target_gene_exact", "target_name"],
                    ascending=[True, False, True],
                )
                .drop_duplicates("pair_id", keep="first")
                .drop(columns=["_target_gene_exact"])
            )
        else:
            frame["seed_rows_collapsed"] = 1
            frame["seed_target_aliases"] = frame["target_name"].map(_clean)
        frames.append(frame)
    if not frames:
        raise ValueError("No seed cohorts were configured")
    return pd.concat(frames, ignore_index=True, sort=False).fillna("")


def _pair_indices(seed):
    by_db_gene = defaultdict(set)
    by_db_uniprot = defaultdict(set)
    by_db_target_name = defaultdict(set)
    by_db = defaultdict(set)
    pair_by_id = {}
    for row in seed.to_dict("records"):
        pair_id = row["pair_id"]
        pair_by_id[pair_id] = row
        dbid = _clean(row.get("drugbank_id"))
        by_db[dbid].add(pair_id)
        gene_values = {
            _target_key(row.get("target_name")),
            _target_key(row.get("target_gene_name")),
        }
        for gene in gene_values:
            if gene:
                by_db_gene[(dbid, gene)].add(pair_id)
        uniprot = _clean(row.get("target_uniprot_id")).upper()
        if uniprot:
            by_db_uniprot[(dbid, uniprot)].add(pair_id)
        target_db_name = _name_key(row.get("target_drugbank_name"))
        if target_db_name:
            by_db_target_name[(dbid, target_db_name)].add(pair_id)
    return {
        "by_db_gene": by_db_gene,
        "by_db_uniprot": by_db_uniprot,
        "by_db_target_name": by_db_target_name,
        "by_db": by_db,
        "pair_by_id": pair_by_id,
    }


def _match_drugbank_target(indices, dbid, gene, uniprot, target_name):
    matches = set()
    match_methods = []
    if uniprot:
        found = indices["by_db_uniprot"].get((dbid, uniprot.upper()), set())
        if found:
            matches.update(found)
            match_methods.append("drugbank_id+uniprot")
    if gene:
        found = indices["by_db_gene"].get((dbid, _target_key(gene)), set())
        if found:
            matches.update(found)
            match_methods.append("drugbank_id+gene_symbol")
    if target_name:
        found = indices["by_db_target_name"].get((dbid, _name_key(target_name)), set())
        if found:
            matches.update(found)
            match_methods.append("drugbank_id+target_name")
    return matches, "+".join(match_methods)


def _evidence_record(
    pair,
    source,
    evidence_type,
    source_tier,
    match_method,
    action_raw="",
    action_normalized="",
    core4_action="",
    source_drug_id="",
    source_target_id="",
    source_drug_name="",
    source_target_name="",
    known_action="",
    evidence_text="",
):
    return {
        "cohort": pair["cohort"],
        "pair_id": pair["pair_id"],
        "drug_name": pair["drug_name"],
        "target_name": pair["target_name"],
        "drugbank_id": pair.get("drugbank_id", ""),
        "target_uniprot_id": pair.get("target_uniprot_id", ""),
        "target_gene_name": pair.get("target_gene_name", ""),
        "legacy_action": pair.get("legacy_action", ""),
        "source": source,
        "evidence_type": evidence_type,
        "source_tier": source_tier,
        "direct_pair_action": source_tier in {DIRECT_HIGH, DIRECT_LOW},
        "match_method": match_method,
        "source_drug_id": source_drug_id,
        "source_target_id": source_target_id,
        "source_drug_name": source_drug_name,
        "source_target_name": source_target_name,
        "action_raw": action_raw,
        "action_normalized": action_normalized,
        "core4_action": core4_action,
        "known_action": known_action,
        "evidence_text": evidence_text,
    }


def extract_drugbank_evidence(xml_path, seed, logger=None):
    """Stream DrugBank and return evidence, metadata, and matched pair ids."""

    path = Path(xml_path).expanduser()
    if not path.exists():
        raise FileNotFoundError("DrugBank XML not found: %s" % path)
    indices = _pair_indices(seed)
    requested = set(indices["by_db"])
    evidence = []
    pair_associations = set()
    metadata = {}
    seen_drugs = 0
    for _, drug in ET.iterparse(str(path), events=("end",)):
        if drug.tag != DRUGBANK_NS + "drug":
            continue
        # DrugBank also nests compact <drug> stubs inside interactions and
        # pathways. Those stubs can reuse a requested DrugBank ID but do not
        # contain the full record, and must not overwrite aliases/cross-links.
        if not _has_explicit_primary_drugbank_id(drug):
            drug.clear()
            continue
        dbid = _primary_drugbank_id(drug)
        if dbid not in requested:
            drug.clear()
            continue
        seen_drugs += 1
        drug_name = _direct_text(drug, "name")
        aliases = [drug_name]
        synonyms = drug.find(DRUGBANK_NS + "synonyms")
        if synonyms is not None:
            aliases.extend(
                _clean(item.text)
                for item in synonyms.findall(DRUGBANK_NS + "synonym")
                if _clean(item.text)
            )
        external_ids = defaultdict(set)
        external = drug.find(DRUGBANK_NS + "external-identifiers")
        if external is not None:
            for item in external.findall(DRUGBANK_NS + "external-identifier"):
                resource = _direct_text(item, "resource")
                identifier = _direct_text(item, "identifier")
                if resource and identifier:
                    external_ids[_name_key(resource)].add(identifier)
        metadata[dbid] = {
            "drugbank_name": drug_name,
            "aliases": sorted(set(filter(None, aliases))),
            "external_ids": {
                key: sorted(values) for key, values in sorted(external_ids.items())
            },
        }

        targets = drug.find(DRUGBANK_NS + "targets")
        if targets is None:
            drug.clear()
            continue
        for target in targets.findall(DRUGBANK_NS + "target"):
            target_name = _direct_text(target, "name")
            known_action = _name_key(_direct_text(target, "known-action"))
            polypeptide = target.find(DRUGBANK_NS + "polypeptide")
            uniprot = _clean(polypeptide.attrib.get("id", "")) if polypeptide is not None else ""
            gene = _direct_text(polypeptide, "gene-name") if polypeptide is not None else ""
            pair_ids, match_method = _match_drugbank_target(
                indices, dbid, gene, uniprot, target_name
            )
            if not pair_ids:
                continue
            pair_associations.update(pair_ids)
            actions = target.find(DRUGBANK_NS + "actions")
            raw_actions = []
            if actions is not None:
                raw_actions = [
                    _clean(item.text)
                    for item in actions.findall(DRUGBANK_NS + "action")
                    if _clean(item.text)
                ]
            if not raw_actions:
                for pair_id in pair_ids:
                    evidence.append(
                        _evidence_record(
                            indices["pair_by_id"][pair_id],
                            source="DrugBank",
                            evidence_type="target_association_without_action",
                            source_tier=ASSOCIATION_ONLY,
                            match_method=match_method,
                            source_drug_id=dbid,
                            source_target_id=uniprot,
                            source_drug_name=drug_name,
                            source_target_name=gene or target_name,
                            known_action=known_action,
                        )
                    )
                continue
            for raw_action in raw_actions:
                detailed, core = normalize_action_terms(raw_action)
                tier = DIRECT_HIGH if known_action == "yes" else DIRECT_LOW
                detailed_values = detailed or [""]
                core_values = core or [""]
                for pair_id in pair_ids:
                    for detailed_action in detailed_values:
                        compatible_core = (
                            detailed_action
                            if detailed_action in CORE4_ACTIONS
                            else (
                                "agonist"
                                if detailed_action in {"partial_agonist", "co_agonist"}
                                else ""
                            )
                        )
                        # Retain a generic parsed core even when there is no
                        # detailed term (mainly defensive for unusual source text).
                        if not compatible_core and len(core_values) == 1:
                            compatible_core = core_values[0]
                        evidence.append(
                            _evidence_record(
                                indices["pair_by_id"][pair_id],
                                source="DrugBank",
                                evidence_type="target_action",
                                source_tier=tier,
                                match_method=match_method,
                                action_raw=raw_action,
                                action_normalized=detailed_action,
                                core4_action=compatible_core,
                                source_drug_id=dbid,
                                source_target_id=uniprot,
                                source_drug_name=drug_name,
                                source_target_name=gene or target_name,
                                known_action=known_action,
                                evidence_text=target_name,
                            )
                        )
        drug.clear()
    if logger:
        logger.info(
            "DrugBank: found %d/%d requested drugs and matched %d/%d seed pairs",
            seen_drugs,
            len(requested),
            len(pair_associations),
            len(seed),
        )
    return evidence, metadata, pair_associations


def _build_drug_alias_indices(seed, drugbank_metadata):
    by_chembl = defaultdict(set)
    exact = defaultdict(set)
    relaxed = defaultdict(set)
    dbids = set(seed["drugbank_id"].map(_clean))
    for dbid in dbids:
        meta = drugbank_metadata.get(dbid, {})
        aliases = set(meta.get("aliases", []))
        subset = seed.loc[seed["drugbank_id"].map(_clean) == dbid]
        for column in ("drug_name", "drugbank_name", "matched_alias"):
            if column in subset.columns:
                aliases.update(subset[column].map(_clean).tolist())
        for alias in filter(None, aliases):
            exact[_name_key(alias)].add(dbid)
            relaxed[_relaxed_key(alias)].add(dbid)
        for resource, identifiers in meta.get("external_ids", {}).items():
            if "chembl" in resource:
                for identifier in identifiers:
                    by_chembl[_clean(identifier).upper()].add(dbid)
    return by_chembl, exact, relaxed


def _match_open_targets_drugs(row, by_chembl, exact, relaxed):
    ot_id = _clean(row.get("drugId")).upper()
    cross_reference_candidates = set(by_chembl.get(ot_id, set()))
    name_candidates = set()
    name_methods = []
    names = []
    for column in ("prefName", "synonyms", "tradeNames"):
        names.extend(_iter_values(row.get(column)))
    for name in names:
        exact_matches = exact.get(_name_key(name), set())
        if len(exact_matches) == 1:
            name_candidates.update(exact_matches)
            name_methods.append("unique_exact_drug_name")
            continue
        relaxed_matches = relaxed.get(_relaxed_key(name), set())
        if len(relaxed_matches) == 1:
            name_candidates.update(relaxed_matches)
            name_methods.append("unique_relaxed_drug_name")

    if len(cross_reference_candidates) == 1:
        return cross_reference_candidates, "drugbank_chembl_cross_reference"
    if len(cross_reference_candidates) > 1:
        narrowed = cross_reference_candidates.intersection(name_candidates)
        if len(narrowed) == 1:
            return (
                narrowed,
                "drugbank_chembl_cross_reference+unique_drug_name_disambiguation",
            )
        return set(), "ambiguous_drugbank_chembl_cross_reference"
    if len(name_candidates) == 1:
        return name_candidates, "+".join(sorted(set(name_methods)))
    return set(), "ambiguous_or_unmatched_drug_name"


def extract_open_targets_evidence(parquet_paths, seed, drugbank_metadata, logger=None):
    """Extract pair-specific Open Targets mechanisms from local parquet files."""

    indices = _pair_indices(seed)
    by_chembl, exact, relaxed = _build_drug_alias_indices(seed, drugbank_metadata)
    evidence = []
    seen = set()
    matched_pairs = set()
    columns = [
        "drugId",
        "targetId",
        "approvedSymbol",
        "targetName",
        "prefName",
        "synonyms",
        "tradeNames",
        "mechanismOfAction",
    ]
    for raw_path in parquet_paths:
        path = Path(raw_path).expanduser()
        if not path.exists():
            raise FileNotFoundError("Open Targets parquet not found: %s" % path)
        frame = pd.read_parquet(path, columns=columns)
        for row in frame.to_dict("records"):
            mechanism = _clean(row.get("mechanismOfAction"))
            if not mechanism:
                continue
            dbids, drug_match = _match_open_targets_drugs(
                row, by_chembl, exact, relaxed
            )
            if not dbids:
                continue
            symbol = _target_key(row.get("approvedSymbol"))
            target_name = _name_key(row.get("targetName"))
            pair_ids = set()
            target_methods = []
            for dbid in dbids:
                if symbol:
                    found = indices["by_db_gene"].get((dbid, symbol), set())
                    if found:
                        pair_ids.update(found)
                        target_methods.append("approved_symbol")
                if target_name:
                    found = indices["by_db_target_name"].get((dbid, target_name), set())
                    if found:
                        pair_ids.update(found)
                        target_methods.append("target_name")
            if not pair_ids:
                continue
            key = (
                _clean(row.get("drugId")),
                _clean(row.get("targetId")),
                mechanism,
                tuple(sorted(pair_ids)),
            )
            if key in seen:
                continue
            seen.add(key)
            matched_pairs.update(pair_ids)
            detailed, core = normalize_action_terms(mechanism)
            detailed_values = detailed or [""]
            for pair_id in pair_ids:
                for detailed_action in detailed_values:
                    compatible_core = (
                        detailed_action
                        if detailed_action in CORE4_ACTIONS
                        else (
                            "agonist"
                            if detailed_action in {"partial_agonist", "co_agonist"}
                            else ""
                        )
                    )
                    if not compatible_core and len(core) == 1:
                        compatible_core = core[0]
                    evidence.append(
                        _evidence_record(
                            indices["pair_by_id"][pair_id],
                            source="Open Targets",
                            evidence_type="target_mechanism",
                            source_tier=DIRECT_HIGH,
                            match_method="+".join(
                                filter(
                                    None,
                                    [
                                        drug_match,
                                        "+".join(sorted(set(target_methods))),
                                    ],
                                )
                            ),
                            action_raw=mechanism,
                            action_normalized=detailed_action,
                            core4_action=compatible_core,
                            source_drug_id=_clean(row.get("drugId")),
                            source_target_id=_clean(row.get("targetId")),
                            source_drug_name=_clean(row.get("prefName")),
                            source_target_name=_clean(row.get("approvedSymbol"))
                            or _clean(row.get("targetName")),
                            evidence_text=mechanism,
                        )
                    )
    if logger:
        logger.info(
            "Open Targets: matched mechanisms for %d/%d seed pairs",
            len(matched_pairs),
            len(seed),
        )
    return evidence, matched_pairs


def _seed_name_target_indices(seed, drugbank_metadata):
    exact_drug = defaultdict(set)
    relaxed_drug = defaultdict(set)
    target_by_db = defaultdict(lambda: defaultdict(set))
    pair_by_id = {}
    for row in seed.to_dict("records"):
        pair_by_id[row["pair_id"]] = row
        dbid = _clean(row["drugbank_id"])
        aliases = {
            _clean(row.get("drug_name")),
            _clean(row.get("drugbank_name")),
            _clean(row.get("matched_alias")),
        }
        aliases.update(drugbank_metadata.get(dbid, {}).get("aliases", []))
        for alias in filter(None, aliases):
            exact_drug[_name_key(alias)].add(dbid)
            relaxed_drug[_relaxed_key(alias)].add(dbid)
        for value in (
            row.get("target_name"),
            row.get("target_gene_name"),
            row.get("target_uniprot_id"),
        ):
            key = _target_key(value)
            if key:
                target_by_db[dbid][key].add(row["pair_id"])
    return exact_drug, relaxed_drug, target_by_db, pair_by_id


def extract_repurposing_hub_evidence(tsv_path, seed, drugbank_metadata, logger=None):
    """Extract lower-tier action support from the Broad Repurposing Hub."""

    path = Path(tsv_path).expanduser()
    if not path.exists():
        raise FileNotFoundError("Repurposing Hub file not found: %s" % path)
    exact, relaxed, targets, pair_by_id = _seed_name_target_indices(
        seed, drugbank_metadata
    )
    frame = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    required = {"Name", "MOA", "Target"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            "Repurposing Hub file lacks columns: %s" % ", ".join(sorted(missing))
        )
    evidence = []
    seen = set()
    matched_pairs = set()
    for row in frame.to_dict("records"):
        name = _clean(row["Name"])
        dbids = exact.get(_name_key(name), set())
        method = "unique_exact_drug_name"
        if len(dbids) != 1:
            dbids = relaxed.get(_relaxed_key(name), set())
            method = "unique_relaxed_drug_name"
        if len(dbids) != 1:
            continue
        dbid = next(iter(dbids))
        raw_targets = [
            item.strip()
            for item in re.split(r"[,;|]", _clean(row["Target"]))
            if item.strip()
        ]
        pair_ids = set()
        for target in raw_targets:
            pair_ids.update(targets[dbid].get(_target_key(target), set()))
        if not pair_ids:
            continue
        moa = _clean(row["MOA"])
        detailed, core = normalize_action_terms(moa)
        detailed_values = detailed or [""]
        for pair_id in pair_ids:
            matched_pairs.add(pair_id)
            for detailed_action in detailed_values:
                compatible_core = (
                    detailed_action
                    if detailed_action in CORE4_ACTIONS
                    else (
                        "agonist"
                        if detailed_action in {"partial_agonist", "co_agonist"}
                        else ""
                    )
                )
                if not compatible_core and len(core) == 1:
                    compatible_core = core[0]
                key = (pair_id, moa, detailed_action, compatible_core)
                if key in seen:
                    continue
                seen.add(key)
                evidence.append(
                    _evidence_record(
                        pair_by_id[pair_id],
                        source="Broad Repurposing Hub",
                        evidence_type="drug_moa_with_target_list",
                        source_tier=SUPPORTING,
                        match_method=method + "+target_symbol",
                        action_raw=moa,
                        action_normalized=detailed_action,
                        core4_action=compatible_core,
                        source_drug_name=name,
                        source_target_name=", ".join(raw_targets),
                        evidence_text=moa,
                    )
                )
    if logger:
        logger.info(
            "Repurposing Hub: matched supporting evidence for %d/%d seed pairs",
            len(matched_pairs),
            len(seed),
        )
    return evidence, matched_pairs


def extract_primekg_associations(csv_path, seed, logger=None, chunksize=200000):
    """Extract PrimeKG association corroboration without creating labels."""

    path = Path(csv_path).expanduser()
    if not path.exists():
        raise FileNotFoundError("PrimeKG CSV not found: %s" % path)
    indices = _pair_indices(seed)
    requested = set(indices["by_db"])
    evidence = []
    matched_pairs = set()
    seen = set()
    usecols = [
        "relation",
        "display_relation",
        "x_id",
        "x_type",
        "x_name",
        "x_source",
        "y_id",
        "y_type",
        "y_name",
        "y_source",
    ]
    for chunk in pd.read_csv(
        path, usecols=usecols, dtype=str, chunksize=chunksize
    ):
        chunk = chunk.fillna("")
        for row in chunk.to_dict("records"):
            x_type = _name_key(row["x_type"])
            y_type = _name_key(row["y_type"])
            if x_type == "drug" and y_type in {"gene/protein", "gene", "protein"}:
                dbid = _clean(row["x_id"])
                target_id = _clean(row["y_id"])
                target_name = _clean(row["y_name"])
                drug_name = _clean(row["x_name"])
            elif y_type == "drug" and x_type in {"gene/protein", "gene", "protein"}:
                dbid = _clean(row["y_id"])
                target_id = _clean(row["x_id"])
                target_name = _clean(row["x_name"])
                drug_name = _clean(row["y_name"])
            else:
                continue
            if dbid not in requested:
                continue
            pair_ids = set()
            target_key = _target_key(target_name)
            if target_key:
                pair_ids.update(indices["by_db_gene"].get((dbid, target_key), set()))
            target_id_key = _target_key(target_id)
            if target_id_key:
                pair_ids.update(
                    indices["by_db_gene"].get((dbid, target_id_key), set())
                )
            if not pair_ids:
                continue
            for pair_id in pair_ids:
                key = (
                    pair_id,
                    _clean(row["relation"]),
                    _clean(row["display_relation"]),
                )
                if key in seen:
                    continue
                seen.add(key)
                matched_pairs.add(pair_id)
                evidence.append(
                    _evidence_record(
                        indices["pair_by_id"][pair_id],
                        source="PrimeKG",
                        evidence_type="drug_protein_association",
                        source_tier=ASSOCIATION_ONLY,
                        match_method="drugbank_id+target_symbol",
                        source_drug_id=dbid,
                        source_target_id=target_id,
                        source_drug_name=drug_name,
                        source_target_name=target_name,
                        evidence_text="%s | %s"
                        % (
                            _clean(row["relation"]),
                            _clean(row["display_relation"]),
                        ),
                    )
                )
    if logger:
        logger.info(
            "PrimeKG: corroborated %d/%d seed pairs (association only)",
            len(matched_pairs),
            len(seed),
        )
    return evidence, matched_pairs


def build_consensus(seed, evidence_frame):
    """Build conservative gold/silver/conflict consensus for every seed pair."""

    grouped = {
        pair_id: group.copy()
        for pair_id, group in evidence_frame.groupby("pair_id", sort=False)
    }
    rows = []
    for pair in seed.to_dict("records"):
        evidence = grouped.get(pair["pair_id"], pd.DataFrame())
        if evidence.empty:
            high_labels = set()
            lower_labels = set()
            all_labels = set()
            action_sources = set()
            sources = set()
        else:
            core = evidence.loc[
                evidence["core4_action"].map(_clean) != ""
            ].copy()
            high_labels = set(
                core.loc[
                    core["source_tier"] == DIRECT_HIGH, "core4_action"
                ].map(_clean)
            )
            lower_labels = set(
                core.loc[
                    core["source_tier"].isin([DIRECT_LOW, SUPPORTING]),
                    "core4_action",
                ].map(_clean)
            )
            all_labels = set(core["core4_action"].map(_clean))
            action_sources = set(core["source"].map(_clean))
            sources = set(evidence["source"].map(_clean))
        high_labels.discard("")
        lower_labels.discard("")
        all_labels.discard("")
        action_sources.discard("")
        sources.discard("")

        consensus = ""
        status = "unresolved"
        tier = ""
        reason = "no mappable pair-specific core-four action evidence"
        if len(high_labels) > 1:
            status = "conflict"
            tier = "excluded"
            reason = "conflicting direct high-tier labels"
        elif len(high_labels) == 1:
            consensus = next(iter(high_labels))
            conflicting_lower = lower_labels.difference({consensus})
            if conflicting_lower:
                status = "gold_with_lower_tier_conflict"
                tier = "gold"
                reason = (
                    "one direct high-tier label; conflicting lower-tier evidence retained"
                )
            else:
                status = "gold"
                tier = "gold"
                reason = "one unambiguous direct high-tier label"
        elif len(lower_labels) > 1:
            status = "conflict"
            tier = "excluded"
            reason = "conflicting lower-tier labels without high-tier resolution"
        elif len(lower_labels) == 1:
            consensus = next(iter(lower_labels))
            status = "silver"
            tier = "silver"
            reason = "one unambiguous lower-tier pair-scoped label"

        rows.append(
            {
                "cohort": pair["cohort"],
                "pair_id": pair["pair_id"],
                "drug_name": pair["drug_name"],
                "target_name": pair["target_name"],
                "drugbank_id": pair.get("drugbank_id", ""),
                "target_uniprot_id": pair.get("target_uniprot_id", ""),
                "target_gene_name": pair.get("target_gene_name", ""),
                "legacy_action": pair.get("legacy_action", ""),
                "seed_rows_collapsed": int(pair.get("seed_rows_collapsed", 1)),
                "seed_target_aliases": pair.get("seed_target_aliases", ""),
                "consensus_action": consensus,
                "consensus_status": status,
                "ground_truth_tier": tier,
                "consensus_reason": reason,
                "direct_high_labels": "|".join(sorted(high_labels)),
                "lower_tier_labels": "|".join(sorted(lower_labels)),
                "all_evidence_labels": "|".join(sorted(all_labels)),
                "action_sources": "|".join(sorted(action_sources)),
                "all_sources": "|".join(sorted(sources)),
                "n_action_sources": len(action_sources),
                "n_evidence_rows": 0 if evidence.empty else len(evidence),
                "legacy_agrees": bool(
                    consensus and consensus == _clean(pair.get("legacy_action"))
                ),
            }
        )
    return pd.DataFrame(rows)


def build_drugmap7_consensus(seed, evidence_frame):
    """Build a conservative DrugMAP-aligned seven-class consensus."""

    mapped = evidence_frame.copy()
    mapped["drugmap7_action"] = mapped.apply(
        lambda row: drugmap7_action(
            row.get("action_normalized", ""),
            row.get("action_raw", ""),
        ),
        axis=1,
    )
    grouped = {
        pair_id: group.copy()
        for pair_id, group in mapped.groupby("pair_id", sort=False)
    }
    rows = []
    for pair in seed.to_dict("records"):
        evidence = grouped.get(pair["pair_id"], pd.DataFrame())
        if evidence.empty:
            high_raw = set()
            lower_raw = set()
            action_sources = set()
        else:
            eligible = evidence.loc[
                evidence["drugmap7_action"].map(_clean) != ""
            ]
            high_raw = set(
                eligible.loc[
                    eligible["source_tier"] == DIRECT_HIGH,
                    "drugmap7_action",
                ].map(_clean)
            )
            lower_raw = set(
                eligible.loc[
                    eligible["source_tier"].isin([DIRECT_LOW, SUPPORTING]),
                    "drugmap7_action",
                ].map(_clean)
            )
            action_sources = set(eligible["source"].map(_clean))
        high_raw.discard("")
        lower_raw.discard("")
        action_sources.discard("")
        high, high_rule = _resolve_drugmap7_classes(high_raw)
        lower, lower_rule = _resolve_drugmap7_classes(lower_raw)

        action = ""
        status = "unresolved"
        tier = ""
        reason = "no mappable DrugMAP-seven action"
        if len(high) > 1:
            status = "conflict"
            tier = "excluded"
            reason = "multiple incompatible direct high-tier DrugMAP-seven actions"
        elif len(high) == 1:
            action = next(iter(high))
            lower_conflicts = lower.difference({action})
            if lower_conflicts:
                status = "gold_with_lower_tier_conflict"
                reason = "unique direct action with lower-tier disagreement"
            else:
                status = "gold"
                reason = "unique direct high-tier DrugMAP-seven action"
            tier = "gold"
        elif len(lower) > 1:
            status = "conflict"
            tier = "excluded"
            reason = "multiple incompatible lower-tier DrugMAP-seven actions"
        elif len(lower) == 1:
            action = next(iter(lower))
            status = "silver"
            tier = "silver"
            reason = "unique lower-tier DrugMAP-seven action"

        rows.append(
            {
                "cohort": pair["cohort"],
                "pair_id": pair["pair_id"],
                "drug_name": pair["drug_name"],
                "target_name": pair["target_name"],
                "drugbank_id": pair.get("drugbank_id", ""),
                "target_uniprot_id": pair.get("target_uniprot_id", ""),
                "target_gene_name": pair.get("target_gene_name", ""),
                "legacy_action": pair.get("legacy_action", ""),
                "seed_rows_collapsed": int(pair.get("seed_rows_collapsed", 1)),
                "seed_target_aliases": pair.get("seed_target_aliases", ""),
                "action": action,
                "consensus_status": status,
                "ground_truth_tier": tier,
                "consensus_reason": reason,
                "direct_high_classes_raw": "|".join(sorted(high_raw)),
                "direct_high_classes_resolved": "|".join(sorted(high)),
                "direct_high_resolution_rule": high_rule,
                "lower_tier_classes_raw": "|".join(sorted(lower_raw)),
                "lower_tier_classes_resolved": "|".join(sorted(lower)),
                "lower_tier_resolution_rule": lower_rule,
                "action_sources": "|".join(sorted(action_sources)),
                "n_action_sources": len(action_sources),
                "legacy_agrees": bool(
                    action and action == _clean(pair.get("legacy_action"))
                ),
            }
        )
    return pd.DataFrame(rows), mapped


def _drugmap7_class_coverage(consensus, mapped_evidence):
    rows = []
    for cohort, cohort_consensus in consensus.groupby("cohort", sort=True):
        cohort_evidence = mapped_evidence.loc[
            mapped_evidence["cohort"] == cohort
        ]
        for tier in ("gold", "silver"):
            tier_rows = cohort_consensus.loc[
                cohort_consensus["ground_truth_tier"] == tier
            ]
            for action in sorted(DRUGMAP7_ACTIONS):
                action_rows = tier_rows.loc[tier_rows["action"] == action]
                evidence_rows = cohort_evidence.loc[
                    cohort_evidence["drugmap7_action"] == action
                ]
                rows.append(
                    {
                        "cohort": cohort,
                        "ground_truth_tier": tier,
                        "action": action,
                        "pair_count": int(len(action_rows)),
                        "drug_count": int(action_rows["drug_name"].nunique()),
                        "target_count": int(action_rows["target_name"].nunique()),
                        "drugbank_evidence_pairs": int(
                            evidence_rows.loc[
                                evidence_rows["source"] == "DrugBank", "pair_id"
                            ].nunique()
                        ),
                        "open_targets_evidence_pairs": int(
                            evidence_rows.loc[
                                evidence_rows["source"] == "Open Targets",
                                "pair_id",
                            ].nunique()
                        ),
                        "repurposing_hub_evidence_pairs": int(
                            evidence_rows.loc[
                                evidence_rows["source"]
                                == "Broad Repurposing Hub",
                                "pair_id",
                            ].nunique()
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _replacement_cohort(seed_cohort, consensus_cohort, tier):
    keep = consensus_cohort.loc[
        consensus_cohort["ground_truth_tier"] == tier
    ].copy()
    keep_columns = [
        "pair_id",
        "consensus_action",
        "consensus_status",
        "ground_truth_tier",
        "consensus_reason",
        "action_sources",
        "n_action_sources",
        "legacy_action",
        "legacy_agrees",
    ]
    base = seed_cohort.drop(columns=["legacy_action"], errors="ignore").copy()
    merged = base.merge(keep[keep_columns], on="pair_id", how="inner")
    merged["action"] = merged["consensus_action"]
    preferred = ["drug_name", "target_name", "action", "pair_id"]
    remaining = [column for column in merged.columns if column not in preferred]
    return merged[preferred + remaining]


def _drugmap7_replacement_cohort(seed_cohort, consensus_cohort, tier):
    keep = consensus_cohort.loc[
        consensus_cohort["ground_truth_tier"] == tier
    ].copy()
    metadata_columns = [
        "pair_id",
        "action",
        "consensus_status",
        "ground_truth_tier",
        "consensus_reason",
        "action_sources",
        "n_action_sources",
        "legacy_action",
        "legacy_agrees",
    ]
    base = seed_cohort.drop(
        columns=["action", "legacy_action"], errors="ignore"
    ).copy()
    merged = base.merge(keep[metadata_columns], on="pair_id", how="inner")
    preferred = ["drug_name", "target_name", "action", "pair_id"]
    remaining = [column for column in merged.columns if column not in preferred]
    return merged[preferred + remaining]


def _source_coverage(consensus, evidence):
    rows = []
    for cohort, cohort_consensus in consensus.groupby("cohort", sort=True):
        cohort_evidence = evidence.loc[evidence["cohort"] == cohort]
        seed_pairs = set(cohort_consensus["pair_id"])
        row = {
            "cohort": cohort,
            "seed_pairs": len(seed_pairs),
            "pairs_with_any_evidence": int(cohort_evidence["pair_id"].nunique()),
        }
        for source in (
            "DrugBank",
            "Open Targets",
            "Broad Repurposing Hub",
            "PrimeKG",
        ):
            source_key = re.sub(
                r"[^a-z0-9]+", "_", source.casefold()
            ).strip("_")
            source_rows = cohort_evidence.loc[cohort_evidence["source"] == source]
            row["pairs_%s_any_evidence" % source_key] = int(
                source_rows["pair_id"].nunique()
            )
            row["pairs_%s_core4_action" % source_key] = int(
                source_rows.loc[
                    source_rows["core4_action"].map(_clean) != "", "pair_id"
                ].nunique()
            )
            row["pairs_%s_direct_high" % source_key] = int(
                source_rows.loc[
                    (source_rows["source_tier"] == DIRECT_HIGH)
                    & (source_rows["core4_action"].map(_clean) != ""),
                    "pair_id",
                ].nunique()
            )
        for status in (
            "gold",
            "gold_with_lower_tier_conflict",
            "silver",
            "conflict",
            "unresolved",
        ):
            row["status_%s" % status] = int(
                (cohort_consensus["consensus_status"] == status).sum()
            )
        for action in sorted(CORE4_ACTIONS):
            row["gold_%s" % action] = int(
                (
                    (cohort_consensus["ground_truth_tier"] == "gold")
                    & (cohort_consensus["consensus_action"] == action)
                ).sum()
            )
        gold = cohort_consensus["ground_truth_tier"] == "gold"
        strict_gold = cohort_consensus["consensus_status"] == "gold"
        row["gold_pairs"] = int(gold.sum())
        row["strict_gold_pairs"] = int(strict_gold.sum())
        row["gold_retention_fraction"] = (
            float(gold.mean()) if len(cohort_consensus) else 0.0
        )
        row["strict_gold_retention_fraction"] = (
            float(strict_gold.mean()) if len(cohort_consensus) else 0.0
        )
        row["gold_legacy_agreement_fraction"] = (
            float(cohort_consensus.loc[gold, "legacy_agrees"].mean())
            if gold.any()
            else 0.0
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _source_agreement(evidence):
    action_evidence = evidence.loc[
        evidence["core4_action"].map(_clean) != ""
    ].copy()
    per_pair_source = (
        action_evidence.groupby(["cohort", "pair_id", "source"])["core4_action"]
        .agg(lambda values: "|".join(sorted(set(map(_clean, values)))))
        .reset_index()
    )
    sources = sorted(per_pair_source["source"].unique())
    rows = []
    for cohort in sorted(per_pair_source["cohort"].unique()):
        subset = per_pair_source.loc[per_pair_source["cohort"] == cohort]
        for index, left in enumerate(sources):
            for right in sources[index + 1 :]:
                left_rows = subset.loc[subset["source"] == left, ["pair_id", "core4_action"]]
                right_rows = subset.loc[
                    subset["source"] == right, ["pair_id", "core4_action"]
                ]
                overlap = left_rows.merge(
                    right_rows,
                    on="pair_id",
                    suffixes=("_left", "_right"),
                )
                if overlap.empty:
                    continue
                agreement = (
                    overlap["core4_action_left"] == overlap["core4_action_right"]
                )
                rows.append(
                    {
                        "cohort": cohort,
                        "source_left": left,
                        "source_right": right,
                        "overlap_pairs": len(overlap),
                        "agreement_pairs": int(agreement.sum()),
                        "conflict_pairs": int((~agreement).sum()),
                        "agreement_fraction": float(agreement.mean()),
                    }
                )
    return pd.DataFrame(rows)


def _source_manifest(source_paths):
    rows = []
    for source, paths in source_paths.items():
        for raw_path in paths:
            path = Path(raw_path).expanduser()
            stat = path.stat()
            rows.append(
                {
                    "source": source,
                    "path": str(path),
                    "size_bytes": int(stat.st_size),
                    "modified_utc": datetime.fromtimestamp(
                        stat.st_mtime, timezone.utc
                    ).isoformat(),
                }
            )
    return rows


def _frame_to_markdown(frame):
    """Render a compact Markdown table without an optional tabulate dependency."""

    if frame.empty:
        return "No rows."
    rendered = frame.copy()
    for column in rendered.columns:
        rendered[column] = rendered[column].map(
            lambda value: ("%.4f" % value).rstrip("0").rstrip(".")
            if isinstance(value, float)
            else str(value)
        )
    header = "| " + " | ".join(map(str, rendered.columns)) + " |"
    separator = "| " + " | ".join(["---"] * len(rendered.columns)) + " |"
    body = [
        "| "
        + " | ".join(
            str(value).replace("|", "\\|").replace("\n", " ")
            for value in row
        )
        + " |"
        for row in rendered.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator] + body)


def _build_markdown_report(coverage, agreement, output_dir):
    lines = [
        "# Pair-specific action ground-truth rebuild",
        "",
        "## Decision rule",
        "",
        "- Gold: exactly one core-four action supported by DrugBank target action "
        "with `known-action=yes` and/or an Open Targets target mechanism.",
        "- Silver: exactly one lower-tier core-four action, with no direct high-tier label.",
        "- Conflict: more than one eligible action at the governing evidence tier; excluded.",
        "- PrimeKG: association corroboration only and never a label source.",
        "- The original drug-level action is retained only as `legacy_action` for auditing.",
        "",
        "## Coverage",
        "",
        _frame_to_markdown(coverage),
        "",
        "## Cross-source agreement",
        "",
        _frame_to_markdown(agreement),
        "",
        "## Output use",
        "",
        "Use each cohort's `pair_action_ground_truth_gold.csv` as the primary "
        "four-class action cohort. Silver and conflict files are diagnostic and "
        "must not be mixed into the primary benchmark without a predeclared sensitivity analysis.",
        "",
        "Output directory: `%s`" % output_dir,
        "",
    ]
    return "\n".join(lines)


def _build_drugmap7_report(class_coverage, consensus, output_dir):
    summary = (
        consensus.groupby(
            ["cohort", "ground_truth_tier", "consensus_status"],
            dropna=False,
        )
        .size()
        .reset_index(name="pair_count")
    )
    lines = [
        "# DrugMAP-aligned seven-class action audit",
        "",
        "## Ontology",
        "",
        "The seven interaction classes are `inhibitor`, `ligand`, `substrate`, "
        "`allosteric_modulator`, `antagonist`, `agonist`, and `inducer`. "
        "Blocker is merged into inhibitor; binder into ligand; generic and "
        "signed allosteric modulators into allosteric_modulator; partial/inverse "
        "agonists into agonist; and activator/potentiator into inducer.",
        "",
        "Unlike the public DrugMAP preprocessing, incompatible remaining classes "
        "are never resolved by random choice. Ligand or substrate is discarded "
        "only when exactly one more specific action is available.",
        "",
        "## Consensus summary",
        "",
        _frame_to_markdown(summary),
        "",
        "## Class feasibility",
        "",
        _frame_to_markdown(class_coverage),
        "",
        "## Interpretation",
        "",
        "A class should enter formal repeated entity-disjoint evaluation only "
        "when its pair, drug, and target counts support every requested fold. "
        "Rare classes remain valid ground-truth records but are not automatically "
        "valid supervised benchmark classes.",
        "",
        "Output directory: `%s`" % output_dir,
        "",
    ]
    return "\n".join(lines)


def _write_replacement_configs(cfg, cohort_outputs, output_dir):
    """Write full evaluation configs that point at rebuilt pair tables."""

    base_configs = cfg.get("evaluation_base_configs", {})
    if not base_configs:
        return {}
    config_dir = ensure_dir(output_dir / "evaluation_configs")
    evaluation_root = Path(
        cfg.get(
            "evaluation_output_root",
            output_dir.parent / "pair_specific_evaluation",
        )
    ).expanduser()
    outputs = {}
    for cohort_name, base_path_raw in base_configs.items():
        if cohort_name not in cohort_outputs:
            continue
        base_path = Path(base_path_raw).expanduser()
        with base_path.open("r", encoding="utf-8") as handle:
            base = json.load(handle)
        config_specs = [
            ("gold", "gold_path", "core4"),
            ("gold_strict", "strict_gold_path", "core4"),
            ("drugmap7_gold", "drugmap7_gold_path", "drugmap7"),
            (
                "drugmap7_gold_strict",
                "drugmap7_strict_gold_path",
                "drugmap7",
            ),
            ("drugmap3_gold", "drugmap7_gold_path", "drugmap3"),
            (
                "drugmap3_gold_strict",
                "drugmap7_strict_gold_path",
                "drugmap3",
            ),
        ]
        if cohort_name == "development":
            config_specs.append(
                (
                    "drugmap6_exploratory_gold",
                    "drugmap7_gold_path",
                    "drugmap6",
                )
            )
        for tier, pair_path_key, label_space in config_specs:
            updated = json.loads(json.dumps(base))
            updated["project_name"] = "%s_pair_specific_%s" % (
                updated.get("project_name", cohort_name),
                tier,
            )
            run_root = evaluation_root / cohort_name / tier
            updated.setdefault("paths", {})["reports_dir"] = str(
                run_root / "reports"
            )
            updated["paths"]["logs_dir"] = str(run_root / "logs")
            pair_path = cohort_outputs[cohort_name][pair_path_key]
            for section_name in ("pair_evaluation", "pair_multimodal"):
                section = updated.get(section_name)
                if not isinstance(section, dict):
                    continue
                section["pair_table_path"] = pair_path
                section["cohort_name"] = "%s_pair_specific_%s" % (
                    cohort_name,
                    tier,
                )
                section["reuse_completed_results"] = False
                if label_space in {"drugmap7", "drugmap3", "drugmap6"}:
                    if label_space == "drugmap3":
                        allowed_classes = [
                            "agonist",
                            "antagonist",
                            "inhibitor",
                        ]
                    elif label_space == "drugmap6":
                        allowed_classes = sorted(
                            DRUGMAP7_ACTIONS.difference({"substrate"})
                        )
                    else:
                        allowed_classes = sorted(DRUGMAP7_ACTIONS)
                    section["label_tasks"] = {
                        "%s_action" % label_space: {
                            "label_column": "action",
                            "min_class_count": 5,
                            "allowed_classes": allowed_classes,
                        }
                    }
            kag = updated.get("kag_evaluation")
            if isinstance(kag, dict):
                kag["pair_table_path"] = pair_path
                kag["output_dir"] = str(run_root / "reports" / "kag")
            updated["pair_action_ground_truth_provenance"] = {
                "pair_table_path": pair_path,
                "tier": tier,
                "label_space": label_space,
                "consensus_manifest": str(
                    output_dir / "pair_action_ground_truth_manifest.json"
                ),
                "legacy_label_used_for_consensus": False,
            }
            output_path = config_dir / ("%s_%s.json" % (cohort_name, tier))
            write_json(output_path, updated)
            outputs["%s_%s" % (cohort_name, tier)] = str(output_path)
    return outputs


def build_pair_action_ground_truth(config, logger):
    """Run the complete pair-specific action ground-truth rebuild."""

    cfg = config.get("pair_action_ground_truth", {})
    output_dir = ensure_dir(Path(cfg["output_dir"]).expanduser())
    cohort_paths = cfg.get("cohorts", {})
    seed = _load_seed_cohorts(cohort_paths)
    logger.info(
        "Loaded %d seed rows across %d cohorts",
        len(seed),
        seed["cohort"].nunique(),
    )

    drugbank_path = cfg["drugbank_xml"]
    open_targets_paths = cfg.get("open_targets_files", [])
    repurposing_path = cfg["repurposing_hub_tsv"]
    primekg_path = cfg["primekg_csv"]

    evidence_rows, drugbank_metadata, _ = extract_drugbank_evidence(
        drugbank_path, seed, logger
    )
    open_targets_rows, _ = extract_open_targets_evidence(
        open_targets_paths, seed, drugbank_metadata, logger
    )
    repurposing_rows, _ = extract_repurposing_hub_evidence(
        repurposing_path, seed, drugbank_metadata, logger
    )
    primekg_rows, _ = extract_primekg_associations(primekg_path, seed, logger)
    evidence_rows.extend(open_targets_rows)
    evidence_rows.extend(repurposing_rows)
    evidence_rows.extend(primekg_rows)

    evidence_columns = list(
        _evidence_record(
            seed.iloc[0].to_dict(),
            source="",
            evidence_type="",
            source_tier="",
            match_method="",
        ).keys()
    )
    evidence = pd.DataFrame(evidence_rows, columns=evidence_columns).fillna("")
    evidence = evidence.drop_duplicates().sort_values(
        ["cohort", "pair_id", "source", "source_tier", "core4_action", "action_raw"]
    )
    consensus = build_consensus(seed, evidence)
    drugmap7_consensus, mapped_evidence = build_drugmap7_consensus(
        seed, evidence
    )
    drugmap7_coverage = _drugmap7_class_coverage(
        drugmap7_consensus, mapped_evidence
    )
    coverage = _source_coverage(consensus, evidence)
    agreement = _source_agreement(evidence)

    evidence.to_csv(output_dir / "pair_action_evidence.csv", index=False)
    consensus.to_csv(output_dir / "pair_action_consensus_all.csv", index=False)
    consensus.loc[consensus["ground_truth_tier"] == "gold"].to_csv(
        output_dir / "pair_action_consensus_gold.csv", index=False
    )
    consensus.loc[consensus["ground_truth_tier"] == "silver"].to_csv(
        output_dir / "pair_action_consensus_silver.csv", index=False
    )
    consensus.loc[consensus["consensus_status"] == "conflict"].to_csv(
        output_dir / "pair_action_conflicts.csv", index=False
    )
    consensus.loc[consensus["consensus_status"] == "unresolved"].to_csv(
        output_dir / "pair_action_unresolved.csv", index=False
    )
    coverage.to_csv(output_dir / "source_coverage.csv", index=False)
    agreement.to_csv(output_dir / "source_agreement.csv", index=False)
    drugmap7_consensus.to_csv(
        output_dir / "pair_action_drugmap7_consensus_all.csv", index=False
    )
    drugmap7_consensus.loc[
        drugmap7_consensus["ground_truth_tier"] == "gold"
    ].to_csv(
        output_dir / "pair_action_drugmap7_consensus_gold.csv", index=False
    )
    drugmap7_consensus.loc[
        drugmap7_consensus["ground_truth_tier"] == "silver"
    ].to_csv(
        output_dir / "pair_action_drugmap7_consensus_silver.csv", index=False
    )
    drugmap7_consensus.loc[
        drugmap7_consensus["consensus_status"] == "conflict"
    ].to_csv(output_dir / "pair_action_drugmap7_conflicts.csv", index=False)
    drugmap7_coverage.to_csv(
        output_dir / "pair_action_drugmap7_class_coverage.csv", index=False
    )
    mapping_rows = [
        {
            "detailed_action": detailed,
            "drugmap7_action": drugmap7_action(detailed, ""),
        }
        for detailed in (
            "activator",
            "agonist",
            "antagonist",
            "binder",
            "blocker",
            "co_agonist",
            "inducer",
            "inhibitor",
            "inverse_agonist",
            "modulator",
            "negative_allosteric_modulator",
            "partial_agonist",
            "positive_allosteric_modulator",
            "potentiator",
            "substrate",
            "suppressor",
        )
    ]
    pd.DataFrame(mapping_rows).to_csv(
        output_dir / "pair_action_drugmap7_ontology_mapping.csv", index=False
    )

    cohort_outputs = {}
    for cohort_name, seed_cohort in seed.groupby("cohort", sort=True):
        cohort_dir = ensure_dir(output_dir / cohort_name)
        cohort_consensus = consensus.loc[consensus["cohort"] == cohort_name]
        gold = _replacement_cohort(seed_cohort, cohort_consensus, "gold")
        silver = _replacement_cohort(seed_cohort, cohort_consensus, "silver")
        gold_path = cohort_dir / "pair_action_ground_truth_gold.csv"
        strict_gold_path = (
            cohort_dir / "pair_action_ground_truth_gold_strict.csv"
        )
        silver_path = cohort_dir / "pair_action_ground_truth_silver.csv"
        gold.to_csv(gold_path, index=False)
        strict_gold = gold.loc[gold["consensus_status"] == "gold"].copy()
        strict_gold.to_csv(strict_gold_path, index=False)
        silver.to_csv(silver_path, index=False)
        cohort_outputs[cohort_name] = {
            "gold_path": str(gold_path),
            "strict_gold_path": str(strict_gold_path),
            "silver_path": str(silver_path),
            "seed_pairs": int(len(seed_cohort)),
            "gold_pairs": int(len(gold)),
            "strict_gold_pairs": int(len(strict_gold)),
            "silver_pairs": int(len(silver)),
        }
        drugmap7_cohort = drugmap7_consensus.loc[
            drugmap7_consensus["cohort"] == cohort_name
        ]
        drugmap7_gold = _drugmap7_replacement_cohort(
            seed_cohort, drugmap7_cohort, "gold"
        )
        drugmap7_strict = drugmap7_gold.loc[
            drugmap7_gold["consensus_status"] == "gold"
        ].copy()
        drugmap7_silver = _drugmap7_replacement_cohort(
            seed_cohort, drugmap7_cohort, "silver"
        )
        drugmap7_gold_path = (
            cohort_dir / "pair_action_ground_truth_drugmap7_gold.csv"
        )
        drugmap7_strict_path = (
            cohort_dir / "pair_action_ground_truth_drugmap7_gold_strict.csv"
        )
        drugmap7_silver_path = (
            cohort_dir / "pair_action_ground_truth_drugmap7_silver.csv"
        )
        drugmap7_gold.to_csv(drugmap7_gold_path, index=False)
        drugmap7_strict.to_csv(drugmap7_strict_path, index=False)
        drugmap7_silver.to_csv(drugmap7_silver_path, index=False)
        cohort_outputs[cohort_name].update(
            {
                "drugmap7_gold_path": str(drugmap7_gold_path),
                "drugmap7_strict_gold_path": str(drugmap7_strict_path),
                "drugmap7_silver_path": str(drugmap7_silver_path),
                "drugmap7_gold_pairs": int(len(drugmap7_gold)),
                "drugmap7_strict_gold_pairs": int(len(drugmap7_strict)),
                "drugmap7_silver_pairs": int(len(drugmap7_silver)),
            }
        )

    evaluation_configs = _write_replacement_configs(
        cfg, cohort_outputs, output_dir
    )
    source_paths = {
        "DrugBank": [drugbank_path],
        "Open Targets": open_targets_paths,
        "Broad Repurposing Hub": [repurposing_path],
        "PrimeKG": [primekg_path],
        "seed cohort": list(cohort_paths.values()),
    }
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "core4_actions": sorted(CORE4_ACTIONS),
        "drugmap7_actions": sorted(DRUGMAP7_ACTIONS),
        "policy": {
            "gold_sources": [
                "DrugBank target action with known-action=yes",
                "Open Targets target-level mechanismOfAction",
            ],
            "silver_sources": [
                "DrugBank target action without known-action=yes",
                "Broad Repurposing Hub drug MOA with target list",
            ],
            "association_only_sources": ["PrimeKG"],
            "conflicts_excluded": True,
            "legacy_label_used_for_consensus": False,
        },
        "source_files": _source_manifest(source_paths),
        "outputs": cohort_outputs,
        "evaluation_configs": evaluation_configs,
    }
    write_json(output_dir / "pair_action_ground_truth_manifest.json", manifest)
    write_text(
        output_dir / "pair_action_ground_truth_report.md",
        _build_markdown_report(coverage, agreement, output_dir),
    )
    write_text(
        output_dir / "pair_action_drugmap7_report.md",
        _build_drugmap7_report(
            drugmap7_coverage, drugmap7_consensus, output_dir
        ),
    )
    logger.info(
        "Ground truth complete: %d gold, %d silver, %d conflicts, %d unresolved",
        int((consensus["ground_truth_tier"] == "gold").sum()),
        int((consensus["ground_truth_tier"] == "silver").sum()),
        int((consensus["consensus_status"] == "conflict").sum()),
        int((consensus["consensus_status"] == "unresolved").sum()),
    )
    logger.info(
        "DrugMAP-seven ground truth: %d gold, %d silver, %d conflicts, %d unresolved",
        int((drugmap7_consensus["ground_truth_tier"] == "gold").sum()),
        int((drugmap7_consensus["ground_truth_tier"] == "silver").sum()),
        int((drugmap7_consensus["consensus_status"] == "conflict").sum()),
        int((drugmap7_consensus["consensus_status"] == "unresolved").sum()),
    )
    return {
        "output_dir": str(output_dir),
        "cohorts": cohort_outputs,
        "coverage": coverage.to_dict("records"),
    }
