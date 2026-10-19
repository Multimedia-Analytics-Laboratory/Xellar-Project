"""Build leakage-controlled drug/target sequence inputs from DrugBank XML.

Only drug identity, SMILES, target identity, and target amino-acid sequences
are exported. DrugBank ``actions`` are intentionally ignored because they are
the labels that the downstream MoA task is intended to predict.
"""

from __future__ import annotations

import json
import re
import unicodedata
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import pandas as pd

from .io_utils import ensure_dir, write_json


DRUGBANK_NAMESPACE = "http://www.drugbank.ca"
NS = "{%s}" % DRUGBANK_NAMESPACE


def _clean_text(value):
    if value is None:
        return ""
    return unicodedata.normalize("NFKC", str(value)).strip()


def _exact_name(value):
    return re.sub(r"\s+", " ", _clean_text(value)).casefold()


def _relaxed_name(value):
    return re.sub(r"[^a-z0-9]+", "", _exact_name(value))


def _direct_text(element, tag):
    child = element.find(NS + tag)
    if child is None or child.text is None:
        return ""
    return _clean_text(child.text)


def _primary_drugbank_id(drug):
    for element in drug.findall(NS + "drugbank-id"):
        if element.attrib.get("primary") == "true":
            return _clean_text(element.text)
    first = drug.find(NS + "drugbank-id")
    return _clean_text(first.text if first is not None else "")


def _drug_aliases(drug):
    aliases = []
    primary_name = _direct_text(drug, "name")
    if primary_name:
        aliases.append(("primary_name", primary_name))
    synonyms = drug.find(NS + "synonyms")
    if synonyms is not None:
        for synonym in synonyms.findall(NS + "synonym"):
            text = _clean_text(synonym.text)
            if text:
                aliases.append(("synonym", text))
    products = drug.find(NS + "products")
    if products is not None:
        for product in products.findall(NS + "product"):
            text = _direct_text(product, "name")
            if text:
                aliases.append(("product_name", text))
    return aliases


def _canonical_smiles(drug):
    calculated = drug.find(NS + "calculated-properties")
    if calculated is None:
        return ""
    preferred = {}
    for prop in calculated.findall(NS + "property"):
        kind = _direct_text(prop, "kind").casefold()
        value = _direct_text(prop, "value")
        if kind and value:
            preferred[kind] = value
    for key in (
        "smiles",
        "canonical smiles",
        "isomeric smiles",
    ):
        if key in preferred:
            return preferred[key]
    return ""


def _clean_fasta_sequence(value):
    text = _clean_text(value)
    if not text:
        return ""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines and lines[0].startswith(">"):
        lines = lines[1:]
    return re.sub(r"[^A-Za-z]", "", "".join(lines)).upper()


def _target_sequences(drug):
    targets = drug.find(NS + "targets")
    if targets is None:
        return []
    records = []
    seen = set()
    for target in targets.findall(NS + "target"):
        polypeptide = target.find(NS + "polypeptide")
        if polypeptide is None:
            continue
        sequence_element = polypeptide.find(NS + "amino-acid-sequence")
        sequence = _clean_fasta_sequence(
            sequence_element.text if sequence_element is not None else ""
        )
        if not sequence:
            continue
        target_id = _clean_text(polypeptide.attrib.get("id", ""))
        target_name = _direct_text(target, "name")
        key = (target_id, sequence)
        if key in seen:
            continue
        seen.add(key)
        records.append({
            "target_id": target_id,
            "target_name": target_name,
            "sequence": sequence,
        })
    return records


def _benchmark_name_indices(drug_names):
    exact = defaultdict(set)
    relaxed = defaultdict(set)
    for drug_name in drug_names:
        exact[_exact_name(drug_name)].add(drug_name)
        relaxed[_relaxed_name(drug_name)].add(drug_name)
    return exact, relaxed


def _candidate_matches(aliases, exact_index, relaxed_index):
    candidates = {}
    priority = {
        ("primary_name", "exact"): 0,
        ("synonym", "exact"): 1,
        ("product_name", "exact"): 2,
        ("primary_name", "relaxed"): 3,
        ("synonym", "relaxed"): 4,
        ("product_name", "relaxed"): 5,
    }
    for alias_type, alias in aliases:
        for benchmark_name in exact_index.get(_exact_name(alias), set()):
            key = (alias_type, "exact")
            current = candidates.get(benchmark_name)
            if current is None or priority[key] < current["priority"]:
                candidates[benchmark_name] = {
                    "priority": priority[key],
                    "match_type": "%s_%s" % key,
                    "matched_alias": alias,
                }
        relaxed_key = _relaxed_name(alias)
        # Relaxed matching is only accepted when the benchmark key itself is
        # unique. This avoids silently merging distinct salts or formulations.
        relaxed_names = relaxed_index.get(relaxed_key, set())
        if len(relaxed_names) == 1:
            benchmark_name = next(iter(relaxed_names))
            key = (alias_type, "relaxed")
            current = candidates.get(benchmark_name)
            if current is None or priority[key] < current["priority"]:
                candidates[benchmark_name] = {
                    "priority": priority[key],
                    "match_type": "%s_%s" % key,
                    "matched_alias": alias,
                }
    return candidates


def build_drugbank_multimodal_table(config, logger):
    """Map benchmark drugs to DrugBank SMILES and target sequences."""

    multimodal_cfg = config.get("multimodal", {})
    xml_path = Path(multimodal_cfg["drugbank_xml"]).expanduser()
    if not xml_path.exists():
        raise FileNotFoundError("DrugBank XML not found: %s" % xml_path)

    group_root = Path(config["paths"]["intermediate_dir"]) / "triples" / "groups"
    moa_triples = pd.read_csv(group_root / "moa.csv").drop_duplicates()
    benchmark_drugs = sorted(moa_triples["head"].astype(str).unique().tolist())
    exact_index, relaxed_index = _benchmark_name_indices(benchmark_drugs)

    candidates_by_benchmark = defaultdict(list)
    logger.info(
        "Streaming DrugBank XML for %d benchmark drugs: %s",
        len(benchmark_drugs),
        xml_path,
    )
    for _, element in ET.iterparse(str(xml_path), events=("end",)):
        if element.tag != NS + "drug":
            continue
        aliases = _drug_aliases(element)
        matches = _candidate_matches(aliases, exact_index, relaxed_index)
        if matches:
            drugbank_id = _primary_drugbank_id(element)
            drugbank_name = _direct_text(element, "name")
            smiles = _canonical_smiles(element)
            targets = _target_sequences(element)
            for benchmark_name, match in matches.items():
                candidates_by_benchmark[benchmark_name].append({
                    "benchmark_drug": benchmark_name,
                    "drugbank_id": drugbank_id,
                    "drugbank_name": drugbank_name,
                    "drug_type": element.attrib.get("type", ""),
                    "smiles": smiles,
                    "targets": targets,
                    **match,
                })
        element.clear()

    label_rows = moa_triples.set_index("head")[["relation", "tail"]].to_dict("index")
    output_rows = []
    ambiguous = {}
    for benchmark_name in benchmark_drugs:
        candidates = candidates_by_benchmark.get(benchmark_name, [])
        candidates = sorted(
            candidates,
            key=lambda row: (
                int(row["priority"]),
                not bool(row["smiles"]),
                not bool(row["targets"]),
                row["drugbank_id"],
            ),
        )
        if not candidates:
            continue
        best = candidates[0]
        equally_ranked = [
            row for row in candidates
            if int(row["priority"]) == int(best["priority"])
        ]
        if len({row["drugbank_id"] for row in equally_ranked}) > 1:
            ambiguous[benchmark_name] = [
                {
                    "drugbank_id": row["drugbank_id"],
                    "drugbank_name": row["drugbank_name"],
                    "match_type": row["match_type"],
                    "matched_alias": row["matched_alias"],
                }
                for row in equally_ranked
            ]
            continue

        labels = label_rows[benchmark_name]
        output_rows.append({
            "drug_name": benchmark_name,
            "drugbank_id": best["drugbank_id"],
            "drugbank_name": best["drugbank_name"],
            "drug_type": best["drug_type"],
            "match_type": best["match_type"],
            "matched_alias": best["matched_alias"],
            "smiles": best["smiles"],
            "target_count": len(best["targets"]),
            "target_ids_json": json.dumps(
                [record["target_id"] for record in best["targets"]],
                ensure_ascii=False,
            ),
            "target_names_json": json.dumps(
                [record["target_name"] for record in best["targets"]],
                ensure_ascii=False,
            ),
            "target_sequences_json": json.dumps(
                [record["sequence"] for record in best["targets"]],
                ensure_ascii=False,
            ),
            "superclass_4": str(labels["relation"]),
            "detailed_moa": str(labels["tail"]),
        })

    output_dir = ensure_dir(
        Path(config["paths"]["intermediate_dir"]) / "multimodal"
    )
    output_path = output_dir / "drugbank_multimodal_benchmark.csv"
    output_frame = pd.DataFrame(output_rows)
    output_frame.to_csv(output_path, index=False)

    matched = set(output_frame["drug_name"].tolist()) if not output_frame.empty else set()
    with_smiles = (
        int(output_frame["smiles"].fillna("").astype(str).str.len().gt(0).sum())
        if not output_frame.empty else 0
    )
    with_targets = (
        int(output_frame["target_count"].gt(0).sum())
        if not output_frame.empty else 0
    )
    manifest = {
        "drugbank_xml": str(xml_path),
        "benchmark_drugs": len(benchmark_drugs),
        "matched_unambiguous": len(output_frame),
        "matched_with_smiles": with_smiles,
        "matched_with_targets": with_targets,
        "matched_with_smiles_and_targets": int(
            (
                output_frame["smiles"].fillna("").astype(str).str.len().gt(0)
                & output_frame["target_count"].gt(0)
            ).sum()
        ) if not output_frame.empty else 0,
        "unmatched_drugs": sorted(set(benchmark_drugs) - matched - set(ambiguous)),
        "ambiguous_drugs": ambiguous,
        "label_source": str(group_root / "moa.csv"),
        "leakage_control": "DrugBank action fields were not read or exported.",
    }
    write_json(output_dir / "drugbank_multimodal_manifest.json", manifest)
    logger.info("Wrote multimodal benchmark to %s", output_path)
    logger.info(
        "DrugBank coverage: matched=%d/%d, smiles=%d, targets=%d, both=%d",
        len(output_frame),
        len(benchmark_drugs),
        with_smiles,
        with_targets,
        manifest["matched_with_smiles_and_targets"],
    )
    return {
        "table_path": str(output_path),
        "manifest_path": str(output_dir / "drugbank_multimodal_manifest.json"),
        "manifest": manifest,
    }
