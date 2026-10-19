#!/usr/bin/env python3
"""Validate and summarize post-lock matched drug-only controls from OOF rows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


CLASSES = ("agonist", "antagonist", "blocker", "inhibitor")
SPLITS = ("random_pair", "drug_disjoint", "target_disjoint", "double_disjoint")
KEY = ("split", "repeat", "drug_name", "target_name", "true_label")


def read_rows(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path, rows):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def key(row):
    return tuple(row[field] for field in KEY)


def metric(rows):
    scores = []
    for label in CLASSES:
        tp = sum(r["true_label"] == label and r["predicted_label"] == label for r in rows)
        fp = sum(r["true_label"] != label and r["predicted_label"] == label for r in rows)
        fn = sum(r["true_label"] == label and r["predicted_label"] != label for r in rows)
        scores.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    return {"f1_macro": statistics.mean(scores), "accuracy": sum(r["true_label"] == r["predicted_label"] for r in rows) / len(rows)}


def normalize(rows, strategy):
    result = []
    for r in rows:
        assert r["task"] == "core4_action"
        if r.get("seed"):
            assert int(r["seed"]) == 42 + int(r["repeat"])
        s = {field:r[field] for field in KEY}
        s.update({"cohort":"external","strategy":strategy,"seed":str(42+int(r["repeat"])),"fold":r["fold"],"predicted_label":r["predicted_label"]})
        for label in CLASSES:
            s["score__"+label] = r["score__"+label]
        result.append(s)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--legacy-only", action="store_true")
    args = parser.parse_args()
    root = args.project_root.resolve()
    output = root / "runs/manuscript_revision_20260906/matched_control"
    output.mkdir(parents=True, exist_ok=True)
    reference_path = root / "runs/pair_action_locked_multiview_3repeat/external/routing_oof_predictions.csv"
    legacy_path = root / "runs/pair_action_target_context_sensitivity/external/action_pair_oof_predictions.csv"
    reference = read_rows(reference_path)
    legacy = read_rows(legacy_path)
    pair_gate = normalize([r for r in reference if r["strategy"] == "pair_conditioned_gate"], "pair_gate")
    pair_full = normalize([r for r in reference if r["strategy"] == "expert::full_label_free"], "pair_fullkg_rf")
    old_drug = normalize(legacy, "legacy_drug_only_fullkg_rf")
    source_paths = [reference_path, legacy_path]
    groups = {"pair_gate":pair_gate,"pair_fullkg_rf":pair_full,"legacy_drug_only_fullkg_rf":old_drug}
    if not args.legacy_only:
        raw = []
        for split in SPLITS:
            path = output / "external" / split / "routing_oof_predictions.csv"
            source_paths.append(path)
            split_rows = read_rows(path)
            assert {r["split"] for r in split_rows} == {split}
            raw.extend(split_rows)
        groups["drug_only_gate"] = normalize([r for r in raw if r["strategy"] == "pair_conditioned_gate"], "drug_only_gate")
        groups["drug_only_fullkg_rf"] = normalize([r for r in raw if r["strategy"] == "expert::full_label_free"], "drug_only_fullkg_rf")

    expected = {key(r):r for r in pair_gate}
    assert len(expected) == 898 * 3 * 4
    audit = {"cohort":"external","pair_count":898,"drug_count":284,"target_count":209,"splits":list(SPLITS),"repeats":3,"split_seeds":[42,43,44],"rf_template_random_state":42,"row_and_fold_alignment":{},"source_hashes":{str(p):digest(p) for p in source_paths}}
    for name, rows in groups.items():
        indexed = {key(r):r for r in rows}
        assert len(indexed) == len(rows) == len(expected)
        assert indexed.keys() == expected.keys(), name
        assert all(r["fold"] == expected[k]["fold"] for k,r in indexed.items()), name
        for split in SPLITS:
            for repeat in range(3):
                selected = [r for r in rows if r["split"] == split and int(r["repeat"]) == repeat]
                assert len(selected) == 898
                assert len({r["drug_name"] for r in selected}) == 284
                assert len({r["target_name"] for r in selected}) == 209
                assert {int(r["fold"]) for r in selected} == set(range(25 if split == "double_disjoint" else 5))
        for r in rows:
            scores = [float(r["score__"+label]) for label in CLASSES]
            assert all(math.isfinite(v) and 0 <= v <= 1 for v in scores)
            assert abs(sum(scores)-1) < 1e-10
            assert r["predicted_label"] == CLASSES[max(range(4), key=lambda i:scores[i])]
        audit["row_and_fold_alignment"][name] = {"all_keys_labels_and_folds_equal_reference":True,"oof_rows":len(rows)}

    repeats = []
    for strategy, rows in groups.items():
        for split in SPLITS:
            for repeat in range(3):
                selected = [r for r in rows if r["split"] == split and int(r["repeat"]) == repeat]
                repeats.append({"cohort":"external","split":split,"strategy":strategy,"repeat":repeat,"seed":42+repeat,"pair_count":898,**metric(selected)})
    summary = []
    for strategy in groups:
        for split in SPLITS:
            values = [r for r in repeats if r["strategy"] == strategy and r["split"] == split]
            summary.append({"cohort":"external","split":split,"strategy":strategy,"cv_repeats":3,"pair_count":898,"f1_macro_mean":statistics.mean(r["f1_macro"] for r in values),"f1_macro_sd":statistics.stdev(r["f1_macro"] for r in values),"accuracy_mean":statistics.mean(r["accuracy"] for r in values)})
    lookup = {(r["strategy"],r["split"]):r for r in summary}
    contrasts = []
    comparisons = [("fullkg_legacy_drug_feature_control","pair_fullkg_rf","legacy_drug_only_fullkg_rf")]
    if not args.legacy_only:
        comparisons += [("matched_five_view_gate_drug_feature_control","pair_gate","drug_only_gate"),("matched_fullkg_rf_drug_feature_control","pair_fullkg_rf","drug_only_fullkg_rf")]
        new_drug = {key(r):r for r in groups["drug_only_fullkg_rf"]}
        old_drug_indexed = {key(r):r for r in old_drug}
        maximum = max(abs(float(r["score__"+label])-float(old_drug_indexed[k]["score__"+label])) for k,r in new_drug.items() for label in CLASSES)
        disagreements = sum(r["predicted_label"] != old_drug_indexed[k]["predicted_label"] for k,r in new_drug.items())
        audit["legacy_vs_new_fullkg_drug_only"] = {"maximum_absolute_score_difference":maximum,"prediction_disagreements":disagreements,"fold_and_sample_identity_verified":True,"equal_within_1e_12":maximum <= 1e-12 and disagreements == 0,"matched_comparison_uses_new_routing_expert":True}
    for contrast, pair, drug in comparisons:
        for split in SPLITS:
            left, right = lookup[(pair,split)], lookup[(drug,split)]
            contrasts.append({"cohort":"external","split":split,"comparison":contrast,"pair_strategy":pair,"drug_only_strategy":drug,"pair_macro_f1_mean":left["f1_macro_mean"],"pair_macro_f1_sd":left["f1_macro_sd"],"drug_only_macro_f1_mean":right["f1_macro_mean"],"drug_only_macro_f1_sd":right["f1_macro_sd"],"pair_minus_drug_only_macro_f1":left["f1_macro_mean"]-right["f1_macro_mean"],"cv_repeats":3,"post_lock_sensitivity":True})

    prefix = "legacy_fullkg_" if args.legacy_only else "matched_control_"
    write_rows(output / (prefix+"repeat_metrics.csv"), repeats)
    write_rows(output / (prefix+"summary.csv"), summary)
    write_rows(output / (prefix+"contrasts.csv"), contrasts)
    (output / (prefix+"validation.json")).write_text(json.dumps(audit,indent=2)+"\n")
    if not args.legacy_only:
        write_rows(output / "matched_control_oof_predictions.csv", [r for strategy,rows in groups.items() if strategy != "legacy_drug_only_fullkg_rf" for r in rows])
        report = [
            "# External post-lock matched drug-only controls", "",
            "Completed all four external split schemes, three repeats each, using 898 observations, 284 drugs, and 209 targets. The original pair-gate strategy and its predictions were preserved. The new controls changed only the explicit feature blocks, retaining the same five cached RGCN views, RF settings, gate, and nested split implementation.", "",
            "Macro-F1 values below are mean ± sample standard deviation over seeds 42, 43, and 44; Δ is pair minus drug-only. These are descriptive sensitivity results.", "",
            "| Split | Pair gate | Drug-only gate | Δ gate | Pair full-KG RF | Drug-only full-KG RF | Δ RF |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        def fmt(strategy, split):
            record = lookup[(strategy, split)]
            return "%.4f ± %.4f" % (record["f1_macro_mean"], record["f1_macro_sd"])
        for split in SPLITS:
            gate_delta = lookup[("pair_gate",split)]["f1_macro_mean"] - lookup[("drug_only_gate",split)]["f1_macro_mean"]
            rf_delta = lookup[("pair_fullkg_rf",split)]["f1_macro_mean"] - lookup[("drug_only_fullkg_rf",split)]["f1_macro_mean"]
            report.append("| %s | %s | %s | %+.4f | %s | %s | %+.4f |" % (
                split,fmt("pair_gate",split),fmt("drug_only_gate",split),gate_delta,
                fmt("pair_fullkg_rf",split),fmt("drug_only_fullkg_rf",split),rf_delta,
            ))
        reproduction = audit["legacy_vs_new_fullkg_drug_only"]
        report += [
            "", "## Validation and interpretation", "",
            "Each of the four compared strategies has 10,776 OOF records, and all observation identities, labels, and fold indices match the original gate. The combined comparison file has 43,104 records. Preflight additionally confirmed all 120 outer folds and 810 inner validation partitions, including entity isolation and exact pair-to-drug feature correspondence for all five views.", "",
            "The legacy and new full-KG drug-only experts have %d predicted-label disagreements; the maximum absolute probability difference is %.3g. Final matched full-KG contrasts use the new routing expert." % (reproduction["prediction_disagreements"],reproduction["maximum_absolute_score_difference"]), "",
            "The ablation removes explicit target-containing feature blocks. The cached drug embeddings retain graph-derived target context. All reference action labels remain constant within each drug across targets, so the result does not establish within-drug target-dependent action switching.", "",
            "## Files", "",
            "- `matched_control_contrasts.csv`: matched mean, sample SD, and pair-minus-drug-only differences.",
            "- `matched_control_summary.csv` and `matched_control_repeat_metrics.csv`: four primary comparison strategies plus the legacy drug-only reproducibility reference.",
            "- `matched_control_oof_predictions.csv`: aligned records for the four comparison strategies.",
            "- `matched_control_validation.json`: row/fold checks, score agreement, and input hashes.",
            "- `matched_data_fold_preflight.json`: embedding hashes, feature checks, RF settings, and complete outer/inner index hashes.",
            "- `configs/` and `external/{split}/`: exact executed configurations and downloaded raw reports.",
            "- `METHODS.md`: methods, interpretation limits, seeding, and execution provenance.", "",
            "Original predictive results, the strategy lock, development evaluation, and manuscript were not modified by this analysis. No P values or hypothesis tests were added.", "",
        ]
        (output / "README.md").write_text("\n".join(report))
        completed_files = [p for p in output.glob("matched_control_*") if p.is_file()]
        completed_files += [output / "README.md", output / "METHODS.md"]
        (output / "matched_output_hashes.json").write_text(json.dumps({str(p.relative_to(output)):digest(p) for p in sorted(completed_files)},indent=2)+"\n")
    print(json.dumps({"outputs_prefix":prefix,"contrasts":contrasts,"validation":audit.get("legacy_vs_new_fullkg_drug_only")},indent=2))


if __name__ == "__main__":
    main()
