import tempfile
import unittest
from pathlib import Path

import pandas as pd

from kgml_pipeline.pair_action_ground_truth import (
    _match_open_targets_drugs,
    build_consensus,
    build_drugmap7_consensus,
    drugmap7_action,
    extract_drugbank_evidence,
    normalize_action_terms,
)


class PairActionGroundTruthTest(unittest.TestCase):
    def test_action_normalization_does_not_force_distinct_pharmacology(self):
        detailed, core = normalize_action_terms("positive allosteric modulator")
        self.assertEqual(detailed, ["positive_allosteric_modulator"])
        self.assertEqual(core, [])

        detailed, core = normalize_action_terms("partial agonist")
        self.assertEqual(detailed, ["partial_agonist"])
        self.assertEqual(core, ["agonist"])

        detailed, core = normalize_action_terms("inverse agonist")
        self.assertEqual(detailed, ["inverse_agonist"])
        self.assertEqual(core, [])

    def test_drugbank_target_action_is_pair_specific(self):
        xml = """<?xml version="1.0"?>
<drugbank xmlns="http://www.drugbank.ca">
  <drug type="small molecule">
    <drugbank-id primary="true">DBTEST1</drugbank-id>
    <name>Test drug</name>
    <targets>
      <target>
        <id>1</id>
        <name>Target alpha</name>
        <actions><action>inhibitor</action></actions>
        <known-action>yes</known-action>
        <polypeptide id="P00001">
          <gene-name>GENEA</gene-name>
        </polypeptide>
      </target>
      <target>
        <id>2</id>
        <name>Target beta</name>
        <actions><action>agonist</action></actions>
        <known-action>yes</known-action>
        <polypeptide id="P00002">
          <gene-name>GENEB</gene-name>
        </polypeptide>
      </target>
    </targets>
  </drug>
</drugbank>
"""
        seed = pd.DataFrame(
            [
                {
                    "cohort": "test",
                    "pair_id": "pair-a",
                    "drug_name": "test drug",
                    "target_name": "genea",
                    "action": "agonist",
                    "legacy_action": "agonist",
                    "drugbank_id": "DBTEST1",
                    "target_uniprot_id": "P00001",
                    "target_gene_name": "GENEA",
                    "target_drugbank_name": "Target alpha",
                },
                {
                    "cohort": "test",
                    "pair_id": "pair-b",
                    "drug_name": "test drug",
                    "target_name": "geneb",
                    "action": "agonist",
                    "legacy_action": "agonist",
                    "drugbank_id": "DBTEST1",
                    "target_uniprot_id": "P00002",
                    "target_gene_name": "GENEB",
                    "target_drugbank_name": "Target beta",
                },
            ]
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "drugbank.xml"
            path.write_text(xml, encoding="utf-8")
            rows, _, matched = extract_drugbank_evidence(path, seed)
        actions = {
            (row["pair_id"], row["core4_action"])
            for row in rows
            if row["core4_action"]
        }
        self.assertEqual(actions, {("pair-a", "inhibitor"), ("pair-b", "agonist")})
        self.assertEqual(matched, {"pair-a", "pair-b"})

    def test_high_tier_conflict_is_excluded(self):
        seed = pd.DataFrame(
            [
                {
                    "cohort": "test",
                    "pair_id": "p1",
                    "drug_name": "d",
                    "target_name": "t",
                    "drugbank_id": "DB1",
                    "target_uniprot_id": "P1",
                    "target_gene_name": "T",
                    "legacy_action": "inhibitor",
                }
            ]
        )
        evidence = pd.DataFrame(
            [
                {
                    "pair_id": "p1",
                    "source": "DrugBank",
                    "source_tier": "direct_high",
                    "core4_action": "inhibitor",
                },
                {
                    "pair_id": "p1",
                    "source": "Open Targets",
                    "source_tier": "direct_high",
                    "core4_action": "antagonist",
                },
            ]
        )
        result = build_consensus(seed, evidence).iloc[0]
        self.assertEqual(result["consensus_status"], "conflict")
        self.assertEqual(result["ground_truth_tier"], "excluded")
        self.assertEqual(result["consensus_action"], "")

    def test_open_targets_cross_reference_takes_precedence_over_aliases(self):
        row = {
            "drugId": "CHEMBL1",
            "prefName": "Shared active name",
            "synonyms": [],
            "tradeNames": [],
        }
        candidates, method = _match_open_targets_drugs(
            row,
            {"CHEMBL1": {"DB1"}},
            {"shared active name": {"DB2"}},
            {},
        )
        self.assertEqual(candidates, {"DB1"})
        self.assertEqual(method, "drugbank_chembl_cross_reference")

    def test_drugmap7_mapping_matches_public_preprocessing(self):
        self.assertEqual(drugmap7_action("blocker"), "inhibitor")
        self.assertEqual(drugmap7_action("binder"), "ligand")
        self.assertEqual(
            drugmap7_action("positive_allosteric_modulator"),
            "allosteric_modulator",
        )
        self.assertEqual(drugmap7_action("partial_agonist"), "agonist")
        self.assertEqual(drugmap7_action("potentiator"), "inducer")

    def test_drugmap7_drops_ligand_only_for_one_specific_action(self):
        seed = pd.DataFrame(
            [
                {
                    "cohort": "test",
                    "pair_id": "p1",
                    "drug_name": "d",
                    "target_name": "t",
                    "drugbank_id": "DB1",
                    "target_uniprot_id": "P1",
                    "target_gene_name": "T",
                    "legacy_action": "inhibitor",
                }
            ]
        )
        evidence = pd.DataFrame(
            [
                {
                    "cohort": "test",
                    "pair_id": "p1",
                    "source": "DrugBank",
                    "source_tier": "direct_high",
                    "action_normalized": "binder",
                    "action_raw": "binder",
                },
                {
                    "cohort": "test",
                    "pair_id": "p1",
                    "source": "Open Targets",
                    "source_tier": "direct_high",
                    "action_normalized": "inhibitor",
                    "action_raw": "target inhibitor",
                },
            ]
        )
        result, _ = build_drugmap7_consensus(seed, evidence)
        self.assertEqual(result.iloc[0]["action"], "inhibitor")
        self.assertEqual(result.iloc[0]["ground_truth_tier"], "gold")


if __name__ == "__main__":
    unittest.main()
