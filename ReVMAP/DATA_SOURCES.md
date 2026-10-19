# Data sources and access

## Released files

`data/graphs/{development,cpg0004}/` contains the frozen graph groups used in the
experiments. Gzip compression is lossless; uncompressed checksums and source paths
relative to the project are in `data/DATA_MANIFEST.json`. The `target` and `moa`
tables derive from Drug Repurposing Hub annotations. The `indication`, `anatomy`,
and `pathway` groups derive from PrimeKG. Normalization, deduplication, cohort
filtering, and the SSRI-to-inhibitor mapping are implemented in the preparation code.

The cohort tables expose drug names, target names, and reference action labels.
Their membership reflects the original common-input matching procedure, but the
released tables do not reproduce DrugBank structures or sequences.
Prediction files are model outputs with observation identifiers, reference actions,
folds, repeats, and class probabilities. Summary files are derived study results.

## Original resources

| Resource | Role and source | Data licence/access |
| --- | --- | --- |
| Drug Repurposing Hub | Targets and reference actions; [Broad portal](https://repo-hub.broadinstitute.org/repurposing) | The portal identifies compound metadata as CC BY 4.0. Retain attribution to Corsello et al. (2017), and consult the portal's research-use and commercial-redistribution notices. |
| PrimeKG | Indication, anatomy, and pathway relations; [Harvard Dataverse, DOI 10.7910/DVN/IXA7BM](https://doi.org/10.7910/DVN/IXA7BM) | Dataverse release 2.1 declares CC0 1.0. Its machine-readable licence record is preserved in `data/primekg_license_record.json`. Cite Chandak, Huang, and Zitnik (2023), DOI 10.1038/s41597-023-01960-3. |
| cpg0004/LINCS | Compound-cohort metadata; [LINCS Cell Painting](https://github.com/broadinstitute/lincs-cell-painting) | Source repository declares CC0 1.0 for data/results/figures and BSD-3-Clause for its code. Cite Way et al. (2022), DOI 10.1016/j.cels.2022.10.001. No Cell Painting images or L1000 profiles are required for ReVMAP. |
| DrugBank 5.1.13 | Original compound matching, SMILES/protein inputs for adapted baselines, and separate action evidence; [DrugBank](https://go.drugbank.com/releases/latest) | Obtain the corresponding XML release under your applicable DrugBank licence. The XML, structures, sequences, and extracted source records are not bundled. DrugBank offers different dataset licences; no blanket redistribution licence is inferred here. |
| Open Targets | Separate pharmacological action evidence; [data access](https://platform-docs.opentargets.org/data-access) | [Platform licence](https://platform-docs.opentargets.org/licence) marks data CC0 1.0. Original archived inputs are identified in the configuration; the bulk source is not bundled. |

Drug Repurposing Hub reference: Corsello et al., *The Drug Repurposing Hub: a
next-generation drug library and information resource*, Nature Medicine (2017),
DOI 10.1038/nm.4306. The annotation-derived files are provided for research with
this attribution; the software's MIT licence does not relicense source annotations.

## Rebuilding from source databases

The portable configurations expect:

```text
data/raw/cpg0004/combined_metadata.csv
data/raw/repurposing_hub/Repurposing_Hub_export.txt
data/raw/primekg/kg.csv
data/raw/drugbank/full_database.xml
data/raw/open_targets/<archived parquet files>
```

The originally used PrimeKG `kg.csv` has 981,751,236 bytes and Dataverse MD5
`aac8191d4fbc5bf09cdf8c3c78b4e75f`. The Repurposing Hub input SHA-256 was
`97f6f7a0f8658c30e1fc28584f90e7d86aac6fb45f11b2248b248eb2146e8078`;
the combined-metadata input SHA-256 was
`b005ea0089c43c9dbd45856b027f2e28e5d4ad92d0feb106a95bb0683d16f5d5`.
Source exports change over time. Use the released frozen graph groups and cohorts
to preserve the manuscript's graph/cohort selection; rebuilding from a newer
source release may produce a different benchmark.

The supplementary AI assessments are machine-generated evaluations, not human
expert ratings. Released AI summary records retain the differing outcomes of
the assessed rounds; they are not used as proof of biological correctness.
