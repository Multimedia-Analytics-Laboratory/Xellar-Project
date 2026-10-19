# ReVMAP

Code and analysis data for **ReVMAP: Heterogeneous knowledge graph learning for
target-contextualized drug-action prediction**.

ReVMAP predicts inhibitor, agonist, antagonist, or blocker annotations for
drug–target observations. Five label-free graph views supply RGCN representations
and Random Forest experts. A gate combines their probabilities, with gate fitting
based on nested out-of-fold predictions. Separate modules evaluate predictive
confidence and construct structured reports of model contributions, graph context,
and agreement with pharmacological records.

The development cohort contains 1,614 observations from 738 drugs. The cpg0004
evaluation cohort contains 898 observations from 284 drugs. The reference action
is the compound-level Repurposing Hub annotation associated with each observation.
Molecular structures and protein sequences are inputs to the adapted external
baselines; ReVMAP uses graph representations.

## Quick verification

Download or clone this repository, then enter `ReVMAP/`. Python 3.10 or newer is
sufficient for these two commands; they require no downloaded databases or GPU:

```bash
python scripts/verify_release.py
python scripts/reproduce_tables.py
```

The second command verifies probability vectors and common evaluation observations,
and recomputes macro-F1 for the saved ReVMAP and baseline predictions across three
repeats. It writes `reproduced_results/prediction_macro_f1.csv`.

| Split | ReVMAP mean macro-F1 |
| --- | ---: |
| Random pair | 0.966231 |
| Drug disjoint | 0.776191 |
| Target disjoint | 0.935125 |
| Double disjoint | 0.717995 |

## Contents

- `src/kgml_pipeline/`: data preparation, graph embedding, action prediction,
  view fusion, adapted baselines, confidence analysis, and evidence-report modules.
- `config/`: portable manuscript configurations and frozen-graph configurations.
- `data/graphs/`: exact processed graph groups for development and cpg0004.
  Action labels are in a separate `moa` table and excluded from embedding views.
- `data/cohorts/`: matched observation identifiers and reference actions; no
  DrugBank molecular structures or protein sequences are redistributed here.
- `data/predictions/`: compressed OOF predictions, fold identifiers, calibrated
  probabilities, and matched-control outputs.
- `results/`: saved aggregate results and supplementary analyses. Comparisons
  omitted from the main manuscript remain available in these records.
- `tests/`: checks of splits, fitting logic, label exclusion, and analysis functions.
- `kag_project/`: typed-path retrieval utilities for the optional OpenSPG KAG adapter.

See [REPRODUCE.md](REPRODUCE.md) for training and additional analyses, and
[DATA_SOURCES.md](DATA_SOURCES.md) for provenance, source access, and data licences.

The import/package name `kgml_pipeline` and historical file identifiers are retained
for compatibility. The model and current manuscript are named ReVMAP.

## Licence and citation

Original ReVMAP software is released under the [MIT licence](LICENSE). Third-party
data retain their source licences, described in [DATA_SOURCES.md](DATA_SOURCES.md).
The software licence does not replace those data terms. Citation metadata are in
`CITATION.cff`; the accompanying manuscript has not yet been assigned a publication DOI.
