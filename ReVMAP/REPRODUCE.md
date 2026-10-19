# Reproducing ReVMAP

Run all commands from `ReVMAP/`. Keep the saved data unchanged and write new outputs
under `runs/` or `reproduced_results/`.

## 1. Recompute saved prediction metrics

```bash
python scripts/verify_release.py
python scripts/reproduce_tables.py
```

These commands need only the Python standard library. Saved OOF predictions allow
verification of the reported performance without model retraining. They include
all saved routing strategies and the task-adapted DeepDTA, GraphDTA, and
DruGNNosis-inspired baselines, with the original observation/fold identifiers.

## 2. Install the modelling environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
python -m pytest tests
```

Python 3.10 was used for the recorded modelling runs. `pyproject.toml` and
`environment.yml` specify supported dependencies, not a bit-for-bit historical
environment lock. Preserved matched-control provenance records Python 3.10.19,
NumPy 2.2.5, pandas 2.2.3, and scikit-learn 1.6.1. Library/hardware differences can
change retrained embeddings and downstream scores. The original predictions are
the fixed numerical reference. GPU training is recommended for the embedding and
neural-baseline stages.

## 3. Train ReVMAP from the frozen graph subsets

The frozen input route avoids downloading full source databases. The graph groups
and observation lists are already included. Build graph views and train RGCN for
each cohort:

```bash
kgml-pipeline --config config/frozen_development_embeddings.json build-triples
kgml-pipeline --config config/frozen_development_embeddings.json train-embeddings
kgml-pipeline --config config/frozen_cpg0004_embeddings.json build-triples
kgml-pipeline --config config/frozen_cpg0004_embeddings.json train-embeddings
kgml-pipeline --config config/frozen_multiview_protocol.json select-locked-multiview
kgml-pipeline --config config/frozen_multiview_protocol.json evaluate-locked-multiview-external
```

The two final commands reproduce the development-selection procedure followed by
evaluation of the selected strategy. They write a fresh strategy lock for that run;
they do not claim that a new training run must select the historical design. The
manuscript selected `pair_conditioned_gate`. The exact historical candidate scores
are under `results/pair_action_locked_multiview_3repeat/`.

The frozen configurations restrict embedding training to RGCN. The corresponding
`manuscript_drugmap_style_*.json` configurations retain the six-model screening
settings. Action labels remain outside all embedding views. Graph embeddings are
transductive and fixed across classifier folds, as described in the manuscript.

## 4. Adapted neural baselines and full source rebuilding

Acquire the original resources listed in `DATA_SOURCES.md`, then set the portable
input paths to your licensed/local files. The pair-baseline implementation creates
the matched multimodal cohort from DrugBank and predicts the same four reference
actions; its architectures and fitting procedures are implemented in
`pair_multimodal_evaluation.py` and `multimodal_models.py`.

```bash
kgml-pipeline --config config/manuscript_pair_action_cpg0004_extension.json evaluate-pair-multimodal
```

The supplied manuscript baseline config covers random-pair, drug-disjoint, and
target-disjoint settings. For the double-disjoint baseline run, copy that config,
set `pair_multimodal.splits` to `["double_disjoint"]`, and give its reports/logs
separate output directories. Saved baseline predictions for all four settings
are included and checked by `reproduce_tables.py`.

## 5. Confidence, controls, and reports

`action_reliability.py` implements development-fitted temperature calibration,
selective prediction, empirical label sets, and report/evidence assessments.
`relation_routing.py` implements view contributions and perturbation analyses.
The related configurations describe all required input files. The released
calibrated predictions and aggregate results support downstream inspection;
regenerating every evidence report also requires the separately acquired source
records. No human-expert evaluation is asserted by this release.

`scripts/run_matched_control_20260906.py` supplies the matched pair/drug-only
control workflow. Historical analysis scripts retain their original `runs/`
layout; they require restoring their documented inputs, whereas the quick
verification commands above operate directly on the compressed release data.

The OpenSPG KAG integration is optional. Its adapter requires a separately
configured KAG service/environment. No service credentials or language-model
API keys are included in this repository.
