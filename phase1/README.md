# Phase 1: CAD Tabular Rule-XAI

This folder keeps only the defensible Phase 1 path:

- CAD tabular datasets with angiographic CAD-style labels.
- Leakage-free association-rule mining inside each CV fold.
- A protective score based on unique matched clinical features.
- Logistic baseline vs hybrid logistic as a reference check, not the primary result.
- Cross-fold rule stability.

Dropped from the main path:

- Diamond-Forrester comparison: useful only as a caveated appendix because mappings differ across datasets.
- Framingham/SCORE audit: most variables are missing.
- DCA/calibration plots: not central because the hybrid model does not materially improve prediction.

Main run:

```powershell
python -m phase1.run_cad_rule_pipeline --out-dir artifacts\phase1
```

## Archived Exploratory Probes

These are kept for reference only. They are not part of the main Phase 1 path.

Transferability probes:

```powershell
python -m phase1.archive_experiments.run_transferability_probes --out-dir artifacts\phase1\transferability_probes
```

Interpret these probes as supplementary evidence only unless `verdict.json` reports
`keep_as_main: true`. Current results support transferability/supplementary use,
not the primary Phase 1 result.

Advanced probes:

```powershell
python -m phase1.archive_experiments.run_robustness_probes --out-dir artifacts\phase1\advanced_probes
```

These test harmonized Z-Alizadeh/UCI transfer, shuffled-label negative controls,
and bootstrap-stable rule scores. Treat as advanced supplementary evidence only after
checking `verdict.json`; the default run uses a moderate compute budget.

Grouped CAD counterfactual bridge:

```powershell
python -m phase1.archive_experiments.run_grouped_ecg_bridge --out-dir artifacts\phase1\archive\grouped_cad_counterfactual_failed
```

This bridge is archived because the ECG-ischemia group did not become a stable
main signal across datasets.

Rule-memory counterfactual probe:

```powershell
python -m phase1.archive_experiments.run_rule_memory_probe --out-dir artifacts\phase1\archive\rule_memory_counterfactual_failed
```

The current run is archived because phenotype edits work mostly by changing
diagnostic markers, actionable edits are weak, and immutable-control edits are
not consistently low. Do not use this as a primary result.

If you already have Z-Alizadeh locally:

```powershell
python -m phase1.run_cad_rule_pipeline --z-path Data\z-alizadeh.csv --out-dir artifacts\phase1
```

Dependencies not in the old project requirements:

```powershell
pip install -r phase1\requirements.txt
```

Module layout:

- `cad_tabular_config.py`: dataset URLs, binning specs, rule parameters.
- `cad_tabular_data.py`: Z-Alizadeh and UCI Heart loaders.
- `cad_tabular_preprocessing.py`: imputation, discretization, one-hot encoding.
- `cad_rule_mining.py`: association-rule mining and protective-score construction.
- `cad_rule_evaluation.py`: leakage-free OOF scoring, logistic baselines, stability.
- `cad_rule_pipeline.py`: dataset-level orchestration and artifact saving.
- `cad_rule_api.py`: backward-compatible facade for older imports.
- `archive_experiments/`: supplementary and archived Phase 1 probes.

