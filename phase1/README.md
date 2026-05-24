# Phase 1: CAD Tabular Analysis

This README covers the tracked Phase 1 files in `phase1/`.

The current workflow includes:

- tabular CAD datasets with angiographic CAD labels
- association-rule mining inside each cross-validation fold
- a protective score built from matched clinical features
- logistic baseline and hybrid reference metrics
- cross-fold rule stability

Main run:

```powershell
python -m phase1.run_cad_rule_pipeline --out-dir artifacts\phase1
```

If you already have Z-Alizadeh locally:

```powershell
python -m phase1.run_cad_rule_pipeline --z-path Data\z-alizadeh.csv --out-dir artifacts\phase1
```

Additional dependencies:

```powershell
pip install -r phase1\requirements.txt
```

Module layout:

- `cad_tabular_config.py`: dataset URLs, binning specs, and rule parameters
- `cad_tabular_data.py`: Z-Alizadeh and UCI Heart loaders
- `cad_tabular_preprocessing.py`: imputation, discretization, and one-hot encoding
- `cad_rule_mining.py`: association-rule mining and protective-score construction
- `cad_rule_evaluation.py`: out-of-fold scoring, logistic baselines, and stability metrics
- `cad_rule_pipeline.py`: dataset-level orchestration and artifact saving
- `cad_rule_api.py`: compatibility layer for older imports
