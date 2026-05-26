# Phase 2 ECG Analysis

This README covers the tracked Phase 2 files in `phase2/`.

The current workflow centers on the PTB-XL `NORM vs STTC` task and the external
Georgia check.

`prepare_ptbxl_multilead_data.py` now builds the PTB-XL Phase 2 CSV splits
directly from raw PTB-XL, without requiring a pre-existing single-lead
`Generated_PTBXL` folder.

## Files

| File | Role |
|---|---|
| `ecg_features.py` | ECG loading and feature extraction |
| `ecg_mechanism_core.py` | shared grouping, model, metric, and counterfactual helpers |
| `prototype_memory.py` | prototype memory used by the counterfactual analyses |
| `prepare_ptbxl_multilead_data.py` | PTB-XL 12-lead CSV preparation |
| `download_georgia_dataset.py` | Georgia dataset download and resume helper |
| `run_ptbxl_main_experiment.py` | PTB-XL classification, feature-level controls, and reports |
| `run_georgia_external_validation.py` | Georgia external validation with the PTB-XL model |
| `run_georgia_control_analysis.py` | Georgia feature-level counterfactual controls |
| `run_ptbxl_waveform_counterfactual.py` | PTB-XL waveform counterfactual analysis |
| `run_georgia_waveform_counterfactuals.py` | Georgia waveform counterfactual analysis |

## Runs

PTB-XL main experiment:

```powershell
python -m phase2.run_ptbxl_main_experiment `
  --csv-root Data\Generated_PTBXL_12Lead `
  --out-dir artifacts\phase2\ptbxl_final `
  --prototypes-per-class 12 `
  --permutation-repeats 8 `
  --n-boot 2000 `
  --n-cases 10
```

Georgia external validation with strict-normal negatives:

```powershell
python -m phase2.run_georgia_external_validation `
  --ptbxl-root Data\Generated_PTBXL_12Lead `
  --georgia-root Data\PhysioNet_Challenge_2020_Georgia `
  --out-dir artifacts\phase2\external_georgia_full_strict_normal `
  --negative-policy strict_normal `
  --max-per-class 0 `
  --target-fs 100
```

Georgia external validation with `no_st_t` negatives:

```powershell
python -m phase2.run_georgia_external_validation `
  --ptbxl-root Data\Generated_PTBXL_12Lead `
  --georgia-root Data\PhysioNet_Challenge_2020_Georgia `
  --out-dir artifacts\phase2\external_georgia_full_no_st_t `
  --negative-policy no_st_t `
  --max-per-class 0 `
  --target-fs 100
```

Georgia feature-level controls:

```powershell
python -m phase2.run_georgia_control_analysis `
  --ptbxl-root Data\Generated_PTBXL_12Lead `
  --georgia-root Data\PhysioNet_Challenge_2020_Georgia `
  --out-dir artifacts\phase2\georgia_counterfactual_controls `
  --target-fs 100 `
  --n-boot 1000
```

PTB-XL waveform analysis:

```powershell
python -m phase2.run_ptbxl_waveform_counterfactual `
  --out-dir artifacts\phase2\waveform_counterfactual_sttc `
  --template-max-records 800 `
  --n-boot 1000
```

Georgia waveform analysis:

```powershell
python -m phase2.run_georgia_waveform_counterfactuals `
  --out-dir artifacts\phase2\georgia_waveform_counterfactual_sttc_full `
  --max-positive 0 `
  --template-max-records 800 `
  --n-boot 1000
```

## Key Outputs

- `artifacts/phase2/ptbxl_final/ptbxl_final_report.json`
- `artifacts/phase2/ptbxl_final/negative_control_summary.csv`
- `artifacts/phase2/external_georgia_full_strict_normal/georgia_validation_report.json`
- `artifacts/phase2/external_georgia_full_no_st_t/georgia_validation_report.json`
- `artifacts/phase2/georgia_counterfactual_controls/georgia_counterfactual_controls_report.json`
- `artifacts/phase2/waveform_counterfactual_sttc/waveform_counterfactual_report.json`
- `artifacts/phase2/georgia_waveform_counterfactual_sttc_full/georgia_waveform_counterfactual_report.json`
