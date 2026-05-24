# Phase 2 ECG Mechanism Validation

Focused thesis module for the ECG part of the project.

## Single Claim

The Phase 2 claim is deliberately narrow: a simple counterfactual-memory ECG
pipeline identifies a testable ST/T mechanism for the PTB-XL `NORM vs STTC`
task, and the same mechanism remains directionally consistent on external
Georgia data.

This is an XAI/mechanism-validation study. It is not a classifier ranking,
clinical deployment claim, new physiology claim, or general ECG diagnosis
claim.

## Main Code

The active Phase 2 path is limited to these modules:

| File | Role |
|---|---|
| `ecg_features.py` | ECG loading and feature extraction |
| `ecg_mechanism_core.py` | Shared grouping, model, metric, and counterfactual helpers |
| `prototype_memory.py` | Counterfactual prototype memory |
| `prepare_ptbxl_multilead_data.py` | PTB-XL 12-lead CSV preparation |
| `download_georgia_dataset.py` | Georgia dataset download/resume helper |
| `run_ptbxl_main_experiment.py` | PTB-XL `NORM vs STTC` classification and feature-level counterfactuals |
| `run_georgia_external_validation.py` | External Georgia classification check |
| `run_georgia_control_analysis.py` | Full Georgia feature-level ST/T counterfactual controls |
| `run_ptbxl_waveform_counterfactual.py` | PTB-XL waveform-level ST/T intervention |
| `run_georgia_waveform_counterfactuals.py` | Georgia waveform-level ST/T intervention |

## Main Runs

PTB-XL internal benchmark and feature-level counterfactual proof:

```powershell
python -m phase2.run_ptbxl_main_experiment `
  --csv-root Data\Generated_PTBXL_12Lead `
  --out-dir artifacts\phase2\ptbxl_final `
  --prototypes-per-class 12 `
  --permutation-repeats 8 `
  --n-boot 2000 `
  --n-cases 10
```

External Georgia validation, strict normal negatives:

```powershell
python -m phase2.run_georgia_external_validation `
  --ptbxl-root Data\Generated_PTBXL_12Lead `
  --georgia-root Data\PhysioNet_Challenge_2020_Georgia `
  --out-dir artifacts\phase2\external_georgia_full_strict_normal `
  --negative-policy strict_normal `
  --max-per-class 0 `
  --target-fs 100
```

External Georgia validation, hard negatives:

```powershell
python -m phase2.run_georgia_external_validation `
  --ptbxl-root Data\Generated_PTBXL_12Lead `
  --georgia-root Data\PhysioNet_Challenge_2020_Georgia `
  --out-dir artifacts\phase2\external_georgia_full_no_st_t `
  --negative-policy no_st_t `
  --max-per-class 0 `
  --target-fs 100
```

Full Georgia feature-level ST/T control analysis:

```powershell
python -m phase2.run_georgia_control_analysis `
  --ptbxl-root Data\Generated_PTBXL_12Lead `
  --georgia-root Data\PhysioNet_Challenge_2020_Georgia `
  --out-dir artifacts\phase2\georgia_counterfactual_controls `
  --target-fs 100 `
  --n-boot 1000
```

PTB-XL waveform-level ST/T counterfactual:

```powershell
python -m phase2.run_ptbxl_waveform_counterfactual `
  --out-dir artifacts\phase2\waveform_counterfactual_sttc `
  --template-max-records 800 `
  --n-boot 1000
```

External Georgia waveform-level ST/T counterfactual:

```powershell
python -m phase2.run_georgia_waveform_counterfactuals `
  --out-dir artifacts\phase2\georgia_waveform_counterfactual_sttc_full `
  --max-positive 0 `
  --template-max-records 800 `
  --n-boot 1000
```

## Main Artifacts

| Artifact directory | Keep because |
|---|---|
| `artifacts/phase2/ptbxl_final` | PTB-XL internal classification and `st_segment+t_wave` feature-level counterfactual |
| `artifacts/phase2/external_georgia_full_strict_normal` | Georgia strict-normal external classification |
| `artifacts/phase2/external_georgia_full_no_st_t` | Georgia hard-negative external classification |
| `artifacts/phase2/georgia_counterfactual_controls` | Full Georgia `st_segment+t_wave` feature-level controls |
| `artifacts/phase2/waveform_counterfactual_sttc` | PTB-XL waveform ST/T target, controls, and dose response |
| `artifacts/phase2/georgia_waveform_counterfactual_sttc_full` | Georgia waveform ST/T target, controls, and dose response |

## Result Snapshot

The values below match the focused `st_segment+t_wave` thesis interpretation.

| Layer | PTB-XL | Georgia |
|---|---:|---:|
| Classification AUC | 0.9628 | 0.9088 strict-normal / 0.8228 hard-negative |
| Feature-level ST/T target drop | 0.5240 | 0.5034 |
| Feature-level CCS | 0.3840 | 0.4100 |
| Waveform ST/T target drop | 0.5817 | 0.6032 |
| Waveform pre-QRS control drop | -0.0838 | -0.0601 |

## Archived

Everything outside the focused path is archived:

- exploratory `shape_template` summaries and figures
- XAI baseline side branches
- PTB-XL mechanism-map and non-STTC disease probes
- collapse, spatial, deployable, breakthrough, shortcut, MI, and subgroup trials
- MIMIC bootstrap utility
- stale final-asset builders and focused-claim summaries

Code archives live in `phase2/archive_experiments/`.
Artifact archives live in `artifacts/phase2/archive/`.
