# Phase 2 ECG Mechanism Validation

Focused thesis module.

## Single Claim

Simple clinical counterfactual memory identifies an ST/T mechanism for ST/T abnormality that is strong on PTB-XL and remains consistent on external Georgia data. The strongest evidence is direct waveform-level ST/T counterfactual editing with negative controls and dose-response behavior. This is an XAI/mechanism-validation study, not a classification ranking or deployment statement.

## Main Runs

PTB-XL internal benchmark and counterfactual proof:

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

Supportive PTB-XL disease mechanism map:

```powershell
python -m phase2.run_ptbxl_mechanism_analysis `
  --csv-root Data\Generated_PTBXL_12Lead `
  --out-dir artifacts\phase2\ptbxl_mechanism_map `
  --permutation-repeats 4 `
  --prototypes-per-class 12
```

Focused result summary: Counterfactual Consistency Score, external consistency,
and clinical alignment score.

```powershell
python -m phase2.summarize_phase2_results `
  --out-dir artifacts\phase2\focused_result_summary
```

Final tables and figures:

```powershell
python -m phase2.build_final_assets `
  --out-dir artifacts\phase2\final_assets
```

Official-XAI-style baselines for PTB-XL STTC. This compares ExtraTrees
classification, feature importance, group permutation, TreeSHAP when installed,
and prototype-memory-only scoring against the counterfactual-memory result:

```powershell
python -m phase2.run_xai_baseline_analysis `
  --out-dir artifacts\phase2\xai_baselines_sttc `
  --permutation-repeats 4 `
  --shap-samples 80
```

Waveform-level PTB-XL counterfactual. This directly edits ST/T samples in the
ECG waveform toward a normal template, then re-extracts features and compares
the probability drop against pre-QRS/P-PR/QRS waveform controls:

```powershell
python -m phase2.run_ptbxl_waveform_counterfactual `
  --out-dir artifacts\phase2\waveform_counterfactual_sttc `
  --template-max-records 800 `
  --n-boot 1000
```

External Georgia waveform-level counterfactual. This applies the same PTB-XL
model and normal waveform template to all Georgia ST/T-positive ECGs:

```powershell
python -m phase2.run_georgia_waveform_counterfactuals `
  --out-dir artifacts\phase2\georgia_waveform_counterfactual_sttc_full `
  --max-positive 0 `
  --template-max-records 800 `
  --n-boot 1000
```

Optional external CCS control run. This is slower because it extracts all Georgia
ST/T-positive records and compares ST+shape against non-ST/shape controls:

```powershell
python -m phase2.run_georgia_control_analysis `
  --ptbxl-root Data\Generated_PTBXL_12Lead `
  --georgia-root Data\PhysioNet_Challenge_2020_Georgia `
  --out-dir artifacts\phase2\georgia_counterfactual_controls `
  --target-fs 100 `
  --n-boot 1000
```

Keep the main Phase 2 path limited to the runs above.

## Data Utilities

Create 12-lead PTB-XL CSVs from the existing generated split:

```powershell
python -m phase2.prepare_ptbxl_multilead_data `
  --out-root Data\Generated_PTBXL_12Lead
```

Download or resume Georgia:

```powershell
python -m phase2.download_georgia_dataset `
  --out-root Data\PhysioNet_Challenge_2020_Georgia `
  --workers 4 `
  --timeout 120 `
  --retries 5
```

Bootstrap a tiny MIMIC-IV-ECG sample and verify local WFDB loading:

```powershell
python -m phase2.download_mimic_ecg_sample `
  --out-root Data\MIMIC_IV_ECG_bootstrap `
  --n-records 3 `
  --preview-rows 10
```

## What Is Archived

The following are intentionally out of the main path and live under `archive_experiments/`:

- bridge analyses back to Phase 1
- pre-QRS and shortcut falsification probes
- DL comparison side branches
- Chapman stress tests and older one-off experiments
- thesis-only figure helpers and planning notes

Non-final artifacts are under `artifacts/phase2/archive/`. They are not part of the focused main result.
