# Results Summary

This is the final result snapshot for the focused thesis pipeline.

## Scope

A simple non-deep-learning ECG model learns a clinically coherent ST/T mechanism for ST/T abnormality. The mechanism is supported by feature-level counterfactuals, direct waveform-level ST/T interventions, dose-response behavior, negative controls, and external Georgia validation.

This is not a classification ranking or deployment-readiness statement.

## Final Artifacts

| Component | Artifact |
|---|---|
| PTB-XL internal benchmark | `artifacts/phase2/ptbxl_final/ptbxl_final_report.json` |
| PTB-XL waveform counterfactual | `artifacts/phase2/waveform_counterfactual_sttc/waveform_counterfactual_report.json` |
| Georgia strict-normal external validation | `artifacts/phase2/external_georgia_full_strict_normal/georgia_validation_report.json` |
| Georgia hard-negative external validation | `artifacts/phase2/external_georgia_full_no_st_t/georgia_validation_report.json` |
| Georgia feature-level counterfactual controls | `artifacts/phase2/georgia_counterfactual_controls/georgia_counterfactual_controls_report.json` |
| Georgia waveform counterfactual full | `artifacts/phase2/georgia_waveform_counterfactual_sttc_full/georgia_waveform_counterfactual_report.json` |
| XAI baselines | `artifacts/phase2/xai_baselines_sttc/xai_baseline_report.json` |
| PTB-XL mechanism map | `artifacts/phase2/ptbxl_mechanism_map/mechanism_summary.csv` |
| Focused result summary | `artifacts/phase2/focused_result_summary/focused_result_report.json` |

## Classification

| Dataset / policy | N | AUC | AUPRC | Balanced accuracy | Notes |
|---|---:|---:|---:|---:|---|
| PTB-XL NORM vs STTC | 1,082 | 0.963 | 0.897 | 0.889 | Internal test |
| Georgia strict-normal | 6,341 | 0.909 | 0.960 | 0.790 | External, normal-only negatives |
| Georgia hard-negative | 10,327 | 0.823 | 0.767 | 0.690 | External, all no-ST/T negatives |

Interpretation: classifier transfer is useful but not deployment-ready, especially under hard negatives.

## Feature-Level Counterfactuals

| Dataset | Positive N | Target | Target mean drop | Best control drop | Gap / CCS | Drop ratio |
|---|---:|---|---:|---:|---:|---:|
| PTB-XL | 213 | ST segment + shape | 0.651 | 0.110 | 0.541 | 5.93x |
| Georgia | 4,595 | ST segment + shape | 0.658 | 0.079 | 0.579 | 8.32x |

Interpretation: feature-level ST+shape interventions dominate non-ST/shape controls on both internal and external data.

## Waveform-Level Counterfactuals

| Dataset | Positive N | Target | Mean drop | Median drop | 95% CI | Flip rate | Positive drop rate | Best control drop | Pre-QRS control drop | Gap vs pre-QRS |
|---|---:|---|---:|---:|---|---:|---:|---:|---:|---:|
| PTB-XL | 213 | ST/T waveform edit | 0.582 | 0.685 | [0.533, 0.632] | 0.484 | 0.925 | 0.025 | -0.084 | 0.666 |
| Georgia | 4,596 | ST/T waveform edit | 0.603 | 0.718 | [0.594, 0.614] | 0.391 | 0.946 | 0.025 | -0.060 | 0.663 |

Interpretation: direct waveform ST/T edits produce large probability reductions on both PTB-XL and external Georgia. The pre-QRS control reverses the effect on both datasets, which is the cleanest negative-control evidence that the result is not just an artifact of editing the waveform.

## Waveform Dose Response

| Dataset | Alpha 0.25 | Alpha 0.50 | Alpha 0.75 | Alpha 1.00 | Monotone |
|---|---:|---:|---:|---:|---|
| PTB-XL ST/T edit mean drop | 0.126 | 0.307 | 0.489 | 0.582 | Yes |
| Georgia ST/T edit mean drop | 0.070 | 0.232 | 0.442 | 0.603 | Yes |

Interpretation: increasing the amount of ST/T waveform normalization increases the probability drop, supporting intervention-style explanation faithfulness.

## XAI Baselines

| Baseline | Result |
|---|---|
| ExtraTrees classification | AUC 0.954, below main HGBDT AUC 0.963 |
| ExtraTrees impurity importance | Top group: shape template; top two expected groups include shape and ST |
| ExtraTrees group permutation | Top group: ST segment + shape template; AUC drop 0.303 |
| TreeSHAP | Top group: shape template, second: ST segment |
| Prototype-memory-only | AUC 0.869 |

Interpretation: official-style XAI baselines identify the same broad ST/shape mechanism, but they do not provide waveform intervention, dose-response, or external counterfactual control evidence.

## Final Assets

Final tables and figures are under `artifacts/phase2/final_assets/`.

| Asset | Purpose |
|---|---|
| `table_classification.csv` | PTB-XL and Georgia classification metrics |
| `table_feature_counterfactual.csv` | Feature-level counterfactual target vs controls |
| `table_waveform_counterfactual.csv` | Waveform-level target, controls, pre-QRS negative control |
| `figure_waveform_dose_response.png` | ST/T alpha dose-response on PTB-XL and Georgia |
| `figure_waveform_controls.png` | ST/T target vs waveform controls |
| `figure_validation_layers.png` | Feature-level and waveform-level consistency |

## Supportive Mechanism Map

| Task | Best single group | Best pair | Notes |
|---|---|---|---|
| STTC | ST segment | ST segment + shape template | Main task |
| CD | QRS | Shape template + QRS | Clinically coherent |
| MI | Shape template | ST segment + shape template | Permutation discordance should be discussed |
| HYP | Skipped | Skipped | Current split lacks enough pure HYP support |

## Safe Statements

- The method validates a clinically coherent ST/T mechanism using simple ML.
- Waveform-level counterfactual editing provides stronger XAI evidence than attribution alone.
- The ST/T mechanism remains consistent from PTB-XL to Georgia.
- Negative controls and alpha dose response support explanation faithfulness.

## Do Not State

- Absolute ranking against all published classifiers.
- Clinical deployment readiness.
- True physiological causality.
- A general result across all ECG diagnoses.
- LLCS as the main contribution unless lead-localized analysis is added.

## Thesis Verdict

For an undergraduate thesis, this is strong and sufficient. For a journal-style writeup, this is suitable as an XAI/mechanism-validation study if written with careful limitations. For top-tier clinical venues, additional clinical expert validation or MIMIC-IV-ECG validation would still be needed.
