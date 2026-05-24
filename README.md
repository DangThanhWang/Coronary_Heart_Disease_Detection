# Coronary Heart Disease Detection

Repo này dùng cho hai phần thí nghiệm chính:

- `phase1`: phân tích CAD từ dữ liệu bảng, dùng rule mining để tạo protective score.
- `phase2`: kiểm chứng cơ chế ST/T trên ECG bằng counterfactual, có kiểm tra trên PTB-XL và Georgia.

## Cài Đặt

Tạo môi trường Python:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Cài thư viện:

```powershell
pip install -r requirements.txt
pip install -r phase1\requirements.txt
```

Nếu cần tự build dữ liệu ECG từ raw WFDB/PTB-XL thì cài thêm:

```powershell
pip install wfdb
```

## Chạy Phase 1

Pipeline chính:

```powershell
python -m phase1.run_cad_rule_pipeline --out-dir artifacts\phase1
```

Nếu có sẵn file Z-Alizadeh local:

```powershell
python -m phase1.run_cad_rule_pipeline `
  --z-path Data\z-alizadeh.csv `
  --out-dir artifacts\phase1
```

Chạy nhanh một phần dataset:

```powershell
python -m phase1.run_cad_rule_pipeline --skip-z --out-dir artifacts\phase1
python -m phase1.run_cad_rule_pipeline --skip-cleveland --skip-hungarian --out-dir artifacts\phase1
```

Tạo hình tổng hợp:

```powershell
python -m phase1.build_thesis_phase1_assets `
  --phase1-root artifacts\phase1 `
  --out-dir artifacts\phase1\thesis_assets
```

File cần xem sau khi chạy:

- `artifacts\phase1\summary.csv`
- `artifacts\phase1\summary.json`
- `artifacts\phase1\<dataset>\...`
- `artifacts\phase1\thesis_assets\figure_phase1_summary.png`

## Dữ Liệu Cho Phase 2

Các script Phase 2 giả định có:

- `Data\Generated_PTBXL_12Lead`
- `Data\PhysioNet_Challenge_2020_Georgia`

Nếu cần tạo lại PTB-XL 12-lead:

```powershell
python -m phase2.prepare_ptbxl_multilead_data `
  --out-root Data\Generated_PTBXL_12Lead
```

Nếu cần tải lại Georgia:

```powershell
python -m phase2.download_georgia_dataset `
  --out-root Data\PhysioNet_Challenge_2020_Georgia `
  --workers 4 `
  --timeout 120 `
  --retries 5
```

## Chạy Phase 2

Chạy theo thứ tự dưới đây. Bước 1–3 và 5–7 có thể chạy song song với nhau (mỗi script tự train độc lập trên PTB-XL).

1. PTB-XL internal experiment:

```powershell
python -m phase2.run_ptbxl_main_experiment `
  --csv-root Data\Generated_PTBXL_12Lead `
  --out-dir artifacts\phase2\ptbxl_final `
  --prototypes-per-class 12 `
  --permutation-repeats 8 `
  --n-boot 2000 `
  --n-cases 10
```

2. Georgia external validation, nhóm âm tính strict normal:

```powershell
python -m phase2.run_georgia_external_validation `
  --ptbxl-root Data\Generated_PTBXL_12Lead `
  --georgia-root Data\PhysioNet_Challenge_2020_Georgia `
  --out-dir artifacts\phase2\external_georgia_full_strict_normal `
  --negative-policy strict_normal `
  --max-per-class 0 `
  --target-fs 100
```

3. Georgia external validation, nhóm âm tính hard negative `no_st_t`:

```powershell
python -m phase2.run_georgia_external_validation `
  --ptbxl-root Data\Generated_PTBXL_12Lead `
  --georgia-root Data\PhysioNet_Challenge_2020_Georgia `
  --out-dir artifacts\phase2\external_georgia_full_no_st_t `
  --negative-policy no_st_t `
  --max-per-class 0 `
  --target-fs 100
```

4. Georgia feature-level counterfactual controls:

```powershell
python -m phase2.run_georgia_control_analysis `
  --ptbxl-root Data\Generated_PTBXL_12Lead `
  --georgia-root Data\PhysioNet_Challenge_2020_Georgia `
  --out-dir artifacts\phase2\georgia_counterfactual_controls `
  --target-fs 100 `
  --n-boot 1000
```

5. Waveform counterfactual trên PTB-XL:

```powershell
python -m phase2.run_ptbxl_waveform_counterfactual `
  --out-dir artifacts\phase2\waveform_counterfactual_sttc `
  --template-max-records 800 `
  --n-boot 1000
```

6. Waveform counterfactual trên Georgia:

```powershell
python -m phase2.run_georgia_waveform_counterfactuals `
  --out-dir artifacts\phase2\georgia_waveform_counterfactual_sttc_full `
  --max-positive 0 `
  --template-max-records 800 `
  --n-boot 1000
```

7. Tạo hình luận văn (sau khi tất cả 6 bước trên xong):

```powershell
python thesis_figures\build_figures.py
```

Hình được xuất vào `thesis_figures\output\` và tự copy vào `Thesis\figures\`.

File cần xem sau khi chạy:

- `artifacts\phase2\ptbxl_final\ptbxl_final_report.json`
- `artifacts\phase2\ptbxl_final\negative_control_summary.csv`
- `artifacts\phase2\external_georgia_full_strict_normal\georgia_validation_report.json`
- `artifacts\phase2\external_georgia_full_no_st_t\georgia_validation_report.json`
- `artifacts\phase2\georgia_counterfactual_controls\georgia_counterfactual_controls_report.json`
- `artifacts\phase2\waveform_counterfactual_sttc\waveform_counterfactual_summary.csv`
- `artifacts\phase2\georgia_waveform_counterfactual_sttc_full\georgia_waveform_counterfactual_summary.csv`

README chi tiết hơn nằm ở:

- `phase1\README.md`
- `phase2\README.md`

