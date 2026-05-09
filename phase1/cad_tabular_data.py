from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from phase1.cad_tabular_config import Z_ALIZADEH_UCI_ZIP
from phase1.cad_tabular_preprocessing import basic_impute


def load_z_alizadeh(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".xls", ".xlsx"}:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    return normalize_z_alizadeh(df)


def load_z_alizadeh_from_uci(cache_dir: str | Path = "Data/phase1_cache") -> pd.DataFrame:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / "z_alizadeh_sani.zip"
    if not zip_path.exists():
        urllib.request.urlretrieve(Z_ALIZADEH_UCI_ZIP, zip_path)

    with zipfile.ZipFile(zip_path) as zf:
        xlsx_names = [n for n in zf.namelist() if n.lower().endswith((".xlsx", ".xls"))]
        if not xlsx_names:
            raise RuntimeError("UCI Z-Alizadeh archive does not contain an Excel file.")
        with zf.open(xlsx_names[0]) as fh:
            data = io.BytesIO(fh.read())

    try:
        df = pd.read_excel(data)
    except ImportError as exc:
        raise ImportError("Reading Z-Alizadeh Excel needs openpyxl. Install: pip install openpyxl") from exc
    return normalize_z_alizadeh(df)


def normalize_z_alizadeh(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]

    if "Sex" in out.columns:
        out["Sex"] = out["Sex"].replace({"Fmale": "Female", "female": "Female", "male": "Male"})

    if "Cath" not in out.columns:
        raise ValueError("Z-Alizadeh data must contain a Cath column.")

    out = basic_impute(out)

    bin_01 = [
        "DM",
        "HTN",
        "Current Smoker",
        "EX-Smoker",
        "Edema",
        "Q Wave",
        "St Elevation",
        "St Depression",
        "Tinversion",
    ]
    for col in bin_01:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)

    bin_yn = [
        "Obesity",
        "CRF",
        "CVA",
        "Airway disease",
        "Thyroid Disease",
        "CHF",
        "DLP",
        "Weak Peripheral Pulse",
        "Lung rales",
        "Systolic Murmur",
        "Diastolic Murmur",
        "Dyspnea",
        "Typical Chest Pain",
        "Atypical",
        "Nonanginal",
        "LowTH Ang",
        "LVH",
        "Poor R Progression",
    ]
    for col in bin_yn:
        if col in out.columns:
            values = out[col].astype(str).str.strip().str.upper()
            out[col] = values.isin({"Y", "YES", "1", "TRUE"}).astype(int)

    cath = out["Cath"]
    if not pd.api.types.is_numeric_dtype(cath):
        out["Cath"] = cath.astype(str).str.strip().str.lower().map({"cad": 1, "normal": 0})
    out["Cath"] = pd.to_numeric(out["Cath"], errors="coerce").astype(int)
    return out


def load_uci_heart(url: str) -> pd.DataFrame:
    cols = [
        "Age",
        "Sex",
        "CP",
        "Trestbps",
        "Chol",
        "FBS",
        "Restecg",
        "Thalach",
        "Exang",
        "Oldpeak",
        "Slope",
        "Ca",
        "Thal",
        "Num",
    ]
    df = pd.read_csv(url, header=None, names=cols, na_values="?")
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = basic_impute(df)
    df["Cath"] = (df["Num"] > 0).astype(int)
    return df.drop(columns=["Num"])


