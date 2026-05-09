from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from phase1.cad_tabular_config import BinSpec


def basic_impute(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if not out[col].isna().any():
            continue
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].fillna(out[col].median())
        else:
            mode = out[col].mode(dropna=True)
            fill_value = mode.iloc[0] if not mode.empty else "missing"
            out[col] = out[col].fillna(fill_value)
    return out


def infer_keep_columns(df: pd.DataFrame, bin_spec: BinSpec, max_categories: int = 10) -> List[str]:
    keep: List[str] = []
    binned = set(bin_spec.bins)
    for col in df.columns:
        if col == "Cath" or col in binned:
            continue
        nunique = df[col].nunique(dropna=True)
        if not pd.api.types.is_numeric_dtype(df[col]) or nunique <= max_categories:
            keep.append(col)
    return keep


def discretize(df: pd.DataFrame, bin_spec: BinSpec, keep_cols: Optional[Sequence[str]] = None) -> pd.DataFrame:
    if "Cath" not in df.columns:
        raise ValueError("DataFrame must contain Cath coded 0/1.")

    df = basic_impute(df)
    disc = pd.DataFrame(index=df.index)

    for col, (bins, labels) in bin_spec.bins.items():
        if col not in df.columns:
            continue
        cut = pd.cut(df[col], bins=bins, labels=labels, include_lowest=True)
        disc[f"{col}_bin"] = cut.astype("object").where(cut.notna(), "missing").astype(str)

    selected = list(keep_cols) if keep_cols is not None else infer_keep_columns(df, bin_spec)
    for col in selected:
        if col in df.columns:
            disc[col] = df[col].astype(str)

    y = df["Cath"].astype(int).to_numpy()
    disc["Cath_Cad"] = np.where(y == 1, "Y", "N")
    disc["Cath_Normal"] = np.where(y == 0, "Y", "N")
    return disc


def onehot_items(disc: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(disc.astype(str), dtype=bool)


