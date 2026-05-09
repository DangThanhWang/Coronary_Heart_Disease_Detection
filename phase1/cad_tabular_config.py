from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple


Z_ALIZADEH_UCI_ZIP = "https://archive.ics.uci.edu/static/public/412/z%2Balizadeh%2Bsani.zip"
CLEVELAND_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
HUNGARIAN_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data"


@dataclass(frozen=True)
class RuleParams:
    min_support: float = 0.06
    min_confidence: float = 0.60
    min_lift: float = 1.40
    max_len: int = 3
    top_n_rules: int = 10
    threshold: int = 2
    n_splits: int = 5
    random_state: int = 0


@dataclass(frozen=True)
class BinSpec:
    bins: Dict[str, Tuple[List[float], List[str]]]


BIN_Z = BinSpec(
    bins={
        "Age": ([-math.inf, 40, 60, math.inf], ["young", "mid", "old"]),
        "BMI": ([-math.inf, 25, 30, math.inf], ["normal", "over", "obese"]),
        "BP": ([-math.inf, 90, 130, math.inf], ["normal", "high", "very_high"]),
        "PR": ([-math.inf, 60, 100, math.inf], ["brady", "normal", "tachy"]),
        "FBS": ([-math.inf, 100, 126, math.inf], ["normal", "preDM", "DM"]),
        "LDL": ([-math.inf, 100, 130, 160, math.inf], ["optimal", "near_opt", "border", "high"]),
        "HDL": ([-math.inf, 40, 60, math.inf], ["low", "mid", "high"]),
        "TG": ([-math.inf, 150, 200, math.inf], ["normal", "high", "vhigh"]),
        "EF-TTE": ([-math.inf, 35, 50, math.inf], ["poor", "mid", "good"]),
    }
)


BIN_UCI = BinSpec(
    bins={
        "Age": ([-math.inf, 40, 60, math.inf], ["young", "mid", "old"]),
        "Trestbps": ([-math.inf, 120, 140, math.inf], ["normal", "elevated", "high"]),
        "Chol": ([-math.inf, 200, 240, math.inf], ["desirable", "borderline", "high"]),
        "Thalach": ([-math.inf, 120, 160, math.inf], ["low", "mid", "high"]),
        "Oldpeak": ([-math.inf, 0.5, 1.5, math.inf], ["none", "mild", "significant"]),
    }
)

