from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score

from real_ml_xai_llcs.features import Sample, iter_csv_samples, iter_train_env_samples


BASE_GROUP_ORDER = [
    "st_segment",
    "shape_template",
    "t_wave",
    "global_morph",
    "p_wave",
    "pr_segment",
    "qrs",
    "rhythm",
]

LEAD_RE = re.compile(r"lead\d+_(Lead_[A-Z0-9_]+)_")
REGION_LEADS = {
    "inferior": frozenset({"II", "III", "AVF"}),
    "anterior": frozenset({"V1", "V2", "V3", "V4"}),
    "lateral": frozenset({"I", "AVL", "V5", "V6"}),
    "septal": frozenset({"V1", "V2"}),
}


def map_y(raw: np.ndarray) -> np.ndarray:
    return np.asarray([0 if int(v) == 0 else 1 for v in raw], dtype=int)


def load_ptbxl_norm_sttc_samples(csv_root: Path) -> tuple[list[Sample], list[Sample], list[Sample], list[Sample]]:
    train = [s for s in iter_train_env_samples(csv_root) if s.label in (0, 2)]
    val = [s for s in iter_csv_samples(csv_root, ["ValID"]) if s.label in (0, 2)]
    test = [s for s in iter_csv_samples(csv_root, ["Eval_ID"]) if s.label in (0, 2)]
    ood = [s for s in iter_csv_samples(csv_root, ["Eval_OOD"]) if s.label == 4]
    return train, val, test, ood


def make_hgbdt(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=250,
        learning_rate=0.05,
        class_weight="balanced",
        l2_regularization=0.05,
        random_state=seed,
    )


def select_threshold(y: np.ndarray, prob_positive: np.ndarray) -> float:
    cand = np.unique(prob_positive)
    if cand.size > 512:
        cand = np.quantile(cand, np.linspace(0.01, 0.99, 512))
    best_t, best = 0.5, -np.inf
    for t in cand:
        pred = (prob_positive >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        score = sens + spec - 1.0
        if score > best:
            best = score
            best_t = float(t)
    return best_t


def cls_metrics(y: np.ndarray, prob_positive: np.ndarray, threshold: float) -> dict:
    pred = prob_positive >= threshold
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "auc": float(roc_auc_score(y, prob_positive)),
        "auprc": float(average_precision_score(y, prob_positive)),
        "threshold": float(threshold),
        "positive_rate": float(np.mean(y)),
        "predicted_positive_rate": float(np.mean(pred)),
    }


def bootstrap_metric_ci(
    y: np.ndarray,
    prob_positive: np.ndarray,
    threshold: float,
    metric: str,
    n_boot: int,
    seed: int,
) -> dict:
    y = np.asarray(y)
    prob_positive = np.asarray(prob_positive)
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), size=len(y))
        yy = y[idx]
        pp = prob_positive[idx]
        if len(np.unique(yy)) < 2:
            continue
        if metric == "auc":
            stats.append(float(roc_auc_score(yy, pp)))
        elif metric == "auprc":
            stats.append(float(average_precision_score(yy, pp)))
        elif metric == "balanced_accuracy":
            stats.append(float(balanced_accuracy_score(yy, pp >= threshold)))
        elif metric == "macro_f1":
            stats.append(float(f1_score(yy, pp >= threshold, average="macro", zero_division=0)))
        else:
            raise ValueError(f"Unknown metric: {metric}")
    lo, hi = np.percentile(stats, [2.5, 97.5])
    full = cls_metrics(y, prob_positive, threshold)
    return {"mean": float(full[metric]), "ci95_low": float(lo), "ci95_high": float(hi), "n_boot_valid": len(stats)}


def bootstrap_mean_ci(values: np.ndarray, n_boot: int, seed: int) -> dict:
    values = np.asarray(values)
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(values), size=len(values))
        stats.append(float(np.mean(values[idx])))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return {"mean": float(np.mean(values)), "ci95_low": float(lo), "ci95_high": float(hi)}


def paired_cohen_d(diff: np.ndarray) -> float:
    diff = np.asarray(diff, dtype=float)
    return float(np.mean(diff) / max(np.std(diff, ddof=1), 1e-12))


def feature_groups(feature_names: Sequence[str]) -> dict[str, list[int]]:
    groups = {
        "rhythm": [],
        "p_wave": [],
        "pr_segment": [],
        "qrs": [],
        "st_segment": [],
        "t_wave": [],
        "global_morph": [],
        "shape_template": [],
    }
    for i, name in enumerate(feature_names):
        if name in {"n_peaks", "fs", "rr_mean", "rr_std", "rr_cv", "heart_rate"}:
            groups["rhythm"].append(i)
        elif "_p_" in name:
            groups["p_wave"].append(i)
        elif "_pr_" in name:
            groups["pr_segment"].append(i)
        elif "_qrs_" in name:
            groups["qrs"].append(i)
        elif "_st_" in name:
            groups["st_segment"].append(i)
        elif "_t_" in name:
            groups["t_wave"].append(i)
        elif any(name.endswith(s) for s in ("_mean", "_std", "_min", "_max")):
            groups["global_morph"].append(i)
        else:
            groups["shape_template"].append(i)
    return {k: v for k, v in groups.items() if v}


def lead_name_from_feature(feature_name: str) -> str | None:
    match = LEAD_RE.search(feature_name)
    if not match:
        return None
    return match.group(1).replace("Lead_", "")


def augment_feature_groups_with_regions(
    feature_names: Sequence[str],
    group_cols: dict[str, list[int]],
    regions: dict[str, frozenset[str]] | None = None,
) -> dict[str, list[int]]:
    regions = regions or REGION_LEADS
    out = {name: sorted(set(cols)) for name, cols in group_cols.items()}
    for idx, name in enumerate(feature_names):
        lead = lead_name_from_feature(name)
        if lead is None:
            continue
        if "_st_" in name:
            base_group = "st_segment"
        else:
            base_group = "shape_template"
        if base_group not in {"st_segment", "shape_template"}:
            continue
        for region_name, region_leads in regions.items():
            if lead not in region_leads:
                continue
            out.setdefault(f"{region_name}_{base_group}", []).append(idx)
    return {name: sorted(set(cols)) for name, cols in out.items() if cols}


def ordered_group_names(group_cols: dict[str, list[int]]) -> list[str]:
    base = [name for name in BASE_GROUP_ORDER if name in group_cols]
    extras = sorted(name for name in group_cols if name not in BASE_GROUP_ORDER)
    return base + extras


def apply_counterfactual(
    x: np.ndarray,
    norm_proto: np.ndarray,
    groups: tuple[str, ...],
    group_cols: dict[str, list[int]],
    alpha: float,
) -> np.ndarray:
    out = x.copy()
    cols: list[int] = []
    for group in groups:
        cols.extend(group_cols[group])
    if cols:
        out[:, cols] = (1.0 - alpha) * out[:, cols] + alpha * norm_proto[:, cols]
    return out


def group_columns(group_cols: dict[str, list[int]], groups: tuple[str, ...]) -> list[int]:
    cols: list[int] = []
    for group in groups:
        cols.extend(group_cols[group])
    return sorted(set(cols))
