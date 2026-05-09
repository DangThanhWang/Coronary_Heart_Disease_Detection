from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from phase1.cad_tabular_config import RuleParams
from phase1.cad_rule_mining import item_to_feature, mine_rules, protective_score_unique_features, split_cad_normal_rules, top_antecedents


def safe_spearman(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    if len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    rho, p_value = spearmanr(x, y)
    return float(rho), float(p_value)


def threshold_summary(y: np.ndarray, scores: np.ndarray, threshold: int) -> Dict[str, float]:
    high = scores >= threshold
    low = ~high

    a = int((high & (y == 1)).sum())
    b = int((high & (y == 0)).sum())
    c = int((low & (y == 1)).sum())
    d = int((low & (y == 0)).sum())

    aa, bb, cc, dd = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    or_high_vs_low = (aa * dd) / (bb * cc)
    or_low_vs_high = 1.0 / or_high_vs_low if or_high_vs_low else float("inf")
    _, p_value = fisher_exact([[a, b], [c, d]])

    return {
        "n_high_score": int(high.sum()),
        "n_low_score": int(low.sum()),
        "cad_rate_high_score": float(y[high].mean()) if high.any() else float("nan"),
        "cad_rate_low_score": float(y[low].mean()) if low.any() else float("nan"),
        "or_high_vs_low": float(or_high_vs_low),
        "or_low_vs_high": float(or_low_vs_high),
        "fisher_p": float(p_value),
        "high_cad": a,
        "high_normal": b,
        "low_cad": c,
        "low_normal": d,
    }


def score_cad_rate_table(y: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    tmp = pd.DataFrame({"score": scores, "Cath": y})
    return tmp.groupby("score", as_index=False)["Cath"].agg(cad_rate="mean", n="count")


def oof_rule_score(onehot: pd.DataFrame, y: np.ndarray, rp: RuleParams) -> Tuple[np.ndarray, pd.DataFrame]:
    skf = StratifiedKFold(n_splits=rp.n_splits, shuffle=True, random_state=rp.random_state)
    scores = np.zeros(len(y), dtype=int)
    fold_rows = []
    onehot_bool = onehot.astype(bool)

    for fold, (train_idx, test_idx) in enumerate(skf.split(onehot, y), start=1):
        rules = mine_rules(onehot.iloc[train_idx], rp)
        _, normal_rules = split_cad_normal_rules(rules)
        ants = top_antecedents(normal_rules, rp.top_n_rules)
        scores[test_idx] = protective_score_unique_features(onehot_bool.iloc[test_idx], ants)
        fold_rows.append(
            {
                "fold": fold,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "n_rules_total": int(len(rules)),
                "n_protective_rules": int(len(normal_rules)),
                "n_antecedents_used": int(len(ants)),
            }
        )

    return scores, pd.DataFrame(fold_rows)


def oof_logistic_metrics(
    onehot: pd.DataFrame,
    y: np.ndarray,
    rp: RuleParams,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    x_full = onehot[[col for col in onehot.columns if not col.startswith("Cath_")]].astype(float)
    skf = StratifiedKFold(n_splits=rp.n_splits, shuffle=True, random_state=rp.random_state)
    onehot_bool = onehot.astype(bool)

    base_probs = np.zeros(len(y), dtype=float)
    hybrid_probs = np.zeros(len(y), dtype=float)
    hybrid_scores = np.zeros(len(y), dtype=int)

    for train_idx, test_idx in skf.split(x_full, y):
        x_train = x_full.iloc[train_idx]
        x_test = x_full.iloc[test_idx]
        y_train = y[train_idx]

        rules = mine_rules(onehot.iloc[train_idx], rp)
        _, normal_rules = split_cad_normal_rules(rules)
        ants = top_antecedents(normal_rules, rp.top_n_rules)

        score_train = protective_score_unique_features(onehot_bool.iloc[train_idx], ants)
        score_test = protective_score_unique_features(onehot_bool.iloc[test_idx], ants)
        hybrid_scores[test_idx] = score_test

        base = LogisticRegression(max_iter=2000, solver="liblinear")
        base.fit(x_train, y_train)
        base_probs[test_idx] = base.predict_proba(x_test)[:, 1]

        x_train_h = x_train.copy()
        x_test_h = x_test.copy()
        x_train_h["protective_score"] = score_train
        x_test_h["protective_score"] = score_test

        hybrid = LogisticRegression(max_iter=2000, solver="liblinear")
        hybrid.fit(x_train_h, y_train)
        hybrid_probs[test_idx] = hybrid.predict_proba(x_test_h)[:, 1]

    metrics = {
        "baseline_roc_auc": float(roc_auc_score(y, base_probs)),
        "baseline_pr_auc": float(average_precision_score(y, base_probs)),
        "baseline_brier": float(brier_score_loss(y, base_probs)),
        "hybrid_roc_auc": float(roc_auc_score(y, hybrid_probs)),
        "hybrid_pr_auc": float(average_precision_score(y, hybrid_probs)),
        "hybrid_brier": float(brier_score_loss(y, hybrid_probs)),
        "delta_roc_auc": float(roc_auc_score(y, hybrid_probs) - roc_auc_score(y, base_probs)),
        "delta_brier": float(brier_score_loss(y, hybrid_probs) - brier_score_loss(y, base_probs)),
    }
    preds = pd.DataFrame(
        {
            "y": y,
            "oof_baseline_prob": base_probs,
            "oof_hybrid_prob": hybrid_probs,
            "oof_hybrid_score": hybrid_scores,
        }
    )
    return metrics, preds


def rule_stability(onehot: pd.DataFrame, y: np.ndarray, rp: RuleParams, target: str = "normal") -> pd.DataFrame:
    skf = StratifiedKFold(n_splits=rp.n_splits, shuffle=True, random_state=rp.random_state)
    rows: Dict[str, Dict[str, object]] = {}

    for fold, (train_idx, _) in enumerate(skf.split(onehot, y), start=1):
        rules = mine_rules(onehot.iloc[train_idx], rp)
        cad_rules, normal_rules = split_cad_normal_rules(rules)
        selected = normal_rules if target == "normal" else cad_rules
        top_rules = selected.drop_duplicates(subset=["antecedents"]).head(rp.top_n_rules)

        for _, row in top_rules.iterrows():
            ant = tuple(sorted(row["antecedents"]))
            key = " | ".join(ant)
            item = rows.setdefault(
                key,
                {
                    "antecedent": key,
                    "features": " | ".join(sorted({item_to_feature(x) for x in ant})),
                    "fold_count": 0,
                    "folds": [],
                    "mean_support": [],
                    "mean_confidence": [],
                    "mean_lift": [],
                },
            )
            item["fold_count"] = int(item["fold_count"]) + 1
            item["folds"].append(fold)
            item["mean_support"].append(float(row["support"]))
            item["mean_confidence"].append(float(row["confidence"]))
            item["mean_lift"].append(float(row["lift"]))

    out_rows = []
    for item in rows.values():
        out_rows.append(
            {
                "antecedent": item["antecedent"],
                "features": item["features"],
                "fold_count": item["fold_count"],
                "fold_rate": float(item["fold_count"]) / rp.n_splits,
                "folds": ",".join(map(str, item["folds"])),
                "mean_support": float(np.mean(item["mean_support"])),
                "mean_confidence": float(np.mean(item["mean_confidence"])),
                "mean_lift": float(np.mean(item["mean_lift"])),
            }
        )
    if not out_rows:
        return pd.DataFrame()
    return pd.DataFrame(out_rows).sort_values(
        ["fold_count", "mean_lift", "mean_confidence"], ascending=False
    ).reset_index(drop=True)


