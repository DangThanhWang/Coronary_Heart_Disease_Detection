from __future__ import annotations

import warnings
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from phase1.cad_tabular_config import RuleParams


Antecedent = Tuple[str, ...]


def mine_rules(onehot: pd.DataFrame, rp: RuleParams) -> pd.DataFrame:
    try:
        from mlxtend.frequent_patterns import apriori, association_rules
    except ImportError as exc:
        raise ImportError("Phase 1 rule mining needs mlxtend. Install: pip install mlxtend") from exc

    freq = apriori(
        onehot.astype(bool),
        min_support=rp.min_support,
        use_colnames=True,
        max_len=rp.max_len,
        low_memory=True,
    )
    if freq.empty:
        return pd.DataFrame()

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="invalid value encountered in divide", category=RuntimeWarning)
        try:
            rules = association_rules(freq, metric="confidence", min_threshold=rp.min_confidence)
        except TypeError:
            rules = association_rules(
                freq,
                num_itemsets=len(onehot),
                metric="confidence",
                min_threshold=rp.min_confidence,
            )

    if rules.empty:
        return pd.DataFrame()
    rules = rules[rules["lift"] >= rp.min_lift].copy()
    return rules.sort_values(["lift", "confidence", "support"], ascending=False).reset_index(drop=True)


def split_cad_normal_rules(rules: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if rules.empty:
        return pd.DataFrame(), pd.DataFrame()

    cad_target = frozenset({"Cath_Cad_Y"})
    normal_target = frozenset({"Cath_Normal_Y"})

    cad = rules[
        rules["consequents"].apply(lambda s: s == cad_target)
        & rules["antecedents"].apply(_clean_antecedent)
    ].copy()
    normal = rules[
        rules["consequents"].apply(lambda s: s == normal_target)
        & rules["antecedents"].apply(_clean_antecedent)
    ].copy()

    cad = cad.sort_values(["lift", "confidence", "support"], ascending=False).reset_index(drop=True)
    normal = normal.sort_values(["lift", "confidence", "support"], ascending=False).reset_index(drop=True)
    return cad, normal


def top_antecedents(rules: pd.DataFrame, top_n: int) -> List[Antecedent]:
    if rules.empty:
        return []
    dedup = rules.drop_duplicates(subset=["antecedents"]).head(top_n)
    return [tuple(sorted(a)) for a in dedup["antecedents"].tolist()]


def item_to_feature(item: str) -> str:
    if "_bin_" in item:
        return item.split("_bin_")[0] + "_bin"
    parts = item.split("_")
    return "_".join(parts[:-1]) if len(parts) >= 2 else item


def build_rule_match_matrix(onehot_bool: pd.DataFrame, ants: Sequence[Antecedent]) -> np.ndarray:
    n_samples = len(onehot_bool)
    matches = np.zeros((n_samples, len(ants)), dtype=bool)
    for idx, ant in enumerate(ants):
        cols = list(ant)
        if all(col in onehot_bool.columns for col in cols):
            matches[:, idx] = onehot_bool[cols].all(axis=1).to_numpy()
    return matches


def protective_score_unique_features(onehot_bool: pd.DataFrame, ants: Sequence[Antecedent]) -> np.ndarray:
    if not ants:
        return np.zeros(len(onehot_bool), dtype=int)

    rule_match = build_rule_match_matrix(onehot_bool, ants)
    feature_to_rules: Dict[str, List[int]] = {}
    for rule_idx, ant in enumerate(ants):
        for item in ant:
            feature_to_rules.setdefault(item_to_feature(item), []).append(rule_idx)

    scores = np.zeros(len(onehot_bool), dtype=int)
    for rule_idxs in feature_to_rules.values():
        scores += rule_match[:, rule_idxs].any(axis=1).astype(int)
    return scores


def serialize_rules(rules: pd.DataFrame, limit: int = 50) -> pd.DataFrame:
    if rules.empty:
        return pd.DataFrame()
    keep_cols = [
        col
        for col in ["antecedents", "consequents", "support", "confidence", "lift", "leverage", "conviction"]
        if col in rules.columns
    ]
    out = rules[keep_cols].head(limit).copy()
    for col in ["antecedents", "consequents"]:
        if col in out.columns:
            out[col] = out[col].apply(lambda s: " | ".join(sorted(s)))
    return out


def _clean_antecedent(ant: frozenset[str]) -> bool:
    return all(not item.startswith("Cath_") for item in ant)


