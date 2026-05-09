"""Backward-compatible facade for the split Phase 1 modules."""

from phase1.cad_tabular_config import BIN_UCI, BIN_Z, CLEVELAND_URL, HUNGARIAN_URL, Z_ALIZADEH_UCI_ZIP, BinSpec, RuleParams
from phase1.cad_tabular_data import load_uci_heart, load_z_alizadeh, load_z_alizadeh_from_uci, normalize_z_alizadeh
from phase1.cad_rule_evaluation import (
    oof_logistic_metrics,
    oof_rule_score,
    rule_stability,
    safe_spearman,
    score_cad_rate_table,
    threshold_summary,
)
from phase1.cad_rule_pipeline import run_dataset_analysis, save_dataset_result, summary_row
from phase1.cad_rule_plots import plot_score_cad_rate
from phase1.cad_tabular_preprocessing import basic_impute, discretize, infer_keep_columns, onehot_items
from phase1.cad_rule_mining import (
    Antecedent,
    build_rule_match_matrix,
    item_to_feature,
    mine_rules,
    protective_score_unique_features,
    serialize_rules,
    split_cad_normal_rules,
    top_antecedents,
)

__all__ = [
    "Antecedent",
    "BIN_UCI",
    "BIN_Z",
    "CLEVELAND_URL",
    "HUNGARIAN_URL",
    "Z_ALIZADEH_UCI_ZIP",
    "BinSpec",
    "RuleParams",
    "basic_impute",
    "build_rule_match_matrix",
    "discretize",
    "infer_keep_columns",
    "item_to_feature",
    "load_uci_heart",
    "load_z_alizadeh",
    "load_z_alizadeh_from_uci",
    "mine_rules",
    "normalize_z_alizadeh",
    "onehot_items",
    "oof_logistic_metrics",
    "oof_rule_score",
    "plot_score_cad_rate",
    "protective_score_unique_features",
    "rule_stability",
    "run_dataset_analysis",
    "safe_spearman",
    "save_dataset_result",
    "score_cad_rate_table",
    "serialize_rules",
    "split_cad_normal_rules",
    "summary_row",
    "threshold_summary",
    "top_antecedents",
]

