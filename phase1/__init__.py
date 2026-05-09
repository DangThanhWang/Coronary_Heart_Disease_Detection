"""Phase 1 CAD tabular rule-XAI module."""

from phase1.cad_tabular_config import BIN_UCI, BIN_Z, CLEVELAND_URL, HUNGARIAN_URL, BinSpec, RuleParams
from phase1.cad_tabular_data import load_uci_heart, load_z_alizadeh, load_z_alizadeh_from_uci
from phase1.cad_rule_pipeline import run_dataset_analysis, save_dataset_result, summary_row

__all__ = [
    "BIN_UCI",
    "BIN_Z",
    "CLEVELAND_URL",
    "HUNGARIAN_URL",
    "BinSpec",
    "RuleParams",
    "load_uci_heart",
    "load_z_alizadeh",
    "load_z_alizadeh_from_uci",
    "run_dataset_analysis",
    "save_dataset_result",
    "summary_row",
]

