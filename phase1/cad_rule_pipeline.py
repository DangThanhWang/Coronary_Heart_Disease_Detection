from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict

import pandas as pd

from phase1.cad_tabular_config import BinSpec, RuleParams
from phase1.cad_rule_evaluation import (
    oof_logistic_metrics,
    oof_rule_score,
    rule_stability,
    safe_spearman,
    score_cad_rate_table,
    threshold_summary,
)
from phase1.cad_rule_plots import plot_score_cad_rate
from phase1.cad_tabular_preprocessing import basic_impute, discretize, onehot_items
from phase1.cad_rule_mining import mine_rules, serialize_rules, split_cad_normal_rules


def run_dataset_analysis(
    name: str,
    df: pd.DataFrame,
    bin_spec: BinSpec,
    rp: RuleParams,
    make_plots: bool = True,
) -> Dict[str, object]:
    del make_plots  # Plotting happens in save_dataset_result.

    df = basic_impute(df)
    y = df["Cath"].astype(int).to_numpy()
    disc = discretize(df, bin_spec)
    onehot = onehot_items(disc)

    full_rules = mine_rules(onehot, rp)
    cad_rules, normal_rules = split_cad_normal_rules(full_rules)

    oof_scores, fold_summary = oof_rule_score(onehot, y, rp)
    rho, rho_p = safe_spearman(oof_scores, y)
    thr = threshold_summary(y, oof_scores, rp.threshold)
    score_table = score_cad_rate_table(y, oof_scores)
    baseline_metrics, oof_predictions = oof_logistic_metrics(onehot, y, rp)

    report = {
        "dataset": name,
        "n": int(len(df)),
        "cad_prevalence": float(y.mean()),
        "n_features_original": int(df.shape[1] - 1),
        "n_items": int(onehot.shape[1]),
        "rule_params": asdict(rp),
        "full_rules": {
            "n_total": int(len(full_rules)),
            "n_cad": int(len(cad_rules)),
            "n_protective": int(len(normal_rules)),
        },
        "oof_protective_score": {
            "spearman_rho_vs_cad": rho,
            "spearman_p": rho_p,
            "min": int(oof_scores.min()) if len(oof_scores) else 0,
            "max": int(oof_scores.max()) if len(oof_scores) else 0,
            **thr,
        },
        "baseline_vs_hybrid": baseline_metrics,
    }

    return {
        "report": report,
        "oof_scores": pd.DataFrame({"Cath": y, "oof_protective_score": oof_scores}),
        "oof_predictions": oof_predictions,
        "fold_summary": fold_summary,
        "score_cad_rate": score_table,
        "rules_cad_top": serialize_rules(cad_rules),
        "rules_protective_top": serialize_rules(normal_rules),
        "stable_rules_cad": rule_stability(onehot, y, rp, target="cad"),
        "stable_rules_protective": rule_stability(onehot, y, rp, target="normal"),
    }


def save_dataset_result(result: Dict[str, object], out_dir: str | Path, make_plots: bool = True) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "report.json").open("w", encoding="utf-8") as fh:
        json.dump(result["report"], fh, indent=2)

    for name, value in result.items():
        if name == "report":
            continue
        if isinstance(value, pd.DataFrame):
            value.to_csv(out_dir / f"{name}.csv", index=False)

    if make_plots:
        plot_score_cad_rate(result["score_cad_rate"], out_dir / "figure_cad_rate_by_oof_score.png")


def summary_row(report: Dict[str, object]) -> Dict[str, float | int | str]:
    score = report["oof_protective_score"]
    baseline = report["baseline_vs_hybrid"]
    full_rules = report["full_rules"]
    return {
        "dataset": str(report["dataset"]),
        "n": int(report["n"]),
        "cad_prevalence": float(report["cad_prevalence"]),
        "n_rules_cad": int(full_rules["n_cad"]),
        "n_rules_protective": int(full_rules["n_protective"]),
        "oof_score_rho": float(score["spearman_rho_vs_cad"]),
        "oof_score_or_high_vs_low": float(score["or_high_vs_low"]),
        "oof_score_or_low_vs_high": float(score["or_low_vs_high"]),
        "cad_rate_high_score": float(score["cad_rate_high_score"]),
        "cad_rate_low_score": float(score["cad_rate_low_score"]),
        "baseline_auc": float(baseline["baseline_roc_auc"]),
        "hybrid_auc": float(baseline["hybrid_roc_auc"]),
        "delta_auc": float(baseline["delta_roc_auc"]),
        "baseline_brier": float(baseline["baseline_brier"]),
        "hybrid_brier": float(baseline["hybrid_brier"]),
        "delta_brier": float(baseline["delta_brier"]),
    }


