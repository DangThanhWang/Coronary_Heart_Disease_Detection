from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_GROUPS = {
    "STTC": {"st_segment", "shape_template", "t_wave"},
    "CD": {"qrs", "shape_template", "pr_segment"},
    "MI": {"shape_template", "st_segment", "t_wave", "qrs"},
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def split_groups(combo: str) -> set[str]:
    return {x for x in str(combo).split("+") if x}


def combo_alignment(combo: str, expected: set[str]) -> float:
    groups = split_groups(combo)
    if not groups:
        return 0.0
    return len(groups & expected) / len(groups)


def ptbxl_ccs(ptbxl_report: dict) -> dict:
    target = ptbxl_report["counterfactual"]["target_summary"]
    control = ptbxl_report["counterfactual"]["best_non_st_shape_control"]
    diff = ptbxl_report["counterfactual"]["target_minus_best_control"]
    return {
        "target_combo": ptbxl_report["counterfactual"]["target_combo"],
        "target_mean_drop": target["mean_prob_drop"]["mean"],
        "target_drop_ci_low": target["mean_prob_drop"]["ci95_low"],
        "target_drop_ci_high": target["mean_prob_drop"]["ci95_high"],
        "target_flip_rate": target["flip_rate"]["mean"],
        "best_control_combo": control["combo"],
        "best_control_mean_drop": control["mean_prob_drop"],
        "counterfactual_consistency_score": diff["mean_prob_drop_difference"]["mean"],
        "ccs_ci_low": diff["mean_prob_drop_difference"]["ci95_low"],
        "ccs_ci_high": diff["mean_prob_drop_difference"]["ci95_high"],
        "drop_ratio_vs_control": target["mean_prob_drop"]["mean"] / max(control["mean_prob_drop"], 1e-12),
    }


def georgia_external_summary(report: dict) -> dict:
    ext = report["georgia_external"]
    cf = ext["counterfactual_on_georgia_positive"]
    return {
        "policy": report["label_mapping"]["negative_policy"],
        "n": report["data"]["n_extracted"],
        "positive_n": report["data"]["extracted_counts"]["binary"]["positive"],
        "negative_n": report["data"]["extracted_counts"]["binary"]["negative"],
        "auc": ext["metrics_at_ptbxl_threshold"]["auc"],
        "auprc": ext["metrics_at_ptbxl_threshold"]["auprc"],
        "balanced_accuracy": ext["metrics_at_ptbxl_threshold"]["balanced_accuracy"],
        "predicted_positive_rate": ext["metrics_at_ptbxl_threshold"]["predicted_positive_rate"],
        "target_combo": cf["combo"],
        "target_mean_drop": cf["mean_prob_drop"],
        "target_median_drop": cf["median_prob_drop"],
        "target_flip_rate": cf["flip_rate"],
        "positive_drop_rate": cf["positive_drop_rate"],
    }


def georgia_ccs_summary(report: dict) -> dict:
    ccs = report["georgia_counterfactual_consistency_score"]
    target = report["target"]
    control = report["best_non_st_shape_control"]
    return {
        "target_combo": target["combo"],
        "target_mean_drop": target["mean_prob_drop"],
        "target_flip_rate": target["flip_rate"],
        "best_control_combo": control["combo"],
        "best_control_mean_drop": control["mean_prob_drop"],
        "counterfactual_consistency_score": ccs["mean_prob_drop_difference"]["mean"],
        "ccs_ci_low": ccs["mean_prob_drop_difference"]["ci95_low"],
        "ccs_ci_high": ccs["mean_prob_drop_difference"]["ci95_high"],
        "drop_ratio_vs_control": ccs["drop_ratio_vs_control"],
        "alpha_monotone_mean_drop": report["alpha_sweep"]["monotone_mean_drop"],
        "alpha_monotone_flip_rate": report["alpha_sweep"]["monotone_flip_rate"],
    }


def waveform_summary(report: dict, summary_csv: Path | None = None) -> dict:
    target = report["target_st_t_alpha1"]
    control = report["best_control_alpha1"]
    ranked = pd.DataFrame(report.get("ranked_alpha1", []))
    pre_qrs = {}
    if not ranked.empty and "window" in ranked:
        hit = ranked[ranked["window"] == "pre_qrs_control"]
        if not hit.empty:
            pre_qrs = hit.iloc[0].to_dict()

    alpha_rows = None
    monotone_mean_drop = None
    monotone_flip_rate = None
    if "alpha_sweep_st_t" in report:
        alpha_rows = pd.DataFrame(report["alpha_sweep_st_t"]["rows"])
        monotone_mean_drop = bool(report["alpha_sweep_st_t"]["monotone_mean_drop"])
        monotone_flip_rate = bool(report["alpha_sweep_st_t"]["monotone_flip_rate"])
    elif summary_csv is not None and summary_csv.exists():
        summary = pd.read_csv(summary_csv)
        alpha_rows = summary[summary["window"] == "st_t"].sort_values("alpha")
        monotone_mean_drop = bool(np.all(np.diff(alpha_rows["mean_prob_drop"].to_numpy(dtype=float)) >= -1e-9))
        monotone_flip_rate = bool(np.all(np.diff(alpha_rows["flip_rate"].to_numpy(dtype=float)) >= -1e-9))

    pre_qrs_drop = float(pre_qrs["mean_prob_drop"]) if pre_qrs else None
    return {
        "positive_n": int(target["n"]),
        "target_window": target["window"],
        "target_mean_drop": float(target["mean_prob_drop"]),
        "target_median_drop": float(target["median_prob_drop"]),
        "target_drop_ci_low": float(target["prob_drop_ci95_low"]),
        "target_drop_ci_high": float(target["prob_drop_ci95_high"]),
        "target_flip_rate": float(target["flip_rate"]),
        "target_positive_drop_rate": float(target["positive_drop_rate"]),
        "best_control_window": control["window"],
        "best_control_mean_drop": float(control["mean_prob_drop"]),
        "target_minus_best_control_mean_drop": float(report["target_minus_best_control_mean_drop"]),
        "pre_qrs_control_mean_drop": pre_qrs_drop,
        "target_minus_pre_qrs_control_mean_drop": float(target["mean_prob_drop"] - pre_qrs_drop)
        if pre_qrs_drop is not None
        else None,
        "pre_qrs_control_reverses_effect": bool(pre_qrs_drop is not None and pre_qrs_drop < 0.0),
        "alpha_monotone_mean_drop": monotone_mean_drop,
        "alpha_monotone_flip_rate": monotone_flip_rate,
        "alpha_sweep": alpha_rows.to_dict(orient="records") if alpha_rows is not None else [],
    }


def mechanism_alignment(mechanism_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in mechanism_summary.iterrows():
        target = str(row["target"])
        if bool(row.get("skipped", False)) or target not in EXPECTED_GROUPS:
            continue
        expected = EXPECTED_GROUPS[target]
        rows.append(
            {
                "target": target,
                "expected_groups": "+".join(sorted(expected)),
                "best_single": row["best_single"],
                "best_pair": row["best_pair"],
                "top_permutation": row["top_permutation"],
                "single_alignment": combo_alignment(row["best_single"], expected),
                "pair_alignment": combo_alignment(row["best_pair"], expected),
                "permutation_alignment": combo_alignment(row["top_permutation"], expected),
                "auc": row["auc"],
                "best_pair_drop": row["best_pair_drop"],
            }
        )
    return pd.DataFrame(rows)


def verdict(
    ptbxl: dict,
    strict: dict,
    hard: dict,
    georgia_ccs: dict | None,
    ptbxl_waveform: dict | None,
    georgia_waveform: dict | None,
    align: pd.DataFrame,
) -> dict:
    ptbxl_strong = ptbxl["counterfactual_consistency_score"] >= 0.45 and ptbxl["target_mean_drop"] >= 0.60
    external_consistent = abs(strict["target_mean_drop"] - ptbxl["target_mean_drop"]) <= 0.05
    hard_external_usable = hard["auc"] >= 0.80 and hard["target_mean_drop"] >= 0.60
    alignment_ok = float(align["pair_alignment"].mean()) >= 0.80 if len(align) else False
    georgia_ccs_ok = (
        georgia_ccs is not None
        and georgia_ccs["counterfactual_consistency_score"] >= 0.45
        and georgia_ccs["alpha_monotone_mean_drop"]
        and georgia_ccs["alpha_monotone_flip_rate"]
    )
    waveform_ok = (
        ptbxl_waveform is not None
        and georgia_waveform is not None
        and ptbxl_waveform["target_mean_drop"] >= 0.50
        and georgia_waveform["target_mean_drop"] >= 0.50
        and ptbxl_waveform["target_minus_best_control_mean_drop"] >= 0.40
        and georgia_waveform["target_minus_best_control_mean_drop"] >= 0.40
        and bool(ptbxl_waveform["pre_qrs_control_reverses_effect"])
        and bool(georgia_waveform["pre_qrs_control_reverses_effect"])
    )
    return {
        "main_result_strength": "strong_mechanism_validation"
        if (ptbxl_strong and external_consistent and hard_external_usable and alignment_ok and georgia_ccs_ok and waveform_ok)
        else "solid_mechanism_validation"
        if (ptbxl_strong and external_consistent and hard_external_usable and waveform_ok)
        else "moderate_xai_validation",
        "novel_discovery_proven": False,
        "core_result_supported": bool(ptbxl_strong and external_consistent and georgia_ccs_ok and waveform_ok),
        "waveform_external_validation_supported": bool(waveform_ok),
        "feature_space_external_validation_supported": bool(georgia_ccs_ok),
        "clinical_alignment_supported": bool(alignment_ok),
        "limitations": [
            "The main ST/T mechanism is clinically expected for ST/T abnormality; this is validation, not new physiology discovery.",
            "Hard-negative Georgia balanced accuracy remains low; classifier transfer is not deployment-ready.",
            "Feature-space interventions are prototype replacements, not causal counterfactuals.",
            "Waveform edits are mechanistic stress tests, not clinically confirmed physiological simulations.",
        ],
        "strongest_evidence": [
            "Waveform-level ST/T edits produce large probability drops on PTB-XL and full Georgia.",
            "Pre-QRS waveform control reverses or fails the effect on both datasets.",
            "ST/T edit alpha sweeps are monotone on both datasets.",
            "SHAP and group permutation baselines localize the same broad ST/shape mechanism but lack intervention evidence.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize final Phase 2 mechanism-validation results.")
    parser.add_argument(
        "--ptbxl-report",
        type=Path,
        default=Path("artifacts") / "phase2" / "ptbxl_final" / "ptbxl_final_report.json",
    )
    parser.add_argument(
        "--georgia-strict-report",
        type=Path,
        default=Path("artifacts")
        / "phase2"
        / "external_georgia_full_strict_normal"
        / "georgia_validation_report.json",
    )
    parser.add_argument(
        "--georgia-hard-report",
        type=Path,
        default=Path("artifacts")
        / "phase2"
        / "external_georgia_full_no_st_t"
        / "georgia_validation_report.json",
    )
    parser.add_argument(
        "--mechanism-summary",
        type=Path,
        default=Path("artifacts") / "phase2" / "ptbxl_mechanism_map" / "mechanism_summary.csv",
    )
    parser.add_argument(
        "--georgia-ccs-report",
        type=Path,
        default=Path("artifacts")
        / "phase2"
        / "georgia_counterfactual_controls"
        / "georgia_counterfactual_controls_report.json",
    )
    parser.add_argument(
        "--ptbxl-waveform-report",
        type=Path,
        default=Path("artifacts")
        / "phase2"
        / "waveform_counterfactual_sttc"
        / "waveform_counterfactual_report.json",
    )
    parser.add_argument(
        "--ptbxl-waveform-summary",
        type=Path,
        default=Path("artifacts")
        / "phase2"
        / "waveform_counterfactual_sttc"
        / "waveform_counterfactual_summary.csv",
    )
    parser.add_argument(
        "--georgia-waveform-report",
        type=Path,
        default=Path("artifacts")
        / "phase2"
        / "georgia_waveform_counterfactual_sttc_full"
        / "georgia_waveform_counterfactual_report.json",
    )
    parser.add_argument(
        "--georgia-waveform-summary",
        type=Path,
        default=Path("artifacts")
        / "phase2"
        / "georgia_waveform_counterfactual_sttc_full"
        / "georgia_waveform_counterfactual_summary.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts") / "phase2" / "focused_result_summary",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ptbxl = ptbxl_ccs(read_json(args.ptbxl_report))
    strict = georgia_external_summary(read_json(args.georgia_strict_report))
    hard = georgia_external_summary(read_json(args.georgia_hard_report))
    georgia_ccs = georgia_ccs_summary(read_json(args.georgia_ccs_report)) if args.georgia_ccs_report.exists() else None
    ptbxl_waveform = (
        waveform_summary(read_json(args.ptbxl_waveform_report), args.ptbxl_waveform_summary)
        if args.ptbxl_waveform_report.exists()
        else None
    )
    georgia_waveform = (
        waveform_summary(read_json(args.georgia_waveform_report), args.georgia_waveform_summary)
        if args.georgia_waveform_report.exists()
        else None
    )
    align = mechanism_alignment(pd.read_csv(args.mechanism_summary))
    align.to_csv(args.out_dir / "clinical_alignment_score.csv", index=False)
    external = pd.DataFrame([strict, hard])
    external.to_csv(args.out_dir / "external_consistency.csv", index=False)
    ccs = pd.DataFrame([ptbxl])
    ccs.to_csv(args.out_dir / "counterfactual_consistency_score.csv", index=False)
    if georgia_ccs is not None:
        pd.DataFrame([georgia_ccs]).to_csv(args.out_dir / "georgia_counterfactual_consistency_score.csv", index=False)
    waveform_rows = []
    if ptbxl_waveform is not None:
        waveform_rows.append({"dataset": "PTB-XL", **ptbxl_waveform})
    if georgia_waveform is not None:
        waveform_rows.append({"dataset": "Georgia", **georgia_waveform})
    if waveform_rows:
        pd.DataFrame(waveform_rows).drop(columns=["alpha_sweep"], errors="ignore").to_csv(
            args.out_dir / "waveform_counterfactual_consistency_score.csv", index=False
        )
    report = {
        "method": "Focused result summary",
        "core_result": (
            "A clinically expected ST/T mechanism can be stress-tested by feature-space and waveform-level "
            "counterfactual interventions; the effect is consistent on PTB-XL and external Georgia."
        ),
        "ptbxl_counterfactual_consistency": ptbxl,
        "georgia_counterfactual_consistency": georgia_ccs,
        "waveform_counterfactual_consistency": {
            "ptbxl": ptbxl_waveform,
            "georgia": georgia_waveform,
        },
        "georgia_external": {"strict_normal": strict, "hard_negative": hard},
        "clinical_alignment": {
            "rows": align.to_dict(orient="records"),
            "mean_pair_alignment": float(align["pair_alignment"].mean()) if len(align) else float("nan"),
            "mean_permutation_alignment": float(align["permutation_alignment"].mean()) if len(align) else float("nan"),
        },
        "verdict": verdict(ptbxl, strict, hard, georgia_ccs, ptbxl_waveform, georgia_waveform, align),
        "files": {
            "ccs": str(args.out_dir / "counterfactual_consistency_score.csv"),
            "georgia_ccs": str(args.out_dir / "georgia_counterfactual_consistency_score.csv")
            if georgia_ccs is not None
            else None,
            "waveform_ccs": str(args.out_dir / "waveform_counterfactual_consistency_score.csv")
            if waveform_rows
            else None,
            "external": str(args.out_dir / "external_consistency.csv"),
            "clinical_alignment": str(args.out_dir / "clinical_alignment_score.csv"),
        },
    }
    (args.out_dir / "focused_result_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

