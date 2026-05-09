from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def pick_spatial_report(mode: str) -> Path:
    if mode == "quick":
        return Path("artifacts") / "phase2" / "direction2_spatial_v2_quick" / "direction2_spatial_v2_report.json"
    return Path("artifacts") / "phase2" / "direction2_spatial_v2_full" / "direction2_spatial_v2_report.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Mechanism collapse benchmark for current Phase 2 ECG pipeline.")
    parser.add_argument("--mode", choices=["quick", "full"], default="full")
    parser.add_argument(
        "--focused-report",
        type=Path,
        default=Path("artifacts") / "phase2" / "focused_result_summary" / "focused_result_report.json",
    )
    parser.add_argument(
        "--xai-report",
        type=Path,
        default=Path("artifacts") / "phase2" / "xai_baselines_sttc" / "xai_baseline_report.json",
    )
    parser.add_argument(
        "--direction1-report",
        type=Path,
        default=Path("artifacts") / "phase2" / "archive" / "direction1_mechanism_discovery_scan" / "direction1_scan_report.json",
    )
    parser.add_argument("--spatial-report", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts") / "phase2" / "mechanism_collapse_benchmark")
    args = parser.parse_args()

    spatial_path = args.spatial_report or pick_spatial_report(args.mode)
    focused = load_json(args.focused_report)
    xai = load_json(args.xai_report)
    d1 = load_json(args.direction1_report)
    d2 = load_json(spatial_path)

    georgia_hard = focused["georgia_external"]["hard_negative"]
    wf_ptb = focused["waveform_counterfactual_consistency"]["ptbxl"]
    wf_geo = focused["waveform_counterfactual_consistency"]["georgia"]

    unknown_min = float(d1["best_unknown_combo"].get("cross_dataset_min_drop", 0.0))
    spatial_ratio_ptb = float(d2["ptbxl_spatial_score"]["regional_vs_global_ratio"])
    spatial_ratio_geo = float(d2["georgia_spatial_score"]["regional_vs_global_ratio"])
    spatial_only_geo = float(d2["georgia_spatial_score"]["spatial_only_vs_global_ratio"])

    checks = {
        "counterfactual_chain_strong": bool(
            wf_ptb["target_minus_best_control_mean_drop"] >= 0.40
            and wf_geo["target_minus_best_control_mean_drop"] >= 0.40
            and wf_ptb["alpha_monotone_mean_drop"]
            and wf_geo["alpha_monotone_mean_drop"]
        ),
        "classification_transfer_strong": bool(georgia_hard["balanced_accuracy"] >= 0.75),
        "unknown_mechanism_present": bool(unknown_min >= 0.20),
        "spatial_localization_strong": bool(spatial_ratio_ptb >= 0.85 and spatial_ratio_geo >= 0.85),
        "spatial_signal_nontrivial": bool(spatial_only_geo >= 0.35),
        "baseline_xai_insufficient": bool(
            not xai["verdict"]["baseline_has_actionable_counterfactual_evidence"]
            and not xai["verdict"]["baseline_has_external_counterfactual_control"]
        ),
    }
    pass_count = int(sum(1 for ok in checks.values() if ok))

    verdict = "mechanism_audit_strong_with_collapse"
    if checks["classification_transfer_strong"] and checks["unknown_mechanism_present"] and checks["spatial_localization_strong"]:
        verdict = "collapse_reduced_candidate_extension"

    report = {
        "task": "Mechanism Collapse Benchmark",
        "mode": args.mode,
        "inputs": {
            "focused_report": str(args.focused_report),
            "xai_report": str(args.xai_report),
            "direction1_report": str(args.direction1_report),
            "spatial_report": str(spatial_path),
        },
        "key_metrics": {
            "ptbxl_target_minus_best_control": wf_ptb["target_minus_best_control_mean_drop"],
            "georgia_target_minus_best_control": wf_geo["target_minus_best_control_mean_drop"],
            "georgia_hard_balanced_accuracy": georgia_hard["balanced_accuracy"],
            "best_unknown_cross_dataset_min_drop": unknown_min,
            "ptbxl_regional_vs_global_ratio": spatial_ratio_ptb,
            "georgia_regional_vs_global_ratio": spatial_ratio_geo,
            "georgia_spatial_only_vs_global_ratio": spatial_only_geo,
        },
        "checks": checks,
        "pass_count": pass_count,
        "total_checks": int(len(checks)),
        "collapse_summary": {
            "st_t_dominance_confirmed": bool(checks["counterfactual_chain_strong"]),
            "unknown_mechanism_gap": bool(not checks["unknown_mechanism_present"]),
            "spatial_localization_gap": bool(not checks["spatial_localization_strong"]),
            "external_transfer_gap": bool(not checks["classification_transfer_strong"]),
        },
        "verdict": verdict,
    }

    out_dir = args.out_dir / args.mode
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "mechanism_collapse_benchmark_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"verdict": verdict, "pass_count": pass_count, "report": str(out)}, indent=2))


if __name__ == "__main__":
    main()
