from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from phase2.ecg_mechanism_core import (
    BASE_GROUP_ORDER,
    apply_counterfactual,
    bootstrap_mean_ci,
    feature_groups,
    load_ptbxl_norm_sttc_samples,
    make_hgbdt,
    map_y,
    paired_cohen_d,
    select_threshold,
)
from phase2.prototype_memory import CounterfactualMemory
from phase2.run_georgia_external_validation import (
    build_georgia_matrix,
    discover_georgia,
)
from phase2.ecg_features import build_feature_matrix


TARGET_COMBO = ("st_segment", "t_wave")


def combo_name(combo: tuple[str, ...]) -> str:
    return "+".join(combo)


def control_combos(group_cols: dict[str, list[int]]) -> list[tuple[str, ...]]:
    names = [g for g in BASE_GROUP_ORDER if g in group_cols and g not in set(TARGET_COMBO)]
    combos = [(g,) for g in names]
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            combos.append((first, second))
    return combos


def nearest_norm_prototypes(
    scaler: StandardScaler,
    memory: CounterfactualMemory,
    x: np.ndarray,
) -> np.ndarray:
    x_z = scaler.transform(x)
    norm_idx, _ = memory.nearest_by_label(x_z, 0)
    return scaler.inverse_transform(memory.prototype_matrix_[norm_idx])  # type: ignore[index]


def apply_combo(
    x: np.ndarray,
    norm_proto: np.ndarray,
    group_cols: dict[str, list[int]],
    combo: tuple[str, ...],
    alpha: float,
) -> np.ndarray:
    return apply_counterfactual(x, norm_proto, combo, group_cols, alpha=alpha)


def per_case_drop(
    model: object,
    x: np.ndarray,
    norm_proto: np.ndarray,
    group_cols: dict[str, list[int]],
    combo: tuple[str, ...],
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base_prob = model.predict_proba(x)[:, 1]
    x_cf = apply_combo(x, norm_proto, group_cols, combo, alpha)
    cf_prob = model.predict_proba(x_cf)[:, 1]
    return base_prob - cf_prob, base_prob, cf_prob


def summarize_combo(
    model: object,
    x: np.ndarray,
    norm_proto: np.ndarray,
    threshold: float,
    group_cols: dict[str, list[int]],
    combo: tuple[str, ...],
    alpha: float,
) -> dict:
    drop, base_prob, cf_prob = per_case_drop(model, x, norm_proto, group_cols, combo, alpha)
    return {
        "combo": combo_name(combo),
        "n_groups": len(combo),
        "n_features": int(sum(len(group_cols[g]) for g in combo)),
        "alpha": float(alpha),
        "n": int(len(x)),
        "base_positive_rate": float(np.mean(base_prob >= threshold)),
        "mean_prob_drop": float(np.mean(drop)),
        "median_prob_drop": float(np.median(drop)),
        "positive_drop_rate": float(np.mean(drop > 0)),
        "flip_rate": float(np.mean((base_prob >= threshold) & (cf_prob < threshold))),
        "mean_cf_prob": float(np.mean(cf_prob)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Georgia feature-level counterfactual controls.")
    parser.add_argument("--ptbxl-root", type=Path, default=Path("Data") / "Generated_PTBXL_12Lead")
    parser.add_argument("--georgia-root", type=Path, default=Path("Data") / "PhysioNet_Challenge_2020_Georgia")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts") / "phase2" / "georgia_counterfactual_controls",
    )
    parser.add_argument("--pre", type=int, default=80)
    parser.add_argument("--post", type=int, default=120)
    parser.add_argument("--downsample", type=int, default=32)
    parser.add_argument("--target-fs", type=float, default=100.0)
    parser.add_argument("--prototypes-per-class", type=int, default=12)
    parser.add_argument("--max-positive", type=int, default=0, help="0 means all Georgia ST/T positives.")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_s, val_s, _, _ = load_ptbxl_norm_sttc_samples(args.ptbxl_root)
    x_train, y_train_raw, feature_names, train_ok, train_fail = build_feature_matrix(
        train_s, args.pre, args.post, args.downsample
    )
    x_val, y_val_raw, _, _, val_fail = build_feature_matrix(val_s, args.pre, args.post, args.downsample)
    y_train = map_y(y_train_raw)
    y_val = map_y(y_val_raw)

    model = make_hgbdt(args.seed)
    model.fit(x_train, y_train)
    threshold = select_threshold(y_val, model.predict_proba(x_val)[:, 1])
    scaler = StandardScaler()
    memory = CounterfactualMemory(prototypes_per_class=args.prototypes_per_class, seed=args.seed)
    memory.fit(scaler.fit_transform(x_train), y_train, [s.path.name for s in train_ok])
    group_cols = feature_groups(feature_names)

    rows = [row for row in discover_georgia(args.georgia_root, "no_st_t", None, args.seed) if row[1] == 1]
    if args.max_positive and len(rows) > args.max_positive:
        rng = np.random.default_rng(args.seed)
        idx = sorted(rng.choice(len(rows), size=args.max_positive, replace=False).tolist())
        rows = [rows[i] for i in idx]
    x_geo, y_geo, geo_names, ok_rows, geo_fail = build_georgia_matrix(
        rows, args.pre, args.post, args.downsample, args.target_fs
    )
    if feature_names != geo_names:
        raise RuntimeError("Feature schema mismatch between PTB-XL and Georgia.")

    norm_proto = nearest_norm_prototypes(scaler, memory, x_geo)
    target_summary = summarize_combo(model, x_geo, norm_proto, threshold, group_cols, TARGET_COMBO, alpha=1.0)
    controls = [
        summarize_combo(model, x_geo, norm_proto, threshold, group_cols, combo, alpha=1.0)
        for combo in control_combos(group_cols)
    ]
    controls_df = pd.DataFrame(controls).sort_values(["mean_prob_drop", "flip_rate"], ascending=False)
    controls_df.to_csv(args.out_dir / "georgia_negative_control_summary.csv", index=False)

    best_control_name = str(controls_df.iloc[0]["combo"])
    best_control_combo = tuple(best_control_name.split("+"))
    target_drop, target_base, target_cf = per_case_drop(
        model, x_geo, norm_proto, group_cols, TARGET_COMBO, alpha=1.0
    )
    control_drop, _, control_cf = per_case_drop(
        model, x_geo, norm_proto, group_cols, best_control_combo, alpha=1.0
    )
    diff = target_drop - control_drop
    flip_diff = ((target_base >= threshold) & (target_cf < threshold)).astype(float) - (
        (target_base >= threshold) & (control_cf < threshold)
    ).astype(float)

    alpha_rows = []
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        alpha_rows.append(summarize_combo(model, x_geo, norm_proto, threshold, group_cols, TARGET_COMBO, alpha=alpha))
    alpha_df = pd.DataFrame(alpha_rows)
    alpha_df.to_csv(args.out_dir / "georgia_target_alpha_sweep.csv", index=False)

    report = {
        "method": "Georgia feature-level counterfactual controls",
        "data": {
            "georgia_positive_n": int(len(y_geo)),
            "failures": geo_fail[:20],
            "max_positive": args.max_positive or None,
            "target_fs": args.target_fs,
            "ptbxl_train_failures": train_fail[:10],
            "ptbxl_val_failures": val_fail[:10],
        },
        "target": target_summary,
        "best_control": controls_df.iloc[0].to_dict(),
        "target_vs_best_control": {
            "mean_prob_drop_difference": bootstrap_mean_ci(diff, args.n_boot, args.seed),
            "flip_rate_difference": bootstrap_mean_ci(flip_diff, args.n_boot, args.seed),
            "paired_cohen_d_prob_drop": paired_cohen_d(diff),
            "drop_ratio_vs_control": float(target_summary["mean_prob_drop"] / max(float(controls_df.iloc[0]["mean_prob_drop"]), 1e-12)),
            "best_control_combo": best_control_name,
        },
        "alpha_sweep": {
            "rows": alpha_df.to_dict(orient="records"),
            "monotone_mean_drop": bool(np.all(np.diff(alpha_df["mean_prob_drop"].to_numpy()) >= -1e-9)),
            "monotone_flip_rate": bool(np.all(np.diff(alpha_df["flip_rate"].to_numpy()) >= -1e-9)),
        },
        "files": {
            "controls": str(args.out_dir / "georgia_negative_control_summary.csv"),
            "alpha_sweep": str(args.out_dir / "georgia_target_alpha_sweep.csv"),
        },
    }
    (args.out_dir / "georgia_counterfactual_controls_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

