from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from phase2.ecg_mechanism_core import (
    apply_counterfactual,
    augment_feature_groups_with_regions,
    bootstrap_mean_ci,
    bootstrap_metric_ci,
    cls_metrics,
    feature_groups,
    group_columns,
    load_ptbxl_norm_sttc_samples,
    make_hgbdt,
    map_y,
    ordered_group_names,
    paired_cohen_d,
    select_threshold,
)
from phase2.prototype_memory import CounterfactualMemory
from phase2.run_ptbxl_mechanism_analysis import load_diagnostic_codes, load_superclasses, load_task_samples
from real_ml_xai_llcs.features import build_feature_matrix


DEFAULT_TARGET_COMBOS = {
    "STTC": ("st_segment", "shape_template"),
    "CD": ("shape_template", "qrs"),
    "MI": ("st_segment", "shape_template"),
    "HYP": ("shape_template", "global_morph"),
}


def combo_name(combo: tuple[str, ...]) -> str:
    return "+".join(combo)


def resolve_target_combo(target: str, raw: str | None) -> tuple[str, ...]:
    if raw:
        return tuple(part.strip() for part in raw.split("+") if part.strip())
    return DEFAULT_TARGET_COMBOS[target]


def min_memory_distance(memory: CounterfactualMemory, scaler: StandardScaler, x: np.ndarray) -> np.ndarray:
    if x.shape[0] == 0:
        return np.asarray([], dtype=float)
    return np.min(memory.distances(scaler.transform(x)), axis=1)


def counterfactual_cases(
    model: object,
    scaler: StandardScaler,
    memory: CounterfactualMemory,
    x: np.ndarray,
    threshold: float,
    group_cols: dict[str, list[int]],
    combo: tuple[str, ...],
    alpha: float,
) -> pd.DataFrame:
    x_z = scaler.transform(x)
    norm_idx, _ = memory.nearest_by_label(x_z, 0)
    norm_proto = scaler.inverse_transform(memory.prototype_matrix_[norm_idx])  # type: ignore[index]
    base_prob = model.predict_proba(x)[:, 1]
    x_cf = apply_counterfactual(x, norm_proto, combo, group_cols, alpha=alpha)
    cf_prob = model.predict_proba(x_cf)[:, 1]
    return pd.DataFrame(
        {
            "base_prob": base_prob,
            "cf_prob": cf_prob,
            "prob_drop": base_prob - cf_prob,
            "flipped": (base_prob >= threshold) & (cf_prob < threshold),
            "orig_memory_distance": min_memory_distance(memory, scaler, x),
            "cf_memory_distance": min_memory_distance(memory, scaler, x_cf),
        }
    )


def summarize_case_effect(case: pd.DataFrame, n_boot: int, seed: int) -> dict:
    return {
        "n": int(len(case)),
        "mean_prob_drop": bootstrap_mean_ci(case["prob_drop"].to_numpy(), n_boot=n_boot, seed=seed),
        "flip_rate": bootstrap_mean_ci(case["flipped"].to_numpy(dtype=float), n_boot=n_boot, seed=seed),
        "median_prob_drop": float(case["prob_drop"].median()),
        "positive_drop_rate": float((case["prob_drop"] > 0).mean()),
        "mean_cf_prob": float(case["cf_prob"].mean()),
        "mean_memory_distance_delta": float((case["cf_memory_distance"] - case["orig_memory_distance"]).mean()),
    }


def alpha_sweep(
    model: object,
    scaler: StandardScaler,
    memory: CounterfactualMemory,
    x_pos: np.ndarray,
    threshold: float,
    group_cols: dict[str, list[int]],
    val95_distance: float,
    alphas: list[float],
    target_combo: tuple[str, ...],
) -> pd.DataFrame:
    rows = []
    probe_combos = [tuple([group]) for group in target_combo] + [target_combo]
    seen: set[tuple[str, ...]] = set()
    for combo in probe_combos:
        if combo in seen:
            continue
        seen.add(combo)
        for alpha in alphas:
            case = counterfactual_cases(model, scaler, memory, x_pos, threshold, group_cols, combo, alpha)
            rows.append(
                {
                    "combo": "+".join(combo),
                    "alpha": float(alpha),
                    "mean_prob_drop": float(case["prob_drop"].mean()),
                    "median_prob_drop": float(case["prob_drop"].median()),
                    "flip_rate": float(case["flipped"].mean()),
                    "positive_drop_rate": float((case["prob_drop"] > 0).mean()),
                    "mean_cf_prob": float(case["cf_prob"].mean()),
                    "cf_outlier_rate_val95": float((case["cf_memory_distance"] > val95_distance).mean()),
                }
            )
    return pd.DataFrame(rows)


def control_combos(group_cols: dict[str, list[int]], group_order: list[str], target_combo: tuple[str, ...]) -> list[tuple[str, ...]]:
    names = [g for g in group_order if g in group_cols and g not in set(target_combo)]
    combos = [(g,) for g in names]
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            combos.append((first, second))
    return combos


def control_summary(
    model: object,
    scaler: StandardScaler,
    memory: CounterfactualMemory,
    x_pos: np.ndarray,
    threshold: float,
    group_cols: dict[str, list[int]],
    group_order: list[str],
    target_combo: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    per_case_rows = []
    for combo in control_combos(group_cols, group_order, target_combo):
        case = counterfactual_cases(model, scaler, memory, x_pos, threshold, group_cols, combo, alpha=1.0)
        name = "+".join(combo)
        rows.append(
            {
                "combo": name,
                "n_groups": len(combo),
                "n_features": int(sum(len(group_cols[g]) for g in combo)),
                "mean_prob_drop": float(case["prob_drop"].mean()),
                "median_prob_drop": float(case["prob_drop"].median()),
                "flip_rate": float(case["flipped"].mean()),
                "positive_drop_rate": float((case["prob_drop"] > 0).mean()),
            }
        )
        per_case = case[["prob_drop", "flipped"]].copy()
        per_case.insert(0, "combo", name)
        per_case_rows.append(per_case)
    return pd.DataFrame(rows).sort_values(["mean_prob_drop", "flip_rate"], ascending=False), pd.concat(per_case_rows)


def group_permutation(
    model: object,
    x_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float,
    group_cols: dict[str, list[int]],
    group_order: list[str],
    seed: int,
    repeats: int,
    target_combo: tuple[str, ...],
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = cls_metrics(y_test, model.predict_proba(x_test)[:, 1], threshold)
    rows = []
    combos = [(g,) for g in group_order if g in group_cols] + [target_combo]
    for combo in combos:
        cols = group_columns(group_cols, combo)
        for rep in range(repeats):
            x_perm = x_test.copy()
            perm = rng.permutation(x_perm.shape[0])
            x_perm[:, cols] = x_perm[perm][:, cols]
            m = cls_metrics(y_test, model.predict_proba(x_perm)[:, 1], threshold)
            rows.append(
                {
                    "combo": "+".join(combo),
                    "repeat": rep,
                    "auc_drop": base["auc"] - m["auc"],
                    "auprc_drop": base["auprc"] - m["auprc"],
                    "balanced_accuracy_drop": base["balanced_accuracy"] - m["balanced_accuracy"],
                    "macro_f1_drop": base["macro_f1"] - m["macro_f1"],
                }
            )
    return (
        pd.DataFrame(rows)
        .groupby("combo", as_index=False)
        .agg(
            auc_drop_mean=("auc_drop", "mean"),
            auc_drop_std=("auc_drop", "std"),
            auprc_drop_mean=("auprc_drop", "mean"),
            balanced_accuracy_drop_mean=("balanced_accuracy_drop", "mean"),
            macro_f1_drop_mean=("macro_f1_drop", "mean"),
        )
        .sort_values("auc_drop_mean", ascending=False)
    )


def ablation_retrain(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    group_cols: dict[str, list[int]],
    group_order: list[str],
    seed: int,
    target_combo: tuple[str, ...],
) -> pd.DataFrame:
    combos = [()] + [(g,) for g in group_order if g in group_cols] + [target_combo]
    rows = []
    for combo in combos:
        drop_cols = group_columns(group_cols, combo) if combo else []
        keep = np.ones(x_train.shape[1], dtype=bool)
        keep[drop_cols] = False
        model = make_hgbdt(seed)
        model.fit(x_train[:, keep], y_train)
        val_prob = model.predict_proba(x_val[:, keep])[:, 1]
        threshold = select_threshold(y_val, val_prob)
        test_prob = model.predict_proba(x_test[:, keep])[:, 1]
        rows.append(
            {
                "dropped": "+".join(combo) if combo else "none",
                "n_dropped_features": int(len(drop_cols)),
                "n_kept_features": int(np.sum(keep)),
                **cls_metrics(y_test, test_prob, threshold),
            }
        )
    out = pd.DataFrame(rows)
    base = out[out["dropped"] == "none"].iloc[0]
    for metric in ["auc", "auprc", "balanced_accuracy", "macro_f1"]:
        out[f"{metric}_drop_vs_full"] = float(base[metric]) - out[metric]
    return out.sort_values("auc_drop_vs_full", ascending=False)


def select_target_combo_on_validation(
    model: object,
    scaler: StandardScaler,
    memory: CounterfactualMemory,
    x_val: np.ndarray,
    y_val: np.ndarray,
    threshold: float,
    group_cols: dict[str, list[int]],
    group_order: list[str],
) -> tuple[tuple[str, ...], pd.DataFrame]:
    x_pos = x_val[y_val == 1]
    rows = []
    candidates = [(g,) for g in group_order if g in group_cols]
    for i, first in enumerate(group_order):
        if first not in group_cols:
            continue
        for second in group_order[i + 1 :]:
            if second in group_cols:
                candidates.append((first, second))
    for combo in candidates:
        case = counterfactual_cases(model, scaler, memory, x_pos, threshold, group_cols, combo, alpha=1.0)
        rows.append(
            {
                "combo": combo_name(combo),
                "n_groups": len(combo),
                "mean_prob_drop": float(case["prob_drop"].mean()),
                "flip_rate": float(case["flipped"].mean()),
                "positive_drop_rate": float((case["prob_drop"] > 0).mean()),
            }
        )
    ranking = pd.DataFrame(rows).sort_values(
        ["mean_prob_drop", "flip_rate", "positive_drop_rate"],
        ascending=False,
    ).reset_index(drop=True)
    best = tuple(str(part) for part in ranking.iloc[0]["combo"].split("+"))
    return best, ranking


def leave_environment_out(
    x_train: np.ndarray,
    y_train: np.ndarray,
    envs: list[str],
    x_val: np.ndarray,
    y_val: np.ndarray,
    group_cols: dict[str, list[int]],
    train_sources: list[str],
    seed: int,
    prototypes_per_class: int,
    target_combo: tuple[str, ...],
) -> pd.DataFrame:
    rows = []
    env_arr = np.asarray(envs)
    for holdout in sorted(set(envs)):
        train_mask = env_arr != holdout
        test_mask = env_arr == holdout
        model = make_hgbdt(seed)
        model.fit(x_train[train_mask], y_train[train_mask])
        threshold = select_threshold(y_val, model.predict_proba(x_val)[:, 1])
        prob = model.predict_proba(x_train[test_mask])[:, 1]
        metrics = cls_metrics(y_train[test_mask], prob, threshold)

        scaler = StandardScaler()
        memory = CounterfactualMemory(prototypes_per_class=prototypes_per_class, seed=seed)
        memory.fit(scaler.fit_transform(x_train[train_mask]), y_train[train_mask], [s for i, s in enumerate(train_sources) if train_mask[i]])
        x_pos = x_train[test_mask][y_train[test_mask] == 1]
        case = counterfactual_cases(model, scaler, memory, x_pos, threshold, group_cols, target_combo, alpha=1.0)
        rows.append(
            {
                "heldout_env": holdout,
                "train_n": int(train_mask.sum()),
                "test_n": int(test_mask.sum()),
                "positive_n": int(len(x_pos)),
                **metrics,
                "target_mean_prob_drop": float(case["prob_drop"].mean()),
                "target_flip_rate": float(case["flipped"].mean()),
            }
        )
    return pd.DataFrame(rows)


def export_case_studies(
    model: object,
    scaler: StandardScaler,
    memory: CounterfactualMemory,
    x_test: np.ndarray,
    y_test: np.ndarray,
    test_files: list[str],
    threshold: float,
    group_cols: dict[str, list[int]],
    feature_names: list[str],
    out_dir: Path,
    n_cases: int,
    target_name: str,
    target_combo: tuple[str, ...],
) -> pd.DataFrame:
    pos_idx = np.flatnonzero(y_test == 1)
    x_pos = x_test[pos_idx]
    x_z = scaler.transform(x_pos)
    norm_idx, norm_dist = memory.nearest_by_label(x_z, 0)
    pos_label_idx, pos_label_dist = memory.nearest_by_label(x_z, 1)
    norm_proto = scaler.inverse_transform(memory.prototype_matrix_[norm_idx])  # type: ignore[index]
    base_prob = model.predict_proba(x_pos)[:, 1]
    x_cf = apply_counterfactual(x_pos, norm_proto, target_combo, group_cols, alpha=1.0)
    cf_prob = model.predict_proba(x_cf)[:, 1]
    prob_drop = base_prob - cf_prob
    order = np.argsort(prob_drop)[::-1][:n_cases]
    cols = group_columns(group_cols, target_combo)
    rows = []
    details = []
    for rank, local_i in enumerate(order, start=1):
        delta = norm_proto[local_i, cols] - x_pos[local_i, cols]
        top = np.argsort(np.abs(delta))[::-1][:12]
        deltas = [
            {
                "feature": feature_names[cols[int(i)]],
                "original": float(x_pos[local_i, cols[int(i)]]),
                "norm_prototype": float(norm_proto[local_i, cols[int(i)]]),
                "delta_to_norm": float(delta[int(i)]),
            }
            for i in top
        ]
        row = {
            "rank": rank,
            "test_file": test_files[int(pos_idx[local_i])],
            f"base_{target_name.lower()}_prob": float(base_prob[local_i]),
            f"counterfactual_{target_name.lower()}_prob": float(cf_prob[local_i]),
            "prob_drop": float(prob_drop[local_i]),
            "flipped": bool(base_prob[local_i] >= threshold and cf_prob[local_i] < threshold),
            "nearest_norm_source": memory.cases_[int(norm_idx[local_i])].source,
            "nearest_norm_distance": float(norm_dist[local_i]),
            "nearest_norm_support": int(memory.cases_[int(norm_idx[local_i])].support),
            "nearest_positive_source": memory.cases_[int(pos_label_idx[local_i])].source,
            "nearest_positive_distance": float(pos_label_dist[local_i]),
            "nearest_positive_support": int(memory.cases_[int(pos_label_idx[local_i])].support),
            "top_delta_features": "; ".join(f"{d['feature']}={d['delta_to_norm']:.3g}" for d in deltas[:5]),
        }
        rows.append(row)
        details.append({**row, "feature_deltas": deltas})
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "case_studies_summary.csv", index=False)
    (out_dir / "case_studies_detail.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Final clean PTB-XL pipeline for Clinical Counterfactual Memory.")
    parser.add_argument("--csv-root", type=Path, default=Path("Data") / "Generated_PTBXL_12Lead")
    parser.add_argument(
        "--ptbxl-root",
        type=Path,
        default=Path("Data") / "PTBXL" / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts") / "phase2" / "ptbxl_final")
    parser.add_argument("--target", type=str, default="STTC", choices=["STTC", "CD", "HYP", "MI"])
    parser.add_argument("--pre", type=int, default=80)
    parser.add_argument("--post", type=int, default=120)
    parser.add_argument("--downsample", type=int, default=32)
    parser.add_argument("--prototypes-per-class", type=int, default=12)
    parser.add_argument("--permutation-repeats", type=int, default=8)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--n-cases", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--target-combo",
        type=str,
        default=None,
        help="Optional clinical group combo override, e.g. st_segment+shape_template.",
    )
    parser.add_argument(
        "--mi-cohort",
        type=str,
        default="broad",
        choices=["broad", "single_code", "imi_only", "asmi_only", "inferior_single", "anterior_single"],
        help="Optional MI cohort narrowing; only used when --target MI.",
    )
    parser.add_argument(
        "--augment-mi-regions",
        action="store_true",
        help="Add lead-territory feature groups for MI analysis.",
    )
    parser.add_argument(
        "--auto-target-combo",
        action="store_true",
        help="Select the target combo on validation positives instead of using the default or --target-combo.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.target == "STTC":
        train_s, val_s, test_s, ood_s = load_ptbxl_norm_sttc_samples(args.csv_root)
    else:
        superclasses = load_superclasses(args.ptbxl_root)
        diagnostic_codes = load_diagnostic_codes(args.ptbxl_root) if args.target == "MI" else None
        train_s, val_s, test_s = load_task_samples(
            args.csv_root,
            superclasses,
            args.target,
            diagnostic_codes=diagnostic_codes,
            mi_cohort=args.mi_cohort,
        )
        ood_s = []

    x_train, y_train_raw, feature_names, train_ok, train_fail = build_feature_matrix(train_s, args.pre, args.post, args.downsample)
    x_val, y_val_raw, _, _, val_fail = build_feature_matrix(val_s, args.pre, args.post, args.downsample)
    x_test, y_test_raw, _, test_ok, test_fail = build_feature_matrix(test_s, args.pre, args.post, args.downsample)
    if ood_s:
        x_ood, _, _, ood_ok, ood_fail = build_feature_matrix(ood_s, args.pre, args.post, args.downsample)
    else:
        x_ood = np.zeros((0, x_test.shape[1]), dtype=float)
        ood_ok = []
        ood_fail = []

    if args.target == "STTC":
        y_train = map_y(y_train_raw)
        y_val = map_y(y_val_raw)
        y_test = map_y(y_test_raw)
    else:
        y_train = np.asarray(y_train_raw, dtype=int)
        y_val = np.asarray(y_val_raw, dtype=int)
        y_test = np.asarray(y_test_raw, dtype=int)

    group_cols = feature_groups(feature_names)
    if args.target == "MI" and args.augment_mi_regions:
        group_cols = augment_feature_groups_with_regions(feature_names, group_cols)
    group_order = ordered_group_names(group_cols)
    model = make_hgbdt(args.seed)
    model.fit(x_train, y_train)
    threshold = select_threshold(y_val, model.predict_proba(x_val)[:, 1])
    test_prob = model.predict_proba(x_test)[:, 1]
    test_metrics = cls_metrics(y_test, test_prob, threshold)
    test_metrics_ci = {
        metric: bootstrap_metric_ci(y_test, test_prob, threshold, metric, args.n_boot, args.seed)
        for metric in ["auc", "auprc", "balanced_accuracy", "macro_f1"]
    }

    scaler = StandardScaler()
    memory = CounterfactualMemory(prototypes_per_class=args.prototypes_per_class, seed=args.seed)
    memory.fit(scaler.fit_transform(x_train), y_train, [s.path.name for s in train_ok])
    val95_distance = float(np.quantile(min_memory_distance(memory, scaler, x_val), 0.95))

    target_combo = resolve_target_combo(args.target, args.target_combo)
    val_target_ranking = pd.DataFrame()
    if args.auto_target_combo:
        target_combo, val_target_ranking = select_target_combo_on_validation(
            model,
            scaler,
            memory,
            x_val,
            y_val,
            threshold,
            group_cols,
            group_order,
        )
    target_combo_name = combo_name(target_combo)

    pos_idx = np.flatnonzero(y_test == 1)
    x_pos = x_test[pos_idx]
    target_case = counterfactual_cases(model, scaler, memory, x_pos, threshold, group_cols, target_combo, alpha=1.0)
    target_case.to_csv(args.out_dir / "target_counterfactual_per_case.csv", index=False)
    target_summary = summarize_case_effect(target_case, args.n_boot, args.seed)

    controls, control_cases = control_summary(model, scaler, memory, x_pos, threshold, group_cols, group_order, target_combo)
    controls.to_csv(args.out_dir / "negative_control_summary.csv", index=False)
    best_control_name = str(controls.iloc[0]["combo"])
    best_control_case = control_cases[control_cases["combo"] == best_control_name].reset_index(drop=True)
    diff_drop = target_case["prob_drop"].to_numpy() - best_control_case["prob_drop"].to_numpy()
    diff_flip = target_case["flipped"].to_numpy(dtype=float) - best_control_case["flipped"].to_numpy(dtype=float)

    alpha_df = alpha_sweep(
        model, scaler, memory, x_pos, threshold, group_cols, val95_distance, [0.0, 0.25, 0.5, 0.75, 1.0], target_combo
    )
    alpha_df.to_csv(args.out_dir / "alpha_sweep.csv", index=False)
    permutation_df = group_permutation(
        model,
        x_test,
        y_test,
        threshold,
        group_cols,
        group_order,
        args.seed,
        args.permutation_repeats,
        target_combo,
    )
    permutation_df.to_csv(args.out_dir / "group_permutation_summary.csv", index=False)
    ablation_df = ablation_retrain(
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
        group_cols,
        group_order,
        args.seed,
        target_combo,
    )
    ablation_df.to_csv(args.out_dir / "ablation_retrain.csv", index=False)
    env_df = leave_environment_out(
        x_train,
        y_train,
        [s.env for s in train_ok],
        x_val,
        y_val,
        group_cols,
        [s.path.name for s in train_ok],
        args.seed,
        args.prototypes_per_class,
        target_combo,
    )
    env_df.to_csv(args.out_dir / "leave_environment_out.csv", index=False)

    id_score = min_memory_distance(memory, scaler, x_test)
    ood_score = min_memory_distance(memory, scaler, x_ood) if len(ood_ok) else np.asarray([], dtype=float)
    case_df = export_case_studies(
        model,
        scaler,
        memory,
        x_test,
        y_test,
        [s.path.name for s in test_ok],
        threshold,
        group_cols,
        feature_names,
        args.out_dir,
        args.n_cases,
        args.target,
        target_combo,
    )

    target_alpha = alpha_df[alpha_df["combo"] == target_combo_name].sort_values("alpha")
    report = {
        "method": "Clinical Counterfactual Memory",
        "task": f"PTB-XL 12-lead NORM vs {args.target}",
        "scope_note": "Final PTB-XL internal benchmark; do not generalize without external validation.",
        "data": {
            "csv_root": str(args.csv_root),
            "ptbxl_root": str(args.ptbxl_root),
            "target": args.target,
            "mi_cohort": args.mi_cohort if args.target == "MI" else None,
            "train_n": int(len(y_train)),
            "val_n": int(len(y_val)),
            "test_n": int(len(y_test)),
            "positive_test_n": int(len(pos_idx)),
            "ood_n": int(len(ood_ok)),
            "n_features": int(len(feature_names)),
            "feature_group_sizes": {k: len(v) for k, v in group_cols.items()},
            "failures": {"train": train_fail[:10], "val": val_fail[:10], "test": test_fail[:10], "ood": ood_fail[:10]},
        },
        "model": {
            "name": "HistGradientBoostingClassifier",
            "threshold_selected_on_val": float(threshold),
            "augment_mi_regions": bool(args.target == "MI" and args.augment_mi_regions),
            "auto_target_combo": bool(args.auto_target_combo),
        },
        "classification": {"test_metrics": test_metrics, "bootstrap_ci": test_metrics_ci},
        "counterfactual": {
            "target_combo": target_combo_name,
            "validation_target_combo_ranking": val_target_ranking.head(10).to_dict(orient="records") if not val_target_ranking.empty else [],
            "target_summary": target_summary,
            "best_non_target_control": controls.iloc[0].to_dict(),
            "best_non_st_shape_control": controls.iloc[0].to_dict(),
            "target_minus_best_control": {
                "best_control_combo": best_control_name,
                "mean_prob_drop_difference": bootstrap_mean_ci(diff_drop, args.n_boot, args.seed),
                "flip_rate_difference": bootstrap_mean_ci(diff_flip, args.n_boot, args.seed),
                "paired_cohen_d_prob_drop": paired_cohen_d(diff_drop),
            },
            "target_alpha_monotone_mean_drop": bool(np.all(np.diff(target_alpha["mean_prob_drop"].to_numpy()) >= -1e-9)),
            "target_alpha_monotone_flip_rate": bool(np.all(np.diff(target_alpha["flip_rate"].to_numpy()) >= -1e-9)),
            "st_shape_alpha_monotone_mean_drop": bool(np.all(np.diff(target_alpha["mean_prob_drop"].to_numpy()) >= -1e-9)),
            "st_shape_alpha_monotone_flip_rate": bool(np.all(np.diff(target_alpha["flip_rate"].to_numpy()) >= -1e-9)),
        },
        "xai_controls": {
            "top_permutation_combo": permutation_df.iloc[0].to_dict(),
            "drop_target_retrain": ablation_df[ablation_df["dropped"] == target_combo_name].iloc[0].to_dict(),
            "drop_st_shape_retrain": ablation_df[ablation_df["dropped"] == target_combo_name].iloc[0].to_dict(),
        },
        "robustness": {
            "leave_env_out_mean_auc": float(env_df["auc"].mean()),
            "leave_env_out_min_auc": float(env_df["auc"].min()),
            "leave_env_out_mean_target_prob_drop": float(env_df["target_mean_prob_drop"].mean()),
            "memory_val95_threshold": val95_distance,
            "memory_id_flag_rate": float(np.mean(id_score > val95_distance)),
            "memory_ptbxl_ood_flag_rate": float(np.mean(ood_score > val95_distance)) if ood_score.size else float("nan"),
        },
        "case_studies": {
            "n_cases": int(len(case_df)),
            "mean_selected_prob_drop": float(case_df["prob_drop"].mean()) if len(case_df) else float("nan"),
        },
        "files": {
            "target_per_case": str(args.out_dir / "target_counterfactual_per_case.csv"),
            "negative_controls": str(args.out_dir / "negative_control_summary.csv"),
            "alpha_sweep": str(args.out_dir / "alpha_sweep.csv"),
            "permutation": str(args.out_dir / "group_permutation_summary.csv"),
            "ablation": str(args.out_dir / "ablation_retrain.csv"),
            "leave_env_out": str(args.out_dir / "leave_environment_out.csv"),
            "case_studies": str(args.out_dir / "case_studies_summary.csv"),
        },
    }
    (args.out_dir / "ptbxl_final_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()


