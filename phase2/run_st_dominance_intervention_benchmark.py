from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from phase2.ecg_mechanism_core import (
    BASE_GROUP_ORDER,
    apply_counterfactual,
    cls_metrics,
    feature_groups,
    load_ptbxl_norm_sttc_samples,
    make_hgbdt,
    map_y,
    select_threshold,
)
from phase2.prototype_memory import CounterfactualMemory
from phase2.run_georgia_external_validation import build_georgia_matrix, discover_georgia
from phase2.ecg_features import build_feature_matrix


TARGET = ("st_segment", "shape_template")
KNOWN = {"st_segment", "shape_template", "t_wave"}


def sample_cap(items: list, cap: int, seed: int) -> list:
    if cap <= 0 or len(items) <= cap:
        return items
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(items), size=cap, replace=False))
    return [items[int(i)] for i in idx]


def make_model(name: str, seed: int):
    if name == "hgbdt":
        return make_hgbdt(seed)
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=500,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
    if name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=600,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
    if name == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(256, 64),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=200,
            random_state=seed,
        )
    raise ValueError(f"Unknown model: {name}")


def unknown_combos(groups: dict[str, list[int]]) -> list[tuple[str, ...]]:
    names = [g for g in BASE_GROUP_ORDER if g in groups and g not in KNOWN]
    combos = [(g,) for g in names]
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            combos.append((a, b))
    return combos


def cf_drop(model: object, x: np.ndarray, proto: np.ndarray, groups: dict[str, list[int]], combo: tuple[str, ...]) -> np.ndarray:
    base = model.predict_proba(x)[:, 1]
    x_cf = apply_counterfactual(x, proto, combo, groups, alpha=1.0)
    cf = model.predict_proba(x_cf)[:, 1]
    return base - cf


def weaken_training_features(
    x: np.ndarray,
    cols: list[int],
    method: str,
    strength: float,
    seed: int,
) -> np.ndarray:
    out = x.copy()
    if method == "none" or strength <= 0:
        return out
    rng = np.random.default_rng(seed)
    col_idx = np.asarray(cols, dtype=int)
    if method == "mask":
        row_mask = rng.random(out.shape[0]) < strength
        out[np.ix_(row_mask, col_idx)] = 0.0
        return out
    if method == "dropout":
        keep = (rng.random((out.shape[0], len(col_idx))) >= strength).astype(float)
        out[:, col_idx] = out[:, col_idx] * keep
        return out
    if method == "noise":
        sd = np.std(out[:, col_idx], axis=0)
        noise = rng.normal(0.0, strength, size=(out.shape[0], len(col_idx))) * sd
        out[:, col_idx] = out[:, col_idx] + noise
        return out
    raise ValueError(f"Unknown weakening method: {method}")


def evaluate_variant(
    model_name: str,
    variant: str,
    method: str,
    strength: float,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    x_geo_pos: np.ndarray,
    groups: dict[str, list[int]],
    dominance_cols: list[int],
    seed: int,
) -> dict:
    x_train_variant = weaken_training_features(x_train, dominance_cols, method=method, strength=strength, seed=seed)
    model = make_model(model_name, seed)
    model.fit(x_train_variant, y_train)

    val_prob = model.predict_proba(x_val)[:, 1]
    threshold = select_threshold(y_val, val_prob)
    test_prob = model.predict_proba(x_test)[:, 1]
    metrics = cls_metrics(y_test, test_prob, threshold)

    scaler = StandardScaler()
    memory = CounterfactualMemory(prototypes_per_class=12, seed=seed)
    memory.fit(scaler.fit_transform(x_train_variant), y_train, [f"train_{i}" for i in range(len(y_train))])

    x_test_pos = x_test[y_test == 1]
    test_z = scaler.transform(x_test_pos)
    idx_t, _ = memory.nearest_by_label(test_z, 0)
    proto_test = scaler.inverse_transform(memory.prototype_matrix_[idx_t])  # type: ignore[index]

    geo_z = scaler.transform(x_geo_pos)
    idx_g, _ = memory.nearest_by_label(geo_z, 0)
    proto_geo = scaler.inverse_transform(memory.prototype_matrix_[idx_g])  # type: ignore[index]

    target_ptb = float(np.mean(cf_drop(model, x_test_pos, proto_test, groups, TARGET)))
    target_geo = float(np.mean(cf_drop(model, x_geo_pos, proto_geo, groups, TARGET)))

    best = {"combo": "", "ptbxl": -1.0, "georgia": -1.0, "min_drop": -1.0}
    for combo in unknown_combos(groups):
        d_ptb = float(np.mean(cf_drop(model, x_test_pos, proto_test, groups, combo)))
        d_geo = float(np.mean(cf_drop(model, x_geo_pos, proto_geo, groups, combo)))
        mn = min(d_ptb, d_geo)
        if mn > best["min_drop"]:
            best = {"combo": "+".join(combo), "ptbxl": d_ptb, "georgia": d_geo, "min_drop": mn}

    ratio_ptb = float(best["ptbxl"] / max(target_ptb, 1e-12))
    ratio_geo = float(best["georgia"] / max(target_geo, 1e-12))
    collapse = bool(target_ptb >= 0.40 and target_geo >= 0.40 and best["min_drop"] < 0.20 and ratio_ptb < 0.35 and ratio_geo < 0.35)
    return {
        "model": model_name,
        "variant": variant,
        "method": method,
        "strength": float(strength),
        "auc": float(metrics["auc"]),
        "balanced_accuracy": float(metrics["balanced_accuracy"]),
        "target_drop_ptbxl": target_ptb,
        "target_drop_georgia": target_geo,
        "best_unknown_combo": best["combo"],
        "best_unknown_drop_ptbxl": best["ptbxl"],
        "best_unknown_drop_georgia": best["georgia"],
        "best_unknown_min_drop": best["min_drop"],
        "unknown_ratio_ptbxl": ratio_ptb,
        "unknown_ratio_georgia": ratio_geo,
        "collapse_present": collapse,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Test causal hypothesis: ST-dominance drives mechanism collapse.")
    parser.add_argument("--csv-root", type=Path, default=Path("Data") / "Generated_PTBXL_12Lead")
    parser.add_argument("--georgia-root", type=Path, default=Path("Data") / "PhysioNet_Challenge_2020_Georgia")
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts") / "phase2" / "st_dominance_intervention_benchmark")
    parser.add_argument("--models", nargs="+", default=["hgbdt", "random_forest", "extra_trees", "mlp"])
    parser.add_argument("--pre", type=int, default=80)
    parser.add_argument("--post", type=int, default=120)
    parser.add_argument("--downsample", type=int, default=32)
    parser.add_argument("--target-fs", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--max-val", type=int, default=0)
    parser.add_argument("--max-test", type=int, default=0)
    parser.add_argument("--max-georgia-positive", type=int, default=0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_s, val_s, test_s, _ = load_ptbxl_norm_sttc_samples(args.csv_root)
    train_s = sample_cap(train_s, args.max_train, args.seed)
    val_s = sample_cap(val_s, args.max_val, args.seed + 1)
    test_s = sample_cap(test_s, args.max_test, args.seed + 2)

    x_train, y_train_raw, feature_names, _, _ = build_feature_matrix(train_s, args.pre, args.post, args.downsample)
    x_val, y_val_raw, _, _, _ = build_feature_matrix(val_s, args.pre, args.post, args.downsample)
    x_test, y_test_raw, _, _, _ = build_feature_matrix(test_s, args.pre, args.post, args.downsample)
    y_train = map_y(y_train_raw)
    y_val = map_y(y_val_raw)
    y_test = map_y(y_test_raw)

    groups = feature_groups(feature_names)
    dominance_cols = sorted(set(groups.get("st_segment", []) + groups.get("shape_template", []) + groups.get("t_wave", [])))

    geo_rows = [r for r in discover_georgia(args.georgia_root, "no_st_t", None, args.seed) if r[1] == 1]
    geo_rows = sample_cap(geo_rows, args.max_georgia_positive, args.seed + 3)
    x_geo, y_geo, geo_names, _, _ = build_georgia_matrix(geo_rows, args.pre, args.post, args.downsample, args.target_fs)
    if geo_names != feature_names:
        raise RuntimeError("Feature schema mismatch between PTB-XL and Georgia.")
    x_geo_pos = x_geo[y_geo == 1]

    variants = [
        ("baseline", "none", 0.0),
        ("mask_40", "mask", 0.40),
        ("dropout_40", "dropout", 0.40),
        ("noise_50", "noise", 0.50),
    ]

    rows = []
    for mi, model_name in enumerate(args.models):
        for vi, (variant, method, strength) in enumerate(variants):
            rows.append(
                evaluate_variant(
                    model_name,
                    variant,
                    method,
                    strength,
                    x_train,
                    y_train,
                    x_val,
                    y_val,
                    x_test,
                    y_test,
                    x_geo_pos,
                    groups,
                    dominance_cols,
                    seed=args.seed + 100 * mi + vi,
                )
            )
    table = pd.DataFrame(rows)
    table.to_csv(args.out_dir / "st_dominance_intervention_table.csv", index=False)

    summary_rows = []
    for model_name in args.models:
        sub = table[table["model"] == model_name].copy()
        base = sub[sub["variant"] == "baseline"].iloc[0]
        best = sub[sub["variant"] != "baseline"].sort_values("unknown_ratio_georgia", ascending=False).iloc[0]
        summary_rows.append(
            {
                "model": model_name,
                "baseline_collapse": bool(base["collapse_present"]),
                "best_variant": str(best["variant"]),
                "baseline_target_drop_georgia": float(base["target_drop_georgia"]),
                "best_target_drop_georgia": float(best["target_drop_georgia"]),
                "baseline_unknown_ratio_georgia": float(base["unknown_ratio_georgia"]),
                "best_unknown_ratio_georgia": float(best["unknown_ratio_georgia"]),
                "unknown_ratio_gain": float(best["unknown_ratio_georgia"] - base["unknown_ratio_georgia"]),
                "target_drop_change": float(best["target_drop_georgia"] - base["target_drop_georgia"]),
                "collapse_flipped_to_false": bool(base["collapse_present"] and not bool(best["collapse_present"])),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.out_dir / "st_dominance_intervention_summary.csv", index=False)

    base_collapse = int(summary["baseline_collapse"].sum())
    flipped = int(summary["collapse_flipped_to_false"].sum())
    ratio_gain_models = int((summary["unknown_ratio_gain"] > 0.05).sum())
    causal_supported = bool(base_collapse >= 2 and flipped >= 2 and ratio_gain_models >= 2)
    report = {
        "task": "ST-dominance intervention benchmark",
        "models_tested": args.models,
        "sample_caps": {
            "max_train": int(args.max_train),
            "max_val": int(args.max_val),
            "max_test": int(args.max_test),
            "max_georgia_positive": int(args.max_georgia_positive),
        },
        "counts": {
            "train": int(len(y_train)),
            "val": int(len(y_val)),
            "test": int(len(y_test)),
            "ptbxl_positive_test": int(np.sum(y_test == 1)),
            "georgia_positive_used": int(len(x_geo_pos)),
        },
        "results": {
            "baseline_collapse_models": base_collapse,
            "collapse_flipped_models": flipped,
            "models_with_unknown_ratio_gain_gt_0_05": ratio_gain_models,
        },
        "causal_link_supported": causal_supported,
        "verdict": "st_dominance_link_supported" if causal_supported else "st_dominance_link_inconclusive",
        "files": {
            "table": str(args.out_dir / "st_dominance_intervention_table.csv"),
            "summary": str(args.out_dir / "st_dominance_intervention_summary.csv"),
        },
    }
    (args.out_dir / "st_dominance_intervention_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
