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


def combo_name(combo: tuple[str, ...]) -> str:
    return "+".join(combo)


def cf_drop(
    model: object,
    x: np.ndarray,
    norm_proto: np.ndarray,
    groups: dict[str, list[int]],
    combo: tuple[str, ...],
) -> np.ndarray:
    base = model.predict_proba(x)[:, 1]
    x_cf = apply_counterfactual(x, norm_proto, combo, groups, alpha=1.0)
    cf = model.predict_proba(x_cf)[:, 1]
    return base - cf


def unknown_combos(groups: dict[str, list[int]]) -> list[tuple[str, ...]]:
    names = [g for g in BASE_GROUP_ORDER if g in groups and g not in KNOWN]
    combos = [(g,) for g in names]
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            combos.append((a, b))
    return combos


def summarize_model(
    name: str,
    model: object,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    x_geo_pos: np.ndarray,
    groups: dict[str, list[int]],
    norm_proto_test: np.ndarray,
    norm_proto_geo: np.ndarray,
) -> dict:
    val_prob = model.predict_proba(x_val)[:, 1]
    thr = select_threshold(y_val, val_prob)
    test_prob = model.predict_proba(x_test)[:, 1]
    m = cls_metrics(y_test, test_prob, thr)

    test_pos = x_test[y_test == 1]
    target_test = cf_drop(model, test_pos, norm_proto_test, groups, TARGET)
    target_geo = cf_drop(model, x_geo_pos, norm_proto_geo, groups, TARGET)

    best_unknown = {"combo": "", "ptbxl_drop": -1.0, "georgia_drop": -1.0, "min_drop": -1.0}
    for combo in unknown_combos(groups):
        d_ptb = float(np.mean(cf_drop(model, test_pos, norm_proto_test, groups, combo)))
        d_geo = float(np.mean(cf_drop(model, x_geo_pos, norm_proto_geo, groups, combo)))
        mn = min(d_ptb, d_geo)
        if mn > best_unknown["min_drop"]:
            best_unknown = {"combo": combo_name(combo), "ptbxl_drop": d_ptb, "georgia_drop": d_geo, "min_drop": mn}

    target_ptb = float(np.mean(target_test))
    target_geo_mean = float(np.mean(target_geo))
    collapse = bool(
        target_ptb >= 0.40
        and target_geo_mean >= 0.40
        and best_unknown["min_drop"] < 0.20
        and (best_unknown["ptbxl_drop"] / max(target_ptb, 1e-12)) < 0.35
        and (best_unknown["georgia_drop"] / max(target_geo_mean, 1e-12)) < 0.35
    )
    return {
        "model": name,
        "auc": float(m["auc"]),
        "auprc": float(m["auprc"]),
        "balanced_accuracy": float(m["balanced_accuracy"]),
        "target_drop_ptbxl": target_ptb,
        "target_drop_georgia": target_geo_mean,
        "best_unknown_combo": best_unknown["combo"],
        "best_unknown_drop_ptbxl": best_unknown["ptbxl_drop"],
        "best_unknown_drop_georgia": best_unknown["georgia_drop"],
        "best_unknown_min_drop": best_unknown["min_drop"],
        "unknown_to_target_ratio_ptbxl": float(best_unknown["ptbxl_drop"] / max(target_ptb, 1e-12)),
        "unknown_to_target_ratio_georgia": float(best_unknown["georgia_drop"] / max(target_geo_mean, 1e-12)),
        "collapse_present": collapse,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Test whether mechanism collapse is universal across multiple models.")
    parser.add_argument("--csv-root", type=Path, default=Path("Data") / "Generated_PTBXL_12Lead")
    parser.add_argument("--georgia-root", type=Path, default=Path("Data") / "PhysioNet_Challenge_2020_Georgia")
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts") / "phase2" / "multimodel_collapse_benchmark")
    parser.add_argument("--models", nargs="+", default=["hgbdt", "random_forest", "extra_trees", "mlp"])
    parser.add_argument("--pre", type=int, default=80)
    parser.add_argument("--post", type=int, default=120)
    parser.add_argument("--downsample", type=int, default=32)
    parser.add_argument("--target-fs", type=float, default=100.0)
    parser.add_argument("--prototypes-per-class", type=int, default=12)
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

    x_train, y_train_raw, feature_names, train_ok, _ = build_feature_matrix(train_s, args.pre, args.post, args.downsample)
    x_val, y_val_raw, _, _, _ = build_feature_matrix(val_s, args.pre, args.post, args.downsample)
    x_test, y_test_raw, _, _, _ = build_feature_matrix(test_s, args.pre, args.post, args.downsample)
    y_train = map_y(y_train_raw)
    y_val = map_y(y_val_raw)
    y_test = map_y(y_test_raw)

    groups = feature_groups(feature_names)
    scaler = StandardScaler()
    memory = CounterfactualMemory(prototypes_per_class=args.prototypes_per_class, seed=args.seed)
    memory.fit(scaler.fit_transform(x_train), y_train, [s.path.name for s in train_ok])
    x_test_pos = x_test[y_test == 1]
    test_z = scaler.transform(x_test_pos)
    norm_idx_test, _ = memory.nearest_by_label(test_z, 0)
    norm_proto_test = scaler.inverse_transform(memory.prototype_matrix_[norm_idx_test])  # type: ignore[index]

    geo_rows = [r for r in discover_georgia(args.georgia_root, "no_st_t", None, args.seed) if r[1] == 1]
    geo_rows = sample_cap(geo_rows, args.max_georgia_positive, args.seed + 3)
    x_geo, y_geo, geo_names, _, _ = build_georgia_matrix(geo_rows, args.pre, args.post, args.downsample, args.target_fs)
    if geo_names != feature_names:
        raise RuntimeError("Feature schema mismatch between PTB-XL and Georgia.")
    x_geo_pos = x_geo[y_geo == 1]
    geo_z = scaler.transform(x_geo_pos)
    norm_idx_geo, _ = memory.nearest_by_label(geo_z, 0)
    norm_proto_geo = scaler.inverse_transform(memory.prototype_matrix_[norm_idx_geo])  # type: ignore[index]

    rows = []
    for i, model_name in enumerate(args.models):
        model = make_model(model_name, args.seed + i)
        model.fit(x_train, y_train)
        rows.append(
            summarize_model(
                model_name,
                model,
                x_val,
                y_val,
                x_test,
                y_test,
                x_geo_pos,
                groups,
                norm_proto_test,
                norm_proto_geo,
            )
        )

    table = pd.DataFrame(rows).sort_values(["collapse_present", "target_drop_georgia"], ascending=[False, False])
    table.to_csv(args.out_dir / "multimodel_collapse_table.csv", index=False)
    n_models = int(len(table))
    n_collapse = int(table["collapse_present"].sum())
    universal = bool(n_models >= 4 and n_collapse >= 4)
    report = {
        "task": "Multimodel mechanism collapse benchmark",
        "models_tested": table["model"].tolist(),
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
        "collapse_models": int(n_collapse),
        "total_models": n_models,
        "collapse_universal_supported": universal,
        "verdict": "collapse_pattern_consistent" if universal else "collapse_pattern_inconclusive",
        "files": {"table": str(args.out_dir / "multimodel_collapse_table.csv")},
    }
    (args.out_dir / "multimodel_collapse_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
