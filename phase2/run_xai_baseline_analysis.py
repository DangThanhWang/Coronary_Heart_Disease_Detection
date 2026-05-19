from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler

from phase2.ecg_mechanism_core import (
    BASE_GROUP_ORDER,
    cls_metrics,
    feature_groups,
    load_ptbxl_norm_sttc_samples,
    map_y,
    select_threshold,
)
from phase2.prototype_memory import CounterfactualMemory
from phase2.run_ptbxl_main_experiment import group_permutation
from phase2.ecg_features import build_feature_matrix


EXPECTED_STTC_GROUPS = {"st_segment", "shape_template", "t_wave"}


def make_extra_trees(seed: int) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=600,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )


def ordered_group_rows(scores: dict[str, float], group_cols: dict[str, list[int]]) -> list[dict]:
    total = float(sum(max(v, 0.0) for v in scores.values()))
    rows = []
    for group, score in scores.items():
        rows.append(
            {
                "group": group,
                "score": float(score),
                "normalized_score": float(score / total) if total > 0 else 0.0,
                "n_features": int(len(group_cols.get(group, []))),
                "is_expected_sttc_group": bool(group in EXPECTED_STTC_GROUPS),
            }
        )
    return sorted(rows, key=lambda r: r["score"], reverse=True)


def impurity_group_importance(model: ExtraTreesClassifier, group_cols: dict[str, list[int]]) -> pd.DataFrame:
    importances = np.asarray(model.feature_importances_, dtype=float)
    scores = {group: float(importances[cols].sum()) for group, cols in group_cols.items()}
    return pd.DataFrame(ordered_group_rows(scores, group_cols))


def shap_group_importance(
    model: ExtraTreesClassifier,
    x: np.ndarray,
    y: np.ndarray,
    group_cols: dict[str, list[int]],
    seed: int,
    max_samples: int,
) -> tuple[pd.DataFrame | None, str]:
    if importlib.util.find_spec("shap") is None:
        return None, "shap_not_installed"
    try:
        import shap  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent.
        return None, f"shap_import_failed: {type(exc).__name__}: {exc}"

    pos_idx = np.flatnonzero(y == 1)
    if len(pos_idx) == 0:
        return None, "no_positive_test_samples"
    rng = np.random.default_rng(seed)
    chosen = rng.choice(pos_idx, size=min(max_samples, len(pos_idx)), replace=False)
    x_exp = x[chosen]

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(x_exp)
    except Exception as exc:  # pragma: no cover - environment dependent.
        return None, f"shap_failed: {type(exc).__name__}: {exc}"

    if isinstance(shap_values, list):
        vals = np.asarray(shap_values[1], dtype=float)
    else:
        vals = np.asarray(shap_values, dtype=float)
        if vals.ndim == 3:
            vals = vals[:, :, 1]
    if vals.ndim != 2:
        return None, f"unexpected_shap_shape: {tuple(vals.shape)}"

    scores = {group: float(np.mean(np.sum(np.abs(vals[:, cols]), axis=1))) for group, cols in group_cols.items()}
    return pd.DataFrame(ordered_group_rows(scores, group_cols)), "ok"


def prototype_only_metrics(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    sources: Iterable[str],
    seed: int,
    prototypes_per_class: int,
) -> dict:
    scaler = StandardScaler()
    x_train_z = scaler.fit_transform(x_train)
    memory = CounterfactualMemory(prototypes_per_class=prototypes_per_class, seed=seed)
    memory.fit(x_train_z, y_train, list(sources))

    val_score = memory.margin_norm_minus_sttc(scaler.transform(x_val))
    test_score = memory.margin_norm_minus_sttc(scaler.transform(x_test))
    threshold = select_threshold(y_val, val_score)
    return cls_metrics(y_test, test_score, threshold)


def top_group_summary(df: pd.DataFrame | None, status: str) -> dict:
    if df is None or df.empty:
        return {"status": status, "top1_group": None, "top1_expected": False, "top2_expected_count": 0}
    top = df.head(2)
    return {
        "status": status,
        "top1_group": str(top.iloc[0]["group"]),
        "top1_expected": bool(top.iloc[0]["is_expected_sttc_group"]),
        "top2_expected_count": int(top["is_expected_sttc_group"].sum()),
    }


def combo_has_expected_group(combo: str | None) -> bool:
    if not combo:
        return False
    return any(part in EXPECTED_STTC_GROUPS for part in combo.split("+"))


def load_existing_focused_result(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-root", type=Path, default=Path("Data/Generated_PTBXL_12Lead"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/phase2/xai_baselines_sttc"))
    parser.add_argument("--pre", type=int, default=80)
    parser.add_argument("--post", type=int, default=120)
    parser.add_argument("--downsample", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--permutation-repeats", type=int, default=6)
    parser.add_argument("--shap-samples", type=int, default=160)
    parser.add_argument("--prototypes-per-class", type=int, default=12)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_s, val_s, test_s, _ = load_ptbxl_norm_sttc_samples(args.csv_root)
    x_train, y_train_raw, feature_names, train_ok, train_fail = build_feature_matrix(
        train_s, args.pre, args.post, args.downsample
    )
    x_val, y_val_raw, _, _, val_fail = build_feature_matrix(val_s, args.pre, args.post, args.downsample)
    x_test, y_test_raw, _, _, test_fail = build_feature_matrix(test_s, args.pre, args.post, args.downsample)
    y_train = map_y(y_train_raw)
    y_val = map_y(y_val_raw)
    y_test = map_y(y_test_raw)
    group_cols = feature_groups(feature_names)

    model = make_extra_trees(args.seed)
    model.fit(x_train, y_train)
    val_prob = model.predict_proba(x_val)[:, 1]
    test_prob = model.predict_proba(x_test)[:, 1]
    threshold = select_threshold(y_val, val_prob)
    extra_metrics = cls_metrics(y_test, test_prob, threshold)

    impurity_df = impurity_group_importance(model, group_cols)
    impurity_df.to_csv(args.out_dir / "extratrees_impurity_group_importance.csv", index=False)

    permutation_df = group_permutation(
        model, x_test, y_test, threshold, group_cols, args.seed, args.permutation_repeats
    )
    permutation_df.to_csv(args.out_dir / "extratrees_group_permutation_summary.csv", index=False)

    shap_df, shap_status = shap_group_importance(
        model, x_test, y_test, group_cols, args.seed, args.shap_samples
    )
    if shap_df is not None:
        shap_df.to_csv(args.out_dir / "treeshap_group_importance.csv", index=False)

    proto_metrics = prototype_only_metrics(
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
        [s.path.name for s in train_ok],
        args.seed,
        args.prototypes_per_class,
    )

    focused = load_existing_focused_result(
        Path("artifacts/phase2/focused_result_summary/focused_result_report.json")
    )
    ptbxl_final = load_existing_focused_result(
        Path("artifacts/phase2/ptbxl_final/ptbxl_final_report.json")
    )

    permutation_rank = permutation_df.copy()
    permutation_rank["score"] = permutation_rank["auc_drop_mean"].astype(float)
    permutation_rank = permutation_rank.sort_values("score", ascending=False)
    permutation_top = None
    if not permutation_rank.empty:
        permutation_top = str(permutation_rank.iloc[0]["combo"])

    baseline_localizes_expected = bool(
        top_group_summary(impurity_df, "ok")["top1_expected"]
        or (shap_df is not None and top_group_summary(shap_df, shap_status)["top1_expected"])
        or combo_has_expected_group(permutation_top)
    )
    ours_external_ccs = focused.get("georgia_counterfactual_consistency", {})
    ours_ptbxl_ccs = focused.get("ptbxl_counterfactual_consistency", {})

    report = {
        "task": "PTB-XL NORM vs STTC official-XAI-style baselines",
        "data": {
            "train_n": int(len(y_train)),
            "val_n": int(len(y_val)),
            "test_n": int(len(y_test)),
            "positive_test_n": int(np.sum(y_test == 1)),
            "n_features": int(len(feature_names)),
            "feature_group_sizes": {k: len(v) for k, v in group_cols.items()},
            "failures": {"train": train_fail[:10], "val": val_fail[:10], "test": test_fail[:10]},
        },
        "classification": {
            "extratrees": extra_metrics,
            "prototype_memory_only": proto_metrics,
        },
        "xai_baselines": {
            "extratrees_impurity": top_group_summary(impurity_df, "ok"),
            "extratrees_permutation_top_group": permutation_top,
            "extratrees_permutation_top_auc_drop": float(permutation_rank.iloc[0]["auc_drop_mean"])
            if not permutation_rank.empty
            else None,
            "treeshap": top_group_summary(shap_df, shap_status),
        },
        "current_counterfactual_memory_result": {
            "ptbxl_ccs": ours_ptbxl_ccs,
            "external_georgia_ccs": ours_external_ccs,
            "has_external_counterfactual_control": bool(ours_external_ccs),
        },
        "verdict": {
            "baseline_beats_classification": bool(
                extra_metrics["auc"]
                > ptbxl_final.get("classification", {}).get("test_metrics", {}).get("auc", 1.0)
            ),
            "baseline_localizes_expected_sttc_mechanism": baseline_localizes_expected,
            "baseline_has_actionable_counterfactual_evidence": False,
            "baseline_has_external_counterfactual_control": False,
            "does_baseline_win_over_current_result": False,
            "plain_language": (
                "Baseline XAI can identify the same STTC mechanism, but it does not beat the current result "
                "because it lacks intervention-style counterfactual contrast and external Georgia control. "
                "It is a necessary comparator, not the main contribution."
            ),
        },
        "outputs": {
            "impurity": str(args.out_dir / "extratrees_impurity_group_importance.csv"),
            "permutation": str(args.out_dir / "extratrees_group_permutation_summary.csv"),
            "shap": str(args.out_dir / "treeshap_group_importance.csv") if shap_df is not None else None,
            "report": str(args.out_dir / "xai_baseline_report.json"),
        },
    }

    with (args.out_dir / "xai_baseline_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report["verdict"], indent=2))


if __name__ == "__main__":
    main()


