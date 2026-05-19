from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from phase2.ecg_mechanism_core import (
    BASE_GROUP_ORDER,
    apply_counterfactual,
    cls_metrics,
    feature_groups,
    group_columns,
    make_hgbdt,
    select_threshold,
)
from phase2.prototype_memory import CounterfactualMemory
from phase2.ecg_features import Sample, build_feature_matrix, iter_csv_samples, iter_train_env_samples


ECG_ID_RE = re.compile(r"ptbxl_(\d{5})_")
TASKS = ["STTC", "CD", "HYP", "MI"]
MI_COHORTS = ["broad", "single_code", "imi_only", "asmi_only", "inferior_single", "anterior_single"]


def ecg_id_from_path(path: Path) -> int:
    match = ECG_ID_RE.search(path.name)
    if not match:
        raise ValueError(f"Cannot parse ECG id from {path.name}")
    return int(match.group(1))


def load_superclasses(ptbxl_root: Path) -> dict[int, set[str]]:
    db = pd.read_csv(ptbxl_root / "ptbxl_database.csv")
    scp = pd.read_csv(ptbxl_root / "scp_statements.csv", index_col=0)
    diagnostic = scp[scp["diagnostic"] == 1.0]
    code_to_class = diagnostic["diagnostic_class"].dropna().to_dict()
    out: dict[int, set[str]] = {}
    for _, row in db.iterrows():
        codes = ast.literal_eval(row["scp_codes"])
        classes = {str(code_to_class[c]) for c in codes if c in code_to_class}
        out[int(row["ecg_id"])] = classes
    return out


def load_diagnostic_codes(ptbxl_root: Path) -> dict[int, tuple[str, ...]]:
    db = pd.read_csv(ptbxl_root / "ptbxl_database.csv")
    scp = pd.read_csv(ptbxl_root / "scp_statements.csv", index_col=0)
    diagnostic = scp[scp["diagnostic"] == 1.0]
    code_to_class = diagnostic["diagnostic_class"].dropna().to_dict()
    out: dict[int, tuple[str, ...]] = {}
    for _, row in db.iterrows():
        codes = ast.literal_eval(row["scp_codes"])
        mi_codes = sorted(str(code) for code in codes if code_to_class.get(code) == "MI")
        out[int(row["ecg_id"])] = tuple(mi_codes)
    return out


def keep_mi_record(mi_codes: tuple[str, ...], cohort: str) -> bool:
    if cohort == "broad":
        return True
    if cohort == "single_code":
        return len(mi_codes) == 1
    if cohort == "imi_only":
        return mi_codes == ("IMI",)
    if cohort == "asmi_only":
        return mi_codes == ("ASMI",)
    if cohort == "inferior_single":
        return len(mi_codes) == 1 and mi_codes[0] in {"IMI", "ILMI", "IPLMI", "IPMI"}
    if cohort == "anterior_single":
        return len(mi_codes) == 1 and mi_codes[0] in {"ASMI", "ALMI", "AMI"}
    raise ValueError(f"Unknown MI cohort: {cohort}")


def relabel_samples(
    samples: list[Sample],
    superclasses: dict[int, set[str]],
    target: str,
    diagnostic_codes: dict[int, tuple[str, ...]] | None = None,
    mi_cohort: str = "broad",
) -> list[Sample]:
    out: list[Sample] = []
    for sample in samples:
        ecg_id = ecg_id_from_path(sample.path)
        classes = superclasses.get(ecg_id, set())
        if classes == {"NORM"}:
            out.append(Sample(path=sample.path, label=0, split=sample.split, env=sample.env))
        elif classes == {target}:
            if target == "MI" and diagnostic_codes is not None:
                mi_codes = diagnostic_codes.get(ecg_id, ())
                if not keep_mi_record(mi_codes, mi_cohort):
                    continue
            out.append(Sample(path=sample.path, label=1, split=sample.split, env=sample.env))
    return out


def load_task_samples(
    csv_root: Path,
    superclasses: dict[int, set[str]],
    target: str,
    diagnostic_codes: dict[int, tuple[str, ...]] | None = None,
    mi_cohort: str = "broad",
) -> tuple[list[Sample], list[Sample], list[Sample]]:
    train_all = iter_train_env_samples(csv_root)
    val_all = iter_csv_samples(csv_root, ["ValID"])
    test_all = iter_csv_samples(csv_root, ["Eval_ID"])
    return (
        relabel_samples(train_all, superclasses, target, diagnostic_codes=diagnostic_codes, mi_cohort=mi_cohort),
        relabel_samples(val_all, superclasses, target, diagnostic_codes=diagnostic_codes, mi_cohort=mi_cohort),
        relabel_samples(test_all, superclasses, target, diagnostic_codes=diagnostic_codes, mi_cohort=mi_cohort),
    )


def counterfactual_cases(
    model: object,
    scaler: StandardScaler,
    memory: CounterfactualMemory,
    x: np.ndarray,
    threshold: float,
    group_cols: dict[str, list[int]],
    combo: tuple[str, ...],
) -> pd.DataFrame:
    x_z = scaler.transform(x)
    norm_idx, _ = memory.nearest_by_label(x_z, 0)
    norm_proto = scaler.inverse_transform(memory.prototype_matrix_[norm_idx])  # type: ignore[index]
    base_prob = model.predict_proba(x)[:, 1]
    x_cf = apply_counterfactual(x, norm_proto, combo, group_cols, alpha=1.0)
    cf_prob = model.predict_proba(x_cf)[:, 1]
    return pd.DataFrame(
        {
            "prob_drop": base_prob - cf_prob,
            "flipped": (base_prob >= threshold) & (cf_prob < threshold),
            "positive_drop": (base_prob - cf_prob) > 0,
            "cf_prob": cf_prob,
        }
    )


def clinical_combos(group_cols: dict[str, list[int]], max_groups: int) -> list[tuple[str, ...]]:
    names = [g for g in BASE_GROUP_ORDER if g in group_cols]
    combos: list[tuple[str, ...]] = [(g,) for g in names]
    if max_groups >= 2:
        for i, first in enumerate(names):
            for second in names[i + 1 :]:
                combos.append((first, second))
    return combos


def counterfactual_summary(
    model: object,
    scaler: StandardScaler,
    memory: CounterfactualMemory,
    x_pos: np.ndarray,
    threshold: float,
    group_cols: dict[str, list[int]],
    max_groups: int,
) -> pd.DataFrame:
    rows = []
    for combo in clinical_combos(group_cols, max_groups):
        case = counterfactual_cases(model, scaler, memory, x_pos, threshold, group_cols, combo)
        rows.append(
            {
                "combo": "+".join(combo),
                "n_groups": len(combo),
                "n_features": int(sum(len(group_cols[g]) for g in combo)),
                "mean_prob_drop": float(case["prob_drop"].mean()),
                "median_prob_drop": float(case["prob_drop"].median()),
                "flip_rate": float(case["flipped"].mean()),
                "positive_drop_rate": float(case["positive_drop"].mean()),
                "mean_cf_prob": float(case["cf_prob"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["mean_prob_drop", "flip_rate"], ascending=False)


def group_permutation(
    model: object,
    x_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float,
    group_cols: dict[str, list[int]],
    seed: int,
    repeats: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = cls_metrics(y_test, model.predict_proba(x_test)[:, 1], threshold)
    rows = []
    combos = [(g,) for g in BASE_GROUP_ORDER if g in group_cols]
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
                }
            )
    return (
        pd.DataFrame(rows)
        .groupby("combo", as_index=False)
        .agg(
            auc_drop_mean=("auc_drop", "mean"),
            auprc_drop_mean=("auprc_drop", "mean"),
            balanced_accuracy_drop_mean=("balanced_accuracy_drop", "mean"),
        )
        .sort_values("auc_drop_mean", ascending=False)
    )


def run_task(
    target: str,
    csv_root: Path,
    ptbxl_root: Path,
    superclasses: dict[int, set[str]],
    out_dir: Path,
    pre: int,
    post: int,
    downsample: int,
    prototypes_per_class: int,
    max_groups: int,
    permutation_repeats: int,
    seed: int,
) -> dict:
    task_dir = out_dir / target.lower()
    task_dir.mkdir(parents=True, exist_ok=True)
    train_s, val_s, test_s = load_task_samples(csv_root, superclasses, target)
    if len(train_s) < 100 or len(val_s) < 20 or len(test_s) < 20:
        return {"target": target, "skipped": True, "reason": "too few samples", "counts": {"train": len(train_s), "val": len(val_s), "test": len(test_s)}}

    x_train, y_train, feature_names, train_ok, train_fail = build_feature_matrix(train_s, pre, post, downsample)
    x_val, y_val, _, _, val_fail = build_feature_matrix(val_s, pre, post, downsample)
    x_test, y_test, _, _, test_fail = build_feature_matrix(test_s, pre, post, downsample)
    y_train = y_train.astype(int)
    y_val = y_val.astype(int)
    y_test = y_test.astype(int)

    if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2 or len(np.unique(y_test)) < 2:
        return {
            "target": target,
            "skipped": True,
            "reason": "missing a class in train/val/test",
            "counts": {"train": len(y_train), "val": len(y_val), "test": len(y_test)},
        }

    model = make_hgbdt(seed)
    model.fit(x_train, y_train)
    threshold = select_threshold(y_val, model.predict_proba(x_val)[:, 1])
    test_prob = model.predict_proba(x_test)[:, 1]
    metrics = cls_metrics(y_test, test_prob, threshold)

    scaler = StandardScaler()
    memory = CounterfactualMemory(prototypes_per_class=prototypes_per_class, seed=seed)
    memory.fit(scaler.fit_transform(x_train), y_train, [s.path.name for s in train_ok])
    groups = feature_groups(feature_names)
    x_pos = x_test[y_test == 1]

    cf = counterfactual_summary(model, scaler, memory, x_pos, threshold, groups, max_groups=max_groups)
    perm = group_permutation(model, x_test, y_test, threshold, groups, seed=seed, repeats=permutation_repeats)
    cf.to_csv(task_dir / "counterfactual_groups.csv", index=False)
    perm.to_csv(task_dir / "permutation_groups.csv", index=False)

    best_single = cf[cf["n_groups"] == 1].iloc[0].to_dict()
    best_pair = cf[cf["n_groups"] == 2].iloc[0].to_dict() if (cf["n_groups"] == 2).any() else {}
    report = {
        "target": target,
        "skipped": False,
        "data": {
            "train_n": int(len(y_train)),
            "val_n": int(len(y_val)),
            "test_n": int(len(y_test)),
            "positive_train_n": int(np.sum(y_train == 1)),
            "positive_val_n": int(np.sum(y_val == 1)),
            "positive_test_n": int(np.sum(y_test == 1)),
            "n_features": int(len(feature_names)),
            "failures": {"train": train_fail[:10], "val": val_fail[:10], "test": test_fail[:10]},
        },
        "classification": metrics,
        "mechanism": {
            "best_single_group": best_single,
            "best_pair_group": best_pair,
            "top_permutation_group": perm.iloc[0].to_dict(),
            "top5_counterfactual": cf.head(5).to_dict(orient="records"),
            "top5_permutation": perm.head(5).to_dict(orient="records"),
        },
        "files": {
            "counterfactual_groups": str(task_dir / "counterfactual_groups.csv"),
            "permutation_groups": str(task_dir / "permutation_groups.csv"),
        },
    }
    (task_dir / "task_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="PTB-XL disease-specific clinical counterfactual mechanism map.")
    parser.add_argument("--csv-root", type=Path, default=Path("Data") / "Generated_PTBXL_12Lead")
    parser.add_argument(
        "--ptbxl-root",
        type=Path,
        default=Path("Data") / "PTBXL" / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts") / "phase2" / "ptbxl_mechanism_map",
    )
    parser.add_argument("--tasks", nargs="+", default=TASKS, choices=TASKS)
    parser.add_argument("--pre", type=int, default=80)
    parser.add_argument("--post", type=int, default=120)
    parser.add_argument("--downsample", type=int, default=32)
    parser.add_argument("--prototypes-per-class", type=int, default=12)
    parser.add_argument("--max-groups", type=int, default=2)
    parser.add_argument("--permutation-repeats", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    superclasses = load_superclasses(args.ptbxl_root)
    reports = [
        run_task(
            target,
            args.csv_root,
            args.ptbxl_root,
            superclasses,
            args.out_dir,
            args.pre,
            args.post,
            args.downsample,
            args.prototypes_per_class,
            args.max_groups,
            args.permutation_repeats,
            args.seed,
        )
        for target in args.tasks
    ]
    summary_rows = []
    for report in reports:
        if report.get("skipped"):
            summary_rows.append({"target": report["target"], "skipped": True, "reason": report.get("reason")})
            continue
        summary_rows.append(
            {
                "target": report["target"],
                "skipped": False,
                "train_n": report["data"]["train_n"],
                "test_n": report["data"]["test_n"],
                "positive_test_n": report["data"]["positive_test_n"],
                "auc": report["classification"]["auc"],
                "auprc": report["classification"]["auprc"],
                "balanced_accuracy": report["classification"]["balanced_accuracy"],
                "best_single": report["mechanism"]["best_single_group"]["combo"],
                "best_single_drop": report["mechanism"]["best_single_group"]["mean_prob_drop"],
                "best_pair": report["mechanism"]["best_pair_group"].get("combo", ""),
                "best_pair_drop": report["mechanism"]["best_pair_group"].get("mean_prob_drop", np.nan),
                "top_permutation": report["mechanism"]["top_permutation_group"]["combo"],
                "top_permutation_auc_drop": report["mechanism"]["top_permutation_group"]["auc_drop_mean"],
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.out_dir / "mechanism_summary.csv", index=False)
    final = {
        "method": "PTB-XL disease-specific mechanism map",
        "policy": "pure superclass only: NORM-only negatives vs target-only positives; mixed-label records excluded",
        "reports": reports,
        "files": {"summary": str(args.out_dir / "mechanism_summary.csv")},
    }
    (args.out_dir / "mechanism_map_report.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()


