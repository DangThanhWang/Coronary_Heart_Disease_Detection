from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from phase2.ecg_mechanism_core import (
    bootstrap_mean_ci,
    cls_metrics,
    load_ptbxl_norm_sttc_samples,
    make_hgbdt,
    map_y,
    select_threshold,
)
from phase2.ecg_features import (
    Sample,
    _fs_from_time,
    _lead_columns,
    _median_beat,
    build_feature_matrix,
    extract_ecg_features_from_frame,
)


def rel_windows(pre: int, fs: float) -> dict[str, list[tuple[int, int]]]:
    return {
        "st_t": [(pre + int(0.10 * fs), pre + int(0.72 * fs))],
        "st_only": [(pre + int(0.10 * fs), pre + int(0.32 * fs))],
        "t_only": [(pre + int(0.32 * fs), pre + int(0.72 * fs))],
        "pre_qrs_control": [(pre - int(0.62 * fs), pre)],
        "p_pr_control": [(pre - int(0.22 * fs), pre)],
        "qrs_control": [(pre - int(0.04 * fs), pre + int(0.10 * fs))],
    }


def clipped_window(a: int, b: int, length: int) -> tuple[int, int] | None:
    aa = max(0, int(a))
    bb = min(length, int(b))
    if bb <= aa + 1:
        return None
    return aa, bb


def taper(n: int, ramp: int) -> np.ndarray:
    w = np.ones(n, dtype=float)
    r = min(int(ramp), n // 2)
    if r > 0:
        edge = np.linspace(0.0, 1.0, r + 2, dtype=float)[1:-1]
        w[:r] = edge
        w[-r:] = edge[::-1]
    return w


def beat_z_template(df: pd.DataFrame, pre: int, post: int) -> dict[str, np.ndarray]:
    peaks = df.index[df["Peak"] == 3].to_numpy(dtype=int)
    out = {}
    for col in _lead_columns(df):
        raw = df[col].to_numpy(dtype=float)
        beat = _median_beat(raw, peaks, pre=pre, post=post)
        if beat is None:
            continue
        beat = beat - float(np.median(beat[: min(20, beat.size)]))
        beat_std = float(np.std(beat))
        out[col] = (beat - float(np.mean(beat))) / max(beat_std, 1e-6)
    return out


def build_normal_template(samples: list[Sample], pre: int, post: int, max_records: int) -> dict[str, np.ndarray]:
    by_lead: dict[str, list[np.ndarray]] = {}
    used = 0
    for sample in samples:
        if sample.label != 0:
            continue
        try:
            df = pd.read_csv(sample.path)
            tmpl = beat_z_template(df, pre, post)
        except Exception:
            continue
        if not tmpl:
            continue
        for lead, vec in tmpl.items():
            by_lead.setdefault(lead, []).append(vec)
        used += 1
        if max_records > 0 and used >= max_records:
            break
    if not by_lead:
        raise RuntimeError("Could not build any normal waveform templates.")
    return {lead: np.median(np.stack(rows, axis=0), axis=0) for lead, rows in by_lead.items()}


def apply_waveform_counterfactual(
    df: pd.DataFrame,
    normal_template: dict[str, np.ndarray],
    window_name: str,
    alpha: float,
    pre: int,
    post: int,
    ramp: int,
) -> pd.DataFrame:
    out = df.copy()
    fs = _fs_from_time(df)
    length = pre + post
    windows = rel_windows(pre, fs)[window_name]
    peaks = df.index[df["Peak"] == 3].to_numpy(dtype=int)
    leads = [lead for lead in _lead_columns(df) if lead in normal_template]

    for lead in leads:
        raw = df[lead].to_numpy(dtype=float)
        beat = _median_beat(raw, peaks, pre=pre, post=post)
        if beat is None:
            continue
        baseline = float(np.median(beat[: min(20, beat.size)]))
        beat_centered = beat - baseline
        scale = max(float(np.std(beat_centered)), 1e-6)
        target_beat = baseline + scale * normal_template[lead]
        edited = out[lead].to_numpy(dtype=float)

        for peak in peaks:
            beat_start = int(peak) - pre
            beat_end = int(peak) + post
            if beat_start < 0 or beat_end > len(edited):
                continue
            for a, b in windows:
                win = clipped_window(a, b, length)
                if win is None:
                    continue
                aa, bb = win
                src = target_beat[aa:bb]
                dst_a = beat_start + aa
                dst_b = beat_start + bb
                weights = alpha * taper(dst_b - dst_a, ramp)
                edited[dst_a:dst_b] = (1.0 - weights) * edited[dst_a:dst_b] + weights * src
        out[lead] = edited
    return out


def predict_counterfactual_batch(
    model: object,
    samples: list[Sample],
    base_prob: np.ndarray,
    normal_template: dict[str, np.ndarray],
    window_name: str,
    alpha: float,
    pre: int,
    post: int,
    downsample: int,
    ramp: int,
    threshold: float,
) -> tuple[pd.DataFrame, list[str]]:
    rows = []
    failures = []
    for i, sample in enumerate(samples):
        try:
            df = pd.read_csv(sample.path)
            cf_df = apply_waveform_counterfactual(df, normal_template, window_name, alpha, pre, post, ramp)
            x_cf, _ = extract_ecg_features_from_frame(
                cf_df,
                source=f"{sample.path.name}:{window_name}:{alpha}",
                pre=pre,
                post=post,
                downsample=downsample,
            )
            cf_prob = float(model.predict_proba(x_cf[None, :])[:, 1][0])
        except Exception as exc:
            failures.append(f"{sample.path.name}: {type(exc).__name__}: {exc}")
            continue
        before = float(base_prob[i])
        drop = before - cf_prob
        rows.append(
            {
                "sample": sample.path.name,
                "window": window_name,
                "alpha": float(alpha),
                "base_prob": before,
                "cf_prob": cf_prob,
                "prob_drop": float(drop),
                "flipped": bool(before >= threshold and cf_prob < threshold),
                "positive_drop": bool(drop > 0),
            }
        )
    return pd.DataFrame(rows), failures


def summarize_rows(df: pd.DataFrame, n_boot: int, seed: int) -> dict:
    if df.empty:
        return {"n": 0}
    ci = bootstrap_mean_ci(df["prob_drop"].to_numpy(dtype=float), n_boot=n_boot, seed=seed)
    return {
        "n": int(len(df)),
        "mean_prob_drop": float(df["prob_drop"].mean()),
        "median_prob_drop": float(df["prob_drop"].median()),
        "prob_drop_ci95_low": float(ci["ci95_low"]),
        "prob_drop_ci95_high": float(ci["ci95_high"]),
        "flip_rate": float(df["flipped"].mean()),
        "positive_drop_rate": float(df["positive_drop"].mean()),
        "mean_cf_prob": float(df["cf_prob"].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the PTB-XL waveform counterfactual analysis.")
    parser.add_argument("--csv-root", type=Path, default=Path("Data/Generated_PTBXL_12Lead"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/phase2/waveform_counterfactual_sttc"))
    parser.add_argument("--pre", type=int, default=80)
    parser.add_argument("--post", type=int, default=120)
    parser.add_argument("--downsample", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--template-max-records", type=int, default=800)
    parser.add_argument("--max-positive", type=int, default=0)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--ramp", type=int, default=5)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_s, val_s, test_s, _ = load_ptbxl_norm_sttc_samples(args.csv_root)

    x_train, y_train_raw, _, _, train_fail = build_feature_matrix(train_s, args.pre, args.post, args.downsample)
    x_val, y_val_raw, _, _, val_fail = build_feature_matrix(val_s, args.pre, args.post, args.downsample)
    x_test, y_test_raw, _, test_ok, test_fail = build_feature_matrix(test_s, args.pre, args.post, args.downsample)
    y_train = map_y(y_train_raw)
    y_val = map_y(y_val_raw)
    y_test = map_y(y_test_raw)

    model = make_hgbdt(args.seed)
    model.fit(x_train, y_train)
    threshold = select_threshold(y_val, model.predict_proba(x_val)[:, 1])
    test_prob = model.predict_proba(x_test)[:, 1]
    test_metrics = cls_metrics(y_test, test_prob, threshold)

    pos_idx = np.flatnonzero(y_test == 1)
    if args.max_positive > 0:
        rng = np.random.default_rng(args.seed)
        pos_idx = np.sort(rng.choice(pos_idx, size=min(args.max_positive, len(pos_idx)), replace=False))
    pos_samples = [test_ok[int(i)] for i in pos_idx]
    pos_prob = test_prob[pos_idx]

    normal_template = build_normal_template(train_s, args.pre, args.post, args.template_max_records)
    alpha_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    windows = ["st_t", "st_only", "t_only", "pre_qrs_control", "p_pr_control", "qrs_control"]
    all_rows = []
    all_failures: dict[str, list[str]] = {}
    summaries = []

    for window in windows:
        for alpha in alpha_values:
            rows, failures = predict_counterfactual_batch(
                model,
                pos_samples,
                pos_prob,
                normal_template,
                window,
                alpha,
                args.pre,
                args.post,
                args.downsample,
                args.ramp,
                threshold,
            )
            if not rows.empty:
                all_rows.append(rows)
            all_failures[f"{window}@{alpha}"] = failures[:10]
            summaries.append({"window": window, "alpha": alpha, **summarize_rows(rows, args.n_boot, args.seed)})

    per_case = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    summary_df = pd.DataFrame(summaries)
    per_case.to_csv(args.out_dir / "waveform_counterfactual_per_case.csv", index=False)
    summary_df.to_csv(args.out_dir / "waveform_counterfactual_summary.csv", index=False)

    alpha1 = summary_df[summary_df["alpha"] == 1.0].sort_values("mean_prob_drop", ascending=False)
    target = summary_df[(summary_df["window"] == "st_t") & (summary_df["alpha"] == 1.0)].iloc[0].to_dict()
    control = summary_df[
        (summary_df["alpha"] == 1.0) & (~summary_df["window"].isin(["st_t", "st_only", "t_only"]))
    ].sort_values("mean_prob_drop", ascending=False)
    best_control = control.iloc[0].to_dict()
    target_minus_control = float(target["mean_prob_drop"] - best_control["mean_prob_drop"])

    report = {
        "method": "PTB-XL waveform ST/T counterfactual",
        "task": "PTB-XL 12-lead NORM vs STTC",
        "data": {
            "train_n": int(len(y_train)),
            "val_n": int(len(y_val)),
            "test_n": int(len(y_test)),
            "positive_test_n_used": int(len(pos_samples)),
            "template_max_records": int(args.template_max_records),
            "failures": {"train": train_fail[:10], "val": val_fail[:10], "test": test_fail[:10], **all_failures},
        },
        "classification": test_metrics,
        "target_st_t_alpha1": target,
        "best_control_alpha1": best_control,
        "target_minus_best_control_mean_drop": target_minus_control,
        "ranked_alpha1": alpha1.to_dict(orient="records"),
        "selection_checks": {
            "target_drop_gt_0_30": bool(target["mean_prob_drop"] > 0.30),
            "target_minus_control_gt_0_15": bool(target_minus_control > 0.15),
            "positive_drop_rate_gt_0_85": bool(target["positive_drop_rate"] > 0.85),
        },
        "outputs": {
            "summary": str(args.out_dir / "waveform_counterfactual_summary.csv"),
            "per_case": str(args.out_dir / "waveform_counterfactual_per_case.csv"),
            "report": str(args.out_dir / "waveform_counterfactual_report.json"),
        },
    }
    with (args.out_dir / "waveform_counterfactual_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report["selection_checks"], indent=2))


if __name__ == "__main__":
    main()

