from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

from phase2.ecg_mechanism_core import (
    bootstrap_mean_ci,
    cls_metrics,
    load_ptbxl_norm_sttc_samples,
    make_hgbdt,
    map_y,
    select_threshold,
)
from phase2.run_georgia_external_validation import (
    LEADS_12,
    CODE_NAMES,
    detect_r_peaks,
    discover_georgia,
    lead_col,
    resample_signal,
)
from phase2.run_ptbxl_waveform_counterfactual import (
    apply_waveform_counterfactual,
    build_normal_template,
)
from phase2.ecg_features import build_feature_matrix, extract_ecg_features_from_frame


def read_georgia_frame(record_stem: Path, target_fs: float) -> pd.DataFrame:
    sig, meta = wfdb.rdsamp(str(record_stem))
    fs = float(meta["fs"])
    sig, fs = resample_signal(sig, fs, target_fs)
    names = [str(n) for n in meta["sig_name"]]
    lead_to_idx = {n.upper(): i for i, n in enumerate(names)}
    missing = [lead for lead in LEADS_12 if lead.upper() not in lead_to_idx]
    if missing:
        raise ValueError(f"missing leads: {missing}")

    lead_ii = sig[:, lead_to_idx["II"]].astype(float)
    peaks = detect_r_peaks(lead_ii, fs)
    if len(peaks) < 3:
        raise ValueError("too few detected R peaks")

    n = sig.shape[0]
    df = pd.DataFrame({"Time": np.arange(n, dtype=float) / fs, "Voltage": lead_ii, "Peak": np.zeros(n, dtype=int)})
    df.loc[peaks, "Peak"] = 3
    for lead in LEADS_12:
        df[lead_col(lead)] = sig[:, lead_to_idx[lead.upper()]].astype(float)
    return df


def predict_base_georgia(
    model: object,
    rows: list[tuple[Path, int, list[str]]],
    pre: int,
    post: int,
    downsample: int,
    target_fs: float,
) -> tuple[list[pd.DataFrame], np.ndarray, list[tuple[Path, int, list[str]]], list[str]]:
    frames = []
    probs = []
    ok_rows = []
    failures = []
    for stem, label, dx in rows:
        try:
            df = read_georgia_frame(stem, target_fs)
            x, _ = extract_ecg_features_from_frame(df, source=str(stem), pre=pre, post=post, downsample=downsample)
            prob = float(model.predict_proba(x[None, :])[:, 1][0])
        except Exception as exc:
            failures.append(f"{stem.name}: {type(exc).__name__}: {exc}")
            continue
        frames.append(df)
        probs.append(prob)
        ok_rows.append((stem, label, dx))
    return frames, np.asarray(probs, dtype=float), ok_rows, failures


def predict_cf_for_frames(
    model: object,
    frames: list[pd.DataFrame],
    rows: list[tuple[Path, int, list[str]]],
    base_prob: np.ndarray,
    normal_template: dict[str, np.ndarray],
    window: str,
    alpha: float,
    pre: int,
    post: int,
    downsample: int,
    ramp: int,
    threshold: float,
) -> tuple[pd.DataFrame, list[str]]:
    out = []
    failures = []
    for i, (df, row) in enumerate(zip(frames, rows)):
        stem, _, dx = row
        try:
            cf_df = apply_waveform_counterfactual(df, normal_template, window, alpha, pre, post, ramp)
            x_cf, _ = extract_ecg_features_from_frame(
                cf_df, source=f"{stem.name}:{window}:{alpha}", pre=pre, post=post, downsample=downsample
            )
            cf_prob = float(model.predict_proba(x_cf[None, :])[:, 1][0])
        except Exception as exc:
            failures.append(f"{stem.name}: {type(exc).__name__}: {exc}")
            continue
        before = float(base_prob[i])
        drop = before - cf_prob
        out.append(
            {
                "record": str(stem),
                "window": window,
                "alpha": float(alpha),
                "base_prob": before,
                "cf_prob": cf_prob,
                "prob_drop": float(drop),
                "flipped": bool(before >= threshold and cf_prob < threshold),
                "positive_drop": bool(drop > 0),
                "dx_codes": ",".join(dx),
                "dx_names": ",".join(CODE_NAMES.get(code, code) for code in dx),
            }
        )
    return pd.DataFrame(out), failures


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
        "mean_base_prob": float(df["base_prob"].mean()),
        "mean_cf_prob": float(df["cf_prob"].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Georgia waveform counterfactual analysis.")
    parser.add_argument("--ptbxl-root", type=Path, default=Path("Data") / "Generated_PTBXL_12Lead")
    parser.add_argument("--georgia-root", type=Path, default=Path("Data") / "PhysioNet_Challenge_2020_Georgia")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts") / "phase2" / "georgia_waveform_counterfactual_sttc",
    )
    parser.add_argument("--pre", type=int, default=80)
    parser.add_argument("--post", type=int, default=120)
    parser.add_argument("--downsample", type=int, default=32)
    parser.add_argument("--target-fs", type=float, default=100.0)
    parser.add_argument("--template-max-records", type=int, default=800)
    parser.add_argument("--max-positive", type=int, default=600, help="0 means all Georgia ST/T positives.")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--ramp", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_s, val_s, test_s, _ = load_ptbxl_norm_sttc_samples(args.ptbxl_root)
    x_train, y_train_raw, _, _, train_fail = build_feature_matrix(train_s, args.pre, args.post, args.downsample)
    x_val, y_val_raw, _, _, val_fail = build_feature_matrix(val_s, args.pre, args.post, args.downsample)
    x_test, y_test_raw, _, _, test_fail = build_feature_matrix(test_s, args.pre, args.post, args.downsample)
    y_train = map_y(y_train_raw)
    y_val = map_y(y_val_raw)
    y_test = map_y(y_test_raw)

    model = make_hgbdt(args.seed)
    model.fit(x_train, y_train)
    threshold = select_threshold(y_val, model.predict_proba(x_val)[:, 1])
    ptbxl_metrics = cls_metrics(y_test, model.predict_proba(x_test)[:, 1], threshold)
    normal_template = build_normal_template(train_s, args.pre, args.post, args.template_max_records)

    rows = [row for row in discover_georgia(args.georgia_root, "no_st_t", None, args.seed) if row[1] == 1]
    if args.max_positive and len(rows) > args.max_positive:
        rng = np.random.default_rng(args.seed)
        idx = sorted(rng.choice(len(rows), size=args.max_positive, replace=False).tolist())
        rows = [rows[i] for i in idx]

    frames, base_prob, ok_rows, base_failures = predict_base_georgia(
        model, rows, args.pre, args.post, args.downsample, args.target_fs
    )
    windows = ["st_t", "st_only", "t_only", "pre_qrs_control", "p_pr_control", "qrs_control"]
    alpha_values = [0.0, 0.25, 0.5, 0.75, 1.0]

    all_rows = []
    failures: dict[str, list[str]] = {"base": base_failures[:20]}
    summary_rows = []
    for window in windows:
        for alpha in alpha_values:
            df, fail = predict_cf_for_frames(
                model,
                frames,
                ok_rows,
                base_prob,
                normal_template,
                window,
                alpha,
                args.pre,
                args.post,
                args.downsample,
                args.ramp,
                threshold,
            )
            failures[f"{window}@{alpha}"] = fail[:10]
            if not df.empty:
                all_rows.append(df)
            summary_rows.append({"window": window, "alpha": alpha, **summarize_rows(df, args.n_boot, args.seed)})

    per_case = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    summary_df = pd.DataFrame(summary_rows)
    per_case.to_csv(args.out_dir / "georgia_waveform_counterfactual_per_case.csv", index=False)
    summary_df.to_csv(args.out_dir / "georgia_waveform_counterfactual_summary.csv", index=False)

    target = summary_df[(summary_df["window"] == "st_t") & (summary_df["alpha"] == 1.0)].iloc[0].to_dict()
    controls = summary_df[
        (summary_df["alpha"] == 1.0) & (~summary_df["window"].isin(["st_t", "st_only", "t_only"]))
    ].sort_values("mean_prob_drop", ascending=False)
    best_control = controls.iloc[0].to_dict()
    alpha_st_t = summary_df[summary_df["window"] == "st_t"].sort_values("alpha")

    report = {
        "method": "Georgia waveform ST/T counterfactual",
        "data": {
            "candidate_positive_n": int(len(rows)),
            "usable_positive_n": int(len(ok_rows)),
            "max_positive": args.max_positive or None,
            "target_fs": float(args.target_fs),
            "template_max_records": int(args.template_max_records),
            "failures": failures,
            "ptbxl_failures": {"train": train_fail[:10], "val": val_fail[:10], "test": test_fail[:10]},
        },
        "ptbxl_reference": {"classification": ptbxl_metrics, "threshold": float(threshold)},
        "georgia_base_positive_prob": {
            "mean": float(np.mean(base_prob)) if len(base_prob) else None,
            "median": float(np.median(base_prob)) if len(base_prob) else None,
            "predicted_positive_rate": float(np.mean(base_prob >= threshold)) if len(base_prob) else None,
        },
        "target_st_t_alpha1": target,
        "best_control_alpha1": best_control,
        "target_minus_best_control_mean_drop": float(target["mean_prob_drop"] - best_control["mean_prob_drop"]),
        "alpha_sweep_st_t": {
            "rows": alpha_st_t.to_dict(orient="records"),
            "monotone_mean_drop": bool(np.all(np.diff(alpha_st_t["mean_prob_drop"].to_numpy(dtype=float)) >= -1e-9)),
            "monotone_flip_rate": bool(np.all(np.diff(alpha_st_t["flip_rate"].to_numpy(dtype=float)) >= -1e-9)),
        },
        "ranked_alpha1": summary_df[summary_df["alpha"] == 1.0]
        .sort_values("mean_prob_drop", ascending=False)
        .to_dict(orient="records"),
        "selection_checks": {
            "target_drop_gt_0_30": bool(target["mean_prob_drop"] > 0.30),
            "target_minus_control_gt_0_15": bool(
                (target["mean_prob_drop"] - best_control["mean_prob_drop"]) > 0.15
            ),
            "positive_drop_rate_gt_0_80": bool(target["positive_drop_rate"] > 0.80),
        },
        "outputs": {
            "summary": str(args.out_dir / "georgia_waveform_counterfactual_summary.csv"),
            "per_case": str(args.out_dir / "georgia_waveform_counterfactual_per_case.csv"),
            "report": str(args.out_dir / "georgia_waveform_counterfactual_report.json"),
        },
    }
    with (args.out_dir / "georgia_waveform_counterfactual_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report["selection_checks"], indent=2))


if __name__ == "__main__":
    main()

