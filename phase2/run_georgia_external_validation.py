from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from math import gcd

from scipy.signal import butter, filtfilt, find_peaks, resample_poly
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from phase2.prototype_memory import CounterfactualMemory
from phase2.ecg_mechanism_core import (
    apply_counterfactual,
    feature_groups,
    load_ptbxl_norm_sttc_samples,
    make_hgbdt,
    map_y,
    select_threshold,
)
from phase2.ecg_features import build_feature_matrix, extract_ecg_features_from_frame


NORMAL_CODE = "426783006"  # Sinus rhythm / normal sinus rhythm in Challenge 2020 mappings.
ST_T_CODES = {
    "428750005",  # ST-T change / nonspecific ST-T abnormality.
    "164934002",  # T wave abnormal.
    "59931005",  # T wave inversion.
    "164930006",  # ST elevation.
    "164931005",  # ST depression / ST segment abnormality in ECG mapping.
}
CODE_NAMES = {
    "426783006": "normal_sinus_rhythm",
    "428750005": "st_t_change",
    "164934002": "t_wave_abnormal",
    "59931005": "t_wave_inversion",
    "164930006": "st_elevation",
    "164931005": "st_depression_or_st_abnormality",
}
LEADS_12 = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def make_model(seed: int) -> HistGradientBoostingClassifier:
    return make_hgbdt(seed)


def cls_metrics(y: np.ndarray, prob: np.ndarray, threshold: float) -> dict:
    pred = prob >= threshold
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "auc": float(roc_auc_score(y, prob)),
        "auprc": float(average_precision_score(y, prob)),
        "threshold": float(threshold),
        "positive_rate": float(np.mean(y)),
        "predicted_positive_rate": float(np.mean(pred)),
    }


def read_dx(hea_path: Path) -> list[str]:
    dx: list[str] = []
    for line in hea_path.read_text(errors="ignore").splitlines():
        low = line.lower()
        if low.startswith("# dx:") or low.startswith("#dx:"):
            dx = [x.strip() for x in line.split(":", 1)[1].split(",") if x.strip()]
            break
    return dx


def georgia_label(dx: list[str], negative_policy: str) -> int | None:
    dx_set = set(dx)
    if dx_set & ST_T_CODES:
        return 1
    if negative_policy == "strict_normal":
        return 0 if dx == [NORMAL_CODE] else None
    if negative_policy == "no_st_t":
        return 0 if not (dx_set & ST_T_CODES) else None
    raise ValueError(f"Unknown negative policy: {negative_policy}")


def bandpass_filter(signal: np.ndarray, fs: float) -> np.ndarray:
    nyq = 0.5 * fs
    high = min(40.0, nyq * 0.95)
    b, a = butter(2, [0.5 / nyq, high / nyq], btype="band")
    return filtfilt(b, a, signal)


def detect_r_peaks(signal: np.ndarray, fs: float) -> np.ndarray:
    filtered = bandpass_filter(signal.astype(float), fs)
    filtered = (filtered - np.mean(filtered)) / max(np.std(filtered), 1e-9)
    diff = np.diff(filtered, prepend=filtered[0])
    squared = diff**2
    window = max(1, int(0.15 * fs))
    integrated = np.convolve(squared, np.ones(window) / window, mode="same")
    peaks, _ = find_peaks(
        integrated,
        distance=max(1, int(0.25 * fs)),
        height=np.percentile(integrated, 85),
    )
    if len(peaks) < 3:
        return peaks
    rr = np.diff(peaks) / fs
    keep = np.ones(len(peaks), dtype=bool)
    bad = (rr < 0.3) | (rr > 2.0)
    keep[1:][bad] = False
    return peaks[keep]


def lead_col(lead: str) -> str:
    return f"Lead_{lead.replace('-', '_')}"


def resample_signal(sig: np.ndarray, fs: float, target_fs: float) -> tuple[np.ndarray, float]:
    if abs(fs - target_fs) < 1e-6:
        return sig, fs
    fs_i = int(round(fs))
    target_i = int(round(target_fs))
    div = gcd(fs_i, target_i)
    up = target_i // div
    down = fs_i // div
    return resample_poly(sig, up=up, down=down, axis=0), float(target_i)


def read_georgia_features(
    record_stem: Path,
    pre: int,
    post: int,
    downsample: int,
    target_fs: float,
) -> tuple[np.ndarray, list[str]]:
    sig, meta = wfdb.rdsamp(str(record_stem))
    fs = float(meta["fs"])
    sig, fs = resample_signal(sig, fs, target_fs)
    names = [str(n) for n in meta["sig_name"]]
    lead_to_idx = {n.upper(): i for i, n in enumerate(names)}
    missing = [ld for ld in LEADS_12 if ld.upper() not in lead_to_idx]
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
    return extract_ecg_features_from_frame(df, source=str(record_stem), pre=pre, post=post, downsample=downsample)


def discover_georgia(root: Path, negative_policy: str, max_per_class: int | None, seed: int) -> list[tuple[Path, int, list[str]]]:
    rows: list[tuple[Path, int, list[str]]] = []
    for hea in sorted(root.rglob("*.hea")):
        if not hea.with_suffix(".mat").exists():
            continue
        dx = read_dx(hea)
        label = georgia_label(dx, negative_policy)
        if label is None:
            continue
        rows.append((hea.with_suffix(""), label, dx))
    if max_per_class is None:
        return rows
    rng = np.random.default_rng(seed)
    capped: list[tuple[Path, int, list[str]]] = []
    for label in [0, 1]:
        group = [r for r in rows if r[1] == label]
        if len(group) > max_per_class:
            idx = rng.choice(len(group), size=max_per_class, replace=False)
            group = [group[int(i)] for i in idx]
        capped.extend(group)
    return sorted(capped, key=lambda x: str(x[0]))


def build_georgia_matrix(
    rows: list[tuple[Path, int, list[str]]],
    pre: int,
    post: int,
    downsample: int,
    target_fs: float,
) -> tuple[np.ndarray, np.ndarray, list[str], list[tuple[Path, int, list[str]]], list[str]]:
    xs: list[np.ndarray] = []
    ys: list[int] = []
    ok: list[tuple[Path, int, list[str]]] = []
    names: list[str] | None = None
    failures: list[str] = []
    for stem, label, dx in rows:
        try:
            x, n = read_georgia_features(stem, pre=pre, post=post, downsample=downsample, target_fs=target_fs)
        except Exception as exc:
            failures.append(f"{stem.name}: {exc}")
            continue
        if not np.all(np.isfinite(x)):
            failures.append(f"{stem.name}: non-finite features")
            continue
        if names is None:
            names = n
        xs.append(x)
        ys.append(label)
        ok.append((stem, label, dx))
    if not xs:
        raise RuntimeError("No Georgia feature rows extracted.")
    return np.stack(xs), np.asarray(ys, dtype=int), names or [], ok, failures


def min_memory_distance(memory: CounterfactualMemory, scaler: StandardScaler, x: np.ndarray) -> np.ndarray:
    return np.min(memory.distances(scaler.transform(x)), axis=1)


def counterfactual_summary(
    model: object,
    scaler: StandardScaler,
    memory: CounterfactualMemory,
    x_pos: np.ndarray,
    threshold: float,
    groups: dict[str, list[int]],
) -> dict:
    combo = ("st_segment", "t_wave")
    x_z = scaler.transform(x_pos)
    norm_idx, _ = memory.nearest_by_label(x_z, 0)
    norm_proto = scaler.inverse_transform(memory.prototype_matrix_[norm_idx])  # type: ignore[index]
    base_prob = model.predict_proba(x_pos)[:, 1]
    x_cf = apply_counterfactual(x_pos, norm_proto, combo, groups, alpha=1.0)
    cf_prob = model.predict_proba(x_cf)[:, 1]
    prob_drop = base_prob - cf_prob
    return {
        "combo": "+".join(combo),
        "n_positive": int(len(x_pos)),
        "base_positive_rate": float(np.mean(base_prob >= threshold)),
        "mean_prob_drop": float(np.mean(prob_drop)),
        "median_prob_drop": float(np.median(prob_drop)),
        "positive_drop_rate": float(np.mean(prob_drop > 0)),
        "flip_rate": float(np.mean((base_prob >= threshold) & (cf_prob < threshold))),
        "mean_cf_prob": float(np.mean(cf_prob)),
    }


def label_counts(rows: list[tuple[Path, int, list[str]]]) -> dict:
    out = {"negative": 0, "positive": 0}
    code_counts: dict[str, int] = {}
    for _, label, dx in rows:
        out["positive" if label else "negative"] += 1
        for code in dx:
            code_counts[CODE_NAMES.get(code, code)] = code_counts.get(CODE_NAMES.get(code, code), 0) + 1
    return {"binary": out, "codes": dict(sorted(code_counts.items(), key=lambda kv: kv[1], reverse=True)[:30])}


def main() -> None:
    parser = argparse.ArgumentParser(description="External Georgia ST/T validation for PTB-XL clinical counterfactual memory.")
    parser.add_argument("--ptbxl-root", type=Path, default=Path("Data") / "Generated_PTBXL_12Lead")
    parser.add_argument("--georgia-root", type=Path, default=Path("Data") / "PhysioNet_Challenge_2020_Georgia")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts") / "phase2" / "external_georgia_validation",
    )
    parser.add_argument("--pre", type=int, default=80)
    parser.add_argument("--post", type=int, default=120)
    parser.add_argument("--downsample", type=int, default=32)
    parser.add_argument("--prototypes-per-class", type=int, default=12)
    parser.add_argument("--negative-policy", choices=["strict_normal", "no_st_t"], default="strict_normal")
    parser.add_argument("--max-per-class", type=int, default=600)
    parser.add_argument("--target-fs", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_s, val_s, test_s, _ = load_ptbxl_norm_sttc_samples(args.ptbxl_root)
    x_train, y_train_raw, feature_names, train_ok, train_fail = build_feature_matrix(
        train_s, args.pre, args.post, args.downsample
    )
    x_val, y_val_raw, _, _, val_fail = build_feature_matrix(val_s, args.pre, args.post, args.downsample)
    x_test, y_test_raw, _, _, test_fail = build_feature_matrix(test_s, args.pre, args.post, args.downsample)
    y_train = map_y(y_train_raw)
    y_val = map_y(y_val_raw)
    y_test = map_y(y_test_raw)

    model = make_model(args.seed)
    model.fit(x_train, y_train)
    threshold = select_threshold(y_val, model.predict_proba(x_val)[:, 1])
    ptbxl_prob = model.predict_proba(x_test)[:, 1]
    ptbxl_metrics = cls_metrics(y_test, ptbxl_prob, threshold)

    scaler = StandardScaler()
    x_train_z = scaler.fit_transform(x_train)
    memory = CounterfactualMemory(prototypes_per_class=args.prototypes_per_class, seed=args.seed)
    memory.fit(x_train_z, y_train, [s.path.name for s in train_ok])
    val95_memory_distance = float(np.quantile(min_memory_distance(memory, scaler, x_val), 0.95))

    max_per_class = None if args.max_per_class is not None and args.max_per_class <= 0 else args.max_per_class
    candidate_rows = discover_georgia(args.georgia_root, args.negative_policy, max_per_class, args.seed)
    x_geo, y_geo, geo_names, ok_rows, geo_fail = build_georgia_matrix(
        candidate_rows, args.pre, args.post, args.downsample, args.target_fs
    )
    if feature_names != geo_names:
        raise RuntimeError("Feature schema mismatch between PTB-XL 12-lead and Georgia.")

    geo_prob = model.predict_proba(x_geo)[:, 1]
    geo_metrics = cls_metrics(y_geo, geo_prob, threshold)
    geo_z = scaler.transform(x_geo)
    mem_dist = min_memory_distance(memory, scaler, x_geo)
    _, d_norm = memory.nearest_by_label(geo_z, 0)
    _, d_sttc = memory.nearest_by_label(geo_z, 1)
    memory_margin = d_norm - d_sttc
    pos_idx = np.flatnonzero(y_geo == 1)
    cf = counterfactual_summary(model, scaler, memory, x_geo[pos_idx], threshold, feature_groups(feature_names))

    pred_df = pd.DataFrame(
        {
            "record": [str(r[0]) for r in ok_rows],
            "label": y_geo,
            "sttc_prob": geo_prob,
            "pred_positive": geo_prob >= threshold,
            "memory_distance": mem_dist,
            "memory_margin_norm_minus_sttc": memory_margin,
            "distance_to_norm": d_norm,
            "distance_to_sttc": d_sttc,
            "memory_ood_flag": mem_dist > val95_memory_distance,
            "dx_codes": [",".join(r[2]) for r in ok_rows],
            "dx_names": [",".join(CODE_NAMES.get(c, c) for c in r[2]) for r in ok_rows],
        }
    )
    pred_df.to_csv(args.out_dir / "georgia_predictions.csv", index=False)

    report = {
        "method": "External Georgia ST/T validation",
        "interpretation_limits": [
            "Partial Georgia download; results are provisional until all g1-g11 records are present.",
            "Georgia labels are multi-label SNOMED findings, not the same PTB-XL superclass ontology.",
            "Positive class is ST/T-related SNOMED finding; negative class uses the selected policy.",
        ],
        "label_mapping": {
            "positive_codes": {code: CODE_NAMES.get(code, code) for code in sorted(ST_T_CODES)},
            "negative_policy": args.negative_policy,
            "negative_code": {NORMAL_CODE: CODE_NAMES[NORMAL_CODE]},
        },
        "data": {
            "georgia_root": str(args.georgia_root),
            "candidate_counts_before_feature_extraction": label_counts(candidate_rows),
            "extracted_counts": label_counts(ok_rows),
            "n_extracted": int(len(y_geo)),
            "n_failures": int(len(geo_fail)),
            "failures": geo_fail[:20],
            "max_per_class": max_per_class,
            "n_features": int(len(feature_names)),
            "target_fs": float(args.target_fs),
        },
        "ptbxl_reference": {
            "test_metrics": ptbxl_metrics,
            "threshold": float(threshold),
            "failures": {"train": train_fail[:10], "val": val_fail[:10], "test": test_fail[:10]},
        },
        "georgia_external": {
            "metrics_at_ptbxl_threshold": geo_metrics,
            "mean_prob_by_label": {
                "negative": float(np.mean(geo_prob[y_geo == 0])),
                "positive": float(np.mean(geo_prob[y_geo == 1])),
            },
            "median_prob_by_label": {
                "negative": float(np.median(geo_prob[y_geo == 0])),
                "positive": float(np.median(geo_prob[y_geo == 1])),
            },
            "memory_ood": {
                "val95_threshold": val95_memory_distance,
                "flag_rate_all": float(np.mean(mem_dist > val95_memory_distance)),
                "flag_rate_negative": float(np.mean(mem_dist[y_geo == 0] > val95_memory_distance)),
                "flag_rate_positive": float(np.mean(mem_dist[y_geo == 1] > val95_memory_distance)),
                "mean_distance_all": float(np.mean(mem_dist)),
            },
            "counterfactual_on_georgia_positive": cf,
        },
        "files": {"predictions": str(args.out_dir / "georgia_predictions.csv")},
    }
    (args.out_dir / "georgia_validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()


