from __future__ import annotations

import argparse
import ast
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from scipy.signal import butter, filtfilt, find_peaks


LABEL_MAP = {
    "NORM": 0,
    "MI": 1,
    "STTC": 2,
    "CD": 3,
}
LEADS_12 = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
KNOWN_CLASSES = set(LABEL_MAP)


def normalize_lead(name: str) -> str:
    return str(name).upper().replace(" ", "")


def lead_col(lead: str) -> str:
    return f"Lead_{lead.replace('-', '_')}"


def bandpass_filter(signal: np.ndarray, fs: int, low: float = 0.5, high: float = 40.0) -> np.ndarray:
    if len(signal) == 0:
        return signal
    nyq = 0.5 * fs
    b, a = butter(2, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, signal)


def detect_r_peaks(signal: np.ndarray, fs: int) -> np.ndarray:
    if len(signal) == 0:
        return np.asarray([], dtype=int)

    filtered = bandpass_filter(signal, fs)
    filtered = (filtered - np.mean(filtered)) / (np.std(filtered) + 1e-9)
    diff = np.diff(filtered, prepend=filtered[0])
    squared = diff ** 2

    window = max(1, int(0.15 * fs))
    integrated = np.convolve(squared, np.ones(window) / window, mode="same")
    peaks, _ = find_peaks(
        integrated,
        distance=int(0.25 * fs),
        height=np.percentile(integrated, 85),
    )
    if len(peaks) < 3:
        return np.asarray(peaks, dtype=int)

    rr = np.diff(peaks)
    median_rr = np.median(rr)
    if median_rr <= 0:
        return np.asarray(peaks, dtype=int)

    keep = [int(peaks[0])]
    for peak in peaks[1:]:
        rr_i = int(peak) - keep[-1]
        if 0.6 * median_rr <= rr_i <= 1.6 * median_rr:
            keep.append(int(peak))
    return np.asarray(keep, dtype=int)


def site_to_env(site_value: float | int | None) -> str:
    if site_value is None or pd.isna(site_value):
        return "Env4"
    site = int(site_value)
    if site == 0:
        return "Env1"
    if site == 1:
        return "Env2"
    if site == 2:
        return "Env3"
    return "Env4"


def fold_to_split(fold: int) -> str | None:
    if 1 <= fold <= 8:
        return "train"
    if fold == 9:
        return "ValID"
    if fold == 10:
        return "Eval_ID"
    return None


def load_diagnostic_superclasses(ptbxl_root: Path) -> pd.DataFrame:
    records = pd.read_csv(ptbxl_root / "ptbxl_database.csv")
    scp_df = pd.read_csv(ptbxl_root / "scp_statements.csv", index_col=0)
    scp_df = scp_df[scp_df["diagnostic"] == 1.0]
    code_to_class = {
        str(code): row["diagnostic_class"]
        for code, row in scp_df.iterrows()
        if isinstance(row["diagnostic_class"], str)
    }

    def aggregate_classes(raw_codes: str) -> list[str]:
        codes = ast.literal_eval(raw_codes)
        return sorted({code_to_class[c] for c in codes if c in code_to_class})

    records["diagnostic_superclasses"] = records["scp_codes"].apply(aggregate_classes)
    return records


def classify_record(classes: list[str]) -> tuple[bool, int | None, str | None]:
    if len(classes) == 1 and classes[0] in KNOWN_CLASSES:
        superclass = classes[0]
        return True, LABEL_MAP[superclass], superclass
    return False, None, None


def resolve_lead_index(sig_names: list[str], want: str) -> int | None:
    normalized = normalize_lead(want)
    if normalized in sig_names:
        return sig_names.index(normalized)
    aliases = {
        "AVR": ["AVR", "A VR"],
        "AVL": ["AVL", "A VL"],
        "AVF": ["AVF", "A VF"],
    }
    for variant in aliases.get(normalized, []):
        variant_n = normalize_lead(variant)
        if variant_n in sig_names:
            return sig_names.index(variant_n)
    return None


def load_record_signal(ptbxl_root: Path, filename_lr: str) -> tuple[np.ndarray, dict, dict[str, int]]:
    signal, meta = wfdb.rdsamp(str(ptbxl_root / filename_lr))
    sig_names = [normalize_lead(name) for name in meta["sig_name"]]
    lead_to_idx = {name: idx for idx, name in enumerate(sig_names)}
    return signal, meta, lead_to_idx


def write_record(
    row: pd.Series,
    ptbxl_root: Path,
    out_root: Path,
    leads: list[str],
) -> tuple[bool, dict]:
    is_known, label, label_name = classify_record(row["diagnostic_superclasses"])
    if not is_known or label is None or label_name is None:
        return False, {}

    split_name = fold_to_split(int(row["strat_fold"]))
    if split_name is None:
        return False, {}

    env_name = site_to_env(row["site"])
    try:
        signal, meta, lead_to_idx = load_record_signal(ptbxl_root, str(row["filename_lr"]))
    except Exception:
        return False, {}

    missing = [lead for lead in leads if resolve_lead_index(list(lead_to_idx), lead) is None]
    if missing:
        return False, {}

    fs = int(meta["fs"])
    n_samples = int(signal.shape[0])
    lead_data: dict[str, np.ndarray] = {}
    for lead in leads:
        idx = resolve_lead_index(list(lead_to_idx), lead)
        if idx is None:
            return False, {}
        values = np.asarray(signal[:, idx], dtype=float)
        if values.shape[0] != n_samples:
            return False, {}
        lead_data[lead] = values

    ref_ii = lead_data["II"]
    peaks = detect_r_peaks(ref_ii, fs=fs)
    if len(peaks) < 3:
        return False, {}

    time = np.arange(n_samples, dtype=float) / float(fs)
    peak_col = np.zeros(n_samples, dtype=int)
    peak_col[peaks] = 3
    out = {
        "Time": time,
        "Voltage": ref_ii,
        "Peak": peak_col,
    }
    for lead in leads:
        out[lead_col(lead)] = lead_data[lead]

    split_dir = env_name if split_name == "train" else split_name
    out_dir = out_root / split_dir / "csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    file_stub = f"ptbxl_{int(row['ecg_id']):05d}_{label}_{env_name}_{split_name}"
    pd.DataFrame(out).to_csv(out_dir / f"{file_stub}.csv", index=False)

    info = {
        "split": split_name,
        "env": env_name,
        "label": int(label),
        "label_name": label_name,
        "patient_id": None if pd.isna(row["patient_id"]) else int(row["patient_id"]),
        "site": None if pd.isna(row["site"]) else int(row["site"]),
    }
    return True, info


def build_meta(selected: list[dict], out_root: Path, ptbxl_root: Path, leads: list[str], failed: int) -> dict:
    train_counter: dict[str, Counter] = defaultdict(Counter)
    val_counter = Counter()
    eval_counter = Counter()
    patient_by_split: dict[str, set[int]] = defaultdict(set)

    for item in selected:
        if item["patient_id"] is not None:
            patient_by_split[item["split"]].add(item["patient_id"])
        if item["split"] == "train":
            train_counter[item["env"]][str(item["label"])] += 1
        elif item["split"] == "ValID":
            val_counter[str(item["label"])] += 1
        elif item["split"] == "Eval_ID":
            eval_counter[str(item["label"])] += 1

    return {
        "source": {
            "ptbxl_root": str(ptbxl_root),
            "sampling_rate": 100,
        },
        "leads": leads,
        "n_lead_blocks": len(leads),
        "label_map": LABEL_MAP,
        "env_definition": {
            "Env1": "site == 0",
            "Env2": "site == 1",
            "Env3": "site == 2",
            "Env4": "site >= 3 or site missing",
        },
        "counts": {
            "written": len(selected),
            "failed": failed,
            "train_by_env_label": {env: dict(counter) for env, counter in train_counter.items()},
            "val_by_label": dict(val_counter),
            "eval_id_by_label": dict(eval_counter),
        },
        "patient_overlap": {
            "train_vs_val": len(patient_by_split["train"] & patient_by_split["ValID"]),
            "train_vs_eval_id": len(patient_by_split["train"] & patient_by_split["Eval_ID"]),
            "val_vs_eval_id": len(patient_by_split["ValID"] & patient_by_split["Eval_ID"]),
        },
        "output_root": str(out_root),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build PTB-XL 12-lead Phase 2 CSV splits directly from raw PTB-XL.")
    parser.add_argument(
        "--ptbxl-root",
        type=Path,
        default=Path("Data") / "PTBXL" / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1",
    )
    parser.add_argument("--out-root", type=Path, default=Path("Data") / "Generated_PTBXL_12Lead")
    parser.add_argument("--leads", nargs="*", default=LEADS_12)
    parser.add_argument("--max-records", type=int, default=None)
    args = parser.parse_args()

    leads = [normalize_lead(lead) for lead in args.leads]
    if "II" not in leads:
        raise ValueError("The lead set must include II for R-peak detection and Voltage output.")
    if len(set(leads)) != len(leads):
        raise ValueError("Lead names must be unique.")

    args.out_root.mkdir(parents=True, exist_ok=True)
    records = load_diagnostic_superclasses(args.ptbxl_root)
    selected: list[dict] = []
    failed = 0

    for idx, row in records.iterrows():
        if args.max_records is not None and int(idx) >= int(args.max_records):
            break
        ok, info = write_record(row, args.ptbxl_root, args.out_root, leads)
        if ok:
            selected.append(info)
        elif fold_to_split(int(row["strat_fold"])) is not None:
            failed += 1
        if (idx + 1) % 1000 == 0:
            print(f"processed={idx + 1} written={len(selected)} failed={failed}")

    meta = build_meta(selected, args.out_root, args.ptbxl_root, leads, failed)
    (args.out_root / "multilead_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
