import argparse
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import wfdb


LABEL_PATTERN = re.compile(r"_(\d)_")


def map_symbol(symbol: str) -> int | None:
    # MIT-BIH beat symbol mapping (AAMI-inspired)
    # 0: Normal (N, L, R, e, j)
    # 1: Ventricular ectopic (V)
    # 2: Supraventricular (A, a, J, S)
    # 3: Fusion (F)
    if symbol in {"N", "L", "R", "e", "j"}:
        return 0
    if symbol in {"V"}:
        return 1
    if symbol in {"A", "a", "J", "S"}:
        return 2
    if symbol in {"F"}:
        return 3
    return None


def iter_records(root: Path) -> Iterable[str]:
    for dat in sorted(root.glob("*.dat")):
        yield dat.stem


def assign_env(record: str, envs: List[str]) -> str:
    # Deterministic split by record id
    try:
        rec_id = int(record)
    except ValueError:
        rec_id = sum(ord(c) for c in record)
    return envs[rec_id % len(envs)]


def build_windows(
    signal: np.ndarray,
    fs: int,
    r_peaks: np.ndarray,
    r_labels: List[int],
    window_seconds: int,
    min_beats: int,
    min_majority: float,
) -> List[Tuple[int, int, int]]:
    window_len = window_seconds * fs
    n_samples = signal.shape[0]
    windows = []
    for start in range(0, n_samples - window_len + 1, window_len):
        end = start + window_len
        mask = (r_peaks >= start) & (r_peaks < end)
        if not np.any(mask):
            continue
        labels = [r_labels[i] for i in np.where(mask)[0]]
        if len(labels) < min_beats:
            continue
        counts: Dict[int, int] = {}
        for lab in labels:
            counts[lab] = counts.get(lab, 0) + 1
        major_label, major_count = max(counts.items(), key=lambda x: x[1])
        if major_count / len(labels) < min_majority:
            continue
        windows.append((start, end, major_label))
    return windows


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare MIT-BIH windows as CSV for LLCS/XAI.")
    parser.add_argument("--mitdb-root", type=Path, default=Path("Data") / "mitdb" / "mit-bih-arrhythmia-database-1.0.0")
    parser.add_argument("--out-root", type=Path, default=Path("Data") / "mitbih_generated")
    parser.add_argument("--window-seconds", type=int, default=120)
    parser.add_argument("--mode", choices=["window", "beat"], default="window")
    parser.add_argument("--beat-seconds", type=float, default=3.0)
    parser.add_argument("--max-per-label", type=int, default=0)
    parser.add_argument("--min-beats", type=int, default=10)
    parser.add_argument("--min-majority", type=float, default=0.6)
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--max-windows-per-record", type=int, default=0)
    parser.add_argument("--eval-copy", action="store_true", help="Also copy windows into Eval/csv.")
    args = parser.parse_args()

    envs = ["Env1", "Env2", "Env3", "Env4", "Eval"]
    out_root = args.out_root
    for env in envs:
        (out_root / env / "csv").mkdir(parents=True, exist_ok=True)

    records = list(iter_records(args.mitdb_root))
    if args.max_records > 0:
        records = records[: args.max_records]

    total_csv = 0
    for rec in records:
        rec_path = args.mitdb_root / rec
        record = wfdb.rdrecord(str(rec_path))
        ann = wfdb.rdann(str(rec_path), "atr")
        fs = int(record.fs)
        signal = record.p_signal[:, args.channel]

        # Map annotations to labels
        mapped = []
        for idx, sym in zip(ann.sample, ann.symbol):
            lab = map_symbol(sym)
            if lab is None:
                continue
            mapped.append((idx, lab))
        if not mapped:
            continue
        r_peaks = np.array([m[0] for m in mapped], dtype=int)
        r_labels = [m[1] for m in mapped]

        env = assign_env(rec, envs[:-1])
        eval_env = "Eval"

        if args.mode == "window":
            windows = build_windows(
                signal,
                fs,
                r_peaks,
                r_labels,
                args.window_seconds,
                args.min_beats,
                args.min_majority,
            )
            if not windows:
                continue

            if args.max_windows_per_record > 0:
                windows = windows[: args.max_windows_per_record]

            for w_idx, (start, end, label) in enumerate(windows):
                t = np.arange(end - start) / fs
                v = signal[start:end]
                peaks = np.zeros_like(v, dtype=int)
                in_win = (r_peaks >= start) & (r_peaks < end)
                for rp in r_peaks[in_win]:
                    peaks[rp - start] = 3

                df = pd.DataFrame({"Time": t, "Voltage": v, "Peak": peaks})
                name = f"{rec}_{start:06d}_{label}_w{w_idx:04d}.csv"
                df.to_csv(out_root / env / "csv" / name, index=False)
                total_csv += 1
                if args.eval_copy:
                    df.to_csv(out_root / eval_env / "csv" / name, index=False)
                    total_csv += 1
        else:
            # Beat-centered windows labeled by the beat symbol.
            half = int((args.beat_seconds * fs) / 2)
            per_label = {}
            for b_idx, (rp, lab) in enumerate(zip(r_peaks, r_labels)):
                if args.max_per_label > 0:
                    cnt = per_label.get(lab, 0)
                    if cnt >= args.max_per_label:
                        continue
                start = rp - half
                end = rp + half
                if start < 0 or end >= signal.shape[0]:
                    continue
                t = np.arange(end - start) / fs
                v = signal[start:end]
                peaks = np.zeros_like(v, dtype=int)
                peaks[rp - start] = 3
                df = pd.DataFrame({"Time": t, "Voltage": v, "Peak": peaks})
                name = f"{rec}_{rp:06d}_{lab}_b{b_idx:04d}.csv"
                df.to_csv(out_root / env / "csv" / name, index=False)
                total_csv += 1
                if args.eval_copy:
                    df.to_csv(out_root / eval_env / "csv" / name, index=False)
                    total_csv += 1
                per_label[lab] = per_label.get(lab, 0) + 1

    print(f"MIT-BIH windows exported: {total_csv} CSVs")


if __name__ == "__main__":
    main()
