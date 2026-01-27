import argparse
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


LABEL_PATTERN = re.compile(r"_(\d)_")


def extract_label(filename):
    match = LABEL_PATTERN.search(filename)
    if not match:
        raise ValueError(f"Cannot extract label from filename: {filename}")
    return int(match.group(1))


def create_matrix(x_axis, y_axis, filename):
    grid_size = 30
    x_min, x_max = 400, 1400
    y_min, y_max = 400, 1400

    feature_matrix = np.zeros((grid_size, grid_size), dtype=float)

    x_step = (x_max - x_min) / grid_size
    y_step = (y_max - y_min) / grid_size

    for x, y in zip(x_axis, y_axis):
        if x_min <= x < x_max and y_min <= y < y_max:
            x_idx = int((x - x_min) / x_step)
            y_idx = int((y - y_min) / y_step)
            feature_matrix[y_idx, x_idx] = 1

    new_row = np.zeros((1, grid_size), dtype=float)
    # Activity flag removed to avoid leakage/shortcut learning.
    new_row[0, 0] = 0.0
    updated_array = np.append(feature_matrix, new_row, axis=0)
    return updated_array


def csv_to_npy(csv_path, npy_path, overwrite=False):
    if npy_path.exists() and not overwrite:
        return False
    ecg_data = pd.read_csv(csv_path)
    ecg_data = ecg_data[ecg_data["Time"] <= 120]
    r_peaks = ecg_data[ecg_data["Peak"] == 3]
    rr_intervals = r_peaks["Time"].diff().dropna().reset_index(drop=True)
    if rr_intervals.shape[0] < 2:
        return False
    rr_intervals_ms = rr_intervals * 1000
    rr_n_ms = rr_intervals_ms[:-1]
    rr_n1_ms = rr_intervals_ms[1:]

    input_data = create_matrix(rr_n_ms, rr_n1_ms, csv_path.name)
    np.save(npy_path, input_data)
    return True


def copy_csvs(source_dir, target_dir, labels, limit_per_label=None):
    target_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for label in sorted(labels):
        candidates = [
            p for p in sorted(source_dir.glob("*.csv"))
            if LABEL_PATTERN.search(p.name) and extract_label(p.name) == label
        ]
        if limit_per_label:
            candidates = candidates[:limit_per_label]
        for csv_path in candidates:
            dest = target_dir / csv_path.name
            if not dest.exists():
                shutil.copy2(csv_path, dest)
                copied += 1
    return copied


def summarize_labels(csv_dir):
    counts = {}
    for csv_path in csv_dir.glob("*.csv"):
        try:
            label = extract_label(csv_path.name)
        except ValueError:
            continue
        counts[label] = counts.get(label, 0) + 1
    return counts


def convert_env(csv_dir, npy_dir, overwrite=False):
    npy_dir.mkdir(parents=True, exist_ok=True)
    created = 0
    skipped = 0
    for csv_path in sorted(csv_dir.glob("*.csv")):
        npy_path = npy_dir / f"{csv_path.stem}.npy"
        ok = csv_to_npy(csv_path, npy_path, overwrite=overwrite)
        if ok:
            created += 1
        else:
            skipped += 1
    return created, skipped


def main():
    parser = argparse.ArgumentParser(
        description="Split ECG CSVs into environments and convert to numpy."
    )
    parser.add_argument(
        "--generated-root",
        type=Path,
        default=Path("Data") / "Generated",
        help="Root folder for generated CSVs (default: Data/Generated).",
    )
    parser.add_argument(
        "--disease-root",
        type=Path,
        default=Path("Data") / "Disease_dataset",
        help="Root folder for output NumpyData (default: Data/Disease_dataset).",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("Data") / "Generated" / "All" / "csv",
        help="Source CSV folder with all labels (default: Data/Generated/All/csv).",
    )
    parser.add_argument(
        "--no-split",
        action="store_true",
        help="Skip splitting and only convert existing Env*/csv folders.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .npy files.",
    )
    parser.add_argument(
        "--limit-per-label",
        type=int,
        default=0,
        help="Limit number of CSVs copied per label when splitting.",
    )
    parser.add_argument(
        "--clear-target",
        action="store_true",
        help="Clear target Env*/csv folders before splitting.",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    generated_root = args.generated_root if args.generated_root.is_absolute() else base_dir / args.generated_root
    disease_root = args.disease_root if args.disease_root.is_absolute() else base_dir / args.disease_root

    # Class mapping from report:
    # class 1 -> label 0 (Resting-Normal)
    # class 2 -> label 1 (Resting-Abnormal)
    # class 3 -> label 2 (Working-Normal)
    # class 4 -> label 3 (Working-Abnormal)
    env_labels = {
        "Env1": {0, 2, 3},
        "Env2": {0, 1, 3},
        "Env3": {0, 1, 2},
        "Env4": {0, 1, 3},
        "Eval": {0, 1, 2, 3, 4},
    }

    if not args.no_split and args.source.exists():
        limit = args.limit_per_label if args.limit_per_label > 0 else None
        for env, labels in env_labels.items():
            target = generated_root / env / "csv"
            if args.clear_target and target.exists():
                shutil.rmtree(target)
            copied = copy_csvs(args.source, target, labels, limit_per_label=limit)
            print(f"[split] {env}: copied {copied} files to {target}")
    else:
        if not args.no_split and not args.source.exists():
            print(f"[split] Source not found: {args.source}. Skipping split.")

    for env, labels in env_labels.items():
        csv_dir = generated_root / env / "csv"
        npy_dir = disease_root / env / "NumpyData"
        if not csv_dir.exists():
            print(f"[convert] {env}: missing {csv_dir}, skipping.")
            continue
        counts = summarize_labels(csv_dir)
        created, skipped = convert_env(csv_dir, npy_dir, overwrite=args.overwrite)
        print(
            f"[convert] {env}: created {created}, skipped {skipped}, labels {counts}"
        )


if __name__ == "__main__":
    main()
