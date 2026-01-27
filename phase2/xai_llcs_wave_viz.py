import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def iter_csv_paths(root: Path):
    for name in os.listdir(root):
        if name.endswith(".csv"):
            yield root / name


def load_ecg_template(csv_path: Path, pre: int, post: int) -> np.ndarray | None:
    df = pd.read_csv(csv_path)
    if "Voltage" not in df.columns or "Peak" not in df.columns:
        return None
    voltage = df["Voltage"].to_numpy(dtype=float)
    peaks = df.index[df["Peak"] == 3].to_numpy(dtype=int)
    if len(peaks) == 0:
        return None
    beats = []
    for idx in peaks:
        start = idx - pre
        end = idx + post
        if start < 0 or end >= len(voltage):
            continue
        beats.append(voltage[start:end])
    if not beats:
        return None
    beats = np.stack(beats, axis=0)
    return np.median(beats, axis=0)


def find_csv_by_name(root: Path, name: str) -> Path | None:
    candidate = root / name
    if candidate.exists():
        return candidate
    for path in iter_csv_paths(root):
        if path.name == name:
            return path
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Unknown vs Prototype templates for LLCS waveform XAI.")
    parser.add_argument("--csv-root", type=Path, default=Path("Data") / "Generated_HardOOD3")
    parser.add_argument(
        "--explain-csv",
        type=Path,
        default=Path("artifacts") / "phase2" / "xai_llcs_wave_unknown_explain.csv",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts") / "phase2" / "figures")
    parser.add_argument("--pre", type=int, default=80)
    parser.add_argument("--post", type=int, default=120)
    parser.add_argument("--max-unknown", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    explain = pd.read_csv(args.explain_csv)
    unknown_files = explain["unknown_file"].drop_duplicates().head(args.max_unknown).tolist()

    for unknown_file in unknown_files:
        rows = explain[explain["unknown_file"] == unknown_file].head(args.top_k)
        unknown_path = find_csv_by_name(args.csv_root / "Eval" / "csv", unknown_file)
        if unknown_path is None:
            continue
        unk_template = load_ecg_template(unknown_path, args.pre, args.post)
        if unk_template is None:
            continue

        for _, row in rows.iterrows():
            proto_env = row["proto_env"]
            proto_file = row["proto_file"]
            proto_label = int(row["proto_label"])
            proto_path = find_csv_by_name(args.csv_root / proto_env / "csv", proto_file)
            if proto_path is None:
                continue
            proto_template = load_ecg_template(proto_path, args.pre, args.post)
            if proto_template is None:
                continue

            fig, ax = plt.subplots(figsize=(6, 3))
            ax.plot(unk_template, label="Unknown", linewidth=2)
            ax.plot(proto_template, label=f"Proto L{proto_label} {proto_env}", alpha=0.8)
            ax.set_title(f"{unknown_file} vs {proto_file}")
            ax.set_xlabel("Sample")
            ax.set_ylabel("Voltage")
            ax.legend(loc="best", fontsize=8)
            ax.grid(True, alpha=0.2)
            out_name = f"{unknown_file}_vs_{proto_file}.png".replace(":", "_")
            fig.tight_layout()
            fig.savefig(out_dir / out_name, dpi=150)
            plt.close(fig)

    print(f"Saved figures to {out_dir}")


if __name__ == "__main__":
    main()
