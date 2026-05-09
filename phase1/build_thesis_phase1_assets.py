from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def build_phase1_summary_figure(summary_csv: Path, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    df = pd.read_csv(summary_csv)
    datasets = df["dataset"].tolist()
    x = np.arange(len(datasets))
    width = 0.34

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    ax = axes[0]
    low = df["cad_rate_low_score"].to_numpy() * 100.0
    high = df["cad_rate_high_score"].to_numpy() * 100.0
    ax.bar(x - width / 2, low, width, label="Nhóm điểm thấp", color="#b23a48")
    ax.bar(x + width / 2, high, width, label="Nhóm điểm cao", color="#2a9d8f")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("Tỷ lệ CAD (%)")
    ax.set_title("Tỷ lệ CAD theo nhóm protective score")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    for idx, value in enumerate(low):
        ax.text(idx - width / 2, value + 1.2, f"{value:.1f}", ha="center", va="bottom", fontsize=9)
    for idx, value in enumerate(high):
        ax.text(idx + width / 2, value + 1.2, f"{value:.1f}", ha="center", va="bottom", fontsize=9)

    ax = axes[1]
    baseline = df["baseline_auc"].to_numpy()
    hybrid = df["hybrid_auc"].to_numpy()
    ax.bar(x - width / 2, baseline, width, label="Logistic baseline", color="#577590")
    ax.bar(x + width / 2, hybrid, width, label="Hybrid logistic + score", color="#f4a261")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylim(0.84, 0.96)
    ax.set_ylabel("ROC-AUC")
    ax.set_title("So sánh baseline và hybrid")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    for idx, value in enumerate(baseline):
        ax.text(idx - width / 2, value + 0.002, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    for idx, value in enumerate(hybrid):
        ax.text(idx + width / 2, value + 0.002, f"{value:.3f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1-root", default="artifacts/phase1")
    parser.add_argument("--out-dir", default="artifacts/phase1/thesis_assets")
    args = parser.parse_args()

    phase1_root = Path(args.phase1_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    build_phase1_summary_figure(
        phase1_root / "summary.csv",
        out_dir / "figure_phase1_summary.png",
    )


if __name__ == "__main__":
    main()
