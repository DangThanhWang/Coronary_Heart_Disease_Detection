from __future__ import annotations

from pathlib import Path

import pandas as pd


def plot_score_cad_rate(score_table: pd.DataFrame, path: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(score_table["score"], score_table["cad_rate"] * 100.0, marker="o")
    ax.set_xlabel("OOF protective score")
    ax.set_ylabel("CAD rate (%)")
    ax.set_title("CAD rate by leakage-free protective score")
    ax.grid(True, alpha=0.3)
    for _, row in score_table.iterrows():
        ax.annotate(
            str(int(row["n"])),
            (row["score"], row["cad_rate"] * 100.0),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
        )
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)

