from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


FINAL_ROOT = Path("artifacts") / "phase2"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def classification_table(focused: dict) -> pd.DataFrame:
    ptbxl = read_json(FINAL_ROOT / "ptbxl_final" / "ptbxl_final_report.json")
    rows = [
        {
            "dataset": "PTB-XL",
            "policy": "internal test",
            "n": ptbxl["data"]["test_n"],
            "auc": ptbxl["classification"]["test_metrics"]["auc"],
            "auprc": ptbxl["classification"]["test_metrics"]["auprc"],
            "balanced_accuracy": ptbxl["classification"]["test_metrics"]["balanced_accuracy"],
            "note": "Internal NORM vs STTC",
        }
    ]
    for name, item in focused["georgia_external"].items():
        rows.append(
            {
                "dataset": "Georgia",
                "policy": name,
                "n": item["n"],
                "auc": item["auc"],
                "auprc": item["auprc"],
                "balanced_accuracy": item["balanced_accuracy"],
                "note": "External threshold transfer",
            }
        )
    return pd.DataFrame(rows)


def feature_counterfactual_table(focused: dict) -> pd.DataFrame:
    rows = []
    for dataset, key in [
        ("PTB-XL", "ptbxl_counterfactual_consistency"),
        ("Georgia", "georgia_counterfactual_consistency"),
    ]:
        item = focused.get(key)
        if not item:
            continue
        rows.append(
            {
                "dataset": dataset,
                "target": item["target_combo"],
                "target_mean_drop": item["target_mean_drop"],
                "best_control": item["best_control_combo"],
                "best_control_mean_drop": item["best_control_mean_drop"],
                "target_control_gap": item["counterfactual_consistency_score"],
                "drop_ratio_vs_control": item["drop_ratio_vs_control"],
            }
        )
    return pd.DataFrame(rows)


def waveform_counterfactual_table(focused: dict) -> pd.DataFrame:
    rows = []
    wave = focused["waveform_counterfactual_consistency"]
    for dataset, key in [("PTB-XL", "ptbxl"), ("Georgia", "georgia")]:
        item = wave.get(key)
        if not item:
            continue
        rows.append(
            {
                "dataset": dataset,
                "positive_n": item["positive_n"],
                "target": item["target_window"],
                "target_mean_drop": item["target_mean_drop"],
                "target_median_drop": item["target_median_drop"],
                "target_ci95": f"[{item['target_drop_ci_low']:.3f}, {item['target_drop_ci_high']:.3f}]",
                "flip_rate": item["target_flip_rate"],
                "positive_drop_rate": item["target_positive_drop_rate"],
                "best_control": item["best_control_window"],
                "best_control_mean_drop": item["best_control_mean_drop"],
                "pre_qrs_control_mean_drop": item["pre_qrs_control_mean_drop"],
                "target_control_gap": item["target_minus_best_control_mean_drop"],
                "target_pre_qrs_gap": item["target_minus_pre_qrs_control_mean_drop"],
                "monotone_mean_drop": item["alpha_monotone_mean_drop"],
                "monotone_flip_rate": item["alpha_monotone_flip_rate"],
            }
        )
    return pd.DataFrame(rows)


def load_waveform_summaries() -> pd.DataFrame:
    rows = []
    for dataset, path in [
        ("PTB-XL", FINAL_ROOT / "waveform_counterfactual_sttc" / "waveform_counterfactual_summary.csv"),
        (
            "Georgia",
            FINAL_ROOT
            / "georgia_waveform_counterfactual_sttc_full"
            / "georgia_waveform_counterfactual_summary.csv",
        ),
    ]:
        df = pd.read_csv(path)
        df.insert(0, "dataset", dataset)
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def plot_dose_response(summary: pd.DataFrame, out_path: Path) -> None:
    st_t = summary[summary["window"] == "st_t"].copy()
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=180)
    for dataset, group in st_t.groupby("dataset"):
        ax.plot(group["alpha"], group["mean_prob_drop"], marker="o", linewidth=2.5, label=dataset)
        ax.fill_between(
            group["alpha"].to_numpy(dtype=float),
            group["prob_drop_ci95_low"].to_numpy(dtype=float),
            group["prob_drop_ci95_high"].to_numpy(dtype=float),
            alpha=0.16,
        )
    ax.set_title("ST/T Waveform Counterfactual Dose Response")
    ax.set_xlabel("Counterfactual strength alpha")
    ax.set_ylabel("Mean probability drop")
    ax.set_ylim(-0.03, 0.68)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_waveform_controls(summary: pd.DataFrame, out_path: Path) -> None:
    alpha1 = summary[summary["alpha"] == 1.0].copy()
    keep = ["st_t", "st_only", "t_only", "qrs_control", "p_pr_control", "pre_qrs_control"]
    alpha1 = alpha1[alpha1["window"].isin(keep)]
    order = {name: i for i, name in enumerate(keep)}
    alpha1["order"] = alpha1["window"].map(order)
    alpha1 = alpha1.sort_values(["dataset", "order"])

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), dpi=180, sharey=True)
    colors = {
        "st_t": "#1f77b4",
        "st_only": "#5aa3d8",
        "t_only": "#9ecae1",
        "qrs_control": "#bdbdbd",
        "p_pr_control": "#969696",
        "pre_qrs_control": "#d62728",
    }
    labels = {
        "st_t": "ST/T",
        "st_only": "ST only",
        "t_only": "T only",
        "qrs_control": "QRS ctrl",
        "p_pr_control": "P/PR ctrl",
        "pre_qrs_control": "Pre-QRS ctrl",
    }
    for ax, dataset in zip(axes, ["PTB-XL", "Georgia"]):
        group = alpha1[alpha1["dataset"] == dataset]
        xs = range(len(group))
        ax.bar(xs, group["mean_prob_drop"], color=[colors[w] for w in group["window"]])
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xticks(list(xs), [labels[w] for w in group["window"]], rotation=35, ha="right")
        ax.set_title(dataset)
        ax.grid(axis="y", alpha=0.22)
    axes[0].set_ylabel("Mean probability drop")
    fig.suptitle("Waveform Counterfactual Target vs Controls", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_validation_layers(feature_df: pd.DataFrame, waveform_df: pd.DataFrame, out_path: Path) -> None:
    rows = []
    for _, row in feature_df.iterrows():
        rows.append({"dataset": row["dataset"], "layer": "Feature-level", "mean_drop": row["target_mean_drop"]})
    for _, row in waveform_df.iterrows():
        rows.append({"dataset": row["dataset"], "layer": "Waveform-level", "mean_drop": row["target_mean_drop"]})
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(7.4, 4.4), dpi=180)
    datasets = ["PTB-XL", "Georgia"]
    layers = ["Feature-level", "Waveform-level"]
    width = 0.34
    x = range(len(datasets))
    for j, layer in enumerate(layers):
        vals = [float(df[(df["dataset"] == ds) & (df["layer"] == layer)]["mean_drop"].iloc[0]) for ds in datasets]
        ax.bar([i + (j - 0.5) * width for i in x], vals, width=width, label=layer)
    ax.set_xticks(list(x), datasets)
    ax.set_ylabel("Target mean probability drop")
    ax.set_title("Consistent Counterfactual Effect Across Validation Layers")
    ax.set_ylim(0, 0.72)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create final tables and figures for the Phase 2 ECG study.")
    parser.add_argument(
        "--focused-report",
        type=Path,
        default=FINAL_ROOT / "focused_result_summary" / "focused_result_report.json",
    )
    parser.add_argument("--out-dir", type=Path, default=FINAL_ROOT / "final_assets")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    focused = read_json(args.focused_report)
    classification = classification_table(focused)
    feature_cf = feature_counterfactual_table(focused)
    waveform_cf = waveform_counterfactual_table(focused)
    waveform_summary = load_waveform_summaries()

    classification.to_csv(args.out_dir / "table_classification.csv", index=False)
    feature_cf.to_csv(args.out_dir / "table_feature_counterfactual.csv", index=False)
    waveform_cf.to_csv(args.out_dir / "table_waveform_counterfactual.csv", index=False)
    waveform_summary.to_csv(args.out_dir / "table_waveform_dose_response_long.csv", index=False)

    plot_dose_response(waveform_summary, args.out_dir / "figure_waveform_dose_response.png")
    plot_waveform_controls(waveform_summary, args.out_dir / "figure_waveform_controls.png")
    plot_validation_layers(feature_cf, waveform_cf, args.out_dir / "figure_validation_layers.png")

    manifest = {
        "tables": {
            "classification": str(args.out_dir / "table_classification.csv"),
            "feature_counterfactual": str(args.out_dir / "table_feature_counterfactual.csv"),
            "waveform_counterfactual": str(args.out_dir / "table_waveform_counterfactual.csv"),
            "waveform_dose_response_long": str(args.out_dir / "table_waveform_dose_response_long.csv"),
        },
        "figures": {
            "waveform_dose_response": str(args.out_dir / "figure_waveform_dose_response.png"),
            "waveform_controls": str(args.out_dir / "figure_waveform_controls.png"),
            "validation_layers": str(args.out_dir / "figure_validation_layers.png"),
        },
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

