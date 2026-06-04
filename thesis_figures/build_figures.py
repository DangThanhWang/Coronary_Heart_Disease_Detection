from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
OUT = Path(__file__).resolve().parent / "output"
THESIS_FIGS = ROOT / "Thesis" / "figures"

PHASE1 = ARTIFACTS / "phase1"
PHASE2 = ARTIFACTS / "phase2"

GENERATED_FIGURE_PATTERNS = [
    "figure_phase1_rule_stability.png",
    "figure_phase1_summary.png",
    "figure_phase1_zalizadeh_score.png",
    "figure_phase2_cf_comparison.png",
    "figure_phase2_external_validation.png",
    "figure_phase2_preqrs_negative_control.png",
    "figure_waveform_*.png",
    "figure_validation_layers.png",
]

PALETTE = {
    "blue": "#2f6db2",
    "teal": "#2f8f83",
    "orange": "#db9230",
    "red": "#c94d4d",
    "pink": "#cc4f79",
    "green": "#5a944f",
    "gray": "#7c8794",
    "dark": "#263445",
    "light": "#eef2f6",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def clean_generated_outputs() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for root in [OUT, THESIS_FIGS]:
        for pattern in GENERATED_FIGURE_PATTERNS:
            for path in root.glob(pattern):
                path.unlink()


def keep_static_pipeline_figures() -> None:
    for name in ["figure_phase1_pipeline.png", "figure_phase2_pipeline.png"]:
        src = THESIS_FIGS / name
        dst = OUT / name
        if not src.exists():
            raise FileNotFoundError(f"Missing static pipeline figure: {src}")
        if not dst.exists():
            shutil.copy2(src, dst)


def clean_optional_archived_figures() -> None:
    for root in [OUT, THESIS_FIGS]:
        for path in root.glob("figure_phase2_mechanism_map.png"):
            path.unlink()


def copy_phase1_figures() -> list[Path]:
    copied: list[Path] = []
    sources = [
        (PHASE1 / "thesis_assets" / "figure_phase1_summary.png", "figure_phase1_summary.png"),
        (PHASE1 / "Z_Alizadeh" / "figure_cad_rate_by_oof_score.png", "figure_phase1_zalizadeh_score.png"),
    ]
    for src, name in sources:
        if not src.exists():
            raise FileNotFoundError(f"Missing required Phase 1 figure source: {src}")
        dst = OUT / name
        shutil.copy2(src, dst)
        copied.append(dst)
    return copied


def save_phase1_rule_stability() -> Path:
    out_path = OUT / "figure_phase1_rule_stability.png"
    sources = {
        "Z-Alizadeh": PHASE1 / "Z_Alizadeh" / "stable_rules_protective.csv",
        "Cleveland": PHASE1 / "UCI_Cleveland" / "stable_rules_protective.csv",
        "Hungarian": PHASE1 / "UCI_Hungarian" / "stable_rules_protective.csv",
    }
    if not all(p.exists() for p in sources.values()):
        print(f"  SKIP figure_phase1_rule_stability (artifacts/phase1 not available)")
        return out_path

    def nice(text: str) -> str:
        mapping = {
            "EF-TTE_bin": "EF-TTE", "Typical Chest Pain": "Typical pain",
            "Atypical": "Atypical pain", "HTN": "No HTN", "Age_bin": "Age",
            "CP": "Chest pain", "Ca": "CA", "Oldpeak_bin": "Oldpeak",
            "Sex": "Sex", "Trestbps_bin": "Resting BP", "Slope": "ST slope",
            "Thal": "Thal", "Exang": "Exercise angina", "Thalach_bin": "Max HR",
            "Chol_bin": "Cholesterol",
        }
        parts = [mapping.get(part.strip(), part.strip()) for part in text.split(" | ")]
        return " + ".join(parts)

    fig, axes = plt.subplots(1, 3, figsize=(13.8, 5.1), dpi=220, sharex=True)
    for ax, (dataset, path) in zip(axes, sources.items()):
        df = pd.read_csv(path)
        df = df.sort_values(["fold_rate", "mean_lift"], ascending=[False, False]).head(5).copy()
        df["label"] = df["features"].apply(nice)
        df = df.sort_values("fold_rate")
        ax.barh(df["label"], df["fold_rate"], color=PALETTE["green"])
        for y, value in enumerate(df["fold_rate"]):
            ax.text(value + 0.02, y, f"{value:.1f}", va="center", fontsize=8)
        ax.set_title(dataset, fontsize=11, fontweight="bold")
        ax.set_xlim(0, 1.12)
        ax.grid(axis="x", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="y", labelsize=8)
    axes[0].set_ylabel("Stable protective combination", fontsize=10)
    fig.supxlabel("Selection rate across CV folds", fontsize=11)
    fig.suptitle("Phase 1 rule stability: top protective combinations repeated across folds",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Phase 2 figures
# ---------------------------------------------------------------------------

def _load_waveform_summaries() -> pd.DataFrame:
    rows = []
    for dataset, path in [
        ("PTB-XL", PHASE2 / "waveform_counterfactual_sttc" / "waveform_counterfactual_summary.csv"),
        ("Georgia", PHASE2 / "georgia_waveform_counterfactual_sttc_full" / "georgia_waveform_counterfactual_summary.csv"),
    ]:
        df = pd.read_csv(path)
        df.insert(0, "dataset", dataset)
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def save_phase2_dose_response() -> Path:
    out_path = OUT / "figure_waveform_dose_response.png"
    summary = _load_waveform_summaries()
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
    ax.set_ylim(-0.03, 0.70)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def save_phase2_waveform_controls() -> Path:
    out_path = OUT / "figure_waveform_controls.png"
    summary = _load_waveform_summaries()
    alpha1 = summary[summary["alpha"] == 1.0].copy()
    keep = ["st_t", "st_only", "t_only", "qrs_control", "p_pr_control", "pre_qrs_control"]
    alpha1 = alpha1[alpha1["window"].isin(keep)]
    order = {name: i for i, name in enumerate(keep)}
    alpha1["order"] = alpha1["window"].map(order)
    alpha1 = alpha1.sort_values(["dataset", "order"])

    colors = {
        "st_t": "#1f77b4", "st_only": "#5aa3d8", "t_only": "#9ecae1",
        "qrs_control": "#bdbdbd", "p_pr_control": "#969696", "pre_qrs_control": "#d62728",
    }
    labels = {
        "st_t": "ST/T", "st_only": "ST only", "t_only": "T only",
        "qrs_control": "QRS ctrl", "p_pr_control": "P/PR ctrl", "pre_qrs_control": "Pre-QRS ctrl",
    }
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), dpi=180, sharey=True)
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
    return out_path


def save_phase2_preqrs_negative_control() -> Path:
    out_path = OUT / "figure_phase2_preqrs_negative_control.png"
    summary = _load_waveform_summaries()
    alpha1 = summary[summary["alpha"] == 1.0]

    def get(ds, win, col="mean_prob_drop"):
        row = alpha1[(alpha1["dataset"] == ds) & (alpha1["window"] == win)]
        return float(row[col].iloc[0])

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.8), dpi=220, sharey=True)
    for ax, title in [(axes[0], "PTB-XL"), (axes[1], "Georgia")]:
        target = get(title, "st_t")
        preqrs = get(title, "pre_qrs_control")
        bars = ax.bar(["ST/T target", "pre-QRS control"], [target, preqrs],
                      color=[PALETTE["blue"], PALETTE["pink"]], width=0.55)
        ax.axhline(0, color=PALETTE["dark"], linewidth=1.0)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(-0.14, 0.72)
        for bar, value in zip(bars, [target, preqrs]):
            if value >= 0:
                ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02,
                        f"{value:+.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
            else:
                ax.text(bar.get_x() + bar.get_width() / 2, value - 0.02,
                        f"{value:+.3f}", ha="center", va="top", fontsize=10, fontweight="bold")
        ax.annotate(
            "sign reversal",
            xy=(1, preqrs), xytext=(0.45, 0.74), textcoords="axes fraction",
            arrowprops=dict(arrowstyle="->", color=PALETTE["pink"], lw=1.5),
            color=PALETTE["pink"], fontsize=9, fontweight="bold",
        )
    axes[0].set_ylabel("Mean probability drop", fontsize=11)
    fig.suptitle("Waveform negative control: pre-QRS reverses the effect on both datasets",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_phase2_validation_layers() -> Path:
    out_path = OUT / "figure_validation_layers.png"

    ptbxl_report = read_json(PHASE2 / "ptbxl_final" / "ptbxl_final_report.json")
    georgia_report = read_json(PHASE2 / "georgia_counterfactual_controls" / "georgia_counterfactual_controls_report.json")
    wave_summary = _load_waveform_summaries()
    wave_alpha1 = wave_summary[wave_summary["alpha"] == 1.0]

    def wave_drop(ds):
        row = wave_alpha1[(wave_alpha1["dataset"] == ds) & (wave_alpha1["window"] == "st_t")]
        return float(row["mean_prob_drop"].iloc[0])

    feat_ptbxl = ptbxl_report["counterfactual"]["target_summary"]["mean_prob_drop"]["mean"]
    feat_georgia = georgia_report["target"]["mean_prob_drop"]
    wave_ptbxl = wave_drop("PTB-XL")
    wave_georgia = wave_drop("Georgia")

    datasets = ["PTB-XL", "Georgia"]
    layers = ["Feature-level", "Waveform-level"]
    vals = {
        "Feature-level": [feat_ptbxl, feat_georgia],
        "Waveform-level": [wave_ptbxl, wave_georgia],
    }
    width = 0.34
    x = np.arange(len(datasets))

    fig, ax = plt.subplots(figsize=(7.4, 4.4), dpi=180)
    for j, layer in enumerate(layers):
        ax.bar([i + (j - 0.5) * width for i in x], vals[layer], width=width, label=layer)
    ax.set_xticks(list(x), datasets)
    ax.set_ylabel("Target mean probability drop")
    ax.set_title("Target Counterfactual Drop Across Validation Layers")
    ax.set_ylim(0, 0.74)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def save_phase2_external_validation() -> Path:
    out_path = OUT / "figure_phase2_external_validation.png"
    ptbxl_report = read_json(PHASE2 / "ptbxl_final" / "ptbxl_final_report.json")
    strict_report = read_json(PHASE2 / "external_georgia_full_strict_normal" / "georgia_validation_report.json")
    hard_report = read_json(PHASE2 / "external_georgia_full_no_st_t" / "georgia_validation_report.json")

    rows = [
        {
            "dataset": "PTB-XL\ninternal",
            "n": int(ptbxl_report["data"]["test_n"]),
            **ptbxl_report["classification"]["test_metrics"],
        },
        {
            "dataset": "Georgia\nstrict normal",
            "n": int(strict_report["data"]["n_extracted"]),
            **strict_report["georgia_external"]["metrics_at_ptbxl_threshold"],
        },
        {
            "dataset": "Georgia\nhard negative",
            "n": int(hard_report["data"]["n_extracted"]),
            **hard_report["georgia_external"]["metrics_at_ptbxl_threshold"],
        },
    ]
    df = pd.DataFrame(rows)
    metrics = [
        ("auc", "ROC-AUC", PALETTE["blue"]),
        ("auprc", "PR-AUC", PALETTE["teal"]),
        ("balanced_accuracy", "Balanced acc.", PALETTE["orange"]),
    ]

    fig, ax = plt.subplots(figsize=(8.6, 4.8), dpi=220)
    x = np.arange(len(df))
    width = 0.24
    for idx, (col, label, color) in enumerate(metrics):
        offsets = x + (idx - 1) * width
        bars = ax.bar(offsets, df[col].to_numpy(dtype=float), width=width, color=color, label=label)
        for bar, value in zip(bars, df[col].to_numpy(dtype=float)):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.012,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=0,
            )

    ax.set_xticks(x, [f"{row.dataset}\n(n={row.n:,})" for row in df.itertuples(index=False)])
    ax.set_ylim(0.60, 1.02)
    ax.set_ylabel("Score")
    ax.set_title("Phase 2 classification transfer: PTB-XL internal vs Georgia validation")
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.17))
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_phase2_cf_comparison() -> Path:
    out_path = OUT / "figure_phase2_cf_comparison.png"
    ctrl_path = PHASE2 / "ptbxl_final" / "negative_control_summary.csv"
    report_path = PHASE2 / "ptbxl_final" / "ptbxl_final_report.json"
    if not ctrl_path.exists() or not report_path.exists():
        print(f"  SKIP figure_phase2_cf_comparison (missing {ctrl_path})")
        return out_path

    controls = pd.read_csv(ctrl_path)
    report = read_json(report_path)

    target_drop = report["counterfactual"]["target_summary"]["mean_prob_drop"]["mean"]
    target_ci_lo = report["counterfactual"]["target_summary"]["mean_prob_drop"]["ci95_low"]
    target_ci_hi = report["counterfactual"]["target_summary"]["mean_prob_drop"]["ci95_high"]
    target_n_feat = int(report["counterfactual"].get("target_n_features", 247))

    top_ctrl = controls.head(8).copy()
    combos = [r"st\_segment" + "\n" + r"+t\_wave" + "\n(target)"] + [
        c.replace("+", "\n+") for c in top_ctrl["combo"].tolist()
    ]
    drops = [target_drop] + top_ctrl["mean_prob_drop"].tolist()
    n_feats = [target_n_feat] + top_ctrl["n_features"].tolist()
    colors = [PALETTE["red"]] + [
        PALETTE["orange"] if row["combo"] == "global_morph+beat_baseline" else PALETTE["gray"]
        for _, row in top_ctrl.iterrows()
    ]

    fig, ax = plt.subplots(figsize=(11, 6), dpi=220)
    y = np.arange(len(combos))[::-1]
    bars = ax.barh(y, drops, color=colors, height=0.65, zorder=3)

    for bar, v, nf in zip(bars, drops, n_feats):
        ax.text(v + 0.008, bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}  ({nf} feat)", va="center", fontsize=8.5, fontweight="bold")

    ax.errorbar(target_drop, y[0],
                xerr=[[target_drop - target_ci_lo], [target_ci_hi - target_drop]],
                fmt="none", color="black", capsize=5, linewidth=1.5, zorder=5)

    ax.set_yticks(y)
    ax.set_yticklabels(combos, fontsize=8.5)
    ax.set_xlabel("Mean probability drop (STTC → lower)", fontsize=11)
    ax.set_xlim(0, max(drops) * 1.22)
    ax.grid(axis="x", alpha=0.25, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    best_ctrl_drop = top_ctrl.iloc[0]["mean_prob_drop"]
    best_ctrl_nfeat = int(top_ctrl.iloc[0]["n_features"])
    ratio = target_drop / max(best_ctrl_drop, 1e-6)
    ax.axvline(best_ctrl_drop, color=PALETTE["orange"], linewidth=1.2, linestyle="--", alpha=0.7)
    ax.text(best_ctrl_drop + 0.005, y[-1] - 0.6,
            f"Best control\n({best_ctrl_nfeat} feat)", color=PALETTE["orange"], fontsize=8)

    target_p = mpatches.Patch(color=PALETTE["red"], label=f"Target: st_segment+t_wave ({target_n_feat} feat)")
    ctrl_p = mpatches.Patch(color=PALETTE["orange"], label=f"Best control: {top_ctrl.iloc[0]['combo']} ({best_ctrl_nfeat} feat)")
    other_p = mpatches.Patch(color=PALETTE["gray"], label="Other controls")
    ax.legend(handles=[target_p, ctrl_p, other_p], fontsize=8.5, loc="lower right")

    ax.set_title(
        f"Feature-level counterfactual: ST/T target group vs all control groups (PTB-XL, n=213)\n"
        f"Target ({target_n_feat} feat) → {target_drop:.3f} drop  ·  "
        f"Best control ({best_ctrl_nfeat} feat) → {best_ctrl_drop:.3f} drop  ·  Ratio {ratio:.1f}×",
        fontsize=10.5, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def copy_to_thesis(names: list[str]) -> None:
    THESIS_FIGS.mkdir(parents=True, exist_ok=True)
    for name in names:
        src = OUT / name
        if src.exists():
            shutil.copy2(src, THESIS_FIGS / name)
            print(f"  copied {name} -> Thesis/figures/")
        else:
            print(f"  SKIP copy {name} (not generated)")


def main() -> None:
    clean_generated_outputs()
    clean_optional_archived_figures()
    keep_static_pipeline_figures()

    generated: list[str] = []

    generated.append(save_phase1_rule_stability().name)
    generated.extend(p.name for p in copy_phase1_figures())

    phase2_data_fns = [
        save_phase2_external_validation,
        save_phase2_dose_response,
        save_phase2_waveform_controls,
        save_phase2_preqrs_negative_control,
        save_phase2_validation_layers,
        save_phase2_cf_comparison,
    ]
    for fn in phase2_data_fns:
        p = fn()
        generated.append(p.name)

    copy_to_thesis(generated)

    print(json.dumps({"output_dir": str(OUT), "files": generated}, indent=2))


if __name__ == "__main__":
    main()
