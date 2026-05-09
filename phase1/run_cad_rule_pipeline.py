from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from phase1.cad_tabular_config import (
    BIN_UCI,
    BIN_Z,
    CLEVELAND_URL,
    HUNGARIAN_URL,
    RuleParams,
)
from phase1.cad_tabular_data import (
    load_uci_heart,
    load_z_alizadeh,
    load_z_alizadeh_from_uci,
)
from phase1.cad_rule_pipeline import (
    run_dataset_analysis,
    save_dataset_result,
    summary_row,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 1: leakage-free CAD protective rule-XAI on tabular CAD datasets."
    )
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts") / "phase1")
    parser.add_argument("--cache-dir", type=Path, default=Path("Data") / "phase1_cache")
    parser.add_argument("--z-path", type=Path, default=None, help="Optional local Z-Alizadeh CSV/XLSX.")
    parser.add_argument("--skip-z", action="store_true", help="Skip Z-Alizadeh.")
    parser.add_argument("--skip-cleveland", action="store_true")
    parser.add_argument("--skip-hungarian", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--min-support", type=float, default=0.06)
    parser.add_argument("--min-confidence", type=float, default=0.60)
    parser.add_argument("--min-lift", type=float, default=1.40)
    parser.add_argument("--max-len", type=int, default=3)
    parser.add_argument("--top-n-rules", type=int, default=10)
    parser.add_argument("--threshold", type=int, default=2)
    parser.add_argument("--n-splits", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rp = RuleParams(
        min_support=args.min_support,
        min_confidence=args.min_confidence,
        min_lift=args.min_lift,
        max_len=args.max_len,
        top_n_rules=args.top_n_rules,
        threshold=args.threshold,
        n_splits=args.n_splits,
    )

    datasets = []
    if not args.skip_z:
        if args.z_path:
            datasets.append(("Z_Alizadeh", load_z_alizadeh(args.z_path), BIN_Z))
        else:
            datasets.append(("Z_Alizadeh", load_z_alizadeh_from_uci(args.cache_dir), BIN_Z))
    if not args.skip_cleveland:
        datasets.append(("UCI_Cleveland", load_uci_heart(CLEVELAND_URL), BIN_UCI))
    if not args.skip_hungarian:
        datasets.append(("UCI_Hungarian", load_uci_heart(HUNGARIAN_URL), BIN_UCI))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for name, df, bin_spec in datasets:
        print(f"\n[phase1] Running {name}: n={len(df)}, CAD prevalence={df['Cath'].mean():.3f}")
        result = run_dataset_analysis(name, df, bin_spec, rp, make_plots=not args.no_plots)
        save_dataset_result(result, args.out_dir / name, make_plots=not args.no_plots)
        row = summary_row(result["report"])
        summary_rows.append(row)
        print(json.dumps(row, indent=2))

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.out_dir / "summary.csv", index=False)
    with (args.out_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary_rows, fh, indent=2)

    print(f"\n[phase1] Saved summary: {args.out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()

