from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb


RECORD_RE = re.compile(r"ptbxl_(\d{5})_(\d)_([^_]+)_(.+)\.csv$")
LEADS_12 = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def normalize_lead(name: str) -> str:
    return str(name).upper().replace(" ", "")


def lead_col(lead: str) -> str:
    return f"Lead_{lead.replace('-', '_')}"


def find_singlelead_csvs(root: Path):
    for split in ["Env1", "Env2", "Env3", "Env4", "ValID", "Eval_ID", "Eval_OOD"]:
        csv_dir = root / split / "csv"
        if not csv_dir.exists():
            continue
        for path in sorted(csv_dir.glob("*.csv")):
            yield split, path


def load_ptbxl_index(ptbxl_root: Path) -> dict[int, str]:
    db = pd.read_csv(ptbxl_root / "ptbxl_database.csv")
    return {int(row.ecg_id): str(row.filename_lr) for _, row in db.iterrows()}


def read_signal(ptbxl_root: Path, filename_lr: str) -> tuple[np.ndarray, dict, dict[str, int]]:
    sig, meta = wfdb.rdsamp(str(ptbxl_root / filename_lr))
    names = [normalize_lead(n) for n in meta["sig_name"]]
    idx = {n: i for i, n in enumerate(names)}
    return sig, meta, idx


def convert_one(
    path: Path,
    split: str,
    out_root: Path,
    ptbxl_root: Path,
    ecg_to_filename: dict[int, str],
    leads: list[str],
) -> bool:
    m = RECORD_RE.match(path.name)
    if not m:
        return False
    ecg_id = int(m.group(1))
    filename_lr = ecg_to_filename.get(ecg_id)
    if filename_lr is None:
        return False
    try:
        sig, meta, idx = read_signal(ptbxl_root, filename_lr)
    except Exception:
        return False
    missing = [ld for ld in leads if normalize_lead(ld) not in idx]
    if missing:
        return False
    df = pd.read_csv(path)
    n = len(df)
    if sig.shape[0] != n:
        # Generated CSV and WFDB LR should both be 1000 samples. Skip if inconsistent.
        return False
    out = df.copy()
    for ld in leads:
        out[lead_col(ld)] = np.asarray(sig[:, idx[normalize_lead(ld)]], dtype=float)
    out_dir = out_root / split / "csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / path.name, index=False)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Create multilead generated PTB-XL CSVs from existing single-lead split.")
    parser.add_argument("--single-root", type=Path, default=Path("Data") / "Generated_PTBXL")
    parser.add_argument("--ptbxl-root", type=Path, default=Path("Data") / "PTBXL" / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1")
    parser.add_argument("--out-root", type=Path, default=Path("Data") / "Generated_PTBXL_12Lead")
    parser.add_argument("--leads", nargs="*", default=LEADS_12)
    parser.add_argument("--max-files", type=int, default=None)
    args = parser.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    ecg_to_filename = load_ptbxl_index(args.ptbxl_root)
    counts = {"written": 0, "failed": 0}
    by_split: dict[str, int] = {}
    for i, (split, path) in enumerate(find_singlelead_csvs(args.single_root)):
        if args.max_files is not None and i >= args.max_files:
            break
        ok = convert_one(path, split, args.out_root, args.ptbxl_root, ecg_to_filename, args.leads)
        if ok:
            counts["written"] += 1
            by_split[split] = by_split.get(split, 0) + 1
        else:
            counts["failed"] += 1
        if (i + 1) % 1000 == 0:
            print(f"processed={i+1} written={counts['written']} failed={counts['failed']}")
    meta = {
        "source_single_root": str(args.single_root),
        "ptbxl_root": str(args.ptbxl_root),
        "leads": args.leads,
        "n_lead_blocks": len(args.leads),
        "counts": counts,
        "by_split": by_split,
    }
    (args.out_root / "multilead_meta.json").write_text(pd.Series(meta).to_json(indent=2), encoding="utf-8")
    print(meta)


if __name__ == "__main__":
    main()

