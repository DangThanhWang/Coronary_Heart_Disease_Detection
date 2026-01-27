import argparse
import json
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

LABEL_PATTERN = re.compile(r"_(\d)_")


def extract_label(filename: str) -> int:
    match = LABEL_PATTERN.search(filename)
    if not match:
        return -1
    return int(match.group(1))


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


def align_baseline(vec: np.ndarray, window: int) -> np.ndarray:
    if window <= 0:
        return vec
    window = min(window, vec.shape[0])
    baseline = float(np.median(vec[:window]))
    return vec - baseline


def zscore_vector(vec: np.ndarray) -> np.ndarray:
    mean = float(np.mean(vec))
    std = float(np.std(vec))
    if std < 1e-9:
        return vec * 0.0
    return (vec - mean) / std


def build_train_matrix(csv_root: Path, envs: List[str], pre: int, post: int, baseline_window: int):
    metas = []
    vecs = []
    for env in envs:
        env_root = csv_root / env / "csv"
        for path in env_root.glob("*.csv"):
            label = extract_label(path.name)
            if label not in (0, 1, 2, 3):
                continue
            templ = load_ecg_template(path, pre, post)
            if templ is None:
                continue
            vec = zscore_vector(align_baseline(templ, baseline_window))
            metas.append({"env": env, "label": label, "filename": path.name})
            vecs.append(vec)
    if not vecs:
        raise SystemExit("No training templates found for KNN baseline.")
    return metas, np.stack(vecs, axis=0)


def knn_predict(vec: np.ndarray, matrix: np.ndarray, metas: List[dict], k: int) -> Tuple[int, List[int], List[int], List[float]]:
    d = np.linalg.norm(matrix - vec[None, :], axis=1)
    idx = np.argsort(d)[:k]
    labels = [metas[i]["label"] for i in idx]
    return labels[0], labels, idx.tolist(), d[idx].tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description="KNN baseline on ECG templates.")
    parser.add_argument("--csv-root", type=Path, default=Path("Data") / "Generated_HardOOD3")
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts") / "phase3" / "knn_baseline")
    parser.add_argument("--pre", type=int, default=80)
    parser.add_argument("--post", type=int, default=120)
    parser.add_argument("--baseline-window", type=int, default=20)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    envs = ["Env1", "Env2", "Env3", "Env4"]
    metas, matrix = build_train_matrix(args.csv_root, envs, args.pre, args.post, args.baseline_window)

    eval_root = args.csv_root / "Eval" / "csv"
    known_rows = []
    unknown_rows = []

    for path in eval_root.glob("*.csv"):
        label = extract_label(path.name)
        templ = load_ecg_template(path, args.pre, args.post)
        if templ is None:
            continue
        vec = zscore_vector(align_baseline(templ, args.baseline_window))
        pred, topk_labels, topk_idx, topk_dist = knn_predict(vec, matrix, metas, args.k)
        row = {
            "file": path.name,
            "true_label": label,
            "pred_label": pred,
            "topk_labels": "|".join(str(v) for v in topk_labels),
            "topk_distances": "|".join(f"{v:.6f}" for v in topk_dist),
            "distance_top1": float(topk_dist[0]),
            "topk_hit": int(label in topk_labels),
            "correct": int(label == pred),
        }
        if label == 4:
            unknown_rows.append(row)
        elif label in (0, 1, 2, 3):
            known_rows.append(row)

    known_df = pd.DataFrame(known_rows)
    unknown_df = pd.DataFrame(unknown_rows)

    known_df.to_csv(out_dir / "knn_eval_known.csv", index=False)
    unknown_df.to_csv(out_dir / "knn_eval_unknown.csv", index=False)

    summary = {
        "eval_known_n": int(len(known_df)),
        "eval_unknown_n": int(len(unknown_df)),
        "accuracy_top1": float(known_df["correct"].mean()) if len(known_df) else float("nan"),
        "coverage_topk": float(known_df["topk_hit"].mean()) if len(known_df) else float("nan"),
        "avg_dist_top1_known": float(known_df["distance_top1"].mean()) if len(known_df) else float("nan"),
        "avg_dist_top1_unknown": float(unknown_df["distance_top1"].mean()) if len(unknown_df) else float("nan"),
    }

    with open(out_dir / "knn_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("KNN baseline saved:")
    print(f"- {out_dir / 'knn_eval_known.csv'}")
    print(f"- {out_dir / 'knn_eval_unknown.csv'}")
    print(f"- {out_dir / 'knn_summary.json'}")


if __name__ == "__main__":
    main()
