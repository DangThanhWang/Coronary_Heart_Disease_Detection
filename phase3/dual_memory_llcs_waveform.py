import argparse
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

import sys

ROOT = Path(__file__).resolve().parents[1]
LLCS_ROOT = ROOT / "LLCS"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(LLCS_ROOT))

from graph import Graph
from node import Node
from constant import MODEL_CONSTANT

LABEL_PATTERN = re.compile(r"_(\d)_")


@dataclass
class DataConfig:
    data_root: Path
    csv_root: Path
    env_order: Tuple[str, ...] = ("Env1", "Env2", "Env3", "Env4")
    eval_env: str = "Eval"


def extract_label(filename: str) -> int:
    match = LABEL_PATTERN.search(filename)
    if not match:
        raise ValueError(f"Cannot extract label from filename: {filename}")
    return int(match.group(1))


def iter_npy_paths(root: Path) -> Iterable[Path]:
    for name in os.listdir(root):
        if name.endswith(".npy"):
            yield root / name


def iter_csv_paths(root: Path) -> Iterable[Path]:
    for name in os.listdir(root):
        if name.endswith(".csv"):
            yield root / name


def load_samples(root: Path, env: str) -> List[Tuple[np.ndarray, int, str, str]]:
    samples = []
    for path in iter_npy_paths(root):
        label = extract_label(path.name)
        x = np.load(path)
        samples.append((x, label, path.name, env))
    return samples


def load_ecg_template(
    csv_path: Path,
    pre: int,
    post: int,
    cache: Dict[Tuple[str, int, int], np.ndarray | None] | None = None,
) -> np.ndarray | None:
    if cache is not None:
        key = (str(csv_path), pre, post)
        if key in cache:
            return cache[key]
    df = pd.read_csv(csv_path)
    if "Voltage" not in df.columns or "Peak" not in df.columns:
        if cache is not None:
            cache[key] = None
        return None
    voltage = df["Voltage"].to_numpy(dtype=float)
    peaks = df.index[df["Peak"] == 3].to_numpy(dtype=int)
    if len(peaks) == 0:
        if cache is not None:
            cache[key] = None
        return None
    beats = []
    for idx in peaks:
        start = idx - pre
        end = idx + post
        if start < 0 or end >= len(voltage):
            continue
        beats.append(voltage[start:end])
    if not beats:
        if cache is not None:
            cache[key] = None
        return None
    beats = np.stack(beats, axis=0)
    templ = np.median(beats, axis=0)
    if cache is not None:
        cache[key] = templ
    return templ


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


def corr_distance(vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    v = vec - np.mean(vec)
    v_norm = np.linalg.norm(v)
    if v_norm < 1e-9:
        return np.ones(matrix.shape[0], dtype=float)
    m = matrix - np.mean(matrix, axis=1, keepdims=True)
    m_norm = np.linalg.norm(m, axis=1)
    denom = v_norm * m_norm
    corr = np.where(denom < 1e-9, 0.0, (m @ v) / denom)
    return 1.0 - corr


def hybrid_distance(vec: np.ndarray, matrix: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    # Normalize Euclid to RMS to keep scale comparable to (1 - corr)
    d_euclid = np.sqrt(np.mean((matrix - vec[None, :]) ** 2, axis=1))
    d_corr = corr_distance(vec, matrix)
    return alpha * d_euclid + (1.0 - alpha) * d_corr


def one_hot(label: int, num_classes: int = 4) -> np.ndarray:
    arr = np.zeros((1, num_classes), dtype=float)
    if 0 <= label < num_classes:
        arr[0, label] = 1.0
    return arr


def train_llcs(
    graph: Graph,
    samples: List[Tuple[np.ndarray, int, str, str]],
    epochs: int,
    win_log: Dict[int, Dict],
) -> None:
    t_ins = 0
    for _ in range(epochs):
        random.shuffle(samples)
        for x, label, name, env in samples:
            target = one_hot(label)
            input_node = Node(x, target=target)
            first, second = graph.find_best_and_second_best_node(input_node)
            if first is None or second is None:
                continue

            node_id = id(first)
            node_entry = win_log.setdefault(node_id, {"node": first, "wins": [], "counts": {}})
            dist = float(input_node.get_input_distance(first))
            node_entry["wins"].append((env, label, name, dist))
            node_entry["counts"][label] = node_entry["counts"].get(label, 0) + 1

            first.update_input_weight(input_node, best_node=True)
            for neighbor in first.get_neighbors():
                neighbor.get_node().update_input_weight(input_node, best_node=False)

            graph.activate_node(input_node)
            graph.update_out_weight(input_node)

            t_ins += 1
            if t_ins == MODEL_CONSTANT.THETA * graph.get_graph_size():
                graph.update_BI()
                graph.get_q_and_f_node(t_ins)
                t_ins = 0

            graph.check_deletion_criteria()
            first.update_error_counter(input_node)
            first.decrease_age_for_winner()
            first.decrease_insertion_threshold_for_winner()
            if not first.is_neighbor(second):
                first.add_neighbor(second)
            first.update_edge_for_winner(second)

            if graph.get_graph_size() > 2:
                graph.update_node()


def consolidate_env(
    cfg: DataConfig,
    win_log: Dict[int, Dict],
    env: str,
    pre: int,
    post: int,
    wins_min: int,
    purity_min: float,
    template_cache: Dict[Tuple[str, int, int], np.ndarray | None],
    consolidated: Dict[Tuple[int, str], Dict],
) -> None:
    for node_id, entry in win_log.items():
        wins = entry["wins"]
        if len(wins) < wins_min:
            continue
        label_counts = entry["counts"]
        major_label, major_count = max(label_counts.items(), key=lambda x: x[1])
        purity = major_count / max(1, len(wins))
        if purity < purity_min:
            continue

        env_wins = [(name, label, dist) for env_name, label, name, dist in wins if env_name == env]
        if not env_wins:
            continue
        best_name, best_label, best_dist = min(env_wins, key=lambda x: x[2])
        key = (node_id, env)
        if key in consolidated:
            continue
        csv_path = cfg.csv_root / env / "csv" / best_name.replace(".npy", ".csv")
        if not csv_path.exists():
            continue
        templ = load_ecg_template(csv_path, pre, post, cache=template_cache)
        if templ is None:
            continue
        consolidated[key] = {
            "node_id": node_id,
            "env": env,
            "label": best_label,
            "major_label": major_label,
            "filename": csv_path.name,
            "wins_env": len(env_wins),
            "wins_total": len(wins),
            "purity": float(purity),
            "distance_to_node": float(best_dist),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Dual-memory LLCS -> waveform XAI.")
    parser.add_argument("--data-root", type=Path, default=Path("Data") / "Disease_dataset_hardood")
    parser.add_argument("--csv-root", type=Path, default=Path("Data") / "Generated_HardOOD2")
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts") / "phase3" / "dual_memory")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--theta", type=int, default=5)
    parser.add_argument("--deletion-threshold", type=float, default=0.02)
    parser.add_argument("--max-edge-age", type=int, default=50)
    parser.add_argument("--pre", type=int, default=80)
    parser.add_argument("--post", type=int, default=120)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--wins-min", type=int, default=5)
    parser.add_argument("--purity-min", type=float, default=0.8)
    parser.add_argument("--baseline-window", type=int, default=20)
    parser.add_argument("--ood-threshold-percentile", type=float, default=95.0)
    parser.add_argument("--ood-threshold-absolute", type=float, default=-1.0)
    parser.add_argument("--tau-fit-percentile", type=float, default=95.0)
    parser.add_argument("--tau-fit-mad-k", type=float, default=3.0)
    parser.add_argument("--tau-min-wins", type=int, default=10)
    parser.add_argument("--tau-fit-mode", type=str, default="percentile", choices=("percentile", "mad"))
    parser.add_argument(
        "--distance-metric",
        type=str,
        default="hybrid",
        choices=("euclid", "corr", "hybrid"),
    )
    parser.add_argument("--hybrid-alpha", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    cfg = DataConfig(data_root=args.data_root, csv_root=args.csv_root)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    MODEL_CONSTANT.THETA = args.theta
    MODEL_CONSTANT.DELETION_THRESHOLD = args.deletion_threshold
    MODEL_CONSTANT.MAXIMUM_EDGE_AGE = args.max_edge_age

    env_samples = {}
    for env in cfg.env_order:
        root = cfg.data_root / env / "NumpyData"
        env_samples[env] = load_samples(root, env)

    graph = Graph()
    win_log: Dict[int, Dict] = {}
    consolidated: Dict[Tuple[int, str], Dict] = {}
    template_cache: Dict[Tuple[str, int, int], np.ndarray | None] = {}

    for env in cfg.env_order:
        train_llcs(graph, env_samples[env], epochs=args.epochs, win_log=win_log)
        consolidate_env(
            cfg,
            win_log,
            env,
            pre=args.pre,
            post=args.post,
            wins_min=args.wins_min,
            purity_min=args.purity_min,
            template_cache=template_cache,
            consolidated=consolidated,
        )

    if not consolidated:
        raise SystemExit("No consolidated prototypes found. Try lowering --wins-min or check CSV root.")

    proto_meta = list(consolidated.values())
    node_templates = []
    for row in proto_meta:
        csv_path = cfg.csv_root / row["env"] / "csv" / row["filename"]
        templ = load_ecg_template(csv_path, args.pre, args.post, cache=template_cache)
        if templ is None:
            raise SystemExit(f"Template missing for {csv_path}")
        node_templates.append(templ)

    norm_matrix = np.stack(
        [zscore_vector(align_baseline(t, args.baseline_window)) for t in node_templates],
        axis=0,
    )

    # Stability: same node across envs
    stability_rows = []
    base_env = cfg.env_order[0]
    for cls in (0, 1, 2, 3):
        for env in cfg.env_order[1:]:
            distances = []
            for node_id in {m["node_id"] for m in proto_meta}:
                idx_base = [i for i, m in enumerate(proto_meta) if m["node_id"] == node_id and m["env"] == base_env and m["major_label"] == cls]
                idx_env = [i for i, m in enumerate(proto_meta) if m["node_id"] == node_id and m["env"] == env and m["major_label"] == cls]
                if not idx_base or not idx_env:
                    continue
                vec_a = norm_matrix[idx_base[0]]
                vec_b = norm_matrix[idx_env[0]]
                distances.append(float(np.linalg.norm(vec_a - vec_b)))
            if distances:
                stability_rows.append(
                    {
                        "label": cls,
                        "env_a": base_env,
                        "env_b": env,
                        "mean_min_distance": float(np.mean(distances)),
                        "median_min_distance": float(np.median(distances)),
                        "max_min_distance": float(np.max(distances)),
                        "nodes": len(distances),
                    }
                )

    def compute_distances(vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        if args.distance_metric == "euclid":
            return np.linalg.norm(matrix - vec[None, :], axis=1)
        if args.distance_metric == "corr":
            return corr_distance(vec, matrix)
        return hybrid_distance(vec, matrix, alpha=args.hybrid_alpha)

    # Global threshold from proto-proto distances (fallback)
    if args.ood_threshold_absolute > 0:
        global_threshold = float(args.ood_threshold_absolute)
    elif norm_matrix.shape[0] > 1:
        dist_matrix = np.zeros((norm_matrix.shape[0], norm_matrix.shape[0]), dtype=float)
        for i in range(norm_matrix.shape[0]):
            dist_matrix[i] = compute_distances(norm_matrix[i], norm_matrix)
        np.fill_diagonal(dist_matrix, np.inf)
        nearest = np.min(dist_matrix, axis=1)
        global_threshold = float(np.percentile(nearest, args.ood_threshold_percentile))
    else:
        global_threshold = float("inf")

    # Fit tau per node using template distances on wins (same space as inference)
    node_to_proto_idx: Dict[int, List[int]] = {}
    for i, meta in enumerate(proto_meta):
        node_to_proto_idx.setdefault(meta["node_id"], []).append(i)

    node_thresholds: Dict[int, float] = {}
    for node_id, entry in win_log.items():
        if node_id not in node_to_proto_idx:
            continue
        wins = entry["wins"]
        if len(wins) < args.tau_min_wins:
            continue
        distances: List[float] = []
        for env_name, _, name, _ in wins:
            csv_path = cfg.csv_root / env_name / "csv" / name.replace(".npy", ".csv")
            if not csv_path.exists():
                continue
            templ = load_ecg_template(csv_path, args.pre, args.post, cache=template_cache)
            if templ is None:
                continue
            vec = zscore_vector(align_baseline(templ, args.baseline_window))
            idx = node_to_proto_idx[node_id]
            d = compute_distances(vec, norm_matrix[idx])
            if d.size == 0:
                continue
            distances.append(float(np.min(d)))
        if len(distances) < args.tau_min_wins:
            continue
        dist_arr = np.asarray(distances, dtype=float)
        if args.tau_fit_mode == "mad":
            med = float(np.median(dist_arr))
            mad = float(np.median(np.abs(dist_arr - med)))
            node_thresholds[node_id] = med + args.tau_fit_mad_k * mad
        else:
            node_thresholds[node_id] = float(np.percentile(dist_arr, args.tau_fit_percentile))

    # Class thresholds from node thresholds (fallback)
    class_thresholds: Dict[int, float] = {}
    for cls in {m["major_label"] for m in proto_meta}:
        cls_nodes = [m["node_id"] for m in proto_meta if m["major_label"] == cls]
        cls_vals = [node_thresholds[n] for n in cls_nodes if n in node_thresholds]
        if cls_vals:
            class_thresholds[int(cls)] = float(np.median(cls_vals))
        else:
            class_thresholds[int(cls)] = global_threshold

    # Unknown explanations using long-term memory
    unknown_rows = []
    eval_root = cfg.csv_root / cfg.eval_env / "csv"
    for path in iter_csv_paths(eval_root):
        label = extract_label(path.name)
        if label != 4:
            continue
        templ = load_ecg_template(path, args.pre, args.post, cache=template_cache)
        if templ is None:
            continue
        vec = zscore_vector(align_baseline(templ, args.baseline_window))
        d = compute_distances(vec, norm_matrix)
        idx = np.argsort(d)[: args.top_k]
        if idx.size == 0:
            continue
        pred_label = int(proto_meta[idx[0]]["major_label"])
        pred_node = int(proto_meta[idx[0]]["node_id"])
        tau = float(
            node_thresholds.get(
                pred_node,
                class_thresholds.get(pred_label, global_threshold),
            )
        )
        is_ood = bool(float(d[idx[0]]) > tau)
        for rank, i in enumerate(idx, start=1):
            meta = proto_meta[i]
            unknown_rows.append(
                {
                    "unknown_file": path.name,
                    "rank": rank,
                    "proto_env": meta["env"],
                    "proto_label": meta["major_label"],
                    "proto_file": meta["filename"],
                    "node_id": meta["node_id"],
                    "distance": float(d[i]),
                    "ood_threshold": tau,
                    "is_ood": is_ood,
                }
            )

    with open(out_dir / "dual_memory_prototypes.csv", "w", encoding="utf-8") as f:
        f.write("node_id,env,label,major_label,filename,wins_env,wins_total,purity,distance_to_node\n")
        for row in proto_meta:
            f.write(
                f"{row['node_id']},{row['env']},{row['label']},{row['major_label']},{row['filename']},{row['wins_env']},{row['wins_total']},{row['purity']:.3f},{row['distance_to_node']:.6f}\n"
            )

    with open(out_dir / "dual_memory_stability.csv", "w", encoding="utf-8") as f:
        f.write("label,env_a,env_b,mean_min_distance,median_min_distance,max_min_distance,nodes\n")
        for row in stability_rows:
            f.write(
                f"{row['label']},{row['env_a']},{row['env_b']},{row['mean_min_distance']:.6f},{row['median_min_distance']:.6f},{row['max_min_distance']:.6f},{row['nodes']}\n"
            )

    with open(out_dir / "dual_memory_node_tau.csv", "w", encoding="utf-8") as f:
        f.write("node_id,tau\n")
        for node_id, tau in sorted(node_thresholds.items(), key=lambda x: x[0]):
            f.write(f"{node_id},{tau:.6f}\n")

    with open(out_dir / "dual_memory_unknown_explain.csv", "w", encoding="utf-8") as f:
        f.write("unknown_file,rank,proto_env,proto_label,proto_file,node_id,distance,ood_threshold,is_ood\n")
        for row in unknown_rows:
            f.write(
                f"{row['unknown_file']},{row['rank']},{row['proto_env']},{row['proto_label']},{row['proto_file']},{row['node_id']},{row['distance']:.6f},{row['ood_threshold']:.6f},{int(row['is_ood'])}\n"
            )

    print("Dual-memory LLCS waveform XAI saved:")
    print(f"- {out_dir / 'dual_memory_prototypes.csv'}")
    print(f"- {out_dir / 'dual_memory_stability.csv'}")
    print(f"- {out_dir / 'dual_memory_node_tau.csv'}")
    print(f"- {out_dir / 'dual_memory_unknown_explain.csv'}")


if __name__ == "__main__":
    main()
