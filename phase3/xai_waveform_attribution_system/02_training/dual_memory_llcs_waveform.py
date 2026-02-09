import argparse
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional

import numpy as np
import pandas as pd

import sys

# Self-contained: LLCS library is in same folder
SCRIPT_DIR = Path(__file__).resolve().parent
LLCS_ROOT = SCRIPT_DIR / "LLCS"
ROOT = SCRIPT_DIR.parents[2]  # Project root for data paths

sys.path.insert(0, str(LLCS_ROOT))
sys.path.insert(0, str(ROOT))

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
    rr_filter_k: float = 3.0,
    adaptive_window: bool = False,
    adaptive_scale_min: float = 0.7,
    adaptive_scale_max: float = 1.3,
    min_peaks: int = 2,
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
    if len(peaks) < min_peaks:
        if cache is not None:
            cache[key] = None
        return None
    if len(peaks) >= 3 and rr_filter_k > 0:
        rr = np.diff(peaks)
        med = float(np.median(rr))
        mad = float(np.median(np.abs(rr - med)))
        thresh = rr_filter_k * mad if mad > 1e-9 else rr_filter_k * max(1.0, med * 0.1)
        valid = [peaks[0]]
        for i in range(1, len(peaks)):
            rr_i = peaks[i] - peaks[i - 1]
            if abs(rr_i - med) <= thresh:
                valid.append(peaks[i])
        peaks = np.asarray(valid, dtype=int)
        if len(peaks) < min_peaks:
            if cache is not None:
                cache[key] = None
            return None
    beats = []
    rr_med = None
    if adaptive_window and len(peaks) >= 2:
        rr_med = float(np.median(np.diff(peaks)))
    for i, idx in enumerate(peaks):
        pre_i = pre
        post_i = post
        if adaptive_window and rr_med and rr_med > 1e-6 and i > 0:
            rr_i = float(peaks[i] - peaks[i - 1])
            scale = min(adaptive_scale_max, max(adaptive_scale_min, rr_i / rr_med))
            pre_i = int(round(pre * scale))
            post_i = int(round(post * scale))
        start = idx - pre_i
        end = idx + post_i
        if start < 0 or end >= len(voltage):
            continue
        seg = voltage[start:end]
        if seg.shape[0] != pre_i + post_i:
            continue
        if pre_i + post_i != pre + post:
            seg = np.interp(
                np.linspace(0, 1, pre + post, endpoint=False),
                np.linspace(0, 1, seg.shape[0], endpoint=False),
                seg,
            )
        beats.append(seg)
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


def detrend_poly(vec: np.ndarray, order: int = 3) -> np.ndarray:
    if vec.shape[0] <= order + 1:
        return vec
    x = np.arange(vec.shape[0], dtype=float)
    coeffs = np.polyfit(x, vec, deg=order)
    baseline = np.polyval(coeffs, x)
    return vec - baseline


def zscore_vector(vec: np.ndarray, min_std: float = 1e-6) -> np.ndarray:
    mean = float(np.mean(vec))
    std = float(np.std(vec))
    if std < min_std:
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
    node_uid_map: Dict[int, int],
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

            node_key = id(first)
            if node_key not in node_uid_map:
                node_uid_map[node_key] = len(node_uid_map)
            node_entry = win_log.setdefault(
                node_key,
                {"node": first, "wins": [], "counts": {}, "uid": node_uid_map[node_key]},
            )
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
    wins_min_total: int,
    wins_min_env: int,
    purity_min: float,
    template_cache: Dict[Tuple[str, int, int], np.ndarray | None],
    consolidated: Dict[Tuple[int, str], Dict],
    rr_filter_k: float,
    adaptive_window: bool,
    adaptive_scale_min: float,
    adaptive_scale_max: float,
    min_peaks: int,
    baseline_method: str,
    baseline_window: int,
    baseline_poly_order: int,
    min_std: float,
    consolidation_method: str,
    consolidation_outlier_iqr: float,
    consolidation_max_candidates: int,
) -> None:
    for node_id, entry in win_log.items():
        wins = entry["wins"]
        if len(wins) < wins_min_total:
            continue
        label_counts = entry["counts"]
        major_label, major_count = max(label_counts.items(), key=lambda x: x[1])
        purity = major_count / max(1, len(wins))
        if purity < purity_min:
            continue

        env_wins = [(name, label, dist) for env_name, label, name, dist in wins if env_name == env]
        if len(env_wins) < wins_min_env:
            continue
        if not env_wins:
            continue
        env_wins_sorted = sorted(env_wins, key=lambda x: x[2])
        if consolidation_outlier_iqr > 0 and len(env_wins_sorted) >= 4:
            dists = np.asarray([d for _, _, d in env_wins_sorted], dtype=float)
            q1, q3 = np.percentile(dists, [25, 75])
            iqr = q3 - q1
            cutoff = q3 + consolidation_outlier_iqr * iqr
            env_wins_sorted = [w for w in env_wins_sorted if w[2] <= cutoff] or env_wins_sorted

        candidates = env_wins_sorted
        if consolidation_max_candidates > 0 and len(candidates) > consolidation_max_candidates:
            candidates = candidates[:consolidation_max_candidates]

        best_name: Optional[str] = None
        best_label: Optional[int] = None
        best_dist: Optional[float] = None
        if consolidation_method == "medoid" and len(candidates) >= 3:
            vectors = []
            meta = []
            for name, label, dist in candidates:
                csv_path = cfg.csv_root / env / "csv" / name.replace(".npy", ".csv")
                if not csv_path.exists():
                    continue
                templ = load_ecg_template(
                    csv_path,
                    pre,
                    post,
                    cache=template_cache,
                    rr_filter_k=rr_filter_k,
                    adaptive_window=adaptive_window,
                    adaptive_scale_min=adaptive_scale_min,
                    adaptive_scale_max=adaptive_scale_max,
                    min_peaks=min_peaks,
                )
                if templ is None:
                    continue
                if baseline_method == "poly":
                    vec = detrend_poly(templ, order=baseline_poly_order)
                elif baseline_method == "window":
                    vec = align_baseline(templ, baseline_window)
                else:
                    vec = templ
                vec = zscore_vector(vec, min_std=min_std)
                vectors.append(vec)
                meta.append((name, label, dist))
            if len(vectors) >= 3:
                mat = np.stack(vectors, axis=0)
                dist_mat = np.linalg.norm(mat[:, None, :] - mat[None, :, :], axis=2)
                medoid_idx = int(np.argmin(np.sum(dist_mat, axis=1)))
                best_name, best_label, best_dist = meta[medoid_idx]
        if best_name is None:
            best_name, best_label, best_dist = min(env_wins_sorted, key=lambda x: x[2])

        key = (entry["uid"], env)
        if key in consolidated:
            continue
        csv_path = cfg.csv_root / env / "csv" / best_name.replace(".npy", ".csv")
        if not csv_path.exists():
            continue
        templ = load_ecg_template(
            csv_path,
            pre,
            post,
            cache=template_cache,
            rr_filter_k=rr_filter_k,
            adaptive_window=adaptive_window,
            adaptive_scale_min=adaptive_scale_min,
            adaptive_scale_max=adaptive_scale_max,
            min_peaks=min_peaks,
        )
        if templ is None:
            continue
        consolidated[key] = {
            "node_id": entry["uid"],
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
    parser.add_argument("--wins-min-env", type=int, default=3)
    parser.add_argument("--purity-min", type=float, default=0.8)
    parser.add_argument("--baseline-window", type=int, default=20)
    parser.add_argument("--baseline-method", type=str, default="window", choices=("window", "poly", "none"))
    parser.add_argument("--baseline-poly-order", type=int, default=3)
    parser.add_argument("--zscore-min-std", type=float, default=1e-6)
    parser.add_argument("--rr-filter-k", type=float, default=3.0)
    parser.add_argument("--min-peaks", type=int, default=2)
    parser.add_argument("--adaptive-window", action="store_true")
    parser.add_argument("--adaptive-scale-min", type=float, default=0.7)
    parser.add_argument("--adaptive-scale-max", type=float, default=1.3)
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
    parser.add_argument("--consolidation-method", type=str, default="medoid", choices=("nearest", "medoid"))
    parser.add_argument("--consolidation-outlier-iqr", type=float, default=1.5)
    parser.add_argument("--consolidation-max-candidates", type=int, default=50)
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
    node_uid_map: Dict[int, int] = {}

    for env in cfg.env_order:
        train_llcs(
            graph,
            env_samples[env],
            epochs=args.epochs,
            win_log=win_log,
            node_uid_map=node_uid_map,
        )
        consolidate_env(
            cfg,
            win_log,
            env,
            pre=args.pre,
            post=args.post,
            wins_min_total=args.wins_min,
            wins_min_env=args.wins_min_env,
            purity_min=args.purity_min,
            template_cache=template_cache,
            consolidated=consolidated,
            rr_filter_k=args.rr_filter_k,
            adaptive_window=args.adaptive_window,
            adaptive_scale_min=args.adaptive_scale_min,
            adaptive_scale_max=args.adaptive_scale_max,
            min_peaks=args.min_peaks,
            baseline_method=args.baseline_method,
            baseline_window=args.baseline_window,
            baseline_poly_order=args.baseline_poly_order,
            min_std=args.zscore_min_std,
            consolidation_method=args.consolidation_method,
            consolidation_outlier_iqr=args.consolidation_outlier_iqr,
            consolidation_max_candidates=args.consolidation_max_candidates,
        )

    if not consolidated:
        raise SystemExit("No consolidated prototypes found. Try lowering --wins-min or check CSV root.")

    proto_meta = list(consolidated.values())
    node_templates = []
    for row in proto_meta:
        csv_path = cfg.csv_root / row["env"] / "csv" / row["filename"]
        templ = load_ecg_template(
            csv_path,
            args.pre,
            args.post,
            cache=template_cache,
            rr_filter_k=args.rr_filter_k,
            adaptive_window=args.adaptive_window,
            adaptive_scale_min=args.adaptive_scale_min,
            adaptive_scale_max=args.adaptive_scale_max,
            min_peaks=args.min_peaks,
        )
        if templ is None:
            raise SystemExit(f"Template missing for {csv_path}")
        node_templates.append(templ)

    def normalize_template(vec: np.ndarray) -> np.ndarray:
        if args.baseline_method == "poly":
            vec = detrend_poly(vec, order=args.baseline_poly_order)
        elif args.baseline_method == "window":
            vec = align_baseline(vec, args.baseline_window)
        return zscore_vector(vec, min_std=args.zscore_min_std)

    norm_matrix = np.stack(
        [normalize_template(t) for t in node_templates],
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
        node_uid = entry["uid"]
        if node_uid not in node_to_proto_idx:
            continue
        wins = entry["wins"]
        if len(wins) < args.tau_min_wins:
            continue
        distances: List[float] = []
        for env_name, _, name, _ in wins:
            csv_path = cfg.csv_root / env_name / "csv" / name.replace(".npy", ".csv")
            if not csv_path.exists():
                continue
            templ = load_ecg_template(
                csv_path,
                args.pre,
                args.post,
                cache=template_cache,
                rr_filter_k=args.rr_filter_k,
                adaptive_window=args.adaptive_window,
                adaptive_scale_min=args.adaptive_scale_min,
                adaptive_scale_max=args.adaptive_scale_max,
                min_peaks=args.min_peaks,
            )
            if templ is None:
                continue
            vec = normalize_template(templ)
            idx = node_to_proto_idx[node_uid]
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
            node_thresholds[node_uid] = med + args.tau_fit_mad_k * mad
        else:
            node_thresholds[node_uid] = float(np.percentile(dist_arr, args.tau_fit_percentile))

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
        templ = load_ecg_template(
            path,
            args.pre,
            args.post,
            cache=template_cache,
            rr_filter_k=args.rr_filter_k,
            adaptive_window=args.adaptive_window,
            adaptive_scale_min=args.adaptive_scale_min,
            adaptive_scale_max=args.adaptive_scale_max,
            min_peaks=args.min_peaks,
        )
        if templ is None:
            continue
        vec = normalize_template(templ)
        d = compute_distances(vec, norm_matrix)
        idx = np.argsort(d)[: args.top_k]
        if idx.size == 0:
            continue
        pred_label = int(proto_meta[idx[0]]["major_label"])
        pred_node = int(proto_meta[idx[0]]["node_id"])
        tau_best = float(
            node_thresholds.get(
                pred_node,
                class_thresholds.get(pred_label, global_threshold),
            )
        )
        is_ood_best = bool(float(d[idx[0]]) > tau_best)
        for rank, i in enumerate(idx, start=1):
            meta = proto_meta[i]
            tau_rank = float(
                node_thresholds.get(
                    int(meta["node_id"]),
                    class_thresholds.get(int(meta["major_label"]), global_threshold),
                )
            )
            is_ood_rank = bool(float(d[i]) > tau_rank)
            unknown_rows.append(
                {
                    "unknown_file": path.name,
                    "rank": rank,
                    "proto_env": meta["env"],
                    "proto_label": meta["major_label"],
                    "proto_file": meta["filename"],
                    "node_id": meta["node_id"],
                    "distance": float(d[i]),
                    "ood_threshold": tau_rank,
                    "is_ood": is_ood_rank,
                    "ood_threshold_best": tau_best,
                    "is_ood_best": is_ood_best,
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
        f.write(
            "unknown_file,rank,proto_env,proto_label,proto_file,node_id,distance,"
            "ood_threshold,is_ood,ood_threshold_best,is_ood_best\n"
        )
        for row in unknown_rows:
            f.write(
                f"{row['unknown_file']},{row['rank']},{row['proto_env']},{row['proto_label']},"
                f"{row['proto_file']},{row['node_id']},{row['distance']:.6f},"
                f"{row['ood_threshold']:.6f},{int(row['is_ood'])},"
                f"{row['ood_threshold_best']:.6f},{int(row['is_ood_best'])}\n"
            )

    print("Dual-memory LLCS waveform XAI saved:")
    print(f"- {out_dir / 'dual_memory_prototypes.csv'}")
    print(f"- {out_dir / 'dual_memory_stability.csv'}")
    print(f"- {out_dir / 'dual_memory_node_tau.csv'}")
    print(f"- {out_dir / 'dual_memory_unknown_explain.csv'}")


if __name__ == "__main__":
    main()
