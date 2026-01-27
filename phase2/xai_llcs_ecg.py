import argparse
import math
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


def load_samples(root: Path) -> List[Tuple[np.ndarray, int, str]]:
    samples = []
    for path in iter_npy_paths(root):
        label = extract_label(path.name)
        x = np.load(path)
        samples.append((x, label, path.name))
    return samples


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
    candidate = root / name.replace(".npy", ".csv")
    if candidate.exists():
        return candidate
    for path in iter_csv_paths(root):
        if path.name == name.replace(".npy", ".csv"):
            return path
    return None


def standardize(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    std = np.where(std < 1e-9, 1.0, std)
    return (matrix - mean) / std, mean, std


def apply_standardize(matrix: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    std = np.where(std < 1e-9, 1.0, std)
    return (matrix - mean) / std


def one_hot(label: int, num_classes: int = 4) -> np.ndarray:
    arr = np.zeros((1, num_classes), dtype=float)
    if 0 <= label < num_classes:
        arr[0, label] = 1.0
    return arr


def train_llcs(graph: Graph, samples: List[Tuple[np.ndarray, int, str]], epochs: int) -> None:
    t_ins = 0
    for _ in range(epochs):
        random.shuffle(samples)
        for x, label, _ in samples:
            target = one_hot(label)
            input_node = Node(x, target=target)
            first, second = graph.find_best_and_second_best_node(input_node)
            if first is None or second is None:
                continue
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


def nearest_sample(node_vec: np.ndarray, samples: List[Tuple[np.ndarray, int, str, str]]) -> Tuple[np.ndarray, int, str, str, float]:
    best = None
    for x, label, name, env in samples:
        d = float(np.linalg.norm(node_vec - x))
        if best is None or d < best[-1]:
            best = (x, label, name, env, d)
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="LLCS-backed XAI using ECG morphology templates.")
    parser.add_argument("--data-root", type=Path, default=Path("Data") / "Disease_dataset_hardood")
    parser.add_argument("--csv-root", type=Path, default=Path("Data") / "Generated_HardOOD2")
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts") / "phase2")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--theta", type=int, default=2)
    parser.add_argument("--deletion-threshold", type=float, default=0.0)
    parser.add_argument("--max-edge-age", type=int, default=200)
    parser.add_argument("--pre", type=int, default=80)
    parser.add_argument("--post", type=int, default=120)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    cfg = DataConfig(data_root=args.data_root, csv_root=args.csv_root)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Relax deletion/insertion to keep more nodes for XAI coverage.
    MODEL_CONSTANT.THETA = args.theta
    MODEL_CONSTANT.DELETION_THRESHOLD = args.deletion_threshold
    MODEL_CONSTANT.MAXIMUM_EDGE_AGE = args.max_edge_age

    # load train samples by env
    env_samples = {}
    all_samples = []
    for env in cfg.env_order:
        root = cfg.data_root / env / "NumpyData"
        samples = load_samples(root)
        env_samples[env] = samples
        for x, label, name in samples:
            all_samples.append((x, label, name, env))

    # train LLCS sequentially
    graph = Graph()
    for env in cfg.env_order:
        train_llcs(graph, env_samples[env], epochs=args.epochs)

    # map nodes -> nearest sample
    node_meta = []
    node_templates = []
    for node in graph.graph:
        nearest = nearest_sample(node.input_weight, all_samples)
        if nearest is None:
            continue
        _, label, name, env, dist = nearest
        csv_path = find_csv_by_name(cfg.csv_root / env / "csv", name)
        if csv_path is None:
            continue
        templ = load_ecg_template(csv_path, args.pre, args.post)
        if templ is None:
            continue
        node_meta.append({"env": env, "label": label, "filename": csv_path.name, "llcs_distance": dist})
        node_templates.append(templ)

    if not node_templates:
        raise SystemExit("No node templates extracted.")

    template_matrix = np.stack(node_templates, axis=0)
    norm_matrix, mean, std = standardize(template_matrix)

    # stability per class (Env1 vs others)
    stability_rows = []
    for cls in (0, 1, 2, 3):
        env_vectors = {}
        for env in cfg.env_order:
            vecs = [vec for vec, meta in zip(norm_matrix, node_meta) if meta["env"] == env and meta["label"] == cls]
            env_vectors[env] = np.array(vecs) if vecs else np.empty((0, 0))
        base_env = cfg.env_order[0]
        base = env_vectors[base_env]
        for env in cfg.env_order[1:]:
            other = env_vectors[env]
            if base.size == 0 or other.size == 0:
                continue
            d = np.linalg.norm(base[:, None, :] - other[None, :, :], axis=2)
            min_d = d.min(axis=1)
            stability_rows.append(
                {
                    "label": cls,
                    "env_a": base_env,
                    "env_b": env,
                    "mean_min_distance": float(np.mean(min_d)),
                    "median_min_distance": float(np.median(min_d)),
                    "max_min_distance": float(np.max(min_d)),
                }
            )

    # unknown explanations
    unknown_rows = []
    eval_root = cfg.csv_root / cfg.eval_env / "csv"
    for path in iter_csv_paths(eval_root):
        label = extract_label(path.name)
        if label != 4:
            continue
        templ = load_ecg_template(path, args.pre, args.post)
        if templ is None:
            continue
        vec = apply_standardize(templ[None, :], mean, std)[0]
        d = np.linalg.norm(norm_matrix - vec[None, :], axis=1)
        idx = np.argsort(d)[: args.top_k]
        for rank, i in enumerate(idx, start=1):
            meta = node_meta[i]
            unknown_rows.append(
                {
                    "unknown_file": path.name,
                    "rank": rank,
                    "proto_env": meta["env"],
                    "proto_label": meta["label"],
                    "proto_file": meta["filename"],
                    "distance": float(d[i]),
                }
            )

    # save
    with open(out_dir / "xai_llcs_stability.csv", "w", encoding="utf-8") as f:
        f.write("label,env_a,env_b,mean_min_distance,median_min_distance,max_min_distance\n")
        for row in stability_rows:
            f.write(
                f"{row['label']},{row['env_a']},{row['env_b']},{row['mean_min_distance']:.6f},{row['median_min_distance']:.6f},{row['max_min_distance']:.6f}\n"
            )

    with open(out_dir / "xai_llcs_unknown_explain.csv", "w", encoding="utf-8") as f:
        f.write("unknown_file,rank,proto_env,proto_label,proto_file,distance\n")
        for row in unknown_rows:
            f.write(
                f"{row['unknown_file']},{row['rank']},{row['proto_env']},{row['proto_label']},{row['proto_file']},{row['distance']:.6f}\n"
            )

    print("LLCS-backed XAI reports saved:")
    print(f"- {out_dir / 'xai_llcs_stability.csv'}")
    print(f"- {out_dir / 'xai_llcs_unknown_explain.csv'}")


if __name__ == "__main__":
    main()
