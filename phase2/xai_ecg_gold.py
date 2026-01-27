import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

LABEL_PATTERN = re.compile(r"_(\d)_")


@dataclass
class DataConfig:
    csv_root: Path
    env_order: Tuple[str, ...] = ("Env1", "Env2", "Env3", "Env4")
    eval_env: str = "Eval"


def extract_label(filename: str) -> int:
    match = LABEL_PATTERN.search(filename)
    if not match:
        raise ValueError(f"Cannot extract label from filename: {filename}")
    return int(match.group(1))


def iter_csv_paths(root: Path) -> Iterable[Path]:
    for name in os.listdir(root):
        if name.endswith(".csv"):
            yield root / name


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


def standardize(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    std = np.where(std < 1e-9, 1.0, std)
    return (matrix - mean) / std, mean, std


def apply_standardize(matrix: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    std = np.where(std < 1e-9, 1.0, std)
    return (matrix - mean) / std


def pairwise_distances(matrix: np.ndarray) -> np.ndarray:
    diffs = matrix[:, None, :] - matrix[None, :, :]
    return np.sqrt(np.sum(diffs * diffs, axis=2))


def select_medoids(matrix: np.ndarray, k: int) -> List[int]:
    n = matrix.shape[0]
    if n == 0:
        return []
    if k >= n:
        return list(range(n))
    dists = pairwise_distances(matrix)
    avg = dists.mean(axis=1)
    medoids = [int(np.argmin(avg))]
    while len(medoids) < k:
        min_to_medoids = np.min(dists[:, medoids], axis=1)
        next_idx = int(np.argmax(min_to_medoids))
        if next_idx in medoids:
            break
        medoids.append(next_idx)
    return medoids


def nearest_prototypes(x: np.ndarray, proto_matrix: np.ndarray, proto_meta: List[dict], k: int) -> List[dict]:
    if proto_matrix.size == 0:
        return []
    d = np.linalg.norm(proto_matrix - x[None, :], axis=1)
    idx = np.argsort(d)[:k]
    results = []
    for i in idx:
        item = dict(proto_meta[i])
        item["distance"] = float(d[i])
        results.append(item)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="XAI gold (ECG morphology templates).")
    parser.add_argument("--csv-root", type=Path, default=Path("Data") / "Generated_HardOOD2")
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts") / "phase2")
    parser.add_argument("--prototypes-per-class", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--pre", type=int, default=80)
    parser.add_argument("--post", type=int, default=120)
    args = parser.parse_args()

    cfg = DataConfig(csv_root=args.csv_root)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    env_data = {}
    all_templates = []
    all_labels = []

    for env in cfg.env_order:
        env_root = cfg.csv_root / env / "csv"
        templates = []
        labels = []
        names = []
        for path in iter_csv_paths(env_root):
            label = extract_label(path.name)
            templ = load_ecg_template(path, args.pre, args.post)
            if templ is None:
                continue
            templates.append(templ)
            labels.append(label)
            names.append(path.name)
        env_data[env] = {"templates": templates, "labels": labels, "names": names}
        all_templates.extend(templates)
        all_labels.extend(labels)

    if not all_templates:
        raise SystemExit("No templates extracted.")

    template_matrix = np.stack(all_templates, axis=0)
    norm_matrix, mean, std = standardize(template_matrix)

    # class mean template for attribution
    class_mean = {}
    class_std = {}
    for cls in (0, 1, 2, 3):
        idx = [i for i, y in enumerate(all_labels) if y == cls]
        if not idx:
            continue
        Xc = norm_matrix[idx]
        class_mean[cls] = Xc.mean(axis=0)
        class_std[cls] = np.where(Xc.std(axis=0) < 1e-9, 1.0, Xc.std(axis=0))

    prototype_rows = []
    prototype_meta = []
    prototype_vectors = []

    offset = 0
    env_offsets = {}
    for env in cfg.env_order:
        env_offsets[env] = offset
        offset += len(env_data[env]["templates"])

    for env in cfg.env_order:
        templates = env_data[env]["templates"]
        labels = env_data[env]["labels"]
        names = env_data[env]["names"]
        if not templates:
            continue
        X = apply_standardize(np.stack(templates, axis=0), mean, std)
        base_idx = env_offsets[env]
        for cls in (0, 1, 2, 3):
            idx = [i for i, y in enumerate(labels) if y == cls]
            if not idx:
                continue
            Xm = X[idx]
            medoid_idx = select_medoids(Xm, args.prototypes_per_class)
            for local_i in medoid_idx:
                env_i = idx[local_i]
                vec = X[env_i]
                proto = {
                    "env": env,
                    "label": cls,
                    "filename": names[env_i],
                }
                prototype_meta.append(proto)
                prototype_vectors.append(vec)

                stats_mean = class_mean.get(cls)
                stats_std = class_std.get(cls)
                if stats_mean is not None and stats_std is not None:
                    z = (vec - stats_mean) / stats_std
                    top_idx = np.argsort(np.abs(z))[-5:][::-1]
                    top_feats = [f"t{int(i)}={z[i]:.2f}" for i in top_idx]
                else:
                    top_feats = []
                prototype_rows.append(
                    {
                        "env": env,
                        "label": cls,
                        "filename": names[env_i],
                        "top_features": " | ".join(top_feats),
                    }
                )

    proto_matrix = np.array(prototype_vectors) if prototype_vectors else np.empty((0, 0))

    # stability
    stability_rows = []
    for cls in (0, 1, 2, 3):
        env_protos = {}
        for env in cfg.env_order:
            env_vectors = [vec for vec, meta in zip(prototype_vectors, prototype_meta) if meta["env"] == env and meta["label"] == cls]
            env_protos[env] = np.array(env_vectors) if env_vectors else np.empty((0, 0))
        base_env = cfg.env_order[0]
        base = env_protos[base_env]
        for env in cfg.env_order[1:]:
            other = env_protos[env]
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

    # Unknown explanations
    eval_root = cfg.csv_root / cfg.eval_env / "csv"
    unknown_rows = []
    for path in iter_csv_paths(eval_root):
        label = extract_label(path.name)
        if label != 4:
            continue
        templ = load_ecg_template(path, args.pre, args.post)
        if templ is None:
            continue
        vec = apply_standardize(templ[None, :], mean, std)[0]
        nearest = nearest_prototypes(vec, proto_matrix, prototype_meta, args.top_k)
        for rank, item in enumerate(nearest, start=1):
            unknown_rows.append(
                {
                    "unknown_file": path.name,
                    "rank": rank,
                    "proto_env": item["env"],
                    "proto_label": item["label"],
                    "proto_file": item["filename"],
                    "distance": item["distance"],
                }
            )

    with open(out_dir / "xai_ecg_prototypes.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "prototypes": prototype_rows,
                "stability": stability_rows,
                "unknown_explanations": unknown_rows,
            },
            f,
            indent=2,
        )

    with open(out_dir / "xai_ecg_prototypes.csv", "w", encoding="utf-8") as f:
        f.write("env,label,filename,top_features\n")
        for row in prototype_rows:
            f.write(f"{row['env']},{row['label']},{row['filename']},\"{row['top_features']}\"\n")

    with open(out_dir / "xai_ecg_stability.csv", "w", encoding="utf-8") as f:
        f.write("label,env_a,env_b,mean_min_distance,median_min_distance,max_min_distance\n")
        for row in stability_rows:
            f.write(
                f"{row['label']},{row['env_a']},{row['env_b']},{row['mean_min_distance']:.6f},{row['median_min_distance']:.6f},{row['max_min_distance']:.6f}\n"
            )

    with open(out_dir / "xai_ecg_unknown_explain.csv", "w", encoding="utf-8") as f:
        f.write("unknown_file,rank,proto_env,proto_label,proto_file,distance\n")
        for row in unknown_rows:
            f.write(
                f"{row['unknown_file']},{row['rank']},{row['proto_env']},{row['proto_label']},{row['proto_file']},{row['distance']:.6f}\n"
            )

    print("XAI ECG reports saved:")
    print(f"- {out_dir / 'xai_ecg_prototypes.json'}")
    print(f"- {out_dir / 'xai_ecg_prototypes.csv'}")
    print(f"- {out_dir / 'xai_ecg_stability.csv'}")
    print(f"- {out_dir / 'xai_ecg_unknown_explain.csv'}")


if __name__ == "__main__":
    main()
