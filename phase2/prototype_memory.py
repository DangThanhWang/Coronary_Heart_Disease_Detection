from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances


@dataclass(frozen=True)
class MemoryCase:
    label: int
    source: str
    train_index: int
    support: int


class CounterfactualMemory:
    """Case-based prototype memory in standardized clinical-feature space."""

    def __init__(self, prototypes_per_class: int = 16, seed: int = 42):
        self.prototypes_per_class = int(prototypes_per_class)
        self.seed = int(seed)
        self.prototype_matrix_: np.ndarray | None = None
        self.cases_: list[MemoryCase] = []

    def fit(self, x_z: np.ndarray, y: np.ndarray, sources: Sequence[str]) -> "CounterfactualMemory":
        rows: list[np.ndarray] = []
        cases: list[MemoryCase] = []
        for label in sorted(int(v) for v in np.unique(y)):
            idx = np.flatnonzero(y == label)
            x_cls = x_z[idx]
            k = min(self.prototypes_per_class, len(idx))
            if k == len(idx):
                assignments = np.arange(len(idx))
                medoid_local = np.arange(len(idx))
            else:
                km = KMeans(n_clusters=k, random_state=self.seed, n_init=10)
                assignments = km.fit_predict(x_cls)
                medoids = []
                for c in range(k):
                    members = np.flatnonzero(assignments == c)
                    center = km.cluster_centers_[c]
                    d = np.linalg.norm(x_cls[members] - center[None, :], axis=1)
                    medoids.append(int(members[int(np.argmin(d))]))
                medoid_local = np.asarray(medoids, dtype=int)
            for li in medoid_local:
                gi = int(idx[int(li)])
                support = int(np.sum(assignments == assignments[int(li)])) if len(assignments) else 1
                rows.append(x_z[gi])
                cases.append(MemoryCase(label=label, source=str(sources[gi]), train_index=gi, support=support))
        self.prototype_matrix_ = np.stack(rows, axis=0)
        self.cases_ = cases
        return self

    def distances(self, x_z: np.ndarray) -> np.ndarray:
        if self.prototype_matrix_ is None:
            raise RuntimeError("CounterfactualMemory is not fitted.")
        return pairwise_distances(x_z, self.prototype_matrix_, metric="euclidean")

    def nearest_by_label(self, x_z: np.ndarray, label: int) -> tuple[np.ndarray, np.ndarray]:
        d = self.distances(x_z)
        candidates = [i for i, c in enumerate(self.cases_) if c.label == label]
        if not candidates:
            raise RuntimeError(f"No prototypes for label {label}")
        sub = d[:, candidates]
        local = np.argmin(sub, axis=1)
        proto_idx = np.asarray([candidates[int(i)] for i in local], dtype=int)
        proto_dist = d[np.arange(d.shape[0]), proto_idx]
        return proto_idx, proto_dist

    def margin_norm_minus_sttc(self, x_z: np.ndarray) -> np.ndarray:
        _, d_norm = self.nearest_by_label(x_z, 0)
        _, d_sttc = self.nearest_by_label(x_z, 1)
        return d_norm - d_sttc

    def nearest_pair(self, x_z: np.ndarray) -> list[dict]:
        norm_i, norm_d = self.nearest_by_label(x_z, 0)
        sttc_i, sttc_d = self.nearest_by_label(x_z, 1)
        out = []
        for i in range(x_z.shape[0]):
            ni = int(norm_i[i])
            si = int(sttc_i[i])
            out.append(
                {
                    "nearest_norm_source": self.cases_[ni].source,
                    "nearest_norm_distance": float(norm_d[i]),
                    "nearest_norm_support": int(self.cases_[ni].support),
                    "nearest_sttc_source": self.cases_[si].source,
                    "nearest_sttc_distance": float(sttc_d[i]),
                    "nearest_sttc_support": int(self.cases_[si].support),
                    "margin_norm_minus_sttc": float(norm_d[i] - sttc_d[i]),
                }
            )
        return out

