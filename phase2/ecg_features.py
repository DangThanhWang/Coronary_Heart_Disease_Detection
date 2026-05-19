from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


LABEL_RE = re.compile(r"_(\d)_")


@dataclass(frozen=True)
class Sample:
    path: Path
    label: int
    split: str
    env: str


def extract_label(path_or_name: Path | str) -> int:
    name = Path(path_or_name).name
    m = LABEL_RE.search(name)
    return int(m.group(1)) if m else -1


def iter_csv_samples(csv_root: Path, split_names: Sequence[str]) -> list[Sample]:
    out: list[Sample] = []
    for split in split_names:
        root = csv_root / split / "csv"
        if not root.exists():
            continue
        for path in sorted(root.glob("*.csv")):
            out.append(Sample(path=path, label=extract_label(path), split=split, env=split))
    return out


def iter_train_env_samples(csv_root: Path, envs: Sequence[str] = ("Env1", "Env2", "Env3", "Env4")) -> list[Sample]:
    out: list[Sample] = []
    for env in envs:
        root = csv_root / env / "csv"
        if not root.exists():
            continue
        for path in sorted(root.glob("*.csv")):
            out.append(Sample(path=path, label=extract_label(path), split="train", env=env))
    return out


def stratified_cap(samples: Sequence[Sample], max_per_class: int | None, labels: Iterable[int], seed: int) -> list[Sample]:
    if max_per_class is None:
        return [s for s in samples if s.label in set(labels)]
    rng = np.random.default_rng(seed)
    keep: list[Sample] = []
    for label in labels:
        group = [s for s in samples if s.label == label]
        if len(group) <= max_per_class:
            keep.extend(group)
            continue
        idx = rng.choice(len(group), size=max_per_class, replace=False)
        keep.extend(group[int(i)] for i in idx)
    return sorted(keep, key=lambda s: str(s.path))


def _lead_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c.startswith("Lead_")]
    if not cols and "Voltage" in df.columns:
        return ["Voltage"]
    if "Voltage" in df.columns:
        return ["Voltage"] + cols
    return cols


def _fs_from_time(df: pd.DataFrame) -> float:
    if "Time" not in df.columns or len(df) < 3:
        return 100.0
    t = df["Time"].to_numpy(dtype=float)
    dt = np.diff(t[: min(len(t), 100)])
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if len(dt) == 0:
        return 100.0
    return float(1.0 / np.median(dt))


def _clip_segment(length: int, start: int, end: int) -> tuple[int, int] | None:
    start = max(0, int(start))
    end = min(length, int(end))
    if end <= start + 1:
        return None
    return start, end


def _median_beat(values: np.ndarray, peaks: np.ndarray, pre: int, post: int) -> np.ndarray | None:
    beats = []
    for p in peaks:
        start = int(p) - pre
        end = int(p) + post
        if start < 0 or end > len(values):
            continue
        beats.append(values[start:end])
    if not beats:
        return None
    return np.median(np.stack(beats, axis=0), axis=0)


def _downsample(vec: np.ndarray, n: int) -> np.ndarray:
    if vec.size == n:
        return vec.astype(float)
    x_old = np.linspace(0.0, 1.0, vec.size)
    x_new = np.linspace(0.0, 1.0, n)
    return np.interp(x_new, x_old, vec).astype(float)


def extract_ecg_features_from_frame(
    df: pd.DataFrame,
    source: str = "<frame>",
    pre: int = 80,
    post: int = 120,
    downsample: int = 24,
) -> tuple[np.ndarray, list[str]]:
    if "Peak" not in df.columns:
        raise ValueError(f"Missing Peak column: {source}")
    peaks = df.index[df["Peak"] == 3].to_numpy(dtype=int)
    if len(peaks) < 3:
        raise ValueError(f"Too few peaks: {source}")

    fs = _fs_from_time(df)
    rr = np.diff(peaks) / fs
    rr_mean = float(np.mean(rr)) if len(rr) else 0.0
    rr_std = float(np.std(rr)) if len(rr) else 0.0
    base_values = [
        len(peaks),
        fs,
        rr_mean,
        rr_std,
        rr_std / max(rr_mean, 1e-6),
        60.0 / max(rr_mean, 1e-6),
    ]
    names = ["n_peaks", "fs", "rr_mean", "rr_std", "rr_cv", "heart_rate"]

    values: list[float] = list(base_values)
    length = pre + post
    segments = {
        "p": (pre - int(0.22 * fs), pre - int(0.08 * fs)),
        "pr": (pre - int(0.08 * fs), pre),
        "qrs": (pre - int(0.04 * fs), pre + int(0.10 * fs)),
        "st": (pre + int(0.10 * fs), pre + int(0.32 * fs)),
        "t": (pre + int(0.32 * fs), pre + int(0.72 * fs)),
    }

    for lead_i, col in enumerate(_lead_columns(df)):
        raw = df[col].to_numpy(dtype=float)
        beat = _median_beat(raw, peaks, pre=pre, post=post)
        if beat is None:
            raise ValueError(f"Cannot build beat: {source}")
        beat = beat - float(np.median(beat[: min(20, beat.size)]))
        std = float(np.std(beat))
        beat_z = (beat - float(np.mean(beat))) / max(std, 1e-6)

        prefix = f"lead{lead_i}_{col}"
        values.extend([float(np.mean(beat)), float(np.std(beat)), float(np.min(beat)), float(np.max(beat))])
        names.extend([f"{prefix}_mean", f"{prefix}_std", f"{prefix}_min", f"{prefix}_max"])

        for seg_name, (a, b) in segments.items():
            span = _clip_segment(length, a, b)
            if span is None:
                seg = beat_z
            else:
                seg = beat_z[span[0] : span[1]]
            values.extend(
                [
                    float(np.mean(seg)),
                    float(np.std(seg)),
                    float(np.min(seg)),
                    float(np.max(seg)),
                    float(np.trapezoid(np.abs(seg))),
                ]
            )
            names.extend(
                [
                    f"{prefix}_{seg_name}_mean",
                    f"{prefix}_{seg_name}_std",
                    f"{prefix}_{seg_name}_min",
                    f"{prefix}_{seg_name}_max",
                    f"{prefix}_{seg_name}_abs_area",
                ]
            )

        ds = _downsample(beat_z, downsample)
        values.extend(float(x) for x in ds)
        names.extend([f"{prefix}_shape_{i:02d}" for i in range(downsample)])

    return np.asarray(values, dtype=np.float32), names


def extract_ecg_features(csv_path: Path, pre: int = 80, post: int = 120, downsample: int = 24) -> tuple[np.ndarray, list[str]]:
    df = pd.read_csv(csv_path)
    return extract_ecg_features_from_frame(df, source=str(csv_path), pre=pre, post=post, downsample=downsample)


def build_feature_matrix(
    samples: Sequence[Sample],
    pre: int,
    post: int,
    downsample: int,
) -> tuple[np.ndarray, np.ndarray, list[str], list[Sample], list[str]]:
    xs: list[np.ndarray] = []
    ys: list[int] = []
    ok: list[Sample] = []
    names: list[str] | None = None
    failures: list[str] = []
    for sample in samples:
        try:
            x, n = extract_ecg_features(sample.path, pre=pre, post=post, downsample=downsample)
        except Exception as exc:
            failures.append(f"{sample.path.name}: {exc}")
            continue
        if names is None:
            names = n
        xs.append(x)
        ys.append(sample.label)
        ok.append(sample)
    if not xs:
        raise RuntimeError("No feature rows could be extracted.")
    return np.stack(xs, axis=0), np.asarray(ys, dtype=int), names or [], ok, failures


def csv_to_npy_path(csv_path: Path, csv_root: Path, npy_root: Path) -> Path:
    rel = csv_path.relative_to(csv_root)
    parts = list(rel.parts)
    if len(parts) >= 3 and parts[1] == "csv":
        parts[1] = "NumpyData"
    return npy_root.joinpath(*parts).with_suffix(".npy")
