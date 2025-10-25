"""Utility helpers for loading datasets and experiment logs."""
from __future__ import annotations

import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .config import DATA_ROOT, LOG_ROOT

_LABEL_PATTERN = re.compile(r"_(\d)_")


@dataclass
class DatasetSplit:
    """Container holding data and labels for a dataset split."""

    name: str
    data: np.ndarray
    labels: np.ndarray
    source_path: Path


def list_environments() -> List[str]:
    """Return all dataset environment directories that contain Numpy data."""
    environments: List[str] = []
    for child in sorted(DATA_ROOT.iterdir()):
        if not child.is_dir():
            continue
        if (child / "NumpyData").exists():
            environments.append(child.name)
    return environments


def load_numpy_samples(
    env_name: str,
    sample_size: Optional[int] = None,
    seed: Optional[int] = 42,
) -> DatasetSplit:
    """Load ECG numpy samples for a given environment.

    Parameters
    ----------
    env_name: str
        Name of the environment folder, e.g. "Env1" or "Eval".
    sample_size: Optional[int]
        Limit the number of samples returned. None keeps all samples.
    seed: Optional[int]
        Seed for reproducible sampling when sample_size is specified.
    """
    env_path = DATA_ROOT / env_name / "NumpyData"
    if not env_path.exists():
        raise FileNotFoundError(f"Environment '{env_name}' not found at {env_path}")

    files = sorted(f for f in env_path.iterdir() if f.suffix == ".npy")
    if not files:
        raise RuntimeError(f"No .npy files found in {env_path}")

    if sample_size is not None and sample_size < len(files):
        rng = random.Random(seed)
        files = rng.sample(files, sample_size)

    data: List[np.ndarray] = []
    labels: List[int] = []

    for file_path in files:
        arr = np.load(file_path)
        data.append(arr)
        label = _extract_label(file_path.name)
        labels.append(label)

    stacked = np.asarray(data)
    label_array = np.asarray(labels)
    return DatasetSplit(name=env_name, data=stacked, labels=label_array, source_path=env_path)


def summarize_environment(env_name: str) -> Dict[str, object]:
    """Return quick statistics for an environment without loading arrays."""
    env_path = DATA_ROOT / env_name / "NumpyData"
    if not env_path.exists():
        raise FileNotFoundError(f"Environment '{env_name}' not found at {env_path}")

    counts = Counter()
    total = 0
    for file_path in env_path.iterdir():
        if file_path.suffix != ".npy":
            continue
        total += 1
        label = _extract_label(file_path.name)
        counts[label] += 1
    return {"total": total, "label_distribution": counts}


def _extract_label(filename: str) -> int:
    match = _LABEL_PATTERN.search(filename)
    if not match:
        raise ValueError(f"Cannot parse label from '{filename}'")
    return int(match.group(1))


@dataclass
class LogRun:
    """Metadata for an experiment log run."""

    experiment: str
    run_id: str
    step_files: Sequence[Path]


def list_log_runs(experiment: Optional[str] = None) -> List[LogRun]:
    """Enumerate available log runs under Thesis_log.

    Parameters
    ----------
    experiment: Optional[str]
        Filter runs by experiment folder name, e.g. "DatasetA".
    """
    results: List[LogRun] = []
    experiment_dirs = [LOG_ROOT / experiment] if experiment else LOG_ROOT.iterdir()

    for exp_dir in experiment_dirs:
        if not exp_dir.is_dir():
            continue
        if experiment and exp_dir.name != experiment:
            continue
        for run_dir in sorted(exp_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            step_files = sorted(run_dir.glob("*.json"))
            if not step_files:
                continue
            results.append(LogRun(experiment=exp_dir.name, run_id=run_dir.name, step_files=step_files))
    return results


def load_log_file(log_path: Path) -> List[Dict[str, float]]:
    """Load a single JSON log file containing metric snapshots."""
    with log_path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"Unexpected log structure in {log_path}")
    return data
