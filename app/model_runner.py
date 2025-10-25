"""Lightweight wrappers around research models for UI usage."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from KSOM.ksom import KSOM


@dataclass
class TrainingReport:
    """Summary of a training invocation."""

    duration_s: float
    epochs: int
    steps: int
    success_rate: float


def train_ksom(
    data: np.ndarray,
    labels: np.ndarray,
    *,
    grid_size: int = 10,
    learning_rate: float = 0.1,
    radius: float = 1.0,
    epochs: int = 3,
    dataset_name: str = "UI_RUN",
    seed: Optional[int] = 42,
) -> Dict[str, object]:
    """Train a KSOM model and return the fitted instance with a report."""
    # Flatten samples once so the SOM receives consistent vectors.
    flattened = np.asarray([sample.flatten() for sample in data])

    if seed is not None:
        np.random.seed(seed)

    som = KSOM(
        grid_size=grid_size,
        dim=flattened.shape[1],
        learning_rate=learning_rate,
        radius=radius,
        max_iter=epochs,
        dataset_name=dataset_name,
    )

    start = time.time()
    som.train(flattened, labels)
    duration = time.time() - start

    report = TrainingReport(
        duration_s=duration,
        epochs=epochs,
        steps=som.step,
        success_rate=som.true_detect / max(1, som.step),
    )

    return {"model": som, "report": report}


def evaluate_ksom(som: KSOM, data: np.ndarray, labels: np.ndarray) -> Dict[str, object]:
    """Evaluate a trained KSOM model on a labelled dataset."""
    flattened = np.asarray([sample.flatten() for sample in data])

    predictions = []
    for row in flattened:
        predictions.append(som.predict(row))
    predictions = np.asarray(predictions)

    accuracy = float((predictions == labels).mean())
    return {"predictions": predictions, "accuracy": accuracy}
