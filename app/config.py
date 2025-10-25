"""Shared configuration for the interactive UI tier."""
from pathlib import Path

# Root of the dataset relative to repository
DATA_ROOT = Path(__file__).resolve().parent.parent / "Data" / "Disease_dataset"
# Location that stores experiment logs exported during training
LOG_ROOT = Path(__file__).resolve().parent.parent / "Thesis_log"

# Default evaluation split name used across helpers
EVAL_ENV = "Eval"

# For models that need to persist artifacts (temporary cache, etc.)
ARTIFACT_ROOT = Path(__file__).resolve().parent.parent / "artifacts"
ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
