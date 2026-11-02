"""Canonical repository paths, independent of the current working directory."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
CHECKPOINT_DIR = REPO_ROOT / "artifacts" / "checkpoints"
INT16_DIR = REPO_ROOT / "artifacts" / "int16"
DEFAULT_CHECKPOINT = CHECKPOINT_DIR / "mnist_snn_static.pt"
