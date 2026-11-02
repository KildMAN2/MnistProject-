"""Compatibility imports for the canonical implementation in :mod:`model`."""

from model import (
    DEFAULT_BETA,
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_NUM_STEPS,
    DEFAULT_THRESHOLD,
    INPUT_SIZE,
    NUM_CLASSES,
    MNISTSNN,
    count_output_spikes,
    encode_images,
    predict_from_spikes,
)

HIDDEN_SIZE = DEFAULT_HIDDEN_SIZE
BETA = DEFAULT_BETA
THRESHOLD = DEFAULT_THRESHOLD
NUM_STEPS = DEFAULT_NUM_STEPS

__all__ = [
    "BETA",
    "HIDDEN_SIZE",
    "INPUT_SIZE",
    "MNISTSNN",
    "NUM_CLASSES",
    "NUM_STEPS",
    "THRESHOLD",
    "count_output_spikes",
    "encode_images",
    "predict_from_spikes",
]
