"""Tiny numerical helpers for RidgeFT."""
from __future__ import annotations

from typing import Iterable

import numpy as np


def as_float32(x: np.ndarray) -> np.ndarray:
    """Return a C-contiguous float32 array."""
    return np.asarray(x, dtype=np.float32, order="C")


def as_float64(x: np.ndarray) -> np.ndarray:
    """Return a C-contiguous float64 array."""
    return np.asarray(x, dtype=np.float64, order="C")


def check_2d(name: str, x: np.ndarray) -> np.ndarray:
    """Validate and return a 2-D float32 matrix."""
    x = as_float32(x)
    if x.ndim != 2:
        raise ValueError(f"{name} must be 2-D, got shape {x.shape}")
    return x


def check_labels(y: np.ndarray) -> np.ndarray:
    """Validate and return a one-dimensional int64 label array."""
    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError(f"labels must be 1-D, got shape {y.shape}")
    if not np.issubdtype(y.dtype, np.integer):
        raise ValueError("RidgeFT expects integer class ids. "
                         "Encode string labels before training.")
    return y.astype(np.int64, copy=False)


def row_layer_norm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Layer-normalize each row independently.

    ``y_i = (x_i - mean(x_i)) / (std(x_i) + eps)``.
    """
    x = as_float32(x)
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True) + eps
    return as_float32((x - mean) / std)


def unique_sorted(y: Iterable[int]) -> list[int]:
    """Return sorted unique integer labels."""
    return sorted(int(v) for v in np.unique(np.asarray(list(y))))
