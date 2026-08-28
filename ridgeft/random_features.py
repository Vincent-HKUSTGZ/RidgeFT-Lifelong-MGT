r"""Isotropic ReLU random feature lift (Module 2: Random Feature Expansion).

Given a calibrated embedding ``tilde h \in R^{d_h}`` produced by
:class:`ridgeft.spectral.FractionalWhitening`, the lift is

    phi(x) = LayerNorm( ReLU( R tilde h ) )    in R^{d_phi},
    R_ij ~ N(0, 1/d_h),     d_phi = 4096.

This is parameter-free, fixed once at base time, and never updated.
The high-dimensional ReLU+LN representation is what gives the
downstream closed-form ridge enough capacity to express non-linear
generator-discriminative boundaries.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .utils import as_float32, check_2d, row_layer_norm


@dataclass
class RandomFeatureLift:
    """Fixed isotropic Gaussian ReLU + LayerNorm random feature map."""

    R: np.ndarray
    seed: int = 42

    @classmethod
    def fit(
        cls,
        input_dim: int,
        *,
        output_dim: int = 4096,
        seed: int = 42,
    ) -> "RandomFeatureLift":
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if output_dim <= 0:
            raise ValueError("output_dim must be positive")
        rng = np.random.default_rng(int(seed) + 1009)
        R = rng.standard_normal((int(output_dim), int(input_dim))).astype(np.float32)
        R *= math.sqrt(1.0 / max(int(input_dim), 1))
        return cls(R=R, seed=int(seed))

    @property
    def input_dim(self) -> int:
        return int(self.R.shape[1])

    @property
    def output_dim(self) -> int:
        return int(self.R.shape[0])

    def transform(self, H: np.ndarray) -> np.ndarray:
        """phi(H) = LayerNorm( ReLU( H R^T ) ) row-wise."""
        H = check_2d("H", H)
        if H.shape[1] != self.input_dim:
            raise ValueError(
                f"feature dimension mismatch: expected {self.input_dim}, "
                f"got {H.shape[1]}")
        Z = np.maximum(H @ self.R.T, 0.0)
        return row_layer_norm(as_float32(Z))
