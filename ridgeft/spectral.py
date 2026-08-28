"""Within-class fractional whitening (Module 1: Covariance Calibration)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .utils import as_float32, check_2d, check_labels


@dataclass
class FractionalWhitening:
    r"""Trace-shrunk within-class fractional whitening.

    Given base-class embeddings ``H \in R^{N x d}`` and labels ``y``, we
    estimate the within-class covariance

        S_w = (1/(N-C)) * sum_c sum_{i: y_i=c} (h_i - mu_c)(h_i - mu_c)^T,

    apply trace-scaled shrinkage

        S_w^{shrink} = (1 - alpha) S_w + alpha * (tr(S_w)/d) * I,

    eigen-decompose ``S_w^{shrink} = U diag(sigma_j) U^T``, and define the
    fractional whitening map

        tilde h = U diag((sigma_j + eps)^{-delta}) U^T (h - mu).

    Defaults
    --------
    * ``delta = 0.5``    — full standard whitening on the shrunk covariance.
    * ``shrinkage = 0.05`` — light Ledoit-Wolf-style shrinkage that stabilises
      the eigen-decomposition without touching the dominant directions.
    """

    mean: np.ndarray
    eigvals: np.ndarray
    eigvecs: np.ndarray
    delta: float = 0.5
    shrinkage: float = 0.05
    eps: float = 1e-5

    @classmethod
    def fit(
        cls,
        H: np.ndarray,
        y: np.ndarray,
        *,
        delta: float = 0.5,
        shrinkage: float = 0.05,
        eps: float = 1e-5,
    ) -> "FractionalWhitening":
        H = check_2d("H", H)
        y = check_labels(y)
        if H.shape[0] != y.shape[0]:
            raise ValueError(
                f"H and y length mismatch: {H.shape[0]} != {y.shape[0]}")

        X = H.astype(np.float64, copy=False)
        mean = X.mean(axis=0)
        d = X.shape[1]
        sw = np.zeros((d, d), dtype=np.float64)
        denom = 0
        for c in sorted(np.unique(y)):
            Xc = X[y == c]
            if Xc.shape[0] == 0:
                continue
            muc = Xc.mean(axis=0)
            centered = Xc - muc
            sw += centered.T @ centered
            denom += max(Xc.shape[0] - 1, 1)
        sw /= max(denom, 1)

        trace_scale = float(np.trace(sw) / max(d, 1))
        if not np.isfinite(trace_scale) or trace_scale <= 0.0:
            trace_scale = 1.0
        alpha = float(np.clip(shrinkage, 0.0, 1.0))
        sw = (1.0 - alpha) * sw + alpha * trace_scale * np.eye(d, dtype=np.float64)

        vals, vecs = np.linalg.eigh((sw + sw.T) * 0.5)
        vals = np.maximum(vals, eps)
        return cls(
            mean=mean.astype(np.float32),
            eigvals=vals.astype(np.float32),
            eigvecs=vecs.astype(np.float32),
            delta=float(delta),
            shrinkage=alpha,
            eps=float(eps),
        )

    # ------------------------------------------------------------------
    def transform(self, H: np.ndarray, *, delta: float | None = None) -> np.ndarray:
        """Apply fractional whitening to a batch of encoder embeddings.

        Parameters
        ----------
        H        : (N, d) embedding matrix.
        delta    : optional override of the calibrated exponent.
        """
        H = check_2d("H", H)
        power = self.delta if delta is None else float(delta)
        centered = H.astype(np.float32, copy=False) - self.mean
        coeff = centered @ self.eigvecs
        coeff *= np.power(self.eigvals + self.eps, -power).astype(np.float32)
        return as_float32(coeff @ self.eigvecs.T)
