"""End-to-end RidgeFT pipeline.

Three frozen analytical stages:

    h          : encoder embedding  (d_h)
    tilde h    = FractionalWhitening(delta=0.5).transform(h)         [Module 1]
    phi(x)     = LayerNorm( ReLU( R tilde h ) )    in R^{d_phi=4096} [Module 2]
    y_hat      = argmax_c (phi(x)^T W)_c                              [Module 3]

The encoder is fine-tuned on base classes and then *frozen forever*.
All three modules (whitening basis, R, ridge weights) are computed in
closed form. Adding a new generator class amounts to a single
:py:meth:`update_manyshot` call which only touches the per-class
sufficient statistics ``(A_c, b_c, N_c)`` and re-solves the
``(d_phi x d_phi)`` ridge system.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .classifier import ClassBalancedRidgeClassifier
from .random_features import RandomFeatureLift
from .spectral import FractionalWhitening
from .utils import check_2d, check_labels


@dataclass
class RidgeFTModel:
    whitening: FractionalWhitening
    rf: RandomFeatureLift
    ridge: ClassBalancedRidgeClassifier
    config: dict

    # ------------------------------------------------------------------
    @classmethod
    def fit_base(
        cls,
        H_base: np.ndarray,
        y_base: np.ndarray,
        *,
        # Module 1 — covariance calibration
        delta: float = 0.5,
        shrinkage: float = 0.05,
        # Module 2 — random feature lift
        total_dim: int = 4096,
        # Module 3 — class-balanced ridge
        beta: float = 1.0,
        tau_smoothing: float = 0.0,
        ridge_lam: float = 1.0,
        # numerical / reproducibility
        seed: int = 42,
        eps: float = 1e-5,
    ) -> "RidgeFTModel":
        """Fit the three RidgeFT stages on the base-class embeddings.

        ``H_base`` is the ``(N, d_h)`` encoder-output matrix on the base
        classes; ``y_base`` are integer class ids.
        """
        H_base = check_2d("H_base", H_base)
        y_base = check_labels(y_base)
        if H_base.shape[0] != y_base.shape[0]:
            raise ValueError(
                f"H_base and y_base length mismatch: "
                f"{H_base.shape[0]} != {y_base.shape[0]}")

        # --- Module 1 --------------------------------------------------
        whitening = FractionalWhitening.fit(
            H_base, y_base,
            delta=delta, shrinkage=shrinkage, eps=eps)
        H_white = whitening.transform(H_base)

        # --- Module 2 --------------------------------------------------
        rf = RandomFeatureLift.fit(
            H_white.shape[1], output_dim=total_dim, seed=seed)
        Phi_base = rf.transform(H_white)

        # --- Module 3 --------------------------------------------------
        ridge = ClassBalancedRidgeClassifier(
            lam=ridge_lam, beta=beta, tau_smoothing=tau_smoothing,
        ).fit(Phi_base, y_base)

        cfg = dict(
            delta=float(delta), shrinkage=float(shrinkage),
            total_dim=int(total_dim),
            beta=float(beta), tau_smoothing=float(tau_smoothing),
            ridge_lam=float(ridge_lam),
            seed=int(seed), eps=float(eps),
        )
        return cls(whitening=whitening, rf=rf, ridge=ridge, config=cfg)

    # ------------------------------------------------------------------
    def transform(self, H: np.ndarray) -> np.ndarray:
        """Encoder embeddings -> RidgeFT random features ``phi(x)``."""
        return self.rf.transform(self.whitening.transform(H))

    def update_manyshot(self, H_new: np.ndarray, y_new: np.ndarray) -> "RidgeFTModel":
        """Add one or more new classes via per-class sufficient stats.

        Only ``(A_c, b_c, N_c)`` of the new class(es) are computed and
        merged; old per-class statistics are kept byte-identical, then the
        ``(d_phi x d_phi)`` ridge system is re-solved in closed form.
        """
        H_new = check_2d("H_new", H_new)
        y_new = check_labels(y_new)
        if H_new.shape[0] != y_new.shape[0]:
            raise ValueError(
                f"H_new and y_new length mismatch: "
                f"{H_new.shape[0]} != {y_new.shape[0]}")
        self.ridge.add_manyshot(self.transform(H_new), y_new)
        return self

    # ------------------------------------------------------------------
    def scores(self, H: np.ndarray) -> np.ndarray:
        return self.ridge.scores(self.transform(H))

    def predict(self, H: np.ndarray) -> np.ndarray:
        return self.ridge.predict(self.transform(H))

    @property
    def class_ids(self) -> list[int]:
        return list(self.ridge.class_ids)

    def diagnostics(self) -> dict:
        return {
            **self.config,
            "ridge_info": self.ridge.info(),
            "rf_input_dim": self.rf.input_dim,
            "rf_output_dim": self.rf.output_dim,
        }
