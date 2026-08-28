"""Class-balanced closed-form ridge classifier.

Module 3 of RidgeFT: per-class sufficient statistics ``(A_c, b_c, N_c)``
re-weighted by ``omega_c = (N_c + tau)^{-beta}`` at solve time, normalised
so that ``mean_c omega_c = 1`` (this keeps the effective regularisation
scale ``lambda`` fixed when the class balance changes).

Mathematically:

    A_c        = Phi_c^T Phi_c        (per-class second-moment matrix)
    b_c        = sum_{i: y_i=c} phi_i (per-class linear sum)
    N_c        = number of class-c samples
    omega_c    = (N_c + tau)^{-beta} / mean_{c'} (N_{c'} + tau)^{-beta}
    A_bar      = sum_c omega_c A_c
    B_bar[:c]  = omega_c b_c
    W          = (A_bar + lambda I)^{-1} B_bar
    y_hat(x)   = argmax_c (phi(x)^T W)_c

When ``beta = 0`` this reduces exactly to the global ridge over all
samples (so the module is *free* on balanced streams). When ``beta > 0``
the per-class weights amplify under-represented classes, preventing the
"statistical swamping" problem that hits closed-form CIL when a new
generator arrives in few-shot mode.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .utils import as_float32, check_2d, check_labels, unique_sorted


@dataclass
class ClassBalancedRidgeClassifier:
    lam: float = 1.0
    beta: float = 1.0
    tau_smoothing: float = 0.0
    class_ids: list[int] = field(default_factory=list)
    A_per_class: dict[int, np.ndarray] = field(default_factory=dict)
    b_per_class: dict[int, np.ndarray] = field(default_factory=dict)
    n_per_class: dict[int, int] = field(default_factory=dict)
    W: np.ndarray | None = None
    n_features_: int = 0

    # ------------------------------------------------------------------
    def _accumulate(self, Phi: np.ndarray, y: np.ndarray) -> None:
        Phi = check_2d("Phi", Phi)
        y = check_labels(y)
        if Phi.shape[0] != y.shape[0]:
            raise ValueError(
                f"Phi and y length mismatch: {Phi.shape[0]} != {y.shape[0]}")
        if self.n_features_ == 0:
            self.n_features_ = int(Phi.shape[1])
        elif Phi.shape[1] != self.n_features_:
            raise ValueError(
                f"feature dim mismatch: expected {self.n_features_}, "
                f"got {Phi.shape[1]}")
        X = Phi.astype(np.float64, copy=False)
        for c in unique_sorted(y):
            Xc = X[y == c]
            n_c = int(Xc.shape[0])
            if n_c == 0:
                continue
            Ac = Xc.T @ Xc
            bc = Xc.sum(axis=0)
            if c in self.A_per_class:
                self.A_per_class[c] += Ac
                self.b_per_class[c] += bc
                self.n_per_class[c] += n_c
            else:
                self.A_per_class[c] = Ac
                self.b_per_class[c] = bc
                self.n_per_class[c] = n_c
                if c not in self.class_ids:
                    self.class_ids.append(int(c))

    def _omega(self) -> dict[int, float]:
        ids = sorted(self.class_ids)
        beta = float(self.beta)
        tau = float(self.tau_smoothing)
        raw = {c: (float(self.n_per_class[c]) + tau) ** (-beta) for c in ids}
        mean_w = float(np.mean(list(raw.values()))) or 1.0
        return {c: raw[c] / mean_w for c in ids}

    def solve(self) -> None:
        if not self.class_ids:
            raise RuntimeError("no classes accumulated")
        d = self.n_features_
        omega = self._omega()
        ids_sorted = sorted(self.class_ids)
        self.class_ids = list(ids_sorted)
        A_bar = np.zeros((d, d), dtype=np.float64)
        B_bar = np.zeros((d, len(ids_sorted)), dtype=np.float64)
        for col, c in enumerate(ids_sorted):
            w = omega[c]
            A_bar += w * self.A_per_class[c]
            B_bar[:, col] = w * self.b_per_class[c]
        reg = A_bar + float(self.lam) * np.eye(d, dtype=np.float64)
        self.W = np.linalg.solve(reg, B_bar).astype(np.float32)

    # ------------------------------------------------------------------
    def fit(
        self,
        Phi: np.ndarray,
        y: np.ndarray,
        class_ids: list[int] | None = None,
    ) -> "ClassBalancedRidgeClassifier":
        y = check_labels(y)
        if class_ids is not None:
            for c in class_ids:
                if int(c) not in self.class_ids:
                    self.class_ids.append(int(c))
        self._accumulate(Phi, y)
        self.solve()
        return self

    def add_manyshot(
        self,
        Phi_new: np.ndarray,
        y_new: np.ndarray,
    ) -> "ClassBalancedRidgeClassifier":
        if not self.A_per_class:
            raise RuntimeError("fit must be called before add_manyshot")
        self._accumulate(Phi_new, y_new)
        self.solve()
        return self

    # ------------------------------------------------------------------
    def scores(self, Phi: np.ndarray) -> np.ndarray:
        if self.W is None:
            raise RuntimeError("classifier is not fitted")
        Phi = check_2d("Phi", Phi)
        if Phi.shape[1] != self.W.shape[0]:
            raise ValueError(
                f"feature dim mismatch: expected {self.W.shape[0]}, "
                f"got {Phi.shape[1]}")
        return as_float32(Phi) @ self.W

    def predict(self, Phi: np.ndarray) -> np.ndarray:
        ids = np.asarray(sorted(self.class_ids), dtype=np.int64)
        return ids[self.scores(Phi).argmax(axis=1)]

    def info(self) -> dict:
        return {
            "beta": float(self.beta),
            "tau_smoothing": float(self.tau_smoothing),
            "lam": float(self.lam),
            "n_per_class": {int(c): int(v) for c, v in self.n_per_class.items()},
            "omega": self._omega() if self.class_ids else {},
        }
