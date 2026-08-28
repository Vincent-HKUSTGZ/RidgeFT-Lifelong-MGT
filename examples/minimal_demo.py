"""Minimal end-to-end demo: base fit -> add a new class -> predict.

Synthetic Gaussian "embeddings" stand in for encoder outputs so the demo
runs in seconds on CPU and has no external dependency beyond numpy.
"""
from __future__ import annotations

import numpy as np

from ridgeft import RidgeFTModel


def _make_class(rng: np.random.Generator, mu: np.ndarray, n: int, sigma: float = 0.5) -> np.ndarray:
    return rng.standard_normal((n, mu.size)).astype(np.float32) * sigma + mu


def main() -> None:
    rng = np.random.default_rng(0)
    d_h = 64
    centers = rng.standard_normal((6, d_h)).astype(np.float32) * 1.5

    # ---- base classes 0..4: 200 samples each -------------------------
    H_base = np.concatenate([
        _make_class(rng, centers[c], 200) for c in range(5)
    ], axis=0)
    y_base = np.concatenate([np.full(200, c, dtype=np.int64) for c in range(5)])

    model = RidgeFTModel.fit_base(
        H_base, y_base,
        delta=0.5, shrinkage=0.05,
        total_dim=4096,
        beta=1.0, ridge_lam=1.0,
        seed=42,
    )

    # ---- evaluate on held-out base data ------------------------------
    H_test = np.concatenate([_make_class(rng, centers[c], 100) for c in range(5)])
    y_test = np.concatenate([np.full(100, c) for c in range(5)])
    acc_base = float(np.mean(model.predict(H_test) == y_test))
    print(f"[base] 5-class accuracy = {acc_base:.3f}")

    # ---- add a 6th class with only 30 (few-shot) samples -------------
    H_new = _make_class(rng, centers[5], 30)
    y_new = np.full(30, 5, dtype=np.int64)
    model.update_manyshot(H_new, y_new)

    H_test6 = np.concatenate([_make_class(rng, centers[c], 100) for c in range(6)])
    y_test6 = np.concatenate([np.full(100, c) for c in range(6)])
    acc6 = float(np.mean(model.predict(H_test6) == y_test6))
    print(f"[+1 class] 6-class accuracy = {acc6:.3f}")
    print("class ids:", model.class_ids)


if __name__ == "__main__":
    main()
