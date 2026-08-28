"""Lightweight smoke tests for the public API."""
from __future__ import annotations

import numpy as np

from ridgeft import (
    ClassBalancedRidgeClassifier,
    FractionalWhitening,
    RandomFeatureLift,
    RidgeFTModel,
)


def _make_data(n_classes: int, n_per_class: int, d_h: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_classes, d_h)).astype(np.float32) * 1.5
    H = np.concatenate([
        rng.standard_normal((n_per_class, d_h)).astype(np.float32) * 0.4 + centers[c]
        for c in range(n_classes)
    ])
    y = np.concatenate([np.full(n_per_class, c, dtype=np.int64) for c in range(n_classes)])
    return H, y


def test_pipeline_fit_predict_update():
    H, y = _make_data(n_classes=5, n_per_class=120, d_h=48)
    model = RidgeFTModel.fit_base(H, y, total_dim=512, seed=1)
    assert sorted(model.class_ids) == [0, 1, 2, 3, 4]

    H_new = np.random.default_rng(7).standard_normal((30, 48)).astype(np.float32) * 0.4 + 3.0
    y_new = np.full(30, 5, dtype=np.int64)
    model.update_manyshot(H_new, y_new)
    assert sorted(model.class_ids) == [0, 1, 2, 3, 4, 5]

    yhat = model.predict(H[:10])
    assert yhat.shape == (10,)
    assert yhat.dtype == np.int64


def test_module_shapes():
    H, y = _make_data(3, 50, 32)
    w = FractionalWhitening.fit(H, y, delta=0.5)
    Hw = w.transform(H)
    assert Hw.shape == H.shape

    rf = RandomFeatureLift.fit(Hw.shape[1], output_dim=256, seed=2)
    Phi = rf.transform(Hw)
    assert Phi.shape == (H.shape[0], 256)

    clf = ClassBalancedRidgeClassifier(beta=1.0).fit(Phi, y)
    assert clf.W is not None and clf.W.shape == (256, 3)
    assert clf.predict(Phi).shape == (H.shape[0],)


def test_balanced_beta_zero_matches_global_ridge():
    """beta=0 should reproduce a vanilla closed-form ridge to machine precision."""
    H, y = _make_data(4, 80, 24, seed=3)
    w = FractionalWhitening.fit(H, y, delta=0.5)
    rf = RandomFeatureLift.fit(H.shape[1], output_dim=128, seed=4)
    Phi = rf.transform(w.transform(H))

    cb = ClassBalancedRidgeClassifier(beta=0.0, lam=1.0).fit(Phi, y)
    A_bar = sum(cb.A_per_class[c] for c in cb.class_ids)
    B_full = np.zeros((128, len(cb.class_ids)), dtype=np.float64)
    for col, c in enumerate(sorted(cb.class_ids)):
        B_full[:, col] = cb.b_per_class[c]
    W_ref = np.linalg.solve(A_bar + np.eye(128), B_full).astype(np.float32)
    np.testing.assert_allclose(cb.W, W_ref, rtol=1e-4, atol=1e-4)
