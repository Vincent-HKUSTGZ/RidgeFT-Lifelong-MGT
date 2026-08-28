"""RidgeFT: exemplar-free analytical continual MGT attribution.

Public API
----------

    from ridgeft import RidgeFTModel
    model = RidgeFTModel.fit_base(H_base, y_base)
    model.update_manyshot(H_new, y_new)
    y_hat = model.predict(H_test)
"""
from .classifier import ClassBalancedRidgeClassifier
from .model import RidgeFTModel
from .random_features import RandomFeatureLift
from .spectral import FractionalWhitening

__all__ = [
    "RidgeFTModel",
    "FractionalWhitening",
    "RandomFeatureLift",
    "ClassBalancedRidgeClassifier",
]

__version__ = "1.0.0"
