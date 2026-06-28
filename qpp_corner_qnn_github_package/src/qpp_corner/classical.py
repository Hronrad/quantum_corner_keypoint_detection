"""Classical baselines for patch-level corner/keypoint classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


@dataclass
class SklearnScoreModel:
    model: Any

    def predict_scores(self, x: np.ndarray) -> np.ndarray:
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(x)[:, 1].astype(np.float64)
        if hasattr(self.model, "decision_function"):
            scores = self.model.decision_function(x)
            return (1.0 / (1.0 + np.exp(-np.clip(scores, -80, 80)))).astype(np.float64)
        raise TypeError("Model does not expose predict_proba or decision_function.")


@dataclass
class ThresholdScoreModel:
    sign: float = 1.0

    def predict_scores(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float64)
        if arr.ndim == 2:
            arr = arr[:, 0]
        return self.sign * arr


def fit_logistic(x_train: np.ndarray, y_train: np.ndarray, *, seed: int = 0, max_iter: int = 1000) -> SklearnScoreModel:
    model = LogisticRegression(max_iter=max_iter, class_weight="balanced", random_state=seed)
    model.fit(x_train, y_train)
    return SklearnScoreModel(model)


def fit_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    seed: int = 0,
    hidden_layer_sizes: tuple[int, ...] = (32, 16),
    max_iter: int = 300,
) -> SklearnScoreModel:
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=max_iter,
        early_stopping=True,
        random_state=seed,
    )
    model.fit(x_train, y_train)
    return SklearnScoreModel(model)


def threshold_baseline(sign: float = 1.0) -> ThresholdScoreModel:
    return ThresholdScoreModel(sign=sign)


def opencv_available() -> bool:
    try:
        import cv2  # noqa: F401
    except Exception:
        return False
    return True
