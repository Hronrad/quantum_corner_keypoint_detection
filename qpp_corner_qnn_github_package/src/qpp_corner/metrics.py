"""Binary classification metrics and validation threshold selection."""

from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def sigmoid(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(arr, -80, 80)))


def safe_roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return math.nan
    return float(roc_auc_score(y_true, scores))


def safe_pr_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return math.nan
    return float(average_precision_score(y_true, scores))


def choose_threshold_by_f1(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=np.float64)
    if len(s) == 0:
        raise ValueError("Cannot choose a threshold from an empty validation set.")
    candidates = np.unique(s)
    if len(candidates) > 512:
        candidates = np.quantile(s, np.linspace(0.0, 1.0, 512))
        candidates = np.unique(candidates)
    candidates = np.unique(np.concatenate([[s.min() - 1e-9], candidates, [s.max() + 1e-9]]))
    best_threshold = float(candidates[0])
    best_f1 = -1.0
    for threshold in candidates:
        pred = (s >= threshold).astype(int)
        score = f1_score(y, pred, zero_division=0)
        if score > best_f1:
            best_f1 = float(score)
            best_threshold = float(threshold)
    return best_threshold, best_f1


def binary_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=np.float64)
    pred = (s >= float(threshold)).astype(int)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": safe_roc_auc(y, s),
        "pr_auc": safe_pr_auc(y, s),
    }
