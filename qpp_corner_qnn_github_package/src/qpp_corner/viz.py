"""Visualization helpers for experiment outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import precision_recall_curve


def _plt():
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - depends on optional plotting stack.
        raise RuntimeError(f"matplotlib is required for plots: {exc}") from exc
    return plt


def plot_pr_curve(y_true, scores, path: str | Path, *, title: str = "Precision-Recall") -> None:
    plt = _plt()
    precision, recall, _ = precision_recall_curve(y_true, scores)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(recall, precision)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_training_curves(history: list[dict[str, float]], path: str | Path) -> None:
    if not history:
        return
    plt = _plt()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]
    fig, ax = plt.subplots(figsize=(6, 4))
    if "loss" in history[0]:
        ax.plot(epochs, [row["loss"] for row in history], label="loss")
    if "val_pr_auc" in history[0]:
        ax.plot(epochs, [row["val_pr_auc"] for row in history], label="val PR-AUC")
    if "train_pr_auc" in history[0]:
        ax.plot(epochs, [row["train_pr_auc"] for row in history], label="train PR-AUC")
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_feature_scatter(x: np.ndarray, y: np.ndarray, feature_names: list[str], path: str | Path) -> None:
    if x.shape[1] < 2:
        return
    plt = _plt()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    y = np.asarray(y).astype(int)
    ax.scatter(x[y == 0, 0], x[y == 0, 1], s=8, alpha=0.45, label="negative")
    ax.scatter(x[y == 1, 0], x[y == 1, 1], s=8, alpha=0.65, label="positive")
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.legend()
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_patch_preview(patches: np.ndarray, labels: np.ndarray, scores: np.ndarray, path: str | Path, *, n: int = 12) -> None:
    plt = _plt()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = min(int(n), len(patches))
    if n <= 0:
        return
    cols = min(6, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.7, rows * 1.9))
    axes = np.asarray(axes).reshape(-1)
    order = np.argsort(scores)[::-1][:n]
    for ax, idx in zip(axes, order):
        ax.imshow(patches[idx], cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"y={int(labels[idx])} p={scores[idx]:.2f}", fontsize=8)
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
