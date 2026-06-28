"""Training and evaluation helpers."""

from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np

from .metrics import binary_metrics, safe_pr_auc, sigmoid
from .qnn_torch import TORCH_AVAILABLE, require_torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def _torch_device(device: str = "auto"):
    require_torch()
    import torch

    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def predict_torch(model, x: np.ndarray, *, batch_size: int = 512, device: str = "auto") -> np.ndarray:
    require_torch()
    import torch

    dev = _torch_device(device)
    model = model.to(dev)
    model.eval()
    scores = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.as_tensor(x[start : start + batch_size], dtype=torch.float32, device=dev)
            logits = model(xb).detach().cpu().numpy()
            scores.append(sigmoid(logits))
    return np.concatenate(scores) if scores else np.asarray([], dtype=np.float64)


def _make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool):
    require_torch()
    import torch

    dataset = torch.utils.data.TensorDataset(
        torch.as_tensor(x, dtype=torch.float32),
        torch.as_tensor(y, dtype=torch.float32),
    )
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_torch_classifier(
    model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    out_dir: str | Path | None = None,
    lr: float = 3e-3,
    batch_size: int = 64,
    epochs: int = 50,
    patience: int = 8,
    monitor: str = "val_pr_auc",
    seed: int = 0,
    device: str = "auto",
) -> tuple[object, list[dict[str, float]]]:
    """Train a PyTorch binary classifier with early stopping."""

    require_torch()
    import torch

    set_seed(seed)
    dev = _torch_device(device)
    model = model.to(dev)
    loader = _make_loader(x_train, y_train, batch_size=batch_size, shuffle=True)
    y_mean = float(np.mean(y_train))
    pos_weight = None
    if 0.0 < y_mean < 1.0:
        pos_weight = torch.tensor([(1.0 - y_mean) / max(y_mean, 1e-6)], dtype=torch.float32, device=dev)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    out_path = Path(out_dir) if out_dir is not None else None
    if out_path is not None:
        out_path.mkdir(parents=True, exist_ok=True)
    best_value = -float("inf")
    best_state = None
    stale = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, int(epochs) + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            xb = xb.to(dev)
            yb = yb.to(dev)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        train_scores = predict_torch(model, x_train, batch_size=batch_size, device=str(dev))
        val_scores = predict_torch(model, x_val, batch_size=batch_size, device=str(dev))
        train_pr = safe_pr_auc(y_train, train_scores)
        val_pr = safe_pr_auc(y_val, val_scores)
        row = {
            "epoch": float(epoch),
            "loss": float(np.mean(losses)) if losses else float("nan"),
            "train_pr_auc": float(train_pr),
            "val_pr_auc": float(val_pr),
            "val_f1_at_0_5": binary_metrics(y_val, val_scores, 0.5)["f1"],
        }
        history.append(row)
        monitor_value = row.get(monitor, row["val_pr_auc"])
        if np.isnan(monitor_value):
            monitor_value = -float("inf")
        if monitor_value > best_value:
            best_value = float(monitor_value)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
            if out_path is not None:
                torch.save(best_state, out_path / "best_model.pt")
        else:
            stale += 1
            if stale >= int(patience):
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def write_history_csv(path: str | Path, history: list[dict[str, float]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not history:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
