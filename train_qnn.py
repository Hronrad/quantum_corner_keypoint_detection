"""QNN 训练入口与可复用训练函数。

本文件负责把 `.npz` 数据接口、FeatureNormalizer、DataReuploadingQNN、
loss、optimizer、验证指标和 checkpoint 保存串起来。它不负责图像读取、
梯度计算、Harris/FAST/ORB 特征提取；这些步骤应由前处理模块产出：

    X_train, y_train, X_val, y_val

然后本脚本把这些特征作为 patch-level 二分类数据训练 QNN。
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from config import DEFAULT_FEATURE_NAMES, QNNConfig, TrainConfig
from preprocessing import FeatureNormalizer
from qnn_circuit import DataReuploadingQNN


ArrayDataset = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
ArrayDatasetWithTest = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def set_random_seed(seed: int) -> None:
    """设置 Python、NumPy 和 PyTorch 随机种子，提升实验可复现性。"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_npz_dataset(path: str | Path) -> ArrayDataset:
    """读取第一版约定的 `.npz` 数据文件。

    Args:
        path: `.npz` 文件路径，必须包含 ``X_train``, ``y_train``,
            ``X_val``, ``y_val`` 四个字段。

    Returns:
        ``(X_train, y_train, X_val, y_val)``。X 的 shape 为 ``(N, d)``，
        y 的 shape 为 ``(N,)``。
    """

    data = np.load(path)
    required = ["X_train", "y_train", "X_val", "y_val"]
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"Dataset {path} is missing fields: {missing}.")

    X_train = np.asarray(data["X_train"], dtype=np.float32)
    y_train = np.asarray(data["y_train"], dtype=np.float32).reshape(-1)
    X_val = np.asarray(data["X_val"], dtype=np.float32)
    y_val = np.asarray(data["y_val"], dtype=np.float32).reshape(-1)

    _validate_dataset_shapes(X_train, y_train, X_val, y_val)
    return X_train, y_train, X_val, y_val


def load_npz_dataset_with_test(path: str | Path) -> ArrayDatasetWithTest:
    """读取包含 train/val/test 三个划分的 `.npz` 数据文件。"""

    data = np.load(path)
    required = ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test"]
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"Dataset {path} is missing fields: {missing}.")

    X_train = np.asarray(data["X_train"], dtype=np.float32)
    y_train = np.asarray(data["y_train"], dtype=np.float32).reshape(-1)
    X_val = np.asarray(data["X_val"], dtype=np.float32)
    y_val = np.asarray(data["y_val"], dtype=np.float32).reshape(-1)
    X_test = np.asarray(data["X_test"], dtype=np.float32)
    y_test = np.asarray(data["y_test"], dtype=np.float32).reshape(-1)

    _validate_dataset_shapes_with_test(X_train, y_train, X_val, y_val, X_test, y_test)
    return X_train, y_train, X_val, y_val, X_test, y_test


def select_feature_columns(X: np.ndarray, feature_indices: Optional[Sequence[int]]) -> np.ndarray:
    """按列选择当前实验使用的特征。

    Args:
        X: 原始特征矩阵，shape 为 ``(N, d_total)``。
        feature_indices: 需要保留的列索引。``None`` 表示保留全部列。

    Returns:
        子特征矩阵，shape 为 ``(N, d_selected)``。
    """

    if feature_indices is None:
        return X
    return X[:, list(feature_indices)]


def make_dataloader(Phi: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    """把角度特征和标签包装为 PyTorch DataLoader。

    Phi 已经是 angle encoding 输入，shape 为 ``(N, d)``；y 是二分类标签，
    shape 为 ``(N,)``。
    """

    dataset = TensorDataset(
        torch.as_tensor(Phi, dtype=torch.float32),
        torch.as_tensor(y, dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def compute_pos_weight(y_train: np.ndarray) -> torch.Tensor:
    """根据训练标签计算 BCEWithLogitsLoss 的 pos_weight。

    PyTorch 中 ``pos_weight = num_negative / num_positive``，用于提高正样本
    角点/keypoint 的损失权重。若某个 toy 数据恰好没有正样本，则回退为 1。
    """

    positives = float(np.sum(y_train == 1))
    negatives = float(np.sum(y_train == 0))
    if positives <= 0:
        return torch.tensor(1.0, dtype=torch.float32)
    return torch.tensor(negatives / positives, dtype=torch.float32)


def compute_binary_metrics(y_true: np.ndarray, logits: np.ndarray) -> Dict[str, float]:
    """计算 patch-level 二分类指标。

    Args:
        y_true: 真实标签，shape 为 ``(N,)``。
        logits: 模型输出 logits，shape 为 ``(N,)``。

    Returns:
        指标字典。ROC-AUC 或 PR-AUC 在单类别验证集上无法定义时返回 NaN。
    """

    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= 0.5).astype(np.int64)
    y_int = y_true.astype(np.int64)

    metrics = {
        "accuracy": float(accuracy_score(y_int, preds)),
        "precision": float(precision_score(y_int, preds, zero_division=0)),
        "recall": float(recall_score(y_int, preds, zero_division=0)),
        "f1": float(f1_score(y_int, preds, zero_division=0)),
    }

    try:
        metrics["roc_auc"] = float(roc_auc_score(y_int, probs))
    except ValueError:
        metrics["roc_auc"] = float("nan")

    try:
        metrics["pr_auc"] = float(average_precision_score(y_int, probs))
    except ValueError:
        metrics["pr_auc"] = float("nan")

    return metrics


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, Dict[str, float], np.ndarray]:
    """在验证集上计算 loss、指标和 logits。

    Returns:
        ``(avg_loss, metrics, logits)``。
    """

    model.eval()
    total_loss = 0.0
    total_count = 0
    all_logits: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []

    with torch.no_grad():
        for Phi_batch, y_batch in dataloader:
            Phi_batch = Phi_batch.to(device)
            y_batch = y_batch.to(device)
            logits = model(Phi_batch)
            loss = criterion(logits, y_batch)

            total_loss += float(loss.item()) * y_batch.numel()
            total_count += y_batch.numel()
            all_logits.append(logits.detach().cpu().numpy())
            all_targets.append(y_batch.detach().cpu().numpy())

    logits_np = np.concatenate(all_logits, axis=0)
    targets_np = np.concatenate(all_targets, axis=0)
    metrics = compute_binary_metrics(targets_np, logits_np)
    return total_loss / max(total_count, 1), metrics, logits_np


def train_qnn_experiment(
    data_path: str | Path,
    output_dir: str | Path,
    qnn_config: QNNConfig,
    train_config: TrainConfig,
    feature_indices: Optional[Sequence[int]] = None,
    feature_names: Optional[Sequence[str]] = None,
    device_name: str = "auto",
) -> Dict[str, float]:
    """训练一个 QNN 实验并保存结果。

    Args:
        data_path: `.npz` 数据文件。
        output_dir: 输出目录，保存 normalizer、best model、metrics 和 config。
        qnn_config: QNN 结构配置。
        train_config: 训练超参数配置。
        feature_indices: 当前实验使用的特征列。``None`` 表示使用全部列。
        feature_names: 与特征列对应的可读名称，写入日志方便交接。
        device_name: ``"auto"``, ``"cpu"`` 或 ``"cuda"``。

    Returns:
        best validation metrics 字典。
    """

    set_random_seed(train_config.seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        X_train, y_train, X_val, y_val, X_test, y_test = load_npz_dataset_with_test(data_path)
        test_split_source = "test"
    except KeyError:
        X_train, y_train, X_val, y_val = load_npz_dataset(data_path)
        X_test, y_test = X_val.copy(), y_val.copy()
        test_split_source = "val_fallback"
    X_train = select_feature_columns(X_train, feature_indices)
    X_val = select_feature_columns(X_val, feature_indices)
    X_test = select_feature_columns(X_test, feature_indices)

    # 第一版采用 n_qubits = input_dim，避免特征消融时做 padding 或隐式丢列。
    input_dim = X_train.shape[1]
    qnn_config.n_qubits = input_dim

    normalizer = FeatureNormalizer(clip_value=train_config.clip_value)
    Phi_train = normalizer.fit_transform(X_train)
    Phi_val = normalizer.transform(X_val)
    Phi_test = normalizer.transform(X_test)
    normalizer.save(output_path / "normalizer.npz")

    train_loader = make_dataloader(Phi_train, y_train, train_config.batch_size, shuffle=True)
    val_loader = make_dataloader(Phi_val, y_val, train_config.batch_size, shuffle=False)
    test_loader = make_dataloader(Phi_test, y_test, train_config.batch_size, shuffle=False)

    device = _select_device(device_name)
    model = DataReuploadingQNN(**qnn_config.to_dict()).to(device)
    pos_weight = compute_pos_weight(y_train).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config.learning_rate)

    history: List[Dict[str, float]] = []
    best_f1 = -1.0
    best_metrics: Dict[str, float] = {}
    best_state_dict: Dict[str, torch.Tensor] | None = None

    for epoch in range(1, train_config.epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0

        for Phi_batch, y_batch in train_loader:
            Phi_batch = Phi_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(Phi_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * y_batch.numel()
            total_count += y_batch.numel()

        train_loss = total_loss / max(total_count, 1)
        val_loss, val_metrics, _ = evaluate_model(model, val_loader, criterion, device)
        row = {
            "epoch": float(epoch),
            "train_loss": train_loss,
            "val_loss": val_loss,
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(row)

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_metrics = {"val_loss": val_loss, **val_metrics}
            best_state_dict = copy.deepcopy(model.state_dict())
            torch.save(
                {
                    "model_state_dict": best_state_dict,
                    "qnn_config": qnn_config.to_dict(),
                    "train_config": train_config.to_dict(),
                    "feature_indices": list(feature_indices) if feature_indices is not None else None,
                    "feature_names": list(feature_names) if feature_names is not None else None,
                    "best_metrics": best_metrics,
                },
                output_path / "best_model.pt",
            )

        print(
            f"epoch={epoch:03d} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_f1={val_metrics['f1']:.4f} val_pr_auc={val_metrics['pr_auc']:.4f}"
        )

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    test_loss, test_metrics, _ = evaluate_model(model, test_loader, criterion, device)
    test_metrics = {"test_loss": test_loss, **test_metrics}

    _save_json(
        output_path / "metrics.json",
        {
            "best": best_metrics,
            "test": test_metrics,
            "test_split_source": test_split_source,
            "history": history,
        },
    )
    _save_json(
        output_path / "config.json",
        {
            "data_path": str(data_path),
            "qnn_config": qnn_config.to_dict(),
            "train_config": train_config.to_dict(),
            "feature_indices": list(feature_indices) if feature_indices is not None else None,
            "feature_names": list(feature_names) if feature_names is not None else None,
            "pos_weight": float(pos_weight.detach().cpu()),
            "test_split_source": test_split_source,
        },
    )
    return {"best": best_metrics, "test": test_metrics}


def parse_feature_indices(text: Optional[str]) -> Optional[List[int]]:
    """解析命令行传入的逗号分隔特征列索引，例如 ``0,1,4``。"""

    if text is None or text.strip() == "":
        return None
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def main() -> None:
    """命令行入口：训练默认 QNN 或自定义 QNN 配置。"""

    parser = argparse.ArgumentParser(description="Train a patch-level data re-uploading QNN.")
    parser.add_argument("--data", default="data/synthetic_corner_data.npz", help="Path to .npz dataset.")
    parser.add_argument("--output-dir", default="outputs/default_run", help="Directory to save outputs.")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--encoding", choices=["ry", "ryrz"], default="ryrz")
    parser.add_argument("--entanglement", choices=["none", "linear", "ring"], default="ring")
    parser.add_argument("--readout", choices=["single", "all", "all_zz"], default="all")
    parser.add_argument("--shots", type=int, default=None)
    parser.add_argument("--trainable-input-scaling", action="store_true")
    parser.add_argument("--init-scale", type=float, default=0.01)
    parser.add_argument("--feature-indices", default=None, help="Comma-separated feature columns, e.g. 0,1,4.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    feature_indices = parse_feature_indices(args.feature_indices)
    feature_names = _names_from_indices(DEFAULT_FEATURE_NAMES, feature_indices)
    qnn_config = QNNConfig(
        n_qubits=5,
        n_layers=args.layers,
        encoding_type=args.encoding,
        entanglement=args.entanglement,
        readout=args.readout,
        shots=args.shots,
        trainable_input_scaling=args.trainable_input_scaling,
        init_scale=args.init_scale,
    )
    train_config = TrainConfig(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        seed=args.seed,
        output_dir=args.output_dir,
    )

    train_qnn_experiment(
        data_path=args.data,
        output_dir=args.output_dir,
        qnn_config=qnn_config,
        train_config=train_config,
        feature_indices=feature_indices,
        feature_names=feature_names,
        device_name=args.device,
    )


def _validate_dataset_shapes(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> None:
    """检查 `.npz` 中训练/验证特征和标签 shape 是否匹配。"""

    if X_train.ndim != 2 or X_val.ndim != 2:
        raise ValueError("X_train and X_val must have shape (N, d).")
    if y_train.ndim != 1 or y_val.ndim != 1:
        raise ValueError("y_train and y_val must have shape (N,).")
    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError("X_train and y_train have inconsistent sample counts.")
    if X_val.shape[0] != y_val.shape[0]:
        raise ValueError("X_val and y_val have inconsistent sample counts.")
    if X_train.shape[1] != X_val.shape[1]:
        raise ValueError("X_train and X_val must have the same feature dimension.")


def _validate_dataset_shapes_with_test(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """检查 train/val/test 特征和标签 shape 是否匹配。"""

    _validate_dataset_shapes(X_train, y_train, X_val, y_val)
    if X_test.ndim != 2:
        raise ValueError("X_test must have shape (N, d).")
    if y_test.ndim != 1:
        raise ValueError("y_test must have shape (N,).")
    if X_test.shape[0] != y_test.shape[0]:
        raise ValueError("X_test and y_test have inconsistent sample counts.")
    if X_test.shape[1] != X_train.shape[1]:
        raise ValueError("X_test must have the same feature dimension as X_train.")


def _select_device(device_name: str) -> torch.device:
    """选择 PyTorch 设备。PennyLane exact simulator 通常 CPU 已足够。"""

    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _save_json(path: str | Path, payload: dict) -> None:
    """保存 JSON 文件，允许 NaN 指标以便记录单类别验证集情况。"""

    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, allow_nan=True)


def _names_from_indices(names: Sequence[str], indices: Optional[Sequence[int]]) -> List[str]:
    """根据特征列索引生成日志里的特征名。"""

    if indices is None:
        return list(names)
    return [names[i] if i < len(names) else f"feature_{i}" for i in indices]


if __name__ == "__main__":
    main()
