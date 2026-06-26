"""生成用于 QNN 训练链路自检的 synthetic patch-level 数据。

这个脚本不试图模拟完整图像生成与真实角点标注，只生成与默认特征顺序一致的
五维特征：

    [Ix, Iy, lambda1, lambda2, R]

正样本被设计成两个 structure tensor 特征值都较大、Harris response 较高；
负样本包含平坦区域和边缘区域。它的目的只是帮助确认：

1. `.npz` 数据接口正确；
2. FeatureNormalizer 能工作；
3. QNN 训练脚本能端到端跑通。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np


def generate_synthetic_corner_data(
    n_train: int = 512,
    n_val: int = 256,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """生成 toy 训练集和验证集。

    Args:
        n_train: 训练样本数。
        n_val: 验证样本数。
        seed: 随机种子。

    Returns:
        ``X_train, y_train, X_val, y_val``，其中 X shape 为 ``(N, 5)``，
        y shape 为 ``(N,)``。
    """

    rng = np.random.default_rng(seed)
    X_train, y_train = _make_split(rng, n_train)
    X_val, y_val = _make_split(rng, n_val)
    return X_train, y_train, X_val, y_val


def main() -> None:
    """命令行入口，写出符合训练脚本接口的 `.npz` 文件。"""

    parser = argparse.ArgumentParser(description="Generate synthetic QNN corner/keypoint data.")
    parser.add_argument("--output", default="data/synthetic_corner_data.npz")
    parser.add_argument("--n-train", type=int, default=512)
    parser.add_argument("--n-val", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    X_train, y_train, X_val, y_val = generate_synthetic_corner_data(args.n_train, args.n_val, args.seed)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)
    print(f"saved synthetic dataset to {output_path}")
    print(f"X_train={X_train.shape}, y_train={y_train.shape}, X_val={X_val.shape}, y_val={y_val.shape}")


def _make_split(rng: np.random.Generator, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """生成一个数据划分；正负样本比例约为 1:3，模拟角点稀少。"""

    y = (rng.random(n_samples) < 0.25).astype(np.float32)
    X = np.zeros((n_samples, 5), dtype=np.float32)

    for i, label in enumerate(y):
        if label == 1:
            # corner/keypoint：两个方向梯度都较明显，lambda1/lambda2 都大，R 高。
            ix = rng.normal(1.0, 0.45)
            iy = rng.normal(1.0, 0.45)
            lambda1 = rng.normal(2.2, 0.35)
            lambda2 = rng.normal(1.8, 0.35)
            response = lambda1 * lambda2 - 0.04 * (lambda1 + lambda2) ** 2 + rng.normal(0.0, 0.12)
        else:
            if rng.random() < 0.5:
                # flat：梯度和特征值都小。
                ix = rng.normal(0.0, 0.25)
                iy = rng.normal(0.0, 0.25)
                lambda1 = rng.normal(0.25, 0.12)
                lambda2 = rng.normal(0.15, 0.08)
            else:
                # edge：一个方向强、另一个方向弱，lambda2 小。
                ix = rng.normal(1.2, 0.45)
                iy = rng.normal(0.1, 0.25)
                lambda1 = rng.normal(1.8, 0.35)
                lambda2 = rng.normal(0.2, 0.10)
            response = lambda1 * lambda2 - 0.04 * (lambda1 + lambda2) ** 2 + rng.normal(0.0, 0.12)

        X[i] = np.array([ix, iy, max(lambda1, 0.0), max(lambda2, 0.0), response], dtype=np.float32)

    # 打乱顺序，避免 batch 中标签按生成顺序聚集。
    order = rng.permutation(n_samples)
    return X[order], y[order]


if __name__ == "__main__":
    main()
