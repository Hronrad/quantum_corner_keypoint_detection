"""特征归一化与角度映射模块。

QNN 线路中的 angle encoding 不能直接使用未缩放的图像特征。这个模块负责：

1. 只在训练集上估计每一维特征的均值和标准差；
2. 使用同一组统计量变换训练集、验证集和测试集；
3. 将标准化后的特征截断并映射为量子旋转门角度；
4. 保存和加载归一化参数，方便其他前处理模块与 QNN 模块对接。

注意：归一化参数必须只从训练集拟合，不能从验证集或测试集重新拟合，否则会造成
数据泄漏。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class FeatureNormalizer:
    """将经典局部特征转换为 angle encoding 可用的角度特征。

    输入 shape:
        X: ``(N, d)``，N 是样本数，d 是特征维度。

    输出 shape:
        Phi: ``(N, d)``，每个元素是量子旋转角，范围为 ``[-pi, pi]``。

    变换流程:
        ``X -> z-score -> clip(-clip_value, clip_value) -> angle_scale * z``。

    Args:
        clip_value: z-score 截断范围。默认 3.0。
        angle_scale: 角度缩放因子。默认 pi / 3，使 clip 后范围为 [-pi, pi]。
        eps: 防止标准差为 0 的小常数。
    """

    clip_value: float = 3.0
    angle_scale: float = np.pi / 3.0
    eps: float = 1e-8
    mean_: Optional[np.ndarray] = None
    std_: Optional[np.ndarray] = None

    def fit(self, X_train: np.ndarray) -> "FeatureNormalizer":
        """在训练集上计算并保存均值和标准差。

        Args:
            X_train: 训练特征矩阵，shape 为 ``(N_train, d)``。

        Returns:
            self，便于链式调用。
        """

        X = self._validate_2d_array(X_train, name="X_train")
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """使用已拟合的训练集统计量将特征映射为量子旋转角。

        Args:
            X: 待变换特征矩阵，shape 为 ``(N, d)``。

        Returns:
            Phi: angle encoding 特征矩阵，shape 为 ``(N, d)``，
            dtype 为 ``float32``，元素范围为 ``[-pi, pi]``。

        Raises:
            RuntimeError: 如果尚未调用 ``fit`` 或 ``load``。
        """

        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("FeatureNormalizer must be fitted or loaded before transform().")

        X_arr = self._validate_2d_array(X, name="X")
        if X_arr.shape[1] != self.mean_.shape[0]:
            raise ValueError(
                f"Feature dimension mismatch: got d={X_arr.shape[1]}, "
                f"expected d={self.mean_.shape[0]}."
            )

        z = (X_arr - self.mean_) / (self.std_ + self.eps)
        z = np.clip(z, -self.clip_value, self.clip_value)
        phi = self.angle_scale * z
        return phi.astype(np.float32)

    def fit_transform(self, X_train: np.ndarray) -> np.ndarray:
        """先在训练集上拟合统计量，再返回训练集角度特征。"""

        return self.fit(X_train).transform(X_train)

    def save(self, path: str | Path) -> None:
        """保存归一化参数。

        Args:
            path: 输出 `.npz` 文件路径。文件中包含 mean、std、clip_value、
                angle_scale 和 eps。
        """

        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Cannot save FeatureNormalizer before fit().")

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            output_path,
            mean=self.mean_,
            std=self.std_,
            clip_value=np.array(self.clip_value, dtype=np.float64),
            angle_scale=np.array(self.angle_scale, dtype=np.float64),
            eps=np.array(self.eps, dtype=np.float64),
        )

    @classmethod
    def load(cls, path: str | Path) -> "FeatureNormalizer":
        """从 `.npz` 文件加载归一化参数。

        Args:
            path: 由 ``save`` 写出的 `.npz` 文件路径。

        Returns:
            已拟合状态的 ``FeatureNormalizer``。
        """

        data = np.load(path)
        normalizer = cls(
            clip_value=float(data["clip_value"]),
            angle_scale=float(data["angle_scale"]),
            eps=float(data["eps"]),
        )
        normalizer.mean_ = data["mean"].astype(np.float64)
        normalizer.std_ = data["std"].astype(np.float64)
        return normalizer

    @staticmethod
    def _validate_2d_array(X: np.ndarray, name: str) -> np.ndarray:
        """检查输入是否为二维数值数组，并转换为 float64 便于统计计算。"""

        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"{name} must be a 2D array with shape (N, d), got {arr.shape}.")
        if arr.shape[0] == 0 or arr.shape[1] == 0:
            raise ValueError(f"{name} must not be empty, got {arr.shape}.")
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} contains NaN or infinite values.")
        return arr
