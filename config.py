"""QNN 实验默认配置。

本文件只保存与 patch-level QNN 分类器相关的默认参数，不包含图像读取、
Harris/FAST/ORB 前处理或 NMS 后处理。其他脚本可以导入这些 dataclass，
也可以通过命令行参数覆盖默认值。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


DEFAULT_FEATURE_NAMES: List[str] = ["Ix", "Iy", "lambda1", "lambda2", "R"]


@dataclass
class QNNConfig:
    """Data re-uploading QNN 的结构配置。

    Attributes:
        n_qubits: 量子比特数。默认等于输入特征维度 d。
        n_layers: data re-uploading 层数 L。
        encoding_type: 输入编码方式，支持 ``"ry"`` 和 ``"ryrz"``。
        entanglement: 纠缠结构，支持 ``"none"``, ``"linear"``, ``"ring"``。
        readout: readout 方式，支持 ``"single"`` 和 ``"all"``。
        shots: ``None`` 表示 exact expectation；整数表示有限 shots 模拟。
    """

    n_qubits: int = 5
    n_layers: int = 3
    encoding_type: str = "ryrz"
    entanglement: str = "ring"
    readout: str = "all"
    shots: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """返回可 JSON 序列化的配置字典。"""

        return asdict(self)


@dataclass
class TrainConfig:
    """训练过程默认配置。

    Attributes:
        learning_rate: Adam 学习率。
        batch_size: mini-batch 大小。
        epochs: 最大训练轮数。
        seed: 随机种子，用于保证实验可复现。
        output_dir: 训练结果、checkpoint 和指标文件的保存目录。
        clip_value: 标准化后 z-score 的截断范围，最终角度落在 [-pi, pi]。
    """

    learning_rate: float = 1e-2
    batch_size: int = 64
    epochs: int = 80
    seed: int = 42
    output_dir: str = "outputs/default_run"
    clip_value: float = 3.0

    def to_dict(self) -> Dict[str, Any]:
        """返回可 JSON 序列化的配置字典。"""

        return asdict(self)


@dataclass
class DataConfig:
    """输入数据与特征列配置。

    Attributes:
        data_path: `.npz` 数据文件路径。
        feature_names: 当前实验使用的特征名，默认顺序为
            [Ix, Iy, lambda1, lambda2, R]。
    """

    data_path: str = "data/synthetic_corner_data.npz"
    feature_names: List[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.feature_names is None:
            self.feature_names = list(DEFAULT_FEATURE_NAMES)

    def to_dict(self) -> Dict[str, Any]:
        """返回可 JSON 序列化的配置字典。"""

        return asdict(self)
