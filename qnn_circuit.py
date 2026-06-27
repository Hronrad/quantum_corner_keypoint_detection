"""Data re-uploading QNN 线路模块。

本模块实现计划书中的 patch-level 量子神经网络分类器：

    angle encoding -> trainable rotations -> entanglement -> measurement -> linear readout

模型接收已经归一化并映射为角度的特征 ``Phi``，shape 为 ``(B, d)``，
输出 logits，shape 为 ``(B,)``。这里输出的是 logits，不是 sigmoid 概率；
训练时应直接送入 ``torch.nn.BCEWithLogitsLoss``。

第一版使用 PennyLane 的 ``default.qubit`` exact expectation，便于与 PyTorch 自动求导
训练循环对接。有限 shots 与噪声模型可以在后续版本扩展。
"""

from __future__ import annotations

import os
from typing import Iterable, Optional

# Windows + Anaconda 环境中，PyTorch、PennyLane/scipy 可能加载两份 Intel OpenMP
# runtime，导致仅仅 import/forward 就触发 "OMP Error #15"。这里设置兼容开关，
# 目的是保证交接代码在常见本地科研环境中可运行。若后续部署到干净虚拟环境或
# Linux 服务器，可以删除该环境变量并使用单一 OpenMP runtime。
if os.name == "nt":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pennylane as qml
import torch
from torch import nn


class DataReuploadingQNN(nn.Module):
    """低深度 data re-uploading 变分量子分类器。

    Args:
        n_qubits: 量子比特数。第一版建议等于输入特征维度 d。
        n_layers: data re-uploading 层数 L。
        encoding_type: ``"ry"`` 或 ``"ryrz"``。``"ryrz"`` 会对每个特征依次施加
            ``RY(phi_j)`` 和 ``RZ(phi_j)``。
        entanglement: ``"none"``, ``"linear"`` 或 ``"ring"``。
        readout: ``"single"`` 只使用 ``<Z0>``；``"all"`` 使用所有 qubit 的 ``<Zj>``；
            ``"all_zz"`` 同时使用所有 ``<Zj>`` 和相邻 qubit 的 ``<ZjZk>``。
        shots: ``None`` 表示 exact expectation；整数表示有限 shots 模拟。

    Input:
        Phi: ``torch.Tensor``，shape 为 ``(B, d)``。Phi 应该已经由
        ``FeatureNormalizer`` 转换到角度区间。

    Output:
        logits: ``torch.Tensor``，shape 为 ``(B,)``。需要概率时在外部调用
        ``torch.sigmoid(logits)``。
    """

    VALID_ENCODINGS = {"ry", "ryrz"}
    VALID_ENTANGLEMENTS = {"none", "linear", "ring"}
    VALID_READOUTS = {"single", "all", "all_zz"}

    def __init__(
        self,
        n_qubits: int,
        n_layers: int = 3,
        encoding_type: str = "ryrz",
        entanglement: str = "ring",
        readout: str = "all",
        shots: Optional[int] = None,
        trainable_input_scaling: bool = False,
        init_scale: float = 0.01,
    ) -> None:
        super().__init__()
        self.n_qubits = int(n_qubits)
        self.n_layers = int(n_layers)
        self.encoding_type = encoding_type
        self.entanglement = entanglement
        self.readout = readout
        self.shots = shots
        self.trainable_input_scaling = bool(trainable_input_scaling)
        self.init_scale = float(init_scale)

        self._validate_config()

        # theta[l, j, :] 对应第 l 层第 j 个 qubit 上的 RZ-RY-RZ 三个可训练角度。
        # 小随机初始化可以避免所有 qubit 在初始状态下完全对称。
        self.theta = nn.Parameter(self.init_scale * torch.randn(self.n_layers, self.n_qubits, 3))
        if self.trainable_input_scaling:
            self.input_scale_y = nn.Parameter(torch.ones(self.n_qubits))
            self.input_scale_z = nn.Parameter(torch.ones(self.n_qubits))
        else:
            self.register_buffer("input_scale_y", torch.ones(self.n_qubits))
            self.register_buffer("input_scale_z", torch.ones(self.n_qubits))

        readout_dim = self._readout_dim()
        self.readout_weights = nn.Parameter(torch.zeros(readout_dim))
        self.readout_bias = nn.Parameter(torch.zeros(()))

        self.device = qml.device("default.qubit", wires=self.n_qubits, shots=self.shots)
        self._qnode = self._build_qnode()

    def forward(self, Phi: torch.Tensor) -> torch.Tensor:
        """对一个 batch 的角度特征执行 QNN forward。

        Args:
            Phi: angle encoding 输入，shape 为 ``(B, d)``。当前实现要求
                ``d == n_qubits``。

        Returns:
            logits: shape 为 ``(B,)``，可直接传给 ``BCEWithLogitsLoss``。
        """

        if Phi.ndim != 2:
            raise ValueError(f"Phi must have shape (B, d), got {tuple(Phi.shape)}.")
        if Phi.shape[1] != self.n_qubits:
            raise ValueError(
                f"Input feature dimension d={Phi.shape[1]} must equal n_qubits={self.n_qubits}."
            )

        # PennyLane 的 QNode 在这里逐样本执行。这样写最直观，便于其他人对照线路公式。
        # 后续如果样本量很大，可以再做 vectorized/batched QNode 优化。
        z_values = []
        for sample_phi in Phi:
            expvals = self._qnode(sample_phi, self.theta, self.input_scale_y, self.input_scale_z)
            if self.readout == "single":
                z = torch.stack([expvals]) if expvals.ndim == 0 else expvals.reshape(1)
            else:
                z = torch.stack(list(expvals)) if isinstance(expvals, (list, tuple)) else expvals
            z_values.append(z.to(dtype=Phi.dtype))

        z_batch = torch.stack(z_values, dim=0)
        logits = z_batch @ self.readout_weights.to(dtype=Phi.dtype) + self.readout_bias.to(dtype=Phi.dtype)
        return logits.reshape(-1)

    def predict_proba(self, Phi: torch.Tensor) -> torch.Tensor:
        """返回 corner/keypoint 概率。

        训练时不建议调用本函数；训练应使用 logits 版本以获得更稳定的 BCE loss。
        """

        return torch.sigmoid(self.forward(Phi))

    def get_config(self) -> dict:
        """返回模型结构配置，便于 checkpoint 和实验日志保存。"""

        return {
            "n_qubits": self.n_qubits,
            "n_layers": self.n_layers,
            "encoding_type": self.encoding_type,
            "entanglement": self.entanglement,
            "readout": self.readout,
            "shots": self.shots,
            "trainable_input_scaling": self.trainable_input_scaling,
            "init_scale": self.init_scale,
        }

    def _build_qnode(self):
        """构造 PennyLane QNode。

        QNode 内部严格对应计划书中的每层结构：

        1. encoding: ``U_enc(phi)``；
        2. variational layer: ``U_var(theta_l)``；
        3. entanglement: ``U_ent``；
        4. measurement: 返回 Pauli-Z expectation。
        """

        @qml.qnode(self.device, interface="torch", diff_method="best")
        def circuit(phi: torch.Tensor, theta: torch.Tensor, scale_y: torch.Tensor, scale_z: torch.Tensor):
            # Data re-uploading：同一个输入 phi 在每一层重新作为门参数施加。
            # 这里没有重置量子态，也没有重新制备 |0...0>；每层都作用在当前态上。
            for layer in range(self.n_layers):
                self._apply_encoding(phi, scale_y, scale_z)
                self._apply_variational_layer(theta[layer])
                self._apply_entanglement()

            if self.readout == "single":
                return qml.expval(qml.PauliZ(0))
            if self.readout == "all_zz":
                z_terms = [qml.expval(qml.PauliZ(wire)) for wire in range(self.n_qubits)]
                zz_terms = [
                    qml.expval(qml.PauliZ(a) @ qml.PauliZ(b))
                    for a, b in self._neighbor_edges()
                ]
                return tuple(z_terms + zz_terms)
            return tuple(qml.expval(qml.PauliZ(wire)) for wire in range(self.n_qubits))

        return circuit

    def _apply_encoding(self, phi: torch.Tensor, scale_y: torch.Tensor, scale_z: torch.Tensor) -> None:
        """Angle encoding 层。

        ``ry``: 对每个 qubit 施加 ``RY(phi_j)``。
        ``ryrz``: 对每个 qubit 施加 ``RY(phi_j)`` 后再施加 ``RZ(phi_j)``。
        """

        for wire in range(self.n_qubits):
            qml.RY(scale_y[wire] * phi[wire], wires=wire)
            if self.encoding_type == "ryrz":
                qml.RZ(scale_z[wire] * phi[wire], wires=wire)

    def _apply_variational_layer(self, layer_theta: torch.Tensor) -> None:
        """可训练单比特旋转层，对应每个 qubit 的 RZ-RY-RZ。"""

        for wire in range(self.n_qubits):
            qml.RZ(layer_theta[wire, 0], wires=wire)
            qml.RY(layer_theta[wire, 1], wires=wire)
            qml.RZ(layer_theta[wire, 2], wires=wire)

    def _apply_entanglement(self) -> None:
        """纠缠层，支持 none、linear 和 ring 三种结构。"""

        if self.entanglement == "none":
            return

        for control, target in self._entanglement_edges():
            qml.CNOT(wires=[control, target])

    def _entanglement_edges(self) -> Iterable[tuple[int, int]]:
        """返回当前纠缠结构的 CNOT 边列表。"""

        if self.entanglement == "linear":
            return [(j, j + 1) for j in range(self.n_qubits - 1)]
        if self.entanglement == "ring":
            return [(j, (j + 1) % self.n_qubits) for j in range(self.n_qubits)]
        return []

    def _neighbor_edges(self) -> list[tuple[int, int]]:
        """返回 readout 使用的相邻 ZZ 测量边。"""

        if self.n_qubits == 1:
            return []
        return [(j, j + 1) for j in range(self.n_qubits - 1)] + [(self.n_qubits - 1, 0)]

    def _readout_dim(self) -> int:
        """返回当前 readout 的经典线性头输入维度。"""

        if self.readout == "single":
            return 1
        if self.readout == "all_zz":
            return self.n_qubits + len(self._neighbor_edges())
        return self.n_qubits

    def _validate_config(self) -> None:
        """检查模型配置是否合法，尽早暴露拼写或维度错误。"""

        if self.n_qubits <= 0:
            raise ValueError("n_qubits must be positive.")
        if self.n_layers <= 0:
            raise ValueError("n_layers must be positive.")
        if self.encoding_type not in self.VALID_ENCODINGS:
            raise ValueError(f"encoding_type must be one of {self.VALID_ENCODINGS}.")
        if self.entanglement not in self.VALID_ENTANGLEMENTS:
            raise ValueError(f"entanglement must be one of {self.VALID_ENTANGLEMENTS}.")
        if self.readout not in self.VALID_READOUTS:
            raise ValueError(f"readout must be one of {self.VALID_READOUTS}.")
