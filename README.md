# 面向角点与关键点检测的 QNN 模块

本仓库实现一个 patch-level QNN 二分类器，用于判断候选中心点是否为
corner/keypoint。第一版只负责 QNN 模型搭建、训练、验证和消融实验，不负责
完整图像前处理。

换句话说，前处理模块需要先产出局部特征矩阵和标签；本仓库接收这些矩阵后训练：

```text
X_train, y_train, X_val, y_val -> FeatureNormalizer -> DataReuploadingQNN -> logits
```

模型输出的是 logits。需要概率时再做 `sigmoid(logits)`；训练时直接使用
`BCEWithLogitsLoss`，不要在模型内部提前 sigmoid。

## 环境安装

建议使用 Python 3.10 或 3.11。

```bash
pip install -r requirements.txt
```

第一版依赖：

- `PennyLane`：构建 data re-uploading VQC。
- `PyTorch`：训练循环、loss、optimizer 和 checkpoint。
- `scikit-learn`：分类指标和经典 baseline。
- `numpy/pandas`：数据读写与实验汇总。

## 数据格式

默认读取 `.npz` 文件，必须包含四个字段：

```text
X_train: shape (N_train, d)
y_train: shape (N_train,)
X_val:   shape (N_val, d)
y_val:   shape (N_val,)
```

第一版默认 `d=5`，特征顺序固定为：

```text
[Ix, Iy, lambda1, lambda2, R]
```

含义：

- `Ix`：候选中心点附近的水平梯度。
- `Iy`：候选中心点附近的竖直梯度。
- `lambda1`：局部 structure tensor 的较大特征值。
- `lambda2`：局部 structure tensor 的较小特征值。
- `R`：Harris response。

如果其他同学负责图像前处理，只需要确保最终保存出上述 `.npz` 文件即可。

## 快速运行

先生成 toy 数据，检查训练链路：

```bash
python generate_synthetic_data.py --output data/synthetic_corner_data.npz
```

训练默认 QNN：

```bash
python train_qnn.py --data data/synthetic_corner_data.npz --output-dir outputs/default_run
```

运行精简消融实验：

```bash
python run_ablation.py --data data/synthetic_corner_data.npz --output-dir outputs/ablation
```

训练输出包括：

- `best_model.pt`：验证集 F1 最好的模型 checkpoint。
- `normalizer.npz`：训练集拟合得到的归一化参数。
- `metrics.json`：每个 epoch 的 loss 和验证指标。
- `config.json`：本次实验的模型配置、训练配置和特征列。

## 模块说明

### `FeatureNormalizer`

文件：`preprocessing.py`

职责：

1. 在训练集上计算每一维特征的均值和标准差。
2. 对训练集、验证集和测试集使用同一组统计量。
3. 将 z-score 截断到 `[-3, 3]`。
4. 映射为角度 `phi = pi / 3 * z`，因此输出范围为 `[-pi, pi]`。
5. 保存和加载归一化参数。

典型用法：

```python
from preprocessing import FeatureNormalizer

normalizer = FeatureNormalizer()
Phi_train = normalizer.fit_transform(X_train)
Phi_val = normalizer.transform(X_val)
normalizer.save("outputs/default_run/normalizer.npz")
```

注意：只能在训练集上 `fit`，验证集和测试集必须只调用 `transform`，否则会造成数据泄漏。

### `DataReuploadingQNN`

文件：`qnn_circuit.py`

职责：实现低深度 data re-uploading VQC。

默认结构：

```text
angle encoding -> RZ-RY-RZ trainable rotations -> entanglement
```

并在每一层重复输入同一个 `phi`。这就是 data re-uploading。它不是重置量子态，也不是重新制备输入态，而是在当前态上继续施加由同一组经典特征控制的编码门。

支持配置：

- `encoding_type="ry"` 或 `"ryrz"`
- `entanglement="none"`、`"linear"` 或 `"ring"`
- `readout="single"` 或 `"all"`
- `n_layers=1/2/3/...`
- `shots=None` 表示 exact expectation

输入输出：

```text
input:  Phi, shape (B, d)
output: logits, shape (B,)
```

第一版采用 `n_qubits = d`。例如默认五维特征对应 5 个 qubit。

### `train_qnn.py`

职责：

1. 读取 `.npz` 数据。
2. 选择特征列。
3. 拟合 `FeatureNormalizer`。
4. 构建 `DataReuploadingQNN`。
5. 使用 `BCEWithLogitsLoss(pos_weight=neg/pos)` 训练。
6. 输出 patch-level 指标。
7. 保存 best model、normalizer、metrics 和 config。

常用参数：

```bash
python train_qnn.py \
  --data data/synthetic_corner_data.npz \
  --output-dir outputs/default_run \
  --layers 3 \
  --encoding ryrz \
  --entanglement ring \
  --batch-size 64 \
  --epochs 80
```

只用 `[Ix, Iy]` 两维特征训练：

```bash
python train_qnn.py \
  --data data/synthetic_corner_data.npz \
  --output-dir outputs/gradients_only \
  --feature-indices 0,1
```

### `run_ablation.py`

职责：运行一个规模可控的消融矩阵，并保存 `summary.csv`。

默认比较：

- 特征组：`[Ix, Iy]` 与 `[Ix, Iy, lambda1, lambda2, R]`
- 编码：`ry` 与 `ryrz`
- 纠缠：`none`、`linear`、`ring`
- 层数：`L=1` 与 `L=3`
- baseline：logistic regression 与 MLP with same features

如果 QNN 不能超过使用相同输入特征的 MLP，不能声称 QNN 有性能优势，只能作为量子线路结构基准或负结果分析。

## 指标

第一阶段报告 patch-level 分类指标：

- accuracy
- precision
- recall
- F1
- ROC-AUC
- PR-AUC

由于角点样本通常远少于非角点，PR-AUC、F1、precision、recall 比 accuracy 更重要。

第二阶段接入真实 keypoint pipeline 后，可以再报告：

- repeatability
- localization error
- matching score
- number of detected keypoints
- RANSAC inlier ratio

## 对接真实前处理模块

前处理模块只需要保存：

```python
import numpy as np

np.savez(
    "data/real_features.npz",
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
)
```

其中 `X_*` 的列顺序必须和实验配置一致。默认顺序为：

```text
[Ix, Iy, lambda1, lambda2, R]
```

如果后续扩展到八维特征：

```text
[Ix, Iy, Ix^2, Iy^2, IxIy, lambda1, lambda2, R]
```

需要同步调整：

1. `.npz` 中 `X_*` 的列顺序；
2. `run_ablation.py` 中的特征组索引；
3. 实验报告里的 feature config。

## 从 logits 到 keypoints

本仓库第一版只输出 patch-level logits。真实 keypoint 检测阶段可以在外部做：

```text
logits -> sigmoid -> threshold -> NMS -> keypoint set
```

NMS、空间均匀化、descriptor matching 和 SLAM 前端验证不在第一版范围内。

## 实验记录建议

每次实验至少保存：

- 使用的数据文件路径；
- 特征列和特征顺序；
- QNN 配置：qubit 数、层数、encoding、entanglement、readout；
- 训练配置：seed、学习率、batch size、epoch；
- best checkpoint；
- PR-AUC、F1、precision、recall；
- 是否与 logistic regression / MLP 做了相同特征对比。

## Windows/Anaconda 排障

如果在手动导入 `torch` 和 `pennylane` 时遇到：

```text
OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
```

这是 Windows + Anaconda 环境中常见的 OpenMP runtime 冲突。`qnn_circuit.py`
已经在 Windows 下设置 `KMP_DUPLICATE_LIB_OK=TRUE`，保证训练脚本和正常导入路径可运行。
如果你自己写交互式脚本，建议先导入本仓库模块，再导入 `torch`：

```python
from qnn_circuit import DataReuploadingQNN
import torch
```

如果必须先导入 `torch`，就在脚本最顶部手动设置：

```python
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
```

更干净的长期做法是新建独立虚拟环境，只安装一套 PyTorch/PennyLane 依赖。

## 目录结构

```text
.
├── config.py
├── preprocessing.py
├── qnn_circuit.py
├── train_qnn.py
├── run_ablation.py
├── generate_synthetic_data.py
├── requirements.txt
└── README.md
```
