# 面向角点与关键点检测的 QNN 模块

本仓库实现一个 patch-level QNN 二分类器，用于判断候选中心点是否为
corner/keypoint。第一版只负责 QNN 模型搭建、训练、验证和消融实验，不负责
完整图像前处理。

换句话说，前处理模块需要先产出局部特征矩阵和标签；本仓库接收这些矩阵后训练：

```text
.
├── README.md
├── .gitignore
├── config.py
├── generate_synthetic_data.py
├── preprocessing.py
├── qnn_circuit.py
├── requirements.txt
├── quantum_corner_keypoint_detection.pdf
├── run_ablation.py
├── train_qnn.py
├── outputs/
│   ├── day1_overlay_comparison.png
│   ├── baseline_metrics.json
│   ├── baseline_metrics.png
│   ├── fast_overlay.png
│   ├── harris_overlay.png
│   ├── mlp_metrics.json
│   ├── mlp_overlay.png
│   ├── orb_overlay.png
│   └── mlp_training_curve.png
├── qcd_data/
│   ├── __init__.py
│   ├── baselines.py
│   ├── features.py
│   ├── synthetic.py
│   └── visualize.py
└── scripts/
  ├── generate_synthetic_dataset.py
  └── run_day1_baselines.py
```

- `qcd_data/synthetic.py`: synthetic scene generation, label generation, patch extraction, and a PyTorch `Dataset` wrapper.
- `qcd_data/features.py`: 9-D patch feature extraction for MLP/QNN-compatible inputs.
- `qcd_data/baselines.py`: Harris, FAST, point matching metrics, and overlay plotting helpers. It uses OpenCV when available and NumPy fallbacks otherwise.
- `qcd_data/visualize.py`: preview-grid rendering for generated samples.
- `scripts/generate_synthetic_dataset.py`: command-line tool for writing image/label datasets to disk.
- `scripts/run_day1_baselines.py`: one-command Day 1 pipeline for synthetic data, patches, features, Harris/FAST, MLP, and overlay figures.
- `outputs/`: checked-in Day 1 baseline result figures and metrics.
- `quantum_corner_keypoint_detection.pdf`: project paper/report artifact.
- `data/`: generated datasets are written here by default and intentionally ignored by Git.

## Install

```bash
python -m pip install -r requirements.txt
```

`opencv-python` is included so the Day 1 baseline script can run native Harris, FAST, and ORB detectors. The Harris/FAST helpers still include NumPy fallbacks for lightweight smoke tests, but ORB evaluation requires OpenCV.

## Generate Data

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

## Day 1 Classical Baselines
```text
├── README.md
├── .gitignore
├── config.py
├── generate_synthetic_data.py
├── preprocessing.py
├── qnn_circuit.py
├── requirements.txt
├── quantum_corner_keypoint_detection.pdf
├── run_ablation.py
├── train_qnn.py
├── outputs/
│   ├── day1_overlay_comparison.png
│   ├── baseline_metrics.json
│   ├── baseline_metrics.png
│   ├── fast_overlay.png
│   ├── harris_overlay.png
│   ├── mlp_metrics.json
│   ├── mlp_overlay.png
│   ├── orb_overlay.png
│   └── mlp_training_curve.png
├── qcd_data/
│   ├── __init__.py
│   ├── baselines.py
│   ├── features.py
│   ├── synthetic.py
│   └── visualize.py
└── scripts/
  ├── generate_synthetic_dataset.py
  └── run_day1_baselines.py
```
- `outputs/baselines/day1/figures/day1_overlay_comparison.png`: GT/Harris/FAST/ORB/MLP comparison figure.
- `outputs/baselines/day1/metrics/baseline_metrics.json`: Precision/Recall/F1 for Harris, FAST, ORB, and MLP over all 300 synthetic images.
- `outputs/baselines/day1/figures/baseline_metrics.png`: bar-chart summary of the same baseline metrics.

Current full-image baseline evaluation from the checked-in run:

```text
method   precision   recall   f1
Harris   0.1666      0.9600   0.2839
FAST     0.0666      0.8733   0.1238
ORB      0.0296      1.0000   0.0576
MLP      0.1219      0.9067   0.2149
```

![Day 1 overlay comparison](outputs/baselines/day1/figures/day1_overlay_comparison.png)
![Day 1 baseline metrics](outputs/baselines/day1/figures/baseline_metrics.png)

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
├── README.md
├── .gitignore
├── requirements.txt
├── quantum_corner_keypoint_detection.pdf
├── outputs/
│   ├── day1_overlay_comparison.png
│   ├── baseline_metrics.json
│   ├── baseline_metrics.png
│   ├── fast_overlay.png
│   ├── harris_overlay.png
│   ├── mlp_metrics.json
│   ├── mlp_overlay.png
│   ├── orb_overlay.png
│   └── mlp_training_curve.png
├── qcd_data/
│   ├── __init__.py
│   ├── baselines.py
│   ├── features.py
│   ├── synthetic.py
│   └── visualize.py
└── scripts/
    ├── generate_synthetic_dataset.py
    └── run_day1_baselines.py
```

- `qcd_data/synthetic.py`：合成场景生成、标签生成、patch 提取，以及 PyTorch `Dataset` 封装。
- `qcd_data/features.py`：提取 9 维 patch 特征，供 MLP/QNN 使用同一输入接口。
- `qcd_data/baselines.py`：Harris、FAST、点匹配指标和 overlay 绘图工具；有 OpenCV 时优先使用 OpenCV，没有时使用 NumPy 版本作为回退。
- `qcd_data/visualize.py`：生成样本的预览图绘制工具。
- `scripts/generate_synthetic_dataset.py`：将合成图像和标签写入磁盘的命令行脚本。
- `scripts/run_day1_baselines.py`：一键运行第一天闭环，包括合成数据、patch、特征、Harris/FAST、MLP 和结果图。
- `outputs/`：已纳入仓库的第一天 baseline 展示图和指标。
- `quantum_corner_keypoint_detection.pdf`：项目论文/报告文件。
- `data/`：默认的数据集输出目录，已被 Git 忽略。

## 安装依赖

```bash
python -m pip install -r requirements.txt
```

`opencv-python` 已列入依赖，用于运行原生 Harris、FAST 和 ORB baseline。Harris/FAST 工具函数仍保留 NumPy 回退，方便轻量 smoke test；但 ORB 评估需要 OpenCV。

## 生成数据

```bash
python scripts/generate_synthetic_dataset.py --count 200 --out data/synthetic_keypoints --seed 7
```

如需生成矩形图像，可显式指定宽度和高度：

```bash
python scripts/generate_synthetic_dataset.py --count 200 --width 192 --height 96 --out data/synthetic_keypoints_192x96
```

如需生成更小且带模糊的数据，可减小 patch 尺寸并增加模糊强度：

```bash
python scripts/generate_synthetic_dataset.py --count 64 --width 48 --height 48 --patch-size 15 --line-width-min 1 --line-width-max 3 --blur-probability 1.0 --blur-radius-min 1.2 --blur-radius-max 2.0 --noise-std-max 0.04 --out data/synthetic_keypoints_small_blur
```

脚本会输出：

- `images/sample_XXXXXX.png`：灰度合成图像。
- `labels/sample_XXXXXX.npz`：稠密标签和 patch 级标签。
- `manifest.jsonl`：图像/标签路径以及场景元数据。
- `preview.png`：叠加关键点的快速预览图。
- `config.json`：数据生成参数。

`--image-size 128` 是方形图像的快捷参数。当需要非方形图像时，`--width` 和 `--height` 会覆盖它。

## 标签格式

每个 `.npz` 标签文件包含：

- `image`：形状为 `(H, W)` 的 float32 数组，取值范围为 `[0, 1]`。
- `points_xy`：形状为 `(N, 2)` 的 float32 数组，按 `(x, y)` 像素坐标存储。
- `type_ids`：形状为 `(N,)` 的 int64 数组；`1=corner`，`2=t_junction`，`3=x_junction`。
- `heatmap`：形状为 `(H, W)` 的 float32 数组，由关键点周围的高斯响应取最大值合成。
- `patch_centers_xy`：形状为 `(P, 2)` 的 float32 数组。
- `patch_labels`：形状为 `(P,)` 的 int64 数组，用于 patch 级二分类。
- `patches`：形状为 `(P, patch_size, patch_size)` 的 float32 数组。
- `scene_type`：合成场景类别。

## 当前合成场景

- `l_corner`：L 形角点。
- `t_junction`：T 形连接点。
- `x_junction`：X 形交叉点。
- `checkerboard`：棋盘格角点。
- `polygon`：多边形顶点。
- `line_intersections`：随机线段交点。

这些场景刻意保持简单，用于在引入 Harris/FAST 伪标签和真实验证图像之前，先调试标签、数据流程和基线模型。

## 第一天经典 Baseline

运行完整的“合成数据到经典 baseline”闭环：

```bash
python scripts/run_day1_baselines.py
```

脚本会生成 300 张 64x64 灰度合成图像：

- 100 张 `l_corner`
- 100 张 `t_junction`
- 100 张 `x_junction`

同时保存：

- `data/synthetic_images.npz`：图像、ground-truth 关键点和场景标签。
- `data/patch_dataset.npz`：`X_patches [7500, 9, 9]`、`y [7500]`、`centers [7500, 2]`、`image_ids [7500]`。
- `data/feature_dataset.npz`：`X_features [7500, 9]`、`y [7500]` 和特征名。
- `outputs/baselines/day1/overlays/harris_overlay.png`：单张样本上的 Harris 检测结果。
- `outputs/baselines/day1/overlays/fast_overlay.png`：单张样本上的 FAST 检测结果。
- `outputs/baselines/day1/overlays/orb_overlay.png`：单张样本上的 ORB 检测结果。
- `outputs/baselines/day1/overlays/mlp_overlay.png`：单张样本上的 MLP 检测结果。
- `outputs/baselines/day1/figures/mlp_training_curve.png`：MLP 二元交叉熵训练曲线。
- `outputs/baselines/day1/metrics/mlp_metrics.json`：数据集摘要、验证指标和 overlay 点匹配指标。
- `outputs/baselines/day1/metrics/baseline_metrics.json`：300 张合成图上 Harris、FAST、ORB、MLP 的 Precision/Recall/F1。
- `outputs/baselines/day1/figures/baseline_metrics.png`：上述指标的柱状图。
- `outputs/baselines/day1/figures/day1_overlay_comparison.png`：GT/Harris/FAST/ORB/MLP 对比图。

当前已纳入仓库的全图 baseline 评估结果：

```text
method   precision   recall   f1
Harris   0.1666      0.9600   0.2839
FAST     0.0666      0.8733   0.1238
ORB      0.0296      1.0000   0.0576
MLP      0.1219      0.9067   0.2149
```

![第一天 overlay 对比图](outputs/baselines/day1/figures/day1_overlay_comparison.png)
![第一天 baseline 指标](outputs/baselines/day1/figures/baseline_metrics.png)

## 第二天 QNN 与统一特征接口

运行完整的 Day 2 pipeline：

```bash
python scripts/run_day2_pipeline.py --seed 27
```

该脚本使用计划书推荐的 5 维输入接口：

```text
[Ix, Iy, lambda1, lambda2, R]
```

MLP 和 QNN 使用同一份 `data/feature_dataset.npz`。文件包含：

- `X_train`, `y_train`
- `X_val`, `y_val`
- `X_test`, `y_test`
- `feature_names`
- `train_centers`, `train_image_ids`
- `val_centers`, `val_image_ids`
- `test_centers`, `test_image_ids`

主要输出：

- `outputs/day2/metrics/day2_mlp_metrics.json`
- `outputs/day2/figures/day2_mlp_training_curve.png`
- `outputs/day2/metrics/day2_qnn_metrics.json`
- `outputs/day2/artifacts/day2_qnn_normalizer.npz`
- `outputs/day2/figures/day2_qnn_training_curve.png`
- `outputs/day2/figures/day2_qnn_overlay.png`
- `outputs/day2/figures/day2_comparison_overlay.png`
- `outputs/day2/figures/day2_pipeline_flow.png`
- `outputs/day2/figures/day2_data_samples.png`
- `outputs/day2/reports/day2_progress_summary.md`
- `outputs/day2/tables/day2_result_table.csv`

当前 clean test / held-out image 结果：

```text
Method  Input          Precision  Recall  F1      PR-AUC
Harris  image          0.1652     0.9500  0.2815  0.8953
FAST    image          0.0789     0.7833  0.1433  0.7925
ORB     image          0.0412     1.0000  0.0791  0.7759
MLP     same features  0.9347     0.9067  0.9205  0.9749
QNN     same features  0.5238     0.6875  0.5946  0.4672
```

![第二天 QNN 对比图](outputs/day2/figures/day2_comparison_overlay.png)
![第二天 QNN 检测图](outputs/day2/figures/day2_qnn_overlay.png)

## QNN 提升实验

进一步提升实验入口：

```bash
python scripts/run_improvement_experiments.py --seed 37
```

该实验加入：

- 8 维输入特征：`[Ix, Iy, Ix2, Iy2, IxIy, lambda1, lambda2, R]`
- QNN `Z+ZZ` readout
- trainable input scaling
- small-angle initialization
- `L=1/2/3` 与 `none/linear/ring` 消融
- Gaussian / blur / salt-and-pepper 噪声鲁棒性测试
- HTML demo 页面

主要产物：

- `outputs/qnn_improvement/ablation/qnn_ablation_results.csv`
- `outputs/qnn_improvement/ablation/qnn_ablation_results.png`
- `outputs/qnn_improvement/noise/noise_robustness_results.csv`
- `outputs/qnn_improvement/noise/noise_robustness.png`
- `outputs/qnn_improvement/metrics/improved_qnn_metrics.json`
- `outputs/qnn_improvement/figures/improved_qnn_training_curve.png`
- `outputs/qnn_improvement/figures/improved_comparison_overlay.png`
- `outputs/qnn_improvement/demo/qnn_improvement_demo.html`

当前改进主模型结果：

```text
Model                  Precision  Recall  F1      PR-AUC
MLP, same 8-D features 0.9406     0.9500  0.9453  0.9756
Improved QNN           0.6538     0.8500  0.7391  0.7909
```

![QNN 消融结果](outputs/qnn_improvement/ablation/qnn_ablation_results.png)
![噪声鲁棒性](outputs/qnn_improvement/noise/noise_robustness.png)
