# Quantum Corner Keypoint Detection

Synthetic data utilities for corner/keypoint detection. The project currently focuses on generating controlled grayscale images with known keypoint annotations, so classical baselines and later quantum neural network experiments can use the same supervision.

The generated samples include point coordinates, keypoint type labels, dense Gaussian heatmaps, and patch-level binary labels. This makes the dataset useful for both dense keypoint localization and local patch classification experiments.

## Project Structure

```text
.
├── README.md
├── .gitignore
├── quantum_corner_keypoint_detection.pdf
├── qcd_data/
│   ├── __init__.py
│   ├── synthetic.py
│   └── visualize.py
└── scripts/
    └── generate_synthetic_dataset.py
```

- `qcd_data/synthetic.py`: synthetic scene generation, label generation, patch extraction, and a PyTorch `Dataset` wrapper.
- `qcd_data/visualize.py`: preview-grid rendering for generated samples.
- `scripts/generate_synthetic_dataset.py`: command-line tool for writing image/label datasets to disk.
- `quantum_corner_keypoint_detection.pdf`: project paper/report artifact.
- `data/`: generated datasets are written here by default and intentionally ignored by Git.

## Generate Data

```bash
python scripts/generate_synthetic_dataset.py --count 200 --out data/synthetic_keypoints --seed 7
```

For rectangular images, set width and height explicitly:

```bash
python scripts/generate_synthetic_dataset.py --count 200 --width 192 --height 96 --out data/synthetic_keypoints_192x96
```

For small blurry images, reduce the patch size and increase blur:

```bash
python scripts/generate_synthetic_dataset.py --count 64 --width 48 --height 48 --patch-size 15 --line-width-min 1 --line-width-max 3 --blur-probability 1.0 --blur-radius-min 1.2 --blur-radius-max 2.0 --noise-std-max 0.04 --out data/synthetic_keypoints_small_blur
```

The script writes:

- `images/sample_XXXXXX.png`: grayscale synthetic image.
- `labels/sample_XXXXXX.npz`: dense and patch-level labels.
- `manifest.jsonl`: image/label paths and scene metadata.
- `preview.png`: quick visual check with keypoints overlaid.
- `config.json`: generation parameters.

`--image-size 128` remains a square-image shortcut. `--width` and `--height` override it when either dimension needs to be different.

## Label Format

Each `.npz` label file contains:

- `image`: float32 array with shape `(H, W)`, values in `[0, 1]`.
- `points_xy`: float32 array with shape `(N, 2)`, stored as `(x, y)` pixel coordinates.
- `type_ids`: int64 array with shape `(N,)`; `1=corner`, `2=t_junction`, `3=x_junction`.
- `heatmap`: float32 array with shape `(H, W)`, max-composed Gaussian targets around each point.
- `patch_centers_xy`: float32 array with shape `(P, 2)`.
- `patch_labels`: int64 array with shape `(P,)`, binary labels for patch-level classification.
- `patches`: float32 array with shape `(P, patch_size, patch_size)`.
- `scene_type`: synthetic scene family.

## Current Scene Families

- `l_corner`
- `t_junction`
- `x_junction`
- `checkerboard`
- `polygon`
- `line_intersections`

These scenes are intentionally simple. They provide a controlled place to debug labels and baselines before moving to pseudo-labels from Harris/FAST and real validation frames.

---

# 量子角点关键点检测

本项目提供角点/关键点检测所需的合成数据生成工具。当前阶段的重点是生成带有已知关键点标注的可控灰度图像，使经典基线方法和后续量子神经网络实验能够共享同一套监督信号。

生成样本包含关键点坐标、关键点类别、稠密高斯热力图以及局部图像块的二分类标签，因此既可用于稠密关键点定位，也可用于局部 patch 分类实验。

## 目录结构

```text
.
├── README.md
├── .gitignore
├── quantum_corner_keypoint_detection.pdf
├── qcd_data/
│   ├── __init__.py
│   ├── synthetic.py
│   └── visualize.py
└── scripts/
    └── generate_synthetic_dataset.py
```

- `qcd_data/synthetic.py`：合成场景生成、标签生成、patch 提取，以及 PyTorch `Dataset` 封装。
- `qcd_data/visualize.py`：生成样本的预览图绘制工具。
- `scripts/generate_synthetic_dataset.py`：将合成图像和标签写入磁盘的命令行脚本。
- `quantum_corner_keypoint_detection.pdf`：项目论文/报告文件。
- `data/`：默认的数据集输出目录，已被 Git 忽略。

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
