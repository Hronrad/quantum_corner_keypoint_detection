# Quantum Corner Detection: Synthetic Data

This workspace starts with synthetic data for corner/keypoint detection. The first goal is to produce images with known keypoint labels so classical baselines and later QNN experiments can share the same supervision.

## Generate Data

```powershell
python scripts\generate_synthetic_dataset.py --count 200 --out data\synthetic_keypoints --seed 7
```

For rectangular images, set width and height explicitly:

```powershell
python scripts\generate_synthetic_dataset.py --count 200 --width 192 --height 96 --out data\synthetic_keypoints_192x96
```

For small blurry images, reduce the patch size and increase blur:

```powershell
python scripts\generate_synthetic_dataset.py --count 64 --width 48 --height 48 --patch-size 15 --line-width-min 1 --line-width-max 3 --blur-probability 1.0 --blur-radius-min 1.2 --blur-radius-max 2.0 --noise-std-max 0.04 --out data\synthetic_keypoints_small_blur
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

These are intentionally simple. They give us a controlled place to debug labels and baselines before moving to pseudo-labels from Harris/FAST and real validation frames.
