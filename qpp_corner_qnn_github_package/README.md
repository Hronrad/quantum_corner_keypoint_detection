# QPP Corner QNN

Reproducible Python experiments for QPP-inspired / data-reuploading quantum neural networks on patch-level corner and keypoint detection.

The default task is binary classification: given a local patch or candidate center point, predict whether it is a corner, junction, or keypoint. All positive keypoint categories are treated as one positive class.

## Install

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

PyTorch is required for the QNN configs. Classical inspection, feature building, threshold baselines, logistic regression, and MLP baselines run without quantum-specific packages.

## Data

Place the uploaded archive at `data/raw/data.zip`, then extract it:

```bash
python -m zipfile -e data/raw/data.zip data/raw
```

This should create `data/raw/data/` with:

- `smoke_readme_pipeline`
- `synthetic_keypoints_smoke`
- `synthetic_keypoints`
- `synthetic_keypoints_small_blur`
- `synthetic_keypoints_rect_smoke`
- `synthetic_keypoints_square_smoke`

The scripts also work if `data/raw/data/` is already extracted.

## Inspect Data

```bash
python scripts/inspect_data.py --data-root data/raw
python scripts/inspect_data.py --data-root data/raw --dataset synthetic_keypoints_smoke
```

## Build Feature Caches

```bash
python scripts/build_features.py --data-root data/raw --dataset synthetic_keypoints_smoke --feature-set logS_eta --force
python scripts/build_features.py --data-root data/raw --dataset synthetic_keypoints --feature-set lambda12 --force
```

## Smoke Runs

```bash
python scripts/run_experiment.py --config configs/smoke_2q_logS_eta.yaml
python scripts/run_experiment.py --config configs/smoke_2q_lambda12.yaml
python scripts/run_experiment.py --config configs/smoke_1q_scalar.yaml
python scripts/run_ablation.py --config configs/ablation_smoke.yaml
pytest -q
```

Each run writes artifacts under `outputs/runs/<run_id>/`:

- `config_resolved.yaml`
- `metrics.json`
- `history.csv`
- `normalizer.json` when normalization is enabled
- `best_model.pt` for QNN runs
- `predictions.csv`
- `summary.csv`
- plots such as `pr_curve.png`, `feature_scatter.png`, and `patch_preview.png`

## Full Runs

```bash
python scripts/run_experiment.py --config configs/full_2q_logS_eta.yaml
python scripts/run_experiment.py --config configs/full_2q_lambda12.yaml
python scripts/run_experiment.py --config configs/full_1q_scalar.yaml
python scripts/run_ablation.py --config configs/ablation_full.yaml
```

The full configs train on `synthetic_keypoints` with image-level 70/15/15 train/validation/test splits.

## Keypoint Preview/Evaluation

After a run produces `predictions.csv`, evaluate sparse patch-center detections against `points_xy`:

```bash
python scripts/eval_keypoints.py --run-dir outputs/runs/<run_id> --data-root data/raw --split test --match-radius 4 --nms-radius 3
```

## Feature Sets

- `lambda12`: `[lambda1, lambda2]`
- `logS_eta`: `[logS, eta]`
- `scalar`: one-dimensional `t`, with modes `logS_plus_c_eta`, `lambda2`, `R`, and `learned_logS_eta`
- `ixiy`: `[Ix_center, Iy_center]`, a deliberately weak ablation
- `ref5`: `[Ix_center, Iy_center, lambda1, lambda2, R]`
- `R`, `lambda2`, `S`, `eta`, `logS_plus_eta`: single-score threshold baselines

Structure-tensor features are computed per patch using optional Gaussian smoothing, Sobel or finite-difference gradients, Gaussian-weighted tensor sums, sorted eigenvalues, Harris response, `S`, `eta`, and `logS`. Hessian features are computed for later ablations.

## Leakage Controls

- Splits are group-aware by image/sample id.
- Normalizers are fitted only on the training split.
- Classification thresholds are selected only on validation scores by best F1.
- Test metrics use the validation-selected threshold.
