from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qcd_data.baselines import evaluate_points, run_fast, run_harris, save_overlay, save_overlay_comparison
from qcd_data.features import FEATURE_NAMES, extract_patch_features
from qcd_data.synthetic import SyntheticKeypointConfig, extract_patches, generate_sample


def main() -> None:
    seed = 26
    rng = np.random.default_rng(seed)
    data_dir = ROOT / "data"
    outputs_dir = ROOT / "outputs"
    data_dir.mkdir(exist_ok=True)
    outputs_dir.mkdir(exist_ok=True)

    config = SyntheticKeypointConfig(
        image_size=64,
        min_margin=10,
        line_width=(1, 3),
        patch_size=9,
        patches_per_image=25,
        positive_fraction=0.2,
        positive_radius=1.5,
        negative_radius=5.0,
        noise_std=(0.0, 0.04),
        blur_probability=0.15,
        blur_radius=(0.2, 0.6),
    )

    samples = []
    scene_types = ["l_corner", "t_junction", "x_junction"]
    for scene_type in scene_types:
        for _ in range(100):
            sample_seed = int(rng.integers(0, 2**32 - 1))
            samples.append(generate_sample(sample_seed, config, scene_type=scene_type))

    images = np.stack([sample.image for sample in samples]).astype(np.float32)
    keypoints = np.stack([sample.points_xy for sample in samples]).astype(np.float32)
    scene_labels = np.array([sample.scene_type for sample in samples])

    patches = []
    labels = []
    centers = []
    image_ids = []
    for image_id, sample in enumerate(samples):
        sample_patches = extract_patches(sample.image, sample.patch_centers_xy, config.patch_size)
        patches.append(sample_patches)
        labels.append(sample.patch_labels)
        centers.append(sample.patch_centers_xy)
        image_ids.append(np.full(len(sample.patch_labels), image_id, dtype=np.int64))

    patches_array = np.concatenate(patches, axis=0).astype(np.float32)
    labels_array = np.concatenate(labels, axis=0).astype(np.int64)
    centers_array = np.concatenate(centers, axis=0).astype(np.float32)
    image_ids_array = np.concatenate(image_ids, axis=0).astype(np.int64)
    features = extract_patch_features(patches_array)

    np.savez_compressed(
        data_dir / "synthetic_images.npz",
        images=images,
        keypoints=keypoints,
        scene_types=scene_labels,
        image_size=np.array([config.height, config.width], dtype=np.int64),
    )
    np.savez_compressed(
        data_dir / "patch_dataset.npz",
        X_patches=patches_array,
        y=labels_array,
        centers=centers_array,
        image_ids=image_ids_array,
    )
    np.savez_compressed(
        data_dir / "feature_dataset.npz",
        X_features=features,
        y=labels_array,
        feature_names=np.array(FEATURE_NAMES),
    )

    x_train, x_val, y_train, y_val = train_test_split(
        features,
        labels_array,
        test_size=0.25,
        random_state=seed,
        stratify=labels_array,
    )
    mlp = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(32, 16),
            activation="relu",
            solver="adam",
            max_iter=300,
            random_state=seed,
            learning_rate_init=0.003,
            batch_size=128,
        ),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        mlp.fit(x_train, y_train)
    val_prob = mlp.predict_proba(x_val)[:, 1]
    val_pred = (val_prob >= 0.5).astype(np.int64)

    display_index = 0
    display_image = images[display_index]
    display_gt = samples[display_index].points_xy
    harris_points = run_harris(display_image, threshold_rel=0.01, max_points=40)
    fast_points = run_fast(display_image, threshold=20, max_points=40)
    mlp_points, mlp_probabilities = predict_keypoints_with_mlp(display_image, mlp, config.patch_size)

    save_overlay(display_image, harris_points, outputs_dir / "harris_overlay.png", display_gt, "Harris")
    save_overlay(display_image, fast_points, outputs_dir / "fast_overlay.png", display_gt, "FAST")
    save_overlay(display_image, mlp_points, outputs_dir / "mlp_overlay.png", display_gt, "MLP")
    save_overlay_comparison(
        display_image,
        display_gt,
        harris_points,
        fast_points,
        mlp_points,
        outputs_dir / "day1_overlay_comparison.png",
    )
    save_training_curve(mlp.named_steps["mlpclassifier"].loss_curve_, outputs_dir / "mlp_training_curve.png")

    harris_metrics = evaluate_points(harris_points, display_gt)
    fast_metrics = evaluate_points(fast_points, display_gt)
    mlp_point_metrics = evaluate_points(mlp_points, display_gt)
    metrics = {
        "dataset": {
            "num_images": int(len(images)),
            "image_shape": [int(config.height), int(config.width)],
            "scene_counts": {scene_type: 100 for scene_type in scene_types},
            "num_patches": int(len(labels_array)),
            "positive_patches": int(labels_array.sum()),
            "negative_patches": int(len(labels_array) - labels_array.sum()),
            "feature_names": FEATURE_NAMES,
        },
        "mlp_validation": {
            "precision": float(precision_score(y_val, val_pred, zero_division=0)),
            "recall": float(recall_score(y_val, val_pred, zero_division=0)),
            "f1": float(f1_score(y_val, val_pred, zero_division=0)),
            "loss_final": float(mlp.named_steps["mlpclassifier"].loss_),
            "probability_min": float(np.min(val_prob)),
            "probability_max": float(np.max(val_prob)),
        },
        "overlay_image": {
            "image_id": display_index,
            "scene_type": str(scene_labels[display_index]),
            "gt_points": display_gt.tolist(),
            "harris": _metrics_dict(harris_metrics),
            "fast": _metrics_dict(fast_metrics),
            "mlp": _metrics_dict(mlp_point_metrics),
            "mlp_probability_threshold": 0.5,
            "mlp_probability_max": float(np.max(mlp_probabilities)),
        },
    }
    (outputs_dir / "mlp_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("Day 1 baseline pipeline complete.")
    print(f"Wrote {data_dir / 'patch_dataset.npz'}")
    print(f"Wrote {data_dir / 'feature_dataset.npz'}")
    print(f"Wrote {outputs_dir / 'day1_overlay_comparison.png'}")
    print(json.dumps(metrics["mlp_validation"], indent=2))


def predict_keypoints_with_mlp(
    image: np.ndarray,
    mlp,
    patch_size: int,
    probability_threshold: float = 0.5,
    max_points: int = 12,
    min_distance: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    half = patch_size // 2
    centers = np.array(
        [(x, y) for y in range(half, image.shape[0] - half) for x in range(half, image.shape[1] - half)],
        dtype=np.float32,
    )
    patches = extract_patches(image, centers, patch_size)
    features = extract_patch_features(patches)
    probabilities = mlp.predict_proba(features)[:, 1]
    candidate_indices = np.flatnonzero(probabilities >= probability_threshold)
    scored = [(float(probabilities[index]), float(centers[index, 0]), float(centers[index, 1])) for index in candidate_indices]
    points = []
    for _score, x, y in sorted(scored, reverse=True):
        point = np.array([x, y], dtype=np.float32)
        if all(np.linalg.norm(point - np.array(existing, dtype=np.float32)) >= min_distance for existing in points):
            points.append((x, y))
        if len(points) >= max_points:
            break
    if not points:
        best = np.argsort(probabilities)[-min(5, len(probabilities)) :]
        points = [(float(centers[index, 0]), float(centers[index, 1])) for index in best[::-1]]
    return np.array(points, dtype=np.float32), probabilities


def save_training_curve(loss_curve: list[float], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(np.arange(1, len(loss_curve) + 1), loss_curve, color="tab:blue")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Binary cross entropy")
    ax.set_title("MLP training loss")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _metrics_dict(metrics) -> dict[str, float | int]:
    return {
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
        "true_positives": metrics.true_positives,
        "false_positives": metrics.false_positives,
        "false_negatives": metrics.false_negatives,
    }


if __name__ == "__main__":
    main()
