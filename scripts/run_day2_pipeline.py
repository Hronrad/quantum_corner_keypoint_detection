from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import QNNConfig, TrainConfig
from preprocessing import FeatureNormalizer
from qcd_data.baselines import evaluate_points, harris_response_numpy, run_fast, run_harris, run_orb
from qcd_data.features import STRUCTURE_TENSOR_FEATURE_NAMES, extract_structure_tensor_features
from qcd_data.synthetic import SyntheticKeypointConfig, extract_patches, generate_sample
from qnn_circuit import DataReuploadingQNN
from train_qnn import train_qnn_experiment


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    data_dir = ROOT / "data"
    outputs_dir = ROOT / "outputs"
    data_dir.mkdir(exist_ok=True)
    outputs_dir.mkdir(exist_ok=True)

    samples = generate_clean_samples(rng, args.images_per_scene)
    images = np.stack([sample.image for sample in samples]).astype(np.float32)
    keypoints = np.stack([sample.points_xy for sample in samples]).astype(np.float32)
    scene_labels = np.array([sample.scene_type for sample in samples])

    patches, patch_labels, centers, image_ids = build_patch_dataset(samples, patch_size=args.patch_size)
    features = extract_structure_tensor_features(patches)
    train_images, val_images, test_images = split_image_ids(scene_labels, args.seed)

    split_payload = make_feature_splits(features, patch_labels, centers, image_ids, train_images, val_images, test_images)
    feature_dataset_path = data_dir / "feature_dataset.npz"
    np.savez_compressed(
        feature_dataset_path,
        **split_payload,
        feature_names=np.array(STRUCTURE_TENSOR_FEATURE_NAMES),
        images=images,
        keypoints=keypoints,
        scene_types=scene_labels,
        split_train_image_ids=train_images,
        split_val_image_ids=val_images,
        split_test_image_ids=test_images,
    )

    mlp, mlp_metrics = train_mlp_baseline(split_payload, outputs_dir, args.seed)
    qnn_metrics = train_qnn_baseline(feature_dataset_path, outputs_dir, data_dir, args)
    qnn_predictor = load_qnn_predictor(outputs_dir / "day2_qnn_run")
    normalizer = FeatureNormalizer.load(outputs_dir / "day2_qnn_run" / "normalizer.npz")

    test_image_id = int(test_images[0])
    comparison_points = build_comparison_points(images[test_image_id], mlp, qnn_predictor, normalizer, args.patch_size)
    classical_rows = evaluate_classical_methods(images, keypoints, split_payload, test_images)
    mlp_patch_row = metrics_row("MLP", "same features", mlp_metrics)
    qnn_patch_row = metrics_row("QNN", "same features", qnn_metrics["test"])
    result_rows = [*classical_rows, mlp_patch_row, qnn_patch_row]

    write_result_table(outputs_dir / "day2_result_table.csv", result_rows)
    save_qnn_overlay(
        images,
        keypoints,
        test_images[: min(3, len(test_images))],
        qnn_predictor,
        normalizer,
        args.patch_size,
        outputs_dir / "day2_qnn_overlay.png",
    )
    save_day2_comparison_overlay(
        images[test_image_id],
        keypoints[test_image_id],
        comparison_points,
        outputs_dir / "day2_comparison_overlay.png",
    )
    save_pipeline_flow(outputs_dir / "day2_pipeline_flow.png")
    save_data_samples(images, keypoints, scene_labels, test_images, outputs_dir / "day2_data_samples.png")
    write_progress_summary(outputs_dir / "day2_progress_summary.md", result_rows, qnn_metrics)

    print("Day 2 pipeline complete.")
    print(f"Wrote {feature_dataset_path}")
    print(f"Wrote {outputs_dir / 'day2_result_table.csv'}")
    print(json.dumps({"mlp": mlp_metrics, "qnn": qnn_metrics["test"]}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Day 2 QNN + baseline pipeline.")
    parser.add_argument("--seed", type=int, default=27)
    parser.add_argument("--images-per-scene", type=int, default=100)
    parser.add_argument("--patch-size", type=int, default=9)
    parser.add_argument("--qnn-epochs", type=int, default=20)
    parser.add_argument("--qnn-train-limit", type=int, default=160)
    parser.add_argument("--qnn-val-limit", type=int, default=80)
    parser.add_argument("--qnn-test-limit", type=int, default=80)
    return parser.parse_args()


def generate_clean_samples(rng: np.random.Generator, images_per_scene: int):
    config = SyntheticKeypointConfig(
        image_size=64,
        min_margin=10,
        line_width=(1, 3),
        patch_size=9,
        patches_per_image=25,
        positive_fraction=0.2,
        positive_radius=1.5,
        negative_radius=5.0,
        noise_std=(0.0, 0.0),
        blur_probability=0.0,
        contrast_range=(0.9, 1.1),
        brightness_range=(-0.02, 0.02),
    )
    samples = []
    for scene_type in ["l_corner", "t_junction", "x_junction"]:
        for _ in range(images_per_scene):
            samples.append(generate_sample(int(rng.integers(0, 2**32 - 1)), config, scene_type=scene_type))
    return samples


def build_patch_dataset(samples, patch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    patches, labels, centers, image_ids = [], [], [], []
    for image_id, sample in enumerate(samples):
        patches.append(extract_patches(sample.image, sample.patch_centers_xy, patch_size))
        labels.append(sample.patch_labels)
        centers.append(sample.patch_centers_xy)
        image_ids.append(np.full(len(sample.patch_labels), image_id, dtype=np.int64))
    return (
        np.concatenate(patches, axis=0).astype(np.float32),
        np.concatenate(labels, axis=0).astype(np.int64),
        np.concatenate(centers, axis=0).astype(np.float32),
        np.concatenate(image_ids, axis=0).astype(np.int64),
    )


def split_image_ids(scene_labels: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_ids, val_ids, test_ids = [], [], []
    for scene_type in sorted(set(scene_labels.tolist())):
        ids = np.flatnonzero(scene_labels == scene_type)
        train_scene, temp_scene = train_test_split(ids, test_size=0.4, random_state=seed, shuffle=True)
        val_scene, test_scene = train_test_split(temp_scene, test_size=0.5, random_state=seed, shuffle=True)
        train_ids.extend(train_scene.tolist())
        val_ids.extend(val_scene.tolist())
        test_ids.extend(test_scene.tolist())
    return np.array(sorted(train_ids)), np.array(sorted(val_ids)), np.array(sorted(test_ids))


def make_feature_splits(
    features: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
    image_ids: np.ndarray,
    train_images: np.ndarray,
    val_images: np.ndarray,
    test_images: np.ndarray,
) -> dict[str, np.ndarray]:
    payload = {}
    for name, ids in [("train", train_images), ("val", val_images), ("test", test_images)]:
        mask = np.isin(image_ids, ids)
        payload[f"X_{name}"] = features[mask].astype(np.float32)
        payload[f"y_{name}"] = labels[mask].astype(np.int64)
        payload[f"{name}_centers"] = centers[mask].astype(np.float32)
        payload[f"{name}_image_ids"] = image_ids[mask].astype(np.int64)
    return payload


def train_mlp_baseline(payload: dict[str, np.ndarray], outputs_dir: Path, seed: int):
    mlp = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(32, 16),
            activation="relu",
            solver="adam",
            max_iter=300,
            learning_rate_init=0.003,
            random_state=seed,
            batch_size=128,
        ),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        mlp.fit(payload["X_train"], payload["y_train"].astype(int))

    probabilities = mlp.predict_proba(payload["X_test"])[:, 1]
    metrics = compute_probability_metrics(payload["y_test"], probabilities)
    metrics.update(
        {
            "train_samples": int(len(payload["y_train"])),
            "val_samples": int(len(payload["y_val"])),
            "test_samples": int(len(payload["y_test"])),
            "feature_names": STRUCTURE_TENSOR_FEATURE_NAMES,
        }
    )
    (outputs_dir / "day2_mlp_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    loss_curve = mlp.named_steps["mlpclassifier"].loss_curve_
    save_loss_curve(loss_curve, outputs_dir / "day2_mlp_training_curve.png", "MLP training loss")
    return mlp, metrics


def train_qnn_baseline(feature_dataset_path: Path, outputs_dir: Path, data_dir: Path, args: argparse.Namespace) -> dict:
    qnn_subset_path, subset_counts = save_qnn_subset(
        feature_dataset_path,
        data_dir / "feature_dataset_qnn_subset.npz",
        train_limit=args.qnn_train_limit,
        val_limit=args.qnn_val_limit,
        test_limit=args.qnn_test_limit,
        seed=args.seed,
    )
    qnn_output_dir = outputs_dir / "day2_qnn_run"
    metrics = train_qnn_experiment(
        data_path=qnn_subset_path,
        output_dir=qnn_output_dir,
        qnn_config=QNNConfig(n_qubits=5, n_layers=3, encoding_type="ryrz", entanglement="ring", readout="all"),
        train_config=TrainConfig(learning_rate=1e-2, batch_size=32, epochs=args.qnn_epochs, seed=args.seed),
        feature_indices=None,
        feature_names=STRUCTURE_TENSOR_FEATURE_NAMES,
        device_name="cpu",
    )
    run_metrics = json.loads((qnn_output_dir / "metrics.json").read_text(encoding="utf-8"))
    run_metrics["subset_counts"] = subset_counts
    run_metrics["source_dataset"] = str(feature_dataset_path)
    shutil.copyfile(qnn_output_dir / "normalizer.npz", outputs_dir / "day2_qnn_normalizer.npz")
    (outputs_dir / "day2_qnn_metrics.json").write_text(json.dumps(run_metrics, indent=2), encoding="utf-8")
    save_qnn_training_curve(run_metrics["history"], outputs_dir / "day2_qnn_training_curve.png")
    return run_metrics


def save_qnn_subset(
    source_path: Path,
    subset_path: Path,
    train_limit: int,
    val_limit: int,
    test_limit: int,
    seed: int,
) -> tuple[Path, dict[str, int]]:
    data = np.load(source_path)
    payload = {"feature_names": data["feature_names"]}
    counts = {}
    for split, limit in [("train", train_limit), ("val", val_limit), ("test", test_limit)]:
        indices = stratified_limit_indices(data[f"y_{split}"], limit, seed + len(split))
        payload[f"X_{split}"] = data[f"X_{split}"][indices]
        payload[f"y_{split}"] = data[f"y_{split}"][indices]
        counts[f"{split}_samples"] = int(len(indices))
        counts[f"{split}_positives"] = int(np.sum(payload[f"y_{split}"] == 1))
    np.savez_compressed(subset_path, **payload)
    return subset_path, counts


def stratified_limit_indices(y: np.ndarray, limit: int, seed: int) -> np.ndarray:
    y = np.asarray(y).reshape(-1)
    if limit <= 0 or limit >= len(y):
        return np.arange(len(y))
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    pos_count = max(1, min(len(pos), int(round(limit * len(pos) / len(y)))))
    neg_count = max(1, min(len(neg), limit - pos_count))
    chosen = np.concatenate([rng.choice(pos, pos_count, replace=False), rng.choice(neg, neg_count, replace=False)])
    return np.sort(chosen)


def load_qnn_predictor(run_dir: Path) -> DataReuploadingQNN:
    checkpoint = torch.load(run_dir / "best_model.pt", map_location="cpu")
    model = DataReuploadingQNN(**checkpoint["qnn_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def predict_with_qnn(features: np.ndarray, model: DataReuploadingQNN, normalizer: FeatureNormalizer) -> np.ndarray:
    phi = normalizer.transform(features)
    with torch.no_grad():
        probabilities = model.predict_proba(torch.as_tensor(phi, dtype=torch.float32)).detach().cpu().numpy()
    return probabilities.reshape(-1)


def build_comparison_points(
    image: np.ndarray,
    mlp,
    qnn_model: DataReuploadingQNN,
    normalizer: FeatureNormalizer,
    patch_size: int,
) -> dict[str, np.ndarray]:
    centers = sliding_centers(image.shape, patch_size)
    patches = extract_patches(image, centers, patch_size)
    features = extract_structure_tensor_features(patches)
    mlp_points = select_points_from_scores(centers, mlp.predict_proba(features)[:, 1], threshold=0.5)
    qnn_points = select_points_from_scores(centers, predict_with_qnn(features, qnn_model, normalizer), threshold=0.5)
    return {
        "Harris": run_harris(image, threshold_rel=0.01, max_points=40),
        "FAST": run_fast(image, threshold=20, max_points=40),
        "ORB": run_orb(image, max_points=40),
        "MLP": mlp_points,
        "QNN": qnn_points,
    }


def evaluate_classical_methods(
    images: np.ndarray,
    keypoints: np.ndarray,
    payload: dict[str, np.ndarray],
    test_image_ids: np.ndarray,
) -> list[dict[str, str | float]]:
    totals = {
        "Harris": {"tp": 0, "fp": 0, "fn": 0},
        "FAST": {"tp": 0, "fp": 0, "fn": 0},
        "ORB": {"tp": 0, "fp": 0, "fn": 0},
    }
    patch_scores = {name: [] for name in totals}
    y_true = payload["y_test"].astype(int)

    for image_id in test_image_ids:
        image = images[int(image_id)]
        gt = keypoints[int(image_id)]
        detections = {
            "Harris": run_harris(image, threshold_rel=0.01, max_points=40),
            "FAST": run_fast(image, threshold=20, max_points=40),
            "ORB": run_orb(image, max_points=40),
        }
        for name, points in detections.items():
            metrics = evaluate_points(points, gt)
            totals[name]["tp"] += metrics.true_positives
            totals[name]["fp"] += metrics.false_positives
            totals[name]["fn"] += metrics.false_negatives

        mask = payload["test_image_ids"] == image_id
        centers = payload["test_centers"][mask]
        patch_scores["Harris"].extend(sample_response_at_centers(harris_response_numpy(image), centers).tolist())
        patch_scores["FAST"].extend(sparse_detector_scores(image, centers, method="fast").tolist())
        patch_scores["ORB"].extend(sparse_detector_scores(image, centers, method="orb").tolist())

    rows = []
    for name, values in totals.items():
        tp, fp, fn = values["tp"], values["fp"], values["fn"]
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
        rows.append(
            {
                "Method": name,
                "Input": "image",
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
                "PR-AUC": safe_average_precision(y_true, np.asarray(patch_scores[name], dtype=np.float32)),
            }
        )
    return rows


def sample_response_at_centers(response: np.ndarray, centers: np.ndarray) -> np.ndarray:
    scores = []
    for x, y in centers:
        cx, cy = int(round(float(x))), int(round(float(y)))
        y0, y1 = max(0, cy - 1), min(response.shape[0], cy + 2)
        x0, x1 = max(0, cx - 1), min(response.shape[1], cx + 2)
        scores.append(float(np.max(response[y0:y1, x0:x1])))
    return np.array(scores, dtype=np.float32)


def sparse_detector_scores(image: np.ndarray, centers: np.ndarray, method: str) -> np.ndarray:
    if method == "fast":
        points = run_fast(image, threshold=20, max_points=80)
    elif method == "orb":
        points = run_orb(image, max_points=80)
    else:
        raise ValueError(method)
    scores = np.zeros(len(centers), dtype=np.float32)
    if len(points) == 0:
        return scores
    for index, center in enumerate(centers):
        distances = np.linalg.norm(points - center[None, :], axis=1)
        scores[index] = float(np.max(np.clip(4.0 - distances, 0.0, 4.0)))
    return scores


def sliding_centers(shape: tuple[int, int], patch_size: int) -> np.ndarray:
    half = patch_size // 2
    height, width = shape
    return np.array([(x, y) for y in range(half, height - half) for x in range(half, width - half)], dtype=np.float32)


def select_points_from_scores(
    centers: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    max_points: int = 12,
    min_distance: float = 5.0,
) -> np.ndarray:
    indices = np.flatnonzero(scores >= threshold)
    if len(indices) == 0:
        indices = np.argsort(scores)[-min(5, len(scores)) :]
    chosen = []
    for index in sorted(indices, key=lambda i: float(scores[i]), reverse=True):
        point = centers[index]
        if all(np.linalg.norm(point - existing) >= min_distance for existing in chosen):
            chosen.append(point)
        if len(chosen) >= max_points:
            break
    return np.array(chosen, dtype=np.float32).reshape(-1, 2)


def compute_probability_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    y_int = np.asarray(y_true, dtype=int).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    preds = (probabilities >= 0.5).astype(int)
    return {
        "precision": float(precision_score(y_int, preds, zero_division=0)),
        "recall": float(recall_score(y_int, preds, zero_division=0)),
        "f1": float(f1_score(y_int, preds, zero_division=0)),
        "pr_auc": safe_average_precision(y_int, probabilities),
    }


def safe_average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    try:
        return float(average_precision_score(np.asarray(y_true, dtype=int), np.asarray(scores, dtype=float)))
    except ValueError:
        return float("nan")


def metrics_row(method: str, input_name: str, metrics: dict) -> dict[str, str | float]:
    return {
        "Method": method,
        "Input": input_name,
        "Precision": float(metrics["precision"]),
        "Recall": float(metrics["recall"]),
        "F1": float(metrics["f1"]),
        "PR-AUC": float(metrics["pr_auc"]),
    }


def write_result_table(path: Path, rows: list[dict[str, str | float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Method", "Input", "Precision", "Recall", "F1", "PR-AUC"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_loss_curve(loss_curve: list[float], path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(np.arange(1, len(loss_curve) + 1), loss_curve, color="tab:blue")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_qnn_training_curve(history: list[dict], path: Path) -> None:
    epochs = [row["epoch"] for row in history]
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(epochs, [row["train_loss"] for row in history], label="train")
    ax.plot(epochs, [row["val_loss"] for row in history], label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCEWithLogitsLoss")
    ax.set_title("QNN training loss")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_qnn_overlay(
    images: np.ndarray,
    keypoints: np.ndarray,
    image_ids: np.ndarray,
    qnn_model: DataReuploadingQNN,
    normalizer: FeatureNormalizer,
    patch_size: int,
    path: Path,
) -> None:
    columns = len(image_ids)
    fig, axes = plt.subplots(1, columns, figsize=(4 * columns, 4), squeeze=False)
    for ax, image_id in zip(axes.flat, image_ids):
        image = images[int(image_id)]
        centers = sliding_centers(image.shape, patch_size)
        scores = predict_with_qnn(extract_structure_tensor_features(extract_patches(image, centers, patch_size)), qnn_model, normalizer)
        points = select_points_from_scores(centers, scores, threshold=0.5)
        ax.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
        ax.scatter(keypoints[int(image_id)][:, 0], keypoints[int(image_id)][:, 1], c="lime", s=42, marker="o", facecolors="none", linewidths=1.5)
        if len(points):
            ax.scatter(points[:, 0], points[:, 1], c="red", s=30, marker="x", linewidths=1.4)
        ax.set_title(f"QNN image {int(image_id)}")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_day2_comparison_overlay(
    image: np.ndarray,
    gt_xy: np.ndarray,
    points_by_method: dict[str, np.ndarray],
    path: Path,
) -> None:
    panels = [("GT", gt_xy, "lime", "o"), *[(name, points, "red", "x") for name, points in points_by_method.items()]]
    fig, axes = plt.subplots(2, 3, figsize=(10, 7))
    for ax, (title, points, color, marker) in zip(axes.flat, panels):
        ax.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
        points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        if len(points):
            if marker == "o":
                ax.scatter(points[:, 0], points[:, 1], c=color, s=44, marker=marker, facecolors="none", linewidths=1.6)
            else:
                ax.scatter(points[:, 0], points[:, 1], c=color, s=30, marker=marker, linewidths=1.4)
        ax.set_title(f"{title} ({len(points)})")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_pipeline_flow(path: Path) -> None:
    labels = ["Synthetic images", "Patch sampling", "5-D features", "MLP / QNN", "Metrics + overlays"]
    fig, ax = plt.subplots(figsize=(10, 2.6))
    ax.axis("off")
    xs = np.linspace(0.08, 0.92, len(labels))
    for index, (x, label) in enumerate(zip(xs, labels)):
        ax.text(x, 0.55, label, ha="center", va="center", bbox={"boxstyle": "round,pad=0.35", "fc": "#eef5ff", "ec": "#4777aa"})
        if index < len(labels) - 1:
            ax.annotate("", xy=(xs[index + 1] - 0.08, 0.55), xytext=(x + 0.08, 0.55), arrowprops={"arrowstyle": "->", "lw": 1.4})
    ax.set_title("Task flow")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_data_samples(images: np.ndarray, keypoints: np.ndarray, scene_labels: np.ndarray, image_ids: np.ndarray, path: Path) -> None:
    chosen = image_ids[:6]
    fig, axes = plt.subplots(2, 3, figsize=(8, 5))
    for ax, image_id in zip(axes.flat, chosen):
        ax.imshow(images[int(image_id)], cmap="gray", vmin=0.0, vmax=1.0)
        pts = keypoints[int(image_id)]
        ax.scatter(pts[:, 0], pts[:, 1], c="lime", s=35, marker="o", facecolors="none", linewidths=1.4)
        ax.set_title(str(scene_labels[int(image_id)]))
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_progress_summary(path: Path, rows: list[dict[str, str | float]], qnn_metrics: dict) -> None:
    table = "\n".join(
        f"| {row['Method']} | {row['Input']} | {float(row['Precision']):.4f} | {float(row['Recall']):.4f} | {float(row['F1']):.4f} | {float(row['PR-AUC']):.4f} |"
        for row in rows
    )
    text = f"""# Day 2 Progress Summary

已完成：
1. 合成角点/交点数据
2. patch 采样
3. classical baseline，包括 Harris / FAST / ORB
4. 5-D feature extraction: Ix, Iy, lambda1, lambda2, R
5. QNN 接入同一特征接口
6. QNN 第一轮训练与可视化

下一步：
1. 噪声鲁棒性实验
2. QNN 消融实验
3. demo 整合

| Method | Input | Precision | Recall | F1 | PR-AUC |
| --- | --- | ---: | ---: | ---: | ---: |
{table}

QNN subset: {json.dumps(qnn_metrics.get('subset_counts', {}), ensure_ascii=False)}
"""
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
