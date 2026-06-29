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
from PIL import Image, ImageFilter
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import QNNConfig, TrainConfig
from preprocessing import FeatureNormalizer
from qcd_data.baselines import evaluate_points, run_fast, run_harris, run_orb
from qcd_data.features import (
    EXTENDED_STRUCTURE_TENSOR_FEATURE_NAMES,
    STRUCTURE_TENSOR_FEATURE_NAMES,
    extract_extended_structure_tensor_features,
    extract_structure_tensor_features,
)
from qcd_data.synthetic import extract_patches
from qnn_circuit import DataReuploadingQNN
from scripts.run_day2_pipeline import (
    build_patch_dataset,
    generate_clean_samples,
    make_feature_splits,
    split_image_ids,
)
from train_qnn import train_qnn_experiment


NOISE_CASES = [
    ("clean", "none", 0.0),
    ("gaussian_0.04", "gaussian", 0.04),
    ("gaussian_0.08", "gaussian", 0.08),
    ("blur_0.9", "blur", 0.9),
    ("saltpepper_0.03", "saltpepper", 0.03),
]


def main() -> None:
    args = parse_args()
    data_dir = ROOT / "data"
    output_dir = args.output_dir
    run_root = output_dir / "feature_dim_noise_comparison_runs"
    data_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    run_root.mkdir(exist_ok=True)

    dataset5_path, dataset8_path = ensure_feature_datasets(args, data_dir, output_dir)
    dataset5 = load_feature_dataset(dataset5_path)
    dataset8 = load_feature_dataset(dataset8_path)
    validate_shared_protocol(dataset5, dataset8)

    run_specs = [
        {
            "feature_dim": "5D",
            "dataset": dataset5,
            "dataset_path": dataset5_path,
            "feature_names": STRUCTURE_TENSOR_FEATURE_NAMES,
            "extractor": extract_structure_tensor_features,
            "qnn_config": QNNConfig(n_qubits=5, n_layers=3, encoding_type="ryrz", entanglement="ring", readout="all"),
            "train_limit": args.qnn5_train_limit,
            "val_limit": args.qnn5_val_limit,
            "test_limit": args.qnn5_test_limit,
            "mlp_hidden": (32, 16),
        },
        {
            "feature_dim": "8D",
            "dataset": dataset8,
            "dataset_path": dataset8_path,
            "feature_names": EXTENDED_STRUCTURE_TENSOR_FEATURE_NAMES,
            "extractor": extract_extended_structure_tensor_features,
            "qnn_config": QNNConfig(
                n_qubits=8,
                n_layers=2,
                encoding_type="ryrz",
                entanglement="ring",
                readout="all_zz",
                trainable_input_scaling=True,
                init_scale=0.01,
            ),
            "train_limit": args.qnn8_train_limit,
            "val_limit": args.qnn8_val_limit,
            "test_limit": args.qnn8_test_limit,
            "mlp_hidden": (48, 24),
        },
    ]

    rows = []
    for spec in run_specs:
        print(f"Preparing {spec['feature_dim']} models...")
        mlp = train_mlp(spec["dataset"], spec["mlp_hidden"], args.seed)
        mlp_val_scores = mlp.predict_proba(spec["dataset"]["X_val"])[:, 1]
        mlp_threshold = choose_threshold_by_f1(spec["dataset"]["y_val"], mlp_val_scores)

        qnn_run_dir = train_or_load_qnn(spec, run_root, args)
        qnn_model, qnn_normalizer = load_qnn_artifacts(qnn_run_dir)
        qnn_val_scores = qnn_probabilities(qnn_model, qnn_normalizer, spec["dataset"]["X_val"])
        qnn_threshold = choose_threshold_by_f1(spec["dataset"]["y_val"], qnn_val_scores)

        rows.extend(
            evaluate_feature_dimension(
                spec=spec,
                mlp=mlp,
                mlp_threshold=mlp_threshold,
                qnn_model=qnn_model,
                qnn_normalizer=qnn_normalizer,
                qnn_threshold=qnn_threshold,
                args=args,
            )
        )

    write_rows(output_dir / "feature_dim_noise_comparison.csv", rows)
    (output_dir / "feature_dim_noise_comparison.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    save_comparison_chart(rows, output_dir / "feature_dim_noise_comparison.png")
    write_summary(rows, output_dir / "feature_dim_noise_summary.md")

    print(f"Wrote {output_dir / 'feature_dim_noise_comparison.csv'}")
    print(f"Wrote {output_dir / 'feature_dim_noise_comparison.png'}")
    print(f"Wrote {output_dir / 'feature_dim_noise_summary.md'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare 5D and 8D QNN noise robustness against baselines.")
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--images-per-scene", type=int, default=100)
    parser.add_argument("--patch-size", type=int, default=9)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--force-rebuild-data", action="store_true")
    parser.add_argument("--reuse-qnn", action="store_true", help="Reuse compatible QNN run directories if present.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--qnn-epochs", type=int, default=14)
    parser.add_argument("--qnn-batch-size", type=int, default=24)
    parser.add_argument("--qnn-learning-rate", type=float, default=0.01)
    parser.add_argument("--qnn5-train-limit", type=int, default=160)
    parser.add_argument("--qnn5-val-limit", type=int, default=80)
    parser.add_argument("--qnn5-test-limit", type=int, default=80)
    parser.add_argument("--qnn8-train-limit", type=int, default=220)
    parser.add_argument("--qnn8-val-limit", type=int, default=100)
    parser.add_argument("--qnn8-test-limit", type=int, default=100)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a fast smoke protocol with fewer images, epochs, and QNN training samples.",
    )
    args = parser.parse_args()
    if args.smoke:
        args.images_per_scene = min(args.images_per_scene, 12)
        args.qnn_epochs = min(args.qnn_epochs, 2)
        args.qnn5_train_limit = min(args.qnn5_train_limit, 24)
        args.qnn5_val_limit = min(args.qnn5_val_limit, 16)
        args.qnn5_test_limit = min(args.qnn5_test_limit, 16)
        args.qnn8_train_limit = min(args.qnn8_train_limit, 24)
        args.qnn8_val_limit = min(args.qnn8_val_limit, 16)
        args.qnn8_test_limit = min(args.qnn8_test_limit, 16)
    return args


def ensure_feature_datasets(args: argparse.Namespace, data_dir: Path, output_dir: Path) -> tuple[Path, Path]:
    dataset5_path = data_dir / "feature_dataset.npz"
    dataset8_path = data_dir / "feature_dataset_extended.npz"
    if args.force_rebuild_data:
        generated_dir = output_dir / "feature_dim_noise_comparison_data"
        generated_dir.mkdir(parents=True, exist_ok=True)
        dataset5_path = generated_dir / "feature_dataset_5d.npz"
        dataset8_path = generated_dir / "feature_dataset_8d.npz"
    elif dataset5_path.exists() and dataset8_path.exists():
        try:
            validate_shared_protocol(load_feature_dataset(dataset5_path), load_feature_dataset(dataset8_path))
            return dataset5_path, dataset8_path
        except (KeyError, ValueError) as exc:
            print(f"Existing data/ feature datasets are not protocol-aligned: {exc}")
            print("Writing aligned comparison datasets under the output directory instead.")
            generated_dir = output_dir / "feature_dim_noise_comparison_data"
            generated_dir.mkdir(parents=True, exist_ok=True)
            dataset5_path = generated_dir / "feature_dataset_5d.npz"
            dataset8_path = generated_dir / "feature_dataset_8d.npz"

    rng = np.random.default_rng(args.seed)
    samples = generate_clean_samples(rng, args.images_per_scene)
    images = np.stack([sample.image for sample in samples]).astype(np.float32)
    keypoints = np.stack([sample.points_xy for sample in samples]).astype(np.float32)
    scene_labels = np.array([sample.scene_type for sample in samples])
    patches, labels, centers, image_ids = build_patch_dataset(samples, args.patch_size)
    train_images, val_images, test_images = split_image_ids(scene_labels, args.seed)

    save_feature_dataset(
        dataset5_path,
        extract_structure_tensor_features(patches),
        STRUCTURE_TENSOR_FEATURE_NAMES,
        labels,
        centers,
        image_ids,
        train_images,
        val_images,
        test_images,
        images,
        keypoints,
        scene_labels,
    )
    save_feature_dataset(
        dataset8_path,
        extract_extended_structure_tensor_features(patches),
        EXTENDED_STRUCTURE_TENSOR_FEATURE_NAMES,
        labels,
        centers,
        image_ids,
        train_images,
        val_images,
        test_images,
        images,
        keypoints,
        scene_labels,
    )
    return dataset5_path, dataset8_path


def save_feature_dataset(
    path: Path,
    features: np.ndarray,
    feature_names: list[str],
    labels: np.ndarray,
    centers: np.ndarray,
    image_ids: np.ndarray,
    train_images: np.ndarray,
    val_images: np.ndarray,
    test_images: np.ndarray,
    images: np.ndarray,
    keypoints: np.ndarray,
    scene_labels: np.ndarray,
) -> None:
    payload = make_feature_splits(features, labels, centers, image_ids, train_images, val_images, test_images)
    np.savez_compressed(
        path,
        **payload,
        feature_names=np.array(feature_names),
        images=images,
        keypoints=keypoints,
        scene_types=scene_labels,
        split_train_image_ids=train_images,
        split_val_image_ids=val_images,
        split_test_image_ids=test_images,
    )


def load_feature_dataset(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def validate_shared_protocol(dataset5: dict[str, np.ndarray], dataset8: dict[str, np.ndarray]) -> None:
    keys = [
        "y_train",
        "y_val",
        "y_test",
        "train_centers",
        "val_centers",
        "test_centers",
        "train_image_ids",
        "val_image_ids",
        "test_image_ids",
        "split_train_image_ids",
        "split_val_image_ids",
        "split_test_image_ids",
    ]
    for key in keys:
        if key not in dataset5 or key not in dataset8:
            raise KeyError(f"Both feature datasets must include {key}.")
        if not np.array_equal(dataset5[key], dataset8[key]):
            raise ValueError(f"5D and 8D datasets do not share the same protocol for {key}.")


def train_mlp(dataset: dict[str, np.ndarray], hidden_layers: tuple[int, ...], seed: int):
    mlp = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            activation="relu",
            solver="adam",
            max_iter=350,
            learning_rate_init=0.003,
            batch_size=128,
            random_state=seed,
        ),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        mlp.fit(dataset["X_train"], dataset["y_train"].astype(int))
    return mlp


def train_or_load_qnn(spec: dict, run_root: Path, args: argparse.Namespace) -> Path:
    feature_dim = str(spec["feature_dim"])
    run_dir = run_root / f"{feature_dim.lower()}_qnn"
    if args.reuse_qnn and (run_dir / "best_model.pt").exists() and (run_dir / "normalizer.npz").exists():
        return run_dir

    if run_dir.exists():
        shutil.rmtree(run_dir)
    subset_path = run_root / f"{feature_dim.lower()}_qnn_train_subset.npz"
    save_qnn_subset(
        source_path=spec["dataset_path"],
        subset_path=subset_path,
        train_limit=int(spec["train_limit"]),
        val_limit=int(spec["val_limit"]),
        test_limit=int(spec["test_limit"]),
        seed=args.seed + (5 if feature_dim == "5D" else 8),
    )
    train_qnn_experiment(
        data_path=subset_path,
        output_dir=run_dir,
        qnn_config=spec["qnn_config"],
        train_config=TrainConfig(
            learning_rate=args.qnn_learning_rate,
            batch_size=args.qnn_batch_size,
            epochs=args.qnn_epochs,
            seed=args.seed,
        ),
        feature_names=spec["feature_names"],
        device_name=args.device,
    )
    return run_dir


def save_qnn_subset(source_path: Path, subset_path: Path, train_limit: int, val_limit: int, test_limit: int, seed: int) -> None:
    data = np.load(source_path, allow_pickle=True)
    payload = {"feature_names": data["feature_names"]}
    for split, limit in [("train", train_limit), ("val", val_limit), ("test", test_limit)]:
        indices = stratified_indices(data[f"y_{split}"], limit, seed + len(split))
        payload[f"X_{split}"] = data[f"X_{split}"][indices]
        payload[f"y_{split}"] = data[f"y_{split}"][indices]
    np.savez_compressed(subset_path, **payload)


def stratified_indices(y: np.ndarray, limit: int, seed: int) -> np.ndarray:
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


def load_qnn_artifacts(run_dir: Path) -> tuple[DataReuploadingQNN, FeatureNormalizer]:
    checkpoint = torch.load(run_dir / "best_model.pt", map_location="cpu")
    model = DataReuploadingQNN(**checkpoint["qnn_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, FeatureNormalizer.load(run_dir / "normalizer.npz")


def evaluate_feature_dimension(
    spec: dict,
    mlp,
    mlp_threshold: float,
    qnn_model: DataReuploadingQNN,
    qnn_normalizer: FeatureNormalizer,
    qnn_threshold: float,
    args: argparse.Namespace,
) -> list[dict]:
    dataset = spec["dataset"]
    rows = []
    for case_name, noise_type, value in NOISE_CASES:
        noisy_images = apply_noise_to_images(dataset["images"], noise_type, value, args.seed)
        test_features = extract_features_for_centers(
            noisy_images,
            dataset["test_image_ids"],
            dataset["test_centers"],
            args.patch_size,
            spec["extractor"],
        )
        labels = dataset["y_test"].astype(int)
        mlp_scores = mlp.predict_proba(test_features)[:, 1]
        qnn_scores = qnn_probabilities(qnn_model, qnn_normalizer, test_features)

        rows.append(
            result_row(
                feature_dim=spec["feature_dim"],
                case_name=case_name,
                method="MLP",
                train_samples=len(dataset["y_train"]),
                test_samples=len(labels),
                threshold=mlp_threshold,
                metrics=probability_metrics(labels, mlp_scores, mlp_threshold),
            )
        )
        rows.append(
            result_row(
                feature_dim=spec["feature_dim"],
                case_name=case_name,
                method="QNN",
                train_samples=int(spec["train_limit"]) if int(spec["train_limit"]) > 0 else len(dataset["y_train"]),
                test_samples=len(labels),
                threshold=qnn_threshold,
                metrics=probability_metrics(labels, qnn_scores, qnn_threshold),
            )
        )
        if spec["feature_dim"] == "5D":
            rows.extend(
                evaluate_classical_baselines(
                    noisy_images=noisy_images,
                    keypoints=dataset["keypoints"],
                    test_images=dataset["split_test_image_ids"],
                    case_name=case_name,
                    train_samples=len(dataset["y_train"]),
                )
            )
    return rows


def apply_noise_to_images(images: np.ndarray, noise_type: str, value: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + int(value * 1000))
    out = np.asarray(images, dtype=np.float32).copy()
    if noise_type == "none":
        return out
    if noise_type == "gaussian":
        return np.clip(out + rng.normal(0.0, value, size=out.shape).astype(np.float32), 0.0, 1.0)
    if noise_type == "saltpepper":
        mask = rng.random(out.shape)
        out[mask < value / 2.0] = 0.0
        out[(mask >= value / 2.0) & (mask < value)] = 1.0
        return out
    if noise_type == "blur":
        blurred = []
        for image in out:
            pil = Image.fromarray(np.uint8(np.clip(image, 0.0, 1.0) * 255))
            blurred.append(np.asarray(pil.filter(ImageFilter.GaussianBlur(radius=value)), dtype=np.float32) / 255.0)
        return np.stack(blurred).astype(np.float32)
    raise ValueError(noise_type)


def extract_features_for_centers(
    images: np.ndarray,
    image_ids: np.ndarray,
    centers: np.ndarray,
    patch_size: int,
    extractor,
) -> np.ndarray:
    patches = []
    for image_id, center in zip(image_ids, centers):
        patch = extract_patches(images[int(image_id)], np.asarray([center], dtype=np.float32), patch_size)[0]
        patches.append(patch)
    return extractor(np.stack(patches).astype(np.float32))


def qnn_probabilities(model: DataReuploadingQNN, normalizer: FeatureNormalizer, features: np.ndarray) -> np.ndarray:
    phi = normalizer.transform(features)
    with torch.no_grad():
        return model.predict_proba(torch.as_tensor(phi, dtype=torch.float32)).detach().cpu().numpy().reshape(-1)


def choose_threshold_by_f1(y_true: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=int).reshape(-1)
    s = np.asarray(scores, dtype=float).reshape(-1)
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], s)))
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in candidates:
        f1 = f1_score(y, (s >= threshold).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_threshold = float(threshold)
    return best_threshold


def probability_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int).reshape(-1)
    s = np.asarray(scores, dtype=float).reshape(-1)
    pred = (s >= threshold).astype(int)
    return {
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "pr_auc": safe_average_precision(y, s),
    }


def evaluate_classical_baselines(
    noisy_images: np.ndarray,
    keypoints: np.ndarray,
    test_images: np.ndarray,
    case_name: str,
    train_samples: int,
) -> list[dict]:
    rows = []
    for method in ["Harris", "FAST", "ORB"]:
        tp = fp = fn = 0
        unavailable = False
        for image_id in test_images:
            image = noisy_images[int(image_id)]
            try:
                if method == "Harris":
                    points = run_harris(image, threshold_rel=0.01, max_points=40)
                elif method == "FAST":
                    points = run_fast(image, threshold=20, max_points=40)
                else:
                    points = run_orb(image, max_points=40)
            except ImportError:
                unavailable = True
                break
            metrics = evaluate_points(points, keypoints[int(image_id)])
            tp += metrics.true_positives
            fp += metrics.false_positives
            fn += metrics.false_negatives
        if unavailable:
            precision = recall = f1 = float("nan")
        else:
            precision = tp / max(1, tp + fp)
            recall = tp / max(1, tp + fn)
            f1 = 2 * precision * recall / max(1e-12, precision + recall)
        rows.append(
            result_row(
                feature_dim="image",
                case_name=case_name,
                method=method,
                train_samples=train_samples,
                test_samples=int(len(test_images)),
                threshold=float("nan"),
                metrics={"precision": precision, "recall": recall, "f1": f1, "pr_auc": float("nan")},
            )
        )
    return rows


def result_row(
    feature_dim: str,
    case_name: str,
    method: str,
    train_samples: int,
    test_samples: int,
    threshold: float,
    metrics: dict[str, float],
) -> dict:
    return {
        "feature_dim": feature_dim,
        "case": case_name,
        "method": method,
        "train_samples": int(train_samples),
        "test_samples": int(test_samples),
        "threshold": float(threshold),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1": float(metrics["f1"]),
        "pr_auc": float(metrics["pr_auc"]),
    }


def safe_average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    try:
        return float(average_precision_score(np.asarray(y_true, dtype=int), np.asarray(scores, dtype=float)))
    except ValueError:
        return float("nan")


def write_rows(path: Path, rows: list[dict]) -> None:
    fieldnames = ["feature_dim", "case", "method", "train_samples", "test_samples", "threshold", "precision", "recall", "f1", "pr_auc"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_comparison_chart(rows: list[dict], path: Path) -> None:
    cases = [case for case, _, _ in NOISE_CASES]
    learned_series = [
        ("5D", "QNN"),
        ("5D", "MLP"),
        ("8D", "QNN"),
        ("8D", "MLP"),
    ]
    salt_case = "saltpepper_0.03"
    baseline_rows = [
        row
        for row in rows
        if row["case"] == salt_case and (row["feature_dim"], row["method"]) in [("5D", "QNN"), ("5D", "MLP"), ("8D", "QNN"), ("8D", "MLP")]
    ]
    baseline_rows += [row for row in rows if row["case"] == salt_case and row["feature_dim"] == "image"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax = axes[0]
    for feature_dim, method in learned_series:
        values = [
            find_metric(rows, feature_dim=feature_dim, method=method, case_name=case, metric="f1")
            for case in cases
        ]
        ax.plot(cases, values, marker="o", label=f"{feature_dim} {method}")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("F1")
    ax.set_title("Clean-trained models under noisy test features")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    ax.tick_params(axis="x", rotation=25)

    ax = axes[1]
    labels = [f"{row['feature_dim']} {row['method']}" if row["feature_dim"] != "image" else row["method"] for row in baseline_rows]
    values = [float(row["f1"]) for row in baseline_rows]
    ax.bar(np.arange(len(values)), values, color="tab:orange")
    ax.set_xticks(np.arange(len(values)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("F1")
    ax.set_title("Salt-and-pepper comparison")
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_summary(rows: list[dict], path: Path) -> None:
    lines = [
        "# 5D/8D QNN Noise Robustness Summary",
        "",
        "Protocol: clean training, clean-validation threshold selection, noisy held-out test evaluation.",
        "",
        "## Main Answers",
        "",
        answer_qnn_vs_mlp(rows, "5D"),
        answer_qnn_vs_mlp(rows, "8D"),
        answer_5d_vs_8d(rows),
        "",
        "## Clean and Salt-Pepper Snapshot",
        "",
        "| Feature | Method | Clean F1 | Salt-Pepper F1 | Clean PR-AUC | Salt-Pepper PR-AUC |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for feature_dim in ["5D", "8D"]:
        for method in ["QNN", "MLP"]:
            clean_f1 = find_metric(rows, feature_dim, method, "clean", "f1")
            sp_f1 = find_metric(rows, feature_dim, method, "saltpepper_0.03", "f1")
            clean_pr = find_metric(rows, feature_dim, method, "clean", "pr_auc")
            sp_pr = find_metric(rows, feature_dim, method, "saltpepper_0.03", "pr_auc")
            lines.append(f"| {feature_dim} | {method} | {clean_f1:.4f} | {sp_f1:.4f} | {clean_pr:.4f} | {sp_pr:.4f} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def answer_qnn_vs_mlp(rows: list[dict], feature_dim: str) -> str:
    qnn_clean = find_metric(rows, feature_dim, "QNN", "clean", "f1")
    qnn_sp = find_metric(rows, feature_dim, "QNN", "saltpepper_0.03", "f1")
    mlp_clean = find_metric(rows, feature_dim, "MLP", "clean", "f1")
    mlp_sp = find_metric(rows, feature_dim, "MLP", "saltpepper_0.03", "f1")
    qnn_drop = qnn_clean - qnn_sp
    mlp_drop = mlp_clean - mlp_sp
    verdict = "more stable than" if qnn_drop < mlp_drop else "less stable than"
    return f"- {feature_dim} QNN is {verdict} {feature_dim} MLP by F1 drop from clean to salt-pepper ({qnn_drop:.4f} vs {mlp_drop:.4f})."


def answer_5d_vs_8d(rows: list[dict]) -> str:
    qnn5_clean = find_metric(rows, "5D", "QNN", "clean", "f1")
    qnn8_clean = find_metric(rows, "8D", "QNN", "clean", "f1")
    qnn5_sp = find_metric(rows, "5D", "QNN", "saltpepper_0.03", "f1")
    qnn8_sp = find_metric(rows, "8D", "QNN", "saltpepper_0.03", "f1")
    direction = "improves" if qnn8_sp > qnn5_sp else "does not improve"
    return f"- Moving from 5D to 8D {direction} QNN salt-pepper F1 ({qnn5_sp:.4f} -> {qnn8_sp:.4f}); clean F1 changes from {qnn5_clean:.4f} to {qnn8_clean:.4f}."


def find_metric(rows: list[dict], feature_dim: str, method: str, case_name: str, metric: str) -> float:
    for row in rows:
        if row["feature_dim"] == feature_dim and row["method"] == method and row["case"] == case_name:
            return float(row[metric])
    return float("nan")


if __name__ == "__main__":
    main()
