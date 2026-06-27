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
    extract_extended_structure_tensor_features,
)
from qcd_data.synthetic import extract_patches
from qnn_circuit import DataReuploadingQNN
from scripts.run_day2_pipeline import (
    build_patch_dataset,
    generate_clean_samples,
    make_feature_splits,
    save_day2_comparison_overlay,
    save_qnn_training_curve,
    select_points_from_scores,
    sliding_centers,
    split_image_ids,
)
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
    patches, labels, centers, image_ids = build_patch_dataset(samples, args.patch_size)
    features = extract_extended_structure_tensor_features(patches)
    train_images, val_images, test_images = split_image_ids(scene_labels, args.seed)
    payload = make_feature_splits(features, labels, centers, image_ids, train_images, val_images, test_images)
    feature_dataset_path = data_dir / "feature_dataset_extended.npz"
    np.savez_compressed(
        feature_dataset_path,
        **payload,
        feature_names=np.array(EXTENDED_STRUCTURE_TENSOR_FEATURE_NAMES),
        images=images,
        keypoints=keypoints,
        scene_types=scene_labels,
        split_train_image_ids=train_images,
        split_val_image_ids=val_images,
        split_test_image_ids=test_images,
    )

    mlp, mlp_metrics = train_extended_mlp(payload, outputs_dir, args.seed)
    ablation_rows, best_run = run_qnn_ablation(feature_dataset_path, outputs_dir, data_dir, args)
    improved_metrics = write_improved_metrics(best_run, outputs_dir)
    qnn_model, normalizer = load_qnn_artifacts(best_run["run_dir"])

    noise_rows = run_noise_robustness(
        images=images,
        keypoints=keypoints,
        payload=payload,
        test_images=test_images,
        mlp=mlp,
        qnn_model=qnn_model,
        normalizer=normalizer,
        args=args,
    )
    write_table(outputs_dir / "noise_robustness_results.csv", noise_rows)
    (outputs_dir / "noise_robustness_results.json").write_text(json.dumps(noise_rows, indent=2), encoding="utf-8")
    save_noise_chart(noise_rows, outputs_dir / "noise_robustness.png")

    test_image_id = int(test_images[0])
    comparison_points = build_improved_comparison_points(images[test_image_id], mlp, qnn_model, normalizer, args.patch_size)
    save_day2_comparison_overlay(
        images[test_image_id],
        keypoints[test_image_id],
        comparison_points,
        outputs_dir / "improved_comparison_overlay.png",
    )
    write_improvement_report(outputs_dir / "qnn_improvement_demo.html", ablation_rows, noise_rows, mlp_metrics, improved_metrics)
    write_improvement_markdown(outputs_dir / "qnn_improvement_summary.md", ablation_rows, noise_rows, mlp_metrics, improved_metrics)

    print("Improvement experiments complete.")
    print(f"Wrote {outputs_dir / 'qnn_ablation_results.csv'}")
    print(f"Wrote {outputs_dir / 'noise_robustness_results.csv'}")
    print(f"Wrote {outputs_dir / 'qnn_improvement_demo.html'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run QNN improvement experiments.")
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--images-per-scene", type=int, default=100)
    parser.add_argument("--patch-size", type=int, default=9)
    parser.add_argument("--ablation-epochs", type=int, default=5)
    parser.add_argument("--improved-epochs", type=int, default=14)
    parser.add_argument("--ablation-train-limit", type=int, default=72)
    parser.add_argument("--ablation-val-limit", type=int, default=36)
    parser.add_argument("--ablation-test-limit", type=int, default=36)
    parser.add_argument("--improved-train-limit", type=int, default=220)
    parser.add_argument("--improved-val-limit", type=int, default=100)
    parser.add_argument("--improved-test-limit", type=int, default=100)
    return parser.parse_args()


def train_extended_mlp(payload: dict[str, np.ndarray], outputs_dir: Path, seed: int):
    mlp = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(48, 24),
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
        mlp.fit(payload["X_train"], payload["y_train"].astype(int))
    probs = mlp.predict_proba(payload["X_test"])[:, 1]
    metrics = probability_metrics(payload["y_test"], probs)
    metrics.update(
        {
            "feature_names": EXTENDED_STRUCTURE_TENSOR_FEATURE_NAMES,
            "train_samples": int(len(payload["y_train"])),
            "test_samples": int(len(payload["y_test"])),
        }
    )
    (outputs_dir / "improved_mlp_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_loss_curve(mlp.named_steps["mlpclassifier"].loss_curve_, outputs_dir / "improved_mlp_training_curve.png", "Improved MLP training loss")
    return mlp, metrics


def run_qnn_ablation(feature_dataset_path: Path, outputs_dir: Path, data_dir: Path, args: argparse.Namespace):
    configs = [
        {"name": "L1_ring_Z", "layers": 1, "entanglement": "ring", "readout": "all", "trainable_input_scaling": False, "init_scale": 0.01, "epochs": args.ablation_epochs},
        {"name": "L2_ring_Z", "layers": 2, "entanglement": "ring", "readout": "all", "trainable_input_scaling": False, "init_scale": 0.01, "epochs": args.ablation_epochs},
        {"name": "L3_ring_Z", "layers": 3, "entanglement": "ring", "readout": "all", "trainable_input_scaling": False, "init_scale": 0.01, "epochs": args.ablation_epochs},
        {"name": "L2_none_ZZ", "layers": 2, "entanglement": "none", "readout": "all_zz", "trainable_input_scaling": False, "init_scale": 0.01, "epochs": args.ablation_epochs},
        {"name": "L2_linear_ZZ", "layers": 2, "entanglement": "linear", "readout": "all_zz", "trainable_input_scaling": False, "init_scale": 0.01, "epochs": args.ablation_epochs},
        {"name": "L2_ring_ZZ", "layers": 2, "entanglement": "ring", "readout": "all_zz", "trainable_input_scaling": False, "init_scale": 0.01, "epochs": args.ablation_epochs},
        {"name": "L2_ring_ZZ_scale", "layers": 2, "entanglement": "ring", "readout": "all_zz", "trainable_input_scaling": True, "init_scale": 0.01, "epochs": args.ablation_epochs},
        {"name": "improved_L2_ring_ZZ_scale_more_data", "layers": 2, "entanglement": "ring", "readout": "all_zz", "trainable_input_scaling": True, "init_scale": 0.01, "epochs": args.improved_epochs, "improved": True},
    ]
    rows = []
    best_run = None
    for index, cfg in enumerate(configs):
        improved = bool(cfg.get("improved", False))
        subset_path = data_dir / f"qnn_{cfg['name']}_subset.npz"
        counts = save_subset(
            feature_dataset_path,
            subset_path,
            train_limit=args.improved_train_limit if improved else args.ablation_train_limit,
            val_limit=args.improved_val_limit if improved else args.ablation_val_limit,
            test_limit=args.improved_test_limit if improved else args.ablation_test_limit,
            seed=args.seed + index,
        )
        run_dir = outputs_dir / f"qnn_improvement_{cfg['name']}"
        metrics = train_qnn_experiment(
            data_path=subset_path,
            output_dir=run_dir,
            qnn_config=QNNConfig(
                n_qubits=len(EXTENDED_STRUCTURE_TENSOR_FEATURE_NAMES),
                n_layers=int(cfg["layers"]),
                encoding_type="ryrz",
                entanglement=str(cfg["entanglement"]),
                readout=str(cfg["readout"]),
                trainable_input_scaling=bool(cfg["trainable_input_scaling"]),
                init_scale=float(cfg["init_scale"]),
            ),
            train_config=TrainConfig(
                learning_rate=0.01,
                batch_size=24,
                epochs=int(cfg["epochs"]),
                seed=args.seed,
            ),
            feature_names=EXTENDED_STRUCTURE_TENSOR_FEATURE_NAMES,
            device_name="cpu",
        )
        history = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        row = {
            "name": str(cfg["name"]),
            "layers": int(cfg["layers"]),
            "entanglement": str(cfg["entanglement"]),
            "readout": str(cfg["readout"]),
            "trainable_input_scaling": bool(cfg["trainable_input_scaling"]),
            "epochs": int(cfg["epochs"]),
            "train_samples": counts["train_samples"],
            "test_samples": counts["test_samples"],
            **prefix_metrics("test", metrics["test"]),
            **prefix_metrics("best_val", metrics["best"]),
        }
        rows.append(row)
        if improved:
            best_run = {"name": cfg["name"], "run_dir": run_dir, "metrics": metrics, "history": history, "counts": counts}

    write_table(outputs_dir / "qnn_ablation_results.csv", rows)
    (outputs_dir / "qnn_ablation_results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    save_ablation_chart(rows, outputs_dir / "qnn_ablation_results.png")
    if best_run is None:
        raise RuntimeError("Improved QNN run was not executed.")
    return rows, best_run


def save_subset(source_path: Path, subset_path: Path, train_limit: int, val_limit: int, test_limit: int, seed: int) -> dict[str, int]:
    data = np.load(source_path)
    payload = {"feature_names": data["feature_names"]}
    counts = {}
    for split, limit in [("train", train_limit), ("val", val_limit), ("test", test_limit)]:
        indices = stratified_indices(data[f"y_{split}"], limit, seed + len(split))
        payload[f"X_{split}"] = data[f"X_{split}"][indices]
        payload[f"y_{split}"] = data[f"y_{split}"][indices]
        counts[f"{split}_samples"] = int(len(indices))
        counts[f"{split}_positives"] = int(np.sum(payload[f"y_{split}"] == 1))
    np.savez_compressed(subset_path, **payload)
    return counts


def stratified_indices(y: np.ndarray, limit: int, seed: int) -> np.ndarray:
    y = np.asarray(y).reshape(-1)
    if limit >= len(y):
        return np.arange(len(y))
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    pos_count = max(1, min(len(pos), int(round(limit * len(pos) / len(y)))))
    neg_count = max(1, min(len(neg), limit - pos_count))
    chosen = np.concatenate([rng.choice(pos, pos_count, replace=False), rng.choice(neg, neg_count, replace=False)])
    return np.sort(chosen)


def write_improved_metrics(best_run: dict, outputs_dir: Path) -> dict:
    metrics = json.loads((best_run["run_dir"] / "metrics.json").read_text(encoding="utf-8"))
    metrics["subset_counts"] = best_run["counts"]
    metrics["model_name"] = best_run["name"]
    metrics["feature_names"] = EXTENDED_STRUCTURE_TENSOR_FEATURE_NAMES
    shutil.copyfile(best_run["run_dir"] / "normalizer.npz", outputs_dir / "improved_qnn_normalizer.npz")
    shutil.copyfile(best_run["run_dir"] / "metrics.json", outputs_dir / "improved_qnn_raw_metrics.json")
    save_qnn_training_curve(metrics["history"], outputs_dir / "improved_qnn_training_curve.png")
    (outputs_dir / "improved_qnn_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def load_qnn_artifacts(run_dir: Path) -> tuple[DataReuploadingQNN, FeatureNormalizer]:
    checkpoint = torch.load(run_dir / "best_model.pt", map_location="cpu")
    model = DataReuploadingQNN(**checkpoint["qnn_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, FeatureNormalizer.load(run_dir / "normalizer.npz")


def run_noise_robustness(
    images: np.ndarray,
    keypoints: np.ndarray,
    payload: dict[str, np.ndarray],
    test_images: np.ndarray,
    mlp,
    qnn_model: DataReuploadingQNN,
    normalizer: FeatureNormalizer,
    args: argparse.Namespace,
) -> list[dict[str, str | float]]:
    cases = [
        ("clean", "none", 0.0),
        ("gaussian_0.04", "gaussian", 0.04),
        ("gaussian_0.08", "gaussian", 0.08),
        ("blur_0.9", "blur", 0.9),
        ("saltpepper_0.03", "saltpepper", 0.03),
    ]
    subset = stratified_indices(payload["y_test"], min(120, len(payload["y_test"])), args.seed + 991)
    rows = []
    for case_name, noise_type, value in cases:
        noisy_images = apply_noise_to_images(images, noise_type, value, args.seed)
        labels = payload["y_test"][subset].astype(int)
        centers = payload["test_centers"][subset]
        ids = payload["test_image_ids"][subset]
        features = extract_features_for_centers(noisy_images, ids, centers, args.patch_size)
        mlp_probs = mlp.predict_proba(features)[:, 1]
        qnn_probs = qnn_probabilities(qnn_model, normalizer, features)
        rows.append({"case": case_name, "method": "MLP", **probability_metrics(labels, mlp_probs)})
        rows.append({"case": case_name, "method": "QNN", **probability_metrics(labels, qnn_probs)})

        for method in ["Harris", "FAST", "ORB"]:
            precision, recall, f1 = evaluate_image_detector(noisy_images, keypoints, test_images, method)
            rows.append({"case": case_name, "method": method, "precision": precision, "recall": recall, "f1": f1, "pr_auc": float("nan")})
    return rows


def apply_noise_to_images(images: np.ndarray, noise_type: str, value: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + int(value * 1000))
    out = images.copy()
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


def extract_features_for_centers(images: np.ndarray, image_ids: np.ndarray, centers: np.ndarray, patch_size: int) -> np.ndarray:
    patches = []
    for image_id, center in zip(image_ids, centers):
        patches.append(extract_patches(images[int(image_id)], np.asarray([center], dtype=np.float32), patch_size)[0])
    return extract_extended_structure_tensor_features(np.stack(patches).astype(np.float32))


def evaluate_image_detector(images: np.ndarray, keypoints: np.ndarray, test_images: np.ndarray, method: str) -> tuple[float, float, float]:
    tp = fp = fn = 0
    for image_id in test_images:
        image = images[int(image_id)]
        if method == "Harris":
            points = run_harris(image, threshold_rel=0.01, max_points=40)
        elif method == "FAST":
            points = run_fast(image, threshold=20, max_points=40)
        elif method == "ORB":
            points = run_orb(image, max_points=40)
        else:
            raise ValueError(method)
        metrics = evaluate_points(points, keypoints[int(image_id)])
        tp += metrics.true_positives
        fp += metrics.false_positives
        fn += metrics.false_negatives
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return precision, recall, f1


def build_improved_comparison_points(image: np.ndarray, mlp, qnn_model: DataReuploadingQNN, normalizer: FeatureNormalizer, patch_size: int):
    centers = sliding_centers(image.shape, patch_size)
    features = extract_extended_structure_tensor_features(extract_patches(image, centers, patch_size))
    return {
        "Harris": run_harris(image, threshold_rel=0.01, max_points=40),
        "FAST": run_fast(image, threshold=20, max_points=40),
        "ORB": run_orb(image, max_points=40),
        "MLP": select_points_from_scores(centers, mlp.predict_proba(features)[:, 1], threshold=0.5),
        "QNN": select_points_from_scores(centers, qnn_probabilities(qnn_model, normalizer, features), threshold=0.5),
    }


def qnn_probabilities(model: DataReuploadingQNN, normalizer: FeatureNormalizer, features: np.ndarray) -> np.ndarray:
    phi = normalizer.transform(features)
    with torch.no_grad():
        return model.predict_proba(torch.as_tensor(phi, dtype=torch.float32)).detach().cpu().numpy().reshape(-1)


def probability_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int).reshape(-1)
    p = np.asarray(probabilities, dtype=float).reshape(-1)
    pred = (p >= 0.5).astype(int)
    return {
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y, p)),
    }


def prefix_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": float(value) for key, value in metrics.items() if isinstance(value, (int, float))}


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


def save_ablation_chart(rows: list[dict], path: Path) -> None:
    names = [row["name"] for row in rows]
    f1 = [float(row["test_f1"]) for row in rows]
    pr_auc = [float(row["test_pr_auc"]) for row in rows]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(x - 0.18, f1, width=0.36, label="F1")
    ax.bar(x + 0.18, pr_auc, width=0.36, label="PR-AUC")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_title("QNN ablation results")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_noise_chart(rows: list[dict], path: Path) -> None:
    cases = []
    methods = []
    for row in rows:
        if row["case"] not in cases:
            cases.append(row["case"])
        if row["method"] in ["MLP", "QNN"] and row["method"] not in methods:
            methods.append(row["method"])
    fig, ax = plt.subplots(figsize=(8, 4))
    for method in methods:
        values = [float(next(row["f1"] for row in rows if row["case"] == case and row["method"] == method)) for case in cases]
        ax.plot(cases, values, marker="o", label=method)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1")
    ax.set_title("Noise robustness on held-out patch subset")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_table(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_improvement_markdown(path: Path, ablation_rows: list[dict], noise_rows: list[dict], mlp_metrics: dict, improved_metrics: dict) -> None:
    best = improved_metrics["test"]
    text = f"""# QNN Improvement Experiments

## Main Improved Model

- Features: 8-D `[Ix, Iy, Ix2, Iy2, IxIy, lambda1, lambda2, R]`
- QNN: L=2, ring entanglement, RyRz, Z+ZZ readout, trainable input scaling
- MLP F1: {mlp_metrics['f1']:.4f}, PR-AUC: {mlp_metrics['pr_auc']:.4f}
- Improved QNN F1: {best['f1']:.4f}, PR-AUC: {best['pr_auc']:.4f}

## Key Figures

![Ablation](qnn_ablation_results.png)

![Noise](noise_robustness.png)

![Overlay](improved_comparison_overlay.png)
"""
    path.write_text(text, encoding="utf-8")


def write_improvement_report(path: Path, ablation_rows: list[dict], noise_rows: list[dict], mlp_metrics: dict, improved_metrics: dict) -> None:
    best = improved_metrics["test"]
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>QNN Improvement Demo</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #17202a; }}
    h1, h2 {{ margin-bottom: 8px; }}
    .grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 20px; align-items: start; }}
    img {{ max-width: 100%; border: 1px solid #d8dee9; }}
    table {{ border-collapse: collapse; width: 100%; margin: 14px 0 24px; }}
    th, td {{ border: 1px solid #d8dee9; padding: 8px 10px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    .note {{ background: #f5f7fb; padding: 12px 14px; border-left: 4px solid #4978bc; }}
  </style>
</head>
<body>
  <h1>QNN Improvement Demo</h1>
  <p class="note">当前目标不是宣称量子优势，而是展示 QNN 结构改进、噪声鲁棒性与消融分析。MLP 仍是 clean 条件下最强 baseline。</p>

  <h2>主模型结果</h2>
  <table>
    <tr><th>Model</th><th>Precision</th><th>Recall</th><th>F1</th><th>PR-AUC</th></tr>
    <tr><td>MLP, same 8-D features</td><td>{mlp_metrics['precision']:.4f}</td><td>{mlp_metrics['recall']:.4f}</td><td>{mlp_metrics['f1']:.4f}</td><td>{mlp_metrics['pr_auc']:.4f}</td></tr>
    <tr><td>Improved QNN</td><td>{best['precision']:.4f}</td><td>{best['recall']:.4f}</td><td>{best['f1']:.4f}</td><td>{best['pr_auc']:.4f}</td></tr>
  </table>

  <div class="grid">
    <section><h2>QNN 消融</h2><img src="qnn_ablation_results.png" /></section>
    <section><h2>噪声鲁棒性</h2><img src="noise_robustness.png" /></section>
    <section><h2>Overlay 对比</h2><img src="improved_comparison_overlay.png" /></section>
    <section><h2>训练曲线</h2><img src="improved_qnn_training_curve.png" /></section>
  </div>

  <h2>结论</h2>
  <ul>
    <li>8-D 特征、Z+ZZ readout、trainable input scaling 已接入并完成训练。</li>
    <li>消融覆盖 L=1/2/3、none/linear/ring entanglement、Z vs Z+ZZ readout。</li>
    <li>噪声实验覆盖 clean、Gaussian、blur、salt-and-pepper。</li>
    <li>后续重点应放在 hard negative mining、soft labels、layerwise training 与更多训练样本。</li>
  </ul>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
