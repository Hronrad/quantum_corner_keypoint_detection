from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
QPP_SRC = ROOT / "qpp_corner_qnn_github_package" / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(QPP_SRC) not in sys.path:
    sys.path.insert(0, str(QPP_SRC))

from qpp_corner.classical import fit_logistic, threshold_baseline
from qpp_corner.metrics import binary_metrics, choose_threshold_by_f1
from qpp_corner.normalizer import FeatureNormalizer
from qpp_corner.qnn_torch import DataReuploadingQNN1, DataReuploadingQNN2
from qpp_corner.train import predict_torch, set_seed, train_torch_classifier
from scripts.run_improvement_experiments import apply_noise_to_images, extract_features_for_centers


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    payload = np.load(args.data_path, allow_pickle=True)
    feature_names = [str(name) for name in payload["feature_names"].tolist()]

    models = train_models(payload, feature_names, args)
    cases = [
        ("clean", "none", 0.0),
        ("gaussian_0.04", "gaussian", 0.04),
        ("gaussian_0.08", "gaussian", 0.08),
        ("blur_0.9", "blur", 0.9),
        ("saltpepper_0.03", "saltpepper", 0.03),
    ]

    rows = []
    for case_name, noise_type, value in cases:
        if noise_type == "none":
            x_test = payload["X_test"].astype(np.float32)
        else:
            noisy_images = apply_noise_to_images(payload["images"], noise_type, value, args.seed)
            x_test = extract_features_for_centers(
                noisy_images,
                payload["test_image_ids"],
                payload["test_centers"],
                args.patch_size,
            )
        qpp_test = qpp_feature_sets_from_extended(x_test, feature_names)
        y_test = payload["y_test"].astype(int)
        for item in models:
            scores = predict_scores(item, qpp_test[item["feature_set"]], args)
            metrics = binary_metrics(y_test, scores, item["threshold"])
            rows.append(
                {
                    "case": case_name,
                    "name": item["name"],
                    "feature_set": item["feature_set"],
                    "model": item["model"],
                    "n_qubits": item["n_qubits"],
                    "layers": item["layers"],
                    "test_samples": int(len(y_test)),
                    "threshold": float(item["threshold"]),
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "roc_auc": metrics["roc_auc"],
                    "pr_auc": metrics["pr_auc"],
                }
            )

    args.output_dir.mkdir(exist_ok=True)
    write_rows(args.output_dir / "qpp_noise_robustness_results.csv", rows)
    (args.output_dir / "qpp_noise_robustness_results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    save_noise_chart(rows, args.output_dir / "qpp_noise_robustness.png")
    print(json.dumps(rows, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate QPP-style few-qubit models under image noise.")
    parser.add_argument("--data-path", type=Path, default=ROOT / "data" / "feature_dataset_extended.npz")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=9)
    return parser.parse_args()


def qpp_feature_sets_from_extended(x: np.ndarray, feature_names: list[str]) -> dict[str, np.ndarray]:
    lookup = {name: idx for idx, name in enumerate(feature_names)}
    lambda1 = np.clip(x[:, lookup["lambda1"]], 0.0, None)
    lambda2 = np.clip(x[:, lookup["lambda2"]], 0.0, None)
    s = lambda1 + lambda2
    eta = np.clip(4.0 * lambda1 * lambda2 / (s * s + 1e-8), 0.0, 1.0)
    log_s = np.log(s + 1e-8)
    return {
        "lambda2": lambda2.reshape(-1, 1).astype(np.float32),
        "lambda12": np.column_stack([lambda1, lambda2]).astype(np.float32),
        "logS_eta": np.column_stack([log_s, eta]).astype(np.float32),
        "scalar_c4": (log_s + 4.0 * eta).reshape(-1, 1).astype(np.float32),
    }


def train_models(payload, feature_names: list[str], args: argparse.Namespace) -> list[dict]:
    qpp = {
        split: qpp_feature_sets_from_extended(payload[f"X_{split}"].astype(np.float32), feature_names)
        for split in ["train", "val", "test"]
    }
    y_train = payload["y_train"].astype(int)
    y_val = payload["y_val"].astype(int)
    models = []

    threshold_model = threshold_baseline()
    threshold_scores = threshold_model.predict_scores(qpp["val"]["lambda2"])
    threshold, _ = choose_threshold_by_f1(y_val, threshold_scores)
    models.append(
        {
            "name": "lambda2_threshold",
            "feature_set": "lambda2",
            "model": "threshold",
            "n_qubits": 0,
            "layers": 0,
            "predictor": threshold_model,
            "threshold": threshold,
        }
    )

    logistic_normalizer = FeatureNormalizer()
    x_train = logistic_normalizer.fit_transform(qpp["train"]["logS_eta"], ["logS", "eta"])
    x_val = logistic_normalizer.transform(qpp["val"]["logS_eta"])
    logistic_model = fit_logistic(x_train, y_train, seed=args.seed)
    threshold, _ = choose_threshold_by_f1(y_val, logistic_model.predict_scores(x_val))
    models.append(
        {
            "name": "logistic_logS_eta",
            "feature_set": "logS_eta",
            "model": "logistic",
            "n_qubits": 0,
            "layers": 0,
            "predictor": logistic_model,
            "normalizer": logistic_normalizer,
            "threshold": threshold,
        }
    )

    scalar_normalizer = FeatureNormalizer()
    x_train = scalar_normalizer.to_angles(scalar_normalizer.fit_transform(qpp["train"]["scalar_c4"], ["logS_plus_4_eta"]))
    x_val = scalar_normalizer.to_angles(scalar_normalizer.transform(qpp["val"]["scalar_c4"]))
    qnn1 = DataReuploadingQNN1(n_layers=2, encoding="ryrz")
    qnn1, _ = train_torch_classifier(
        qnn1,
        x_train,
        y_train,
        x_val,
        y_val,
        out_dir=args.output_dir / "qpp_noise_1q_scalar_c4_run",
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        seed=args.seed,
        device="cpu",
    )
    threshold, _ = choose_threshold_by_f1(y_val, predict_torch(qnn1, x_val, batch_size=args.batch_size, device="cpu"))
    models.append(
        {
            "name": "qpp_1q_scalar_c4_L2",
            "feature_set": "scalar_c4",
            "model": "qpp_qnn1",
            "n_qubits": 1,
            "layers": 2,
            "predictor": qnn1,
            "normalizer": scalar_normalizer,
            "threshold": threshold,
        }
    )

    lambda_normalizer = FeatureNormalizer()
    x_train = lambda_normalizer.to_angles(lambda_normalizer.fit_transform(qpp["train"]["lambda12"], ["lambda1", "lambda2"]))
    x_val = lambda_normalizer.to_angles(lambda_normalizer.transform(qpp["val"]["lambda12"]))
    qnn2 = DataReuploadingQNN2(n_layers=2, encoding="ryrz", entanglement="linear_01", readout="z_z_zz")
    qnn2, _ = train_torch_classifier(
        qnn2,
        x_train,
        y_train,
        x_val,
        y_val,
        out_dir=args.output_dir / "qpp_noise_2q_lambda12_run",
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        seed=args.seed,
        device="cpu",
    )
    threshold, _ = choose_threshold_by_f1(y_val, predict_torch(qnn2, x_val, batch_size=args.batch_size, device="cpu"))
    models.append(
        {
            "name": "qpp_2q_lambda12_L2",
            "feature_set": "lambda12",
            "model": "qpp_qnn2",
            "n_qubits": 2,
            "layers": 2,
            "predictor": qnn2,
            "normalizer": lambda_normalizer,
            "threshold": threshold,
        }
    )
    return models


def predict_scores(model_item: dict, x: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    if model_item["model"] == "threshold":
        return model_item["predictor"].predict_scores(x)
    if model_item["model"] == "logistic":
        return model_item["predictor"].predict_scores(model_item["normalizer"].transform(x))
    phi = model_item["normalizer"].to_angles(model_item["normalizer"].transform(x))
    return predict_torch(model_item["predictor"], phi, batch_size=args.batch_size, device="cpu")


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_noise_chart(rows: list[dict], path: Path) -> None:
    cases = []
    names = []
    for row in rows:
        if row["case"] not in cases:
            cases.append(row["case"])
        if row["name"] not in names:
            names.append(row["name"])
    fig, ax = plt.subplots(figsize=(10, 4))
    for name in names:
        values = [float(next(row["f1"] for row in rows if row["case"] == case and row["name"] == name)) for case in cases]
        ax.plot(cases, values, marker="o", label=name.replace("qpp_", ""))
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1")
    ax.set_title("QPP-style noise robustness on full test split")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
