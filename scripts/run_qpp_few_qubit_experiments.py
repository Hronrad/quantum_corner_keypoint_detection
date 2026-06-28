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


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    payload = np.load(args.data_path, allow_pickle=True)
    features = build_qpp_features(payload)
    outputs_dir = args.output_dir
    outputs_dir.mkdir(exist_ok=True)

    rows = []
    rows.append(run_threshold("lambda2_threshold", "lambda2", features["lambda2"], payload))
    rows.append(run_logistic("logistic_logS_eta", "logS_eta", features["logS_eta"], payload, args.seed))
    rows.append(run_qnn1("qpp_1q_scalar_c2_L2", "scalar_c2", features["scalar_c2"], payload, args))
    rows.append(run_qnn1("qpp_1q_scalar_c4_L2", "scalar_c4", features["scalar_c4"], payload, args))
    rows.append(run_qnn2("qpp_2q_logS_eta_L2", "logS_eta", features["logS_eta"], payload, args))
    rows.append(run_qnn2("qpp_2q_lambda12_L2", "lambda12", features["lambda12"], payload, args))

    csv_path = outputs_dir / "qpp_few_qubit_results.csv"
    json_path = outputs_dir / "qpp_few_qubit_results.json"
    png_path = outputs_dir / "qpp_few_qubit_results.png"
    write_rows(csv_path, rows)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    save_chart(rows, png_path)
    print(json.dumps(rows, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run QPP-inspired few-qubit experiments on the current full feature split.")
    parser.add_argument("--data-path", type=Path, default=ROOT / "data" / "feature_dataset_extended.npz")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=8)
    return parser.parse_args()


def build_qpp_features(payload) -> dict[str, dict[str, np.ndarray | list[str]]]:
    names = [str(name) for name in payload["feature_names"].tolist()]
    lookup = {name: idx for idx, name in enumerate(names)}

    def split_base(split: str) -> dict[str, np.ndarray]:
        x = payload[f"X_{split}"].astype(np.float32)
        lambda1 = np.clip(x[:, lookup["lambda1"]], 0.0, None)
        lambda2 = np.clip(x[:, lookup["lambda2"]], 0.0, None)
        s = lambda1 + lambda2
        eta = np.clip(4.0 * lambda1 * lambda2 / (s * s + 1e-8), 0.0, 1.0)
        log_s = np.log(s + 1e-8)
        return {
            "lambda1": lambda1,
            "lambda2": lambda2,
            "S": s,
            "eta": eta,
            "logS": log_s,
        }

    base = {split: split_base(split) for split in ["train", "val", "test"]}

    def pack(columns: list[str], feature_names: list[str]):
        return {
            "feature_names": feature_names,
            **{
                split: np.column_stack([base[split][column] for column in columns]).astype(np.float32)
                for split in ["train", "val", "test"]
            },
        }

    return {
        "lambda2": pack(["lambda2"], ["lambda2"]),
        "lambda12": pack(["lambda1", "lambda2"], ["lambda1", "lambda2"]),
        "logS_eta": pack(["logS", "eta"], ["logS", "eta"]),
        "scalar_c2": {
            "feature_names": ["logS_plus_2_eta"],
            **{
                split: (base[split]["logS"] + 2.0 * base[split]["eta"]).reshape(-1, 1).astype(np.float32)
                for split in ["train", "val", "test"]
            },
        },
        "scalar_c4": {
            "feature_names": ["logS_plus_4_eta"],
            **{
                split: (base[split]["logS"] + 4.0 * base[split]["eta"]).reshape(-1, 1).astype(np.float32)
                for split in ["train", "val", "test"]
            },
        },
    }


def labels(payload, split: str) -> np.ndarray:
    return payload[f"y_{split}"].astype(int)


def run_threshold(name: str, feature_set: str, x_pack: dict, payload) -> dict:
    model = threshold_baseline()
    val_scores = model.predict_scores(x_pack["val"])
    threshold, val_f1 = choose_threshold_by_f1(labels(payload, "val"), val_scores)
    test_scores = model.predict_scores(x_pack["test"])
    return result_row(
        name=name,
        feature_set=feature_set,
        model="threshold",
        n_qubits=0,
        layers=0,
        val_f1=val_f1,
        threshold=threshold,
        metrics=binary_metrics(labels(payload, "test"), test_scores, threshold),
        train_samples=len(labels(payload, "train")),
        test_samples=len(labels(payload, "test")),
    )


def run_logistic(name: str, feature_set: str, x_pack: dict, payload, seed: int) -> dict:
    normalizer = FeatureNormalizer()
    x_train = normalizer.fit_transform(x_pack["train"], x_pack["feature_names"])
    x_val = normalizer.transform(x_pack["val"])
    x_test = normalizer.transform(x_pack["test"])
    model = fit_logistic(x_train, labels(payload, "train"), seed=seed)
    val_scores = model.predict_scores(x_val)
    threshold, val_f1 = choose_threshold_by_f1(labels(payload, "val"), val_scores)
    test_scores = model.predict_scores(x_test)
    return result_row(
        name=name,
        feature_set=feature_set,
        model="logistic",
        n_qubits=0,
        layers=0,
        val_f1=val_f1,
        threshold=threshold,
        metrics=binary_metrics(labels(payload, "test"), test_scores, threshold),
        train_samples=len(labels(payload, "train")),
        test_samples=len(labels(payload, "test")),
    )


def run_qnn1(name: str, feature_set: str, x_pack: dict, payload, args: argparse.Namespace) -> dict:
    normalizer = FeatureNormalizer()
    x_train = normalizer.to_angles(normalizer.fit_transform(x_pack["train"], x_pack["feature_names"]))
    x_val = normalizer.to_angles(normalizer.transform(x_pack["val"]))
    x_test = normalizer.to_angles(normalizer.transform(x_pack["test"]))
    model = DataReuploadingQNN1(n_layers=2, encoding="ryrz")
    run_dir = args.output_dir / f"{name}_run"
    model, _ = train_torch_classifier(
        model,
        x_train,
        labels(payload, "train"),
        x_val,
        labels(payload, "val"),
        out_dir=run_dir,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        seed=args.seed,
        device="cpu",
    )
    val_scores = predict_torch(model, x_val, batch_size=args.batch_size, device="cpu")
    threshold, val_f1 = choose_threshold_by_f1(labels(payload, "val"), val_scores)
    test_scores = predict_torch(model, x_test, batch_size=args.batch_size, device="cpu")
    return result_row(
        name=name,
        feature_set=feature_set,
        model="qpp_qnn1",
        n_qubits=1,
        layers=2,
        val_f1=val_f1,
        threshold=threshold,
        metrics=binary_metrics(labels(payload, "test"), test_scores, threshold),
        train_samples=len(labels(payload, "train")),
        test_samples=len(labels(payload, "test")),
    )


def run_qnn2(name: str, feature_set: str, x_pack: dict, payload, args: argparse.Namespace) -> dict:
    normalizer = FeatureNormalizer()
    x_train = normalizer.to_angles(normalizer.fit_transform(x_pack["train"], x_pack["feature_names"]))
    x_val = normalizer.to_angles(normalizer.transform(x_pack["val"]))
    x_test = normalizer.to_angles(normalizer.transform(x_pack["test"]))
    model = DataReuploadingQNN2(n_layers=2, encoding="ryrz", entanglement="linear_01", readout="z_z_zz")
    run_dir = args.output_dir / f"{name}_run"
    model, _ = train_torch_classifier(
        model,
        x_train,
        labels(payload, "train"),
        x_val,
        labels(payload, "val"),
        out_dir=run_dir,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        seed=args.seed,
        device="cpu",
    )
    val_scores = predict_torch(model, x_val, batch_size=args.batch_size, device="cpu")
    threshold, val_f1 = choose_threshold_by_f1(labels(payload, "val"), val_scores)
    test_scores = predict_torch(model, x_test, batch_size=args.batch_size, device="cpu")
    return result_row(
        name=name,
        feature_set=feature_set,
        model="qpp_qnn2",
        n_qubits=2,
        layers=2,
        val_f1=val_f1,
        threshold=threshold,
        metrics=binary_metrics(labels(payload, "test"), test_scores, threshold),
        train_samples=len(labels(payload, "train")),
        test_samples=len(labels(payload, "test")),
    )


def result_row(
    *,
    name: str,
    feature_set: str,
    model: str,
    n_qubits: int,
    layers: int,
    val_f1: float,
    threshold: float,
    metrics: dict[str, float],
    train_samples: int,
    test_samples: int,
) -> dict:
    return {
        "name": name,
        "feature_set": feature_set,
        "model": model,
        "n_qubits": int(n_qubits),
        "layers": int(layers),
        "train_samples": int(train_samples),
        "test_samples": int(test_samples),
        "val_f1": float(val_f1),
        "threshold": float(threshold),
        "test_accuracy": float(metrics["accuracy"]),
        "test_precision": float(metrics["precision"]),
        "test_recall": float(metrics["recall"]),
        "test_f1": float(metrics["f1"]),
        "test_roc_auc": float(metrics["roc_auc"]),
        "test_pr_auc": float(metrics["pr_auc"]),
    }


def write_rows(path: Path, rows: list[dict]) -> None:
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_chart(rows: list[dict], path: Path) -> None:
    names = [row["name"].replace("qpp_", "").replace("_", "\n") for row in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - 0.18, [float(row["test_f1"]) for row in rows], width=0.36, label="F1")
    ax.bar(x + 0.18, [float(row["test_pr_auc"]) for row in rows], width=0.36, label="PR-AUC")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("QPP-style few-qubit experiments on full split")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
