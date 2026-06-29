from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import QNNConfig
from qcd_data.features import (
    EXTENDED_STRUCTURE_TENSOR_FEATURE_NAMES,
    STRUCTURE_TENSOR_FEATURE_NAMES,
    extract_extended_structure_tensor_features,
    extract_structure_tensor_features,
)
from scripts.run_5d8d_noise_comparison import (
    apply_noise_to_images,
    choose_threshold_by_f1,
    ensure_feature_datasets,
    extract_features_for_centers,
    load_feature_dataset,
    load_qnn_artifacts,
    probability_metrics,
    qnn_probabilities,
    train_mlp,
    train_or_load_qnn,
    validate_shared_protocol,
)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    data_dir = ROOT / "data"
    run_root = output_dir / "feature_dim_noise_comparison_runs"
    output_dir.mkdir(exist_ok=True)
    run_root.mkdir(exist_ok=True)

    dataset5_path, dataset8_path = ensure_feature_datasets(args, data_dir, output_dir)
    dataset5 = load_feature_dataset(dataset5_path)
    dataset8 = load_feature_dataset(dataset8_path)
    validate_shared_protocol(dataset5, dataset8)

    specs = [
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
    for spec in specs:
        print(f"Preparing {spec['feature_dim']} salt-pepper sweep...")
        mlp = train_mlp(spec["dataset"], spec["mlp_hidden"], args.seed)
        mlp_threshold = choose_threshold_by_f1(
            spec["dataset"]["y_val"],
            mlp.predict_proba(spec["dataset"]["X_val"])[:, 1],
        )

        qnn_run_dir = train_or_load_qnn(spec, run_root, args)
        qnn_model, qnn_normalizer = load_qnn_artifacts(qnn_run_dir)
        qnn_threshold = choose_threshold_by_f1(
            spec["dataset"]["y_val"],
            qnn_probabilities(qnn_model, qnn_normalizer, spec["dataset"]["X_val"]),
        )

        for value in args.saltpepper_values:
            noisy_images = apply_noise_to_images(spec["dataset"]["images"], "saltpepper", value, args.seed)
            test_features = extract_features_for_centers(
                noisy_images,
                spec["dataset"]["test_image_ids"],
                spec["dataset"]["test_centers"],
                args.patch_size,
                spec["extractor"],
            )
            labels = spec["dataset"]["y_test"].astype(int)
            rows.append(
                sweep_row(
                    spec["feature_dim"],
                    "MLP",
                    value,
                    len(spec["dataset"]["y_train"]),
                    len(labels),
                    mlp_threshold,
                    probability_metrics(labels, mlp.predict_proba(test_features)[:, 1], mlp_threshold),
                )
            )
            rows.append(
                sweep_row(
                    spec["feature_dim"],
                    "QNN",
                    value,
                    int(spec["train_limit"]),
                    len(labels),
                    qnn_threshold,
                    probability_metrics(labels, qnn_probabilities(qnn_model, qnn_normalizer, test_features), qnn_threshold),
                )
            )

    csv_path = output_dir / "saltpepper_sweep_5d8d.csv"
    json_path = output_dir / "saltpepper_sweep_5d8d.json"
    png_path = output_dir / "saltpepper_sweep_5d8d.png"
    md_path = output_dir / "saltpepper_sweep_5d8d_summary.md"
    write_rows(csv_path, rows)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    save_sweep_chart(rows, png_path)
    write_summary(rows, md_path)
    print(f"Wrote {csv_path}")
    print(f"Wrote {png_path}")
    print(f"Wrote {md_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep salt-and-pepper noise strength for 5D/8D QNN and MLP.")
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--images-per-scene", type=int, default=100)
    parser.add_argument("--patch-size", type=int, default=9)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--force-rebuild-data", action="store_true")
    parser.add_argument("--reuse-qnn", action="store_true")
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
        "--saltpepper-values",
        type=parse_float_list,
        default=[0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15],
        help="Comma-separated salt-pepper probabilities, e.g. 0,0.01,0.03,0.05.",
    )
    parser.add_argument("--smoke", action="store_true")
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
        args.saltpepper_values = [0.0, 0.03, 0.08]
    return args


def parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def sweep_row(
    feature_dim: str,
    method: str,
    saltpepper_value: float,
    train_samples: int,
    test_samples: int,
    threshold: float,
    metrics: dict[str, float],
) -> dict:
    return {
        "feature_dim": feature_dim,
        "method": method,
        "saltpepper_value": float(saltpepper_value),
        "train_samples": int(train_samples),
        "test_samples": int(test_samples),
        "threshold": float(threshold),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1": float(metrics["f1"]),
        "pr_auc": float(metrics["pr_auc"]),
    }


def write_rows(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "feature_dim",
        "method",
        "saltpepper_value",
        "train_samples",
        "test_samples",
        "threshold",
        "precision",
        "recall",
        "f1",
        "pr_auc",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_sweep_chart(rows: list[dict], path: Path) -> None:
    series = [("5D", "QNN"), ("5D", "MLP"), ("8D", "QNN"), ("8D", "MLP")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric in zip(axes, ["f1", "pr_auc"]):
        for feature_dim, method in series:
            subset = sorted(
                [row for row in rows if row["feature_dim"] == feature_dim and row["method"] == method],
                key=lambda row: float(row["saltpepper_value"]),
            )
            ax.plot(
                [float(row["saltpepper_value"]) for row in subset],
                [float(row[metric]) for row in subset],
                marker="o",
                label=f"{feature_dim} {method}",
            )
        ax.set_xlabel("Salt-pepper probability")
        ax.set_ylabel(metric.upper() if metric == "f1" else "PR-AUC")
        ax.set_ylim(0.0, 1.05)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    axes[0].set_title("F1 degradation under stronger salt-pepper noise")
    axes[1].set_title("Ranking robustness under stronger salt-pepper noise")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_summary(rows: list[dict], path: Path) -> None:
    lines = [
        "# Salt-Pepper Sweep: 5D/8D QNN vs MLP",
        "",
        "Protocol: clean training, clean-validation threshold selection, fixed thresholds for all salt-pepper levels.",
        "",
        "## Trend Summary",
        "",
        trend_sentence(rows, "5D"),
        trend_sentence(rows, "8D"),
        "",
        "## F1 Table",
        "",
        "| Salt-pepper | 5D QNN | 5D MLP | 8D QNN | 8D MLP |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    values = sorted({float(row["saltpepper_value"]) for row in rows})
    for value in values:
        lines.append(
            "| "
            + f"{value:.2f} | "
            + f"{metric_at(rows, '5D', 'QNN', value, 'f1'):.4f} | "
            + f"{metric_at(rows, '5D', 'MLP', value, 'f1'):.4f} | "
            + f"{metric_at(rows, '8D', 'QNN', value, 'f1'):.4f} | "
            + f"{metric_at(rows, '8D', 'MLP', value, 'f1'):.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def trend_sentence(rows: list[dict], feature_dim: str) -> str:
    values = sorted({float(row["saltpepper_value"]) for row in rows})
    start, end = values[0], values[-1]
    qnn_start = metric_at(rows, feature_dim, "QNN", start, "f1")
    qnn_end = metric_at(rows, feature_dim, "QNN", end, "f1")
    mlp_start = metric_at(rows, feature_dim, "MLP", start, "f1")
    mlp_end = metric_at(rows, feature_dim, "MLP", end, "f1")
    qnn_drop = qnn_start - qnn_end
    mlp_drop = mlp_start - mlp_end
    verdict = "supports stronger QNN salt-pepper robustness" if qnn_drop < mlp_drop else "does not support stronger QNN salt-pepper robustness"
    return f"- {feature_dim}: F1 drop from {start:.2f} to {end:.2f} is QNN {qnn_drop:.4f} vs MLP {mlp_drop:.4f}; this {verdict}."


def metric_at(rows: list[dict], feature_dim: str, method: str, value: float, metric: str) -> float:
    for row in rows:
        if row["feature_dim"] == feature_dim and row["method"] == method and abs(float(row["saltpepper_value"]) - value) < 1e-12:
            return float(row[metric])
    return float("nan")


if __name__ == "__main__":
    main()
