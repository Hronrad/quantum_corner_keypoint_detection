from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.exceptions import ConvergenceWarning

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_improvement_experiments import (
    apply_noise_to_images,
    extract_features_for_centers,
    load_qnn_artifacts,
    probability_metrics,
    qnn_probabilities,
    train_extended_mlp,
)


def main() -> None:
    args = parse_args()
    data = np.load(args.data_path, allow_pickle=True)
    payload = {key: data[key] for key in data.files if key.startswith(("X_", "y_"))}

    outputs_dir = args.output_dir
    outputs_dir.mkdir(exist_ok=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        mlp, _ = train_extended_mlp(payload, outputs_dir, args.seed)

    qnn_model, normalizer = load_qnn_artifacts(args.qnn_run_dir)
    noisy_images = apply_noise_to_images(data["images"], "saltpepper", args.noise_value, args.seed)
    features = extract_features_for_centers(
        noisy_images,
        data["test_image_ids"],
        data["test_centers"],
        args.patch_size,
    )
    labels = data["y_test"].astype(int)

    rows = [
        {
            "case": f"saltpepper_{args.noise_value:.2f}",
            "method": "MLP",
            "samples": int(len(labels)),
            "positives": int(np.sum(labels == 1)),
            **probability_metrics(labels, mlp.predict_proba(features)[:, 1]),
        },
        {
            "case": f"saltpepper_{args.noise_value:.2f}",
            "method": "QNN",
            "samples": int(len(labels)),
            "positives": int(np.sum(labels == 1)),
            **probability_metrics(labels, qnn_probabilities(qnn_model, normalizer, features)),
        },
    ]

    write_rows(outputs_dir / "saltpepper_full_validation.csv", rows)
    (outputs_dir / "saltpepper_full_validation.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    save_validation_chart(rows, outputs_dir / "saltpepper_full_validation.png")
    print(json.dumps(rows, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate salt-and-pepper robustness on the full held-out patch split.")
    parser.add_argument("--data-path", type=Path, default=ROOT / "data" / "feature_dataset_extended.npz")
    parser.add_argument(
        "--qnn-run-dir",
        type=Path,
        default=ROOT / "outputs" / "qnn_improvement_improved_L2_ring_ZZ_scale_more_data",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--noise-value", type=float, default=0.03)
    parser.add_argument("--patch-size", type=int, default=9)
    parser.add_argument("--seed", type=int, default=37)
    return parser.parse_args()


def write_rows(path: Path, rows: list[dict]) -> None:
    fieldnames = ["case", "method", "samples", "positives", "precision", "recall", "f1", "pr_auc"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_validation_chart(rows: list[dict], path: Path) -> None:
    methods = [row["method"] for row in rows]
    x = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    ax.bar(x - 0.18, [float(row["f1"]) for row in rows], width=0.36, label="F1")
    ax.bar(x + 0.18, [float(row["pr_auc"]) for row in rows], width=0.36, label="PR-AUC")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Salt-and-pepper full test validation")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
