"""精简版消融实验入口。

本脚本自动运行计划书第一版要求的核心消融：

1. 特征组：[Ix, Iy] 与 [Ix, Iy, lambda1, lambda2, R]；
2. 编码方式：ry 与 ryrz；
3. 纠缠结构：none、linear、ring；
4. 层数：L=1 与 L=3；
5. 经典 baseline：logistic regression 与 MLP with same features。

为了避免组合爆炸，默认只运行一个可控规模的实验矩阵。后续如果需要更完整的
ablation，可以在 ABLATION_GRID 中继续添加配置。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import DEFAULT_FEATURE_NAMES, QNNConfig, TrainConfig
from train_qnn import compute_binary_metrics, load_npz_dataset, select_feature_columns, train_qnn_experiment


FEATURE_GROUPS = {
    "gradients": [0, 1],
    "default5": [0, 1, 2, 3, 4],
}

ABLATION_GRID = [
    {"encoding": "ry", "entanglement": "ring", "layers": 3},
    {"encoding": "ryrz", "entanglement": "none", "layers": 3},
    {"encoding": "ryrz", "entanglement": "linear", "layers": 3},
    {"encoding": "ryrz", "entanglement": "ring", "layers": 1},
    {"encoding": "ryrz", "entanglement": "ring", "layers": 3},
]


def run_classical_baselines(
    data_path: str | Path,
    output_dir: str | Path,
    feature_group: str,
    feature_indices: Sequence[int],
    seed: int,
) -> List[Dict[str, float | str]]:
    """运行 logistic regression 和 MLP baseline。

    两个 baseline 使用与 QNN 完全相同的特征列。这样可以判断 QNN 的表现是否
    真的来自量子线路结构，而不是来自输入特征工程。
    """

    X_train, y_train, X_val, y_val = load_npz_dataset(data_path)
    X_train = select_feature_columns(X_train, feature_indices)
    X_val = select_feature_columns(X_val, feature_indices)

    baselines = {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed),
        ),
        "mlp_same_features": make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(16, 8),
                activation="relu",
                max_iter=500,
                random_state=seed,
                early_stopping=True,
            ),
        ),
    }

    rows: List[Dict[str, float | str]] = []
    for name, model in baselines.items():
        model.fit(X_train, y_train.astype(int))
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_val)[:, 1]
            logits = np.log(np.clip(probs, 1e-7, 1.0 - 1e-7) / np.clip(1.0 - probs, 1e-7, 1.0))
        else:
            logits = model.decision_function(X_val)

        metrics = compute_binary_metrics(y_val, logits)
        row: Dict[str, float | str] = {
            "model": name,
            "feature_group": feature_group,
            "encoding": "classical",
            "entanglement": "classical",
            "layers": "classical",
            **metrics,
        }
        rows.append(row)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with (output_path / f"classical_{feature_group}.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2, allow_nan=True)

    return rows


def main() -> None:
    """命令行入口，运行精简消融矩阵并保存 summary。"""

    parser = argparse.ArgumentParser(description="Run QNN ablation experiments.")
    parser.add_argument("--data", default="data/synthetic_corner_data.npz")
    parser.add_argument("--output-dir", default="outputs/ablation")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, float | str]] = []
    for feature_group, feature_indices in FEATURE_GROUPS.items():
        feature_names = [DEFAULT_FEATURE_NAMES[i] for i in feature_indices]
        rows.extend(
            run_classical_baselines(
                data_path=args.data,
                output_dir=output_path,
                feature_group=feature_group,
                feature_indices=feature_indices,
                seed=args.seed,
            )
        )

        for cfg in ABLATION_GRID:
            run_name = (
                f"{feature_group}_enc-{cfg['encoding']}_ent-{cfg['entanglement']}_L{cfg['layers']}"
            )
            run_dir = output_path / run_name
            qnn_config = QNNConfig(
                n_qubits=len(feature_indices),
                n_layers=int(cfg["layers"]),
                encoding_type=str(cfg["encoding"]),
                entanglement=str(cfg["entanglement"]),
                readout="all",
                shots=None,
            )
            train_config = TrainConfig(
                learning_rate=args.learning_rate,
                batch_size=args.batch_size,
                epochs=args.epochs,
                seed=args.seed,
                output_dir=str(run_dir),
            )
            metrics = train_qnn_experiment(
                data_path=args.data,
                output_dir=run_dir,
                qnn_config=qnn_config,
                train_config=train_config,
                feature_indices=feature_indices,
                feature_names=feature_names,
                device_name=args.device,
            )
            rows.append(
                {
                    "model": "qnn",
                    "feature_group": feature_group,
                    "encoding": str(cfg["encoding"]),
                    "entanglement": str(cfg["entanglement"]),
                    "layers": str(cfg["layers"]),
                    **metrics,
                }
            )

    summary = pd.DataFrame(rows)
    summary.to_csv(output_path / "summary.csv", index=False, encoding="utf-8-sig")
    with (output_path / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2, allow_nan=True)

    print(summary)
    print(f"saved ablation summary to {output_path / 'summary.csv'}")


if __name__ == "__main__":
    main()
