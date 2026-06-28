"""Experiment orchestration shared by CLI scripts."""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .classical import fit_logistic, fit_mlp, threshold_baseline
from .data import load_patch_table, make_group_splits, save_splits
from .features import FeatureOptions, build_feature_matrix
from .metrics import binary_metrics, choose_threshold_by_f1
from .normalizer import FeatureNormalizer
from .qnn_torch import TORCH_AVAILABLE
from .train import predict_torch, set_seed, train_torch_classifier, write_history_csv


SUMMARY_COLUMNS = [
    "dataset",
    "split",
    "feature_set",
    "model",
    "n_qubits",
    "L",
    "encoding",
    "entanglement",
    "readout",
    "val_pr_auc",
    "val_f1",
    "test_pr_auc",
    "test_f1",
    "test_roc_auc",
    "threshold",
    "notes",
]


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config {path} must contain a YAML mapping.")
    return payload


def _safe_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _safe_json_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_safe_json_value(v) for v in value]
    if isinstance(value, tuple):
        return [_safe_json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_safe_json_value(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _run_id(config: dict[str, Any]) -> str:
    base = str(config.get("run", {}).get("id") or config.get("name") or "run")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{base}"


def _split_name_array(n: int, splits: dict[str, np.ndarray]) -> np.ndarray:
    names = np.full(n, "unused", dtype=object)
    for split, idx in splits.items():
        names[idx] = split
    return names


def _save_predictions(
    path: Path,
    table,
    split_names: np.ndarray,
    scores: np.ndarray,
) -> None:
    df = pd.DataFrame(
        {
            "dataset": table.dataset,
            "sample_id": table.groups,
            "center_x": table.centers_xy[:, 0],
            "center_y": table.centers_xy[:, 1],
            "scene_type": table.scene_types,
            "label": table.labels.astype(int),
            "split": split_names,
            "score": scores,
        }
    )
    df.to_csv(path, index=False)


def _prepare_features(config: dict[str, Any]):
    data_cfg = config.get("data", {})
    feature_cfg = config.get("features", {})
    split_cfg = config.get("split", {})
    seed = int(config.get("run", {}).get("seed", 0))
    dataset = str(data_cfg.get("dataset", "smoke_readme_pipeline"))
    table = load_patch_table(
        data_cfg.get("data_root", "data/raw"),
        dataset,
        max_images=data_cfg.get("max_images"),
        sample_per_image=data_cfg.get("sample_per_image"),
        seed=seed,
    )
    options = FeatureOptions(
        gradient=str(feature_cfg.get("gradient", "sobel")),
        smooth_sigma=float(feature_cfg.get("smooth_sigma", 0.6)),
        tensor_sigma=feature_cfg.get("tensor_sigma"),
        harris_k=float(feature_cfg.get("harris_k", 0.04)),
        eps=float(feature_cfg.get("eps", 1e-8)),
    )
    x, feature_names, base, base_names = build_feature_matrix(
        table.patches,
        str(feature_cfg.get("feature_set", "logS_eta")),
        options=options,
        scalar_mode=str(feature_cfg.get("scalar_mode", "logS_plus_c_eta")),
        scalar_c=float(feature_cfg.get("scalar_c", 1.0)),
    )
    splits = make_group_splits(
        table.groups,
        train=float(split_cfg.get("train", 0.70)),
        val=float(split_cfg.get("val", 0.15)),
        test=float(split_cfg.get("test", 0.15)),
        seed=seed,
    )
    return table, x, feature_names, base, base_names, splits


def _transform_features(
    x: np.ndarray,
    y: np.ndarray,
    splits: dict[str, np.ndarray],
    feature_names: list[str],
    model_type: str,
    normalizer_cfg: dict[str, Any],
    out_dir: Path,
) -> tuple[dict[str, np.ndarray], FeatureNormalizer | None]:
    enabled = bool(normalizer_cfg.get("enabled", model_type != "threshold"))
    use_angles = bool(normalizer_cfg.get("angles", model_type.startswith("qnn")))
    split_x = {name: x[idx] for name, idx in splits.items()}
    if not enabled:
        return split_x, None
    normalizer = FeatureNormalizer(clip=float(normalizer_cfg.get("clip", 3.0)))
    z_train = normalizer.fit_transform(split_x["train"], feature_names)
    transformed = {"train": z_train}
    for name in ["val", "test"]:
        transformed[name] = normalizer.transform(split_x[name])
    if use_angles:
        transformed = {name: normalizer.to_angles(value) for name, value in transformed.items()}
    normalizer.save_json(out_dir / "normalizer.json")
    return transformed, normalizer


def _make_model(config: dict[str, Any], input_dim: int):
    model_cfg = config.get("model", {})
    model_type = str(model_cfg.get("type", "qnn2")).lower()
    seed = int(config.get("run", {}).get("seed", 0))
    if model_type == "logistic":
        return "classical", lambda x_train, y_train: fit_logistic(
            x_train,
            y_train,
            seed=seed,
            max_iter=int(model_cfg.get("max_iter", 1000)),
        )
    if model_type == "mlp":
        hidden = tuple(int(v) for v in model_cfg.get("hidden", [32, 16]))
        return "classical", lambda x_train, y_train: fit_mlp(
            x_train,
            y_train,
            seed=seed,
            hidden_layer_sizes=hidden,
            max_iter=int(model_cfg.get("max_iter", 300)),
        )
    if model_type == "threshold":
        return "classical", lambda x_train, y_train: threshold_baseline(sign=float(model_cfg.get("sign", 1.0)))

    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is not installed; QNN experiments require `pip install torch`.")

    from .qnn_torch import DataReuploadingQNN1, DataReuploadingQNN2

    n_layers = int(model_cfg.get("L", model_cfg.get("n_layers", 2)))
    if model_type == "qnn2":
        if input_dim != 2:
            raise ValueError(f"qnn2 expects exactly 2 features, got {input_dim}")
        model = DataReuploadingQNN2(
            n_layers=n_layers,
            encoding=str(model_cfg.get("encoding", "ryrz")),
            entanglement=str(model_cfg.get("entanglement", "linear_01")),
            readout=str(model_cfg.get("readout", "z_z_zz")),
        )
        return "torch", model
    if model_type == "qnn1":
        model = DataReuploadingQNN1(
            n_layers=n_layers,
            encoding=str(model_cfg.get("encoding", "rz")),
            input_dim=input_dim,
            learnable_projection=bool(model_cfg.get("learnable_projection", False)),
        )
        return "torch", model
    raise ValueError(f"Unknown model type: {model_type}")


def _summary_row(
    config: dict[str, Any],
    metrics: dict[str, Any],
    *,
    notes: str = "",
) -> dict[str, Any]:
    model_cfg = config.get("model", {})
    feature_cfg = config.get("features", {})
    model_type = str(model_cfg.get("type", ""))
    n_qubits = 2 if model_type == "qnn2" else 1 if model_type == "qnn1" else 0
    return {
        "dataset": config.get("data", {}).get("dataset", ""),
        "split": "test",
        "feature_set": feature_cfg.get("feature_set", ""),
        "model": model_type,
        "n_qubits": n_qubits,
        "L": model_cfg.get("L", model_cfg.get("n_layers", "")),
        "encoding": model_cfg.get("encoding", ""),
        "entanglement": model_cfg.get("entanglement", ""),
        "readout": model_cfg.get("readout", ""),
        "val_pr_auc": metrics.get("val", {}).get("pr_auc"),
        "val_f1": metrics.get("val", {}).get("f1"),
        "test_pr_auc": metrics.get("test", {}).get("pr_auc"),
        "test_f1": metrics.get("test", {}).get("f1"),
        "test_roc_auc": metrics.get("test", {}).get("roc_auc"),
        "threshold": metrics.get("threshold"),
        "notes": notes,
    }


def run_config(
    config: dict[str, Any],
    *,
    output_root: str | Path = "outputs/runs",
    run_id: str | None = None,
    make_plots: bool | None = None,
) -> dict[str, Any]:
    """Run one resolved experiment config and save artifacts."""

    set_seed(int(config.get("run", {}).get("seed", 0)))
    out_dir = Path(output_root) / (run_id or _run_id(config))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config_resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    table, x, feature_names, _base, _base_names, splits = _prepare_features(config)
    save_splits(out_dir / "splits.json", splits, table.groups)
    y = table.labels.astype(int)
    model_cfg = config.get("model", {})
    model_type = str(model_cfg.get("type", "qnn2")).lower()
    normalizer_cfg = config.get("normalizer", {})
    x_by_split, normalizer = _transform_features(
        x,
        y,
        splits,
        feature_names,
        model_type,
        normalizer_cfg,
        out_dir,
    )
    y_by_split = {name: y[idx] for name, idx in splits.items()}

    kind, model_or_factory = _make_model(config, input_dim=x_by_split["train"].shape[1])
    history: list[dict[str, float]] = []
    if kind == "torch":
        train_cfg = config.get("train", {})
        model, history = train_torch_classifier(
            model_or_factory,
            x_by_split["train"],
            y_by_split["train"],
            x_by_split["val"],
            y_by_split["val"],
            out_dir=out_dir,
            lr=float(train_cfg.get("lr", 3e-3)),
            batch_size=int(train_cfg.get("batch_size", 64)),
            epochs=int(train_cfg.get("epochs", 50)),
            patience=int(train_cfg.get("patience", 8)),
            monitor=str(train_cfg.get("monitor", "val_pr_auc")),
            seed=int(config.get("run", {}).get("seed", 0)),
            device=str(train_cfg.get("device", "auto")),
        )
        val_scores = predict_torch(model, x_by_split["val"], batch_size=int(config.get("train", {}).get("batch_size", 64)))
        test_scores = predict_torch(model, x_by_split["test"], batch_size=int(config.get("train", {}).get("batch_size", 64)))
        all_x = np.zeros_like(x_by_split["train"][:0])
        all_scores = np.zeros(len(y), dtype=np.float64)
        for split, idx in splits.items():
            all_scores[idx] = predict_torch(
                model,
                x_by_split[split],
                batch_size=int(config.get("train", {}).get("batch_size", 64)),
            )
    else:
        model = model_or_factory(x_by_split["train"], y_by_split["train"])
        val_scores = model.predict_scores(x_by_split["val"])
        test_scores = model.predict_scores(x_by_split["test"])
        all_scores = np.zeros(len(y), dtype=np.float64)
        for split, idx in splits.items():
            all_scores[idx] = model.predict_scores(x_by_split[split])

    threshold, _ = choose_threshold_by_f1(y_by_split["val"], val_scores)
    metrics = {
        "run_id": out_dir.name,
        "output_dir": str(out_dir),
        "dataset": table.dataset,
        "feature_set": config.get("features", {}).get("feature_set"),
        "feature_names": feature_names,
        "model": model_cfg,
        "n_train": int(len(splits["train"])),
        "n_val": int(len(splits["val"])),
        "n_test": int(len(splits["test"])),
        "threshold": float(threshold),
        "val": binary_metrics(y_by_split["val"], val_scores, threshold),
        "test": binary_metrics(y_by_split["test"], test_scores, threshold),
    }
    (out_dir / "metrics.json").write_text(json.dumps(_safe_json_value(metrics), indent=2), encoding="utf-8")
    write_history_csv(out_dir / "history.csv", history)
    split_names = _split_name_array(len(y), splits)
    _save_predictions(out_dir / "predictions.csv", table, split_names, all_scores)

    if make_plots is None:
        make_plots = bool(config.get("outputs", {}).get("plots", True))
    if make_plots:
        try:
            from .viz import plot_feature_scatter, plot_patch_preview, plot_pr_curve, plot_training_curves

            plot_pr_curve(y_by_split["test"], test_scores, out_dir / "pr_curve.png", title=f"{out_dir.name} test PR")
            plot_training_curves(history, out_dir / "training_curves.png")
            plot_feature_scatter(x, y, feature_names, out_dir / "feature_scatter.png")
            plot_patch_preview(table.patches[splits["test"]], y_by_split["test"], test_scores, out_dir / "patch_preview.png")
        except Exception as exc:
            (out_dir / "plot_warning.txt").write_text(str(exc), encoding="utf-8")

    row = _summary_row(config, metrics)
    pd.DataFrame([row], columns=SUMMARY_COLUMNS).to_csv(out_dir / "summary.csv", index=False)
    return {"out_dir": out_dir, "metrics": metrics, "summary_row": row, "normalizer": normalizer}


def failed_summary_row(config: dict[str, Any], note: str) -> dict[str, Any]:
    empty_metrics = {"val": {}, "test": {}, "threshold": None}
    return _summary_row(config, empty_metrics, notes=note)
