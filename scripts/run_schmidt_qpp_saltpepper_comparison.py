"""Matched Schmidt-QPP versus TinyMLP salt-and-pepper robustness study.

The protocol is deliberately strict:

* both models receive the same ordered ``[lambda1, lambda2]`` features;
* both use the same image-disjoint train/validation/test split and all 4,500
  training patches by default;
* decision thresholds are selected once on the clean validation split;
* noisy held-out images are corrupted before patch features are re-extracted;
* the clean-validation thresholds stay fixed throughout the noise sweep; and
* repeated training and noise seeds are retained in the raw output.

The explicit Schmidt-QPP circuit prepares
``sqrt(mu1)|00> + sqrt(mu2)|11>`` and has 25 trainable parameters.  Two
two-input, five-hidden-unit TinyMLPs (21 parameters each) provide classical
controls: one receives the raw ordered spectrum and the other receives exactly
the normalized spectrum retained by the Schmidt state.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
QPP_SRC = ROOT / "qpp_corner_qnn_github_package" / "src"
for path in [ROOT, QPP_SRC]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from qcd_data.features import extract_extended_structure_tensor_features
from qcd_data.synthetic import extract_patches
from qpp_corner.qnn_torch import SchmidtQPPQNN2


MODEL_SCHMIDT = "Schmidt-QPP QNN L2"
MODEL_MLP_RAW = "TinyMLP h5 (lambda12)"
MODEL_MLP_MU = "TinyMLP h5 (mu12)"


class TinyMLP(nn.Module):
    """Parameter-matched classical reference: 2 -> 5 -> 1 (21 parameters)."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(2, 5), nn.Tanh(), nn.Linear(5, 1))

    def forward(self, x):
        return self.network(x.to(dtype=torch.float32)).squeeze(-1)


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_float_list(value: str) -> list[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("At least one noise value is required.")
    return sorted(set(values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare explicit Schmidt-QPP with a matched lambda12 TinyMLP under a salt-pepper sweep."
    )
    parser.add_argument("--data-path", type=Path, default=ROOT / "data" / "feature_dataset_extended.npz")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "schmidt_qpp_noise")
    parser.add_argument("--training-seeds", type=parse_int_list, default=parse_int_list("17,29,41,53,67"))
    parser.add_argument("--noise-seeds", type=parse_int_list, default=parse_int_list("1001,2003,3001"))
    parser.add_argument(
        "--saltpepper-values",
        type=parse_float_list,
        default=parse_float_list("0,0.01,0.02,0.03,0.05,0.08,0.10,0.15"),
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--patch-size", type=int, default=9)
    parser.add_argument("--mlp-clip", type=float, default=3.0)
    parser.add_argument("--train-limit", type=int, default=0, help="Stratified training limit; 0 uses all samples.")
    parser.add_argument("--val-limit", type=int, default=0, help="Stratified validation limit; 0 uses all samples.")
    parser.add_argument("--test-limit", type=int, default=0, help="Stratified test limit; 0 uses all samples.")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--smoke", action="store_true", help="Run a small protocol for implementation checks.")
    args = parser.parse_args()
    if args.smoke:
        args.training_seeds = args.training_seeds[:1]
        args.noise_seeds = args.noise_seeds[:1]
        args.saltpepper_values = [0.0, 0.03, 0.08]
        args.epochs = min(args.epochs, 3)
        args.patience = min(args.patience, 2)
        args.train_limit = args.train_limit or 320
        args.val_limit = args.val_limit or 240
        args.test_limit = args.test_limit or 300
    if 0.0 not in args.saltpepper_values:
        args.saltpepper_values = sorted([0.0, *args.saltpepper_values])
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(value)


def sigmoid(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(values, -80.0, 80.0)))


def average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=np.float64)
    positives = int(np.sum(y == 1))
    if positives == 0:
        return math.nan
    order = np.argsort(-s, kind="mergesort")
    ranked_y = y[order]
    ranked_scores = s[order]
    distinct_ends = np.r_[np.flatnonzero(np.diff(ranked_scores)), len(ranked_scores) - 1]
    true_positives = np.cumsum(ranked_y == 1)[distinct_ends]
    false_positives = (distinct_ends + 1) - true_positives
    precision = true_positives / np.maximum(1, true_positives + false_positives)
    recall = true_positives / positives
    recall_increments = np.diff(np.r_[0.0, recall])
    return float(np.sum(recall_increments * precision))


def classification_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    pred = (np.asarray(scores) >= float(threshold)).astype(int)
    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    return {
        "accuracy": float((tp + tn) / max(1, len(y))),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "pr_auc": average_precision(y, scores),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def choose_threshold_by_f1(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=np.float64)
    candidates = np.unique(s)
    if len(candidates) > 512:
        candidates = np.unique(np.quantile(s, np.linspace(0.0, 1.0, 512)))
    candidates = np.concatenate([[s.min() - 1e-9], candidates, [s.max() + 1e-9]])
    best_threshold = float(candidates[0])
    best_f1 = -1.0
    for threshold in candidates:
        current = classification_metrics(y, s, float(threshold))["f1"]
        if current > best_f1:
            best_f1 = current
            best_threshold = float(threshold)
    return best_threshold, float(best_f1)


def stratified_indices(y: np.ndarray, limit: int, seed: int) -> np.ndarray:
    if limit <= 0 or limit >= len(y):
        return np.arange(len(y), dtype=int)
    rng = np.random.default_rng(seed)
    labels, counts = np.unique(y, return_counts=True)
    target_counts = np.floor(limit * counts / counts.sum()).astype(int)
    while target_counts.sum() < limit:
        remainders = limit * counts / counts.sum() - target_counts
        target_counts[int(np.argmax(remainders))] += 1
    selected = []
    for label, count in zip(labels, target_counts):
        indices = np.flatnonzero(y == label)
        selected.extend(rng.choice(indices, size=int(count), replace=False).tolist())
    return np.asarray(sorted(selected), dtype=int)


def ordered_spectrum(features: np.ndarray, feature_names: list[str]) -> np.ndarray:
    lookup = {name: index for index, name in enumerate(feature_names)}
    values = np.column_stack([features[:, lookup["lambda1"]], features[:, lookup["lambda2"]]])
    values = np.clip(values, 0.0, None).astype(np.float32)
    return np.column_stack([np.maximum(values[:, 0], values[:, 1]), np.minimum(values[:, 0], values[:, 1])]).astype(
        np.float32
    )


def normalized_spectrum(spectrum: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(spectrum, dtype=np.float32), 0.0, None)
    total = values.sum(axis=1, keepdims=True)
    mu = values / np.maximum(total, 1e-8)
    vacuum = total[:, 0] <= 1e-8
    mu[vacuum, 0] = 1.0
    mu[vacuum, 1] = 0.0
    return mu.astype(np.float32)


def fit_standardizer(x: np.ndarray, clip: float) -> dict[str, np.ndarray | float]:
    mean = np.asarray(x, dtype=np.float32).mean(axis=0)
    std = np.asarray(x, dtype=np.float32).std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return {"mean": mean, "std": std, "clip": float(clip)}


def apply_standardizer(x: np.ndarray, normalizer: dict[str, np.ndarray | float]) -> np.ndarray:
    values = (np.asarray(x, dtype=np.float32) - normalizer["mean"]) / normalizer["std"]
    return np.clip(values, -float(normalizer["clip"]), float(normalizer["clip"])).astype(np.float32)


def predict(model: nn.Module, x: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    values = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = torch.as_tensor(x[start : start + batch_size], dtype=torch.float32, device=device)
            values.append(model(batch).detach().cpu().numpy())
    return sigmoid(np.concatenate(values))


def train_classifier(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    seed: int,
    epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
) -> tuple[nn.Module, list[dict[str, float]]]:
    set_seed(seed)
    model = model.to(device)
    dataset = torch.utils.data.TensorDataset(
        torch.as_tensor(x_train, dtype=torch.float32), torch.as_tensor(y_train, dtype=torch.float32)
    )
    generator = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
    positive_rate = float(np.mean(y_train))
    pos_weight = torch.tensor([(1.0 - positive_rate) / max(positive_rate, 1e-8)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_value = -float("inf")
    best_state = None
    stale = 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        train_scores = predict(model, x_train, batch_size, device)
        val_scores = predict(model, x_val, batch_size, device)
        row = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "train_pr_auc": average_precision(y_train, train_scores),
            "val_pr_auc": average_precision(y_val, val_scores),
        }
        history.append(row)
        if row["val_pr_auc"] > best_value + 1e-8:
            best_value = row["val_pr_auc"]
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def apply_saltpepper(images: np.ndarray, probability: float, seed: int) -> np.ndarray:
    if probability <= 0.0:
        return np.asarray(images, dtype=np.float32).copy()
    rng = np.random.default_rng(seed + int(round(probability * 1000)))
    out = np.asarray(images, dtype=np.float32).copy()
    mask = rng.random(out.shape)
    out[mask < probability / 2.0] = 0.0
    out[(mask >= probability / 2.0) & (mask < probability)] = 1.0
    return out


def extract_test_spectrum(
    images: np.ndarray,
    image_ids: np.ndarray,
    centers: np.ndarray,
    feature_names: list[str],
    patch_size: int,
) -> np.ndarray:
    patches = [
        extract_patches(images[int(image_id)], np.asarray([center], dtype=np.float32), patch_size)[0]
        for image_id, center in zip(image_ids, centers)
    ]
    features = extract_extended_structure_tensor_features(np.stack(patches).astype(np.float32))
    return ordered_spectrum(features, feature_names)


def prepare_noise_cache(
    payload,
    feature_names: list[str],
    test_indices: np.ndarray,
    args: argparse.Namespace,
) -> dict[tuple[float, int | None], np.ndarray]:
    cache: dict[tuple[float, int | None], np.ndarray] = {}
    clean = ordered_spectrum(payload["X_test"].astype(np.float32), feature_names)[test_indices]
    cache[(0.0, None)] = clean
    image_ids = payload["test_image_ids"][test_indices]
    centers = payload["test_centers"][test_indices]
    for probability in args.saltpepper_values:
        if probability == 0.0:
            continue
        for noise_seed in args.noise_seeds:
            noisy_images = apply_saltpepper(payload["images"], probability, noise_seed)
            cache[(probability, noise_seed)] = extract_test_spectrum(
                noisy_images, image_ids, centers, feature_names, args.patch_size
            )
    return cache


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=1)) if len(array) > 1 else 0.0


def aggregate_rows(rows: list[dict]) -> list[dict]:
    seed_groups: dict[tuple[str, float, int], list[dict]] = defaultdict(list)
    for row in rows:
        seed_groups[
            (str(row["model"]), float(row["saltpepper_probability"]), int(row["training_seed"]))
        ].append(row)
    groups: dict[tuple[str, float], list[dict[str, float]]] = defaultdict(list)
    for (model, probability, _), subset in seed_groups.items():
        groups[(model, probability)].append(
            {
                metric: float(np.mean([float(row[metric]) for row in subset]))
                for metric in ["precision", "recall", "f1", "pr_auc", "f1_drop", "f1_retention"]
            }
        )
    output = []
    for (model, probability), seed_means in sorted(groups.items(), key=lambda item: (item[0][1], item[0][0])):
        raw_subset = [
            row
            for row in rows
            if str(row["model"]) == model and float(row["saltpepper_probability"]) == probability
        ]
        item = {
            "model": model,
            "saltpepper_probability": probability,
            "n_raw_evaluations": len(raw_subset),
            "n_training_seeds": len(seed_means),
            "n_noise_seeds_per_training_seed": 1 if probability == 0.0 else len(
                {str(row["noise_seed"]) for row in raw_subset}
            ),
            "trainable_parameters": int(raw_subset[0]["trainable_parameters"]),
        }
        for metric in ["precision", "recall", "f1", "pr_auc", "f1_drop", "f1_retention"]:
            item[f"{metric}_mean"], item[f"{metric}_std"] = mean_std(
                [float(seed_mean[metric]) for seed_mean in seed_means]
            )
            raw_values = [float(row[metric]) for row in raw_subset]
            _, item[f"{metric}_raw_std"] = mean_std(raw_values)
            within_seed_stds = []
            for training_seed in sorted({int(row["training_seed"]) for row in raw_subset}):
                seed_values = [
                    float(row[metric]) for row in raw_subset if int(row["training_seed"]) == training_seed
                ]
                _, seed_std = mean_std(seed_values)
                within_seed_stds.append(seed_std)
            item[f"{metric}_within_noise_std_mean"] = float(np.mean(within_seed_stds))
        output.append(item)
    return output


def paired_differences(rows: list[dict]) -> list[dict]:
    lookup = {
        (int(row["training_seed"]), str(row["noise_seed"]), float(row["saltpepper_probability"]), str(row["model"])): row
        for row in rows
    }
    output = []
    for comparator in [MODEL_MLP_RAW, MODEL_MLP_MU]:
        for probability in sorted({float(row["saltpepper_probability"]) for row in rows}):
            seed_differences: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
            keys = {
                (int(row["training_seed"]), str(row["noise_seed"]))
                for row in rows
                if float(row["saltpepper_probability"]) == probability
            }
            for training_seed, noise_seed in keys:
                qnn = lookup[(training_seed, noise_seed, probability, MODEL_SCHMIDT)]
                mlp = lookup[(training_seed, noise_seed, probability, comparator)]
                for metric in ["precision", "recall", "f1", "pr_auc", "f1_retention"]:
                    seed_differences[training_seed][metric].append(float(qnn[metric]) - float(mlp[metric]))
            item = {
                "comparator": comparator,
                "saltpepper_probability": probability,
                "n_raw_pairs": len(keys),
                "n_training_seed_pairs": len(seed_differences),
            }
            for metric in ["precision", "recall", "f1", "pr_auc", "f1_retention"]:
                values = [float(np.mean(metrics[metric])) for metrics in seed_differences.values()]
                item[f"delta_{metric}_mean"], item[f"delta_{metric}_std"] = mean_std(values)
            output.append(item)
    return output


def save_chart(summary: list[dict], path: Path) -> None:
    colors = {MODEL_SCHMIDT: "#4c78a8", MODEL_MLP_RAW: "#f58518", MODEL_MLP_MU: "#54a24b"}
    styles = {
        MODEL_MLP_RAW: {"linestyle": "-", "marker": "o", "zorder": 2},
        MODEL_MLP_MU: {"linestyle": ":", "marker": "x", "zorder": 3},
        MODEL_SCHMIDT: {"linestyle": "--", "marker": "o", "markerfacecolor": "white", "zorder": 4},
    }
    fig, axes = plt.subplots(2, 2, figsize=(12, 9.5), sharex=True)
    specifications = [
        ("f1", "Fixed-threshold F1", (0.0, 1.02)),
        ("pr_auc", "PR-AUC", (0.0, 1.02)),
        ("f1_retention", "F1 retention relative to clean", (0.0, 1.08)),
        ("precision", "Precision at the fixed clean threshold", (0.0, 1.02)),
    ]
    for ax, (metric, title, limits) in zip(axes.flat, specifications):
        for model in [MODEL_MLP_RAW, MODEL_MLP_MU, MODEL_SCHMIDT]:
            subset = sorted([row for row in summary if row["model"] == model], key=lambda row: row["saltpepper_probability"])
            x = [float(row["saltpepper_probability"]) for row in subset]
            y = [float(row[f"{metric}_mean"]) for row in subset]
            error = [float(row[f"{metric}_raw_std"]) for row in subset]
            ax.errorbar(
                x,
                y,
                yerr=error,
                capsize=3,
                linewidth=2,
                color=colors[model],
                label=model,
                **styles[model],
            )
        ax.set_title(title)
        ax.set_ylim(*limits)
        ax.grid(alpha=0.25)
        ax.set_xlabel("Salt-and-pepper probability")
    axes[0, 0].set_ylabel("Score")
    axes[1, 0].set_ylabel("Score")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=3, frameon=False)
    fig.suptitle("Matched lambda12 robustness: explicit Schmidt-QPP versus TinyMLP", y=0.995)
    fig.text(
        0.5,
        0.905,
        "Dashed blue and dotted green curves overlap because Schmidt-QPP and the mu12-TinyMLP give the same metrics.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.875])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_history(path: Path, history: list[dict]) -> None:
    write_csv(path, history)


def run(args: argparse.Namespace) -> None:
    started = time.time()
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = np.load(args.data_path, allow_pickle=True)
    feature_names = [str(name) for name in payload["feature_names"].tolist()]
    y_train_all = payload["y_train"].astype(int)
    y_val_all = payload["y_val"].astype(int)
    y_test_all = payload["y_test"].astype(int)
    train_indices = stratified_indices(y_train_all, args.train_limit, seed=991)
    val_indices = stratified_indices(y_val_all, args.val_limit, seed=992)
    test_indices = stratified_indices(y_test_all, args.test_limit, seed=993)
    y_train = y_train_all[train_indices]
    y_val = y_val_all[val_indices]
    y_test = y_test_all[test_indices]
    x_train_raw = ordered_spectrum(payload["X_train"].astype(np.float32), feature_names)[train_indices]
    x_val_raw = ordered_spectrum(payload["X_val"].astype(np.float32), feature_names)[val_indices]
    x_train_mu = normalized_spectrum(x_train_raw)
    x_val_mu = normalized_spectrum(x_val_raw)
    normalizer = fit_standardizer(x_train_raw, args.mlp_clip)
    x_train_mlp = apply_standardizer(x_train_raw, normalizer)
    x_val_mlp = apply_standardizer(x_val_raw, normalizer)
    print("Preparing noisy held-out spectra...", flush=True)
    noise_cache = prepare_noise_cache(payload, feature_names, test_indices, args)
    rows: list[dict] = []
    run_root = args.output_dir / "runs"
    for training_seed in args.training_seeds:
        print(f"Training seed {training_seed}: Schmidt-QPP", flush=True)
        set_seed(training_seed)
        schmidt = SchmidtQPPQNN2(n_layers=2)
        schmidt, schmidt_history = train_classifier(
            schmidt,
            x_train_raw,
            y_train,
            x_val_raw,
            y_val,
            seed=training_seed,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=device,
        )
        print(f"Training seed {training_seed}: TinyMLP", flush=True)
        set_seed(training_seed)
        mlp = TinyMLP()
        mlp, mlp_history = train_classifier(
            mlp,
            x_train_mlp,
            y_train,
            x_val_mlp,
            y_val,
            seed=training_seed,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=device,
        )
        print(f"Training seed {training_seed}: normalized-spectrum TinyMLP", flush=True)
        set_seed(training_seed)
        mlp_mu = TinyMLP()
        mlp_mu, mlp_mu_history = train_classifier(
            mlp_mu,
            x_train_mu,
            y_train,
            x_val_mu,
            y_val,
            seed=training_seed,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=device,
        )
        seed_dir = run_root / f"seed_{training_seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        torch.save(schmidt.state_dict(), seed_dir / "schmidt_qpp_best_model.pt")
        torch.save(mlp.state_dict(), seed_dir / "tiny_mlp_best_model.pt")
        torch.save(mlp_mu.state_dict(), seed_dir / "tiny_mlp_mu_best_model.pt")
        save_history(seed_dir / "schmidt_qpp_history.csv", schmidt_history)
        save_history(seed_dir / "tiny_mlp_history.csv", mlp_history)
        save_history(seed_dir / "tiny_mlp_mu_history.csv", mlp_mu_history)
        val_schmidt = predict(schmidt, x_val_raw, args.batch_size, device)
        val_mlp = predict(mlp, x_val_mlp, args.batch_size, device)
        val_mlp_mu = predict(mlp_mu, x_val_mu, args.batch_size, device)
        thresholds = {
            MODEL_SCHMIDT: choose_threshold_by_f1(y_val, val_schmidt)[0],
            MODEL_MLP_RAW: choose_threshold_by_f1(y_val, val_mlp)[0],
            MODEL_MLP_MU: choose_threshold_by_f1(y_val, val_mlp_mu)[0],
        }
        models = {
            MODEL_SCHMIDT: (schmidt, lambda values: values, x_train_raw),
            MODEL_MLP_RAW: (mlp, lambda values: apply_standardizer(values, normalizer), x_train_mlp),
            MODEL_MLP_MU: (mlp_mu, normalized_spectrum, x_train_mu),
        }
        clean_metrics: dict[str, dict[str, float]] = {}
        for model_name, (model, transform, _) in models.items():
            scores = predict(model, transform(noise_cache[(0.0, None)]), args.batch_size, device)
            clean_metrics[model_name] = classification_metrics(y_test, scores, thresholds[model_name])
        for probability in args.saltpepper_values:
            evaluation_noise_seeds: list[int | None] = [None] if probability == 0.0 else list(args.noise_seeds)
            for noise_seed in evaluation_noise_seeds:
                spectrum = noise_cache[(probability, noise_seed)]
                for model_name, (model, transform, _) in models.items():
                    scores = predict(model, transform(spectrum), args.batch_size, device)
                    metrics = classification_metrics(y_test, scores, thresholds[model_name])
                    clean_f1 = clean_metrics[model_name]["f1"]
                    rows.append(
                        {
                            "model": model_name,
                            "training_seed": training_seed,
                            "noise_seed": "clean" if noise_seed is None else noise_seed,
                            "saltpepper_probability": probability,
                            "threshold": thresholds[model_name],
                            "trainable_parameters": sum(
                                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
                            ),
                            "train_samples": len(y_train),
                            "validation_samples": len(y_val),
                            "test_samples": len(y_test),
                            **metrics,
                            "clean_f1": clean_f1,
                            "f1_drop": clean_f1 - metrics["f1"],
                            "f1_retention": metrics["f1"] / max(clean_f1, 1e-12),
                        }
                    )
        print(
            f"Seed {training_seed} complete: clean F1 "
            f"Schmidt={clean_metrics[MODEL_SCHMIDT]['f1']:.4f}, "
            f"lambda-MLP={clean_metrics[MODEL_MLP_RAW]['f1']:.4f}, "
            f"mu-MLP={clean_metrics[MODEL_MLP_MU]['f1']:.4f}",
            flush=True,
        )
    summary = aggregate_rows(rows)
    differences = paired_differences(rows)
    write_csv(args.output_dir / "schmidt_qpp_saltpepper_raw.csv", rows)
    write_csv(args.output_dir / "schmidt_qpp_saltpepper_summary.csv", summary)
    write_csv(args.output_dir / "schmidt_qpp_saltpepper_paired_differences.csv", differences)
    (args.output_dir / "schmidt_qpp_saltpepper_results.json").write_text(
        json.dumps({"raw": rows, "summary": summary, "paired_differences": differences}, indent=2), encoding="utf-8"
    )
    save_chart(summary, args.output_dir / "schmidt_qpp_saltpepper_comparison.png")
    protocol = {
        "data_path": str(args.data_path.resolve()),
        "training_seeds": args.training_seeds,
        "noise_seeds": args.noise_seeds,
        "saltpepper_values": args.saltpepper_values,
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "device": str(device),
        "train_samples": len(y_train),
        "validation_samples": len(y_val),
        "test_samples": len(y_test),
        "positive_fraction_train": float(np.mean(y_train)),
        "schmidt_trainable_parameters": 25,
        "tiny_mlp_trainable_parameters": 21,
        "normalized_spectrum_tiny_mlp_trainable_parameters": 21,
        "threshold_protocol": "best F1 on clean validation; frozen for all noisy tests",
        "noise_application": "corrupt held-out images before re-extracting lambda1 and lambda2",
        "elapsed_seconds": time.time() - started,
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
    }
    (args.output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(f"Wrote results to {args.output_dir}", flush=True)


if __name__ == "__main__":
    run(parse_args())
