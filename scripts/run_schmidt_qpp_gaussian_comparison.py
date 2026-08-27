"""Matched Schmidt-QPP versus TinyMLP Gaussian-noise robustness study.

This script reuses the model, training, metric, and feature helpers from the
salt-and-pepper study while applying zero-mean pixel Gaussian noise with
increasing standard deviation.  Models are trained only on clean data and the
clean-validation F1 thresholds remain frozen for every noisy test condition.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
QPP_SRC = ROOT / "qpp_corner_qnn_github_package" / "src"
for path in [ROOT, QPP_SRC]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from qpp_corner.qnn_torch import SchmidtQPPQNN2
from scripts.run_schmidt_qpp_saltpepper_comparison import (
    MODEL_MLP_MU,
    MODEL_MLP_RAW,
    MODEL_SCHMIDT,
    TinyMLP,
    apply_standardizer,
    choose_threshold_by_f1,
    classification_metrics,
    extract_test_spectrum,
    fit_standardizer,
    mean_std,
    normalized_spectrum,
    ordered_spectrum,
    parse_float_list,
    parse_int_list,
    predict,
    resolve_device,
    save_history,
    set_seed,
    stratified_indices,
    train_classifier,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare explicit Schmidt-QPP with matched TinyMLPs under increasing Gaussian image noise."
    )
    parser.add_argument("--data-path", type=Path, default=ROOT / "data" / "feature_dataset_extended.npz")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "schmidt_qpp_gaussian_noise")
    parser.add_argument("--training-seeds", type=parse_int_list, default=parse_int_list("17,29,41,53,67"))
    parser.add_argument("--noise-seeds", type=parse_int_list, default=parse_int_list("1001,2003,3001"))
    parser.add_argument(
        "--gaussian-sigmas",
        type=parse_float_list,
        default=parse_float_list("0,0.01,0.02,0.04,0.06,0.08,0.10,0.15"),
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
    parser.add_argument("--smoke", action="store_true", help="Run a small implementation-check protocol.")
    args = parser.parse_args()
    if args.smoke:
        args.training_seeds = args.training_seeds[:1]
        args.noise_seeds = args.noise_seeds[:1]
        args.gaussian_sigmas = [0.0, 0.04, 0.08]
        args.epochs = min(args.epochs, 3)
        args.patience = min(args.patience, 2)
        args.train_limit = args.train_limit or 320
        args.val_limit = args.val_limit or 240
        args.test_limit = args.test_limit or 300
    if 0.0 not in args.gaussian_sigmas:
        args.gaussian_sigmas = sorted([0.0, *args.gaussian_sigmas])
    return args


def apply_gaussian(images: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    """Add zero-mean Gaussian pixel noise and clip image intensities to [0, 1]."""

    values = np.asarray(images, dtype=np.float32)
    if sigma <= 0.0:
        return values.copy()
    rng = np.random.default_rng(seed + int(round(sigma * 1000)))
    noise = rng.normal(0.0, sigma, size=values.shape).astype(np.float32)
    return np.clip(values + noise, 0.0, 1.0).astype(np.float32)


def prepare_noise_cache(
    payload,
    feature_names: list[str],
    test_indices: np.ndarray,
    args: argparse.Namespace,
) -> dict[tuple[float, int | None], np.ndarray]:
    cache: dict[tuple[float, int | None], np.ndarray] = {}
    cache[(0.0, None)] = ordered_spectrum(payload["X_test"].astype(np.float32), feature_names)[test_indices]
    image_ids = payload["test_image_ids"][test_indices]
    centers = payload["test_centers"][test_indices]
    for sigma in args.gaussian_sigmas:
        if sigma == 0.0:
            continue
        for noise_seed in args.noise_seeds:
            noisy_images = apply_gaussian(payload["images"], sigma, noise_seed)
            cache[(sigma, noise_seed)] = extract_test_spectrum(
                noisy_images, image_ids, centers, feature_names, args.patch_size
            )
    return cache


def aggregate_rows(rows: list[dict]) -> list[dict]:
    """Average noise repeats within a training seed, then summarize training seeds."""

    seed_groups: dict[tuple[str, float, int], list[dict]] = defaultdict(list)
    for row in rows:
        seed_groups[(str(row["model"]), float(row["gaussian_sigma"]), int(row["training_seed"]))].append(row)
    groups: dict[tuple[str, float], list[dict[str, float]]] = defaultdict(list)
    metrics = ["precision", "recall", "f1", "pr_auc", "f1_drop", "f1_retention"]
    for (model, sigma, _), subset in seed_groups.items():
        groups[(model, sigma)].append(
            {metric: float(np.mean([float(row[metric]) for row in subset])) for metric in metrics}
        )
    output = []
    for (model, sigma), seed_means in sorted(groups.items(), key=lambda item: (item[0][1], item[0][0])):
        raw_subset = [row for row in rows if row["model"] == model and float(row["gaussian_sigma"]) == sigma]
        item = {
            "model": model,
            "gaussian_sigma": sigma,
            "n_raw_evaluations": len(raw_subset),
            "n_training_seeds": len(seed_means),
            "n_noise_seeds_per_training_seed": 1
            if sigma == 0.0
            else len({str(row["noise_seed"]) for row in raw_subset}),
            "trainable_parameters": int(raw_subset[0]["trainable_parameters"]),
        }
        for metric in metrics:
            item[f"{metric}_mean"], item[f"{metric}_std"] = mean_std(
                [float(seed_mean[metric]) for seed_mean in seed_means]
            )
            _, item[f"{metric}_raw_std"] = mean_std([float(row[metric]) for row in raw_subset])
            within_noise_stds = []
            for training_seed in sorted({int(row["training_seed"]) for row in raw_subset}):
                seed_values = [
                    float(row[metric]) for row in raw_subset if int(row["training_seed"]) == training_seed
                ]
                _, seed_std = mean_std(seed_values)
                within_noise_stds.append(seed_std)
            item[f"{metric}_within_noise_std_mean"] = float(np.mean(within_noise_stds))
        output.append(item)
    return output


def paired_differences(rows: list[dict]) -> list[dict]:
    lookup = {
        (int(row["training_seed"]), str(row["noise_seed"]), float(row["gaussian_sigma"]), str(row["model"])): row
        for row in rows
    }
    output = []
    metrics = ["precision", "recall", "f1", "pr_auc", "f1_retention"]
    for comparator in [MODEL_MLP_RAW, MODEL_MLP_MU]:
        for sigma in sorted({float(row["gaussian_sigma"]) for row in rows}):
            seed_differences: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
            keys = {
                (int(row["training_seed"]), str(row["noise_seed"]))
                for row in rows
                if float(row["gaussian_sigma"]) == sigma
            }
            for training_seed, noise_seed in keys:
                qnn = lookup[(training_seed, noise_seed, sigma, MODEL_SCHMIDT)]
                mlp = lookup[(training_seed, noise_seed, sigma, comparator)]
                for metric in metrics:
                    seed_differences[training_seed][metric].append(float(qnn[metric]) - float(mlp[metric]))
            item = {
                "comparator": comparator,
                "gaussian_sigma": sigma,
                "n_raw_pairs": len(keys),
                "n_training_seed_pairs": len(seed_differences),
            }
            for metric in metrics:
                values = [float(np.mean(seed_metrics[metric])) for seed_metrics in seed_differences.values()]
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
            subset = sorted([row for row in summary if row["model"] == model], key=lambda row: row["gaussian_sigma"])
            x = [float(row["gaussian_sigma"]) for row in subset]
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
        ax.set_xlabel("Gaussian noise standard deviation")
    axes[0, 0].set_ylabel("Score")
    axes[1, 0].set_ylabel("Score")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=3, frameon=False)
    fig.suptitle("Matched lambda12 robustness under increasing Gaussian image noise", y=0.995)
    fig.text(
        0.5,
        0.905,
        "Models are clean-trained; each clean-validation F1 threshold is frozen across all noise levels.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.875])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


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
    print("Preparing Gaussian-noisy held-out spectra...", flush=True)
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
        print(f"Training seed {training_seed}: raw-spectrum TinyMLP", flush=True)
        set_seed(training_seed)
        mlp_raw = TinyMLP()
        mlp_raw, mlp_raw_history = train_classifier(
            mlp_raw,
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
        torch.save(mlp_raw.state_dict(), seed_dir / "tiny_mlp_lambda_best_model.pt")
        torch.save(mlp_mu.state_dict(), seed_dir / "tiny_mlp_mu_best_model.pt")
        save_history(seed_dir / "schmidt_qpp_history.csv", schmidt_history)
        save_history(seed_dir / "tiny_mlp_lambda_history.csv", mlp_raw_history)
        save_history(seed_dir / "tiny_mlp_mu_history.csv", mlp_mu_history)
        thresholds = {
            MODEL_SCHMIDT: choose_threshold_by_f1(
                y_val, predict(schmidt, x_val_raw, args.batch_size, device)
            )[0],
            MODEL_MLP_RAW: choose_threshold_by_f1(
                y_val, predict(mlp_raw, x_val_mlp, args.batch_size, device)
            )[0],
            MODEL_MLP_MU: choose_threshold_by_f1(
                y_val, predict(mlp_mu, x_val_mu, args.batch_size, device)
            )[0],
        }
        models = {
            MODEL_SCHMIDT: (schmidt, lambda values: values),
            MODEL_MLP_RAW: (mlp_raw, lambda values: apply_standardizer(values, normalizer)),
            MODEL_MLP_MU: (mlp_mu, normalized_spectrum),
        }
        clean_metrics: dict[str, dict[str, float]] = {}
        for model_name, (model, transform) in models.items():
            scores = predict(model, transform(noise_cache[(0.0, None)]), args.batch_size, device)
            clean_metrics[model_name] = classification_metrics(y_test, scores, thresholds[model_name])
        for sigma in args.gaussian_sigmas:
            evaluation_noise_seeds: list[int | None] = [None] if sigma == 0.0 else list(args.noise_seeds)
            for noise_seed in evaluation_noise_seeds:
                spectrum = noise_cache[(sigma, noise_seed)]
                for model_name, (model, transform) in models.items():
                    scores = predict(model, transform(spectrum), args.batch_size, device)
                    metrics = classification_metrics(y_test, scores, thresholds[model_name])
                    clean_f1 = clean_metrics[model_name]["f1"]
                    rows.append(
                        {
                            "model": model_name,
                            "training_seed": training_seed,
                            "noise_seed": "clean" if noise_seed is None else noise_seed,
                            "gaussian_sigma": sigma,
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
    write_csv(args.output_dir / "schmidt_qpp_gaussian_raw.csv", rows)
    write_csv(args.output_dir / "schmidt_qpp_gaussian_summary.csv", summary)
    write_csv(args.output_dir / "schmidt_qpp_gaussian_paired_differences.csv", differences)
    (args.output_dir / "schmidt_qpp_gaussian_results.json").write_text(
        json.dumps({"raw": rows, "summary": summary, "paired_differences": differences}, indent=2), encoding="utf-8"
    )
    save_chart(summary, args.output_dir / "schmidt_qpp_gaussian_comparison.png")
    protocol = {
        "data_path": str(args.data_path.resolve()),
        "training_seeds": args.training_seeds,
        "noise_seeds": args.noise_seeds,
        "gaussian_sigmas": args.gaussian_sigmas,
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
        "threshold_protocol": "best F1 on clean validation; frozen for all Gaussian-noisy tests",
        "noise_application": "add N(0, sigma^2) to held-out images, clip to [0,1], then re-extract lambda1/lambda2",
        "elapsed_seconds": time.time() - started,
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
    }
    (args.output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(f"Wrote results to {args.output_dir}", flush=True)


if __name__ == "__main__":
    run(parse_args())
