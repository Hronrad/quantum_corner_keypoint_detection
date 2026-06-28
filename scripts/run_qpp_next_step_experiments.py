from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier

ROOT = Path(__file__).resolve().parents[1]
QPP_SRC = ROOT / "qpp_corner_qnn_github_package" / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(QPP_SRC) not in sys.path:
    sys.path.insert(0, str(QPP_SRC))

from qpp_corner.classical import fit_logistic
from qpp_corner.metrics import binary_metrics, choose_threshold_by_f1
from qpp_corner.normalizer import FeatureNormalizer
from qpp_corner.qnn_torch import (
    DataReuploadingQNN1,
    DataReuploadingQNN2,
    Ry,
    Rz,
    apply_cnot,
    apply_single_qubit_gate,
    expectation_z,
    expectation_z0z1,
    zero_state,
)
from qpp_corner.train import predict_torch, set_seed, train_torch_classifier
from scripts.run_improvement_experiments import apply_noise_to_images, extract_features_for_centers

try:
    import torch
    from torch import nn
except Exception as exc:  # pragma: no cover
    raise ImportError("PyTorch is required for QPP next-step experiments.") from exc


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(exist_ok=True)
    payload = np.load(args.data_path, allow_pickle=True)
    feature_names = [str(name) for name in payload["feature_names"].tolist()]
    qpp = {
        split: qpp_feature_sets_from_extended(payload[f"X_{split}"].astype(np.float32), feature_names)
        for split in ["train", "val", "test"]
    }

    low_rows = run_low_sample(payload, qpp, args)
    write_rows(args.output_dir / "qpp_low_sample_results.csv", low_rows)
    write_json(args.output_dir / "qpp_low_sample_results.json", low_rows)
    save_grouped_curve(
        low_rows,
        args.output_dir / "qpp_low_sample_results.png",
        x_key="train_positives",
        series_key="model_name",
        y_keys=("f1", "pr_auc"),
        title="Low-sample comparison",
    )

    noise_aware_rows = run_noise_aware_training(payload, feature_names, args)
    write_rows(args.output_dir / "qpp_noise_aware_results.csv", noise_aware_rows)
    write_json(args.output_dir / "qpp_noise_aware_results.json", noise_aware_rows)
    save_bar_chart(
        noise_aware_rows,
        args.output_dir / "qpp_noise_aware_results.png",
        x_key="setting",
        y_key="f1",
        title="Noise-aware QPP QNN on salt-and-pepper test",
    )

    structure_rows = run_structure_ablation(payload, qpp, args)
    write_rows(args.output_dir / "qpp_structure_ablation_results.csv", structure_rows)
    write_json(args.output_dir / "qpp_structure_ablation_results.json", structure_rows)
    save_bar_chart(
        structure_rows,
        args.output_dir / "qpp_structure_ablation_results.png",
        x_key="name",
        y_key="f1",
        title="QPP QNN structure ablation",
    )

    phase_rows = run_phase_mapping(payload, qpp, args)
    write_rows(args.output_dir / "qpp_phase_mapping_results.csv", phase_rows)
    write_json(args.output_dir / "qpp_phase_mapping_results.json", phase_rows)
    save_bar_chart(
        phase_rows,
        args.output_dir / "qpp_phase_mapping_results.png",
        x_key="name",
        y_key="f1",
        title="QPP phase mapping ablation",
    )

    resource_rows = run_resource_validation(payload, qpp, feature_names, args)
    write_rows(args.output_dir / "qpp_resource_advantage_results.csv", resource_rows)
    write_json(args.output_dir / "qpp_resource_advantage_results.json", resource_rows)
    save_grouped_bars(
        resource_rows,
        args.output_dir / "qpp_resource_advantage_results.png",
        group_key="condition",
        series_key="model_name",
        y_key="f1",
        title="Resource-limited comparison",
    )

    final_rows = build_final_comparison(args.output_dir)
    write_rows(args.output_dir / "final_comparison_results.csv", final_rows)
    save_bar_chart(
        final_rows,
        args.output_dir / "final_comparison_results.png",
        x_key="method",
        y_key="f1",
        title="Final clean-test comparison",
    )

    print("QPP next-step experiments complete.")
    print(json.dumps({"outputs": [str(p) for p in sorted(args.output_dir.glob("qpp_*_results.csv"))]}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run QPP next-step experiments requested for the current project.")
    parser.add_argument("--data-path", type=Path, default=ROOT / "data" / "feature_dataset_extended.npz")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--seed", type=int, default=57)
    parser.add_argument("--epochs", type=int, default=26)
    parser.add_argument("--low-sample-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--patch-size", type=int, default=9)
    parser.add_argument("--train-positives", type=int, nargs="+", default=[20, 50, 100, 200])
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
        "scalar_c2": (log_s + 2.0 * eta).reshape(-1, 1).astype(np.float32),
        "scalar_c4": (log_s + 4.0 * eta).reshape(-1, 1).astype(np.float32),
        "log_lambda12": np.column_stack([np.log(lambda1 + 1e-8), np.log(lambda2 + 1e-8)]).astype(np.float32),
        "symmetry_interleaved": np.column_stack([log_s, eta]).astype(np.float32),
    }


def labels(payload, split: str) -> np.ndarray:
    return payload[f"y_{split}"].astype(int)


def stratified_positive_subset(y: np.ndarray, positive_count: int, negative_ratio: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    n_pos = min(len(pos), int(positive_count))
    n_neg = min(len(neg), int(positive_count) * int(negative_ratio))
    chosen = np.concatenate([rng.choice(pos, n_pos, replace=False), rng.choice(neg, n_neg, replace=False)])
    rng.shuffle(chosen)
    return chosen


def fit_mlp_score(x_train: np.ndarray, y_train: np.ndarray, seed: int) -> object:
    model = MLPClassifier(
        hidden_layer_sizes=(8,),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate_init=0.01,
        batch_size=min(64, max(16, len(y_train))),
        max_iter=450,
        random_state=seed,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(x_train, y_train)
    return model


def run_low_sample(payload, qpp: dict, args: argparse.Namespace) -> list[dict]:
    rows = []
    y_train = labels(payload, "train")
    y_val = labels(payload, "val")
    y_test = labels(payload, "test")
    for positive_count in args.train_positives:
        idx = stratified_positive_subset(y_train, positive_count, negative_ratio=4, seed=args.seed + positive_count)
        train_samples = int(len(idx))
        actual_pos = int(np.sum(y_train[idx] == 1))
        train_pack = {
            "train": qpp["train"]["lambda12"][idx],
            "val": qpp["val"]["lambda12"],
            "test": qpp["test"]["lambda12"],
        }
        rows.append(
            run_logistic_row(
                name=f"logistic_{positive_count}pos",
                model_name="logistic",
                feature_set="lambda12",
                train_pack=train_pack,
                y_train=y_train[idx],
                y_val=y_val,
                y_test=y_test,
                seed=args.seed,
                extra={"train_positives": actual_pos, "train_samples": train_samples, "condition": "clean"},
            )
        )
        rows.append(
            run_mlp_row(
                name=f"mlp_{positive_count}pos",
                model_name="mlp",
                feature_set="lambda12",
                train_pack=train_pack,
                y_train=y_train[idx],
                y_val=y_val,
                y_test=y_test,
                seed=args.seed,
                extra={"train_positives": actual_pos, "train_samples": train_samples, "condition": "clean"},
            )
        )
        rows.append(
            run_qnn2_row(
                name=f"qpp_qnn2_{positive_count}pos",
                model_name="qpp_qnn2",
                feature_set="lambda12",
                train_pack=train_pack,
                y_train=y_train[idx],
                y_val=y_val,
                y_test=y_test,
                args=args,
                n_layers=2,
                entanglement="linear_01",
                readout="z_z_zz",
                run_suffix=f"low_{positive_count}",
                epochs=args.low_sample_epochs,
                extra={"train_positives": actual_pos, "train_samples": train_samples, "condition": "clean"},
            )
        )
    return rows


def run_noise_aware_training(payload, feature_names: list[str], args: argparse.Namespace) -> list[dict]:
    clean_train = qpp_feature_sets_from_extended(payload["X_train"].astype(np.float32), feature_names)
    clean_val = qpp_feature_sets_from_extended(payload["X_val"].astype(np.float32), feature_names)
    y_train = labels(payload, "train")
    y_val = labels(payload, "val")
    y_test = labels(payload, "test")
    salt_test_x = noisy_extended_features(payload, "test", "saltpepper", 0.03, args)
    salt_val_x = noisy_extended_features(payload, "val", "saltpepper", 0.03, args)
    salt_test = qpp_feature_sets_from_extended(salt_test_x, feature_names)
    salt_val = qpp_feature_sets_from_extended(salt_val_x, feature_names)

    rows = []
    clean_pack = {
        "train": clean_train["lambda12"],
        "val": clean_val["lambda12"],
        "test": salt_test["lambda12"],
    }
    rows.append(
        run_qnn2_row(
            name="clean_train_saltpepper_test",
            model_name="qpp_qnn2_clean_train",
            feature_set="lambda12",
            train_pack=clean_pack,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            args=args,
            n_layers=2,
            entanglement="linear_01",
            readout="z_z_zz",
            run_suffix="noiseaware_clean",
            epochs=args.epochs,
            extra={"setting": "clean train"},
        )
    )

    train_parts = [payload["X_train"].astype(np.float32)]
    val_parts = [payload["X_val"].astype(np.float32), salt_val_x]
    train_y_parts = [y_train]
    val_y_parts = [y_val, y_val]
    for noise_type, value in [("gaussian", 0.04), ("gaussian", 0.08), ("blur", 0.9), ("saltpepper", 0.03)]:
        train_parts.append(noisy_extended_features(payload, "train", noise_type, value, args))
        train_y_parts.append(y_train)
    aug_train = qpp_feature_sets_from_extended(np.vstack(train_parts), feature_names)
    aug_val = qpp_feature_sets_from_extended(np.vstack(val_parts), feature_names)
    aug_y_train = np.concatenate(train_y_parts)
    aug_y_val = np.concatenate(val_y_parts)
    aug_pack = {"train": aug_train["lambda12"], "val": aug_val["lambda12"], "test": salt_test["lambda12"]}
    rows.append(
        run_qnn2_row(
            name="noise_aware_train_saltpepper_test",
            model_name="qpp_qnn2_noise_aware",
            feature_set="lambda12",
            train_pack=aug_pack,
            y_train=aug_y_train,
            y_val=aug_y_val,
            y_test=y_test,
            args=args,
            n_layers=2,
            entanglement="linear_01",
            readout="z_z_zz",
            run_suffix="noiseaware_aug",
            epochs=args.epochs,
            extra={
                "setting": "noise-aware train",
                "train_samples": int(len(aug_y_train)),
                "val_samples": int(len(aug_y_val)),
            },
        )
    )
    return rows


def noisy_extended_features(payload, split: str, noise_type: str, value: float, args: argparse.Namespace) -> np.ndarray:
    images = apply_noise_to_images(payload["images"], noise_type, value, args.seed)
    return extract_features_for_centers(
        images,
        payload[f"{split}_image_ids"],
        payload[f"{split}_centers"],
        args.patch_size,
    ).astype(np.float32)


def run_structure_ablation(payload, qpp: dict, args: argparse.Namespace) -> list[dict]:
    y_train = labels(payload, "train")
    y_val = labels(payload, "val")
    y_test = labels(payload, "test")
    rows = []
    rows.append(
        run_qnn1_row(
            "1q_scalar_c4_L2_Z",
            "qpp_qnn1",
            "scalar_c4",
            pack_for(qpp, "scalar_c4"),
            y_train,
            y_val,
            y_test,
            args,
            n_layers=2,
            run_suffix="struct_1q_c4",
            extra={"ablation": "qubits"},
        )
    )
    for layers in [1, 2, 3]:
        rows.append(
            run_qnn2_flexible_row(
                name=f"2q_lambda12_linear_L{layers}_Z_ZZ",
                model_name="qpp_qnn2",
                feature_set="lambda12",
                train_pack=pack_for(qpp, "lambda12"),
                y_train=y_train,
                y_val=y_val,
                y_test=y_test,
                args=args,
                n_layers=layers,
                entanglement="linear_01",
                readout="z_z_zz",
                run_suffix=f"struct_lambda12_linear_L{layers}",
                extra={"ablation": "layers"},
            )
        )
    for feature_set in ["lambda12", "logS_eta"]:
        for entanglement in ["none", "linear_01", "bidirectional"]:
            rows.append(
                run_qnn2_flexible_row(
                    name=f"2q_{feature_set}_{entanglement}_Z_ZZ",
                    model_name="qpp_qnn2",
                    feature_set=feature_set,
                    train_pack=pack_for(qpp, feature_set),
                    y_train=y_train,
                    y_val=y_val,
                    y_test=y_test,
                    args=args,
                    n_layers=2,
                    entanglement=entanglement,
                    readout="z_z_zz",
                    run_suffix=f"struct_{feature_set}_{entanglement}_zz",
                    extra={"ablation": "entanglement"},
                )
            )
        for readout in ["z_only", "z_z_zz", "xyz_zz"]:
            rows.append(
                run_qnn2_flexible_row(
                    name=f"2q_{feature_set}_linear_{readout}",
                    model_name="qpp_qnn2",
                    feature_set=feature_set,
                    train_pack=pack_for(qpp, feature_set),
                    y_train=y_train,
                    y_val=y_val,
                    y_test=y_test,
                    args=args,
                    n_layers=2,
                    entanglement="linear_01",
                    readout=readout,
                    run_suffix=f"struct_{feature_set}_linear_{readout}",
                    extra={"ablation": "readout"},
                )
            )
    return rows


def run_phase_mapping(payload, qpp: dict, args: argparse.Namespace) -> list[dict]:
    y_train = labels(payload, "train")
    y_val = labels(payload, "val")
    y_test = labels(payload, "test")
    rows = []
    rows.append(
        run_qnn1_row(
            "fixed_scalar_c4",
            "qpp_qnn1",
            "scalar_c4",
            pack_for(qpp, "scalar_c4"),
            y_train,
            y_val,
            y_test,
            args,
            n_layers=2,
            run_suffix="phase_scalar_c4",
            extra={"phase_mapping": "fixed scalarizer"},
        )
    )
    rows.append(
        run_qnn1_row(
            "learnable_scalarizer_logS_eta",
            "qpp_qnn1_learnable",
            "logS_eta",
            pack_for(qpp, "logS_eta"),
            y_train,
            y_val,
            y_test,
            args,
            n_layers=2,
            run_suffix="phase_learnable_scalarizer",
            learnable_projection=True,
            extra={"phase_mapping": "learnable scalarizer"},
        )
    )
    rows.append(
        run_qnn2_flexible_row(
            "log_eigenvalue_encoding",
            "qpp_qnn2",
            "log_lambda12",
            pack_for(qpp, "log_lambda12"),
            y_train,
            y_val,
            y_test,
            args,
            n_layers=2,
            entanglement="linear_01",
            readout="z_z_zz",
            run_suffix="phase_log_lambda12",
            extra={"phase_mapping": "log eigenvalue encoding"},
        )
    )
    rows.append(
        run_qnn2_flexible_row(
            "symmetry_preserving_logS_eta",
            "qpp_qnn2",
            "logS_eta",
            pack_for(qpp, "logS_eta"),
            y_train,
            y_val,
            y_test,
            args,
            n_layers=2,
            entanglement="linear_01",
            readout="z_z_zz",
            run_suffix="phase_symmetry",
            extra={"phase_mapping": "symmetry-preserving"},
        )
    )
    rows.append(
        run_interleaved_row(
            "interleaved_phase_logS_eta",
            "qpp_qnn1_interleaved",
            "logS_eta",
            pack_for(qpp, "logS_eta"),
            y_train,
            y_val,
            y_test,
            args,
            run_suffix="phase_interleaved",
            extra={"phase_mapping": "junction interleaved phase"},
        )
    )
    return rows


def run_resource_validation(payload, qpp: dict, feature_names: list[str], args: argparse.Namespace) -> list[dict]:
    y_train = labels(payload, "train")
    idx = stratified_positive_subset(y_train, 100, negative_ratio=4, seed=args.seed + 1000)
    y_sub = y_train[idx]
    y_val = labels(payload, "val")
    y_test = labels(payload, "test")
    noisy_test_x = noisy_extended_features(payload, "test", "saltpepper", 0.03, args)
    noisy_qpp_test = qpp_feature_sets_from_extended(noisy_test_x, feature_names)
    rows = []
    for condition, test_pack in [("clean", qpp["test"]), ("saltpepper_0.03", noisy_qpp_test)]:
        pack = {
            "train": qpp["train"]["lambda12"][idx],
            "val": qpp["val"]["lambda12"],
            "test": test_pack["lambda12"],
        }
        common = {
            "condition": condition,
            "train_positives": int(np.sum(y_sub == 1)),
            "train_samples": int(len(y_sub)),
            "shots": 1024,
            "resource_note": "2 input features; QPP uses 2 qubits/2 layers with scores quantized to 1024-shot resolution; MLP uses one 8-unit hidden layer",
        }
        rows.append(
            run_logistic_row(
                "resource_logistic",
                "logistic",
                "lambda12",
                pack,
                y_sub,
                y_val,
                y_test,
                args.seed,
                extra=common,
            )
        )
        rows.append(
            run_mlp_row("resource_mlp8", "mlp8", "lambda12", pack, y_sub, y_val, y_test, args.seed, extra=common)
        )
        rows.append(
            run_qnn2_row(
                "resource_qpp_qnn2",
                "qpp_qnn2",
                "lambda12",
                pack,
                y_sub,
                y_val,
                y_test,
                args,
                n_layers=2,
                entanglement="linear_01",
                readout="z_z_zz",
                run_suffix=f"resource_{condition}",
                epochs=args.low_sample_epochs,
                shots=1024,
                extra=common,
            )
        )
    return rows


def pack_for(qpp: dict, feature_set: str) -> dict[str, np.ndarray]:
    return {"train": qpp["train"][feature_set], "val": qpp["val"][feature_set], "test": qpp["test"][feature_set]}


def run_logistic_row(
    name: str,
    model_name: str,
    feature_set: str,
    train_pack: dict[str, np.ndarray],
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    *,
    extra: dict | None = None,
) -> dict:
    normalizer = FeatureNormalizer()
    x_train = normalizer.fit_transform(train_pack["train"], [f"f{i}" for i in range(train_pack["train"].shape[1])])
    x_val = normalizer.transform(train_pack["val"])
    x_test = normalizer.transform(train_pack["test"])
    model = fit_logistic(x_train, y_train, seed=seed)
    val_scores = model.predict_scores(x_val)
    threshold, val_f1 = choose_threshold_by_f1(y_val, val_scores)
    test_scores = model.predict_scores(x_test)
    return result_row(
        name,
        model_name,
        feature_set,
        n_qubits=0,
        layers=0,
        val_f1=val_f1,
        threshold=threshold,
        metrics=binary_metrics(y_test, test_scores, threshold),
        train_samples=len(y_train),
        test_samples=len(y_test),
        extra=extra,
    )


def run_mlp_row(
    name: str,
    model_name: str,
    feature_set: str,
    train_pack: dict[str, np.ndarray],
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    *,
    extra: dict | None = None,
) -> dict:
    normalizer = FeatureNormalizer()
    x_train = normalizer.fit_transform(train_pack["train"], [f"f{i}" for i in range(train_pack["train"].shape[1])])
    x_val = normalizer.transform(train_pack["val"])
    x_test = normalizer.transform(train_pack["test"])
    model = fit_mlp_score(x_train, y_train, seed)
    val_scores = model.predict_proba(x_val)[:, 1]
    threshold, val_f1 = choose_threshold_by_f1(y_val, val_scores)
    test_scores = model.predict_proba(x_test)[:, 1]
    return result_row(
        name,
        model_name,
        feature_set,
        n_qubits=0,
        layers=1,
        val_f1=val_f1,
        threshold=threshold,
        metrics=binary_metrics(y_test, test_scores, threshold),
        train_samples=len(y_train),
        test_samples=len(y_test),
        extra=extra,
    )


def run_qnn1_row(
    name: str,
    model_name: str,
    feature_set: str,
    train_pack: dict[str, np.ndarray],
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    args: argparse.Namespace,
    *,
    n_layers: int,
    run_suffix: str,
    learnable_projection: bool = False,
    extra: dict | None = None,
) -> dict:
    model = DataReuploadingQNN1(
        n_layers=n_layers,
        encoding="ryrz",
        input_dim=train_pack["train"].shape[1],
        learnable_projection=learnable_projection,
    )
    return run_torch_row(
        name,
        model_name,
        feature_set,
        model,
        train_pack,
        y_train,
        y_val,
        y_test,
        args,
        n_qubits=1,
        layers=n_layers,
        run_suffix=run_suffix,
        extra=extra,
    )


def run_qnn2_row(
    name: str,
    model_name: str,
    feature_set: str,
    train_pack: dict[str, np.ndarray],
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    args: argparse.Namespace,
    *,
    n_layers: int,
    entanglement: str,
    readout: str,
    run_suffix: str,
    epochs: int | None = None,
    shots: int | None = None,
    extra: dict | None = None,
) -> dict:
    model = DataReuploadingQNN2(n_layers=n_layers, encoding="ryrz", entanglement=entanglement, readout=readout)
    return run_torch_row(
        name,
        model_name,
        feature_set,
        model,
        train_pack,
        y_train,
        y_val,
        y_test,
        args,
        n_qubits=2,
        layers=n_layers,
        run_suffix=run_suffix,
        epochs=epochs,
        shots=shots,
        extra=extra,
    )


def run_qnn2_flexible_row(
    name: str,
    model_name: str,
    feature_set: str,
    train_pack: dict[str, np.ndarray],
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    args: argparse.Namespace,
    *,
    n_layers: int,
    entanglement: str,
    readout: str,
    run_suffix: str,
    extra: dict | None = None,
) -> dict:
    model = FlexibleReadoutQNN2(n_layers=n_layers, encoding="ryrz", entanglement=entanglement, readout=readout)
    return run_torch_row(
        name,
        model_name,
        feature_set,
        model,
        train_pack,
        y_train,
        y_val,
        y_test,
        args,
        n_qubits=2,
        layers=n_layers,
        run_suffix=run_suffix,
        extra=extra,
    )


def run_interleaved_row(
    name: str,
    model_name: str,
    feature_set: str,
    train_pack: dict[str, np.ndarray],
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    args: argparse.Namespace,
    *,
    run_suffix: str,
    extra: dict | None = None,
) -> dict:
    model = InterleavedPhaseQNN1(n_layers=2)
    return run_torch_row(
        name,
        model_name,
        feature_set,
        model,
        train_pack,
        y_train,
        y_val,
        y_test,
        args,
        n_qubits=1,
        layers=2,
        run_suffix=run_suffix,
        extra=extra,
    )


def run_torch_row(
    name: str,
    model_name: str,
    feature_set: str,
    model: nn.Module,
    train_pack: dict[str, np.ndarray],
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    args: argparse.Namespace,
    *,
    n_qubits: int,
    layers: int,
    run_suffix: str,
    epochs: int | None = None,
    shots: int | None = None,
    extra: dict | None = None,
) -> dict:
    normalizer = FeatureNormalizer()
    x_train = normalizer.to_angles(normalizer.fit_transform(train_pack["train"], [f"f{i}" for i in range(train_pack["train"].shape[1])]))
    x_val = normalizer.to_angles(normalizer.transform(train_pack["val"]))
    x_test = normalizer.to_angles(normalizer.transform(train_pack["test"]))
    model, _ = train_torch_classifier(
        model,
        x_train,
        y_train,
        x_val,
        y_val,
        out_dir=args.output_dir / f"qpp_next_{run_suffix}",
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs if epochs is None else epochs,
        patience=args.patience,
        seed=args.seed,
        device="cpu",
    )
    val_scores = predict_torch(model, x_val, batch_size=args.batch_size, device="cpu")
    if shots is not None:
        val_scores = quantize_scores(val_scores, shots)
    threshold, val_f1 = choose_threshold_by_f1(y_val, val_scores)
    test_scores = predict_torch(model, x_test, batch_size=args.batch_size, device="cpu")
    if shots is not None:
        test_scores = quantize_scores(test_scores, shots)
    return result_row(
        name,
        model_name,
        feature_set,
        n_qubits=n_qubits,
        layers=layers,
        val_f1=val_f1,
        threshold=threshold,
        metrics=binary_metrics(y_test, test_scores, threshold),
        train_samples=len(y_train),
        test_samples=len(y_test),
        extra=extra,
    )


def quantize_scores(scores: np.ndarray, shots: int) -> np.ndarray:
    clipped = np.clip(np.asarray(scores, dtype=np.float64), 0.0, 1.0)
    return np.round(clipped * int(shots)) / float(shots)


class FlexibleReadoutQNN2(nn.Module):
    def __init__(self, n_layers: int, *, encoding: str, entanglement: str, readout: str) -> None:
        super().__init__()
        self.n_layers = int(n_layers)
        self.encoding = encoding
        self.entanglement = entanglement
        self.readout = readout
        self.theta = nn.Parameter(0.05 * torch.randn(self.n_layers, 2, 3))
        self.head = nn.Linear(self.readout_dim, 1)

    @property
    def readout_dim(self) -> int:
        return {"z_only": 2, "z_z_zz": 3, "xyz_zz": 7}[self.readout]

    def _encode(self, state, phi):
        for qubit in range(2):
            angle = phi[:, qubit]
            if self.encoding == "ryrz":
                state = apply_single_qubit_gate(state, Ry(angle), qubit, 2)
                state = apply_single_qubit_gate(state, Rz(angle), qubit, 2)
            elif self.encoding == "qpp_z":
                state = apply_single_qubit_gate(state, Rz(angle), qubit, 2)
                state = apply_single_qubit_gate(state, Ry(torch.full_like(angle, math.pi / 2)), qubit, 2)
            else:
                raise ValueError(f"Unknown encoding: {self.encoding}")
        return state

    def _trainable(self, state, layer: int):
        for qubit in range(2):
            a, b, c = self.theta[layer, qubit]
            state = apply_single_qubit_gate(state, Rz(a), qubit, 2)
            state = apply_single_qubit_gate(state, Ry(b), qubit, 2)
            state = apply_single_qubit_gate(state, Rz(c), qubit, 2)
        return state

    def _entangle(self, state):
        if self.entanglement == "none":
            return state
        if self.entanglement == "linear_01":
            return apply_cnot(state, 0, 1, 2)
        if self.entanglement == "bidirectional":
            state = apply_cnot(state, 0, 1, 2)
            return apply_cnot(state, 1, 0, 2)
        raise ValueError(f"Unknown entanglement: {self.entanglement}")

    def statevector(self, phi):
        if phi.shape[-1] != 2:
            raise ValueError(f"2q QNN expects 2 features, got {tuple(phi.shape)}")
        state = zero_state(phi.shape[0], 2, phi.device)
        for layer in range(self.n_layers):
            state = self._encode(state, phi)
            state = self._trainable(state, layer)
            state = self._entangle(state)
        return state

    def observables(self, phi):
        state = self.statevector(phi.to(dtype=torch.float32))
        z0 = expectation_z(state, 0, 2)
        z1 = expectation_z(state, 1, 2)
        if self.readout == "z_only":
            obs = [z0, z1]
        elif self.readout == "z_z_zz":
            obs = [z0, z1, expectation_z0z1(state)]
        elif self.readout == "xyz_zz":
            obs = [
                pauli_expectation(state, "X", 0, 2),
                pauli_expectation(state, "Y", 0, 2),
                z0,
                pauli_expectation(state, "X", 1, 2),
                pauli_expectation(state, "Y", 1, 2),
                z1,
                expectation_z0z1(state),
            ]
        else:
            raise ValueError(f"Unknown readout: {self.readout}")
        return torch.stack(obs, dim=1).to(dtype=torch.float32)

    def forward(self, phi):
        return self.head(self.observables(phi)).squeeze(-1)


class InterleavedPhaseQNN1(nn.Module):
    def __init__(self, n_layers: int) -> None:
        super().__init__()
        self.n_layers = int(n_layers)
        self.theta = nn.Parameter(0.05 * torch.randn(self.n_layers, 3))
        self.head = nn.Linear(1, 1)

    def statevector(self, x):
        if x.shape[-1] != 2:
            raise ValueError("InterleavedPhaseQNN1 expects [logS, eta].")
        x = x.to(dtype=torch.float32)
        state = zero_state(x.shape[0], 1, x.device)
        for layer in range(self.n_layers):
            log_s = x[:, 0]
            eta = x[:, 1]
            a, b, c = self.theta[layer]
            state = apply_single_qubit_gate(state, Rz(log_s), 0, 1)
            state = apply_single_qubit_gate(state, Ry(a + eta), 0, 1)
            state = apply_single_qubit_gate(state, Rz(eta), 0, 1)
            state = apply_single_qubit_gate(state, Ry(b + log_s * eta), 0, 1)
            state = apply_single_qubit_gate(state, Rz(c), 0, 1)
        return state

    def observables(self, x):
        return expectation_z(self.statevector(x), 0, 1).to(dtype=torch.float32).unsqueeze(1)

    def forward(self, x):
        return self.head(self.observables(x)).squeeze(-1)


def pauli_expectation(state, pauli: str, qubit: int, n_qubits: int):
    dim = 2**n_qubits
    indices = torch.arange(dim, device=state.device)
    mask = 1 << (n_qubits - 1 - qubit)
    if pauli == "Z":
        return expectation_z(state, qubit, n_qubits)
    mapped = indices ^ mask
    bit_is_one = (indices & mask) != 0
    out = torch.zeros_like(state)
    if pauli == "X":
        phase = torch.ones(dim, dtype=torch.complex64, device=state.device)
    elif pauli == "Y":
        phase = torch.where(
            bit_is_one,
            torch.full((dim,), -1j, dtype=torch.complex64, device=state.device),
            torch.full((dim,), 1j, dtype=torch.complex64, device=state.device),
        )
    else:
        raise ValueError(f"Unknown Pauli: {pauli}")
    out[:, mapped] = state * phase
    return (state.conj() * out).sum(dim=1).real


def result_row(
    name: str,
    model_name: str,
    feature_set: str,
    *,
    n_qubits: int,
    layers: int,
    val_f1: float,
    threshold: float,
    metrics: dict[str, float],
    train_samples: int,
    test_samples: int,
    extra: dict | None = None,
) -> dict:
    row = {
        "name": name,
        "model_name": model_name,
        "feature_set": feature_set,
        "n_qubits": int(n_qubits),
        "layers": int(layers),
        "train_samples": int(train_samples),
        "test_samples": int(test_samples),
        "val_f1": float(val_f1),
        "threshold": float(threshold),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1": float(metrics["f1"]),
        "roc_auc": float(metrics["roc_auc"]),
        "pr_auc": float(metrics["pr_auc"]),
    }
    if extra:
        row.update(extra)
    return row


def build_final_comparison(output_dir: Path) -> list[dict]:
    rows: list[dict] = []
    day2_path = output_dir / "day2_result_table.csv"
    if day2_path.exists():
        with day2_path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                method = row["Method"]
                rows.append(
                    {
                        "method": method,
                        "input": row["Input"],
                        "precision": float(row["Precision"]),
                        "recall": float(row["Recall"]),
                        "f1": float(row["F1"]),
                        "pr_auc": float(row["PR-AUC"]),
                        "source": "classical_and_day2_baseline",
                    }
                )
    qpp_path = output_dir / "qpp_few_qubit_results.csv"
    if qpp_path.exists():
        wanted = {
            "lambda2_threshold": "Threshold lambda2",
            "logistic_logS_eta": "Logistic logS+eta",
            "qpp_1q_scalar_c4_L2": "QPP QNN 1q scalar c4",
            "qpp_2q_lambda12_L2": "QPP QNN 2q lambda12",
        }
        with qpp_path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row["name"] in wanted:
                    rows.append(
                        {
                            "method": wanted[row["name"]],
                            "input": row["feature_set"],
                            "precision": float(row["test_precision"]),
                            "recall": float(row["test_recall"]),
                            "f1": float(row["test_f1"]),
                            "pr_auc": float(row["test_pr_auc"]),
                            "source": "qpp_full_split",
                        }
                    )
    return rows


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, rows: list[dict]) -> None:
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def save_grouped_curve(
    rows: list[dict],
    path: Path,
    *,
    x_key: str,
    series_key: str,
    y_keys: tuple[str, str],
    title: str,
) -> None:
    series = sorted({str(row[series_key]) for row in rows})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    for ax, y_key in zip(axes, y_keys):
        for label in series:
            data = sorted((row for row in rows if str(row[series_key]) == label), key=lambda item: float(item[x_key]))
            ax.plot([float(row[x_key]) for row in data], [float(row[y_key]) for row in data], marker="o", label=label)
        ax.set_title(y_key.upper())
        ax.set_xlabel(x_key.replace("_", " "))
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("score")
    axes[1].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_bar_chart(rows: list[dict], path: Path, *, x_key: str, y_key: str, title: str) -> None:
    labels = [str(row[x_key]) for row in rows]
    values = [float(row[y_key]) for row in rows]
    fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(labels)), 4))
    ax.bar(np.arange(len(labels)), values, color="#4C78A8")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(y_key.upper())
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_grouped_bars(
    rows: list[dict],
    path: Path,
    *,
    group_key: str,
    series_key: str,
    y_key: str,
    title: str,
) -> None:
    groups = sorted({str(row[group_key]) for row in rows})
    series = sorted({str(row[series_key]) for row in rows})
    width = 0.8 / max(1, len(series))
    x = np.arange(len(groups))
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, label in enumerate(series):
        values = []
        for group in groups:
            found = next(row for row in rows if str(row[group_key]) == group and str(row[series_key]) == label)
            values.append(float(found[y_key]))
        ax.bar(x + (i - (len(series) - 1) / 2) * width, values, width=width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(y_key.upper())
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
