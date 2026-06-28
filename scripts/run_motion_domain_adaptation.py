from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageFilter


ROOT = Path(__file__).resolve().parents[1]
QPP_SRC = ROOT / "qpp_corner_qnn_github_package" / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(QPP_SRC) not in sys.path:
    sys.path.insert(0, str(QPP_SRC))

from qcd_data.baselines import evaluate_points, run_fast, run_harris, run_orb
from qcd_data.features import EXTENDED_STRUCTURE_TENSOR_FEATURE_NAMES, extract_extended_structure_tensor_features
from qpp_corner.classical import fit_logistic
from qpp_corner.metrics import binary_metrics, choose_threshold_by_f1
from qpp_corner.normalizer import FeatureNormalizer
from qpp_corner.qnn_torch import DataReuploadingQNN2
from qpp_corner.train import predict_torch, train_torch_classifier, write_history_csv
from scripts.build_realdata_and_noise_demos import load_noise_qpp_detectors, sliding_patches, write_rows, write_video
from scripts.run_qpp_next_step_experiments import qpp_feature_sets_from_extended
from scripts.run_synthetic_motion_benchmark import FrameRecord, build_dataset, make_comparison_frame


@dataclass
class LearningDetector:
    name: str
    feature_set: str
    normalizer: FeatureNormalizer
    model: object
    kind: str


@dataclass(frozen=True)
class SelectionConfig:
    quantile: float
    nms_radius: float
    max_points: int
    min_threshold: float | None = None


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.data_dir.mkdir(parents=True, exist_ok=True)
    run_dir = args.output_dir / "motion_domain_adaptation_qpp_run"
    run_dir.mkdir(parents=True, exist_ok=True)

    records = build_dataset(args)
    splits = split_records(records, train_ratio=0.70, val_ratio=0.15)
    detectors = load_noise_qpp_detectors()

    train_pack = build_motion_patch_pack(splits["train"], args, detectors["qpp_2q"], augment=True)
    val_pack = build_motion_patch_pack(splits["val"], args, detectors["qpp_2q"], augment=False)
    test_pack = build_motion_patch_pack(splits["test"], args, detectors["qpp_2q"], augment=False)
    save_patch_dataset(args.data_dir / "synthetic_motion_finetune_dataset.npz", train_pack, val_pack, test_pack)

    qpp_ft = train_finetuned_qpp(train_pack, val_pack, args, run_dir)
    logistic_motion = train_motion_logistic(train_pack)

    detector_specs = build_detector_specs(detectors, qpp_ft, logistic_motion)
    tuned_configs = tune_selection_configs(splits["val"], detector_specs, args)
    classical_configs = tune_classical_configs(splits["val"], args)

    test_rows, frame_rows = evaluate_adapted(splits["test"], detector_specs, tuned_configs, classical_configs, args)
    all_video_frames = build_video_frames(records, detector_specs, tuned_configs, classical_configs, args)

    write_rows(args.output_dir / "synthetic_motion_adapted_metrics.csv", test_rows)
    write_rows(args.output_dir / "synthetic_motion_adapted_frame_metrics.csv", frame_rows)
    (args.output_dir / "synthetic_motion_adapted_metrics.json").write_text(
        json.dumps({"metrics": test_rows, "configs": serialize_configs(tuned_configs, classical_configs)}, indent=2),
        encoding="utf-8",
    )
    save_adapted_plot(test_rows, args.output_dir / "synthetic_motion_adapted_metrics.png")
    save_adaptation_report(
        args.output_dir / "synthetic_motion_adaptation_report.md",
        test_rows,
        train_pack,
        val_pack,
        test_pack,
        tuned_configs,
        classical_configs,
        args,
    )

    for sequence, frames in all_video_frames.items():
        if frames:
            write_video(args.output_dir / f"synthetic_motion_{sequence}_adapted_comparison.mp4", frames, fps=args.fps)
            Image.fromarray(frames[len(frames) // 2]).save(args.output_dir / f"synthetic_motion_{sequence}_adapted_preview.png")

    print("Motion/domain adaptation complete.")
    print(args.output_dir / "synthetic_motion_adapted_metrics.csv")
    print(args.output_dir / "synthetic_motion_2d_adapted_comparison.mp4")
    print(args.output_dir / "synthetic_motion_3d_adapted_comparison.mp4")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune QPP QNN on synthetic motion data and tune adaptive threshold/NMS.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--size", type=int, default=192)
    parser.add_argument("--seed", type=int, default=91)
    parser.add_argument("--patch-size", type=int, default=9)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--max-points", type=int, default=60)
    parser.add_argument("--tolerance", type=float, default=5.0)
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--lr", type=float, default=1.5e-3)
    parser.add_argument("--pos-jitters", type=int, default=4)
    parser.add_argument("--random-negatives-per-positive", type=float, default=1.6)
    parser.add_argument("--hard-negatives-per-frame", type=int, default=24)
    parser.add_argument("--augmentations", type=int, default=3)
    return parser.parse_args()


def split_records(records: list[FrameRecord], train_ratio: float, val_ratio: float) -> dict[str, list[FrameRecord]]:
    out = {"train": [], "val": [], "test": []}
    by_sequence: dict[str, list[FrameRecord]] = {}
    for record in records:
        by_sequence.setdefault(record.sequence, []).append(record)
    for sequence_records in by_sequence.values():
        ordered = sorted(sequence_records, key=lambda item: item.frame)
        n = len(ordered)
        train_end = int(round(n * train_ratio))
        val_end = int(round(n * (train_ratio + val_ratio)))
        out["train"].extend(ordered[:train_end])
        out["val"].extend(ordered[train_end:val_end])
        out["test"].extend(ordered[val_end:])
    return out


def build_motion_patch_pack(records: list[FrameRecord], args: argparse.Namespace, hard_detector: dict, *, augment: bool) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(args.seed + (11 if augment else 23))
    centers: list[tuple[float, float]] = []
    patches: list[np.ndarray] = []
    labels: list[int] = []
    image_ids: list[int] = []
    sequences: list[str] = []

    for image_id, record in enumerate(records):
        variants = [(record.image, record.gt)]
        if augment:
            for _ in range(args.augmentations):
                variants.append((augment_image(record.image, rng), record.gt))
        hard_negative_centers = hard_negative_points(record, hard_detector, args)
        for image, gt in variants:
            local_pos = sample_positive_centers(gt, args, rng, image.shape)
            local_neg = sample_random_negative_centers(gt, int(math.ceil(len(local_pos) * args.random_negatives_per_positive)), args, rng, image.shape)
            local_hard = hard_negative_centers[: args.hard_negatives_per_frame] if augment else hard_negative_centers[: max(4, args.hard_negatives_per_frame // 3)]
            for label, sampled in [(1, local_pos), (0, local_neg), (0, local_hard)]:
                for center in sampled:
                    patch = crop_patch(image, center, args.patch_size)
                    if patch is None:
                        continue
                    centers.append((float(center[0]), float(center[1])))
                    patches.append(patch)
                    labels.append(label)
                    image_ids.append(image_id)
                    sequences.append(record.sequence)

    patch_arr = np.asarray(patches, dtype=np.float32)
    features = extract_extended_structure_tensor_features(patch_arr)
    qpp = qpp_feature_sets_from_extended(features, EXTENDED_STRUCTURE_TENSOR_FEATURE_NAMES)
    return {
        "features_extended": features.astype(np.float32),
        "lambda12": qpp["lambda12"].astype(np.float32),
        "logS_eta": qpp["logS_eta"].astype(np.float32),
        "y": np.asarray(labels, dtype=np.int64),
        "centers": np.asarray(centers, dtype=np.float32),
        "image_ids": np.asarray(image_ids, dtype=np.int32),
        "sequence": np.asarray(sequences),
    }


def augment_image(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = np.asarray(image, dtype=np.float32).copy()
    gain = rng.uniform(0.82, 1.20)
    bias = rng.uniform(-0.04, 0.04)
    out = np.clip(out * gain + bias, 0.0, 1.0)
    if rng.random() < 0.75:
        out = np.clip(out + rng.normal(0.0, rng.uniform(0.006, 0.035), size=out.shape), 0.0, 1.0)
    if rng.random() < 0.45:
        pil = Image.fromarray(np.uint8(out * 255))
        out = np.asarray(pil.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.15, 0.9)))), dtype=np.float32) / 255.0
    if rng.random() < 0.25:
        rank = rng.random(out.shape)
        mask = rank < rng.uniform(0.002, 0.012)
        values = (rng.random(out.shape) > 0.5).astype(np.float32)
        out[mask] = values[mask]
    return out.astype(np.float32)


def sample_positive_centers(gt: np.ndarray, args: argparse.Namespace, rng: np.random.Generator, shape: tuple[int, int]) -> list[np.ndarray]:
    centers = []
    for point in np.asarray(gt, dtype=np.float32).reshape(-1, 2):
        centers.append(point.copy())
        for _ in range(args.pos_jitters):
            centers.append(point + rng.normal(0.0, 1.6, size=2).astype(np.float32))
    return [c for c in centers if patch_inside(c, args.patch_size, shape)]


def sample_random_negative_centers(
    gt: np.ndarray,
    count: int,
    args: argparse.Namespace,
    rng: np.random.Generator,
    shape: tuple[int, int],
) -> list[np.ndarray]:
    radius = args.patch_size // 2
    out: list[np.ndarray] = []
    gt_arr = np.asarray(gt, dtype=np.float32).reshape(-1, 2)
    attempts = 0
    while len(out) < count and attempts < count * 80:
        attempts += 1
        point = np.asarray([rng.uniform(radius, shape[1] - radius), rng.uniform(radius, shape[0] - radius)], dtype=np.float32)
        if len(gt_arr) and np.min(np.linalg.norm(gt_arr - point[None, :], axis=1)) < args.tolerance * 2.0:
            continue
        out.append(point)
    return out


def hard_negative_points(record: FrameRecord, detector: dict, args: argparse.Namespace) -> list[np.ndarray]:
    centers, patches = sliding_patches(record.image, args.patch_size, args.stride)
    if len(patches) == 0:
        return []
    features = extract_extended_structure_tensor_features(patches)
    qpp = qpp_feature_sets_from_extended(features, EXTENDED_STRUCTURE_TENSOR_FEATURE_NAMES)
    x = qpp[str(detector["feature_set"])]
    angles = detector["normalizer"].to_angles(detector["normalizer"].transform(x))
    scores = predict_torch(detector["model"], angles, batch_size=512, device="cpu")
    order = np.argsort(scores)[::-1]
    gt = np.asarray(record.gt, dtype=np.float32).reshape(-1, 2)
    out: list[np.ndarray] = []
    for idx in order:
        point = centers[int(idx)]
        if len(gt) and np.min(np.linalg.norm(gt - point[None, :], axis=1)) <= args.tolerance * 1.5:
            continue
        if all(np.linalg.norm(point - other) >= 5.0 for other in out):
            out.append(point.astype(np.float32))
        if len(out) >= args.hard_negatives_per_frame:
            break
    return out


def crop_patch(image: np.ndarray, center: np.ndarray, patch_size: int) -> np.ndarray | None:
    radius = patch_size // 2
    x, y = int(round(float(center[0]))), int(round(float(center[1])))
    if y - radius < 0 or x - radius < 0 or y + radius >= image.shape[0] or x + radius >= image.shape[1]:
        return None
    return image[y - radius : y + radius + 1, x - radius : x + radius + 1].astype(np.float32)


def patch_inside(center: np.ndarray, patch_size: int, shape: tuple[int, int]) -> bool:
    radius = patch_size // 2
    x, y = float(center[0]), float(center[1])
    return radius <= x < shape[1] - radius and radius <= y < shape[0] - radius


def save_patch_dataset(path: Path, train_pack: dict, val_pack: dict, test_pack: dict) -> None:
    np.savez_compressed(
        path,
        X_train=train_pack["lambda12"],
        y_train=train_pack["y"],
        X_val=val_pack["lambda12"],
        y_val=val_pack["y"],
        X_test=test_pack["lambda12"],
        y_test=test_pack["y"],
        feature_names=np.asarray(["lambda1", "lambda2"]),
        train_centers=train_pack["centers"],
        val_centers=val_pack["centers"],
        test_centers=test_pack["centers"],
    )


def train_finetuned_qpp(train_pack: dict, val_pack: dict, args: argparse.Namespace, run_dir: Path) -> LearningDetector:
    normalizer = FeatureNormalizer()
    x_train = normalizer.fit_transform(train_pack["lambda12"], ["lambda1", "lambda2"])
    x_val = normalizer.transform(val_pack["lambda12"])
    angles_train = normalizer.to_angles(x_train)
    angles_val = normalizer.to_angles(x_val)

    model = DataReuploadingQNN2(n_layers=2, encoding="ryrz", entanglement="linear_01", readout="z_z_zz")
    state_path = ROOT / "outputs" / "qpp_2q_lambda12_L2_run" / "best_model.pt"
    if state_path.exists():
        model.load_state_dict(torch.load(state_path, map_location="cpu"))
    model, history = train_torch_classifier(
        model,
        angles_train,
        train_pack["y"],
        angles_val,
        val_pack["y"],
        out_dir=run_dir,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=8,
        seed=args.seed,
        device="cpu",
    )
    write_history_csv(run_dir / "history.csv", history)
    normalizer.save_json(run_dir / "normalizer.json")
    val_scores = predict_torch(model, angles_val, batch_size=512, device="cpu")
    threshold, val_f1 = choose_threshold_by_f1(val_pack["y"], val_scores)
    (run_dir / "patch_val_metrics.json").write_text(
        json.dumps({"threshold": threshold, "val_f1": val_f1, **binary_metrics(val_pack["y"], val_scores, threshold)}, indent=2),
        encoding="utf-8",
    )
    save_training_curve(history, run_dir / "training_curve.png")
    return LearningDetector("QPP QNN2 fine-tuned", "lambda12", normalizer, model, "qnn")


def train_motion_logistic(train_pack: dict) -> LearningDetector:
    normalizer = FeatureNormalizer()
    x_train = normalizer.fit_transform(train_pack["lambda12"], ["lambda1", "lambda2"])
    model = fit_logistic(x_train, train_pack["y"], seed=123)
    return LearningDetector("Logistic motion-tuned", "lambda12", normalizer, model, "logistic")


def build_detector_specs(detectors: dict, qpp_ft: LearningDetector, logistic_motion: LearningDetector) -> dict[str, LearningDetector]:
    return {
        "Logistic original": LearningDetector(
            "Logistic original",
            str(detectors["logistic"]["feature_set"]),
            detectors["logistic"]["normalizer"],
            detectors["logistic"]["model"],
            "logistic",
        ),
        "QPP QNN2 original": LearningDetector(
            "QPP QNN2 original",
            str(detectors["qpp_2q"]["feature_set"]),
            detectors["qpp_2q"]["normalizer"],
            detectors["qpp_2q"]["model"],
            "qnn",
        ),
        logistic_motion.name: logistic_motion,
        qpp_ft.name: qpp_ft,
    }


def tune_selection_configs(records: list[FrameRecord], detector_specs: dict[str, LearningDetector], args: argparse.Namespace) -> dict[str, SelectionConfig]:
    configs: dict[str, SelectionConfig] = {}
    for name, detector in detector_specs.items():
        scored = [(record, *score_learning_frame(record.image, detector, args)) for record in records]
        best = (SelectionConfig(0.95, 8.0, 24, None), -1.0)
        for quantile in [0.88, 0.90, 0.93, 0.95, 0.97, 0.985]:
            for radius in [6.0, 8.0, 10.0, 12.0, 14.0]:
                for max_points in [10, 14, 18, 24, 32, 44]:
                    config = SelectionConfig(quantile=quantile, nms_radius=radius, max_points=max_points, min_threshold=None)
                    f1 = aggregate_learning_f1(scored, config, args.tolerance)
                    if f1 > best[1]:
                        best = (config, f1)
        configs[name] = best[0]
    return configs


def tune_classical_configs(records: list[FrameRecord], args: argparse.Namespace) -> dict[str, dict]:
    grids = {
        "Harris tuned": [
            {"threshold_rel": t, "min_distance": d, "max_points": m}
            for t in [0.004, 0.008, 0.012, 0.02, 0.035]
            for d in [4.0, 6.0, 8.0, 10.0, 12.0]
            for m in [12, 18, 24, 32, 44]
        ],
        "FAST tuned": [
            {"threshold": t, "min_distance": d, "max_points": m}
            for t in [12, 18, 24, 32, 44]
            for d in [4.0, 6.0, 8.0, 10.0, 12.0]
            for m in [12, 18, 24, 32, 44]
        ],
        "ORB tuned": [
            {"fast_threshold": t, "min_distance": d, "max_points": m, "nfeatures": max(80, m * 3)}
            for t in [5, 10, 16, 24]
            for d in [4.0, 6.0, 8.0, 10.0, 12.0]
            for m in [12, 18, 24, 32, 44]
        ],
    }
    best: dict[str, dict] = {}
    for method, candidates in grids.items():
        best_config = candidates[0]
        best_f1 = -1.0
        for config in candidates:
            totals = {"tp": 0, "fp": 0, "fn": 0}
            for record in records:
                points = detect_classical(method, record.image, config)
                metrics = evaluate_points(points, record.gt, tolerance=args.tolerance)
                totals["tp"] += metrics.true_positives
                totals["fp"] += metrics.false_positives
                totals["fn"] += metrics.false_negatives
            f1 = aggregate_counts_f1(totals["tp"], totals["fp"], totals["fn"])
            if f1 > best_f1:
                best_f1 = f1
                best_config = dict(config)
        best[method] = best_config
    return best


def score_learning_frame(image: np.ndarray, detector: LearningDetector, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    centers, patches = sliding_patches(image, args.patch_size, args.stride)
    if len(patches) == 0:
        return centers, np.asarray([], dtype=np.float64)
    features = extract_extended_structure_tensor_features(patches)
    qpp = qpp_feature_sets_from_extended(features, EXTENDED_STRUCTURE_TENSOR_FEATURE_NAMES)
    x = qpp[detector.feature_set]
    if detector.kind == "qnn":
        angles = detector.normalizer.to_angles(detector.normalizer.transform(x))
        scores = predict_torch(detector.model, angles, batch_size=512, device="cpu")
    else:
        normed = detector.normalizer.transform(x)
        scores = detector.model.predict_scores(normed)
    return centers, np.asarray(scores, dtype=np.float64)


def aggregate_learning_f1(scored_records: list[tuple[FrameRecord, np.ndarray, np.ndarray]], config: SelectionConfig, tolerance: float) -> float:
    tp = fp = fn = 0
    for record, centers, scores in scored_records:
        points = select_adaptive_points(centers, scores, config)
        metrics = evaluate_points(points, record.gt, tolerance=tolerance)
        tp += metrics.true_positives
        fp += metrics.false_positives
        fn += metrics.false_negatives
    return aggregate_counts_f1(tp, fp, fn)


def select_adaptive_points(centers: np.ndarray, scores: np.ndarray, config: SelectionConfig) -> np.ndarray:
    centers = np.asarray(centers, dtype=np.float32).reshape(-1, 2)
    scores = np.asarray(scores, dtype=np.float64)
    if len(scores) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    threshold = float(np.quantile(scores, config.quantile))
    if config.min_threshold is not None:
        threshold = max(threshold, float(config.min_threshold))
    candidates = np.flatnonzero(scores >= threshold)
    if len(candidates) == 0:
        candidates = np.asarray([int(np.argmax(scores))])
    order = candidates[np.argsort(scores[candidates])[::-1]]
    selected: list[np.ndarray] = []
    for idx in order:
        point = centers[int(idx)]
        if all(np.linalg.norm(point - other) >= config.nms_radius for other in selected):
            selected.append(point)
        if len(selected) >= config.max_points:
            break
    if not selected:
        return np.zeros((0, 2), dtype=np.float32)
    return np.vstack(selected).astype(np.float32)


def detect_classical(method: str, image: np.ndarray, config: dict) -> np.ndarray:
    if method.startswith("Harris"):
        return run_harris(
            image,
            threshold_rel=float(config["threshold_rel"]),
            max_points=int(config["max_points"]),
            min_distance=float(config["min_distance"]),
        )
    if method.startswith("FAST"):
        return run_fast(
            image,
            threshold=int(config["threshold"]),
            max_points=int(config["max_points"]),
            min_distance=float(config["min_distance"]),
        )
    if method.startswith("ORB"):
        return run_orb(
            image,
            nfeatures=int(config["nfeatures"]),
            fast_threshold=int(config["fast_threshold"]),
            max_points=int(config["max_points"]),
            min_distance=float(config["min_distance"]),
        )
    raise ValueError(method)


def evaluate_adapted(
    records: list[FrameRecord],
    detector_specs: dict[str, LearningDetector],
    tuned_configs: dict[str, SelectionConfig],
    classical_configs: dict[str, dict],
    args: argparse.Namespace,
) -> tuple[list[dict], list[dict]]:
    frame_rows: list[dict] = []
    totals: dict[tuple[str, str], dict[str, float]] = {}
    methods = list(classical_configs.keys()) + list(detector_specs.keys())
    for record in records:
        detections: dict[str, np.ndarray] = {}
        for method, config in classical_configs.items():
            detections[method] = detect_classical(method, record.image, config)
        for method, detector in detector_specs.items():
            centers, scores = score_learning_frame(record.image, detector, args)
            detections[method] = select_adaptive_points(centers, scores, tuned_configs[method])
        for method in methods:
            metrics = evaluate_points(detections[method], record.gt, tolerance=args.tolerance)
            key = (record.sequence, method)
            item = totals.setdefault(key, {"tp": 0.0, "fp": 0.0, "fn": 0.0, "frames": 0.0, "points": 0.0, "gt": 0.0})
            item["tp"] += metrics.true_positives
            item["fp"] += metrics.false_positives
            item["fn"] += metrics.false_negatives
            item["frames"] += 1
            item["points"] += len(detections[method])
            item["gt"] += len(record.gt)
            frame_rows.append(
                {
                    "split": "test",
                    "sequence": record.sequence,
                    "frame": record.frame,
                    "method": method,
                    "gt_points": len(record.gt),
                    "detected_points": len(detections[method]),
                    "tp": metrics.true_positives,
                    "fp": metrics.false_positives,
                    "fn": metrics.false_negatives,
                    "precision": metrics.precision,
                    "recall": metrics.recall,
                    "f1": metrics.f1,
                }
            )
    rows = []
    for (sequence, method), item in sorted(totals.items()):
        precision = item["tp"] / max(1.0, item["tp"] + item["fp"])
        recall = item["tp"] / max(1.0, item["tp"] + item["fn"])
        f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
        rows.append(
            {
                "split": "test",
                "sequence": sequence,
                "method": method,
                "frames": int(item["frames"]),
                "gt_points": int(item["gt"]),
                "mean_detected_points": item["points"] / max(1.0, item["frames"]),
                "tp": int(item["tp"]),
                "fp": int(item["fp"]),
                "fn": int(item["fn"]),
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return rows, frame_rows


def build_video_frames(
    records: list[FrameRecord],
    detector_specs: dict[str, LearningDetector],
    tuned_configs: dict[str, SelectionConfig],
    classical_configs: dict[str, dict],
    args: argparse.Namespace,
) -> dict[str, list[np.ndarray]]:
    frames: dict[str, list[np.ndarray]] = {"2d": [], "3d": []}
    for record in records:
        detections = {
            "Harris tuned": detect_classical("Harris tuned", record.image, classical_configs["Harris tuned"]),
            "FAST tuned": detect_classical("FAST tuned", record.image, classical_configs["FAST tuned"]),
        }
        for method in ["Logistic motion-tuned", "QPP QNN2 original", "QPP QNN2 fine-tuned"]:
            centers, scores = score_learning_frame(record.image, detector_specs[method], args)
            detections[method] = select_adaptive_points(centers, scores, tuned_configs[method])
        frames[record.sequence].append(make_comparison_frame(record, detections))
    return frames


def aggregate_counts_f1(tp: int | float, fp: int | float, fn: int | float) -> float:
    precision = tp / max(1.0, tp + fp)
    recall = tp / max(1.0, tp + fn)
    return 2.0 * precision * recall / max(1e-12, precision + recall)


def serialize_configs(tuned_configs: dict[str, SelectionConfig], classical_configs: dict[str, dict]) -> dict:
    return {
        "learning": {name: config.__dict__ for name, config in tuned_configs.items()},
        "classical": classical_configs,
    }


def save_training_curve(history: list[dict[str, float]], path: Path) -> None:
    if not history:
        return
    fig, ax1 = plt.subplots(figsize=(6, 4))
    epochs = [row["epoch"] for row in history]
    ax1.plot(epochs, [row["loss"] for row in history], label="loss", color="#ef4444")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax2 = ax1.twinx()
    ax2.plot(epochs, [row["val_pr_auc"] for row in history], label="val PR-AUC", color="#2563eb")
    ax2.plot(epochs, [row["val_f1_at_0_5"] for row in history], label="val F1@0.5", color="#16a34a")
    ax2.set_ylabel("Validation")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_adapted_plot(rows: list[dict], path: Path) -> None:
    methods = ["Harris tuned", "FAST tuned", "ORB tuned", "Logistic motion-tuned", "QPP QNN2 original", "QPP QNN2 fine-tuned"]
    sequences = ["2d", "3d"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, sequence in zip(axes, sequences):
        values = []
        labels = []
        for method in methods:
            row = next((item for item in rows if item["sequence"] == sequence and item["method"] == method), None)
            if row is not None:
                values.append(float(row["f1"]))
                labels.append(method.replace(" tuned", "").replace("QPP QNN2 ", "QNN2 "))
        ax.bar(labels, values, color="#3b82f6")
        ax.set_ylim(0.0, 1.0)
        ax.set_title(f"{sequence.upper()} adapted test F1")
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("F1")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_adaptation_report(
    path: Path,
    rows: list[dict],
    train_pack: dict,
    val_pack: dict,
    test_pack: dict,
    tuned_configs: dict[str, SelectionConfig],
    classical_configs: dict[str, dict],
    args: argparse.Namespace,
) -> None:
    lines = [
        "# Motion Domain Adaptation Results",
        "",
        "This run completes motion/domain-randomized fine-tuning, adaptive per-frame thresholding, and stronger NMS for the synthetic 2D/3D motion benchmark.",
        "",
        "## What Changed",
        "",
        "- Fine-tuning data uses positives around geometric GT points, random background negatives, and hard negatives from original QPP QNN false detections.",
        "- Domain randomization adds brightness/contrast jitter, Gaussian noise, mild blur, and occasional salt-and-pepper corruption without changing GT geometry.",
        "- Adaptive threshold chooses a per-frame score quantile on the validation split.",
        "- Stronger NMS tunes radius and max points on the validation split.",
        "",
        "## Patch Dataset",
        "",
        f"- Train samples: {len(train_pack['y'])}, positives: {int(np.sum(train_pack['y']))}",
        f"- Val samples: {len(val_pack['y'])}, positives: {int(np.sum(val_pack['y']))}",
        f"- Test patch samples: {len(test_pack['y'])}, positives: {int(np.sum(test_pack['y']))}",
        f"- Frame split: first 70% train, next 15% validation, final 15% test for each of 2D/3D.",
        "",
        "## Tuned Configs",
        "",
        f"- Learning detectors: `{json.dumps({k: v.__dict__ for k, v in tuned_configs.items()}, ensure_ascii=False)}`",
        f"- Classical detectors: `{json.dumps(classical_configs, ensure_ascii=False)}`",
        "",
        "## Test Results",
        "",
        "| Sequence | Method | Precision | Recall | F1 | Mean Detected |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['sequence']} | {row['method']} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} | {row['mean_detected_points']:.1f} |"
        )
    lines.extend(
        [
            "",
            "The fine-tuned QNN should be read as a motion-domain adapted detector, not a clean-test replacement. It uses the same 2-qubit QPP architecture but updates parameters on motion-domain patches.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
