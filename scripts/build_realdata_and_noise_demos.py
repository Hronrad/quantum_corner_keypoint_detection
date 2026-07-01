from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import urllib.request
import zipfile
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

from qcd_data.baselines import run_fast, run_harris, run_orb
from qcd_data.features import EXTENDED_STRUCTURE_TENSOR_FEATURE_NAMES, extract_extended_structure_tensor_features
from qpp_corner.classical import fit_logistic
from qpp_corner.normalizer import FeatureNormalizer
from qpp_corner.qnn_torch import DataReuploadingQNN1, DataReuploadingQNN2
from qpp_corner.train import predict_torch
from scripts.run_qpp_next_step_experiments import qpp_feature_sets_from_extended


HPATCHES_MONTAGE_URL = "https://raw.githubusercontent.com/hpatches/hpatches-dataset/master/img/images.png"
KITTI_SMALL_ZIP_URL = (
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0001/"
    "2011_09_26_drive_0001_sync.zip"
)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    real_dir = args.data_dir / "real_preview"
    real_dir.mkdir(parents=True, exist_ok=True)

    hpatches_frames = prepare_hpatches_frames(real_dir / "hpatches")
    kitti_frames, kitti_status = prepare_kitti_frames(real_dir / "kitti", args)
    real_rows = []
    for dataset_name, frames in [("HPatches example sequence", hpatches_frames), ("KITTI drive 0001", kitti_frames)]:
        if not frames:
            continue
        rows = run_real_preview(dataset_name, frames, args)
        real_rows.extend(rows)

    write_rows(demo_output_path(args.output_dir, "realdata_preview_metrics.csv"), real_rows)
    write_real_report(demo_output_path(args.output_dir, "realdata_preview_report.md"), real_rows, kitti_status)
    build_noise_demo(args)
    print("Demo build complete.")
    print(demo_output_path(args.output_dir, "realdata_preview_report.md"))
    print(demo_output_path(args.output_dir, "realdata_hpatches_qpp_overlay.mp4"))
    print(demo_output_path(args.output_dir, "dynamic_noise_robustness_demo.mp4"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build real-data preview overlays and dynamic noise robustness demos.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--seed", type=int, default=67)
    parser.add_argument("--patch-size", type=int, default=9)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--max-points", type=int, default=35)
    parser.add_argument("--noise-image-id", type=int, default=30)
    parser.add_argument("--noise-stride", type=int, default=2)
    parser.add_argument("--download-kitti", action="store_true", help="Download the 459MB official KITTI raw mini drive zip.")
    parser.add_argument("--kitti-max-frames", type=int, default=72)
    return parser.parse_args()


def demo_output_path(base: Path, filename: str) -> Path:
    if filename == "realdata_preview_metrics.csv":
        return base / "demos" / "realdata" / "metrics" / filename
    if filename in {"realdata_preview_report.md", "real_dataset_samples_report.md"}:
        return base / "demos" / "realdata" / "reports" / filename
    if filename.startswith("real_dataset_samples"):
        return base / "demos" / "realdata" / "samples" / filename
    if filename.startswith("realdata_hpatches_example_sequence_"):
        return base / "demos" / "realdata" / "frames" / "hpatches" / filename
    if filename.startswith("realdata_kitti_drive_0001_"):
        return base / "demos" / "realdata" / "frames" / "kitti" / filename
    if filename.startswith("realdata_") and filename.endswith((".mp4", ".gif")):
        return base / "demos" / "realdata" / "videos" / filename
    if filename == "dynamic_noise_demo_metrics.csv":
        return base / "demos" / "dynamic_noise" / "metrics" / filename
    if filename.startswith("dynamic_noise_") and filename.endswith((".mp4", ".gif")):
        return base / "demos" / "dynamic_noise" / "videos" / filename
    if filename.startswith("dynamic_noise_") and filename.endswith(".png"):
        return base / "demos" / "dynamic_noise" / "figures" / filename
    return base / filename


def prepare_hpatches_frames(target_dir: Path) -> list[Path]:
    target_dir.mkdir(parents=True, exist_ok=True)
    montage_path = target_dir / "example_sequence_montage.png"
    if not montage_path.exists():
        download_file(HPATCHES_MONTAGE_URL, montage_path)
    image = Image.open(montage_path).convert("RGB")
    width, height = image.size
    frames: list[Path] = []
    for index in range(6):
        left = round(index * width / 6)
        right = round((index + 1) * width / 6)
        crop = image.crop((left, 0, right, height))
        frame_path = target_dir / f"hpatches_example_{index:02d}.png"
        crop.save(frame_path)
        frames.append(frame_path)
    return frames


def prepare_kitti_frames(target_dir: Path, args: argparse.Namespace) -> tuple[list[Path], dict]:
    target_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(target_dir.glob("kitti_*.png"))
    if len(existing) >= args.kitti_max_frames:
        return existing[: args.kitti_max_frames], {"status": "used cached frames", "frames": len(existing)}
    if not args.download_kitti:
        if existing:
            return existing, {
                "status": "used partial cached frames",
                "frames": len(existing),
                "reason": "Use --download-kitti to refresh/extract more frames.",
            }
        return [], {
            "status": "skipped",
            "reason": "Official KITTI mini drive zip is available but about 459MB; rerun with --download-kitti to fetch it.",
            "url": KITTI_SMALL_ZIP_URL,
        }
    zip_path = target_dir / "2011_09_26_drive_0001_sync.zip"
    if not zip_path.exists() or zip_path.stat().st_size < 400_000_000:
        download_file(KITTI_SMALL_ZIP_URL, zip_path)
    extracted = extract_kitti_frames(zip_path, target_dir, args.kitti_max_frames)
    return extracted, {"status": "downloaded and extracted", "frames": len(extracted), "url": KITTI_SMALL_ZIP_URL}


def extract_kitti_frames(zip_path: Path, target_dir: Path, max_frames: int) -> list[Path]:
    frames: list[Path] = []
    with zipfile.ZipFile(zip_path) as zf:
        names = sorted(
            (name for name in zf.namelist() if "/image_02/data/" in name and name.endswith(".png")),
            key=lambda name: int(Path(name).stem),
        )
        for index, name in enumerate(names[:max_frames]):
            with zf.open(name) as handle:
                image = Image.open(handle).convert("RGB")
                out_path = target_dir / f"kitti_{index:02d}.png"
                image.save(out_path)
                frames.append(out_path)
    return frames


def run_real_preview(dataset_name: str, frame_paths: list[Path], args: argparse.Namespace) -> list[dict]:
    if "KITTI" in dataset_name:
        return run_kitti_final_preview(dataset_name, frame_paths, args)
    model, normalizer = load_qpp_model(args)
    rows = []
    overlay_frames = []
    for index, path in enumerate(frame_paths):
        image = load_gray(path, target_width=480)
        harris = run_harris(image, max_points=args.max_points, min_distance=4.0)
        fast = run_fast(image, max_points=args.max_points, min_distance=4.0)
        orb = run_orb(image, max_points=args.max_points, min_distance=4.0)
        qpp_points, qpp_scores = qpp_points_for_image(image, model, normalizer, args)
        overlay = make_real_overlay_frame(image, dataset_name, index, harris, fast, orb, qpp_points)
        overlay_frames.append(overlay)
        frame_out = demo_output_path(args.output_dir, f"realdata_{slug(dataset_name)}_{index:02d}_overlay.png")
        Image.fromarray(overlay).save(frame_out)
        rows.append(
            {
                "dataset": dataset_name,
                "frame": index,
                "harris_points": len(harris),
                "fast_points": len(fast),
                "orb_points": len(orb),
                "qpp_points": len(qpp_points),
                "qpp_mean_score": float(np.mean(qpp_scores)) if len(qpp_scores) else 0.0,
                "qpp_top_score": float(np.max(qpp_scores)) if len(qpp_scores) else 0.0,
            }
        )
    if overlay_frames:
        video_name = "realdata_hpatches_qpp_overlay.mp4" if "HPatches" in dataset_name else "realdata_kitti_qpp_overlay.mp4"
        write_video(demo_output_path(args.output_dir, video_name), overlay_frames, fps=2)
    return rows


def run_kitti_final_preview(dataset_name: str, frame_paths: list[Path], args: argparse.Namespace) -> list[dict]:
    detectors = load_noise_qpp_detectors()
    rows = []
    overlay_frames = []
    for index, path in enumerate(frame_paths):
        image = load_gray(path, target_width=640)
        logistic_points, logistic_scores = logistic_threshold_points_for_image(image, detectors["logistic"], args)
        harris = run_harris(image, max_points=args.max_points, min_distance=4.0)
        fast = run_fast(image, max_points=args.max_points, min_distance=4.0)
        qpp_points, qpp_scores = qpp_threshold_points_for_image(image, detectors["qpp_2q"], args)
        overlay = make_kitti_final_frame(
            image=image,
            dataset_name=dataset_name,
            index=index,
            logistic=logistic_points,
            harris=harris,
            fast=fast,
            qpp_2q=qpp_points,
            gt_xy=None,
        )
        overlay_frames.append(overlay)
        frame_out = demo_output_path(args.output_dir, f"realdata_{slug(dataset_name)}_{index:02d}_overlay.png")
        Image.fromarray(overlay).save(frame_out)
        rows.append(
            {
                "dataset": dataset_name,
                "frame": index,
                "logistic_points": len(logistic_points),
                "harris_points": len(harris),
                "fast_points": len(fast),
                "qpp_2q_points": len(qpp_points),
                "logistic_mean_score": float(np.mean(logistic_scores)) if len(logistic_scores) else 0.0,
                "logistic_top_score": float(np.max(logistic_scores)) if len(logistic_scores) else 0.0,
                "qpp_2q_mean_score": float(np.mean(qpp_scores)) if len(qpp_scores) else 0.0,
                "qpp_2q_top_score": float(np.max(qpp_scores)) if len(qpp_scores) else 0.0,
            }
        )
    if overlay_frames:
        write_video(
            demo_output_path(args.output_dir, "realdata_kitti_qpp_overlay.mp4"),
            overlay_frames,
            fps=kitti_demo_fps(len(overlay_frames)),
        )
    return rows


def kitti_demo_fps(frame_count: int, target_seconds: float = 10.0) -> float:
    return max(1.0, float(frame_count) / float(target_seconds))


def load_qpp_model(args: argparse.Namespace):
    data = np.load(ROOT / "data" / "feature_dataset_extended.npz", allow_pickle=True)
    feature_names = [str(name) for name in data["feature_names"].tolist()]
    qpp_train = qpp_feature_sets_from_extended(data["X_train"].astype(np.float32), feature_names)
    normalizer = FeatureNormalizer()
    normalizer.fit_transform(qpp_train["lambda12"], ["lambda1", "lambda2"])
    model = DataReuploadingQNN2(n_layers=2, encoding="ryrz", entanglement="linear_01", readout="z_z_zz")
    state_path = ROOT / "outputs" / "runs" / "qpp" / "few_qubit" / "qpp_2q_lambda12_L2_run" / "best_model.pt"
    state = torch.load(state_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model, normalizer


def load_noise_qpp_detectors() -> dict[str, dict]:
    data = np.load(ROOT / "data" / "feature_dataset_extended.npz", allow_pickle=True)
    feature_names = [str(name) for name in data["feature_names"].tolist()]
    qpp_train = qpp_feature_sets_from_extended(data["X_train"].astype(np.float32), feature_names)
    thresholds = read_qpp_thresholds()

    normalizer_1q = FeatureNormalizer()
    normalizer_1q.fit_transform(qpp_train["scalar_c4"], ["logS_plus_4_eta"])
    model_1q = DataReuploadingQNN1(n_layers=2, encoding="ryrz")
    model_1q.load_state_dict(
        torch.load(
            ROOT / "outputs" / "runs" / "qpp" / "few_qubit" / "qpp_1q_scalar_c4_L2_run" / "best_model.pt",
            map_location="cpu",
        )
    )
    model_1q.eval()

    normalizer_2q = FeatureNormalizer()
    normalizer_2q.fit_transform(qpp_train["lambda12"], ["lambda1", "lambda2"])
    model_2q = DataReuploadingQNN2(n_layers=2, encoding="ryrz", entanglement="linear_01", readout="z_z_zz")
    model_2q.load_state_dict(
        torch.load(
            ROOT / "outputs" / "runs" / "qpp" / "few_qubit" / "qpp_2q_lambda12_L2_run" / "best_model.pt",
            map_location="cpu",
        )
    )
    model_2q.eval()

    return {
        "logistic": {
            "label": "Logistic",
            "model": fit_logistic(normalize_feature_pack(qpp_train["logS_eta"], ["logS", "eta"])[0], data["y_train"].astype(int), seed=47),
            "normalizer": normalize_feature_pack(qpp_train["logS_eta"], ["logS", "eta"])[1],
            "feature_set": "logS_eta",
            "threshold": thresholds["logistic_logS_eta"],
        },
        "qpp_1q": {
            "label": "1-qubit QNN",
            "model": model_1q,
            "normalizer": normalizer_1q,
            "feature_set": "scalar_c4",
            "threshold": thresholds["qpp_1q_scalar_c4_L2"],
        },
        "qpp_2q": {
            "label": "2-qubit QNN",
            "model": model_2q,
            "normalizer": normalizer_2q,
            "feature_set": "lambda12",
            "threshold": thresholds["qpp_2q_lambda12_L2"],
        },
    }


def read_qpp_thresholds() -> dict[str, float]:
    path = ROOT / "outputs" / "qpp" / "few_qubit" / "qpp_few_qubit_results.csv"
    thresholds: dict[str, float] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["name"] in {"logistic_logS_eta", "qpp_1q_scalar_c4_L2", "qpp_2q_lambda12_L2"}:
                thresholds[row["name"]] = float(row["threshold"])
    missing = {"logistic_logS_eta", "qpp_1q_scalar_c4_L2", "qpp_2q_lambda12_L2"} - set(thresholds)
    if missing:
        raise RuntimeError(f"Missing QPP thresholds in {path}: {sorted(missing)}")
    return thresholds


def normalize_feature_pack(x: np.ndarray, feature_names: list[str]) -> tuple[np.ndarray, FeatureNormalizer]:
    normalizer = FeatureNormalizer()
    return normalizer.fit_transform(x, feature_names), normalizer


def qpp_points_for_image(image: np.ndarray, model, normalizer: FeatureNormalizer, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    centers, patches = sliding_patches(image, args.patch_size, args.stride)
    if len(patches) == 0:
        return np.zeros((0, 2), dtype=np.float32), np.asarray([], dtype=np.float32)
    features = extract_extended_structure_tensor_features(patches)
    qpp = qpp_feature_sets_from_extended(features, EXTENDED_STRUCTURE_TENSOR_FEATURE_NAMES)["lambda12"]
    angles = normalizer.to_angles(normalizer.transform(qpp))
    scores = predict_torch(model, angles, batch_size=256, device="cpu")
    points = top_points(centers, scores, args.max_points, min_distance=9.0)
    return points, scores


def sliding_patches(image: np.ndarray, patch_size: int, stride: int) -> tuple[np.ndarray, np.ndarray]:
    radius = patch_size // 2
    centers = []
    patches = []
    for y in range(radius, image.shape[0] - radius, stride):
        for x in range(radius, image.shape[1] - radius, stride):
            patch = image[y - radius : y + radius + 1, x - radius : x + radius + 1]
            if patch.shape == (patch_size, patch_size):
                centers.append((float(x), float(y)))
                patches.append(patch)
    if not patches:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, patch_size, patch_size), dtype=np.float32)
    return np.asarray(centers, dtype=np.float32), np.asarray(patches, dtype=np.float32)


def top_points(centers: np.ndarray, scores: np.ndarray, max_points: int, min_distance: float) -> np.ndarray:
    order = np.argsort(np.asarray(scores))[::-1]
    selected: list[np.ndarray] = []
    for idx in order:
        point = centers[int(idx)]
        if all(np.linalg.norm(point - other) >= min_distance for other in selected):
            selected.append(point)
        if len(selected) >= max_points:
            break
    if not selected:
        return np.zeros((0, 2), dtype=np.float32)
    return np.vstack(selected).astype(np.float32)


def threshold_points(
    centers: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    max_points: int,
    min_distance: float,
) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    candidates = np.flatnonzero(scores >= float(threshold))
    if len(candidates) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    order = candidates[np.argsort(scores[candidates])[::-1]]
    selected: list[np.ndarray] = []
    for idx in order:
        point = centers[int(idx)]
        if all(np.linalg.norm(point - other) >= min_distance for other in selected):
            selected.append(point)
        if len(selected) >= max_points:
            break
    if not selected:
        return np.zeros((0, 2), dtype=np.float32)
    return np.vstack(selected).astype(np.float32)


def make_real_overlay_frame(
    image: np.ndarray,
    dataset_name: str,
    index: int,
    harris: np.ndarray,
    fast: np.ndarray,
    orb: np.ndarray,
    qpp: np.ndarray,
) -> np.ndarray:
    panels = [("Harris", harris), ("FAST", fast), ("ORB", orb), ("QPP QNN", qpp)]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))
    for ax, (title, pts) in zip(axes, panels):
        ax.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
        pts = np.asarray(pts).reshape(-1, 2)
        if len(pts):
            ax.scatter(pts[:, 0], pts[:, 1], s=16, c="#ff2d2d", marker="x", linewidths=1.0)
        ax.set_title(f"{title} ({len(pts)})", fontsize=9)
        ax.axis("off")
    fig.suptitle(f"{dataset_name} frame {index}: real-data keypoint preview", fontsize=11)
    fig.tight_layout()
    return fig_to_rgb(fig)


def make_kitti_final_frame(
    *,
    image: np.ndarray,
    dataset_name: str,
    index: int,
    logistic: np.ndarray,
    harris: np.ndarray,
    fast: np.ndarray,
    qpp_2q: np.ndarray,
    gt_xy: np.ndarray | None,
) -> np.ndarray:
    panels = [
        ("Logistic", logistic),
        ("Harris", harris),
        ("FAST", fast),
        ("2-qubit QNN", qpp_2q),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.1))
    gt = None if gt_xy is None else np.asarray(gt_xy, dtype=np.float32).reshape(-1, 2)
    for ax, (title, pts) in zip(axes.flat, panels):
        ax.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
        if gt is not None and len(gt):
            ax.scatter(gt[:, 0], gt[:, 1], s=18, c="lime", marker="o", linewidths=0.0)
        pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
        if len(pts):
            ax.scatter(pts[:, 0], pts[:, 1], s=18, c="#ff2d2d", marker="x", linewidths=1.0)
        ax.set_title(f"{title} ({len(pts)})", fontsize=10)
        ax.axis("off")
    fig.suptitle(f"{dataset_name}: final keypoint demo, frame {index:02d}", fontsize=12)
    fig.tight_layout()
    return fig_to_rgb(fig)


def build_noise_demo(args: argparse.Namespace) -> None:
    data = np.load(ROOT / "data" / "feature_dataset_extended.npz", allow_pickle=True)
    qpp_detectors = load_noise_qpp_detectors()
    image_id = int(args.noise_image_id)
    test_ids = {int(item) for item in data["split_test_image_ids"].astype(int)}
    if image_id not in test_ids:
        image_id = int(data["split_test_image_ids"][0])
    base_image = data["images"][image_id].astype(np.float32)
    keypoint = data["keypoints"][image_id].astype(np.float32)
    rng = np.random.default_rng(args.seed)
    gaussian_field = rng.normal(0.0, 1.0, size=base_image.shape).astype(np.float32)
    saltpepper_rank = rng.random(base_image.shape)
    saltpepper_value = rng.random(base_image.shape) > 0.5
    frames = []
    rows = []
    schedule = [("clean", "none", 0.0)] * 5 + [
        ("gaussian", "gaussian", float(v)) for v in np.linspace(0.005, 0.08, 16)
    ]
    for noise_name, noise_type, value in schedule:
        image = gradual_noise_frame(
            base_image,
            noise_type,
            value,
            gaussian_field=gaussian_field,
            saltpepper_rank=saltpepper_rank,
            saltpepper_value=saltpepper_value,
        )
        fast = run_fast(image, max_points=args.max_points, min_distance=4.0)
        logistic_points, logistic_scores = logistic_threshold_points_for_image(image, qpp_detectors["logistic"], args)
        qpp_1q_points, qpp_1q_scores = qpp_threshold_points_for_image(image, qpp_detectors["qpp_1q"], args)
        qpp_2q_points, qpp_2q_scores = qpp_threshold_points_for_image(image, qpp_detectors["qpp_2q"], args)
        frame = make_noise_frame(
            image,
            keypoint,
            logistic_points,
            fast,
            qpp_1q_points,
            qpp_2q_points,
            noise_name,
            value,
            image_id,
        )
        frames.append(frame)
        rows.append(
            {
                "frame": len(rows),
                "source_image": image_id,
                "noise": noise_name,
                "value": float(value),
                "logistic_points": len(logistic_points),
                "fast_points": len(fast),
                "qpp_1q_points": len(qpp_1q_points),
                "qpp_2q_points": len(qpp_2q_points),
                "logistic_threshold": float(qpp_detectors["logistic"]["threshold"]),
                "qpp_1q_threshold": float(qpp_detectors["qpp_1q"]["threshold"]),
                "qpp_2q_threshold": float(qpp_detectors["qpp_2q"]["threshold"]),
                "logistic_mean_score": float(np.mean(logistic_scores)) if len(logistic_scores) else 0.0,
                "logistic_top_score": float(np.max(logistic_scores)) if len(logistic_scores) else 0.0,
                "qpp_1q_mean_score": float(np.mean(qpp_1q_scores)) if len(qpp_1q_scores) else 0.0,
                "qpp_1q_top_score": float(np.max(qpp_1q_scores)) if len(qpp_1q_scores) else 0.0,
                "qpp_2q_mean_score": float(np.mean(qpp_2q_scores)) if len(qpp_2q_scores) else 0.0,
                "qpp_2q_top_score": float(np.max(qpp_2q_scores)) if len(qpp_2q_scores) else 0.0,
                "logistic_nearest_gt": nearest_distance(logistic_points, keypoint),
                "qpp_1q_nearest_gt": nearest_distance(qpp_1q_points, keypoint),
                "qpp_2q_nearest_gt": nearest_distance(qpp_2q_points, keypoint),
            }
        )
    write_rows(demo_output_path(args.output_dir, "dynamic_noise_demo_metrics.csv"), rows)
    write_video(demo_output_path(args.output_dir, "dynamic_noise_robustness_demo.mp4"), frames, fps=3)
    if frames:
        Image.fromarray(frames[min(4, len(frames) - 1)]).save(
            demo_output_path(args.output_dir, "dynamic_noise_robustness_demo_preview.png")
        )


def gradual_noise_frame(
    base_image: np.ndarray,
    noise_type: str,
    value: float,
    *,
    gaussian_field: np.ndarray,
    saltpepper_rank: np.ndarray,
    saltpepper_value: np.ndarray,
) -> np.ndarray:
    if noise_type == "none":
        return base_image.copy()
    if noise_type == "gaussian":
        return np.clip(base_image + gaussian_field * float(value), 0.0, 1.0).astype(np.float32)
    if noise_type == "blur":
        pil = Image.fromarray(np.uint8(np.clip(base_image, 0.0, 1.0) * 255))
        return (np.asarray(pil.filter(ImageFilter.GaussianBlur(radius=float(value))), dtype=np.float32) / 255.0).astype(np.float32)
    if noise_type == "saltpepper":
        out = base_image.copy()
        mask = saltpepper_rank < float(value)
        out[mask] = saltpepper_value[mask].astype(np.float32)
        return out.astype(np.float32)
    raise ValueError(noise_type)


def make_noise_frame(
    image: np.ndarray,
    gt: np.ndarray,
    logistic: np.ndarray,
    fast: np.ndarray,
    qpp_1q: np.ndarray,
    qpp_2q: np.ndarray,
    noise_name: str,
    value: float,
    image_index: int,
) -> np.ndarray:
    panels = [
        ("Logistic", logistic, "#f97316", "x"),
        ("FAST", fast, "#ff2d2d", "x"),
        ("1-qubit QNN", qpp_1q, "#00a6ff", "x"),
        ("2-qubit QNN", qpp_2q, "#a855f7", "x"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(12.8, 3.2))
    for ax, (title, pts, color, marker) in zip(axes, panels):
        ax.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
        gt = np.asarray(gt, dtype=np.float32).reshape(-1, 2)
        if len(gt):
            ax.scatter(gt[:, 0], gt[:, 1], s=64, facecolors="none", edgecolors="lime", marker="o", linewidths=2.0)
        pts = np.asarray(pts).reshape(-1, 2)
        if len(pts):
            ax.scatter(pts[:, 0], pts[:, 1], s=28, c=color, marker=marker, linewidths=1.2)
        ax.set_title(f"{title} ({len(pts)})", fontsize=9)
        ax.axis("off")
    fig.suptitle(f"Gaussian noise demo: {noise_name}={value:.3f}, test image {image_index}", fontsize=11)
    fig.tight_layout()
    return fig_to_rgb(fig)


def logistic_threshold_points_for_image(image: np.ndarray, detector: dict, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    centers, patches = sliding_patches(image, args.patch_size, args.noise_stride)
    if len(patches) == 0:
        return np.zeros((0, 2), dtype=np.float32), np.asarray([], dtype=np.float32)
    features = extract_extended_structure_tensor_features(patches)
    qpp = qpp_feature_sets_from_extended(features, EXTENDED_STRUCTURE_TENSOR_FEATURE_NAMES)
    x = detector["normalizer"].transform(qpp[str(detector["feature_set"])])
    scores = detector["model"].predict_scores(x)
    points = threshold_points(
        centers,
        scores,
        float(detector["threshold"]),
        max_points=args.max_points,
        min_distance=4.0,
    )
    return points, scores


def qpp_threshold_points_for_image(image: np.ndarray, detector: dict, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    centers, patches = sliding_patches(image, args.patch_size, args.noise_stride)
    if len(patches) == 0:
        return np.zeros((0, 2), dtype=np.float32), np.asarray([], dtype=np.float32)
    features = extract_extended_structure_tensor_features(patches)
    qpp = qpp_feature_sets_from_extended(features, EXTENDED_STRUCTURE_TENSOR_FEATURE_NAMES)
    x = qpp[str(detector["feature_set"])]
    angles = detector["normalizer"].to_angles(detector["normalizer"].transform(x))
    scores = predict_torch(detector["model"], angles, batch_size=512, device="cpu")
    points = threshold_points(
        centers,
        scores,
        float(detector["threshold"]),
        max_points=args.max_points,
        min_distance=4.0,
    )
    return points, scores


def nearest_distance(points: np.ndarray, gt: np.ndarray) -> float:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    gt = np.asarray(gt, dtype=np.float32).reshape(-1, 2)
    if len(points) == 0 or len(gt) == 0:
        return float("nan")
    distances = np.linalg.norm(points[:, None, :] - gt[None, :, :], axis=2)
    return float(np.min(distances))


def load_gray(path: Path, target_width: int | None = None) -> np.ndarray:
    image = Image.open(path).convert("L")
    if target_width and image.width != target_width:
        height = max(1, round(image.height * target_width / image.width))
        image = image.resize((target_width, height), Image.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return arr


def fig_to_rgb(fig) -> np.ndarray:
    fig.canvas.draw()
    arr = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[..., :3].copy()
    plt.close(fig)
    return arr


def write_video(path: Path, frames: list[np.ndarray], fps: float) -> None:
    if not frames:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("ffmpeg"):
        write_video_ffmpeg(path, frames, fps)
        return
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for frame in frames:
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height))
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def write_video_ffmpeg(path: Path, frames: list[np.ndarray], fps: float) -> None:
    frame_dir = path.parent / f".{path.stem}_frames"
    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True)
    height, width = frames[0].shape[:2]
    try:
        for index, frame in enumerate(frames):
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height))
            Image.fromarray(frame).save(frame_dir / f"frame_{index:05d}.png")
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            str(fps),
            "-i",
            str(frame_dir / "frame_%05d.png"),
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2,format=yuv420p",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-movflags",
            "+faststart",
            str(path),
        ]
        subprocess.run(cmd, check=True)
        gif_path = path.with_suffix(".gif")
        gif_cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-vf",
            f"fps={fps},scale=1280:-1:flags=lanczos",
            str(gif_path),
        ]
        subprocess.run(gif_cmd, check=True)
    finally:
        shutil.rmtree(frame_dir, ignore_errors=True)


def download_file(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as response, path.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_real_report(path: Path, rows: list[dict], kitti_status: dict) -> None:
    datasets = sorted({row["dataset"] for row in rows})
    lines = [
        "# Real-Data Preview Demo",
        "",
        "This preview runs the current QPP few-qubit detector on small real-image samples without ground-truth labels.",
        "Metrics are therefore descriptive counts and QPP score summaries, not precision/recall.",
        "",
        "## Sources",
        "",
        f"- HPatches example sequence montage: `{HPATCHES_MONTAGE_URL}`",
        f"- KITTI official mini drive: `{KITTI_SMALL_ZIP_URL}`",
        f"- KITTI status: `{json.dumps(kitti_status, ensure_ascii=False)}`",
        "",
        "## Outputs",
        "",
        "- `outputs/demos/realdata/videos/realdata_hpatches_qpp_overlay.mp4`",
        "- `outputs/demos/realdata/videos/realdata_hpatches_qpp_overlay.gif`",
        "- `outputs/demos/realdata/videos/realdata_kitti_qpp_overlay.mp4`",
        "- `outputs/demos/realdata/videos/realdata_kitti_qpp_overlay.gif`",
        "- `outputs/demos/dynamic_noise/videos/dynamic_noise_robustness_demo.mp4`",
        "- `outputs/demos/dynamic_noise/videos/dynamic_noise_robustness_demo.gif`",
        "- `outputs/demos/dynamic_noise/figures/dynamic_noise_robustness_demo_preview.png`",
        "",
        "Videos are encoded as H.264/yuv420p for browser and presentation compatibility; GIF files are fallback previews.",
        "The KITTI preview is encoded from 72 frames at 7.2 fps, so it plays in about 10 seconds.",
        "",
    ]
    for dataset in datasets:
        subset = [row for row in rows if row["dataset"] == dataset]
        qpp_key = "qpp_points" if "qpp_points" in subset[0] else "qpp_2q_points"
        qpp_counts = [int(row[qpp_key]) for row in subset]
        harris_counts = [int(row["harris_points"]) for row in subset]
        lines.extend(
            [
                f"## {dataset}",
                "",
                f"- Frames: {len(subset)}",
                f"- {'2-qubit QNN' if qpp_key == 'qpp_2q_points' else 'QPP'} points per frame: min {min(qpp_counts)}, mean {np.mean(qpp_counts):.1f}, max {max(qpp_counts)}",
                f"- Harris points per frame: min {min(harris_counts)}, mean {np.mean(harris_counts):.1f}, max {max(harris_counts)}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def slug(text: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in text).strip("_")


if __name__ == "__main__":
    main()
