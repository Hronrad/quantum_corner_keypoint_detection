from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


@dataclass(frozen=True)
class DetectionMetrics:
    precision: float
    recall: float
    f1: float
    true_positives: int
    false_positives: int
    false_negatives: int


def run_harris(
    image: np.ndarray,
    block_size: int = 2,
    ksize: int = 3,
    k: float = 0.04,
    threshold_rel: float = 0.01,
    max_points: int = 80,
    min_distance: float = 2.0,
) -> np.ndarray:
    """Run an OpenCV Harris detector when available, otherwise a NumPy Harris fallback."""
    image = np.asarray(image, dtype=np.float32)
    if cv2 is not None:
        response = cv2.cornerHarris(image, block_size, ksize, k)
    else:
        response = harris_response_numpy(image, window_size=max(3, block_size + 1), k=k)
    return _response_to_points(response, threshold_rel, max_points, min_distance)


def run_fast(
    image: np.ndarray,
    threshold: int = 20,
    nonmax_suppression: bool = True,
    max_points: int = 80,
    min_distance: float = 2.0,
) -> np.ndarray:
    """Run OpenCV FAST when available, otherwise a small FAST-9 style fallback."""
    image_u8 = np.uint8(np.clip(image, 0.0, 1.0) * 255)
    if cv2 is not None:
        detector = cv2.FastFeatureDetector_create(threshold=threshold, nonmaxSuppression=nonmax_suppression)
        keypoints = detector.detect(image_u8, None)
        scored = [(float(kp.response), float(kp.pt[0]), float(kp.pt[1])) for kp in keypoints]
        return _select_scored_points(scored, max_points, min_distance)
    return fast9_numpy(image_u8, threshold=threshold, max_points=max_points, min_distance=min_distance)


def run_orb(
    image: np.ndarray,
    nfeatures: int = 80,
    fast_threshold: int = 5,
    max_points: int = 80,
    min_distance: float = 2.0,
) -> np.ndarray:
    """Run ORB keypoint detection. Requires OpenCV."""
    if cv2 is None:
        raise ImportError("run_orb requires opencv-python.")
    image_u8 = np.uint8(np.clip(image, 0.0, 1.0) * 255)
    detector = cv2.ORB_create(
        nfeatures=nfeatures,
        edgeThreshold=4,
        patchSize=15,
        fastThreshold=fast_threshold,
    )
    keypoints = detector.detect(image_u8, None)
    scored = [(float(kp.response), float(kp.pt[0]), float(kp.pt[1])) for kp in keypoints]
    return _select_scored_points(scored, max_points, min_distance)


def harris_response_numpy(image: np.ndarray, window_size: int = 3, k: float = 0.04) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    iy, ix = np.gradient(image)
    ix2 = _box_filter(ix * ix, window_size)
    iy2 = _box_filter(iy * iy, window_size)
    ixiy = _box_filter(ix * iy, window_size)
    det = ix2 * iy2 - ixiy * ixiy
    trace = ix2 + iy2
    return det - k * trace * trace


def fast9_numpy(
    image_u8: np.ndarray,
    threshold: int = 20,
    max_points: int = 80,
    min_distance: float = 2.0,
) -> np.ndarray:
    image_u8 = np.asarray(image_u8, dtype=np.uint8)
    height, width = image_u8.shape
    offsets = [
        (0, -3),
        (1, -3),
        (2, -2),
        (3, -1),
        (3, 0),
        (3, 1),
        (2, 2),
        (1, 3),
        (0, 3),
        (-1, 3),
        (-2, 2),
        (-3, 1),
        (-3, 0),
        (-3, -1),
        (-2, -2),
        (-1, -3),
    ]
    scored: list[tuple[float, float, float]] = []
    for y in range(3, height - 3):
        for x in range(3, width - 3):
            center = int(image_u8[y, x])
            circle = np.array([int(image_u8[y + dy, x + dx]) for dx, dy in offsets], dtype=np.int16)
            bright = circle > center + threshold
            dark = circle < center - threshold
            if _has_contiguous_run(bright, 9) or _has_contiguous_run(dark, 9):
                score = float(np.max(np.abs(circle - center)))
                scored.append((score, float(x), float(y)))
    return _select_scored_points(scored, max_points, min_distance)


def evaluate_points(detected_xy: np.ndarray, gt_xy: np.ndarray, tolerance: float = 3.0) -> DetectionMetrics:
    detected_xy = np.asarray(detected_xy, dtype=np.float32).reshape(-1, 2)
    gt_xy = np.asarray(gt_xy, dtype=np.float32).reshape(-1, 2)
    matched_gt: set[int] = set()
    true_positives = 0

    for point in detected_xy:
        if len(gt_xy) == 0:
            break
        distances = np.linalg.norm(gt_xy - point[None, :], axis=1)
        order = np.argsort(distances)
        for gt_index in order:
            if int(gt_index) not in matched_gt and distances[gt_index] <= tolerance:
                matched_gt.add(int(gt_index))
                true_positives += 1
                break

    false_positives = int(len(detected_xy) - true_positives)
    false_negatives = int(len(gt_xy) - true_positives)
    precision = true_positives / max(1, true_positives + false_positives)
    recall = true_positives / max(1, true_positives + false_negatives)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    return DetectionMetrics(precision, recall, f1, true_positives, false_positives, false_negatives)


def save_overlay(
    image: np.ndarray,
    detected_xy: np.ndarray,
    path: Path,
    gt_xy: np.ndarray | None = None,
    title: str = "Detection",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
    if gt_xy is not None and len(gt_xy):
        ax.scatter(gt_xy[:, 0], gt_xy[:, 1], c="lime", s=42, marker="o", facecolors="none", linewidths=1.5, label="GT")
    detected_xy = np.asarray(detected_xy, dtype=np.float32).reshape(-1, 2)
    if len(detected_xy):
        ax.scatter(detected_xy[:, 0], detected_xy[:, 1], c="red", s=30, marker="x", linewidths=1.4, label="Detected")
    ax.set_title(f"{title} ({len(detected_xy)} points)", fontsize=10)
    ax.axis("off")
    if gt_xy is not None:
        ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_overlay_comparison(
    image: np.ndarray,
    gt_xy: np.ndarray,
    harris_xy: np.ndarray,
    fast_xy: np.ndarray,
    orb_xy: np.ndarray,
    mlp_xy: np.ndarray,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    panels = [
        ("GT keypoints", gt_xy, "lime", "o"),
        ("Harris", harris_xy, "red", "x"),
        ("FAST", fast_xy, "red", "x"),
        ("ORB", orb_xy, "red", "x"),
        ("MLP", mlp_xy, "red", "x"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(10, 7))
    axes.flat[-1].axis("off")
    for ax, (title, points, color, marker) in zip(axes.flat, panels):
        ax.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
        points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        if len(points):
            if marker == "o":
                ax.scatter(points[:, 0], points[:, 1], c=color, s=44, marker=marker, facecolors="none", linewidths=1.6)
            else:
                ax.scatter(points[:, 0], points[:, 1], c=color, s=30, marker=marker, linewidths=1.4)
        ax.set_title(f"{title} ({len(points)})", fontsize=10)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _box_filter(values: np.ndarray, size: int) -> np.ndarray:
    size = max(1, int(size))
    pad = size // 2
    padded = np.pad(values, pad, mode="reflect")
    output = np.zeros_like(values, dtype=np.float32)
    for y in range(values.shape[0]):
        for x in range(values.shape[1]):
            output[y, x] = float(np.sum(padded[y : y + size, x : x + size]))
    return output


def _response_to_points(response: np.ndarray, threshold_rel: float, max_points: int, min_distance: float) -> np.ndarray:
    response = np.asarray(response, dtype=np.float32)
    max_response = float(np.max(response))
    if not np.isfinite(max_response) or max_response <= 0:
        return np.zeros((0, 2), dtype=np.float32)
    threshold = threshold_rel * max_response
    scored = []
    height, width = response.shape
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            value = float(response[y, x])
            if value < threshold:
                continue
            local = response[y - 1 : y + 2, x - 1 : x + 2]
            if value >= float(np.max(local)):
                scored.append((value, float(x), float(y)))
    return _select_scored_points(scored, max_points, min_distance)


def _select_scored_points(scored: list[tuple[float, float, float]], max_points: int, min_distance: float) -> np.ndarray:
    points: list[tuple[float, float]] = []
    for _score, x, y in sorted(scored, reverse=True):
        point = np.array([x, y], dtype=np.float32)
        if all(np.linalg.norm(point - np.array(existing, dtype=np.float32)) >= min_distance for existing in points):
            points.append((x, y))
        if len(points) >= max_points:
            break
    if not points:
        return np.zeros((0, 2), dtype=np.float32)
    return np.array(points, dtype=np.float32)


def _has_contiguous_run(flags: np.ndarray, length: int) -> bool:
    doubled = np.concatenate([flags, flags])
    run = 0
    for value in doubled:
        run = run + 1 if bool(value) else 0
        if run >= length:
            return True
    return False
