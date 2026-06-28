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
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
QPP_SRC = ROOT / "qpp_corner_qnn_github_package" / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(QPP_SRC) not in sys.path:
    sys.path.insert(0, str(QPP_SRC))

from qcd_data.baselines import evaluate_points, run_fast, run_harris, run_orb
from scripts.build_realdata_and_noise_demos import (
    load_noise_qpp_detectors,
    logistic_threshold_points_for_image,
    qpp_threshold_points_for_image,
    write_rows,
    write_video,
)


@dataclass
class FrameRecord:
    sequence: str
    frame: int
    image: np.ndarray
    gt: np.ndarray


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.data_dir.mkdir(parents=True, exist_ok=True)

    records = build_dataset(args)
    save_dataset(records, args.data_dir / "synthetic_motion_sequences.npz")

    detectors = load_noise_qpp_detectors()
    detection_args = argparse.Namespace(
        patch_size=args.patch_size,
        noise_stride=args.stride,
        max_points=args.max_points,
    )
    frame_rows, metrics_rows, preview_frames = evaluate_dataset(records, detectors, detection_args, args)

    write_rows(args.output_dir / "synthetic_motion_frame_metrics.csv", frame_rows)
    write_rows(args.output_dir / "synthetic_motion_metrics.csv", metrics_rows)
    (args.output_dir / "synthetic_motion_metrics.json").write_text(
        json.dumps(metrics_rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    save_metrics_plot(metrics_rows, args.output_dir / "synthetic_motion_metrics.png")
    save_report(args.output_dir / "synthetic_motion_report.md", metrics_rows, records, args)

    for sequence, frames in preview_frames.items():
        if frames:
            write_video(args.output_dir / f"synthetic_motion_{sequence}_comparison.mp4", frames, fps=args.fps)
            Image.fromarray(frames[len(frames) // 2]).save(args.output_dir / f"synthetic_motion_{sequence}_preview.png")

    print("Synthetic motion benchmark complete.")
    print(args.data_dir / "synthetic_motion_sequences.npz")
    print(args.output_dir / "synthetic_motion_metrics.csv")
    print(args.output_dir / "synthetic_motion_2d_comparison.mp4")
    print(args.output_dir / "synthetic_motion_3d_comparison.mp4")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate moving 2D/3D geometric sequences and benchmark keypoint detectors.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--size", type=int, default=192)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--patch-size", type=int, default=9)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--max-points", type=int, default=60)
    parser.add_argument("--tolerance", type=float, default=5.0)
    parser.add_argument("--fps", type=int, default=4)
    return parser.parse_args()


def build_dataset(args: argparse.Namespace) -> list[FrameRecord]:
    rng = np.random.default_rng(args.seed)
    records = []
    for index in range(args.frames):
        image, gt = render_2d_motion_frame(index, args.frames, args.size, rng)
        records.append(FrameRecord("2d", index, image, gt))
    for index in range(args.frames):
        image, gt = render_3d_motion_frame(index, args.frames, args.size)
        records.append(FrameRecord("3d", index, image, gt))
    return records


def render_2d_motion_frame(index: int, frames: int, size: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    del rng
    canvas = np.full((size, size), 0.07, dtype=np.float32)
    t = index / max(1, frames - 1)
    gt: list[tuple[float, float]] = []

    square = np.array([[-22, -22], [22, -22], [22, 22], [-22, 22]], dtype=np.float32)
    square = transform_2d(square, angle=2.1 * math.pi * t, scale=1.0 + 0.12 * math.sin(2 * math.pi * t))
    square += np.array([58 + 18 * math.sin(2 * math.pi * t), 62 + 10 * math.cos(2 * math.pi * t)], dtype=np.float32)
    draw_polyline(canvas, square, closed=True, thickness=4)
    gt.extend(points_in_bounds(square, size))

    tri = np.array([[0, -26], [28, 25], [-28, 25]], dtype=np.float32)
    tri = transform_2d(tri, angle=-1.6 * math.pi * t + 0.4, scale=0.95)
    tri += np.array([132 - 24 * math.sin(2 * math.pi * t), 58 + 16 * math.sin(4 * math.pi * t)], dtype=np.float32)
    draw_polyline(canvas, tri, closed=True, thickness=4)
    gt.extend(points_in_bounds(tri, size))

    l_corner = np.array([[0, 0], [34, 0], [0, 0], [0, 34]], dtype=np.float32)
    l_corner = transform_2d(l_corner, angle=1.2 * math.pi * t - 0.5, scale=1.0)
    l_corner += np.array([58 + 12 * math.cos(2 * math.pi * t), 135 + 13 * math.sin(2 * math.pi * t)], dtype=np.float32)
    draw_segments(canvas, l_corner.reshape(2, 2, 2), thickness=4)
    gt.append(tuple(l_corner[0]))

    x_cross = np.array([[-24, -24], [24, 24], [-24, 24], [24, -24]], dtype=np.float32)
    x_cross = transform_2d(x_cross, angle=1.8 * math.pi * t, scale=0.95)
    x_center = np.array([134 + 17 * math.sin(2 * math.pi * t + 0.7), 135 + 10 * math.cos(2 * math.pi * t)], dtype=np.float32)
    x_cross += x_center
    draw_segments(canvas, x_cross.reshape(2, 2, 2), thickness=4)
    gt.append(tuple(x_center))

    image = finalize_canvas(canvas)
    return image, dedupe_points(np.asarray(gt, dtype=np.float32), min_distance=4.0, size=size)


def render_3d_motion_frame(index: int, frames: int, size: int) -> tuple[np.ndarray, np.ndarray]:
    canvas = np.full((size, size), 0.065, dtype=np.float32)
    t = index / max(1, frames - 1)
    gt: list[tuple[float, float]] = []

    cube_vertices = np.array(
        [
            [-0.8, -0.8, -0.8],
            [0.8, -0.8, -0.8],
            [0.8, 0.8, -0.8],
            [-0.8, 0.8, -0.8],
            [-0.8, -0.8, 0.8],
            [0.8, -0.8, 0.8],
            [0.8, 0.8, 0.8],
            [-0.8, 0.8, 0.8],
        ],
        dtype=np.float32,
    )
    cube_edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    cube = rotate_3d(cube_vertices, 1.8 * math.pi * t, 1.2 * math.pi * t + 0.2, 0.8 * math.pi * t)
    cube[:, 0] += -0.65 + 0.25 * math.sin(2 * math.pi * t)
    cube[:, 1] += 0.15 * math.cos(2 * math.pi * t)
    cube[:, 2] += 4.3 + 0.35 * math.sin(2 * math.pi * t + 0.3)
    cube_xy = project_3d(cube, size=size, focal=145.0)
    draw_edges(canvas, cube_xy, cube_edges, thickness=3)
    gt.extend(points_in_bounds(cube_xy, size))

    pyramid_vertices = np.array(
        [
            [-0.75, -0.75, -0.65],
            [0.75, -0.75, -0.65],
            [0.75, 0.75, -0.65],
            [-0.75, 0.75, -0.65],
            [0.0, 0.0, 0.9],
        ],
        dtype=np.float32,
    )
    pyramid_edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (1, 4), (2, 4), (3, 4)]
    pyramid = rotate_3d(pyramid_vertices, -1.3 * math.pi * t, 1.5 * math.pi * t, 0.9 * math.pi * t + 0.6)
    pyramid[:, 0] += 0.75 + 0.2 * math.cos(2 * math.pi * t)
    pyramid[:, 1] += 0.1 * math.sin(2 * math.pi * t)
    pyramid[:, 2] += 4.1 + 0.25 * math.cos(2 * math.pi * t)
    pyramid_xy = project_3d(pyramid, size=size, focal=145.0)
    draw_edges(canvas, pyramid_xy, pyramid_edges, thickness=3)
    gt.extend(points_in_bounds(pyramid_xy, size))

    image = finalize_canvas(canvas)
    return image, dedupe_points(np.asarray(gt, dtype=np.float32), min_distance=4.0, size=size)


def transform_2d(points: np.ndarray, angle: float, scale: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    matrix = np.array([[c, -s], [s, c]], dtype=np.float32) * float(scale)
    return points @ matrix.T


def rotate_3d(points: np.ndarray, ax: float, ay: float, az: float) -> np.ndarray:
    cx, sx = math.cos(ax), math.sin(ax)
    cy, sy = math.cos(ay), math.sin(ay)
    cz, sz = math.cos(az), math.sin(az)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
    return points @ (rz @ ry @ rx).T


def project_3d(points: np.ndarray, size: int, focal: float) -> np.ndarray:
    z = np.maximum(points[:, 2], 0.3)
    x = size / 2 + focal * points[:, 0] / z
    y = size / 2 - focal * points[:, 1] / z
    return np.column_stack([x, y]).astype(np.float32)


def draw_polyline(canvas: np.ndarray, points: np.ndarray, *, closed: bool, thickness: int) -> None:
    lines = [(points[i], points[(i + 1) % len(points)]) for i in range(len(points) - (0 if closed else 1))]
    draw_segments(canvas, np.asarray(lines, dtype=np.float32), thickness=thickness)


def draw_edges(canvas: np.ndarray, points: np.ndarray, edges: list[tuple[int, int]], thickness: int) -> None:
    segments = np.asarray([(points[a], points[b]) for a, b in edges], dtype=np.float32)
    draw_segments(canvas, segments, thickness=thickness)


def draw_segments(canvas: np.ndarray, segments: np.ndarray, thickness: int) -> None:
    image_u8 = np.uint8(np.clip(canvas, 0.0, 1.0) * 255)
    for start, end in np.asarray(segments, dtype=np.float32):
        p0 = tuple(np.round(start).astype(int))
        p1 = tuple(np.round(end).astype(int))
        cv2.line(image_u8, p0, p1, color=235, thickness=thickness, lineType=cv2.LINE_AA)
    canvas[:] = image_u8.astype(np.float32) / 255.0


def finalize_canvas(canvas: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(canvas, (3, 3), sigmaX=0.35)
    return np.clip(blur, 0.0, 1.0).astype(np.float32)


def points_in_bounds(points: np.ndarray, size: int, margin: float = 4.0) -> list[tuple[float, float]]:
    out = []
    for x, y in np.asarray(points, dtype=np.float32).reshape(-1, 2):
        if margin <= x < size - margin and margin <= y < size - margin:
            out.append((float(x), float(y)))
    return out


def dedupe_points(points: np.ndarray, min_distance: float, size: int) -> np.ndarray:
    selected: list[np.ndarray] = []
    for point in np.asarray(points, dtype=np.float32).reshape(-1, 2):
        if not (0 <= point[0] < size and 0 <= point[1] < size):
            continue
        if all(np.linalg.norm(point - other) >= min_distance for other in selected):
            selected.append(point)
    if not selected:
        return np.zeros((0, 2), dtype=np.float32)
    return np.vstack(selected).astype(np.float32)


def save_dataset(records: list[FrameRecord], path: Path) -> None:
    images = np.stack([record.image for record in records]).astype(np.float32)
    max_gt = max(len(record.gt) for record in records)
    gt_points = np.zeros((len(records), max_gt, 2), dtype=np.float32)
    gt_mask = np.zeros((len(records), max_gt), dtype=bool)
    for idx, record in enumerate(records):
        count = len(record.gt)
        gt_points[idx, :count] = record.gt
        gt_mask[idx, :count] = True
    np.savez_compressed(
        path,
        images=images,
        gt_points=gt_points,
        gt_mask=gt_mask,
        sequence=np.asarray([record.sequence for record in records]),
        frame_index=np.asarray([record.frame for record in records], dtype=np.int32),
        description="2D/3D geometric motion sequences with GT vertices/intersections.",
    )


def evaluate_dataset(
    records: list[FrameRecord],
    detectors: dict,
    detection_args: argparse.Namespace,
    args: argparse.Namespace,
) -> tuple[list[dict], list[dict], dict[str, list[np.ndarray]]]:
    frame_rows: list[dict] = []
    totals: dict[tuple[str, str], dict[str, float]] = {}
    preview_frames: dict[str, list[np.ndarray]] = {"2d": [], "3d": []}
    for record in records:
        detections = detect_all(record.image, detectors, detection_args)
        preview_frames[record.sequence].append(make_comparison_frame(record, detections))
        for method, points in detections.items():
            metrics = evaluate_points(points, record.gt, tolerance=args.tolerance)
            key = (record.sequence, method)
            item = totals.setdefault(key, {"tp": 0.0, "fp": 0.0, "fn": 0.0, "frames": 0.0, "points": 0.0, "gt": 0.0})
            item["tp"] += metrics.true_positives
            item["fp"] += metrics.false_positives
            item["fn"] += metrics.false_negatives
            item["frames"] += 1
            item["points"] += len(points)
            item["gt"] += len(record.gt)
            frame_rows.append(
                {
                    "sequence": record.sequence,
                    "frame": record.frame,
                    "method": method,
                    "gt_points": len(record.gt),
                    "detected_points": len(points),
                    "tp": metrics.true_positives,
                    "fp": metrics.false_positives,
                    "fn": metrics.false_negatives,
                    "precision": metrics.precision,
                    "recall": metrics.recall,
                    "f1": metrics.f1,
                }
            )

    metrics_rows = []
    for (sequence, method), item in sorted(totals.items()):
        precision = item["tp"] / max(1.0, item["tp"] + item["fp"])
        recall = item["tp"] / max(1.0, item["tp"] + item["fn"])
        f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
        metrics_rows.append(
            {
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
                "tolerance_px": float(args.tolerance),
            }
        )
    return frame_rows, metrics_rows, preview_frames


def detect_all(image: np.ndarray, detectors: dict, args: argparse.Namespace) -> dict[str, np.ndarray]:
    logistic, _ = logistic_threshold_points_for_image(image, detectors["logistic"], args)
    qpp_2q, _ = qpp_threshold_points_for_image(image, detectors["qpp_2q"], args)
    return {
        "Harris": run_harris(image, threshold_rel=0.01, max_points=args.max_points, min_distance=4.0),
        "FAST": run_fast(image, threshold=20, max_points=args.max_points, min_distance=4.0),
        "ORB": run_orb(image, max_points=args.max_points, min_distance=4.0),
        "Logistic": logistic,
        "QPP QNN2": qpp_2q,
    }


def make_comparison_frame(record: FrameRecord, detections: dict[str, np.ndarray]) -> np.ndarray:
    panels = [("GT", record.gt)] + list(detections.items())
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 7.0))
    for ax, (title, points) in zip(axes.flat, panels):
        ax.imshow(record.image, cmap="gray", vmin=0.0, vmax=1.0)
        gt = np.asarray(record.gt, dtype=np.float32).reshape(-1, 2)
        if title != "GT" and len(gt):
            ax.scatter(gt[:, 0], gt[:, 1], s=36, facecolors="none", edgecolors="#39ff14", marker="o", linewidths=1.5)
        points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        if len(points):
            if title == "GT":
                ax.scatter(points[:, 0], points[:, 1], s=42, facecolors="none", edgecolors="#39ff14", marker="o", linewidths=1.7)
            else:
                ax.scatter(points[:, 0], points[:, 1], s=22, c="#ff2d2d", marker="x", linewidths=1.2)
        ax.set_title(f"{title} ({len(points)})", fontsize=10)
        ax.axis("off")
    fig.suptitle(f"Synthetic {record.sequence.upper()} motion sequence, frame {record.frame:02d}", fontsize=12)
    fig.tight_layout()
    fig.canvas.draw()
    arr = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[..., :3].copy()
    plt.close(fig)
    return arr


def save_metrics_plot(rows: list[dict], path: Path) -> None:
    methods = ["Harris", "FAST", "ORB", "Logistic", "QPP QNN2"]
    sequences = ["2d", "3d"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, sequence in zip(axes, sequences):
        values = []
        for method in methods:
            row = next((item for item in rows if item["sequence"] == sequence and item["method"] == method), None)
            values.append(0.0 if row is None else float(row["f1"]))
        ax.bar(methods, values, color=["#64748b", "#94a3b8", "#cbd5e1", "#f97316", "#7c3aed"])
        ax.set_ylim(0.0, 1.0)
        ax.set_title(f"{sequence.upper()} motion F1")
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("F1")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_report(path: Path, rows: list[dict], records: list[FrameRecord], args: argparse.Namespace) -> None:
    lines = [
        "# Synthetic 2D/3D Motion Benchmark",
        "",
        "This benchmark creates continuous synthetic image sequences from moving geometric objects and evaluates classical keypoint detectors and the current QPP 2-qubit QNN.",
        "",
        "## Dataset",
        "",
        f"- Frames: {len(records)} total, {args.frames} per sequence type.",
        f"- Image size: {args.size} x {args.size}.",
        "- 2D sequence: rotating/translating/scaling square, triangle, L-corner, and X-junction.",
        "- 3D sequence: perspective projection of rotating/translating wireframe cube and pyramid.",
        "- GT keypoints: rendered geometric vertices and junction/intersection centers.",
        "",
        "## Outputs",
        "",
        "- `data/synthetic_motion_sequences.npz`",
        "- `outputs/synthetic_motion_metrics.csv`",
        "- `outputs/synthetic_motion_frame_metrics.csv`",
        "- `outputs/synthetic_motion_metrics.png`",
        "- `outputs/synthetic_motion_2d_comparison.mp4`",
        "- `outputs/synthetic_motion_3d_comparison.mp4`",
        "",
        "## Aggregate Results",
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
            "Note: this is a geometric-motion generalization test. QPP QNN uses the existing model trained on the earlier synthetic corner/junction patches; it is not retrained on these motion sequences.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
