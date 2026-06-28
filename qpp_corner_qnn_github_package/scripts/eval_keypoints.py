from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qpp_corner.data import load_label_npz, read_manifest
from qpp_corner.metrics import choose_threshold_by_f1


def nms_sparse(points: np.ndarray, scores: np.ndarray, radius: float) -> np.ndarray:
    order = np.argsort(scores)[::-1]
    keep: list[int] = []
    suppressed = np.zeros(len(points), dtype=bool)
    for idx in order:
        if suppressed[idx]:
            continue
        keep.append(int(idx))
        dist = np.linalg.norm(points - points[idx], axis=1)
        suppressed |= dist <= radius
    return np.asarray(keep, dtype=int)


def match_points(detections: np.ndarray, truth: np.ndarray, radius: float) -> tuple[int, int, int]:
    used = np.zeros(len(truth), dtype=bool)
    tp = 0
    for det in detections:
        if len(truth) == 0:
            continue
        dist = np.linalg.norm(truth - det, axis=1)
        candidates = np.where((dist <= radius) & (~used))[0]
        if len(candidates):
            best = candidates[np.argmin(dist[candidates])]
            used[best] = True
            tp += 1
    fp = len(detections) - tp
    fn = len(truth) - tp
    return tp, fp, fn


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate sparse patch-center scores as keypoint detections.")
    parser.add_argument("--run-dir", required=True, type=Path, help="Run directory containing predictions.csv.")
    parser.add_argument("--data-root", default="data/raw", type=Path, help="Path to data/raw or data/raw/data.")
    parser.add_argument("--dataset", default=None, help="Dataset name; defaults to dataset column in predictions.")
    parser.add_argument("--split", default="test", help="Prediction split to evaluate.")
    parser.add_argument("--threshold", type=float, default=None, help="Score threshold; if omitted, choose by best patch F1.")
    parser.add_argument("--match-radius", type=float, default=4.0, help="Detection-to-truth match radius in pixels.")
    parser.add_argument("--nms-radius", type=float, default=3.0, help="Sparse NMS radius in pixels.")
    args = parser.parse_args()

    predictions = pd.read_csv(args.run_dir / "predictions.csv")
    dataset = args.dataset or str(predictions["dataset"].iloc[0])
    split_df = predictions[predictions["split"] == args.split].copy()
    if split_df.empty:
        raise SystemExit(f"No predictions for split {args.split!r}")
    threshold = args.threshold
    if threshold is None:
        threshold, _ = choose_threshold_by_f1(split_df["label"].to_numpy(), split_df["score"].to_numpy())

    records = {record.sample_id: record for record in read_manifest(args.data_root, dataset)}
    totals = {"tp": 0, "fp": 0, "fn": 0}
    for sample_id, group in split_df.groupby("sample_id"):
        points = group[["center_x", "center_y"]].to_numpy(dtype=float)
        scores = group["score"].to_numpy(dtype=float)
        active = scores >= threshold
        det_points = points[active]
        det_scores = scores[active]
        if len(det_points):
            keep = nms_sparse(det_points, det_scores, args.nms_radius)
            det_points = det_points[keep]
        truth = np.asarray(load_label_npz(records[str(sample_id)])["points_xy"], dtype=float)
        tp, fp, fn = match_points(det_points, truth, args.match_radius)
        totals["tp"] += tp
        totals["fp"] += fp
        totals["fn"] += fn

    precision = totals["tp"] / max(totals["tp"] + totals["fp"], 1)
    recall = totals["tp"] / max(totals["tp"] + totals["fn"], 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    result = {
        "dataset": dataset,
        "split": args.split,
        "threshold": float(threshold),
        "match_radius": args.match_radius,
        "nms_radius": args.nms_radius,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        **totals,
    }
    out = args.run_dir / f"keypoint_eval_{args.split}.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
