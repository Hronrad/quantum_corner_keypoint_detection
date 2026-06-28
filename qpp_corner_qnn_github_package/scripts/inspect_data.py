from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qpp_corner.data import discover_datasets, load_label_npz, read_manifest


def inspect_dataset(data_root: Path, dataset: str) -> dict[str, object]:
    records = read_manifest(data_root, dataset)
    scene_counts = Counter(record.scene_type for record in records)
    patch_shapes = Counter()
    label_counts = Counter()
    npz_keys: list[str] = []
    patch_total = 0
    for record in records:
        item = load_label_npz(record)
        if not npz_keys:
            npz_keys = sorted(item.keys())
        patches = np.asarray(item["patches"])
        labels = (np.asarray(item["patch_labels"]).reshape(-1) > 0).astype(int)
        patch_shapes[str(tuple(patches.shape[1:]))] += len(labels)
        label_counts.update(labels.tolist())
        patch_total += len(labels)
    return {
        "dataset": dataset,
        "images": len(records),
        "patches": patch_total,
        "scene_counts": dict(scene_counts),
        "patch_shapes": dict(patch_shapes),
        "label_counts": {str(k): int(v) for k, v in label_counts.items()},
        "positive_fraction": float(label_counts[1] / patch_total) if patch_total else None,
        "npz_keys": npz_keys,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect extracted QPP-corner datasets.")
    parser.add_argument("--data-root", default="data/raw", type=Path, help="Path to data/raw or data/raw/data.")
    parser.add_argument("--dataset", default=None, help="Optional subdataset name to inspect.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args()

    datasets = discover_datasets(args.data_root)
    if args.dataset:
        names = [args.dataset]
    else:
        names = sorted(datasets)
    summaries = [inspect_dataset(args.data_root, name) for name in names]
    if args.json:
        print(json.dumps(summaries, indent=2))
        return
    print(f"Data root: {args.data_root}")
    for summary in summaries:
        print(f"\n[{summary['dataset']}]")
        print(f"  images: {summary['images']}")
        print(f"  patches: {summary['patches']}")
        print(f"  label_counts: {summary['label_counts']} pos_frac={summary['positive_fraction']:.3f}")
        print(f"  scene_counts: {summary['scene_counts']}")
        print(f"  patch_shapes: {summary['patch_shapes']}")
        print(f"  npz_keys: {summary['npz_keys']}")


if __name__ == "__main__":
    main()
