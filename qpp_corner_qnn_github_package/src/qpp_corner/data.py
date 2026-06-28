"""Dataset discovery, manifest loading, and leakage-safe group splits."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class ManifestRecord:
    """One image-level manifest entry."""

    dataset: str
    sample_id: str
    image_path: Path
    label_path: Path
    scene_type: str
    raw: dict[str, Any]


@dataclass
class PatchTable:
    """Flattened patch-level data with image/sample groups preserved."""

    patches: np.ndarray
    labels: np.ndarray
    groups: np.ndarray
    centers_xy: np.ndarray
    scene_types: np.ndarray
    dataset: str
    records: list[ManifestRecord]


def _candidate_dataset_roots(data_root: str | Path) -> list[Path]:
    root = Path(data_root).expanduser().resolve()
    return [root, root / "data"]


def discover_datasets(data_root: str | Path) -> dict[str, Path]:
    """Return ``dataset_name -> dataset_dir`` for an extracted data root.

    The supplied archive may be addressed as either ``data/raw`` or
    ``data/raw/data``. This function accepts both.
    """

    datasets: dict[str, Path] = {}
    for candidate_root in _candidate_dataset_roots(data_root):
        if not candidate_root.exists():
            continue
        if (candidate_root / "manifest.jsonl").exists():
            datasets[candidate_root.name] = candidate_root
        for child in sorted(candidate_root.iterdir()):
            if child.is_dir() and (child / "manifest.jsonl").exists():
                datasets[child.name] = child
    return datasets


def find_dataset_dir(data_root: str | Path, dataset: str | None = None) -> Path:
    datasets = discover_datasets(data_root)
    if dataset is None:
        if len(datasets) == 1:
            return next(iter(datasets.values()))
        names = ", ".join(sorted(datasets)) or "<none>"
        raise ValueError(f"Specify --dataset; discovered datasets: {names}")
    if dataset not in datasets:
        names = ", ".join(sorted(datasets)) or "<none>"
        raise FileNotFoundError(f"Dataset {dataset!r} not found under {data_root}. Found: {names}")
    return datasets[dataset]


def _resolve_manifest_path(dataset_dir: Path, raw_path: str) -> Path:
    """Resolve paths written with several common relative conventions."""

    path = Path(raw_path)
    if path.is_absolute() and path.exists():
        return path

    candidates = [
        dataset_dir / path,
        dataset_dir.parent / path,
        dataset_dir.parent.parent / path,
        Path.cwd() / path,
    ]
    if path.parts and path.parts[0] == dataset_dir.name:
        candidates.append(dataset_dir.parent / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    # Fall back to the most likely path for a helpful downstream error.
    return candidates[0].resolve()


def read_manifest(data_root: str | Path, dataset: str | None = None) -> list[ManifestRecord]:
    dataset_dir = find_dataset_dir(data_root, dataset)
    records: list[ManifestRecord] = []
    with (dataset_dir / "manifest.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            sample_id = str(item.get("id") or item.get("sample_id") or Path(item["label"]).stem)
            records.append(
                ManifestRecord(
                    dataset=dataset_dir.name,
                    sample_id=sample_id,
                    image_path=_resolve_manifest_path(dataset_dir, item["image"]),
                    label_path=_resolve_manifest_path(dataset_dir, item["label"]),
                    scene_type=str(item.get("scene_type", "")),
                    raw=item,
                )
            )
    return records


def load_label_npz(record_or_path: ManifestRecord | str | Path) -> dict[str, Any]:
    path = record_or_path.label_path if isinstance(record_or_path, ManifestRecord) else Path(record_or_path)
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def load_patch_table(
    data_root: str | Path,
    dataset: str,
    *,
    max_images: int | None = None,
    sample_per_image: int | None = None,
    seed: int = 0,
) -> PatchTable:
    """Load all patches from a subdataset into a flattened table."""

    records = read_manifest(data_root, dataset)
    if max_images is not None:
        records = records[:max_images]
    rng = np.random.default_rng(seed)

    patches: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    groups: list[np.ndarray] = []
    centers: list[np.ndarray] = []
    scene_types: list[np.ndarray] = []
    kept_records: list[ManifestRecord] = []

    for record in records:
        item = load_label_npz(record)
        p = np.asarray(item["patches"], dtype=np.float32)
        y = (np.asarray(item["patch_labels"]).reshape(-1) > 0).astype(np.int64)
        c = np.asarray(item["patch_centers_xy"], dtype=np.float32)
        if sample_per_image is not None and sample_per_image < len(y):
            idx = rng.choice(len(y), size=sample_per_image, replace=False)
            idx.sort()
            p = p[idx]
            y = y[idx]
            c = c[idx]
        n = len(y)
        patches.append(p)
        labels.append(y)
        groups.append(np.full(n, record.sample_id, dtype=object))
        centers.append(c)
        scene_types.append(np.full(n, record.scene_type, dtype=object))
        kept_records.append(record)

    if not patches:
        raise ValueError(f"No patches loaded for dataset {dataset!r}")

    return PatchTable(
        patches=np.concatenate(patches, axis=0),
        labels=np.concatenate(labels, axis=0),
        groups=np.concatenate(groups, axis=0),
        centers_xy=np.concatenate(centers, axis=0),
        scene_types=np.concatenate(scene_types, axis=0),
        dataset=dataset,
        records=kept_records,
    )


def make_group_splits(
    groups: Iterable[Any],
    *,
    train: float = 0.70,
    val: float = 0.15,
    test: float = 0.15,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Create train/val/test index arrays from image/sample groups."""

    if train <= 0 or val < 0 or test < 0:
        raise ValueError("Split fractions must be non-negative and train must be positive.")
    unique = np.array(sorted({str(group) for group in groups}), dtype=object)
    if len(unique) < 3:
        raise ValueError("At least three groups are required for train/val/test splitting.")

    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n = len(unique)
    n_train = max(1, int(round(n * train)))
    n_val = max(1, int(round(n * val)))
    if n_train + n_val >= n:
        n_train = max(1, n - 2)
        n_val = 1
    train_groups = set(unique[:n_train])
    val_groups = set(unique[n_train : n_train + n_val])
    test_groups = set(unique[n_train + n_val :])
    if not test_groups:
        test_groups = {unique[-1]}
        train_groups.discard(unique[-1])

    group_array = np.asarray([str(group) for group in groups], dtype=object)
    return {
        "train": np.flatnonzero(np.isin(group_array, list(train_groups))),
        "val": np.flatnonzero(np.isin(group_array, list(val_groups))),
        "test": np.flatnonzero(np.isin(group_array, list(test_groups))),
    }


def save_splits(path: str | Path, splits: dict[str, np.ndarray], groups: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    group_array = np.asarray(groups)
    payload = {
        name: sorted({str(group_array[i]) for i in indices.tolist()})
        for name, indices in splits.items()
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_splits(path: str | Path, groups: np.ndarray) -> dict[str, np.ndarray]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    group_array = np.asarray([str(group) for group in groups], dtype=object)
    return {
        name: np.flatnonzero(np.isin(group_array, [str(group) for group in split_groups]))
        for name, split_groups in payload.items()
    }
