from __future__ import annotations

from pathlib import Path

import pytest

from qpp_corner.data import discover_datasets, load_patch_table, make_group_splits, read_manifest


DATA_ROOT = Path("data/raw")


pytestmark = pytest.mark.skipif(not (DATA_ROOT / "data").exists(), reason="sample data is not extracted")


def test_manifest_resolves_from_raw_root():
    datasets = discover_datasets(DATA_ROOT)
    assert "smoke_readme_pipeline" in datasets
    records = read_manifest(DATA_ROOT, "smoke_readme_pipeline")
    assert records
    assert records[0].label_path.exists()
    assert records[0].image_path.exists()


def test_patch_table_and_group_split_are_image_level():
    table = load_patch_table(DATA_ROOT, "smoke_readme_pipeline")
    assert table.patches.ndim == 3
    assert table.labels.shape[0] == table.patches.shape[0]
    splits = make_group_splits(table.groups, train=0.5, val=0.25, test=0.25, seed=1)
    split_groups = [set(table.groups[idx].tolist()) for idx in splits.values()]
    assert split_groups[0].isdisjoint(split_groups[1])
    assert split_groups[0].isdisjoint(split_groups[2])
    assert split_groups[1].isdisjoint(split_groups[2])
