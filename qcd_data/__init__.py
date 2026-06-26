"""Synthetic data utilities for quantum corner/keypoint detection."""

from .synthetic import (
    CLASS_TO_ID,
    ID_TO_CLASS,
    SyntheticKeypointConfig,
    SyntheticKeypointDataset,
    SyntheticSample,
    generate_sample,
    sample_patch_labels,
)

__all__ = [
    "CLASS_TO_ID",
    "ID_TO_CLASS",
    "SyntheticKeypointConfig",
    "SyntheticKeypointDataset",
    "SyntheticSample",
    "generate_sample",
    "sample_patch_labels",
]
