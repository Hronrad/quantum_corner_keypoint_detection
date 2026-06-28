from __future__ import annotations

import numpy as np

from qpp_corner.features import FeatureOptions, build_feature_matrix, compute_patch_features


def test_corner_patch_features_have_sorted_nonnegative_eigenvalues():
    patch = np.zeros((21, 21), dtype=np.float32)
    patch[10:, 10] = 1.0
    patch[10, 10:] = 1.0
    features = compute_patch_features(patch, FeatureOptions(smooth_sigma=0.0))
    assert features["lambda1"] >= features["lambda2"] >= 0.0
    assert 0.0 <= features["eta"] <= 1.0
    assert np.isfinite(features["R"])


def test_feature_set_selection_shapes():
    patches = np.random.default_rng(0).normal(size=(5, 9, 9)).astype(np.float32)
    x, names, base, base_names = build_feature_matrix(patches, "ref5")
    assert x.shape == (5, 5)
    assert names == ["Ix_center", "Iy_center", "lambda1", "lambda2", "R"]
    assert base.shape[0] == 5
    assert "logS" in base_names


def test_scalar_modes():
    patches = np.random.default_rng(1).normal(size=(4, 9, 9)).astype(np.float32)
    x, names, *_ = build_feature_matrix(patches, "scalar", scalar_mode="logS_plus_c_eta", scalar_c=2.0)
    assert x.shape == (4, 1)
    assert names == ["logS_plus_2_eta"]
