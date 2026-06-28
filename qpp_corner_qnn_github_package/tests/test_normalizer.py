from __future__ import annotations

import numpy as np

from qpp_corner.normalizer import FeatureNormalizer


def test_normalizer_clips_and_roundtrips(tmp_path):
    x = np.asarray([[0.0, 1.0], [1.0, 3.0], [100.0, -100.0]], dtype=np.float32)
    norm = FeatureNormalizer(clip=3.0).fit(x, ["a", "b"])
    z = norm.transform(x)
    assert z.max() <= 3.0
    assert z.min() >= -3.0
    phi = norm.to_angles(z)
    assert np.all(phi <= np.pi)
    path = tmp_path / "normalizer.json"
    norm.save_json(path)
    loaded = FeatureNormalizer.load_json(path)
    np.testing.assert_allclose(loaded.transform(x), z)
    assert loaded.feature_names == ["a", "b"]
