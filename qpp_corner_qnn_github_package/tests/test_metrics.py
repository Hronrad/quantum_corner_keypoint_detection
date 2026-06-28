from __future__ import annotations

import numpy as np

from qpp_corner.metrics import binary_metrics, choose_threshold_by_f1


def test_threshold_selection_uses_validation_scores():
    y = np.asarray([0, 0, 1, 1])
    scores = np.asarray([0.1, 0.2, 0.8, 0.9])
    threshold, f1 = choose_threshold_by_f1(y, scores)
    assert 0.2 <= threshold <= 0.8
    assert f1 == 1.0
    metrics = binary_metrics(y, scores, threshold)
    assert metrics["f1"] == 1.0
    assert metrics["pr_auc"] == 1.0
