from __future__ import annotations

import numpy as np


FEATURE_NAMES = [
    "ix_center",
    "iy_center",
    "ix2_sum",
    "iy2_sum",
    "ixiy_sum",
    "grad_mag_center",
    "harris_response",
    "patch_mean",
    "patch_std",
]

STRUCTURE_TENSOR_FEATURE_NAMES = [
    "Ix",
    "Iy",
    "lambda1",
    "lambda2",
    "R",
]


def extract_patch_features(patches: np.ndarray, harris_k: float = 0.04) -> np.ndarray:
    """Convert image patches into the shared 9-D classical/QNN feature vector."""
    patches = np.asarray(patches, dtype=np.float32)
    if patches.ndim != 3:
        raise ValueError("patches must have shape (N, H, W).")

    features = np.zeros((patches.shape[0], len(FEATURE_NAMES)), dtype=np.float32)
    for index, patch in enumerate(patches):
        iy, ix = np.gradient(patch)
        center_y = patch.shape[0] // 2
        center_x = patch.shape[1] // 2
        ix2_sum = float(np.sum(ix * ix))
        iy2_sum = float(np.sum(iy * iy))
        ixiy_sum = float(np.sum(ix * iy))
        det = ix2_sum * iy2_sum - ixiy_sum * ixiy_sum
        trace = ix2_sum + iy2_sum
        harris_response = det - harris_k * trace * trace

        features[index] = np.array(
            [
                ix[center_y, center_x],
                iy[center_y, center_x],
                ix2_sum,
                iy2_sum,
                ixiy_sum,
                float(np.hypot(ix[center_y, center_x], iy[center_y, center_x])),
                harris_response,
                float(np.mean(patch)),
                float(np.std(patch)),
            ],
            dtype=np.float32,
        )

    return features


def extract_structure_tensor_features(patches: np.ndarray, harris_k: float = 0.04) -> np.ndarray:
    """Extract the canonical 5-D QNN feature vector from image patches.

    The feature order follows the QNN plan PDF:
    ``[Ix, Iy, lambda1, lambda2, R]``.
    """
    patches = np.asarray(patches, dtype=np.float32)
    if patches.ndim != 3:
        raise ValueError("patches must have shape (N, H, W).")

    features = np.zeros((patches.shape[0], len(STRUCTURE_TENSOR_FEATURE_NAMES)), dtype=np.float32)
    for index, patch in enumerate(patches):
        iy, ix = np.gradient(patch)
        center_y = patch.shape[0] // 2
        center_x = patch.shape[1] // 2
        ix2_sum = float(np.sum(ix * ix))
        iy2_sum = float(np.sum(iy * iy))
        ixiy_sum = float(np.sum(ix * iy))
        structure_tensor = np.array([[ix2_sum, ixiy_sum], [ixiy_sum, iy2_sum]], dtype=np.float64)
        eigenvalues = np.linalg.eigvalsh(structure_tensor)
        lambda2, lambda1 = float(eigenvalues[0]), float(eigenvalues[1])
        det = ix2_sum * iy2_sum - ixiy_sum * ixiy_sum
        trace = ix2_sum + iy2_sum
        harris_response = det - harris_k * trace * trace

        features[index] = np.array(
            [
                ix[center_y, center_x],
                iy[center_y, center_x],
                lambda1,
                lambda2,
                harris_response,
            ],
            dtype=np.float32,
        )

    return features
