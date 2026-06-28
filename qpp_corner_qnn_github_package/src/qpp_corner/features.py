"""Patch-level image features for corner/keypoint classification."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


BASE_FEATURE_NAMES = [
    "Ix_center",
    "Iy_center",
    "lambda1",
    "lambda2",
    "R",
    "S",
    "eta",
    "logS",
    "Ixx",
    "Ixy",
    "Iyy",
    "detH",
    "trH",
]


FEATURE_SETS = {
    "lambda12": ["lambda1", "lambda2"],
    "logS_eta": ["logS", "eta"],
    "ixiy": ["Ix_center", "Iy_center"],
    "ref5": ["Ix_center", "Iy_center", "lambda1", "lambda2", "R"],
    "R": ["R"],
    "lambda2": ["lambda2"],
    "S": ["S"],
    "eta": ["eta"],
    "logS_plus_eta": ["logS_plus_eta"],
    "hessian": ["Ixx", "Ixy", "Iyy", "detH", "trH"],
    "all": BASE_FEATURE_NAMES,
}


@dataclass(frozen=True)
class FeatureOptions:
    gradient: str = "sobel"
    smooth_sigma: float = 0.6
    tensor_sigma: float | None = None
    harris_k: float = 0.04
    eps: float = 1e-8


def gaussian_weights(shape: tuple[int, int], sigma: float | None = None) -> np.ndarray:
    h, w = shape
    yy, xx = np.mgrid[:h, :w]
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    if sigma is None:
        sigma = max(h, w) / 4.0
    sigma = max(float(sigma), 1e-6)
    weights = np.exp(-0.5 * (((yy - cy) / sigma) ** 2 + ((xx - cx) / sigma) ** 2))
    return (weights / weights.sum()).astype(np.float32)


def compute_gradients(
    patch: np.ndarray,
    *,
    gradient: str = "sobel",
    smooth_sigma: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image = np.asarray(patch, dtype=np.float32)
    if image.ndim == 3:
        image = image.mean(axis=-1)
    if smooth_sigma and smooth_sigma > 0:
        image = ndimage.gaussian_filter(image, sigma=float(smooth_sigma), mode="nearest")

    if gradient == "finite":
        iy, ix = np.gradient(image)
    elif gradient == "sobel":
        ix = ndimage.sobel(image, axis=1, mode="nearest") / 8.0
        iy = ndimage.sobel(image, axis=0, mode="nearest") / 8.0
    else:
        raise ValueError(f"Unknown gradient method: {gradient}")
    return image.astype(np.float32), ix.astype(np.float32), iy.astype(np.float32)


def structure_tensor_features(
    ix: np.ndarray,
    iy: np.ndarray,
    *,
    tensor_sigma: float | None = None,
    harris_k: float = 0.04,
    eps: float = 1e-8,
) -> dict[str, float]:
    weights = gaussian_weights(ix.shape, tensor_sigma)
    a = float(np.sum(weights * ix * ix))
    b = float(np.sum(weights * ix * iy))
    c = float(np.sum(weights * iy * iy))
    trace = max(a + c, 0.0)
    det = max(a * c - b * b, 0.0)
    delta = max((a - c) * (a - c) + 4.0 * b * b, 0.0)
    root = float(np.sqrt(delta))
    lambda1 = max(0.5 * (trace + root), 0.0)
    lambda2 = max(0.5 * (trace - root), 0.0)
    if lambda2 > lambda1:
        lambda1, lambda2 = lambda2, lambda1
    response = det - float(harris_k) * trace * trace
    eta = 4.0 * lambda1 * lambda2 / (trace * trace + eps)
    eta = float(np.clip(eta, 0.0, 1.0))
    return {
        "lambda1": float(lambda1),
        "lambda2": float(lambda2),
        "R": float(response),
        "S": float(trace),
        "eta": eta,
        "logS": float(np.log(trace + eps)),
    }


def hessian_features(image: np.ndarray) -> dict[str, float]:
    gy, gx = np.gradient(image.astype(np.float32))
    gyy, gyx = np.gradient(gy)
    gxy, gxx = np.gradient(gx)
    cy = image.shape[0] // 2
    cx = image.shape[1] // 2
    ixx = float(gxx[cy, cx])
    ixy = float(0.5 * (gxy[cy, cx] + gyx[cy, cx]))
    iyy = float(gyy[cy, cx])
    return {
        "Ixx": ixx,
        "Ixy": ixy,
        "Iyy": iyy,
        "detH": float(ixx * iyy - ixy * ixy),
        "trH": float(ixx + iyy),
    }


def compute_patch_features(patch: np.ndarray, options: FeatureOptions | None = None) -> dict[str, float]:
    options = options or FeatureOptions()
    image, ix, iy = compute_gradients(
        patch,
        gradient=options.gradient,
        smooth_sigma=options.smooth_sigma,
    )
    cy = ix.shape[0] // 2
    cx = ix.shape[1] // 2
    features: dict[str, float] = {
        "Ix_center": float(ix[cy, cx]),
        "Iy_center": float(iy[cy, cx]),
    }
    features.update(
        structure_tensor_features(
            ix,
            iy,
            tensor_sigma=options.tensor_sigma,
            harris_k=options.harris_k,
            eps=options.eps,
        )
    )
    features.update(hessian_features(image))
    return features


def compute_base_feature_matrix(
    patches: np.ndarray,
    options: FeatureOptions | None = None,
) -> tuple[np.ndarray, list[str]]:
    rows = []
    for patch in np.asarray(patches):
        item = compute_patch_features(patch, options)
        rows.append([item[name] for name in BASE_FEATURE_NAMES])
    return np.asarray(rows, dtype=np.float32), list(BASE_FEATURE_NAMES)


def append_derived_columns(base: np.ndarray, names: list[str], *, scalar_c: float = 1.0) -> tuple[np.ndarray, list[str]]:
    values = {name: base[:, idx] for idx, name in enumerate(names)}
    extra = []
    extra_names = []
    if "logS" in values and "eta" in values:
        extra.append(values["logS"] + scalar_c * values["eta"])
        extra_names.append("logS_plus_eta" if scalar_c == 1.0 else f"logS_plus_{scalar_c:g}_eta")
    if extra:
        return np.column_stack([base, *extra]).astype(np.float32), names + extra_names
    return base, names


def select_feature_set(
    base: np.ndarray,
    names: list[str],
    feature_set: str,
    *,
    scalar_mode: str = "logS_plus_c_eta",
    scalar_c: float = 1.0,
) -> tuple[np.ndarray, list[str]]:
    """Select a named feature set from a base feature matrix."""

    all_values = {name: base[:, idx] for idx, name in enumerate(names)}
    if feature_set == "scalar":
        if scalar_mode == "logS_plus_c_eta":
            col = all_values["logS"] + float(scalar_c) * all_values["eta"]
            name = f"logS_plus_{scalar_c:g}_eta"
        elif scalar_mode == "lambda2":
            col = all_values["lambda2"]
            name = "lambda2"
        elif scalar_mode == "R":
            col = all_values["R"]
            name = "R"
        elif scalar_mode == "learned_logS_eta":
            return np.column_stack([all_values["logS"], all_values["eta"]]).astype(np.float32), ["logS", "eta"]
        else:
            raise ValueError(f"Unknown scalar mode: {scalar_mode}")
        return col.reshape(-1, 1).astype(np.float32), [name]

    expanded, expanded_names = append_derived_columns(base, names, scalar_c=scalar_c)
    values = {name: expanded[:, idx] for idx, name in enumerate(expanded_names)}
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature_set {feature_set!r}. Available: {sorted(FEATURE_SETS)}")
    selected = FEATURE_SETS[feature_set]
    return np.column_stack([values[name] for name in selected]).astype(np.float32), list(selected)


def build_feature_matrix(
    patches: np.ndarray,
    feature_set: str,
    *,
    options: FeatureOptions | None = None,
    scalar_mode: str = "logS_plus_c_eta",
    scalar_c: float = 1.0,
) -> tuple[np.ndarray, list[str], np.ndarray, list[str]]:
    base, names = compute_base_feature_matrix(patches, options)
    x, feature_names = select_feature_set(
        base,
        names,
        feature_set,
        scalar_mode=scalar_mode,
        scalar_c=scalar_c,
    )
    return x, feature_names, base, names
