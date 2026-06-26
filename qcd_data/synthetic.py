from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter
from torch.utils.data import Dataset


CLASS_TO_ID = {
    "corner": 1,
    "t_junction": 2,
    "x_junction": 3,
}
ID_TO_CLASS = {value: key for key, value in CLASS_TO_ID.items()}


@dataclass(frozen=True)
class SyntheticKeypointConfig:
    # `image_size` is kept as the square-image shortcut. Set image_width and
    # image_height when you want rectangular images.
    image_size: int = 128
    image_width: int | None = None
    image_height: int | None = None
    min_margin: int = 16
    line_width: tuple[int, int] = (2, 5)
    antialias_scale: int = 4
    gaussian_sigma: float = 2.0
    positive_radius: float = 3.0
    negative_radius: float = 10.0
    patch_size: int = 21
    patches_per_image: int = 96
    positive_fraction: float = 0.5
    noise_std: tuple[float, float] = (0.0, 0.06)
    blur_probability: float = 0.25
    blur_radius: tuple[float, float] = (0.2, 0.8)
    contrast_range: tuple[float, float] = (0.75, 1.25)
    brightness_range: tuple[float, float] = (-0.08, 0.08)

    @property
    def width(self) -> int:
        return int(self.image_width or self.image_size)

    @property
    def height(self) -> int:
        return int(self.image_height or self.image_size)

    @property
    def shape(self) -> tuple[int, int]:
        # Numpy/torch arrays use (H, W), while point labels use (x, y).
        return self.height, self.width

    @property
    def min_side(self) -> int:
        return min(self.width, self.height)

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Image width and height must be positive.")
        if self.patch_size <= 0 or self.patch_size % 2 == 0:
            raise ValueError("patch_size must be a positive odd integer.")
        if self.patch_size >= self.min_side:
            raise ValueError("patch_size must be smaller than both image dimensions.")
        if self.line_width[0] <= 0 or self.line_width[0] > self.line_width[1]:
            raise ValueError("line_width must be a positive (min, max) range.")
        if self.noise_std[0] < 0 or self.noise_std[0] > self.noise_std[1]:
            raise ValueError("noise_std must be a nonnegative (min, max) range.")
        if self.blur_radius[0] < 0 or self.blur_radius[0] > self.blur_radius[1]:
            raise ValueError("blur_radius must be a nonnegative (min, max) range.")


@dataclass(frozen=True)
class SyntheticSample:
    # image/heatmap follow numpy convention: (height, width).
    image: np.ndarray
    # points_xy and patch_centers_xy follow vision convention: (x, y).
    points_xy: np.ndarray
    type_ids: np.ndarray
    heatmap: np.ndarray
    scene_type: str
    patch_centers_xy: np.ndarray
    patch_labels: np.ndarray


def generate_sample(
    rng: np.random.Generator | int | None = None,
    config: SyntheticKeypointConfig | None = None,
    scene_type: str | None = None,
) -> SyntheticSample:
    """Generate one synthetic image and all labels needed for early baselines."""
    config = config or SyntheticKeypointConfig()
    rng = _as_rng(rng)

    generators: dict[str, Callable[[np.random.Generator, SyntheticKeypointConfig], tuple[np.ndarray, np.ndarray, list[str]]]] = {
        "l_corner": _draw_l_corner,
        "t_junction": _draw_t_junction,
        "x_junction": _draw_x_junction,
        "checkerboard": _draw_checkerboard,
        "polygon": _draw_polygon,
        "line_intersections": _draw_line_intersections,
    }
    if scene_type is None:
        scene_type = rng.choice(list(generators))
    if scene_type not in generators:
        valid = ", ".join(sorted(generators))
        raise ValueError(f"Unknown scene_type={scene_type!r}. Expected one of: {valid}")

    image, points_xy, classes = generators[scene_type](rng, config)
    image = _augment_image(image, rng, config)
    heatmap = make_heatmap(points_xy, config.shape, sigma=config.gaussian_sigma)
    patch_centers_xy, patch_labels = sample_patch_labels(points_xy, rng, config)

    return SyntheticSample(
        image=image.astype(np.float32),
        points_xy=points_xy.astype(np.float32),
        type_ids=np.array([CLASS_TO_ID[name] for name in classes], dtype=np.int64),
        heatmap=heatmap.astype(np.float32),
        scene_type=scene_type,
        patch_centers_xy=patch_centers_xy.astype(np.float32),
        patch_labels=patch_labels.astype(np.int64),
    )


def make_heatmap(points_xy: np.ndarray, image_shape: int | tuple[int, int], sigma: float = 2.0) -> np.ndarray:
    """Create a max-composed Gaussian heatmap from point labels."""
    height, width = _resolve_shape(image_shape)
    yy, xx = np.mgrid[0:height, 0:width]
    heatmap = np.zeros((height, width), dtype=np.float32)
    if len(points_xy) == 0:
        return heatmap

    sigma2 = float(sigma) ** 2
    for x, y in points_xy:
        response = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2.0 * sigma2))
        heatmap = np.maximum(heatmap, response.astype(np.float32))
    return heatmap


def sample_patch_labels(
    points_xy: np.ndarray,
    rng: np.random.Generator | int | None = None,
    config: SyntheticKeypointConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample patch centers and binary labels for patch-level classification."""
    config = config or SyntheticKeypointConfig()
    rng = _as_rng(rng)
    half = config.patch_size // 2
    low = np.array([half, half], dtype=np.float32)
    high = np.array([config.width - half - 1, config.height - half - 1], dtype=np.float32)
    total = config.patches_per_image
    num_pos = int(round(total * config.positive_fraction))

    centers: list[tuple[float, float]] = []
    labels: list[int] = []

    # Positive patch centers are jittered around real keypoints; negative patch
    # centers are sampled away from every keypoint by negative_radius pixels.
    if len(points_xy) > 0:
        for _ in range(num_pos):
            point = points_xy[rng.integers(0, len(points_xy))]
            jitter = rng.normal(0.0, max(1.0, config.positive_radius / 2.0), size=2)
            center = np.clip(point + jitter, low, high)
            centers.append((float(center[0]), float(center[1])))
            labels.append(1)

    attempts = 0
    while len(labels) < total and attempts < total * 200:
        attempts += 1
        center = rng.uniform(low, high)
        if _min_distance(center, points_xy) >= config.negative_radius:
            centers.append((float(center[0]), float(center[1])))
            labels.append(0)

    while len(labels) < total:
        center = rng.uniform(low, high)
        centers.append((float(center[0]), float(center[1])))
        labels.append(0)

    order = rng.permutation(len(labels))
    return np.array(centers, dtype=np.float32)[order], np.array(labels, dtype=np.int64)[order]


class SyntheticKeypointDataset(Dataset):
    """Torch dataset that yields generated images with keypoint labels."""

    def __init__(
        self,
        length: int,
        config: SyntheticKeypointConfig | None = None,
        seed: int = 0,
        transform: Callable[[SyntheticSample], object] | None = None,
    ) -> None:
        self.length = int(length)
        self.config = config or SyntheticKeypointConfig()
        self.seed = int(seed)
        self.transform = transform

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = generate_sample(self.seed + index, self.config)
        if self.transform is not None:
            return self.transform(sample)

        return {
            "image": torch.from_numpy(sample.image[None, :, :]),
            "heatmap": torch.from_numpy(sample.heatmap[None, :, :]),
            "points_xy": torch.from_numpy(sample.points_xy),
            "type_ids": torch.from_numpy(sample.type_ids),
            "scene_type": sample.scene_type,
            "patch_centers_xy": torch.from_numpy(sample.patch_centers_xy),
            "patch_labels": torch.from_numpy(sample.patch_labels),
        }


def extract_patches(image: np.ndarray, centers_xy: np.ndarray, patch_size: int) -> np.ndarray:
    """Extract square patches centered at floating-point xy coordinates."""
    half = patch_size // 2
    padded = np.pad(image, half, mode="reflect")
    patches = []
    for x, y in centers_xy:
        cx = int(round(float(x))) + half
        cy = int(round(float(y))) + half
        patch = padded[cy - half : cy + half + 1, cx - half : cx + half + 1]
        patches.append(patch)
    return np.stack(patches, axis=0).astype(np.float32)


def save_sample_npz(path: Path, sample: SyntheticSample, config: SyntheticKeypointConfig) -> None:
    patches = extract_patches(sample.image, sample.patch_centers_xy, config.patch_size)
    np.savez_compressed(
        path,
        image=sample.image,
        points_xy=sample.points_xy,
        type_ids=sample.type_ids,
        heatmap=sample.heatmap,
        scene_type=np.array(sample.scene_type),
        patch_centers_xy=sample.patch_centers_xy,
        patch_labels=sample.patch_labels,
        patches=patches,
    )


def _draw_l_corner(
    rng: np.random.Generator, config: SyntheticKeypointConfig
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    pivot = _random_point(rng, config)
    orientation = int(rng.integers(0, 4))
    length_a = _random_length(rng, config, 0.25, 0.50)
    length_b = _random_length(rng, config, 0.25, 0.50)
    directions = [((1, 0), (0, 1)), ((-1, 0), (0, 1)), ((-1, 0), (0, -1)), ((1, 0), (0, -1))]
    d1, d2 = directions[orientation]
    p1 = _clip_point(pivot + np.array(d1) * length_a, config)
    p2 = _clip_point(pivot + np.array(d2) * length_b, config)

    return _draw_lines(
        config.shape,
        [(tuple(pivot), tuple(p1)), (tuple(pivot), tuple(p2))],
        rng,
        config,
    ), np.array([pivot], dtype=np.float32), ["corner"]


def _draw_t_junction(
    rng: np.random.Generator, config: SyntheticKeypointConfig
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    center = _random_point(rng, config)
    angle = rng.uniform(0, np.pi)
    bar_len = _random_length(rng, config, 0.28, 0.45)
    stem_len = _random_length(rng, config, 0.22, 0.42)
    direction = np.array([np.cos(angle), np.sin(angle)])
    normal = np.array([-direction[1], direction[0]])

    a = _clip_point(center - direction * bar_len, config)
    b = _clip_point(center + direction * bar_len, config)
    c = _clip_point(center + normal * stem_len, config)
    image = _draw_lines(config.shape, [(tuple(a), tuple(b)), (tuple(center), tuple(c))], rng, config)
    return image, np.array([center], dtype=np.float32), ["t_junction"]


def _draw_x_junction(
    rng: np.random.Generator, config: SyntheticKeypointConfig
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    center = _random_point(rng, config)
    angle = rng.uniform(0, np.pi)
    length = _random_length(rng, config, 0.28, 0.48)
    dirs = [
        np.array([np.cos(angle), np.sin(angle)]),
        np.array([np.cos(angle + np.pi / 2.0), np.sin(angle + np.pi / 2.0)]),
    ]
    lines = []
    for direction in dirs:
        a = _clip_point(center - direction * length, config)
        b = _clip_point(center + direction * length, config)
        lines.append((tuple(a), tuple(b)))
    image = _draw_lines(config.shape, lines, rng, config)
    return image, np.array([center], dtype=np.float32), ["x_junction"]


def _draw_checkerboard(
    rng: np.random.Generator, config: SyntheticKeypointConfig
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    height, width = config.shape
    cells_x = int(rng.integers(5, 9))
    cells_y = int(rng.integers(5, 9))
    square_x = width / cells_x
    square_y = height / cells_y
    image = np.zeros((height, width), dtype=np.float32)
    low, high = _random_background_foreground(rng)
    image[:, :] = low
    for row in range(cells_y):
        for col in range(cells_x):
            if (row + col) % 2 == 0:
                x0 = int(round(col * square_x))
                x1 = int(round((col + 1) * square_x))
                y0 = int(round(row * square_y))
                y1 = int(round((row + 1) * square_y))
                image[y0:y1, x0:x1] = high

    points = []
    for row in range(1, cells_y):
        for col in range(1, cells_x):
            points.append((col * square_x, row * square_y))
    image = _smooth_render(image, config)
    return image, np.array(points, dtype=np.float32), ["x_junction"] * len(points)


def _draw_polygon(
    rng: np.random.Generator, config: SyntheticKeypointConfig
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    n = int(rng.integers(3, 7))
    center = np.array([config.width / 2, config.height / 2], dtype=np.float32)
    center += rng.normal(0, config.min_side * 0.07, size=2)
    angles = np.sort(rng.uniform(0, 2 * np.pi, size=n))
    radii = rng.uniform(config.min_side * 0.22, config.min_side * 0.43, size=n)
    points = np.stack([center[0] + np.cos(angles) * radii, center[1] + np.sin(angles) * radii], axis=1)
    points = np.array([_clip_point(point, config) for point in points], dtype=np.float32)
    lines = [(tuple(points[i]), tuple(points[(i + 1) % n])) for i in range(n)]
    image = _draw_lines(config.shape, lines, rng, config)
    return image, points, ["corner"] * len(points)


def _draw_line_intersections(
    rng: np.random.Generator, config: SyntheticKeypointConfig
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    num_lines = int(rng.integers(3, 6))
    raw_lines: list[tuple[np.ndarray, np.ndarray]] = []
    draw_lines: list[tuple[tuple[float, float], tuple[float, float]]] = []

    for _ in range(num_lines):
        p = _random_point(rng, config)
        angle = rng.uniform(0, np.pi)
        direction = np.array([np.cos(angle), np.sin(angle)])
        length = max(config.width, config.height) * 0.9
        a = _clip_point(p - direction * length, config)
        b = _clip_point(p + direction * length, config)
        raw_lines.append((a, b))
        draw_lines.append((tuple(a), tuple(b)))

    intersections = []
    for i in range(len(raw_lines)):
        for j in range(i + 1, len(raw_lines)):
            point = _line_intersection(*raw_lines[i], *raw_lines[j])
            if point is not None and _inside_margin(point, config):
                if all(np.linalg.norm(point - existing) > 5 for existing in intersections):
                    intersections.append(point)

    if not intersections:
        return _draw_x_junction(rng, config)

    image = _draw_lines(config.shape, draw_lines, rng, config)
    return image, np.array(intersections, dtype=np.float32), ["x_junction"] * len(intersections)


def _draw_lines(
    shape: tuple[int, int],
    lines: Iterable[tuple[tuple[float, float], tuple[float, float]]],
    rng: np.random.Generator,
    config: SyntheticKeypointConfig,
) -> np.ndarray:
    height, width = shape
    scale = config.antialias_scale
    bg, fg = _random_background_foreground(rng)
    image = Image.new("L", (width * scale, height * scale), color=int(bg * 255))
    draw = ImageDraw.Draw(image)
    line_width = int(rng.integers(config.line_width[0], config.line_width[1] + 1)) * scale
    for a, b in lines:
        a_scaled = (float(a[0]) * scale, float(a[1]) * scale)
        b_scaled = (float(b[0]) * scale, float(b[1]) * scale)
        draw.line([a_scaled, b_scaled], fill=int(fg * 255), width=line_width, joint="curve")
    return _downsample(image, shape)


def _smooth_render(image: np.ndarray, config: SyntheticKeypointConfig) -> np.ndarray:
    pil = Image.fromarray(np.uint8(np.clip(image, 0.0, 1.0) * 255), mode="L")
    pil = pil.resize((config.width * config.antialias_scale, config.height * config.antialias_scale), Image.Resampling.BICUBIC)
    return _downsample(pil, config.shape)


def _augment_image(image: np.ndarray, rng: np.random.Generator, config: SyntheticKeypointConfig) -> np.ndarray:
    image = image.astype(np.float32)
    contrast = rng.uniform(*config.contrast_range)
    brightness = rng.uniform(*config.brightness_range)
    image = np.clip((image - 0.5) * contrast + 0.5 + brightness, 0.0, 1.0)

    noise_std = rng.uniform(*config.noise_std)
    if noise_std > 0:
        image = image + rng.normal(0.0, noise_std, size=image.shape).astype(np.float32)

    if rng.random() < config.blur_probability:
        radius = rng.uniform(*config.blur_radius)
        pil = Image.fromarray(np.uint8(np.clip(image, 0.0, 1.0) * 255), mode="L")
        image = np.asarray(pil.filter(ImageFilter.GaussianBlur(radius=radius)), dtype=np.float32) / 255.0

    return np.clip(image, 0.0, 1.0)


def _downsample(image: Image.Image, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    image = image.resize((width, height), Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.float32) / 255.0


def _random_background_foreground(rng: np.random.Generator) -> tuple[float, float]:
    if rng.random() < 0.5:
        return rng.uniform(0.02, 0.18), rng.uniform(0.72, 0.98)
    return rng.uniform(0.76, 0.96), rng.uniform(0.03, 0.25)


def _resolve_shape(image_shape: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(image_shape, int):
        return image_shape, image_shape
    height, width = image_shape
    return int(height), int(width)


def _effective_margin(config: SyntheticKeypointConfig) -> float:
    # Keep the requested margin when possible, but relax it for small images so
    # the random sampler still has a valid interior region.
    return float(min(config.min_margin, max(2, (config.min_side - 2) // 3)))


def _random_length(
    rng: np.random.Generator, config: SyntheticKeypointConfig, low_ratio: float, high_ratio: float
) -> int:
    low = max(4, int(config.min_side * low_ratio))
    high = max(low + 1, int(config.min_side * high_ratio))
    return int(rng.integers(low, high))


def _random_point(rng: np.random.Generator, config: SyntheticKeypointConfig) -> np.ndarray:
    margin = _effective_margin(config)
    low = np.array([margin, margin], dtype=np.float32)
    high = np.array([config.width - margin, config.height - margin], dtype=np.float32)
    return rng.uniform(low, high).astype(np.float32)


def _clip_point(point: np.ndarray, config: SyntheticKeypointConfig) -> np.ndarray:
    low = np.array([1, 1], dtype=np.float32)
    high = np.array([config.width - 2, config.height - 2], dtype=np.float32)
    return np.clip(np.asarray(point, dtype=np.float32), low, high)


def _inside_margin(point: np.ndarray, config: SyntheticKeypointConfig) -> bool:
    margin = _effective_margin(config)
    return bool(
        margin <= point[0] <= config.width - margin
        and margin <= point[1] <= config.height - margin
    )


def _line_intersection(
    a1: np.ndarray, a2: np.ndarray, b1: np.ndarray, b2: np.ndarray
) -> np.ndarray | None:
    da = a2 - a1
    db = b2 - b1
    matrix = np.array([[da[0], -db[0]], [da[1], -db[1]]], dtype=np.float32)
    det = float(np.linalg.det(matrix))
    if abs(det) < 1e-6:
        return None
    t, u = np.linalg.solve(matrix, b1 - a1)
    if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
        return a1 + t * da
    return None


def _min_distance(point_xy: np.ndarray, points_xy: np.ndarray) -> float:
    if len(points_xy) == 0:
        return float("inf")
    return float(np.min(np.linalg.norm(points_xy - point_xy[None, :], axis=1)))


def _as_rng(rng: np.random.Generator | int | None) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)
