from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qcd_data.synthetic import SyntheticKeypointConfig, generate_sample, save_sample_npz
from qcd_data.visualize import save_preview_grid


def main() -> None:
    args = parse_args()
    config = SyntheticKeypointConfig(
        image_size=args.image_size,
        image_width=args.width,
        image_height=args.height,
        line_width=(args.line_width_min, args.line_width_max),
        patch_size=args.patch_size,
        patches_per_image=args.patches_per_image,
        gaussian_sigma=args.gaussian_sigma,
        positive_radius=args.positive_radius,
        negative_radius=args.negative_radius,
        noise_std=(args.noise_std_min, args.noise_std_max),
        blur_probability=args.blur_probability,
        blur_radius=(args.blur_radius_min, args.blur_radius_max),
    )

    out_dir = args.out
    image_dir = out_dir / "images"
    label_dir = out_dir / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    preview_samples = []
    manifest_path = out_dir / "manifest.jsonl"

    with manifest_path.open("w", encoding="utf-8") as manifest:
        for index in range(args.count):
            sample_seed = int(rng.integers(0, 2**32 - 1))
            sample = generate_sample(sample_seed, config)
            stem = f"sample_{index:06d}"
            image_path = image_dir / f"{stem}.png"
            label_path = label_dir / f"{stem}.npz"

            image_u8 = np.uint8(np.clip(sample.image, 0.0, 1.0) * 255)
            Image.fromarray(image_u8).save(image_path)
            save_sample_npz(label_path, sample, config)

            manifest.write(
                json.dumps(
                    {
                        "id": stem,
                        "image": str(image_path.as_posix()),
                        "label": str(label_path.as_posix()),
                        "scene_type": sample.scene_type,
                        "num_points": int(len(sample.points_xy)),
                        "height": int(sample.image.shape[0]),
                        "width": int(sample.image.shape[1]),
                        "seed": sample_seed,
                    },
                    ensure_ascii=True,
                )
                + "\n"
            )

            if len(preview_samples) < args.preview_count:
                preview_samples.append(sample)

    config_path = out_dir / "config.json"
    config_data = {
        **config.__dict__,
        "resolved_width": config.width,
        "resolved_height": config.height,
    }
    config_path.write_text(json.dumps(config_data, indent=2, ensure_ascii=True), encoding="utf-8")

    if preview_samples:
        save_preview_grid(preview_samples, out_dir / "preview.png", columns=args.preview_columns)

    print(f"Wrote {args.count} samples to {out_dir}")
    print(f"Manifest: {manifest_path}")
    if preview_samples:
        print(f"Preview: {out_dir / 'preview.png'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic corner/keypoint data.")
    parser.add_argument("--out", type=Path, default=Path("data/synthetic_keypoints"))
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--image-size", type=int, default=128, help="Square image size shortcut.")
    parser.add_argument("--width", type=int, default=None, help="Image width. Overrides --image-size when set.")
    parser.add_argument("--height", type=int, default=None, help="Image height. Overrides --image-size when set.")
    parser.add_argument("--line-width-min", type=int, default=2)
    parser.add_argument("--line-width-max", type=int, default=5)
    parser.add_argument("--patch-size", type=int, default=21)
    parser.add_argument("--patches-per-image", type=int, default=96)
    parser.add_argument("--gaussian-sigma", type=float, default=2.0)
    parser.add_argument("--positive-radius", type=float, default=3.0)
    parser.add_argument("--negative-radius", type=float, default=10.0)
    parser.add_argument("--noise-std-min", type=float, default=0.0)
    parser.add_argument("--noise-std-max", type=float, default=0.06)
    parser.add_argument("--blur-probability", type=float, default=0.25)
    parser.add_argument("--blur-radius-min", type=float, default=0.2)
    parser.add_argument("--blur-radius-max", type=float, default=0.8)
    parser.add_argument("--preview-count", type=int, default=16)
    parser.add_argument("--preview-columns", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    main()
