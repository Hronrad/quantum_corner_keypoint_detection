from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qpp_corner.data import load_patch_table
from qpp_corner.features import FeatureOptions, build_feature_matrix


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and cache patch feature matrices.")
    parser.add_argument("--data-root", default="data/raw", type=Path, help="Path to data/raw or data/raw/data.")
    parser.add_argument("--dataset", required=True, help="Subdataset name.")
    parser.add_argument("--feature-set", default="logS_eta", help="Feature set name.")
    parser.add_argument("--output-dir", default="data/processed", type=Path, help="Cache output directory.")
    parser.add_argument("--max-images", type=int, default=None, help="Limit number of images.")
    parser.add_argument("--sample-per-image", type=int, default=None, help="Optional patch subsampling per image.")
    parser.add_argument("--smooth-sigma", type=float, default=0.6, help="Gaussian smoothing before gradients.")
    parser.add_argument("--tensor-sigma", type=float, default=None, help="Gaussian structure tensor weight sigma.")
    parser.add_argument("--scalar-mode", default="logS_plus_c_eta", help="Scalar feature mode for feature_set=scalar.")
    parser.add_argument("--scalar-c", type=float, default=1.0, help="c in t=logS+c*eta.")
    parser.add_argument("--seed", type=int, default=0, help="Subsampling seed.")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing cache.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / f"{args.dataset}_{args.feature_set}.npz"
    meta_out = args.output_dir / f"{args.dataset}_{args.feature_set}.json"
    if out.exists() and not args.force:
        raise SystemExit(f"Cache already exists: {out}. Use --force to overwrite.")

    table = load_patch_table(
        args.data_root,
        args.dataset,
        max_images=args.max_images,
        sample_per_image=args.sample_per_image,
        seed=args.seed,
    )
    options = FeatureOptions(smooth_sigma=args.smooth_sigma, tensor_sigma=args.tensor_sigma)
    x, names, base, base_names = build_feature_matrix(
        table.patches,
        args.feature_set,
        options=options,
        scalar_mode=args.scalar_mode,
        scalar_c=args.scalar_c,
    )
    np.savez_compressed(
        out,
        X=x,
        y=table.labels,
        groups=table.groups.astype(str),
        centers_xy=table.centers_xy,
        feature_names=np.asarray(names, dtype=object),
        base_features=base,
        base_feature_names=np.asarray(base_names, dtype=object),
    )
    meta_out.write_text(
        json.dumps(
            {
                "dataset": args.dataset,
                "feature_set": args.feature_set,
                "feature_names": names,
                "n": int(len(table.labels)),
                "positive_fraction": float(np.mean(table.labels)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
