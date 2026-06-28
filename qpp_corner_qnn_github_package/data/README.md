# Data not included

This package intentionally excludes data contents.

Expected local data layout after downloading/extracting the shared dataset:

data/raw/data/
  smoke_readme_pipeline/
  synthetic_keypoints/
  synthetic_keypoints_smoke/
  synthetic_keypoints_small_blur/
  synthetic_keypoints_rect_smoke/
  synthetic_keypoints_square_smoke/

Run data inspection with:

python scripts/inspect_data.py --data-root data/raw
