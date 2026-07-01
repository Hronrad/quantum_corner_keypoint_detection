# Real-Data Preview Demo

This preview runs the current QPP few-qubit detector on small real-image samples without ground-truth labels.
Metrics are therefore descriptive counts and QPP score summaries, not precision/recall.

## Sources

- HPatches example sequence montage: `https://raw.githubusercontent.com/hpatches/hpatches-dataset/master/img/images.png`
- KITTI official mini drive: `https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0001/2011_09_26_drive_0001_sync.zip`
- KITTI status: `{"status": "downloaded and extracted", "frames": 72, "url": "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0001/2011_09_26_drive_0001_sync.zip"}`

## Outputs

- `outputs/demos/realdata/videos/realdata_hpatches_qpp_overlay.mp4`
- `outputs/demos/realdata/videos/realdata_hpatches_qpp_overlay.gif`
- `outputs/demos/realdata/videos/realdata_kitti_qpp_overlay.mp4`
- `outputs/demos/realdata/videos/realdata_kitti_qpp_overlay.gif`
- `outputs/demos/dynamic_noise/videos/dynamic_noise_robustness_demo.mp4`
- `outputs/demos/dynamic_noise/videos/dynamic_noise_robustness_demo.gif`
- `outputs/demos/dynamic_noise/figures/dynamic_noise_robustness_demo_preview.png`

Videos are encoded as H.264/yuv420p for browser and presentation compatibility; GIF files are fallback previews.
The KITTI preview is encoded from 72 frames at 7.2 fps, so it plays in about 10 seconds.

## HPatches example sequence

- Frames: 6
- QPP points per frame: min 35, mean 35.0, max 35
- Harris points per frame: min 35, mean 35.0, max 35

## KITTI drive 0001

- Frames: 72
- 2-qubit QNN points per frame: min 35, mean 35.0, max 35
- Harris points per frame: min 35, mean 35.0, max 35
