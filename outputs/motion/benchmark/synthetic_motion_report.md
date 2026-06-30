# Synthetic 2D/3D Motion Benchmark

This benchmark creates continuous synthetic image sequences from moving geometric objects and evaluates classical keypoint detectors and the current QPP 2-qubit QNN.

## Dataset

- Frames: 80 total, 40 per sequence type.
- Image size: 192 x 192.
- 2D sequence: rotating/translating/scaling square, triangle, L-corner, and X-junction.
- 3D sequence: perspective projection of rotating/translating wireframe cube and pyramid.
- GT keypoints: rendered geometric vertices and junction/intersection centers.

## Outputs

- `data/synthetic_motion_sequences.npz`
- `outputs/synthetic_motion_metrics.csv`
- `outputs/synthetic_motion_frame_metrics.csv`
- `outputs/synthetic_motion_metrics.png`
- `outputs/synthetic_motion_2d_comparison.mp4`
- `outputs/synthetic_motion_3d_comparison.mp4`

## Aggregate Results

| Sequence | Method | Precision | Recall | F1 | Mean Detected |
| --- | --- | ---: | ---: | ---: | ---: |
| 2d | FAST | 0.3005 | 0.9417 | 0.4556 | 28.2 |
| 2d | Harris | 0.3346 | 0.9778 | 0.4986 | 26.3 |
| 2d | Logistic | 0.1475 | 0.9833 | 0.2565 | 60.0 |
| 2d | ORB | 0.2887 | 0.8333 | 0.4289 | 26.0 |
| 2d | QPP QNN2 | 0.1467 | 0.9778 | 0.2551 | 60.0 |
| 3d | FAST | 0.2573 | 0.9207 | 0.4022 | 46.2 |
| 3d | Harris | 0.2467 | 0.7253 | 0.3682 | 38.0 |
| 3d | Logistic | 0.1821 | 0.8453 | 0.2996 | 60.0 |
| 3d | ORB | 0.1777 | 0.3946 | 0.2450 | 28.7 |
| 3d | QPP QNN2 | 0.1750 | 0.8124 | 0.2880 | 60.0 |

Note: this is a geometric-motion generalization test. QPP QNN uses the existing model trained on the earlier synthetic corner/junction patches; it is not retrained on these motion sequences.
