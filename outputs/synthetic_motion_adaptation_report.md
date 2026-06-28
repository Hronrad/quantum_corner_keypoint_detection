# Motion Domain Adaptation Results

This run completes motion/domain-randomized fine-tuning, adaptive per-frame thresholding, and stronger NMS for the synthetic 2D/3D motion benchmark.

## What Changed

- Fine-tuning data uses positives around geometric GT points, random background negatives, and hard negatives from original QPP QNN false detections.
- Domain randomization adds brightness/contrast jitter, Gaussian noise, mild blur, and occasional salt-and-pepper corruption without changing GT geometry.
- Adaptive threshold chooses a per-frame score quantile on the validation split.
- Stronger NMS tunes radius and max points on the validation split.

## Patch Dataset

- Train samples: 37136, positives: 12260
- Val samples: 1800, positives: 657
- Test patch samples: 1805, positives: 659
- Frame split: first 70% train, next 15% validation, final 15% test for each of 2D/3D.

## Tuned Configs

- Learning detectors: `{"Logistic original": {"quantile": 0.97, "nms_radius": 6.0, "max_points": 32, "min_threshold": null}, "QPP QNN2 original": {"quantile": 0.88, "nms_radius": 6.0, "max_points": 24, "min_threshold": null}, "Logistic motion-tuned": {"quantile": 0.985, "nms_radius": 14.0, "max_points": 18, "min_threshold": null}, "QPP QNN2 fine-tuned": {"quantile": 0.88, "nms_radius": 10.0, "max_points": 14, "min_threshold": null}}`
- Classical detectors: `{"Harris tuned": {"threshold_rel": 0.02, "min_distance": 8.0, "max_points": 32}, "FAST tuned": {"threshold": 32, "min_distance": 6.0, "max_points": 44}, "ORB tuned": {"fast_threshold": 5, "min_distance": 6.0, "max_points": 44, "nfeatures": 132}}`

## Test Results

| Sequence | Method | Precision | Recall | F1 | Mean Detected |
| --- | --- | ---: | ---: | ---: | ---: |
| 2d | FAST tuned | 0.3247 | 0.9259 | 0.4808 | 25.7 |
| 2d | Harris tuned | 0.3571 | 0.7407 | 0.4819 | 18.7 |
| 2d | Logistic motion-tuned | 0.3452 | 0.5370 | 0.4203 | 14.0 |
| 2d | Logistic original | 0.3333 | 0.9074 | 0.4876 | 24.5 |
| 2d | ORB tuned | 0.2892 | 0.8889 | 0.4364 | 27.7 |
| 2d | QPP QNN2 fine-tuned | 0.4167 | 0.6481 | 0.5072 | 14.0 |
| 2d | QPP QNN2 original | 0.3333 | 0.8889 | 0.4848 | 24.0 |
| 3d | FAST tuned | 0.3443 | 0.9359 | 0.5034 | 35.3 |
| 3d | Harris tuned | 0.3566 | 0.5897 | 0.4444 | 21.5 |
| 3d | Logistic motion-tuned | 0.3803 | 0.3462 | 0.3624 | 11.8 |
| 3d | Logistic original | 0.3312 | 0.6538 | 0.4397 | 25.7 |
| 3d | ORB tuned | 0.2154 | 0.5385 | 0.3077 | 32.5 |
| 3d | QPP QNN2 fine-tuned | 0.4405 | 0.4744 | 0.4568 | 14.0 |
| 3d | QPP QNN2 original | 0.3125 | 0.5769 | 0.4054 | 24.0 |

The fine-tuned QNN should be read as a motion-domain adapted detector, not a clean-test replacement. It uses the same 2-qubit QPP architecture but updates parameters on motion-domain patches.
