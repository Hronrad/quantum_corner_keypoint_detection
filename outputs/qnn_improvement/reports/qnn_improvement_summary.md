# QNN Improvement Summary

## Main Improved Model

- Features: 8-D `[Ix, Iy, Ix2, Iy2, IxIy, lambda1, lambda2, R]`
- QNN: L=2, ring entanglement, RyRz, Z+ZZ readout, trainable input scaling
- MLP F1: 0.9453, PR-AUC: 0.9756
- Improved QNN F1: 0.7391, PR-AUC: 0.7909

## Metrics

- Precision: predicted keypoints that are correct; low precision means many false positives.
- Recall: GT keypoints that are detected; low recall means many misses.
- F1: harmonic mean of Precision and Recall.
- PR-AUC: threshold-independent ranking quality under the Precision-Recall curve.

## Test Coverage

The held-out test split is not only ordinary corners. It includes 60 images: 20 L-corner, 20 T-junction, and 20 X-junction images. The task is still binary keypoint detection, not multi-class keypoint type classification.

## ORB Note

ORB performs poorly because this benchmark is sparse synthetic geometry, not textured natural imagery. Its FAST-based detector fires on many high-contrast edge and antialiasing responses; those extra stable points become false positives because the GT only marks geometric junctions.

## Salt-and-pepper Full Validation

| Method | Samples | Positives | Precision | Recall | F1 | PR-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MLP | 1500 | 300 | 0.4708 | 0.9667 | 0.6332 | 0.6832 |
| QNN | 1500 | 300 | 0.5699 | 0.8833 | 0.6928 | 0.5172 |

The QNN has better fixed-threshold F1 under salt-and-pepper in this full held-out run, but the MLP keeps better PR-AUC, so the result should be validated with more random seeds.

## Key Figures

![Ablation](qnn_ablation_results.png)

![Noise](noise_robustness.png)

![Salt-and-pepper full validation](saltpepper_full_validation.png)

![Overlay](improved_comparison_overlay.png)
