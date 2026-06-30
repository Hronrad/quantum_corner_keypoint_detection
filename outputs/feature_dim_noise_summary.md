# 5D/8D QNN Noise Robustness Summary

Protocol: clean training, clean-validation threshold selection, noisy held-out test evaluation.

## Main Answers

- 5D QNN is more stable than 5D MLP by F1 drop from clean to salt-pepper (0.1048 vs 0.3425).
- 8D QNN is more stable than 8D MLP by F1 drop from clean to salt-pepper (0.1512 vs 0.3529).
- Moving from 5D to 8D does not improve QNN salt-pepper F1 (0.6842 -> 0.6593); clean F1 changes from 0.7890 to 0.8105.

## Clean and Salt-Pepper Snapshot

| Feature | Method | Clean F1 | Salt-Pepper F1 | Clean PR-AUC | Salt-Pepper PR-AUC |
| --- | --- | ---: | ---: | ---: | ---: |
| 5D | QNN | 0.7890 | 0.6842 | 0.7271 | 0.5664 |
| 5D | MLP | 0.9386 | 0.5961 | 0.9814 | 0.7111 |
| 8D | QNN | 0.8105 | 0.6593 | 0.8289 | 0.5198 |
| 8D | MLP | 0.9265 | 0.5736 | 0.9756 | 0.6832 |
