# Salt-Pepper Sweep: 5D/8D QNN vs MLP

Protocol: clean training, clean-validation threshold selection, fixed thresholds for all salt-pepper levels.

## Trend Summary

- 5D: F1 drop from 0.00 to 0.15 is QNN 0.4387 vs MLP 0.5958; this supports stronger QNN salt-pepper robustness.
- 8D: F1 drop from 0.00 to 0.15 is QNN 0.4193 vs MLP 0.5879; this supports stronger QNN salt-pepper robustness.

## F1 Table

| Salt-pepper | 5D QNN | 5D MLP | 8D QNN | 8D MLP |
| ---: | ---: | ---: | ---: | ---: |
| 0.00 | 0.7890 | 0.9386 | 0.8105 | 0.9265 |
| 0.01 | 0.7726 | 0.8028 | 0.7592 | 0.7898 |
| 0.02 | 0.7430 | 0.6760 | 0.7197 | 0.6497 |
| 0.03 | 0.6842 | 0.5961 | 0.6593 | 0.5736 |
| 0.05 | 0.5592 | 0.4897 | 0.6011 | 0.4551 |
| 0.08 | 0.4443 | 0.4011 | 0.4982 | 0.3824 |
| 0.10 | 0.3847 | 0.3755 | 0.4592 | 0.3589 |
| 0.15 | 0.3503 | 0.3429 | 0.3911 | 0.3386 |
