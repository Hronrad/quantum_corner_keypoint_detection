# Day 2 Progress Summary

已完成：
1. 合成角点/交点数据
2. patch 采样
3. classical baseline，包括 Harris / FAST / ORB
4. 5-D feature extraction: Ix, Iy, lambda1, lambda2, R
5. QNN 接入同一特征接口
6. QNN 第一轮训练与可视化

下一步：
1. 噪声鲁棒性实验
2. QNN 消融实验
3. demo 整合

| Method | Input | Precision | Recall | F1 | PR-AUC |
| --- | --- | ---: | ---: | ---: | ---: |
| Harris | image | 0.1652 | 0.9500 | 0.2815 | 0.8953 |
| FAST | image | 0.0789 | 0.7833 | 0.1433 | 0.7925 |
| ORB | image | 0.0412 | 1.0000 | 0.0791 | 0.7759 |
| MLP | same features | 0.9347 | 0.9067 | 0.9205 | 0.9749 |
| QNN | same features | 0.5238 | 0.6875 | 0.5946 | 0.4672 |

QNN subset: {"train_samples": 160, "train_positives": 32, "val_samples": 80, "val_positives": 16, "test_samples": 80, "test_positives": 16}
