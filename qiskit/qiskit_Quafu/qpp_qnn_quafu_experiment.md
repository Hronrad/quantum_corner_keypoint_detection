# QPP QNN 转为 Quafu 真机任务的流程与实验结果

本记录对应 2026-06-30 的一次 Quafu `Baihua` 真机批量实验。实验目标是把当前项目中效果最好的 QPP few-qubit QNN 推理线路转成 Quafu 可提交的 OpenQASM 任务，并验证 Quafu 返回的测量 counts 能否接回原 QPP QNN 的经典读出头。

## 使用的模型

- 模型：`qpp_2q_lambda12_linear_L3_Z_ZZ`
- 输入特征：`lambda12 = [lambda1, lambda2]`
- 量子比特数：2
- 数据重上传层数：3
- 编码：每层对两个 qubit 分别执行 `Ry(phi_j), Rz(phi_j)`
- 可训练量子门：每层每个 qubit 一个训练好的 `Rz-Ry-Rz`
- 纠缠：每层一个 `CX q[0], q[1]`
- 读出：由测量 counts 估计 `<Z0>, <Z1>, <Z0Z1>`，再接训练好的线性 head 和 sigmoid

模型参数来自 `qiskit/qpp_qnn_qiskit.py` 中固化的 best QPP checkpoint。训练仍然来自原 PyTorch 版本；这里做的是真机推理和读出验证。

## 转换为 Quafu 任务

1. 从 5D patch 数据集中读取一个样本。
2. 只保留 `[lambda1, lambda2]`，因为该 QPP QNN 的 best 版本只使用结构张量的两个特征值。
3. 使用训练集统计量做归一化：

```text
z = clip((lambda12 - mean) / std, -3, 3)
phi = pi * z / 3
```

4. 用 Qiskit 构建 2-qubit、3-layer 的数据重上传线路。
5. 添加测量：

```text
measure q[0] -> c[0]
measure q[1] -> c[1]
```

6. 用 `qiskit.qasm2.dumps(circuit)` 导出 OpenQASM 2.0。
7. 按 Quafu demo 的格式提交：

```python
from quark import Task

tmgr = Task(token)
task = {
    "chip": "Baihua",
    "name": "QPP_QNN_batch_01_idx308",
    "circuit": qasm,
    "compile": True,
    "shots": 1024,
}
tid = tmgr.run(task)
res = tmgr.result(tid, timeout=60)
counts = res["count"]
```

脚本会优先从环境变量 `QPU_API_TOKEN` / `QUAFU_API_TOKEN` 读取 token；如果没有，则从 `qiskit/README.md` 的 API Token 行读取。输出 JSON、CSV 和 Markdown 中不会保存 token。

## Counts 到 QPP 输出概率

Quafu 真机返回的是 bitstring counts，例如：

```text
{"00": 842, "01": 59, "10": 78, "11": 45}
```

由于 Qiskit bitstring 的右侧是 `c[0]`，而线路中 `q[0] -> c[0]`、`q[1] -> c[1]`，因此：

- `q0` 是 bitstring 的最右位
- `q1` 是 bitstring 的倒数第二位
- 测到 `0` 时，对 Z 观测量贡献 `+1`
- 测到 `1` 时，对 Z 观测量贡献 `-1`

于是可以从 counts 估计：

```text
<Z0>, <Z1>, <Z0Z1>
```

然后复用原 QPP QNN 的训练读出头：

```text
logit = w0 * <Z0> + w1 * <Z1> + w2 * <Z0Z1> + b
p = sigmoid(logit)
```

这一步很关键：真机上执行的是量子线路和测量；分类所需的 logits 和 sigmoid 仍然在经典端用训练好的 head 计算。

## 批量实验设置

运行命令：

```powershell
python qiskit\run_quafu_qpp_qnn_batch.py --submit --selection mixed --n-per-label 4 --shots 1024 --poll-seconds 60 --output-dir outputs\quafu_qpp_qnn_batch
```

样本选择方式：

- 每个类别各 4 个样本
- 每类 2 个 high-confidence 样本
- 每类 2 个 borderline 样本
- 共提交 8 个 Quafu 任务
- 每个任务 `1024` shots

本实验不是完整 test split 评估，而是一次小批量真机链路验证。high-confidence 样本用于验证真机 counts 是否保持正确分类趋势；borderline 样本用于观察有限 shots 和硬件噪声对阈值附近样本的影响。

## 实验结果汇总

- 任务数：8
- 完成并返回 counts：8
- 失败或超时：0
- statevector 在这 8 个样本上的 accuracy：0.625
- Quafu counts 在这 8 个样本上的 accuracy：0.750
- Quafu counts precision / recall / F1：0.667 / 1.000 / 0.800
- Quafu 概率相对 statevector 的平均偏移：+0.0542
- Quafu 概率相对 statevector 的 MAE：0.0637
- 最大绝对偏移：0.1065

需要注意：这 8 个样本是人为混合挑选的展示样本，不应直接作为模型整体性能指标。完整性能仍应参考 full split 的 QPP QNN/VQC 结果。

## 单任务结果

| run | idx | y | 类型 | lambda1 | lambda2 | statevector p | Quafu p | delta | exact/hw pred | counts |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---|---|
| 1 | 308 | 1 | confident | 2.8588 | 1.2191 | 0.9766 | 0.9568 | -0.0198 | 1/1 | 00=842, 01=59, 10=78, 11=45 |
| 2 | 519 | 1 | confident | 2.8364 | 1.2337 | 0.9764 | 0.9584 | -0.0181 | 1/1 | 00=863, 01=42, 10=78, 11=41 |
| 3 | 903 | 1 | borderline | 1.5532 | 0.3667 | 0.4963 | 0.6029 | +0.1065 | 0/1 | 00=353, 01=128, 10=398, 11=145 |
| 4 | 641 | 1 | borderline | 5.8799 | 0.9525 | 0.5213 | 0.6130 | +0.0917 | 1/1 | 00=248, 01=364, 10=377, 11=35 |
| 5 | 462 | 0 | confident | 0.0693 | 0.0000 | 0.0341 | 0.0917 | +0.0576 | 0/0 | 00=139, 01=17, 10=839, 11=29 |
| 6 | 438 | 0 | confident | 0.0440 | 0.0000 | 0.0341 | 0.0812 | +0.0471 | 0/0 | 00=133, 01=6, 10=862, 11=23 |
| 7 | 1408 | 0 | borderline | 6.5167 | 0.2111 | 0.5066 | 0.5843 | +0.0776 | 1/1 | 00=79, 01=422, 10=322, 11=201 |
| 8 | 1264 | 0 | borderline | 1.8445 | 0.4120 | 0.5229 | 0.6137 | +0.0908 | 1/1 | 00=378, 01=92, 10=395, 11=159 |

## 图表

概率对比：

![QPP QNN statevector vs Quafu probability](../outputs/quafu_qpp_qnn_batch/run_20260630_215120/quafu_qpp_qnn_probabilities.png)

测量 counts：

![Quafu bitstring counts](../outputs/quafu_qpp_qnn_batch/run_20260630_215120/quafu_qpp_qnn_counts.png)

混淆矩阵：

![Quafu hardware confusion matrix](../outputs/quafu_qpp_qnn_batch/run_20260630_215120/quafu_qpp_qnn_confusion.png)

## 观察

1. 高置信正样本在 Quafu 上仍然保持高概率，两个样本均预测为 1。
2. 高置信负样本的 Quafu 概率从约 0.034 上升到约 0.08-0.09，但仍低于 0.5，因此分类不变。
3. 临界样本对真机噪声和有限 shots 更敏感。本次实验中，Quafu 概率整体相对 statevector 偏高，导致一个 statevector 低于 0.5 的正样本被推到 0.6029，反而纠正了该样本；两个本来就略高于 0.5 的负样本继续被判为 1。
4. 这说明链路已经跑通：QPP QNN 的量子部分可以作为 Quafu OpenQASM 任务执行，真机 counts 也可以接回原来的 `<Z0>, <Z1>, <Z0Z1>` 读出头。但在展示性能时，应区分“真机链路验证”和“完整模型评估”。

## 输出文件

- 批量实验脚本：`qiskit/run_quafu_qpp_qnn_batch.py`
- 单样本提交脚本：`qiskit/submit_qpp_qnn_quafu.py`
- 本次实验目录：`outputs/quafu_qpp_qnn_batch/run_20260630_215120`
- CSV：`outputs/quafu_qpp_qnn_batch/run_20260630_215120/quafu_qpp_qnn_batch_results.csv`
- JSON：`outputs/quafu_qpp_qnn_batch/run_20260630_215120/quafu_qpp_qnn_batch_results.json`
- 自动报告：`outputs/quafu_qpp_qnn_batch/run_20260630_215120/quafu_qpp_qnn_report.md`
