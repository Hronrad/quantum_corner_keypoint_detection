# QPP QNN to Quafu real-device experiment

- Created: 2026-06-30T21:53:02
- Model: `qpp_2q_lambda12_linear_L3_Z_ZZ`
- Backend chip: `Baihua`
- Shots per task: `1024`
- Dataset split: `test`
- Sample selection: `mixed`, `n_per_label=4`
- CSV: `quafu_qpp_qnn_batch_results.csv`
- JSON: `quafu_qpp_qnn_batch_results.json`

## Conversion path

1. Read one patch feature vector from the 5D dataset.
2. Keep only `lambda12 = [lambda1, lambda2]`, because the selected QPP QNN was trained on these two structure-tensor eigenvalue features.
3. Apply the train-split normalizer: `z = clip((lambda12 - mean) / std, -3, 3)` and map to circuit angles with `phi = pi * z / 3`.
4. Build the 2-qubit, 3-layer data-reuploading QPP circuit in Qiskit. Each layer applies `Ry(phi_j), Rz(phi_j)` on both qubits, then the trained `Rz-Ry-Rz` rotations, then `CX q[0], q[1]`.
5. Add measurements `q[0] -> c[0]` and `q[1] -> c[1]`, then export the circuit with `qiskit.qasm2.dumps(circuit)`.
6. Submit the OpenQASM string through Quafu's `quark.Task` API with a task dictionary containing `chip`, `name`, `circuit`, `compile`, and `shots`.
7. Convert returned bitstring counts back to observables. With Qiskit-style bitstrings, `q0` is the rightmost bit and `q1` is the second rightmost bit. A measured `0` contributes `+1` to Z and `1` contributes `-1`.
8. Reuse the trained affine readout head: `logit = w0 <Z0> + w1 <Z1> + w2 <Z0Z1> + b`, then `probability = sigmoid(logit)`.

## Aggregate result

- Selected tasks: `8`
- Finished with counts: `8`
- Failed or timed out: `0`
- Statevector accuracy on selected samples: `0.625`
- Quafu-count accuracy on finished samples: `0.750`
- Quafu-count precision / recall / F1: `0.667` / `1.000` / `0.800`
- Probability delta mean: `+0.0542`
- Probability delta MAE: `0.0637`
- Probability delta max abs: `0.1065`

## Per-task results

| run | idx | y | bucket | lambda1 | lambda2 | exact p | Quafu p | delta | pred exact/hw | counts |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---|---|
| 1 | 308 | 1 | confident | 2.8588 | 1.2191 | 0.9766 | 0.9568 | -0.0198 | 1/1 | 00=842, 01=59, 10=78, 11=45 |
| 2 | 519 | 1 | confident | 2.8364 | 1.2337 | 0.9764 | 0.9584 | -0.0181 | 1/1 | 00=863, 01=42, 10=78, 11=41 |
| 3 | 903 | 1 | borderline | 1.5532 | 0.3667 | 0.4963 | 0.6029 | +0.1065 | 0/1 | 00=353, 01=128, 10=398, 11=145 |
| 4 | 641 | 1 | borderline | 5.8799 | 0.9525 | 0.5213 | 0.6130 | +0.0917 | 1/1 | 00=248, 01=364, 10=377, 11=35 |
| 5 | 462 | 0 | confident | 0.0693 | 0.0000 | 0.0341 | 0.0917 | +0.0576 | 0/0 | 00=139, 01=17, 10=839, 11=29 |
| 6 | 438 | 0 | confident | 0.0440 | 0.0000 | 0.0341 | 0.0812 | +0.0471 | 0/0 | 00=133, 01=6, 10=862, 11=23 |
| 7 | 1408 | 0 | borderline | 6.5167 | 0.2111 | 0.5066 | 0.5843 | +0.0776 | 1/1 | 00=79, 01=422, 10=322, 11=201 |
| 8 | 1264 | 0 | borderline | 1.8445 | 0.4120 | 0.5229 | 0.6137 | +0.0908 | 1/1 | 00=378, 01=92, 10=395, 11=159 |

## Figures

![Statevector vs Quafu probability](quafu_qpp_qnn_probabilities.png)

![Quafu bitstring counts](quafu_qpp_qnn_counts.png)

![Quafu confusion matrix](quafu_qpp_qnn_confusion.png)

## Reading the result

The high-confidence samples are mainly a hardware sanity check: the Quafu-count probability should stay on the same side of the 0.5 decision threshold as the exact statevector result. Borderline samples are deliberately harder; small device noise, compilation differences, and finite-shot fluctuation can move them across the threshold.

This experiment therefore checks two things at once: whether the QPP-QNN circuit can be submitted as a Quafu task, and whether the classical QPP readout head can consume real-device bitstring counts without changing the trained model.