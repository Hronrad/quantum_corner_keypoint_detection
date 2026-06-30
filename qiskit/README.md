# Qiskit port of the QPP few-qubit QNN

This folder contains a Qiskit forward-pass implementation of the current
best-clean-F1 QPP model used in the project:

- model: `qpp_2q_lambda12_linear_L3_Z_ZZ`
- input: `lambda12 = [lambda1, lambda2]`
- qubits: 2
- layers: 3
- circuit: `Ry/Rz` data re-uploading, trainable `Rz-Ry-Rz`, `CNOT(q0 -> q1)`
- readout: `<Z0>`, `<Z1>`, `<Z0 Z1>`, followed by the trained affine head

The training path remains the PyTorch exact differentiable statevector code.
The Qiskit version is meant for circuit inspection, finite-shot inference,
transpilation, and real-backend execution.

## Why it can run on NISQ hardware

The circuit uses only 2 qubits, shallow single-qubit rotations, one CNOT per
layer, computational-basis measurement, and classical post-processing of Pauli-Z
expectations.  Even at `L=3`, this is still a small NISQ-friendly circuit that
can be transpiled to IBM/Qiskit-compatible backends.  On hardware, the
probability is estimated from finite-shot counts rather than exact statevectors.

## Quick local check

```bash
python3 qiskit/qpp_qnn_qiskit.py
```

This prints the circuit and one exact statevector probability.  It requires
`qiskit`; finite-shot local execution also requires `qiskit-aer`.

## Real backend sketch

```python
import sys
from pathlib import Path
from qiskit_ibm_runtime import QiskitRuntimeService

sys.path.insert(0, str(Path("qiskit").resolve()))
from qpp_qnn_qiskit import predict_probability_backend

service = QiskitRuntimeService()
backend = service.least_busy(operational=True, simulator=False, min_num_qubits=2)

lambda12 = [1.2, 0.5]
probability, counts = predict_probability_backend(lambda12, backend, shots=2048)
print(probability, counts)
```

For a full hardware experiment, run a batch of patch features, then apply the
same threshold and NMS logic used by the project demo scripts.
