"""Qiskit forward-pass port of the current best QPP few-qubit QNN.

This file mirrors the best clean-test-F1 PyTorch exact differentiable
statevector model used in
``outputs/qpp_next_struct_lambda12_linear_L3/best_model.pt``:

    feature set: lambda12 = [lambda1, lambda2]
    qubits:      2
    layers:      3
    encoding:    Ry(phi), Rz(phi) data re-uploading
    trainable:   Rz-Ry-Rz on each qubit
    entangle:    CNOT(q0 -> q1)
    readout:     <Z0>, <Z1>, <Z0 Z1> + affine head + sigmoid

Training is still done by the PyTorch implementation.  This Qiskit version is
for circuit inspection, finite-shot inference, transpilation, and real NISQ
backend runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, pi
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


BEST_QPP_LAMBDA12_L3 = {
    "name": "qpp_2q_lambda12_linear_L3_Z_ZZ",
    "feature_set": "lambda12",
    "n_qubits": 2,
    "n_layers": 3,
    "encoding": "ryrz",
    "entanglement": "linear_01",
    "readout": "z_z_zz",
    "theta": [
        [
            [0.5120954513549805, 0.25936442613601685, 0.048623740673065186],
            [0.12168067693710327, -0.7518175840377808, 0.7989760637283325],
        ],
        [
            [-0.15672539174556732, -0.02371535263955593, -0.15216878056526184],
            [-0.2911463975906372, -0.5240213871002197, 0.49349266290664673],
        ],
        [
            [-0.015621397644281387, 0.6927890181541443, 0.16573002934455872],
            [0.5262488126754761, -0.3649025857448578, 0.03523566946387291],
        ],
    ],
    "head_weight": [-0.7785586714744568, 2.079404592514038, 1.6216249465942383],
    "head_bias": 0.9510509967803955,
    "normalizer": {
        "feature_names": ["lambda1", "lambda2"],
        "clip": 3.0,
        "mean": [1.2037864284740631, 0.4748739992054601],
        "std": [1.9432498635425994, 1.053883332853748],
    },
}

REFERENCE_QPP_LAMBDA12_L2 = {
    "name": "qpp_2q_lambda12_L2",
    "feature_set": "lambda12",
    "n_qubits": 2,
    "n_layers": 2,
    "encoding": "ryrz",
    "entanglement": "linear_01",
    "readout": "z_z_zz",
    "theta": [
        [
            [0.06461842358112335, 0.7566940188407898, 0.42490383982658386],
            [-0.009072994813323021, 1.729751706123352, -0.3706171214580536],
        ],
        [
            [0.9580743312835693, 0.9252015352249146, -0.04179465025663376],
            [1.0074002742767334, -0.4577445685863495, -0.019011465832591057],
        ],
    ],
    "head_weight": [-2.4366261959075928, -2.6549079418182373, -2.5185344219207764],
    "head_bias": 1.682659387588501,
}


@dataclass(frozen=True)
class Lambda12Normalizer:
    """Train-split normalizer and angle map used by the QPP QNN."""

    mean: np.ndarray
    std: np.ndarray
    clip: float = 3.0

    @classmethod
    def from_best_model(cls) -> "Lambda12Normalizer":
        payload = BEST_QPP_LAMBDA12_L3["normalizer"]
        return cls(
            mean=np.asarray(payload["mean"], dtype=np.float64),
            std=np.asarray(payload["std"], dtype=np.float64),
            clip=float(payload["clip"]),
        )

    def transform(self, lambda12: Sequence[float] | np.ndarray) -> np.ndarray:
        x = np.asarray(lambda12, dtype=np.float64)
        z = (x - self.mean) / self.std
        return np.clip(z, -self.clip, self.clip)

    def to_angles(self, lambda12: Sequence[float] | np.ndarray) -> np.ndarray:
        z = self.transform(lambda12)
        return (pi * z / self.clip).astype(np.float64)


@dataclass(frozen=True)
class QPPQNNWeights:
    """Parameters for the 2-qubit QPP data-reuploading QNN."""

    theta: np.ndarray
    head_weight: np.ndarray
    head_bias: float

    @classmethod
    def from_best_model(cls) -> "QPPQNNWeights":
        return cls(
            theta=np.asarray(BEST_QPP_LAMBDA12_L3["theta"], dtype=np.float64),
            head_weight=np.asarray(BEST_QPP_LAMBDA12_L3["head_weight"], dtype=np.float64),
            head_bias=float(BEST_QPP_LAMBDA12_L3["head_bias"]),
        )

    @classmethod
    def from_torch_checkpoint(cls, checkpoint_path: str | Path) -> "QPPQNNWeights":
        """Load weights from the PyTorch checkpoint produced by the training code."""

        import torch

        state = torch.load(str(checkpoint_path), map_location="cpu")
        return cls(
            theta=state["theta"].detach().cpu().numpy().astype(np.float64),
            head_weight=state["head.weight"].detach().cpu().numpy().reshape(-1).astype(np.float64),
            head_bias=float(state["head.bias"].detach().cpu().numpy().reshape(-1)[0]),
        )


def sigmoid(value: float) -> float:
    if value >= 0:
        z = exp(-value)
        return 1.0 / (1.0 + z)
    z = exp(value)
    return z / (1.0 + z)


def build_qpp_qnn_circuit(
    phi: Sequence[float] | np.ndarray,
    weights: QPPQNNWeights | None = None,
    *,
    entanglement: str = "linear_01",
    measure: bool = False,
):
    """Build the Qiskit circuit for one normalized ``lambda12`` sample.

    ``phi`` must already be angle-mapped.  Use ``Lambda12Normalizer.to_angles``
    for raw ``[lambda1, lambda2]`` features.
    """

    from qiskit import QuantumCircuit

    params = weights or QPPQNNWeights.from_best_model()
    theta = np.asarray(params.theta, dtype=np.float64)
    phi_arr = np.asarray(phi, dtype=np.float64).reshape(-1)
    if phi_arr.shape[0] != 2:
        raise ValueError(f"Expected two angle features, got shape {phi_arr.shape}.")

    circuit = QuantumCircuit(2, 2 if measure else 0)
    for layer in range(theta.shape[0]):
        for qubit in range(2):
            circuit.ry(float(phi_arr[qubit]), qubit)
            circuit.rz(float(phi_arr[qubit]), qubit)
        for qubit in range(2):
            a, b, c = theta[layer, qubit]
            circuit.rz(float(a), qubit)
            circuit.ry(float(b), qubit)
            circuit.rz(float(c), qubit)
        if entanglement in {"linear", "linear_01"}:
            circuit.cx(0, 1)
        elif entanglement == "bidirectional":
            circuit.cx(0, 1)
            circuit.cx(1, 0)
        elif entanglement == "none":
            pass
        else:
            raise ValueError(f"Unknown entanglement mode: {entanglement}")

    if measure:
        circuit.measure([0, 1], [0, 1])
    return circuit


def observables_from_statevector(circuit) -> np.ndarray:
    """Compute <Z0>, <Z1>, and <Z0 Z1> exactly from a Qiskit statevector."""

    from qiskit.quantum_info import Statevector

    state = Statevector.from_instruction(circuit)
    probs = np.asarray(state.probabilities(), dtype=np.float64)
    z0 = 0.0
    z1 = 0.0
    zz = 0.0
    for basis_index, prob in enumerate(probs):
        q0 = (basis_index >> 0) & 1
        q1 = (basis_index >> 1) & 1
        v0 = 1.0 if q0 == 0 else -1.0
        v1 = 1.0 if q1 == 0 else -1.0
        z0 += prob * v0
        z1 += prob * v1
        zz += prob * v0 * v1
    return np.asarray([z0, z1, zz], dtype=np.float64)


def observables_from_counts(counts: dict[str, int]) -> np.ndarray:
    """Estimate <Z0>, <Z1>, and <Z0 Z1> from measured bitstring counts."""

    shots = float(sum(counts.values()))
    if shots <= 0:
        raise ValueError("Counts are empty.")

    z0 = 0.0
    z1 = 0.0
    zz = 0.0
    for bitstring, count in counts.items():
        compact = bitstring.replace(" ", "")
        if len(compact) < 2:
            raise ValueError(f"Expected two measured bits, got {bitstring!r}.")
        # Qiskit returns classical bitstrings as c[n-1]...c[0].  The circuit
        # measures q0 -> c0 and q1 -> c1, so q0 is the rightmost bit.
        q0 = int(compact[-1])
        q1 = int(compact[-2])
        v0 = 1.0 if q0 == 0 else -1.0
        v1 = 1.0 if q1 == 0 else -1.0
        weight = count / shots
        z0 += weight * v0
        z1 += weight * v1
        zz += weight * v0 * v1
    return np.asarray([z0, z1, zz], dtype=np.float64)


def probability_from_observables(obs: Sequence[float], weights: QPPQNNWeights | None = None) -> float:
    params = weights or QPPQNNWeights.from_best_model()
    obs_arr = np.asarray(obs, dtype=np.float64)
    logit = float(np.dot(params.head_weight, obs_arr) + params.head_bias)
    return sigmoid(logit)


def predict_probability_statevector(
    lambda12: Sequence[float] | np.ndarray,
    *,
    normalizer: Lambda12Normalizer | None = None,
    weights: QPPQNNWeights | None = None,
) -> float:
    """Exact simulator prediction for one raw ``[lambda1, lambda2]`` feature."""

    norm = normalizer or Lambda12Normalizer.from_best_model()
    params = weights or QPPQNNWeights.from_best_model()
    phi = norm.to_angles(lambda12)
    circuit = build_qpp_qnn_circuit(phi, params, measure=False)
    obs = observables_from_statevector(circuit)
    return probability_from_observables(obs, params)


def predict_probability_backend(
    lambda12: Sequence[float] | np.ndarray,
    backend=None,
    *,
    shots: int = 2048,
    normalizer: Lambda12Normalizer | None = None,
    weights: QPPQNNWeights | None = None,
) -> tuple[float, dict[str, int]]:
    """Finite-shot prediction using a Qiskit backend.

    Pass a real backend object, for example from ``QiskitRuntimeService``.  If no
    backend is supplied, the function tries ``qiskit_aer.AerSimulator``.
    """

    from qiskit import transpile

    if backend is None:
        from qiskit_aer import AerSimulator

        backend = AerSimulator()

    norm = normalizer or Lambda12Normalizer.from_best_model()
    params = weights or QPPQNNWeights.from_best_model()
    phi = norm.to_angles(lambda12)
    circuit = build_qpp_qnn_circuit(phi, params, measure=True)
    transpiled = transpile(circuit, backend=backend, optimization_level=1)
    job = backend.run(transpiled, shots=int(shots))
    counts = job.result().get_counts()
    obs = observables_from_counts(counts)
    return probability_from_observables(obs, params), counts


def build_batch_circuits(
    lambda12_batch: Iterable[Sequence[float]],
    *,
    measure: bool = True,
    normalizer: Lambda12Normalizer | None = None,
    weights: QPPQNNWeights | None = None,
) -> list:
    """Build one measured/unmeasured circuit per raw ``lambda12`` feature."""

    norm = normalizer or Lambda12Normalizer.from_best_model()
    params = weights or QPPQNNWeights.from_best_model()
    circuits = []
    for lambda12 in lambda12_batch:
        phi = norm.to_angles(lambda12)
        circuits.append(build_qpp_qnn_circuit(phi, params, measure=measure))
    return circuits


def main() -> None:
    weights = QPPQNNWeights.from_best_model()
    normalizer = Lambda12Normalizer.from_best_model()
    example_lambda12 = np.asarray(normalizer.mean, dtype=np.float64)
    phi = normalizer.to_angles(example_lambda12)
    circuit = build_qpp_qnn_circuit(phi, weights, measure=True)
    print(circuit.draw(output="text"))
    print("Example probability:", predict_probability_statevector(example_lambda12))


if __name__ == "__main__":
    main()
