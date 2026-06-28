"""Exact differentiable one- and two-qubit data-reuploading QNNs in PyTorch."""

from __future__ import annotations

import math

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover - exercised in environments without torch.
    torch = None
    nn = None


TORCH_AVAILABLE = torch is not None


def require_torch() -> None:
    if torch is None:
        raise ImportError("PyTorch is required for qpp_corner.qnn_torch. Install with `pip install torch`.")


if torch is not None:

    def _complex(theta):
        return theta.to(dtype=torch.complex64)


    def Ry(theta):
        theta = torch.as_tensor(theta)
        c = torch.cos(theta / 2.0)
        s = torch.sin(theta / 2.0)
        mat = torch.zeros(theta.shape + (2, 2), dtype=torch.complex64, device=theta.device)
        mat[..., 0, 0] = _complex(c)
        mat[..., 0, 1] = _complex(-s)
        mat[..., 1, 0] = _complex(s)
        mat[..., 1, 1] = _complex(c)
        return mat


    def Rz(theta):
        theta = torch.as_tensor(theta)
        mat = torch.zeros(theta.shape + (2, 2), dtype=torch.complex64, device=theta.device)
        mat[..., 0, 0] = torch.exp(_complex(-0.5 * theta) * 1j)
        mat[..., 1, 1] = torch.exp(_complex(0.5 * theta) * 1j)
        return mat


    def _broadcast_gate(gate, batch_size: int):
        if gate.dim() == 2:
            return gate.unsqueeze(0).expand(batch_size, -1, -1)
        return gate


    def apply_single_qubit_gate(state, gate, qubit: int, n_qubits: int):
        batch = state.shape[0]
        gate = _broadcast_gate(gate, batch)
        tensor = state.reshape((batch,) + (2,) * n_qubits)
        axis = 1 + qubit
        tensor = tensor.movedim(axis, -1)
        updated = torch.einsum("bij,b...j->b...i", gate, tensor)
        updated = updated.movedim(-1, axis)
        return updated.reshape(batch, 2**n_qubits)


    def apply_cnot(state, control: int, target: int, n_qubits: int = 2):
        dim = 2**n_qubits
        device = state.device
        indices = torch.arange(dim, device=device)
        control_mask = 1 << (n_qubits - 1 - control)
        target_mask = 1 << (n_qubits - 1 - target)
        mapped = torch.where((indices & control_mask) != 0, indices ^ target_mask, indices)
        out = torch.empty_like(state)
        out[:, mapped] = state
        return out


    def expectation_z(state, qubit: int, n_qubits: int):
        dim = 2**n_qubits
        indices = torch.arange(dim, device=state.device)
        mask = 1 << (n_qubits - 1 - qubit)
        signs = torch.where((indices & mask) == 0, 1.0, -1.0).to(state.device)
        probs = (state.conj() * state).real
        return probs @ signs


    def expectation_z0z1(state):
        dim = 4
        indices = torch.arange(dim, device=state.device)
        bit0 = torch.where((indices & 0b10) == 0, 1.0, -1.0).to(state.device)
        bit1 = torch.where((indices & 0b01) == 0, 1.0, -1.0).to(state.device)
        probs = (state.conj() * state).real
        return probs @ (bit0 * bit1)


    def zero_state(batch_size: int, n_qubits: int, device):
        state = torch.zeros((batch_size, 2**n_qubits), dtype=torch.complex64, device=device)
        state[:, 0] = 1.0 + 0.0j
        return state


    class DataReuploadingQNN2(nn.Module):
        """Two-qubit data-reuploading QNN with affine readout."""

        def __init__(
            self,
            n_layers: int = 2,
            *,
            encoding: str = "ryrz",
            entanglement: str = "linear_01",
            readout: str = "z_z_zz",
        ) -> None:
            super().__init__()
            self.n_layers = int(n_layers)
            self.encoding = encoding
            self.entanglement = entanglement
            self.readout = readout
            self.theta = nn.Parameter(0.05 * torch.randn(self.n_layers, 2, 3))
            self.head = nn.Linear(3, 1)

        def _encode(self, state, phi):
            for qubit in range(2):
                angle = phi[:, qubit]
                if self.encoding == "ry":
                    state = apply_single_qubit_gate(state, Ry(angle), qubit, 2)
                elif self.encoding == "ryrz":
                    state = apply_single_qubit_gate(state, Ry(angle), qubit, 2)
                    state = apply_single_qubit_gate(state, Rz(angle), qubit, 2)
                elif self.encoding == "qpp_z":
                    state = apply_single_qubit_gate(state, Rz(angle), qubit, 2)
                    state = apply_single_qubit_gate(state, Ry(torch.full_like(angle, math.pi / 2)), qubit, 2)
                else:
                    raise ValueError(f"Unknown encoding: {self.encoding}")
            return state

        def _trainable_rotations(self, state, layer: int):
            for qubit in range(2):
                a, b, c = self.theta[layer, qubit]
                state = apply_single_qubit_gate(state, Rz(a), qubit, 2)
                state = apply_single_qubit_gate(state, Ry(b), qubit, 2)
                state = apply_single_qubit_gate(state, Rz(c), qubit, 2)
            return state

        def _entangle(self, state):
            if self.entanglement == "none":
                return state
            if self.entanglement == "linear_01":
                return apply_cnot(state, 0, 1, 2)
            if self.entanglement == "bidirectional":
                state = apply_cnot(state, 0, 1, 2)
                return apply_cnot(state, 1, 0, 2)
            raise ValueError(f"Unknown entanglement: {self.entanglement}")

        def statevector(self, phi):
            if phi.shape[-1] != 2:
                raise ValueError(f"2q QNN expects two angle features, got shape {tuple(phi.shape)}")
            phi = phi.to(dtype=torch.float32)
            state = zero_state(phi.shape[0], 2, phi.device)
            for layer in range(self.n_layers):
                state = self._encode(state, phi)
                state = self._trainable_rotations(state, layer)
                state = self._entangle(state)
            return state

        def observables(self, phi):
            state = self.statevector(phi)
            z0 = expectation_z(state, 0, 2)
            z1 = expectation_z(state, 1, 2)
            zz = expectation_z0z1(state)
            return torch.stack([z0, z1, zz], dim=1).to(dtype=torch.float32)

        def forward(self, phi):
            obs = self.observables(phi)
            return self.head(obs).squeeze(-1)


    class DataReuploadingQNN1(nn.Module):
        """One-qubit scalar data-reuploading QNN with affine readout."""

        def __init__(
            self,
            n_layers: int = 2,
            *,
            encoding: str = "rz",
            input_dim: int = 1,
            learnable_projection: bool = False,
        ) -> None:
            super().__init__()
            self.n_layers = int(n_layers)
            self.encoding = encoding
            self.input_dim = int(input_dim)
            self.learnable_projection = bool(learnable_projection)
            self.project = nn.Linear(self.input_dim, 1) if self.learnable_projection else None
            self.theta = nn.Parameter(0.05 * torch.randn(self.n_layers, 3))
            self.head = nn.Linear(1, 1)

        def _scalar(self, x):
            if self.project is not None:
                return self.project(x.to(dtype=torch.float32)).squeeze(-1)
            if x.shape[-1] != 1:
                raise ValueError("1q QNN expects one scalar angle unless learnable_projection=True.")
            return x[:, 0].to(dtype=torch.float32)

        def _encode(self, state, t):
            if self.encoding == "rz":
                return apply_single_qubit_gate(state, Rz(t), 0, 1)
            if self.encoding == "ryrz":
                state = apply_single_qubit_gate(state, Ry(t), 0, 1)
                return apply_single_qubit_gate(state, Rz(t), 0, 1)
            raise ValueError(f"Unknown 1q encoding: {self.encoding}")

        def statevector(self, x):
            x = x.to(dtype=torch.float32)
            t = self._scalar(x)
            state = zero_state(x.shape[0], 1, x.device)
            for layer in range(self.n_layers):
                state = self._encode(state, t)
                a, b, c = self.theta[layer]
                state = apply_single_qubit_gate(state, Rz(a), 0, 1)
                state = apply_single_qubit_gate(state, Ry(b), 0, 1)
                state = apply_single_qubit_gate(state, Rz(c), 0, 1)
            return state

        def observables(self, x):
            state = self.statevector(x)
            z = expectation_z(state, 0, 1).to(dtype=torch.float32).unsqueeze(1)
            return z

        def forward(self, x):
            z = self.observables(x)
            return self.head(z).squeeze(-1)

else:

    def Ry(theta):  # type: ignore[no-redef]
        require_torch()


    def Rz(theta):  # type: ignore[no-redef]
        require_torch()


    def apply_single_qubit_gate(state, gate, qubit: int, n_qubits: int):  # type: ignore[no-redef]
        require_torch()


    def apply_cnot(state, control: int, target: int, n_qubits: int = 2):  # type: ignore[no-redef]
        require_torch()


    def expectation_z(state, qubit: int, n_qubits: int):  # type: ignore[no-redef]
        require_torch()


    def expectation_z0z1(state):  # type: ignore[no-redef]
        require_torch()


    class DataReuploadingQNN2:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            require_torch()


    class DataReuploadingQNN1:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            require_torch()
