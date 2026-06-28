from __future__ import annotations

import pytest

from qpp_corner import qnn_torch


pytestmark = pytest.mark.skipif(not qnn_torch.TORCH_AVAILABLE, reason="PyTorch is not installed")


def _assert_all_trainable_params_have_gradients(model):
    import torch

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"missing gradient for {name}"
            assert torch.isfinite(param.grad).all(), f"non-finite gradient for {name}"


def test_two_qubit_forward_observables_and_backward():
    import torch

    torch.manual_seed(0)
    model = qnn_torch.DataReuploadingQNN2(n_layers=2, encoding="ryrz", entanglement="bidirectional")
    x = torch.randn((5, 2), dtype=torch.float32)
    logits = model(x)
    assert logits.shape == (5,)
    assert torch.isfinite(logits).all()

    obs = model.observables(x)
    assert obs.shape == (5, 3)
    assert torch.isfinite(obs).all()
    assert torch.all(obs <= 1.0 + 1e-5)
    assert torch.all(obs >= -1.0 - 1e-5)

    y = torch.tensor([0, 1, 0, 1, 1], dtype=torch.float32)
    loss = torch.nn.BCEWithLogitsLoss()(logits, y)
    loss.backward()
    _assert_all_trainable_params_have_gradients(model)


def test_one_qubit_forward_observable_and_backward():
    import torch

    torch.manual_seed(1)
    model = qnn_torch.DataReuploadingQNN1(n_layers=3, encoding="ryrz")
    x = torch.randn((6, 1), dtype=torch.float32)
    logits = model(x)
    assert logits.shape == (6,)
    assert torch.isfinite(logits).all()

    obs = model.observables(x)
    assert obs.shape == (6, 1)
    assert torch.isfinite(obs).all()
    assert torch.all(obs <= 1.0 + 1e-5)
    assert torch.all(obs >= -1.0 - 1e-5)

    y = torch.tensor([0, 1, 1, 0, 1, 0], dtype=torch.float32)
    loss = torch.nn.BCEWithLogitsLoss()(logits, y)
    loss.backward()
    _assert_all_trainable_params_have_gradients(model)


def test_expectation_helpers_bound_outputs_on_normalized_states():
    import torch

    torch.manual_seed(2)
    raw = torch.randn((8, 4), dtype=torch.complex64)
    state = raw / torch.linalg.vector_norm(raw, dim=1, keepdim=True)
    z0 = qnn_torch.expectation_z(state, 0, 2)
    z1 = qnn_torch.expectation_z(state, 1, 2)
    zz = qnn_torch.expectation_z0z1(state)
    for value in [z0, z1, zz]:
        assert torch.isfinite(value).all()
        assert torch.all(value <= 1.0 + 1e-5)
        assert torch.all(value >= -1.0 - 1e-5)
