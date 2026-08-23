import inspect

import numpy as np
import pytest
import torch

from clv_gatefree_lowdim_model import GateFreeLowDimNVLightGCN


def _adj(n_users=3, n_items=4):
    edges = [(0, 0), (0, 1), (1, 1), (1, 2), (2, 3)]
    rows, cols = [], []
    for user, item in edges:
        rows.extend([user, n_users + item])
        cols.extend([n_users + item, user])
    indices = torch.tensor([rows, cols], dtype=torch.long)
    values = torch.ones(len(rows), dtype=torch.float32)
    raw = torch.sparse_coo_tensor(
        indices, values, (n_users + n_items,) * 2
    ).coalesce()
    degree = torch.sparse.sum(raw, dim=1).to_dense().clamp_min(1.0)
    norm = values / torch.sqrt(degree[indices[0]] * degree[indices[1]])
    return torch.sparse_coo_tensor(indices, norm, raw.shape).coalesce()


def _model(layers=1, axis_budget=0.1):
    return GateFreeLowDimNVLightGCN(
        n_users=3,
        n_items=4,
        user_activity=np.array(
            [[-1.0, 0.2], [0.0, 0.5], [1.0, 0.8]], np.float32
        ),
        user_value=np.array(
            [[1.0, 0.7], [0.0, 0.5], [-1.0, 0.3]], np.float32
        ),
        user_activity_valid=np.ones(3, bool),
        user_value_valid=np.ones(3, bool),
        q_n=np.array([0.1, 0.5, 0.9], np.float32),
        q_v=np.array([0.9, 0.5, 0.1], np.float32),
        adj=_adj(),
        id_dim=6,
        axis_dim=4,
        hidden_dim=5,
        n_layers=layers,
        axis_budget=axis_budget,
        pref_reg=1e-4,
    )


def test_layer0_keeps_id_and_adds_two_bounded_four_dimensional_axes():
    model = _model(layers=0)
    user, item = model.layer0_embeddings()

    assert user.shape == (3, 14)
    assert item.shape == (4, 14)
    assert torch.all(user[:, 6:].abs() <= np.sqrt(0.1) + 1e-7)
    assert torch.all(item[:, 6:].abs() <= np.sqrt(0.1) + 1e-7)
    torch.testing.assert_close(user[:, 6:10].mean(0), torch.zeros(4), atol=1e-6, rtol=0)
    torch.testing.assert_close(user[:, 10:14].mean(0), torch.zeros(4), atol=1e-6, rtol=0)


def test_model_has_no_item_economic_inputs_gate_or_learned_axis_weight():
    parameters = inspect.signature(GateFreeLowDimNVLightGCN).parameters
    model = _model(layers=0)

    assert "item_profile" not in parameters
    assert not hasattr(model, "gate_n")
    assert not hasattr(model, "gate_v")
    assert not hasattr(model, "sqrt_gamma_n")
    assert not hasattr(model, "sqrt_gamma_v")
    assert model.axis_budget == pytest.approx(0.1)


def test_point_zero_five_budget_scales_each_axis_by_square_root():
    model = _model(layers=0, axis_budget=0.05)
    user, item = model.layer0_embeddings()

    assert torch.all(user[:, 6:].abs() <= np.sqrt(0.05) + 1e-7)
    assert torch.all(item[:, 6:].abs() <= np.sqrt(0.05) + 1e-7)
    assert model.representation_diagnostics()["axis_budget"] == pytest.approx(0.05)


def test_one_plain_bpr_loss_trains_id_user_axes_and_item_responses():
    model = _model(layers=1)
    loss, diagnostics = model.bpr_loss(
        torch.tensor([0, 1]),
        torch.tensor([0, 2]),
        torch.tensor([3, 0]),
        None,
        0.0,
        None,
    )
    loss.backward()

    assert diagnostics["objective"] == "plain_bpr"
    assert model.E_u.weight.grad.abs().sum() > 0
    assert model.E_i.weight.grad.abs().sum() > 0
    assert model.activity_user.net[0].weight.grad.abs().sum() > 0
    assert model.value_user.net[0].weight.grad.abs().sum() > 0
    assert model.activity_item.weight.grad.abs().sum() > 0
    assert model.value_item.weight.grad.abs().sum() > 0


def test_m4_sample_weights_are_rejected():
    model = _model()
    with pytest.raises(ValueError, match="M4"):
        model.bpr_loss(
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([3]),
            None,
            0.0,
            torch.tensor([2.0]),
        )
