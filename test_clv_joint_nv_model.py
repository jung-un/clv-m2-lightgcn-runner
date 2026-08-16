import numpy as np
import pytest
import torch

from clv_dual_axis_model import DualItemProfile
from clv_joint_nv_model import JointNVLightGCN


def _adj(n_users=3, n_items=4):
    edges = [(0, 0), (0, 1), (1, 1), (1, 2), (2, 3)]
    rows, cols = [], []
    for user, item in edges:
        rows.extend([user, n_users + item])
        cols.extend([n_users + item, user])
    indices = torch.tensor([rows, cols], dtype=torch.long)
    values = torch.ones(len(rows), dtype=torch.float32)
    raw = torch.sparse_coo_tensor(indices, values, (n_users + n_items,) * 2).coalesce()
    degree = torch.sparse.sum(raw, dim=1).to_dense().clamp_min(1.0)
    norm = values / torch.sqrt(degree[indices[0]] * degree[indices[1]])
    return torch.sparse_coo_tensor(indices, norm, raw.shape).coalesce()


def _inputs():
    user_n = np.array([[0.1, 0.2, 0.3], [0.8, 0.7, 0.9], [0.4, 0.5, 0.6]], np.float32)
    user_v = np.array([[0.9, 0.8], [0.2, 0.1], [0.5, 0.6]], np.float32)
    item = DualItemProfile(
        activity=np.array(
            [[0.1, 0.2, 1.0], [0.8, 0.7, 1.0], [0.4, 0.6, 0.0], [0.2, 0.9, 1.0]],
            np.float32,
        ),
        value=np.array(
            [[0.8, 0.9, 0.7, 0.6], [0.2, 0.1, 0.3, 0.4], [0.5, 0.4, 0.6, 0.5], [0.9, 0.7, 0.8, 0.7]],
            np.float32,
        ),
        valid_item=np.ones(4, bool),
        activity_names=("a", "b", "c"),
        value_names=("d", "e", "f", "g"),
    )
    q_n = np.array([0.1, 0.9, 0.5], np.float32)
    q_v = np.array([0.9, 0.1, 0.5], np.float32)
    return user_n, user_v, item, q_n, q_v


def _model(variant="joint_nv", gate_shape="high", layers=1, gamma_init=0.01):
    user_n, user_v, item, q_n, q_v = _inputs()
    return JointNVLightGCN(
        n_users=3,
        n_items=4,
        user_activity=user_n,
        user_value=user_v,
        item_profile=item,
        q_n=q_n,
        q_v=q_v,
        adj=_adj(),
        id_dim=6,
        axis_dim=3,
        hidden_dim=5,
        n_layers=layers,
        variant=variant,
        gate_shape=gate_shape,
        shuffle_seed=42,
        pref_reg=1e-4,
        gamma_init=gamma_init,
    )


def test_joint_embedding_is_propagated_as_one_concatenated_space():
    model = _model(layers=1)
    layer0_u, layer0_i = model.layer0_embeddings()
    final_u, final_i, zero_u, zero_i = model.embeddings()

    assert layer0_u.shape == (3, 12)
    assert layer0_i.shape == (4, 12)
    assert final_u.shape == layer0_u.shape
    assert final_i.shape == layer0_i.shape
    assert not torch.allclose(final_u, layer0_u)
    assert zero_u.shape == (3, 1)
    assert zero_i.shape == (4, 1)


def test_score_level_gamma_scales_both_user_and_item_axis_blocks_symmetrically():
    model = _model(layers=0, gamma_init=0.01)
    user, item = model.layer0_embeddings()
    user_n = user[:, 6:9]
    user_v = user[:, 9:12]
    item_n = item[:, 6:9]
    item_v = item[:, 9:12]

    np.testing.assert_allclose(float(model.gamma_n.detach()), 0.01, rtol=1e-5)
    np.testing.assert_allclose(float(model.gamma_v.detach()), 0.01, rtol=1e-5)
    np.testing.assert_allclose(
        user_n.norm(dim=1).detach().numpy(), [0.02, 0.18, 0.10], rtol=1e-5
    )
    np.testing.assert_allclose(
        user_v.norm(dim=1).detach().numpy(), [0.18, 0.02, 0.10], rtol=1e-5
    )
    np.testing.assert_allclose(
        item_n.norm(dim=1).detach().numpy(), np.full(4, 0.10), rtol=1e-5
    )
    np.testing.assert_allclose(
        item_v.norm(dim=1).detach().numpy(), np.full(4, 0.10), rtol=1e-5
    )


def test_plain_bpr_sends_one_loss_gradient_to_id_n_and_v_paths():
    model = _model()
    model.raw_gamma_n.data.fill_(0.0)
    model.raw_gamma_v.data.fill_(0.0)
    loss, diagnostics = model.bpr_loss(
        torch.tensor([0, 1]),
        torch.tensor([0, 2]),
        torch.tensor([3, 0]),
        None,
        0.0,
        None,
    )
    loss.backward()

    assert diagnostics["bpr"] > 0
    assert model.E_u.weight.grad.abs().sum() > 0
    assert model.activity_user.net[0].weight.grad.abs().sum() > 0
    assert model.activity_item.net[0].weight.grad.abs().sum() > 0
    assert model.value_user.net[0].weight.grad.abs().sum() > 0
    assert model.value_item.net[0].weight.grad.abs().sum() > 0
    assert model.raw_gamma_n.grad.abs() > 0
    assert model.raw_gamma_v.grad.abs() > 0


def test_fixed_gate_and_controls_have_distinct_identifiable_meaning():
    normal = _model("joint_nv", gate_shape="high")
    shuffled = _model("joint_shuffled_user", gate_shape="high")
    constant = _model("joint_constant_user", gate_shape="high")

    assert torch.allclose(normal.gate_n, torch.tensor([0.2, 1.8, 1.0]))
    assert not torch.allclose(normal.user_activity, shuffled.user_activity)
    assert not torch.allclose(normal.gate_n, shuffled.gate_n)
    assert torch.allclose(constant.gate_n, torch.ones(3))
    assert torch.allclose(constant.gate_v, torch.ones(3))
    assert torch.allclose(constant.user_activity[0], constant.user_activity[1])
    assert torch.allclose(constant.user_value[0], constant.user_value[2])


def test_loss_rejects_sample_weights_so_m4_cannot_leak_into_m2():
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
