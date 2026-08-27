import numpy as np
import pandas as pd
import pytest
import torch

from clv_fixed_composition_nv_model import (
    FixedCompositionNVLightGCN,
    ItemAxisAffinity,
    build_popularity_controlled_item_affinities,
    fixed_axis_composition,
)


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
    normalized = values / torch.sqrt(degree[indices[0]] * degree[indices[1]])
    return torch.sparse_coo_tensor(indices, normalized, raw.shape).coalesce()


def _model():
    item = ItemAxisAffinity(
        activity=np.array([[-1.0], [0.5], [1.0], [-0.5]], np.float32),
        value=np.array([[0.5], [-1.0], [1.0], [0.2]], np.float32),
        activity_valid=np.ones(4, bool),
        value_valid=np.ones(4, bool),
        diagnostics={},
    )
    return FixedCompositionNVLightGCN(
        n_users=3,
        n_items=4,
        user_activity=np.array(
            [[0.1, 0.2], [0.8, 0.7], [0.4, 0.5]], np.float32
        ),
        user_value=np.array([[0.9], [0.2], [0.5]], np.float32),
        user_activity_valid=np.ones(3, bool),
        user_value_valid=np.ones(3, bool),
        item_affinity=item,
        q_n=np.array([0.9, 0.1, 0.5], np.float32),
        q_v=np.array([0.1, 0.9, 0.5], np.float32),
        adj=_adj(),
        id_dim=6,
        axis_dim=2,
        hidden_dim=4,
        n_layers=1,
        rho=0.05,
        pref_reg=1e-3,
    )


def test_fixed_composition_allocates_one_nonnegative_budget_per_valid_user():
    pi_n, pi_v = fixed_axis_composition(
        np.array([0.9, 0.1, 0.5, 0.5]),
        np.array([0.1, 0.9, 0.5, 0.5]),
        np.array([True, True, True, False]),
        np.array([True, True, False, False]),
    )

    assert pi_n[0] > pi_v[0]
    assert pi_n[1] < pi_v[1]
    assert pi_n[2] == pytest.approx(1.0)
    assert pi_v[2] == pytest.approx(0.0)
    assert pi_n[3] == pytest.approx(0.0)
    assert pi_v[3] == pytest.approx(0.0)
    np.testing.assert_allclose((pi_n + pi_v)[:3], 1.0)


def test_item_affinity_uses_unique_buyers_and_removes_degree_direction():
    rows = []
    categories = [0, 0, 1, 1, 1, 0]
    buyers = [
        [0],
        [0, 1],
        [0, 1, 2],
        [1, 2, 3, 4],
        [0, 1, 2, 3, 4],
        [2, 3, 4],
    ]
    for item, users in enumerate(buyers):
        for user in users:
            rows.append({"u_idx": user, "i_idx": item, "cat_idx": categories[item]})
    train = pd.DataFrame(rows)
    duplicated = pd.concat([train, train.iloc[[0, 0, 0]]], ignore_index=True)
    kwargs = {
        "n_items": 6,
        "q_n": np.array([0.1, 0.3, 0.5, 0.7, 0.9]),
        "q_v": np.array([0.9, 0.7, 0.5, 0.3, 0.1]),
        "user_activity_valid": np.ones(5, bool),
        "user_value_valid": np.ones(5, bool),
    }

    first = build_popularity_controlled_item_affinities(train, **kwargs)
    second = build_popularity_controlled_item_affinities(duplicated, **kwargs)

    np.testing.assert_allclose(first.activity, second.activity)
    np.testing.assert_allclose(first.value, second.value)
    assert first.activity.shape == (6, 1)
    assert first.value.shape == (6, 1)
    assert first.activity[first.activity_valid, 0].mean() == pytest.approx(0.0, abs=1e-6)
    assert first.activity[first.activity_valid, 0].std() == pytest.approx(1.0, abs=1e-6)
    assert first.value[first.value_valid, 0].mean() == pytest.approx(0.0, abs=1e-6)
    assert first.value[first.value_valid, 0].std() == pytest.approx(1.0, abs=1e-6)


def test_layer0_is_id_n_v_with_fixed_rho_and_normalized_axis_blocks():
    model = _model()
    user, item = model.layer0_embeddings()

    assert user.shape == (3, 10)
    assert item.shape == (4, 10)
    assert model.total_dim == 10
    assert "rho" not in dict(model.named_parameters())
    np.testing.assert_allclose(
        user[:, 6:8].norm(dim=1).detach().numpy(),
        np.sqrt(0.05) * model.pi_n.detach().numpy(),
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        user[:, 8:10].norm(dim=1).detach().numpy(),
        np.sqrt(0.05) * model.pi_v.detach().numpy(),
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        item[:, 6:8].norm(dim=1).detach().numpy(),
        np.full(4, np.sqrt(0.05)),
        rtol=1e-5,
    )


def test_one_plain_bpr_updates_id_and_all_four_axis_encoders():
    model = _model()
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
    assert model.activity_item.net[0].weight.grad.abs().sum() > 0
    assert model.value_item.net[0].weight.grad.abs().sum() > 0


def test_m4_sample_weights_cannot_enter_m2_loss():
    model = _model()
    with pytest.raises(ValueError, match="M4"):
        model.bpr_loss(
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([3]),
            weights=torch.ones(1),
        )
