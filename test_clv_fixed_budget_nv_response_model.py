import numpy as np
import pytest
import torch

from clv_fixed_budget_nv_response_model import FixedBudgetNVResponseLightGCN


def _adj(n_users=3, n_items=4):
    edges = [(0, 0), (0, 1), (1, 1), (1, 2), (2, 3)]
    edges = [
        (user, item)
        for user, item in edges
        if user < n_users and item < n_items
    ]
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


def _model(*, rho=0.05, layers=1):
    torch.manual_seed(11)
    return FixedBudgetNVResponseLightGCN(
        n_users=3,
        n_items=4,
        q_n=np.array([0.9, 0.5, 0.1], np.float32),
        q_v=np.array([0.1, 0.5, 0.9], np.float32),
        q_c=np.array([0.8, 0.5, 0.2], np.float32),
        user_clv_valid=np.ones(3, bool),
        item_overall_price=np.array([0.0, 0.25, 0.75, 1.0], np.float32),
        item_price_valid=np.ones(4, bool),
        adj=_adj(),
        id_dim=6,
        rho=rho,
        n_layers=layers,
        pref_reg=1e-4,
        price_scale_initial=0.9,
    )


def test_clv_level_is_exactly_allocated_between_n_and_v():
    model = _model(layers=0)
    b_n, b_v = model.clv_user_budget()

    torch.testing.assert_close(b_n + b_v, model.q_c, atol=1e-7, rtol=0)
    assert b_n[0] > b_v[0]
    assert b_n[1] == pytest.approx(b_v[1])
    assert b_n[2] < b_v[2]


def test_item_responses_are_bounded_and_price_response_has_fixed_direction():
    model = _model(layers=0)
    with torch.no_grad():
        model.item_n_projection.weight.zero_()
        model.item_n_projection.weight[0, 0] = 3.0

    r_n, r_v = model.item_responses()

    assert torch.all(r_n.abs() <= 1.0)
    assert torch.all(r_v.abs() <= 1.0)
    assert torch.all(r_v[1:] > r_v[:-1])
    expected = torch.sigmoid(model.price_scale_raw) * torch.tensor(
        [-1.0, -0.5, 0.5, 1.0]
    )
    torch.testing.assert_close(r_v, expected)


def test_layer0_has_id_plus_two_fixed_budget_coordinates():
    model = _model(layers=0)
    user, item = model.layer0_embeddings()
    b_n, b_v = model.clv_user_budget()
    r_n, r_v = model.item_responses()
    scale = np.sqrt(0.05)

    assert user.shape == (3, 8)
    assert item.shape == (4, 8)
    torch.testing.assert_close(user[:, :6], model.E_u.weight)
    torch.testing.assert_close(item[:, :6], model.E_i.weight)
    torch.testing.assert_close(user[:, 6], scale * b_n)
    torch.testing.assert_close(user[:, 7], scale * b_v)
    torch.testing.assert_close(item[:, 6], scale * r_n)
    torch.testing.assert_close(item[:, 7], scale * r_v)
    assert model.layer0_auxiliary_scores().abs().max() <= 0.05 + 1e-7


def test_rho_zero_is_exact_ordinary_lightgcn():
    model = _model(rho=0.0, layers=1)
    full_user, full_item, *_ = model.embeddings()
    id_user, id_item = model.id_embeddings()

    torch.testing.assert_close(full_user[:, :6], id_user, atol=0, rtol=0)
    torch.testing.assert_close(full_item[:, :6], id_item, atol=0, rtol=0)
    torch.testing.assert_close(full_user[:, 6:], torch.zeros_like(full_user[:, 6:]))
    torch.testing.assert_close(full_item[:, 6:], torch.zeros_like(full_item[:, 6:]))


def test_one_bpr_trains_id_n_response_and_positive_price_scale():
    model = _model(layers=1)
    users = torch.tensor([0, 1, 2])
    positives = torch.tensor([0, 1, 3])
    negatives = torch.tensor([2, 3, 0])

    loss, diagnostics = model.bpr_loss(
        users, positives, negatives, None, 0.0, None
    )
    loss.backward()

    assert diagnostics["objective"] == "plain_bpr"
    assert model.E_u.weight.grad.abs().sum() > 0
    assert model.E_i.weight.grad.abs().sum() > 0
    assert model.item_n_projection.weight.grad.abs().sum() > 0
    assert model.price_scale_raw.grad.abs().sum() > 0


def test_n_and_v_views_partition_the_two_auxiliary_coordinates():
    model = _model(layers=1)
    n_user, n_item = model.component_embeddings("n")
    v_user, v_item = model.component_embeddings("v")

    torch.testing.assert_close(n_user[:, 7], torch.zeros_like(n_user[:, 7]))
    torch.testing.assert_close(n_item[:, 7], torch.zeros_like(n_item[:, 7]))
    torch.testing.assert_close(v_user[:, 6], torch.zeros_like(v_user[:, 6]))
    torch.testing.assert_close(v_item[:, 6], torch.zeros_like(v_item[:, 6]))


def test_invalid_inputs_are_exact_nonintervention():
    model = FixedBudgetNVResponseLightGCN(
        n_users=2,
        n_items=2,
        q_n=np.array([0.8, 0.0], np.float32),
        q_v=np.array([0.2, 0.0], np.float32),
        q_c=np.array([0.7, 0.0], np.float32),
        user_clv_valid=np.array([True, False]),
        item_overall_price=np.array([0.8, 0.5], np.float32),
        item_price_valid=np.array([True, False]),
        adj=_adj(2, 2),
        id_dim=4,
        n_layers=0,
    )

    b_n, b_v = model.clv_user_budget()
    _, r_v = model.item_responses()
    assert b_n[1] == 0
    assert b_v[1] == 0
    assert r_v[1] == 0


def test_m3_m4_and_external_score_paths_are_rejected():
    model = _model()
    users = torch.tensor([0])
    positives = torch.tensor([0])
    negatives = torch.tensor([3])

    with pytest.raises(ValueError, match="M4"):
        model.bpr_loss(
            users, positives, negatives, None, 0.0, torch.tensor([2.0])
        )
    with pytest.raises(ValueError, match="외부"):
        model.bpr_loss(users, positives, negatives, None, 0.1, None)
