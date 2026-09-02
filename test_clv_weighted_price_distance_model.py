import numpy as np
import pytest
import torch

from clv_weighted_price_distance_model import (
    CLVWeightedPriceDistanceLightGCN,
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


def _model(*, rho=0.05, layers=1):
    torch.manual_seed(17)
    return CLVWeightedPriceDistanceLightGCN(
        n_users=3,
        n_items=4,
        q_c=np.array([0.9, 0.5, 0.0], np.float32),
        user_clv_valid=np.array([True, True, False]),
        user_overall_price=np.array([0.2, 0.8, 0.5], np.float32),
        user_price_valid=np.array([True, True, False]),
        item_overall_price=np.array([0.1, 0.4, 0.7, 0.5], np.float32),
        item_price_valid=np.array([True, True, True, False]),
        adj=_adj(),
        id_dim=6,
        auxiliary_dim=2,
        rho=rho,
        price_scale_initial=0.9,
        n_layers=layers,
        pref_reg=1e-4,
    )


def test_layer0_is_id_plus_only_two_price_distance_coordinates():
    model = _model(layers=0)
    user, item = model.layer0_embeddings()

    assert user.shape == (3, 8)
    assert item.shape == (4, 8)
    assert model.total_dim == 8
    assert not hasattr(model, "item_relation_projection")
    assert not hasattr(model, "item_price_logits")
    torch.testing.assert_close(user[:, :6], model.E_u.weight)
    torch.testing.assert_close(item[:, :6], model.E_i.weight)
    torch.testing.assert_close(user[2, 6:], torch.zeros(2))
    torch.testing.assert_close(item[3, 6:], torch.zeros(2))


def test_layer0_pairwise_difference_is_exact_negative_squared_price_distance():
    model = _model(layers=0)
    users = torch.tensor([0, 0, 1, 1])
    left = torch.tensor([0, 1, 1, 2])
    right = torch.tensor([1, 2, 0, 1])

    actual = model.layer0_auxiliary_scores(users, left) - model.layer0_auxiliary_scores(
        users, right
    )
    q_c = model.q_c[users]
    user_price = model.user_overall_price[users]
    left_distance = (model.item_overall_price[left] - user_price).square()
    right_distance = (model.item_overall_price[right] - user_price).square()
    expected = (
        -model.rho
        * model.price_scale()
        * q_c
        * (left_distance - right_distance)
    )

    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)


def test_clv_level_controls_strength_not_preferred_price_location():
    model = _model(layers=0)
    with torch.no_grad():
        model.q_c[:] = torch.tensor([1.0, 0.25, 0.0])
        model.user_overall_price[:] = 0.2
        model.user_clv_valid[:] = 1.0
        model.user_price_valid[:] = 1.0

    close = torch.tensor([0, 0, 0])
    far = torch.tensor([2, 2, 2])
    users = torch.arange(3)
    advantage = model.layer0_auxiliary_scores(users, close) - model.layer0_auxiliary_scores(
        users, far
    )

    assert advantage[0] > advantage[1] > advantage[2]
    assert advantage[2] == 0.0


def test_price_scale_is_positive_bounded_and_receives_plain_bpr_gradient():
    model = _model(layers=1)
    users = torch.tensor([0, 1, 1])
    positives = torch.tensor([0, 2, 1])
    negatives = torch.tensor([2, 0, 0])

    loss, diagnostics = model.bpr_loss(
        users, positives, negatives, None, 0.0, None
    )
    loss.backward()

    assert diagnostics["objective"] == "plain_bpr"
    assert 0.0 < float(model.price_scale().detach()) < 1.0
    gradients = model.training_gradient_diagnostics()
    assert gradients["id_user_gradient_norm"] > 0.0
    assert gradients["id_item_gradient_norm"] > 0.0
    assert gradients["price_scale_gradient_norm"] > 0.0


def test_rho_zero_is_exact_ordinary_lightgcn():
    model = _model(rho=0.0, layers=1)
    full_user, full_item, *_ = model.embeddings()
    id_user, id_item = model.id_embeddings()

    torch.testing.assert_close(full_user[:, :6], id_user, atol=0, rtol=0)
    torch.testing.assert_close(full_item[:, :6], id_item, atol=0, rtol=0)
    torch.testing.assert_close(full_user[:, 6:], torch.zeros_like(full_user[:, 6:]))
    torch.testing.assert_close(full_item[:, 6:], torch.zeros_like(full_item[:, 6:]))
    assert model.representation_diagnostics()["rho_zero_auxiliary_max_abs"] == 0.0


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
