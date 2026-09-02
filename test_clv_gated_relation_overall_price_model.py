import numpy as np
import pytest
import torch

from clv_gated_relation_overall_price_model import (
    GatedRelationOverallPriceLightGCN,
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
    torch.manual_seed(13)
    return GatedRelationOverallPriceLightGCN(
        n_users=3,
        n_items=4,
        q_n=np.array([0.9, 0.5, 0.2], np.float32),
        q_v=np.array([0.2, 0.5, 0.9], np.float32),
        q_c=np.array([0.9, 0.5, 0.2], np.float32),
        user_clv_valid=np.ones(3, bool),
        user_overall_price=np.array([0.2, 0.5, 0.9], np.float32),
        user_price_valid=np.ones(3, bool),
        item_overall_price=np.array([0.1, 0.4, 0.7, 1.0], np.float32),
        item_price_valid=np.ones(4, bool),
        adj=_adj(),
        id_dim=6,
        auxiliary_dim=3,
        rho=rho,
        price_budget=0.25,
        n_layers=layers,
        pref_reg=1e-4,
    )


def test_layer0_is_id_plus_gated_relation_two_dim_and_price_one_dim():
    model = _model(layers=0)
    user, item = model.layer0_embeddings()

    assert user.shape == (3, 9)
    assert item.shape == (4, 9)
    assert model.total_dim == 9
    torch.testing.assert_close(user[:, :6], model.E_u.weight)
    torch.testing.assert_close(item[:, :6], model.E_i.weight)
    assert torch.all(user[:, 6:].norm(dim=1) <= np.sqrt(0.05) + 1e-6)
    assert torch.all(item[:, 6:].norm(dim=1) <= np.sqrt(0.05) + 1e-6)


def test_relation_gate_is_continuous_bounded_and_has_positive_level_slope():
    model = _model(layers=0)
    gate = model.relation_gate()

    assert torch.all((0.0 < gate) & (gate < 1.0))
    assert gate[0] > gate[2]
    assert model.representation_diagnostics()["relation_level_slope"] > 0.0


def test_price_coordinate_uses_overall_user_item_price_fit_with_fixed_sign():
    model = _model(layers=0)
    user = model.auxiliary_user_embeddings()
    item = model.auxiliary_item_embeddings()

    assert user[0, 2] < 0 < user[2, 2]
    assert item[0, 2] < 0 < item[3, 2]
    assert 0.5 <= float(model.item_price_scale().detach()) <= 1.5


def test_rho_zero_is_exact_ordinary_lightgcn():
    model = _model(rho=0.0, layers=1)
    full_user, full_item, *_ = model.embeddings()
    id_user, id_item = model.id_embeddings()

    torch.testing.assert_close(full_user[:, :6], id_user, atol=0, rtol=0)
    torch.testing.assert_close(full_item[:, :6], id_item, atol=0, rtol=0)
    torch.testing.assert_close(full_user[:, 6:], torch.zeros_like(full_user[:, 6:]))
    torch.testing.assert_close(full_item[:, 6:], torch.zeros_like(full_item[:, 6:]))
    assert model.representation_diagnostics()["rho_zero_auxiliary_max_abs"] == 0.0


def test_one_plain_bpr_trains_id_relation_gate_and_price_scale_together():
    model = _model(layers=1)
    users = torch.tensor([0, 1, 2])
    positives = torch.tensor([0, 1, 3])
    negatives = torch.tensor([2, 3, 0])

    loss, diagnostics = model.bpr_loss(
        users, positives, negatives, None, 0.0, None
    )
    loss.backward()

    assert diagnostics["objective"] == "plain_bpr"
    gradients = model.training_gradient_diagnostics()
    assert gradients["id_user_gradient_norm"] > 0
    assert gradients["id_item_gradient_norm"] > 0
    assert gradients["item_relation_projection_gradient_norm"] > 0
    assert gradients["relation_gate_gradient_norm"] > 0
    assert gradients["item_price_scale_gradient_norm"] > 0


def test_component_views_zero_only_the_other_auxiliary_coordinates():
    model = _model(layers=1)
    relation_user, relation_item = model.component_embeddings("relation")
    price_user, price_item = model.component_embeddings("price")

    torch.testing.assert_close(
        relation_user[:, 8:], torch.zeros_like(relation_user[:, 8:])
    )
    torch.testing.assert_close(
        relation_item[:, 8:], torch.zeros_like(relation_item[:, 8:])
    )
    torch.testing.assert_close(
        price_user[:, 6:8], torch.zeros_like(price_user[:, 6:8])
    )
    torch.testing.assert_close(
        price_item[:, 6:8], torch.zeros_like(price_item[:, 6:8])
    )


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
