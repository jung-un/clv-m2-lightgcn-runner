import numpy as np
import pytest
import torch

from clv_gradient_isolated_economic_interaction_model import (
    GradientIsolatedCLVEconomicInteractionLightGCN,
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
    torch.manual_seed(11)
    return GradientIsolatedCLVEconomicInteractionLightGCN(
        n_users=3,
        n_items=4,
        q_n=np.array([0.1, 0.6, 0.9], np.float32),
        q_v=np.array([0.9, 0.5, 0.2], np.float32),
        q_c=np.array([0.2, 0.6, 0.9], np.float32),
        user_clv_valid=np.ones(3, bool),
        item_price_percentile=np.array([0.1, 0.4, 0.7, 0.9], np.float32),
        item_price_valid=np.ones(4, bool),
        adj=_adj(),
        id_dim=6,
        relation_dim=3,
        rho=rho,
        n_layers=layers,
        pref_reg=1e-4,
    )


def test_model_has_id_relation_and_price_coordinates():
    model = _model(layers=0)
    user, item, *_ = model.embeddings()

    assert user.shape == (3, 10)
    assert item.shape == (4, 10)
    assert model.total_dim == 10
    assert 1.0 / 3.0 <= float(model.price_calibration()) <= 1.0


def test_auxiliary_path_does_not_backpropagate_into_id_tables():
    model = _model(layers=1)
    id_user, id_item = model.id_embeddings()
    relation_user, relation_item, price_user, price_item = (
        model.auxiliary_embeddings(id_user, id_item)
    )
    auxiliary = (
        relation_user.sum()
        + relation_item.sum()
        + price_user.sum()
        + price_item.sum()
    )
    auxiliary.backward()

    assert model.E_u.weight.grad is None
    assert model.E_i.weight.grad is None
    assert model.user_projection.weight.grad.abs().sum() > 0
    assert model.item_projection.weight.grad.abs().sum() > 0
    assert model.condition_mixer.weight.grad.abs().sum() > 0
    assert model.raw_price_calibration.grad.abs().sum() > 0


def test_single_plain_bpr_trains_id_and_auxiliary_parameters_together():
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
    assert model.user_projection.weight.grad.abs().sum() > 0
    assert model.item_projection.weight.grad.abs().sum() > 0
    assert model.condition_mixer.weight.grad.abs().sum() > 0
    assert model.raw_price_calibration.grad.abs().sum() > 0


def test_rho_zero_is_exact_ordinary_lightgcn_score():
    model = _model(rho=0.0, layers=1)
    users = torch.tensor([0, 1, 2])
    items = torch.tensor([3, 0, 1])
    full_user, full_item, *_ = model.embeddings()
    id_user, id_item = model.id_embeddings()

    torch.testing.assert_close(full_user, id_user, atol=0, rtol=0)
    torch.testing.assert_close(full_item, id_item, atol=0, rtol=0)
    torch.testing.assert_close(
        (full_user[users] * full_item[items]).sum(1),
        (id_user[users] * id_item[items]).sum(1),
        atol=0,
        rtol=0,
    )
    assert model.representation_diagnostics()["rho_zero_auxiliary_max_abs"] == 0


def test_price_direction_and_budget_are_structurally_bounded():
    model = _model(layers=0)
    _, _, price_user, price_item = model.auxiliary_embeddings()

    assert price_user[0] > 0  # high V
    assert price_user[2] < 0  # low V
    assert price_item[0] < 0  # low price
    assert price_item[3] > 0  # high price
    assert price_user.abs().max() <= 1.25 + 1e-6
    assert price_item.abs().max() <= 1.0 + 1e-6


def test_invalid_users_and_items_have_zero_auxiliary_coordinates():
    model = GradientIsolatedCLVEconomicInteractionLightGCN(
        n_users=3,
        n_items=4,
        q_n=np.array([0.1, 0.0, 0.9], np.float32),
        q_v=np.array([0.9, 0.0, 0.2], np.float32),
        q_c=np.array([0.2, 0.0, 0.9], np.float32),
        user_clv_valid=np.array([True, False, True]),
        item_price_percentile=np.array([0.1, 0.5, 0.7, 0.9], np.float32),
        item_price_valid=np.array([True, False, True, True]),
        adj=_adj(),
        id_dim=6,
        relation_dim=3,
        n_layers=0,
    )
    relation_user, _, price_user, price_item = model.auxiliary_embeddings()

    torch.testing.assert_close(relation_user[1], torch.zeros_like(relation_user[1]))
    assert price_user[1] == 0
    assert price_item[1] == 0


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
