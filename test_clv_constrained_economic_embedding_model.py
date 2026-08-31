import numpy as np
import pytest
import torch

from clv_constrained_economic_embedding_model import (
    ConstrainedCLVEconomicLightGCN,
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
    return ConstrainedCLVEconomicLightGCN(
        n_users=3,
        n_items=4,
        q_n=np.array([0.9, 0.5, 0.1], np.float32),
        q_v=np.array([0.1, 0.5, 0.9], np.float32),
        q_c=np.array([0.8, 0.5, 0.2], np.float32),
        user_clv_valid=np.ones(3, bool),
        item_economic_features=np.array(
            [[-1.0, -0.8], [-0.3, 0.1], [0.4, 0.5], [1.0, 0.9]],
            np.float32,
        ),
        item_economic_valid=np.ones(4, bool),
        adj=_adj(),
        id_dim=6,
        clv_dim=4,
        rho=rho,
        n_layers=layers,
        pref_reg=1e-4,
    )


def test_user_clv_norm_is_total_level_and_direction_distinguishes_n_from_v():
    model = _model(layers=0)
    with torch.no_grad():
        model.user_clv_projection.weight.zero_()
        model.user_clv_projection.weight[0, 0] = 1.0
        model.user_clv_projection.weight[1, 1] = 1.0

    user = model.clv_user_embeddings()

    torch.testing.assert_close(
        user.norm(dim=1), torch.tensor([0.8, 0.5, 0.2]), atol=1e-6, rtol=0
    )
    assert user[0, 0] > user[0, 1]  # N-dominant
    assert user[2, 1] > user[2, 0]  # V-dominant


def test_item_clv_block_can_only_use_two_price_positions():
    model = _model(layers=0)
    assert not hasattr(model, "item_response")
    with torch.no_grad():
        model.item_economic_projection.weight.zero_()
        model.item_economic_projection.weight[0, 0] = 1.0
        model.item_economic_projection.weight[1, 1] = 1.0

    item = model.clv_item_embeddings()

    assert item[0, 0] < 0 < item[3, 0]
    assert item[0, 1] < 0 < item[3, 1]
    torch.testing.assert_close(item.norm(dim=1), torch.ones(4), atol=1e-6, rtol=0)


def test_layer0_is_id_plus_one_bounded_four_dimensional_block():
    model = _model(layers=0)
    user, item = model.layer0_embeddings()

    assert user.shape == (3, 10)
    assert item.shape == (4, 10)
    torch.testing.assert_close(user[:, :6], model.E_u.weight)
    torch.testing.assert_close(item[:, :6], model.E_i.weight)
    assert torch.all(user[:, 6:].norm(dim=1) <= np.sqrt(0.05) + 1e-6)
    assert torch.all(item[:, 6:].norm(dim=1) <= np.sqrt(0.05) + 1e-6)


def test_rho_zero_is_exact_ordinary_lightgcn():
    model = _model(rho=0.0, layers=1)
    full_user, full_item, *_ = model.embeddings()
    id_user, id_item = model.id_embeddings()

    torch.testing.assert_close(full_user[:, :6], id_user, atol=0, rtol=0)
    torch.testing.assert_close(full_item[:, :6], id_item, atol=0, rtol=0)
    torch.testing.assert_close(full_user[:, 6:], torch.zeros_like(full_user[:, 6:]))
    torch.testing.assert_close(full_item[:, 6:], torch.zeros_like(full_item[:, 6:]))


def test_one_bpr_trains_id_and_both_constrained_projections():
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
    assert model.user_clv_projection.weight.grad.abs().sum() > 0
    assert model.item_economic_projection.weight.grad.abs().sum() > 0


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
