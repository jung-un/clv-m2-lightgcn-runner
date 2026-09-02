import numpy as np
import pytest
import torch

from clv_economic_quartile_distribution_model import (
    CLVEconomicQuartileDistributionLightGCN,
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
    torch.manual_seed(7)
    user_profile = np.array(
        [
            [0.30, -0.10, -0.10, -0.10],
            [-0.10, 0.30, -0.10, -0.10],
            [-0.10, -0.10, -0.10, 0.30],
        ],
        dtype=np.float32,
    )
    item_basis = np.eye(4, dtype=np.float32) - 0.25
    return CLVEconomicQuartileDistributionLightGCN(
        n_users=3,
        n_items=4,
        q_c=np.array([0.9, 0.5, 0.2], dtype=np.float32),
        user_clv_valid=np.ones(3, dtype=bool),
        user_economic_profile=user_profile,
        user_profile_valid=np.ones(3, dtype=bool),
        item_economic_basis=item_basis,
        item_economic_valid=np.ones(4, dtype=bool),
        adj=_adj(),
        id_dim=6,
        rho=rho,
        n_layers=layers,
        pref_reg=1e-4,
    )


def test_layer0_has_one_four_bin_block_with_fixed_total_weight():
    model = _model(layers=0)
    user, item = model.layer0_embeddings()
    weights = model.economic_bin_weights()

    assert user.shape == (3, 10)
    assert item.shape == (4, 10)
    torch.testing.assert_close(user[:, :6], model.E_u.weight)
    torch.testing.assert_close(item[:, :6], model.E_i.weight)
    torch.testing.assert_close(weights.sum(), torch.tensor(1.0))
    torch.testing.assert_close(weights, torch.full((4,), 0.25))
    assert model.layer0_economic_scores().abs().max() <= model.rho + 1e-7


def test_qc_scales_user_profile_without_changing_its_bin_pattern():
    model = _model(layers=0)
    user, _ = model.economic_coordinates()

    expected_first = (
        torch.tensor([0.30, -0.10, -0.10, -0.10])
        * torch.sqrt(torch.full((4,), 0.25))
        * 0.9
    )
    torch.testing.assert_close(user[0], expected_first)
    assert user[0].argmax() == 0
    assert user[1].argmax() == 1
    assert user[2].argmax() == 3


def test_rho_zero_is_exact_ordinary_lightgcn():
    model = _model(rho=0.0, layers=1)
    full_user, full_item, *_ = model.embeddings()
    id_user, id_item = model.id_embeddings()

    torch.testing.assert_close(full_user[:, :6], id_user, atol=0, rtol=0)
    torch.testing.assert_close(full_item[:, :6], id_item, atol=0, rtol=0)
    torch.testing.assert_close(full_user[:, 6:], torch.zeros_like(full_user[:, 6:]))
    torch.testing.assert_close(full_item[:, 6:], torch.zeros_like(full_item[:, 6:]))


def test_one_bpr_jointly_trains_id_and_relative_bin_weights():
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
    assert model.economic_bin_logits.grad is not None
    assert model.economic_bin_logits.grad.abs().sum() > 0


def test_invalid_clv_and_m3_m4_paths_are_rejected():
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
