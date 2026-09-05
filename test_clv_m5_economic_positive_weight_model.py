import numpy as np
import pytest
import torch
import torch.nn.functional as F

from clv_m5_economic_positive_weight_model import (
    M5EconomicLightGCN,
    positive_row_weights,
    weighted_multi_negative_bpr,
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


def _model(*, rho=0.15, layers=1):
    torch.manual_seed(7)
    user_input = np.array(
        [
            [0.10, -0.05, -0.03, -0.02, 0.7],
            [-0.08, 0.12, -0.02, -0.02, -0.3],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    item_input = np.array(
        [[-1.0, -0.5], [-0.2, 0.1], [0.4, 0.3], [1.0, 0.8]],
        dtype=np.float32,
    )
    return M5EconomicLightGCN(
        n_users=3,
        n_items=4,
        user_economic_input=user_input,
        user_economic_valid=np.array([True, True, False]),
        item_economic_input=item_input,
        item_economic_valid=np.ones(4, dtype=bool),
        adj=_adj(),
        id_dim=6,
        economic_dim=4,
        rho=rho,
        n_layers=layers,
        pref_reg=1e-4,
    )


def test_bounded_projection_preserves_zero_and_caps_norm():
    model = _model(layers=0)
    user, item = model.economic_coordinates()

    torch.testing.assert_close(user[2], torch.zeros(4))
    assert float(user.norm(dim=1).max().detach()) <= 1.0
    assert float(item.norm(dim=1).max().detach()) <= 1.0


def test_rho_zero_is_exact_id_lightgcn():
    model = _model(rho=0.0, layers=2)
    full_user, full_item = model.propagated_embeddings()
    id_user, id_item = model.id_embeddings()

    torch.testing.assert_close(full_user[:, :6], id_user, atol=0, rtol=0)
    torch.testing.assert_close(full_item[:, :6], id_item, atol=0, rtol=0)
    torch.testing.assert_close(full_user[:, 6:], torch.zeros_like(full_user[:, 6:]))
    torch.testing.assert_close(full_item[:, 6:], torch.zeros_like(full_item[:, 6:]))


def test_one_ranking_loss_updates_id_and_both_economic_projections():
    model = _model(layers=1)
    users = torch.tensor([0, 1])
    positives = torch.tensor([0, 1])
    negatives = torch.tensor([[2, 3], [0, 3]])
    user_z, item_z = model.propagated_embeddings()
    positive_scores = (user_z[users] * item_z[positives]).sum(1)
    negative_scores = (user_z[users, None, :] * item_z[negatives]).sum(2)

    loss, _ = weighted_multi_negative_bpr(
        positive_scores, negative_scores, torch.ones(2)
    )
    loss.backward()

    assert model.E_u.weight.grad.abs().sum() > 0
    assert model.E_i.weight.grad.abs().sum() > 0
    assert model.user_economic_projection.weight.grad.abs().sum() > 0
    assert model.item_economic_projection.weight.grad.abs().sum() > 0


def test_positive_weights_follow_fixed_formula_and_global_normalization():
    q_c = torch.tensor([0.0, 1.0, 1.0])
    amount_percentile = torch.tensor([0.0, 0.5, 1.0])
    raw = torch.tensor([1.0, 1.0, 1.5])
    normalizer = float(raw.mean())

    actual = positive_row_weights(
        q_c,
        amount_percentile,
        train_mean_raw_weight=normalizer,
        lambda_=0.5,
    )

    torch.testing.assert_close(actual, raw / normalizer)


def test_weighted_bpr_averages_per_negative_losses_before_row_weighting():
    positive = torch.tensor([1.0, 0.5])
    negative = torch.tensor([[0.0, 2.0], [0.0, 1.0]])
    row_weight = torch.tensor([1.5, 0.5])
    expected = (
        row_weight * F.softplus(negative - positive[:, None]).mean(dim=1)
    ).mean()

    actual, diagnostics = weighted_multi_negative_bpr(
        positive, negative, row_weight
    )

    torch.testing.assert_close(actual, expected)
    assert diagnostics["negative_count"] == 2
    assert float(diagnostics["row_weight_mean"]) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "q_c,amount,normalizer,match",
    [
        (torch.tensor([-0.1]), torch.tensor([0.5]), 1.0, "q_c"),
        (torch.tensor([0.5]), torch.tensor([1.1]), 1.0, "amount"),
        (torch.tensor([0.5]), torch.tensor([0.5]), 0.0, "normalizer"),
    ],
)
def test_positive_weights_reject_invalid_inputs(q_c, amount, normalizer, match):
    with pytest.raises(ValueError, match=match):
        positive_row_weights(
            q_c,
            amount,
            train_mean_raw_weight=normalizer,
            lambda_=0.5,
        )
