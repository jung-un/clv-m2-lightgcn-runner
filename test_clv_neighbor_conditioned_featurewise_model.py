import inspect

import numpy as np
import pytest
import torch

from clv_neighbor_conditioned_featurewise_model import (
    CLVNeighborConditionedFeaturewiseLightGCN,
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


def _model(*, rho=0.05, activity_valid=None, value_valid=None, layers=0):
    return CLVNeighborConditionedFeaturewiseLightGCN(
        n_users=3,
        n_items=4,
        q_n=np.array([0.1, 0.5, 0.9], np.float32),
        q_v=np.array([0.9, 0.5, 0.1], np.float32),
        user_activity_valid=(
            np.ones(3, bool) if activity_valid is None else activity_valid
        ),
        user_value_valid=(
            np.ones(3, bool) if value_valid is None else value_valid
        ),
        adj=_adj(),
        embedding_dim=6,
        rho=rho,
        n_layers=layers,
        pref_reg=1e-4,
    )


def test_conditions_are_centred_once_over_valid_train_users():
    valid = np.array([True, False, True])
    model = _model(activity_valid=valid, value_valid=valid)

    torch.testing.assert_close(
        model.c_n, torch.tensor([-0.4, 0.0, 0.4]), atol=1e-7, rtol=0
    )
    torch.testing.assert_close(
        model.c_v, torch.tensor([0.4, 0.0, -0.4]), atol=1e-7, rtol=0
    )
    assert model.c_n[valid].mean() == pytest.approx(0.0, abs=1e-7)
    assert model.c_v[valid].mean() == pytest.approx(0.0, abs=1e-7)


def test_featurewise_correction_is_history_times_two_axis_vectors():
    model = _model()
    history = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            [3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        ]
    )
    model.purchase_neighbour_expression = lambda: history
    with torch.no_grad():
        model.activity_axis_vector.copy_(torch.tensor([1, 2, 3, 4, 5, 6]))
        model.value_axis_vector.copy_(torch.tensor([6, 5, 4, 3, 2, 1]))

    activity, value = model.axis_corrections()

    torch.testing.assert_close(
        activity, history * model.c_n[:, None] * model.activity_axis_vector
    )
    torch.testing.assert_close(
        value, history * model.c_v[:, None] * model.value_axis_vector
    )


def test_model_has_only_two_axis_vectors_not_low_rank_transforms():
    model = _model()
    parameters = inspect.signature(
        CLVNeighborConditionedFeaturewiseLightGCN
    ).parameters

    assert "transform_rank" not in parameters
    assert not hasattr(model, "activity_transform")
    assert not hasattr(model, "value_transform")
    assert model.activity_axis_vector.shape == (6,)
    assert model.value_axis_vector.shape == (6,)
    assert model.activity_axis_vector.numel() + model.value_axis_vector.numel() == 12


def test_rho_zero_exactly_matches_id_layer0_and_item_is_unchanged():
    model = _model(rho=0.0)
    with torch.no_grad():
        model.activity_axis_vector.fill_(2.0)
        model.value_axis_vector.fill_(-3.0)

    user, item = model.layer0_embeddings()

    torch.testing.assert_close(user, model.E_u.weight, atol=0, rtol=0)
    torch.testing.assert_close(item, model.E_i.weight, atol=0, rtol=0)


def test_nonzero_correction_preserves_each_user_id_norm():
    model = _model()
    with torch.no_grad():
        model.activity_axis_vector.fill_(0.4)
        model.value_axis_vector.fill_(-0.2)

    user, _ = model.layer0_embeddings()

    torch.testing.assert_close(
        user.norm(dim=1), model.E_u.weight.norm(dim=1), atol=1e-7, rtol=1e-6
    )
    assert (user - model.E_u.weight).abs().sum() > 0


def test_one_bpr_trains_ids_and_both_axis_vectors_without_extra_weighting():
    model = _model(layers=1)
    users = torch.tensor([0, 2])
    positives = torch.tensor([0, 3])
    negatives = torch.tensor([2, 1])

    loss, diagnostics = model.bpr_loss(users, positives, negatives, None, 0.0, None)
    loss.backward()

    assert diagnostics["objective"] == "plain_bpr"
    assert model.E_u.weight.grad.abs().sum() > 0
    assert model.E_i.weight.grad.abs().sum() > 0
    assert model.activity_axis_vector.grad.abs().sum() > 0
    assert model.value_axis_vector.grad.abs().sum() > 0

    with pytest.raises(ValueError, match="M4"):
        model.bpr_loss(users, positives, negatives, None, 0.0, torch.ones(2))
    with pytest.raises(ValueError, match="외부 점수"):
        model.bpr_loss(users, positives, negatives, None, 0.1, None)
