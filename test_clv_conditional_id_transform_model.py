import inspect

import numpy as np
import pytest
import torch

from clv_conditional_id_transform_model import (
    CLVConditionalIDTransformLightGCN,
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
    normalized = values / torch.sqrt(
        degree[indices[0]] * degree[indices[1]]
    )
    return torch.sparse_coo_tensor(indices, normalized, raw.shape).coalesce()


def _model(*, rho=0.05, activity_valid=None, value_valid=None, layers=1):
    return CLVConditionalIDTransformLightGCN(
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
        transform_rank=2,
        rho=rho,
        n_layers=layers,
        pref_reg=1e-4,
    )


def _make_transforms_nonzero(model):
    with torch.no_grad():
        model.activity_transform.up.weight.fill_(0.2)
        model.value_transform.up.weight.fill_(-0.15)


def test_model_keeps_one_64d_score_space_and_has_no_item_side_clv_features():
    parameters = inspect.signature(
        CLVConditionalIDTransformLightGCN
    ).parameters
    model = _model(layers=0)
    user, item = model.layer0_embeddings()

    assert user.shape == (3, 6)
    assert item.shape == (4, 6)
    assert "item_profile" not in parameters
    assert not hasattr(model, "activity_item")
    assert not hasattr(model, "value_item")
    assert not hasattr(model, "gate_n")
    assert not hasattr(model, "gate_v")
    torch.testing.assert_close(item, model.E_i.weight)


def test_rho_zero_exactly_matches_the_ordinary_id_layer0():
    model = _model(rho=0.0, layers=0)
    _make_transforms_nonzero(model)
    user, item = model.layer0_embeddings()

    torch.testing.assert_close(user, model.E_u.weight, atol=0, rtol=0)
    torch.testing.assert_close(item, model.E_i.weight, atol=0, rtol=0)


def test_nonzero_transformation_preserves_each_user_embedding_norm():
    model = _model(layers=0)
    _make_transforms_nonzero(model)
    user, _ = model.layer0_embeddings()

    torch.testing.assert_close(
        user.norm(dim=1), model.E_u.weight.norm(dim=1), atol=1e-7, rtol=1e-6
    )
    assert (user - model.E_u.weight).abs().sum() > 0


def test_invalid_user_receives_no_axis_conditioning():
    valid = np.array([True, False, True])
    model = _model(activity_valid=valid, value_valid=valid, layers=0)
    _make_transforms_nonzero(model)
    user, _ = model.layer0_embeddings()

    torch.testing.assert_close(user[1], model.E_u.weight[1], atol=0, rtol=0)
    assert model.condition_n[1] == 0
    assert model.condition_v[1] == 0


def test_n_and_v_conditions_are_signed_and_kept_separate():
    model = _model(layers=0)

    torch.testing.assert_close(
        model.condition_n, torch.tensor([-0.8, 0.0, 0.8])
    )
    torch.testing.assert_close(
        model.condition_v, torch.tensor([0.8, 0.0, -0.8])
    )


def test_one_plain_bpr_jointly_trains_id_and_both_low_rank_transforms():
    model = _model(layers=1)
    users = torch.tensor([0, 2])
    positives = torch.tensor([0, 3])
    negatives = torch.tensor([2, 1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    loss, diagnostics = model.bpr_loss(
        users, positives, negatives, None, 0.0, None
    )
    loss.backward()
    assert diagnostics["objective"] == "plain_bpr"
    assert model.E_u.weight.grad.abs().sum() > 0
    assert model.E_i.weight.grad.abs().sum() > 0
    assert model.activity_transform.up.weight.grad.abs().sum() > 0
    assert model.value_transform.up.weight.grad.abs().sum() > 0

    optimizer.step()
    optimizer.zero_grad()
    loss, _ = model.bpr_loss(users, positives, negatives, None, 0.0, None)
    loss.backward()
    assert model.activity_transform.down.weight.grad.abs().sum() > 0
    assert model.value_transform.down.weight.grad.abs().sum() > 0


def test_m3_m4_and_external_score_interventions_are_rejected():
    model = _model()
    users = torch.tensor([0])
    positives = torch.tensor([0])
    negatives = torch.tensor([3])

    with pytest.raises(ValueError, match="M4"):
        model.bpr_loss(
            users, positives, negatives, None, 0.0, torch.tensor([2.0])
        )
    with pytest.raises(ValueError, match="외부 점수"):
        model.bpr_loss(users, positives, negatives, None, 0.1, None)
