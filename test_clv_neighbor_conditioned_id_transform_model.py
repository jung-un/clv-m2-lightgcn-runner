import inspect

import numpy as np
import pytest
import torch

from clv_neighbor_conditioned_id_transform_model import (
    CLVNeighborConditionedIDTransformLightGCN,
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


def _model(*, rho=0.05, activity_valid=None, value_valid=None, layers=1):
    return CLVNeighborConditionedIDTransformLightGCN(
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


def test_model_uses_no_explicit_item_features_or_separate_score_space():
    parameters = inspect.signature(
        CLVNeighborConditionedIDTransformLightGCN
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


def test_purchase_neighbour_expression_is_one_hop_item_id_aggregate():
    model = _model(layers=0)
    empty = torch.zeros_like(model.E_u.weight)
    manual = torch.sparse.mm(
        model.adj, torch.cat([empty, model.E_i.weight], dim=0)
    )[: model.n_users]
    manual = torch.nn.functional.normalize(manual, p=2, dim=1, eps=1e-12)

    torch.testing.assert_close(model.purchase_neighbour_expression(), manual)


def test_rho_zero_exactly_matches_ordinary_id_layer0():
    model = _model(rho=0.0, layers=0)
    _make_transforms_nonzero(model)
    user, item = model.layer0_embeddings()

    torch.testing.assert_close(user, model.E_u.weight, atol=0, rtol=0)
    torch.testing.assert_close(item, model.E_i.weight, atol=0, rtol=0)


def test_nonzero_correction_preserves_each_user_embedding_norm():
    model = _model(layers=0)
    _make_transforms_nonzero(model)
    user, _ = model.layer0_embeddings()

    torch.testing.assert_close(
        user.norm(dim=1), model.E_u.weight.norm(dim=1), atol=1e-7, rtol=1e-6
    )
    assert (user - model.E_u.weight).abs().sum() > 0


def test_invalid_user_receives_exactly_zero_axis_correction():
    valid = np.array([True, False, True])
    model = _model(activity_valid=valid, value_valid=valid, layers=0)
    _make_transforms_nonzero(model)
    activity, value = model.axis_corrections()

    torch.testing.assert_close(activity[1], torch.zeros_like(activity[1]))
    torch.testing.assert_close(value[1], torch.zeros_like(value[1]))


def test_axis_corrections_are_population_centred_over_valid_users():
    model = _model(layers=0)
    _make_transforms_nonzero(model)
    activity, value = model.axis_corrections()

    torch.testing.assert_close(
        activity[model.user_activity_valid.bool()].mean(dim=0),
        torch.zeros(model.embedding_dim),
        atol=1e-7,
        rtol=0,
    )
    torch.testing.assert_close(
        value[model.user_value_valid.bool()].mean(dim=0),
        torch.zeros(model.embedding_dim),
        atol=1e-7,
        rtol=0,
    )


def test_n_and_v_conditions_are_nonnegative_and_separate():
    model = _model(layers=0)

    torch.testing.assert_close(model.q_n, torch.tensor([0.1, 0.5, 0.9]))
    torch.testing.assert_close(model.q_v, torch.tensor([0.9, 0.5, 0.1]))
    assert (model.q_n >= 0).all()
    assert (model.q_v >= 0).all()


def test_one_plain_bpr_jointly_trains_id_items_and_both_transforms():
    model = _model(layers=1)
    users = torch.tensor([0, 2])
    positives = torch.tensor([0, 3])
    negatives = torch.tensor([2, 1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    loss, diagnostics = model.bpr_loss(users, positives, negatives, None, 0.0, None)
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


def test_liveness_diagnostics_report_gradient_and_effective_correction_ratios():
    model = _model(layers=1)
    users = torch.tensor([0, 2])
    positives = torch.tensor([0, 3])
    negatives = torch.tensor([2, 1])

    loss, _ = model.bpr_loss(users, positives, negatives, None, 0.0, None)
    loss.backward()
    gradients = model.training_gradient_diagnostics()
    representation = model.representation_diagnostics()

    assert gradients["activity_up_gradient_norm"] > 0
    assert gradients["value_up_gradient_norm"] > 0
    assert representation["activity_effective_ratio_to_id"] >= 0
    assert representation["value_effective_ratio_to_id"] >= 0


def test_existing_l2_regularises_sampled_id_but_not_transform_parameters():
    model = _model(layers=0)
    users = torch.tensor([0, 2])
    positives = torch.tensor([0, 3])
    negatives = torch.tensor([2, 1])
    before = model.batch_l2(users, positives, negatives)
    with torch.no_grad():
        model.activity_transform.up.weight.fill_(2.0)
        model.value_transform.up.weight.fill_(2.0)
    after_transform_change = model.batch_l2(users, positives, negatives)
    with torch.no_grad():
        model.E_u.weight[users] = model.E_u.weight[users] * 2.0
    after_id_change = model.batch_l2(users, positives, negatives)

    torch.testing.assert_close(after_transform_change, before)
    assert after_id_change > after_transform_change


def test_m3_m4_and_external_score_interventions_are_rejected():
    model = _model()
    users = torch.tensor([0])
    positives = torch.tensor([0])
    negatives = torch.tensor([3])

    with pytest.raises(ValueError, match="M4"):
        model.bpr_loss(users, positives, negatives, None, 0.0, torch.tensor([2.0]))
    with pytest.raises(ValueError, match="외부 점수"):
        model.bpr_loss(users, positives, negatives, None, 0.1, None)
