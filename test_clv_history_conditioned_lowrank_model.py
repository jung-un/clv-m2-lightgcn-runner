import inspect

import numpy as np
import pytest
import torch

from clv_history_conditioned_lowrank_model import (
    CLVHistoryConditionedLowRankLightGCN,
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


def _model(*, rho=0.05, layers=1, activity_valid=None, value_valid=None):
    return CLVHistoryConditionedLowRankLightGCN(
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


def test_model_has_no_free_user_id_or_explicit_item_economic_features():
    parameters = inspect.signature(
        CLVHistoryConditionedLowRankLightGCN
    ).parameters
    model = _model(layers=0)

    assert not hasattr(model, "E_u")
    assert hasattr(model, "E_i")
    assert "item_profile" not in parameters
    assert not hasattr(model, "activity_item")
    assert not hasattr(model, "value_item")


def test_user_base_is_normalized_purchase_history_of_learned_item_ids():
    model = _model(layers=0)
    empty_users = torch.zeros((model.n_users, model.embedding_dim))
    manual = torch.sparse.mm(
        model.adj, torch.cat([empty_users, model.E_i.weight], dim=0)
    )[: model.n_users]
    manual = torch.nn.functional.normalize(manual, p=2, dim=1, eps=1e-12)

    torch.testing.assert_close(model.purchase_history_expression(), manual)


def test_conditions_are_centred_once_over_valid_train_users():
    valid = np.array([True, True, False])
    model = _model(activity_valid=valid, value_valid=valid, layers=0)

    torch.testing.assert_close(model.c_n, torch.tensor([-0.2, 0.2, 0.0]))
    torch.testing.assert_close(model.c_v, torch.tensor([0.2, -0.2, 0.0]))
    assert model.c_n[valid].mean() == pytest.approx(0.0)
    assert model.c_v[valid].mean() == pytest.approx(0.0)


def test_rho_zero_is_exactly_the_history_only_representation():
    model = _model(rho=0.0, layers=0)
    _make_transforms_nonzero(model)
    user, item = model.layer0_embeddings()

    torch.testing.assert_close(
        user, model.purchase_history_expression(), atol=0, rtol=0
    )
    torch.testing.assert_close(item, model.E_i.weight, atol=0, rtol=0)


def test_rho_zero_bypasses_conditional_maps_and_norm_rescaling(monkeypatch):
    class FailIfCalled(torch.nn.Module):
        def forward(self, values):
            raise AssertionError("rho=0에서 조건부 변환을 호출하면 안 됩니다")

    model = _model(rho=0.0, layers=0)
    history = torch.randn(model.n_users, model.embedding_dim)
    monkeypatch.setattr(model, "purchase_history_expression", lambda: history)
    model.activity_transform = FailIfCalled()
    model.value_transform = FailIfCalled()

    user, _ = model.layer0_embeddings()

    assert user is history


def test_clv_cannot_create_a_user_direction_without_purchase_history():
    model = _model(layers=0)
    _make_transforms_nonzero(model)
    with torch.no_grad():
        model.E_i.weight.zero_()

    user, _ = model.layer0_embeddings()
    torch.testing.assert_close(user, torch.zeros_like(user), atol=0, rtol=0)


def test_nonzero_conditioning_preserves_history_norm_and_changes_direction():
    model = _model(layers=0)
    _make_transforms_nonzero(model)
    history = model.purchase_history_expression()
    user, _ = model.layer0_embeddings()

    torch.testing.assert_close(
        user.norm(dim=1), history.norm(dim=1), atol=1e-7, rtol=1e-6
    )
    assert (user - history).abs().sum() > 0


def test_one_plain_bpr_jointly_trains_items_and_both_conditional_maps():
    model = _model(layers=1)
    users = torch.tensor([0, 2])
    positives = torch.tensor([0, 3])
    negatives = torch.tensor([2, 1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    loss, diagnostics = model.bpr_loss(users, positives, negatives, None, 0.0, None)
    loss.backward()
    assert diagnostics["objective"] == "plain_bpr"
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
        model.bpr_loss(users, positives, negatives, None, 0.0, torch.tensor([2.0]))
    with pytest.raises(ValueError, match="외부 점수"):
        model.bpr_loss(users, positives, negatives, None, 0.1, None)
