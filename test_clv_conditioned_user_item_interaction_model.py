import inspect

import numpy as np
import pytest
import torch

from clv_conditioned_user_item_interaction_model import (
    CLVConditionedUserItemInteractionLightGCN,
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


def _model(*, rho=0.05, valid=None, layers=1):
    torch.manual_seed(7)
    return CLVConditionedUserItemInteractionLightGCN(
        n_users=3,
        n_items=4,
        q_c=np.array([0.2, 0.6, 0.9], np.float32),
        d_nv=np.array([-0.7, 0.0, 0.8], np.float32),
        user_clv_valid=np.ones(3, bool) if valid is None else valid,
        adj=_adj(),
        id_dim=6,
        context_dim=3,
        rho=rho,
        n_layers=layers,
        pref_reg=1e-4,
    )


def test_model_contains_one_interaction_block_without_item_economic_inputs():
    model = _model(layers=0)
    parameters = inspect.signature(
        CLVConditionedUserItemInteractionLightGCN
    ).parameters
    user, item, *_ = model.embeddings()

    assert user.shape == (3, 9)
    assert item.shape == (4, 9)
    assert "item_profile" not in parameters
    assert not hasattr(model, "activity_item")
    assert not hasattr(model, "value_item")
    assert not hasattr(model, "gate_n")
    assert not hasattr(model, "gate_v")


def test_rho_zero_score_exactly_matches_ordinary_lightgcn_id_score():
    model = _model(rho=0.0, layers=1)
    users = torch.tensor([0, 1, 2])
    items = torch.tensor([3, 0, 1])
    full_user, full_item, *_ = model.embeddings()
    id_user, id_item = model.id_embeddings()

    full_score = (full_user[users] * full_item[items]).sum(1)
    id_score = (id_user[users] * id_item[items]).sum(1)
    torch.testing.assert_close(full_score, id_score, atol=0, rtol=0)
    assert model.representation_diagnostics()["rho_zero_auxiliary_max_abs"] == 0.0


def test_interaction_is_bounded_by_each_users_overall_clv_level():
    model = _model(layers=1)
    users = torch.arange(3).repeat_interleave(4)
    items = torch.arange(4).repeat(3)
    _, interaction, weighted = model.candidate_score_components(users, items)

    assert torch.all(interaction.abs() <= model.q_c[users] + 1e-6)
    assert torch.all(weighted.abs() <= model.rho * model.q_c[users] + 1e-6)


def test_interaction_changes_by_user_and_candidate_item():
    model = _model(layers=0)
    users = torch.tensor([0, 0, 1, 1])
    items = torch.tensor([0, 1, 0, 1])
    _, interaction, _ = model.candidate_score_components(users, items)

    assert interaction[0] != interaction[1]
    assert interaction[0] != interaction[2]


def test_invalid_user_has_exactly_zero_interaction():
    valid = np.array([True, False, True])
    model = CLVConditionedUserItemInteractionLightGCN(
        n_users=3,
        n_items=4,
        q_c=np.array([0.2, 0.0, 0.9], np.float32),
        d_nv=np.array([-0.7, 0.0, 0.8], np.float32),
        user_clv_valid=valid,
        adj=_adj(),
        id_dim=6,
        context_dim=3,
        n_layers=0,
    )
    user_interaction, _, _ = model.interaction_embeddings()

    torch.testing.assert_close(
        user_interaction[1], torch.zeros_like(user_interaction[1])
    )


def test_one_plain_bpr_jointly_trains_id_projection_and_context_parameters():
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
    assert model.overall_clv_context.grad.abs().sum() > 0
    assert model.nv_composition_context.grad.abs().sum() > 0


def test_m3_m4_and_external_score_paths_are_not_available():
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


def test_invalid_user_inputs_must_already_be_zero_masked():
    with pytest.raises(ValueError, match="계산 불가"):
        CLVConditionedUserItemInteractionLightGCN(
            n_users=3,
            n_items=4,
            q_c=np.array([0.2, 0.4, 0.9], np.float32),
            d_nv=np.array([-0.7, 0.2, 0.8], np.float32),
            user_clv_valid=np.array([True, False, True]),
            adj=_adj(),
            id_dim=6,
            context_dim=3,
        )

