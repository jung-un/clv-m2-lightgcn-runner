import numpy as np
import pandas as pd
import pytest
import torch

from clv_history_item_fit_model import (
    HistoryItemFitLightGCN,
    build_personal_history_weights,
)


def _adj(n_users=2, n_items=3):
    edges = [(0, 0), (0, 1), (1, 1), (1, 2)]
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


def _history_frame():
    return pd.DataFrame(
        [
            {"u_idx": 0, "i_idx": 0, "b_raw": "a", "v": 10.0},
            {"u_idx": 0, "i_idx": 0, "b_raw": "b", "v": 20.0},
            {"u_idx": 0, "i_idx": 1, "b_raw": "c", "v": 70.0},
            {"u_idx": 1, "i_idx": 1, "b_raw": "d", "v": 40.0},
            {"u_idx": 1, "i_idx": 2, "b_raw": "e", "v": 60.0},
        ]
    )


def _model():
    history = build_personal_history_weights(
        _history_frame(), n_users=2, n_items=3
    )
    return HistoryItemFitLightGCN(
        n_users=2,
        n_items=3,
        history=history,
        q_n=np.array([0.8, 0.2], np.float32),
        q_v=np.array([0.3, 0.9], np.float32),
        activity_valid=np.ones(2, bool),
        value_valid=np.ones(2, bool),
        adj=_adj(),
        id_dim=6,
        axis_dim=2,
        n_layers=1,
        rho=0.05,
    )


def test_personal_history_weights_are_normalized_within_each_user():
    history = build_personal_history_weights(
        _history_frame(), n_users=2, n_items=3
    )
    key = history.users * 3 + history.items
    by_key = {
        int(pair): (float(n), float(v))
        for pair, n, v in zip(
            key, history.activity_share, history.value_share, strict=True
        )
    }

    assert by_key[0][0] == pytest.approx(2 / 3)
    assert by_key[1][0] == pytest.approx(1 / 3)
    assert by_key[0][1] == pytest.approx(0.3)
    assert by_key[1][1] == pytest.approx(0.7)
    assert history.diagnostics["activity_row_sum_max_error"] < 1e-7
    assert history.diagnostics["value_row_sum_max_error"] < 1e-7


def test_positive_item_is_removed_and_remaining_history_is_renormalized():
    model = _model()
    users = torch.tensor([0])
    positives = torch.tensor([0])
    profile_n, _, profile_v, _ = model._training_profiles(users, positives)
    source_n = model._unit_rows(model.activity_source.weight)
    source_v = model._unit_rows(model.value_source.weight)

    torch.testing.assert_close(profile_n[0], 0.8 * source_n[1])
    torch.testing.assert_close(profile_v[0], 0.3 * source_v[1])


def test_one_bpr_updates_id_and_all_source_target_axis_tables():
    model = _model()
    loss, diagnostics = model.bpr_loss(
        torch.tensor([0, 1]),
        torch.tensor([0, 2]),
        torch.tensor([2, 0]),
        weights=None,
    )
    loss.backward()

    assert diagnostics["objective"] == "plain_bpr"
    for parameter in (
        model.E_u.weight,
        model.E_i.weight,
        model.activity_source.weight,
        model.activity_target.weight,
        model.value_source.weight,
        model.value_target.weight,
    ):
        assert parameter.grad is not None
        assert parameter.grad.abs().sum() > 0


def test_evaluation_embeddings_form_one_id_n_v_dot_space():
    model = _model()
    users, items, _, _ = model.embeddings()

    assert users.shape == (2, 10)
    assert items.shape == (3, 10)
    assert model.total_dim == 10
    assert "rho" not in dict(model.named_parameters())


def test_m4_sample_weights_cannot_enter_the_m2_loss():
    model = _model()
    with pytest.raises(ValueError, match="M4"):
        model.bpr_loss(
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([2]),
            weights=torch.ones(1),
        )
