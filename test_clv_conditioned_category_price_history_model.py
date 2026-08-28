import numpy as np
import pandas as pd
import pytest
import torch

from clv_conditioned_category_price_history_model import (
    ConditionedCategoryPriceHistoryLightGCN,
    build_conditioned_history_features,
)


def _frame():
    return pd.DataFrame(
        [
            dict(u_idx=0, i_idx=0, cat_idx=0, b_raw="a", t=1, v=10.0, up=1.0),
            dict(u_idx=0, i_idx=1, cat_idx=1, b_raw="b", t=4, v=30.0, up=3.0),
            dict(u_idx=1, i_idx=2, cat_idx=0, b_raw="c", t=4, v=20.0, up=2.0),
        ]
    )


def _edges():
    return np.array([0, 0, 1]), np.array([0, 1, 2])


def _adj():
    edge_users, edge_items = _edges()
    n_users, n_items = 2, 3
    user_degree = np.bincount(edge_users, minlength=n_users)
    item_degree = np.bincount(edge_items, minlength=n_items)
    weight = 1.0 / np.sqrt(user_degree[edge_users] * item_degree[edge_items])
    rows = np.concatenate([edge_users, n_users + edge_items])
    cols = np.concatenate([n_users + edge_items, edge_users])
    values = np.concatenate([weight, weight]).astype(np.float32)
    return torch.sparse_coo_tensor(
        torch.from_numpy(np.stack([rows, cols])),
        torch.from_numpy(values),
        (n_users + n_items, n_users + n_items),
    ).coalesce()


def _features():
    return build_conditioned_history_features(
        _frame(), n_users=2, n_items=3, n_categories=2, is_date=False
    )


def _model(rho=0.1):
    edge_users, edge_items = _edges()
    return ConditionedCategoryPriceHistoryLightGCN(
        n_users=2,
        n_items=3,
        n_categories=2,
        features=_features(),
        edge_users=edge_users,
        edge_items=edge_items,
        adj=_adj(),
        id_dim=6,
        category_dim=2,
        n_layers=2,
        rho=rho,
    )


def test_train_only_behavior_composites_and_history_are_well_formed():
    features = _features()

    assert features.user_state.shape == (2, 4)
    assert np.all((features.q_n >= 0) & (features.q_n <= 1))
    assert np.all((features.q_v >= 0) & (features.q_v <= 1))
    assert features.diagnostics["category_history_row_sum_max_error"] < 1e-7
    assert features.item_price_percentile.min() >= 0
    assert features.item_price_percentile.max() <= 1
    assert features.item_price_signal.min() >= -1
    assert features.item_price_signal.max() <= 1
    assert features.unique_item_count.tolist() == [2, 1]
    assert features.auxiliary_valid.tolist() == [True, False]


def test_zero_initialized_condition_mixer_starts_at_equal_composition():
    model = _model()
    gate = model._gate()

    torch.testing.assert_close(gate, torch.full_like(gate, 0.5))


def test_single_unique_item_user_has_zero_auxiliary_in_training_and_evaluation():
    model = _model()
    user_aux, _, hcat, hprice, category, gate = model._layer0_blocks()

    torch.testing.assert_close(user_aux[1], torch.zeros_like(user_aux[1]))
    loo = model._leave_one_out_auxiliary(
        torch.tensor([1]),
        torch.tensor([2]),
        hcat,
        hprice,
        category,
        gate,
    )
    torch.testing.assert_close(loo[0], torch.zeros_like(loo[0]))


def test_pairwise_leave_one_out_matches_brute_force_two_layer_propagation():
    model = _model()
    users = torch.tensor([0])
    positives = torch.tensor([0])
    negatives = torch.tensor([2])

    pair_user, pair_positive, pair_negative = model._pair_embeddings_with_exact_loo(
        users, positives, negatives
    )
    user_aux, item_aux, hcat, hprice, category, gate = model._layer0_blocks()
    loo_aux = model._leave_one_out_auxiliary(
        users, positives, hcat, hprice, category, gate
    )
    user_aux = user_aux.clone()
    user_aux[0] = loo_aux[0]
    scale = np.sqrt(model.rho)
    user0 = torch.cat([model.E_u.weight, scale * user_aux], dim=1)
    item0 = torch.cat([model.E_i.weight, scale * item_aux], dim=1)
    current = torch.cat([user0, item0], dim=0)
    total = current
    for _ in range(2):
        current = torch.sparse.mm(model.adj, current)
        total = total + current
    total = total / 3.0

    torch.testing.assert_close(pair_user[0], total[0], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(pair_positive[0], total[2], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(pair_negative[0], total[4], atol=1e-6, rtol=1e-6)


def test_rho_zero_is_exact_id_only_score_for_same_id_state():
    model = _model(rho=0.0)
    user, item, _, _ = model.embeddings()
    user_id, item_id = model.id_only_embeddings()

    torch.testing.assert_close(user[:, : model.id_dim], user_id)
    torch.testing.assert_close(item[:, : model.id_dim], item_id)
    torch.testing.assert_close(
        user[:, model.id_dim :], torch.zeros_like(user[:, model.id_dim :])
    )
    torch.testing.assert_close(
        item[:, model.id_dim :], torch.zeros_like(item[:, model.id_dim :])
    )
    torch.testing.assert_close(user @ item.T, user_id @ item_id.T)


def test_one_plain_bpr_updates_id_category_and_condition_mixer():
    model = _model()
    loss, diagnostics = model.bpr_loss(
        torch.tensor([0]), torch.tensor([0]), torch.tensor([2]), weights=None
    )
    loss.backward()

    assert diagnostics["objective"] == "plain_bpr"
    for parameter in (
        model.E_u.weight,
        model.E_i.weight,
        model.category_embedding.weight,
        model.condition_mixer.weight,
    ):
        assert parameter.grad is not None
        assert parameter.grad.abs().sum() > 0


def test_m3_m4_controls_cannot_enter_the_m2_loss():
    model = _model()
    with pytest.raises(ValueError, match="M4"):
        model.bpr_loss(
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([2]),
            weights=torch.ones(1),
        )
    with pytest.raises(ValueError, match="M4"):
        model.bpr_loss(
            torch.tensor([0]), torch.tensor([0]), torch.tensor([2]), lam=0.1
        )
