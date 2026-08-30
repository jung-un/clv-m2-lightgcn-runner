import numpy as np
import pandas as pd
import pytest
import torch

from clv_conditioned_category_price_history_model import (
    build_conditioned_history_features,
)
from clv_conditioned_centered_balanced_history_model import (
    CenteredBalancedHistoryLightGCN,
)


def _frame():
    rows = []
    values = ((0, 0, 0, 10.0), (0, 1, 1, 30.0), (1, 1, 1, 20.0),
              (1, 2, 0, 50.0), (2, 2, 0, 15.0), (2, 3, 1, 60.0))
    for basket, (user, item, category, amount) in enumerate(values):
        rows.append(
            dict(
                u_idx=user,
                i_idx=item,
                cat_idx=category,
                b_raw=f"b{basket}",
                t=1 + basket,
                v=amount,
                up=amount,
            )
        )
    return pd.DataFrame(rows)


def _edges():
    frame = _frame()
    return frame.u_idx.to_numpy(np.int64), frame.i_idx.to_numpy(np.int64)


def _adj():
    edge_users, edge_items = _edges()
    n_users, n_items = 3, 4
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


def _model():
    edge_users, edge_items = _edges()
    features = build_conditioned_history_features(
        _frame(), n_users=3, n_items=4, n_categories=2, is_date=False
    )
    return CenteredBalancedHistoryLightGCN(
        n_users=3,
        n_items=4,
        n_categories=2,
        features=features,
        edge_users=edge_users,
        edge_items=edge_items,
        adj=_adj(),
        id_dim=6,
        category_dim=2,
        n_layers=2,
        rho=0.1,
        warmup_epochs=20,
    )


def test_bounded_mixer_starts_equal_and_never_drops_a_branch():
    model = _model()
    torch.testing.assert_close(model._gate(), torch.full((3, 2), 0.5))
    with torch.no_grad():
        model.condition_mixer.weight.fill_(100.0)
    gate = model._gate()
    assert float(gate.min().detach()) >= 0.25
    assert float(gate.max().detach()) <= 0.75
    torch.testing.assert_close(gate.sum(1), torch.ones(3))


def test_rho_warmup_reaches_declared_maximum():
    model = _model()
    model.set_training_epoch(1)
    assert model.rho == pytest.approx(0.005)
    model.set_training_epoch(10)
    assert model.rho == pytest.approx(0.05)
    model.set_training_epoch(20)
    assert model.rho == pytest.approx(0.1)
    model.set_training_epoch(100)
    assert model.rho == pytest.approx(0.1)


def test_histories_are_population_centered_then_row_normalized():
    model = _model()
    _, _, category_history, price_history, _, _ = model._layer0_blocks()
    valid = model.auxiliary_valid.bool()
    assert torch.all(category_history[valid].norm(dim=1) <= 1.000001)
    assert torch.all(price_history[valid].norm(dim=1) <= 1.000001)
    diagnostics = model.representation_diagnostics()
    assert diagnostics["category_centered_population_mean_norm"] < 1e-6
    assert diagnostics["price_centered_population_mean_norm"] < 1e-6


def test_exact_leave_one_out_matches_brute_force_propagation():
    model = _model()
    users = torch.tensor([0])
    positives = torch.tensor([0])
    negatives = torch.tensor([3])
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
    current = torch.cat(
        [
            torch.cat([model.E_u.weight, scale * user_aux], dim=1),
            torch.cat([model.E_i.weight, scale * item_aux], dim=1),
        ],
        dim=0,
    )
    total = current
    for _ in range(2):
        current = torch.sparse.mm(model.adj, current)
        total = total + current
    total = total / 3.0
    torch.testing.assert_close(pair_user[0], total[0], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(pair_positive[0], total[3], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(pair_negative[0], total[6], atol=1e-6, rtol=1e-6)


def test_one_bpr_backward_reaches_id_history_and_condition_parameters():
    model = _model()
    loss, _ = model.bpr_loss(
        torch.tensor([0, 1]),
        torch.tensor([0, 1]),
        torch.tensor([3, 0]),
    )
    loss.backward()
    for parameter in (
        model.E_u.weight,
        model.E_i.weight,
        model.category_embedding.weight,
        model.condition_mixer.weight,
    ):
        assert parameter.grad is not None
        assert parameter.grad.abs().sum() > 0
