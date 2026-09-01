import numpy as np
import pandas as pd

from clv_m3_clv_conditioned_category_transition_graph import (
    ARM_ACTUAL,
    ARM_GENERAL,
    ARM_SHUFFLE,
    _transition_evidence,
    build_clv_conditioned_category_transition_graph,
)


def _train() -> tuple[pd.DataFrame, int]:
    rows = []
    next_item = 1
    for user in range(6):
        low = user < 3
        value = 1.0 if low else 10.0
        target_category = 1 if low else 2
        baskets = [
            (100 + 10 * user, 1, 0, 0),
            (101 + 10 * user, 2, next_item, target_category),
            (102 + 10 * user, 3, next_item + 1, 0),
        ]
        next_item += 2
        for basket, time, item, category in baskets:
            rows.append(
                {
                    "u_idx": user,
                    "i_idx": item,
                    "cat_idx": category,
                    "b_raw": basket,
                    "t": time,
                    "v": value,
                }
            )
    return pd.DataFrame(rows), next_item


def test_clv_changes_target_category_direction_with_cross_fitted_evidence():
    train, n_items = _train()
    graph = build_clv_conditioned_category_transition_graph(
        train,
        n_users=6,
        n_items=n_items,
        n_cat=3,
        kappa=0.0,
        min_support_users=1,
        log_lift_cap=np.log(10.0),
        shuffle_degree_bins=1,
        cross_fit_folds=2,
    )

    actual = graph.user_category_operators[ARM_ACTUAL].to_dense().numpy()
    general = graph.user_category_operators[ARM_GENERAL].to_dense().numpy()
    assert actual[0, 1] > actual[0, 2]
    assert actual[5, 2] > actual[5, 1]
    np.testing.assert_allclose(general.sum(axis=1), 1.0)
    np.testing.assert_allclose(actual.sum(axis=1), 1.0)

    diagnostics = graph.diagnostics
    assert diagnostics["definition"]["item_price_used"] is False
    assert diagnostics["settings"]["cross_fit_folds"] == 2
    assert diagnostics["arms"][ARM_ACTUAL]["max_active_row_mass_error"] < 1e-7
    assert all(
        fold["supported_edges"] > 0
        for fold in diagnostics["arms"][ARM_ACTUAL]["folds"]
    )


def test_full_historical_clv_and_shuffle_are_literal_and_catalog_is_preserved():
    train, n_items = _train()
    graph = build_clv_conditioned_category_transition_graph(
        train,
        n_users=6,
        n_items=n_items,
        n_cat=3,
        kappa=1.0,
        min_support_users=1,
        shuffle_degree_bins=1,
        cross_fit_folds=2,
    )

    np.testing.assert_allclose(graph.clv_proxy, graph.n_hat * graph.v_hat)
    np.testing.assert_allclose(
        np.sort(graph.clv_percentile),
        np.sort(graph.clv_shuffle_percentile),
    )
    assert not np.array_equal(
        graph.clv_percentile, graph.clv_shuffle_percentile
    )
    assert graph.user_category_operators[ARM_SHUFFLE].shape == (6, 3)
    assert graph.category_item_operator.shape == (3, n_items)
    np.testing.assert_allclose(
        graph.category_item_operator.to_dense().numpy().sum(axis=1),
        1.0,
    )


def test_minimum_support_removes_auxiliary_relation_not_catalog_items():
    train, n_items = _train()
    graph = build_clv_conditioned_category_transition_graph(
        train,
        n_users=6,
        n_items=n_items,
        n_cat=3,
        min_support_users=6,
        cross_fit_folds=2,
    )

    assert graph.user_category_operators[ARM_ACTUAL]._nnz() == 0
    assert graph.user_category_operators[ARM_GENERAL]._nnz() == 0
    assert graph.category_item_operator._nnz() == n_items


def test_transition_evidence_uses_user_date_when_basket_id_is_absent():
    train, _ = _train()
    hm_style = train.drop(columns="b_raw")

    pair, recent, diagnostics = _transition_evidence(hm_style, n_users=6)

    assert not pair.empty
    assert recent.groupby("u_idx").recent_share.sum().eq(1.0).all()
    assert diagnostics["n_baskets"] == hm_style[["u_idx", "t"]].drop_duplicates().shape[0]
