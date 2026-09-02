import numpy as np
import pandas as pd
import pytest
import torch

from clv_m3_clv_conditioned_candidate_item_graph import (
    ARM_ACTUAL,
    ARM_GENERAL,
    ARM_SHUFFLE,
    RELATION_MODE_COMMON_SUPPORT,
    RELATION_MODE_SUPPLEMENTAL,
    build_clv_conditioned_common_support_candidate_item_graph,
    build_clv_conditioned_candidate_item_graph,
    build_clv_conditioned_supplemental_candidate_item_graph,
)


def _train() -> pd.DataFrame:
    rows = []
    for user in range(8):
        high_clv = user >= 4
        value = 10.0 if high_clv else 1.0
        if high_clv:
            target_item = 2 if user % 2 == 0 else 5
        else:
            target_item = 1 if user % 2 == 0 else 4
        for basket, time, item, category in (
            (100 + user * 10, 1, 0, 0),
            (101 + user * 10, 2, target_item, 1),
            (102 + user * 10, 3, 3, 0),
        ):
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
    return pd.DataFrame(rows)


def _supplemental_train() -> pd.DataFrame:
    rows = []
    for user in range(12):
        high_clv = user >= 6
        value = 10.0 if high_clv else 1.0
        target_item = (4 if high_clv else 1) + (user % 3)
        for basket, time, item, category in (
            (100 + user * 10, 1, 0, 0),
            (101 + user * 10, 2, target_item, 1),
            (102 + user * 10, 3, 7, 0),
        ):
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
    return pd.DataFrame(rows)


def _graph():
    return build_clv_conditioned_candidate_item_graph(
        _train(),
        n_users=8,
        n_items=6,
        n_cat=2,
        category_kappa=0.0,
        category_min_support_users=1,
        item_kappa=0.0,
        item_min_support_users=1,
        shuffle_degree_bins=1,
        cross_fit_folds=2,
        max_target_categories=2,
        max_candidate_items=3,
    )


def _common_support_graph():
    return build_clv_conditioned_common_support_candidate_item_graph(
        _train(),
        n_users=8,
        n_items=6,
        n_cat=2,
        category_kappa=0.0,
        category_min_support_users=1,
        item_kappa=0.0,
        item_min_support_users=1,
        shuffle_degree_bins=1,
        cross_fit_folds=2,
        max_target_categories=2,
        max_candidate_items=3,
    )


def _supplemental_graph():
    return build_clv_conditioned_supplemental_candidate_item_graph(
        _supplemental_train(),
        n_users=12,
        n_items=8,
        n_cat=2,
        category_kappa=0.0,
        category_min_support_users=1,
        item_kappa=0.0,
        item_min_support_users=1,
        shuffle_degree_bins=1,
        cross_fit_folds=2,
        max_target_categories=2,
        base_candidate_items=1,
        supplemental_candidate_items=1,
    )


def test_clv_changes_direct_candidate_item_direction():
    graph = _graph()
    actual = graph.user_item_operators[ARM_ACTUAL].to_dense().numpy()

    assert actual[0, 4] > actual[0, 5]
    assert actual[4, 5] > actual[4, 4]
    np.testing.assert_allclose(actual.sum(axis=1), 1.0)
    assert graph.diagnostics["definition"]["item_price_used"] is False
    assert graph.diagnostics["arms"][ARM_ACTUAL][
        "max_active_row_mass_error"
    ] < 1e-7


def test_candidate_edges_exclude_every_users_train_items():
    train = _train()
    graph = _graph()
    seen = set(
        zip(
            train["u_idx"].to_numpy(np.int64),
            train["i_idx"].to_numpy(np.int64),
            strict=True,
        )
    )
    for operator in graph.user_item_operators.values():
        users, items = operator.indices().numpy()
        assert not any((int(user), int(item)) in seen for user, item in zip(users, items))


def test_general_and_shuffle_controls_are_distinct_and_catalog_is_preserved():
    graph = _graph()
    general = graph.user_item_operators[ARM_GENERAL].to_dense().numpy()
    actual = graph.user_item_operators[ARM_ACTUAL].to_dense().numpy()
    shuffle = graph.user_item_operators[ARM_SHUFFLE].to_dense().numpy()

    assert not np.array_equal(actual, general)
    assert not np.array_equal(actual, shuffle)
    np.testing.assert_allclose(
        np.sort(graph.clv_percentile),
        np.sort(graph.clv_shuffle_percentile),
    )
    assert graph.diagnostics["m1_catalog_items_preserved"] == 6


def test_minimum_item_support_removes_auxiliary_edges_not_catalog_items():
    graph = build_clv_conditioned_candidate_item_graph(
        _train(),
        n_users=8,
        n_items=6,
        n_cat=2,
        category_min_support_users=1,
        item_min_support_users=8,
        cross_fit_folds=2,
    )

    assert graph.user_item_operators[ARM_ACTUAL]._nnz() == 0
    assert graph.diagnostics["m1_catalog_items_preserved"] == 6


def test_common_support_is_exactly_shared_while_clv_changes_weights():
    graph = _common_support_graph()
    operators = graph.user_item_operators
    general = operators[ARM_GENERAL]
    actual = operators[ARM_ACTUAL]
    shuffle = operators[ARM_SHUFFLE]

    assert torch.equal(general.indices(), actual.indices())
    assert torch.equal(general.indices(), shuffle.indices())
    assert not torch.equal(general.values(), actual.values())
    assert not torch.equal(actual.values(), shuffle.values())
    for operator in operators.values():
        np.testing.assert_allclose(
            operator.to_dense().numpy().sum(axis=1),
            1.0,
        )
    assert graph.diagnostics["settings"]["relation_mode"] == (
        RELATION_MODE_COMMON_SUPPORT
    )
    assert graph.diagnostics["definition"]["positive_excess_clipping"] is False
    assert graph.diagnostics["common_support"]["exact_common_edge_support"] is True


def test_common_support_uses_clv_for_direction_not_candidate_availability():
    graph = _common_support_graph()
    general = graph.user_item_operators[ARM_GENERAL].to_dense().numpy()
    actual = graph.user_item_operators[ARM_ACTUAL].to_dense().numpy()

    np.testing.assert_array_equal(general > 0, actual > 0)
    assert actual[0, 4] > actual[0, 5]
    assert actual[4, 5] > actual[4, 4]


def test_supplemental_graph_preserves_base_and_matches_block_mass():
    graph = _supplemental_graph()
    operators = graph.user_item_operators

    for operator in operators.values():
        dense = operator.to_dense().numpy()
        np.testing.assert_allclose(dense.sum(axis=1), 1.0)
        assert np.all((dense > 0).sum(axis=1) == 2)
    support = graph.diagnostics["supplemental_support"]
    assert support["base_edges_identical"] is True
    assert support["base_mass"] == pytest.approx(0.5)
    assert support["extra_mass"] == pytest.approx(0.5)
    assert support["max_base_mass_error"] < 1e-7
    assert support["max_extra_mass_error"] < 1e-7
    assert graph.diagnostics["settings"]["relation_mode"] == (
        RELATION_MODE_SUPPLEMENTAL
    )


def test_supplemental_candidates_are_outside_base_and_train_pairs():
    graph = _supplemental_graph()
    support = graph.diagnostics["supplemental_support"]

    assert support["base_extra_overlap"] == 0
    assert support["train_pair_edges"] == 0
    assert support["edges_per_active_user"] == 2


def test_supplemental_graph_fails_when_positive_excess_is_insufficient():
    with pytest.raises(RuntimeError, match="positive excess candidates"):
        build_clv_conditioned_supplemental_candidate_item_graph(
            _supplemental_train(),
            n_users=12,
            n_items=8,
            n_cat=2,
            base_candidate_items=4,
            supplemental_candidate_items=2,
            category_kappa=0.0,
            category_min_support_users=1,
            item_kappa=0.0,
            item_min_support_users=1,
            shuffle_degree_bins=1,
            cross_fit_folds=2,
            max_target_categories=2,
        )
