import numpy as np
import pandas as pd

from clv_m3_tfidf_neighbor_graph import (
    build_degree_matched_random_neighbor_operator,
    build_historical_clv_gates,
    build_ordinary_copurchase_operator,
    build_tfidf_neighbor_operator,
    top_candidate_items,
)


def _train() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "u_idx": [0, 0, 1, 1, 2, 2],
            "i_idx": [0, 1, 0, 1, 2, 3],
            "b_raw": [10, 11, 20, 21, 30, 31],
            "t": [1, 2, 1, 2, 1, 2],
            "v": [2.0, 6.0, 4.0, 8.0, 3.0, 9.0],
        }
    )


def test_tfidf_topk_selects_same_taste_user_and_normalizes_rows():
    relation, diagnostics = build_tfidf_neighbor_operator(
        _train(), n_users=3, n_items=4, top_k=1
    )

    dense = relation.toarray()
    assert dense[0, 1] == 1.0
    assert dense[1, 0] == 1.0
    assert dense[2].sum() == 0.0
    assert np.allclose(dense.sum(axis=1), [1.0, 1.0, 0.0])
    assert diagnostics["eligible_user_count"] == 2
    assert diagnostics["self_edge_count"] == 0


def test_historical_clv_is_distinct_baskets_times_mean_basket_value():
    gates = build_historical_clv_gates(
        _train(), n_users=3, shuffle_degree_bins=1, shuffle_seed=42
    )

    np.testing.assert_allclose(gates.n_hat, [2.0, 2.0, 2.0])
    np.testing.assert_allclose(gates.v_hat, [4.0, 6.0, 6.0])
    np.testing.assert_allclose(gates.clv_proxy, [8.0, 12.0, 12.0])
    np.testing.assert_allclose(
        np.sort(gates.clv_percentile),
        np.sort(gates.clv_shuffle_percentile),
    )
    assert np.isclose(gates.constant_gate.mean(), gates.clv_percentile.mean())
    assert np.all((gates.degree_percentile > 0) & (gates.degree_percentile < 1))


def test_ordinary_copurchase_is_m1_symmetric_normalized_two_hop():
    relation, diagnostics = build_ordinary_copurchase_operator(
        _train(), n_users=3, n_items=4, top_k=1
    )

    dense = relation.toarray()
    assert dense[0, 1] == 1.0
    assert dense[1, 0] == 1.0
    assert dense[2].sum() == 0.0
    assert diagnostics["normalization"] == "m1_symmetric_two_hop_then_topk_row"


def test_random_neighbors_stay_inside_degree_strata_and_exclude_self():
    operator, diagnostics = build_degree_matched_random_neighbor_operator(
        np.array([2, 2, 2, 2]), top_k=2, n_bins=1, seed=42
    )

    dense = operator.toarray()
    np.testing.assert_allclose(dense.sum(axis=1), np.ones(4))
    np.testing.assert_array_equal(np.diag(dense), np.zeros(4))
    assert diagnostics["neighbor_count_max"] == 2


def test_candidate_items_exclude_users_observed_train_pairs():
    relation, _ = build_tfidf_neighbor_operator(
        _train(), n_users=3, n_items=4, top_k=1
    )
    candidates = top_candidate_items(
        relation, _train(), n_users=3, n_items=4, candidate_count=2
    )

    assert set(candidates[0]).isdisjoint({0, 1})
    assert set(candidates[1]).isdisjoint({0, 1})
    assert len(candidates[2]) == 0

