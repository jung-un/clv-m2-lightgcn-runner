import numpy as np
import pandas as pd
from scipy import sparse

from clv_m3_clv_taste_neighbor_graph import (
    ACTUAL_CLV,
    CLV_SHUFFLE,
    DEGREE_RELATION,
    PREFERENCE_RELATION,
    build_clv_taste_neighbor_graph,
    build_historical_clv_features,
    build_neighbor_operators,
)


def _candidate_matrix():
    return sparse.csr_matrix(
        (
            [0.90, 0.80, 0.70, 0.90, 0.80, 0.90, 0.80, 0.90, 0.80],
            (
                [0, 0, 0, 1, 1, 2, 2, 3, 3],
                [1, 2, 3, 0, 2, 0, 1, 0, 2],
            ),
        ),
        shape=(4, 4),
    )


def test_clv_features_keep_total_level_and_nv_composition_separate():
    train = pd.DataFrame(
        {
            "u_idx": [0, 0, 0, 1, 1, 2, 2, 3, 3],
            "i_idx": [0, 1, 2, 0, 3, 1, 3, 0, 2],
            "b_raw": [10, 10, 11, 20, 21, 30, 31, 40, 41],
            "v": [4.0, 6.0, 30.0, 50.0, 50.0, 15.0, 15.0, 2.0, 8.0],
        }
    )
    features = build_historical_clv_features(
        train,
        n_users=4,
        n_items=4,
        reliability_kappa=5.0,
        degree_bins=1,
        shuffle_seed=42,
    )

    np.testing.assert_allclose(features.n_hat, [2.0, 2.0, 2.0, 2.0])
    np.testing.assert_allclose(features.v_hat, [20.0, 50.0, 15.0, 5.0])
    np.testing.assert_allclose(features.clv_proxy, [40.0, 100.0, 30.0, 10.0])
    np.testing.assert_allclose(
        features.composition_coordinate,
        (features.q_n - features.q_v + 1.0) / 2.0,
    )
    assert np.all((features.q_clv >= 0) & (features.q_clv <= 1))
    assert np.all((features.composition_coordinate >= 0) & (features.composition_coordinate <= 1))
    assert np.all(features.reliability < 1.0)


def test_tuple_shuffle_preserves_every_clv_coordinate_within_degree_stratum():
    train = pd.DataFrame(
        {
            "u_idx": np.repeat(np.arange(6), 2),
            "i_idx": np.tile([0, 1], 6),
            "b_raw": np.arange(12),
            "v": np.arange(1, 13, dtype=float),
        }
    )
    features = build_historical_clv_features(
        train,
        n_users=6,
        n_items=2,
        degree_bins=1,
        shuffle_seed=42,
    )
    actual = np.column_stack(
        [features.q_clv, features.q_n, features.q_v, features.composition_coordinate]
    )
    shuffled = np.column_stack(
        [
            features.shuffled_q_clv,
            features.shuffled_q_n,
            features.shuffled_q_v,
            features.shuffled_composition_coordinate,
        ]
    )

    np.testing.assert_allclose(
        actual[np.lexsort(actual.T[::-1])],
        shuffled[np.lexsort(shuffled.T[::-1])],
    )
    assert features.diagnostics["shuffle_preserves_tuple_multiset_by_degree_stratum"]
    assert features.diagnostics["shuffle_source_changed_user_share"] > 0


def test_clv_only_reorders_neighbors_inside_the_fixed_taste_shortlist():
    candidates = _candidate_matrix()
    operators, diagnostics = build_neighbor_operators(
        candidates,
        q_clv=np.array([0.90, 0.10, 0.90, 0.50]),
        composition_coordinate=np.array([0.80, 0.20, 0.80, 0.50]),
        shuffled_q_clv=np.array([0.90, 0.90, 0.10, 0.50]),
        shuffled_composition_coordinate=np.array([0.80, 0.80, 0.20, 0.50]),
        degree_percentile=np.array([0.60, 0.55, 0.10, 0.90]),
        reliability=np.full(4, 0.90),
        final_neighbors=2,
    )

    pref_neighbors = set(operators[PREFERENCE_RELATION][0].indices)
    actual_neighbors = set(operators[ACTUAL_CLV][0].indices)
    candidate_neighbors = set(candidates[0].indices)
    assert pref_neighbors == {1, 2}
    assert actual_neighbors == {2, 3}
    assert actual_neighbors <= candidate_neighbors
    assert diagnostics["all_arms_use_preference_candidate_support"]


def test_every_arm_has_same_neighbor_count_and_unit_row_mass():
    operators, diagnostics = build_neighbor_operators(
        _candidate_matrix(),
        q_clv=np.array([0.90, 0.10, 0.90, 0.50]),
        composition_coordinate=np.array([0.80, 0.20, 0.80, 0.50]),
        shuffled_q_clv=np.array([0.90, 0.90, 0.10, 0.50]),
        shuffled_composition_coordinate=np.array([0.80, 0.80, 0.20, 0.50]),
        degree_percentile=np.array([0.60, 0.55, 0.10, 0.90]),
        reliability=np.full(4, 0.90),
        final_neighbors=2,
    )

    reference_count = np.diff(operators[PREFERENCE_RELATION].indptr)
    for arm in (PREFERENCE_RELATION, ACTUAL_CLV, CLV_SHUFFLE, DEGREE_RELATION):
        operator = operators[arm]
        np.testing.assert_array_equal(np.diff(operator.indptr), reference_count)
        mass = np.asarray(operator.sum(axis=1)).ravel()
        np.testing.assert_allclose(mass[reference_count > 0], 1.0)
        assert operator.diagonal().sum() == 0
    assert diagnostics["same_neighbor_count_all_arms"]
    assert diagnostics["same_row_mass_all_arms"]


def test_end_to_end_graph_uses_binary_purchase_tfidf_and_excludes_self():
    train = pd.DataFrame(
        {
            "u_idx": [0, 0, 1, 1, 2, 2, 3, 3],
            "i_idx": [0, 1, 0, 2, 0, 3, 4, 5],
            "b_raw": np.arange(8),
            "v": [10.0, 20.0, 9.0, 30.0, 8.0, 40.0, 5.0, 5.0],
        }
    )
    graph = build_clv_taste_neighbor_graph(
        train,
        n_users=4,
        n_items=6,
        candidate_neighbors=3,
        final_neighbors=2,
        reliability_kappa=5.0,
        degree_bins=1,
        shuffle_seed=42,
    )

    assert graph.preference_candidates.shape == (4, 4)
    assert graph.preference_candidates.diagonal().sum() == 0
    assert set(graph.operators) == {
        PREFERENCE_RELATION,
        ACTUAL_CLV,
        CLV_SHUFFLE,
        DEGREE_RELATION,
    }
    assert graph.diagnostics["quality_passed"]
