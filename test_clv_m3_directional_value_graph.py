import numpy as np
import pandas as pd

from clv_m3_directional_value_graph import (
    ARM_ACTUAL,
    ARM_RELATION_ONLY,
    ARM_SHUFFLE,
    build_directional_value_graph,
    build_mass_preserving_coefficients,
)


def _train() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "u_idx": [0, 0, 0, 0, 1, 1, 1, 1, 1, 2],
            "i_idx": [0, 1, 0, 1, 0, 1, 2, 0, 2, 2],
            "b_raw": [10, 10, 11, 11, 20, 20, 20, 21, 21, 30],
            "t": [1, 1, 2, 2, 1, 1, 1, 2, 2, 1],
            "v": [2.0, 8.0, 9.0, 1.0, 1.0, 1.0, 8.0, 8.0, 12.0, 5.0],
        }
    )


def test_train_only_clv_and_edge_contribution_are_literal():
    graph = build_directional_value_graph(
        _train(), 3, 3, target_strength=0.02, shuffle_degree_bins=1
    )

    np.testing.assert_allclose(graph.n_hat, [2.0, 2.0, 1.0])
    np.testing.assert_allclose(graph.v_hat, [10.0, 15.0, 5.0])
    np.testing.assert_allclose(graph.clv_proxy, [20.0, 30.0, 5.0])
    np.testing.assert_allclose(graph.clv_proxy, graph.n_hat * graph.v_hat)
    np.testing.assert_allclose(graph.clv_percentile, [0.5, 5 / 6, 1 / 6])

    edge = list(zip(graph.edge_users.tolist(), graph.edge_items.tolist()))
    contribution = dict(zip(edge, graph.edge_contribution.tolist()))
    assert np.isclose(contribution[(0, 0)], 0.55)
    assert np.isclose(contribution[(0, 1)], 0.45)
    assert np.isclose(contribution[(1, 0)], 0.25)
    assert np.isclose(contribution[(1, 1)], 0.10)
    assert np.isclose(contribution[(1, 2)], 0.70)
    assert np.isclose(contribution[(2, 2)], 1.00)


def test_relation_is_ranked_only_inside_each_users_purchased_items():
    graph = build_directional_value_graph(
        _train(), 3, 3, target_strength=0.02, shuffle_degree_bins=1
    )
    edge = list(zip(graph.edge_users.tolist(), graph.edge_items.tolist()))
    relation = dict(zip(edge, graph.within_user_relation.tolist()))

    assert np.isclose(relation[(0, 0)], 0.5)
    assert np.isclose(relation[(0, 1)], -0.5)
    assert np.isclose(relation[(1, 0)], 0.0)
    assert np.isclose(relation[(1, 1)], -2 / 3)
    assert np.isclose(relation[(1, 2)], 2 / 3)
    assert np.isclose(relation[(2, 2)], 0.0)


def test_every_active_arm_preserves_each_users_m1_first_hop_mass():
    graph = build_directional_value_graph(
        _train(), 3, 3, target_strength=0.02, shuffle_degree_bins=1
    )
    base_mass = np.bincount(
        graph.edge_users, weights=graph.base_coefficients, minlength=3
    )
    for arm in (ARM_RELATION_ONLY, ARM_ACTUAL, ARM_SHUFFLE):
        adjusted_mass = np.bincount(
            graph.edge_users,
            weights=graph.user_from_item_coefficients[arm],
            minlength=3,
        )
        np.testing.assert_allclose(adjusted_mass, base_mass, atol=1e-7)
        assert graph.diagnostics["arms"][arm]["max_user_mass_abs_error"] < 1e-7


def test_zero_clv_gate_is_exact_binary_m1_for_that_user():
    edge_users = np.array([0, 0, 1], dtype=np.int64)
    base = np.array([0.2, 0.4, 0.5], dtype=np.float64)
    relation = np.array([-0.5, 0.5, 0.0], dtype=np.float64)
    gate = np.array([0.0, 1.0], dtype=np.float64)

    adjusted = build_mass_preserving_coefficients(
        base, edge_users, relation, gate, beta=2.0, n_users=2
    )
    np.testing.assert_array_equal(adjusted[:2], base[:2])


def test_degree_matched_shuffle_preserves_values_within_strata_but_reassigns_them():
    graph = build_directional_value_graph(
        _train(),
        3,
        3,
        target_strength=0.02,
        shuffle_degree_bins=1,
        shuffle_seed=42,
    )
    np.testing.assert_allclose(
        np.sort(graph.clv_percentile), np.sort(graph.clv_shuffle_percentile)
    )
    assert not np.array_equal(graph.clv_percentile, graph.clv_shuffle_percentile)
    assert np.all(graph.clv_shuffle_stratum == 0)


def test_all_three_active_arms_are_matched_to_the_same_first_hop_strength():
    target = 0.02
    graph = build_directional_value_graph(
        _train(),
        3,
        3,
        target_strength=target,
        beta_cap=20.0,
        shuffle_degree_bins=1,
    )
    for arm in (ARM_RELATION_ONLY, ARM_ACTUAL, ARM_SHUFFLE):
        assert np.isclose(
            graph.diagnostics["arms"][arm]["first_hop_strength"],
            target,
            atol=1e-6,
        )

