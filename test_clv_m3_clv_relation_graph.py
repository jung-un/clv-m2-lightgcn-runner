import numpy as np
import pandas as pd

from clv_m3_clv_relation_graph import build_clv_relation_graph


def _train() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "u_idx": [0, 0, 0, 1, 1, 2, 2, 2],
            "i_idx": [0, 0, 1, 0, 2, 1, 2, 2],
            "up": [2.0, 2.0, 8.0, 2.0, 5.0, 8.0, 5.0, 5.0],
        }
    )


def test_all_variants_share_binary_edges_and_user_mean_one():
    graph = build_clv_relation_graph(
        _train(), 3, 3, np.array([0.1, 0.5, 0.9]), target_strength=0.075
    )
    assert graph.edge_users.tolist() == [0, 0, 1, 1, 2, 2]
    assert graph.edge_items.tolist() == [0, 1, 0, 2, 1, 2]
    for weights in (
        graph.relation_only_weights,
        graph.clv_gate_weights,
        graph.allocated_relation_only_weights,
        graph.clv_allocated_gate_weights,
    ):
        assert np.all(np.isfinite(weights))
        assert np.all(weights > 0)
        for user in range(3):
            assert np.isclose(weights[graph.edge_users == user].mean(), 1.0)


def test_higher_clv_moves_farther_from_binary_relation_graph():
    graph = build_clv_relation_graph(
        _train(), 3, 3, np.array([0.1, 0.5, 0.9]), target_strength=0.075
    )
    for user in range(3):
        mask = graph.edge_users == user
        gate_distance = np.abs(graph.clv_gate_weights[mask] - 1.0).mean()
        relation_distance = np.abs(graph.relation_only_weights[mask] - 1.0).mean()
        if relation_distance > 0:
            assert gate_distance > 0
    low = np.abs(graph.clv_gate_weights[graph.edge_users == 0] - 1.0).mean()
    high = np.abs(graph.clv_gate_weights[graph.edge_users == 2] - 1.0).mean()
    assert high > low


def test_allocated_proposal_uses_clv_inside_relation_signal():
    low_first = build_clv_relation_graph(
        _train(), 3, 3, np.array([0.1, 0.5, 0.9]), target_strength=0.075
    )
    high_first = build_clv_relation_graph(
        _train(), 3, 3, np.array([0.9, 0.5, 0.1]), target_strength=0.075
    )
    assert not np.allclose(
        low_first.allocated_relation_signal,
        high_first.allocated_relation_signal,
    )
    assert np.allclose(
        low_first.relation_signal,
        high_first.relation_signal,
    )


def test_weights_do_not_depend_on_price_values():
    train = _train()
    first = build_clv_relation_graph(
        train, 3, 3, np.array([0.1, 0.5, 0.9]), target_strength=0.075
    )
    changed = train.copy()
    changed["up"] = changed["up"] * np.array([1, 3, 2, 5, 7, 11, 13, 17])
    second = build_clv_relation_graph(
        changed, 3, 3, np.array([0.1, 0.5, 0.9]), target_strength=0.075
    )
    for name in (
        "relation_only_weights",
        "clv_gate_weights",
        "allocated_relation_only_weights",
        "clv_allocated_gate_weights",
    ):
        assert np.allclose(getattr(first, name), getattr(second, name))


def test_degree_stratified_shuffle_preserves_clv_values_inside_each_stratum():
    graph = build_clv_relation_graph(
        _train(),
        3,
        3,
        np.array([0.1, 0.5, 0.9]),
        target_strength=0.075,
        shuffle_seed=42,
        shuffle_degree_bins=10,
    )

    assert not np.array_equal(
        graph.clv_percentile,
        graph.clv_shuffle_percentile,
    )
    for stratum in np.unique(graph.clv_shuffle_stratum):
        mask = graph.clv_shuffle_stratum == stratum
        np.testing.assert_allclose(
            np.sort(graph.clv_percentile[mask]),
            np.sort(graph.clv_shuffle_percentile[mask]),
        )


def test_shuffled_clv_is_recomputed_in_both_allocated_relation_and_gate():
    first = build_clv_relation_graph(
        _train(),
        3,
        3,
        np.array([0.1, 0.5, 0.9]),
        target_strength=0.075,
        shuffle_seed=42,
    )
    repeated = build_clv_relation_graph(
        _train(),
        3,
        3,
        np.array([0.1, 0.5, 0.9]),
        target_strength=0.075,
        shuffle_seed=42,
    )

    np.testing.assert_allclose(
        first.clv_allocated_gate_shuffle_weights,
        repeated.clv_allocated_gate_shuffle_weights,
    )
    assert not np.allclose(
        first.allocated_relation_signal,
        first.allocated_relation_shuffle_signal,
    )
    assert not np.allclose(
        first.clv_allocated_gate_weights,
        first.clv_allocated_gate_shuffle_weights,
    )
    for user in range(3):
        mask = first.edge_users == user
        assert np.isclose(
            first.clv_allocated_gate_shuffle_weights[mask].mean(),
            1.0,
        )
