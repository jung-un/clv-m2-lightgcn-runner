import numpy as np
import pandas as pd
import torch

from clv_m3_mass_preserving_graph import (
    MODES,
    build_directional_torch_adj,
    build_mass_preserving_clv_graph,
)


def _train() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "u_idx": [0, 0, 0, 1, 1, 2, 2, 2],
            "i_idx": [0, 1, 1, 0, 2, 0, 1, 2],
            "b_raw": [10, 10, 11, 20, 21, 30, 31, 32],
            "t": [1, 1, 2, 1, 2, 1, 2, 3],
            "v": [4.0, 6.0, 20.0, 5.0, 5.0, 10.0, 10.0, 10.0],
        }
    )


def test_n_v_and_clv_use_train_baskets_exactly():
    graph = build_mass_preserving_clv_graph(_train(), 3, 3)
    assert np.allclose(graph.n_hat, [2.0, 2.0, 3.0])
    assert np.allclose(graph.v_hat, [15.0, 5.0, 10.0])
    assert np.allclose(graph.clv_proxy, [30.0, 10.0, 30.0])
    assert np.allclose(graph.clv_proxy, graph.n_hat * graph.v_hat)


def test_each_mode_preserves_every_item_incoming_mass():
    graph = build_mass_preserving_clv_graph(_train(), 3, 3)
    base_mass = np.bincount(
        graph.edge_items, weights=graph.base_coefficients, minlength=3
    )
    for mode in MODES:
        adjusted_mass = np.bincount(
            graph.edge_items,
            weights=graph.item_user_coefficients[mode],
            minlength=3,
        )
        assert np.allclose(adjusted_mass, base_mass, atol=1e-7)
        assert graph.diagnostics["modes"][mode]["max_item_mass_abs_error"] < 1e-12


def test_constant_factor_is_exact_m1_operator():
    train = _train().copy()
    train["v"] = 10.0
    train["b_raw"] = [10, 11, 12, 20, 21, 30, 31, 32]
    # All users have different N in the original data, so use the V-only arm:
    # one-line baskets make V identical and average-rank percentiles give c == 1.
    graph = build_mass_preserving_clv_graph(train, 3, 3)
    assert np.array_equal(
        graph.item_user_coefficients["v_only"], graph.base_coefficients
    )

    adj = build_directional_torch_adj(graph, "v_only", 3, 3, torch.device("cpu"))
    dense = adj.to_dense().numpy()
    assert np.allclose(dense, dense.T)


def test_only_item_receiving_direction_is_redistributed():
    graph = build_mass_preserving_clv_graph(_train(), 3, 3)
    adj = build_directional_torch_adj(graph, "clv", 3, 3, torch.device("cpu"))
    dense = adj.to_dense().numpy()
    for edge, (user, item) in enumerate(zip(graph.edge_users, graph.edge_items)):
        assert dense[user, 3 + item] == graph.base_coefficients[edge]
        assert dense[3 + item, user] == graph.item_user_coefficients["clv"][edge]


def test_shuffle_preserves_clv_factor_distribution_but_changes_assignment():
    graph = build_mass_preserving_clv_graph(_train(), 3, 3)
    assert np.allclose(
        np.sort(graph.user_factors["clv"]),
        np.sort(graph.user_factors["clv_shuffle"]),
    )
    assert not np.array_equal(
        graph.user_factors["clv"], graph.user_factors["clv_shuffle"]
    )
