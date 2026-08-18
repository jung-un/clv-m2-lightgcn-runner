import numpy as np
import pandas as pd
import pytest

from clv_m3_axis_adaptive_graph import build_m3_axis_adaptive_graph


def _train_rows():
    return pd.DataFrame(
        {
            "u_idx": [0, 0, 0, 0, 0, 1, 1, 1, 2, 2],
            "i_idx": [0, 1, 2, 0, 3, 1, 1, 2, 0, 3],
            "b_raw": [10, 10, 11, 12, 12, 20, 21, 21, 30, 31],
            "cat_idx": [0, 0, 1, 0, 1, 0, 0, 1, 0, 1],
            "t": [0, 0, 1, 10, 10, 0, 1, 1, 0, 20],
            "v": [9.0, 1.0, 5.0, 8.0, 2.0, 3.0, 3.0, 2.0, 4.0, 4.0],
        }
    )


def test_axis_adaptive_graph_preserves_m1_edges_and_user_mass():
    graph = build_m3_axis_adaptive_graph(_train_rows(), n_users=3, n_items=4)
    expected = np.array([0, 1, 2, 3, 5, 6, 8, 11])

    np.testing.assert_array_equal(graph.edge_users * 4 + graph.edge_items, expected)
    assert np.all(graph.weights > 0)
    assert np.all(graph.v_only_weights > 0)
    for user in np.unique(graph.edge_users):
        current = graph.weights[graph.edge_users == user]
        assert current.mean() == pytest.approx(1.0, abs=1e-6)
        v_only = graph.v_only_weights[graph.edge_users == user]
        assert v_only.mean() == pytest.approx(1.0, abs=1e-6)
    assert graph.diagnostics["propagation_strength"] == pytest.approx(
        graph.diagnostics["target_propagation_strength"], abs=1e-8
    )
    assert graph.diagnostics["v_only_propagation_strength"] == pytest.approx(
        graph.diagnostics["target_propagation_strength"], abs=1e-8
    )


def test_n_relation_is_temporal_residualized_and_not_old_repeat_signal():
    graph = build_m3_axis_adaptive_graph(_train_rows(), n_users=3, n_items=4)

    assert graph.diagnostics["next_horizon_days"] == 7.0
    assert graph.diagnostics["n_item_relation_unique"] > 1
    assert graph.diagnostics["pair_basket_observations"]["median"] >= 1.0
    assert np.std(graph.n_component) > 0
    assert np.std(graph.v_component) > 0
    assert not np.array_equal(graph.n_component, graph.v_component)


def test_axis_adaptive_graph_rejects_missing_basket_time():
    with pytest.raises(ValueError, match="t"):
        build_m3_axis_adaptive_graph(
            _train_rows().drop(columns="t"), n_users=3, n_items=4
        )
