import numpy as np
import pandas as pd
import pytest

from clv_m3_direct_value_graph import build_direct_clv_value_graph


def _train_rows():
    return pd.DataFrame(
        {
            "u_idx": [0, 0, 0, 1, 1, 2],
            "i_idx": [0, 0, 1, 0, 2, 1],
            "v": [2.0, 3.0, 1.0, 4.0, 8.0, 2.0],
            "up": [2.0, 3.0, 1.0, 4.0, 8.0, 2.0],
        }
    )


def test_direct_graph_preserves_edges_and_normalizes_every_arm():
    graph = build_direct_clv_value_graph(
        _train_rows(), 3, 3, np.array([0.5, 2.0, 1.0]), alpha=1.0
    )

    np.testing.assert_array_equal(
        graph.edge_users * 3 + graph.edge_items, np.array([0, 1, 3, 5, 7])
    )
    for weights in (
        graph.user_clv_weights,
        graph.spend_only_weights,
        graph.clv_spend_weights,
    ):
        assert np.all(np.isfinite(weights))
        assert np.all(weights > 0)
        assert weights.mean() == pytest.approx(1.0, abs=1e-6)


def test_user_and_relationship_weights_encode_the_intended_ordering():
    graph = build_direct_clv_value_graph(
        _train_rows(), 3, 3, np.array([0.5, 2.0, 1.0]), alpha=1.0
    )

    user0 = graph.edge_users == 0
    assert np.unique(graph.user_clv_weights[user0]).size == 1
    assert graph.user_clv_weights[graph.edge_users == 1].mean() > graph.user_clv_weights[user0].mean()
    user1 = np.flatnonzero(graph.edge_users == 1)
    assert graph.spend_only_weights[user1[1]] > graph.spend_only_weights[user1[0]]
    assert graph.clv_spend_weights[user1[1]] > graph.clv_spend_weights[user1[0]]


def test_spend_control_does_not_depend_on_clv_assignment():
    first = build_direct_clv_value_graph(
        _train_rows(), 3, 3, np.array([0.5, 2.0, 1.0]), alpha=1.0
    )
    shuffled = build_direct_clv_value_graph(
        _train_rows(), 3, 3, np.array([2.0, 0.5, 1.0]), alpha=1.0
    )

    np.testing.assert_array_equal(first.spend_only_weights, shuffled.spend_only_weights)
    assert not np.array_equal(first.user_clv_weights, shuffled.user_clv_weights)
    assert not np.array_equal(first.clv_spend_weights, shuffled.clv_spend_weights)


@pytest.mark.parametrize("alpha", [-0.1, np.nan])
def test_direct_graph_rejects_invalid_alpha(alpha):
    with pytest.raises(ValueError, match="alpha"):
        build_direct_clv_value_graph(
            _train_rows(), 3, 3, np.ones(3), alpha=alpha
        )
