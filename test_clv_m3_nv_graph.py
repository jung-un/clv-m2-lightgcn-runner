import numpy as np
import pandas as pd
import pytest

from clv_m3_nv_graph import build_clv_nv_graph


def _train_rows():
    return pd.DataFrame(
        {
            "u_idx": [0, 0, 0, 0, 0, 1, 1],
            "i_idx": [0, 1, 0, 2, 0, 1, 1],
            "b_raw": [10, 10, 11, 11, 12, 20, 21],
            "t": [0, 0, 1, 1, 2, 0, 10],
            "v": [2.0, 8.0, 2.0, 98.0, 2.0, 5.0, 5.0],
        }
    )


def test_clv_nv_graph_preserves_edges_and_composes_repeat_and_basket_context():
    graph = build_clv_nv_graph(_train_rows(), n_users=2, n_items=3)
    keys = graph.edge_users * 3 + graph.edge_items

    np.testing.assert_array_equal(keys, np.array([0, 1, 2, 4]))
    assert graph.n_relation[0] > graph.n_relation[1]
    assert graph.n_relation[1] == 0.0
    assert graph.v_relation[2] > graph.v_relation[1]
    assert graph.q_n[0] > graph.q_n[1]
    assert graph.q_v[0] > graph.q_v[1]
    assert np.isfinite(graph.weights).all()
    assert np.all(graph.weights >= 0.25)
    assert np.all(graph.weights <= 4.0)
    assert not np.allclose(graph.weights, 1.0)
    assert graph.diagnostics["n_edges"] == 4
    assert graph.diagnostics["weight_mean"] == pytest.approx(
        float(graph.weights.mean())
    )


def test_clv_nv_graph_rejects_missing_basket_identity():
    with pytest.raises(ValueError, match="b_raw"):
        build_clv_nv_graph(
            _train_rows().drop(columns="b_raw"), n_users=2, n_items=3
        )
