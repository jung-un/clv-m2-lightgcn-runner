import numpy as np
import pandas as pd
import pytest

from clv_m3_transfer_graph import build_m3_transfer_graphs


def _train_rows():
    return pd.DataFrame(
        {
            "u_idx": [0, 0, 0, 0, 1, 1],
            "i_idx": [0, 1, 0, 2, 1, 2],
            "b_raw": [10, 10, 11, 12, 20, 21],
            "cat_idx": [0, 0, 0, 1, 0, 1],
            "t": [0, 0, 1, 2, 0, 10],
            "v": [9.0, 1.0, 10.0, 5.0, 3.0, 3.0],
        }
    )


def test_transfer_graph_preserves_edges_and_matches_effective_strength():
    graph = build_m3_transfer_graphs(_train_rows(), n_users=2, n_items=3)
    keys = graph.edge_users * 3 + graph.edge_items

    np.testing.assert_array_equal(keys, np.array([0, 1, 2, 4, 5]))
    assert graph.n_weights.mean() == pytest.approx(1.0, abs=1e-6)
    assert graph.v_weights.mean() == pytest.approx(1.0, abs=1e-6)
    assert graph.clv_composition_weights.mean() == pytest.approx(1.0, abs=1e-6)
    assert np.all(graph.n_weights > 0)
    assert np.all(graph.v_weights > 0)
    assert np.all(graph.clv_composition_weights > 0)
    assert graph.n_weights.max() / graph.n_weights.min() < 1.7
    assert graph.v_weights.max() / graph.v_weights.min() < 1.7
    assert graph.diagnostics["n_propagation_strength"] == pytest.approx(
        graph.diagnostics["v_propagation_strength"], abs=1e-8
    )
    assert graph.diagnostics[
        "clv_composition_propagation_strength"
    ] == pytest.approx(graph.diagnostics["target_propagation_strength"], abs=1e-8)


def test_clv_composition_uses_full_magnitude_and_user_specific_axis_mix():
    graph = build_m3_transfer_graphs(_train_rows(), n_users=2, n_items=3)

    assert np.allclose(graph.pi_n + graph.pi_v, 1.0, atol=1e-6)
    assert np.unique(graph.q_clv).size > 1
    assert not np.array_equal(graph.clv_composition_signal, graph.n_signal)
    assert not np.array_equal(graph.clv_composition_signal, graph.v_signal)
    for user in np.unique(graph.edge_users):
        current = graph.clv_composition_weights[graph.edge_users == user]
        assert current.mean() == pytest.approx(1.0, abs=1e-6)


def test_n_transfer_uses_category_repeatability_not_exact_edge_repeat():
    graph = build_m3_transfer_graphs(_train_rows(), n_users=2, n_items=3)

    # u0-i0 is repeated, u0-i1 is not; both are category 0 and must receive
    # the same N-transfer signal.  Category 0 repeats more than category 1.
    assert graph.n_signal[0] == pytest.approx(graph.n_signal[1])
    assert graph.n_signal[0] > graph.n_signal[2]
    assert graph.diagnostics["n_signal_zero_share"] == 0.0


def test_v_contribution_uses_line_share_not_whole_basket_context():
    graph = build_m3_transfer_graphs(_train_rows(), n_users=2, n_items=3)

    # In basket 10, i0 contributes 90% and i1 10%; i0 also owns all of basket
    # 11.  The V signal must therefore be larger for i0 than i1 for user 0.
    assert graph.v_signal[0] > graph.v_signal[1]


def test_transfer_graph_rejects_missing_category():
    with pytest.raises(ValueError, match="cat_idx"):
        build_m3_transfer_graphs(
            _train_rows().drop(columns="cat_idx"), n_users=2, n_items=3
        )
