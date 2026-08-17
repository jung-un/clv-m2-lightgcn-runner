import numpy as np
import pandas as pd

from clv_m3_nv_diagnostics import analyze_clv_nv_graph
from clv_m3_nv_graph import CLVNVGraphWeights


def _fixture():
    train = pd.DataFrame(
        {
            "u_idx": [0, 0, 1, 1],
            "i_idx": [0, 1, 0, 2],
            "b_raw": [10, 11, 20, 21],
            "v": [1.0, 10.0, 1.0, 10.0],
            "up": [1.0, 10.0, 1.0, 10.0],
            "i_raw": [100, 101, 100, 102],
            "cat_raw": ["staple", "premium", "staple", "premium"],
        }
    )
    graph = CLVNVGraphWeights(
        edge_users=np.array([0, 0, 1, 1]),
        edge_items=np.array([0, 1, 0, 2]),
        weights=np.array([2.0, 0.5, 2.0, 0.5], np.float32),
        n_relation=np.array([1.0, 0.0, 1.0, 0.0], np.float32),
        v_relation=np.array([0.0, 1.0, 0.0, 1.0], np.float32),
        n_component=np.array([2.0, 0.0, 2.0, 0.0], np.float32),
        v_component=np.array([0.0, 2.0, 0.0, 2.0], np.float32),
        q_n=np.array([0.5, 1.0], np.float32),
        q_v=np.array([0.5, 1.0], np.float32),
        diagnostics={"repeat_edge_share": 0.5},
    )
    return train, graph


def test_diagnostic_detects_popularity_and_low_price_amplification():
    train, graph = _fixture()

    report = analyze_clv_nv_graph(train, graph, n_users=2, n_items=3)
    correlations = report["correlations"].set_index("comparison")["spearman"]

    assert correlations["raw_weight__item_user_degree"] > 0.9
    assert correlations["raw_weight__item_price_percentile"] < -0.9
    assert report["summary"]["n_component_zero_share"] == 0.5
    assert report["top_items"].iloc[0]["i_idx"] == 0


def test_diagnostic_reports_degree_normalized_propagation_and_deciles():
    train, graph = _fixture()

    report = analyze_clv_nv_graph(train, graph, n_users=2, n_items=3)

    assert np.isfinite(report["summary"]["propagation_ratio_std"])
    assert report["summary"]["n_edges"] == 4
    assert report["weight_deciles"]["edge_count"].sum() == 4
    assert {
        "raw_weight_mean",
        "propagation_ratio_mean",
        "item_user_degree_mean",
        "item_price_percentile_mean",
    }.issubset(report["weight_deciles"].columns)
