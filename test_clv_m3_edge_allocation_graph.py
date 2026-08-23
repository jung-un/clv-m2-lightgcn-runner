import numpy as np
import pandas as pd
import torch

from clv_m3_edge_allocation_graph import (
    build_directional_torch_adj,
    build_edge_allocated_clv_graph,
)


def _train():
    return pd.DataFrame(
        {
            "u_idx": [0, 0, 0, 1, 1, 2],
            "i_idx": [0, 0, 1, 0, 2, 2],
            "b_raw": [10, 11, 11, 20, 21, 30],
            "t": [1, 2, 2, 1, 2, 1],
            "v": [4.0, 5.0, 1.0, 10.0, 2.0, 3.0],
        }
    )


def test_clv_is_allocated_once_across_each_users_edges():
    graph = build_edge_allocated_clv_graph(_train(), 3, 3)
    allocated = np.bincount(
        graph.edge_users, weights=graph.edge_clv_allocation, minlength=3
    )
    np.testing.assert_allclose(allocated, graph.clv_proxy, rtol=0, atol=1e-10)
    shares = np.bincount(
        graph.edge_users, weights=graph.relationship_share, minlength=3
    )
    np.testing.assert_allclose(shares, np.ones(3), rtol=0, atol=1e-10)


def test_edge_set_and_item_message_mass_match_m1():
    graph = build_edge_allocated_clv_graph(_train(), 3, 3)
    assert list(zip(graph.edge_users, graph.edge_items, strict=True)) == [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 2),
        (2, 2),
    ]
    base_mass = np.bincount(
        graph.edge_items, weights=graph.base_coefficients, minlength=3
    )
    changed_mass = np.bincount(
        graph.edge_items, weights=graph.item_user_coefficients, minlength=3
    )
    np.testing.assert_allclose(changed_mass, base_mass, rtol=0, atol=1e-10)


def test_degree_one_items_are_exactly_m1_and_all_coefficients_are_positive():
    graph = build_edge_allocated_clv_graph(_train(), 3, 3)
    degree = np.bincount(graph.edge_items, minlength=3)
    mask = degree[graph.edge_items] == 1
    np.testing.assert_array_equal(
        graph.item_user_coefficients[mask], graph.base_coefficients[mask]
    )
    assert np.isfinite(graph.item_user_coefficients).all()
    assert (graph.item_user_coefficients > 0).all()


def test_directional_operator_keeps_user_rows_at_m1():
    graph = build_edge_allocated_clv_graph(_train(), 3, 3)
    adj = build_directional_torch_adj(graph, 3, 3, torch.device("cpu")).to_dense()
    for (user, item), base in zip(
        zip(graph.edge_users, graph.edge_items, strict=True),
        graph.base_coefficients,
        strict=True,
    ):
        assert adj[user, 3 + item].item() == np.float32(base)
