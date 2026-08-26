import numpy as np
import pandas as pd

from clv_m3_next_new_transition import (
    build_historical_clv,
    build_transition_graphs,
    build_user_transition_events,
)


def _transactions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            # user 0: {0,1} -> {1,2}; only item 2 is new
            (0, 0, 1, "u0-b1", 2.0),
            (0, 1, 1, "u0-b1", 3.0),
            (0, 1, 2, "u0-b2", 4.0),
            (0, 2, 2, "u0-b2", 6.0),
            # next basket has no first-purchase item and must contribute nothing
            (0, 2, 3, "u0-b3", 7.0),
            # user 1: {1} -> {2,3}
            (1, 1, 1, "u1-b1", 5.0),
            (1, 2, 2, "u1-b2", 5.0),
            (1, 3, 2, "u1-b2", 10.0),
        ],
        columns=["u_idx", "i_idx", "t", "basket_id", "v"],
    )


def test_next_new_targets_exclude_repeat_items_and_empty_pairs():
    events = build_user_transition_events(_transactions(), n_users=2)

    observed = set(
        zip(
            events.user_idx.tolist(),
            events.source_item_idx.tolist(),
            events.target_item_idx.tolist(),
        )
    )
    assert observed == {(0, 0, 2), (0, 1, 2), (1, 1, 2), (1, 1, 3)}
    np.testing.assert_array_equal(events.eligible_pair_count_by_user, [1, 1])


def test_transition_contributions_are_basket_and_user_normalized():
    events = build_user_transition_events(_transactions(), n_users=2)

    for user in (0, 1):
        mask = events.user_idx == user
        assert np.isclose(events.contribution[mask].sum(), 1.0)
        np.testing.assert_allclose(events.contribution[mask], [0.5, 0.5])


def test_historical_clv_uses_distinct_baskets_and_mean_basket_value():
    clv, shuffled = build_historical_clv(_transactions(), n_users=2)

    np.testing.assert_allclose(clv.n_hat, [3.0, 2.0])
    np.testing.assert_allclose(clv.v_hat, [(5.0 + 10.0 + 7.0) / 3.0, 10.0])
    np.testing.assert_allclose(clv.clv_proxy, clv.n_hat * clv.v_hat)
    assert np.isclose(clv.coefficient.mean(), 1.0)
    np.testing.assert_allclose(np.sort(shuffled), np.sort(clv.coefficient))


def test_clv_shuffle_preserves_coefficients_within_activity_decile():
    rows = []
    for user in range(20):
        for basket in range(user + 1):
            rows.append((user, user % 4, basket, f"u{user}-b{basket}", float(user + 1)))
    tx = pd.DataFrame(rows, columns=["u_idx", "i_idx", "t", "basket_id", "v"])

    clv, shuffled = build_historical_clv(tx, n_users=20)

    for decile in np.unique(clv.activity_decile):
        mask = clv.activity_decile == decile
        np.testing.assert_allclose(
            np.sort(shuffled[mask]), np.sort(clv.coefficient[mask])
        )


def test_transition_graphs_are_row_normalized_without_pruning():
    events = build_user_transition_events(_transactions(), n_users=2)
    graphs = build_transition_graphs(
        events,
        clv_coefficient=np.array([0.5, 1.5]),
        shuffled_coefficient=np.array([1.5, 0.5]),
        n_items=5,
    )

    for matrix in (
        graphs.global_relation,
        graphs.clv_relation,
        graphs.shuffled_clv_relation,
    ):
        row_sums = np.asarray(matrix.sum(axis=1)).ravel()
        np.testing.assert_allclose(row_sums[[0, 1]], [1.0, 1.0])
        np.testing.assert_allclose(row_sums[[2, 3, 4]], 0.0)
        assert matrix.shape == (5, 5)

    # Source item 1 has contributions from both users, so correct CLV assignment
    # must change how much mass goes to targets 2 and 3.
    assert not np.allclose(
        graphs.global_relation.getrow(1).toarray(),
        graphs.clv_relation.getrow(1).toarray(),
    )
    assert graphs.edge_support[1, 2] == 2
    assert graphs.edge_support[1, 3] == 1
