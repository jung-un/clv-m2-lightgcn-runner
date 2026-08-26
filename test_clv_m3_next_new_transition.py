import numpy as np
import pandas as pd

from clv_m3_next_new_transition import (
    build_historical_clv,
    build_transition_graphs,
    build_user_transition_events,
    count_transition_candidates,
    decide_pilot,
    evaluate_transition_ranking,
    rank_transition_candidates,
    reachable_truth_share,
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


def test_candidate_ranking_averages_last_basket_rows_and_excludes_seen_items():
    from scipy import sparse

    relation = sparse.csr_matrix(
        np.array(
            [
                [0.0, 0.0, 0.4, 0.6, 0.0],
                [0.0, 0.0, 0.8, 0.0, 0.2],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
    )
    ranked = rank_transition_candidates(
        relation,
        last_basket_items={0: np.array([0, 1]), 1: np.array([2])},
        seen_items={0: np.array([2]), 1: np.array([], dtype=int)},
        eval_users=np.array([0, 1]),
        top_k=10,
    )

    # Mean scores are item 2=.6 (seen), item 3=.3, item 4=.1.
    np.testing.assert_array_equal(ranked[0], [3, 4])
    # No positive candidates means an empty list: no popularity backfill.
    assert ranked[1].size == 0
    counts = count_transition_candidates(
        relation,
        last_basket_items={0: np.array([0, 1]), 1: np.array([2])},
        seen_items={0: np.array([2]), 1: np.array([], dtype=int)},
        eval_users=np.array([0, 1]),
    )
    assert counts == {0: 2, 1: 0}


def test_candidate_ranking_breaks_equal_scores_by_item_index():
    from scipy import sparse

    relation = sparse.csr_matrix(np.array([[0.0, 0.0, 0.5, 0.5]] * 4))
    ranked = rank_transition_candidates(
        relation,
        last_basket_items={0: np.array([0])},
        seen_items={0: np.array([], dtype=int)},
        eval_users=np.array([0]),
        top_k=2,
    )
    np.testing.assert_array_equal(ranked[0], [2, 3])


def test_ranking_metrics_and_reachable_truth_are_computed_per_user():
    from scipy import sparse

    rankings = {0: np.array([2, 3]), 1: np.array([3])}
    truth = {0: np.array([2, 4]), 1: np.array([4])}
    metrics, per_user = evaluate_transition_ranking(
        rankings,
        truth=truth,
        n_items=5,
        ks=(1, 2),
    )
    assert np.isclose(metrics["recall@1"], 0.25)
    assert np.isclose(metrics["recall@2"], 0.25)
    assert metrics["n_distinct@1"] == 2
    assert np.isclose(metrics["top10_share@1"], 1.0)
    assert len(per_user) == 2

    relation = sparse.csr_matrix(
        np.array(
            [
                [0, 0, 1, 1, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=float,
        )
    )
    share = reachable_truth_share(
        relation,
        last_basket_items={0: np.array([0]), 1: np.array([2])},
        seen_items={0: np.array([], dtype=int), 1: np.array([], dtype=int)},
        truth=truth,
    )
    assert np.isclose(share, 2 / 3)


def _pilot_row(model_id: str, scale: float = 1.0) -> dict:
    row = {
        "model_id": model_id,
        "reachable_truth_share": 0.5,
        "n_distinct@10": 100,
        "top10_share@10": 0.2,
    }
    for metric in ("recall", "ndcg"):
        for k in (10, 20, 50):
            row[f"{metric}@{k}"] = scale
    return row


def test_pilot_decision_requires_all_predeclared_guards():
    table = pd.DataFrame(
        [
            _pilot_row("transition_global", 1.0),
            _pilot_row("transition_clv", 1.01),
            _pilot_row("transition_clv_shuffle", 1.005),
        ]
    )
    decision = decide_pilot(table)
    assert decision["passes_pilot"] is True
    assert all(decision["checks"].values())

    table.loc[table.model_id == "transition_clv", "recall@50"] = 0.98
    failed = decide_pilot(table)
    assert failed["passes_pilot"] is False
    assert failed["checks"]["accuracy_guard"] is False
