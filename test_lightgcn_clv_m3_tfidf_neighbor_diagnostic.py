import numpy as np
import pandas as pd

from lightgcn_clv_m3_tfidf_neighbor_diagnostic import (
    build_anchor_truth,
    historical_anchor_ends,
    mechanism_reading,
)


def test_last_five_anchor_windows_are_nonoverlapping_and_train_only():
    assert historical_anchor_ends(683, horizon_days=7, n_anchors=5) == [
        648,
        655,
        662,
        669,
        676,
    ]


def test_anchor_truth_keeps_only_seen_items_and_new_user_item_pairs():
    frame = pd.DataFrame(
        {
            "u_idx": [0, 0, 1, 0, 0, 1, 1],
            "i_idx": [0, 1, 1, 0, 2, 1, 3],
            "t": [1, 2, 2, 3, 3, 3, 3],
        }
    )
    past, truth = build_anchor_truth(
        frame, anchor_end=2, horizon_days=1, n_users=2, n_items=4
    )

    assert set(map(tuple, past[["u_idx", "i_idx"]].to_numpy())) == {
        (0, 0),
        (0, 1),
        (1, 1),
    }
    # (0,0) and (1,1) are repeats; items 2 and 3 were not available by anchor.
    assert truth == {}


def test_mechanism_reading_requires_both_comparators_at_all_declared_levels():
    rows = []
    for anchor in range(5):
        for user, group, degree in [(0, "low", 0), (1, "high", 1)]:
            rows.append(
                {
                    "anchor_end": anchor,
                    "u_idx": user,
                    "clv_group": group,
                    "degree_stratum": degree,
                    "eligible_tfidf": True,
                    "tfidf_topk_neighbor": 0.30 if group == "low" else 0.50,
                    "ordinary_copurchase_propagation": 0.25
                    if group == "low"
                    else 0.35,
                    "degree_matched_random_neighbor": 0.20
                    if group == "low"
                    else 0.30,
                    "q_clv": 0.2 if group == "low" else 0.8,
                }
            )
    reading = mechanism_reading(pd.DataFrame(rows))

    assert reading["precheck_passed"] is True
    for comparator in (
        "ordinary_copurchase_propagation",
        "degree_matched_random_neighbor",
    ):
        assert reading["comparisons"][comparator]["overall_mean_delta"] > 0
        assert reading["comparisons"][comparator]["high_clv_mean_delta"] > 0
        assert reading["comparisons"][comparator]["positive_high_clv_anchor_count"] == 5


def test_mechanism_reading_fails_if_high_clv_does_not_beat_one_control():
    rows = pd.DataFrame(
        {
            "anchor_end": np.repeat(np.arange(5), 2),
            "u_idx": np.tile([0, 1], 5),
            "clv_group": np.tile(["low", "high"], 5),
            "degree_stratum": np.tile([0, 1], 5),
            "eligible_tfidf": True,
            "tfidf_topk_neighbor": np.tile([0.30, 0.30], 5),
            "ordinary_copurchase_propagation": np.tile([0.20, 0.20], 5),
            "degree_matched_random_neighbor": np.tile([0.20, 0.40], 5),
            "q_clv": np.tile([0.2, 0.8], 5),
        }
    )

    assert mechanism_reading(rows)["precheck_passed"] is False

