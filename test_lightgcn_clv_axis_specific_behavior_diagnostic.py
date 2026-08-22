import numpy as np
import pandas as pd

import lightgcn_clv_axis_specific_behavior_diagnostic as diagnostic


def test_assign_nv_groups_uses_train_axis_percentiles():
    groups = diagnostic.assign_nv_groups(
        np.array([0.1, 0.9, 0.1, 0.9, 0.9]),
        np.array([0.1, 0.1, 0.9, 0.9, 0.9]),
        np.array([True, True, True, True, False]),
    )
    assert groups.tolist() == [
        "low_n_low_v",
        "high_n_low_v",
        "low_n_high_v",
        "high_n_high_v",
        "invalid_axis",
    ]


def test_role_items_separates_promoted_hits_and_displaced_hits():
    roles = diagnostic._role_items(
        0,
        truth=np.array([2, 4, 9]),
        m1=np.array([1, 2, 3, 4]),
        m2=np.array([2, 5, 4, 9]),
    )
    assert roles["m2_promoted_top10"].tolist() == [5, 9]
    assert roles["m2_promoted_hit"].tolist() == [9]
    assert roles["m2_promoted_miss"].tolist() == [5]
    assert roles["m1_displaced_top10"].tolist() == [1, 3]
    assert roles["m1_displaced_hit"].tolist() == []


def test_truth_rank_summary_keeps_seed_as_unit():
    frame = pd.DataFrame(
        {
            "seed": [42, 42, 43, 43],
            "group_id": ["high_n_low_v"] * 4,
            "m1_rank_capped_51": [11, 9, 20, 8],
            "m2_rank_capped_51": [8, 12, 10, 9],
            "rank_improvement": [3, -3, 10, -1],
            "entered_top10": [True, False, True, False],
            "left_top10": [False, True, False, True],
            "entered_top20": [False, False, True, False],
            "left_top20": [False, False, False, False],
        }
    )
    by_seed, mean = diagnostic._truth_rank_summary(frame)
    assert len(by_seed) == 2
    assert mean.loc[0, "n_seeds"] == 2
    assert np.isclose(mean.loc[0, "mean_rank_improvement"], 2.25)
    assert np.isclose(mean.loc[0, "entered_top10_share"], 0.5)
