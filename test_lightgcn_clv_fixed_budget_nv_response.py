import numpy as np
import pandas as pd
import pytest

import lightgcn_clv_fixed_budget_nv_response as runner


def test_preflight_freezes_minimal_m2_and_protected_split():
    cfg = runner.configure_fixed_budget_nv_response_run(
        out_dir="/tmp/m2-fixed-budget-nv",
        baseline_result_dir="/tmp/m1-result",
    )
    summary = runner.preflight_summary(cfg)

    assert summary["trained_models"] == [
        runner.MATCHED_MODEL_ID,
        runner.MODEL_ID,
        runner.SHUFFLED_MODEL_ID,
    ]
    assert summary["historical_development_split"]["final_test_constructed"] is False
    assert summary["historical_development_split"]["holdout_constructed"] is False
    assert summary["m2"]["architecture"] == "ID(64)|fixed-budget N response(1)|fixed-budget V-price response(1)"
    assert summary["m2"]["total_dim"] == 66
    assert summary["m2"]["clv_budget_identity"] == "b_N(u)+b_V(u)=q_C(u)"
    assert summary["m2"]["learned_user_projection"] is False
    assert summary["m2"]["free_item_response_embedding"] is False
    assert summary["m2"]["repeatshare_input"] is False
    assert summary["m2"]["item_popularity_input"] is False
    assert summary["fixed"]["graph"] == "binary"
    assert summary["fixed"]["negative_sampling"] == "uniform"
    assert summary["fixed"]["sample_weighting"] is False
    assert summary["fixed"]["new_loss_term"] is False


@pytest.mark.parametrize(
    "override",
    [
        {"seed": 43},
        {"rho": 0.10},
        {"id_dim": 60},
        {"n_layers": 1},
        {"time_cutoff": 697},
        {"include_degree_matched_shuffle": False},
    ],
)
def test_fast_screen_rejects_unplanned_overrides(override):
    with pytest.raises(ValueError, match="빠른 M2 screen"):
        runner.configure_fixed_budget_nv_response_run(
            out_dir="/tmp/m2-fixed-budget-nv",
            baseline_result_dir="/tmp/m1-result",
            **override,
        )


def test_degree_matched_shuffle_moves_joint_clv_tuples_only_within_strata():
    n_users = 12
    rows = []
    for user in range(n_users):
        for item in range(user + 1):
            rows.append({"u_idx": user, "i_idx": item})
    q_n = np.linspace(0.05, 0.95, n_users, dtype=np.float32)
    q_v = np.linspace(0.95, 0.05, n_users, dtype=np.float32)
    q_c = np.arange(n_users, dtype=np.float32) / n_users
    prepared = {
        "data": {
            "train": pd.DataFrame(rows),
            "n_users": n_users,
        },
        "q_n": q_n,
        "q_v": q_v,
        "q_c": q_c,
        "clv_valid": np.ones(n_users, dtype=bool),
    }
    cfg = runner.FixedBudgetNVResponseConfig(
        shuffle_degree_bins=3,
        shuffle_seed=7,
    )

    shuffled = runner._degree_matched_clv_shuffle(prepared, cfg)
    source = shuffled["source_user"]

    assert shuffled["changed_valid_user_share"] > 0.0
    np.testing.assert_array_equal(shuffled["q_n"], q_n[source])
    np.testing.assert_array_equal(shuffled["q_v"], q_v[source])
    np.testing.assert_array_equal(shuffled["q_c"], q_c[source])
    np.testing.assert_array_equal(shuffled["stratum"], shuffled["stratum"][source])
