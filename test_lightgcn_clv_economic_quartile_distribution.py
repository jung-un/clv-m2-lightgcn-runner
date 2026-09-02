import numpy as np
import pandas as pd
import pytest

import lightgcn_clv_economic_quartile_distribution as runner


def _train_frame():
    rows = []
    prices = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    for item, price in enumerate(prices):
        rows.append(
            {
                "u_idx": item % 3,
                "i_idx": item,
                "up": price,
                "v": price,
            }
        )
    rows.extend(
        [
            {"u_idx": 0, "i_idx": 0, "up": 1.0, "v": 2.0},
            {"u_idx": 1, "i_idx": 7, "up": 8.0, "v": 16.0},
        ]
    )
    return pd.DataFrame(rows)


def test_preflight_freezes_m2_boundary_and_protected_split():
    cfg = runner.configure_economic_quartile_run(
        out_dir="/tmp/m2-economic-quartile",
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
    assert summary["m2"]["architecture"] == "ID(64)|CLV-conditioned economic quartiles(4)"
    assert summary["m2"]["total_dim"] == 68
    assert summary["m2"]["learned_global_scale"] is False
    assert summary["m2"]["category_input"] is False
    assert summary["m2"]["repeatshare_input"] is False
    assert summary["fixed"]["graph"] == "binary"
    assert summary["fixed"]["negative_sampling"] == "uniform"
    assert summary["fixed"]["sample_weighting"] is False
    assert summary["fixed"]["new_loss_term"] is False


@pytest.mark.parametrize(
    "override",
    [
        {"seed": 43},
        {"rho": 0.10},
        {"economic_bins": 3},
        {"shrinkage_strength": 0.0},
        {"n_layers": 1},
        {"include_degree_matched_shuffle": False},
    ],
)
def test_fast_screen_rejects_unplanned_overrides(override):
    with pytest.raises(ValueError, match="빠른 M2 screen"):
        runner.configure_economic_quartile_run(
            out_dir="/tmp/m2-economic-quartile",
            baseline_result_dir="/tmp/m1-result",
            **override,
        )


def test_economic_bins_have_equal_item_counts_and_profiles_sum_to_zero():
    inputs = runner.build_economic_quartile_inputs(
        _train_frame(),
        n_users=3,
        n_items=8,
        n_bins=4,
        shrinkage_strength=10.0,
    )

    np.testing.assert_array_equal(
        np.bincount(inputs["item_economic_bin"], minlength=4),
        np.array([2, 2, 2, 2]),
    )
    np.testing.assert_allclose(
        inputs["user_economic_profile"].sum(axis=1), 0.0, atol=1e-7
    )
    np.testing.assert_allclose(
        inputs["item_economic_basis"].sum(axis=1), 0.0, atol=1e-7
    )
    assert inputs["economic_input_diagnostics"]["item_count_bin_imbalance"] == 0


def test_shrinkage_reduces_sparse_user_profile_without_unit_renormalization():
    frame = _train_frame()
    shrunk = runner.build_economic_quartile_inputs(
        frame,
        n_users=3,
        n_items=8,
        n_bins=4,
        shrinkage_strength=10.0,
    )
    unshrunk = runner.build_economic_quartile_inputs(
        frame,
        n_users=3,
        n_items=8,
        n_bins=4,
        shrinkage_strength=0.0,
    )

    shrunk_norm = np.linalg.norm(shrunk["user_economic_profile"], axis=1)
    unshrunk_norm = np.linalg.norm(unshrunk["user_economic_profile"], axis=1)
    assert np.all(shrunk_norm < unshrunk_norm)
    assert np.all(shrunk["user_profile_reliability"] < 1.0)
