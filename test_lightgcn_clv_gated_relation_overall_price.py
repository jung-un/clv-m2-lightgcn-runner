import numpy as np
import pandas as pd
import pytest

from lightgcn_clv_gated_relation_overall_price import (
    ID_ONLY_MODEL_ID,
    MATCHED_MODEL_ID,
    MODEL_ID,
    PRICE_ONLY_MODEL_ID,
    RELATION_ONLY_MODEL_ID,
    build_overall_price_fit_inputs,
    configure_gated_relation_price_run,
    preflight_summary,
)


def test_preflight_freezes_the_bounded_m2_and_protected_development_split():
    cfg = configure_gated_relation_price_run(
        out_dir="/tmp/m2-gated-relation-price",
        baseline_result_dir="/tmp/m1-result",
    )
    summary = preflight_summary(cfg)

    assert summary["trained_models"] == [MATCHED_MODEL_ID, MODEL_ID]
    assert summary["historical_development_split"]["final_test_constructed"] is False
    assert summary["historical_development_split"]["holdout_constructed"] is False
    assert summary["m2"]["architecture"] == (
        "ID(64)|gated CLV relation(2)|overall price fit(1)"
    )
    assert summary["m2"]["total_dim"] == 67
    assert summary["m2"]["within_category_price_input"] is False
    assert summary["m2"]["repeatshare_input"] is False
    assert summary["m2"]["item_popularity_input"] is False
    assert summary["fixed"]["graph"] == "binary"
    assert summary["fixed"]["negative_sampling"] == "uniform"
    assert summary["fixed"]["sample_weighting"] is False
    assert summary["fixed"]["new_loss_term"] is False
    assert ID_ONLY_MODEL_ID == "m2_jointly_trained_id_only"
    assert RELATION_ONLY_MODEL_ID == "m2_id_plus_gated_relation_only"
    assert PRICE_ONLY_MODEL_ID == "m2_id_plus_overall_price_only"


@pytest.mark.parametrize(
    "override",
    [
        {"seed": 43},
        {"rho": 0.1},
        {"price_budget": 0.5},
        {"auxiliary_dim": 4},
        {"id_dim": 60},
        {"time_cutoff": 697},
    ],
)
def test_fast_screen_rejects_unplanned_overrides(override):
    with pytest.raises(ValueError, match="빠른 M2 screen"):
        configure_gated_relation_price_run(
            out_dir="/tmp/m2-gated-relation-price",
            baseline_result_dir="/tmp/m1-result",
            **override,
        )


def test_overall_price_fit_uses_item_mean_price_and_user_amount_weights():
    train = pd.DataFrame(
        {
            "u_idx": [0, 0, 1, 1],
            "i_idx": [0, 1, 1, 2],
            "up": [10.0, 30.0, 30.0, 50.0],
            "v": [90.0, 10.0, 20.0, 80.0],
        }
    )
    inputs = build_overall_price_fit_inputs(train, n_users=3, n_items=4)

    np.testing.assert_array_equal(
        inputs["item_price_valid"], [True, True, True, False]
    )
    np.testing.assert_array_equal(
        inputs["user_price_valid"], [True, True, False]
    )
    assert inputs["item_overall_price"][0] < inputs["item_overall_price"][1]
    assert inputs["item_overall_price"][1] < inputs["item_overall_price"][2]
    assert inputs["user_overall_price"][0] < inputs["user_overall_price"][1]
    assert inputs["item_overall_price"][3] == 0.5
    assert inputs["user_overall_price"][2] == 0.5
