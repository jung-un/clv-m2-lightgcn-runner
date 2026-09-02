import pytest

from lightgcn_clv_weighted_price_distance import (
    ID_ONLY_MODEL_ID,
    MATCHED_MODEL_ID,
    MODEL_ID,
    configure_price_distance_run,
    preflight_summary,
)


def test_preflight_freezes_minimal_price_distance_m2_and_protected_split():
    cfg = configure_price_distance_run(
        out_dir="/tmp/m2-price-distance",
        baseline_result_dir="/tmp/m1-result",
    )
    summary = preflight_summary(cfg)

    assert summary["trained_models"] == [MATCHED_MODEL_ID, MODEL_ID]
    assert summary["historical_development_split"]["final_test_constructed"] is False
    assert summary["historical_development_split"]["holdout_constructed"] is False
    assert summary["m2"]["architecture"] == (
        "ID(64)|CLV-weighted overall price distance(2)"
    )
    assert summary["m2"]["total_dim"] == 66
    assert summary["m2"]["relation_block"] is False
    assert summary["m2"]["q_n_minus_q_v_input"] is False
    assert summary["m2"]["within_category_price_input"] is False
    assert summary["m2"]["item_id_projection_in_auxiliary"] is False
    assert summary["m2"]["pairwise_intervention_max"] == 0.05
    assert summary["m2"]["repeatshare_input"] is False
    assert summary["m2"]["item_popularity_input"] is False
    assert summary["fixed"]["graph"] == "binary"
    assert summary["fixed"]["negative_sampling"] == "uniform"
    assert summary["fixed"]["sample_weighting"] is False
    assert summary["fixed"]["new_loss_term"] is False
    assert ID_ONLY_MODEL_ID == "m2_jointly_trained_id_only"


@pytest.mark.parametrize(
    "override",
    [
        {"seed": 43},
        {"rho": 0.1},
        {"price_scale_initial": 0.5},
        {"auxiliary_dim": 3},
        {"id_dim": 60},
        {"time_cutoff": 697},
    ],
)
def test_fast_screen_reject_unplanned_overrides(override):
    with pytest.raises(ValueError, match="빠른 M2 screen"):
        configure_price_distance_run(
            out_dir="/tmp/m2-price-distance",
            baseline_result_dir="/tmp/m1-result",
            **override,
        )
