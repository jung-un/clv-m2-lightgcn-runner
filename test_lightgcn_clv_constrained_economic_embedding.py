import pytest

from lightgcn_clv_constrained_economic_embedding import (
    ID_ONLY_MODEL_ID,
    MATCHED_MODEL_ID,
    MODEL_ID,
    PRICE_ONLY_MODEL_ID,
    RELATION_ONLY_MODEL_ID,
    configure_constrained_economic_run,
    preflight_summary,
)


def test_preflight_freezes_the_two_removed_shortcuts_and_protected_split():
    cfg = configure_constrained_economic_run(
        out_dir="/tmp/m2-constrained-economic",
        baseline_result_dir="/tmp/m1-result",
    )
    summary = preflight_summary(cfg)

    assert summary["trained_models"] == [MATCHED_MODEL_ID, MODEL_ID]
    assert summary["historical_development_split"]["final_test_constructed"] is False
    assert summary["historical_development_split"]["holdout_constructed"] is False
    assert summary["m2"]["architecture"] == "ID(64)|one CLV-conditioned hybrid item block(4)"
    assert summary["m2"]["total_dim"] == 68
    assert summary["m2"]["user_tanh"] is False
    assert summary["m2"]["free_item_response_embedding"] is False
    assert summary["m2"]["item_inputs"] == [
        "existing item ID embedding projected to 2 dimensions",
        "overall price percentile",
        "within-category price percentile",
    ]
    assert summary["m2"]["item_price_budget"] == 0.25
    assert ID_ONLY_MODEL_ID == "m2_jointly_trained_id_only"
    assert RELATION_ONLY_MODEL_ID == "m2_id_plus_item_relation_only"
    assert PRICE_ONLY_MODEL_ID == "m2_id_plus_item_price_only"
    assert summary["fixed"]["graph"] == "binary"
    assert summary["fixed"]["negative_sampling"] == "uniform"
    assert summary["fixed"]["new_loss_term"] is False


@pytest.mark.parametrize(
    "override",
    [
        {"seed": 43},
        {"rho": 0.1},
        {"item_price_budget": 0.5},
        {"clv_dim": 8},
        {"id_dim": 60},
        {"time_cutoff": 697},
    ],
)
def test_fast_screen_rejects_unplanned_overrides(override):
    with pytest.raises(ValueError, match="빠른 M2 screen"):
        configure_constrained_economic_run(
            out_dir="/tmp/m2-constrained-economic",
            baseline_result_dir="/tmp/m1-result",
            **override,
        )
