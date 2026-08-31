import pytest

from lightgcn_clv_constrained_economic_embedding import (
    MATCHED_MODEL_ID,
    MODEL_ID,
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
    assert summary["m2"]["architecture"] == "ID(64)|one constrained CLV-economic block(4)"
    assert summary["m2"]["total_dim"] == 68
    assert summary["m2"]["user_tanh"] is False
    assert summary["m2"]["free_item_response_embedding"] is False
    assert summary["m2"]["item_inputs"] == [
        "overall price percentile",
        "within-category price percentile",
    ]
    assert summary["fixed"]["graph"] == "binary"
    assert summary["fixed"]["negative_sampling"] == "uniform"
    assert summary["fixed"]["new_loss_term"] is False


@pytest.mark.parametrize(
    "override",
    [
        {"seed": 43},
        {"rho": 0.1},
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
