import pytest

import lightgcn_clv_temporal_history_item_fit as runner


def test_preflight_adds_only_fixed_relationship_time_decay(tmp_path):
    cfg = runner.configure_temporal_history_item_fit_run(
        out_dir=str(tmp_path / "new"),
        baseline_result_dir=str(tmp_path / "baseline"),
        current_m2_result_dir=str(tmp_path / "current"),
    )
    summary = runner.preflight_summary(cfg)

    assert summary["trained_models"] == ["m2_flv_temporal_personal_history_fit"]
    assert summary["reused_comparators"] == [
        "m1_64",
        "m2_nv_personal_history_candidate_fit",
    ]
    assert summary["historical_development_split"] == {
        "train_end_inclusive": 683,
        "evaluation_start_inclusive": 684,
        "evaluation_end_inclusive": 690,
        "final_test_constructed": False,
        "holdout_constructed": False,
    }
    assert summary["m2"]["relationship_time_decay"] == (
        "exp(-user-item recency / user mean distinct-basket gap)"
    )
    assert summary["m2"]["time_decay_learned"] is False
    assert summary["m2"]["invalid_gap_fallback"] == "original N/V history shares"
    assert summary["m2"]["learned_attention"] is False
    assert summary["m2"]["fixed_axis_scale"] == pytest.approx(0.05)
    assert summary["fixed"]["graph"] == "binary"
    assert summary["fixed"]["negative_sampling"] == "uniform"
    assert summary["fixed"]["sample_weighting"] is False
    assert summary["fixed"]["new_loss_term"] is False


def test_config_rejects_new_strength_or_capacity_search(tmp_path):
    common = {
        "out_dir": str(tmp_path / "new"),
        "baseline_result_dir": str(tmp_path / "baseline"),
        "current_m2_result_dir": str(tmp_path / "current"),
    }
    with pytest.raises(ValueError, match="rho=0.05"):
        runner.configure_temporal_history_item_fit_run(**common, rho=0.1)
    with pytest.raises(ValueError, match="axis_dim=4"):
        runner.configure_temporal_history_item_fit_run(**common, axis_dim=8)


def test_base_config_keeps_historical_development_and_m2_boundaries(tmp_path):
    cfg = runner.configure_temporal_history_item_fit_run(
        out_dir=str(tmp_path / "new"),
        baseline_result_dir=str(tmp_path / "baseline"),
        current_m2_result_dir=str(tmp_path / "current"),
    )
    base = runner._base_config(cfg)

    assert base["TIME_CUTOFF"] == 690
    assert base["TRAIN_ON_VAL"] is True
    assert base["TEST_DAYS"] == 7
    assert base["HOLDOUT_DAYS"] == 0
    assert base["GRAPH_MODE"] == "binary"
    assert base["NEG_MODE"] == "uniform"
    assert base["LOSS_MODE"] == "plain"
    assert base["MIN_ITEM_INTER"] == 1
    assert base["EPOCHS"] == 100
