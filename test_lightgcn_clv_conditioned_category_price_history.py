import pytest

import lightgcn_clv_conditioned_category_price_history as runner


def test_preflight_locks_historical_screen_and_m2_boundaries(tmp_path):
    cfg = runner.configure_conditioned_history_run(
        out_dir=str(tmp_path / "new"),
        baseline_result_dir=str(tmp_path / "baseline"),
    )
    summary = runner.preflight_summary(cfg)

    assert summary["trained_models"] == [
        "m2_clv_conditioned_category_price_history"
    ]
    assert summary["reused_comparator"] == "m1_64"
    assert summary["historical_development_split"] == {
        "train_end_inclusive": 683,
        "evaluation_start_inclusive": 684,
        "evaluation_end_inclusive": 690,
        "final_test_constructed": False,
        "holdout_constructed": False,
    }
    assert summary["m2"]["rho"] == pytest.approx(0.1)
    assert summary["m2"]["total_dim"] == 72
    assert summary["m2"]["learned_global_axis_weight"] is False
    assert summary["m2"]["raw_item_popularity_input"] is False
    assert summary["m2"]["raw_repeatshare_input"] is False
    assert summary["fixed"]["graph"] == "binary"
    assert summary["fixed"]["negative_sampling"] == "uniform"
    assert summary["fixed"]["sample_weighting"] is False
    assert summary["fixed"]["new_loss_term"] is False
    assert summary["fixed"]["one_training_loop_and_optimizer"] is True
    assert summary["fixed"]["min_item_interactions"] == 1


def test_config_rejects_posthoc_strength_or_capacity_changes(tmp_path):
    common = {
        "out_dir": str(tmp_path / "new"),
        "baseline_result_dir": str(tmp_path / "baseline"),
    }
    with pytest.raises(ValueError, match="rho=0.1"):
        runner.configure_conditioned_history_run(**common, rho=0.05)
    with pytest.raises(ValueError, match="category_dim=4"):
        runner.configure_conditioned_history_run(**common, category_dim=8)


def test_base_config_blocks_final_test_holdout_m3_and_m4(tmp_path):
    cfg = runner.configure_conditioned_history_run(
        out_dir=str(tmp_path / "new"),
        baseline_result_dir=str(tmp_path / "baseline"),
    )
    base = runner._base_config(cfg)

    assert base["TIME_CUTOFF"] == 690
    assert base["TRAIN_ON_VAL"] is True
    assert base["TEST_DAYS"] == 7
    assert base["HOLDOUT_DAYS"] == 0
    assert base["EVAL_TEST"] is True
    assert base["EVAL_HOLDOUT"] is False
    assert base["GRAPH_MODE"] == "binary"
    assert base["NEG_MODE"] == "uniform"
    assert base["LOSS_MODE"] == "plain"
    assert base["MIN_USER_INTER"] == 1
    assert base["MIN_ITEM_INTER"] == 1
    assert base["EPOCHS"] == 100
