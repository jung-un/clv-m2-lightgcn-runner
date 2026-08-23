import lightgcn_clv_repeatshare_backtest as backtest


def test_repeatshare_backtest_is_locked_to_unseen_historical_development_split(
    tmp_path,
):
    cfg = backtest.configure_repeatshare_backtest(out_dir=str(tmp_path))
    summary = backtest.preflight_summary(cfg)

    assert summary["models"] == [
        "m1_64",
        "m2_raw_repeatshare",
        "m2_popularity_controlled_repeatshare",
    ]
    assert summary["historical_development_split"] == {
        "train_end_inclusive": 683,
        "evaluation_start_inclusive": 684,
        "evaluation_end_inclusive": 690,
        "original_validation_test_holdout_constructed": False,
    }
    assert summary["fixed"]["graph"] == "binary"
    assert summary["fixed"]["negative_sampling"] == "uniform"
    assert summary["fixed"]["sample_weighting"] is False
    assert summary["fixed"]["validation_or_epoch_selection"] is False


def test_repeatshare_backtest_base_config_preserves_m2_boundaries(tmp_path):
    cfg = backtest.configure_repeatshare_backtest(out_dir=str(tmp_path))
    base = backtest._base_config(cfg)

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
