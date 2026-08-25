import pytest

import lightgcn_clv_history_conditioned_lowrank as runner


def test_preflight_freezes_fast_seed42_screen_and_m2_boundaries(tmp_path):
    cfg = runner.configure_history_conditioned_lowrank_run(
        out_dir=str(tmp_path / "new"),
        baseline_result_dir=str(tmp_path / "old"),
    )
    summary = runner.preflight_summary(cfg)

    assert summary["trained_models"] == [
        "m2_history_conditioned_lowrank_transform"
    ]
    assert summary["reused_comparator"] == "m1_64"
    assert summary["controls_trained"] == []
    assert summary["historical_development_split"] == {
        "train_end_inclusive": 683,
        "evaluation_start_inclusive": 684,
        "evaluation_end_inclusive": 690,
        "original_validation_test_holdout_constructed": False,
    }
    assert summary["m2"]["free_user_id_embedding"] is False
    assert summary["m2"]["user_base"] == "normalized_purchase_history"
    assert summary["m2"]["conditional_map"] == (
        "[I + rho(c_N A_N + c_V A_V)] H_u"
    )
    assert summary["m2"]["transform_rank"] == 4
    assert summary["m2"]["rho"] == pytest.approx(0.05)
    assert summary["m2"]["explicit_item_features"] is False
    assert summary["m2"]["separate_axis_scores"] is False
    assert summary["fixed"]["graph"] == "binary"
    assert summary["fixed"]["negative_sampling"] == "uniform"
    assert summary["fixed"]["sample_weighting"] is False
    assert summary["fixed"]["new_loss_term"] is False
    assert summary["fixed"]["validation_or_epoch_selection"] is False


def test_shared_historical_config_keeps_new_item_task_and_boundaries(tmp_path):
    cfg = runner.configure_history_conditioned_lowrank_run(
        out_dir=str(tmp_path / "new"),
        baseline_result_dir=str(tmp_path / "old"),
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


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"rho": 0.1}, "rho=0.05"),
        ({"transform_rank": 8}, "transform_rank=4"),
        ({"embedding_dim": 96}, "embedding_dim=64"),
        ({"seed": 43}, "seed=42"),
    ],
)
def test_unplanned_screen_variants_are_rejected(tmp_path, override, message):
    with pytest.raises(ValueError, match=message):
        runner.configure_history_conditioned_lowrank_run(
            out_dir=str(tmp_path / "new"),
            baseline_result_dir=str(tmp_path / "old"),
            **override,
        )
