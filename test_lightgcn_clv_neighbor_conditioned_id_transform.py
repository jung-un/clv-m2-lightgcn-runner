import pytest

import lightgcn_clv_neighbor_conditioned_id_transform as runner


def test_preflight_freezes_architecture_and_m2_boundaries(tmp_path):
    cfg = runner.configure_neighbor_conditioned_id_transform_run(
        out_dir=str(tmp_path / "new"),
        baseline_result_dir=str(tmp_path / "old"),
    )
    summary = runner.preflight_summary(cfg)

    assert summary["trained_models"] == [
        "m2_neighbor_conditioned_id_transform"
    ]
    assert summary["reused_comparator"] == "m1_64"
    assert summary["historical_development_split"] == {
        "train_end_inclusive": 683,
        "evaluation_start_inclusive": 684,
        "evaluation_end_inclusive": 690,
        "original_validation_test_holdout_constructed": False,
    }
    assert summary["m2"]["embedding_dim"] == 64
    assert summary["m2"]["transform_rank"] == 4
    assert summary["m2"]["rho"] == pytest.approx(0.05)
    assert summary["m2"]["population_mean_correction_removed"] is True
    assert summary["m2"]["explicit_item_features"] is False
    assert summary["m2"]["existing_l2_extended_to_transforms"] is False
    assert summary["m2"]["transform_regularization"] == "none"
    assert summary["m2"]["epoch_liveness_diagnostics"] is True
    assert summary["m2"]["liveness_fail_closed"] == {
        "min_max_effective_correction_ratio": pytest.approx(1e-6),
        "min_mean_user_representation_change": pytest.approx(1e-8),
    }
    assert summary["fixed"]["graph"] == "binary"
    assert summary["fixed"]["negative_sampling"] == "uniform"
    assert summary["fixed"]["sample_weighting"] is False
    assert summary["fixed"]["new_loss_term"] is False
    assert summary["fixed"]["validation_or_epoch_selection"] is False


def test_shared_historical_config_keeps_new_item_task_and_boundaries(tmp_path):
    cfg = runner.configure_neighbor_conditioned_id_transform_run(
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
    assert base["DIM"] == 64


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
        runner.configure_neighbor_conditioned_id_transform_run(
            out_dir=str(tmp_path / "new"),
            baseline_result_dir=str(tmp_path / "old"),
            **override,
        )


def test_liveness_reading_passes_only_when_correction_changes_representation():
    alive = runner._liveness_reading(
        {
            "activity_effective_ratio_to_id": 2e-4,
            "value_effective_ratio_to_id": 3e-4,
            "mean_user_representation_change": 5e-4,
        }
    )
    collapsed = runner._liveness_reading(
        {
            "activity_effective_ratio_to_id": 2e-12,
            "value_effective_ratio_to_id": 3e-12,
            "mean_user_representation_change": 5e-13,
        }
    )

    assert alive["passed"] is True
    assert collapsed["passed"] is False
    assert collapsed["status"] == "collapsed_correction_path"
