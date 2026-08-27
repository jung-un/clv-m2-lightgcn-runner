import pytest

import lightgcn_clv_fixed_composition_nv as runner


def test_preflight_locks_historical_screen_and_m2_boundaries(tmp_path):
    cfg = runner.configure_fixed_composition_run(
        out_dir=str(tmp_path / "new"),
        baseline_result_dir=str(tmp_path / "baseline"),
    )
    summary = runner.preflight_summary(cfg)

    assert summary["trained_models"] == ["m2_total_level_composition_nv"]
    assert summary["reused_comparator"] == "m1_64"
    assert summary["historical_development_split"] == {
        "train_end_inclusive": 683,
        "evaluation_start_inclusive": 684,
        "evaluation_end_inclusive": 690,
        "final_test_constructed": False,
        "holdout_constructed": False,
    }
    assert summary["m2"]["architecture"] == "ID(64)|activity(4)|transaction-value(4)"
    assert summary["m2"]["fixed_max_axis_scale"] == pytest.approx(0.05)
    assert "q_C" in summary["m2"]["user_total_axis_level"]
    assert summary["m2"]["user_axis_allocation"] == (
        "b_N=q_C*pi_N, b_V=q_C*pi_V; b_N+b_V=q_C"
    )
    assert summary["m2"]["learned_global_axis_weight"] is False
    assert summary["m2"]["raw_repeatshare_input"] is False
    assert summary["m2"]["raw_item_popularity_input"] is False
    assert summary["m2"]["total_dim"] == 72
    assert summary["fixed"]["graph"] == "binary"
    assert summary["fixed"]["negative_sampling"] == "uniform"
    assert summary["fixed"]["sample_weighting"] is False
    assert summary["fixed"]["new_loss_term"] is False
    assert summary["fixed"]["one_training_loop_and_optimizer"] is True
    assert summary["fixed"]["min_item_interactions"] == 1
    assert summary["fixed"]["validation_or_epoch_selection"] is False


def test_config_rejects_posthoc_capacity_or_strength_change(tmp_path):
    common = {
        "out_dir": str(tmp_path / "new"),
        "baseline_result_dir": str(tmp_path / "baseline"),
    }
    with pytest.raises(ValueError, match="rho=0.05"):
        runner.configure_fixed_composition_run(**common, rho=0.1)
    with pytest.raises(ValueError, match="axis_dim=4"):
        runner.configure_fixed_composition_run(**common, axis_dim=8)


def test_base_config_preserves_graph_loss_sampling_and_protected_splits(tmp_path):
    cfg = runner.configure_fixed_composition_run(
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


def test_comparison_uses_only_public_metric_whitelist():
    baseline = {
        "model_id": "m1_64",
        "recall@10": 0.1,
        "ndcg@10": 0.2,
        "rho": 123.0,
    }
    arm = {
        "metrics": {"recall@10": 0.11, "ndcg@10": 0.19},
        "diagnostics": {"rho": 0.05},
    }

    comparison = runner._comparison(baseline, arm)

    assert set(comparison.metric) == {"recall@10", "ndcg@10"}
    assert "rho" not in set(comparison.metric)
