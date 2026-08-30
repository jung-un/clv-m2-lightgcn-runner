import pytest

import lightgcn_clv_conditioned_centered_balanced_history as runner


def test_preflight_locks_m2_boundaries_and_new_representation(tmp_path):
    cfg = runner.configure_centered_balanced_history_run(
        out_dir=str(tmp_path / "new"),
        baseline_result_dir=str(tmp_path / "baseline"),
    )
    summary = runner.preflight_summary(cfg)
    assert summary["trained_models"] == [
        "m2_clv_conditioned_centered_balanced_history"
    ]
    assert summary["historical_development_split"]["final_test_constructed"] is False
    assert summary["historical_development_split"]["holdout_constructed"] is False
    assert summary["m2"]["mixer_bounds"] == [0.25, 0.75]
    assert summary["m2"]["rho_max"] == 0.1
    assert summary["m2"]["rho_warmup_epochs"] == 20
    assert summary["fixed"]["graph"] == "binary"
    assert summary["fixed"]["negative_sampling"] == "uniform"
    assert summary["fixed"]["sample_weighting"] is False
    assert summary["fixed"]["new_loss_term"] is False
    assert summary["fixed"]["min_item_interactions"] == 1


def test_config_rejects_posthoc_changes(tmp_path):
    common = {
        "out_dir": str(tmp_path / "new"),
        "baseline_result_dir": str(tmp_path / "baseline"),
    }
    with pytest.raises(ValueError, match="warmup_epochs=20"):
        runner.configure_centered_balanced_history_run(**common, warmup_epochs=10)
    with pytest.raises(ValueError, match="rho=0.1"):
        runner.configure_centered_balanced_history_run(**common, rho=0.05)


def test_base_config_blocks_test_holdout_graph_and_loss_changes(tmp_path):
    cfg = runner.configure_centered_balanced_history_run(
        out_dir=str(tmp_path / "new"),
        baseline_result_dir=str(tmp_path / "baseline"),
    )
    base = runner.base._base_config(cfg)
    assert base["TIME_CUTOFF"] == 690
    assert base["HOLDOUT_DAYS"] == 0
    assert base["EVAL_HOLDOUT"] is False
    assert base["GRAPH_MODE"] == "binary"
    assert base["NEG_MODE"] == "uniform"
    assert base["LOSS_MODE"] == "plain"
    assert base["MIN_ITEM_INTER"] == 1
