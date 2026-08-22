import pytest

import lightgcn_clv_postprop_gate_test1 as runner


def test_protocol_is_seed42_fixed_epoch_test_only(tmp_path):
    cfg = runner.configure_test1_run(out_dir=str(tmp_path))
    summary = runner.preflight_summary(cfg)
    base = runner._base_config(cfg)

    assert cfg.seed == 42
    assert cfg.epochs == 100
    assert summary["models"] == ["m1_64", "m2_postprop_axis_gate"]
    assert summary["validation"].startswith("not constructed")
    assert summary["holdout"].startswith("disabled")
    assert base["TRAIN_ON_VAL"] is True
    assert base["EVAL_TEST"] is True
    assert base["EVAL_HOLDOUT"] is False
    assert base["MIN_USER_INTER"] == 1
    assert base["MIN_ITEM_INTER"] == 1


@pytest.mark.parametrize(
    ("name", "value"),
    [("seed", 43), ("epochs", 99), ("dataset", "hm"), ("gate_shape", "equal")],
)
def test_protocol_rejects_changes(tmp_path, name, value):
    with pytest.raises(ValueError):
        runner.configure_test1_run(out_dir=str(tmp_path), **{name: value})


def test_preflight_declares_actual_current_feature_schema(tmp_path):
    summary = runner.preflight_summary(
        runner.configure_test1_run(out_dir=str(tmp_path))
    )

    assert summary["m2_feature_schema"] == runner.FEATURE_SCHEMA
    assert summary["m2_feature_schema"]["user_activity"][0] == (
        "repeat_transaction_count"
    )
    assert summary["m2_feature_schema"]["item_value"][-1] == (
        "mean_transaction_value_share"
    )
    assert summary["evidence_status"].startswith("exploratory")
