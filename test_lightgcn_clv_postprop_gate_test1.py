import json
from pathlib import Path

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


def test_colab_is_pinned_and_has_one_unguarded_run_cell():
    notebook_path = Path(__file__).with_name(
        "clv_m2_postprop_gate_dunnhumby_test1_colab.ipynb"
    )
    notebook = json.loads(notebook_path.read_text())
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )

    assert "d1f913bb806c657ea4995a2355d64ae2662f3cf7" in source
    assert source.count("result_df = run_test1(cfg)") == 1
    assert "ACKNOWLEDGE_HIGH_COST" not in source
    assert "holdout은 생성·평가하지 않음" in source
