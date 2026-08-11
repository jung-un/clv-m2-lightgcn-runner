import dataclasses

import pytest


def test_default_single_screening_is_seed42_validation_only():
    import lightgcn_clv_single as single

    cfg = single.configure_single_run("dunnhumby")
    summary = single.preflight_summary(cfg)
    assert cfg.seed_list == (42,)
    assert cfg.eval_test is False and cfg.eval_holdout is False
    assert summary["primary_model_id"] == "single_full"
    assert summary["required_controls"] == [
        "single_zero_user",
        "single_shuffled_user",
        "single_base_only",
    ]
    assert summary["mechanism_controls"] == ["single_zero_item"]
    assert summary["graph_mode"] == "binary"
    assert summary["loss_mode"] == "plain"


@pytest.mark.parametrize("field", ["eval_test", "eval_holdout"])
def test_direct_dataclass_cannot_open_protected_splits(field):
    import lightgcn_clv_moe as moe
    import lightgcn_clv_single as single

    cfg = dataclasses.replace(moe.MoEConfig(), **{field: True})
    with pytest.raises(ValueError, match="screening-only"):
        single.validate_single_config(cfg)


def test_single_screening_decision_requires_full_to_beat_required_controls():
    import lightgcn_clv_single as single

    selected = {
        "single_full": 1.0,
        "single_zero_user": 1.0,
        "single_shuffled_user": 0.5,
        "single_zero_item": 1.0,
        "single_base_only": 0.5,
        "pref_continue": 0.0,
    }
    rows = [
        {
            "seed": 42,
            "split": "val",
            "model_id": model_id,
            "lambda": selected[model_id],
            "revenue@10": revenue,
        }
        for model_id, revenue in {
            "single_full": 1.10,
            "single_zero_user": 1.04,
            "single_shuffled_user": 1.03,
            "single_zero_item": 1.12,
            "single_base_only": 1.02,
            "pref_continue": 1.01,
        }.items()
    ]
    success = {model_id: True for model_id in selected}
    decision = single.single_screening_decision(rows, selected, success)
    assert decision["success"] is True
    assert decision["mechanism_comparison"]["single_zero_item"] == 1.12
    rows[1]["revenue@10"] = 1.11
    decision = single.single_screening_decision(rows, selected, success)
    assert decision["success"] is False
    assert decision["failed_controls"] == ["single_zero_user"]


def test_decision_tie_with_required_control_is_failure():
    import lightgcn_clv_single as single

    selected = {
        "single_full": 1.0,
        "single_zero_user": 1.0,
        "single_shuffled_user": 0.5,
        "single_zero_item": 1.0,
        "single_base_only": 0.5,
    }
    values = {
        "single_full": 1.10,
        "single_zero_user": 1.10,
        "single_shuffled_user": 1.03,
        "single_zero_item": 1.12,
        "single_base_only": 1.02,
    }
    rows = [
        {
            "seed": 42,
            "split": "val",
            "model_id": model_id,
            "lambda": selected[model_id],
            "revenue@10": revenue,
        }
        for model_id, revenue in values.items()
    ]
    success = {model_id: True for model_id in selected}
    decision = single.single_screening_decision(rows, selected, success)
    assert decision["success"] is False


def test_validate_rejects_changed_lambda_grid():
    import lightgcn_clv_single as single

    cfg = single.configure_single_run("dunnhumby")
    with pytest.raises(ValueError, match="lambda grid"):
        single.validate_single_config(
            dataclasses.replace(cfg, lambda_eval=(0.0, 1.0))
        )
