import dataclasses
import json
from pathlib import Path

import pytest


def test_dual_runner_is_seed42_validation_only_and_has_four_models():
    import lightgcn_clv_dual as dual

    cfg = dual.configure_dual_run("hm", short_hm=True)
    summary = dual.preflight_summary(cfg)
    assert summary["seed_list"] == [42]
    assert summary["window_days"] == 60
    assert summary["eval_test"] is False
    assert summary["eval_holdout"] is False
    assert summary["models"] == [
        "m1",
        "dual_clv_fixed",
        "dual_shuffled_gate",
        "dual_base_only",
    ]


def test_dual_runner_rejects_protected_splits_and_extra_seeds_before_data():
    import lightgcn_clv_dual as dual

    cfg = dual.configure_dual_run("dunnhumby")
    with pytest.raises(ValueError, match="validation-only"):
        dual.validate_dual_config(dataclasses.replace(cfg, eval_test=True))
    with pytest.raises(ValueError, match="seed 42"):
        dual.validate_dual_config(dataclasses.replace(cfg, seed_list=(42, 43)))


def test_screening_decision_requires_both_controls_to_be_lower():
    import lightgcn_clv_dual as dual

    selected = {
        "dual_clv_fixed": 0.5,
        "dual_shuffled_gate": 1.0,
        "dual_base_only": 1.0,
    }
    rows = [
        {"model_id": "dual_clv_fixed", "lambda": 0.5, "revenue@10": 1.10},
        {"model_id": "dual_shuffled_gate", "lambda": 1.0, "revenue@10": 1.05},
        {"model_id": "dual_base_only", "lambda": 1.0, "revenue@10": 1.11},
    ]
    decision = dual.screening_decision(
        rows,
        selected,
        {
            "dual_clv_fixed": True,
            "dual_shuffled_gate": True,
            "dual_base_only": True,
        },
    )
    assert decision["success"] is False
    assert decision["failed_controls"] == ["dual_base_only"]


def test_colab_has_only_two_presets_four_models_and_no_acknowledgement_gate():
    notebook = json.loads(Path("clv_dual_axis_colab.ipynb").read_text())
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert "DATASET_PRESET = 'hm_w60'" in source
    assert "'dunnhumby'" in source
    assert "ACKNOWLEDGE_HIGH_COST" not in source
    for model_id in (
        "dual_clv_fixed",
        "dual_shuffled_gate",
        "dual_base_only",
    ):
        assert model_id in source
