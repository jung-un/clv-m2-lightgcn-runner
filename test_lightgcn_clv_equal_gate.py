from dataclasses import replace
import json
from pathlib import Path

import pytest

import lightgcn_clv_equal_gate as equal


def _row(model_id, rule, weighted_hit, accuracy=1.0):
    return {
        "model_id": model_id,
        "selection_rule": rule,
        "revenue@10": weighted_hit,
        **{metric: accuracy for metric in equal.GUARD_METRICS},
    }


def test_equal_gate_preset_is_validation_only_and_changes_one_gate():
    cfg = equal.configure_equal_gate_dunnhumby_run()
    summary = equal.preflight_summary(cfg)

    assert cfg.dataset == "dunnhumby"
    assert cfg.seed == 42
    assert cfg.gate_shape == "equal"
    assert cfg.preference_preserving is True
    assert cfg.gamma_init == 0.1
    assert cfg.eval_test is False
    assert cfg.eval_holdout is False
    assert summary["models"] == ["m1", equal.MODEL_ID]
    assert summary["m4_sample_weighting"] is False


@pytest.mark.parametrize("field", ["eval_test", "eval_holdout"])
def test_equal_gate_rejects_protected_splits(field):
    cfg = equal.configure_equal_gate_dunnhumby_run()
    with pytest.raises(ValueError):
        equal.validate_equal_gate_config(replace(cfg, **{field: True}))


def test_decision_reports_primary_and_economic_views_separately():
    rows = [
        _row("m1", equal.SELECTION_PRIMARY, 1.0),
        _row(equal.MODEL_ID, equal.SELECTION_PRIMARY, 1.02),
        _row("m1", equal.SELECTION_ECONOMIC, 1.01),
        _row(equal.MODEL_ID, equal.SELECTION_ECONOMIC, 1.03),
    ]
    decision = equal.screening_decision(rows)

    assert decision["success"] is True
    assert decision["classification"] == "strong_success"
    assert decision["views"][equal.SELECTION_PRIMARY]["success"] is True
    assert decision["views"][equal.SELECTION_ECONOMIC]["success"] is True


def test_economic_only_gain_is_not_called_strong_success():
    rows = [
        _row("m1", equal.SELECTION_PRIMARY, 1.0),
        _row(equal.MODEL_ID, equal.SELECTION_PRIMARY, 0.99),
        _row("m1", equal.SELECTION_ECONOMIC, 1.01),
        _row(equal.MODEL_ID, equal.SELECTION_ECONOMIC, 1.03),
    ]
    decision = equal.screening_decision(rows)

    assert decision["success"] is False
    assert decision["classification"] == "conditional_economic_success"


def test_colab_has_one_explicit_run_and_no_high_cost_gate():
    notebook = json.loads(
        Path("clv_m2_equal_gate_dunnhumby_colab.ipynb").read_text(
            encoding="utf-8"
        )
    )
    sources = ["".join(cell.get("source", [])) for cell in notebook["cells"]]
    joined = "\n".join(sources)

    assert sum(source.strip() == "result_df = run_experiment(cfg)" for source in sources) == 1
    assert "ACKNOWLEDGE_HIGH_COST" not in joined
    assert "REVIEWED_SHA" in joined
    assert "TO_BE_PINNED" in joined or any(
        len(token) == 40 and all(char in "0123456789abcdef" for char in token)
        for token in joined.replace("'", " ").split()
    )
