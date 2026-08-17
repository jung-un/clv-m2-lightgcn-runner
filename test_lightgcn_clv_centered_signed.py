from dataclasses import replace

import pandas as pd
import pytest

import lightgcn_clv_centered_signed as centered


def _metric_row(model_id, revenue=1.0, accuracy=1.0):
    return {
        "model_id": model_id,
        "revenue@10": revenue,
        **{
            metric: accuracy
            for metric in centered.ACCURACY_METRICS
        },
    }


def test_centered_signed_preset_is_dunnhumby_validation_only():
    cfg = centered.configure_centered_signed_dunnhumby_run()
    summary = centered.preflight_summary(cfg)

    assert cfg.dataset == "dunnhumby"
    assert cfg.seed == 42
    assert cfg.gate_shape == "centered"
    assert cfg.preference_preserving is True
    assert cfg.anchor_weight == 0.0
    assert cfg.eval_test is False
    assert cfg.eval_holdout is False
    assert summary["models"] == [
        "m1",
        centered.MODEL_ID,
        centered.SHUFFLED_ID,
    ]
    assert summary["m4_sample_weighting"] is False


@pytest.mark.parametrize("field", ["eval_test", "eval_holdout"])
def test_centered_signed_rejects_protected_splits(field):
    cfg = centered.configure_centered_signed_dunnhumby_run()
    with pytest.raises(ValueError):
        centered.validate_centered_signed_config(
            replace(cfg, **{field: True})
        )


def test_success_requires_m1_gain_and_correct_user_assignment():
    passing = [
        _metric_row("m1", 1.0),
        _metric_row(centered.MODEL_ID, 1.02),
        _metric_row(centered.SHUFFLED_ID, 1.01),
    ]
    decision = centered.screening_decision(passing)
    assert decision["success"] is True
    assert decision["correct_assignment_beats_shuffled"] is True

    shuffled_wins = pd.DataFrame(passing).to_dict("records")
    shuffled_wins[2]["revenue@10"] = 1.03
    decision = centered.screening_decision(shuffled_wins)
    assert decision["success"] is False
    assert decision["correct_assignment_beats_shuffled"] is False
