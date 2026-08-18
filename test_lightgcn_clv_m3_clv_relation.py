import pandas as pd
import pytest

import lightgcn_clv_m3_clv_relation as runner


def test_screening_config_locks_protected_split_and_m1_settings(tmp_path):
    cfg = runner.configure_m3_clv_relation_dunnhumby_run(
        out_dir=str(tmp_path / "dunnhumby")
    )
    assert cfg["SEED_LIST"] == [42]
    assert cfg["MIN_ITEM_INTER"] == 1
    assert cfg["LOSS_MODE"] == "plain"
    assert cfg["NEG_MODE"] == "uniform"
    assert cfg["EVAL_TEST"] is False
    assert cfg["EVAL_HOLDOUT"] is False


@pytest.mark.parametrize(
    "override",
    [
        {"MIN_ITEM_INTER": 10},
        {"LOSS_MODE": "user"},
        {"NEG_MODE": "hard50"},
        {"EVAL_TEST": True},
        {"EVAL_HOLDOUT": True},
    ],
)
def test_screening_config_rejects_invariant_violation(tmp_path, override):
    with pytest.raises(ValueError):
        runner.configure_m3_clv_relation_dunnhumby_run(
            out_dir=str(tmp_path), **override
        )


def _row(model_id, revenue, arp, distinct, top10, precision=0.001):
    row = {
        "model_id": model_id,
        "split": "val",
        "revenue@10": revenue,
        "arp@10": arp,
        "n_distinct@10": distinct,
        "top10_share@10": top10,
        "mean_hits@10": 10 * precision,
        "hit_value@10": revenue / (10 * precision),
    }
    for metric in runner.ACCURACY_METRICS:
        row[metric] = 1.0
    return row


def test_screen_requires_m1_control_and_mechanism_guards():
    frame = pd.DataFrame(
        [
            _row("m1_baseline", 1.00, 0.25, 200, 0.40),
            _row(runner.RELATION_CONTROL_ID, 1.01, 0.25, 200, 0.40),
            _row(runner.GATE_ID, 1.02, 0.25, 200, 0.40),
            _row(runner.ALLOCATED_CONTROL_ID, 1.01, 0.25, 200, 0.40),
            _row(runner.ALLOCATED_GATE_ID, 1.03, 0.25, 200, 0.40),
        ]
    )
    decision = runner.screening_decision(frame)
    assert decision["success"] is True
    assert decision["clv_as_mixture_gate"]["passes_screen"] is True
    assert decision["clv_in_edge_and_gate"]["passes_screen"] is True


def test_price_shift_fails_even_if_weighted_hit_increases():
    frame = pd.DataFrame(
        [
            _row("m1_baseline", 1.00, 0.25, 200, 0.40),
            _row(runner.RELATION_CONTROL_ID, 1.01, 0.25, 200, 0.40),
            _row(runner.GATE_ID, 1.20, 0.30, 200, 0.40),
            _row(runner.ALLOCATED_CONTROL_ID, 1.01, 0.25, 200, 0.40),
            _row(runner.ALLOCATED_GATE_ID, 1.20, 0.30, 200, 0.40),
        ]
    )
    decision = runner.screening_decision(frame)
    assert decision["success"] is False
    assert not decision["clv_as_mixture_gate"]["guards"][
        "recommended_price_percentile"
    ]
