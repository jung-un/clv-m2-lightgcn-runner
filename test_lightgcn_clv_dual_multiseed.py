import dataclasses

import pandas as pd
import pytest


def _rows(revenue_deltas=(0.1, 0.1, 0.1), accuracy_ratio=1.0):
    rows = []
    for seed, revenue_delta in zip((42, 43, 44), revenue_deltas):
        baseline = {
            f"{metric}@{k}": 1.0
            for metric in ("recall", "ndcg")
            for k in (10, 20, 50)
        }
        rows.append(
            {
                "seed": seed,
                "model_id": "m1",
                "revenue@10": 1.0,
                **baseline,
            }
        )
        rows.append(
            {
                "seed": seed,
                "model_id": "dual_clv_fixed",
                "revenue@10": 1.0 + revenue_delta,
                **{key: value * accuracy_ratio for key, value in baseline.items()},
            }
        )
    return pd.DataFrame(rows)


def test_multiseed_presets_are_frozen_and_exclude_protected_runs(tmp_path):
    import lightgcn_clv_dual_multiseed as multiseed

    dun = multiseed.configure_multiseed_validation(
        "dunnhumby", tmp_path / "dun.json"
    )
    hm = multiseed.configure_multiseed_validation(
        "hm", tmp_path / "hm.json", short_hm=True
    )
    assert (dun.gate_shape, dun.fixed_lambda) == ("equal", 2.0)
    assert (hm.window_days, hm.gate_shape, hm.fixed_lambda) == (60, "high", 1.0)
    assert dun.new_seeds == hm.new_seeds == (43, 44)
    assert dun.model_ids == hm.model_ids == ("m1", "dual_clv_fixed")
    assert dun.eval_test is hm.eval_test is False
    assert dun.eval_holdout is hm.eval_holdout is False


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"new_seeds": (42, 43)}, "43, 44"),
        ({"eval_test": True}, "validation-only"),
        ({"eval_holdout": True}, "validation-only"),
        ({"model_ids": ("m1", "dual_clv_fixed", "dual_adapter_only")}, "두 모형"),
        ({"gate_shape": "high"}, "Dunnhumby"),
        ({"fixed_lambda": 1.0}, "Dunnhumby"),
    ],
)
def test_multiseed_config_rejects_protocol_changes(changes, message, tmp_path):
    import lightgcn_clv_dual_multiseed as multiseed

    cfg = multiseed.configure_multiseed_validation(
        "dunnhumby", tmp_path / "dun.json"
    )
    with pytest.raises(ValueError, match=message):
        multiseed.validate_multiseed_config(dataclasses.replace(cfg, **changes))


def test_hm_full_period_is_rejected_for_this_runner(tmp_path):
    import lightgcn_clv_dual_multiseed as multiseed

    with pytest.raises(ValueError, match="60일"):
        multiseed.configure_multiseed_validation(
            "hm", tmp_path / "hm.json", short_hm=False
        )


def test_reproducibility_decision_passes_all_three_frozen_conditions():
    from lightgcn_clv_dual_multiseed import reproducibility_decision

    decision = reproducibility_decision(_rows())
    assert decision["success"] is True
    assert decision["positive_revenue_seed_count"] == 3
    assert decision["mean_revenue_delta"] == pytest.approx(0.1)
    assert decision["failed_conditions"] == []


@pytest.mark.parametrize(
    ("rows", "failed"),
    [
        (_rows((-0.2, 0.05, 0.05)), "mean_revenue_delta_positive"),
        (_rows((0.2, -0.01, -0.01)), "positive_revenue_in_at_least_two_seeds"),
        (_rows(accuracy_ratio=0.98), "six_accuracy_mean_ratios_at_least_0.99"),
    ],
)
def test_reproducibility_decision_reports_each_failure(rows, failed):
    from lightgcn_clv_dual_multiseed import reproducibility_decision

    decision = reproducibility_decision(rows)
    assert decision["success"] is False
    assert failed in decision["failed_conditions"]
