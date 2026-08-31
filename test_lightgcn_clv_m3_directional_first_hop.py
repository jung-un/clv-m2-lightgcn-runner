import pandas as pd
import pytest

import lightgcn_clv_m3_directional_first_hop as runner


def test_preflight_locks_fast_historical_m3_screen_and_four_arms(tmp_path):
    cfg = runner.configure_directional_first_hop_run(
        out_dir=str(tmp_path / "m3"),
        baseline_result_dir=str(tmp_path / "m1"),
    )
    summary = runner.preflight_summary(cfg)

    assert summary["trained_models"] == [
        runner.RELATION_ONLY_ID,
        runner.ACTUAL_ID,
        runner.SHUFFLE_ID,
    ]
    assert summary["reused_comparator"] == runner.M1_ID
    assert summary["historical_development_split"] == {
        "train_end_inclusive": 683,
        "evaluation_start_inclusive": 684,
        "evaluation_end_inclusive": 690,
        "final_test_constructed": False,
        "holdout_constructed": False,
    }
    assert summary["m3"]["historical_clv_proxy"] == "N_hat * V_hat"
    assert summary["m3"]["changed_term"] == "user first-hop only"
    assert summary["m3"]["target_first_hop_strength"] == 0.075
    assert summary["fixed"]["negative_sampling"] == "uniform"
    assert summary["fixed"]["one_training_loop_and_optimizer"] is True
    assert summary["fixed"]["min_item_interactions"] == 1
    assert summary["reading_rule"]["accuracy_guardrails"] is False


@pytest.mark.parametrize(
    "override",
    [
        {"seed": 43},
        {"time_cutoff": 697},
        {"target_strength": 0.1},
        {"n_layers": 1},
        {"epochs": 50},
    ],
)
def test_fast_screen_rejects_unplanned_overrides(tmp_path, override):
    with pytest.raises(ValueError, match="빠른 M3 screen"):
        runner.configure_directional_first_hop_run(
            out_dir=str(tmp_path / "m3"),
            baseline_result_dir=str(tmp_path / "m1"),
            **override,
        )


def _metrics(multiplier):
    return {
        "recall@10": 0.01 * multiplier,
        "ndcg@10": 0.011 * multiplier,
        "recall@20": 0.02 * multiplier,
        "ndcg@20": 0.021 * multiplier,
        "recall@50": 0.04 * multiplier,
        "ndcg@50": 0.03 * multiplier,
        "price_purchase_amount_weighted_hit@10": 0.3 * multiplier,
        "mean_recommended_price_percentile@10": 0.25,
    }


def test_clv_attribution_requires_actual_to_beat_all_three_comparators():
    rows = [
        {"model_id": runner.M1_ID, **_metrics(1.00)},
        {"model_id": runner.RELATION_ONLY_ID, **_metrics(1.01)},
        {"model_id": runner.ACTUAL_ID, **_metrics(1.02)},
        {"model_id": runner.SHUFFLE_ID, **_metrics(1.005)},
    ]
    reading = runner.attribution_reading(pd.DataFrame(rows))
    assert reading["clv_attribution_supported"] is True
    assert reading["six_metric_balance_actual_vs_m1"] > 1
    assert reading["six_metric_balance_actual_vs_relation_only"] > 1
    assert reading["six_metric_balance_actual_vs_shuffle"] > 1
    assert "accuracy_guard" not in reading

    rows[-1] = {"model_id": runner.SHUFFLE_ID, **_metrics(1.03)}
    reading = runner.attribution_reading(pd.DataFrame(rows))
    assert reading["clv_attribution_supported"] is False
    assert reading["six_metric_balance_actual_vs_shuffle"] < 1

