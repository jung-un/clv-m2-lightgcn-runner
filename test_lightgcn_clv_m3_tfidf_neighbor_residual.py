from dataclasses import replace

import pandas as pd
import pytest

from lightgcn_clv_m3_tfidf_neighbor_residual import (
    ACTUAL_ID,
    DEGREE_ID,
    M1_ID,
    RELATION_ID,
    SHUFFLE_ID,
    attribution_reading,
    configure_tfidf_neighbor_residual_run,
    preflight_summary,
    validate_tfidf_neighbor_residual_config,
)


METRICS = (
    "recall@10",
    "ndcg@10",
    "recall@20",
    "ndcg@20",
    "recall@50",
    "ndcg@50",
)


def _frame(actual=1.02, shuffle=1.01):
    rows = []
    for model_id, value, budget in (
        (M1_ID, 1.00, 0.0),
        (RELATION_ID, 1.01, 0.25),
        (ACTUAL_ID, actual, 0.26),
        (SHUFFLE_ID, shuffle, 0.25),
        (DEGREE_ID, 1.005, 0.24),
    ):
        rows.append(
            {
                "model_id": model_id,
                **{metric: value for metric in METRICS},
                "price_purchase_amount_weighted_hit@10": value,
                "mean_recommended_price_percentile@10": 0.25,
                "effective_budget_eligible": budget,
            }
        )
    return pd.DataFrame(rows)


def test_preflight_fixes_graph_loss_split_and_five_arms():
    cfg = configure_tfidf_neighbor_residual_run(
        out_dir="/tmp/m3-tfidf", baseline_result_dir="/tmp/baseline"
    )
    summary = preflight_summary(cfg)

    assert summary["historical_development_split"]["holdout_constructed"] is False
    assert summary["fixed"]["binary_m1_graph_preserved"] is True
    assert summary["fixed"]["negative_sampling"] == "uniform"
    assert summary["fixed"]["sample_weighting"] is False
    assert summary["fixed"]["new_loss_term"] is False
    assert summary["m3"]["historical_clv_proxy"] == "N_hat * V_hat"
    assert set(summary["all_models"]) == {
        M1_ID,
        RELATION_ID,
        ACTUAL_ID,
        SHUFFLE_ID,
        DEGREE_ID,
    }


def test_config_rejects_changed_rho_or_epochs():
    cfg = configure_tfidf_neighbor_residual_run(
        out_dir="/tmp/m3-tfidf", baseline_result_dir="/tmp/baseline"
    )
    with pytest.raises(ValueError):
        validate_tfidf_neighbor_residual_config(replace(cfg, rho=0.2))
    with pytest.raises(ValueError):
        validate_tfidf_neighbor_residual_config(replace(cfg, epochs=99))


def test_attribution_requires_actual_to_beat_m1_and_every_control():
    reading = attribution_reading(_frame())

    assert reading["clv_attribution_supported"] is True
    assert reading["six_metric_balance_actual_vs_m1"] > 1
    assert reading["six_metric_balance_actual_vs_relation_constant"] > 1
    assert reading["six_metric_balance_actual_vs_shuffle"] > 1
    assert reading["six_metric_balance_actual_vs_degree_gate"] > 1
    assert reading["budget_direction_only_warning"] is False


def test_attribution_fails_when_shuffle_is_better():
    reading = attribution_reading(_frame(actual=1.02, shuffle=1.03))

    assert reading["clv_attribution_supported"] is False


def test_budget_warning_uses_predeclared_ten_percent_boundary():
    frame = _frame()
    frame.loc[frame.model_id.eq(ACTUAL_ID), "effective_budget_eligible"] = 0.30

    assert attribution_reading(frame)["budget_direction_only_warning"] is True

