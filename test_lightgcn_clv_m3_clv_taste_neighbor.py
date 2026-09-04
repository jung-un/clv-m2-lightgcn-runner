from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from lightgcn_clv_m3_clv_taste_neighbor_diagnostic import mechanism_reading
import lightgcn_clv_m3_clv_taste_neighbor as runner


METRICS = (
    "recall@10",
    "ndcg@10",
    "recall@20",
    "ndcg@20",
    "recall@50",
    "ndcg@50",
)


def _mechanism_rows(*, high_actual=0.40):
    rows = []
    for anchor in [648, 655, 662, 669, 676]:
        for user, group in [(0, "low"), (1, "middle"), (2, "high")]:
            actual = high_actual if group == "high" else 0.50
            rows.append(
                {
                    "anchor_end": anchor,
                    "u_idx": user,
                    "q_clv": {"low": 0.2, "middle": 0.5, "high": 0.8}[group],
                    "clv_group": group,
                    "degree_stratum": user,
                    "preference_relation": 0.30,
                    "actual_clv": actual,
                    "clv_shuffle": 0.25,
                    "degree_relation": 0.28,
                }
            )
    return pd.DataFrame(rows)


def _relation_rows(*, changed_share=0.20):
    return pd.DataFrame(
        [
            {
                "anchor_end": anchor,
                "is_full_train": anchor == 683,
                "quality_passed": True,
                "same_neighbor_count_all_arms": True,
                "same_row_mass_all_arms": True,
                "actual_shuffle_changed_user_share": changed_share,
            }
            for anchor in [648, 655, 662, 669, 676, 683]
        ]
    )


def test_mechanism_requires_actual_to_beat_all_controls_and_change_neighbors():
    reading = mechanism_reading(_mechanism_rows(), _relation_rows())

    assert reading["precheck_passed"] is True
    assert reading["full_train_actual_shuffle_changed_user_share"] == 0.20
    for control in (
        "preference_relation",
        "clv_shuffle",
        "degree_relation",
    ):
        comparison = reading["comparisons"][control]
        assert comparison["overall_mean_delta"] > 0
        assert comparison["positive_anchor_count"] == 5


def test_high_clv_gain_is_reported_but_is_not_a_gate_for_neighbor_identity_hypothesis():
    reading = mechanism_reading(
        _mechanism_rows(high_actual=0.20), _relation_rows()
    )

    assert reading["precheck_passed"] is True
    assert reading["segment_diagnostics"]["preference_relation"]["high"] < 0


def test_mechanism_fails_when_actual_and_shuffle_neighbors_barely_change():
    reading = mechanism_reading(
        _mechanism_rows(), _relation_rows(changed_share=0.05)
    )

    assert reading["precheck_passed"] is False
    assert reading["checks"]["actual_shuffle_neighbor_change_at_least_10pct"] is False


def _metric_rows(actual=1.03, shuffle=1.01):
    rows = []
    for model_id, value in (
        (runner.M1_ID, 1.00),
        (runner.RELATION_ID, 1.01),
        (runner.ACTUAL_ID, actual),
        (runner.SHUFFLE_ID, shuffle),
        (runner.DEGREE_ID, 1.005),
    ):
        rows.append(
            {
                "model_id": model_id,
                **{metric: value for metric in METRICS},
                "price_purchase_amount_weighted_hit@10": value,
                "mean_recommended_price_percentile@10": 0.25,
            }
        )
    return pd.DataFrame(rows)


def test_preflight_fixes_split_graph_loss_and_clv_neighbor_role(tmp_path):
    cfg = runner.configure_clv_taste_neighbor_run(
        out_dir=str(tmp_path / "m3"),
        baseline_result_dir=str(tmp_path / "m1"),
    )
    summary = runner.preflight_summary(cfg)

    assert summary["historical_development_split"]["holdout_constructed"] is False
    assert summary["m3"]["historical_clv_proxy"] == "N_hat * V_hat"
    assert summary["m3"]["preference_candidate_neighbors"] == 100
    assert summary["m3"]["final_neighbors"] == 20
    assert summary["m3"]["reliability_kappa"] == 5.0
    assert summary["m3"]["gamma"] == 0.075
    assert summary["fixed"]["negative_sampling"] == "uniform"
    assert summary["fixed"]["sample_weighting"] is False
    assert summary["fixed"]["one_training_loop_and_optimizer"] is True
    assert set(summary["all_models"]) == {
        runner.M1_ID,
        runner.RELATION_ID,
        runner.ACTUAL_ID,
        runner.SHUFFLE_ID,
        runner.DEGREE_ID,
    }


def test_fixed_screen_rejects_changed_structure_or_training_settings(tmp_path):
    cfg = runner.configure_clv_taste_neighbor_run(
        out_dir=str(tmp_path / "m3"),
        baseline_result_dir=str(tmp_path / "m1"),
    )
    for changed in (
        replace(cfg, preference_candidate_neighbors=50),
        replace(cfg, final_neighbors=10),
        replace(cfg, reliability_kappa=10.0),
        replace(cfg, gamma=0.1),
        replace(cfg, epochs=50),
    ):
        with pytest.raises(ValueError, match="빠른 M3 screen"):
            runner.validate_clv_taste_neighbor_config(changed)


def test_attribution_requires_actual_to_beat_m1_and_every_control():
    reading = runner.attribution_reading(_metric_rows())

    assert reading["clv_attribution_supported"] is True
    assert reading["six_metric_balance_actual_vs_m1"] > 1
    assert reading["six_metric_balance_actual_vs_preference_relation"] > 1
    assert reading["six_metric_balance_actual_vs_shuffle"] > 1
    assert reading["six_metric_balance_actual_vs_degree_relation"] > 1

    failed = runner.attribution_reading(_metric_rows(actual=1.02, shuffle=1.03))
    assert failed["clv_attribution_supported"] is False


def test_training_is_blocked_before_data_loading_when_precheck_fails(tmp_path):
    cfg = runner.configure_clv_taste_neighbor_run(
        out_dir=str(tmp_path / "m3"),
        baseline_result_dir=str(tmp_path / "m1"),
    )
    with pytest.raises(RuntimeError, match="사전 관계 진단"):
        runner.run_clv_taste_neighbor_screen(
            cfg, mechanism_reading={"precheck_passed": False}
        )
