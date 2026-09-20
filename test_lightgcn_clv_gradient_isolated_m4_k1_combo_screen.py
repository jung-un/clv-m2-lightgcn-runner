import numpy as np

import lightgcn_clv_gradient_isolated_m4_k1_combo_screen as combo


def test_combo_preflight_pins_seed43_k1_and_only_trains_new_arms():
    cfg = combo.configure_combo_screen(
        out_dir="/tmp/combo",
        baseline_result_dir="/tmp/baseline",
        reference_result_dir="/tmp/reference",
    )
    summary = combo.preflight_summary(cfg)

    assert cfg.seed == 43
    assert cfg.negative_count == 1
    assert summary["trained_models"] == [combo.M2_MODEL_ID, combo.M5_MODEL_ID]
    assert summary["reused_models"] == [combo.M1_MODEL_ID, combo.M4_MODEL_ID]
    assert summary["fixed"]["new_item_task"] is True
    assert summary["fixed"]["min_item_interactions"] == 1
    assert summary["fixed"]["final_test_constructed"] is False
    assert summary["fixed"]["holdout_constructed"] is False


def test_directional_decision_requires_replication_and_combination():
    metrics = {
        model_id: {
            "recall@10": 1.0,
            "ndcg@10": 1.0,
            "recall@20": 1.0,
            "ndcg@20": 1.0,
            "recall@50": 1.0,
            "ndcg@50": 1.0,
            "price_purchase_amount_weighted_hit@10": 1.0,
            "vndcg@10": 1.0,
        }
        for model_id in combo.MODEL_IDS
    }
    metrics[combo.M2_MODEL_ID].update(
        {"price_purchase_amount_weighted_hit@10": 1.01, "vndcg@10": 1.01}
    )
    metrics[combo.M4_MODEL_ID].update(
        {"price_purchase_amount_weighted_hit@10": 1.02, "vndcg@10": 1.02}
    )
    metrics[combo.M5_MODEL_ID].update(
        {"price_purchase_amount_weighted_hit@10": 1.03, "vndcg@10": 1.03}
    )

    reading = combo.decision(metrics)

    assert reading["directional_screen_pass"] is True
    assert reading["m2_replication_on_both_economic_metrics"] is True
    assert reading["m5_beats_both_parts_on_both_economic_metrics"] is True
    assert np.isfinite(list(reading["interaction"].values())).all()


def test_directional_decision_rejects_m5_below_best_part():
    metrics = {
        model_id: {
            "recall@10": 1.0,
            "ndcg@10": 1.0,
            "recall@20": 1.0,
            "ndcg@20": 1.0,
            "recall@50": 1.0,
            "ndcg@50": 1.0,
            "price_purchase_amount_weighted_hit@10": 1.0,
            "vndcg@10": 1.0,
        }
        for model_id in combo.MODEL_IDS
    }
    metrics[combo.M2_MODEL_ID].update(
        {"price_purchase_amount_weighted_hit@10": 1.02, "vndcg@10": 1.02}
    )
    metrics[combo.M4_MODEL_ID].update(
        {"price_purchase_amount_weighted_hit@10": 1.03, "vndcg@10": 1.03}
    )
    metrics[combo.M5_MODEL_ID].update(
        {"price_purchase_amount_weighted_hit@10": 1.025, "vndcg@10": 1.025}
    )

    assert combo.decision(metrics)["directional_screen_pass"] is False
