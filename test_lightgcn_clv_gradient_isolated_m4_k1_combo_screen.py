import numpy as np
import pandas as pd

import lightgcn_clv_gradient_isolated_m4_k1_combo_screen as combo


def test_prepare_builds_the_exact_m4_user_bin_fit(monkeypatch, tmp_path):
    train = pd.DataFrame(
        {
            "u_idx": [0, 0, 1, 1, 2, 2, 3, 3],
            "i_idx": [0, 1, 1, 2, 2, 3, 3, 0],
            "cat_idx": [0, 0, 0, 1, 1, 1, 1, 0],
            "v": [1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 1.0],
        }
    )
    base = {
        "data": {
            "train": train,
            "n_users": 4,
            "n_items": 4,
            "tr_u": train["u_idx"].to_numpy(np.int64),
            "tr_i": train["i_idx"].to_numpy(np.int64),
        },
        "q_n": np.array([0.1, 0.4, 0.7, 1.0]),
        "q_v": np.array([0.2, 0.5, 0.8, 1.0]),
        "q_c": np.array([0.1, 0.3, 0.7, 1.0]),
        "clv_valid": np.ones(4, dtype=bool),
        "input_hash": "synthetic-input",
        "revision": "synthetic-revision",
    }
    monkeypatch.setattr(combo.gi, "_prepare", lambda cfg: dict(base))
    cfg = combo.configure_combo_screen(
        out_dir=str(tmp_path / "out"),
        baseline_result_dir=str(tmp_path / "baseline"),
        reference_result_dir=str(tmp_path / "reference"),
    )

    prepared = combo._prepare(cfg)

    assert prepared["user_bin_fit"].shape == (4, cfg.economic_bins)
    weights, diagnostics = combo._weights(prepared, cfg, weighted=True)
    assert weights.shape == (len(train),)
    assert diagnostics["row_weight_cv"] > 0.0


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
