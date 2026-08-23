import pandas as pd
import pytest

import lightgcn_clv_m3_edge_allocation_backtest as pilot
import lightgcn_clv_v3 as v3


def test_pilot_is_locked_to_historical_seed42_without_selection(tmp_path):
    cfg = pilot.configure_m3_edge_allocation_backtest(out_dir=str(tmp_path))
    summary = pilot.preflight_summary(cfg)
    assert summary["models"] == ["m1_baseline", "m3_clv_edge_allocation"]
    assert summary["historical_development_split"] == {
        "train_end_inclusive": 683,
        "evaluation_start_inclusive": 684,
        "evaluation_end_inclusive": 690,
        "final_test_constructed": False,
        "holdout_constructed": False,
    }
    assert summary["seed"] == 42
    assert summary["epochs"] == 100
    assert summary["validation_or_epoch_selection"] is False


def test_base_config_preserves_m3_boundaries(tmp_path):
    previous_cfg = dict(v3.CFG)
    previous_dcfg = v3.DCFG
    cfg = pilot.configure_m3_edge_allocation_backtest(out_dir=str(tmp_path))
    base = pilot._base_config(cfg)
    assert base["TIME_CUTOFF"] == 690
    assert base["TRAIN_ON_VAL"] is True
    assert base["TEST_DAYS"] == 7
    assert base["HOLDOUT_DAYS"] == 0
    assert base["EVAL_TEST"] is True
    assert base["EVAL_HOLDOUT"] is False
    assert base["GRAPH_MODE"] == "binary"
    assert base["LOSS_MODE"] == "plain"
    assert base["NEG_MODE"] == "uniform"
    assert base["MIN_USER_INTER"] == base["MIN_ITEM_INTER"] == 1
    assert v3.CFG == previous_cfg
    assert v3.DCFG is previous_dcfg


@pytest.mark.parametrize(
    ("key", "value"), [("seed", 43), ("epochs", 99), ("time_cutoff", 697)]
)
def test_pilot_config_fails_closed(tmp_path, key, value):
    with pytest.raises(ValueError):
        pilot.configure_m3_edge_allocation_backtest(
            out_dir=str(tmp_path), **{key: value}
        )


def _passing_frame():
    return pd.DataFrame(
        [
            {
                "model_id": "m1_baseline",
                "recall@10": 1.0,
                "ndcg@10": 1.0,
                "recall@20": 1.0,
                "ndcg@20": 1.0,
                "recall@50": 1.0,
                "ndcg@50": 1.0,
                "price_purchase_amount_weighted_hit@10": 2.0,
                "mean_recommended_price_percentile@10": 0.25,
                "n_distinct@10": 200,
                "top10_share@10": 0.40,
            },
            {
                "model_id": "m3_clv_edge_allocation",
                "recall@10": 0.995,
                "ndcg@10": 1.01,
                "recall@20": 1.01,
                "ndcg@20": 1.01,
                "recall@50": 1.01,
                "ndcg@50": 1.01,
                "price_purchase_amount_weighted_hit@10": 2.1,
                "mean_recommended_price_percentile@10": 0.251,
                "n_distinct@10": 195,
                "top10_share@10": 0.405,
            },
        ]
    )


def test_pilot_decision_requires_every_predeclared_guard():
    frame = _passing_frame()
    assert pilot._pilot_decision(frame)["passes_pilot"] is True
    frame.loc[1, "recall@10"] = 0.98
    decision = pilot._pilot_decision(frame)
    assert decision["passes_pilot"] is False
    assert decision["checks"]["accuracy_guard"] is False


@pytest.mark.parametrize(
    ("metric", "value", "guard"),
    [
        ("price_purchase_amount_weighted_hit@10", 2.0, "weighted_hit_improved"),
        ("mean_recommended_price_percentile@10", 0.26, "price_guard"),
        ("n_distinct@10", 189, "catalog_guard"),
        ("top10_share@10", 0.411, "exposure_guard"),
    ],
)
def test_each_nonaccuracy_guard_can_fail(metric, value, guard):
    frame = _passing_frame()
    frame.loc[1, metric] = value
    decision = pilot._pilot_decision(frame)
    assert decision["passes_pilot"] is False
    assert decision["checks"][guard] is False
