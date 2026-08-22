import pandas as pd
import pytest

import lightgcn_clv_m3_mass_preserving as M3


def test_screen_is_fixed_to_dunnhumby_seed42_validation(tmp_path):
    cfg = M3.configure_m3_clv_influence_dunnhumby_run(
        out_dir=str(tmp_path / "dunnhumby")
    )
    assert cfg["DATASET"] == "dunnhumby"
    assert cfg["SEED_LIST"] == [42]
    assert cfg["GRAPH_MODE"] == "clv"
    assert cfg["GRAPH_ALPHA"] == 1.0
    assert cfg["MIN_USER_INTER"] == cfg["MIN_ITEM_INTER"] == 1
    assert cfg["LOSS_MODE"] == "plain"
    assert cfg["NEG_MODE"] == "uniform"
    assert cfg["GATE_MODE"] == "none"
    assert cfg["EVAL_TEST"] is False
    assert cfg["EVAL_HOLDOUT"] is False
    assert M3.preflight_summary(cfg)["models"] == [
        "m1_baseline",
        *M3.MODEL_IDS.values(),
    ]


@pytest.mark.parametrize("key", ["EVAL_TEST", "EVAL_HOLDOUT"])
def test_screen_rejects_protected_splits(tmp_path, key):
    cfg = M3.configure_m3_clv_influence_dunnhumby_run(
        out_dir=str(tmp_path / "dunnhumby")
    )
    cfg[key] = True
    with pytest.raises(ValueError, match=key):
        M3.validate_screening_config(cfg)


def _frame(clv_weighted_hit=1.05):
    common = {
        "split": "val",
        "recall@10": 1.0,
        "ndcg@10": 1.0,
        "recall@20": 1.0,
        "ndcg@20": 1.0,
        "recall@50": 1.0,
        "ndcg@50": 1.0,
        "precision@10": 0.1,
        "precision@20": 0.05,
        "precision@50": 0.02,
        "revenue@20": 1.0,
        "revenue@50": 1.0,
        "arp@10": 0.5,
        "n_distinct@10": 100,
        "top10_share@10": 0.2,
    }
    return M3.normalize_result_schema(
        pd.DataFrame(
            [
                {**common, "model_id": "m1_baseline", "revenue@10": 1.0},
                {
                    **common,
                    "model_id": M3.MODEL_IDS["n_only"],
                    "revenue@10": 1.01,
                },
                {
                    **common,
                    "model_id": M3.MODEL_IDS["v_only"],
                    "revenue@10": 1.02,
                },
                {
                    **common,
                    "model_id": M3.MODEL_IDS["clv"],
                    "revenue@10": clv_weighted_hit,
                },
                {
                    **common,
                    "model_id": M3.MODEL_IDS["clv_shuffle"],
                    "revenue@10": 1.00,
                },
            ]
        )
    )


def test_clv_must_beat_m1_both_axes_and_shuffle():
    decision = M3.screening_decision(_frame())
    assert decision["success"] is True
    assert all(decision["guards"].values())


def test_clv_fails_if_it_does_not_beat_v_only():
    decision = M3.screening_decision(_frame(clv_weighted_hit=1.015))
    assert decision["success"] is False
    assert decision["guards"]["weighted_hit_vs_v_only"] is False
