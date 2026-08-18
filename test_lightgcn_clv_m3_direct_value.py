import pandas as pd
import pytest

import lightgcn_clv_m3_direct_value as M3


def test_direct_screen_is_fixed_to_dunnhumby_seed42_validation(tmp_path):
    cfg = M3.configure_m3_direct_clv_dunnhumby_run(
        out_dir=str(tmp_path / "dunnhumby")
    )
    assert cfg["DATASET"] == "dunnhumby"
    assert cfg["SEED_LIST"] == [42]
    assert cfg["GRAPH_MODE"] == "clv_direct_user_spend"
    assert cfg["GRAPH_ALPHA"] == 1.0
    assert cfg["MIN_USER_INTER"] == cfg["MIN_ITEM_INTER"] == 1
    assert cfg["LOSS_MODE"] == "plain"
    assert cfg["NEG_MODE"] == "uniform"
    assert cfg["EVAL_TEST"] is False
    assert cfg["EVAL_HOLDOUT"] is False
    assert M3.preflight_summary(cfg)["models"] == [
        "m1_baseline",
        M3.USER_ID,
        M3.SPEND_CONTROL_ID,
        M3.CLV_SPEND_ID,
    ]


@pytest.mark.parametrize("key", ["EVAL_TEST", "EVAL_HOLDOUT"])
def test_direct_screen_rejects_protected_splits(tmp_path, key):
    cfg = M3.configure_m3_direct_clv_dunnhumby_run(
        out_dir=str(tmp_path / "dunnhumby")
    )
    cfg[key] = True
    with pytest.raises(ValueError, match=key):
        M3.validate_screening_config(cfg)


def _frame(user_rev=1.1, spend_rev=1.05, joint_rev=1.2, joint_recall=1.0):
    common = {
        "split": "val",
        "recall@10": 1.0,
        "ndcg@10": 1.0,
        "recall@20": 1.0,
        "ndcg@20": 1.0,
        "recall@50": 1.0,
        "ndcg@50": 1.0,
    }
    return pd.DataFrame(
        [
            {**common, "model_id": "m1_baseline", "revenue@10": 1.0},
            {**common, "model_id": M3.USER_ID, "revenue@10": user_rev},
            {**common, "model_id": M3.SPEND_CONTROL_ID, "revenue@10": spend_rev},
            {
                **common,
                "model_id": M3.CLV_SPEND_ID,
                "recall@10": joint_recall,
                "revenue@10": joint_rev,
            },
        ]
    )


def test_decision_reports_both_clv_hypotheses_and_control():
    decision = M3.screening_decision(_frame())
    assert decision["success"] is True
    assert decision["user_clv_only"]["passes_m1_screen"] is True
    assert decision["clv_x_edge_spend"]["success"] is True
    assert decision["clv_x_edge_spend"]["beats_spend_control"] is True


def test_joint_requires_accuracy_and_spend_control_superiority():
    decision = M3.screening_decision(
        _frame(user_rev=0.9, spend_rev=1.15, joint_rev=1.1, joint_recall=0.98)
    )
    assert decision["success"] is False
    assert decision["user_clv_only"]["passes_m1_screen"] is False
    assert decision["clv_x_edge_spend"]["passes_m1_screen"] is False
    assert decision["clv_x_edge_spend"]["beats_spend_control"] is False
    assert decision["clv_x_edge_spend"]["success"] is False
