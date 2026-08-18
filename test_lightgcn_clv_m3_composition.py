import pandas as pd
import pytest

import lightgcn_clv_m3_composition as M3


def test_composition_screening_is_dunnhumby_seed42_validation_only(tmp_path):
    cfg = M3.configure_m3_clv_composition_dunnhumby_run(
        out_dir=str(tmp_path / "dunnhumby")
    )
    assert cfg["DATASET"] == "dunnhumby"
    assert cfg["SEED_LIST"] == [42]
    assert cfg["GRAPH_MODE"] == "clv_composition"
    assert cfg["MIN_USER_INTER"] == cfg["MIN_ITEM_INTER"] == 1
    assert cfg["LOSS_MODE"] == "plain"
    assert cfg["NEG_MODE"] == "uniform"
    assert cfg["EVAL_TEST"] is False
    assert cfg["EVAL_HOLDOUT"] is False
    assert M3.preflight_summary(cfg)["models"] == [
        "m1_baseline",
        "m3_clv_composition",
    ]


@pytest.mark.parametrize("key", ["EVAL_TEST", "EVAL_HOLDOUT"])
def test_composition_screening_rejects_protected_splits(tmp_path, key):
    cfg = M3.configure_m3_clv_composition_dunnhumby_run(
        out_dir=str(tmp_path / "dunnhumby")
    )
    cfg[key] = True
    with pytest.raises(ValueError, match=key):
        M3.validate_screening_config(cfg)


def test_composition_decision_compares_only_model_and_m1():
    common = {
        "split": "val",
        "recall@10": 1.0,
        "ndcg@10": 1.0,
        "recall@20": 1.0,
        "ndcg@20": 1.0,
        "recall@50": 1.0,
        "ndcg@50": 1.0,
    }
    frame = pd.DataFrame(
        [
            {**common, "model_id": "m1_baseline", "revenue@10": 1.0},
            {**common, "model_id": "m3_clv_composition", "revenue@10": 1.1},
        ]
    )
    assert M3.screening_decision(frame)["success"] is True
