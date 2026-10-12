from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

import clv_m2_adaptive_lastweek_final10 as s


def test_frozen_dataset_batches_keep_final_week_and_plain_bpr():
    dh = s.configure("dunnhumby", out_dir="/tmp/m2-dh")
    hm = s.configure("hm", out_dir="/tmp/m2-hm")
    assert dh.seeds == tuple(range(42, 52))
    assert hm.seeds == (49,)
    assert (dh.batch_size, hm.batch_size) == (8192, 131072)
    for cfg in (dh, hm):
        base = s.base_config(cfg)
        assert (base["VAL_DAYS"], base["TEST_DAYS"], base["HOLDOUT_DAYS"]) == (0, 7, 0)
        assert base["TRAIN_ON_VAL"] and base["EVAL_TEST"] and not base["EVAL_HOLDOUT"]
        assert (base["GRAPH_MODE"], base["LOSS_MODE"], base["NEG_MODE"]) == (
            "binary", "plain", "uniform"
        )
        assert base["MIN_ITEM_INTER"] == 1
        assert cfg.epochs == 300 and cfg.negative_count == 1
    with pytest.raises(ValueError):
        s._validate_config(replace(hm, epochs=200))
    with pytest.raises(ValueError):
        s._validate_config(replace(dh, seeds=(42,)))


def test_hm_one_seed_is_partial_not_a_final_mean(tmp_path):
    cfg = s.configure("hm", out_dir=str(tmp_path))
    prepared = {
        "run_dir": Path(tmp_path),
        "protocol": {"compute_seeds": [49]},
    }
    arm = prepared["run_dir"] / "arms" / f"{s.MODEL_ID}_s49.json"
    arm.parent.mkdir()
    s.io._atomic_json(arm, {
        "identity": {
            "stage": s.CODE_VERSION, "model_id": s.MODEL_ID, "seed": 49,
            "config_hash": "test", "source_revision": "source", "input_hash": "input",
        },
        "model_id": s.MODEL_ID, "seed": 49, "epochs": 300,
        "metrics": {f"metric_{i}@10": float(i) for i in range(142)},
        "diagnostics": {"graph": "binary", "row_weighted": False},
    })
    prepared.update(config_hash="test", revision="source", input_hash="input")
    result = s.report(prepared)
    assert not result["complete"]
    assert "summary" not in result
    assert len(pd.read_csv(result["paths"]["absolute"]).columns) == 144
