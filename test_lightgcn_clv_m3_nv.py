import pandas as pd
import pytest
import json
from pathlib import Path

import lightgcn_clv_m3_nv as M3


def test_dunnhumby_screening_config_is_safe_and_comparable(tmp_path):
    cfg = M3.configure_m3_clv_nv_dunnhumby_run(out_dir=str(tmp_path / "dunnhumby"))
    assert cfg["DATASET"] == "dunnhumby"
    assert cfg["SEED_LIST"] == [42]
    assert cfg["ARCH"] == "pref_only"
    assert cfg["GRAPH_MODE"] == "clv_nv"
    assert cfg["LOSS_MODE"] == "plain"
    assert cfg["NEG_MODE"] == "uniform"
    assert cfg["MIN_USER_INTER"] == cfg["MIN_ITEM_INTER"] == 1
    assert cfg["WINDOW_DAYS"] is None
    assert cfg["EVAL_TEST"] is False
    assert cfg["EVAL_HOLDOUT"] is False


@pytest.mark.parametrize("key", ["EVAL_TEST", "EVAL_HOLDOUT"])
def test_screening_rejects_protected_splits(tmp_path, key):
    cfg = M3.configure_m3_clv_nv_dunnhumby_run(out_dir=str(tmp_path / "dunnhumby"))
    cfg[key] = True
    with pytest.raises(ValueError, match=key):
        M3.validate_screening_config(cfg)


def test_decision_requires_economic_gain_and_accuracy_guardrails():
    rows = [
        {"model_id": "m1_baseline", "role": "baseline", "split": "val",
         "recall@10": 1.0, "ndcg@10": 1.0, "recall@20": 1.0,
         "ndcg@20": 1.0, "recall@50": 1.0, "ndcg@50": 1.0,
         "revenue@10": 1.0},
        {"model_id": "m3_clv_nv", "role": "model", "split": "val",
         "recall@10": 1.0, "ndcg@10": 1.0, "recall@20": 0.99,
         "ndcg@20": 1.01, "recall@50": 1.0, "ndcg@50": 1.0,
         "revenue@10": 1.01},
    ]
    decision = M3.screening_decision(pd.DataFrame(rows))
    assert decision["success"] is True
    rows[1]["revenue@10"] = 1.0
    assert M3.screening_decision(pd.DataFrame(rows))["success"] is False


def test_colab_is_pinned_and_runs_only_the_safe_runner():
    path = Path(__file__).with_name("clv_m3_nv_dunnhumby_colab.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert "79a5dd7905ae8917c76107b3b9d810ab00e22a10" in source
    assert "configure_m3_clv_nv_dunnhumby_run" in source
    assert source.count("run_experiment(cfg)") == 1
    assert "EVAL_TEST=False" in source and "EVAL_HOLDOUT=False" in source
