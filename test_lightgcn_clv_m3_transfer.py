import pandas as pd
import pytest
import json
from pathlib import Path

import lightgcn_clv_m3_transfer as M3


def test_transfer_screening_config_is_safe_and_comparable(tmp_path):
    cfg = M3.configure_m3_transfer_dunnhumby_run(
        out_dir=str(tmp_path / "dunnhumby")
    )
    assert cfg["DATASET"] == "dunnhumby"
    assert cfg["SEED_LIST"] == [42]
    assert cfg["GRAPH_MODES"] == ("n_transfer", "v_contribution")
    assert cfg["ARCH"] == "pref_only"
    assert cfg["LOSS_MODE"] == "plain"
    assert cfg["NEG_MODE"] == "uniform"
    assert cfg["MIN_USER_INTER"] == cfg["MIN_ITEM_INTER"] == 1
    assert cfg["EVAL_TEST"] is False
    assert cfg["EVAL_HOLDOUT"] is False


@pytest.mark.parametrize("key", ["EVAL_TEST", "EVAL_HOLDOUT"])
def test_transfer_screening_rejects_protected_splits(tmp_path, key):
    cfg = M3.configure_m3_transfer_dunnhumby_run(
        out_dir=str(tmp_path / "dunnhumby")
    )
    cfg[key] = True
    with pytest.raises(ValueError, match=key):
        M3.validate_screening_config(cfg)


def test_decision_is_independent_for_n_and_v_arms():
    common = {
        "split": "val",
        "recall@10": 1.0,
        "ndcg@10": 1.0,
        "recall@20": 1.0,
        "ndcg@20": 1.0,
        "recall@50": 1.0,
        "ndcg@50": 1.0,
    }
    rows = [
        {**common, "model_id": "m1_baseline", "revenue@10": 1.0},
        {**common, "model_id": "m3_n_transfer", "revenue@10": 1.01},
        {
            **common,
            "model_id": "m3_v_contribution",
            "recall@20": 0.98,
            "revenue@10": 1.02,
        },
    ]
    decision = M3.screening_decision(pd.DataFrame(rows))

    assert decision["arms"]["m3_n_transfer"]["success"] is True
    assert decision["arms"]["m3_v_contribution"]["success"] is False
    assert decision["any_axis_success"] is True


def test_result_schema_exposes_exposure_entropy():
    frame = pd.DataFrame({"entropy@10": [3.0]})
    normalized = M3.normalize_result_schema(frame)
    assert normalized["exposure_entropy@10"].tolist() == [3.0]


def test_colab_is_pinned_and_runs_only_transfer_screening():
    notebook = json.loads(
        Path("clv_m3_transfer_dunnhumby_colab.ipynb").read_text(encoding="utf-8")
    )
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert "2cb09269cfdda9db237090b0b03c2439f318d79e" in source
    assert source.count("run_experiment(cfg)") == 1
    assert "EVAL_TEST" not in source
    assert "ACKNOWLEDGE_HIGH_COST" not in source
