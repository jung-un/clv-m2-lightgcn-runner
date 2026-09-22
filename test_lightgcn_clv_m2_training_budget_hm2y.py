import dataclasses
import json
from pathlib import Path

import pandas as pd
import pytest

import lightgcn_clv_m2_training_budget_hm2y as budget


def test_hm2y_training_budget_contract(tmp_path):
    cfg = budget.configure_hm2y_training_budget(out_dir=str(tmp_path))
    summary = budget.preflight_summary(cfg)
    assert cfg.seed == 43
    assert cfg.epochs == 300
    assert cfg.evaluation_epochs == (100, 200, 300)
    assert summary["fixed"]["test_constructed"] is False
    assert summary["fixed"]["holdout_constructed"] is False
    assert summary["fixed"]["negative_sampling"] == "one uniform unseen item"


@pytest.mark.parametrize(("field", "value"), [
    ("seed", 42), ("epochs", 200), ("evaluation_epochs", (100, 300)),
    ("negative_count", 5), ("eval_test", True), ("eval_holdout", True),
    ("history_rho", 0.1),
])
def test_hm2y_training_budget_rejects_protocol_drift(tmp_path, field, value):
    cfg = budget.configure_hm2y_training_budget(out_dir=str(tmp_path))
    with pytest.raises(ValueError):
        budget.validate_config(dataclasses.replace(cfg, **{field: value}))


def test_gap_table_pairs_same_epoch_and_keeps_high_clv_metrics():
    curve = pd.DataFrame([
        {"model_id": budget.M1_MODEL_ID, "seed": 43, "epoch": 100,
         "recall@10": 0.1, "고CLV_price_purchase_amount_weighted_hit@10": 0.2},
        {"model_id": budget.M2_MODEL_ID, "seed": 43, "epoch": 100,
         "recall@10": 0.11, "고CLV_price_purchase_amount_weighted_hit@10": 0.22},
    ])
    gap = budget.gap_table(curve)
    assert gap.loc[0, "recall@10"] == pytest.approx(0.01)
    assert gap.loc[0, "recall@10_ratio"] == pytest.approx(1.1)
    assert gap.loc[0, "고CLV_price_purchase_amount_weighted_hit@10"] == pytest.approx(0.02)


def test_hm2y_colab_is_pinned_and_runs_once():
    notebook = json.loads(Path("clv_m2_training_budget_hm2y_seed43_colab.ipynb").read_text())
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"]
                       if cell.get("cell_type") == "code")
    assert "'checkout', SOURCE_COMMIT" in source
    assert "configure_hm2y_training_budget" in source
    assert source.count("run_hm2y_training_budget(cfg)") == 1
    assert "EVAL_TEST" not in source
    assert "EVAL_HOLDOUT" not in source
