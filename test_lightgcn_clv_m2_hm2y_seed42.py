from dataclasses import replace
import json
from pathlib import Path

import pandas as pd
import pytest

import lightgcn_clv_m2_hm2y_seed42 as hm2


def test_hm2_seed42_protocol_keeps_test_and_holdout_closed():
    cfg = hm2.configure_hm2y_seed42_run()
    summary = hm2.preflight_summary(cfg)

    assert cfg.dataset == "hm"
    assert cfg.seed == 42
    assert cfg.window_days is None
    assert cfg.epochs == 100
    assert cfg.batch_size == 131_072
    assert cfg.eval_test is False
    assert cfg.eval_holdout is False
    assert summary["models"] == [
        "m1_matched_rho0",
        "m2_clv_level_composition_price_embedding",
        "m2_degree_matched_clv_shuffle",
        "m2_jointly_trained_id_only",
    ]
    assert summary["split"] == "hm2y_validation"
    assert summary["fixed"]["graph"] == "binary"
    assert summary["fixed"]["negative_sampling"] == "uniform"
    assert summary["fixed"]["sample_weighting"] is False


@pytest.mark.parametrize("field", ["eval_test", "eval_holdout"])
def test_hm2_rejects_protected_splits(field):
    cfg = hm2.configure_hm2y_seed42_run()
    with pytest.raises(ValueError):
        hm2.validate_hm2y_seed42_config(replace(cfg, **{field: True}))


def test_hm2_seed42_decision_requires_actual_clv_to_beat_all_controls():
    rows = []
    for model_id, increment in (
        (hm2.MATCHED_MODEL_ID, 0.0000),
        (hm2.ID_ONLY_MODEL_ID, 0.0010),
        (hm2.SHUFFLED_MODEL_ID, 0.0020),
        (hm2.MODEL_ID, 0.0030),
    ):
        row = {"model_id": model_id}
        for metric in hm2.ACCURACY_METRICS + hm2.PRIMARY_METRICS:
            row[metric] = 0.05 + increment
        rows.append(row)

    decision, paired = hm2.seed42_decision(pd.DataFrame(rows))

    assert decision["positive_screen"] is True
    assert paired["passes"].all()
    assert decision["statistical_note"].startswith("H&M 2-year seed 42")

    failing = pd.DataFrame(rows)
    failing.loc[failing.model_id == hm2.MODEL_ID, "고CLV_ndcg@10"] = 0.01
    failed, _ = hm2.seed42_decision(failing)
    assert failed["positive_screen"] is False


def test_hm2_colab_runs_once_and_never_opens_protected_splits():
    notebook = json.loads(
        Path("clv_m2_level_composition_price_hm2y_seed42_colab.ipynb").read_text(
            encoding="utf-8"
        )
    )
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )

    assert source.count("result_df = run_hm2y_seed42(cfg)") == 1
    assert "configure_hm2y_seed42_run" in source
    assert "EVAL_TEST=True" not in source
    assert "EVAL_HOLDOUT=True" not in source
    assert "range(42, 52)" not in source
