from dataclasses import replace
import json
from pathlib import Path

import pandas as pd
import pytest

import lightgcn_clv_m4_hm2y_seed42 as hm4


def test_hm4_seed42_protocol_uses_one_seed_and_three_trained_arms():
    cfg = hm4.configure_hm2y_seed42_run()
    summary = hm4.preflight_summary(cfg)

    assert cfg.dataset == "hm"
    assert cfg.seed == 42
    assert cfg.window_days is None
    assert cfg.epochs == 100
    assert cfg.negative_count == 5
    assert cfg.eval_test is False
    assert cfg.eval_holdout is False
    assert summary["trained_models"] == [
        "m1_multineg_mean_k5",
        "m4_clv_hard_k5",
        "m4_clv_hard_k5_degree_shuffled",
    ]
    assert summary["baseline_source"] == "matching M2 rho=0 H&M seed-42 arm"
    assert summary["split"] == "hm2y_validation"


@pytest.mark.parametrize("field", ["eval_test", "eval_holdout"])
def test_hm4_rejects_protected_splits(field):
    cfg = hm4.configure_hm2y_seed42_run()
    with pytest.raises(ValueError):
        hm4.validate_hm2y_seed42_config(replace(cfg, **{field: True}))


def test_hm4_seed42_decision_requires_clv_assignment_and_m4_effect():
    rows = []
    for model_id, increment in (
        (hm4.K1_MODEL_ID, 0.0000),
        (hm4.MEAN_K5_MODEL_ID, 0.0010),
        (hm4.SHUFFLED_M4_MODEL_ID, 0.0020),
        (hm4.M4_MODEL_ID, 0.0030),
    ):
        row = {"model_id": model_id}
        for metric in hm4.ACCURACY_METRICS + hm4.PRIMARY_METRICS:
            row[metric] = 0.05 + increment
        rows.append(row)

    decision, paired = hm4.seed42_decision(pd.DataFrame(rows))

    assert decision["positive_screen"] is True
    assert set(paired.reference) == {
        hm4.MEAN_K5_MODEL_ID,
        hm4.SHUFFLED_M4_MODEL_ID,
    }
    assert paired["passes"].all()

    failing = pd.DataFrame(rows)
    failing.loc[failing.model_id == hm4.M4_MODEL_ID, "고CLV_recall@10"] = 0.01
    failed, _ = hm4.seed42_decision(failing)
    assert failed["positive_screen"] is False


def test_hm4_colab_runs_once_and_never_opens_protected_splits():
    notebook = json.loads(
        Path("clv_m4_clv_hard_negative_hm2y_seed42_colab.ipynb").read_text(
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
