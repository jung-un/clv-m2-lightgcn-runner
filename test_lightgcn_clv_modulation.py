import json
from dataclasses import replace
from pathlib import Path

import pytest

import lightgcn_clv_modulation as modulation


def test_dunnhumby_preset_is_one_space_validation_only():
    cfg = modulation.configure_modulation_dunnhumby_run()
    summary = modulation.preflight_summary(cfg)

    assert summary["dataset"] == "dunnhumby"
    assert summary["seed"] == 42
    assert summary["models"] == ["m1", "m2_clv_modulation"]
    assert summary["architecture"] == (
        "CLV N/V-conditioned modulation -> one 64d LightGCN -> one dot score"
    )
    assert summary["tau"] == pytest.approx(0.10)
    assert summary["modulation_rank"] == 4
    assert summary["final_embedding_dim"] == 64
    assert summary["eval_test"] is False
    assert summary["eval_holdout"] is False
    assert summary["loss"] == "plain_bpr"


@pytest.mark.parametrize("field", ["eval_test", "eval_holdout"])
def test_public_runner_rejects_protected_splits_before_prepare(monkeypatch, field):
    cfg = replace(modulation.configure_modulation_dunnhumby_run(), **{field: True})
    monkeypatch.setattr(
        modulation, "_prepare", lambda _: pytest.fail("must fail before prepare")
    )

    with pytest.raises(ValueError, match="validation-only"):
        modulation.run_experiment(cfg)


def test_screening_decision_requires_economic_gain_and_all_accuracy_guards():
    baseline = {
        "recall@10": 1.0,
        "ndcg@10": 1.0,
        "recall@20": 1.0,
        "ndcg@20": 1.0,
        "recall@50": 1.0,
        "ndcg@50": 1.0,
        "revenue@10": 1.0,
    }
    model = baseline | {"recall@20": 0.99, "revenue@10": 1.01}

    assert modulation.screening_decision(model, baseline)["success"] is True
    assert modulation.screening_decision(
        model | {"recall@20": 0.989}, baseline
    )["success"] is False
    assert modulation.screening_decision(
        model | {"revenue@10": 1.0}, baseline
    )["success"] is False


def test_colab_is_pinned_and_runs_modulation_once():
    path = Path(__file__).with_name(
        "clv_conditioned_modulation_dunnhumby_colab.ipynb"
    )
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert "TO_BE_PINNED_AFTER_SOURCE_COMMIT" not in source
    assert "configure_modulation_dunnhumby_run" in source
    assert source.count("result_df = run_experiment(cfg)") == 1
    assert "eval_test=False" in source
    assert "eval_holdout=False" in source
