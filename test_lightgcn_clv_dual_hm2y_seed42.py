import dataclasses
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def test_hm2y_config_is_full_period_seed42_validation_only(tmp_path):
    import lightgcn_clv_dual_hm2y_seed42 as hm2y

    cfg = hm2y.configure_hm2y_seed42(
        out_dir=tmp_path / "out",
        m1_checkpoint_dir=tmp_path / "m1",
    )

    assert cfg.dataset == "hm"
    assert cfg.window_days is None
    assert cfg.seed_list == (42,)
    assert cfg.eval_test is cfg.eval_holdout is False
    assert hm2y.GATE_SHAPE == "high"
    assert hm2y.TARGET_RHO == pytest.approx(0.2)


@pytest.mark.parametrize(
    "changes",
    [
        {"dataset": "dunnhumby"},
        {"window_days": 60},
        {"seed_list": (43,)},
        {"eval_test": True},
        {"eval_holdout": True},
    ],
)
def test_hm2y_config_rejects_protocol_changes(changes, tmp_path):
    import lightgcn_clv_dual_hm2y_seed42 as hm2y

    cfg = hm2y.configure_hm2y_seed42(
        out_dir=tmp_path / "out",
        m1_checkpoint_dir=tmp_path / "m1",
    )
    with pytest.raises(ValueError):
        hm2y.validate_hm2y_config(dataclasses.replace(cfg, **changes))


def test_operating_point_uses_frozen_rho_not_validation_metric():
    import lightgcn_clv_dual_hm2y_seed42 as hm2y

    point = hm2y.operating_point(0.4)

    assert point == {
        "gate_shape": "high",
        "rho": pytest.approx(0.2),
        "raw_effective_ratio": pytest.approx(0.4),
        "lambda": pytest.approx(0.5),
        "effective_strength": pytest.approx(0.2),
    }
    with pytest.raises(ValueError):
        hm2y.operating_point(0.0)


def test_seed42_decision_requires_six_accuracy_guards_and_revenue_gain():
    import lightgcn_clv_dual_hm2y_seed42 as hm2y

    baseline = {
        **{f"{metric}@{k}": 1.0 for metric in ("recall", "ndcg") for k in (10, 20, 50)},
        "revenue@10": 2.0,
    }
    model = dict(baseline, **{"revenue@10": 2.1})
    assert hm2y.seed42_decision(baseline, model)["success"] is True

    failed = dict(model, **{"recall@50": 0.98})
    decision = hm2y.seed42_decision(baseline, failed)
    assert decision["success"] is False
    assert "six_accuracy_ratios_at_least_0.99" in decision["failed_conditions"]


def test_hm2y_colab_runs_only_frozen_seed42_main_arm():
    notebook = json.loads(Path("clv_dual_hm2y_seed42_colab.ipynb").read_text())
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )

    assert re.search(r"REVIEWED_SHA = '[0-9a-f]{40}'", source)
    assert "configure_hm2y_seed42" in source
    assert "run_hm2y_seed42" in source
    assert "TARGET_RHO" in source
    assert "torch.cuda.is_available()" in source
    for forbidden in (
        "short_hm=True",
        "eval_test=True",
        "eval_holdout=True",
        "dual_shuffled_user",
        "dual_adapter_only",
        "rho_grid",
        "lambda_eval=",
    ):
        assert forbidden not in source


def test_effective_ratio_and_lambda_are_finite():
    import lightgcn_clv_dual_hm2y_seed42 as hm2y

    for ratio in (0.1, 0.4, 2.0):
        point = hm2y.operating_point(ratio)
        assert np.isfinite(point["lambda"])
        assert point["lambda"] > 0


def test_runner_trains_only_main_and_evaluates_one_normalized_point(
    monkeypatch, tmp_path
):
    import lightgcn_clv_dual_hm2y_seed42 as hm2y

    events = {}

    class FakeModel:
        def set_eval_axes(self, value):
            events["axes"] = value

        def set_gate_shape(self, value):
            events["gate_shape"] = value

    prepared = {
        "baseline_flat": {"revenue@10": 1.0},
        "cache": object(),
        "meta": object(),
        "data": object(),
        "base_cfg": object(),
    }
    monkeypatch.setattr(hm2y.dual, "_prepare", lambda *_args, **_kwargs: prepared)
    monkeypatch.setattr(hm2y.dual, "_fresh_base", lambda *_args, **_kwargs: object())

    def fake_train(model_id, *_args, **kwargs):
        events["trained"] = model_id
        events["train_gate_shapes"] = kwargs["gate_shapes"]
        events["train_lambda_eval"] = kwargs["lambda_eval"]
        return {
            "model": FakeModel(),
            "diagnostics": {
                "gate_shape_diagnostics": {
                    "high": {"effective_total_ratio": 0.4}
                }
            },
        }

    monkeypatch.setattr(hm2y.dual, "_train_variant", fake_train)

    def fake_eval(_model, lam, *_args, **_kwargs):
        events["eval_lambda"] = lam
        return {"revenue@10": 1.1}, {"revenue": np.array([0.1])}

    monkeypatch.setattr(hm2y.moe, "_flat_evaluation", fake_eval)

    def fake_persist(_cfg, _prepared, _run, point, *_args):
        events["point"] = point
        frame = pd.DataFrame()
        frame.attrs["decision"] = {"success": True}
        frame.attrs["result_paths"] = {}
        return frame

    monkeypatch.setattr(hm2y, "_persist", fake_persist)
    cfg = hm2y.configure_hm2y_seed42(
        out_dir=tmp_path / "out", m1_checkpoint_dir=tmp_path / "m1"
    )

    hm2y.run_hm2y_seed42(cfg)

    assert events["trained"] == "dual_clv_fixed"
    assert events["train_gate_shapes"] == ("high",)
    assert events["train_lambda_eval"] == ()
    assert events["axes"] == "n_plus_v"
    assert events["gate_shape"] == "high"
    assert events["eval_lambda"] == pytest.approx(0.5)
    assert events["point"]["effective_strength"] == pytest.approx(0.2)
