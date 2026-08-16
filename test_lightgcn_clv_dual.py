import dataclasses
import inspect
import json
import re
from types import SimpleNamespace
from pathlib import Path

import pytest


def test_train_variant_uses_requested_seed_and_fixed_point(monkeypatch, tmp_path):
    import lightgcn_clv_dual as dual

    calls = {}

    class FakeModel:
        q_n = q_v = has_profile = None

        def to(self, _device):
            return self

        def set_gate_shape(self, shape):
            calls.setdefault("shapes", []).append(shape)

        def state_dict(self):
            return {"weight": 1}

    def fake_constructor(*_args, seed, **_kwargs):
        calls["constructor_seed"] = seed
        return FakeModel()

    def fake_train(*_args, **kwargs):
        calls["train_seed"] = _args[4]
        assert kwargs["freeze_base"] is True
        return {"best_epoch": 1}

    monkeypatch.setattr(dual, "CLVDualAxisEmbeddingModel", fake_constructor)
    monkeypatch.setattr(dual.moe, "train_moe", fake_train)
    monkeypatch.setattr(
        dual,
        "_diagnostics",
        lambda *_: {
            "gate_shape_diagnostics": {
                "equal": {"effective_total_ratio": 0.5}
            }
        },
    )
    monkeypatch.setattr(
        dual.moe,
        "_flat_evaluation",
        lambda *_args, **_kwargs: ({"recall@10": 0.1}, {"recall": [0.1]}),
    )
    monkeypatch.setattr(dual.torch, "save", lambda _blob, path: calls.setdefault("path", path))

    prepared = {
        "user_profile": SimpleNamespace(feature_names=("u",)),
        "item_profile": SimpleNamespace(
            activity_names=("n",), value_names=("v",)
        ),
        "q_n": [0.5],
        "q_v": [0.5],
        "data": {},
        "base_cfg": {"K_LIST": [10]},
        "cache": object(),
        "meta": {},
        "artifact": object(),
        "out_dir": tmp_path,
        "fingerprint": "fixture",
        "revision": "revision",
        "baseline_hash": "baseline",
    }
    cfg = SimpleNamespace(
        dataset="dunnhumby",
        expert_hidden_dim=4,
        expert_dim=2,
        lambda_train=1.0,
        lambda_eval=(0.0, 1.0),
    )

    result = dual._train_variant(
        dual.PRIMARY_MODEL,
        object(),
        prepared,
        cfg,
        seed=43,
        gate_shapes=("equal",),
        lambda_eval=(2.0,),
    )

    assert calls["constructor_seed"] == 43
    assert calls["train_seed"] == 43
    assert calls["shapes"] == ["equal"]
    assert "_s43_" in str(calls["path"])
    assert [(row["seed"], row["gate_shape"], row["lambda"]) for row in result["rows"]] == [
        (43, "equal", 2.0)
    ]


def test_prepare_and_fresh_base_accept_requested_seed():
    import lightgcn_clv_dual as dual

    assert inspect.signature(dual._prepare).parameters["seed"].default == 42
    assert inspect.signature(dual._fresh_base).parameters["seed"].default == 42
    prepare_source = inspect.getsource(dual._prepare)
    fresh_source = inspect.getsource(dual._fresh_base)
    assert "seed=seed" in prepare_source
    assert "s{seed}" in prepare_source
    assert "'pref_only', seed" in prepare_source
    assert "prepared[\"base_context\"], seed" in fresh_source


def test_dual_runner_is_seed42_validation_only_and_has_four_models():
    import lightgcn_clv_dual as dual

    cfg = dual.configure_dual_run("hm", short_hm=True)
    summary = dual.preflight_summary(cfg)
    assert summary["seed_list"] == [42]
    assert summary["window_days"] == 60
    assert summary["eval_test"] is False
    assert summary["eval_holdout"] is False
    assert summary["models"] == [
        "m1",
        "dual_clv_fixed",
        "dual_shuffled_user",
        "dual_adapter_only",
    ]
    assert summary["gate_shapes"] == ["high", "equal", "low"]
    assert summary["lambda_eval"][-2:] == [4.0, 8.0]


def test_dual_runner_rejects_protected_splits_and_extra_seeds_before_data():
    import lightgcn_clv_dual as dual

    cfg = dual.configure_dual_run("dunnhumby")
    with pytest.raises(ValueError, match="validation-only"):
        dual.validate_dual_config(dataclasses.replace(cfg, eval_test=True))
    with pytest.raises(ValueError, match="seed 42"):
        dual.validate_dual_config(dataclasses.replace(cfg, seed_list=(42, 43)))


def test_screening_decision_uses_same_operating_points_not_control_argmax():
    import lightgcn_clv_dual as dual

    baseline = {
        f"{metric}@{k}": 1.0
        for metric in ("recall", "ndcg")
        for k in (10, 20, 50)
    } | {"revenue@10": 1.0}
    rows = []
    for model_id, revenues, strengths in (
        ("dual_clv_fixed", (1.08, 1.10), (0.2, 0.4)),
        ("dual_shuffled_user", (1.05, 1.09), (0.22, 0.41)),
        ("dual_adapter_only", (1.04, 1.11), (0.19, 0.39)),
    ):
        for lam, revenue, strength in zip((0.5, 1.0), revenues, strengths):
            rows.append(
                {
                    "model_id": model_id,
                    "gate_shape": "high",
                    "lambda": lam,
                    "revenue@10": revenue,
                    "effective_strength": strength,
                    **{key: 1.0 for key in baseline if key != "revenue@10"},
                }
            )
    selected, success, table = dual.select_primary_operating_point(rows, baseline)
    assert selected["gate_shape"] == "high" and selected["lambda"] == 1.0
    decision = dual.screening_decision(rows, selected, success, table)
    assert decision["success"] is False
    assert "dual_adapter_only" in decision["failed_controls"]
    assert decision["comparisons"]["dual_adapter_only"]["same_lambda_revenue"] == 1.11


def test_axis_preflight_reports_degenerate_transaction_target_and_axis_correlation():
    import lightgcn_clv_dual as dual

    diagnostics = dual.axis_preflight_diagnostics(
        transaction_targets=[
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        n_hat=[1.0, 1.0, 2.0, 2.0],
        v_hat=[4.0, 3.0, 2.0, 1.0],
        q_n=[0.25, 0.25, 0.75, 0.75],
        q_v=[0.875, 0.625, 0.375, 0.125],
        valid=[True, True, True, True],
        user_repeat_gap_valid=[False, False, True, False],
        item_repeat_gap_valid=[True, False, False],
    )
    assert diagnostics["future_transactions_ge2_share"] == 0.0
    assert diagnostics["q_n_unique_count"] == 2
    assert diagnostics["q_n_max_tie_share"] == 0.5
    assert diagnostics["n_axis_warning"] is True
    assert diagnostics["user_repeat_gap_valid_share"] == 0.25


def test_colab_runs_both_datasets_sequentially_with_four_models():
    notebook = json.loads(Path("clv_dual_axis_colab.ipynb").read_text())
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert "DATASET_PRESETS = ('dunnhumby', 'hm_w60')" in source
    assert "for dataset_preset in DATASET_PRESETS" in source
    assert "results_by_dataset" in source
    assert "ACKNOWLEDGE_HIGH_COST" not in source
    reviewed = re.search(r"REVIEWED_SHA = '([0-9a-f]{40})'", source)
    assert reviewed is not None
    assert reviewed.group(1) == "7049d363137b66f190603dcef83a529585d5ff5c"
    for model_id in (
        "dual_clv_fixed",
        "dual_shuffled_user",
        "dual_adapter_only",
    ):
        assert model_id in source
