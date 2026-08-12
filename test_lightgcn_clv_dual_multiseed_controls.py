import dataclasses

import pandas as pd
import pytest


def _comparison_rows():
    rows = []
    values = {
        42: (1.10, 1.00, 0.98),
        43: (1.08, 1.01, 1.02),
        44: (1.03, 1.04, 1.01),
    }
    for seed, (main, shuffled, adapter) in values.items():
        for model_id, revenue in (
            ("dual_clv_fixed", main),
            ("dual_shuffled_user", shuffled),
            ("dual_adapter_only", adapter),
        ):
            rows.append(
                {
                    "seed": seed,
                    "model_id": model_id,
                    "gate_shape": "equal",
                    "lambda": 2.0,
                    "revenue@10": revenue,
                }
            )
    return pd.DataFrame(rows)


def test_control_presets_run_only_frozen_validation_controls(tmp_path):
    import lightgcn_clv_dual_multiseed_controls as controls

    dun = controls.configure_multiseed_controls(
        "dunnhumby",
        tmp_path / "seed42.json",
        tmp_path / "multiseed.json",
    )
    hm = controls.configure_multiseed_controls(
        "hm",
        tmp_path / "seed42-hm.json",
        tmp_path / "multiseed-hm.json",
        short_hm=True,
    )

    assert (dun.gate_shape, dun.fixed_lambda) == ("equal", 2.0)
    assert (hm.window_days, hm.gate_shape, hm.fixed_lambda) == (60, "high", 1.0)
    assert dun.new_seeds == hm.new_seeds == (43, 44)
    assert dun.control_ids == hm.control_ids == (
        "dual_shuffled_user",
        "dual_adapter_only",
    )
    assert dun.eval_test is hm.eval_test is False
    assert dun.eval_holdout is hm.eval_holdout is False


@pytest.mark.parametrize(
    "changes",
    [
        {"new_seeds": (42, 43)},
        {"eval_test": True},
        {"eval_holdout": True},
        {"control_ids": ("dual_shuffled_user",)},
        {"fixed_lambda": 4.0},
    ],
)
def test_control_config_rejects_protocol_changes(changes, tmp_path):
    import lightgcn_clv_dual_multiseed_controls as controls

    cfg = controls.configure_multiseed_controls(
        "dunnhumby",
        tmp_path / "seed42.json",
        tmp_path / "multiseed.json",
    )
    with pytest.raises(ValueError):
        controls.validate_control_config(dataclasses.replace(cfg, **changes))


def test_control_decision_requires_main_to_beat_each_control_in_mean_and_two_seeds():
    from lightgcn_clv_dual_multiseed_controls import control_reproducibility_decision

    decision = control_reproducibility_decision(_comparison_rows())

    assert decision["success"] is True
    assert decision["comparisons"]["dual_shuffled_user"]["positive_seed_count"] == 2
    assert decision["comparisons"]["dual_adapter_only"]["positive_seed_count"] == 3
    assert decision["comparisons"]["dual_shuffled_user"]["mean_delta"] > 0


def test_control_decision_fails_when_capacity_control_explains_mean_gain():
    import lightgcn_clv_dual_multiseed_controls as controls

    rows = _comparison_rows()
    rows.loc[rows.model_id.eq("dual_adapter_only"), "revenue@10"] = 1.2
    decision = controls.control_reproducibility_decision(rows)

    assert decision["success"] is False
    assert "dual_adapter_only" in decision["failed_controls"]


def test_new_seed_trains_only_two_controls_from_reused_prepared_context(monkeypatch):
    import lightgcn_clv_dual_multiseed_controls as controls

    cfg = controls.ControlValidationConfig(
        dataset="dunnhumby",
        seed42_result_json="seed42.json",
        multiseed_result_json="multiseed.json",
        window_days=None,
        gate_shape="equal",
        fixed_lambda=2.0,
        out_dir="out",
    )
    prepared = {
        "baseline_hash": "base",
        "cache": type("Cache", (), {"users": [1, 2]})(),
        "baseline_per_user": {"revenue": [1.0, 1.0]},
        "m1_checkpoint": "m1.pt",
        "encoder_checkpoint": "encoder.pt",
        "manifest": {},
    }
    calls = []

    monkeypatch.setattr(controls, "_load_reusable_prepared", lambda *_: prepared)
    monkeypatch.setattr(controls.dual, "_fresh_base", lambda *_args, **_kwargs: object())

    def fake_train(model_id, *_args, **_kwargs):
        calls.append(model_id)
        return {
            "rows": [{"seed": 43, "model_id": model_id, "revenue@10": 1.0}],
            "per_user": {("equal", 2.0): {"revenue": [1.0, 1.0]}},
            "checkpoint": f"{model_id}.pt",
            "training": {},
        }

    monkeypatch.setattr(controls.dual, "_train_variant", fake_train)
    result = controls._run_control_seed(cfg, {"config": {}}, {}, 43)

    assert calls == ["dual_shuffled_user", "dual_adapter_only"]
    assert [row["model_id"] for row in result["rows"]] == list(cfg.control_ids)

