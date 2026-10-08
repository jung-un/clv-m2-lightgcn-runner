import json

import pandas as pd
import pytest

import clv_m2_axis_rho_seed42_screen as screen
import lightgcn_clv_m2_capacity_search as search


def _cfg(**overrides):
    values = {"out_dir": "/tmp/axis-rho-screen", **overrides}
    return screen.validate_config(screen.Config(**values))


def test_preflight_fixes_two_isolated_arms_in_execution_order():
    cfg = _cfg()
    summary = screen.preflight_summary(cfg)
    specs = screen._new_specs(screen._search_config(cfg))

    assert summary["new_training_fits"] == 2
    assert summary["final_test_constructed"] is False
    assert [spec["condition"] for spec in specs] == ["axis_wide", "weak_signal"]
    assert [(spec["axis_dim"], spec["rho"]) for spec in specs] == [(8, .05), (4, .025)]
    assert all(spec["id_dim"] == 64 and spec["pref_reg"] == .001 for spec in specs)


def test_config_refuses_parameter_drift():
    with pytest.raises(ValueError):
        _cfg(seed=43)
    with pytest.raises(ValueError):
        _cfg(epochs=100)
    with pytest.raises(ValueError):
        _cfg(conditions=("baseline", "axis_wide"))


def test_reference_audit_requires_exact_completed_baseline():
    spec = {
        "condition": "baseline",
        "model_id": search.M2_MODEL_ID,
        "shared_key": "dim64_l20.001",
        "id_dim": 64,
        "axis_dim": 4,
        "pref_reg": .001,
        "rho": .05,
    }
    payload = {
        **spec,
        "seed": 42,
        "code_version": search.CODE_VERSION,
        "source_revision": "rev",
        "curve": [{"epoch": 100}, {"epoch": 300}],
    }
    screen._audit_arm_payload(payload, spec)

    bad = json.loads(json.dumps(payload))
    bad["rho"] = .025
    with pytest.raises(RuntimeError, match="rho"):
        screen._audit_arm_payload(bad, spec)
    bad = json.loads(json.dumps(payload))
    bad["curve"][-1]["epoch"] = 299
    with pytest.raises(RuntimeError, match="300 epoch"):
        screen._audit_arm_payload(bad, spec)


def _row(condition, model_id, axis_dim, rho, values):
    return {
        "condition": condition,
        "hypothesis": "x",
        "shared_key": "dim64_l20.001",
        "model_id": model_id,
        "id_dim": 64,
        "axis_dim": axis_dim,
        "pref_reg": .001,
        "rho": rho,
        "seed": 42,
        "epoch": 300,
        **values,
    }


def test_fixed_epoch_tables_compare_each_candidate_to_m1_and_axis4():
    metrics = {
        "recall@10": 1.0,
        "ndcg@10": 1.0,
        "recall@50": 1.0,
        "price_purchase_amount_weighted_hit@10": 1.0,
        "price_purchase_amount_weighted_hit@50": 1.0,
        "vndcg@10": 1.0,
    }
    better = {key: value * 1.02 for key, value in metrics.items()}
    curve = pd.DataFrame(
        [
            _row("baseline", search.M1_MODEL_ID, 0, 0.0, metrics),
            _row("baseline", search.M2_MODEL_ID, 4, .05, metrics),
            _row("axis_wide", search.M2_MODEL_ID, 8, .05, better),
            _row("weak_signal", search.M2_MODEL_ID, 4, .025, better),
        ]
    )

    absolute, comparison, reading = screen.fixed_epoch_tables(curve)

    assert len(absolute) == 4
    assert set(comparison["reference"]) == {"m1", "axis4_rho005"}
    assert set(comparison["condition"]) == {"axis_wide", "weak_signal"}
    assert reading["single_seed_screen_condition_met"].all()
    assert not reading["condition_selected"].any()
