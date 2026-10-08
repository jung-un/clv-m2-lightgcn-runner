import ast
import json
from pathlib import Path

import pandas as pd

import clv_m5_history_m2_m3_m4_user_centered_screen as screen


NOTEBOOK = Path("clv_m5_history_m2_m3_m4_user_centered_colab.ipynb")
SOURCE_COMMIT = "d1aa9642628aa92b90da3facdf5de9a4ab6ec33d"


def _synthetic_curve(*, full_economic_delta: float = 0.02) -> pd.DataFrame:
    rows = []
    models = (
        screen.m3.M1_MODEL_ID,
        screen.m3.ARM_VALUE_ACTIVITY,
        screen.M2_M3_MODEL_ID,
        screen.observed.MODEL_ID,
        screen.FULL_MODEL_ID,
    )
    for seed in screen.SEEDS:
        for epoch in (screen.DIAGNOSTIC_EPOCH, screen.FIXED_EPOCH):
            for model_id in models:
                values = {metric: 1.0 for metric in (*screen.ACCURACY, *screen.ECONOMIC)}
                if model_id == screen.m3.ARM_VALUE_ACTIVITY:
                    values = {metric: 1.02 for metric in values}
                elif model_id == screen.M2_M3_MODEL_ID:
                    values = {metric: 1.03 for metric in values}
                elif model_id == screen.observed.MODEL_ID:
                    values = {metric: 1.04 for metric in values}
                elif model_id == screen.FULL_MODEL_ID:
                    values = {metric: 1.04 for metric in values}
                    for metric in screen.ECONOMIC:
                        values[metric] += full_economic_delta
                rows.append({
                    "model_id": model_id,
                    "seed": seed,
                    "epoch": epoch,
                    **values,
                })
    return pd.DataFrame(rows)


def test_preflight_fixes_components_and_joint_training_contract():
    summary = screen.preflight_summary()

    assert summary["seeds"] == [42, 44]
    assert summary["fixed_m2"] == {"axis_dim": 4, "rho": 0.05}
    assert summary["fixed_m4_c"]["lambda"] == 0.5
    assert summary["new_fit_count"] == 4
    assert summary["training"]["joint_from_initialisation"]
    assert not summary["training"]["pretraining_or_freezing"]
    assert not summary["final_test_constructed"]
    assert not summary["holdout_constructed"]


def test_only_the_two_preregistered_development_seeds_are_allowed():
    assert screen.configure(42).seeds == (42,)
    assert screen.configure(44).seeds == (44,)

    try:
        screen.configure(43)
    except ValueError as exc:
        assert "seed" in str(exc)
    else:
        raise AssertionError("seed43 must remain outside the fixed two-seed gate")


def test_comparison_and_factorial_isolate_m2_increment_on_m3_m4_c():
    curve = _synthetic_curve()
    comparison = screen.comparison_table(curve)
    factorial = screen.factorial_table(curve)

    assert set(comparison.contrast) == {
        "M2+M3_minus_M3",
        "FULL_minus_M3+M4-C",
        "FULL_minus_M2+M3",
        "FULL_minus_M3",
        "FULL_minus_M1",
    }
    target = comparison[
        comparison.contrast.eq("FULL_minus_M3+M4-C")
        & comparison.epoch.eq(screen.FIXED_EPOCH)
        & comparison.metric.eq(screen.ECONOMIC[0])
    ]
    assert len(target) == 2
    assert (target.delta > 0).all()

    economic = factorial[factorial.metric.eq(screen.ECONOMIC[0])]
    assert len(economic) == 2
    assert (economic.m2_on_m3_m4_c > 0).all()


def test_strict_gate_requires_both_economic_metrics_in_both_seeds():
    curve = _synthetic_curve()
    comparison = screen.comparison_table(curve)
    passed = screen.reading(curve, comparison)
    assert passed["mean_screen_condition_met"]
    assert passed["strict_two_seed_condition_met"]
    assert passed["full_minus_m3_m4_c_positive_seed_count"] == {
        metric: 2 for metric in screen.ECONOMIC
    }

    failed_curve = curve.copy()
    mask = (
        failed_curve.model_id.eq(screen.FULL_MODEL_ID)
        & failed_curve.seed.eq(44)
        & failed_curve.epoch.eq(screen.FIXED_EPOCH)
    )
    failed_curve.loc[mask, screen.ECONOMIC[1]] = 1.00
    failed_comparison = screen.comparison_table(failed_curve)
    failed = screen.reading(failed_curve, failed_comparison)
    assert not failed["full_both_economic_at10_above_m3_m4_c_mean"]
    assert not failed["strict_two_seed_condition_met"]


def test_colab_is_valid_and_pins_the_reviewed_source_commit():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    ast.parse(code)
    assert f"SOURCE_COMMIT = '{SOURCE_COMMIT}'" in code
    assert "result = screen.run()" in code
    assert "final_test" not in code
    assert "holdout" not in code
