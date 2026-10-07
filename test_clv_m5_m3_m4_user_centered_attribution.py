import numpy as np
import pandas as pd

import clv_m5_m3_m4_user_centered_attribution as screen


def _prepared():
    return {
        "data": {"n_users": 8},
        "signals": {"edge_users": np.array([0, 0, 1, 2, 2, 2, 3, 4, 5, 6, 7])},
        "q_n": np.arange(8, dtype=np.float32) / 10,
        "q_v": np.arange(8, dtype=np.float32)[::-1] / 10,
        "q_c": np.arange(8, dtype=np.float32) / 8,
        "clv_valid": np.array([True, True, False, True, True, False, True, True]),
    }


def test_joint_shuffle_preserves_tuple_multisets_and_degree_strata():
    prepared = _prepared()
    assignment, audit = screen.degree_matched_tuple_shuffle(prepared, seed=7, bins=2)

    assert audit["moved_user_share"] > 0
    assert audit["same_degree_bin"]
    for name in ("q_n", "q_v", "q_c", "clv_valid"):
        assert np.array_equal(np.sort(assignment[name]), np.sort(prepared[name]))
    source = assignment["source_user"]
    assert np.array_equal(assignment["q_n"], prepared["q_n"][source])
    assert np.array_equal(assignment["q_v"], prepared["q_v"][source])
    assert np.array_equal(assignment["q_c"], prepared["q_c"][source])


def test_shuffled_prepared_changes_only_assignment_fields_and_derived_axes():
    prepared = _prepared() | {
        "user_bin_fit": np.arange(24).reshape(8, 3),
        "config_hash": "same",
    }
    assignment, _ = screen.degree_matched_tuple_shuffle(prepared, seed=9, bins=2)
    shuffled = screen._prepared_with_assignment(prepared, assignment)

    assert shuffled["signals"] is prepared["signals"]
    assert shuffled["user_bin_fit"] is prepared["user_bin_fit"]
    assert np.array_equal(shuffled["q_value"], np.where(assignment["clv_valid"], assignment["q_v"], 0))
    assert np.array_equal(shuffled["q_activity"], np.where(assignment["clv_valid"], assignment["q_n"], 0))


def test_reading_requires_accuracy_economics_and_assignment_attribution():
    models = (
        screen.m3.M1_MODEL_ID,
        screen.m3.ARM_VALUE_ACTIVITY,
        screen.M4_C_MODEL_ID,
        screen.observed.MODEL_ID,
        screen.M5_C_SHUFFLED_MODEL_ID,
    )
    metrics = (*screen.ACCURACY, *screen.ECONOMIC)
    rows = []
    for seed in screen.SEEDS:
        for model in models:
            values = {metric: 1.0 for metric in metrics}
            if model == screen.observed.MODEL_ID:
                values.update({metric: 1.02 for metric in screen.ECONOMIC})
            elif model == screen.M5_C_SHUFFLED_MODEL_ID:
                values.update({metric: 1.01 for metric in screen.ECONOMIC})
            rows.append({"model_id": model, "seed": seed, "epoch": 300, **values})
    curve = pd.DataFrame(rows)
    comparison = screen.comparison_table(pd.concat([
        curve.assign(epoch=100), curve
    ], ignore_index=True))

    result = screen.reading(curve, comparison)

    assert result["m5_c_accuracy_guard_99pct_vs_m1_mean"]
    assert result["m5_c_economic_at10_above_m1_and_m3_mean"]
    assert result["observed_assignment_beats_degree_matched_shuffle_on_both_economic_at10_mean"]
    assert result["ready_for_preregistered_final_10seed"]


def test_arm_specs_are_exactly_the_two_missing_fits():
    specs = screen._arm_specs()
    assert [spec["model_id"] for spec in specs] == [
        screen.M4_C_MODEL_ID,
        screen.M5_C_SHUFFLED_MODEL_ID,
    ]
    assert screen.configure(42).epochs == 300
    assert screen.configure(44).epochs == 300
