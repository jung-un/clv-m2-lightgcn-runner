import numpy as np
import pytest

import lightgcn_clv_m5_value_basis_k1_assignment_controls as controls


def _cfg():
    return controls.configure_assignment_controls(
        out_dir="/tmp/controls", baseline_result_dir="/tmp/base"
    )


def test_controls_keep_the_single_negative_and_rho_of_the_screen():
    cfg = _cfg()
    assert cfg.negative_count == 1
    assert cfg.rho == 0.25
    with pytest.raises(ValueError, match="rho"):
        controls.configure_assignment_controls(
            out_dir="/tmp/controls", baseline_result_dir="/tmp/base", rho=0.05
        )


def test_only_the_m2_block_is_permuted_and_m4_keeps_the_observed_clv():
    prepared = {"m2_actual": {"name": "actual"}, "m2_shuffle": {"name": "shuffled"}}
    specs = controls.arm_specifications(prepared, _cfg())

    assert [spec["model_id"] for spec in specs] == list(controls.MODEL_IDS)
    assert [spec["m2_assignment"]["name"] for spec in specs] == [
        "actual",
        "shuffled",
        "actual",
        "shuffled",
    ]
    assert [spec["weighted"] for spec in specs] == [False, False, True, True]
    # the loss weight always reads the observed assignment
    assert all(spec["assignment"] is prepared for spec in specs)
    summary = controls.preflight_summary(_cfg())
    assert summary["control"]["m4_weight_assignment"] == "observed q_C in every arm"
    assert summary["reused_models"] == []


def _metrics(econ):
    return {
        "recall@10": 0.015,
        "ndcg@10": 0.018,
        "price_purchase_amount_weighted_hit@10": econ,
        "vndcg@10": econ / 40,
    }


def _reading(m2_actual, m5_actual, change=0.2):
    rows = {
        controls.M2_ACTUAL_MODEL_ID: _metrics(m2_actual),
        controls.M2_SHUFFLED_MODEL_ID: _metrics(0.385),
        controls.M5_ACTUAL_MODEL_ID: _metrics(m5_actual),
        controls.M5_SHUFFLED_MODEL_ID: _metrics(0.385),
    }
    return controls.attribution_reading(
        rows,
        top10_change_shares={
            f"{controls.M2_ACTUAL_MODEL_ID}_vs_shuffle": change,
            f"{controls.M5_ACTUAL_MODEL_ID}_vs_shuffle": change,
        },
        economic_score_ratios={model_id: 0.05 for model_id in controls.MODEL_IDS},
        band_pairs={},
    )


def test_attribution_requires_both_actual_arms_to_beat_their_control():
    assert _reading(0.390, 0.390)["classification"] == "clv_assignment_supported"
    assert _reading(0.390, 0.380)["classification"] == "partial_clv_assignment_signal"
    assert _reading(0.380, 0.380)["classification"] == "no_clv_assignment_signal"
    assert _reading(0.390, 0.390, change=0.0)["classification"] == (
        "not_evaluable_no_top10_change"
    )


def test_reading_reports_actual_minus_shuffled_deltas():
    reading = _reading(0.390, 0.392)
    assert np.isclose(
        reading["deltas_m2_actual_minus_shuffled"][
            "price_purchase_amount_weighted_hit@10"
        ],
        0.005,
    )
    assert np.isclose(
        reading["deltas_m5_actual_minus_shuffled"][
            "price_purchase_amount_weighted_hit@10"
        ],
        0.007,
    )
    assert reading["clv_assignment_supported"] is True


def test_prepare_rejects_a_permutation_that_breaks_an_invariant(monkeypatch):
    prepared = {
        "degree_bin": np.array([0, 0, 1, 1]),
        "q_n": np.array([0.1, 0.2, 0.6, 0.7], dtype=np.float32),
        "q_v": np.array([0.9, 0.8, 0.4, 0.3], dtype=np.float32),
        "q_c": np.array([0.2, 0.3, 0.7, 0.8], dtype=np.float32),
        "clv_valid": np.array([True, True, False, True]),
        "input_hash": "hash",
        "revision": "rev",
        "m2_actual": {},
    }
    monkeypatch.setattr(controls.screen, "_prepare", lambda cfg: prepared)
    monkeypatch.setattr(
        controls.controls,
        "degree_matched_nv_shuffle",
        lambda prepared, *, seed, degree_bins: {
            "q_n": prepared["q_n"],
            "q_v": prepared["q_v"],
            "q_c": np.array([0.2, 0.3, 0.7, 0.9], dtype=np.float32),
            "clv_valid": prepared["clv_valid"],
            "source_user": np.array([1, 0, 3, 2]),
        },
    )

    with pytest.raises(RuntimeError, match="순열 불변식"):
        controls._prepare(_cfg())
