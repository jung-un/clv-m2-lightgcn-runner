import numpy as np
import pandas as pd
import pytest

import lightgcn_clv_m5_k1_m4_improvement_screen as improvement


def _cfg():
    return improvement.configure_improvement_screen(
        out_dir="/tmp/improve", baseline_result_dir="/tmp/base"
    )


def _prepared():
    train = pd.DataFrame(
        {
            "u_idx": [0, 0, 0, 1],
            "i_idx": [0, 0, 1, 1],
            "t": [1, 2, 2, 1],
        }
    )
    return {
        "data": {
            "train": train,
            "tr_u": train.u_idx.to_numpy(np.int64),
            "tr_i": train.i_idx.to_numpy(np.int64),
        },
        "q_c": np.array([1.0, 0.0], dtype=np.float32),
        "q_v": np.array([0.9, 0.1], dtype=np.float32),
        "clv_valid": np.array([True, True]),
        "item_amount_percentile": np.array([0.8, 0.2], dtype=np.float64),
        "item_economic_valid": np.array([True, True]),
        "item_bin": np.array([3, 0], dtype=np.int64),
        "user_bin_fit": np.array([[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0]]),
    }


def test_first_purchase_flags_mark_only_the_first_row_of_each_pair():
    flags = improvement.first_purchase_row_flags(_prepared()["data"]["train"])

    np.testing.assert_array_equal(flags, [True, False, True, True])


def test_single_negative_and_no_hard_negative_in_the_contract():
    summary = improvement.preflight_summary(_cfg())

    assert summary["loss"]["negative_count"] == 1
    assert summary["loss"]["hard_negative"] is False
    assert summary["reused_models"] == []
    assert summary["trained_models"] == list(improvement.MODEL_IDS)


def test_first_purchase_weight_leaves_repeat_rows_at_the_base_level():
    weights, diagnostics = improvement.row_weights(
        _prepared(), _cfg(), "first_purchase"
    )

    assert diagnostics["weight_mode"] == "first_purchase"
    assert np.isclose(diagnostics["weighted_row_share"], 0.75)
    # user 0 buys item 0 twice: only the first row carries the emphasis
    assert weights[0] > weights[1]
    # user 1 has q_C = 0, so its row stays at the base level like the repeat row
    assert np.isclose(weights[1], weights[3])
    assert np.isclose(float(weights.mean()), 1.0)


def test_complementary_weight_is_largest_where_the_value_basis_fits_worst():
    prepared = _prepared()
    weights, diagnostics = improvement.row_weights(prepared, _cfg(), "complementary")
    fit = improvement.value_basis_fit(prepared, _cfg())

    assert diagnostics["weight_mode"] == "complementary"
    assert 0.0 <= fit.min() and fit.max() <= 1.0
    # user 0 (q_C=1) buying the cheap item 1 fits worst, so it gets the most weight
    assert weights[2] == weights.max()
    assert np.isclose(float(weights.mean()), 1.0)


def test_unweighted_arms_get_flat_weights():
    weights, diagnostics = improvement.row_weights(_prepared(), _cfg(), None)

    assert diagnostics == {"weight_mode": "unweighted"}
    np.testing.assert_allclose(weights, 1.0)
    with pytest.raises(ValueError, match="알 수 없는 개선안"):
        improvement.row_weights(_prepared(), _cfg(), "unknown")


def test_six_arms_cross_the_value_basis_with_each_improvement():
    specs = improvement.arm_specifications(_cfg())

    assert [spec["model_id"] for spec in specs] == list(improvement.MODEL_IDS)
    assert [(spec["rho"] > 0, spec["improvement"]) for spec in specs] == [
        (False, None),
        (True, None),
        (False, "original"),
        (True, "original"),
        (False, "first_purchase"),
        (True, "first_purchase"),
        (False, "complementary"),
        (True, "complementary"),
    ]


def _metrics(econ, accuracy=1.0):
    return {
        "recall@10": 0.015 * accuracy,
        "ndcg@10": 0.018 * accuracy,
        "recall@20": 0.024 * accuracy,
        "ndcg@20": 0.021 * accuracy,
        "recall@50": 0.043 * accuracy,
        "ndcg@50": 0.027 * accuracy,
        "price_purchase_amount_weighted_hit@10": econ,
        "vndcg@10": econ / 40,
    }


def _reading(m5_first_econ, *, change=0.2, m5_first_accuracy=1.0):
    rows = {
        improvement.M1_MODEL_ID: _metrics(0.380),
        improvement.M2_MODEL_ID: _metrics(0.389),
        improvement.M4_ORIGINAL_MODEL_ID: _metrics(0.384),
        improvement.M5_ORIGINAL_MODEL_ID: _metrics(0.386),
        improvement.M4_FIRST_MODEL_ID: _metrics(0.384),
        improvement.M5_FIRST_MODEL_ID: _metrics(
            m5_first_econ, accuracy=m5_first_accuracy
        ),
        improvement.M4_COMPLEMENT_MODEL_ID: _metrics(0.384),
        improvement.M5_COMPLEMENT_MODEL_ID: _metrics(0.386),
    }
    return improvement.improvement_reading(
        rows,
        top10_change_shares={
            improvement.M2_MODEL_ID: 0.19,
            improvement.M5_ORIGINAL_MODEL_ID: 0.2,
            improvement.M5_FIRST_MODEL_ID: change,
            improvement.M5_COMPLEMENT_MODEL_ID: 0.2,
        },
    )


def test_an_improvement_passes_only_when_m5_beats_both_parts_and_keeps_accuracy():
    passing = _reading(0.395)
    assert passing["improvements"]["first_purchase"]["improvement_pass"] is True
    assert passing["classification"] == "first_purchase_passes"
    # 0.386 does not beat M2's 0.389
    assert passing["improvements"]["complementary"]["improvement_pass"] is False

    assert _reading(0.387)["classification"] == "no_improvement_passes"
    assert _reading(0.395, change=0.0)["improvements"]["first_purchase"][
        "evaluable"
    ] is False
    assert _reading(0.395, m5_first_accuracy=0.985)["improvements"]["first_purchase"][
        "m5_accuracy_guard_vs_m4"
    ] is False


def test_reading_compares_each_improved_m5_with_the_original_combination():
    reading = _reading(0.395)
    first = reading["improvements"]["first_purchase"]

    assert np.isclose(
        first["deltas_m5_minus_original_m5"][
            "price_purchase_amount_weighted_hit@10"
        ],
        0.395 - 0.386,
    )
    assert "deltas_m5_minus_original_m5" not in reading["improvements"]["original"]


def test_reading_reports_the_interaction_and_leaves_attribution_untested():
    reading = _reading(0.395)
    first = reading["improvements"]["first_purchase"]

    assert np.isclose(
        first["interaction_m5_minus_m4_minus_m2_minus_m1"][
            "price_purchase_amount_weighted_hit@10"
        ],
        (0.395 - 0.384) - (0.389 - 0.380),
    )
    assert reading["clv_attribution_tested"] is False
