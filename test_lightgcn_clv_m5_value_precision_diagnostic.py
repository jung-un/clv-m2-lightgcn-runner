import numpy as np
import pandas as pd

import lightgcn_clv_fixed_segment_error_diagnostic as fixed
import lightgcn_clv_m5_value_precision_diagnostic as precision


LOW, MID, HIGH = fixed.SEGMENT_ORDER


def test_shrinkage_pulls_rare_buyers_toward_the_population_mean():
    value = np.array([10.0, 10.0, 2.0, 2.0])
    counts = np.array([1.0, 100.0, 1.0, 100.0])
    valid = np.ones(4, dtype=bool)

    shrunken, population = precision.shrunken_mean_value(
        value, counts, valid, strength=5.0
    )

    assert np.isclose(population, np.average(value, weights=counts))
    # the one-transaction users move far, the hundred-transaction users barely.
    assert abs(shrunken[0] - population) < abs(value[0] - population)
    assert abs(shrunken[1] - value[1]) < abs(shrunken[0] - value[0])
    assert np.isclose(
        shrunken[2], (2.0 + 5.0 * population) / 6.0
    )


def test_transaction_counts_read_both_dataset_axis_shapes():
    dunnhumby = {"repeat_transaction_count": np.array([0.0, 3.0])}
    hm = {"n_behavior_score": np.array([4.0, 9.0])}

    np.testing.assert_allclose(precision.transaction_counts(dunnhumby), [1.0, 4.0])
    np.testing.assert_allclose(precision.transaction_counts(hm), [4.0, 9.0])


def test_value_positions_keep_only_valid_users_and_report_movement():
    axes = {
        "v_behavior_score": np.array([1.0, 5.0, 9.0, 0.0]),
        "repeat_transaction_count": np.array([0.0, 4.0, 40.0, 0.0]),
        "valid_user": np.array([True, True, True, False]),
        "value_valid": np.array([True, True, True, False]),
    }
    price = {"user_overall": np.array([0.1, 0.5, 0.9, np.nan])}

    positions, diagnostics = precision.user_value_positions(axes, price, strength=5.0)

    assert np.isnan(positions[precision.RAW_SIGNAL][3])
    assert np.isnan(positions[precision.SHRUNKEN_SIGNAL][3])
    assert diagnostics["valid_user_share"] == 0.75
    assert diagnostics["mean_absolute_percentile_move"] >= 0.0


def _per_user() -> pd.DataFrame:
    rows = []
    for user in range(20):
        segment = LOW if user < 7 else (MID if user < 14 else HIGH)
        for signal, base in (
            (precision.RAW_SIGNAL, 0.50),
            (precision.SHRUNKEN_SIGNAL, 0.60),
            (precision.REFERENCE_SIGNAL, 0.55),
        ):
            rows.append(
                {
                    "user_idx": user,
                    "signal": signal,
                    "fixed_clv_segment": segment,
                    "q_n": 0.5,
                    "q_v": 0.5,
                    "historical_clv_proxy": float(user),
                    "candidate_pair_count": 10,
                    "truth_wins": int(round(base * 10)),
                    "ties": 0,
                    "false_positive_wins": 10 - int(round(base * 10)),
                    "balanced_win_rate": base,
                }
            )
    return pd.DataFrame(rows)


def test_summary_reports_overall_and_each_clv_segment():
    summary = precision.summarize_win_rates(_per_user())

    overall = summary[summary.group_type.eq("overall")]
    assert set(overall.signal) == set(precision.SIGNAL_ORDER)
    assert np.isclose(
        float(
            overall[overall.signal.eq(precision.SHRUNKEN_SIGNAL)][
                "pair_balanced_win_rate"
            ].iloc[0]
        ),
        0.6,
    )
    segments = summary[summary.group_type.eq("fixed_clv_segment")]
    assert set(segments.group) == set(fixed.SEGMENT_ORDER)


def test_bootstrap_detects_a_constant_paired_difference_and_a_flat_slope():
    report = precision.bootstrap_contrasts(_per_user(), samples=200, seed=42)
    contrasts = report["contrasts"]

    shrunken = contrasts["shrunken_minus_raw"]
    assert np.isclose(shrunken["observed"], 0.10)
    assert shrunken["excludes_zero"] is True

    slope = contrasts[f"{precision.RAW_SIGNAL}__high_minus_low_clv"]
    assert np.isclose(slope["observed"], 0.0)
    assert slope["excludes_zero"] is False
    assert report["paired_users"] == 20


def test_first_purchase_rows_count_each_pair_once():
    train = pd.DataFrame(
        {
            "u_idx": [0, 0, 0, 1, 1],
            "i_idx": [5, 5, 6, 7, 8],
            "t": [1, 2, 3, 1, 2],
        }
    )
    membership = pd.DataFrame(
        {
            "user_idx": [0, 1],
            "fixed_clv_segment": [LOW, HIGH],
        }
    )

    stats = precision.first_purchase_row_stats(train, membership)

    assert stats["overall"]["train_rows"] == 5
    assert stats["overall"]["first_purchase_rows"] == 4
    assert np.isclose(stats["overall"]["first_purchase_row_share"], 0.8)
    assert np.isclose(stats["rows_per_unique_pair"], 1.25)
    assert np.isclose(
        stats["evaluation_user_segments"][LOW]["first_purchase_row_share"], 2 / 3
    )
    assert np.isclose(
        stats["evaluation_user_segments"][HIGH]["first_purchase_row_share"], 1.0
    )


def test_pair_rows_use_each_signal_position_against_the_same_items():
    users = np.array([0])
    top10 = np.array([[1, 2]])
    truth = {0: np.array([2, 3])}
    membership = pd.DataFrame(
        {
            "user_idx": [0],
            "fixed_clv_segment": [HIGH],
            "q_n": [0.9],
            "q_v": [0.9],
            "historical_clv_proxy": [12.0],
        }
    )
    item_price = np.array([0.1, 0.2, 0.5, 0.9])
    positions = {
        precision.RAW_SIGNAL: np.array([0.9]),
        precision.SHRUNKEN_SIGNAL: np.array([0.2]),
        precision.REFERENCE_SIGNAL: np.array([np.nan]),
    }

    rows = precision._pair_rows(
        users=users,
        top10=top10,
        truth=truth,
        membership=membership,
        item_price=item_price,
        positions=positions,
    )

    # item 3 is the missed truth, item 1 the false positive; item 2 is both.
    assert set(rows.signal) == {precision.RAW_SIGNAL, precision.SHRUNKEN_SIGNAL}
    raw = rows[rows.signal.eq(precision.RAW_SIGNAL)].iloc[0]
    shrunken = rows[rows.signal.eq(precision.SHRUNKEN_SIGNAL)].iloc[0]
    assert raw["balanced_win_rate"] == 1.0
    assert shrunken["balanced_win_rate"] == 0.0
