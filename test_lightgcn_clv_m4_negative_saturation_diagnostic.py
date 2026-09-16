import numpy as np
import pandas as pd
import torch

import lightgcn_clv_fixed_segment_error_diagnostic as fixed
import lightgcn_clv_m4_negative_saturation_diagnostic as saturation


LOW, MID, HIGH = fixed.SEGMENT_ORDER


def test_positive_sampling_stays_inside_each_user_own_items():
    csr_ptr = np.array([0, 3, 4])
    csr_items = np.array([10, 11, 12, 20])
    rng = np.random.default_rng(0)

    users, items = saturation.sample_user_positives(
        np.array([0, 1]), csr_ptr, csr_items, rng, per_user=2
    )

    assert (users == 0).sum() == 2 and (users == 1).sum() == 1
    assert set(items[users == 0]) <= {10, 11, 12}
    assert set(items[users == 1]) == {20}


def test_signals_are_the_bpr_gradient_factor_for_both_negative_kinds():
    user_embedding = torch.tensor([[1.0, 0.0]])
    item_embedding = torch.tensor([[2.0, 0.0], [1.0, 0.0], [3.0, 0.0]])

    signals = saturation.pair_signals(
        user_embedding,
        item_embedding,
        np.array([0]),
        np.array([0]),  # positive score 2.0
        np.array([[1, 2]]),  # candidate scores 1.0 and 3.0
    )

    assert np.isclose(signals["uniform_signal"][0], 1 / (1 + np.exp(1.0)))
    assert np.isclose(signals["hard_signal"][0], 1 / (1 + np.exp(-1.0)))
    assert signals["correct"][0] == 1.0


def _per_user(low_uniform, high_uniform, low_headroom, high_headroom, users=40):
    rows = []
    for index in range(users):
        segment = LOW if index < users // 2 else HIGH
        uniform = low_uniform if segment == LOW else high_uniform
        headroom = low_headroom if segment == LOW else high_headroom
        rows.append(
            {
                "user_idx": index,
                "uniform_signal": uniform,
                "hard_signal": uniform + headroom,
                "headroom": headroom,
                "p_correct_uniform": 0.9,
                "fixed_clv_segment": segment,
            }
        )
    return pd.DataFrame(rows)


def test_summary_reports_overall_and_each_present_segment():
    summary = saturation.summarize(_per_user(0.20, 0.05, 0.10, 0.30))

    assert set(summary.group) == {"전체", LOW, HIGH}
    high = summary[summary.group.eq(HIGH)].iloc[0]
    assert np.isclose(high.uniform_signal, 0.05)
    assert np.isclose(high.headroom, 0.30)


def test_reading_supports_the_design_only_with_saturation_and_headroom():
    supported = _per_user(0.20, 0.05, 0.10, 0.30)
    summary = saturation.summarize(supported)
    bootstrap = saturation.bootstrap_high_minus_low(supported, samples=200)
    reading = saturation.dataset_reading(summary, bootstrap)

    assert reading["saturation_in_high_clv"] is True
    assert reading["more_headroom_in_high_clv"] is True
    assert reading["design_supported_in_this_dataset"] is True


def test_reading_rejects_when_high_clv_signal_is_not_lower():
    flipped = _per_user(0.05, 0.20, 0.30, 0.10)
    summary = saturation.summarize(flipped)
    bootstrap = saturation.bootstrap_high_minus_low(flipped, samples=200)
    reading = saturation.dataset_reading(summary, bootstrap)

    assert reading["saturation_in_high_clv"] is False
    assert reading["more_headroom_in_high_clv"] is False
    assert reading["design_supported_in_this_dataset"] is False


def test_flat_segments_give_intervals_that_contain_zero():
    flat = _per_user(0.10, 0.10, 0.20, 0.20)
    bootstrap = saturation.bootstrap_high_minus_low(flat, samples=200)

    for contrast in bootstrap["contrasts"].values():
        assert contrast["excludes_zero"] is False


def test_per_user_table_averages_pairs_and_keeps_the_segment():
    membership = pd.DataFrame(
        {"user_idx": [0, 1], "fixed_clv_segment": [LOW, HIGH]}
    )
    signals = {
        "uniform_signal": np.array([0.2, 0.4, 0.1]),
        "hard_signal": np.array([0.5, 0.7, 0.3]),
        "correct": np.array([1.0, 0.0, 1.0]),
    }

    table = saturation.per_user_table(np.array([0, 0, 1]), signals, membership)

    assert list(table.user_idx) == [0, 1]
    assert np.isclose(table.uniform_signal.iloc[0], 0.3)
    assert np.isclose(table.headroom.iloc[0], 0.3)
    assert list(table.fixed_clv_segment) == [LOW, HIGH]
