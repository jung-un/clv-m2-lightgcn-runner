import numpy as np
import pandas as pd

import lightgcn_clv_fixed_segment_error_diagnostic as fixed
import lightgcn_clv_m5_price_band_error_diagnostic as band


LOW, MID, HIGH = fixed.SEGMENT_ORDER


def test_price_bands_are_equal_width_and_keep_missing_out():
    bands = band.price_bands(np.array([0.1, 0.25, 0.49, 0.5, 0.99, 1.0, np.nan]))

    np.testing.assert_array_equal(bands, [0, 1, 1, 2, 3, 3, -1])


def test_train_band_counts_use_distinct_pairs_of_evaluation_users():
    train = pd.DataFrame(
        {"u_idx": [0, 0, 0, 1, 2], "i_idx": [0, 0, 3, 1, 2]}
    )
    item_band = np.array([0, 1, 2, 3])

    counts = band.train_band_counts(train, np.array([0, 1]), item_band)

    np.testing.assert_array_equal(counts, [[1, 0, 0, 1], [0, 1, 0, 0]])


def _single_user_rows():
    # items 0..5 with price percentiles; bands 0,0,1,2,3,3
    item_price = np.array([0.10, 0.20, 0.30, 0.60, 0.76, 0.90])
    item_band = band.price_bands(item_price)
    membership = pd.DataFrame(
        {"user_idx": [0], "fixed_clv_segment": [HIGH]}
    )
    return band._pair_rows(
        users=np.array([0]),
        top10=np.array([[1, 4]]),
        truth={0: np.array([5, 0])},
        membership=membership,
        item_price=item_price,
        item_band=item_band,
        user_position=np.array([0.85]),
        user_band=band.price_bands(np.array([0.85])),
        purchased_by_band=np.array([[0, 1, 0, 0]]),
        band_sizes=np.bincount(item_band, minlength=band.N_BANDS),
    )


def test_pairs_split_into_same_and_cross_band_with_q_v_direction():
    rows = _single_user_rows()
    row = rows.iloc[0]

    # missed truths: 5 (band 3), 0 (band 0); false positives: 1 (band 0), 4 (band 3)
    assert row.candidate_pair_count == 4
    assert row.same_band_pairs == 2
    # cross pairs: (5 vs 1) truth closer to 0.85 -> win; (0 vs 4) -> loss
    assert row.cross_band_wins == 1
    # same pairs: (5 vs 4) truth closer -> win; (0 vs 1) truth farther -> loss
    assert row.same_band_wins == 1


def test_false_negative_lift_compares_value_band_to_all_unbought():
    rows = _single_user_rows()
    row = rows.iloc[0]

    assert row.user_value_band == 3
    assert row.truth_in_value_band == 1
    assert row.unbought_in_value_band == 2
    assert row.unbought_items == 5
    assert row.truth_count == 2

    summary = band.summarize(rows)
    high = summary[summary.group.eq(HIGH)].iloc[0]
    assert np.isclose(high.false_negative_lift, (1 / 2) / (2 / 5))


def test_random_pair_same_band_share_is_one_over_bands_for_equal_counts():
    assert np.isclose(band.random_pair_same_band_share(np.array([5, 5, 5, 5])), 0.25)
    assert np.isclose(band.random_pair_same_band_share(np.array([10, 0, 0, 0])), 1.0)


def test_reading_requires_enough_same_band_error_and_a_role_split():
    summary = pd.DataFrame(
        [
            {
                "group_type": "fixed_clv_segment",
                "group": HIGH,
                "same_band_pair_share": 0.40,
                "same_band_q_v_win_rate": 0.52,
                "cross_band_q_v_win_rate": 0.65,
                "false_negative_lift": 1.8,
            }
        ]
    )
    supported = band.dataset_reading(summary, 0.25)
    summary.loc[0, "same_band_q_v_win_rate"] = 0.70
    not_split = band.dataset_reading(summary, 0.25)

    assert supported["design_supported_in_this_dataset"] is True
    assert not_split["roles_split"] is False
    assert not_split["design_supported_in_this_dataset"] is False
