import numpy as np
import pandas as pd
import pytest

import lightgcn_clv_m5_value_basis_hm2y_screen as hm_screen


def _cfg(**overrides):
    return hm_screen.configure_hm2y_value_basis_screen(
        out_dir="/tmp/hm2y", **overrides
    )


def test_config_fixes_the_hm_scale_settings():
    cfg = _cfg()

    assert (cfg.dataset, cfg.seed, cfg.negative_count) == ("hm", 42, 1)
    assert cfg.batch_size == 131_072
    assert cfg.rho == 0.25
    with pytest.raises(ValueError, match="batch_size"):
        _cfg(batch_size=8_192)


def test_degree_deciles_split_users_into_equal_sized_bins():
    train = pd.DataFrame(
        {"u_idx": [0, 0, 0, 1, 1, 2, 3, 3, 3, 3], "i_idx": [0, 1, 2, 0, 1, 0, 0, 1, 2, 3]}
    )

    bins = hm_screen.degree_deciles(train, n_users=4, bins=2)

    assert bins[2] == 0 and bins[3] == 1
    assert set(np.unique(bins)) <= {0, 1}
    assert (bins == 0).sum() == (bins == 1).sum()


def test_arms_run_m1_then_actual_then_the_shuffle_control():
    prepared = {"m2_actual": {"name": "actual"}, "m2_shuffle": {"name": "shuffled"}}

    arms = hm_screen.arm_specifications(prepared, _cfg())
    without = hm_screen.arm_specifications(prepared, _cfg(include_shuffle=False))

    assert [arm["model_id"] for arm in arms] == list(hm_screen.MODEL_IDS)
    assert [arm["rho"] > 0 for arm in arms] == [False, True, True]
    assert [arm["m2_assignment"]["name"] for arm in arms] == [
        "actual",
        "actual",
        "shuffled",
    ]
    assert [arm["model_id"] for arm in without] == list(hm_screen.MODEL_IDS[:2])
    assert hm_screen.preflight_summary(_cfg(include_shuffle=False))[
        "trained_models"
    ] == list(hm_screen.MODEL_IDS[:2])


def _metrics(econ, accuracy=1.0):
    return {
        "recall@10": 0.0114 * accuracy,
        "ndcg@10": 0.0077 * accuracy,
        "price_purchase_amount_weighted_hit@10": econ,
        "vndcg@10": econ / 8,
    }


def test_portability_needs_both_economic_metrics_and_the_accuracy_guard():
    rows = {
        hm_screen.M1_MODEL_ID: _metrics(0.000885),
        hm_screen.M2_ACTUAL_MODEL_ID: _metrics(0.000905),
    }
    shares = {hm_screen.M2_ACTUAL_MODEL_ID: 0.2}

    portable = hm_screen.portability_reading(rows, top10_change_shares=shares)
    rows[hm_screen.M2_ACTUAL_MODEL_ID] = _metrics(0.000870)
    weaker = hm_screen.portability_reading(rows, top10_change_shares=shares)

    assert portable["classification"] == "portable"
    assert weaker["classification"] == "not_portable"
    assert portable["assignment_signal_tested"] is False


def test_shuffle_arm_adds_the_assignment_comparison():
    rows = {
        hm_screen.M1_MODEL_ID: _metrics(0.000885),
        hm_screen.M2_ACTUAL_MODEL_ID: _metrics(0.000905),
        hm_screen.M2_SHUFFLED_MODEL_ID: _metrics(0.000890),
    }

    reading = hm_screen.portability_reading(
        rows, top10_change_shares={hm_screen.M2_ACTUAL_MODEL_ID: 0.2}
    )

    assert reading["assignment_signal_tested"] is True
    assert reading["m2_beats_shuffle_on_both_economic_metrics"] is True
    assert np.isclose(
        reading["deltas_m2_actual_minus_shuffled"][
            "price_purchase_amount_weighted_hit@10"
        ],
        0.000015,
    )


def test_no_top10_change_is_not_evaluable():
    rows = {
        hm_screen.M1_MODEL_ID: _metrics(0.000885),
        hm_screen.M2_ACTUAL_MODEL_ID: _metrics(0.000905),
    }

    reading = hm_screen.portability_reading(
        rows, top10_change_shares={hm_screen.M2_ACTUAL_MODEL_ID: 0.0}
    )

    assert reading["classification"] == "not_evaluable_no_top10_change"
    assert reading["portability"] is False
