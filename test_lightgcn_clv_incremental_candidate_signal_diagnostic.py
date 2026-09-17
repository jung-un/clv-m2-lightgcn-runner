import numpy as np
import pandas as pd
import pytest

import lightgcn_clv_incremental_candidate_signal_diagnostic as diagnostic


def test_preflight_is_checkpoint_only_and_uses_development_splits():
    cfg = diagnostic.configure_incremental_candidate_signal_diagnostic(
        "dunnhumby",
        out_dir="/tmp/incremental-signal",
        baseline_result_dir="/tmp/m1",
    )

    report = diagnostic.preflight_summary(cfg)

    assert report["training"] is False
    assert report["checkpoint_selection"] is False
    assert report["final_test_executed"] is False
    assert report["holdout_executed"] is False
    assert report["new_item_task"] is True
    assert report["matching"]["popularity_bins"] == 10
    assert report["matching"]["controls_per_truth"] == 5


def test_item_context_uses_distinct_purchasers_for_n_and_train_price_for_v():
    train = pd.DataFrame(
        {
            "u_idx": [0, 0, 1, 2],
            "i_idx": [0, 0, 0, 1],
            "up": [10.0, 14.0, 12.0, 30.0],
        }
    )
    q_n = np.array([0.1, 0.9, 0.5])
    valid = np.ones(3, dtype=bool)

    context = diagnostic.build_item_context(
        train, n_items=2, q_n=q_n, clv_valid=valid
    )

    # Repeated rows from user 0 must not count as two purchasers.
    assert np.isclose(context["buyer_qn_mean"][0], 0.5)
    assert context["distinct_buyer_count"][0] == 2
    assert context["distinct_buyer_count"][1] == 1
    assert context["price_percentile"][1] > context["price_percentile"][0]
    assert set(context["popularity_bin"].tolist()).issubset(set(range(10)))


def test_match_controls_preserves_popularity_and_uses_nearest_m1_scores():
    candidate_items = np.array([0, 1, 2, 3, 4, 5])
    scores = np.array([0.10, 0.19, 0.35, 0.21, 0.80, 0.18])
    popularity_bin = np.array([0, 1, 1, 1, 2, 1])

    matched = diagnostic.match_controls_for_truth(
        truth_item=1,
        truth_score=0.20,
        candidate_items=candidate_items,
        candidate_scores=scores,
        popularity_bin=popularity_bin,
        forbidden_items={1, 3},
        controls_per_truth=2,
    )

    # Item 3 is closer but forbidden; same-bin items 5 and 2 remain.
    np.testing.assert_array_equal(matched, np.array([5, 2]))
    assert np.all(popularity_bin[matched] == popularity_bin[1])


def test_match_controls_returns_empty_when_no_same_bin_control_exists():
    matched = diagnostic.match_controls_for_truth(
        truth_item=0,
        truth_score=0.1,
        candidate_items=np.array([0, 1]),
        candidate_scores=np.array([0.1, 0.2]),
        popularity_bin=np.array([0, 1]),
        forbidden_items={0},
        controls_per_truth=5,
    )
    assert matched.size == 0


def test_signal_values_are_candidate_specific_and_qc_is_not():
    values = diagnostic.candidate_signal_values(
        user_qn=0.8,
        user_qv=0.7,
        user_qc=0.9,
        items=np.array([0, 1]),
        buyer_qn_mean=np.array([0.75, 0.1]),
        price_percentile=np.array([0.65, 0.2]),
    )

    assert values[diagnostic.N_SIGNAL][0] > values[diagnostic.N_SIGNAL][1]
    assert values[diagnostic.V_SIGNAL][0] > values[diagnostic.V_SIGNAL][1]
    assert values[diagnostic.NV_SIGNAL][0] > values[diagnostic.NV_SIGNAL][1]
    np.testing.assert_allclose(values[diagnostic.QC_LEVEL], [0.9, 0.9])


def test_aggregate_pair_rows_counts_wins_ties_losses_and_match_quality():
    rows = pd.DataFrame(
        {
            "user_idx": [0, 0, 1],
            "fixed_clv_segment": ["저CLV", "저CLV", "고CLV"],
            "signal": [diagnostic.N_SIGNAL] * 3,
            "candidate_pair_count": [2, 2, 2],
            "truth_wins": [2, 0, 1],
            "ties": [0, 2, 0],
            "control_wins": [0, 0, 1],
            "score_gap_sum": [0.02, 0.04, 0.10],
            "score_gap_max": [0.01, 0.02, 0.05],
        }
    )

    summary = diagnostic.summarize_pairs(rows)

    overall = summary[
        summary.signal.eq(diagnostic.N_SIGNAL)
        & summary.group_type.eq("overall")
    ].iloc[0]
    assert overall.candidate_pair_count == 6
    assert np.isclose(overall.pair_balanced_win_rate, 4 / 6)
    assert np.isclose(overall.mean_absolute_m1_score_gap, 0.16 / 6)


def test_bootstrap_uses_user_macro_contrasts():
    rows = []
    for user in range(20):
        segment = "저CLV" if user < 7 else ("중CLV" if user < 14 else "고CLV")
        for signal, rate in (
            (diagnostic.N_SIGNAL, 0.6),
            (diagnostic.V_SIGNAL, 0.7),
            (diagnostic.NV_SIGNAL, 0.8),
        ):
            rows.append(
                {
                    "user_idx": user,
                    "fixed_clv_segment": segment,
                    "signal": signal,
                    "candidate_pair_count": 10,
                    "truth_wins": int(rate * 10),
                    "ties": 0,
                    "control_wins": 10 - int(rate * 10),
                    "score_gap_sum": 0.1,
                    "score_gap_max": 0.01,
                }
            )
    report = diagnostic.bootstrap_signal_rates(
        pd.DataFrame(rows), samples=200, seed=42
    )

    assert np.isclose(report["signals"][diagnostic.NV_SIGNAL]["observed"], 0.8)
    assert report["signals"][diagnostic.NV_SIGNAL]["ci_low"] > 0.5


def test_config_rejects_nonpositive_matching_parameters():
    with pytest.raises(ValueError):
        diagnostic.configure_incremental_candidate_signal_diagnostic(
            "hm", candidate_pool_per_bin=0
        )
