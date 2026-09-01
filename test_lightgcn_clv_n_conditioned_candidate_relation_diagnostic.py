import numpy as np
import pandas as pd

import lightgcn_clv_n_conditioned_candidate_relation_diagnostic as diagnostic


def test_preflight_is_checkpoint_only_and_fixes_macro_decision_rule():
    cfg = diagnostic.configure_n_conditioned_candidate_relation_diagnostic(
        "dunnhumby",
        out_dir="/tmp/n-conditioned-relation",
        baseline_result_dir="/tmp/m1",
    )
    summary = diagnostic.preflight_summary(cfg)
    assert summary["training"] is False
    assert summary["checkpoint_selection"] is False
    assert summary["final_test_executed"] is False
    assert summary["holdout_executed"] is False
    assert summary["decision_metric"] == "macro user balanced win rate"
    assert summary["degree_used_only_for_shuffle_control"] is True


def test_degree_matched_shuffle_is_deterministic_and_stays_in_strata():
    q_n = np.linspace(0.1, 1.0, 10)
    valid = np.ones(10, dtype=bool)
    degree = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
    first, first_strata, info = diagnostic._degree_matched_q_n_shuffle(
        q_n, valid, degree, n_bins=2, seed=42
    )
    second, second_strata, _ = diagnostic._degree_matched_q_n_shuffle(
        q_n, valid, degree, n_bins=2, seed=42
    )
    assert np.array_equal(first, second)
    assert np.array_equal(first_strata, second_strata)
    assert np.array_equal(np.sort(first), np.sort(q_n))
    for stratum in np.unique(first_strata):
        index = np.flatnonzero(first_strata == stratum)
        assert set(first[index]) == set(q_n[index])
    assert info["changed_valid_n_user_share"] > 0


def test_actual_n_conditioned_transition_separates_low_and_high_n_patterns():
    pair = pd.DataFrame(
        {
            "u_idx": np.arange(8),
            "c_idx": np.zeros(8, dtype=int),
            "d_idx": np.array([0, 0, 0, 0, 1, 1, 1, 1]),
            "mass": np.ones(8),
        }
    )
    recent = pd.DataFrame(
        {
            "u_idx": np.arange(8),
            "c_idx": np.zeros(8, dtype=int),
            "recent_share": np.ones(8),
        }
    )
    q_n = np.array([0.0] * 4 + [1.0] * 4)
    q_shuffled = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    scores, info = diagnostic._cross_fitted_transition_score_arms(
        pair=pair,
        recent=recent,
        evaluation_users=np.arange(8),
        q_n=q_n,
        q_n_shuffled=q_shuffled,
        valid_n=np.ones(8, dtype=bool),
        n_users=8,
        n_categories=2,
        folds=2,
        min_support_users=1,
        kappa=1.0,
        log_lift_cap=np.log(3.0),
    )
    actual = scores["actual_n_conditioned"]
    assert np.all(actual[:4, 0] > actual[:4, 1])
    assert np.all(actual[4:, 1] > actual[4:, 0])
    assert info["valid_n_evaluation_user_count"] == 8


def test_summary_reports_macro_and_comparison_deltas():
    rows = []
    rates = {
        "pooled": (0.50, 0.50),
        "actual_n_conditioned": (0.60, 0.70),
        "shuffled_n_conditioned": (0.49, 0.51),
    }
    for arm, values in rates.items():
        for user, rate in enumerate(values):
            rows.append(
                {
                    "user_idx": user,
                    "arm": arm,
                    "fixed_clv_segment": "고CLV",
                    "nv_quadrant": "고N·고V",
                    "high_clv_composition": "균형 고CLV",
                    "candidate_pair_count": 10,
                    "truth_wins": int(rate * 10),
                    "ties": 0,
                    "false_positive_wins": 10 - int(rate * 10),
                    "balanced_win_rate": rate,
                    "strict_win_rate": rate,
                    "mean_pair_score_difference": rate - 0.5,
                }
            )
    summary = diagnostic.summarize_pair_directions(pd.DataFrame(rows))
    comparison = diagnostic.compare_arms(summary)
    selected = comparison[
        comparison.reference_arm.eq("pooled")
        & comparison.group_type.eq("overall")
    ].iloc[0]
    assert np.isclose(selected.macro_user_balanced_win_rate_delta, 0.15)
    assert diagnostic._screen_reading(summary)[
        "n_attribution_supported_in_this_dataset"
    ] is True
