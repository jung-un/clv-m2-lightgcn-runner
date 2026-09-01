import numpy as np
import pandas as pd

import lightgcn_clv_candidate_relation_diagnostic as diagnostic


def test_preflight_is_checkpoint_only_and_protects_final_splits():
    cfg = diagnostic.configure_candidate_relation_diagnostic(
        "dunnhumby",
        out_dir="/tmp/candidate-relation",
        baseline_result_dir="/tmp/m1",
    )
    summary = diagnostic.preflight_summary(cfg)
    assert summary["training"] is False
    assert summary["checkpoint_selection"] is False
    assert summary["new_item_task"] is True
    assert summary["final_test_executed"] is False
    assert summary["holdout_executed"] is False
    assert summary["repeatshare_used"] is False
    assert summary["raw_item_degree_used"] is False


def test_cross_fitted_category_lift_prefers_supported_transition():
    pair = pd.DataFrame(
        {
            "u_idx": [0, 1, 2, 3],
            "c_idx": [0, 0, 0, 1],
            "d_idx": [1, 1, 1, 0],
            "mass": [1.0, 1.0, 1.0, 1.0],
        }
    )
    recent = pd.DataFrame(
        {
            "u_idx": [0, 1, 2, 3],
            "c_idx": [0, 0, 0, 1],
            "recent_share": [1.0, 1.0, 1.0, 1.0],
        }
    )
    scores, info = diagnostic._cross_fitted_category_lift_scores(
        pair=pair,
        recent=recent,
        evaluation_users=np.arange(4),
        n_users=4,
        n_categories=2,
        folds=2,
        min_support_users=1,
        kappa=1.0,
        log_lift_cap=np.log(3.0),
    )
    assert scores.shape == (4, 2)
    assert scores[0, 1] > scores[0, 0]
    assert info["transition_rows"] == 4


def test_pair_summary_counts_truth_false_and_ties():
    per_user = pd.DataFrame(
        {
            "user_idx": [0, 1],
            "signal": [diagnostic.SIGNAL_ORDER[0]] * 2,
            "fixed_clv_segment": ["고CLV", "고CLV"],
            "nv_quadrant": ["고N·고V", "고N·고V"],
            "high_clv_composition": ["균형 고CLV", "균형 고CLV"],
            "candidate_pair_count": [4, 2],
            "truth_wins": [3, 0],
            "ties": [0, 2],
            "false_positive_wins": [1, 0],
            "balanced_win_rate": [0.75, 0.5],
            "mean_pair_score_difference": [0.2, 0.0],
        }
    )
    summary = diagnostic.summarize_pair_directions(per_user)
    overall = summary[
        summary.signal.eq(diagnostic.SIGNAL_ORDER[0])
        & summary.group_type.eq("overall")
    ].iloc[0]
    assert overall.candidate_pair_count == 6
    assert overall.truth_wins == 3
    assert overall.ties == 2
    assert np.isclose(overall.pair_balanced_win_rate, 4 / 6)


def test_value_fit_rewards_candidate_closer_to_user_price_position():
    price = {
        "item_overall": np.array([0.75, 0.20]),
        "item_within": np.array([0.70, 0.10]),
        "user_overall": np.array([0.80]),
        "user_within": np.array([0.75]),
    }
    values = diagnostic._candidate_signal_values(
        user_position=0,
        user_id=0,
        items=np.array([0, 1]),
        item_category=np.array([0, 0]),
        activity_scores=np.zeros((1, 1)),
        price=price,
    )
    assert values[diagnostic.SIGNAL_ORDER[1]][0] > values[diagnostic.SIGNAL_ORDER[1]][1]
    assert values[diagnostic.SIGNAL_ORDER[2]][0] > values[diagnostic.SIGNAL_ORDER[2]][1]
