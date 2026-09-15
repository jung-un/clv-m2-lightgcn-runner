import numpy as np
import pandas as pd
import torch

import graphsage_clv_m2_user_clv_weight_screen as runner


def _prepared_for_shuffle():
    users = np.repeat(np.arange(20), 2)
    items = np.tile(np.arange(2), 20)
    return {
        "data": {
            "n_users": 20,
            "train": pd.DataFrame({"u_idx": users, "i_idx": items}),
        },
        "q_v": np.linspace(0.05, 0.95, 20, dtype=np.float32),
        "q_c": np.linspace(0.025, 0.975, 20, dtype=np.float32),
        "clv_valid": np.ones(20, dtype=bool),
    }


def test_preflight_fixes_graphsage_new_item_protocol_and_discloses_rule_change():
    cfg = runner.configure_graphsage_m2_user_clv_weight_screen(
        out_dir="/tmp/graphsage-user-clv-weight",
        reference_result_json="/tmp/old-graphsage.json",
    )
    summary = runner.preflight_summary(cfg)

    assert summary["backbone"] == "graphsage"
    assert summary["fixed"]["new_item_task"] is True
    assert summary["fixed"]["train_pairs_excluded_from_evaluation"] is True
    assert summary["fixed"]["min_item_interactions"] == 1
    assert summary["fixed"]["graph"] == "binary"
    assert summary["fixed"]["negative_sampling"] == "uniform"
    assert summary["fixed"]["final_test_constructed"] is False
    assert summary["fixed"]["holdout_constructed"] is False
    assert summary["modified_m4"]["item_price_or_bin_fit_in_weight"] is False
    assert "post-result" in summary["rule_change_disclosure"]


def test_degree_matched_joint_assignment_preserves_pairs_and_changes_users():
    prepared = _prepared_for_shuffle()
    shuffled = runner.degree_matched_joint_assignment(prepared, seed=42, n_bins=2)
    source = shuffled["source_user"]

    assert shuffled["changed_valid_user_share"] > 0.0
    np.testing.assert_array_equal(shuffled["q_c"], prepared["q_c"][source])
    np.testing.assert_array_equal(shuffled["q_v"], prepared["q_v"][source])
    np.testing.assert_array_equal(
        shuffled["stratum"], shuffled["stratum"][source]
    )


def test_user_clv_row_weight_has_no_item_direction_and_train_mean_is_one():
    prepared = {
        "data": {"tr_u": np.array([0, 0, 1, 2], dtype=np.int64)}
    }
    assignment = {"q_c": np.array([0.2, 0.5, 0.9], dtype=np.float32)}
    normalizer = runner.user_clv_train_weight_normalizer(
        prepared, assignment, 0.5
    )
    users = torch.tensor([0, 0, 1, 2])
    q_c = torch.tensor(assignment["q_c"])[users]
    first = runner.user_clv_positive_row_weights(
        q_c,
        torch.tensor([0.0, 0.2, 0.7, 1.0]),
        train_mean_raw_weight=normalizer,
        lambda_=0.5,
    )
    second = runner.user_clv_positive_row_weights(
        q_c,
        torch.tensor([1.0, 0.8, 0.3, 0.0]),
        train_mean_raw_weight=normalizer,
        lambda_=0.5,
    )

    torch.testing.assert_close(first, second)
    torch.testing.assert_close(first.mean(), torch.tensor(1.0))


def test_arm_specs_only_change_assignment_or_user_clv_loss_weight():
    prepared = {
        "q_v": np.array([0.3, 0.7], dtype=np.float32),
        "q_c": np.array([0.2, 0.8], dtype=np.float32),
        "clv_valid": np.ones(2, dtype=bool),
        "degree_matched_assignment": {
            "q_v": np.array([0.7, 0.3], dtype=np.float32),
            "q_c": np.array([0.8, 0.2], dtype=np.float32),
            "clv_valid": np.ones(2, dtype=bool),
        },
    }
    cfg = runner.configure_graphsage_m2_user_clv_weight_screen(
        out_dir="/tmp/graphsage-user-clv-weight",
        reference_result_json="/tmp/old-graphsage.json",
    )
    specs = runner.arm_specifications(prepared, cfg)

    assert [(spec["m2_active"], spec["weighted"]) for spec in specs] == [
        (False, False),
        (True, False),
        (True, False),
        (False, True),
        (True, True),
    ]
    assert all("user_bin_fit" not in spec["assignment"] for spec in specs)


def test_screening_requires_assignment_and_m5_top10_dominance():
    ids = runner.model_ids()
    rows = {
        ids["m1"]: {metric: 1.0 for metric in runner.TOP10_METRICS},
        ids["m2"]: {metric: 1.2 for metric in runner.TOP10_METRICS},
        ids["m2_shuffle"]: {metric: 1.1 for metric in runner.TOP10_METRICS},
        ids["m4"]: {metric: 1.15 for metric in runner.TOP10_METRICS},
        ids["m5"]: {metric: 1.3 for metric in runner.TOP10_METRICS},
    }
    reading = runner.screening_reading(rows)

    assert reading["m2_assignment_signal"] is True
    assert reading["modified_m5_beats_observed_m2_all_four_top10_metrics"] is True
    assert reading["modified_m5_beats_m1_all_four_top10_metrics"] is True
    assert reading["mechanism_screen_pass"] is True
