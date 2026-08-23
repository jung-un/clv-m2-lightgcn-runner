from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from clv_m3_profile_graph import CLVProfileLightGCN, build_clv_profile_graph
import lightgcn_clv_m3_profile_backtest as runner


def _train_frame():
    return pd.DataFrame(
        {
            "u_idx": [0, 0, 1, 1, 2, 2, 3, 3],
            "i_idx": [0, 1, 0, 2, 1, 2, 0, 3],
            "b_raw": [10, 11, 20, 21, 30, 31, 40, 41],
            "t": [1, 2, 1, 2, 1, 2, 1, 2],
            "v": [5.0, 5.0, 10.0, 10.0, 15.0, 15.0, 20.0, 20.0],
        }
    )


def test_profile_graph_assigns_one_balanced_bin_per_user():
    graph = build_clv_profile_graph(_train_frame(), 4, n_profile_bins=2)
    assert graph.profile_bin.tolist() == [0, 0, 1, 1]
    assert graph.profile_size.tolist() == [2, 2]
    np.testing.assert_allclose(graph.n_hat, [2, 2, 2, 2])
    np.testing.assert_allclose(graph.v_hat, [5, 10, 15, 20])
    np.testing.assert_allclose(graph.clv_proxy, [10, 20, 30, 40])


def test_profile_message_is_other_users_mean_and_excludes_self():
    fake = SimpleNamespace(
        profile_bin=torch.tensor([0, 0, 1, 1]),
        profile_size=torch.tensor([2.0, 2.0]),
    )
    embeddings = torch.tensor([[1.0], [3.0], [10.0], [14.0]])
    message = CLVProfileLightGCN._peer_profile_message(fake, embeddings)
    torch.testing.assert_close(message, torch.tensor([[3.0], [1.0], [14.0], [10.0]]))


def test_profile_graph_rejects_empty_or_invalid_bins():
    with pytest.raises(ValueError):
        build_clv_profile_graph(_train_frame().iloc[0:0], 4, n_profile_bins=2)
    with pytest.raises(ValueError):
        build_clv_profile_graph(_train_frame(), 4, n_profile_bins=1)


def test_config_locks_new_historical_interval_and_protected_settings(tmp_path):
    cfg = runner.configure_m3_clv_profile_backtest(out_dir=str(tmp_path))
    summary = runner.preflight_summary(cfg)
    split = summary["historical_development_split"]
    assert split["train_end_inclusive"] == 676
    assert split["evaluation_start_inclusive"] == 677
    assert split["evaluation_end_inclusive"] == 683
    assert split["previous_days_684_690_constructed"] is False
    assert split["final_test_constructed"] is False
    assert split["holdout_constructed"] is False
    assert cfg.seed == 42
    assert cfg.epochs == 100
    assert cfg.n_profile_bins == 10


@pytest.mark.parametrize(
    "override",
    [
        {"time_cutoff": 690},
        {"evaluation_days": 14},
        {"epochs": 50},
        {"n_profile_bins": 5},
        {"profile_alpha_init": 1.0},
    ],
)
def test_config_rejects_post_hoc_changes(tmp_path, override):
    with pytest.raises(ValueError):
        runner.configure_m3_clv_profile_backtest(out_dir=str(tmp_path), **override)


def _metric_row(model_id: str, weighted_hit: float, accuracy: float = 1.0):
    row = {
        "model_id": model_id,
        "mean_recommended_price_percentile@10": 0.25,
        "n_distinct@10": 200,
        "top10_share@10": 0.4,
        "price_purchase_amount_weighted_hit@10": weighted_hit,
    }
    for metric in runner.ACCURACY_METRICS:
        row[metric] = accuracy
    return row


def test_pilot_requires_accuracy_and_weighted_hit_improvement():
    passing = pd.DataFrame(
        [_metric_row("m1_baseline", 1.0), _metric_row("m3_clv_profile", 1.1)]
    )
    assert runner._pilot_decision(passing)["passes_pilot"] is True

    failing = pd.DataFrame(
        [
            _metric_row("m1_baseline", 1.0),
            _metric_row("m3_clv_profile", 1.1, accuracy=0.98),
        ]
    )
    decision = runner._pilot_decision(failing)
    assert decision["passes_pilot"] is False
    assert decision["checks"]["accuracy_guard"] is False
