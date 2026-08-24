from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from clv_m3_profile_item_graph import (
    CLVProfileItemLightGCN,
    build_clv_profile_item_graph,
)
import lightgcn_clv_m3_profile_item_backtest as runner


def _train_frame():
    return pd.DataFrame(
        {
            "u_idx": [0, 0, 1, 1, 2, 2, 3, 3],
            "i_idx": [0, 2, 0, 2, 1, 2, 1, 2],
            "b_raw": [10, 11, 20, 21, 30, 31, 40, 41],
            "t": [1, 2, 1, 2, 1, 2, 1, 2],
            "v": [5.0, 5.0, 5.0, 5.0, 20.0, 20.0, 20.0, 20.0],
        }
    )


def test_profile_item_graph_uses_positive_pmi_and_removes_nonselective_item():
    graph = build_clv_profile_item_graph(
        _train_frame(), n_users=4, n_items=3, n_profile_bins=2
    )
    dense = graph.profile_item_operator.to_dense().numpy()

    assert graph.profile_bin.tolist() == [0, 0, 1, 1]
    np.testing.assert_allclose(dense[0], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(dense[1], [0.0, 1.0, 0.0])
    assert graph.diagnostics["profile_item_relation"]["n_positive_edges"] == 2
    assert graph.diagnostics["definition"]["item_price_used"] is False


def test_profile_item_message_is_weighted_item_mean_selected_by_user_profile():
    operator = torch.tensor([[0.75, 0.25, 0.0], [0.0, 0.0, 1.0]]).to_sparse()
    fake = SimpleNamespace(
        profile_item_operator=operator,
        profile_bin=torch.tensor([0, 1, 0]),
    )
    item_embedding = torch.tensor([[2.0], [6.0], [10.0]])
    message = CLVProfileItemLightGCN._profile_item_message(fake, item_embedding)
    torch.testing.assert_close(message, torch.tensor([[3.0], [10.0], [3.0]]))


def test_profile_item_graph_rejects_missing_active_users():
    with pytest.raises(ValueError, match="every indexed training user"):
        build_clv_profile_item_graph(
            _train_frame().query("u_idx != 3"),
            n_users=4,
            n_items=3,
            n_profile_bins=2,
        )


def test_config_locks_historical_interval_and_protected_settings(tmp_path):
    cfg = runner.configure_m3_clv_profile_item_backtest(out_dir=str(tmp_path))
    summary = runner.preflight_summary(cfg)
    split = summary["historical_development_split"]
    assert split["train_end_inclusive"] == 676
    assert split["evaluation_start_inclusive"] == 677
    assert split["evaluation_end_inclusive"] == 683
    assert split["previous_days_684_690_constructed"] is False
    assert split["final_test_constructed"] is False
    assert split["holdout_constructed"] is False
    assert summary["graph_intervention"]["purchase_graph"] == (
        "unchanged binary M1 graph and normalization"
    )
    assert summary["graph_intervention"]["profile_item_weight"] == (
        "row-normalized positive pointwise mutual information from train-only unique user-item pairs"
    )
    assert cfg.seed == 42
    assert cfg.epochs == 100


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
        runner.configure_m3_clv_profile_item_backtest(
            out_dir=str(tmp_path), **override
        )


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
        [
            _metric_row("m1_baseline", 1.0),
            _metric_row("m3_clv_profile_item", 1.1),
        ]
    )
    assert runner._pilot_decision(passing)["passes_pilot"] is True

    failing = pd.DataFrame(
        [
            _metric_row("m1_baseline", 1.0),
            _metric_row("m3_clv_profile_item", 1.1, accuracy=0.98),
        ]
    )
    decision = runner._pilot_decision(failing)
    assert decision["passes_pilot"] is False
    assert decision["checks"]["accuracy_guard"] is False
