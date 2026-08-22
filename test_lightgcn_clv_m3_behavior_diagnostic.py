import numpy as np
import pandas as pd

import lightgcn_clv_m3_behavior_diagnostic as diagnostic


def test_percentile_is_rank_based_and_keeps_missing_invalid():
    percentile, valid = diagnostic._percentile(
        np.array([10.0, 30.0, 20.0, np.nan])
    )
    assert valid.tolist() == [True, True, True, False]
    assert np.allclose(percentile[:3], [1 / 6, 5 / 6, 1 / 2])
    assert np.isnan(percentile[3])


def test_segment_summary_uses_same_users_and_paired_differences():
    shared = {
        "seed": [42, 42],
        "user_idx": [0, 1],
        "nv_quadrant": ["low_n_low_v", "low_n_low_v"],
        "nv_quadrant_label": ["low", "low"],
        "clv_quintile": ["Q1", "Q1"],
    }
    metric_values = {
        metric: [0.0, 0.0] for metric in diagnostic.METRICS
    }
    m1 = pd.DataFrame(
        {**shared, "model_id": [diagnostic.M1_ID] * 2, **metric_values}
    )
    m3_values = {metric: [0.0, 0.0] for metric in diagnostic.METRICS}
    m1["recall@10"] = [0.0, 0.5]
    m3_values["recall@10"] = [0.5, 0.0]
    m3 = pd.DataFrame(
        {**shared, "model_id": [diagnostic.M3_ID] * 2, **m3_values}
    )
    summary = diagnostic._segment_metric_summary(pd.concat([m1, m3]))
    row = summary[
        summary.segment_type.eq("nv_quadrant")
        & summary.metric.eq("recall@10")
    ].iloc[0]
    assert row.n_users == 2
    assert np.isclose(row.m1_mean, 0.25)
    assert np.isclose(row.m3_mean, 0.25)
    assert np.isclose(row.mean_delta, 0.0)
    assert np.isclose(row.improved_user_share, 0.5)
    assert np.isclose(row.degraded_user_share, 0.5)


def test_representative_cases_use_predeclared_delta_rules():
    rows = []
    for model_id, ndcg, recall in (
        (diagnostic.M1_ID, [0.0, 0.8], [0.0, 0.5]),
        (diagnostic.M3_ID, [0.6, 0.0], [0.5, 0.0]),
    ):
        for user in (0, 1):
            rows.append(
                {
                    "seed": 42,
                    "model_id": model_id,
                    "user_idx": user,
                    "user_id": f"u{user}",
                    "nv_quadrant": "low_n_low_v",
                    "nv_quadrant_label": "low",
                    "clv_quintile": "Q1",
                    "n_hat": 1,
                    "v_hat": 1,
                    "clv_proxy": 1,
                    "recall@10": recall[user],
                    "ndcg@10": ndcg[user],
                    "recall@50": recall[user],
                }
            )
    per_user = pd.DataFrame(rows)
    truth = pd.DataFrame(
        {"seed": [42, 42], "user_idx": [0, 1], "item_idx": [3, 4]}
    )
    recs = pd.DataFrame(
        {
            "seed": [42, 42, 42, 42],
            "user_idx": [0, 0, 1, 1],
            "model_id": [diagnostic.M1_ID, diagnostic.M3_ID] * 2,
            "rank": [1, 1, 1, 1],
            "item_idx": [1, 3, 4, 2],
        }
    )
    cfg = diagnostic.configure_m3_behavior_diagnostic(
        representative_per_quadrant=1
    )
    summary, details = diagnostic._representative_cases(
        per_user, truth, recs, cfg
    )
    gain = summary[summary.selection_rule.eq("largest_ndcg10_gain")].iloc[0]
    loss = summary[summary.selection_rule.eq("largest_ndcg10_loss")].iloc[0]
    assert gain.user_idx == 0
    assert loss.user_idx == 1
    assert set(details.detail_role) == {"test_truth", "m1_top10", "m3_top10"}
