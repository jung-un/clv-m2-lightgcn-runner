import numpy as np

import lightgcn_clv_residual as residual
from clv_variable_validity import candidate_variables, validate_anchor


def _anchor():
    numeric = np.zeros((6, len(residual.NUMERIC_FEATURES)), np.float32)
    valid = np.ones_like(numeric, dtype=bool)
    values = {
        "basket_count": [1, 2, 3, 4, 5, 6],
        "recency_days": [20, 15, 10, 7, 3, 1],
        "observed_days": [30, 30, 30, 30, 30, 30],
        "gap_mean": [0, 10, 8, 6, 4, 2],
        "avg_basket_value": [10, 20, 30, 40, 50, 60],
        "premium_share": [0, 0, 0.2, 0.4, 0.7, 1.0],
    }
    for name, column in values.items():
        numeric[:, residual.NUMERIC_FEATURES.index(name)] = column
    valid[0, residual.NUMERIC_FEATURES.index("gap_mean")] = False
    future_count = np.array([0, 1, 1, 2, 3, 5], np.float32)
    future_value = np.array([0, 18, 32, 38, 55, 70], np.float32)
    return residual.AnchorExamples(
        7, 0, 1, 2, 3, np.arange(6), numeric, valid,
        (future_count > 0).astype(np.float32),
        future_count * future_value,
        future_count,
        future_value,
    )


def test_candidate_variables_keep_old_and_literature_based_definitions_separate():
    frame = candidate_variables(_anchor())
    required = {
        "old_n_proxy", "old_v_proxy", "old_clv_proxy",
        "repeat_transaction_count", "transaction_recency", "customer_age",
        "mean_transaction_value", "new_n_behavior", "new_v_behavior",
        "new_clv_behavior", "gap_valid",
    }
    assert required.issubset(frame.columns)
    assert frame.loc[0, "repeat_transaction_count"] == 0
    assert frame.loc[5, "repeat_transaction_count"] == 5
    assert frame.loc[0, "transaction_recency"] == 9
    assert frame.loc[0, "gap_valid"] == 0
    assert np.isfinite(frame.to_numpy()).all()


def test_validity_reports_future_n_v_and_total_without_recommender_splits():
    report = validate_anchor(_anchor(), dataset="toy", anchor_label="train_internal_7")
    metrics = report["metrics"]
    assert set(metrics.target) == {
        "future_transaction_count",
        "future_mean_transaction_value",
        "future_total_value",
    }
    assert (metrics.source_split == "train_internal").all()
    n_row = metrics[
        metrics.variable.eq("new_n_behavior")
        & metrics.target.eq("future_transaction_count")
    ].iloc[0]
    v_row = metrics[
        metrics.variable.eq("new_v_behavior")
        & metrics.target.eq("future_mean_transaction_value")
    ].iloc[0]
    assert n_row.spearman > 0
    assert v_row.spearman > 0
    assert len(report["quadrants"]) == 4
