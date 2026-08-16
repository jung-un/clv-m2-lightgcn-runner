"""Train-internal validity checks for literature-grounded CLV N/V variables.

These diagnostics do not train the recommender and never use its official
validation, test, or holdout labels.  They only ask whether variables computed
at an earlier point inside the training period are associated with the next
training-period transaction count and value.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

import lightgcn_clv_residual as residual


def _column(anchor: residual.AnchorExamples, name: str) -> tuple[np.ndarray, np.ndarray]:
    index = residual.NUMERIC_FEATURES.index(name)
    return (
        np.asarray(anchor.numeric[:, index], dtype=np.float32),
        np.asarray(anchor.valid[:, index], dtype=bool),
    )


def _percentile(values: np.ndarray, valid: np.ndarray, *, ascending: bool = True) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(values)
    result = np.zeros(len(values), dtype=np.float32)
    if valid.any():
        ranked = (rankdata(values[valid], method="average") - 0.5) / valid.sum()
        result[valid] = ranked if ascending else 1.0 - ranked
    return result


def _masked_mean(parts: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    numerator = np.zeros(len(parts[0][0]), dtype=np.float32)
    denominator = np.zeros(len(parts[0][0]), dtype=np.float32)
    for values, valid in parts:
        mask = np.asarray(valid, dtype=np.float32)
        numerator += np.asarray(values, dtype=np.float32) * mask
        denominator += mask
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0,
    )


def candidate_variables(anchor: residual.AnchorExamples) -> pd.DataFrame:
    """Return old proxies and current CLV-component behavior variables.

    The current variables follow the non-contractual CLV decomposition:
    repeat transactions ``x``, time of the last transaction ``t_x``, customer
    age ``T`` and mean inter-transaction gap describe the transaction process;
    mean value per transaction describes the monetary process.  The two
    behavior scores are descriptive observed rates and values, not forecasts
    N-hat/V-hat.  The runner converts them to fixed train-history percentiles
    for the gates.
    """

    basket_count, basket_valid = _column(anchor, "basket_count")
    recency_days, recency_valid = _column(anchor, "recency_days")
    observed_days, observed_valid = _column(anchor, "observed_days")
    gap_mean, gap_valid = _column(anchor, "gap_mean")
    avg_basket_value, value_valid = _column(anchor, "avg_basket_value")
    premium_share, premium_valid = _column(anchor, "premium_share")

    repeat_count = np.maximum(basket_count - 1.0, 0.0)
    customer_age = np.maximum(observed_days - 1.0, 0.0)
    transaction_recency = np.clip(customer_age - recency_days, 0.0, customer_age)
    activity_valid = basket_valid & observed_valid
    repeat_transaction_rate = np.divide(
        repeat_count,
        np.maximum(customer_age, 1.0),
        out=np.zeros_like(repeat_count),
        where=activity_valid,
    )

    new_n_behavior = repeat_transaction_rate
    new_v_behavior = np.where(value_valid, avg_basket_value, 0.0)

    old_n_proxy = _masked_mean(
        [
            (_percentile(basket_count, basket_valid), basket_valid),
            (_percentile(observed_days, observed_valid), observed_valid),
            (_percentile(recency_days, recency_valid, ascending=False), recency_valid),
        ]
    )
    old_v_proxy = _masked_mean(
        [
            (_percentile(avg_basket_value, value_valid), value_valid),
            (_percentile(premium_share, premium_valid), premium_valid),
        ]
    )

    return pd.DataFrame(
        {
            "user_id": np.asarray(anchor.user_ids, dtype=np.int64),
            "old_n_proxy": old_n_proxy,
            "old_v_proxy": old_v_proxy,
            "old_clv_proxy": old_n_proxy * old_v_proxy,
            "repeat_transaction_count": repeat_count,
            "repeat_transaction_rate": repeat_transaction_rate,
            "transaction_recency": transaction_recency,
            "customer_age": customer_age,
            "mean_transaction_gap": np.where(gap_valid, gap_mean, 0.0),
            "gap_valid": gap_valid.astype(np.float32),
            "mean_transaction_value": np.where(value_valid, avg_basket_value, 0.0),
            "value_valid": value_valid.astype(np.float32),
            "new_n_behavior": new_n_behavior,
            "new_v_behavior": new_v_behavior,
            "new_clv_behavior": new_n_behavior * new_v_behavior,
        }
    )


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    good = np.isfinite(x) & np.isfinite(y)
    if good.sum() < 3 or np.unique(x[good]).size < 2 or np.unique(y[good]).size < 2:
        return 0.0
    statistic = spearmanr(x[good], y[good]).statistic
    return float(statistic) if np.isfinite(statistic) else 0.0


def validate_anchor(
    anchor: residual.AnchorExamples, *, dataset: str, anchor_label: str
) -> dict[str, pd.DataFrame]:
    """Validate candidate variables against future outcomes inside train."""

    variables = candidate_variables(anchor)
    count = np.asarray(anchor.transaction_target, dtype=np.float32)
    if anchor.mean_transaction_value_target is None:
        total = np.asarray(anchor.amount_target, dtype=np.float32)
        mean_value = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
    else:
        mean_value = np.asarray(anchor.mean_transaction_value_target, dtype=np.float32)
    total = np.asarray(anchor.amount_target, dtype=np.float32)
    targets = {
        "future_transaction_count": count,
        "future_mean_transaction_value": mean_value,
        "future_total_value": total,
    }
    rows = []
    for variable in (
        "old_n_proxy",
        "old_v_proxy",
        "old_clv_proxy",
        "new_n_behavior",
        "new_v_behavior",
        "new_clv_behavior",
    ):
        for target_name, target in targets.items():
            rows.append(
                {
                    "dataset": dataset,
                    "anchor": anchor_label,
                    "source_split": "train_internal",
                    "variable": variable,
                    "target": target_name,
                    "n_users": int(len(variables)),
                    "spearman": _safe_spearman(
                        variables[variable].to_numpy(float), target
                    ),
                }
            )

    n_high = variables.new_n_behavior >= variables.new_n_behavior.median()
    v_high = variables.new_v_behavior >= variables.new_v_behavior.median()
    labels = np.select(
        [~n_high & ~v_high, n_high & ~v_high, ~n_high & v_high, n_high & v_high],
        ["low_low", "activity", "value", "core"],
        default="low_low",
    )
    quadrant_rows = []
    for label in ("low_low", "activity", "value", "core"):
        mask = labels == label
        quadrant_rows.append(
            {
                "dataset": dataset,
                "anchor": anchor_label,
                "quadrant": label,
                "user_count": int(mask.sum()),
                "future_transaction_count_mean": float(count[mask].mean()) if mask.any() else 0.0,
                "future_mean_transaction_value_mean": float(mean_value[mask].mean()) if mask.any() else 0.0,
                "future_total_value_mean": float(total[mask].mean()) if mask.any() else 0.0,
            }
        )
    return {
        "variables": variables,
        "metrics": pd.DataFrame(rows),
        "quadrants": pd.DataFrame(quadrant_rows),
    }
