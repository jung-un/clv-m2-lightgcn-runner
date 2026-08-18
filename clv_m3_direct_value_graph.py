"""Direct historical-CLV edge weights for the M3 graph intervention.

The three weight arrays share the exact binary M1 edge set:

* user CLV only: every edge of a customer receives the same CLV weight;
* spend only: an identification control using cumulative train spend per edge;
* user CLV x spend: customer value scales the value of each user-item relation.

All inputs come from the train split.  The graph changes only propagation;
negative sampling and the plain BPR objective remain untouched.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


@dataclass(frozen=True)
class DirectCLVValueGraph:
    edge_users: np.ndarray
    edge_items: np.ndarray
    user_clv_weights: np.ndarray
    spend_only_weights: np.ndarray
    clv_spend_weights: np.ndarray
    clv_gate: np.ndarray
    edge_spend: np.ndarray
    spend_signal: np.ndarray
    diagnostics: dict


def _mean_one(raw: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float64)
    if raw.size == 0 or not np.all(np.isfinite(raw)) or np.any(raw <= 0):
        raise ValueError("M3 direct CLV graph weights must be finite and positive")
    mean = float(raw.mean())
    if mean <= 0:
        raise ValueError("M3 direct CLV graph weight mean must be positive")
    return (raw / mean).astype(np.float32)


def _stats(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    if np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return 0.0
    value = spearmanr(left, right).statistic
    return float(value) if np.isfinite(value) else 0.0


def build_direct_clv_value_graph(
    train: pd.DataFrame,
    n_users: int,
    n_items: int,
    clv_gate: np.ndarray,
    *,
    alpha: float = 1.0,
) -> DirectCLVValueGraph:
    """Build mean-one CLV and relationship-value graph weights from train only."""
    required = {"u_idx", "i_idx", "v", "up"}
    missing = required - set(train.columns)
    if missing:
        raise ValueError(f"direct CLV graph requires columns {sorted(missing)}")
    if alpha < 0 or not np.isfinite(alpha):
        raise ValueError("alpha must be finite and non-negative")
    clv_gate = np.asarray(clv_gate, dtype=np.float64)
    if clv_gate.shape != (n_users,):
        raise ValueError(f"clv_gate must have shape ({n_users},)")
    if not np.all(np.isfinite(clv_gate)) or np.any(clv_gate < 0):
        raise ValueError("clv_gate must be finite and non-negative")

    grouped = train.groupby(["u_idx", "i_idx"], sort=True).agg(
        spend=("v", "sum")
    )
    if grouped.empty:
        raise ValueError("cannot build an M3 graph from an empty train split")
    edge_users = grouped.index.get_level_values("u_idx").to_numpy(np.int64)
    edge_items = grouped.index.get_level_values("i_idx").to_numpy(np.int64)
    if edge_users.min() < 0 or edge_users.max() >= n_users:
        raise ValueError("train user index is outside n_users")
    if edge_items.min() < 0 or edge_items.max() >= n_items:
        raise ValueError("train item index is outside n_items")

    edge_spend = grouped["spend"].to_numpy(np.float64)
    mean_unit_price = float(pd.to_numeric(train["up"], errors="raise").mean())
    if not np.isfinite(mean_unit_price) or mean_unit_price <= 0:
        raise ValueError("mean train unit price must be finite and positive")
    spend_signal = np.log1p(edge_spend / mean_unit_price)
    edge_clv = clv_gate[edge_users]

    user_clv_weights = _mean_one(1.0 + alpha * edge_clv)
    spend_only_weights = _mean_one(1.0 + alpha * spend_signal)
    clv_spend_weights = _mean_one(1.0 + alpha * edge_clv * spend_signal)

    item_degree = np.bincount(edge_items, minlength=n_items).astype(np.float64)
    diagnostics = {
        "alpha": float(alpha),
        "n_edges": int(len(edge_users)),
        "mean_unit_price": mean_unit_price,
        "clv_gate": _stats(edge_clv),
        "edge_spend": _stats(edge_spend),
        "spend_signal": _stats(spend_signal),
        "user_clv_weights": _stats(user_clv_weights),
        "spend_only_weights": _stats(spend_only_weights),
        "clv_spend_weights": _stats(clv_spend_weights),
        "edge_clv_spend_spearman": _safe_spearman(edge_clv, edge_spend),
        "edge_spend_item_degree_spearman": _safe_spearman(
            edge_spend, item_degree[edge_items]
        ),
        "clv_spend_weight_item_degree_spearman": _safe_spearman(
            clv_spend_weights, item_degree[edge_items]
        ),
    }
    return DirectCLVValueGraph(
        edge_users=edge_users,
        edge_items=edge_items,
        user_clv_weights=user_clv_weights,
        spend_only_weights=spend_only_weights,
        clv_spend_weights=clv_spend_weights,
        clv_gate=clv_gate.astype(np.float32),
        edge_spend=edge_spend.astype(np.float32),
        spend_signal=spend_signal.astype(np.float32),
        diagnostics=diagnostics,
    )
