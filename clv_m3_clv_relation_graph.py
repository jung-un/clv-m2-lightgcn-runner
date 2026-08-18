"""Train-only edge weights for two scalar-CLV M3 graph interventions.

The shared M1 user-item edge set is unchanged.  Four weighted variants isolate
two questions:

* Does scalar historical CLV control how far propagation moves from a binary
  graph toward a price-free relationship graph?
* Does using scalar historical CLV inside the edge relationship itself add
  value beyond the same relationship with CLV removed?

All four variants preserve mean outgoing edge mass one inside every user.
They change LightGCN propagation only; the BPR objective and negative sampling
remain unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from clv_m3_transfer_graph import normalized_propagation_strength

DEFAULT_TARGET_STRENGTH = 0.075
DEFAULT_BETA_CAP = 4.0


@dataclass(frozen=True)
class CLVRelationGraph:
    edge_users: np.ndarray
    edge_items: np.ndarray
    relation_only_weights: np.ndarray
    clv_gate_weights: np.ndarray
    allocated_relation_only_weights: np.ndarray
    clv_allocated_gate_weights: np.ndarray
    clv_percentile: np.ndarray
    relation_signal: np.ndarray
    allocated_relation_signal: np.ndarray
    diagnostics: dict


def _safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.size == 0 or np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return 0.0
    value = spearmanr(left, right).statistic
    return float(value) if np.isfinite(value) else 0.0


def _stats(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "p01": float(np.percentile(values, 1)),
        "median": float(np.median(values)),
        "p99": float(np.percentile(values, 99)),
        "max": float(values.max()),
    }


def _percentile_gate(clv: np.ndarray) -> np.ndarray:
    clv = np.asarray(clv, dtype=np.float64)
    valid = np.isfinite(clv)
    if not valid.any():
        raise ValueError("historical CLV is missing for every user")
    gate = np.zeros(len(clv), dtype=np.float64)
    gate[valid] = (rankdata(clv[valid], method="average") - 0.5) / valid.sum()
    return gate


def _within_user_rank(
    values: np.ndarray, edge_users: np.ndarray, n_users: int
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    out = np.zeros(len(values), dtype=np.float64)
    order = np.argsort(edge_users, kind="stable")
    sorted_users = edge_users[order]
    starts = np.flatnonzero(
        np.r_[True, sorted_users[1:] != sorted_users[:-1]]
    )
    stops = np.r_[starts[1:], len(order)]
    for start, stop in zip(starts, stops):
        idx = order[start:stop]
        n = len(idx)
        if n > 1:
            out[idx] = 2.0 * (rankdata(values[idx], method="average") - 0.5) / n - 1.0
    if len(np.unique(edge_users)) > n_users:
        raise ValueError("edge users exceed n_users")
    return out


def _user_mean_one(
    raw: np.ndarray, edge_users: np.ndarray, n_users: int
) -> np.ndarray:
    total = np.bincount(edge_users, weights=raw, minlength=n_users)
    count = np.bincount(edge_users, minlength=n_users)
    mean = np.ones(n_users, dtype=np.float64)
    valid = count > 0
    mean[valid] = total[valid] / count[valid]
    return raw / mean[edge_users]


def _relationship_weights(
    signal: np.ndarray, edge_users: np.ndarray, n_users: int, beta: float
) -> np.ndarray:
    ranked = _within_user_rank(signal, edge_users, n_users)
    return _user_mean_one(np.exp(beta * ranked), edge_users, n_users)


def _match_strength(
    signal: np.ndarray,
    gate: np.ndarray | None,
    target_strength: float,
    edge_users: np.ndarray,
    edge_items: np.ndarray,
    n_users: int,
    n_items: int,
    beta_cap: float,
) -> tuple[float, np.ndarray, float]:
    def weights_at(beta: float) -> np.ndarray:
        relation = _relationship_weights(signal, edge_users, n_users, beta)
        return relation if gate is None else 1.0 + gate * (relation - 1.0)

    cap_weights = weights_at(beta_cap)
    cap_strength = normalized_propagation_strength(
        edge_users, edge_items, cap_weights, n_users, n_items
    )
    if cap_strength <= target_strength:
        return float(beta_cap), cap_weights, cap_strength
    lo, hi = 0.0, float(beta_cap)
    for _ in range(48):
        mid = (lo + hi) / 2.0
        weights = weights_at(mid)
        strength = normalized_propagation_strength(
            edge_users, edge_items, weights, n_users, n_items
        )
        if strength < target_strength:
            lo = mid
        else:
            hi = mid
    beta = (lo + hi) / 2.0
    weights = weights_at(beta)
    strength = normalized_propagation_strength(
        edge_users, edge_items, weights, n_users, n_items
    )
    return beta, weights, strength


def _item_mean(
    values: np.ndarray,
    edge_items: np.ndarray,
    n_items: int,
    prior_strength: float,
) -> np.ndarray:
    total = np.bincount(edge_items, weights=values, minlength=n_items)
    count = np.bincount(edge_items, minlength=n_items)
    global_mean = float(values.mean())
    return (total + prior_strength * global_mean) / (count + prior_strength)


def build_clv_relation_graph(
    train: pd.DataFrame,
    n_users: int,
    n_items: int,
    clv: np.ndarray,
    *,
    target_strength: float = DEFAULT_TARGET_STRENGTH,
    beta_cap: float = DEFAULT_BETA_CAP,
    prior_strength: float = 20.0,
    epsilon: float = 1e-8,
) -> CLVRelationGraph:
    """Build two CLV graph proposals and their CLV-free controls."""
    required = {"u_idx", "i_idx", "up"}
    missing = required - set(train.columns)
    if missing:
        raise ValueError(f"CLV relation graph requires columns {sorted(missing)}")
    if train.empty:
        raise ValueError("cannot build a graph from an empty train split")
    if target_strength <= 0 or not np.isfinite(target_strength):
        raise ValueError("target_strength must be finite and positive")
    if beta_cap <= 0 or not np.isfinite(beta_cap):
        raise ValueError("beta_cap must be finite and positive")
    if prior_strength < 0 or not np.isfinite(prior_strength):
        raise ValueError("prior_strength must be finite and non-negative")
    if epsilon <= 0 or not np.isfinite(epsilon):
        raise ValueError("epsilon must be finite and positive")

    grouped = train.groupby(["u_idx", "i_idx"], sort=True).size().rename("count")
    edge_users = grouped.index.get_level_values("u_idx").to_numpy(np.int64)
    edge_items = grouped.index.get_level_values("i_idx").to_numpy(np.int64)
    count_ui = grouped.to_numpy(np.float64)
    if edge_users.min() < 0 or edge_users.max() >= n_users:
        raise ValueError("train user index is outside n_users")
    if edge_items.min() < 0 or edge_items.max() >= n_items:
        raise ValueError("train item index is outside n_items")

    count_u = np.bincount(edge_users, weights=count_ui, minlength=n_users)
    count_i = np.bincount(edge_items, weights=count_ui, minlength=n_items)
    total_count = float(count_ui.sum())
    expected = count_u[edge_users] * count_i[edge_items] / total_count
    relation_signal = np.log((count_ui + 0.5) / (expected + 0.5))
    clv_percentile = _percentile_gate(np.asarray(clv, dtype=np.float64))
    clv_gate = clv_percentile[edge_users]
    beta_relation, relation_only, relation_strength = _match_strength(
        relation_signal,
        None,
        target_strength,
        edge_users,
        edge_items,
        n_users,
        n_items,
        beta_cap,
    )
    beta_gate, clv_gate_weights, gate_strength = _match_strength(
        relation_signal,
        clv_gate,
        target_strength,
        edge_users,
        edge_items,
        n_users,
        n_items,
        beta_cap,
    )

    user_share = count_ui / np.maximum(count_u[edge_users], 1.0)
    share_item_mean = _item_mean(
        user_share, edge_items, n_items, prior_strength
    )
    allocated_control_signal = np.log(
        (user_share + epsilon) / (share_item_mean[edge_items] + epsilon)
    )
    beta_allocated_control, allocated_relation_only, allocated_control_strength = (
        _match_strength(
            allocated_control_signal,
            None,
            target_strength,
            edge_users,
            edge_items,
            n_users,
            n_items,
            beta_cap,
        )
    )

    allocated_clv = clv_percentile[edge_users] * user_share
    allocated_item_mean = _item_mean(
        allocated_clv, edge_items, n_items, prior_strength
    )
    allocated_relation_signal = np.log(
        (allocated_clv + epsilon) / (allocated_item_mean[edge_items] + epsilon)
    )
    beta_allocated_gate, clv_allocated_gate, allocated_gate_strength = (
        _match_strength(
            allocated_relation_signal,
            clv_gate,
            target_strength,
            edge_users,
            edge_items,
            n_users,
            n_items,
            beta_cap,
        )
    )

    arrays = {
        "relation_only_weights": relation_only,
        "clv_gate_weights": clv_gate_weights,
        "allocated_relation_only_weights": allocated_relation_only,
        "clv_allocated_gate_weights": clv_allocated_gate,
    }
    for name, values in arrays.items():
        if not np.all(np.isfinite(values)) or np.any(values <= 0):
            raise RuntimeError(f"{name} must be finite and positive")

    item_price = (
        train.groupby("i_idx", sort=True)["up"]
        .median()
        .reindex(np.arange(n_items))
        .fillna(float(train["up"].median()))
        .rank(pct=True)
        .to_numpy(np.float64)
    )
    diagnostics = {
        "target_propagation_strength": float(target_strength),
        "beta_cap": float(beta_cap),
        "matched_beta": {
            "relation_only_weights": float(beta_relation),
            "clv_gate_weights": float(beta_gate),
            "allocated_relation_only_weights": float(beta_allocated_control),
            "clv_allocated_gate_weights": float(beta_allocated_gate),
        },
        "prior_strength": float(prior_strength),
        "n_edges": int(len(edge_users)),
        "clv_percentile": _stats(clv_percentile[np.isfinite(clv)]),
        "relation_signal": _stats(relation_signal),
        "allocated_control_signal": _stats(allocated_control_signal),
        "allocated_relation_signal": _stats(allocated_relation_signal),
        "relation_signal_item_price_spearman": _safe_spearman(
            relation_signal, item_price[edge_items]
        ),
        "allocated_relation_signal_item_price_spearman": _safe_spearman(
            allocated_relation_signal, item_price[edge_items]
        ),
    }
    for name, values in arrays.items():
        diagnostics[name] = _stats(values)
        diagnostics[f"{name}_item_price_spearman"] = _safe_spearman(
            values, item_price[edge_items]
        )
        diagnostics[f"{name}_propagation_strength"] = normalized_propagation_strength(
            edge_users, edge_items, values, n_users, n_items
        )

    diagnostics["matched_strength"] = {
        "relation_only_weights": float(relation_strength),
        "clv_gate_weights": float(gate_strength),
        "allocated_relation_only_weights": float(allocated_control_strength),
        "clv_allocated_gate_weights": float(allocated_gate_strength),
    }

    return CLVRelationGraph(
        edge_users=edge_users,
        edge_items=edge_items,
        relation_only_weights=relation_only.astype(np.float32),
        clv_gate_weights=clv_gate_weights.astype(np.float32),
        allocated_relation_only_weights=allocated_relation_only.astype(np.float32),
        clv_allocated_gate_weights=clv_allocated_gate.astype(np.float32),
        clv_percentile=clv_percentile.astype(np.float32),
        relation_signal=relation_signal.astype(np.float32),
        allocated_relation_signal=allocated_relation_signal.astype(np.float32),
        diagnostics=diagnostics,
    )
