"""Train-only CLV N/V compositional edge weights for M3 LightGCN graphs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import rankdata


@dataclass(frozen=True)
class CLVNVGraphWeights:
    edge_users: np.ndarray
    edge_items: np.ndarray
    weights: np.ndarray
    n_relation: np.ndarray
    v_relation: np.ndarray
    n_component: np.ndarray
    v_component: np.ndarray
    q_n: np.ndarray
    q_v: np.ndarray
    diagnostics: dict


def _percentile(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return np.empty(0, dtype=np.float32)
    ranks = (rankdata(values, method="average") - 0.5) / len(values)
    return ranks.astype(np.float32)


def _customer_age(first_time: pd.Series, end_time) -> np.ndarray:
    if pd.api.types.is_datetime64_any_dtype(first_time.dtype):
        return (pd.Timestamp(end_time) - first_time).dt.days.to_numpy(np.float64)
    return (
        float(end_time) - pd.to_numeric(first_time, errors="raise").to_numpy(np.float64)
    )


def _mean_one(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    mean = float(values.mean()) if len(values) else 0.0
    if mean <= 1e-12:
        return np.zeros_like(values, dtype=np.float64)
    return values / mean


def build_clv_nv_graph(
    train: pd.DataFrame, n_users: int, n_items: int
) -> CLVNVGraphWeights:
    """Build edge weights using historical user N/V and edge N/V context.

    The returned edge order is ascending ``u_idx * n_items + i_idx`` so it can
    be checked directly against the common M1/M3 unique-edge list.
    """
    required = {"u_idx", "i_idx", "b_raw", "t", "v"}
    missing = sorted(required.difference(train.columns))
    if missing:
        raise ValueError(f"CLV-NV graph 필수 컬럼 누락: {', '.join(missing)}")
    if train.empty:
        raise ValueError("CLV-NV graph에 사용할 train 행이 없습니다")
    if n_users <= 0 or n_items <= 0:
        raise ValueError("n_users와 n_items는 양수여야 합니다")

    users = train["u_idx"].to_numpy(np.int64)
    items = train["i_idx"].to_numpy(np.int64)
    if users.min() < 0 or users.max() >= n_users:
        raise ValueError("u_idx가 n_users 범위를 벗어났습니다")
    if items.min() < 0 or items.max() >= n_items:
        raise ValueError("i_idx가 n_items 범위를 벗어났습니다")
    if not np.isfinite(train["v"].to_numpy(np.float64)).all():
        raise ValueError("train 거래가치에 비유한 값이 있습니다")

    basket = (
        train.groupby(["u_idx", "b_raw"], sort=False)
        .agg(basket_value=("v", "sum"), basket_time=("t", "min"))
        .reset_index()
    )
    user_stats = basket.groupby("u_idx", sort=True).agg(
        basket_count=("b_raw", "size"),
        first_time=("basket_time", "min"),
        mean_basket_value=("basket_value", "mean"),
    )
    age = np.maximum(_customer_age(user_stats["first_time"], train["t"].max()), 1.0)
    repeat_rate = np.maximum(
        user_stats["basket_count"].to_numpy(np.float64) - 1.0, 0.0
    ) / age

    q_n = np.zeros(n_users, dtype=np.float32)
    q_v = np.zeros(n_users, dtype=np.float32)
    user_ids = user_stats.index.to_numpy(np.int64)
    q_n[user_ids] = _percentile(repeat_rate)
    q_v[user_ids] = _percentile(
        user_stats["mean_basket_value"].to_numpy(np.float64)
    )

    presence = train[["u_idx", "i_idx", "b_raw"]].drop_duplicates()
    edge = presence.merge(
        basket[["u_idx", "b_raw", "basket_value"]],
        on=["u_idx", "b_raw"],
        how="left",
        validate="many_to_one",
    )
    edge = (
        edge.groupby(["u_idx", "i_idx"], sort=True)
        .agg(
            basket_count=("b_raw", "size"),
            mean_basket_context=("basket_value", "mean"),
        )
        .reset_index()
    )
    edge_users = edge["u_idx"].to_numpy(np.int64)
    edge_items = edge["i_idx"].to_numpy(np.int64)
    edge_keys = edge_users * n_items + edge_items
    if len(np.unique(edge_keys)) != len(edge_keys) or np.any(np.diff(edge_keys) <= 0):
        raise RuntimeError("CLV-NV graph 엣지 인덱스가 고유·오름차순이 아닙니다")

    repeat_strength = np.log1p(
        np.maximum(edge["basket_count"].to_numpy(np.float64) - 1.0, 0.0)
    )
    repeat_series = pd.Series(repeat_strength, index=edge.index)
    n_relation = np.zeros(len(edge), dtype=np.float64)
    repeat_mask = repeat_strength > 0
    if repeat_mask.any():
        repeated = edge.loc[repeat_mask, ["u_idx"]].copy()
        repeated["strength"] = repeat_series[repeat_mask].to_numpy()
        n_relation[repeat_mask] = (
            repeated.groupby("u_idx")["strength"]
            .rank(method="average", pct=True)
            .to_numpy(np.float64)
        )
    v_relation = (
        edge.groupby("u_idx")["mean_basket_context"]
        .rank(method="average", pct=True)
        .to_numpy(np.float64)
    )

    n_component = _mean_one(2.0 * q_n[edge_users] * n_relation)
    v_component = _mean_one(2.0 * q_v[edge_users] * v_relation)
    raw_weight = 1.0 + 0.5 * (n_component + v_component)
    weights = np.clip(raw_weight / raw_weight.mean(), 0.25, 4.0).astype(np.float32)
    arrays = (n_relation, v_relation, n_component, v_component, weights)
    if not all(np.isfinite(values).all() for values in arrays):
        raise RuntimeError("CLV-NV graph 계산 결과에 비유한 값이 있습니다")
    if np.any(weights <= 0):
        raise RuntimeError("CLV-NV graph 가중치는 양수여야 합니다")

    diagnostics = {
        "n_edges": int(len(edge)),
        "repeat_edge_share": float(repeat_mask.mean()),
        "q_n_unique": int(np.unique(q_n[user_ids]).size),
        "q_v_unique": int(np.unique(q_v[user_ids]).size),
        "n_component_mean": float(n_component.mean()),
        "v_component_mean": float(v_component.mean()),
        "weight_mean": float(weights.mean()),
        "weight_median": float(np.median(weights)),
        "weight_min": float(weights.min()),
        "weight_max": float(weights.max()),
        "clip_low_share": float((weights <= 0.25).mean()),
        "clip_high_share": float((weights >= 4.0).mean()),
    }
    return CLVNVGraphWeights(
        edge_users=edge_users,
        edge_items=edge_items,
        weights=weights,
        n_relation=n_relation.astype(np.float32),
        v_relation=v_relation.astype(np.float32),
        n_component=n_component.astype(np.float32),
        v_component=v_component.astype(np.float32),
        q_n=q_n,
        q_v=q_v,
        diagnostics=diagnostics,
    )
