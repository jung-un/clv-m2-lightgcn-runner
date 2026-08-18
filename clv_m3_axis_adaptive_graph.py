"""Train-only CLV factor-adaptive M3 graph.

The user-side CLV proxy is kept factorized: transaction activity controls the
N relation, while mean transaction value controls the V relation.  The N
relation is not raw repeatability.  It is an item-level, user-residualized and
category-shrunk estimate of whether a basket containing the item is followed
within the evaluation horizon by another basket containing previously unseen
items.

No validation/test rows are used.  The unique binary M1 edge set is preserved;
only positive, per-user mean-one propagation weights are changed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from clv_m3_transfer_graph import (
    DEFAULT_CATEGORY_PRIOR_STRENGTH,
    DEFAULT_COMPOSITION_BETA_CAP,
    _match_user_normalized_beta,
    _rank_signal,
    _weight_summary,
    build_m3_transfer_graphs,
)


DEFAULT_NEXT_HORIZON_DAYS = 7.0


@dataclass(frozen=True)
class M3AxisAdaptiveGraphWeights:
    edge_users: np.ndarray
    edge_items: np.ndarray
    weights: np.ndarray
    v_only_weights: np.ndarray
    signal: np.ndarray
    n_component: np.ndarray
    v_component: np.ndarray
    n_item_relation: np.ndarray
    v_edge_relation: np.ndarray
    q_n: np.ndarray
    q_v: np.ndarray
    diagnostics: dict


def _time_gap(current: pd.Series, following: pd.Series) -> np.ndarray:
    if pd.api.types.is_datetime64_any_dtype(current.dtype):
        return (following - current).dt.total_seconds().to_numpy(np.float64) / 86400.0
    return (
        pd.to_numeric(following, errors="coerce").to_numpy(np.float64)
        - pd.to_numeric(current, errors="coerce").to_numpy(np.float64)
    )


def _safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if len(left) < 2 or np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return 0.0
    return float(np.corrcoef(rankdata(left), rankdata(right))[0, 1])


def _percentiles(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return {
        "min": float(values.min()),
        "p10": float(np.percentile(values, 10)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "max": float(values.max()),
        "mean": float(values.mean()),
    }


def build_m3_axis_adaptive_graph(
    train: pd.DataFrame,
    n_users: int,
    n_items: int,
    *,
    next_horizon_days: float = DEFAULT_NEXT_HORIZON_DAYS,
    category_prior_strength: float = DEFAULT_CATEGORY_PRIOR_STRENGTH,
    beta_cap: float = DEFAULT_COMPOSITION_BETA_CAP,
) -> M3AxisAdaptiveGraphWeights:
    """Build the full N/V CLV graph from train-only chronological baskets."""
    required = {"u_idx", "i_idx", "b_raw", "cat_idx", "t", "v"}
    missing = sorted(required.difference(train.columns))
    if missing:
        raise ValueError(f"M3 axis-adaptive graph 필수 컬럼 누락: {', '.join(missing)}")
    if train.empty:
        raise ValueError("M3 axis-adaptive graph에 사용할 train 행이 없습니다")
    if n_users <= 0 or n_items <= 0:
        raise ValueError("n_users와 n_items는 양수여야 합니다")
    if next_horizon_days <= 0 or category_prior_strength < 0 or beta_cap <= 0:
        raise ValueError("시간창·beta는 양수, 축소추정 강도는 0 이상이어야 합니다")

    # Reuse the already audited edge order, customer N/V definitions and the
    # train-only matched propagation-strength reference from the successful V
    # component screening.  The old repeat-category N relation is not used.
    reference = build_m3_transfer_graphs(train, n_users, n_items)
    edge_users = reference.edge_users.astype(np.int64, copy=False)
    edge_items = reference.edge_items.astype(np.int64, copy=False)
    q_n = reference.q_n.astype(np.float64)
    q_v = reference.q_v.astype(np.float64)

    basket = (
        train.groupby(["u_idx", "b_raw"], sort=False)
        .agg(basket_time=("t", "min"), basket_value=("v", "sum"))
        .reset_index()
        .sort_values(["u_idx", "basket_time", "b_raw"], kind="stable")
        .reset_index(drop=True)
    )
    basket["basket_order"] = basket.groupby("u_idx", sort=False).cumcount()

    basket_item = (
        train[["u_idx", "b_raw", "i_idx", "cat_idx"]]
        .drop_duplicates(["u_idx", "b_raw", "i_idx"])
        .merge(
            basket[["u_idx", "b_raw", "basket_order"]],
            on=["u_idx", "b_raw"],
            how="left",
            validate="many_to_one",
        )
    )
    first_order = basket_item.groupby(["u_idx", "i_idx"], sort=False)[
        "basket_order"
    ].transform("min")
    basket_item["is_novel"] = (
        basket_item["basket_order"].to_numpy(np.int64)
        == first_order.to_numpy(np.int64)
    ).astype(np.float64)
    basket_novelty = (
        basket_item.groupby(["u_idx", "b_raw"], sort=False)
        .agg(n_items=("i_idx", "size"), n_novel=("is_novel", "sum"))
        .reset_index()
    )
    basket_novelty["novel_share"] = (
        basket_novelty["n_novel"] / basket_novelty["n_items"]
    )
    basket = basket.merge(
        basket_novelty[["u_idx", "b_raw", "novel_share"]],
        on=["u_idx", "b_raw"],
        how="left",
        validate="one_to_one",
    )
    grouped_basket = basket.groupby("u_idx", sort=False)
    basket["next_time"] = grouped_basket["basket_time"].shift(-1)
    basket["next_novel_share"] = grouped_basket["novel_share"].shift(-1)
    gap = _time_gap(basket["basket_time"], basket["next_time"])
    has_timely_next = (
        np.isfinite(gap)
        & (gap >= 0.0)
        & (gap <= float(next_horizon_days))
    )
    basket["n_outcome"] = np.where(
        has_timely_next,
        basket["next_novel_share"].fillna(0.0).to_numpy(np.float64),
        0.0,
    )

    # Remove each customer's ordinary continuation/novelty level.  An item is
    # rewarded only when baskets containing it precede more continuation with
    # unseen items than is usual for that same customer.  This avoids projecting
    # high-frequency customers onto staple/popular products.
    user_total = basket.groupby("u_idx", sort=False)["n_outcome"].transform("sum")
    user_count = basket.groupby("u_idx", sort=False)["n_outcome"].transform("size")
    global_outcome = float(basket["n_outcome"].mean())
    leave_one_out = np.full(len(basket), global_outcome, dtype=np.float64)
    repeated = user_count.to_numpy(np.int64) > 1
    leave_one_out[repeated] = (
        user_total.to_numpy(np.float64)[repeated]
        - basket["n_outcome"].to_numpy(np.float64)[repeated]
    ) / (user_count.to_numpy(np.float64)[repeated] - 1.0)
    basket["n_residual"] = basket["n_outcome"].to_numpy(np.float64) - leave_one_out

    occurrence = basket_item.merge(
        basket[["u_idx", "b_raw", "n_residual"]],
        on=["u_idx", "b_raw"],
        how="left",
        validate="many_to_one",
    )
    category_prior = occurrence.groupby("cat_idx", sort=True)["n_residual"].mean()
    item_stats = occurrence.groupby("i_idx", sort=True).agg(
        observation_count=("n_residual", "size"),
        residual_mean=("n_residual", "mean"),
        user_count=("u_idx", "nunique"),
    )
    item_category = train.groupby("i_idx", sort=True)["cat_idx"].agg(
        lambda values: int(values.mode().iloc[0])
    )
    item_ids = item_stats.index.to_numpy(np.int64)
    item_prior = category_prior.reindex(
        item_category.reindex(item_ids).to_numpy(np.int64)
    ).to_numpy(np.float64)
    item_n = item_stats["observation_count"].to_numpy(np.float64)
    item_score = (
        item_n * item_stats["residual_mean"].to_numpy(np.float64)
        + float(category_prior_strength) * item_prior
    ) / (item_n + float(category_prior_strength))
    n_item_relation = np.zeros(n_items, dtype=np.float64)
    n_item_relation[item_ids] = item_score
    z_n_item = np.zeros(n_items, dtype=np.float64)
    z_n_item[item_ids] = _rank_signal(item_score)

    # V relation: the item's mean share of this customer's basket value.  The
    # user value percentile is applied separately below, so the relation itself
    # contains no second user gate.
    line = (
        train.groupby(["u_idx", "i_idx", "b_raw"], sort=False)["v"]
        .sum()
        .rename("line_value")
        .reset_index()
        .merge(
            basket[["u_idx", "b_raw", "basket_value"]],
            on=["u_idx", "b_raw"],
            how="left",
            validate="many_to_one",
        )
    )
    valid_value = line["basket_value"].to_numpy(np.float64) > 0.0
    line["share"] = 0.0
    line.loc[valid_value, "share"] = np.clip(
        line.loc[valid_value, "line_value"].to_numpy(np.float64)
        / line.loc[valid_value, "basket_value"].to_numpy(np.float64),
        0.0,
        None,
    )
    v_edge_relation = (
        line.groupby(["u_idx", "i_idx"], sort=True)["share"]
        .mean()
        .reindex(pd.MultiIndex.from_arrays([edge_users, edge_items]))
        .to_numpy(np.float64)
    )
    z_v_edge = _rank_signal(v_edge_relation)

    n_component = q_n[edge_users] * z_n_item[edge_items]
    v_component = q_v[edge_users] * z_v_edge
    signal = n_component + v_component
    target_strength = float(reference.diagnostics["target_propagation_strength"])
    beta, weights, strength = _match_user_normalized_beta(
        signal,
        target_strength,
        edge_users,
        edge_items,
        n_users,
        n_items,
        beta_cap,
    )
    v_only_beta, v_only_weights, v_only_strength = _match_user_normalized_beta(
        v_component,
        target_strength,
        edge_users,
        edge_items,
        n_users,
        n_items,
        beta_cap,
    )

    pair_observations = (
        basket_item.groupby(["u_idx", "i_idx"], sort=False).size().to_numpy(np.int64)
    )
    item_popularity = item_stats["user_count"].to_numpy(np.float64)
    diagnostics = {
        "n_edges": int(len(edge_users)),
        "next_horizon_days": float(next_horizon_days),
        "category_prior_strength": float(category_prior_strength),
        "n_outcome_mean": global_outcome,
        "timely_next_basket_share": float(has_timely_next.mean()),
        "pair_basket_observations": _percentiles(pair_observations),
        "item_basket_observations": _percentiles(item_n),
        "n_item_relation_std": float(item_score.std()),
        "n_item_relation_unique": int(np.unique(item_score).size),
        "n_relation_popularity_spearman": _safe_spearman(
            item_score, item_popularity
        ),
        "q_n_unique": int(np.unique(q_n).size),
        "q_v_unique": int(np.unique(q_v).size),
        "beta": float(beta),
        "v_only_beta": float(v_only_beta),
        "target_propagation_strength": target_strength,
        "propagation_strength": float(strength),
        "v_only_propagation_strength": float(v_only_strength),
        "weights": _weight_summary(weights),
        "v_only_weights": _weight_summary(v_only_weights),
        "nonpositive_basket_line_share": float((~valid_value).mean()),
        "audit_note": (
            "Diagnostics are train-only descriptive outputs and do not block or "
            "select the model. N relation is user-residualized and category-shrunk."
        ),
    }
    return M3AxisAdaptiveGraphWeights(
        edge_users=edge_users,
        edge_items=edge_items,
        weights=weights.astype(np.float32),
        v_only_weights=v_only_weights.astype(np.float32),
        signal=signal.astype(np.float32),
        n_component=n_component.astype(np.float32),
        v_component=v_component.astype(np.float32),
        n_item_relation=n_item_relation.astype(np.float32),
        v_edge_relation=v_edge_relation.astype(np.float32),
        q_n=q_n.astype(np.float32),
        q_v=q_v.astype(np.float32),
        diagnostics=diagnostics,
    )
