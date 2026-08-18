"""Train-only M3 graphs that transfer CLV N/V structure without memorizing it.

Two graph interventions share the exact binary M1 edge set:

* ``n_transfer``: customer transaction-activity percentile x smoothed category
  repeatability.  Category-level repeatability lets recurrence information reach
  unseen items in a recurrent category instead of thickening only an already
  purchased user-item edge.
* ``v_contribution``: customer mean-basket-value percentile x the item's mean
  share of that customer's basket value.  This measures item contribution, not
  the total value of every basket in which the item happened to appear.

Both signals are rank transformed, converted to positive mean-one weights, and
matched on the standard deviation of the resulting normalized propagation
coefficient relative to the binary graph.  No validation/test labels are used.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import rankdata


DEFAULT_BETA_CAP = 0.25
DEFAULT_COMPOSITION_BETA_CAP = 2.0
DEFAULT_CATEGORY_PRIOR_STRENGTH = 20.0


@dataclass(frozen=True)
class M3TransferGraphWeights:
    edge_users: np.ndarray
    edge_items: np.ndarray
    n_weights: np.ndarray
    v_weights: np.ndarray
    clv_composition_weights: np.ndarray
    n_signal: np.ndarray
    v_signal: np.ndarray
    clv_composition_signal: np.ndarray
    q_n: np.ndarray
    q_v: np.ndarray
    q_clv: np.ndarray
    pi_n: np.ndarray
    pi_v: np.ndarray
    diagnostics: dict


def _percentile(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return np.empty(0, dtype=np.float64)
    return (rankdata(values, method="average") - 0.5) / len(values)


def _customer_age(first_time: pd.Series, end_time) -> np.ndarray:
    if pd.api.types.is_datetime64_any_dtype(first_time.dtype):
        return (pd.Timestamp(end_time) - first_time).dt.days.to_numpy(np.float64)
    return (
        float(end_time)
        - pd.to_numeric(first_time, errors="raise").to_numpy(np.float64)
    )


def _rank_signal(signal: np.ndarray) -> np.ndarray:
    """Map an edge signal to a bounded, centred rank score in [-1, 1]."""
    if len(signal) == 0:
        return np.empty(0, dtype=np.float64)
    return 2.0 * _percentile(signal) - 1.0


def _mean_one_exponential(z: np.ndarray, beta: float) -> np.ndarray:
    raw = np.exp(float(beta) * np.asarray(z, dtype=np.float64))
    return raw / raw.mean()


def _user_mean_one_exponential(
    z: np.ndarray, beta: float, edge_users: np.ndarray, n_users: int
) -> np.ndarray:
    """Positive weights whose mean is one inside every user's neighborhood."""
    raw = np.exp(float(beta) * np.asarray(z, dtype=np.float64))
    total = np.bincount(edge_users, weights=raw, minlength=n_users)
    count = np.bincount(edge_users, minlength=n_users)
    user_mean = np.ones(n_users, dtype=np.float64)
    valid = count > 0
    user_mean[valid] = total[valid] / count[valid]
    return raw / user_mean[edge_users]


def normalized_propagation_strength(
    edge_users: np.ndarray,
    edge_items: np.ndarray,
    weights: np.ndarray,
    n_users: int,
    n_items: int,
) -> float:
    """Std of log(weighted normalized coefficient / binary coefficient)."""
    edge_users = np.asarray(edge_users, dtype=np.int64)
    edge_items = np.asarray(edge_items, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float64)
    degree_u = np.bincount(edge_users, minlength=n_users).astype(np.float64)
    degree_i = np.bincount(edge_items, minlength=n_items).astype(np.float64)
    weighted_u = np.bincount(
        edge_users, weights=weights, minlength=n_users
    ).astype(np.float64)
    weighted_i = np.bincount(
        edge_items, weights=weights, minlength=n_items
    ).astype(np.float64)
    ratio = weights * np.sqrt(
        (degree_u[edge_users] * degree_i[edge_items])
        / (weighted_u[edge_users] * weighted_i[edge_items])
    )
    return float(np.std(np.log(np.maximum(ratio, 1e-12))))


def _match_beta(
    z: np.ndarray,
    target_strength: float,
    edge_users: np.ndarray,
    edge_items: np.ndarray,
    n_users: int,
    n_items: int,
    beta_cap: float,
) -> tuple[float, np.ndarray, float]:
    if target_strength <= 1e-12:
        weights = np.ones(len(z), dtype=np.float64)
        return 0.0, weights, 0.0
    lo, hi = 0.0, float(beta_cap)
    for _ in range(48):
        mid = (lo + hi) / 2.0
        weights = _mean_one_exponential(z, mid)
        strength = normalized_propagation_strength(
            edge_users, edge_items, weights, n_users, n_items
        )
        if strength < target_strength:
            lo = mid
        else:
            hi = mid
    beta = (lo + hi) / 2.0
    weights = _mean_one_exponential(z, beta)
    strength = normalized_propagation_strength(
        edge_users, edge_items, weights, n_users, n_items
    )
    return beta, weights, strength


def _match_user_normalized_beta(
    z: np.ndarray,
    target_strength: float,
    edge_users: np.ndarray,
    edge_items: np.ndarray,
    n_users: int,
    n_items: int,
    beta_cap: float,
) -> tuple[float, np.ndarray, float]:
    """Match train-only propagation strength with per-user mean-one weights."""
    if target_strength <= 1e-12:
        weights = np.ones(len(z), dtype=np.float64)
        return 0.0, weights, 0.0
    cap_weights = _user_mean_one_exponential(z, beta_cap, edge_users, n_users)
    cap_strength = normalized_propagation_strength(
        edge_users, edge_items, cap_weights, n_users, n_items
    )
    if cap_strength <= target_strength:
        return float(beta_cap), cap_weights, cap_strength
    lo, hi = 0.0, float(beta_cap)
    for _ in range(48):
        mid = (lo + hi) / 2.0
        weights = _user_mean_one_exponential(z, mid, edge_users, n_users)
        strength = normalized_propagation_strength(
            edge_users, edge_items, weights, n_users, n_items
        )
        if strength < target_strength:
            lo = mid
        else:
            hi = mid
    beta = (lo + hi) / 2.0
    weights = _user_mean_one_exponential(z, beta, edge_users, n_users)
    strength = normalized_propagation_strength(
        edge_users, edge_items, weights, n_users, n_items
    )
    return beta, weights, strength


def _weight_summary(weights: np.ndarray) -> dict:
    return {
        "mean": float(weights.mean()),
        "std": float(weights.std()),
        "min": float(weights.min()),
        "median": float(np.median(weights)),
        "max": float(weights.max()),
        "p01": float(np.percentile(weights, 1)),
        "p99": float(np.percentile(weights, 99)),
    }


def build_m3_transfer_graphs(
    train: pd.DataFrame,
    n_users: int,
    n_items: int,
    *,
    beta_cap: float = DEFAULT_BETA_CAP,
    composition_beta_cap: float = DEFAULT_COMPOSITION_BETA_CAP,
    category_prior_strength: float = DEFAULT_CATEGORY_PRIOR_STRENGTH,
) -> M3TransferGraphWeights:
    """Build matched N-transfer and V-contribution weights from train only."""
    required = {"u_idx", "i_idx", "b_raw", "cat_idx", "t", "v"}
    missing = sorted(required.difference(train.columns))
    if missing:
        raise ValueError(f"M3 transfer graph 필수 컬럼 누락: {', '.join(missing)}")
    if train.empty:
        raise ValueError("M3 transfer graph에 사용할 train 행이 없습니다")
    if n_users <= 0 or n_items <= 0:
        raise ValueError("n_users와 n_items는 양수여야 합니다")
    if beta_cap <= 0 or composition_beta_cap <= 0 or category_prior_strength < 0:
        raise ValueError(
            "beta_cap들은 양수, category_prior_strength는 0 이상이어야 합니다"
        )
    if not np.isfinite(train["v"].to_numpy(np.float64)).all():
        raise ValueError("train 거래가치에 비유한 값이 있습니다")

    users = train["u_idx"].to_numpy(np.int64)
    items = train["i_idx"].to_numpy(np.int64)
    if users.min() < 0 or users.max() >= n_users:
        raise ValueError("u_idx가 n_users 범위를 벗어났습니다")
    if items.min() < 0 or items.max() >= n_items:
        raise ValueError("i_idx가 n_items 범위를 벗어났습니다")

    # Customer-side CLV components: transaction activity N and value per
    # transaction V.  These are historical descriptors, not future CLV labels.
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
    activity = np.maximum(
        user_stats["basket_count"].to_numpy(np.float64) - 1.0, 0.0
    ) / age
    q_n = np.zeros(n_users, dtype=np.float64)
    q_v = np.zeros(n_users, dtype=np.float64)
    q_clv = np.zeros(n_users, dtype=np.float64)
    user_ids = user_stats.index.to_numpy(np.int64)
    q_n[user_ids] = _percentile(activity)
    mean_basket_value = user_stats["mean_basket_value"].to_numpy(np.float64)
    q_v[user_ids] = _percentile(mean_basket_value)
    # A coherent decomposition must use the same N/V quantities for both the
    # total CLV level and the composition shares.  Ratios of independent
    # percentiles are not a decomposition of N*V.  In log space the stabilized
    # product is additive, so pi_N/pi_V are literal shares of that same total.
    transaction_count = np.maximum(
        user_stats["basket_count"].to_numpy(np.float64) - 1.0, 0.0
    )
    log_n = np.log1p(transaction_count)
    log_v = np.log1p(np.maximum(mean_basket_value, 0.0))
    log_clv = log_n + log_v
    q_clv[user_ids] = _percentile(log_clv)
    eps = 1e-6
    pi_n = np.zeros(n_users, dtype=np.float64)
    pi_v = np.zeros(n_users, dtype=np.float64)
    positive_total = log_clv > eps
    local_pi_n = np.full(len(user_ids), 0.5, dtype=np.float64)
    local_pi_n[positive_total] = (
        log_n[positive_total] / log_clv[positive_total]
    )
    pi_n[user_ids] = local_pi_n
    pi_v[user_ids] = 1.0 - local_pi_n

    # Shared unique user-item edge order.  This must match M1 exactly.
    edge = (
        train[["u_idx", "i_idx"]]
        .drop_duplicates()
        .sort_values(["u_idx", "i_idx"])
        .reset_index(drop=True)
    )
    edge_users = edge["u_idx"].to_numpy(np.int64)
    edge_items = edge["i_idx"].to_numpy(np.int64)
    edge_keys = edge_users * n_items + edge_items
    if len(np.unique(edge_keys)) != len(edge_keys) or np.any(np.diff(edge_keys) <= 0):
        raise RuntimeError("M3 transfer graph 엣지가 고유·오름차순가 아닙니다")

    # N relation: smoothed probability that a category purchase is followed by
    # another basket containing that category for the same customer.
    user_basket_category = train[["u_idx", "b_raw", "cat_idx"]].drop_duplicates()
    user_category = (
        user_basket_category.groupby(["u_idx", "cat_idx"], sort=False)
        .size()
        .rename("basket_count")
        .reset_index()
    )
    user_category["repeat_events"] = np.maximum(
        user_category["basket_count"].to_numpy(np.float64) - 1.0, 0.0
    )
    category = user_category.groupby("cat_idx", sort=True).agg(
        repeat_events=("repeat_events", "sum"),
        basket_events=("basket_count", "sum"),
    )
    global_repeatability = float(
        category["repeat_events"].sum() / category["basket_events"].sum()
    )
    category["repeatability"] = (
        category["repeat_events"]
        + category_prior_strength * global_repeatability
    ) / (category["basket_events"] + category_prior_strength)
    item_category = train.groupby("i_idx", sort=True)["cat_idx"].agg(
        lambda values: int(values.mode().iloc[0])
    )
    edge_category = item_category.reindex(edge_items).to_numpy(np.int64)
    category_repeatability = category["repeatability"].reindex(
        edge_category
    ).to_numpy(np.float64)
    n_signal = q_n[edge_users] * category_repeatability

    # V relation: mean share of the customer's basket value attributable to
    # the item.  Non-positive baskets cannot define a monetary share and
    # contribute zero; their frequency is reported for audit.
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
    valid_basket = line["basket_value"].to_numpy(np.float64) > 0
    line["share"] = 0.0
    line.loc[valid_basket, "share"] = np.clip(
        line.loc[valid_basket, "line_value"].to_numpy(np.float64)
        / line.loc[valid_basket, "basket_value"].to_numpy(np.float64),
        0.0,
        None,
    )
    contribution = (
        line.groupby(["u_idx", "i_idx"], sort=True)["share"]
        .mean()
        .reindex(pd.MultiIndex.from_arrays([edge_users, edge_items]))
        .to_numpy(np.float64)
    )
    v_signal = q_v[edge_users] * contribution

    if not all(
        np.isfinite(values).all()
        for values in (n_signal, v_signal, category_repeatability, contribution)
    ):
        raise RuntimeError("M3 transfer graph signal에 비유한 값이 있습니다")

    z_n, z_v = _rank_signal(n_signal), _rank_signal(v_signal)
    n_cap_weights = _mean_one_exponential(z_n, beta_cap)
    v_cap_weights = _mean_one_exponential(z_v, beta_cap)
    n_cap_strength = normalized_propagation_strength(
        edge_users, edge_items, n_cap_weights, n_users, n_items
    )
    v_cap_strength = normalized_propagation_strength(
        edge_users, edge_items, v_cap_weights, n_users, n_items
    )
    target_strength = min(n_cap_strength, v_cap_strength)
    beta_n, n_weights, n_strength = _match_beta(
        z_n, target_strength, edge_users, edge_items, n_users, n_items, beta_cap
    )
    beta_v, v_weights, v_strength = _match_beta(
        z_v, target_strength, edge_users, edge_items, n_users, n_items, beta_cap
    )

    # Full-CLV compositional relation:
    #   magnitude = percentile(log(1+N_u) + log(1+V_u))
    #   composition = user-specific N/V shares
    #   direction = corrected N-transfer and V-contribution edge relations.
    # Unlike an equal N+V sum, the N relation is concentrated on N-driven
    # customers instead of diluting the successful V relation for everyone.
    clv_composition_signal = q_clv[edge_users] * (
        pi_n[edge_users] * z_n + pi_v[edge_users] * z_v
    )
    # The following mean-one normalization preserves each user's total outgoing
    # edge mass.  Therefore q_C changes within-neighborhood sharpness
    # (temperature), not the user's loss weight or all of their edges at once.
    # That is the graph-propagation intervention that distinguishes M3 from M4.
    beta_clv, clv_composition_weights, clv_strength = _match_user_normalized_beta(
        clv_composition_signal,
        target_strength,
        edge_users,
        edge_items,
        n_users,
        n_items,
        composition_beta_cap,
    )

    diagnostics = {
        "n_edges": int(len(edge)),
        "category_prior_strength": float(category_prior_strength),
        "category_global_repeatability": global_repeatability,
        "category_repeatability_min": float(category["repeatability"].min()),
        "category_repeatability_max": float(category["repeatability"].max()),
        "nonpositive_basket_line_share": float((~valid_basket).mean()),
        "q_n_unique": int(np.unique(q_n[user_ids]).size),
        "q_v_unique": int(np.unique(q_v[user_ids]).size),
        "q_clv_unique": int(np.unique(q_clv[user_ids]).size),
        "log_clv_min": float(log_clv.min()),
        "log_clv_max": float(log_clv.max()),
        "pi_n_mean": float(pi_n[user_ids].mean()),
        "pi_n_std": float(pi_n[user_ids].std()),
        "pi_v_mean": float(pi_v[user_ids].mean()),
        "pi_v_std": float(pi_v[user_ids].std()),
        "n_signal_zero_share": float((n_signal == 0).mean()),
        "v_signal_zero_share": float((v_signal == 0).mean()),
        "beta_cap": float(beta_cap),
        "composition_beta_cap": float(composition_beta_cap),
        "beta_n": float(beta_n),
        "beta_v": float(beta_v),
        "beta_clv_composition": float(beta_clv),
        "target_propagation_strength": float(target_strength),
        "n_propagation_strength": float(n_strength),
        "v_propagation_strength": float(v_strength),
        "clv_composition_propagation_strength": float(clv_strength),
        "n_weights": _weight_summary(n_weights),
        "v_weights": _weight_summary(v_weights),
        "clv_composition_weights": _weight_summary(clv_composition_weights),
    }
    return M3TransferGraphWeights(
        edge_users=edge_users,
        edge_items=edge_items,
        n_weights=n_weights.astype(np.float32),
        v_weights=v_weights.astype(np.float32),
        clv_composition_weights=clv_composition_weights.astype(np.float32),
        n_signal=n_signal.astype(np.float32),
        v_signal=v_signal.astype(np.float32),
        clv_composition_signal=clv_composition_signal.astype(np.float32),
        q_n=q_n.astype(np.float32),
        q_v=q_v.astype(np.float32),
        q_clv=q_clv.astype(np.float32),
        pi_n=pi_n.astype(np.float32),
        pi_v=pi_v.astype(np.float32),
        diagnostics=diagnostics,
    )
