"""Reallocate one user-side LightGCN hop away from repeated train items.

The historical purchase-frequency component ``q_N`` is kept unchanged.  It
controls how strongly each user's fixed first-hop coefficient mass is moved
from frequently repeated train items toward less-repeated train items.  The
edge set and every user's total first-hop mass remain identical to the binary
LightGCN operator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from clv_m3_directional_value_graph import (
    build_mass_preserving_coefficients,
    first_hop_strength,
)


DEFAULT_TARGET_STRENGTH = 0.075
DEFAULT_BETA_CAP = 20.0


@dataclass(frozen=True)
class RepeatFrequencyFirstHop:
    edge_users: np.ndarray
    edge_items: np.ndarray
    base_coefficients: np.ndarray
    adjusted_coefficients: np.ndarray
    edge_repeat_count: np.ndarray
    less_repeated_relation: np.ndarray
    q_n: np.ndarray
    diagnostics: dict


def _stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "median": float(np.median(values)),
        "max": float(values.max()),
    }


def _safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    finite = np.isfinite(left) & np.isfinite(right)
    if finite.sum() < 2 or left[finite].std() <= 1e-12 or right[finite].std() <= 1e-12:
        return 0.0
    value = spearmanr(left[finite], right[finite]).statistic
    return float(value) if np.isfinite(value) else 0.0


def _within_user_centered_rank(
    values: np.ndarray,
    edge_users: np.ndarray,
    *,
    n_users: int,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    edge_users = np.asarray(edge_users, dtype=np.int64)
    if values.shape != edge_users.shape:
        raise ValueError("values and edge_users must have the same shape")
    output = np.zeros(len(values), dtype=np.float64)
    order = np.argsort(edge_users, kind="stable")
    sorted_users = edge_users[order]
    starts = np.flatnonzero(np.r_[True, sorted_users[1:] != sorted_users[:-1]])
    stops = np.r_[starts[1:], len(order)]
    for start, stop in zip(starts, stops, strict=True):
        index = order[start:stop]
        count = len(index)
        if count > 1:
            percentile = (rankdata(values[index], method="average") - 0.5) / count
            output[index] = 2.0 * percentile - 1.0
    if len(output) and (edge_users.min() < 0 or edge_users.max() >= n_users):
        raise ValueError("edge user index is outside n_users")
    return output


def _match_beta(
    base: np.ndarray,
    edge_users: np.ndarray,
    relation: np.ndarray,
    q_n: np.ndarray,
    *,
    target_strength: float,
    beta_cap: float,
    n_users: int,
) -> tuple[float, np.ndarray, float, bool]:
    def at(beta: float) -> tuple[np.ndarray, float]:
        adjusted = build_mass_preserving_coefficients(
            base,
            edge_users,
            relation,
            q_n,
            beta=beta,
            n_users=n_users,
        )
        return adjusted, first_hop_strength(adjusted, base)

    cap_adjusted, cap_strength = at(beta_cap)
    if cap_strength < target_strength:
        return float(beta_cap), cap_adjusted, cap_strength, False
    lo, hi = 0.0, float(beta_cap)
    for _ in range(60):
        mid = (lo + hi) / 2.0
        _, strength = at(mid)
        if strength < target_strength:
            lo = mid
        else:
            hi = mid
    beta = (lo + hi) / 2.0
    adjusted, strength = at(beta)
    return beta, adjusted, strength, True


def build_repeat_frequency_first_hop(
    train: pd.DataFrame,
    edge_users: np.ndarray,
    edge_items: np.ndarray,
    base_coefficients: np.ndarray,
    q_n: np.ndarray,
    clv_valid: np.ndarray,
    *,
    n_users: int,
    target_strength: float = DEFAULT_TARGET_STRENGTH,
    beta_cap: float = DEFAULT_BETA_CAP,
) -> RepeatFrequencyFirstHop:
    """Build a mass-preserving, q_N-conditioned first-hop operator."""

    required = {"u_idx", "i_idx", "b_raw"}
    missing = required - set(train.columns)
    if missing:
        raise ValueError(f"repeat-frequency M3 requires columns {sorted(missing)}")
    edge_users = np.asarray(edge_users, dtype=np.int64)
    edge_items = np.asarray(edge_items, dtype=np.int64)
    base = np.asarray(base_coefficients, dtype=np.float64)
    q_n = np.asarray(q_n, dtype=np.float64)
    valid = np.asarray(clv_valid, dtype=bool)
    if not (edge_users.ndim == edge_items.ndim == base.ndim == 1):
        raise ValueError("edge arrays must be one-dimensional")
    if not (edge_users.shape == edge_items.shape == base.shape) or not len(base):
        raise ValueError("non-empty aligned edge arrays are required")
    if q_n.shape != (n_users,) or valid.shape != (n_users,):
        raise ValueError("q_N and valid must have shape [n_users]")
    if not np.isfinite(base).all() or np.any(base <= 0.0):
        raise ValueError("base coefficients must be finite and positive")
    if not np.isfinite(q_n).all() or np.any((q_n < 0.0) | (q_n > 1.0)):
        raise ValueError("q_N must be finite and in [0,1]")
    if np.any(q_n[~valid] != 0.0):
        raise ValueError("invalid users must have q_N=0")
    if target_strength <= 0.0 or beta_cap <= 0.0:
        raise ValueError("target_strength and beta_cap must be positive")

    pair_index = pd.MultiIndex.from_arrays(
        [edge_users, edge_items], names=["u_idx", "i_idx"]
    )
    repeat_count = (
        train.groupby(["u_idx", "i_idx"], sort=True)["b_raw"]
        .nunique()
        .reindex(pair_index)
        .to_numpy(np.float64)
    )
    if not np.isfinite(repeat_count).all() or np.any(repeat_count < 1.0):
        raise RuntimeError("every binary train edge must have at least one basket")

    # Larger relation means less repeated. Equal-frequency histories receive 0.
    relation = _within_user_centered_rank(
        -np.log1p(repeat_count), edge_users, n_users=n_users
    )
    beta, adjusted, strength, target_reached = _match_beta(
        base,
        edge_users,
        relation,
        q_n,
        target_strength=target_strength,
        beta_cap=beta_cap,
        n_users=n_users,
    )

    base_mass = np.bincount(edge_users, weights=base, minlength=n_users)
    adjusted_mass = np.bincount(
        edge_users, weights=adjusted, minlength=n_users
    )
    active = base_mass > 0.0
    mass_error = np.abs(adjusted_mass[active] - base_mass[active])
    ratio = adjusted / base
    varying_user_count = int(
        pd.DataFrame({"u_idx": edge_users, "count": repeat_count})
        .groupby("u_idx")["count"]
        .agg(lambda x: float(x.max()) > float(x.min()))
        .sum()
    )
    diagnostics = {
        "definition": {
            "n_component": (
                "unchanged train-history q_N percentile of repeat transactions "
                "per customer age"
            ),
            "edge_repeat_count": "distinct train baskets per observed user-item pair",
            "relation": (
                "within-user centered midrank of -log(1+edge repeat count); "
                "larger means less repeated"
            ),
            "multiplier": "exp(beta*q_N(user)*relation(user,item))",
            "mass_constraint": (
                "sum_i adjusted[user,item] == sum_i binary_base[user,item]"
            ),
            "changed_path": "user receives item messages at layer 1 only",
        },
        "n_edges": int(len(base)),
        "valid_user_count": int(valid.sum()),
        "users_with_varying_edge_repeat_count": varying_user_count,
        "edge_repeat_count": _stats(repeat_count),
        "single_basket_edge_share": float(np.mean(repeat_count == 1.0)),
        "repeated_edge_share": float(np.mean(repeat_count > 1.0)),
        "less_repeated_relation": _stats(relation),
        "q_n": _stats(q_n[valid]),
        "beta": float(beta),
        "beta_cap": float(beta_cap),
        "target_first_hop_strength": float(target_strength),
        "first_hop_strength": float(strength),
        "target_reached": bool(target_reached),
        "max_user_mass_abs_error": float(mass_error.max(initial=0.0)),
        "changed_edge_share": float(np.mean(~np.isclose(ratio, 1.0))),
        "increased_edge_share": float(np.mean(ratio > 1.0 + 1e-8)),
        "decreased_edge_share": float(np.mean(ratio < 1.0 - 1e-8)),
        "coefficient_ratio": _stats(ratio),
        "repeat_count_vs_coefficient_ratio_spearman": _safe_spearman(
            repeat_count, ratio
        ),
    }
    return RepeatFrequencyFirstHop(
        edge_users=edge_users.copy(),
        edge_items=edge_items.copy(),
        base_coefficients=base.astype(np.float32),
        adjusted_coefficients=adjusted.astype(np.float32),
        edge_repeat_count=repeat_count.astype(np.float32),
        less_repeated_relation=relation.astype(np.float32),
        q_n=q_n.astype(np.float32),
        diagnostics=diagnostics,
    )
