"""Train-only user-first-hop coefficients conditioned on historical CLV.

The binary M1 edge set and its symmetric-normalized coefficients are retained.
For an observed user-item edge, the relationship signal is the item's mean
share of that user's basket value, ranked only against the user's other
observed items.  Historical CLV ``N_hat * V_hat`` controls how strongly that
within-user relationship redistributes the user's first-hop item messages.

Every user's original M1 first-hop coefficient mass is preserved exactly.
The module also builds a CLV-free relationship control and a degree-stratified
CLV shuffle at the same measured first-hop intervention strength.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
import torch


ARM_M1 = "m1"
ARM_RELATION_ONLY = "relation_only"
ARM_ACTUAL = "actual_clv"
ARM_SHUFFLE = "clv_shuffle"
ACTIVE_ARMS = (ARM_RELATION_ONLY, ARM_ACTUAL, ARM_SHUFFLE)
DEFAULT_TARGET_STRENGTH = 0.075
DEFAULT_BETA_CAP = 20.0
DEFAULT_SHUFFLE_SEED = 42
DEFAULT_SHUFFLE_DEGREE_BINS = 10


@dataclass(frozen=True)
class DirectionalValueGraph:
    edge_users: np.ndarray
    edge_items: np.ndarray
    base_coefficients: np.ndarray
    user_from_item_coefficients: dict[str, np.ndarray]
    edge_contribution: np.ndarray
    within_user_relation: np.ndarray
    n_hat: np.ndarray
    v_hat: np.ndarray
    clv_proxy: np.ndarray
    clv_percentile: np.ndarray
    clv_shuffle_percentile: np.ndarray
    clv_shuffle_stratum: np.ndarray
    diagnostics: dict


def _stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "median": float("nan"),
            "max": float("nan"),
        }
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


def _midrank_percentile(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if values.ndim != 1 or values.shape != valid.shape:
        raise ValueError("values and valid must be aligned one-dimensional arrays")
    if not valid.any() or not np.isfinite(values[valid]).all():
        raise ValueError("at least one finite historical CLV value is required")
    output = np.zeros(len(values), dtype=np.float64)
    output[valid] = (
        rankdata(values[valid], method="average") - 0.5
    ) / int(valid.sum())
    return output


def _historical_clv(
    train: pd.DataFrame, n_users: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    basket = (
        train.groupby(["u_idx", "b_raw"], sort=False)["v"]
        .sum()
        .rename("basket_value")
    )
    by_user = basket.groupby(level="u_idx", sort=True)
    summary = pd.DataFrame(
        {"n_hat": by_user.size(), "v_hat": by_user.mean()}
    )
    summary["clv_proxy"] = summary["n_hat"] * summary["v_hat"]

    n_hat = np.full(n_users, np.nan, dtype=np.float64)
    v_hat = np.full(n_users, np.nan, dtype=np.float64)
    clv_proxy = np.full(n_users, np.nan, dtype=np.float64)
    user_ids = summary.index.to_numpy(np.int64)
    if user_ids.min(initial=0) < 0 or user_ids.max(initial=-1) >= n_users:
        raise ValueError("train user index is outside n_users")
    n_hat[user_ids] = summary["n_hat"].to_numpy(np.float64)
    v_hat[user_ids] = summary["v_hat"].to_numpy(np.float64)
    clv_proxy[user_ids] = summary["clv_proxy"].to_numpy(np.float64)
    valid = np.isfinite(clv_proxy)
    return n_hat, v_hat, clv_proxy, valid


def _basket_value_contribution(
    train: pd.DataFrame,
    edge_users: np.ndarray,
    edge_items: np.ndarray,
) -> tuple[np.ndarray, float]:
    basket = (
        train.groupby(["u_idx", "b_raw"], sort=False)["v"]
        .sum()
        .rename("basket_value")
        .reset_index()
    )
    line = (
        train.groupby(["u_idx", "i_idx", "b_raw"], sort=False)["v"]
        .sum()
        .rename("line_value")
        .reset_index()
        .merge(
            basket,
            on=["u_idx", "b_raw"],
            how="left",
            validate="many_to_one",
        )
    )
    basket_value = line["basket_value"].to_numpy(np.float64)
    valid = np.isfinite(basket_value) & (basket_value > 0)
    line["share"] = 0.0
    line.loc[valid, "share"] = np.clip(
        line.loc[valid, "line_value"].to_numpy(np.float64) / basket_value[valid],
        0.0,
        None,
    )
    pair_index = pd.MultiIndex.from_arrays(
        [edge_users, edge_items], names=["u_idx", "i_idx"]
    )
    contribution = (
        line.groupby(["u_idx", "i_idx"], sort=True)["share"]
        .mean()
        .reindex(pair_index)
        .to_numpy(np.float64)
    )
    if not np.isfinite(contribution).all():
        raise RuntimeError("edge basket-value contribution contains non-finite values")
    return contribution, float((~valid).mean())


def _within_user_centered_rank(
    values: np.ndarray, edge_users: np.ndarray, n_users: int
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    edge_users = np.asarray(edge_users, dtype=np.int64)
    if values.shape != edge_users.shape:
        raise ValueError("values and edge_users must have the same shape")
    if edge_users.min(initial=0) < 0 or edge_users.max(initial=-1) >= n_users:
        raise ValueError("edge user index is outside n_users")
    output = np.zeros(len(values), dtype=np.float64)
    order = np.argsort(edge_users, kind="stable")
    sorted_users = edge_users[order]
    starts = np.flatnonzero(np.r_[True, sorted_users[1:] != sorted_users[:-1]])
    stops = np.r_[starts[1:], len(order)]
    for start, stop in zip(starts, stops, strict=True):
        index = order[start:stop]
        count = len(index)
        if count > 1:
            percentile = (
                rankdata(values[index], method="average") - 0.5
            ) / count
            output[index] = 2.0 * percentile - 1.0
    return output


def _degree_stratified_shuffle(
    values: np.ndarray,
    user_degree: np.ndarray,
    *,
    n_bins: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if n_bins <= 0:
        raise ValueError("shuffle_degree_bins must be positive")
    values = np.asarray(values, dtype=np.float64)
    user_degree = np.asarray(user_degree, dtype=np.float64)
    if values.shape != user_degree.shape:
        raise ValueError("CLV percentile and user degree must have the same shape")
    shuffled = values.copy()
    strata = np.full(len(values), -1, dtype=np.int16)
    valid_index = np.flatnonzero((user_degree > 0) & np.isfinite(values))
    if not len(valid_index):
        return shuffled, strata
    degree_rank = rankdata(user_degree[valid_index], method="average")
    assigned = np.floor(
        (degree_rank - 0.5) * n_bins / len(valid_index)
    ).astype(np.int16)
    assigned = np.minimum(assigned, n_bins - 1)
    strata[valid_index] = assigned
    rng = np.random.default_rng(seed)
    for stratum in np.unique(assigned):
        index = valid_index[assigned == stratum]
        if len(index) < 2:
            continue
        source = rng.permutation(index)
        if np.array_equal(source, index):
            source = np.roll(source, 1)
        shuffled[index] = values[source]
    return shuffled, strata


def build_mass_preserving_coefficients(
    base_coefficients: np.ndarray,
    edge_users: np.ndarray,
    relation: np.ndarray,
    user_gate: np.ndarray,
    *,
    beta: float,
    n_users: int,
) -> np.ndarray:
    """Redistribute each user's M1 first-hop mass without changing its sum."""
    base = np.asarray(base_coefficients, dtype=np.float64)
    edge_users = np.asarray(edge_users, dtype=np.int64)
    relation = np.asarray(relation, dtype=np.float64)
    gate = np.asarray(user_gate, dtype=np.float64)
    if base.shape != edge_users.shape or relation.shape != edge_users.shape:
        raise ValueError("edge coefficient arrays must be aligned")
    if gate.shape != (n_users,):
        raise ValueError("user_gate must have shape [n_users]")
    if np.any(base <= 0) or not np.isfinite(base).all():
        raise ValueError("base coefficients must be finite and positive")
    if beta < 0 or not np.isfinite(beta):
        raise ValueError("beta must be finite and non-negative")
    exponent = float(beta) * gate[edge_users] * relation
    raw = np.exp(np.clip(exponent, -60.0, 60.0))
    base_mass = np.bincount(edge_users, weights=base, minlength=n_users)
    weighted_mass = np.bincount(
        edge_users, weights=base * raw, minlength=n_users
    )
    scale = np.ones(n_users, dtype=np.float64)
    active = base_mass > 0
    if np.any(weighted_mass[active] <= 0):
        raise RuntimeError("weighted first-hop mass must remain positive")
    scale[active] = base_mass[active] / weighted_mass[active]
    return base * raw * scale[edge_users]


def first_hop_strength(
    adjusted: np.ndarray, base_coefficients: np.ndarray
) -> float:
    adjusted = np.asarray(adjusted, dtype=np.float64)
    base = np.asarray(base_coefficients, dtype=np.float64)
    if adjusted.shape != base.shape or np.any(adjusted <= 0) or np.any(base <= 0):
        raise ValueError("positive aligned coefficient arrays are required")
    return float(np.std(np.log(adjusted / base)))


def _match_beta(
    base: np.ndarray,
    edge_users: np.ndarray,
    relation: np.ndarray,
    user_gate: np.ndarray,
    *,
    target_strength: float,
    beta_cap: float,
    n_users: int,
) -> tuple[float, np.ndarray, float, bool]:
    def at(beta: float) -> tuple[np.ndarray, float]:
        coefficient = build_mass_preserving_coefficients(
            base,
            edge_users,
            relation,
            user_gate,
            beta=beta,
            n_users=n_users,
        )
        return coefficient, first_hop_strength(coefficient, base)

    cap_coefficients, cap_strength = at(beta_cap)
    if cap_strength <= target_strength:
        return float(beta_cap), cap_coefficients, cap_strength, False
    lo, hi = 0.0, float(beta_cap)
    for _ in range(60):
        mid = (lo + hi) / 2.0
        _, strength = at(mid)
        if strength < target_strength:
            lo = mid
        else:
            hi = mid
    beta = (lo + hi) / 2.0
    coefficients, strength = at(beta)
    return beta, coefficients, strength, True


def build_directional_value_graph(
    train: pd.DataFrame,
    n_users: int,
    n_items: int,
    *,
    target_strength: float = DEFAULT_TARGET_STRENGTH,
    beta_cap: float = DEFAULT_BETA_CAP,
    shuffle_seed: int = DEFAULT_SHUFFLE_SEED,
    shuffle_degree_bins: int = DEFAULT_SHUFFLE_DEGREE_BINS,
) -> DirectionalValueGraph:
    required = {"u_idx", "i_idx", "b_raw", "t", "v"}
    missing = required - set(train.columns)
    if missing:
        raise ValueError(f"directional M3 graph requires columns {sorted(missing)}")
    if train.empty or n_users <= 0 or n_items <= 0:
        raise ValueError("non-empty train and positive graph sizes are required")
    if target_strength <= 0 or not np.isfinite(target_strength):
        raise ValueError("target_strength must be finite and positive")
    if beta_cap <= 0 or not np.isfinite(beta_cap):
        raise ValueError("beta_cap must be finite and positive")
    if not np.isfinite(train["v"].to_numpy(np.float64)).all():
        raise ValueError("train purchase amount contains non-finite values")

    edge = (
        train[["u_idx", "i_idx"]]
        .drop_duplicates()
        .sort_values(["u_idx", "i_idx"], kind="stable")
        .reset_index(drop=True)
    )
    edge_users = edge["u_idx"].to_numpy(np.int64)
    edge_items = edge["i_idx"].to_numpy(np.int64)
    if edge_users.min() < 0 or edge_users.max() >= n_users:
        raise ValueError("train user index is outside n_users")
    if edge_items.min() < 0 or edge_items.max() >= n_items:
        raise ValueError("train item index is outside n_items")
    user_degree = np.bincount(edge_users, minlength=n_users).astype(np.float64)
    item_degree = np.bincount(edge_items, minlength=n_items).astype(np.float64)
    base = 1.0 / np.sqrt(user_degree[edge_users] * item_degree[edge_items])

    n_hat, v_hat, clv_proxy, clv_valid = _historical_clv(train, n_users)
    clv_percentile = _midrank_percentile(clv_proxy, clv_valid)
    contribution, nonpositive_basket_line_share = _basket_value_contribution(
        train, edge_users, edge_items
    )
    relation = _within_user_centered_rank(
        contribution, edge_users, n_users
    )
    shuffled_clv, shuffle_stratum = _degree_stratified_shuffle(
        clv_percentile,
        user_degree,
        n_bins=shuffle_degree_bins,
        seed=shuffle_seed,
    )

    gates = {
        ARM_RELATION_ONLY: np.ones(n_users, dtype=np.float64),
        ARM_ACTUAL: clv_percentile,
        ARM_SHUFFLE: shuffled_clv,
    }
    coefficients: dict[str, np.ndarray] = {ARM_M1: base.astype(np.float32)}
    arm_diagnostics: dict[str, dict] = {}
    base_mass = np.bincount(edge_users, weights=base, minlength=n_users)
    for arm in ACTIVE_ARMS:
        beta, adjusted, strength, reached = _match_beta(
            base,
            edge_users,
            relation,
            gates[arm],
            target_strength=target_strength,
            beta_cap=beta_cap,
            n_users=n_users,
        )
        adjusted_mass = np.bincount(
            edge_users, weights=adjusted, minlength=n_users
        )
        active_users = base_mass > 0
        error = np.abs(adjusted_mass[active_users] - base_mass[active_users])
        coefficients[arm] = adjusted.astype(np.float32)
        arm_diagnostics[arm] = {
            "beta": float(beta),
            "target_reached": bool(reached),
            "first_hop_strength": float(strength),
            "max_user_mass_abs_error": float(error.max(initial=0.0)),
            "coefficient_ratio": _stats(adjusted / base),
        }

    item_price = (
        train.groupby("i_idx", sort=True)["v"]
        .sum()
        .reindex(np.arange(n_items), fill_value=0.0)
        .to_numpy(np.float64)
    )
    diagnostics = {
        "definition": {
            "n_hat": "number of distinct train baskets",
            "v_hat": "mean train basket value",
            "historical_clv_proxy": "n_hat * v_hat",
            "edge_relationship": (
                "mean item share of user basket value, midranked within user"
            ),
            "active_multiplier": "exp(beta * q_CLV(user) * relation(user,item))",
            "mass_constraint": (
                "sum_i adjusted[user,item] == sum_i M1[user,item]"
            ),
            "changed_path": "user receives item messages at layer 1 only",
        },
        "n_edges": int(len(edge_users)),
        "n_active_users": int(clv_valid.sum()),
        "target_first_hop_strength": float(target_strength),
        "beta_cap": float(beta_cap),
        "nonpositive_basket_line_share": nonpositive_basket_line_share,
        "n_hat": _stats(n_hat[clv_valid]),
        "v_hat": _stats(v_hat[clv_valid]),
        "clv_proxy": _stats(clv_proxy[clv_valid]),
        "clv_percentile": _stats(clv_percentile[clv_valid]),
        "edge_contribution": _stats(contribution),
        "within_user_relation": _stats(relation),
        "shuffle": {
            "seed": int(shuffle_seed),
            "degree_bins": int(shuffle_degree_bins),
            "changed_user_count": int(
                np.count_nonzero(clv_percentile != shuffled_clv)
            ),
            "actual_vs_shuffle_spearman": _safe_spearman(
                clv_percentile[clv_valid], shuffled_clv[clv_valid]
            ),
        },
        "edge_contribution_item_total_value_spearman": _safe_spearman(
            contribution, item_price[edge_items]
        ),
        "arms": arm_diagnostics,
    }
    return DirectionalValueGraph(
        edge_users=edge_users,
        edge_items=edge_items,
        base_coefficients=base.astype(np.float32),
        user_from_item_coefficients=coefficients,
        edge_contribution=contribution.astype(np.float32),
        within_user_relation=relation.astype(np.float32),
        n_hat=n_hat.astype(np.float32),
        v_hat=v_hat.astype(np.float32),
        clv_proxy=clv_proxy.astype(np.float32),
        clv_percentile=clv_percentile.astype(np.float32),
        clv_shuffle_percentile=shuffled_clv.astype(np.float32),
        clv_shuffle_stratum=shuffle_stratum,
        diagnostics=diagnostics,
    )


def build_directional_operators(
    graph: DirectionalValueGraph,
    arm: str,
    n_users: int,
    n_items: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return base U<-I, base I<-U, and selected U<-I sparse matrices."""
    if arm not in graph.user_from_item_coefficients:
        raise ValueError(
            f"unknown arm {arm!r}; expected {tuple(graph.user_from_item_coefficients)}"
        )
    indices = torch.from_numpy(
        np.stack([graph.edge_users, graph.edge_items])
    )

    def sparse(values: np.ndarray) -> torch.Tensor:
        return torch.sparse_coo_tensor(
            indices,
            torch.from_numpy(np.asarray(values, dtype=np.float32)),
            size=(n_users, n_items),
            check_invariants=False,
        ).coalesce().to(device)

    base_user_from_item = sparse(graph.base_coefficients)
    base_item_from_user = base_user_from_item.transpose(0, 1).coalesce()
    active_user_from_item = sparse(graph.user_from_item_coefficients[arm])
    return base_user_from_item, base_item_from_user, active_user_from_item

