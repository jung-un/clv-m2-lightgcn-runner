"""Train-only CLV redistribution for the item-receiving LightGCN direction.

The binary edge set and LightGCN's symmetric-normalized coefficients are kept.
Only the composition of each item's incoming user messages changes.  For item
``i`` and user ``u`` the proposal is

    adjusted[i, u] = base[i, u] * c[u]
                       * sum_v base[i, v] / sum_v base[i, v] * c[v]

so ``c == 1`` is exactly the M1 operator and every item's incoming coefficient
mass is preserved.  User <- item propagation always uses ``base`` unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata


MODES = ("n_only", "v_only", "clv", "clv_shuffle")
DEFAULT_SHUFFLE_SEED = 20260822


@dataclass(frozen=True)
class MassPreservingCLVGraph:
    edge_users: np.ndarray
    edge_items: np.ndarray
    base_coefficients: np.ndarray
    item_user_coefficients: dict[str, np.ndarray]
    user_factors: dict[str, np.ndarray]
    n_hat: np.ndarray
    v_hat: np.ndarray
    clv_proxy: np.ndarray
    diagnostics: dict


def _stats(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "p10": float(np.percentile(values, 10)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "max": float(values.max()),
    }


def _percentile_factor(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Return mean-one ``0.5 + percentile`` factors on train-observed users."""
    values = np.asarray(values, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if values.ndim != 1 or valid.shape != values.shape:
        raise ValueError("values and valid must be aligned one-dimensional arrays")
    if not valid.any() or not np.all(np.isfinite(values[valid])):
        raise ValueError("at least one finite train-observed user value is required")

    factor = np.ones(len(values), dtype=np.float64)
    n_valid = int(valid.sum())
    percentile = (rankdata(values[valid], method="average") - 0.5) / n_valid
    raw = 0.5 + percentile
    factor[valid] = raw / raw.mean()
    return factor


def _customer_value_components(
    train: pd.DataFrame, n_users: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute N, V and N*V from train baskets only.

    N is the number of transactions and V is mean transaction value.  A source
    basket id is used when available; otherwise (user, timestamp) is one basket.
    """
    basket_keys = ["u_idx", "b_raw"] if "b_raw" in train.columns else ["u_idx", "t"]
    basket = train.groupby(basket_keys, sort=False)["v"].sum().rename("basket_value")
    by_user = basket.groupby(level="u_idx", sort=False)
    summary = pd.DataFrame({"n_hat": by_user.size(), "v_hat": by_user.mean()})
    summary["clv_proxy"] = summary["n_hat"] * summary["v_hat"]

    n_hat = np.full(n_users, np.nan, dtype=np.float64)
    v_hat = np.full(n_users, np.nan, dtype=np.float64)
    clv_proxy = np.full(n_users, np.nan, dtype=np.float64)
    user_ids = summary.index.to_numpy(np.int64)
    if user_ids.min() < 0 or user_ids.max() >= n_users:
        raise ValueError("train user index is outside n_users")
    n_hat[user_ids] = summary["n_hat"].to_numpy(np.float64)
    v_hat[user_ids] = summary["v_hat"].to_numpy(np.float64)
    clv_proxy[user_ids] = summary["clv_proxy"].to_numpy(np.float64)
    valid = np.isfinite(clv_proxy)
    return n_hat, v_hat, clv_proxy, valid


def _kish_ratio_by_item(
    values: np.ndarray, edge_items: np.ndarray, n_items: int
) -> np.ndarray:
    count = np.bincount(edge_items, minlength=n_items).astype(np.float64)
    total = np.bincount(edge_items, weights=values, minlength=n_items)
    square = np.bincount(edge_items, weights=values**2, minlength=n_items)
    out = np.full(n_items, np.nan, dtype=np.float64)
    valid = (count > 0) & (square > 0)
    out[valid] = total[valid] ** 2 / (count[valid] * square[valid])
    return out


def build_mass_preserving_clv_graph(
    train: pd.DataFrame,
    n_users: int,
    n_items: int,
    *,
    shuffle_seed: int = DEFAULT_SHUFFLE_SEED,
) -> MassPreservingCLVGraph:
    required = {"u_idx", "i_idx", "t", "v"}
    missing = required - set(train.columns)
    if missing:
        raise ValueError(f"CLV redistribution requires columns {sorted(missing)}")
    if train.empty:
        raise ValueError("cannot build a graph from an empty train split")

    pairs = train[["u_idx", "i_idx"]].drop_duplicates().sort_values(
        ["u_idx", "i_idx"], kind="stable"
    )
    edge_users = pairs["u_idx"].to_numpy(np.int64)
    edge_items = pairs["i_idx"].to_numpy(np.int64)
    if edge_users.min() < 0 or edge_users.max() >= n_users:
        raise ValueError("train user index is outside n_users")
    if edge_items.min() < 0 or edge_items.max() >= n_items:
        raise ValueError("train item index is outside n_items")

    user_degree = np.bincount(edge_users, minlength=n_users).astype(np.float64)
    item_degree = np.bincount(edge_items, minlength=n_items).astype(np.float64)
    base = 1.0 / np.sqrt(user_degree[edge_users] * item_degree[edge_items])

    n_hat, v_hat, clv_proxy, valid = _customer_value_components(train, n_users)
    factors = {
        "n_only": _percentile_factor(n_hat, valid),
        "v_only": _percentile_factor(v_hat, valid),
        "clv": _percentile_factor(clv_proxy, valid),
    }
    rng = np.random.default_rng(shuffle_seed)
    shuffled = factors["clv"].copy()
    shuffled[valid] = rng.permutation(shuffled[valid])
    factors["clv_shuffle"] = shuffled

    base_item_mass = np.bincount(edge_items, weights=base, minlength=n_items)
    adjusted: dict[str, np.ndarray] = {}
    diagnostics: dict = {
        "definition": {
            "n_hat": "number of train transactions/baskets",
            "v_hat": "mean train transaction/basket value",
            "clv_proxy": "n_hat * v_hat (equals train-period total purchase value)",
            "factor": "(0.5 + train-user percentile) / active-user mean",
            "item_mass": "sum_u adjusted[i,u] == sum_u base[i,u]",
        },
        "n_edges": int(len(edge_users)),
        "n_active_users": int(valid.sum()),
        "shuffle_seed": int(shuffle_seed),
        "n_hat": _stats(n_hat[valid]),
        "v_hat": _stats(v_hat[valid]),
        "clv_proxy": _stats(clv_proxy[valid]),
        "modes": {},
    }

    for mode, factor in factors.items():
        weighted_item_mass = np.bincount(
            edge_items, weights=base * factor[edge_users], minlength=n_items
        )
        scale = np.ones(n_items, dtype=np.float64)
        present = base_item_mass > 0
        if np.any(weighted_item_mass[present] <= 0):
            raise RuntimeError(f"{mode} produced non-positive item message mass")
        scale[present] = base_item_mass[present] / weighted_item_mass[present]
        coeff = base * factor[edge_users] * scale[edge_items]
        adjusted[mode] = coeff.astype(np.float32)

        adjusted_mass = np.bincount(edge_items, weights=coeff, minlength=n_items)
        mass_error = np.abs(adjusted_mass[present] - base_item_mass[present])
        c_kish = _kish_ratio_by_item(factor[edge_users], edge_items, n_items)
        coeff_kish = _kish_ratio_by_item(coeff, edge_items, n_items)
        diagnostics["modes"][mode] = {
            "user_factor": _stats(factor[valid]),
            "max_item_mass_abs_error": float(mass_error.max(initial=0.0)),
            "item_kish_ratio_from_user_factor": _stats(c_kish[present]),
            "item_kish_ratio_from_user_factor_median": float(
                np.nanmedian(c_kish[present])
            ),
            "item_kish_ratio_from_operator": _stats(coeff_kish[present]),
            "item_kish_ratio_from_operator_median": float(
                np.nanmedian(coeff_kish[present])
            ),
        }

    return MassPreservingCLVGraph(
        edge_users=edge_users,
        edge_items=edge_items,
        base_coefficients=base.astype(np.float32),
        item_user_coefficients=adjusted,
        user_factors={key: value.astype(np.float32) for key, value in factors.items()},
        n_hat=n_hat.astype(np.float32),
        v_hat=v_hat.astype(np.float32),
        clv_proxy=clv_proxy.astype(np.float32),
        diagnostics=diagnostics,
    )


def build_directional_torch_adj(
    graph: MassPreservingCLVGraph,
    mode: str,
    n_users: int,
    n_items: int,
    device: torch.device,
) -> torch.Tensor:
    """Build one block operator with M1 user rows and adjusted item rows."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    eu, ei = graph.edge_users, graph.edge_items
    rows = np.concatenate([eu, ei + n_users])
    cols = np.concatenate([ei + n_users, eu])
    values = np.concatenate(
        [graph.base_coefficients, graph.item_user_coefficients[mode]]
    ).astype(np.float32)
    return torch.sparse_coo_tensor(
        torch.from_numpy(np.stack([rows, cols])),
        torch.from_numpy(values),
        size=(n_users + n_items, n_users + n_items),
        check_invariants=False,
    ).coalesce().to(device)
