"""Train-only CLV allocation across price-free user-item relationships.

The binary edge support and M1 LightGCN coefficients stay fixed.  Only the
item-receiving direction redistributes each item's existing coefficient mass
among its buyers.  User-receiving rows remain exactly M1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import torch


@dataclass(frozen=True)
class EdgeAllocatedCLVGraph:
    edge_users: np.ndarray
    edge_items: np.ndarray
    base_coefficients: np.ndarray
    item_user_coefficients: np.ndarray
    relationship_share: np.ndarray
    edge_clv_allocation: np.ndarray
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
        "median": float(np.median(values)),
        "max": float(values.max()),
    }


def _basket_keys(train: pd.DataFrame) -> list[str]:
    return ["u_idx", "b_raw"] if "b_raw" in train else ["u_idx", "t"]


def _customer_value(
    train: pd.DataFrame, n_users: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    basket = (
        train.groupby(_basket_keys(train), sort=False)["v"]
        .sum()
        .rename("basket_value")
    )
    grouped = basket.groupby(level="u_idx", sort=False)
    summary = pd.DataFrame({"n_hat": grouped.size(), "v_hat": grouped.mean()})
    summary["clv_proxy"] = summary["n_hat"] * summary["v_hat"]

    n_hat = np.full(n_users, np.nan, dtype=np.float64)
    v_hat = np.full(n_users, np.nan, dtype=np.float64)
    clv_proxy = np.full(n_users, np.nan, dtype=np.float64)
    users = summary.index.to_numpy(np.int64)
    n_hat[users] = summary["n_hat"].to_numpy(np.float64)
    v_hat[users] = summary["v_hat"].to_numpy(np.float64)
    clv_proxy[users] = summary["clv_proxy"].to_numpy(np.float64)
    return n_hat, v_hat, clv_proxy


def _safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if len(left) < 2 or np.ptp(left) == 0 or np.ptp(right) == 0:
        return 0.0
    value = spearmanr(left, right).statistic
    return 0.0 if not np.isfinite(value) else float(value)


def build_edge_allocated_clv_graph(
    train: pd.DataFrame, n_users: int, n_items: int
) -> EdgeAllocatedCLVGraph:
    required = {"u_idx", "i_idx", "t", "v"}
    missing = required - set(train.columns)
    if missing:
        raise ValueError(f"CLV edge allocation requires {sorted(missing)}")
    if train.empty:
        raise ValueError("cannot build an M3 graph from empty training data")

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

    basket_id = "b_raw" if "b_raw" in train else "t"
    frequency = (
        train[["u_idx", "i_idx", basket_id]]
        .drop_duplicates()
        .groupby(["u_idx", "i_idx"], sort=False)
        .size()
        .rename("basket_frequency")
    )
    aligned = pairs.merge(
        frequency.reset_index(), on=["u_idx", "i_idx"], how="left"
    )
    f_ui = aligned["basket_frequency"].to_numpy(np.float64)
    n_active_users = int((user_degree > 0).sum())
    idf = np.log((n_active_users + 1.0) / (item_degree[edge_items] + 1.0))
    relationship = np.log1p(f_ui) * np.maximum(idf, 0.0)
    relationship_sum = np.bincount(
        edge_users, weights=relationship, minlength=n_users
    )
    share = np.empty(len(edge_users), dtype=np.float64)
    positive = relationship_sum[edge_users] > 0
    share[positive] = relationship[positive] / relationship_sum[edge_users[positive]]
    share[~positive] = 1.0 / user_degree[edge_users[~positive]]

    n_hat, v_hat, clv_proxy = _customer_value(train, n_users)
    if not np.all(np.isfinite(clv_proxy[edge_users])):
        raise RuntimeError("active training users must have finite CLV proxies")
    edge_allocation = clv_proxy[edge_users] * share

    allocation_frame = pd.DataFrame(
        {"item": edge_items, "allocation": edge_allocation}
    )
    rank = allocation_frame.groupby("item", sort=False)["allocation"].rank(
        method="average"
    ).to_numpy(np.float64)
    within_item_percentile = (rank - 0.5) / item_degree[edge_items]
    factor = 0.5 + within_item_percentile

    base_mass = np.bincount(edge_items, weights=base, minlength=n_items)
    raw = base * factor
    raw_mass = np.bincount(edge_items, weights=raw, minlength=n_items)
    scale = np.ones(n_items, dtype=np.float64)
    present = item_degree > 0
    if np.any(raw_mass[present] <= 0):
        raise RuntimeError("edge allocation produced non-positive item mass")
    scale[present] = base_mass[present] / raw_mass[present]
    adjusted = raw * scale[edge_items]
    adjusted[item_degree[edge_items] == 1] = base[item_degree[edge_items] == 1]

    allocated_by_user = np.bincount(
        edge_users, weights=edge_allocation, minlength=n_users
    )
    active_users = user_degree > 0
    share_by_user = np.bincount(edge_users, weights=share, minlength=n_users)
    adjusted_mass = np.bincount(edge_items, weights=adjusted, minlength=n_items)
    item_mass_error = np.abs(adjusted_mass[present] - base_mass[present])

    diagnostics = {
        "definition": {
            "historical_clv_proxy": "N_hat * V_hat",
            "relationship": (
                "log(1 + distinct baskets containing edge) * "
                "log((active users + 1) / (item buyers + 1))"
            ),
            "allocation": "CLV_u * within-user relationship share",
            "item_message_mass": "exactly preserved from M1",
            "item_price_used": False,
            "free_strength_parameter": False,
        },
        "n_edges": int(len(edge_users)),
        "n_active_users": n_active_users,
        "n_active_items": int(present.sum()),
        "n_hat": _stats(n_hat[active_users]),
        "v_hat": _stats(v_hat[active_users]),
        "clv_proxy": _stats(clv_proxy[active_users]),
        "relationship_share": _stats(share),
        "edge_clv_allocation": _stats(edge_allocation),
        "item_user_coefficient": _stats(adjusted),
        "max_user_share_sum_abs_error": float(
            np.max(np.abs(share_by_user[active_users] - 1.0), initial=0.0)
        ),
        "max_user_clv_allocation_abs_error": float(
            np.max(
                np.abs(allocated_by_user[active_users] - clv_proxy[active_users]),
                initial=0.0,
            )
        ),
        "max_item_message_mass_abs_error": float(
            item_mass_error.max(initial=0.0)
        ),
        "allocation_item_degree_spearman": _safe_spearman(
            edge_allocation, item_degree[edge_items]
        ),
    }
    if "up" in train.columns:
        item_price = train.groupby("i_idx", sort=False)["up"].median()
        price_pct = np.full(n_items, 0.5, dtype=np.float64)
        price_pct[item_price.index.to_numpy(np.int64)] = item_price.rank(
            pct=True
        ).to_numpy(np.float64)
        diagnostics["allocation_item_price_percentile_spearman"] = _safe_spearman(
            edge_allocation, price_pct[edge_items]
        )

    return EdgeAllocatedCLVGraph(
        edge_users=edge_users,
        edge_items=edge_items,
        base_coefficients=base,
        item_user_coefficients=adjusted,
        relationship_share=share,
        edge_clv_allocation=edge_allocation,
        n_hat=n_hat,
        v_hat=v_hat,
        clv_proxy=clv_proxy,
        diagnostics=diagnostics,
    )


def build_directional_torch_adj(
    graph: EdgeAllocatedCLVGraph,
    n_users: int,
    n_items: int,
    device: torch.device,
) -> torch.Tensor:
    rows = np.concatenate([graph.edge_users, graph.edge_items + n_users])
    cols = np.concatenate([graph.edge_items + n_users, graph.edge_users])
    values = np.concatenate(
        [graph.base_coefficients, graph.item_user_coefficients]
    ).astype(np.float32)
    return torch.sparse_coo_tensor(
        torch.from_numpy(np.stack([rows, cols])),
        torch.from_numpy(values),
        size=(n_users + n_items, n_users + n_items),
        check_invariants=False,
    ).coalesce().to(device)
