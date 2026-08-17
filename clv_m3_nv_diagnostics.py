"""No-training diagnostics for the M3 CLV-NV weighted graph."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from clv_m3_nv_graph import CLVNVGraphWeights


def _spearman(left, right) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 3 or np.unique(left[valid]).size < 2 or np.unique(right[valid]).size < 2:
        return 0.0
    value = spearmanr(left[valid], right[valid]).correlation
    return 0.0 if not np.isfinite(value) else float(value)


def _edge_context(train: pd.DataFrame) -> pd.DataFrame:
    basket_keys = ["u_idx", "b_raw"]
    rows = train.copy()
    rows["_basket_total"] = rows.groupby(basket_keys, sort=False)["v"].transform("sum")
    rows["_item_value_share"] = rows["v"].div(
        rows["_basket_total"].where(rows["_basket_total"] > 0)
    ).fillna(0.0)
    return rows.groupby(["u_idx", "i_idx"], sort=True).agg(
        edge_row_count=("i_idx", "size"),
        edge_basket_count=("b_raw", "nunique"),
        mean_basket_context=("_basket_total", "mean"),
        mean_item_value_share=("_item_value_share", "mean"),
    )


def _item_context(train: pd.DataFrame, edge_items: np.ndarray, n_items: int) -> pd.DataFrame:
    item = train.groupby("i_idx", sort=True).agg(
        i_raw=("i_raw", "first"),
        category=("cat_raw", "first"),
        item_row_count=("i_idx", "size"),
        item_median_price=("up", "median"),
    )
    price_percentile = np.zeros(n_items, dtype=np.float64)
    price_percentile[item.index.to_numpy(np.int64)] = item[
        "item_median_price"
    ].rank(method="average", pct=True).to_numpy(np.float64)
    user_degree = np.bincount(edge_items, minlength=n_items)
    item["item_user_degree"] = user_degree[item.index.to_numpy(np.int64)]
    item["item_price_percentile"] = price_percentile[item.index.to_numpy(np.int64)]
    return item


def _normalized_coefficients(
    edge_users: np.ndarray,
    edge_items: np.ndarray,
    weights: np.ndarray,
    n_users: int,
    n_items: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    binary_user_degree = np.bincount(edge_users, minlength=n_users).astype(np.float64)
    binary_item_degree = np.bincount(edge_items, minlength=n_items).astype(np.float64)
    weighted_user_degree = np.bincount(
        edge_users, weights=weights, minlength=n_users
    ).astype(np.float64)
    weighted_item_degree = np.bincount(
        edge_items, weights=weights, minlength=n_items
    ).astype(np.float64)
    binary = 1.0 / np.sqrt(
        binary_user_degree[edge_users] * binary_item_degree[edge_items]
    )
    weighted = weights / np.sqrt(
        weighted_user_degree[edge_users] * weighted_item_degree[edge_items]
    )
    ratio = weighted / np.maximum(binary, 1e-12)
    return binary, weighted, ratio


def analyze_clv_nv_graph(
    train: pd.DataFrame,
    graph: CLVNVGraphWeights,
    *,
    n_users: int,
    n_items: int,
    top_n_items: int = 30,
) -> dict:
    """Measure which relations the current M3 formula strengthens.

    This function rebuilds no model and reads no validation/test labels.  It
    examines only the train graph before and after LightGCN degree
    normalisation.
    """
    required = {"u_idx", "i_idx", "b_raw", "v", "up", "i_raw", "cat_raw"}
    missing = sorted(required.difference(train.columns))
    if missing:
        raise ValueError(f"M3 diagnostic train columns missing: {missing}")
    edge_users = np.asarray(graph.edge_users, dtype=np.int64)
    edge_items = np.asarray(graph.edge_items, dtype=np.int64)
    weights = np.asarray(graph.weights, dtype=np.float64)
    edge_count = len(weights)
    arrays = (
        edge_users,
        edge_items,
        graph.n_relation,
        graph.v_relation,
        graph.n_component,
        graph.v_component,
    )
    if any(len(values) != edge_count for values in arrays):
        raise ValueError("M3 graph arrays must have the same edge count")

    actual_keys = edge_users * n_items + edge_items
    expected = train[["u_idx", "i_idx"]].drop_duplicates().sort_values(
        ["u_idx", "i_idx"]
    )
    expected_keys = (
        expected["u_idx"].to_numpy(np.int64) * n_items
        + expected["i_idx"].to_numpy(np.int64)
    )
    if not np.array_equal(actual_keys, expected_keys):
        raise ValueError("graph edge order does not match train unique edges")

    edge_context = _edge_context(train).reindex(
        pd.MultiIndex.from_arrays(
            [edge_users, edge_items], names=["u_idx", "i_idx"]
        )
    )
    item_context = _item_context(train, edge_items, n_items)
    binary_coef, weighted_coef, propagation_ratio = _normalized_coefficients(
        edge_users, edge_items, weights, n_users, n_items
    )
    edges = pd.DataFrame(
        {
            "u_idx": edge_users,
            "i_idx": edge_items,
            "raw_weight": weights,
            "n_relation": np.asarray(graph.n_relation, np.float64),
            "v_relation": np.asarray(graph.v_relation, np.float64),
            "n_component": np.asarray(graph.n_component, np.float64),
            "v_component": np.asarray(graph.v_component, np.float64),
            "binary_coefficient": binary_coef,
            "weighted_coefficient": weighted_coef,
            "propagation_ratio": propagation_ratio,
            "edge_row_count": edge_context["edge_row_count"].to_numpy(np.float64),
            "edge_basket_count": edge_context["edge_basket_count"].to_numpy(np.float64),
            "mean_basket_context": edge_context["mean_basket_context"].to_numpy(np.float64),
            "mean_item_value_share": edge_context[
                "mean_item_value_share"
            ].to_numpy(np.float64),
        }
    )
    item_lookup = item_context.reindex(edge_items)
    for name in (
        "item_user_degree",
        "item_row_count",
        "item_median_price",
        "item_price_percentile",
    ):
        edges[name] = item_lookup[name].to_numpy(np.float64)

    comparisons = {
        "raw_weight__n_component": ("raw_weight", "n_component"),
        "raw_weight__v_component": ("raw_weight", "v_component"),
        "raw_weight__item_user_degree": ("raw_weight", "item_user_degree"),
        "raw_weight__item_price_percentile": (
            "raw_weight",
            "item_price_percentile",
        ),
        "propagation_ratio__item_user_degree": (
            "propagation_ratio",
            "item_user_degree",
        ),
        "propagation_ratio__item_price_percentile": (
            "propagation_ratio",
            "item_price_percentile",
        ),
        "propagation_ratio__n_component": (
            "propagation_ratio",
            "n_component",
        ),
        "propagation_ratio__v_component": (
            "propagation_ratio",
            "v_component",
        ),
        "n_component__item_user_degree": ("n_component", "item_user_degree"),
        "n_component__item_price_percentile": (
            "n_component",
            "item_price_percentile",
        ),
        "v_component__mean_basket_context": (
            "v_component",
            "mean_basket_context",
        ),
        "v_component__mean_item_value_share": (
            "v_component",
            "mean_item_value_share",
        ),
        "v_component__item_price_percentile": (
            "v_component",
            "item_price_percentile",
        ),
        "n_component__v_component": ("n_component", "v_component"),
    }
    correlations = pd.DataFrame(
        [
            {
                "comparison": label,
                "left": left,
                "right": right,
                "spearman": _spearman(edges[left], edges[right]),
            }
            for label, (left, right) in comparisons.items()
        ]
    )

    percentile = edges["raw_weight"].rank(method="first", pct=True)
    edges["weight_decile"] = np.ceil(percentile * 10).clip(1, 10).astype(int)
    weight_deciles = edges.groupby("weight_decile", as_index=False).agg(
        edge_count=("raw_weight", "size"),
        raw_weight_mean=("raw_weight", "mean"),
        propagation_ratio_mean=("propagation_ratio", "mean"),
        n_component_mean=("n_component", "mean"),
        v_component_mean=("v_component", "mean"),
        repeat_edge_share=("n_relation", lambda values: float((values > 0).mean())),
        item_user_degree_mean=("item_user_degree", "mean"),
        item_price_percentile_mean=("item_price_percentile", "mean"),
        mean_basket_context=("mean_basket_context", "mean"),
        mean_item_value_share=("mean_item_value_share", "mean"),
    )

    top_items = edges.groupby("i_idx", as_index=False).agg(
        edge_count=("raw_weight", "size"),
        raw_weight_mean=("raw_weight", "mean"),
        propagation_ratio_mean=("propagation_ratio", "mean"),
        n_component_mean=("n_component", "mean"),
        v_component_mean=("v_component", "mean"),
    )
    top_items = top_items.join(
        item_context[
            [
                "i_raw",
                "category",
                "item_user_degree",
                "item_row_count",
                "item_median_price",
                "item_price_percentile",
            ]
        ],
        on="i_idx",
    ).sort_values(
        ["propagation_ratio_mean", "raw_weight_mean"], ascending=False
    ).head(top_n_items).reset_index(drop=True)

    corr = correlations.set_index("comparison")["spearman"]
    low_decile = weight_deciles.iloc[0]
    high_decile = weight_deciles.iloc[-1]
    summary = {
        "n_edges": int(edge_count),
        "repeat_edge_share": float((edges["n_relation"] > 0).mean()),
        "n_component_zero_share": float((edges["n_component"] == 0).mean()),
        "v_component_zero_share": float((edges["v_component"] == 0).mean()),
        "raw_weight_mean": float(weights.mean()),
        "raw_weight_std": float(weights.std()),
        "raw_weight_min": float(weights.min()),
        "raw_weight_max": float(weights.max()),
        "propagation_ratio_mean": float(propagation_ratio.mean()),
        "propagation_ratio_std": float(propagation_ratio.std()),
        "propagation_ratio_min": float(propagation_ratio.min()),
        "propagation_ratio_max": float(propagation_ratio.max()),
        "n_component_std": float(edges["n_component"].std()),
        "v_component_std": float(edges["v_component"].std()),
        "n_component_variance_share": float(
            edges["n_component"].var()
            / max(
                float(edges["n_component"].var() + edges["v_component"].var()),
                1e-12,
            )
        ),
        "v_component_variance_share": float(
            edges["v_component"].var()
            / max(
                float(edges["n_component"].var() + edges["v_component"].var()),
                1e-12,
            )
        ),
        "top_vs_bottom_item_degree_ratio": float(
            high_decile["item_user_degree_mean"]
            / max(float(low_decile["item_user_degree_mean"]), 1e-12)
        ),
        "top_minus_bottom_price_percentile": float(
            high_decile["item_price_percentile_mean"]
            - low_decile["item_price_percentile_mean"]
        ),
        "popularity_amplification_flag": bool(
            corr["propagation_ratio__item_user_degree"] > 0.05
        ),
        "low_price_amplification_flag": bool(
            corr["propagation_ratio__item_price_percentile"] < -0.05
        ),
        "v_context_mismatch_flag": bool(
            abs(corr["v_component__mean_item_value_share"])
            < abs(corr["v_component__mean_basket_context"])
        ),
    }
    return {
        "summary": summary,
        "correlations": correlations,
        "weight_deciles": weight_deciles,
        "top_items": top_items,
    }
