"""Minimal M1--M5 factorial using one historical-CLV scalar everywhere."""

from __future__ import annotations

import numpy as np
import pandas as pd

from clv_m5_economic_positive_weight_model import M5EconomicLightGCN
import lightgcn_clv_m5_economic_positive_weight as legacy
import lightgcn_clv_v3 as v3


M1_MODEL_ID = "m1_lightgcn_single_negative_bpr_scalar_clv_factorial"
M2_MODEL_ID = "m2_user_clv_item_buyer_context_embedding"
M3_MODEL_ID = "m3_user_clv_edge_weight"
M4P_MODEL_ID = "m4_user_clv_positive_row_weight"
M5_MODEL_ID = "m5_minimal_scalar_clv_m2_m3_m4"
MODEL_IDS = (M1_MODEL_ID, M2_MODEL_ID, M3_MODEL_ID, M4P_MODEL_ID, M5_MODEL_ID)
COMPARISON_REFERENCE_IDS = (
    M1_MODEL_ID,
    M2_MODEL_ID,
    M3_MODEL_ID,
    M4P_MODEL_ID,
)
INTERACTION_LABEL = "구성요소별 M1 대비 차이와 M5의 단일모형 대비 차이"

ACCURACY_METRICS = legacy.ACCURACY_METRICS
PRIMARY_METRIC = legacy.PRIMARY_METRIC
M5EconomicPositiveConfig = legacy.M5EconomicPositiveConfig
M3_ALPHA = 1.0

_arm_hash = legacy._arm_hash
_arm_paths = legacy._arm_paths
_train_arm = legacy._train_arm


def build_minimal_scalar_clv_inputs(
    train: pd.DataFrame,
    *,
    n_users: int,
    n_items: int,
    q_c: np.ndarray,
    clv_valid: np.ndarray,
) -> dict:
    """Create one user CLV coordinate and one item purchaser-CLV context."""

    q_c = np.asarray(q_c, dtype=np.float64)
    clv_valid = np.asarray(clv_valid, dtype=bool)
    if q_c.shape != (n_users,) or clv_valid.shape != (n_users,):
        raise ValueError("q_C와 valid mask shape이 n_users와 다릅니다")
    if not np.isfinite(q_c).all() or np.any((q_c < 0.0) | (q_c > 1.0)):
        raise ValueError("q_C는 [0,1]의 유한한 백분위여야 합니다")

    centered_user = 2.0 * q_c - 1.0
    centered_user[~clv_valid] = 0.0
    pairs = train[["u_idx", "i_idx"]].drop_duplicates()
    users = pairs["u_idx"].to_numpy(np.int64, copy=False)
    items = pairs["i_idx"].to_numpy(np.int64, copy=False)
    usable = clv_valid[users]
    item_count = np.bincount(items[usable], minlength=n_items).astype(np.float64)
    item_sum = np.bincount(
        items[usable], weights=centered_user[users[usable]], minlength=n_items
    )
    item_context = np.divide(
        item_sum,
        item_count,
        out=np.zeros(n_items, dtype=np.float64),
        where=item_count > 0.0,
    )
    item_valid = item_count > 0.0

    return {
        "q_c": q_c.astype(np.float32),
        "clv_valid": clv_valid,
        "user_economic_input": centered_user[:, None].astype(np.float32),
        "user_economic_valid": clv_valid,
        "item_economic_input": item_context[:, None].astype(np.float32),
        "item_economic_valid": item_valid,
        # The exercised M4 implementation becomes exactly 1 + lambda*q_C
        # when this fixed feature is one. No item price enters this experiment.
        "item_amount_percentile": np.ones(n_items, dtype=np.float32),
        "economic_input_diagnostics": {
            "historical_clv_proxy": "q_C = percentile(n_u * v_u)",
            "user_clv_input_dim": 1,
            "item_clv_context_dim": 1,
            "item_context_definition": "mean centered q_C of train purchasers",
            "item_context_is_item_clv": False,
            "item_context_valid_share": float(item_valid.mean()),
            "user_clv_valid_share": float(clv_valid.mean()),
            "user_clv_centered_mean": float(centered_user[clv_valid].mean()),
            "user_clv_centered_std": float(centered_user[clv_valid].std()),
            "item_purchaser_clv_context_mean": float(item_context[item_valid].mean()),
            "item_purchaser_clv_context_std": float(item_context[item_valid].std()),
            "m4_item_feature": "constant_one_no_item_price",
            "m3_alpha": M3_ALPHA,
        },
    }


def arm_specifications(prepared: dict, cfg: M5EconomicPositiveConfig) -> list[dict]:
    """Return M1 plus the three scalar-CLV locations and their full combination."""

    common = {"assignment": prepared, "assignment_name": "observed_q_c"}
    return [
        {
            "model_id": M1_MODEL_ID,
            "role": "minimal_scalar_clv_m1",
            "rho": 0.0,
            "m3": False,
            "weighted": False,
            **common,
        },
        {
            "model_id": M2_MODEL_ID,
            "role": "minimal_scalar_clv_m2_expression",
            "rho": cfg.rho,
            "m3": False,
            "weighted": False,
            **common,
        },
        {
            "model_id": M3_MODEL_ID,
            "role": "minimal_scalar_clv_m3_graph",
            "rho": 0.0,
            "m3": True,
            "weighted": False,
            **common,
        },
        {
            "model_id": M4P_MODEL_ID,
            "role": "minimal_scalar_clv_m4_loss",
            "rho": 0.0,
            "m3": False,
            "weighted": True,
            **common,
        },
        {
            "model_id": M5_MODEL_ID,
            "role": "minimal_scalar_clv_m5_all",
            "rho": cfg.rho,
            "m3": True,
            "weighted": True,
            **common,
        },
    ]


def _build_model(prepared: dict, cfg: M5EconomicPositiveConfig, spec: dict):
    data = prepared["data"]
    v3.set_seed(cfg.seed)
    return M5EconomicLightGCN(
        n_users=data["n_users"],
        n_items=data["n_items"],
        user_economic_input=prepared["user_economic_input"],
        user_economic_valid=prepared["user_economic_valid"],
        item_economic_input=prepared["item_economic_input"],
        item_economic_valid=prepared["item_economic_valid"],
        adj=prepared["clv_adj"] if spec["m3"] else data["adj"],
        id_dim=cfg.id_dim,
        economic_dim=cfg.economic_dim,
        rho=spec["rho"],
        n_layers=cfg.n_layers,
        pref_reg=cfg.pref_reg,
    ).to(v3.DEVICE)


def interaction_rows(metric_rows: dict[str, dict]) -> pd.DataFrame:
    """Report matched component deltas without claiming a full factorial interaction."""

    m1 = metric_rows[M1_MODEL_ID]
    m2 = metric_rows[M2_MODEL_ID]
    m3 = metric_rows[M3_MODEL_ID]
    m4 = metric_rows[M4P_MODEL_ID]
    m5 = metric_rows[M5_MODEL_ID]
    metrics = ACCURACY_METRICS + (
        PRIMARY_METRIC,
        "price_purchase_amount_weighted_hit@10",
    )
    rows = []
    for metric in metrics:
        if not all(metric in row for row in (m1, m2, m3, m4, m5)):
            continue
        single_values = [m2[metric], m3[metric], m4[metric]]
        rows.append(
            {
                "metric": metric,
                "m2_minus_m1": float(m2[metric] - m1[metric]),
                "m3_minus_m1": float(m3[metric] - m1[metric]),
                "m4_minus_m1": float(m4[metric] - m1[metric]),
                "m5_minus_m1": float(m5[metric] - m1[metric]),
                "m5_minus_best_single": float(m5[metric] - max(single_values)),
            }
        )
    return pd.DataFrame(rows)


def screening_reading(metric_rows: dict[str, dict]) -> dict:
    """Return an explicitly descriptive reading for the exposed seed-42 test."""

    m1 = metric_rows[M1_MODEL_ID]
    metrics = ACCURACY_METRICS + (
        PRIMARY_METRIC,
        "price_purchase_amount_weighted_hit@10",
    )
    deltas = {
        model_id: {
            metric: float(values[metric] - m1[metric])
            for metric in metrics
            if metric in values and metric in m1
        }
        for model_id, values in metric_rows.items()
        if model_id != M1_MODEL_ID
    }
    return {
        "descriptive_only": True,
        "model_selection_permitted": False,
        "deltas_vs_m1": deltas,
        "m5_minus_best_single": interaction_rows(metric_rows).set_index("metric")[
            "m5_minus_best_single"
        ].to_dict(),
        "prior_m3_note": (
            "user-only scalar CLV edge weighting previously showed normalization "
            "weakening; it is included here as a simple location reference and "
            "combination component, not revived as an already supported hypothesis"
        ),
        "statistical_note": (
            "one seed on an exposed test interval; no significance, stability, "
            "generalization, attribution, or final-model claim"
        ),
    }
