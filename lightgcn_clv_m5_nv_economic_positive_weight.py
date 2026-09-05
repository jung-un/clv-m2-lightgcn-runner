"""M5 factorial arms with an explicit role-separated q_N/q_V M2 block."""

from __future__ import annotations

import numpy as np
import pandas as pd

from clv_m5_nv_economic_positive_weight_model import M5NVEconomicLightGCN
import lightgcn_clv_m5_economic_positive_weight as legacy
import lightgcn_clv_v3 as v3


M1_MODEL_ID = "m1_multineg_mean_k5_nv_factorial"
M2_MODEL_ID = "m2_explicit_nv_economic_embedding_multineg_mean_k5"
M4P_MODEL_ID = "m4_clv_positive_amount_weight_k5_nv_factorial"
M5_MODEL_ID = "m5_explicit_nv_economic_embedding_positive_amount_weight_k5"
M5_SHUFFLED_MODEL_ID = "m5_explicit_nv_degree_matched_joint_shuffle"
M5_DEGREE_GATE_MODEL_ID = "m5_explicit_nv_degree_loss_gate"
MODEL_IDS = (
    M1_MODEL_ID,
    M2_MODEL_ID,
    M4P_MODEL_ID,
    M5_MODEL_ID,
    M5_SHUFFLED_MODEL_ID,
    M5_DEGREE_GATE_MODEL_ID,
)
ACCURACY_METRICS = legacy.ACCURACY_METRICS
PRIMARY_METRIC = legacy.PRIMARY_METRIC
M5EconomicPositiveConfig = legacy.M5EconomicPositiveConfig

# The M4 loss, optimizer, negative sampler, checkpointing, and diagnostics stay
# byte-for-byte on the already exercised path.  Only the M2 inputs/model below
# are changed in this experiment.
_arm_hash = legacy._arm_hash
_arm_paths = legacy._arm_paths
_train_arm = legacy._train_arm
positive_row_weights = legacy.positive_row_weights
weighted_multi_negative_bpr = legacy.weighted_multi_negative_bpr


def _population_spend_profile(
    train: pd.DataFrame,
    item_bin: np.ndarray,
    item_valid: np.ndarray,
    *,
    n_users: int,
    n_bins: int,
) -> np.ndarray:
    clean = train[["u_idx", "i_idx", "v"]].copy()
    clean["v"] = pd.to_numeric(clean["v"], errors="coerce")
    valid_amount = np.isfinite(clean["v"].to_numpy()) & (
        clean["v"].to_numpy() > 0.0
    )
    clean = clean.loc[valid_amount]
    users = clean["u_idx"].to_numpy(np.int64, copy=False)
    items = clean["i_idx"].to_numpy(np.int64, copy=False)
    amounts = clean["v"].to_numpy(np.float64, copy=False)
    usable = item_valid[items]
    spend = np.zeros((n_users, n_bins), dtype=np.float64)
    np.add.at(spend, (users[usable], item_bin[items[usable]]), amounts[usable])
    totals = spend.sum(axis=1)
    valid_users = totals > 0.0
    profile = np.divide(
        spend,
        totals[:, None],
        out=np.zeros_like(spend),
        where=valid_users[:, None],
    )
    if not valid_users.any():
        raise RuntimeError("가격구간 지출분포를 계산할 유효 사용자가 없습니다")
    return profile[valid_users].mean(axis=0)


def build_nv_economic_inputs(
    train: pd.DataFrame,
    *,
    n_users: int,
    n_items: int,
    q_n: np.ndarray,
    q_v: np.ndarray,
    q_c: np.ndarray,
    clv_valid: np.ndarray,
    n_bins: int = 4,
    shrinkage_strength: float = 10.0,
    degree_bins: int = 10,
) -> dict[str, np.ndarray | dict]:
    """Build train-only inputs with q_N=strength and V=economic direction."""

    q_n = np.asarray(q_n, dtype=np.float64)
    q_v = np.asarray(q_v, dtype=np.float64)
    clv_valid = np.asarray(clv_valid, dtype=bool)
    if q_n.shape != (n_users,) or q_v.shape != (n_users,):
        raise ValueError("q_N·q_V shape이 n_users와 다릅니다")
    if not np.isfinite(q_n).all() or not np.isfinite(q_v).all():
        raise ValueError("q_N·q_V는 유한값이어야 합니다")
    if np.any((q_n < 0.0) | (q_n > 1.0)) or np.any((q_v < 0.0) | (q_v > 1.0)):
        raise ValueError("q_N·q_V 범위는 [0,1]이어야 합니다")

    built = legacy.build_economic_inputs(
        train,
        n_users=n_users,
        n_items=n_items,
        q_v=q_v,
        q_c=q_c,
        clv_valid=clv_valid,
        n_bins=n_bins,
        shrinkage_strength=shrinkage_strength,
        degree_bins=degree_bins,
    )
    user_valid = np.asarray(built["user_economic_valid"], dtype=bool)
    centered_profile = np.asarray(
        built["user_economic_input"], dtype=np.float64
    )[:, :n_bins]
    centered_q_v = 2.0 * q_v - 1.0
    user_input = np.column_stack([centered_q_v, centered_profile])
    user_input[~user_valid] = 0.0
    activity_gate = np.clip(q_n, 0.0, 1.0)
    activity_gate[~user_valid] = 0.0

    item_valid = np.asarray(built["item_economic_valid"], dtype=bool)
    item_bin = np.asarray(built["item_bin"], dtype=np.int64)
    population = _population_spend_profile(
        train,
        item_bin,
        item_valid,
        n_users=n_users,
        n_bins=n_bins,
    )
    one_hot = np.zeros((n_items, n_bins), dtype=np.float64)
    valid_items = np.flatnonzero(item_valid)
    one_hot[valid_items, item_bin[valid_items]] = 1.0
    item_amount = np.asarray(built["item_amount_percentile"], dtype=np.float64)
    item_input = np.column_stack(
        [2.0 * item_amount - 1.0, one_hot - population[None, :]]
    )
    item_input[~item_valid] = 0.0

    # The shrunken profile equals population + reliability*(raw-population).
    shrunken_profile = population[None, :] + centered_profile
    bin_centers = (np.arange(n_bins, dtype=np.float64) + 0.5) / n_bins
    mean_economic_position = shrunken_profile @ bin_centers
    mean_economic_position[~user_valid] = 0.0

    diagnostics = dict(built["economic_input_diagnostics"])
    diagnostics.update(
        {
            "explicit_q_n_input": True,
            "explicit_q_v_input": True,
            "q_c_excluded_from_m2_input": True,
            "q_n_role": "post_projection_strength_only",
            "v_input": "q_v plus shrunken four-bin spending profile",
            "item_input": "overall amount percentile plus centered four-bin basis",
            "category_relative_amount_used": False,
            "user_economic_input_dim": int(user_input.shape[1]),
            "item_economic_input_dim": int(item_input.shape[1]),
            "q_n_gate_mean": float(activity_gate[user_valid].mean()),
            "q_n_gate_std": float(activity_gate[user_valid].std()),
            "implied_mean_economic_position_mean": float(
                mean_economic_position[user_valid].mean()
            ),
            "implied_mean_economic_position_std": float(
                mean_economic_position[user_valid].std()
            ),
        }
    )
    built.update(
        {
            "q_n": q_n.astype(np.float32),
            "q_v": q_v.astype(np.float32),
            "user_activity_gate": activity_gate.astype(np.float32),
            "user_economic_input": user_input.astype(np.float32),
            "item_economic_input": item_input.astype(np.float32),
            "population_spend_profile": population.astype(np.float32),
            "user_mean_economic_position": mean_economic_position.astype(np.float32),
            "economic_input_diagnostics": diagnostics,
        }
    )
    return built


def joint_degree_matched_shuffle(
    prepared: dict, *, seed: int = 42, degree_bins: int = 10
) -> dict[str, np.ndarray]:
    """Jointly shuffle N, V/profile, and q_C inside user-degree bins."""

    shuffled = legacy.joint_degree_matched_shuffle(
        prepared, seed=seed, degree_bins=degree_bins
    )
    source = shuffled["source_user"]
    shuffled.update(
        {
            "q_n": np.asarray(prepared["q_n"])[source].copy(),
            "user_activity_gate": np.asarray(prepared["user_activity_gate"])[
                source
            ].copy(),
        }
    )
    return shuffled


def arm_specifications(prepared: dict, cfg: M5EconomicPositiveConfig) -> list[dict]:
    return [
        {"model_id": M1_MODEL_ID, "role": "factorial_m1", "rho": 0.0,
         "weighted": False, "assignment": prepared, "assignment_name": "observed"},
        {"model_id": M2_MODEL_ID, "role": "factorial_m2_explicit_nv", "rho": cfg.rho,
         "weighted": False, "assignment": prepared, "assignment_name": "observed"},
        {"model_id": M4P_MODEL_ID, "role": "factorial_m4_prime", "rho": 0.0,
         "weighted": True, "assignment": prepared, "assignment_name": "observed"},
        {"model_id": M5_MODEL_ID, "role": "factorial_m5_explicit_nv", "rho": cfg.rho,
         "weighted": True, "assignment": prepared, "assignment_name": "observed"},
        {"model_id": M5_SHUFFLED_MODEL_ID, "role": "joint_assignment_control",
         "rho": cfg.rho, "weighted": True, "assignment": prepared["joint_shuffle"],
         "assignment_name": "degree_matched_joint_nv_shuffle"},
        {"model_id": M5_DEGREE_GATE_MODEL_ID, "role": "loss_gate_control",
         "rho": cfg.rho, "weighted": True, "assignment": prepared["degree_gate"],
         "assignment_name": "degree_percentile_loss_gate"},
    ]


def _build_model(prepared: dict, cfg: M5EconomicPositiveConfig, spec: dict):
    data = prepared["data"]
    assignment = spec["assignment"]
    v3.set_seed(cfg.seed)
    return M5NVEconomicLightGCN(
        n_users=data["n_users"],
        n_items=data["n_items"],
        user_economic_input=assignment["user_economic_input"],
        user_economic_valid=assignment["user_economic_valid"],
        user_activity_gate=assignment["user_activity_gate"],
        item_economic_input=prepared["item_economic_input"],
        item_economic_valid=prepared["item_economic_valid"],
        adj=data["adj"],
        id_dim=cfg.id_dim,
        economic_dim=cfg.economic_dim,
        rho=spec["rho"],
        n_layers=cfg.n_layers,
        pref_reg=cfg.pref_reg,
    ).to(v3.DEVICE)


def interaction_rows(metric_rows: dict[str, dict]) -> pd.DataFrame:
    a, b = metric_rows[M1_MODEL_ID], metric_rows[M2_MODEL_ID]
    c, d = metric_rows[M4P_MODEL_ID], metric_rows[M5_MODEL_ID]
    metrics = ACCURACY_METRICS + (
        PRIMARY_METRIC,
        "price_purchase_amount_weighted_hit@10",
    )
    rows = []
    for metric in metrics:
        if not all(metric in values for values in (a, b, c, d)):
            continue
        m2_effect = float(b[metric] - a[metric])
        rows.append(
            {
                "metric": metric,
                "m2_effect": m2_effect,
                "m4_prime_effect": float(c[metric] - a[metric]),
                "m5_effect": float(d[metric] - a[metric]),
                "interaction_effect": float((d[metric] - c[metric]) - m2_effect),
            }
        )
    return pd.DataFrame(rows)


def screening_reading(metric_rows: dict[str, dict]) -> dict:
    """Keep interaction descriptive; success is baseline plus CLV attribution."""

    a = metric_rows[M1_MODEL_ID]
    d = metric_rows[M5_MODEL_ID]
    e = metric_rows[M5_SHUFFLED_MODEL_ID]
    f = metric_rows[M5_DEGREE_GATE_MODEL_ID]
    accuracy_ratios = {metric: float(d[metric] / a[metric]) for metric in ACCURACY_METRICS}
    primary_deltas = {
        "vs_m1": float(d[PRIMARY_METRIC] - a[PRIMARY_METRIC]),
        "vs_joint_shuffle": float(d[PRIMARY_METRIC] - e[PRIMARY_METRIC]),
        "vs_degree_gate": float(d[PRIMARY_METRIC] - f[PRIMARY_METRIC]),
    }
    exposure_pass = bool(
        d["coverage@10"] / a["coverage@10"] >= 0.95
        and d["n_distinct@10"] / a["n_distinct@10"] >= 0.95
        and d["top10_share@10"] / a["top10_share@10"] <= 1.05
    )
    primary_pass = all(delta > 0.0 for delta in primary_deltas.values())
    return {
        "positive_screen": bool(primary_pass and exposure_pass),
        "primary_pass": primary_pass,
        "exposure_pass": exposure_pass,
        "interaction_required": False,
        "accuracy_guard_required": False,
        "primary_metric": PRIMARY_METRIC,
        "primary_deltas": primary_deltas,
        "accuracy_ratios_vs_m1_reported_not_gated": accuracy_ratios,
        "primary_interaction_effect_descriptive": float(
            interaction_rows(metric_rows).set_index("metric").at[
                PRIMARY_METRIC, "interaction_effect"
            ]
        ),
        "statistical_note": (
            "test-exposed exploratory seed 42; no significance or final generalization claim"
        ),
    }
