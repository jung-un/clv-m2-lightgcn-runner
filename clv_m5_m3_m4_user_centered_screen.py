"""One-factor M5 screen: fixed M3 plus customer-mass-preserving M4.

This is an exploratory Dunnhumby development run, not another look at the
already exposed final week.  M1 and the two-axis M3 are reused from the exact
finished three-seed run.  The only new fit is M5-C at seed 42.

M4-C keeps the original interaction signal q_C(u) * amount(i) * fit(u, i), but
centres it within each customer's valid positive rows and normalises the
resulting weights within that customer.  Consequently M4-C can reorder which
positive items matter for a customer without changing that customer's total
training mass relative to plain BPR.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import clv_m5_m3_m4_split_nv_screen as prior
import lightgcn_clv_axis_specific_test10 as io
import lightgcn_clv_m3_centered_value_graph as m3
import lightgcn_clv_v3 as v3


CODE_VERSION = "clv-m5-m3-m4-user-centered-dev-v1"
MODEL_ID = "m5_m3nv_graph_user_centered_m4_bpr_k1"
DEFAULT_SEED = 42
ALLOWED_SEEDS = (42, 44)
# Backward-compatible label used by the already published seed42 notebook.
SEED = DEFAULT_SEED
LAMBDA = 0.5
FIXED_EPOCH = 300
DIAGNOSTIC_EPOCH = 100
ACCURACY = tuple(f"{name}@{k}" for name in ("recall", "ndcg") for k in (10, 20, 50))
ECONOMIC = ("price_purchase_amount_weighted_hit@10", "vndcg@10")


def configure(seed: int = DEFAULT_SEED, **overrides):
    root = v3.default_out_dir("dunnhumby")
    defaults = {
        "seeds": (seed,),
        "epochs": FIXED_EPOCH,
        "eval_every": 25,
        "out_dir": f"{root}_clv_m5_m3_m4_user_centered_s{seed}_v1",
    }
    cfg = m3.configure_centered_graph(**(defaults | overrides))
    if (cfg.seeds != (seed,) or seed not in ALLOWED_SEEDS or cfg.epochs != FIXED_EPOCH
            or cfg.eval_every != 25 or cfg.allow_baseline_training):
        raise ValueError(
            f"Dunnhumby 개발 seed{ALLOWED_SEEDS}·300epoch·25간격·기존 M1/M3 재사용만 허용합니다"
        )
    return cfg


def customer_centered_weights(
    edge_users: np.ndarray,
    q_c_by_user: np.ndarray,
    item_term_by_row: np.ndarray,
    valid_rows: np.ndarray,
    n_users: int,
    strength: float = LAMBDA,
) -> tuple[np.ndarray, dict]:
    """Build positive-row weights while preserving every customer's mass."""
    u = np.asarray(edge_users, dtype=np.int64)
    q_c = np.asarray(q_c_by_user, dtype=np.float64)
    item_term = np.asarray(item_term_by_row, dtype=np.float64)
    valid = np.asarray(valid_rows, dtype=bool)
    if (u.ndim != 1 or item_term.shape != u.shape or valid.shape != u.shape):
        raise ValueError("학습행 user/item-term/valid shape이 다릅니다")
    if q_c.shape != (n_users,) or np.any(u < 0) or np.any(u >= n_users):
        raise ValueError("사용자 CLV 또는 학습행 user index가 잘못됐습니다")
    if (not np.isfinite(q_c).all() or not np.isfinite(item_term).all()
            or np.any((q_c < 0) | (q_c > 1)) or strength <= 0):
        raise ValueError("q_C·아이템 항·강도가 유한한 허용범위여야 합니다")

    valid_count = np.bincount(u[valid], minlength=n_users).astype(np.int64)
    valid_sum = np.bincount(u[valid], weights=item_term[valid], minlength=n_users)
    mean_term = np.divide(valid_sum, valid_count, out=np.zeros(n_users), where=valid_count > 0)

    centred = np.zeros(len(u), dtype=np.float64)
    centred[valid] = item_term[valid] - mean_term[u[valid]]
    raw = np.ones(len(u), dtype=np.float64)
    raw[valid] = np.exp(strength * q_c[u[valid]] * centred[valid])

    raw_sum = np.bincount(u[valid], weights=raw[valid], minlength=n_users)
    raw_mean = np.divide(raw_sum, valid_count, out=np.ones(n_users), where=valid_count > 0)
    weights = np.ones(len(u), dtype=np.float64)
    weights[valid] = raw[valid] / raw_mean[u[valid]]
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise RuntimeError("사용자별 정규화 후 비유한·비양수 가중치가 생겼습니다")

    row_count = np.bincount(u, minlength=n_users)
    row_mass = np.bincount(u, weights=weights, minlength=n_users)
    seen = row_count > 0
    mass_ratio = row_mass[seen] / row_count[seen]
    valid_mass = np.bincount(u[valid], weights=weights[valid], minlength=n_users)
    valid_seen = valid_count > 0
    valid_mean = valid_mass[valid_seen] / valid_count[valid_seen]
    audit = {
        "rows": int(len(u)),
        "valid_rows": int(valid.sum()),
        "invalid_rows": int((~valid).sum()),
        "users_with_two_or_more_valid_rows": int((valid_count >= 2).sum()),
        "users_with_nonzero_within_user_signal": int(
            np.unique(u[valid & (np.abs(centred) > 1e-12)]).size),
        "train_mean_weight": float(weights.mean()),
        "row_weight_cv": float(weights.std()),
        "row_weight_min": float(weights.min()),
        "row_weight_max": float(weights.max()),
        "invalid_rows_equal_one": bool(np.all(weights[~valid] == 1.0)),
        "max_valid_within_user_mean_error": float(
            np.max(np.abs(valid_mean - 1.0)) if valid_mean.size else 0.0),
        "max_customer_mass_ratio_error": float(np.max(np.abs(mass_ratio - 1.0))),
        "customer_mass_ratio_cv": float(mass_ratio.std()),
        "sha256": hashlib.sha256(weights.astype(np.float32).tobytes()).hexdigest(),
    }
    if (not audit["invalid_rows_equal_one"]
            or audit["max_customer_mass_ratio_error"] > 1e-10):
        raise RuntimeError(f"사용자별 학습량 보존 검사 실패: {audit}")
    return weights.astype(np.float32), audit


def row_weights(prepared: dict) -> tuple[np.ndarray, dict]:
    data = prepared["data"]
    u = np.asarray(data["tr_u"], dtype=np.int64)
    i = np.asarray(data["tr_i"], dtype=np.int64)
    valid = (
        np.asarray(prepared["clv_valid"], dtype=bool)[u]
        & np.asarray(prepared["user_economic_valid"], dtype=bool)[u]
        & np.asarray(prepared["item_economic_valid"], dtype=bool)[i]
    )
    amount = np.asarray(prepared["item_amount_percentile"], dtype=float)[i]
    fit = np.clip(
        np.asarray(prepared["user_bin_fit"], dtype=float)[
            u, np.asarray(prepared["item_bin"], dtype=np.int64)[i]
        ],
        0.0,
        None,
    )
    return customer_centered_weights(
        u,
        np.asarray(prepared["q_c"], dtype=float),
        amount * fit,
        valid,
        data["n_users"],
    )


def _comparison(curve: pd.DataFrame, seed: int) -> pd.DataFrame:
    index = curve.set_index(["model_id", "seed", "epoch"])
    metrics = [column for column in curve.columns if "@" in column
               or column == "user_value_tendency_recommended_price_alignment"]
    rows = []
    for epoch in (DIAGNOSTIC_EPOCH, FIXED_EPOCH):
        for reference in (m3.M1_MODEL_ID, m3.ARM_VALUE_ACTIVITY):
            candidate = index.loc[(MODEL_ID, seed, epoch)]
            baseline = index.loc[(reference, seed, epoch)]
            for metric in metrics:
                base, value = float(baseline[metric]), float(candidate[metric])
                rows.append({
                    "seed": seed,
                    "epoch": epoch,
                    "model_id": MODEL_ID,
                    "reference": reference,
                    "metric": metric,
                    "reference_value": base,
                    "candidate_value": value,
                    "delta": value - base,
                    "relative_change_pct": 100 * (value / base - 1) if base else np.nan,
                })
    return pd.DataFrame(rows)


def _reading(comparison: pd.DataFrame) -> dict:
    at = comparison[comparison.epoch.eq(FIXED_EPOCH)]
    vs = {
        reference: at[at.reference.eq(reference)].set_index("metric")
        for reference in (m3.M1_MODEL_ID, m3.ARM_VALUE_ACTIVITY)
    }
    accuracy_vs_m1 = bool((vs[m3.M1_MODEL_ID].loc[list(ACCURACY), "candidate_value"]
                           >= .99 * vs[m3.M1_MODEL_ID].loc[list(ACCURACY), "reference_value"]).all())
    accuracy_vs_m3 = bool((vs[m3.ARM_VALUE_ACTIVITY].loc[list(ACCURACY), "candidate_value"]
                           >= .99 * vs[m3.ARM_VALUE_ACTIVITY].loc[list(ACCURACY), "reference_value"]).all())
    economic_vs_both = all(
        float(vs[reference].at[metric, "delta"]) > 0
        for reference in vs for metric in ECONOMIC
    )
    return {
        "development_screen_only": True,
        "repeatedly_exposed_split": True,
        "fixed_epoch": FIXED_EPOCH,
        "m5_c_accuracy_guard_99pct_vs_m1": accuracy_vs_m1,
        "m5_c_accuracy_guard_99pct_vs_m3": accuracy_vs_m3,
        "m5_c_both_economic_at10_above_m1_and_m3": bool(economic_vs_both),
        "screen_condition_met": bool(accuracy_vs_m1 and accuracy_vs_m3 and economic_vs_both),
        "m4_c_standalone_not_trained": True,
        "test_or_holdout_evaluated": False,
        "significance_claim": False,
        "generalization_claim": False,
        "clv_attribution_claim": False,
    }


def self_test() -> None:
    users = np.array([0, 0, 0, 1, 1, 2])
    q_c = np.array([1.0, 0.5, 0.9])
    term = np.array([0.1, 0.4, 0.9, 0.2, 0.8, 0.7])
    valid = np.array([True, True, False, True, True, True])
    weights, audit = customer_centered_weights(users, q_c, term, valid, 3)
    assert np.allclose(np.bincount(users, weights=weights), np.bincount(users))
    assert weights[2] == 1.0 and weights[5] == 1.0
    assert weights[0] < weights[1] and weights[3] < weights[4]
    assert audit["max_customer_mass_ratio_error"] < 1e-10


def run(cfg=None) -> dict:
    cfg = configure() if cfg is None else configure(cfg.seeds[0], **asdict(cfg))
    seed = cfg.seeds[0]
    prepared = m3._prepare(cfg)
    graph_spec = next(spec for spec in m3.arm_specifications()
                      if spec["model_id"] == m3.ARM_VALUE_ACTIVITY)
    graph = m3.build_arm_graph(prepared, cfg, graph_spec)
    old_curve, old_reference = prior._original_curves(cfg, graph["beta"])
    weights, weight_audit = row_weights(prepared)

    old_hash = prepared["config_hash"]
    prepared["config_hash"] = hashlib.sha256(
        f"{CODE_VERSION}:{LAMBDA}:{old_hash}:{weight_audit['sha256']}".encode()
    ).hexdigest()[:12]
    stale = m3.clear_stale_progress(prepared)
    spec = {
        "model_id": MODEL_ID,
        "arm": "value_and_activity",
        "gamma": 1.0,
        "question": "Can within-customer M4 priority add to M3 without reallocating customer mass?",
        "code_version": CODE_VERSION,
        "stage": "m5_m3_m4_user_centered_dev",
    }
    print(json.dumps({
        "scope": "Dunnhumby historical development only",
        "seed": seed,
        "fixed_epoch": FIXED_EPOCH,
        "m3_beta": graph["beta"],
        "m3_audit": graph["audit"],
        "m4_c_weight_audit": weight_audit,
        "new_fits": [MODEL_ID],
        "reused": [m3.M1_MODEL_ID, m3.ARM_VALUE_ACTIVITY],
    }, ensure_ascii=False, indent=2), flush=True)
    arm = m3._run_arm(
        prepared,
        cfg,
        spec,
        {**graph, "row_weights": weights},
        seed,
    )
    curve = pd.concat([old_curve, m3.curve_table([arm], {})], ignore_index=True)
    if curve.duplicated(["model_id", "seed", "epoch"]).any():
        raise RuntimeError("비교 곡선에 중복 model/seed/epoch가 있습니다")
    comparison = _comparison(curve, seed)
    reading = _reading(comparison)

    out = Path(cfg.out_dir)
    stem = f"{CODE_VERSION}_{prepared['config_hash']}"
    paths = {
        "absolute_csv": out / f"{stem}_absolute.csv",
        "comparison_csv": out / f"{stem}_comparison.csv",
        "json": out / f"{stem}.json",
    }
    io._atomic_csv(paths["absolute_csv"], curve)
    io._atomic_csv(paths["comparison_csv"], comparison)
    io._atomic_json(paths["json"], {
        "code_version": CODE_VERSION,
        "config": asdict(cfg),
        "seed": seed,
        "lambda": LAMBDA,
        "source_revision": prepared["revision"],
        "input_hash": prepared["input_hash"],
        "split": "historical_development_days_684_690",
        "final_test": False,
        "holdout": False,
        "new_fits": 1,
        "m4_c": (
            "exp(.5*q_C(u)*(amount(i)*fit(u,i)-customer valid-row mean)); "
            "valid rows normalised to customer mean one; invalid rows one"
        ),
        "m3_beta": graph["beta"],
        "m3_audit": graph["audit"],
        "m4_c_weight_audit": weight_audit,
        "old_m1_m3_reference": old_reference,
        "dropped_stale_progress": stale,
        "reading": reading,
        "limits": (
            "single repeatedly exposed development seed; no significance, generalization, "
            "CLV attribution, test tuning or standalone M4-C claim"
        ),
        "result_paths": {key: str(value) for key, value in paths.items()},
    })
    print(json.dumps({
        "reading": reading,
        "paths": {key: str(value) for key, value in paths.items()},
    }, ensure_ascii=False, indent=2), flush=True)
    return {"absolute": curve, "comparison": comparison, "reading": reading, "paths": paths}


if __name__ == "__main__":
    self_test()
    print("self_test ok")
