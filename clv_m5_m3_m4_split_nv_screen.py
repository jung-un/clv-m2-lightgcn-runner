"""M5 = M3 two-axis graph + M4 loss, two loss variants, Dunnhumby dev seed 43.

Both arms keep the confirmed M3 two-axis graph (V and N edge axes) unchanged
and differ only in the BPR row weight:

* A  original M4:  1 + .5*q_C(u)*amount(i)*fit(u,i)
* B  split N/V:    (1 + .5*q_N(u)) * (1 + .5*q_V(u)*amount(i)*fit(u,i))
                   N sets how much each customer is learned (constant within
                   a customer); V sets which of that customer's items count more.

Invalid rows (CLV, customer or item economics missing) get raw weight 1 in
both arms; weights are normalised to train mean 1.  One model, one optimizer,
from random initialisation; no freezing, no score post-processing.
M1 and M3 (arm B) are read from the finished 3-seed M3 result, never retrained.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

import lightgcn_clv_m3_centered_value_graph as m3
import lightgcn_clv_v3 as v3
import lightgcn_clv_axis_specific_test10 as io
from clv_m3_binary_residual_nv_screen import _original_curves


CODE_VERSION = "clv-m5-m3-m4-split-nv-dev-v1"
LAMBDA = 0.5
SEED = 43
FIXED_EPOCH = 300
DIAGNOSTIC_EPOCH = 100
ARM_A = "m5_m3nv_graph_original_m4_bpr_k1"
ARM_B = "m5_m3nv_graph_split_nv_m4_bpr_k1"
# Audited original M4 lambda=.5 on this exact development train set (2026-09-25/29 records).
ORIGINAL_M4_ROWS = 2_478_857
ORIGINAL_M4_MEAN_RAW = 1.14168269
ORIGINAL_M4_CV = 0.11679748
ECONOMIC = ("price_purchase_amount_weighted_hit@10", "vndcg@10")
ACCURACY = tuple(f"{m}@{k}" for m in ("recall", "ndcg") for k in (10, 20, 50))


def configure(**overrides):
    defaults = {"seeds": (SEED,), "epochs": FIXED_EPOCH, "eval_every": 25,
                "out_dir": f"{v3.default_out_dir('dunnhumby')}_clv_m5_m3_m4_split_nv_s43_v1"}
    cfg = m3.configure_centered_graph(**(defaults | overrides))
    if cfg.seeds != (SEED,) or cfg.epochs != FIXED_EPOCH or cfg.allow_baseline_training:
        raise ValueError("이번 스크린은 seed 43·300 epoch·M1/M3 재사용만 허용합니다")
    return cfg


def raw_row_weights(q_c, q_n, q_v, item_term, valid, variant: str) -> np.ndarray:
    if variant == "original":
        raw = 1.0 + LAMBDA * q_c * item_term
    elif variant == "split_nv":
        raw = (1.0 + LAMBDA * q_n) * (1.0 + LAMBDA * q_v * item_term)
    else:
        raise ValueError(variant)
    return np.where(valid, raw, 1.0)


def row_weights(prepared: dict) -> tuple[dict[str, np.ndarray], dict]:
    data = prepared["data"]
    u = np.asarray(data["tr_u"], np.int64)
    i = np.asarray(data["tr_i"], np.int64)
    valid = (np.asarray(prepared["clv_valid"], bool)[u]
             & np.asarray(prepared["user_economic_valid"], bool)[u]
             & np.asarray(prepared["item_economic_valid"], bool)[i])
    amount = np.asarray(prepared["item_amount_percentile"], float)[i]
    fit = np.clip(np.asarray(prepared["user_bin_fit"], float)[
        u, np.asarray(prepared["item_bin"], np.int64)[i]], 0.0, None)
    item_term = amount * fit
    q = {k: np.asarray(prepared[k], float)[u] for k in ("q_c", "q_n", "q_v")}
    weights, audit = {}, {"rows": int(len(u)), "invalid_rows": int((~valid).sum())}
    for model_id, variant in ((ARM_A, "original"), (ARM_B, "split_nv")):
        raw = raw_row_weights(q["q_c"], q["q_n"], q["q_v"], item_term, valid, variant)
        w = raw / raw.mean()
        if not np.isfinite(w).all() or (w <= 0).any():
            raise RuntimeError(f"{model_id}: 비유한·비양수 가중치 — 학습하지 않습니다")
        weights[model_id] = w
        user_total = np.bincount(u, weights=w, minlength=data["n_users"])
        user_rows = np.bincount(u, minlength=data["n_users"])
        seen = user_rows > 0
        audit[model_id] = {
            "variant": variant, "train_mean_raw_weight": float(raw.mean()),
            "row_weight_cv": float(w.std()), "row_weight_min": float(w.min()),
            "row_weight_max": float(w.max()),
            "invalid_extra_absent": bool(np.all(raw[~valid] == 1.0)),
            # how much the customer-level learning share moves vs plain BPR
            "customer_mass_ratio_cv": float(
                (user_total[seen] / user_rows[seen]).std()),
            "sha256": hashlib.sha256(w.astype(np.float32).tobytes()).hexdigest(),
        }
    return weights, audit


def check_original_m4(audit: dict) -> None:
    """Arm A must be the audited original M4, not a re-implementation drift."""
    a = audit[ARM_A]
    if (audit["rows"] != ORIGINAL_M4_ROWS
            or not np.isclose(a["train_mean_raw_weight"], ORIGINAL_M4_MEAN_RAW, rtol=1e-5)
            or not np.isclose(a["row_weight_cv"], ORIGINAL_M4_CV, rtol=1e-4)
            or not a["invalid_extra_absent"]):
        raise RuntimeError(f"원형 M4 가중치가 기존 감사값과 다릅니다: {a} rows={audit['rows']}")


def self_test() -> None:
    q_c, q_n, q_v = np.array([.9, .5, .5]), np.array([.9, .9, .2]), np.array([.9, .2, .9])
    term = np.array([.8, .8, .8])
    valid = np.array([True, True, False])
    a = raw_row_weights(q_c, q_n, q_v, term, valid, "original")
    b = raw_row_weights(q_c, q_n, q_v, term, valid, "split_nv")
    assert np.allclose(a, [1.36, 1.2, 1.0]) and np.allclose(b, [1.45 * 1.36, 1.45 * 1.08, 1.0])
    # B: within one customer only the V factor changes the ratio between items
    cheap = raw_row_weights(q_c, q_n, q_v, term * 0, valid, "split_nv")
    assert np.allclose(cheap[:2], [1.45, 1.45])


def _comparison(curve: pd.DataFrame) -> pd.DataFrame:
    index = curve.set_index(["model_id", "seed", "epoch"])
    metrics = [c for c in curve.columns if "@" in c
               or c == "user_value_tendency_recommended_price_alignment"]
    rows = []
    pairs = [(ARM_A, m3.M1_MODEL_ID), (ARM_A, m3.ARM_VALUE_ACTIVITY),
             (ARM_B, m3.M1_MODEL_ID), (ARM_B, m3.ARM_VALUE_ACTIVITY), (ARM_B, ARM_A)]
    for epoch in (DIAGNOSTIC_EPOCH, FIXED_EPOCH):
        for model_id, reference in pairs:
            left, right = index.loc[(model_id, SEED, epoch)], index.loc[(reference, SEED, epoch)]
            for metric in metrics:
                base, value = float(right[metric]), float(left[metric])
                rows.append({"seed": SEED, "epoch": epoch, "model_id": model_id,
                             "reference": reference, "metric": metric,
                             "reference_value": base, "candidate_value": value,
                             "delta": value - base,
                             "ratio": value / base if base else float("nan")})
    return pd.DataFrame(rows)


def reading(comparison: pd.DataFrame) -> dict:
    at = comparison[comparison.epoch.eq(FIXED_EPOCH)]
    out = {"development_screen_only": True, "fixed_epoch": FIXED_EPOCH,
           "significance_claim": False, "arm_selected_after_results": False}
    for model_id in (ARM_A, ARM_B):
        m1 = at[at.model_id.eq(model_id) & at.reference.eq(m3.M1_MODEL_ID)].set_index("metric")
        m3b = at[at.model_id.eq(model_id)
                 & at.reference.eq(m3.ARM_VALUE_ACTIVITY)].set_index("metric")
        guard = bool((m1.loc[list(ACCURACY), "ratio"] >= 0.99).all())
        above_m1 = bool((m1.loc[list(ECONOMIC), "delta"] > 0).all())
        noninferior = bool((m3b.loc[list(ECONOMIC), "ratio"] >= 0.99).all())
        out[model_id] = {"accuracy_guard_vs_m1": guard, "both_economic_at10_above_m1": above_m1,
                         "economic_at10_noninferior_99pct_vs_m3": noninferior,
                         "all_three_met": guard and above_m1 and noninferior}
    return out


def run(cfg=None) -> dict:
    cfg = configure() if cfg is None else configure(**asdict(cfg))
    prepared = m3._prepare(cfg)
    prepared["config_hash"] = hashlib.sha256(
        f"{CODE_VERSION}:{LAMBDA}:{prepared['config_hash']}".encode()).hexdigest()[:12]
    graph_spec = next(s for s in m3.arm_specifications()
                      if s["model_id"] == m3.ARM_VALUE_ACTIVITY)
    graph = m3.build_arm_graph(prepared, cfg, graph_spec)
    old_curve, old_reference = _original_curves(cfg, graph["beta"])  # stop before training
    weights, weight_audit = row_weights(prepared)
    check_original_m4(weight_audit)
    print(json.dumps({"beta": graph["beta"], "graph_audit": graph["audit"],
                      "weight_audit": weight_audit}, ensure_ascii=False, indent=2))
    old_stages = m3.clear_stale_progress(prepared)
    arms = []
    for model_id, loss in ((ARM_A, "original M4 (q_C)"), (ARM_B, "split N/V M4")):
        spec = {"model_id": model_id, "arm": "value_and_activity", "gamma": 1.0,
                "question": f"M3 two-axis graph + {loss}: above M1, not below M3?",
                "code_version": CODE_VERSION, "stage": "m5_m3_m4_dev"}
        print(f"\n===== {model_id} | seed {SEED} | {cfg.epochs} epoch =====", flush=True)
        arms.append(m3._run_arm(prepared, cfg, spec,
                                {**graph, "row_weights": weights[model_id]}, SEED))
    curve = pd.concat([m3.curve_table(arms, {}), old_curve], ignore_index=True)
    if curve.duplicated(["model_id", "seed", "epoch"]).any():
        raise RuntimeError("비교 곡선에 중복 모형·시드·epoch가 있습니다")
    comparison = _comparison(curve)
    result_reading = reading(comparison)
    out = Path(cfg.out_dir)
    stem = f"{CODE_VERSION}_{prepared['config_hash']}"
    paths = {"absolute_csv": out / f"{stem}_absolute.csv",
             "comparison_csv": out / f"{stem}_comparison.csv",
             "json": out / f"{stem}.json"}
    io._atomic_csv(paths["absolute_csv"], curve)
    io._atomic_csv(paths["comparison_csv"], comparison)
    io._atomic_json(paths["json"], {
        "code_version": CODE_VERSION, "config": asdict(cfg), "lambda": LAMBDA,
        "source_revision": prepared["revision"], "input_hash": prepared["input_hash"],
        "split": "historical_development_days_684_690", "final_test": False, "holdout": False,
        "beta": graph["beta"], "graph_audit": graph["audit"], "weight_audit": weight_audit,
        "old_m1_m3_reference": old_reference, "dropped_stale_progress": old_stages,
        "reading": result_reading,
        "limits": "single repeatedly exposed development seed; no significance, "
                  "generalization or CLV attribution claim",
        "result_paths": {k: str(v) for k, v in paths.items()},
    })
    print(json.dumps({"reading": result_reading,
                      "paths": {k: str(v) for k, v in paths.items()}},
                     ensure_ascii=False, indent=2))
    return {"absolute": curve, "comparison": comparison, "reading": result_reading,
            "paths": paths}


if __name__ == "__main__":
    self_test()
    print("self_test ok")
