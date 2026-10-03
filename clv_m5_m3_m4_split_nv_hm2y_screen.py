"""H&M M5 = M3 two-axis graph + M4 loss (A original / B split N/V), seed 43, development split.

Same model as the Dunnhumby screen (`clv_m5_m3_m4_split_nv_screen.py`):
both arms keep the M3 two-axis centered value graph and differ only in the BPR
row weight.  On H&M the original M4 is used exactly as in the confirmed H&M M4
run (no extra validity mask: q_C, q_N, q_V are already 0 for CLV-invalid
customers, so their rows stay at raw weight 1 in both arms).

* A  original M4:  1 + .5*q_C(u)*amount(i)*fit(u,i)
* B  split N/V:    (1 + .5*q_N(u)) * (1 + .5*q_V(u)*amount(i)*fit(u,i))

Judgment (fixed before training) at epoch 100, where M1, M3 and M4 seed 43 all
exist: (1) both economic@10 above M1, (2) six accuracy >= 99% of M1,
(3) both economic@10 >= 99% of M4 (the best single model on H&M).
Training state is kept so a later run with stop_epoch=300 resumes from 100
(same source revision only).  One arm per call so A and B can run in parallel.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import lightgcn_clv_hm2y_seed42_common as common
import lightgcn_clv_m2_training_budget_hm2y as budget
import lightgcn_clv_m3_centered_value_graph_hm2y as hm2y
import lightgcn_clv_m5_nv_economic_positive_weight as economic_helpers
import lightgcn_clv_v3 as v3
from clv_m5_m3_m4_split_nv_screen import raw_row_weights


CODE_VERSION = "clv-m5-m3-m4-split-nv-hm2y-dev-v1"
SEED = 43
LAMBDA = 0.5
JUDGE_EPOCH = 100
ARMS = {"original": "m5_m3nv_graph_original_m4_bpr_k1_hm2y_s43",
        "split_nv": "m5_m3nv_graph_split_nv_m4_bpr_k1_hm2y_s43"}
# Audited references (H&M seed 43, development split).
HM_M3_BETA = 0.6088429480285137
HM_M4_MEAN_RAW = 1.2199791520931353
HM_M4_CV = 0.15904585000470794
M1_ID = "m1_bpr_k1_hm2y_training_budget_s43"
M3_ID = hm2y.ARM_VALUE_ACTIVITY
M4_ID = "m4_personalized_positive_weight_actual_qc_bpr_k1_hm2y"
M4_M1_ID = "m1_bpr_k1_hm2y_m4_assignment_control"
ECONOMIC = ("price_purchase_amount_weighted_hit@10", "vndcg@10")
ACCURACY = tuple(f"{m}@{k}" for m in ("recall", "ndcg") for k in (10, 20, 50))
PROTOCOL = ("seed", "batch_size", "lr", "pref_reg", "id_dim", "n_layers", "input_days")


def configure(**overrides):
    root = v3.default_out_dir("hm")
    defaults = {"seed": SEED, "epochs": 300, "evaluation_epochs": (100, 200, 300),
                "reported_epochs": (100, 300), "arms": ("value_and_activity",),
                "out_dir": f"{root}_clv_m5_m3_m4_split_nv_hm2y_s43_v1"}
    cfg = hm2y.configure_centered_graph_hm2y(**(defaults | overrides))
    if cfg.seed != SEED or cfg.epochs != 300 or cfg.allow_baseline_training:
        raise ValueError("seed 43·300 epoch 계획·기준모형 재사용만 허용합니다")
    return cfg


def references(cfg) -> tuple[pd.DataFrame, dict]:
    """M1/M3 (100/200/300) from the H&M M3 run, M4 (100) from the H&M M4 run."""
    root = v3.default_out_dir("hm")
    found = {}
    # Exact seed-43 result files (run hashes from the 2026-09-28 / 09-30 records);
    # the folders also hold other seeds' results, so a glob is ambiguous.
    for name, suffix, filename in (
            ("m3", "_clv_m3_centered_value_graph_s43_v1",
             "clv_m3_centered_value_graph_hm2y_93ab6203637d.json"),
            ("m4", "_m4_k1_assignment_control_hm2y_development_screen_v1",
             "m4_k1_assignment_control_hm2y_7218001fef1d.json")):
        path = Path(root + suffix) / filename
        if not path.is_file():
            raise RuntimeError(f"{name} 기준 결과가 없습니다: {path}")
        raw = path.read_bytes()
        found[name] = (json.loads(raw), {"path": str(path),
                                         "sha256": hashlib.sha256(raw).hexdigest()})
    if found["m4"][0]["config"]["seed"] != SEED or found["m3"][0]["config"]["seed"] != SEED:
        raise RuntimeError("기준 결과가 seed 43이 아닙니다")
    m3, m4 = found["m3"][0], found["m4"][0]
    for name, old in (("m3", m3["config"]), ("m4", m4["config"])):
        if any(old[k] != getattr(cfg, k) for k in PROTOCOL):
            raise RuntimeError(f"{name} 기준 실행의 학습 설정이 다릅니다")
    if m4["config"]["positive_weight_lambda"] != LAMBDA or m4["config"]["epochs"] != JUDGE_EPOCH:
        raise RuntimeError("M4 기준이 λ=.5·100 epoch 실행이 아닙니다")
    if not np.isclose(m3["betas"][M3_ID], HM_M3_BETA, rtol=0, atol=1e-9):
        raise RuntimeError("M3 기준의 그래프 강도가 다릅니다")
    rows = [r for r in m3["curve"] if r["model_id"] in (M1_ID, M3_ID)]
    m4_rows = {r["model_id"]: r for r in m4["absolute_rows"]}
    m1_at_100 = next(r for r in rows if r["model_id"] == M1_ID and r["epoch"] == JUDGE_EPOCH)
    # The two runs' M1 must be the same model, or M4 is compared on another baseline.
    for metric in ACCURACY + ECONOMIC:
        if not np.isclose(m1_at_100[metric], m4_rows[M4_M1_ID][metric], rtol=1e-6, atol=0):
            raise RuntimeError(f"M3 실행과 M4 실행의 M1@100이 다릅니다: {metric}")
    rows.append({**m4_rows[M4_ID], "model_id": M4_ID, "epoch": JUDGE_EPOCH})
    curve = pd.DataFrame(rows)
    return curve, {name: meta for name, (_, meta) in found.items()}


def row_weights(prepared: dict, economic: dict) -> tuple[dict[str, np.ndarray], dict]:
    data = prepared["data"]
    u = np.asarray(data["tr_u"], np.int64)
    i = np.asarray(data["tr_i"], np.int64)
    amount = np.asarray(economic["item_amount_percentile"], float)[i]
    fit = np.clip(np.asarray(economic["user_bin_fit"], float)[
        u, np.asarray(economic["item_bin"], np.int64)[i]], 0.0, None)
    term = amount * fit
    q = {k: np.asarray(prepared[k], float)[u] for k in ("q_c", "q_n", "q_v")}
    every_row = np.ones(len(u), bool)
    weights, audit = {}, {"rows": int(len(u))}
    for variant, model_id in ARMS.items():
        raw = raw_row_weights(q["q_c"], q["q_n"], q["q_v"], term, every_row, variant)
        w = raw / raw.mean()
        if not np.isfinite(w).all() or (w <= 0).any():
            raise RuntimeError(f"{model_id}: 비유한·비양수 가중치 — 학습하지 않습니다")
        weights[variant] = w
        total = np.bincount(u, weights=w, minlength=data["n_users"])
        count = np.bincount(u, minlength=data["n_users"])
        seen = count > 0
        audit[variant] = {"train_mean_raw_weight": float(raw.mean()),
                          "row_weight_cv": float(w.std()),
                          "row_weight_min": float(w.min()), "row_weight_max": float(w.max()),
                          "customer_mass_ratio_cv": float((total[seen] / count[seen]).std()),
                          "sha256": hashlib.sha256(w.astype(np.float32).tobytes()).hexdigest()}
    a = audit["original"]
    if (not np.isclose(a["train_mean_raw_weight"], HM_M4_MEAN_RAW, rtol=1e-5)
            or not np.isclose(a["row_weight_cv"], HM_M4_CV, rtol=1e-4)):
        raise RuntimeError(f"A의 가중치가 확증된 H&M M4와 다릅니다: {a}")
    return weights, audit


def prepare(cfg) -> dict:
    prepared = hm2y._prepare(cfg)
    spec = next(s for s in hm2y.arm_specifications(cfg) if s["arm"] == "value_and_activity")
    graph = hm2y.build_arm_graph(prepared, cfg, spec)
    # beta is calibrated to CV .20 within 1e-4, so platforms can differ in the 4th digit
    # (local 0.608725 vs Colab M3 run 0.608843); a different graph would differ far more.
    if not np.isclose(graph["beta"], HM_M3_BETA, rtol=1e-3, atol=0):
        raise RuntimeError(f"그래프 강도가 M3 기준과 다릅니다: {graph['beta']}")
    curve, sources = references(cfg)
    data = prepared["data"]
    economic = economic_helpers.build_nv_economic_inputs(
        data["train"], n_users=data["n_users"], n_items=data["n_items"],
        q_n=prepared["q_n"], q_v=prepared["q_v"], q_c=prepared["q_c"],
        clv_valid=prepared["clv_valid"], n_bins=4, shrinkage_strength=10.0, degree_bins=10)
    weights, audit = row_weights(prepared, economic)
    prepared.update(m5_graph=graph, m5_weights=weights, m5_weight_audit=audit,
                    m5_reference_curve=curve, m5_reference_sources=sources)
    print(json.dumps({"beta": graph["beta"], "graph_audit": graph["audit"],
                      "weight_audit": audit, "references": sources},
                     ensure_ascii=False, indent=2), flush=True)
    return prepared


def _arm_dir(prepared: dict) -> Path:
    root = Path(prepared["out_dir"]) / "arms" / prepared["run_hash"]
    root.mkdir(parents=True, exist_ok=True)
    return root


def train_arm(cfg, prepared: dict, variant: str, stop_epoch: int = JUDGE_EPOCH) -> dict:
    if variant not in ARMS or stop_epoch not in cfg.evaluation_epochs:
        raise ValueError(f"arm {variant} / stop_epoch {stop_epoch}")
    model_id = ARMS[variant]
    path = _arm_dir(prepared) / f"{model_id}_to{stop_epoch}.json"
    if path.exists():
        print(f"  [cached] {model_id} → epoch {stop_epoch} 결과 재사용", flush=True)
        return json.loads(path.read_text(encoding="utf-8"))
    spec = {"model_id": model_id, "arm": "value_and_activity", "gamma": 1.0, "kind": "m5",
            "question": f"H&M M3 two-axis graph + {variant} M4 loss"}
    weights = prepared["m5_weights"][variant]
    arm_hash = hashlib.sha256(
        f"{prepared['run_hash']}:{CODE_VERSION}:{prepared['m5_weight_audit'][variant]['sha256']}"
        .encode()).hexdigest()[:12]
    model = hm2y._build_model(prepared, cfg, prepared["m5_graph"])
    store = common.progress_store(prepared, cfg, model_id, arm_hash)
    curve = budget._train_curve(model, prepared, cfg, spec, store, row_weights=weights,
                                stop_epoch=stop_epoch)
    payload = {**spec, "variant": variant, "seed": cfg.seed, "stop_epoch": stop_epoch,
               "split": hm2y.SPLIT, "beta": prepared["m5_graph"]["beta"],
               "weight_audit": prepared["m5_weight_audit"][variant],
               "code_version": CODE_VERSION, "source_revision": prepared["revision"],
               "evaluated_at": datetime.now(timezone.utc).isoformat(), "curve": curve}
    common.atomic_json(path, payload)
    if stop_epoch == cfg.epochs:
        store.mark_complete(epoch=cfg.epochs, max_epoch=cfg.epochs, selection="none",
                            checkpoint_path="", result_path=str(path))
    return payload


def report(cfg, prepared: dict) -> dict:
    """Compare every finished arm with M1/M3 at shared epochs and with M4 at 100."""
    arms = []
    for variant, model_id in ARMS.items():
        done = sorted(_arm_dir(prepared).glob(f"{model_id}_to*.json"))
        if done:  # the longest finished run of this arm
            arms.append(max((json.loads(p.read_text(encoding="utf-8")) for p in done),
                            key=lambda a: a["stop_epoch"]))
    rows = [{"model_id": a["model_id"], "epoch": r["epoch"], **r["metrics"]}
            for a in arms for r in a["curve"] if "metrics" in r]
    curve = pd.concat([pd.DataFrame(rows), prepared["m5_reference_curve"]], ignore_index=True)
    index = curve.set_index(["model_id", "epoch"])
    metrics = [c for c in curve.columns if "@" in c
               or c == "user_value_tendency_recommended_price_alignment"]
    comparison = []
    for a in arms:
        for epoch in sorted({r["epoch"] for r in a["curve"] if "metrics" in r}):
            for ref in (M1_ID, M3_ID, M4_ID):
                if (ref, epoch) not in index.index:
                    continue
                left, right = index.loc[(a["model_id"], epoch)], index.loc[(ref, epoch)]
                for metric in metrics:
                    base, value = float(right[metric]), float(left[metric])
                    comparison.append({"epoch": epoch, "model_id": a["model_id"],
                                       "reference": ref, "metric": metric,
                                       "reference_value": base, "candidate_value": value,
                                       "delta": value - base,
                                       "ratio": value / base if base else float("nan")})
    comparison = pd.DataFrame(comparison)
    reading = {"development_screen_only": True, "judge_epoch": JUDGE_EPOCH,
               "significance_claim": False, "arm_selected_after_results": False}
    for a in arms:
        at = comparison[comparison.epoch.eq(JUDGE_EPOCH) & comparison.model_id.eq(a["model_id"])]
        m1 = at[at.reference.eq(M1_ID)].set_index("metric")
        m4 = at[at.reference.eq(M4_ID)].set_index("metric")
        guard = bool((m1.loc[list(ACCURACY), "ratio"] >= 0.99).all())
        above = bool((m1.loc[list(ECONOMIC), "delta"] > 0).all())
        noninferior = bool((m4.loc[list(ECONOMIC), "ratio"] >= 0.99).all())
        reading[a["model_id"]] = {"trained_to_epoch": a["stop_epoch"],
                                  "accuracy_guard_vs_m1": guard,
                                  "both_economic_at10_above_m1": above,
                                  "economic_at10_noninferior_99pct_vs_m4": noninferior,
                                  "all_three_met": guard and above and noninferior}
    out = Path(cfg.out_dir) / "reports"
    stem = f"{CODE_VERSION}_{prepared['run_hash']}"
    paths = {"absolute_csv": out / f"{stem}_absolute.csv",
             "comparison_csv": out / f"{stem}_comparison.csv", "json": out / f"{stem}.json"}
    common.atomic_json(paths["json"], {
        "code_version": CODE_VERSION, "source_revision": prepared["revision"],
        "input_hash": prepared["input_hash"], "split": hm2y.SPLIT,
        "final_test": False, "holdout": False, "lambda": LAMBDA,
        "beta": prepared["m5_graph"]["beta"], "graph_audit": prepared["m5_graph"]["audit"],
        "weight_audit": prepared["m5_weight_audit"],
        "references": prepared["m5_reference_sources"], "reading": reading,
        "limits": "single repeatedly exposed development seed; no significance, "
                  "generalization or CLV attribution claim",
        "result_paths": {k: str(v) for k, v in paths.items()}})
    for key, frame in (("absolute_csv", curve), ("comparison_csv", comparison)):
        paths[key].parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(paths[key], index=False)
    print(json.dumps({"reading": reading, "paths": {k: str(v) for k, v in paths.items()}},
                     ensure_ascii=False, indent=2), flush=True)
    return {"absolute": curve, "comparison": comparison, "reading": reading, "paths": paths}
