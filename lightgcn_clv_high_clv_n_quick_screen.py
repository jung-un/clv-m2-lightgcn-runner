"""Seed-43 directional screen of high-CLV-routed candidate-N M2 and M4."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from clv_high_clv_n_fit_model import HighCLVNFitLightGCN
import lightgcn_clv_candidate_nv_fit_factorial as candidate
import lightgcn_clv_gradient_isolated_economic_interaction as reports
import lightgcn_clv_m4_clv_hard_negative as m4_helpers
import lightgcn_clv_m4_k1_assignment_control_multiseed as prior_m1
import lightgcn_clv_m5_economic_positive_weight as legacy
import lightgcn_clv_v3 as v3


CODE_VERSION = "high-clv-candidate-n-m2-m4-quick-screen-v1"
M1_MODEL_ID = "m1_bpr_k1_reused_seed43"
M2_MODEL_ID = "m2_high_clv_candidate_n_expression_bpr_k1"
M4_MODEL_ID = "m4_high_clv_pairwise_n_weight_bpr_k1"
NEW_MODEL_IDS = (M2_MODEL_ID, M4_MODEL_ID)
ACCURACY_METRICS = (
    "recall@10", "ndcg@10", "recall@20", "ndcg@20", "recall@50", "ndcg@50"
)
PRIMARY_METRICS = (
    "price_purchase_amount_weighted_hit@10",
    "vndcg@10",
)


@dataclass(frozen=True)
class QuickScreenConfig(candidate.CandidateNVFitConfig):
    seed: int = 43
    rho: float = 0.05
    economic_dim: int = 3
    buyer_prior_mass: float = 10.0
    high_clv_percentile: float = 0.8
    reuse_exact_m1: bool = True
    m1_multiseed_result_dir: str = ""


def configure_quick_screen(**overrides) -> QuickScreenConfig:
    root = v3.default_out_dir("dunnhumby")
    defaults = {
        "out_dir": f"{root}_high_clv_candidate_n_m2_m4_quick_screen_v1",
        "baseline_result_dir": f"{root}_m2_repeatshare_historical_backtest_v1",
        "m1_multiseed_result_dir": (
            f"{root}_m4_k1_assignment_control_development_multiseed_v1"
        ),
    }
    return validate_config(QuickScreenConfig(**(defaults | overrides)))


def validate_config(cfg: QuickScreenConfig) -> QuickScreenConfig:
    fixed = {
        "dataset": "dunnhumby", "seed": 43, "time_cutoff": 690,
        "evaluation_days": 7, "epochs": 100, "id_dim": 64,
        "economic_dim": 3, "rho": 0.05, "positive_weight_lambda": 0.5,
        "n_layers": 2, "negative_count": 1, "input_days": 365,
        "buyer_prior_mass": 10.0, "high_clv_percentile": 0.8,
        "reuse_exact_m1": True,
    }
    for name, expected in fixed.items():
        if getattr(cfg, name) != expected:
            raise ValueError(f"빠른 스크린은 {name}={expected!r}이어야 합니다")
    if not cfg.out_dir or not cfg.m1_multiseed_result_dir:
        raise ValueError("결과와 기존 M1 디렉터리가 필요합니다")
    return cfg


def preflight_summary(cfg: QuickScreenConfig) -> dict:
    cfg = validate_config(cfg)
    return {
        "code_version": CODE_VERSION,
        "dataset": cfg.dataset,
        "seed": cfg.seed,
        "split": "historical_development_days_684_690",
        "reused_models": [M1_MODEL_ID],
        "trained_models": list(NEW_MODEL_IDS),
        "m1_reuse": "exact matched K=1 seed-43 row from prior 10-seed run",
        "m2": (
            "after binary LightGCN ID propagation, append rho-scaled RBF(q_N) "
            "and RBF(shrunk item-purchaser q_N); direct block gated by q_C>=0.8"
        ),
        "m4": (
            "batch-normalized 1+lambda*I[q_C>=0.8]*relu(F_N(u,i+)-F_N(u,j))"
        ),
        "fixed": {
            "new_item_task": True, "train_pairs_excluded": True,
            "min_item_interactions": 1, "graph": "binary",
            "negative_sampling": "one uniform unseen item", "epochs": 100,
            "final_test": False, "holdout": False, "external_reranking": False,
        },
        "reading_rule": (
            "direction only: inspect complete metrics; no CLV attribution, "
            "stability, significance or generalization claim"
        ),
        "out_dir": cfg.out_dir,
    }


def build_shrunk_buyer_context(
    train: pd.DataFrame, *, q_n: np.ndarray, n_items: int, prior_mass: float
) -> dict:
    if prior_mass <= 0:
        raise ValueError("prior_mass는 양수여야 합니다")
    pairs = train[["u_idx", "i_idx"]].drop_duplicates()
    users = pairs.u_idx.to_numpy(np.int64)
    items = pairs.i_idx.to_numpy(np.int64)
    values = np.asarray(q_n, dtype=np.float64)
    valid = np.isfinite(values[users])
    users, items = users[valid], items[valid]
    # q_N is a percentile coordinate; 0.5 is its fixed neutral location.
    global_mean = 0.5
    count = np.bincount(items, minlength=n_items).astype(np.float64)
    total = np.bincount(items, weights=values[users], minlength=n_items)
    mean = (total + prior_mass * global_mean) / (count + prior_mass)
    active = count > 0
    return {
        "item_mean": mean.astype(np.float32),
        "buyer_sum": total.astype(np.float64),
        "buyer_count": count.astype(np.float64),
        "global_mean": global_mean,
        "item_valid": active,
        "single_buyer_item_share": float(np.mean(count[active] == 1)) if active.any() else 0.0,
    }


def positive_leave_one_out_context(users, items, q_n, context, *, prior_mass):
    users = np.asarray(users, dtype=np.int64)
    items = np.asarray(items, dtype=np.int64)
    values = np.asarray(q_n, dtype=np.float64)
    total = context["buyer_sum"][items] - values[users]
    count = np.maximum(context["buyer_count"][items] - 1.0, 0.0)
    return (total + prior_mass * context["global_mean"]) / (count + prior_mass)


def pairwise_n_weights(users, positive_fit, negative_fit, high_gate, *, lambda_):
    users = np.asarray(users, dtype=np.int64)
    delta = np.maximum(
        np.asarray(positive_fit, dtype=np.float64)
        - np.asarray(negative_fit, dtype=np.float64),
        0.0,
    )
    raw = 1.0 + float(lambda_) * np.asarray(high_gate, dtype=np.float64)[users] * delta
    return raw, raw / raw.mean()


def _config_hash(cfg, input_hash, revision):
    payload = {"code_version": CODE_VERSION, "config": asdict(cfg),
               "input_hash": input_hash, "source_revision": revision}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:12]


def _prepare(cfg: QuickScreenConfig) -> dict:
    prepared = candidate._prepare(cfg)
    context = build_shrunk_buyer_context(
        prepared["data"]["train"], q_n=prepared["q_n"],
        n_items=prepared["data"]["n_items"], prior_mass=cfg.buyer_prior_mass,
    )
    prepared["buyer_context"] = context
    prepared["item_buyer_q_n"] = context["item_mean"]
    prepared["item_context_valid"] = context["item_valid"]
    prepared["high_clv_gate"] = (
        prepared["clv_valid"] & (prepared["q_c"] >= cfg.high_clv_percentile)
    ).astype(np.float32)
    prepared["economic_input_diagnostics"] = {
        "historical_clv_gate": "q_C percentile >= 0.8",
        "high_clv_user_count": int(prepared["high_clv_gate"].sum()),
        "buyer_prior_mass": cfg.buyer_prior_mass,
        "single_buyer_item_share": context["single_buyer_item_share"],
        "item_context_is_item_clv": False,
    }
    prepared["config_hash"] = _config_hash(
        cfg, prepared["input_hash"], prepared["revision"]
    )
    return prepared


def _build_model(prepared: dict, cfg: QuickScreenConfig, spec: dict):
    data = prepared["data"]
    v3.set_seed(cfg.seed)
    return HighCLVNFitLightGCN(
        n_users=data["n_users"], n_items=data["n_items"],
        user_q_n=prepared["q_n"], high_clv_gate=prepared["high_clv_gate"],
        clv_valid=prepared["clv_valid"], item_buyer_q_n=prepared["item_buyer_q_n"],
        item_context_valid=prepared["item_context_valid"], adj=data["adj"],
        id_dim=cfg.id_dim, rho=spec["rho"], n_layers=cfg.n_layers,
        pref_reg=cfg.pref_reg, basis_bandwidth=cfg.basis_bandwidth,
    ).to(v3.DEVICE)


def _train_arm(model, prepared, cfg, spec, store):
    data = prepared["data"]
    tr_u, tr_i, positive_keys = data["tr_u"], data["tr_i"], data["pos_key"]
    n_train = len(tr_u)
    n_batches = math.ceil(n_train / cfg.batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    rng = np.random.default_rng(cfg.seed)
    restored = store.restore_epoch(model, optimizer, rng)
    start_epoch = 1 if restored is None else int(restored["next_epoch"])
    history = list(restored.get("history", [])) if restored else []
    updates = int(restored.get("updates", 0)) if restored else 0
    samples = int(restored.get("samples", 0)) if restored else 0
    previous_wall = float(restored.get("wall_clock_sec", 0.0)) if restored else 0.0
    if restored:
        print(f"  [{spec['model_id']}] epoch {start_epoch - 1}에서 자동 재개")
    store.mark_stage("running", epoch=start_epoch - 1, max_epoch=cfg.epochs)
    started = time.time()
    last_epoch = start_epoch - 1
    q_n = prepared["q_n"]
    context = prepared["buyer_context"]
    high_gate = prepared["high_clv_gate"]
    for epoch in range(start_epoch, cfg.epochs + 1):
        last_epoch = epoch
        model.train()
        epoch_started = time.time()
        permutation = rng.permutation(n_train)
        totals = {"loss": 0.0, "bpr": 0.0, "p_correct": 0.0,
                  "row_weight_mean": 0.0, "row_weight_cv": 0.0}
        for batch in range(n_batches):
            index = permutation[batch * cfg.batch_size:(batch + 1) * cfg.batch_size]
            users_np, positives_np = tr_u[index], tr_i[index]
            negatives_np = m4_helpers.sample_uniform_negative_matrix(
                users_np, positives_np, data["n_items"], positive_keys, rng, k=1
            )
            users = torch.as_tensor(users_np, dtype=torch.long, device=v3.DEVICE)
            positives = torch.as_tensor(positives_np, dtype=torch.long, device=v3.DEVICE)
            negatives = torch.as_tensor(negatives_np, dtype=torch.long, device=v3.DEVICE)
            user_z, item_z = model.propagated_embeddings()
            pos_score = (user_z[users] * item_z[positives]).sum(dim=1)
            neg_score = (user_z[users, None, :] * item_z[negatives]).sum(dim=2)
            if spec["weighted"]:
                positive_context = positive_leave_one_out_context(
                    users_np, positives_np, q_n, context, prior_mass=cfg.buyer_prior_mass
                )
                negative_context = context["item_mean"][negatives_np[:, 0]]
                pos_fit = 1.0 - np.abs(q_n[users_np] - positive_context)
                neg_fit = 1.0 - np.abs(q_n[users_np] - negative_context)
                raw, normalized = pairwise_n_weights(
                    users_np, pos_fit, neg_fit, high_gate,
                    lambda_=cfg.positive_weight_lambda,
                )
                weights = torch.as_tensor(normalized, dtype=torch.float32, device=v3.DEVICE)
                weight_cv = float(np.std(normalized) / np.mean(normalized))
            else:
                raw = np.ones(len(users_np))
                weights = torch.ones_like(pos_score)
                weight_cv = 0.0
            per_row = torch.nn.functional.softplus(neg_score[:, 0] - pos_score)
            bpr = (weights * per_row).mean()
            loss = bpr + model.sampled_l2(users, positives, negatives)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            totals["loss"] += float(loss.detach())
            totals["bpr"] += float(bpr.detach())
            totals["p_correct"] += float((neg_score[:, 0] < pos_score).float().mean())
            totals["row_weight_mean"] += float(np.mean(raw))
            totals["row_weight_cv"] += weight_cv
            updates += 1
            samples += len(index)
            store.heartbeat(epoch=epoch, max_epoch=cfg.epochs, batch=batch + 1,
                            batches=n_batches, loss=totals["loss"] / (batch + 1))
        record = {"epoch": epoch, **{k: v / n_batches for k, v in totals.items()},
                  "epoch_sec": time.time() - epoch_started,
                  **model.representation_diagnostics()}
        history.append(record)
        print(f"  [{spec['model_id']}] ep {epoch:3d}/{cfg.epochs} | "
              f"loss {record['loss']:.4f} | P(pos>neg) {record['p_correct']:.3f} | "
              f"weight-cv {record['row_weight_cv']:.3f} | {record['epoch_sec']:.0f}s")
        store.save_epoch(model, optimizer, rng, epoch=epoch, best_epoch=epoch,
                         best_metric=0.0, best_state=None, bad=0, updates=updates,
                         samples=samples, history=history,
                         wall_clock_sec=previous_wall + time.time() - started)
    return {"phase": spec["model_id"], "epochs_run": last_epoch, "updates": updates,
            "samples": samples, "negative_count": 1,
            "wall_clock_sec": round(previous_wall + time.time() - started, 1),
            "history": history, "final_diagnostics": history[-1] if history else {}}


def _load_exact_m1(cfg: QuickScreenConfig, prepared: dict) -> tuple[dict, Path]:
    root = Path(cfg.m1_multiseed_result_dir)
    valid = []
    current_hash = legacy.moe.manifest_hash(prepared["manifest"])
    for path in root.glob("m4_k1_assignment_control_multiseed_*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        stored_cfg = payload.get("config", {})
        if payload.get("code_version") != prior_m1.CODE_VERSION:
            continue
        if any(stored_cfg.get(k) != v for k, v in {
            "epochs": 100, "id_dim": 64, "n_layers": 2,
            "negative_count": 1, "input_days": 365, "time_cutoff": 690,
            "evaluation_days": 7,
        }.items()):
            continue
        if legacy.moe.manifest_hash(payload.get("input_manifest", {})) != current_hash:
            continue
        rows = [row for row in payload.get("absolute_rows", [])
                if int(row.get("seed", -1)) == cfg.seed
                and row.get("model_id") == prior_m1.M1_MODEL_ID]
        if len(rows) == 1:
            valid.append((rows[0], path))
    if len(valid) != 1:
        raise FileNotFoundError(
            f"정확히 일치하는 seed {cfg.seed} K=1 M1 결과가 1개여야 합니다: {len(valid)}"
        )
    return valid[0]


def _specs(cfg):
    return [
        {"model_id": M2_MODEL_ID, "role": "high_clv_candidate_n_m2",
         "rho": cfg.rho, "weighted": False, "assignment_name": "q_c_upper20"},
        {"model_id": M4_MODEL_ID, "role": "high_clv_pairwise_n_m4",
         "rho": 0.0, "weighted": True, "assignment_name": "q_c_upper20"},
    ]


def run_quick_screen(cfg: QuickScreenConfig | None = None) -> pd.DataFrame:
    cfg = validate_config(cfg or configure_quick_screen())
    summary = preflight_summary(cfg)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    prepared = _prepare(cfg)
    m1_row, m1_source = _load_exact_m1(cfg, prepared)
    print(f"[reused] seed {cfg.seed} K=1 M1: {m1_source}")
    arms = {}
    for spec in _specs(cfg):
        print(f"\n===== {spec['model_id']} | seed {cfg.seed} | K=1 | 100 epochs =====")
        with patch.object(legacy, "_build_model", _build_model), patch.object(
            legacy, "_train_arm", _train_arm
        ):
            arm, _ = legacy._run_arm(prepared, cfg, spec)
        arms[spec["model_id"]] = arm
    rows = [{**m1_row, "model_id": M1_MODEL_ID,
             "role": "reused_exact_k1_m1", "training_origin": "reused"}]
    for spec in _specs(cfg):
        arm = arms[spec["model_id"]]
        rows.append({"model_id": spec["model_id"], "role": spec["role"],
                     "training_origin": "new", "seed": cfg.seed,
                     "split": arm["split"], "final_epoch": arm["final_epoch"],
                     "rho": arm["rho"], **arm["diagnostics"],
                     **arm["training"].get("final_diagnostics", {}), **arm["metrics"]})
    frame = pd.DataFrame(rows)
    metric_rows = {
        M1_MODEL_ID: {k: m1_row[k] for k in m1_row if "@" in k},
        **{mid: arms[mid]["metrics"] for mid in NEW_MODEL_IDS},
    }
    comparison = reports._metric_comparison(metric_rows, references=(M1_MODEL_ID,))
    decision = {
        "direction_only": True,
        "m2_beats_m1_both_primary": all(metric_rows[M2_MODEL_ID][m] > metric_rows[M1_MODEL_ID][m] for m in PRIMARY_METRICS),
        "m4_beats_m1_both_primary": all(metric_rows[M4_MODEL_ID][m] > metric_rows[M1_MODEL_ID][m] for m in PRIMARY_METRICS),
        "accuracy_guard_99pct": {
            mid: all(metric_rows[mid][m] >= 0.99 * metric_rows[M1_MODEL_ID][m] for m in ACCURACY_METRICS)
            for mid in NEW_MODEL_IDS
        },
        "clv_attribution_tested": False,
        "success_or_failure_final_decision_permitted": False,
    }
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"high_clv_candidate_n_quick_{prepared['config_hash']}"
    paths = {"absolute_csv": out / f"{stem}.csv",
             "comparison_csv": out / f"{stem}_comparison.csv",
             "json": out / f"{stem}.json"}
    legacy.test10._atomic_csv(paths["absolute_csv"], frame)
    legacy.test10._atomic_csv(paths["comparison_csv"], comparison)
    legacy.test10._atomic_json(paths["json"], {
        "code_version": CODE_VERSION, "config": asdict(cfg), "preflight": summary,
        "input_manifest": prepared["manifest"], "m1_source": str(m1_source),
        "absolute_rows": frame.to_dict("records"),
        "comparison_rows": comparison.to_dict("records"), "decision": decision,
        "arms": arms, "result_paths": {k: str(v) for k, v in paths.items()},
    })
    frame.attrs.update(decision=decision, comparison=comparison.to_dict("records"),
                       result_paths={k: str(v) for k, v in paths.items()})
    print("\n단일시드 방향성 판독:")
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    print("\n결과 파일:", json.dumps(frame.attrs["result_paths"], ensure_ascii=False, indent=2))
    return frame


if __name__ == "__main__":
    print(json.dumps(preflight_summary(configure_quick_screen()), ensure_ascii=False, indent=2))
