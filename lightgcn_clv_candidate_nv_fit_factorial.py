"""Six-arm seed-42 development screen with one shared candidate N/V fit.

This is a new, pre-result hypothesis.  It does not reuse or overwrite the
previous value-basis experiment.  Historical CLV is kept as q_N, q_V and
q_C=percentile(n_u*v_u).  Item-side values are training-only buyer-activity
context b_N and purchase-amount position p_V; they are not item CLV.
"""

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

from clv_candidate_nv_fit_model import (
    CandidateNVFitLightGCN,
    candidate_nv_fit_numpy,
    candidate_nv_fit_torch,
)
import lightgcn_clv_gradient_isolated_economic_interaction as report_helpers
import lightgcn_clv_m5_economic_positive_weight as legacy
import lightgcn_clv_v3 as v3


CODE_VERSION = "m5-candidate-specific-nv-fit-factorial-development-screen-v1"
M1_MODEL_ID = "m1_bpr_k1_candidate_nv_factorial"
M2_MODEL_ID = "m2_candidate_nv_expression_bpr_k1"
M3_MODEL_ID = "m3_candidate_nv_edge_weight_bpr_k1"
M4_MODEL_ID = "m4_candidate_nv_positive_weight_bpr_k1"
M5A_MODEL_ID = "m5_candidate_nv_m2_m4_bpr_k1"
M5B_MODEL_ID = "m5_candidate_nv_m2_m3_m4_bpr_k1"
MODEL_IDS = (
    M1_MODEL_ID,
    M2_MODEL_ID,
    M3_MODEL_ID,
    M4_MODEL_ID,
    M5A_MODEL_ID,
    M5B_MODEL_ID,
)
ACCURACY_METRICS = (
    "recall@10",
    "ndcg@10",
    "recall@20",
    "ndcg@20",
    "recall@50",
    "ndcg@50",
)
TOP10_METRICS = (
    "recall@10",
    "ndcg@10",
    "price_purchase_amount_weighted_hit@10",
    "vndcg@10",
)


@dataclass(frozen=True)
class CandidateNVFitConfig(legacy.M5EconomicPositiveConfig):
    economic_dim: int = 2
    negative_count: int = 1
    beta_m3: float = 0.15
    candidate_diagnostic_sample_size: int = 4096


def configure_candidate_nv_fit_screen(**overrides) -> CandidateNVFitConfig:
    root = v3.default_out_dir("dunnhumby")
    defaults = {
        "out_dir": (
            f"{root}_m5_candidate_specific_nv_fit_factorial_"
            "development_screen_v1"
        ),
        "baseline_result_dir": f"{root}_m2_repeatshare_historical_backtest_v1",
    }
    return validate_config(CandidateNVFitConfig(**(defaults | overrides)))


def validate_config(cfg: CandidateNVFitConfig) -> CandidateNVFitConfig:
    fixed = {
        "dataset": "dunnhumby",
        "seed": 42,
        "time_cutoff": 690,
        "evaluation_days": 7,
        "epochs": 100,
        "id_dim": 64,
        "economic_dim": 2,
        "rho": 0.15,
        "positive_weight_lambda": 0.5,
        "beta_m3": 0.15,
        "n_layers": 2,
        "negative_count": 1,
        "input_days": 365,
        "diagnostic_max_k": 50,
        "candidate_diagnostic_sample_size": 4096,
    }
    for key, expected in fixed.items():
        if getattr(cfg, key) != expected:
            raise ValueError(
                f"후보상품별 N/V 적합도 screen은 {key}={expected!r}이어야 합니다"
            )
    if cfg.batch_size <= 0 or cfg.lr <= 0.0 or cfg.pref_reg < 0.0:
        raise ValueError("학습 설정이 잘못됐습니다")
    if not cfg.out_dir or not cfg.baseline_result_dir:
        raise ValueError("out_dir와 baseline_result_dir가 필요합니다")
    return cfg


def preflight_summary(cfg: CandidateNVFitConfig) -> dict:
    cfg = validate_config(cfg)
    return {
        "code_version": CODE_VERSION,
        "dataset": cfg.dataset,
        "seed": cfg.seed,
        "split": "historical_development_days_684_690",
        "trained_models": list(MODEL_IDS),
        "reused_models": [],
        "research_question": (
            "Does one shared candidate-specific fit between historical CLV "
            "composition (q_N,q_V) and train-only item context improve new-item "
            "recommendation when used at M2, M3, M4 and their combinations?"
        ),
        "historical_clv": {
            "q_n": "percentile of train-period transaction frequency n_u",
            "q_v": "percentile of train-period value per transaction v_u",
            "q_c": "percentile(n_u * v_u), not q_N * q_V",
        },
        "item_context_not_item_clv": {
            "b_n": "mean q_N of distinct train purchasers of item i",
            "p_v": "train-only item purchase-amount percentile",
        },
        "shared_fit": "F(u,i)=1-(|q_N-b_N|+|q_V-p_V|)/2",
        "m2": {
            "user": "q_C * [2q_N-1, 2q_V-1]",
            "item": "[2b_N-1, 2p_V-1]",
            "axis_weights": "alpha_N+alpha_V=2; learned with the same BPR loss",
            "rho": cfg.rho,
            "joint_lightgcn_propagation": True,
        },
        "m3": {
            "raw": "exp(beta*q_C*(F-user_mean_F))",
            "normalization": "each user's raw edge-weight mean is one",
            "beta": cfg.beta_m3,
            "edge_set_changed": False,
        },
        "m4": {
            "raw": "1+lambda*q_C*F(u,i+)",
            "normalization": "mean over all train positive rows is one",
            "lambda": cfg.positive_weight_lambda,
        },
        "arms": {
            M1_MODEL_ID: "ID-only LightGCN",
            M2_MODEL_ID: "M2 representation only",
            M3_MODEL_ID: "M3 observed-edge weighting only",
            M4_MODEL_ID: "M4 positive-row weighting only",
            M5A_MODEL_ID: "M2+M4",
            M5B_MODEL_ID: "M2+M3+M4",
        },
        "reading_rule": {
            "baseline_direction": "report every arm versus M1 on all metrics",
            "m5a_increment": "M5-A versus M4",
            "m5b_increment": "M5-B versus M5-A and versus the best single arm",
            "attribution": "not tested in this first screen; add a degree-matched joint CLV shuffle only after a positive direction",
            "selection": "do not tune rho, beta or lambda on this same interval",
        },
        "fixed": {
            "new_item_task": True,
            "train_pairs_excluded_from_evaluation": True,
            "min_item_interactions": 1,
            "graph": "binary except the prespecified M3 edge coefficients",
            "negative_sampling": "one uniform unseen item",
            "epochs": cfg.epochs,
            "validation_or_epoch_selection": False,
            "final_test_constructed": False,
            "holdout_constructed": False,
            "one_training_loop_and_optimizer_per_arm": True,
            "external_reranking": False,
        },
        "statistical_note": (
            "one exposed historical development seed; no significance, stability, "
            "generalization or CLV-attribution claim"
        ),
        "out_dir": cfg.out_dir,
    }


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def _config_hash(cfg: CandidateNVFitConfig, input_hash: str, revision: str) -> str:
    payload = {
        "code_version": CODE_VERSION,
        "config": asdict(cfg),
        "input_hash": input_hash,
        "source_revision": revision,
    }
    return hashlib.sha256(_canonical(payload).encode()).hexdigest()[:12]


def build_candidate_nv_inputs(
    train: pd.DataFrame,
    *,
    n_users: int,
    n_items: int,
    q_n: np.ndarray,
    q_v: np.ndarray,
    q_c: np.ndarray,
    clv_valid: np.ndarray,
) -> dict:
    """Build b_N, p_V, F diagnostics and the mass-preserving M3 graph."""

    q_n = np.asarray(q_n, dtype=np.float64)
    q_v = np.asarray(q_v, dtype=np.float64)
    q_c = np.asarray(q_c, dtype=np.float64)
    valid = np.asarray(clv_valid, dtype=bool)
    if any(value.shape != (n_users,) for value in (q_n, q_v, q_c, valid)):
        raise ValueError("q_N·q_V·q_C·valid shape이 n_users와 다릅니다")
    if not all(np.isfinite(value).all() for value in (q_n, q_v, q_c)):
        raise ValueError("q_N·q_V·q_C는 유한해야 합니다")

    amount_series = (
        train.assign(v=pd.to_numeric(train["v"], errors="coerce"))
        .loc[lambda frame: np.isfinite(frame["v"]) & frame["v"].gt(0.0)]
        .groupby("i_idx", sort=True)["v"]
        .median()
        .reindex(np.arange(n_items))
    )
    amount = amount_series.to_numpy(np.float64)
    amount_valid = np.isfinite(amount) & (amount > 0.0)
    p_v = np.full(n_items, 0.5, dtype=np.float64)
    if amount_valid.any():
        p_v[amount_valid] = (
            pd.Series(np.log1p(amount[amount_valid]))
            .rank(method="average", pct=True)
            .to_numpy(np.float64)
        )
    pairs = train[["u_idx", "i_idx"]].drop_duplicates()
    edge_users = pairs["u_idx"].to_numpy(np.int64, copy=False)
    edge_items = pairs["i_idx"].to_numpy(np.int64, copy=False)
    usable = valid[edge_users]
    buyer_count = np.bincount(
        edge_items[usable], minlength=n_items
    ).astype(np.float64)
    buyer_sum = np.bincount(
        edge_items[usable], weights=q_n[edge_users[usable]], minlength=n_items
    )
    buyer_q_n = np.full(n_items, 0.5, dtype=np.float64)
    np.divide(
        buyer_sum,
        buyer_count,
        out=buyer_q_n,
        where=buyer_count > 0.0,
    )
    item_valid = amount_valid & (buyer_count > 0.0)
    item_degree = np.bincount(edge_items, minlength=n_items).astype(np.float64)
    correlation = pd.Series(buyer_q_n[item_valid]).corr(
        pd.Series(item_degree[item_valid]), method="spearman"
    )
    return {
        "q_n": q_n.astype(np.float32),
        "q_v": q_v.astype(np.float32),
        "q_c": q_c.astype(np.float32),
        "clv_valid": valid,
        "item_buyer_q_n": buyer_q_n.astype(np.float32),
        "item_amount_percentile": p_v.astype(np.float32),
        "item_context_valid": item_valid,
        "item_economic_valid": item_valid,
        "item_degree": item_degree.astype(np.float32),
        "economic_input_diagnostics": {
            "historical_clv_proxy": "q_C = percentile(n_u * v_u)",
            "item_context_is_item_clv": False,
            "item_buyer_context_definition": (
                "mean q_N of distinct training-period purchasers"
            ),
            "item_context_valid_share": float(item_valid.mean()),
            "item_buyer_q_n_mean": float(buyer_q_n[item_valid].mean()),
            "item_buyer_q_n_std": float(buyer_q_n[item_valid].std()),
            "item_buyer_q_n_item_degree_spearman": float(correlation),
        },
    }


def build_candidate_m3_adjacency(
    prepared: dict, *, beta: float
) -> tuple[torch.Tensor, dict]:
    """Reallocate each user's fixed observed-edge mass using centered F."""

    data = prepared["data"]
    n_users, n_items = data["n_users"], data["n_items"]
    keys = np.asarray(data["pos_key"], dtype=np.int64)
    edge_users, edge_items = keys // n_items, keys % n_items
    fit = candidate_nv_fit_numpy(
        prepared["q_n"][edge_users],
        prepared["q_v"][edge_users],
        prepared["item_buyer_q_n"][edge_items],
        prepared["item_amount_percentile"][edge_items],
    ).astype(np.float64)
    degree = np.bincount(edge_users, minlength=n_users).astype(np.float64)
    fit_sum = np.bincount(edge_users, weights=fit, minlength=n_users)
    fit_mean = np.divide(
        fit_sum, degree, out=np.zeros(n_users, dtype=np.float64), where=degree > 0
    )
    raw = np.exp(
        float(beta)
        * prepared["q_c"][edge_users].astype(np.float64)
        * (fit - fit_mean[edge_users])
    )
    raw_sum = np.bincount(edge_users, weights=raw, minlength=n_users)
    raw_mean = np.divide(
        raw_sum, degree, out=np.ones(n_users, dtype=np.float64), where=degree > 0
    )
    coefficient = raw / raw_mean[edge_users]
    final_sum = np.bincount(edge_users, weights=coefficient, minlength=n_users)
    active = degree > 0
    mass_error = float(np.max(np.abs(final_sum[active] - degree[active])))
    diagnostics = {
        "beta_m3": float(beta),
        "edge_count": int(len(coefficient)),
        "changed_edge_share": float(np.mean(np.abs(coefficient - 1.0) > 1e-8)),
        "coefficient_mean": float(coefficient.mean()),
        "coefficient_std": float(coefficient.std()),
        "coefficient_min": float(coefficient.min()),
        "coefficient_max": float(coefficient.max()),
        "user_mass_preservation_max_abs_error": mass_error,
        "observed_edge_fit_mean": float(fit.mean()),
        "observed_edge_fit_std": float(fit.std()),
    }
    adjacency = v3.build_adj(
        edge_users,
        edge_items,
        coefficient.astype(np.float32),
        n_users,
        n_items,
    )
    return adjacency, diagnostics


def _prepare(cfg: CandidateNVFitConfig) -> dict:
    common_cfg = legacy._common_config(cfg)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = legacy.moe.build_input_manifest(v3.SCHEMA[cfg.dataset])
    input_hash = legacy.moe.manifest_hash(manifest)
    revision = legacy.moe.source_revision()
    base_cfg = legacy.common.gatefree._base_config(common_cfg)
    data = v3.prepare_data(base_cfg, v3.DCFG)
    if set(data["splits"]) != {"test"}:
        raise RuntimeError("개발평가 외 split이 섞였습니다")
    if float(data["train"].t.max()) != 683.0:
        raise RuntimeError("개발 train 종료일이 683이 아닙니다")
    if data.get("loss_w") is not None:
        raise RuntimeError("기본 데이터에 기존 M4 가중치가 섞였습니다")
    data["loss_w"] = None
    snapshot = legacy.common.residual.build_final_snapshot(
        data["train"], data["n_users"], v3.DCFG["is_date"], cfg.input_days
    )
    axes = legacy.common.joint.build_user_axis_inputs(snapshot, data["n_users"])
    q_n, q_v, q_c, clv_valid = legacy.common.evaluation.build_clv_inputs(axes)
    meta = v3.item_meta(data["train"], data["n_items"])
    thresholds = v3.segment_thresholds(axes["clv_proxy"], base_cfg["SEG_EDGES"])
    cache = v3.EvalCache(
        *data["splits"]["test"], axes["clv_proxy"], thresholds, data["n_items"]
    )
    prepared = {
        "out_dir": out_dir,
        "manifest": manifest,
        "input_hash": input_hash,
        "revision": revision,
        "base_cfg": base_cfg,
        "data": data,
        "axes": axes,
        "q_n": q_n,
        "q_v": q_v,
        "q_c": q_c,
        "clv_valid": clv_valid,
        "meta": meta,
        "thresholds": thresholds,
        "cache": cache,
    }
    built = build_candidate_nv_inputs(
        data["train"],
        n_users=data["n_users"],
        n_items=data["n_items"],
        q_n=prepared["q_n"],
        q_v=prepared["q_v"],
        q_c=prepared["q_c"],
        clv_valid=prepared["clv_valid"],
    )
    prepared.update(built)
    prepared["m3_adj"], prepared["m3_diagnostics"] = (
        build_candidate_m3_adjacency(prepared, beta=cfg.beta_m3)
    )
    prepared["config_hash"] = _config_hash(cfg, input_hash, revision)
    return prepared


def arm_specifications(prepared: dict, cfg: CandidateNVFitConfig) -> list[dict]:
    def arm(
        model_id: str,
        role: str,
        *,
        m2: bool,
        m3: bool,
        m4: bool,
    ) -> dict:
        return {
            "model_id": model_id,
            "role": role,
            "rho": cfg.rho if m2 else 0.0,
            "m2": m2,
            "m3": m3,
            "weighted": m4,
            "assignment": prepared,
            "assignment_name": "observed_candidate_nv_fit" if m4 else "unweighted",
        }

    return [
        arm(M1_MODEL_ID, "candidate_nv_factorial_m1", m2=False, m3=False, m4=False),
        arm(M2_MODEL_ID, "candidate_nv_factorial_m2", m2=True, m3=False, m4=False),
        arm(M3_MODEL_ID, "candidate_nv_factorial_m3", m2=False, m3=True, m4=False),
        arm(M4_MODEL_ID, "candidate_nv_factorial_m4", m2=False, m3=False, m4=True),
        arm(M5A_MODEL_ID, "candidate_nv_factorial_m2_m4", m2=True, m3=False, m4=True),
        arm(M5B_MODEL_ID, "candidate_nv_factorial_m2_m3_m4", m2=True, m3=True, m4=True),
    ]


def _build_model(prepared: dict, cfg: CandidateNVFitConfig, spec: dict):
    data = prepared["data"]
    v3.set_seed(cfg.seed)
    return CandidateNVFitLightGCN(
        n_users=data["n_users"],
        n_items=data["n_items"],
        user_q_n=prepared["q_n"],
        user_q_v=prepared["q_v"],
        user_q_c=prepared["q_c"],
        user_clv_valid=prepared["clv_valid"],
        item_buyer_q_n=prepared["item_buyer_q_n"],
        item_amount_percentile=prepared["item_amount_percentile"],
        item_context_valid=prepared["item_context_valid"],
        adj=prepared["m3_adj"] if spec["m3"] else data["adj"],
        id_dim=cfg.id_dim,
        rho=spec["rho"],
        n_layers=cfg.n_layers,
        pref_reg=cfg.pref_reg,
    ).to(v3.DEVICE)


def _train_weight_normalizer(prepared: dict, lambda_: float) -> float:
    users = prepared["data"]["tr_u"]
    items = prepared["data"]["tr_i"]
    fit = candidate_nv_fit_numpy(
        prepared["q_n"][users],
        prepared["q_v"][users],
        prepared["item_buyer_q_n"][items],
        prepared["item_amount_percentile"][items],
    )
    raw = 1.0 + float(lambda_) * prepared["q_c"][users] * fit
    mean = float(raw.mean())
    if not np.isfinite(mean) or mean <= 0.0:
        raise RuntimeError("M4 학습행 정규화값이 잘못됐습니다")
    return mean


def _train_arm(model, prepared, cfg, spec, store) -> dict:
    """Use the original one-negative BPR with only the specified M4 row weight."""

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=0.0)
    rng = np.random.default_rng(cfg.seed)
    restored = store.restore_epoch(model, optimizer, rng)
    start_epoch = 1
    history: list[dict] = []
    updates = samples = 0
    previous_wall = 0.0
    if restored is not None:
        start_epoch = int(restored["next_epoch"])
        history = list(restored.get("history", []))
        updates = int(restored.get("updates", 0))
        samples = int(restored.get("samples", 0))
        previous_wall = float(restored.get("wall_clock_sec", 0.0))
        print(f"  [{spec['model_id']}] epoch {start_epoch - 1}에서 자동 재개")
    store.mark_stage("running", epoch=start_epoch - 1, max_epoch=cfg.epochs)

    data = prepared["data"]
    tr_u, tr_i, positive_keys = data["tr_u"], data["tr_i"], data["pos_key"]
    n_train = len(tr_u)
    n_batches = math.ceil(n_train / cfg.batch_size)
    q_n = torch.as_tensor(prepared["q_n"], device=v3.DEVICE)
    q_v = torch.as_tensor(prepared["q_v"], device=v3.DEVICE)
    q_c = torch.as_tensor(prepared["q_c"], device=v3.DEVICE)
    b_n = torch.as_tensor(prepared["item_buyer_q_n"], device=v3.DEVICE)
    p_v = torch.as_tensor(prepared["item_amount_percentile"], device=v3.DEVICE)
    normalizer = _train_weight_normalizer(
        prepared, cfg.positive_weight_lambda
    )
    started = time.time()
    last_epoch = start_epoch - 1
    for epoch in range(start_epoch, cfg.epochs + 1):
        last_epoch = epoch
        model.train()
        epoch_started = time.time()
        permutation = rng.permutation(n_train)
        totals = {
            "loss": 0.0,
            "bpr": 0.0,
            "p_correct": 0.0,
            "row_weight_mean": 0.0,
            "row_weight_std": 0.0,
            "row_weight_cv": 0.0,
            "row_weight_min": 0.0,
            "row_weight_max": 0.0,
            "effective_gradient_mass": 0.0,
        }
        last_gradients: dict[str, float] = {}
        for batch in range(n_batches):
            index = permutation[
                batch * cfg.batch_size : (batch + 1) * cfg.batch_size
            ]
            users_np, positives_np = tr_u[index], tr_i[index]
            negatives_np = legacy.m4_helpers.sample_uniform_negative_matrix(
                users_np,
                positives_np,
                data["n_items"],
                positive_keys,
                rng,
                k=cfg.negative_count,
            )
            users = torch.as_tensor(users_np, dtype=torch.long, device=v3.DEVICE)
            positives = torch.as_tensor(
                positives_np, dtype=torch.long, device=v3.DEVICE
            )
            negatives = torch.as_tensor(
                negatives_np, dtype=torch.long, device=v3.DEVICE
            )
            user_z, item_z = model.propagated_embeddings()
            positive_scores = (user_z[users] * item_z[positives]).sum(dim=1)
            negative_scores = (
                user_z[users, None, :] * item_z[negatives]
            ).sum(dim=2)
            if spec["weighted"]:
                fit = candidate_nv_fit_torch(
                    q_n[users], q_v[users], b_n[positives], p_v[positives]
                )
                row_weights = (
                    1.0 + cfg.positive_weight_lambda * q_c[users] * fit
                ) / normalizer
            else:
                row_weights = torch.ones_like(positive_scores)
            bpr, diagnostics = legacy.weighted_multi_negative_bpr(
                positive_scores, negative_scores, row_weights
            )
            loss = bpr + model.sampled_l2(users, positives, negatives)
            optimizer.zero_grad()
            loss.backward()
            last_gradients = model.training_gradient_diagnostics()
            optimizer.step()
            totals["loss"] += float(loss.detach())
            totals["bpr"] += float(bpr.detach())
            for key in totals:
                if key not in {"loss", "bpr"}:
                    totals[key] += float(diagnostics[key])
            updates += 1
            samples += len(index)
            store.heartbeat(
                epoch=epoch,
                max_epoch=cfg.epochs,
                batch=batch + 1,
                batches=n_batches,
                loss=totals["loss"] / (batch + 1),
            )
        record = {
            "epoch": int(epoch),
            **{key: float(value / n_batches) for key, value in totals.items()},
            "train_mean_raw_weight": normalizer,
            "epoch_sec": float(time.time() - epoch_started),
            **last_gradients,
            **model.representation_diagnostics(),
        }
        history.append(record)
        print(
            f"  [{spec['model_id']}] ep {epoch:3d}/{cfg.epochs} | "
            f"loss {record['loss']:.4f} | P(pos>neg) {record['p_correct']:.3f} | "
            f"weight-cv {record['row_weight_cv']:.3f} | "
            f"{record['epoch_sec']:.0f}s"
        )
        store.save_epoch(
            model,
            optimizer,
            rng,
            epoch=epoch,
            best_epoch=epoch,
            best_metric=0.0,
            best_state=None,
            bad=0,
            updates=updates,
            samples=samples,
            history=history,
            wall_clock_sec=previous_wall + time.time() - started,
        )
    return {
        "phase": spec["model_id"],
        "epochs_run": int(last_epoch),
        "updates": int(updates),
        "samples": int(samples),
        "negative_count": cfg.negative_count,
        "wall_clock_sec": round(previous_wall + time.time() - started, 1),
        "history": history,
        "final_diagnostics": history[-1] if history else {},
    }


def candidate_fit_diagnostics(prepared: dict, *, sample_size: int) -> dict:
    """Measure candidate discrimination on a fixed unseen-item sample."""

    valid_items = np.flatnonzero(prepared["item_context_valid"])
    rng = np.random.default_rng(42)
    selected = np.sort(
        rng.choice(valid_items, size=min(sample_size, len(valid_items)), replace=False)
    )
    positive_keys = np.sort(np.asarray(prepared["data"]["pos_key"], dtype=np.int64))
    n_items = prepared["data"]["n_items"]
    user_stds = []
    user_ranges = []
    for user in prepared["cache"].users.astype(np.int64):
        keys = user * n_items + selected
        position = np.searchsorted(positive_keys, keys)
        seen = (position < len(positive_keys)) & (
            positive_keys[np.minimum(position, len(positive_keys) - 1)] == keys
        )
        items = selected[~seen]
        if len(items) < 2:
            continue
        fit = candidate_nv_fit_numpy(
            np.full(len(items), prepared["q_n"][user]),
            np.full(len(items), prepared["q_v"][user]),
            prepared["item_buyer_q_n"][items],
            prepared["item_amount_percentile"][items],
        )
        user_stds.append(float(fit.std()))
        user_ranges.append(float(fit.max() - fit.min()))
    values = np.asarray(user_stds, dtype=np.float64)
    ranges = np.asarray(user_ranges, dtype=np.float64)
    return {
        "sampled_item_count": int(len(selected)),
        "evaluated_user_count": int(len(values)),
        "within_user_candidate_fit_std_mean": float(values.mean()),
        "within_user_candidate_fit_std_min": float(values.min()),
        "within_user_candidate_fit_std_max": float(values.max()),
        "within_user_candidate_fit_zero_std_share": float(np.mean(values <= 1e-12)),
        "within_user_candidate_fit_range_mean": float(ranges.mean()),
    }


def _m4_weight_diagnostics(prepared: dict, lambda_: float) -> dict:
    users, items = prepared["data"]["tr_u"], prepared["data"]["tr_i"]
    fit = candidate_nv_fit_numpy(
        prepared["q_n"][users],
        prepared["q_v"][users],
        prepared["item_buyer_q_n"][items],
        prepared["item_amount_percentile"][items],
    )
    raw = 1.0 + lambda_ * prepared["q_c"][users] * fit
    weight = raw / raw.mean()
    return {
        "train_row_count": int(len(weight)),
        "weight_mean": float(weight.mean()),
        "weight_std": float(weight.std()),
        "weight_cv": float(weight.std() / weight.mean()),
        "weight_min": float(weight.min()),
        "weight_max": float(weight.max()),
    }


def screening_reading(metric_rows: dict[str, dict]) -> dict:
    m1 = metric_rows[M1_MODEL_ID]

    def beats(left: dict, right: dict) -> bool:
        return all(left[metric] > right[metric] for metric in TOP10_METRICS)

    best_single = {
        metric: max(
            metric_rows[model_id][metric]
            for model_id in (M2_MODEL_ID, M3_MODEL_ID, M4_MODEL_ID)
        )
        for metric in TOP10_METRICS
    }
    deltas = {
        model_id: {
            metric: float(metric_rows[model_id][metric] - m1[metric])
            for metric in TOP10_METRICS
        }
        for model_id in MODEL_IDS
        if model_id != M1_MODEL_ID
    }
    m5a_baseline = beats(metric_rows[M5A_MODEL_ID], m1)
    m5b_baseline = beats(metric_rows[M5B_MODEL_ID], m1)
    m5a_increment = beats(metric_rows[M5A_MODEL_ID], metric_rows[M4_MODEL_ID])
    m5b_increment = all(
        metric_rows[M5B_MODEL_ID][metric] > best_single[metric]
        for metric in TOP10_METRICS
    )
    if m5a_baseline or m5b_baseline:
        classification = "combination_baseline_direction"
    elif any(
        beats(metric_rows[model_id], m1)
        for model_id in (M2_MODEL_ID, M3_MODEL_ID, M4_MODEL_ID)
    ):
        classification = "single_axis_only_direction"
    else:
        classification = "directional_nonpass"
    return {
        "classification": classification,
        "m5a_beats_m1_on_all_four_top10_metrics": m5a_baseline,
        "m5b_beats_m1_on_all_four_top10_metrics": m5b_baseline,
        "m5a_beats_m4_on_all_four_top10_metrics": m5a_increment,
        "m5b_beats_best_single_value_on_all_four_top10_metrics": m5b_increment,
        "top10_deltas_vs_m1": deltas,
        "m5a_minus_m4": {
            metric: float(
                metric_rows[M5A_MODEL_ID][metric]
                - metric_rows[M4_MODEL_ID][metric]
            )
            for metric in TOP10_METRICS
        },
        "m5b_minus_m5a": {
            metric: float(
                metric_rows[M5B_MODEL_ID][metric]
                - metric_rows[M5A_MODEL_ID][metric]
            )
            for metric in TOP10_METRICS
        },
        "clv_attribution_tested": False,
        "success_or_failure_final_decision_permitted": False,
        "next_if_positive_direction": (
            "freeze this structure and add a degree-matched joint q_N/q_V/q_C shuffle"
        ),
        "next_if_nonpass": (
            "stop without tuning rho, beta or lambda on this same development interval"
        ),
        "statistical_note": (
            "one exposed historical development seed; no significance, stability, "
            "generalization or CLV-attribution claim"
        ),
    }


def attach_result_metadata(
    frame: pd.DataFrame,
    *,
    comparison: pd.DataFrame,
    overlap: pd.DataFrame,
    score_diagnostics: pd.DataFrame,
    mechanism: dict,
    reading: dict,
    paths: dict[str, Path],
) -> None:
    """Attach repr-safe result metadata to the returned absolute table."""

    frame.attrs.update(
        comparison=comparison.to_dict("records"),
        top10_overlap=overlap.to_dict("records"),
        score_diagnostics=score_diagnostics.to_dict("records"),
        mechanism_diagnostics=mechanism,
        decision=reading,
        result_paths={key: str(value) for key, value in paths.items()},
    )


def run_candidate_nv_fit_screen(
    cfg: CandidateNVFitConfig | None = None,
) -> pd.DataFrame:
    cfg = validate_config(cfg or configure_candidate_nv_fit_screen())
    summary = preflight_summary(cfg)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    prepared = _prepare(cfg)

    arms: dict[str, dict] = {}
    models: dict[str, object] = {}
    for spec in arm_specifications(prepared, cfg):
        print(
            f"\n===== {spec['model_id']} | seed {cfg.seed} | K={cfg.negative_count} | "
            f"fixed {cfg.epochs} epochs ====="
        )
        with (
            patch.object(legacy, "_build_model", _build_model),
            patch.object(legacy, "_train_arm", _train_arm),
        ):
            arm, model = legacy._run_arm(prepared, cfg, spec)
        arms[spec["model_id"]] = arm
        models[spec["model_id"]] = model

    metric_rows = {model_id: arms[model_id]["metrics"] for model_id in MODEL_IDS}
    rows = []
    for spec in arm_specifications(prepared, cfg):
        arm = arms[spec["model_id"]]
        rows.append(
            {
                "model_id": spec["model_id"],
                "role": arm["role"],
                "seed": arm["seed"],
                "split": arm["split"],
                "final_epoch": arm["final_epoch"],
                "rho": arm["rho"],
                "m2_expression": spec["m2"],
                "m3_edge_weight": spec["m3"],
                "m4_positive_weight": spec["weighted"],
                "beta_m3": cfg.beta_m3 if spec["m3"] else 0.0,
                "positive_weight_lambda": arm["positive_weight_lambda"],
                **arm["diagnostics"],
                **arm["training"].get("final_diagnostics", {}),
                **arm["metrics"],
            }
        )
    frame = pd.DataFrame(rows)
    comparison = report_helpers._metric_comparison(
        metric_rows,
        references=(M1_MODEL_ID, M2_MODEL_ID, M3_MODEL_ID, M4_MODEL_ID),
    )

    topk = {
        model_id: report_helpers._masked_topk(
            models[model_id], prepared, max_k=cfg.diagnostic_max_k
        )
        for model_id in MODEL_IDS
    }
    users = topk[M1_MODEL_ID][0]
    if not all(np.array_equal(users, topk[model_id][0]) for model_id in MODEL_IDS):
        raise RuntimeError("arm별 평가 사용자 순서가 다릅니다")
    overlap_pairs = (
        (M1_MODEL_ID, M2_MODEL_ID),
        (M1_MODEL_ID, M3_MODEL_ID),
        (M1_MODEL_ID, M4_MODEL_ID),
        (M4_MODEL_ID, M5A_MODEL_ID),
        (M5A_MODEL_ID, M5B_MODEL_ID),
        (M1_MODEL_ID, M5B_MODEL_ID),
    )
    overlap = pd.concat(
        [
            report_helpers.topk_overlap_summary(
                topk[reference][1], topk[model_id][1], prepared["cache"].seg
            ).assign(reference=reference, model_id=model_id)
            for reference, model_id in overlap_pairs
        ],
        ignore_index=True,
    )
    score_diagnostics = pd.DataFrame(
        [
            legacy._score_diagnostics(
                models[model_id], users, topk[model_id][1], model_id=model_id
            )
            for model_id in (M2_MODEL_ID, M5A_MODEL_ID, M5B_MODEL_ID)
        ]
    )
    mechanism = {
        "candidate_fit": candidate_fit_diagnostics(
            prepared, sample_size=cfg.candidate_diagnostic_sample_size
        ),
        "m3": prepared["m3_diagnostics"],
        "m4": _m4_weight_diagnostics(
            prepared, cfg.positive_weight_lambda
        ),
        "item_context": prepared["economic_input_diagnostics"],
    }
    reading = screening_reading(metric_rows)

    out = Path(cfg.out_dir)
    stem = f"m5_candidate_nv_fit_{prepared['config_hash']}"
    paths = {
        "absolute_csv": out / f"{stem}.csv",
        "comparison_csv": out / f"{stem}_comparison.csv",
        "top10_overlap_csv": out / f"{stem}_top10_overlap.csv",
        "score_diagnostics_csv": out / f"{stem}_score_diagnostics.csv",
        "json": out / f"{stem}.json",
    }
    legacy.test10._atomic_csv(paths["absolute_csv"], frame)
    legacy.test10._atomic_csv(paths["comparison_csv"], comparison)
    legacy.test10._atomic_csv(paths["top10_overlap_csv"], overlap)
    legacy.test10._atomic_csv(paths["score_diagnostics_csv"], score_diagnostics)
    legacy.test10._atomic_json(
        paths["json"],
        {
            "code_version": CODE_VERSION,
            "source_revision": prepared["revision"],
            "config": asdict(cfg),
            "preflight": summary,
            "input_manifest": prepared["manifest"],
            "data_stats": prepared["data"].get("data_stats", {}),
            "absolute_rows": frame.to_dict("records"),
            "comparison_rows": comparison.to_dict("records"),
            "top10_overlap_rows": overlap.to_dict("records"),
            "score_diagnostic_rows": score_diagnostics.to_dict("records"),
            "mechanism_diagnostics": mechanism,
            "screening_reading": reading,
            "arms": arms,
            "result_paths": {key: str(value) for key, value in paths.items()},
        },
    )
    attach_result_metadata(
        frame,
        comparison=comparison,
        overlap=overlap,
        score_diagnostics=score_diagnostics,
        mechanism=mechanism,
        reading=reading,
        paths=paths,
    )
    print("\n1) 6-arm 절대지표")
    print(frame)
    print("\n2) 사전 판독 범위")
    print(json.dumps(reading, ensure_ascii=False, indent=2))
    print("\n3) 작동 진단")
    print(json.dumps(mechanism, ensure_ascii=False, indent=2))
    print("\n결과 파일:", frame.attrs["result_paths"])
    return frame


if __name__ == "__main__":
    print(
        json.dumps(
            preflight_summary(configure_candidate_nv_fit_screen()),
            ensure_ascii=False,
            indent=2,
        )
    )
