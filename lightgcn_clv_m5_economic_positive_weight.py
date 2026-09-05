"""Seed-42 M5 screen: economic representation plus positive amount weighting."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from clv_m5_economic_positive_weight_model import (
    M5EconomicLightGCN,
    positive_row_weights,
    weighted_multi_negative_bpr,
)
from clv_run_state import ProgressStore, RunIdentity, clone_state, file_sha256
import lightgcn_clv_axis_specific_test10 as test10
import lightgcn_clv_gated_relation_overall_price as common
import lightgcn_clv_gradient_isolated_economic_interaction as report_helpers
import lightgcn_clv_m4_clv_hard_negative as m4_helpers
import lightgcn_clv_moe as moe
import lightgcn_clv_v3 as v3


CODE_VERSION = "m5-clv-economic-positive-weighting-historical-screen-v1"
M1_MODEL_ID = "m1_multineg_mean_k5_economic_factorial"
M2_MODEL_ID = "m2_economic_embedding_multineg_mean_k5"
M4P_MODEL_ID = "m4_clv_positive_amount_weight_k5"
M5_MODEL_ID = "m5_economic_embedding_positive_amount_weight_k5"
M5_SHUFFLED_MODEL_ID = "m5_degree_matched_joint_economic_shuffle"
M5_DEGREE_GATE_MODEL_ID = "m5_degree_gate_positive_amount_weight"
MODEL_IDS = (
    M1_MODEL_ID,
    M2_MODEL_ID,
    M4P_MODEL_ID,
    M5_MODEL_ID,
    M5_SHUFFLED_MODEL_ID,
    M5_DEGREE_GATE_MODEL_ID,
)
ACCURACY_METRICS = (
    "recall@10",
    "ndcg@10",
    "recall@20",
    "ndcg@20",
    "recall@50",
    "ndcg@50",
)
PRIMARY_METRIC = "vndcg@10"


@dataclass(frozen=True)
class M5EconomicPositiveConfig:
    dataset: str = "dunnhumby"
    seed: int = 42
    time_cutoff: int = 690
    evaluation_days: int = 7
    epochs: int = 100
    id_dim: int = 64
    economic_dim: int = 4
    economic_bins: int = 4
    shrinkage_strength: float = 10.0
    rho: float = 0.15
    positive_weight_lambda: float = 0.5
    n_layers: int = 2
    negative_count: int = 5
    batch_size: int = 8192
    lr: float = 5e-4
    pref_reg: float = 1e-3
    input_days: int = 365
    diagnostic_max_k: int = 50
    shuffle_degree_bins: int = 10
    shuffle_seed: int = 42
    out_dir: str = ""
    baseline_result_dir: str = ""


def configure_m5_economic_positive_run(**overrides) -> M5EconomicPositiveConfig:
    defaults = {
        "out_dir": (
            f"{v3.default_out_dir('dunnhumby')}"
            "_m5_economic_positive_weighting_historical_screen_v1"
        ),
        "baseline_result_dir": (
            f"{v3.default_out_dir('dunnhumby')}"
            "_m2_repeatshare_historical_backtest_v1"
        ),
    }
    return validate_config(M5EconomicPositiveConfig(**(defaults | overrides)))


def validate_config(cfg: M5EconomicPositiveConfig) -> M5EconomicPositiveConfig:
    required = {
        "dataset": "dunnhumby",
        "seed": 42,
        "time_cutoff": 690,
        "evaluation_days": 7,
        "epochs": 100,
        "id_dim": 64,
        "economic_dim": 4,
        "economic_bins": 4,
        "shrinkage_strength": 10.0,
        "rho": 0.15,
        "positive_weight_lambda": 0.5,
        "n_layers": 2,
        "negative_count": 5,
        "input_days": 365,
        "diagnostic_max_k": 50,
        "shuffle_degree_bins": 10,
        "shuffle_seed": 42,
    }
    for key, expected in required.items():
        if getattr(cfg, key) != expected:
            raise ValueError(f"빠른 M5 screen은 {key}={expected!r}이어야 합니다")
    if cfg.batch_size <= 0 or cfg.lr <= 0 or cfg.pref_reg < 0:
        raise ValueError("학습 설정이 잘못됐습니다")
    if not cfg.out_dir or not cfg.baseline_result_dir:
        raise ValueError("out_dir와 baseline_result_dir가 필요합니다")
    return cfg


def preflight_summary(cfg: M5EconomicPositiveConfig) -> dict:
    cfg = validate_config(cfg)
    return {
        "code_version": CODE_VERSION,
        "dataset": cfg.dataset,
        "seed": cfg.seed,
        "reported_models": list(MODEL_IDS),
        "research_axis": "M5 partial combination: M2 representation plus M4 loss",
        "m2": {
            "user_input": "shrunken four-bin amount profile plus degree-conditioned V rank",
            "item_input": "overall and within-category representative amount percentile",
            "economic_dim": cfg.economic_dim,
            "projection": "bias-free tanh/2 bounded projection",
            "rho": cfg.rho,
            "layer0_joint_propagation": True,
        },
        "m4_prime": {
            "negative_count": cfg.negative_count,
            "lambda": cfg.positive_weight_lambda,
            "loss": "positive-row weighted mean of per-negative BPR losses",
            "hard_negative": False,
            "mean_loss_mass": 1.0,
        },
        "fixed": {
            "task": "new-item recommendation",
            "graph": "binary",
            "negative_sampling": "uniform",
            "min_user_interactions": 1,
            "min_item_interactions": 1,
            "epochs": cfg.epochs,
            "validation_or_epoch_selection": False,
            "final_test_constructed": False,
            "holdout_constructed": False,
            "external_reranking": False,
            "one_training_loop_and_optimizer_per_arm": True,
        },
        "controls": {
            "factorial": [M1_MODEL_ID, M2_MODEL_ID, M4P_MODEL_ID, M5_MODEL_ID],
            "joint_degree_matched_shuffle": M5_SHUFFLED_MODEL_ID,
            "degree_loss_gate": M5_DEGREE_GATE_MODEL_ID,
        },
        "reading_rule": {
            "primary_metric": PRIMARY_METRIC,
            "primary_label": "price/purchase-amount-weighted NDCG@10",
            "primary": "M5 > M1, M4-prime, joint shuffle, and degree gate",
            "interaction": "(M5-M4-prime)-(M2-M1) > 0",
            "accuracy": "every Recall/NDCG@10/20/50 >= 99% of M1",
            "exposure": "coverage/distinct >=95% and top10 share <=105% of M1",
            "statistical_note": "seed 42 exploratory screen; no significance claim",
        },
        "out_dir": cfg.out_dir,
        "baseline_result_dir": cfg.baseline_result_dir,
    }


def _stable_equal_count_bins(values: np.ndarray, valid: np.ndarray, n_bins: int) -> np.ndarray:
    members = np.flatnonzero(valid)
    if len(members) < n_bins:
        raise ValueError("경제구간을 만들 유효 상품이 부족합니다")
    order = np.lexsort((members, values[members]))
    assigned = np.floor(np.arange(len(members)) * n_bins / len(members))
    bins = np.full(len(values), -1, dtype=np.int16)
    bins[members[order]] = np.minimum(assigned, n_bins - 1).astype(np.int16)
    return bins


def _stable_degree_bins(degree: np.ndarray, n_bins: int) -> np.ndarray:
    users = np.arange(len(degree), dtype=np.int64)
    order = np.lexsort((users, degree))
    assigned = np.floor(np.arange(len(users)) * n_bins / max(len(users), 1))
    bins = np.empty(len(users), dtype=np.int16)
    bins[order] = np.minimum(assigned, n_bins - 1).astype(np.int16)
    return bins


def _rank_percentile(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average", pct=True).to_numpy(np.float64)


def build_economic_inputs(
    train: pd.DataFrame,
    *,
    n_users: int,
    n_items: int,
    q_v: np.ndarray,
    q_c: np.ndarray,
    clv_valid: np.ndarray,
    n_bins: int = 4,
    shrinkage_strength: float = 10.0,
    degree_bins: int = 10,
) -> dict[str, np.ndarray | dict]:
    """Build all train-only economic inputs used by the factorial arms."""

    required = {"u_idx", "i_idx", "cat_idx", "v"}
    missing = required - set(train.columns)
    if missing:
        raise KeyError(f"경제입력 컬럼 누락: {sorted(missing)}")
    if min(n_users, n_items, n_bins, degree_bins) <= 0 or shrinkage_strength < 0:
        raise ValueError("경제입력 차원·구간·축소강도가 잘못됐습니다")
    q_v = np.asarray(q_v, dtype=np.float64)
    q_c = np.asarray(q_c, dtype=np.float64)
    clv_valid = np.asarray(clv_valid, dtype=bool)
    if q_v.shape != (n_users,) or q_c.shape != (n_users,) or clv_valid.shape != (n_users,):
        raise ValueError("CLV 배열 shape이 n_users와 다릅니다")
    if not np.isfinite(q_v).all() or not np.isfinite(q_c).all():
        raise ValueError("CLV 입력은 유한해야 합니다")

    clean = train[["u_idx", "i_idx", "cat_idx", "v"]].copy()
    clean["v"] = pd.to_numeric(clean["v"], errors="coerce")
    positive = np.isfinite(clean["v"].to_numpy()) & (clean["v"].to_numpy() > 0)
    valid_rows = clean.loc[positive]
    item_amount_series = (
        valid_rows.groupby("i_idx", sort=True)["v"]
        .median()
        .reindex(np.arange(n_items))
    )
    item_amount = item_amount_series.to_numpy(np.float64)
    item_valid = np.isfinite(item_amount) & (item_amount > 0.0)
    log_amount = np.zeros(n_items, dtype=np.float64)
    log_amount[item_valid] = np.log1p(item_amount[item_valid])
    item_amount_percentile = np.full(n_items, 0.5, dtype=np.float64)
    item_amount_percentile[item_valid] = _rank_percentile(log_amount[item_valid])
    item_bin = _stable_equal_count_bins(log_amount, item_valid, n_bins)

    item_category = (
        train.groupby("i_idx", sort=True)["cat_idx"]
        .first()
        .reindex(np.arange(n_items))
        .to_numpy()
    )
    item_category_percentile = np.full(n_items, 0.5, dtype=np.float64)
    for category in pd.unique(item_category[item_valid]):
        members = np.flatnonzero(item_valid & (item_category == category))
        item_category_percentile[members] = _rank_percentile(log_amount[members])
    item_economic_input = np.column_stack(
        [2.0 * item_amount_percentile - 1.0, 2.0 * item_category_percentile - 1.0]
    )
    item_economic_input[~item_valid] = 0.0

    row_users = valid_rows["u_idx"].to_numpy(np.int64, copy=False)
    row_items = valid_rows["i_idx"].to_numpy(np.int64, copy=False)
    row_amount = valid_rows["v"].to_numpy(np.float64, copy=False)
    spend = np.zeros((n_users, n_bins), dtype=np.float64)
    usable = item_valid[row_items]
    np.add.at(
        spend,
        (row_users[usable], item_bin[row_items[usable]]),
        row_amount[usable],
    )
    total_spend = spend.sum(axis=1)
    profile_valid = total_spend > 0.0
    raw_profile = np.divide(
        spend,
        total_spend[:, None],
        out=np.zeros_like(spend),
        where=profile_valid[:, None],
    )
    population = raw_profile[profile_valid].mean(axis=0)
    unique_pairs = train[["u_idx", "i_idx"]].drop_duplicates()
    degree = np.bincount(
        unique_pairs["u_idx"].to_numpy(np.int64), minlength=n_users
    ).astype(np.float64)
    reliability = degree / (degree + float(shrinkage_strength))
    centered_profile = reliability[:, None] * (
        raw_profile - population[None, :]
    )

    degree_bin = _stable_degree_bins(degree, degree_bins)
    degree_percentile = _rank_percentile(degree)
    conditioned_v = np.zeros(n_users, dtype=np.float64)
    for group in range(degree_bins):
        members = np.flatnonzero((degree_bin == group) & clv_valid)
        if len(members):
            conditioned_v[members] = 2.0 * _rank_percentile(q_v[members]) - 1.0
    user_valid = clv_valid & profile_valid
    user_economic_input = np.column_stack([centered_profile, conditioned_v])
    user_economic_input[~user_valid] = 0.0
    q_c_clean = np.clip(q_c, 0.0, 1.0)
    q_c_clean[~clv_valid] = 0.0

    counts = np.bincount(item_bin[item_valid], minlength=n_bins)
    diagnostics = {
        "economic_bin_count": int(n_bins),
        "shrinkage_strength": float(shrinkage_strength),
        "item_economic_valid_share": float(item_valid.mean()),
        "user_economic_valid_share": float(user_valid.mean()),
        "user_profile_reliability_mean": float(reliability.mean()),
        "item_count_bin_imbalance": int(counts.max() - counts.min()),
        "item_amount_percentile_min": float(item_amount_percentile[item_valid].min()),
        "item_amount_percentile_max": float(item_amount_percentile[item_valid].max()),
    }
    return {
        "q_v": q_v.astype(np.float32),
        "q_c": q_c_clean.astype(np.float32),
        "clv_valid": clv_valid,
        "user_economic_input": user_economic_input.astype(np.float32),
        "user_economic_valid": user_valid,
        "user_profile_reliability": reliability.astype(np.float32),
        "item_economic_input": item_economic_input.astype(np.float32),
        "item_economic_valid": item_valid,
        "item_amount_percentile": item_amount_percentile.astype(np.float32),
        "item_category_percentile": item_category_percentile.astype(np.float32),
        "item_bin": item_bin,
        "degree": degree.astype(np.float32),
        "degree_bin": degree_bin,
        "degree_percentile": degree_percentile.astype(np.float32),
        "economic_input_diagnostics": diagnostics,
    }


def joint_degree_matched_shuffle(
    prepared: dict, *, seed: int = 42, degree_bins: int = 10
) -> dict[str, np.ndarray]:
    """Jointly permute the complete user economic tuple within degree bins."""

    bins = np.asarray(prepared["degree_bin"])
    if bins.ndim != 1 or len(bins) != len(prepared["q_c"]):
        raise ValueError("degree_bin shape이 잘못됐습니다")
    if bins.min(initial=0) < 0 or bins.max(initial=0) >= degree_bins:
        raise ValueError("degree_bin 범위가 잘못됐습니다")
    rng = np.random.default_rng(seed)
    source = np.arange(len(bins), dtype=np.int64)
    for group in range(degree_bins):
        members = np.flatnonzero(bins == group)
        if len(members) > 1:
            order = rng.permutation(members)
            source[order] = np.roll(order, 1)
    return {
        "q_v": np.asarray(prepared["q_v"])[source].copy(),
        "q_c": np.asarray(prepared["q_c"])[source].copy(),
        "clv_valid": np.asarray(prepared["clv_valid"])[source].copy(),
        "user_economic_input": np.asarray(prepared["user_economic_input"])[source].copy(),
        "user_economic_valid": np.asarray(prepared["user_economic_valid"])[source].copy(),
        "source_user": source,
        "degree_bin": bins.copy(),
    }


def _common_config(cfg: M5EconomicPositiveConfig) -> common.GatedRelationPriceConfig:
    return common.GatedRelationPriceConfig(
        dataset=cfg.dataset,
        seed=cfg.seed,
        time_cutoff=cfg.time_cutoff,
        evaluation_days=cfg.evaluation_days,
        epochs=cfg.epochs,
        id_dim=cfg.id_dim,
        n_layers=cfg.n_layers,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        pref_reg=cfg.pref_reg,
        input_days=cfg.input_days,
        diagnostic_max_k=cfg.diagnostic_max_k,
        out_dir=cfg.out_dir,
        baseline_result_dir=cfg.baseline_result_dir,
    )


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def _config_hash(cfg: M5EconomicPositiveConfig, input_hash: str, revision: str) -> str:
    payload = {
        "code_version": CODE_VERSION,
        "config": asdict(cfg),
        "input_hash": input_hash,
        "source_revision": revision,
    }
    return hashlib.sha256(_canonical(payload).encode()).hexdigest()[:12]


def _prepare(cfg: M5EconomicPositiveConfig) -> dict:
    prepared = common._prepare(_common_config(cfg))
    economic = build_economic_inputs(
        prepared["data"]["train"],
        n_users=prepared["data"]["n_users"],
        n_items=prepared["data"]["n_items"],
        q_v=prepared["q_v"],
        q_c=prepared["q_c"],
        clv_valid=prepared["clv_valid"],
        n_bins=cfg.economic_bins,
        shrinkage_strength=cfg.shrinkage_strength,
        degree_bins=cfg.shuffle_degree_bins,
    )
    prepared.update(economic)
    prepared["joint_shuffle"] = joint_degree_matched_shuffle(
        prepared, seed=cfg.shuffle_seed, degree_bins=cfg.shuffle_degree_bins
    )
    prepared["degree_gate"] = {
        "q_v": prepared["q_v"],
        "q_c": prepared["degree_percentile"],
        "clv_valid": prepared["clv_valid"],
        "user_economic_input": prepared["user_economic_input"],
        "user_economic_valid": prepared["user_economic_valid"],
    }
    prepared["config_hash"] = _config_hash(
        cfg, prepared["input_hash"], prepared["revision"]
    )
    return prepared


def arm_specifications(prepared: dict, cfg: M5EconomicPositiveConfig) -> list[dict]:
    return [
        {"model_id": M1_MODEL_ID, "role": "factorial_m1", "rho": 0.0,
         "weighted": False, "assignment": prepared, "assignment_name": "observed"},
        {"model_id": M2_MODEL_ID, "role": "factorial_m2", "rho": cfg.rho,
         "weighted": False, "assignment": prepared, "assignment_name": "observed"},
        {"model_id": M4P_MODEL_ID, "role": "factorial_m4_prime", "rho": 0.0,
         "weighted": True, "assignment": prepared, "assignment_name": "observed"},
        {"model_id": M5_MODEL_ID, "role": "factorial_m5", "rho": cfg.rho,
         "weighted": True, "assignment": prepared, "assignment_name": "observed"},
        {"model_id": M5_SHUFFLED_MODEL_ID, "role": "joint_assignment_control", "rho": cfg.rho,
         "weighted": True, "assignment": prepared["joint_shuffle"],
         "assignment_name": "degree_matched_joint_shuffle"},
        {"model_id": M5_DEGREE_GATE_MODEL_ID, "role": "loss_gate_control", "rho": cfg.rho,
         "weighted": True, "assignment": prepared["degree_gate"],
         "assignment_name": "degree_percentile_loss_gate"},
    ]


def _build_model(
    prepared: dict,
    cfg: M5EconomicPositiveConfig,
    spec: dict,
) -> M5EconomicLightGCN:
    data = prepared["data"]
    assignment = spec["assignment"]
    v3.set_seed(cfg.seed)
    return M5EconomicLightGCN(
        n_users=data["n_users"],
        n_items=data["n_items"],
        user_economic_input=assignment["user_economic_input"],
        user_economic_valid=assignment["user_economic_valid"],
        item_economic_input=prepared["item_economic_input"],
        item_economic_valid=prepared["item_economic_valid"],
        adj=data["adj"],
        id_dim=cfg.id_dim,
        economic_dim=cfg.economic_dim,
        rho=spec["rho"],
        n_layers=cfg.n_layers,
        pref_reg=cfg.pref_reg,
    ).to(v3.DEVICE)


def _train_weight_normalizer(
    prepared: dict, assignment: dict, lambda_: float
) -> float:
    users = prepared["data"]["tr_u"]
    items = prepared["data"]["tr_i"]
    raw = 1.0 + lambda_ * np.asarray(assignment["q_c"])[users] * (
        2.0 * prepared["item_amount_percentile"][items] - 1.0
    )
    mean = float(np.mean(raw))
    if not np.isfinite(mean) or mean <= 0:
        raise RuntimeError("양성 가중치 전역 정규화값이 잘못됐습니다")
    return mean


def _arm_hash(prepared: dict, cfg: M5EconomicPositiveConfig, spec: dict) -> str:
    return hashlib.sha256(
        _canonical(
            {
                "run": prepared["config_hash"],
                "model_id": spec["model_id"],
                "rho": spec["rho"],
                "weighted": spec["weighted"],
                "assignment": spec["assignment_name"],
                "seed": cfg.seed,
            }
        ).encode()
    ).hexdigest()[:12]


def _arm_paths(prepared: dict, cfg: M5EconomicPositiveConfig, model_id: str) -> dict[str, Path]:
    root = prepared["out_dir"] / "arms" / prepared["config_hash"]
    stem = f"{model_id}_s{cfg.seed}"
    return {"result": root / f"{stem}.json", "checkpoint": root / f"{stem}.pt"}


def _train_arm(
    model: M5EconomicLightGCN,
    prepared: dict,
    cfg: M5EconomicPositiveConfig,
    spec: dict,
    store: ProgressStore,
) -> dict:
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
    q_all = torch.as_tensor(spec["assignment"]["q_c"], device=v3.DEVICE)
    amount_all = torch.as_tensor(
        prepared["item_amount_percentile"], device=v3.DEVICE
    )
    normalizer = _train_weight_normalizer(
        prepared, spec["assignment"], cfg.positive_weight_lambda
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
            negatives_np = m4_helpers.sample_uniform_negative_matrix(
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
                row_weights = positive_row_weights(
                    q_all[users],
                    amount_all[positives],
                    train_mean_raw_weight=normalizer,
                    lambda_=cfg.positive_weight_lambda,
                )
            else:
                row_weights = torch.ones_like(positive_scores)
            bpr, diagnostics = weighted_multi_negative_bpr(
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
            f"weight-cv {record['row_weight_cv']:.3f} | {record['epoch_sec']:.0f}s"
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


def _run_arm(
    prepared: dict, cfg: M5EconomicPositiveConfig, spec: dict
) -> tuple[dict, M5EconomicLightGCN]:
    paths = _arm_paths(prepared, cfg, spec["model_id"])
    model = _build_model(prepared, cfg, spec)
    if paths["result"].exists() and paths["checkpoint"].exists():
        print(f"  [cached] {spec['model_id']} 완료 결과 재사용")
        payload = json.loads(paths["result"].read_text(encoding="utf-8"))
        checkpoint = torch.load(paths["checkpoint"], map_location=v3.DEVICE)
        if checkpoint.get("input_hash") != prepared["input_hash"]:
            raise RuntimeError("cached checkpoint와 현재 입력 hash가 다릅니다")
        model.load_state_dict(checkpoint["state"], strict=True)
        model.eval()
        return payload, model

    store = ProgressStore(
        prepared["out_dir"] / "progress" / prepared["config_hash"],
        RunIdentity(
            stage="historical_development_train",
            model_id=spec["model_id"],
            seed=cfg.seed,
            config_hash=_arm_hash(prepared, cfg, spec),
            source_revision=prepared["revision"],
            input_hash=prepared["input_hash"],
        ),
    )
    training = _train_arm(model, prepared, cfg, spec, store)
    model.eval()
    paths["checkpoint"].parent.mkdir(parents=True, exist_ok=True)
    temporary = paths["checkpoint"].with_suffix(".pt.tmp")
    torch.save(
        {
            "state": clone_state(model),
            "model_id": spec["model_id"],
            "role": spec["role"],
            "rho": spec["rho"],
            "positive_weighted": spec["weighted"],
            "assignment": spec["assignment_name"],
            "config": asdict(cfg),
            "training": training,
            "source_revision": prepared["revision"],
            "input_hash": prepared["input_hash"],
        },
        temporary,
    )
    os.replace(temporary, paths["checkpoint"])
    raw_metrics, _ = moe._flat_evaluation(
        model,
        0.0,
        prepared["cache"],
        prepared["meta"],
        prepared["data"],
        prepared["base_cfg"],
        per_user=False,
    )
    payload = {
        "model_id": spec["model_id"],
        "role": spec["role"],
        "seed": cfg.seed,
        "split": "historical_development_days_684_690",
        "final_epoch": cfg.epochs,
        "rho": spec["rho"],
        "positive_weight_lambda": cfg.positive_weight_lambda if spec["weighted"] else 0.0,
        "negative_count": cfg.negative_count,
        "hard_negative": False,
        "clv_assignment": spec["assignment_name"],
        "metrics": test10._public_metrics(raw_metrics),
        "diagnostics": (
            model.representation_diagnostics()
            | prepared["economic_input_diagnostics"]
        ),
        "training": training,
        "checkpoint": str(paths["checkpoint"]),
        "checkpoint_sha256": file_sha256(paths["checkpoint"]),
    }
    test10._atomic_json(paths["result"], payload)
    store.mark_complete(
        epoch=cfg.epochs,
        max_epoch=cfg.epochs,
        selection="none",
        split="historical_development_days_684_690",
        checkpoint_path=str(paths["checkpoint"]),
        result_path=str(paths["result"]),
    )
    return payload, model


def interaction_rows(metric_rows: dict[str, dict]) -> pd.DataFrame:
    a = metric_rows[M1_MODEL_ID]
    b = metric_rows[M2_MODEL_ID]
    c = metric_rows[M4P_MODEL_ID]
    d = metric_rows[M5_MODEL_ID]
    metrics = ACCURACY_METRICS + (PRIMARY_METRIC, "price_purchase_amount_weighted_hit@10")
    rows = []
    for metric in metrics:
        if metric not in a or metric not in b or metric not in c or metric not in d:
            continue
        m2_effect = float(b[metric] - a[metric])
        m4_effect = float(c[metric] - a[metric])
        m5_effect = float(d[metric] - a[metric])
        rows.append(
            {
                "metric": metric,
                "m2_effect": m2_effect,
                "m4_prime_effect": m4_effect,
                "m5_effect": m5_effect,
                "interaction_effect": float((d[metric] - c[metric]) - m2_effect),
            }
        )
    return pd.DataFrame(rows)


def screening_reading(metric_rows: dict[str, dict]) -> dict:
    a = metric_rows[M1_MODEL_ID]
    c = metric_rows[M4P_MODEL_ID]
    d = metric_rows[M5_MODEL_ID]
    e = metric_rows[M5_SHUFFLED_MODEL_ID]
    f = metric_rows[M5_DEGREE_GATE_MODEL_ID]
    accuracy_ratios = {
        metric: float(d[metric] / a[metric]) for metric in ACCURACY_METRICS
    }
    primary_deltas = {
        "vs_m1": float(d[PRIMARY_METRIC] - a[PRIMARY_METRIC]),
        "vs_m4_prime": float(d[PRIMARY_METRIC] - c[PRIMARY_METRIC]),
        "vs_joint_shuffle": float(d[PRIMARY_METRIC] - e[PRIMARY_METRIC]),
        "vs_degree_gate": float(d[PRIMARY_METRIC] - f[PRIMARY_METRIC]),
    }
    interaction = interaction_rows(metric_rows).set_index("metric")
    interaction_value = float(interaction.at[PRIMARY_METRIC, "interaction_effect"])
    exposure = {
        "coverage@10_ratio_vs_m1": float(d["coverage@10"] / a["coverage@10"]),
        "n_distinct@10_ratio_vs_m1": float(d["n_distinct@10"] / a["n_distinct@10"]),
        "top10_share@10_ratio_vs_m1": float(d["top10_share@10"] / a["top10_share@10"]),
    }
    primary_pass = all(value > 0.0 for value in primary_deltas.values())
    interaction_pass = interaction_value > 0.0
    accuracy_pass = all(value >= 0.99 for value in accuracy_ratios.values())
    exposure_pass = bool(
        exposure["coverage@10_ratio_vs_m1"] >= 0.95
        and exposure["n_distinct@10_ratio_vs_m1"] >= 0.95
        and exposure["top10_share@10_ratio_vs_m1"] <= 1.05
    )
    return {
        "positive_screen": bool(
            primary_pass and interaction_pass and accuracy_pass and exposure_pass
        ),
        "primary_pass": primary_pass,
        "interaction_pass": interaction_pass,
        "accuracy_pass": accuracy_pass,
        "exposure_pass": exposure_pass,
        "primary_metric": PRIMARY_METRIC,
        "primary_deltas": primary_deltas,
        "primary_interaction_effect": interaction_value,
        "accuracy_ratios_vs_m1": accuracy_ratios,
        **exposure,
        "next_if_positive": "repeat frozen development seeds, then H&M seed 42",
        "statistical_note": "seed 42 exploratory screen; no significance claim",
    }


@torch.no_grad()
def _score_diagnostics(
    model: M5EconomicLightGCN,
    users: np.ndarray,
    top50: np.ndarray,
    *,
    model_id: str,
) -> dict:
    width = top50.shape[1]
    pair_users = np.repeat(users.astype(np.int64), width)
    pair_items = top50.reshape(-1).astype(np.int64)
    collected = {key: [] for key in ("id", "economic", "full")}
    for start in range(0, len(pair_users), 65536):
        user_tensor = torch.as_tensor(
            pair_users[start : start + 65536], dtype=torch.long, device=v3.DEVICE
        )
        item_tensor = torch.as_tensor(
            pair_items[start : start + 65536], dtype=torch.long, device=v3.DEVICE
        )
        components = model.candidate_score_components(user_tensor, item_tensor)
        for key in collected:
            collected[key].append(components[key].cpu().numpy())
    values = {key: np.concatenate(value).astype(np.float64) for key, value in collected.items()}
    id_std = float(values["id"].std())
    economic_std = float(values["economic"].std())
    return {
        "model_id": model_id,
        "candidate_pair_count": int(len(pair_users)),
        "id_score_std": id_std,
        "economic_score_std": economic_std,
        "economic_score_std_ratio_to_id": economic_std / id_std if id_std > 0 else np.nan,
        "economic_score_mean_abs": float(np.abs(values["economic"]).mean()),
        "max_full_decomposition_error": float(
            np.max(np.abs(values["full"] - values["id"] - values["economic"]))
        ),
    }


def _economic_recommendation_rows(
    model_id: str,
    users: np.ndarray,
    top50: np.ndarray,
    prepared: dict,
) -> list[dict]:
    top10 = top50[:, :10]
    mean_amount = prepared["item_amount_percentile"][top10].mean(axis=1)
    rows = []
    for grouping, values in (
        ("q_v_quartile", prepared["q_v"][users]),
        ("degree_quartile", prepared["degree_percentile"][users]),
    ):
        bins = np.minimum((np.asarray(values) * 4).astype(int), 3)
        for group in range(4):
            mask = bins == group
            rows.append(
                {
                    "model_id": model_id,
                    "grouping": grouping,
                    "quartile": group + 1,
                    "n_users": int(mask.sum()),
                    "mean_recommended_amount_percentile@10": (
                        float(mean_amount[mask].mean()) if mask.any() else np.nan
                    ),
                }
            )
    return rows


def run_m5_economic_positive_screen(
    cfg: M5EconomicPositiveConfig | None = None,
) -> pd.DataFrame:
    cfg = validate_config(cfg or configure_m5_economic_positive_run())
    summary = preflight_summary(cfg)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    prepared = _prepare(cfg)
    arms: dict[str, dict] = {}
    models: dict[str, M5EconomicLightGCN] = {}
    for spec in arm_specifications(prepared, cfg):
        print(f"\n===== {spec['model_id']} | seed {cfg.seed} | fixed {cfg.epochs} epochs =====")
        arms[spec["model_id"]], models[spec["model_id"]] = _run_arm(
            prepared, cfg, spec
        )

    rows = []
    metric_rows = {}
    for model_id in MODEL_IDS:
        arm = arms[model_id]
        metric_rows[model_id] = arm["metrics"]
        rows.append(
            {
                "model_id": model_id,
                "role": arm["role"],
                "seed": arm["seed"],
                "split": arm["split"],
                "final_epoch": arm["final_epoch"],
                "rho": arm["rho"],
                "positive_weight_lambda": arm["positive_weight_lambda"],
                "clv_assignment": arm["clv_assignment"],
                **arm["diagnostics"],
                **arm["training"].get("final_diagnostics", {}),
                **arm["metrics"],
            }
        )
    frame = pd.DataFrame(rows)
    comparison = report_helpers._metric_comparison(
        metric_rows,
        references=(
            M1_MODEL_ID,
            M4P_MODEL_ID,
            M5_SHUFFLED_MODEL_ID,
            M5_DEGREE_GATE_MODEL_ID,
        ),
    )
    interactions = interaction_rows(metric_rows)
    reading = screening_reading(metric_rows)

    topk: dict[str, np.ndarray] = {}
    users: np.ndarray | None = None
    score_rows = []
    economic_rows = []
    for model_id in MODEL_IDS:
        arm_users, arm_top50 = report_helpers._masked_topk(
            models[model_id], prepared, max_k=cfg.diagnostic_max_k
        )
        if users is None:
            users = arm_users
        elif not np.array_equal(users, arm_users):
            raise RuntimeError("대조군별 평가 사용자가 다릅니다")
        topk[model_id] = arm_top50
        score_rows.append(
            _score_diagnostics(
                models[model_id], arm_users, arm_top50, model_id=model_id
            )
        )
        economic_rows.extend(
            _economic_recommendation_rows(
                model_id, arm_users, arm_top50, prepared
            )
        )
    assert users is not None
    overlap_frames = []
    for reference in (M1_MODEL_ID, M5_SHUFFLED_MODEL_ID, M5_DEGREE_GATE_MODEL_ID):
        overlap = report_helpers.topk_overlap_summary(
            topk[reference], topk[M5_MODEL_ID], prepared["cache"].seg, k=10
        )
        overlap.insert(0, "reference", reference)
        overlap.insert(1, "model_id", M5_MODEL_ID)
        overlap_frames.append(overlap)
    overlap_frame = pd.concat(overlap_frames, ignore_index=True)
    score_frame = pd.DataFrame(score_rows)
    economic_frame = pd.DataFrame(economic_rows)

    m5_score_ratio = float(
        score_frame.set_index("model_id").at[
            M5_MODEL_ID, "economic_score_std_ratio_to_id"
        ]
    )
    m5_weight_cv = float(
        arms[M5_MODEL_ID]["training"]["final_diagnostics"]["row_weight_cv"]
    )
    reading["intervention_checks"] = {
        "economic_score_std_ratio_to_id": m5_score_ratio,
        "economic_score_ratio_at_least_0_02": m5_score_ratio >= 0.02,
        "row_weight_std_over_mean": m5_weight_cv,
        "row_weight_cv_at_least_0_10": m5_weight_cv >= 0.10,
        "intervention_operational": bool(
            m5_score_ratio >= 0.02 and m5_weight_cv >= 0.10
        ),
    }

    out = prepared["out_dir"]
    stem = f"m5_economic_positive_weight_{prepared['config_hash']}"
    paths = {
        "absolute_csv": out / f"{stem}.csv",
        "comparison_csv": out / f"{stem}_comparison.csv",
        "interaction_csv": out / f"{stem}_interaction.csv",
        "score_diagnostics_csv": out / f"{stem}_score_diagnostics.csv",
        "top10_overlap_csv": out / f"{stem}_top10_overlap.csv",
        "economic_recommendations_csv": out / f"{stem}_economic_recommendations.csv",
        "json": out / f"{stem}.json",
    }
    test10._atomic_csv(paths["absolute_csv"], frame)
    test10._atomic_csv(paths["comparison_csv"], comparison)
    test10._atomic_csv(paths["interaction_csv"], interactions)
    test10._atomic_csv(paths["score_diagnostics_csv"], score_frame)
    test10._atomic_csv(paths["top10_overlap_csv"], overlap_frame)
    test10._atomic_csv(paths["economic_recommendations_csv"], economic_frame)
    test10._atomic_json(
        paths["json"],
        {
            "code_version": CODE_VERSION,
            "source_revision": prepared["revision"],
            "config": asdict(cfg),
            "preflight": summary,
            "input_manifest": prepared["manifest"],
            "economic_input_diagnostics": prepared["economic_input_diagnostics"],
            "absolute_rows": frame.to_dict("records"),
            "comparison_rows": comparison.to_dict("records"),
            "interaction_rows": interactions.to_dict("records"),
            "score_diagnostic_rows": score_frame.to_dict("records"),
            "top10_overlap_rows": overlap_frame.to_dict("records"),
            "economic_recommendation_rows": economic_frame.to_dict("records"),
            "screening_reading": reading,
            "joint_shuffle": {
                "method": "joint tuple permutation within degree deciles",
                "source_user": prepared["joint_shuffle"]["source_user"].tolist(),
            },
            "arms": arms,
            "result_paths": {key: str(value) for key, value in paths.items()},
        },
    )
    frame.attrs["comparison"] = comparison
    frame.attrs["interaction"] = interactions
    frame.attrs["score_diagnostics"] = score_frame
    frame.attrs["top10_overlap"] = overlap_frame
    frame.attrs["economic_recommendations"] = economic_frame
    frame.attrs["decision"] = reading
    frame.attrs["result_paths"] = {key: str(value) for key, value in paths.items()}
    return frame
