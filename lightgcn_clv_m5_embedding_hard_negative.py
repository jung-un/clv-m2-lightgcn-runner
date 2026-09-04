"""Minimal M5 screen: the fixed M2 representation under the fixed M4 loss."""

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

from clv_constrained_economic_embedding_model import (
    ConstrainedCLVEconomicLightGCN,
)
from clv_m4_clv_hard_negative_loss import (
    multi_negative_bpr,
    sampled_l2_multineg,
)
from clv_run_state import ProgressStore, RunIdentity, clone_state, file_sha256
import lightgcn_clv_axis_specific_test10 as test10
import lightgcn_clv_constrained_economic_embedding as m2
import lightgcn_clv_gradient_isolated_economic_interaction as report_helpers
import lightgcn_clv_m4_clv_hard_negative as m4
import lightgcn_clv_moe as moe
import lightgcn_clv_v3 as v3


CODE_VERSION = "m5-m2-m4-joint-historical-screen-v1"
M1_K5_MODEL_ID = "m1_multineg_mean_k5"
M2_K5_MODEL_ID = "m2_clv_embedding_multineg_mean_k5"
M4_MODEL_ID = "m4_clv_hard_k5"
M5_MODEL_ID = "m5_clv_embedding_hard_k5"
M5_SHUFFLED_MODEL_ID = "m5_degree_matched_clv_shuffle"
TRAINED_MODEL_IDS = (
    M1_K5_MODEL_ID,
    M2_K5_MODEL_ID,
    M4_MODEL_ID,
    M5_MODEL_ID,
    M5_SHUFFLED_MODEL_ID,
)
ACCURACY_METRICS = (
    "recall@10",
    "ndcg@10",
    "recall@20",
    "ndcg@20",
    "recall@50",
    "ndcg@50",
)
INTERACTION_METRICS = ACCURACY_METRICS + (
    "price_purchase_amount_weighted_hit@10",
    "고CLV_recall@10",
    "고CLV_ndcg@10",
)


@dataclass(frozen=True)
class M5Config:
    dataset: str = "dunnhumby"
    seed: int = 42
    time_cutoff: int = 690
    evaluation_days: int = 7
    epochs: int = 100
    id_dim: int = 64
    clv_dim: int = 3
    rho: float = 0.05
    item_price_budget: float = 0.25
    n_layers: int = 2
    negative_count: int = 5
    batch_size: int = 8192
    lr: float = 5e-4
    pref_reg: float = 1e-3
    input_days: int = 365
    shuffle_degree_bins: int = 10
    shuffle_seed: int = 42
    out_dir: str = ""
    baseline_result_dir: str = ""


def configure_m5_run(**overrides) -> M5Config:
    defaults = {
        "out_dir": (
            f"{v3.default_out_dir('dunnhumby')}"
            "_m5_m2_m4_joint_historical_screen_v1"
        ),
        "baseline_result_dir": (
            f"{v3.default_out_dir('dunnhumby')}"
            "_m2_repeatshare_historical_backtest_v1"
        ),
    }
    return validate_m5_config(M5Config(**(defaults | overrides)))


def validate_m5_config(cfg: M5Config) -> M5Config:
    required = {
        "dataset": "dunnhumby",
        "seed": 42,
        "time_cutoff": 690,
        "evaluation_days": 7,
        "epochs": 100,
        "id_dim": 64,
        "clv_dim": 3,
        "rho": 0.05,
        "item_price_budget": 0.25,
        "n_layers": 2,
        "negative_count": 5,
        "input_days": 365,
        "shuffle_degree_bins": 10,
        "shuffle_seed": 42,
    }
    for key, expected in required.items():
        if getattr(cfg, key) != expected:
            raise ValueError(f"빠른 M5 screen은 {key}={expected!r}이어야 합니다")
    if cfg.batch_size <= 0 or cfg.lr <= 0 or cfg.pref_reg < 0:
        raise ValueError("빠른 M5 screen 학습 설정이 잘못됐습니다")
    if not cfg.out_dir or not cfg.baseline_result_dir:
        raise ValueError("빠른 M5 screen은 out_dir와 baseline_result_dir가 필요합니다")
    return cfg


def preflight_summary(cfg: M5Config) -> dict:
    cfg = validate_m5_config(cfg)
    return {
        "code_version": CODE_VERSION,
        "dataset": cfg.dataset,
        "seed": cfg.seed,
        "trained_models": list(TRAINED_MODEL_IDS),
        "research_axis": "M5 partial combination: M2 representation plus M4 loss",
        "historical_development_split": {
            "train_end_inclusive": 683,
            "evaluation_start_inclusive": 684,
            "evaluation_end_inclusive": 690,
            "final_test_constructed": False,
            "holdout_constructed": False,
        },
        "m2": {
            "architecture": "ID(64)|CLV relation(2)|explicit price fit(1)",
            "rho": cfg.rho,
            "item_price_budget": cfg.item_price_budget,
            "joint_binary_lightgcn_propagation": True,
        },
        "m4": {
            "uniform_negative_count": cfg.negative_count,
            "loss": "(1-q_C)*mean_BPR + q_C*BPR(highest-scored negative)",
            "per_positive_loss_mass": 1.0,
        },
        "attribution_control": {
            "method": "jointly permute (q_N,q_V,q_C,valid) within user-degree deciles",
            "same_permutation_in_m2_and_m4": True,
        },
        "fixed": {
            "task": "new-item recommendation",
            "graph": "binary",
            "negative_sampling": "uniform",
            "m3_edge_weight": False,
            "new_auxiliary_loss": False,
            "external_reranking": False,
            "one_training_loop_and_optimizer_per_arm": True,
            "min_user_interactions": 1,
            "min_item_interactions": 1,
            "epochs": cfg.epochs,
            "validation_or_epoch_selection": False,
        },
        "reading_rule": {
            "accuracy": "M5 accuracy geometric mean >= M4 and every metric >= 99% of M4",
            "economic": "M5 weighted hit@10 > M4 and joint CLV shuffle",
            "high_clv": "M5 high-CLV NDCG@10 > M4 and joint CLV shuffle",
            "interaction": "(M5-M4)-(M2-M1) weighted hit@10 > 0",
            "attribution": "M5 six-metric geometric mean > joint CLV shuffle",
            "exposure": "coverage/distinct >=95% and top10 share <=105% of M4",
            "statistical_note": "seed 42 exploratory screen; no significance claim",
        },
        "out_dir": cfg.out_dir,
        "baseline_result_dir": cfg.baseline_result_dir,
    }


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def _config_hash(cfg: M5Config, input_hash: str, revision: str) -> str:
    payload = {
        "code_version": CODE_VERSION,
        "config": asdict(cfg),
        "input_hash": input_hash,
        "source_revision": revision,
    }
    return hashlib.sha256(_canonical(payload).encode()).hexdigest()[:12]


def _m2_config(cfg: M5Config) -> m2.ConstrainedEconomicConfig:
    return m2.ConstrainedEconomicConfig(
        dataset=cfg.dataset,
        seed=cfg.seed,
        time_cutoff=cfg.time_cutoff,
        evaluation_days=cfg.evaluation_days,
        epochs=cfg.epochs,
        id_dim=cfg.id_dim,
        clv_dim=cfg.clv_dim,
        rho=cfg.rho,
        item_price_budget=cfg.item_price_budget,
        n_layers=cfg.n_layers,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        pref_reg=cfg.pref_reg,
        input_days=cfg.input_days,
        diagnostic_max_k=50,
        include_degree_matched_shuffle=False,
        shuffle_degree_bins=cfg.shuffle_degree_bins,
        shuffle_seed=cfg.shuffle_seed,
        out_dir=cfg.out_dir,
        baseline_result_dir=cfg.baseline_result_dir,
    )


def _prepare(cfg: M5Config) -> dict:
    prepared = m2._prepare(_m2_config(cfg))
    prepared["degree_matched_shuffle"] = m2._degree_matched_clv_shuffle(
        prepared, cfg
    )
    prepared["config_hash"] = _config_hash(
        cfg, prepared["input_hash"], prepared["revision"]
    )
    return prepared


def _build_model(
    prepared: dict,
    cfg: M5Config,
    *,
    rho: float,
    assignment: dict,
) -> ConstrainedCLVEconomicLightGCN:
    data = prepared["data"]
    v3.set_seed(cfg.seed)
    return ConstrainedCLVEconomicLightGCN(
        n_users=data["n_users"],
        n_items=data["n_items"],
        q_n=assignment["q_n"],
        q_v=assignment["q_v"],
        q_c=assignment["q_c"],
        user_clv_valid=assignment["clv_valid"],
        item_economic_features=prepared["item_economic"],
        item_economic_valid=prepared["item_economic_valid"],
        adj=data["adj"],
        id_dim=cfg.id_dim,
        clv_dim=cfg.clv_dim,
        rho=rho,
        item_price_budget=cfg.item_price_budget,
        n_layers=cfg.n_layers,
        pref_reg=cfg.pref_reg,
    ).to(v3.DEVICE)


def arm_specifications(prepared: dict, cfg: M5Config) -> list[dict]:
    observed = prepared
    shuffled = prepared["degree_matched_shuffle"]
    return [
        {
            "model_id": M1_K5_MODEL_ID,
            "role": "factorial_m1",
            "rho": 0.0,
            "assignment": observed,
            "assignment_name": "observed",
            "hard_negative": False,
        },
        {
            "model_id": M2_K5_MODEL_ID,
            "role": "factorial_m2",
            "rho": cfg.rho,
            "assignment": observed,
            "assignment_name": "observed",
            "hard_negative": False,
        },
        {
            "model_id": M4_MODEL_ID,
            "role": "factorial_m4",
            "rho": 0.0,
            "assignment": observed,
            "assignment_name": "observed",
            "hard_negative": True,
        },
        {
            "model_id": M5_MODEL_ID,
            "role": "factorial_m5",
            "rho": cfg.rho,
            "assignment": observed,
            "assignment_name": "observed",
            "hard_negative": True,
        },
        {
            "model_id": M5_SHUFFLED_MODEL_ID,
            "role": "joint_attribution_control",
            "rho": cfg.rho,
            "assignment": shuffled,
            "assignment_name": "degree_matched_shuffle",
            "hard_negative": True,
        },
    ]


def _arm_paths(prepared: dict, cfg: M5Config, model_id: str) -> dict[str, Path]:
    root = prepared["out_dir"] / "arms" / prepared["config_hash"]
    stem = f"{model_id}_s{cfg.seed}"
    return {"result": root / f"{stem}.json", "checkpoint": root / f"{stem}.pt"}


def _arm_hash(prepared: dict, cfg: M5Config, spec: dict) -> str:
    payload = {
        "run": prepared["config_hash"],
        "model_id": spec["model_id"],
        "seed": cfg.seed,
        "rho": spec["rho"],
        "assignment": spec["assignment_name"],
        "hard_negative": spec["hard_negative"],
        "negative_count": cfg.negative_count,
    }
    return hashlib.sha256(_canonical(payload).encode()).hexdigest()[:12]


def _train_arm(
    model: ConstrainedCLVEconomicLightGCN,
    prepared: dict,
    cfg: M5Config,
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
    q_values = (
        np.asarray(spec["assignment"]["q_c"], dtype=np.float32)
        if spec["hard_negative"]
        else np.zeros(data["n_users"], dtype=np.float32)
    )
    q_all = torch.as_tensor(q_values, device=v3.DEVICE)
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
            "hardest_negative_weight_mean": 0.0,
            "positive_hardest_gap": 0.0,
            "effective_gradient_mass": 0.0,
        }
        weight_error = 0.0
        last_gradients: dict[str, float] = {}
        for batch in range(n_batches):
            index = permutation[
                batch * cfg.batch_size : (batch + 1) * cfg.batch_size
            ]
            users_np, positives_np = tr_u[index], tr_i[index]
            negatives_np = m4.sample_uniform_negative_matrix(
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
            positive_scores = (user_z[users] * item_z[positives]).sum(1)
            negative_scores = (
                user_z[users, None, :] * item_z[negatives]
            ).sum(2)
            bpr, diagnostics = multi_negative_bpr(
                positive_scores, negative_scores, q_all[users]
            )
            reg = sampled_l2_multineg(
                model.E_u.weight[users],
                model.E_i.weight[positives],
                model.E_i.weight[negatives],
                coefficient=cfg.pref_reg,
            )
            loss = bpr + reg
            optimizer.zero_grad()
            loss.backward()
            last_gradients = model.training_gradient_diagnostics()
            optimizer.step()
            totals["loss"] += float(loss.detach())
            totals["bpr"] += float(bpr.detach())
            totals["p_correct"] += float(diagnostics["p_correct"])
            totals["hardest_negative_weight_mean"] += float(
                diagnostics["hardest_weight_mean"]
            )
            totals["positive_hardest_gap"] += float(
                diagnostics["positive_hardest_gap"]
            )
            totals["effective_gradient_mass"] += float(
                diagnostics["effective_gradient_mass"]
            )
            weight_error = max(
                weight_error, float(diagnostics["row_weight_sum_error"])
            )
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
            "row_weight_sum_max_error": float(weight_error),
            "epoch_sec": float(time.time() - epoch_started),
            **last_gradients,
            **model.epoch_training_diagnostics(),
        }
        history.append(record)
        print(
            f"  [{spec['model_id']}] ep {epoch:3d}/{cfg.epochs} | "
            f"loss {record['loss']:.4f} | P(pos>neg) {record['p_correct']:.3f} | "
            f"hard-w {record['hardest_negative_weight_mean']:.3f} | "
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


def _run_arm(prepared: dict, cfg: M5Config, spec: dict) -> dict:
    paths = _arm_paths(prepared, cfg, spec["model_id"])
    model = _build_model(
        prepared, cfg, rho=spec["rho"], assignment=spec["assignment"]
    )
    if paths["result"].exists() and paths["checkpoint"].exists():
        print(f"  [cached] {spec['model_id']} 완료 결과 재사용")
        return json.loads(paths["result"].read_text(encoding="utf-8"))
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
            "hard_negative": spec["hard_negative"],
            "clv_assignment": spec["assignment_name"],
            "config": asdict(cfg),
            "training": training,
            "source_revision": prepared["revision"],
            "input_hash": prepared["input_hash"],
        },
        temporary,
    )
    os.replace(temporary, paths["checkpoint"])
    metrics_raw, _ = moe._flat_evaluation(
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
        "negative_count": cfg.negative_count,
        "hard_negative": spec["hard_negative"],
        "clv_assignment": spec["assignment_name"],
        "metrics": test10._public_metrics(metrics_raw),
        "diagnostics": model.representation_diagnostics(),
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
    return payload


def _geomean_ratio(model: dict, reference: dict) -> float:
    ratios = [model[metric] / reference[metric] for metric in ACCURACY_METRICS]
    return float(np.exp(np.mean(np.log(ratios))))


def interaction_rows(metric_rows: dict[str, dict]) -> pd.DataFrame:
    m1 = metric_rows[M1_K5_MODEL_ID]
    m2_metrics = metric_rows[M2_K5_MODEL_ID]
    m4_metrics = metric_rows[M4_MODEL_ID]
    m5_metrics = metric_rows[M5_MODEL_ID]
    rows = []
    for metric in INTERACTION_METRICS:
        m2_effect = float(m2_metrics[metric] - m1[metric])
        m4_effect = float(m4_metrics[metric] - m1[metric])
        m5_effect = float(m5_metrics[metric] - m1[metric])
        rows.append(
            {
                "metric": metric,
                "m2_effect": m2_effect,
                "m4_effect": m4_effect,
                "m5_effect": m5_effect,
                "interaction_effect": float(m5_effect - m2_effect - m4_effect),
            }
        )
    return pd.DataFrame(rows)


def screening_reading(metric_rows: dict[str, dict]) -> dict:
    m4_metrics = metric_rows[M4_MODEL_ID]
    m5_metrics = metric_rows[M5_MODEL_ID]
    shuffled = metric_rows[M5_SHUFFLED_MODEL_ID]
    accuracy_ratios = {
        metric: float(m5_metrics[metric] / m4_metrics[metric])
        for metric in ACCURACY_METRICS
    }
    accuracy_geomean = _geomean_ratio(m5_metrics, m4_metrics)
    attribution_geomean = _geomean_ratio(m5_metrics, shuffled)
    interaction = interaction_rows(metric_rows).set_index("metric")
    weighted_metric = "price_purchase_amount_weighted_hit@10"
    weighted_vs_m4 = float(m5_metrics[weighted_metric] - m4_metrics[weighted_metric])
    weighted_vs_shuffle = float(m5_metrics[weighted_metric] - shuffled[weighted_metric])
    high_ndcg_vs_m4 = float(
        m5_metrics["고CLV_ndcg@10"] - m4_metrics["고CLV_ndcg@10"]
    )
    high_ndcg_vs_shuffle = float(
        m5_metrics["고CLV_ndcg@10"] - shuffled["고CLV_ndcg@10"]
    )
    exposure = {
        "coverage@10_ratio_vs_m4": float(
            m5_metrics["coverage@10"] / m4_metrics["coverage@10"]
        ),
        "n_distinct@10_ratio_vs_m4": float(
            m5_metrics["n_distinct@10"] / m4_metrics["n_distinct@10"]
        ),
        "top10_share@10_ratio_vs_m4": float(
            m5_metrics["top10_share@10"] / m4_metrics["top10_share@10"]
        ),
    }
    accuracy_pass = bool(
        accuracy_geomean >= 1.0
        and all(value >= 0.99 for value in accuracy_ratios.values())
    )
    economic_pass = bool(weighted_vs_m4 > 0.0 and weighted_vs_shuffle > 0.0)
    high_clv_pass = bool(high_ndcg_vs_m4 > 0.0 and high_ndcg_vs_shuffle > 0.0)
    interaction_pass = bool(
        interaction.loc[weighted_metric, "interaction_effect"] > 0.0
    )
    attribution_pass = bool(attribution_geomean > 1.0)
    exposure_pass = bool(
        exposure["coverage@10_ratio_vs_m4"] >= 0.95
        and exposure["n_distinct@10_ratio_vs_m4"] >= 0.95
        and exposure["top10_share@10_ratio_vs_m4"] <= 1.05
    )
    return {
        "positive_screen": bool(
            accuracy_pass
            and economic_pass
            and high_clv_pass
            and interaction_pass
            and attribution_pass
            and exposure_pass
        ),
        "accuracy_pass": accuracy_pass,
        "economic_pass": economic_pass,
        "high_clv_pass": high_clv_pass,
        "interaction_pass": interaction_pass,
        "attribution_pass": attribution_pass,
        "exposure_pass": exposure_pass,
        "accuracy_ratios_vs_m4": accuracy_ratios,
        "accuracy_geomean_ratio_vs_m4": accuracy_geomean,
        "accuracy_geomean_ratio_vs_joint_shuffle": attribution_geomean,
        "weighted_hit@10_delta_vs_m4": weighted_vs_m4,
        "weighted_hit@10_delta_vs_joint_shuffle": weighted_vs_shuffle,
        "high_clv_ndcg@10_delta_vs_m4": high_ndcg_vs_m4,
        "high_clv_ndcg@10_delta_vs_joint_shuffle": high_ndcg_vs_shuffle,
        "weighted_hit@10_interaction_effect": float(
            interaction.loc[weighted_metric, "interaction_effect"]
        ),
        **exposure,
        "next_if_positive": "repeat the frozen M5 on several seeds before any final test",
        "statistical_note": "seed 42 exploratory screen; no significance claim",
    }


def run_m5_screen(cfg: M5Config | None = None) -> pd.DataFrame:
    cfg = validate_m5_config(cfg or configure_m5_run())
    summary = preflight_summary(cfg)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    prepared = _prepare(cfg)
    arms: dict[str, dict] = {}
    for spec in arm_specifications(prepared, cfg):
        print(f"\n===== {spec['model_id']} | seed {cfg.seed} | fixed {cfg.epochs} epochs =====")
        arms[spec["model_id"]] = _run_arm(prepared, cfg, spec)

    rows = []
    for model_id in TRAINED_MODEL_IDS:
        arm = arms[model_id]
        rows.append(
            {
                "model_id": model_id,
                "role": arm["role"],
                "seed": arm["seed"],
                "split": arm["split"],
                "final_epoch": arm["final_epoch"],
                "rho": arm["rho"],
                "negative_count": arm["negative_count"],
                "hard_negative": arm["hard_negative"],
                "clv_assignment": arm["clv_assignment"],
                **arm["diagnostics"],
                **arm["training"].get("final_diagnostics", {}),
                **arm["metrics"],
            }
        )
    frame = pd.DataFrame(rows)
    metric_rows = {model_id: arms[model_id]["metrics"] for model_id in TRAINED_MODEL_IDS}
    comparison = report_helpers._metric_comparison(
        metric_rows,
        references=(M1_K5_MODEL_ID, M4_MODEL_ID, M5_SHUFFLED_MODEL_ID),
    )
    interactions = interaction_rows(metric_rows)
    reading = screening_reading(metric_rows)
    out = prepared["out_dir"]
    stem = f"m5_m2_m4_joint_{prepared['config_hash']}"
    paths = {
        "absolute_csv": out / f"{stem}.csv",
        "comparison_csv": out / f"{stem}_comparison.csv",
        "interaction_csv": out / f"{stem}_interaction.csv",
        "json": out / f"{stem}.json",
    }
    test10._atomic_csv(paths["absolute_csv"], frame)
    test10._atomic_csv(paths["comparison_csv"], comparison)
    test10._atomic_csv(paths["interaction_csv"], interactions)
    test10._atomic_json(
        paths["json"],
        {
            "code_version": CODE_VERSION,
            "source_revision": prepared["revision"],
            "config": asdict(cfg),
            "preflight": summary,
            "input_manifest": prepared["manifest"],
            "absolute_rows": frame.to_dict("records"),
            "comparison_rows": comparison.to_dict("records"),
            "interaction_rows": interactions.to_dict("records"),
            "screening_reading": reading,
            "degree_matched_shuffle": {
                key: value
                for key, value in prepared["degree_matched_shuffle"].items()
                if key
                not in {
                    "q_n",
                    "q_v",
                    "q_c",
                    "clv_valid",
                    "source_user",
                    "stratum",
                    "user_degree",
                }
            },
            "arms": arms,
            "result_paths": {key: str(value) for key, value in paths.items()},
        },
    )
    frame.attrs["comparison"] = comparison.to_dict("records")
    frame.attrs["interaction"] = interactions.to_dict("records")
    frame.attrs["screening_reading"] = reading
    frame.attrs["result_paths"] = {key: str(value) for key, value in paths.items()}

    print("\n1) 절대지표: M1(K=5), M2, M4, M5, M5 CLV 순열")
    print(frame.to_string(index=False))
    print("\n2) 대조군별 비교")
    print(comparison.to_string(index=False))
    print("\n3) M2×M4 상호작용")
    print(interactions.to_string(index=False))
    print("\n4) 사전 판정")
    print(json.dumps(reading, ensure_ascii=False, indent=2))
    print("\n5) 저장 파일")
    print(json.dumps(frame.attrs["result_paths"], ensure_ascii=False, indent=2))
    return frame


if __name__ == "__main__":
    print(json.dumps(preflight_summary(configure_m5_run()), ensure_ascii=False, indent=2))
