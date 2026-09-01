"""Seed-42 historical screen for CLV-conditioned multi-negative BPR."""

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

from clv_m4_clv_hard_negative_loss import (
    multi_negative_bpr,
    sampled_l2_multineg,
)
from clv_run_state import ProgressStore, RunIdentity, clone_state, file_sha256
import lightgcn_clv_axis_specific_test10 as test10
import lightgcn_clv_gatefree_lowdim as gatefree
import lightgcn_clv_gradient_isolated_economic_interaction as report_helpers
import lightgcn_clv_joint_nv as joint
import lightgcn_clv_moe as moe
import lightgcn_clv_residual as residual
import lightgcn_clv_v3 as v3


CODE_VERSION = "m4-clv-conditioned-multinegative-bpr-historical-screen-v1"
K1_MODEL_ID = "m1_64"
MEAN_K5_MODEL_ID = "m1_multineg_mean_k5"
M4_MODEL_ID = "m4_clv_hard_k5"
ACCURACY_METRICS = (
    "recall@10",
    "ndcg@10",
    "recall@20",
    "ndcg@20",
    "recall@50",
    "ndcg@50",
)


@dataclass(frozen=True)
class M4HardNegativeConfig:
    dataset: str = "dunnhumby"
    seed: int = 42
    time_cutoff: int = 690
    evaluation_days: int = 7
    epochs: int = 100
    id_dim: int = 64
    n_layers: int = 2
    negative_count: int = 5
    batch_size: int = 8192
    lr: float = 5e-4
    pref_reg: float = 1e-3
    input_days: int = 365
    out_dir: str = ""
    baseline_result_dir: str = ""


def configure_m4_clv_hard_negative_run(**overrides) -> M4HardNegativeConfig:
    defaults = {
        "out_dir": (
            f"{v3.default_out_dir('dunnhumby')}"
            "_m4_clv_hard_negative_historical_screen_v1"
        ),
        "baseline_result_dir": (
            f"{v3.default_out_dir('dunnhumby')}"
            "_m2_repeatshare_historical_backtest_v1"
        ),
    }
    return validate_m4_config(M4HardNegativeConfig(**(defaults | overrides)))


def validate_m4_config(cfg: M4HardNegativeConfig) -> M4HardNegativeConfig:
    required = {
        "dataset": "dunnhumby",
        "seed": 42,
        "time_cutoff": 690,
        "evaluation_days": 7,
        "epochs": 100,
        "id_dim": 64,
        "n_layers": 2,
        "negative_count": 5,
        "input_days": 365,
    }
    for key, expected in required.items():
        if getattr(cfg, key) != expected:
            raise ValueError(f"빠른 M4 screen은 {key}={expected!r}이어야 합니다")
    if cfg.batch_size <= 0 or cfg.lr <= 0 or cfg.pref_reg < 0:
        raise ValueError("빠른 M4 screen 학습 설정이 잘못됐습니다")
    if not cfg.out_dir or not cfg.baseline_result_dir:
        raise ValueError("빠른 M4 screen은 out_dir와 baseline_result_dir가 필요합니다")
    return cfg


def preflight_summary(cfg: M4HardNegativeConfig) -> dict:
    cfg = validate_m4_config(cfg)
    return {
        "code_version": CODE_VERSION,
        "dataset": cfg.dataset,
        "seed": cfg.seed,
        "trained_models": [MEAN_K5_MODEL_ID, M4_MODEL_ID],
        "reused_comparator": K1_MODEL_ID,
        "research_axis": "M4 loss intervention",
        "historical_development_split": {
            "train_end_inclusive": 683,
            "evaluation_start_inclusive": 684,
            "evaluation_end_inclusive": 690,
            "final_test_constructed": False,
            "holdout_constructed": False,
        },
        "m4": {
            "uniform_negative_count": cfg.negative_count,
            "control": "mean BPR over the same five uniform negatives",
            "intervention": (
                "(1-q_C)*mean_BPR + q_C*BPR(highest-scored negative)"
            ),
            "per_positive_loss_mass": 1.0,
            "historical_clv": "train-only percentile of N_hat*V_hat proxy",
            "n_v_in_loss": False,
            "n_v_segment_diagnostics": True,
        },
        "fixed": {
            "task": "new-item recommendation",
            "graph": "binary",
            "lightgcn": "ID64, 2 layers, layer 0/1/2 mean",
            "negative_sampling": "uniform",
            "m2_representation": False,
            "m3_edge_weight": False,
            "positive_pair_weighting": False,
            "new_auxiliary_loss": False,
            "min_user_interactions": 1,
            "min_item_interactions": 1,
            "epochs": cfg.epochs,
            "validation_or_epoch_selection": False,
        },
        "out_dir": cfg.out_dir,
        "baseline_result_dir": cfg.baseline_result_dir,
    }


def sample_uniform_negative_matrix(
    users: np.ndarray,
    positives: np.ndarray,
    n_items: int,
    positive_keys: np.ndarray,
    rng: np.random.Generator,
    *,
    k: int,
) -> np.ndarray:
    """Sample K independent uniform unseen items for every training row."""

    if k <= 0:
        raise ValueError("K는 1 이상이어야 합니다")
    columns = [
        v3.sample_negatives(
            users,
            positives,
            n_items,
            positive_keys,
            rng,
            "uniform",
        )
        for _ in range(k)
    ]
    result = np.stack(columns, axis=1).astype(np.int64, copy=False)
    sampled_keys = users.astype(np.int64)[:, None] * int(n_items) + result
    if np.isin(sampled_keys, positive_keys).any():
        raise RuntimeError("uniform 음성에 학습 관측상품이 남았습니다")
    return result


def screening_reading(
    baseline: dict,
    mean_k5: dict,
    m4: dict,
) -> dict:
    accuracy_ratios = {
        metric: float(m4[metric] / baseline[metric])
        for metric in ACCURACY_METRICS
    }
    accuracy_pass = all(value >= 0.99 for value in accuracy_ratios.values())
    high_deltas = {
        metric: float(m4[metric] - mean_k5[metric])
        for metric in ("고CLV_recall@10", "고CLV_ndcg@10")
    }
    economic_deltas = {
        "vs_m1_k1": float(
            m4["price_purchase_amount_weighted_hit@10"]
            - baseline["price_purchase_amount_weighted_hit@10"]
        ),
        "vs_mean_k5": float(
            m4["price_purchase_amount_weighted_hit@10"]
            - mean_k5["price_purchase_amount_weighted_hit@10"]
        ),
    }
    coverage_ratio = float(m4["coverage@10"] / baseline["coverage@10"])
    distinct_ratio = float(m4["n_distinct@10"] / baseline["n_distinct@10"])
    top10_ratio = float(m4["top10_share@10"] / baseline["top10_share@10"])
    high_pass = all(delta > 0.0 for delta in high_deltas.values())
    economic_pass = all(delta > 0.0 for delta in economic_deltas.values())
    exposure_pass = (
        coverage_ratio >= 0.95
        and distinct_ratio >= 0.95
        and top10_ratio <= 1.05
    )
    return {
        "positive_screen": bool(
            accuracy_pass and high_pass and economic_pass and exposure_pass
        ),
        "accuracy_pass": bool(accuracy_pass),
        "high_clv_pass": bool(high_pass),
        "economic_pass": bool(economic_pass),
        "exposure_pass": bool(exposure_pass),
        "accuracy_ratios_vs_m1_k1": accuracy_ratios,
        "high_clv_deltas_vs_mean_k5": high_deltas,
        "weighted_hit@10_deltas": economic_deltas,
        "coverage@10_ratio_vs_m1_k1": coverage_ratio,
        "n_distinct@10_ratio_vs_m1_k1": distinct_ratio,
        "top10_share@10_ratio_vs_m1_k1": top10_ratio,
        "next_if_positive": (
            "run degree-matched CLV shuffle, then several development seeds and H&M"
        ),
        "statistical_note": "seed 42 exploratory screen; no significance claim",
    }


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def _config_hash(cfg: M4HardNegativeConfig, input_hash: str, revision: str) -> str:
    payload = {
        "code_version": CODE_VERSION,
        "config": asdict(cfg),
        "input_hash": input_hash,
        "source_revision": revision,
    }
    return hashlib.sha256(_canonical(payload).encode()).hexdigest()[:12]


def _prepare(cfg: M4HardNegativeConfig) -> dict:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = moe.build_input_manifest(v3.SCHEMA[cfg.dataset])
    input_hash = moe.manifest_hash(manifest)
    revision = moe.source_revision()
    base_cfg = gatefree._base_config(cfg)
    data = v3.prepare_data(base_cfg, v3.DCFG)
    if set(data["splits"]) != {"test"}:
        raise RuntimeError(f"historical 개발분할 외 오염: {sorted(data['splits'])}")
    if float(data["train"].t.max()) != 683.0:
        raise RuntimeError(f"historical train 종료일 오류: {data['train'].t.max()}")
    if data.get("loss_w") is not None:
        raise RuntimeError("M4 hard-negative screen에 기존 행 가중치가 섞였습니다")
    data["loss_w"] = None
    snapshot = residual.build_final_snapshot(
        data["train"], data["n_users"], v3.DCFG["is_date"], cfg.input_days
    )
    axes = joint.build_user_axis_inputs(snapshot, data["n_users"])
    _, _, q_c, clv_valid = report_helpers.build_clv_inputs(axes)
    q_c = np.where(clv_valid, q_c, 0.0).astype(np.float32)
    baseline = gatefree._load_compatible_baseline(cfg, manifest)
    x_item, item_cat = v3.item_value_features(
        data["train"], data["n_items"], report=False
    )
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
        "q_c": q_c,
        "clv_valid": clv_valid,
        "baseline": baseline,
        "x_item": x_item,
        "item_cat": item_cat,
        "meta": meta,
        "cache": cache,
    }
    prepared["config_hash"] = _config_hash(cfg, input_hash, revision)
    return prepared


def _build_model(prepared: dict, cfg: M4HardNegativeConfig):
    v3.set_seed(cfg.seed)
    model_cfg = {**prepared["base_cfg"], "ARCH": "pref_only", "DIM": cfg.id_dim}
    model = v3.build_model(
        prepared["data"],
        prepared["data"]["x_val_u"],
        prepared["x_item"],
        prepared["item_cat"],
        model_cfg,
    )
    return model, list(model.pref_params())


def _arm_hash(prepared: dict, cfg: M4HardNegativeConfig, model_id: str) -> str:
    return hashlib.sha256(
        _canonical(
            {
                "run": prepared["config_hash"],
                "model_id": model_id,
                "seed": cfg.seed,
                "negative_count": cfg.negative_count,
            }
        ).encode()
    ).hexdigest()[:12]


def _arm_paths(prepared: dict, cfg: M4HardNegativeConfig, model_id: str) -> dict:
    root = prepared["out_dir"] / "arms" / prepared["config_hash"]
    stem = f"{model_id}_s{cfg.seed}"
    return {"result": root / f"{stem}.json", "checkpoint": root / f"{stem}.pt"}


def _train_arm(
    model,
    params,
    prepared: dict,
    cfg: M4HardNegativeConfig,
    model_id: str,
    store: ProgressStore,
) -> dict:
    data = prepared["data"]
    optimizer = torch.optim.Adam(params, lr=cfg.lr, weight_decay=0.0)
    rng = np.random.default_rng(cfg.seed)
    restored = store.restore_epoch(model, optimizer, rng)
    start_epoch = 1
    history = []
    updates = samples = 0
    previous_wall = 0.0
    if restored is not None:
        start_epoch = int(restored["next_epoch"])
        history = list(restored.get("history", []))
        updates = int(restored.get("updates", 0))
        samples = int(restored.get("samples", 0))
        previous_wall = float(restored.get("wall_clock_sec", 0.0))
        print(f"  [{model_id}] epoch {start_epoch - 1}에서 자동 재개")
    store.mark_stage("running", epoch=start_epoch - 1, max_epoch=cfg.epochs)
    tr_u, tr_i, positive_keys = data["tr_u"], data["tr_i"], data["pos_key"]
    n_train = len(tr_u)
    n_batches = math.ceil(n_train / cfg.batch_size)
    q_all = torch.as_tensor(prepared["q_c"], device=v3.DEVICE)
    started = time.time()
    last_epoch = start_epoch - 1
    for epoch in range(start_epoch, cfg.epochs + 1):
        last_epoch = epoch
        model.train()
        epoch_started = time.time()
        permutation = rng.permutation(n_train)
        loss_sum = bpr_sum = correct_sum = 0.0
        hardest_weight_sum = gap_sum = weight_error = 0.0
        for batch in range(n_batches):
            index = permutation[
                batch * cfg.batch_size : (batch + 1) * cfg.batch_size
            ]
            users_np, positives_np = tr_u[index], tr_i[index]
            negatives_np = sample_uniform_negative_matrix(
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
            user_z, item_z, _, _ = model.embeddings(need_value=False)
            positive_scores = (user_z[users] * item_z[positives]).sum(1)
            negative_scores = (
                user_z[users, None, :] * item_z[negatives]
            ).sum(2)
            q_batch = (
                torch.zeros_like(q_all[users])
                if model_id == MEAN_K5_MODEL_ID
                else q_all[users]
            )
            bpr, diagnostics = multi_negative_bpr(
                positive_scores, negative_scores, q_batch
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
            optimizer.step()
            loss_sum += float(loss.detach())
            bpr_sum += float(bpr.detach())
            correct_sum += float(diagnostics["p_correct"])
            hardest_weight_sum += float(diagnostics["hardest_weight_mean"])
            gap_sum += float(diagnostics["positive_hardest_gap"])
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
                loss=loss_sum / (batch + 1),
            )
        epoch_sec = time.time() - epoch_started
        record = {
            "epoch": int(epoch),
            "loss": float(loss_sum / n_batches),
            "bpr": float(bpr_sum / n_batches),
            "p_correct": float(correct_sum / n_batches),
            "hardest_negative_weight_mean": float(
                hardest_weight_sum / n_batches
            ),
            "positive_hardest_gap": float(gap_sum / n_batches),
            "row_weight_sum_max_error": float(weight_error),
            "epoch_sec": float(epoch_sec),
        }
        history.append(record)
        print(
            f"  [{model_id}] ep {epoch:3d}/{cfg.epochs} | "
            f"loss {record['loss']:.4f} | P(pos>neg) {record['p_correct']:.3f} | "
            f"hard-w {record['hardest_negative_weight_mean']:.3f} | {epoch_sec:.0f}s"
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
        "phase": model_id,
        "epochs_run": int(last_epoch),
        "updates": int(updates),
        "samples": int(samples),
        "negative_count": cfg.negative_count,
        "wall_clock_sec": round(previous_wall + time.time() - started, 1),
        "history": history,
        "final_diagnostics": history[-1] if history else {},
    }


def _run_arm(
    prepared: dict,
    cfg: M4HardNegativeConfig,
    model_id: str,
) -> dict:
    paths = _arm_paths(prepared, cfg, model_id)
    model, params = _build_model(prepared, cfg)
    if paths["result"].exists() and paths["checkpoint"].exists():
        print(f"  [cached] {model_id} 완료 결과 재사용")
        return json.loads(paths["result"].read_text(encoding="utf-8"))
    store = ProgressStore(
        prepared["out_dir"] / "progress" / prepared["config_hash"],
        RunIdentity(
            stage="historical_development_train",
            model_id=model_id,
            seed=cfg.seed,
            config_hash=_arm_hash(prepared, cfg, model_id),
            source_revision=prepared["revision"],
            input_hash=prepared["input_hash"],
        ),
    )
    training = _train_arm(model, params, prepared, cfg, model_id, store)
    model.eval()
    paths["checkpoint"].parent.mkdir(parents=True, exist_ok=True)
    temporary = paths["checkpoint"].with_suffix(".pt.tmp")
    torch.save(
        {
            "state": clone_state(model),
            "model_id": model_id,
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
        "model_id": model_id,
        "role": "multineg_control" if model_id == MEAN_K5_MODEL_ID else "model",
        "seed": cfg.seed,
        "split": "historical_development_days_684_690",
        "final_epoch": cfg.epochs,
        "negative_count": cfg.negative_count,
        "clv_conditioned": model_id == M4_MODEL_ID,
        "metrics": test10._public_metrics(metrics_raw),
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


def run_m4_clv_hard_negative_screen(
    cfg: M4HardNegativeConfig | None = None,
) -> pd.DataFrame:
    cfg = validate_m4_config(cfg or configure_m4_clv_hard_negative_run())
    summary = preflight_summary(cfg)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    prepared = _prepare(cfg)
    print("\n===== K=5 uniform mean control | seed 42 | fixed 100 epochs =====")
    mean_k5 = _run_arm(prepared, cfg, MEAN_K5_MODEL_ID)
    print("\n===== CLV-conditioned hard negative K=5 | seed 42 =====")
    m4 = _run_arm(prepared, cfg, M4_MODEL_ID)

    baseline = dict(prepared["baseline"])
    baseline["model_id"] = K1_MODEL_ID
    baseline["role"] = "reused_baseline_display_only"
    rows = [baseline]
    for arm in (mean_k5, m4):
        rows.append(
            {
                "model_id": arm["model_id"],
                "role": arm["role"],
                "seed": arm["seed"],
                "split": arm["split"],
                "final_epoch": arm["final_epoch"],
                "negative_count": arm["negative_count"],
                **arm["training"].get("final_diagnostics", {}),
                **arm["metrics"],
            }
        )
    frame = pd.DataFrame(rows)
    reading = screening_reading(baseline, mean_k5["metrics"], m4["metrics"])
    metric_rows = {
        K1_MODEL_ID: {
            key: value
            for key, value in baseline.items()
            if "@" in key and isinstance(value, (int, float, np.number))
        },
        MEAN_K5_MODEL_ID: mean_k5["metrics"],
        M4_MODEL_ID: m4["metrics"],
    }
    comparison = report_helpers._metric_comparison(
        metric_rows, references=(K1_MODEL_ID, MEAN_K5_MODEL_ID)
    )
    out = prepared["out_dir"]
    fingerprint = prepared["config_hash"]
    paths = {
        "absolute_csv": out / f"m4_clv_hard_negative_{fingerprint}.csv",
        "comparison_csv": out / f"m4_clv_hard_negative_{fingerprint}_comparison.csv",
        "json": out / f"m4_clv_hard_negative_{fingerprint}.json",
    }
    frame.to_csv(paths["absolute_csv"], index=False)
    comparison.to_csv(paths["comparison_csv"], index=False)
    test10._atomic_json(
        paths["json"],
        {
            "code_version": CODE_VERSION,
            "config": asdict(cfg),
            "preflight": summary,
            "input_manifest": prepared["manifest"],
            "absolute_rows": frame.to_dict("records"),
            "comparison_rows": comparison.to_dict("records"),
            "screening_reading": reading,
            "arms": {MEAN_K5_MODEL_ID: mean_k5, M4_MODEL_ID: m4},
        },
    )
    frame.attrs["comparison"] = comparison.to_dict("records")
    frame.attrs["screening_reading"] = reading
    frame.attrs["training_diagnostics"] = {
        MEAN_K5_MODEL_ID: mean_k5["training"].get("final_diagnostics", {}),
        M4_MODEL_ID: m4["training"].get("final_diagnostics", {}),
    }
    frame.attrs["result_paths"] = {
        key: str(value) for key, value in paths.items()
    }
    return frame
