"""Historical screen for recent-level and CLV-change conditioned LightGCN.

Eight chronological anchors share one model.  At each anchor, only the prior
90 days are used to calculate a historical ``N x V`` proxy, and the change
from the equally long window ending 28 days earlier is supplied separately.
Both fixed conditions transform the ordinary 64-dimensional user-ID embedding
inside the same LightGCN/BPR forward graph.
"""

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

from clv_dynamic_clv_level_change_model import DynamicCLVLevelChangeLightGCN
from clv_run_state import ProgressStore, RunIdentity, clone_state, file_sha256
import lightgcn_clv_axis_specific_test10 as test10
import lightgcn_clv_moe as moe
import lightgcn_clv_v3 as v3


CODE_VERSION = "m2-dynamic-clv-level-change-multianchor-historical-screen-v1"
MODELS = ("m1_wide_multianchor_rho0", "m2_dynamic_clv_level_change")


@dataclass(frozen=True)
class DynamicLevelChangeConfig:
    dataset: str = "dunnhumby"
    seed: int = 42
    time_cutoff: int = 690
    evaluation_days: int = 7
    anchor_count: int = 8
    anchor_spacing_days: int = 28
    target_horizon_days: int = 7
    rolling_window_days: int = 90
    change_lag_days: int = 28
    epochs: int = 100
    embedding_dim: int = 64
    n_layers: int = 2
    rho: float = 0.05
    batch_size: int = 8192
    lr: float = 5e-4
    pref_reg: float = 1e-3
    out_dir: str = ""


def configure_dynamic_level_change(**overrides) -> DynamicLevelChangeConfig:
    defaults = {
        "out_dir": (
            f"{v3.default_out_dir('dunnhumby')}"
            "_m2_dynamic_clv_level_change_multianchor_historical_screen_v1"
        )
    }
    return validate_dynamic_level_change(
        DynamicLevelChangeConfig(**(defaults | overrides))
    )


def validate_dynamic_level_change(
    cfg: DynamicLevelChangeConfig,
) -> DynamicLevelChangeConfig:
    required = {
        "dataset": "dunnhumby",
        "seed": 42,
        "time_cutoff": 690,
        "evaluation_days": 7,
        "anchor_count": 8,
        "anchor_spacing_days": 28,
        "target_horizon_days": 7,
        "rolling_window_days": 90,
        "change_lag_days": 28,
        "epochs": 100,
        "embedding_dim": 64,
        "n_layers": 2,
        "rho": 0.05,
    }
    for key, expected in required.items():
        if getattr(cfg, key) != expected:
            raise ValueError(f"동적 CLV 개발실험은 {key}={expected!r}이어야 합니다")
    if cfg.batch_size <= 0 or cfg.lr <= 0 or cfg.pref_reg < 0:
        raise ValueError("학습 설정이 잘못됐습니다")
    if not cfg.out_dir:
        raise ValueError("out_dir가 필요합니다")
    return cfg


def _anchor_history_ends(cfg: DynamicLevelChangeConfig) -> list[int]:
    train_end = cfg.time_cutoff - cfg.evaluation_days
    final_history_end = train_end - cfg.target_horizon_days
    return [
        final_history_end - cfg.anchor_spacing_days * offset
        for offset in reversed(range(cfg.anchor_count))
    ]


def preflight_summary(cfg: DynamicLevelChangeConfig) -> dict:
    cfg = validate_dynamic_level_change(cfg)
    train_end = cfg.time_cutoff - cfg.evaluation_days
    return {
        "code_version": CODE_VERSION,
        "dataset": cfg.dataset,
        "seed": cfg.seed,
        "models": list(MODELS),
        "research_axis": "M2 embedding/representation intervention",
        "historical_development_split": {
            "anchor_history_ends": _anchor_history_ends(cfg),
            "anchor_target_horizon_days": cfg.target_horizon_days,
            "evaluation_start": train_end + 1,
            "evaluation_end": cfg.time_cutoff,
            "final_test_constructed": False,
            "holdout_constructed": False,
        },
        "historical_clv_proxy": {
            "definition": "N(window) * V(window) = recent-window purchase amount",
            "rolling_window_days": cfg.rolling_window_days,
            "change_lag_days": cfg.change_lag_days,
            "level": "robustly scaled log1p(current-window proxy)",
            "change": (
                "robustly scaled log1p(current proxy)-log1p(previous proxy)"
            ),
            "limitation": "historical CLV proxy, not future or lifetime profit",
        },
        "m2": {
            "formula": (
                "NormPreserve(E_u_ID * (1 + rho*tanh(L_u,t*w_L + D_u,t*w_D)))"
            ),
            "rho": cfg.rho,
            "item_side_CLV_input": False,
            "separate_CLV_score": False,
            "shared_parameters_across_anchors": True,
        },
        "matched_control": (
            "same anchors, labels, graph schedule, parameters and optimizer; rho=0"
        ),
        "fixed": {
            "graph": "binary",
            "negative_sampling": "uniform",
            "sample_weighting": False,
            "loss": "one plain pairwise BPR plus existing sampled L2",
            "new_loss_term": False,
            "one_training_loop_and_optimizer": True,
            "min_user_interactions": 1,
            "min_item_interactions": 1,
            "epochs": cfg.epochs,
            "validation_or_epoch_selection": False,
        },
        "out_dir": cfg.out_dir,
    }


def _basket_table(history: pd.DataFrame) -> pd.DataFrame:
    keys = ["u_idx", "b_raw"] if "b_raw" in history.columns else ["u_idx", "t"]
    return (
        history.groupby(keys, sort=False)
        .agg(basket_value=("v", "sum"), basket_time=("t", "max"))
        .reset_index()
    )


def _window_proxy(
    history: pd.DataFrame,
    *,
    n_users: int,
    window_end: int,
    window_days: int,
) -> np.ndarray:
    window_start = window_end - window_days + 1
    window = history[
        (history["t"] >= window_start) & (history["t"] <= window_end)
    ]
    proxy = np.zeros(n_users, np.float64)
    if window.empty:
        return proxy
    baskets = _basket_table(window)
    summary = baskets.groupby("u_idx", sort=False).agg(
        n=("basket_value", "size"),
        v=("basket_value", "mean"),
    )
    ids = summary.index.to_numpy(np.int64)
    proxy[ids] = (summary["n"] * summary["v"]).to_numpy(np.float64)
    return proxy


def _bounded_robust_z(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    result = np.zeros(len(values), np.float32)
    observed = np.asarray(values, np.float64)[valid]
    if not len(observed):
        return result
    centre = float(np.median(observed))
    mad = float(np.median(np.abs(observed - centre)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < 1e-12:
        scale = float(np.std(observed))
    if not np.isfinite(scale) or scale < 1e-12:
        return result
    result[valid] = np.clip((observed - centre) / scale, -3.0, 3.0) / 3.0
    return result


def rolling_clv_level_change(
    history: pd.DataFrame,
    *,
    n_users: int,
    history_end: int,
    window_days: int,
    change_lag_days: int,
) -> dict:
    """Build leakage-free recent CLV level and within-user change conditions."""
    past = history[history["t"] <= history_end]
    valid = np.zeros(n_users, bool)
    if not past.empty:
        valid[past["u_idx"].unique().astype(np.int64)] = True
    current = _window_proxy(
        past,
        n_users=n_users,
        window_end=history_end,
        window_days=window_days,
    )
    previous = _window_proxy(
        past,
        n_users=n_users,
        window_end=history_end - change_lag_days,
        window_days=window_days,
    )
    log_current = np.log1p(current)
    log_previous = np.log1p(previous)
    return {
        "level_condition": _bounded_robust_z(log_current, valid),
        "change_condition": _bounded_robust_z(
            log_current - log_previous, valid
        ),
        "current_clv_proxy": current,
        "previous_clv_proxy": previous,
        "valid": valid,
    }


def _unique_pair_keys(frame: pd.DataFrame, n_items: int) -> np.ndarray:
    if frame.empty:
        return np.empty(0, np.int64)
    return np.unique(
        frame["u_idx"].to_numpy(np.int64) * n_items
        + frame["i_idx"].to_numpy(np.int64)
    )


def _build_context(
    all_train: pd.DataFrame,
    *,
    history_end: int,
    target_end: int,
    n_users: int,
    n_items: int,
    cfg: DynamicLevelChangeConfig,
) -> dict:
    history = all_train[all_train["t"] <= history_end]
    target = all_train[
        (all_train["t"] > history_end) & (all_train["t"] <= target_end)
    ].copy()
    if history.empty or target.empty:
        raise RuntimeError("anchor history or target is empty")

    history_users = np.zeros(n_users, bool)
    history_items = np.zeros(n_items, bool)
    history_users[history["u_idx"].unique()] = True
    history_items[history["i_idx"].unique()] = True
    target = target[
        history_users[target["u_idx"].to_numpy(np.int64)]
        & history_items[target["i_idx"].to_numpy(np.int64)]
    ]
    history_key = _unique_pair_keys(history, n_items)
    target_key = _unique_pair_keys(target, n_items)
    positions = np.searchsorted(history_key, target_key)
    already_seen = np.zeros(len(target_key), bool)
    in_range = positions < len(history_key)
    already_seen[in_range] = history_key[positions[in_range]] == target_key[in_range]
    target_key = target_key[~already_seen]
    if not len(target_key):
        raise RuntimeError(f"anchor {history_end} has no new-item targets")

    graph_users = (history_key // n_items).astype(np.int64)
    graph_items = (history_key % n_items).astype(np.int64)
    adjacency = v3.build_adj(
        graph_users,
        graph_items,
        np.ones(len(history_key), np.float32),
        n_users,
        n_items,
    )
    conditions = rolling_clv_level_change(
        history,
        n_users=n_users,
        history_end=history_end,
        window_days=cfg.rolling_window_days,
        change_lag_days=cfg.change_lag_days,
    )
    return {
        "name": f"history_le_{history_end}_target_{history_end + 1}_{target_end}",
        "history_end": history_end,
        "target_end": target_end,
        "adj": adjacency,
        **conditions,
        "tr_u": (target_key // n_items).astype(np.int64),
        "tr_i": (target_key % n_items).astype(np.int64),
        "forbidden_key": np.unique(np.concatenate([history_key, target_key])),
        "stats": {
            "history_end": history_end,
            "target_end": target_end,
            "history_edges": int(len(history_key)),
            "new_item_target_pairs": int(len(target_key)),
            "target_users": int(len(np.unique(target_key // n_items))),
            "valid_users": int(conditions["valid"].sum()),
            "recent_proxy_positive_share": float(
                (conditions["current_clv_proxy"][conditions["valid"]] > 0).mean()
            ),
            "level_std": float(
                conditions["level_condition"][conditions["valid"]].std()
            ),
            "change_std": float(
                conditions["change_condition"][conditions["valid"]].std()
            ),
        },
    }


def _base_config(cfg: DynamicLevelChangeConfig) -> dict:
    configured = v3.configure_run(
        cfg.dataset,
        out_dir=cfg.out_dir,
        ARCH="pref_only",
        SEED_LIST=[cfg.seed],
        WINDOW_DAYS=None,
        TIME_CUTOFF=cfg.time_cutoff,
        TRAIN_ON_VAL=True,
        VAL_DAYS=7,
        TEST_DAYS=cfg.evaluation_days,
        HOLDOUT_DAYS=0,
        EVAL_TEST=True,
        EVAL_HOLDOUT=False,
        GRAPH_MODE="binary",
        LOSS_MODE="plain",
        NEG_MODE="uniform",
        MIN_USER_INTER=1,
        MIN_ITEM_INTER=1,
        DIM=cfg.embedding_dim,
        N_LAYERS=cfg.n_layers,
        BATCH_SIZE=cfg.batch_size,
        LR=cfg.lr,
        PREF_REG=cfg.pref_reg,
        EPOCHS=cfg.epochs,
        EARLY_STOP=cfg.epochs,
        REPORT_LEGACY_VALUE_FEATURES=False,
    )
    base = dict(configured)
    required = {
        "TRAIN_ON_VAL": True,
        "EVAL_TEST": True,
        "EVAL_HOLDOUT": False,
        "HOLDOUT_DAYS": 0,
        "GRAPH_MODE": "binary",
        "LOSS_MODE": "plain",
        "NEG_MODE": "uniform",
        "MIN_USER_INTER": 1,
        "MIN_ITEM_INTER": 1,
        "EPOCHS": 100,
    }
    for key, expected in required.items():
        if base[key] != expected:
            raise RuntimeError(f"동적 CLV 기준설정 오염: {key}={base[key]!r}")
    return base


def _input_condition_summary(contexts: list[dict]) -> dict:
    levels = np.stack([context["level_condition"] for context in contexts])
    changes = np.stack([context["change_condition"] for context in contexts])

    def adjacent_correlation(values):
        correlations = []
        for left, right in zip(values[:-1], values[1:]):
            valid = np.isfinite(left) & np.isfinite(right)
            if valid.sum() > 1 and left[valid].std() > 0 and right[valid].std() > 0:
                correlations.append(float(np.corrcoef(left[valid], right[valid])[0, 1]))
        return float(np.mean(correlations)) if correlations else np.nan

    return {
        "level_across_anchor_std_mean": float(levels.std(axis=0).mean()),
        "change_across_anchor_std_mean": float(changes.std(axis=0).mean()),
        "level_changing_user_share": float((levels.std(axis=0) > 1e-8).mean()),
        "change_changing_user_share": float((changes.std(axis=0) > 1e-8).mean()),
        "level_adjacent_correlation_mean": adjacent_correlation(levels),
        "change_adjacent_correlation_mean": adjacent_correlation(changes),
    }


def _prepare(cfg: DynamicLevelChangeConfig) -> dict:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = moe.build_input_manifest(v3.SCHEMA[cfg.dataset])
    input_hash = moe.manifest_hash(manifest)
    revision = moe.source_revision()
    base_cfg = _base_config(cfg)
    data = v3.prepare_data(base_cfg, v3.DCFG)
    if set(data["splits"]) != {"test"} or data.get("loss_w") is not None:
        raise RuntimeError("historical development split or M2 loss boundary is contaminated")
    data["loss_w"] = None
    train_end = cfg.time_cutoff - cfg.evaluation_days
    if int(data["train"]["t"].max()) != train_end:
        raise RuntimeError("unexpected historical train boundary")

    contexts = [
        _build_context(
            data["train"],
            history_end=history_end,
            target_end=history_end + cfg.target_horizon_days,
            n_users=data["n_users"],
            n_items=data["n_items"],
            cfg=cfg,
        )
        for history_end in _anchor_history_ends(cfg)
    ]
    final_conditions = rolling_clv_level_change(
        data["train"],
        n_users=data["n_users"],
        history_end=train_end,
        window_days=cfg.rolling_window_days,
        change_lag_days=cfg.change_lag_days,
    )
    final_context = {
        "name": f"evaluation_history_le_{train_end}",
        "adj": data["adj"],
        **final_conditions,
    }
    input_diagnostics = _input_condition_summary(contexts)
    print("  최근 수준·변화 다중시점 입력 진단:", input_diagnostics)
    for context in contexts:
        print("   ", context["name"], context["stats"])

    meta = v3.item_meta(data["train"], data["n_items"])
    final_proxy = final_conditions["current_clv_proxy"].copy()
    final_proxy[~final_conditions["valid"]] = np.nan
    thresholds = v3.segment_thresholds(final_proxy, base_cfg["SEG_EDGES"])
    cache = v3.EvalCache(
        *data["splits"]["test"], final_proxy, thresholds, data["n_items"]
    )
    config_payload = {
        "code_version": CODE_VERSION,
        "config": asdict(cfg),
        "models": MODELS,
        "input_hash": input_hash,
        "source_revision": revision,
    }
    config_hash = hashlib.sha256(
        json.dumps(config_payload, sort_keys=True, default=str).encode()
    ).hexdigest()[:12]
    return {
        "out_dir": out_dir,
        "manifest": manifest,
        "input_hash": input_hash,
        "revision": revision,
        "config_hash": config_hash,
        "base_cfg": base_cfg,
        "data": data,
        "contexts": contexts,
        "final_context": final_context,
        "input_diagnostics": input_diagnostics,
        "meta": meta,
        "cache": cache,
    }


def _build_model(prepared, cfg, rho):
    all_contexts = prepared["contexts"] + [prepared["final_context"]]
    v3.set_seed(cfg.seed)
    return DynamicCLVLevelChangeLightGCN(
        n_users=prepared["data"]["n_users"],
        n_items=prepared["data"]["n_items"],
        adjacencies=[context["adj"] for context in all_contexts],
        level_conditions=[context["level_condition"] for context in all_contexts],
        change_conditions=[context["change_condition"] for context in all_contexts],
        context_names=[context["name"] for context in all_contexts],
        embedding_dim=cfg.embedding_dim,
        rho=rho,
        n_layers=cfg.n_layers,
        pref_reg=cfg.pref_reg,
    ).to(v3.DEVICE)


def _arm_hash(prepared: dict, model_id: str) -> str:
    return hashlib.sha256(
        json.dumps(
            {"run": prepared["config_hash"], "model_id": model_id, "seed": 42},
            sort_keys=True,
        ).encode()
    ).hexdigest()[:12]


def _train(model, prepared, cfg, model_id, store):
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    rng = np.random.default_rng(cfg.seed)
    restored = store.restore_epoch(model, optimizer, rng)
    start_epoch, history, updates, samples, previous_wall = 1, [], 0, 0, 0.0
    if restored is not None:
        start_epoch = int(restored["next_epoch"])
        history = list(restored.get("history", []))
        updates = int(restored.get("updates", 0))
        samples = int(restored.get("samples", 0))
        previous_wall = float(restored.get("wall_clock_sec", 0.0))
        print(f"  [{model_id}] epoch {start_epoch - 1}에서 자동 재개")
    store.mark_stage("running", epoch=start_epoch - 1, max_epoch=cfg.epochs)
    started = time.time()
    for epoch in range(start_epoch, cfg.epochs + 1):
        model.train()
        epoch_started = time.time()
        totals = {"loss": 0.0, "bpr": 0.0, "p_correct": 0.0, "batches": 0}
        for context_index in rng.permutation(len(prepared["contexts"])):
            context = prepared["contexts"][int(context_index)]
            model.set_context(int(context_index))
            permutation = rng.permutation(len(context["tr_u"]))
            n_batches = math.ceil(len(permutation) / cfg.batch_size)
            for batch in range(n_batches):
                index = permutation[
                    batch * cfg.batch_size : (batch + 1) * cfg.batch_size
                ]
                users, positives = context["tr_u"][index], context["tr_i"][index]
                negatives = v3.sample_negatives(
                    users,
                    positives,
                    prepared["data"]["n_items"],
                    context["forbidden_key"],
                    rng,
                    "uniform",
                )
                loss, diagnostics = model.bpr_loss(
                    torch.as_tensor(users, dtype=torch.long, device=v3.DEVICE),
                    torch.as_tensor(positives, dtype=torch.long, device=v3.DEVICE),
                    torch.as_tensor(negatives, dtype=torch.long, device=v3.DEVICE),
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                totals["loss"] += float(loss)
                totals["bpr"] += diagnostics["bpr"]
                totals["p_correct"] += diagnostics["p_correct"]
                totals["batches"] += 1
                updates += 1
                samples += len(index)
                store.heartbeat(
                    epoch=epoch,
                    max_epoch=cfg.epochs,
                    context=context["name"],
                    batch=batch + 1,
                    batches=n_batches,
                    loss=totals["loss"] / totals["batches"],
                )
        record = {
            "epoch": epoch,
            "loss": totals["loss"] / totals["batches"],
            "bpr": totals["bpr"] / totals["batches"],
            "p_correct": totals["p_correct"] / totals["batches"],
            "epoch_sec": time.time() - epoch_started,
            **model.condition_diagnostics(),
        }
        history.append(record)
        store.save_epoch(
            model,
            optimizer,
            rng,
            epoch=epoch,
            best_epoch=0,
            best_metric=float("nan"),
            history=history,
            updates=updates,
            samples=samples,
            wall_clock_sec=previous_wall + time.time() - started,
            selection="none",
        )
        print(
            f"  [{model_id}] ep {epoch:3d}/{cfg.epochs} | "
            f"loss {record['loss']:.4f} | P(pos>neg) {record['p_correct']:.3f} | "
            f"|wL| {record['level_dimension_weight_abs_mean']:.4f} | "
            f"|wD| {record['change_dimension_weight_abs_mean']:.4f} | "
            f"{record['epoch_sec']:.1f}s"
        )
    return {
        "epochs_run": cfg.epochs,
        "selection": "none",
        "early_stopping": False,
        "updates": updates,
        "samples": samples,
        "wall_clock_sec": previous_wall + time.time() - started,
        "history": history,
    }


def _run_arm(prepared, cfg, model_id):
    root = prepared["out_dir"] / "arms" / prepared["config_hash"]
    result_path = root / f"{model_id}_s{cfg.seed}.json"
    checkpoint_path = root / f"{model_id}_s{cfg.seed}.pt"
    if result_path.exists():
        print(f"  [cached] {model_id} 완료 결과 재사용")
        return json.loads(result_path.read_text(encoding="utf-8"))
    rho = 0.0 if model_id == MODELS[0] else cfg.rho
    model = _build_model(prepared, cfg, rho)
    store = ProgressStore(
        prepared["out_dir"] / "progress" / prepared["config_hash"],
        RunIdentity(
            stage="historical_dynamic_level_change_train",
            model_id=model_id,
            seed=cfg.seed,
            config_hash=_arm_hash(prepared, model_id),
            source_revision=prepared["revision"],
            input_hash=prepared["input_hash"],
        ),
    )
    training = _train(model, prepared, cfg, model_id, store)
    model.eval()
    model.set_context(prepared["final_context"]["name"])
    metrics, _ = moe._flat_evaluation(
        model,
        0.0,
        prepared["cache"],
        prepared["meta"],
        prepared["data"],
        prepared["base_cfg"],
        per_user=False,
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint_path.with_suffix(".pt.tmp")
    torch.save(
        {
            "state": clone_state(model),
            "model_id": model_id,
            "config": asdict(cfg),
            "training": training,
            "diagnostics": model.condition_diagnostics(),
        },
        temporary,
    )
    os.replace(temporary, checkpoint_path)
    payload = {
        "model_id": model_id,
        "role": "baseline" if rho == 0 else "model",
        "seed": cfg.seed,
        "split": "historical_development_days_684_690",
        "final_epoch": cfg.epochs,
        "rho": rho,
        "metrics": test10._public_metrics(metrics),
        "diagnostics": model.condition_diagnostics(),
        "training": training,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
    }
    test10._atomic_json(result_path, payload)
    store.mark_complete(
        epoch=cfg.epochs,
        max_epoch=cfg.epochs,
        selection="none",
        split="historical_development_days_684_690",
        checkpoint_path=str(checkpoint_path),
        result_path=str(result_path),
    )
    return payload


def _row(payload):
    return {
        "model_id": payload["model_id"],
        "role": payload["role"],
        "seed": payload["seed"],
        "split": payload["split"],
        "final_epoch": payload["final_epoch"],
        "rho": payload["rho"],
        **payload["diagnostics"],
        **payload["metrics"],
    }


def _comparison(frame):
    indexed = frame.set_index("model_id")
    baseline, model = indexed.loc[MODELS[0]], indexed.loc[MODELS[1]]
    metric_names = set(frame.columns) - {
        "model_id", "role", "seed", "split", "final_epoch", "rho",
        "level_dimension_weight_abs_mean", "level_dimension_weight_abs_max",
        "change_dimension_weight_abs_mean", "change_dimension_weight_abs_max",
        "scale_min", "scale_max", "level_across_anchor_std_mean",
        "change_across_anchor_std_mean", "level_changing_user_share",
        "change_changing_user_share",
    }
    rows = []
    for metric in metric_names:
        base_value, model_value = baseline[metric], model[metric]
        rows.append(
            {
                "metric": metric,
                MODELS[0]: base_value,
                MODELS[1]: model_value,
                "absolute_delta": model_value - base_value,
                "relative_change_pct": (
                    100 * (model_value - base_value) / base_value
                    if base_value != 0 else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def run_dynamic_level_change(
    cfg: DynamicLevelChangeConfig | None = None,
) -> pd.DataFrame:
    cfg = validate_dynamic_level_change(cfg or configure_dynamic_level_change())
    prepared = _prepare(cfg)
    payloads = [_run_arm(prepared, cfg, model_id) for model_id in MODELS]
    frame = pd.DataFrame([_row(payload) for payload in payloads])
    comparison = _comparison(frame)
    stem = f"m2_dynamic_clv_level_change_{prepared['config_hash']}"
    absolute_path = prepared["out_dir"] / f"{stem}.csv"
    comparison_path = prepared["out_dir"] / f"{stem}_comparison.csv"
    json_path = prepared["out_dir"] / f"{stem}.json"
    frame.to_csv(absolute_path, index=False)
    comparison.to_csv(comparison_path, index=False)

    indexed = comparison.set_index("metric")
    accuracy = {
        metric: float(indexed.at[metric, MODELS[1]])
        / float(indexed.at[metric, MODELS[0]])
        for metric in (
            "recall@10", "ndcg@10", "recall@20", "ndcg@20",
            "recall@50", "ndcg@50",
        )
    }
    weighted_delta = float(
        indexed.at["price_purchase_amount_weighted_hit@10", "absolute_delta"]
    )
    decision = {
        "positive_screen": bool(min(accuracy.values()) >= 0.99 and weighted_delta > 0),
        "accuracy_ratios": accuracy,
        "price_purchase_amount_weighted_hit@10_delta": weighted_delta,
        "next_if_positive": "repeat with several seeds, then freeze before test",
        "statistical_note": "seed 42 historical development screen; no significance claim",
    }
    report = {
        "preflight": preflight_summary(cfg),
        "input_condition_diagnostics": prepared["input_diagnostics"],
        "anchor_stats": [context["stats"] for context in prepared["contexts"]],
        "rows": [_row(payload) for payload in payloads],
        "comparison": comparison.to_dict(orient="records"),
        "decision": decision,
        "result_files": {
            "absolute_csv": str(absolute_path),
            "comparison_csv": str(comparison_path),
            "json": str(json_path),
        },
    }
    test10._atomic_json(json_path, report)
    frame.attrs["comparison"] = comparison
    frame.attrs["decision"] = decision
    frame.attrs["result_files"] = report["result_files"]
    return frame
