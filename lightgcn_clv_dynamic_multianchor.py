"""Historical screen for time-indexed CLV-conditioned LightGCN.

Four training anchors are built before the development week. At every anchor
the binary graph and historical N x V proxy use only the past, while positives
are first-time user-item pairs from the following seven days. One LightGCN is
shared across anchors. The matched M1 arm uses the same protocol with rho=0.
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

from clv_dynamic_clv_modulation_model import DynamicCLVModulationLightGCN
from clv_run_state import ProgressStore, RunIdentity, clone_state, file_sha256
import lightgcn_clv_axis_specific_test10 as test10
import lightgcn_clv_moe as moe
import lightgcn_clv_v3 as v3


CODE_VERSION = "m2-dynamic-clv-multianchor-historical-screen-v1"
MODELS = ("m1_multianchor_rho0", "m2_dynamic_clv")


@dataclass(frozen=True)
class DynamicMultiAnchorConfig:
    dataset: str = "dunnhumby"
    seed: int = 42
    time_cutoff: int = 690
    evaluation_days: int = 7
    anchor_count: int = 4
    anchor_horizon_days: int = 7
    epochs: int = 100
    embedding_dim: int = 64
    n_layers: int = 2
    rho: float = 0.05
    batch_size: int = 8192
    lr: float = 5e-4
    pref_reg: float = 1e-3
    out_dir: str = ""


def configure_dynamic_multianchor(**overrides) -> DynamicMultiAnchorConfig:
    defaults = {
        "out_dir": (
            f"{v3.default_out_dir('dunnhumby')}"
            "_m2_dynamic_clv_multianchor_historical_screen_v1"
        )
    }
    return validate_dynamic_multianchor(
        DynamicMultiAnchorConfig(**(defaults | overrides))
    )


def validate_dynamic_multianchor(
    cfg: DynamicMultiAnchorConfig,
) -> DynamicMultiAnchorConfig:
    required = {
        "dataset": "dunnhumby",
        "seed": 42,
        "time_cutoff": 690,
        "evaluation_days": 7,
        "anchor_count": 4,
        "anchor_horizon_days": 7,
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


def preflight_summary(cfg: DynamicMultiAnchorConfig) -> dict:
    cfg = validate_dynamic_multianchor(cfg)
    train_end = cfg.time_cutoff - cfg.evaluation_days
    first_target_start = train_end - cfg.anchor_count * cfg.anchor_horizon_days + 1
    return {
        "code_version": CODE_VERSION,
        "dataset": cfg.dataset,
        "seed": cfg.seed,
        "models": list(MODELS),
        "research_axis": "M2 embedding/representation intervention",
        "historical_development_split": {
            "multi_anchor_target_start": first_target_start,
            "multi_anchor_target_end": train_end,
            "evaluation_start": train_end + 1,
            "evaluation_end": cfg.time_cutoff,
            "final_test_constructed": False,
            "holdout_constructed": False,
        },
        "historical_clv_proxy": {
            "N": "within-anchor percentile of cumulative basket count",
            "V": "within-anchor percentile of mean basket value",
            "C": "N * V",
            "condition": "2 * within-anchor percentile(C) - 1",
            "limitation": "historical CLV proxy, not future or lifetime profit",
        },
        "m2": {
            "formula": "NormPreserve(E_u_ID * (1 + rho*c_u,t*tanh(w)))",
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


def _base_config(cfg: DynamicMultiAnchorConfig) -> dict:
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
        "TIME_CUTOFF": 690,
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


def _basket_table(history: pd.DataFrame) -> pd.DataFrame:
    keys = ["u_idx", "b_raw"] if "b_raw" in history.columns else ["u_idx", "t"]
    return (
        history.groupby(keys, sort=False)
        .agg(basket_value=("v", "sum"), basket_time=("t", "max"))
        .reset_index()
    )


def historical_clv_condition(history: pd.DataFrame, n_users: int) -> dict:
    """Compute an anchor-specific historical N x V proxy without future rows."""
    baskets = _basket_table(history)
    summary = baskets.groupby("u_idx", sort=False).agg(
        basket_count=("basket_value", "size"),
        mean_basket_value=("basket_value", "mean"),
    )
    summary["n_percentile"] = summary["basket_count"].rank(
        pct=True, method="average"
    )
    summary["v_percentile"] = summary["mean_basket_value"].rank(
        pct=True, method="average"
    )
    summary["clv_proxy"] = summary["n_percentile"] * summary["v_percentile"]
    summary["clv_percentile"] = summary["clv_proxy"].rank(
        pct=True, method="average"
    )
    condition = np.zeros(n_users, np.float32)
    clv_proxy = np.full(n_users, np.nan, np.float64)
    ids = summary.index.to_numpy(np.int64)
    condition[ids] = 2.0 * summary["clv_percentile"].to_numpy(np.float32) - 1.0
    clv_proxy[ids] = summary["clv_proxy"].to_numpy(np.float64)
    return {
        "condition": condition,
        "clv_proxy": clv_proxy,
        "valid": np.isfinite(clv_proxy),
        "n_baskets": int(len(baskets)),
    }


def _unique_pair_keys(frame: pd.DataFrame, n_items: int) -> np.ndarray:
    if frame.empty:
        return np.empty(0, np.int64)
    return np.unique(
        frame["u_idx"].to_numpy(np.int64) * n_items
        + frame["i_idx"].to_numpy(np.int64)
    )


def _build_training_context(
    all_train: pd.DataFrame,
    *,
    history_end: int,
    target_end: int,
    n_users: int,
    n_items: int,
) -> dict:
    history = all_train[all_train["t"] <= history_end].copy()
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
    if len(target_key):
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
    target_users = (target_key // n_items).astype(np.int64)
    target_items = (target_key % n_items).astype(np.int64)
    forbidden_key = np.unique(np.concatenate([history_key, target_key]))
    clv = historical_clv_condition(history, n_users)
    return {
        "name": f"history_le_{history_end}_target_{history_end + 1}_{target_end}",
        "history_end": int(history_end),
        "target_start": int(history_end + 1),
        "target_end": int(target_end),
        "adj": adjacency,
        "condition": clv["condition"],
        "clv_proxy": clv["clv_proxy"],
        "tr_u": target_users,
        "tr_i": target_items,
        "forbidden_key": forbidden_key,
        "stats": {
            "history_rows": int(len(history)),
            "history_edges": int(len(history_key)),
            "history_users": int(history["u_idx"].nunique()),
            "history_items": int(history["i_idx"].nunique()),
            "target_rows_before_new_pair_filter": int(len(target)),
            "new_item_target_pairs": int(len(target_key)),
            "target_users": int(len(np.unique(target_users))),
            "valid_clv_users": int(clv["valid"].sum()),
            "historical_baskets": clv["n_baskets"],
            "condition_mean": float(clv["condition"][clv["valid"]].mean()),
            "condition_std": float(clv["condition"][clv["valid"]].std()),
        },
    }


def _config_hash(cfg, input_hash: str, revision: str) -> str:
    payload = {
        "code_version": CODE_VERSION,
        "config": asdict(cfg),
        "models": MODELS,
        "input_hash": input_hash,
        "source_revision": revision,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()[:12]


def _prepare(cfg: DynamicMultiAnchorConfig) -> dict:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = moe.build_input_manifest(v3.SCHEMA[cfg.dataset])
    input_hash = moe.manifest_hash(manifest)
    revision = moe.source_revision()
    base_cfg = _base_config(cfg)
    data = v3.prepare_data(base_cfg, v3.DCFG)
    if set(data["splits"]) != {"test"}:
        raise RuntimeError("historical development split is contaminated")
    if data.get("loss_w") is not None:
        raise RuntimeError("M4 sample weights are present")
    data["loss_w"] = None
    train_end = cfg.time_cutoff - cfg.evaluation_days
    if float(data["train"]["t"].max()) != float(train_end):
        raise RuntimeError("unexpected historical train boundary")

    contexts = []
    first_target_start = train_end - cfg.anchor_count * cfg.anchor_horizon_days + 1
    for index in range(cfg.anchor_count):
        target_start = first_target_start + index * cfg.anchor_horizon_days
        target_end = target_start + cfg.anchor_horizon_days - 1
        contexts.append(
            _build_training_context(
                data["train"],
                history_end=target_start - 1,
                target_end=target_end,
                n_users=data["n_users"],
                n_items=data["n_items"],
            )
        )
    if contexts[-1]["target_end"] != train_end:
        raise RuntimeError("last anchor does not end at the train boundary")

    final_clv = historical_clv_condition(data["train"], data["n_users"])
    final_context = {
        "name": f"evaluation_history_le_{train_end}",
        "adj": data["adj"],
        "condition": final_clv["condition"],
    }
    meta = v3.item_meta(data["train"], data["n_items"])
    thresholds = v3.segment_thresholds(
        final_clv["clv_proxy"], base_cfg["SEG_EDGES"]
    )
    cache = v3.EvalCache(
        *data["splits"]["test"],
        final_clv["clv_proxy"],
        thresholds,
        data["n_items"],
    )
    print("  다중시점 학습 컨텍스트:")
    for context in contexts:
        print("   ", context["name"], context["stats"])
    return {
        "out_dir": out_dir,
        "manifest": manifest,
        "input_hash": input_hash,
        "revision": revision,
        "config_hash": _config_hash(cfg, input_hash, revision),
        "base_cfg": base_cfg,
        "data": data,
        "contexts": contexts,
        "final_context": final_context,
        "final_clv": final_clv,
        "meta": meta,
        "cache": cache,
    }


def _build_model(prepared: dict, cfg: DynamicMultiAnchorConfig, rho: float):
    all_contexts = prepared["contexts"] + [prepared["final_context"]]
    v3.set_seed(cfg.seed)
    return DynamicCLVModulationLightGCN(
        n_users=prepared["data"]["n_users"],
        n_items=prepared["data"]["n_items"],
        adjacencies=[context["adj"] for context in all_contexts],
        clv_conditions=[context["condition"] for context in all_contexts],
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


def _train(
    model,
    prepared: dict,
    cfg: DynamicMultiAnchorConfig,
    model_id: str,
    store: ProgressStore,
) -> dict:
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
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
    started = time.time()

    for epoch in range(start_epoch, cfg.epochs + 1):
        model.train()
        epoch_started = time.time()
        totals = {"loss": 0.0, "bpr": 0.0, "p_correct": 0.0, "batches": 0}
        for context_index in rng.permutation(len(prepared["contexts"])):
            context = prepared["contexts"][int(context_index)]
            model.set_context(int(context_index))
            tr_u, tr_i = context["tr_u"], context["tr_i"]
            permutation = rng.permutation(len(tr_u))
            n_batches = math.ceil(len(tr_u) / cfg.batch_size)
            for batch in range(n_batches):
                index = permutation[
                    batch * cfg.batch_size : (batch + 1) * cfg.batch_size
                ]
                users, positives = tr_u[index], tr_i[index]
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
            f"|tanh(w)| {record['clv_dimension_weight_abs_mean']:.4f} | "
            f"{record['epoch_sec']:.0f}s"
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


def _run_arm(prepared: dict, cfg: DynamicMultiAnchorConfig, model_id: str) -> dict:
    root = prepared["out_dir"] / "arms" / prepared["config_hash"]
    result_path = root / f"{model_id}_s{cfg.seed}.json"
    checkpoint_path = root / f"{model_id}_s{cfg.seed}.pt"
    if result_path.exists():
        print(f"  [cached] {model_id} 완료 결과 재사용")
        return json.loads(result_path.read_text(encoding="utf-8"))
    rho = 0.0 if model_id == "m1_multianchor_rho0" else cfg.rho
    model = _build_model(prepared, cfg, rho)
    store = ProgressStore(
        prepared["out_dir"] / "progress" / prepared["config_hash"],
        RunIdentity(
            stage="historical_multianchor_train",
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
        "role": "baseline" if rho == 0.0 else "model",
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


def _row(payload: dict) -> dict:
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


def _comparison(frame: pd.DataFrame) -> pd.DataFrame:
    indexed = frame.set_index("model_id")
    baseline = indexed.loc["m1_multianchor_rho0"]
    model = indexed.loc["m2_dynamic_clv"]
    metadata = {
        "model_id", "role", "seed", "split", "final_epoch", "rho",
        "clv_dimension_weight_abs_mean", "clv_dimension_weight_abs_max",
        "scale_min", "scale_max", "condition_across_anchor_std_mean",
        "condition_changing_user_share",
    }
    rows = []
    for metric in frame.columns:
        if metric in metadata:
            continue
        base_value, model_value = baseline[metric], model[metric]
        rows.append(
            {
                "metric": metric,
                "m1_multianchor_rho0": base_value,
                "m2_dynamic_clv": model_value,
                "absolute_delta": model_value - base_value,
                "relative_change_pct": (
                    100.0 * (model_value - base_value) / base_value
                    if base_value != 0 else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def run_dynamic_multianchor(
    cfg: DynamicMultiAnchorConfig | None = None,
) -> pd.DataFrame:
    cfg = validate_dynamic_multianchor(cfg or configure_dynamic_multianchor())
    prepared = _prepare(cfg)
    payloads = [_run_arm(prepared, cfg, model_id) for model_id in MODELS]
    frame = pd.DataFrame([_row(payload) for payload in payloads])
    comparison = _comparison(frame)

    stem = f"m2_dynamic_clv_multianchor_{prepared['config_hash']}"
    absolute_path = prepared["out_dir"] / f"{stem}.csv"
    comparison_path = prepared["out_dir"] / f"{stem}_comparison.csv"
    json_path = prepared["out_dir"] / f"{stem}.json"
    frame.to_csv(absolute_path, index=False)
    comparison.to_csv(comparison_path, index=False)

    indexed = comparison.set_index("metric")
    accuracy = {
        metric: float(indexed.at[metric, "m2_dynamic_clv"])
        / float(indexed.at[metric, "m1_multianchor_rho0"])
        for metric in (
            "recall@10", "ndcg@10", "recall@20", "ndcg@20",
            "recall@50", "ndcg@50",
        )
    }
    weighted_delta = float(
        indexed.at["price_purchase_amount_weighted_hit@10", "absolute_delta"]
    )
    decision = {
        "positive_screen": bool(
            min(accuracy.values()) >= 0.99 and weighted_delta > 0
        ),
        "accuracy_ratios": accuracy,
        "price_purchase_amount_weighted_hit@10_delta": weighted_delta,
        "next_if_positive": (
            "repeat with several seeds, then freeze before protected test"
        ),
        "statistical_note": (
            "seed 42 historical development screen; no significance claim"
        ),
    }
    report = {
        "preflight": preflight_summary(cfg),
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
