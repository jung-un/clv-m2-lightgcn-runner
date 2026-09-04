"""Shared H&M two-year, seed-42 validation helpers for M2 and M4."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from clv_run_state import ProgressStore, RunIdentity, file_sha256
import lightgcn_clv_axis_specific_test10 as test10
import lightgcn_clv_gradient_isolated_economic_interaction as clv_helpers
import lightgcn_clv_joint_nv as joint
import lightgcn_clv_joint_response_embedding as economic
import lightgcn_clv_moe as moe
import lightgcn_clv_residual as residual
import lightgcn_clv_v3 as v3


DEFAULT_BATCH_SIZE = 131_072
BATCH_CANDIDATES = (131_072, 65_536, 32_768)
ACCURACY_METRICS = (
    "recall@10",
    "ndcg@10",
    "recall@20",
    "ndcg@20",
    "recall@50",
    "ndcg@50",
)


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def config_hash(code_version: str, cfg, input_hash: str, revision: str) -> str:
    payload = {
        "code_version": code_version,
        "config": asdict(cfg),
        "input_hash": input_hash,
        "source_revision": revision,
    }
    return hashlib.sha256(canonical(payload).encode()).hexdigest()[:12]


def configure_base(cfg) -> dict:
    configured = v3.configure_run(
        "hm",
        out_dir=cfg.out_dir,
        ARCH="pref_only",
        SEED_LIST=[42],
        WINDOW_DAYS=None,
        TRAIN_ON_VAL=False,
        EVAL_TEST=False,
        EVAL_HOLDOUT=False,
        GRAPH_MODE="binary",
        LOSS_MODE="plain",
        NEG_MODE="uniform",
        MIN_USER_INTER=1,
        MIN_ITEM_INTER=1,
        DIM=64,
        N_LAYERS=2,
        BATCH_SIZE=cfg.batch_size,
        LR=cfg.lr,
        PREF_REG=cfg.pref_reg,
        EPOCHS=cfg.epochs,
        EARLY_STOP=cfg.epochs,
        REPORT_LEGACY_VALUE_FEATURES=False,
    )
    base = dict(configured)
    required = {
        "ARCH": "pref_only",
        "WINDOW_DAYS": None,
        "TRAIN_ON_VAL": False,
        "EVAL_TEST": False,
        "EVAL_HOLDOUT": False,
        "GRAPH_MODE": "binary",
        "LOSS_MODE": "plain",
        "NEG_MODE": "uniform",
        "MIN_USER_INTER": 1,
        "MIN_ITEM_INTER": 1,
        "EPOCHS": 100,
    }
    for key, expected in required.items():
        if base[key] != expected:
            raise RuntimeError(f"H&M seed-42 설정 오염: {key}={base[key]!r}")
    return base


def prepare_hm2y(cfg, *, code_version: str) -> dict:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = moe.build_input_manifest(v3.SCHEMA["hm"])
    input_hash = moe.manifest_hash(manifest)
    revision = moe.source_revision()
    base_cfg = configure_base(cfg)
    data = v3.prepare_data(base_cfg, v3.DCFG)
    if set(data["splits"]) != {"val"}:
        raise RuntimeError(f"H&M validation 외 분할 오염: {sorted(data['splits'])}")
    expected_end = pd.Timestamp("2020-09-01")
    if pd.Timestamp(data["train"].t.max()) != expected_end:
        raise RuntimeError(f"H&M train 종료일 오류: {data['train'].t.max()}")
    if data.get("loss_w") is not None:
        raise RuntimeError("기존 표본 가중치가 섞였습니다")
    data["loss_w"] = None

    snapshot = residual.build_final_snapshot(
        data["train"], data["n_users"], v3.DCFG["is_date"], cfg.input_days
    )
    axes = joint.build_user_axis_inputs(snapshot, data["n_users"])
    q_n, q_v, q_c, clv_valid = clv_helpers.build_clv_inputs(axes)
    q_n = np.where(clv_valid, q_n, 0.0).astype(np.float32)
    q_v = np.where(clv_valid, q_v, 0.0).astype(np.float32)
    q_c = np.where(clv_valid, q_c, 0.0).astype(np.float32)
    item_economic, item_economic_valid = economic.build_item_economic_inputs(
        data["train"], data["n_items"]
    )
    x_item, item_cat = v3.item_value_features(
        data["train"], data["n_items"], report=False
    )
    meta = v3.item_meta(data["train"], data["n_items"])
    thresholds = v3.segment_thresholds(axes["clv_proxy"], base_cfg["SEG_EDGES"])
    cache = v3.EvalCache(
        *data["splits"]["val"], axes["clv_proxy"], thresholds, data["n_items"]
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
        "clv_valid": np.asarray(clv_valid, dtype=bool),
        "item_economic": item_economic,
        "item_economic_valid": item_economic_valid,
        "x_item": x_item,
        "item_cat": item_cat,
        "meta": meta,
        "cache": cache,
    }
    prepared["config_hash"] = config_hash(
        code_version, cfg, input_hash, revision
    )
    train_edges = data["train"][["u_idx", "i_idx"]].drop_duplicates()
    prepared["binary_user_degree"] = np.bincount(
        train_edges["u_idx"].to_numpy(np.int64), minlength=data["n_users"]
    )
    return prepared


def degree_matched_sources(
    valid: np.ndarray,
    user_degree: np.ndarray,
    *,
    n_bins: int,
    seed: int,
) -> dict:
    valid = np.asarray(valid, dtype=bool)
    user_degree = np.asarray(user_degree, dtype=np.int64)
    if valid.shape != user_degree.shape:
        raise ValueError("CLV 유효성·user degree shape이 다릅니다")
    valid_index = np.flatnonzero(valid & (user_degree > 0))
    if len(valid_index) < 2:
        raise RuntimeError("degree-matched 순열의 유효 고객이 부족합니다")
    ranks = pd.Series(user_degree[valid_index]).rank(method="average").to_numpy()
    strata_valid = np.floor((ranks - 0.5) * n_bins / len(valid_index)).astype(
        np.int16
    )
    strata_valid = np.minimum(strata_valid, n_bins - 1)
    strata = np.full(len(valid), -1, dtype=np.int16)
    strata[valid_index] = strata_valid
    source = np.arange(len(valid), dtype=np.int64)
    rng = np.random.default_rng(seed)
    for stratum in np.unique(strata_valid):
        target = valid_index[strata_valid == stratum]
        if len(target) < 2:
            continue
        permuted = rng.permutation(target)
        if np.array_equal(permuted, target):
            permuted = np.roll(permuted, 1)
        source[target] = permuted
    changed = valid_index[source[valid_index] != valid_index]
    if not len(changed):
        raise RuntimeError("degree-matched 순열이 고객 배정을 바꾸지 못했습니다")
    if np.any(strata[changed] != strata[source[changed]]):
        raise RuntimeError("degree-matched 순열이 user-degree 구간을 벗어났습니다")
    return {
        "source_user": source,
        "stratum": strata,
        "changed_valid_user_share": float(len(changed) / len(valid_index)),
    }


def parameter_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    names = set(dict(model.named_parameters()))
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if name in names
    }


def load_parameter_state(
    model: torch.nn.Module, state: dict[str, torch.Tensor]
) -> None:
    parameter_names = set(dict(model.named_parameters()))
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing_parameters = parameter_names.intersection(missing)
    if missing_parameters or unexpected:
        raise RuntimeError(
            "경량 checkpoint 파라미터 불일치: "
            f"missing={sorted(missing_parameters)}, unexpected={unexpected}"
        )


def atomic_torch(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def restore_compact(
    store: ProgressStore,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    rng: np.random.Generator,
) -> dict | None:
    if not store.latest_checkpoint.exists():
        return None
    payload = torch.load(
        store.latest_checkpoint, map_location="cpu", weights_only=False
    )
    store._validate_identity(payload.get("identity", {}))
    load_parameter_state(model, payload["parameter_state"])
    optimizer.load_state_dict(payload["optimizer_state"])
    rng.bit_generator.state = payload["numpy_rng_state"]
    torch.set_rng_state(payload["torch_rng_state"])
    cuda_state = payload.get("cuda_rng_state", [])
    if torch.cuda.is_available() and cuda_state:
        torch.cuda.set_rng_state_all(cuda_state)
    return payload


def save_compact(
    store: ProgressStore,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    rng: np.random.Generator,
    **state,
) -> None:
    atomic_torch(
        store.latest_checkpoint,
        {
            "identity": asdict(store.identity),
            "parameter_state": parameter_state(model),
            "optimizer_state": optimizer.state_dict(),
            "numpy_rng_state": rng.bit_generator.state,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
            ),
            **state,
        },
    )
    store.mark_stage(
        "running",
        epoch=int(state["epoch"]),
        max_epoch=int(state["max_epoch"]),
        checkpoint_path=str(store.latest_checkpoint),
    )


def progress_store(prepared: dict, cfg, model_id: str, arm_hash: str) -> ProgressStore:
    return ProgressStore(
        prepared["out_dir"] / "progress" / prepared["config_hash"],
        RunIdentity(
            stage="hm2y_validation_fixed_epoch_train",
            model_id=model_id,
            seed=42,
            config_hash=arm_hash,
            source_revision=prepared["revision"],
            input_hash=prepared["input_hash"],
        ),
    )


def train_plain_bpr(model, prepared: dict, cfg, model_id: str, store) -> dict:
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=0.0)
    rng = np.random.default_rng(42)
    restored = restore_compact(store, model, optimizer, rng)
    start_epoch = 1 if restored is None else int(restored["epoch"]) + 1
    history = [] if restored is None else list(restored.get("history", []))
    updates = 0 if restored is None else int(restored.get("updates", 0))
    samples = 0 if restored is None else int(restored.get("samples", 0))
    if restored is not None:
        print(f"  [{model_id}] epoch {start_epoch - 1}에서 자동 재개")
    data = prepared["data"]
    tr_u, tr_i, pos_key = data["tr_u"], data["tr_i"], data["pos_key"]
    n_batches = math.ceil(len(tr_u) / cfg.batch_size)
    gate = torch.ones(data["n_users"], device=v3.DEVICE)
    for epoch in range(start_epoch, cfg.epochs + 1):
        model.train()
        started = time.time()
        permutation = rng.permutation(len(tr_u))
        loss_sum = bpr_sum = correct_sum = 0.0
        for batch in range(n_batches):
            index = permutation[
                batch * cfg.batch_size : (batch + 1) * cfg.batch_size
            ]
            users, positives = tr_u[index], tr_i[index]
            negatives = v3.sample_negatives(
                users,
                positives,
                data["n_items"],
                pos_key,
                rng,
                "uniform",
                data["item_cat"],
                data["cat_items"],
            )
            tensors = [
                torch.as_tensor(value, dtype=torch.long, device=v3.DEVICE)
                for value in (users, positives, negatives)
            ]
            loss, diagnostics = model.bpr_loss(*tensors, gate, 0.0, None)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach())
            bpr_sum += float(diagnostics["bpr"])
            correct_sum += float(diagnostics["p_correct"])
            updates += 1
            samples += len(index)
            store.heartbeat(
                epoch=epoch,
                max_epoch=cfg.epochs,
                batch=batch + 1,
                batches=n_batches,
                loss=loss_sum / (batch + 1),
            )
        record = {
            "epoch": epoch,
            "loss": loss_sum / n_batches,
            "bpr": bpr_sum / n_batches,
            "p_correct": correct_sum / n_batches,
            "epoch_sec": time.time() - started,
        }
        if hasattr(model, "epoch_training_diagnostics"):
            record.update(model.epoch_training_diagnostics())
        history.append(record)
        save_compact(
            store,
            model,
            optimizer,
            rng,
            epoch=epoch,
            max_epoch=cfg.epochs,
            history=history,
            updates=updates,
            samples=samples,
        )
        print(
            f"  [{model_id}] ep {epoch:3d}/{cfg.epochs} | "
            f"loss {record['loss']:.4f} | P(pos>neg) {record['p_correct']:.3f} | "
            f"{record['epoch_sec']:.0f}s"
        )
    return {
        "epochs_run": cfg.epochs,
        "selection": "none",
        "early_stopping": False,
        "resumed_from_epoch": start_epoch - 1,
        "updates": updates,
        "samples": samples,
        "history": history,
        "final_diagnostics": history[-1] if history else {},
    }


def evaluate(model, prepared: dict) -> dict:
    raw, _ = moe._flat_evaluation(
        model,
        0.0,
        prepared["cache"],
        prepared["meta"],
        prepared["data"],
        prepared["base_cfg"],
        per_user=False,
    )
    return test10._public_metrics(raw)


def final_parameter_checkpoint(
    path: Path, model, prepared: dict, cfg, model_id: str, training: dict
) -> None:
    atomic_torch(
        path,
        {
            "parameter_state": parameter_state(model),
            "model_id": model_id,
            "config": asdict(cfg),
            "training": training,
            "source_revision": prepared["revision"],
            "input_hash": prepared["input_hash"],
        },
    )


def checkpoint_sha256(path: Path) -> str:
    return file_sha256(path)


def read_progress(out_dir: str | Path) -> dict:
    candidates = sorted(
        Path(out_dir).glob("progress/*/progress.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return {"status": "not_started"}
    payload = json.loads(candidates[0].read_text(encoding="utf-8"))
    payload["progress_path"] = str(candidates[0])
    return payload


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    os.replace(temporary, path)
