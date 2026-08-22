"""Final Dunnhumby test-only runner for the accepted M2 representation.

This module intentionally has no validation selection path.  It merges the
former train and validation intervals, trains every arm for exactly 100
epochs, and evaluates the protected test split only at the final checkpoint.
Completed arm results are cached so reconnecting Colab does not evaluate the
same completed arm again.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
import torch

from clv_dual_axis_model import build_dual_item_profiles
from clv_joint_nv_model import JointNVLightGCN
from clv_run_state import ProgressStore, RunIdentity, clone_state, file_sha256
import lightgcn_clv_joint_nv as joint
import lightgcn_clv_moe as moe
import lightgcn_clv_residual as residual
import lightgcn_clv_v3 as v3


CODE_VERSION = "m2-axis-specific-test10-v1"
SEEDS = tuple(range(42, 52))
MODELS = (
    "m1_64",
    "m2_axis_specific_gate",
    "m1_96",
    "m2_shuffled_user",
)
PUBLIC_METRIC_NAMES = {
    "revenue": "price_purchase_amount_weighted_hit",
    "arp": "mean_recommended_price_percentile",
    "value_alignment": "user_value_tendency_recommended_price_alignment",
}
ACCEPTED_M2_FEATURE_SCHEMA = {
    "user_activity": [
        "repeat_transaction_count",
        "transaction_recency",
        "customer_age",
        "mean_transaction_gap",
        "valid_repeat_transaction_count",
        "valid_transaction_recency",
        "valid_customer_age",
        "valid_mean_transaction_gap",
    ],
    "user_value": [
        "mean_transaction_value",
        "valid_mean_transaction_value",
    ],
    "item_activity": [
        "repeat_purchase_share",
        "log_median_repeat_gap",
        "repeat_gap_valid",
    ],
    "item_value": [
        "price_percentile",
        "category_price_percentile",
        "log_mean_unit_price",
        "mean_transaction_value_share",
    ],
}


@dataclass(frozen=True)
class Test10Config:
    dataset: str = "dunnhumby"
    seeds: tuple[int, ...] = SEEDS
    epochs: int = 100
    id_dim: int = 64
    axis_dim: int = 16
    hidden_dim: int = 32
    n_layers: int = 2
    batch_size: int = 8192
    lr: float = 5e-4
    pref_reg: float = 1e-3
    gamma_init: float = 0.1
    gate_shape: str = "axis_positive"
    input_days: int = 365
    out_dir: str = ""


def configure_test10_run(**overrides) -> Test10Config:
    defaults = {
        "out_dir": (
            f"{v3.default_out_dir('dunnhumby')}"
            "_m2_axis_specific_test10_v1"
        )
    }
    return validate_test10_config(Test10Config(**(defaults | overrides)))


def validate_test10_config(cfg: Test10Config) -> Test10Config:
    """Fail closed if the professor-approved final protocol is altered."""
    required = {
        "dataset": "dunnhumby",
        "seeds": SEEDS,
        "epochs": 100,
        "id_dim": 64,
        "axis_dim": 16,
        "hidden_dim": 32,
        "n_layers": 2,
        "gate_shape": "axis_positive",
        "input_days": 365,
    }
    for key, expected in required.items():
        if getattr(cfg, key) != expected:
            raise ValueError(f"test-only 확증 설정은 {key}={expected!r}이어야 합니다")
    if cfg.batch_size <= 0 or cfg.lr <= 0 or cfg.pref_reg < 0:
        raise ValueError("학습 설정이 잘못됐습니다")
    if not cfg.out_dir:
        raise ValueError("out_dir가 필요합니다")
    return cfg


def preflight_summary(cfg: Test10Config) -> dict:
    cfg = validate_test10_config(cfg)
    return {
        "code_version": CODE_VERSION,
        "dataset": cfg.dataset,
        "seeds": list(cfg.seeds),
        "models": list(MODELS),
        "m2_feature_schema": ACCEPTED_M2_FEATURE_SCHEMA,
        "training_data": "former train + validation",
        "new_item_task": (
            "every user-item pair observed in merged train+validation is "
            "excluded from test truth"
        ),
        "epochs": cfg.epochs,
        "validation_selection": False,
        "early_stopping": False,
        "test_evaluation": "one final checkpoint per seed/model; cached after completion",
        "holdout_evaluation": False,
        "automatic_epoch_resume": True,
        "m2": {
            "architecture": "accepted ID|N|V axis-specific non-negative-gate model",
            "activity_axis_weight": "learned common N-axis weight",
            "transaction_value_axis_weight": "learned common V-axis weight",
            "user_allocation": "positive N and V percentile gates, normalized separately",
        },
        "fixed_boundaries": {
            "graph": "binary",
            "negative_sampling": "uniform",
            "sample_weighting": False,
            "loss": "existing pairwise BPR objective; no added loss term",
            "min_user_interactions": 1,
            "min_item_interactions": 1,
        },
        "reporting": (
            "10-seed means and same-seed paired differences; test is not used "
            "to select an epoch, model, or hyperparameter"
        ),
        "out_dir": cfg.out_dir,
    }


def _base_config(cfg: Test10Config) -> dict:
    configured = v3.configure_run(
        cfg.dataset,
        out_dir=cfg.out_dir,
        ARCH="pref_only",
        SEED_LIST=list(cfg.seeds),
        WINDOW_DAYS=None,
        TRAIN_ON_VAL=True,
        EVAL_TEST=True,
        EVAL_HOLDOUT=False,
        GRAPH_MODE="binary",
        LOSS_MODE="plain",
        NEG_MODE="uniform",
        MIN_USER_INTER=1,
        MIN_ITEM_INTER=1,
        DIM=cfg.id_dim,
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
        "GRAPH_MODE": "binary",
        "LOSS_MODE": "plain",
        "NEG_MODE": "uniform",
        "MIN_USER_INTER": 1,
        "MIN_ITEM_INTER": 1,
        "EPOCHS": 100,
    }
    for key, expected in required.items():
        if base[key] != expected:
            raise RuntimeError(f"최종 test 설정 오염: {key}={base[key]!r}")
    return base


def _config_hash(cfg: Test10Config, input_hash: str, revision: str) -> str:
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


def _prepare(cfg: Test10Config) -> dict:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = moe.build_input_manifest(v3.SCHEMA[cfg.dataset])
    input_hash = moe.manifest_hash(manifest)
    revision = moe.source_revision()
    config_hash = _config_hash(cfg, input_hash, revision)
    base_cfg = _base_config(cfg)
    data = v3.prepare_data(base_cfg, v3.DCFG)
    if set(data["splits"]) != {"test"}:
        raise RuntimeError(
            f"test-only runner에 보호 split 오염: {sorted(data['splits'])}"
        )
    if data.get("loss_w") is not None:
        raise RuntimeError("M2 test에 M4 표본 가중치가 섞였습니다")
    data["loss_w"] = None
    snapshot = residual.build_final_snapshot(
        data["train"], data["n_users"], v3.DCFG["is_date"], cfg.input_days
    )
    axes = joint.build_user_axis_inputs(snapshot, data["n_users"])
    item_profile = build_dual_item_profiles(
        data["train"], data["n_items"], v3.DCFG["is_date"]
    )
    actual_feature_schema = {
        "user_activity": list(axes["activity_names"]),
        "user_value": list(axes["value_names"]),
        "item_activity": list(item_profile.activity_names),
        "item_value": list(item_profile.value_names),
    }
    if actual_feature_schema != ACCEPTED_M2_FEATURE_SCHEMA:
        raise RuntimeError(
            "승인된 M2 사용자·아이템 입력과 실제 입력이 다릅니다: "
            f"expected={ACCEPTED_M2_FEATURE_SCHEMA}, actual={actual_feature_schema}"
        )
    print("  현재 M2 사용자·아이템 입력:")
    for axis, names in actual_feature_schema.items():
        print(f"    {axis}: {names}")
    meta = v3.item_meta(data["train"], data["n_items"])
    thresholds = v3.segment_thresholds(axes["clv_proxy"], base_cfg["SEG_EDGES"])
    cache = v3.EvalCache(
        *data["splits"]["test"],
        axes["clv_proxy"],
        thresholds,
        data["n_items"],
    )
    x_item, item_cat = v3.item_value_features(
        data["train"], data["n_items"], report=False
    )
    return {
        "out_dir": out_dir,
        "manifest": manifest,
        "input_hash": input_hash,
        "revision": revision,
        "config_hash": config_hash,
        "base_cfg": base_cfg,
        "data": data,
        "axes": axes,
        "item_profile": item_profile,
        "feature_schema": actual_feature_schema,
        "meta": meta,
        "cache": cache,
        "x_item": x_item,
        "item_cat": item_cat,
    }


def _model_spec(model_id: str) -> tuple[str, int]:
    if model_id == "m1_64":
        return "m1", 64
    if model_id == "m1_96":
        return "m1", 96
    if model_id == "m2_axis_specific_gate":
        return "joint_nv", 96
    if model_id == "m2_shuffled_user":
        return "joint_shuffled_user", 96
    raise KeyError(model_id)


def _build_model(prepared: dict, cfg: Test10Config, model_id: str, seed: int):
    kind, total_dim = _model_spec(model_id)
    data = prepared["data"]
    v3.set_seed(seed)
    if kind == "m1":
        model_cfg = {**prepared["base_cfg"], "DIM": total_dim}
        model = v3.build_model(
            data,
            data["x_val_u"],
            prepared["x_item"],
            prepared["item_cat"],
            model_cfg,
        )
        return model, list(model.pref_params())
    model = JointNVLightGCN(
        n_users=data["n_users"],
        n_items=data["n_items"],
        user_activity=prepared["axes"]["activity"],
        user_value=prepared["axes"]["value"],
        user_activity_valid=prepared["axes"]["activity_valid"],
        user_value_valid=prepared["axes"]["value_valid"],
        item_profile=prepared["item_profile"],
        q_n=prepared["axes"]["q_n"],
        q_v=prepared["axes"]["q_v"],
        adj=data["adj"],
        id_dim=cfg.id_dim,
        axis_dim=cfg.axis_dim,
        hidden_dim=cfg.hidden_dim,
        n_layers=cfg.n_layers,
        variant=kind,
        gate_shape=cfg.gate_shape,
        shuffle_seed=seed,
        pref_reg=cfg.pref_reg,
        gamma_init=cfg.gamma_init,
        anchor_weight=0.0,
        preference_preserving=True,
    ).to(v3.DEVICE)
    return model, list(model.parameters())


def _arm_hash(
    prepared: dict, cfg: Test10Config, model_id: str, seed: int
) -> str:
    payload = {
        "run": prepared["config_hash"],
        "model_id": model_id,
        "model_spec": _model_spec(model_id),
        "seed": seed,
        "epochs": cfg.epochs,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()[:12]


def _progress_store(
    prepared: dict, cfg: Test10Config, model_id: str, seed: int
) -> ProgressStore:
    return ProgressStore(
        prepared["out_dir"] / "progress" / prepared["config_hash"],
        RunIdentity(
            stage="final_train_test",
            model_id=model_id,
            seed=seed,
            config_hash=_arm_hash(prepared, cfg, model_id, seed),
            source_revision=prepared["revision"],
            input_hash=prepared["input_hash"],
        ),
    )


def _fixed_epoch_train(
    model,
    params,
    prepared: dict,
    cfg: Test10Config,
    model_id: str,
    seed: int,
    store: ProgressStore,
) -> dict:
    """Train exactly 100 epochs without constructing or reading validation."""
    base_cfg, data = prepared["base_cfg"], prepared["data"]
    wd = base_cfg["WD"] if base_cfg["REG_MODE"] == "global_wd" else 0.0
    optimizer = torch.optim.Adam(params, lr=cfg.lr, weight_decay=wd)
    rng = np.random.default_rng(seed)
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
        print(f"  [{model_id} s{seed}] epoch {start_epoch - 1}에서 자동 재개")
    store.mark_stage(
        "running",
        epoch=start_epoch - 1,
        max_epoch=cfg.epochs,
        selection="none",
    )
    tr_u, tr_i, pos_key = data["tr_u"], data["tr_i"], data["pos_key"]
    n_train = len(tr_u)
    n_batches = math.ceil(n_train / cfg.batch_size)
    ones = torch.ones(data["n_users"], dtype=torch.float32, device=v3.DEVICE)
    started = time.time()
    last_epoch = start_epoch - 1
    for epoch in range(start_epoch, cfg.epochs + 1):
        last_epoch = epoch
        model.train()
        epoch_started = time.time()
        permutation = rng.permutation(n_train)
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
            loss, diagnostics = model.bpr_loss(
                torch.as_tensor(users, dtype=torch.long, device=v3.DEVICE),
                torch.as_tensor(positives, dtype=torch.long, device=v3.DEVICE),
                torch.as_tensor(negatives, dtype=torch.long, device=v3.DEVICE),
                ones,
                0.0,
                None,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += float(loss)
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
                selection="none",
            )
        record = {
            "epoch": epoch,
            "loss": loss_sum / n_batches,
            "bpr": bpr_sum / n_batches,
            "p_correct": correct_sum / n_batches,
            "epoch_sec": time.time() - epoch_started,
        }
        if (
            isinstance(model, JointNVLightGCN)
            and model.activity_axis_weight is not None
            and model.transaction_value_axis_weight is not None
        ):
            record.update(
                activity_axis_weight=float(
                    model.activity_axis_weight.detach().cpu()
                ),
                transaction_value_axis_weight=float(
                    model.transaction_value_axis_weight.detach().cpu()
                ),
            )
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
            f"  [{model_id} s{seed}] ep {epoch:3d}/{cfg.epochs} | "
            f"loss {record['loss']:.4f} | P(pos>neg) {record['p_correct']:.3f} | "
            f"{record['epoch_sec']:.0f}s"
        )
    if last_epoch != cfg.epochs:
        raise RuntimeError(f"고정 100 epoch 미완료: {last_epoch}")
    return {
        "epochs_run": cfg.epochs,
        "selection": "none",
        "early_stopping": False,
        "updates": updates,
        "samples": samples,
        "wall_clock_sec": previous_wall + time.time() - started,
        "resumed_from_epoch": start_epoch - 1,
        "history": history,
    }


def _public_metric_key(key: str) -> str:
    if "@" in key:
        name, suffix = key.split("@", 1)
        return f"{PUBLIC_METRIC_NAMES.get(name, name)}@{suffix}"
    return PUBLIC_METRIC_NAMES.get(key, key)


def _public_metrics(metrics: dict) -> dict:
    return {_public_metric_key(key): value for key, value in metrics.items()}


def _public_per_user(per_user: dict) -> dict:
    return {
        PUBLIC_METRIC_NAMES.get(key, key): np.asarray(value)
        for key, value in per_user.items()
    }


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, float_format="%.10g")
    os.replace(temporary, path)


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _arm_paths(prepared: dict, model_id: str, seed: int) -> dict[str, Path]:
    root = prepared["out_dir"] / "arms" / prepared["config_hash"]
    stem = f"{model_id}_s{seed}"
    return {
        "result": root / f"{stem}.json",
        "per_user": root / f"{stem}_per_user.npz",
        "checkpoint": root / f"{stem}.pt",
    }


def _load_cached_arm(paths: dict[str, Path]) -> dict | None:
    if not (paths["result"].exists() and paths["per_user"].exists()):
        return None
    payload = json.loads(paths["result"].read_text(encoding="utf-8"))
    arrays = np.load(paths["per_user"])
    payload["per_user"] = {key: arrays[key] for key in arrays.files}
    print(
        f"  [cached] {payload['model_id']} s{payload['seed']} "
        "완료 결과를 재사용(test 재평가 없음)"
    )
    return payload


def _model_diagnostics(model) -> dict:
    if not isinstance(model, JointNVLightGCN):
        return {
            "activity_axis_weight": None,
            "transaction_value_axis_weight": None,
            "activity_gate_mean": None,
            "activity_gate_std": None,
            "transaction_value_gate_mean": None,
            "transaction_value_gate_std": None,
        }
    return {
        "activity_axis_weight": float(
            model.activity_axis_weight.detach().cpu()
        ),
        "transaction_value_axis_weight": float(
            model.transaction_value_axis_weight.detach().cpu()
        ),
        "activity_gate_mean": float(model.gate_n.mean().detach().cpu()),
        "activity_gate_std": float(model.gate_n.std().detach().cpu()),
        "transaction_value_gate_mean": float(
            model.gate_v.mean().detach().cpu()
        ),
        "transaction_value_gate_std": float(
            model.gate_v.std().detach().cpu()
        ),
    }


def _run_arm(
    prepared: dict,
    cfg: Test10Config,
    model_id: str,
    seed: int,
) -> dict:
    paths = _arm_paths(prepared, model_id, seed)
    cached = _load_cached_arm(paths)
    if cached is not None:
        return cached
    model, params = _build_model(prepared, cfg, model_id, seed)
    store = _progress_store(prepared, cfg, model_id, seed)
    training = _fixed_epoch_train(
        model, params, prepared, cfg, model_id, seed, store
    )
    model.eval()
    checkpoint_payload = {
        "state": clone_state(model),
        "model_id": model_id,
        "seed": seed,
        "training": training,
        "config": asdict(cfg),
        "source_revision": prepared["revision"],
        "input_hash": prepared["input_hash"],
    }
    paths["checkpoint"].parent.mkdir(parents=True, exist_ok=True)
    temporary_checkpoint = paths["checkpoint"].with_suffix(".pt.tmp")
    torch.save(checkpoint_payload, temporary_checkpoint)
    os.replace(temporary_checkpoint, paths["checkpoint"])

    # The only protected-split evaluation call in this arm.  Once its atomic
    # result and per-user files exist, reconnects take the cached path above.
    metrics, per_user = moe._flat_evaluation(
        model,
        0.0,
        prepared["cache"],
        prepared["meta"],
        prepared["data"],
        prepared["base_cfg"],
        per_user=True,
    )
    public_metrics = _public_metrics(metrics)
    public_per_user = _public_per_user(per_user)
    _atomic_npz(paths["per_user"], public_per_user)
    diagnostics = _model_diagnostics(model)
    payload = {
        "model_id": model_id,
        "role": {
            "m1_64": "baseline",
            "m2_axis_specific_gate": "model",
            "m1_96": "capacity_control",
            "m2_shuffled_user": "assignment_control",
        }[model_id],
        "seed": seed,
        "split": "test",
        "final_epoch": cfg.epochs,
        "validation_selection": False,
        "test_evaluation_count": 1,
        "test_evaluated_at": datetime.now(timezone.utc).isoformat(),
        "metrics": public_metrics,
        "diagnostics": diagnostics,
        "training": training,
        "checkpoint": str(paths["checkpoint"]),
        "checkpoint_sha256": file_sha256(paths["checkpoint"]),
        "per_user_path": str(paths["per_user"]),
    }
    _atomic_json(paths["result"], payload)
    payload["per_user"] = public_per_user
    store.mark_complete(
        epoch=cfg.epochs,
        max_epoch=cfg.epochs,
        selection="none",
        split="test",
        test_evaluation_count=1,
        checkpoint_path=str(paths["checkpoint"]),
        result_path=str(paths["result"]),
    )
    return payload


def _absolute_rows(arms: list[dict]) -> pd.DataFrame:
    rows = []
    for arm in arms:
        rows.append(
            {
                "seed": arm["seed"],
                "model_id": arm["model_id"],
                "role": arm["role"],
                "split": "test",
                "epoch": arm["final_epoch"],
                **arm["diagnostics"],
                **arm["metrics"],
            }
        )
    return pd.DataFrame(rows).sort_values(["seed", "model_id"]).reset_index(
        drop=True
    )


def _mean_ci(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    mean = float(values.mean())
    sd = float(values.std(ddof=1)) if n > 1 else 0.0
    half = (
        float(student_t.ppf(0.975, n - 1)) * sd / math.sqrt(n)
        if n > 1
        else 0.0
    )
    return {"n_seeds": n, "mean": mean, "sd": sd, "lo": mean - half, "hi": mean + half}


def _summary_tables(
    absolute: pd.DataFrame, arms: list[dict]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_columns = [
        column
        for column in absolute.columns
        if "@" in column
        or column == "user_value_tendency_recommended_price_alignment"
    ]
    absolute_summary = []
    for model_id, group in absolute.groupby("model_id", sort=False):
        for metric in metric_columns:
            absolute_summary.append(
                {"model_id": model_id, "metric": metric, **_mean_ci(group[metric])}
            )

    arm_map = {(arm["seed"], arm["model_id"]): arm for arm in arms}
    paired_seed_rows = []
    for seed in SEEDS:
        baseline = arm_map[(seed, "m1_64")]
        for model_id in MODELS[1:]:
            compared = arm_map[(seed, model_id)]
            for metric in metric_columns:
                paired_seed_rows.append(
                    {
                        "seed": seed,
                        "model_id": model_id,
                        "reference": "m1_64",
                        "metric": metric,
                        "delta": float(
                            compared["metrics"][metric]
                            - baseline["metrics"][metric]
                        ),
                    }
                )
    paired_seed = pd.DataFrame(paired_seed_rows)
    paired_summary = []
    for (model_id, metric), group in paired_seed.groupby(
        ["model_id", "metric"], sort=False
    ):
        paired_summary.append(
            {
                "model_id": model_id,
                "reference": "m1_64",
                "metric": metric,
                **_mean_ci(group["delta"].to_numpy()),
                "positive_seed_count": int((group["delta"] > 0).sum()),
            }
        )
    return (
        pd.DataFrame(absolute_summary),
        paired_seed,
        pd.DataFrame(paired_summary),
    )


def _persist(
    prepared: dict,
    cfg: Test10Config,
    arms: list[dict],
) -> pd.DataFrame:
    absolute = _absolute_rows(arms)
    absolute_summary, paired_seed, paired_summary = _summary_tables(absolute, arms)
    stem = f"m2_axis_specific_test10_{prepared['config_hash']}"
    paths = {
        "absolute_csv": prepared["out_dir"] / f"{stem}.csv",
        "absolute_summary_csv": prepared["out_dir"] / f"{stem}_mean.csv",
        "paired_seed_csv": prepared["out_dir"] / f"{stem}_paired_seed.csv",
        "paired_summary_csv": prepared["out_dir"] / f"{stem}_paired_mean.csv",
        "json": prepared["out_dir"] / f"{stem}.json",
    }
    seed_dir = prepared["out_dir"] / "seeds" / prepared["config_hash"]
    seed_paths = {
        int(seed): seed_dir / f"seed_{int(seed)}.csv" for seed in cfg.seeds
    }
    _atomic_csv(paths["absolute_csv"], absolute)
    _atomic_csv(paths["absolute_summary_csv"], absolute_summary)
    _atomic_csv(paths["paired_seed_csv"], paired_seed)
    _atomic_csv(paths["paired_summary_csv"], paired_summary)
    for seed, path in seed_paths.items():
        _atomic_csv(path, absolute[absolute["seed"].eq(seed)].copy())
    payload = {
        "code_version": CODE_VERSION,
        "source_revision": prepared["revision"],
        "input_manifest": prepared["manifest"],
        "config": asdict(cfg),
        "preflight": preflight_summary(cfg),
        "data_stats": prepared["data"].get("data_stats", {}),
        "feature_schema": prepared["feature_schema"],
        "absolute_rows": absolute.to_dict("records"),
        "absolute_10seed_summary": absolute_summary.to_dict("records"),
        "same_seed_differences": paired_seed.to_dict("records"),
        "same_seed_10seed_summary": paired_summary.to_dict("records"),
        "result_paths": {name: str(path) for name, path in paths.items()},
        "per_seed_csv": {
            str(seed): str(path) for seed, path in seed_paths.items()
        },
        "interpretation": {
            "selection": "none; test was not used for model or epoch selection",
            "clv": "historical N×V CLV proxy components condition the learned representation",
            "weighted_hit": (
                "price/purchase-amount weighted recommendation hit; not actual "
                "incremental revenue"
            ),
            "significance": (
                "the reported t interval summarizes variation across the 10 "
                "paired seeds"
            ),
        },
    }
    _atomic_json(paths["json"], payload)
    absolute.attrs["absolute_summary"] = absolute_summary
    absolute.attrs["paired_seed"] = paired_seed
    absolute.attrs["paired_summary"] = paired_summary
    absolute.attrs["result_paths"] = {
        **{name: str(path) for name, path in paths.items()},
        **{f"seed_{seed}_csv": str(path) for seed, path in seed_paths.items()},
    }
    return absolute


def run_test10(cfg: Test10Config | None = None) -> pd.DataFrame:
    cfg = validate_test10_config(cfg or configure_test10_run())
    print(json.dumps(preflight_summary(cfg), ensure_ascii=False, indent=2))
    prepared = _prepare(cfg)
    arms = []
    for seed in cfg.seeds:
        for model_id in MODELS:
            print(f"\n===== seed {seed} | {model_id} | final 100-epoch train =====")
            arms.append(_run_arm(prepared, cfg, model_id, seed))
    frame = _persist(prepared, cfg, arms)
    print("\n10시드 test 절대지표:")
    print(frame.to_string(index=False))
    print("\n동일 seed M1@64 대비 10시드 평균 차이:")
    print(frame.attrs["paired_summary"].to_string(index=False))
    print("결과 파일:", frame.attrs["result_paths"])
    return frame


if __name__ == "__main__":
    print(
        json.dumps(
            preflight_summary(configure_test10_run()),
            ensure_ascii=False,
            indent=2,
        )
    )
