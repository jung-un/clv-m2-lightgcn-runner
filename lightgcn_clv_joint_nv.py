"""Seed-42 validation runner for jointly propagated CLV N/V embeddings."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from clv_dual_axis_model import build_dual_item_profiles, fixed_percentile_ranks
from clv_joint_nv_diagnostics import (
    axis_distribution_diagnostics,
    evaluate_block_views,
    find_joint_checkpoint,
    load_joint_checkpoint,
    sampled_block_score_summary,
)
from clv_joint_nv_model import JointNVLightGCN
from clv_run_state import ProgressStore, RunIdentity
from clv_variable_validity import candidate_variables, validate_anchor
import lightgcn_clv_moe as moe
import lightgcn_clv_residual as residual
import lightgcn_clv_v3 as v3


CODE_VERSION = "m2-joint-nv-lightgcn-v1.2"
DIAGNOSTIC_VERSION = "joint-nv-checkpoint-diagnostics-v1"
PRIMARY_MODEL = "joint_nv"
CONTROLS = ()
MODELS = ("m1", PRIMARY_MODEL, *CONTROLS)


@dataclass(frozen=True)
class JointNVConfig:
    dataset: str
    seed: int = 42
    window_days: int | None = None
    input_days: int = 365
    gate_shape: str = "equal"
    id_dim: int = 64
    axis_dim: int = 16
    hidden_dim: int = 32
    n_layers: int = 2
    batch_size: int = 8192
    lr: float = 5e-4
    pref_reg: float = 1e-3
    gamma_init: float = 0.01
    max_epochs: int = 100
    early_stop: int = 20
    eval_test: bool = False
    eval_holdout: bool = False
    out_dir: str = ""
    m1_checkpoint_dir: str = ""


def configure_joint_nv_run(
    dataset: str = "hm", *, short_hm: bool = True, **overrides
) -> JointNVConfig:
    dataset = dataset.lower()
    if dataset not in v3.SCHEMA:
        raise ValueError(f"알 수 없는 dataset: {dataset}")
    if short_hm and dataset != "hm":
        raise ValueError("short_hm은 H&M에서만 사용할 수 있습니다")
    suffix = "hm_w60" if short_hm else dataset
    defaults = {
        "dataset": dataset,
        "window_days": 60 if short_hm else None,
        "input_days": 14 if short_hm else 365,
        "gate_shape": "high" if dataset == "hm" else "equal",
        "out_dir": f"{v3.default_out_dir(dataset)}_m2_joint_nv_{suffix}",
        "m1_checkpoint_dir": v3.default_out_dir(dataset),
    }
    return validate_joint_nv_config(JointNVConfig(**(defaults | overrides)))


def validate_joint_nv_config(cfg: JointNVConfig) -> JointNVConfig:
    if cfg.eval_test or cfg.eval_holdout:
        raise ValueError("M2 joint N/V screening은 seed 42 validation-only입니다")
    if cfg.seed != 42:
        raise ValueError("1차 screening은 seed 42만 허용합니다")
    if cfg.dataset not in v3.SCHEMA:
        raise ValueError(f"알 수 없는 dataset: {cfg.dataset}")
    if cfg.gate_shape not in {"high", "equal", "low"}:
        raise ValueError(f"알 수 없는 gate_shape: {cfg.gate_shape}")
    if min(
        cfg.input_days,
        cfg.id_dim,
        cfg.axis_dim,
        cfg.hidden_dim,
        cfg.batch_size,
        cfg.max_epochs,
    ) <= 0:
        raise ValueError("모델·학습 크기는 양수여야 합니다")
    if cfg.n_layers < 0 or cfg.early_stop <= 0:
        raise ValueError("n_layers/early_stop 설정이 잘못됐습니다")
    if not 0.0 < cfg.gamma_init < 1.0:
        raise ValueError("gamma_init은 0과 1 사이여야 합니다")
    return cfg


def variable_validity_plan(cfg: JointNVConfig) -> dict:
    """Predeclared train-internal windows; official evaluation labels are absent."""
    cfg = validate_joint_nv_config(cfg)
    if cfg.dataset == "hm" and cfg.window_days == 60:
        return {
            "input_days": 14,
            "target_days": 7,
            "anchor_offsets": (21, 14, 7),
        }
    return {
        "input_days": 365,
        "target_days": 90,
        "anchor_offsets": (270, 180, 90),
    }


def preflight_summary(cfg: JointNVConfig) -> dict:
    cfg = validate_joint_nv_config(cfg)
    return {
        "code_version": CODE_VERSION,
        "dataset": cfg.dataset,
        "seed": cfg.seed,
        "window_days": cfg.window_days,
        "input_days": cfg.input_days,
        "models": list(MODELS),
        "architecture": "ID|N|V layer-0 concat -> one binary LightGCN -> one dot score",
        "gamma": {
            "initial_score_strength": cfg.gamma_init,
            "application": "sqrt(gamma) applied symmetrically to user and item N/V blocks",
        },
        "gate_shape": cfg.gate_shape,
        "gate_source": {
            "q_n": "train-history percentile of repeat transactions / customer age",
            "q_v": "train-history percentile of mean transaction value",
        },
        "graph_mode": "binary",
        "negative_sampling": "uniform",
        "loss": "plain_bpr",
        "separate_encoder": False,
        "frozen_or_external_base": False,
        "post_score_residual": False,
        "variable_validity": variable_validity_plan(cfg),
        "variable_validity_source": "train_internal_only",
        "eval_test": cfg.eval_test,
        "eval_holdout": cfg.eval_holdout,
        "out_dir": cfg.out_dir,
    }


def _standardized_axis(raw, masks):
    raw = np.log1p(np.maximum(np.asarray(raw, np.float32), 0.0))
    masks = np.asarray(masks, bool)
    if raw.shape != masks.shape:
        raise ValueError("axis raw와 validity mask shape이 다릅니다")
    transformed = np.zeros_like(raw)
    for column in range(raw.shape[1]):
        good = masks[:, column] & np.isfinite(raw[:, column])
        if good.any():
            mean = float(raw[good, column].mean())
            std = float(raw[good, column].std())
            transformed[good, column] = (raw[good, column] - mean) / max(std, 1e-6)
    return np.concatenate([transformed, masks.astype(np.float32)], axis=1)


def build_user_axis_inputs(snapshot, n_users: int) -> dict:
    activity_names = (
        "repeat_transaction_count",
        "transaction_recency",
        "customer_age",
        "mean_transaction_gap",
    )
    value_names = ("mean_transaction_value",)
    if len(snapshot.user_ids) != len(snapshot.numeric) or len(snapshot.numeric) != len(snapshot.valid):
        raise ValueError("snapshot 사용자·특징 크기가 다릅니다")
    if len(snapshot.user_ids) and (
        snapshot.user_ids.min() < 0 or snapshot.user_ids.max() >= n_users
    ):
        raise ValueError("snapshot user_ids가 n_users 범위를 벗어났습니다")

    candidates = candidate_variables(snapshot)
    base_valid = np.logical_and.reduce(
        [
            snapshot.valid[:, residual.NUMERIC_FEATURES.index(name)]
            for name in ("basket_count", "recency_days", "observed_days")
        ]
    )
    activity_masks = np.column_stack(
        [base_valid, base_valid, base_valid, candidates.gap_valid.to_numpy(bool)]
    )
    value_masks = candidates.value_valid.to_numpy(bool)[:, None]
    local_activity = _standardized_axis(
        candidates.loc[:, activity_names].to_numpy(np.float32), activity_masks
    )
    local_value = _standardized_axis(
        candidates.loc[:, value_names].to_numpy(np.float32), value_masks
    )
    local_n = candidates.new_n_behavior.to_numpy(np.float32)
    local_v = candidates.new_v_behavior.to_numpy(np.float32)

    activity = np.zeros((n_users, local_activity.shape[1]), np.float32)
    value = np.zeros((n_users, local_value.shape[1]), np.float32)
    n_behavior_score = np.zeros(n_users, np.float32)
    v_behavior_score = np.zeros(n_users, np.float32)
    repeat_transaction_count = np.zeros(n_users, np.float32)
    repeat_transaction_rate = np.zeros(n_users, np.float32)
    transaction_recency = np.zeros(n_users, np.float32)
    customer_age = np.zeros(n_users, np.float32)
    mean_transaction_value = np.zeros(n_users, np.float32)
    valid_user = np.zeros(n_users, bool)
    ids = np.asarray(snapshot.user_ids, np.int64)
    activity[ids] = local_activity
    value[ids] = local_value
    n_behavior_score[ids] = local_n
    v_behavior_score[ids] = local_v
    repeat_transaction_count[ids] = candidates.repeat_transaction_count
    repeat_transaction_rate[ids] = candidates.repeat_transaction_rate
    transaction_recency[ids] = candidates.transaction_recency
    customer_age[ids] = candidates.customer_age
    mean_transaction_value[ids] = candidates.mean_transaction_value
    valid_user[ids] = True
    q_n, q_v = fixed_percentile_ranks(
        n_behavior_score, v_behavior_score, valid_user
    )
    return {
        "activity": activity,
        "value": value,
        "n_behavior_score": n_behavior_score,
        "v_behavior_score": v_behavior_score,
        "clv_proxy": n_behavior_score * v_behavior_score,
        "q_n": q_n,
        "q_v": q_v,
        "valid_user": valid_user,
        "repeat_transaction_count": repeat_transaction_count,
        "repeat_transaction_rate": repeat_transaction_rate,
        "transaction_recency": transaction_recency,
        "customer_age": customer_age,
        "mean_transaction_value": mean_transaction_value,
        "activity_names": activity_names + tuple(f"valid_{name}" for name in activity_names),
        "value_names": value_names + tuple(f"valid_{name}" for name in value_names),
    }


def result_row(model_id, role, gate_shape, seed, metrics, diagnostics=None) -> dict:
    return {
        "seed": int(seed),
        "model_id": model_id,
        "split": "val",
        "gate_shape": gate_shape,
        "role": role,
        **(diagnostics or {}),
        **metrics,
    }


def _base_config(cfg: JointNVConfig) -> dict:
    configured = v3.configure_run(
        cfg.dataset,
        out_dir=cfg.m1_checkpoint_dir,
        ARCH="pref_only",
        SEED_LIST=[cfg.seed],
        WINDOW_DAYS=cfg.window_days,
        EVAL_TEST=False,
        EVAL_HOLDOUT=False,
        GRAPH_MODE="binary",
        LOSS_MODE="plain",
        NEG_MODE="uniform",
        DIM=cfg.id_dim,
        N_LAYERS=cfg.n_layers,
        BATCH_SIZE=cfg.batch_size,
        LR=cfg.lr,
        PREF_REG=cfg.pref_reg,
        EPOCHS=cfg.max_epochs,
        EARLY_STOP=cfg.early_stop,
    )
    base = dict(configured)
    required = {
        "ARCH": "pref_only",
        "GRAPH_MODE": "binary",
        "LOSS_MODE": "plain",
        "NEG_MODE": "uniform",
        "EVAL_TEST": False,
        "EVAL_HOLDOUT": False,
    }
    for key, expected in required.items():
        if base[key] != expected:
            raise RuntimeError(f"M2 실험 설정 오염: {key}={base[key]!r}")
    return base


def _config_hash(cfg: JointNVConfig, input_hash: str, revision: str) -> str:
    payload = {"config": asdict(cfg), "input_hash": input_hash, "source": revision}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def _progress_store(root, stage, cfg, config_hash, input_hash, revision):
    return ProgressStore(
        Path(root) / "progress" / config_hash,
        RunIdentity(
            stage=stage,
            model_id=stage,
            seed=cfg.seed,
            config_hash=config_hash,
            source_revision=revision,
            input_hash=input_hash,
        ),
    )


def _evaluate(model, prepared, per_user=True):
    return moe._flat_evaluation(
        model,
        0.0,
        prepared["cache"],
        prepared["meta"],
        prepared["data"],
        prepared["base_cfg"],
        per_user=per_user,
    )


def _prepare(cfg: JointNVConfig) -> dict:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = moe.build_input_manifest(v3.SCHEMA[cfg.dataset])
    input_hash = moe.manifest_hash(manifest)
    revision = moe.source_revision()
    config_hash = _config_hash(cfg, input_hash, revision)
    base_cfg = _base_config(cfg)
    data = v3.prepare_data(base_cfg, v3.DCFG)
    data["loss_w"] = None
    validity_plan = variable_validity_plan(cfg)
    validity_anchors = residual.build_anchor_examples(
        data["train"],
        data["n_users"],
        v3.DCFG["is_date"],
        **validity_plan,
    )
    validity_reports = [
        validate_anchor(
            anchor,
            dataset=cfg.dataset,
            anchor_label=f"train_internal_T-{anchor.offset_days}",
        )
        for anchor in validity_anchors.anchors
    ]
    variable_validity = {
        "metrics": pd.concat(
            [report["metrics"] for report in validity_reports], ignore_index=True
        ),
        "quadrants": pd.concat(
            [report["quadrants"] for report in validity_reports], ignore_index=True
        ),
    }
    snapshot = residual.build_final_snapshot(
        data["train"], data["n_users"], v3.DCFG["is_date"], cfg.input_days
    )
    axes = build_user_axis_inputs(snapshot, data["n_users"])
    item_profile = build_dual_item_profiles(
        data["train"], data["n_items"], v3.DCFG["is_date"]
    )
    meta = v3.item_meta(data["train"], data["n_items"])
    thresholds = v3.segment_thresholds(axes["clv_proxy"], base_cfg["SEG_EDGES"])
    cache = v3.EvalCache(
        *data["splits"]["val"], axes["clv_proxy"], thresholds, data["n_items"]
    )
    x_item, item_cat = v3.item_value_features(data["train"], data["n_items"])
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
        "meta": meta,
        "cache": cache,
        "x_item": x_item,
        "item_cat": item_cat,
        "variable_validity": variable_validity,
    }


def _train_m1(prepared, cfg):
    data, base_cfg = prepared["data"], prepared["base_cfg"]
    gate = torch.ones(data["n_users"], device=v3.DEVICE)
    store = _progress_store(
        prepared["out_dir"], "m1", cfg, prepared["config_hash"],
        prepared["input_hash"], prepared["revision"]
    )
    model, training = v3.get_or_train(
        "pref_only",
        cfg.seed,
        data,
        gate,
        data["x_val_u"],
        prepared["x_item"],
        prepared["item_cat"],
        prepared["meta"],
        prepared["cache"],
        base_cfg,
        progress_store=store,
    )
    model.eval()
    metrics, per_user = _evaluate(model, prepared)
    store.mark_complete(best_metric=float(metrics["recall@10"]))
    return model, training, metrics, per_user


def _build_model(prepared, cfg, variant):
    data, axes = prepared["data"], prepared["axes"]
    return JointNVLightGCN(
        n_users=data["n_users"],
        n_items=data["n_items"],
        user_activity=axes["activity"],
        user_value=axes["value"],
        item_profile=prepared["item_profile"],
        q_n=axes["q_n"],
        q_v=axes["q_v"],
        adj=data["adj"],
        id_dim=cfg.id_dim,
        axis_dim=cfg.axis_dim,
        hidden_dim=cfg.hidden_dim,
        n_layers=cfg.n_layers,
        variant=variant,
        gate_shape=cfg.gate_shape,
        shuffle_seed=cfg.seed,
        pref_reg=cfg.pref_reg,
        gamma_init=cfg.gamma_init,
    ).to(v3.DEVICE)


def _train_variant(prepared, cfg, variant):
    v3.set_seed(cfg.seed)
    model = _build_model(prepared, cfg, variant)
    store = _progress_store(
        prepared["out_dir"], variant, cfg, prepared["config_hash"],
        prepared["input_hash"], prepared["revision"]
    )
    gate = torch.ones(prepared["data"]["n_users"], device=v3.DEVICE)
    training = v3.train_phase(
        model,
        list(model.parameters()),
        prepared["data"],
        gate,
        0.0,
        prepared["base_cfg"],
        cfg.seed,
        variant,
        prepared["cache"],
        prepared["meta"],
        progress_store=store,
    )
    model.eval()
    metrics, per_user = _evaluate(model, prepared)
    diagnostics = model.score_diagnostics(seed=cfg.seed)
    checkpoint = prepared["out_dir"] / (
        f"{variant}_{cfg.dataset}_s{cfg.seed}_{prepared['config_hash']}.pt"
    )
    torch.save(
        {
            "state": model.state_dict(),
            "training": training,
            "diagnostics": diagnostics,
            "config": asdict(cfg),
            "source_revision": prepared["revision"],
            "input_hash": prepared["input_hash"],
        },
        checkpoint,
    )
    store.mark_complete(
        best_metric=float(metrics["recall@10"]), checkpoint_path=str(checkpoint)
    )
    return {
        "model": model,
        "training": training,
        "metrics": metrics,
        "per_user": per_user,
        "diagnostics": diagnostics,
        "checkpoint": str(checkpoint),
    }


def _decision(rows, baseline):
    table = pd.DataFrame(rows).set_index("model_id")
    main = table.loc[PRIMARY_MODEL]
    accuracy = {}
    for metric in ("recall", "ndcg"):
        for k in (10, 20, 50):
            key = f"{metric}@{k}"
            accuracy[key] = float(main[key] / max(float(baseline[key]), 1e-12))
    economic_improved = bool(main["revenue@10"] > baseline["revenue@10"])
    controls = {
        name: bool(main["revenue@10"] > table.loc[name, "revenue@10"])
        for name in CONTROLS
    }
    return {
        "success": bool(economic_improved and min(accuracy.values()) >= 0.99 and all(controls.values())),
        "economic_improved_vs_m1": economic_improved,
        "accuracy_ratios_vs_m1": accuracy,
        "control_dominance": controls,
        "note": "이 조건은 성과를 만드는 학습 제약이 아니라, validation 결과를 판독하는 사후 기준입니다.",
    }


def _persist(prepared, cfg, rows, baseline_per_user, runs, decision):
    frame = pd.DataFrame(rows)
    delta_rows = []
    for name, run in runs.items():
        for metric in ("recall", "ndcg", "revenue", "arp"):
            difference = run["per_user"][metric] - baseline_per_user[metric]
            delta_rows.append(
                {
                    "model_id": name,
                    "split": "val",
                    "metric": metric,
                    **v3.paired_bootstrap([difference], prepared["base_cfg"]["N_BOOT"]),
                }
            )
    stem = f"m2_joint_nv_{cfg.dataset}_{prepared['config_hash']}"
    csv_path = prepared["out_dir"] / f"{stem}.csv"
    delta_path = prepared["out_dir"] / f"{stem}_delta.csv"
    json_path = prepared["out_dir"] / f"{stem}.json"
    validity_path = prepared["out_dir"] / f"{stem}_variable_validity.csv"
    quadrant_path = prepared["out_dir"] / f"{stem}_variable_quadrants.csv"
    frame.to_csv(csv_path, index=False, float_format="%.8f")
    pd.DataFrame(delta_rows).to_csv(delta_path, index=False)
    prepared["variable_validity"]["metrics"].to_csv(validity_path, index=False)
    prepared["variable_validity"]["quadrants"].to_csv(quadrant_path, index=False)
    payload = {
        "code_version": CODE_VERSION,
        "source_revision": prepared["revision"],
        "input_manifest": prepared["manifest"],
        "config": asdict(cfg),
        "preflight": preflight_summary(cfg),
        "data_stats": prepared["data"].get("data_stats", {}),
        "feature_schema": {
            "user_activity": list(prepared["axes"]["activity_names"]),
            "user_value": list(prepared["axes"]["value_names"]),
            "item_activity": list(prepared["item_profile"].activity_names),
            "item_value": list(prepared["item_profile"].value_names),
        },
        "decision": decision,
        "training": {name: run["training"] for name, run in runs.items()},
        "diagnostics": {name: run["diagnostics"] for name, run in runs.items()},
        "checkpoints": {name: run["checkpoint"] for name, run in runs.items()},
        "absolute_rows": frame.to_dict("records"),
        "paired_delta": delta_rows,
        "variable_validity": {
            "source": "train_internal_only",
            "metrics": prepared["variable_validity"]["metrics"].to_dict("records"),
            "quadrants": prepared["variable_validity"]["quadrants"].to_dict("records"),
        },
        "interpretation": {
            "clv": "historical N×V CLV proxy used as conditional representation inputs",
            "revenue": "price/purchase-amount weighted hit, not incremental revenue",
            "item": "item activity/economic attributes, not item CLV",
        },
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    frame.attrs["decision"] = decision
    frame.attrs["result_paths"] = {
        "csv": str(csv_path),
        "delta_csv": str(delta_path),
        "variable_validity_csv": str(validity_path),
        "variable_quadrants_csv": str(quadrant_path),
        "json": str(json_path),
    }
    return frame


def run_experiment(cfg: JointNVConfig | None = None) -> pd.DataFrame:
    cfg = validate_joint_nv_config(cfg or configure_joint_nv_run("hm", short_hm=True))
    print(json.dumps(preflight_summary(cfg), ensure_ascii=False, indent=2))
    prepared = _prepare(cfg)
    _, m1_training, baseline, baseline_per_user = _train_m1(prepared, cfg)
    rows = [result_row("m1", "baseline", "none", cfg.seed, baseline)]
    runs = {}
    for model_id in (PRIMARY_MODEL, *CONTROLS):
        runs[model_id] = _train_variant(prepared, cfg, model_id)
        rows.append(
            result_row(
                model_id,
                "model" if model_id == PRIMARY_MODEL else "control",
                cfg.gate_shape if model_id != "joint_constant_user" else "equal",
                cfg.seed,
                runs[model_id]["metrics"],
                runs[model_id]["diagnostics"],
            )
        )
    decision = _decision(rows, baseline)
    frame = _persist(prepared, cfg, rows, baseline_per_user, runs, decision)
    frame.attrs["m1_training"] = m1_training
    print("최종 validation 판정:", decision)
    print("결과 파일:", frame.attrs["result_paths"])
    return frame


def run_two_dataset_screening() -> dict[str, pd.DataFrame]:
    """Run the cheapest informative comparison before any control ablations."""
    presets = {
        "hm_w60": configure_joint_nv_run("hm", short_hm=True),
        "dunnhumby_full": configure_joint_nv_run("dunnhumby", short_hm=False),
    }
    results = {}
    for label, cfg in presets.items():
        print(f"\n{'=' * 84}\n{label}: M1 vs joint_nv validation 시작\n{'=' * 84}")
        results[label] = run_experiment(cfg)
        print(f"{label}: 완료")
    return results


def _matching_result_payload(out_dir: Path, checkpoint: Path) -> dict | None:
    for path in sorted(out_dir.glob("m2_joint_nv_*.json"), reverse=True):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        recorded = payload.get("checkpoints", {}).get(PRIMARY_MODEL)
        if recorded and Path(recorded).name == checkpoint.name:
            payload["_result_json"] = str(path)
            return payload
    return None


def run_checkpoint_diagnostics(
    cfg: JointNVConfig | None = None, *, checkpoint_path: str | None = None
) -> pd.DataFrame:
    """Evaluate an existing joint checkpoint without any model training."""
    cfg = validate_joint_nv_config(cfg or configure_joint_nv_run("hm", short_hm=True))
    print(json.dumps({
        **preflight_summary(cfg),
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "training": False,
    }, ensure_ascii=False, indent=2))
    prepared = _prepare(cfg)
    checkpoint = (
        Path(checkpoint_path)
        if checkpoint_path
        else find_joint_checkpoint(
            prepared["out_dir"], dataset=cfg.dataset, seed=cfg.seed
        )
    )
    model = _build_model(prepared, cfg, PRIMARY_MODEL)
    checkpoint_payload = load_joint_checkpoint(
        model,
        checkpoint,
        dataset=cfg.dataset,
        seed=cfg.seed,
        input_hash=prepared["input_hash"],
    )

    def evaluator(view):
        metrics, _ = _evaluate(view, prepared, per_user=False)
        return metrics

    block_metrics = evaluate_block_views(model, evaluator)
    rows = [
        {
            "dataset": cfg.dataset,
            "seed": cfg.seed,
            "view": view,
            **metrics,
        }
        for view, metrics in block_metrics.items()
    ]
    frame = pd.DataFrame(rows)
    score_summary = sampled_block_score_summary(model, seed=cfg.seed)
    axis_summary = axis_distribution_diagnostics(
        prepared["axes"]["n_behavior_score"],
        prepared["axes"]["v_behavior_score"],
        prepared["axes"]["valid_user"],
    )
    source_result = _matching_result_payload(prepared["out_dir"], checkpoint)
    baseline_row = None
    if source_result:
        baseline_row = next(
            (
                row for row in source_result.get("absolute_rows", [])
                if row.get("model_id") == "m1"
            ),
            None,
        )

    diagnostic_dir = prepared["out_dir"] / "checkpoint_diagnostics"
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    stem = f"joint_nv_blocks_{cfg.dataset}_s{cfg.seed}_{checkpoint.stem[-12:]}"
    metrics_path = diagnostic_dir / f"{stem}.csv"
    validity_path = diagnostic_dir / f"{stem}_variable_validity.csv"
    json_path = diagnostic_dir / f"{stem}.json"
    frame.to_csv(metrics_path, index=False, float_format="%.8f")
    prepared["variable_validity"]["metrics"].to_csv(validity_path, index=False)
    output = {
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "training_performed": False,
        "checkpoint": str(checkpoint),
        "checkpoint_source_revision": checkpoint_payload.get("source_revision"),
        "current_source_revision": prepared["revision"],
        "input_hash": prepared["input_hash"],
        "config": asdict(cfg),
        "gamma_application": "sqrt(gamma) on both user and item; score multiplier is gamma",
        "block_score_summary": score_summary,
        "axis_distribution": axis_summary,
        "block_metrics": rows,
        "external_m1_from_original_result": baseline_row,
        "source_result_json": source_result.get("_result_json") if source_result else None,
        "variable_validity": prepared["variable_validity"]["metrics"].to_dict("records"),
        "interpretation_guard": {
            "gamma": "gamma alone is not effective strength; use block score std ratios",
            "v_validity": "future mean transaction value correlation is conditional on future buyers",
            "revenue": "price/purchase-amount weighted hit, not incremental revenue",
        },
    }
    json_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    frame.attrs["checkpoint"] = str(checkpoint)
    frame.attrs["block_score_summary"] = score_summary
    frame.attrs["axis_distribution"] = axis_summary
    frame.attrs["external_m1"] = baseline_row
    frame.attrs["result_paths"] = {
        "metrics_csv": str(metrics_path),
        "variable_validity_csv": str(validity_path),
        "json": str(json_path),
    }
    print("체크포인트 진단 완료 (재학습 없음)")
    print("블록 실효강도:", score_summary)
    print("축 분포:", axis_summary)
    print("결과 파일:", frame.attrs["result_paths"])
    return frame


def main_cli():
    print(json.dumps(preflight_summary(configure_joint_nv_run("hm", short_hm=True)), ensure_ascii=False, indent=2))
    print("학습은 Colab notebook에서 시작하세요.")


if __name__ == "__main__":
    main_cli()
