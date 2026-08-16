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
from clv_joint_nv_model import JointNVLightGCN
from clv_run_state import ProgressStore, RunIdentity
import lightgcn_clv_moe as moe
import lightgcn_clv_v3 as v3


CODE_VERSION = "m2-joint-nv-lightgcn-v1.0"
PRIMARY_MODEL = "joint_nv"
CONTROLS = ()
MODELS = ("m1", PRIMARY_MODEL, *CONTROLS)


@dataclass(frozen=True)
class JointNVConfig:
    dataset: str
    seed: int = 42
    window_days: int | None = None
    gate_shape: str = "equal"
    id_dim: int = 64
    axis_dim: int = 16
    hidden_dim: int = 32
    n_layers: int = 2
    batch_size: int = 8192
    lr: float = 5e-4
    pref_reg: float = 1e-3
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
    if min(cfg.id_dim, cfg.axis_dim, cfg.hidden_dim, cfg.batch_size, cfg.max_epochs) <= 0:
        raise ValueError("모델·학습 크기는 양수여야 합니다")
    if cfg.n_layers < 0 or cfg.early_stop <= 0:
        raise ValueError("n_layers/early_stop 설정이 잘못됐습니다")
    return cfg


def preflight_summary(cfg: JointNVConfig) -> dict:
    cfg = validate_joint_nv_config(cfg)
    return {
        "code_version": CODE_VERSION,
        "dataset": cfg.dataset,
        "seed": cfg.seed,
        "window_days": cfg.window_days,
        "models": list(MODELS),
        "architecture": "ID|N|V layer-0 concat -> one binary LightGCN -> one dot score",
        "gate_shape": cfg.gate_shape,
        "graph_mode": "binary",
        "negative_sampling": "uniform",
        "loss": "plain_bpr",
        "separate_encoder": False,
        "frozen_or_external_base": False,
        "post_score_residual": False,
        "eval_test": cfg.eval_test,
        "eval_holdout": cfg.eval_holdout,
        "out_dir": cfg.out_dir,
    }


def build_user_axis_inputs(x_val_u, valid_user) -> dict:
    values = np.asarray(x_val_u, dtype=np.float32)
    valid = np.asarray(valid_user, dtype=bool)
    if values.ndim != 2 or values.shape[1] != 5 or valid.shape != (len(values),):
        raise ValueError("사용자 입력은 [F_p,T_p,R_p,AOV_p,Prem_p] 5차원이어야 합니다")
    if not np.isfinite(values).all():
        raise ValueError("사용자 입력은 유한해야 합니다")
    activity = values[:, :3].copy()
    value = values[:, 3:].copy()
    n_hat = activity.mean(1)
    v_hat = value.mean(1)
    q_n, q_v = fixed_percentile_ranks(n_hat, v_hat, valid)
    return {
        "activity": activity,
        "value": value,
        "n_hat": n_hat,
        "v_hat": v_hat,
        "clv_proxy": n_hat * v_hat,
        "q_n": q_n,
        "q_v": q_v,
        "valid_user": valid,
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
        Path(root) / "progress",
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
    valid_user = np.isfinite(data["clv"])
    axes = build_user_axis_inputs(data["x_val_u"], valid_user)
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
    frame.to_csv(csv_path, index=False, float_format="%.8f")
    pd.DataFrame(delta_rows).to_csv(delta_path, index=False)
    payload = {
        "code_version": CODE_VERSION,
        "source_revision": prepared["revision"],
        "input_manifest": prepared["manifest"],
        "config": asdict(cfg),
        "preflight": preflight_summary(cfg),
        "data_stats": prepared["data"].get("data_stats", {}),
        "feature_schema": {
            "user_activity": ["F_p", "T_p", "R_p"],
            "user_value": ["AOV_p", "Prem_p"],
            "item_activity": list(prepared["item_profile"].activity_names),
            "item_value": list(prepared["item_profile"].value_names),
        },
        "decision": decision,
        "training": {name: run["training"] for name, run in runs.items()},
        "diagnostics": {name: run["diagnostics"] for name, run in runs.items()},
        "checkpoints": {name: run["checkpoint"] for name, run in runs.items()},
        "absolute_rows": frame.to_dict("records"),
        "paired_delta": delta_rows,
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
        "csv": str(csv_path), "delta_csv": str(delta_path), "json": str(json_path)
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


def main_cli():
    print(json.dumps(preflight_summary(configure_joint_nv_run("hm", short_hm=True)), ensure_ascii=False, indent=2))
    print("학습은 Colab notebook에서 시작하세요.")


if __name__ == "__main__":
    main_cli()
