"""Seed-42 historical screen for a gate-free low-dimensional M2.

Only the new M2 arm is trained.  The already-computed M1@64 historical
development result is reused after a fail-closed protocol and input-manifest
identity check.  This is exploratory development, not another test look.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from clv_gatefree_lowdim_model import GateFreeLowDimNVLightGCN
from clv_run_state import ProgressStore, RunIdentity, clone_state, file_sha256
import lightgcn_clv_axis_specific_test10 as test10
import lightgcn_clv_joint_nv as joint
import lightgcn_clv_moe as moe
import lightgcn_clv_residual as residual
import lightgcn_clv_v3 as v3


CODE_VERSION = "m2-gatefree-lowdim-historical-screen-v1"
MODEL_ID = "m2_gatefree_lowdim"
ALLOWED_AXIS_BUDGETS = {0.05, 0.1}


@dataclass(frozen=True)
class GateFreeLowDimConfig:
    dataset: str = "dunnhumby"
    seed: int = 42
    time_cutoff: int = 690
    evaluation_days: int = 7
    epochs: int = 100
    id_dim: int = 64
    axis_dim: int = 4
    hidden_dim: int = 8
    axis_budget: float = 0.1
    n_layers: int = 2
    batch_size: int = 8192
    lr: float = 5e-4
    pref_reg: float = 1e-3
    input_days: int = 365
    out_dir: str = ""
    baseline_result_dir: str = ""


def configure_gatefree_lowdim_run(**overrides) -> GateFreeLowDimConfig:
    defaults = {
        "out_dir": (
            f"{v3.default_out_dir('dunnhumby')}"
            "_m2_gatefree_lowdim_historical_screen_v1"
        ),
        "baseline_result_dir": (
            f"{v3.default_out_dir('dunnhumby')}"
            "_m2_repeatshare_historical_backtest_v1"
        ),
    }
    return validate_config(GateFreeLowDimConfig(**(defaults | overrides)))


def validate_config(cfg: GateFreeLowDimConfig) -> GateFreeLowDimConfig:
    required = {
        "dataset": "dunnhumby",
        "seed": 42,
        "time_cutoff": 690,
        "evaluation_days": 7,
        "epochs": 100,
        "id_dim": 64,
        "axis_dim": 4,
        "hidden_dim": 8,
        "n_layers": 2,
        "input_days": 365,
    }
    for key, expected in required.items():
        if getattr(cfg, key) != expected:
            raise ValueError(f"빠른 M2 screen은 {key}={expected!r}이어야 합니다")
    if cfg.axis_budget not in ALLOWED_AXIS_BUDGETS:
        raise ValueError(
            "축별 고정 계수는 사전 선언된 0.05 또는 0.1이어야 합니다"
        )
    if cfg.batch_size <= 0 or cfg.lr <= 0 or cfg.pref_reg < 0:
        raise ValueError("학습 설정이 잘못됐습니다")
    if not cfg.out_dir or not cfg.baseline_result_dir:
        raise ValueError("out_dir와 baseline_result_dir가 필요합니다")
    return cfg


def preflight_summary(cfg: GateFreeLowDimConfig) -> dict:
    cfg = validate_config(cfg)
    return {
        "code_version": CODE_VERSION,
        "dataset": cfg.dataset,
        "seed": cfg.seed,
        "trained_models": [MODEL_ID],
        "reused_comparator": "m1_64",
        "historical_development_split": {
            "train_end_inclusive": 683,
            "evaluation_start_inclusive": 684,
            "evaluation_end_inclusive": 690,
        },
        "m2": {
            "architecture": "ID(64)|activity(4)|transaction-value(4)",
            "id_dim": cfg.id_dim,
            "activity_dim": cfg.axis_dim,
            "transaction_value_dim": cfg.axis_dim,
            "explicit_item_features": False,
            "item_response": "learned from item ID by the same recommendation loss",
            "user_gate": False,
            "learned_axis_weight": False,
            "fixed_per_axis_budget": cfg.axis_budget,
            "fixed_total_clv_budget": 2.0 * cfg.axis_budget,
            "score_formula": (
                "S_ID + axis_budget*S_N + axis_budget*S_V"
            ),
            "train_user_coordinate_mean": 0.0,
        },
        "fixed": {
            "graph": "binary",
            "negative_sampling": "uniform",
            "sample_weighting": False,
            "loss": "one plain pairwise BPR; no added loss",
            "min_user_interactions": 1,
            "min_item_interactions": 1,
            "epochs": cfg.epochs,
            "validation_or_epoch_selection": False,
        },
        "interpretation": (
            "exploratory historical development screen; if positive, run "
            "M1@72 capacity and shuffled-user controls before attribution"
        ),
        "out_dir": cfg.out_dir,
        "baseline_result_dir": cfg.baseline_result_dir,
    }


def _base_config(cfg: GateFreeLowDimConfig) -> dict:
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
        "TIME_CUTOFF": 690,
        "TRAIN_ON_VAL": True,
        "TEST_DAYS": 7,
        "HOLDOUT_DAYS": 0,
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
            raise RuntimeError(f"M2 historical screen 설정 오염: {key}={base[key]!r}")
    return base


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def _normalise_metric_names(row: dict) -> dict:
    aliases = {
        "revenue": "price_purchase_amount_weighted_hit",
        "arp": "mean_recommended_price_percentile",
        "value_alignment": "user_value_tendency_recommended_price_alignment",
    }
    normalised = {}
    for key, value in row.items():
        if "@" in key:
            name, suffix = key.split("@", 1)
            key = f"{aliases.get(name, name)}@{suffix}"
        else:
            key = aliases.get(key, key)
        normalised[key] = value
    return normalised


def _load_compatible_baseline(
    cfg: GateFreeLowDimConfig, current_manifest: list[dict]
) -> dict:
    root = Path(cfg.baseline_result_dir)
    candidates = sorted(root.glob("m2_repeatshare_backtest_*.json"))
    expected_split = {
        "train_end_inclusive": 683,
        "evaluation_start_inclusive": 684,
        "evaluation_end_inclusive": 690,
        "original_validation_test_holdout_constructed": False,
    }
    compatible = []
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            config = payload["config"]
            fixed = payload["preflight"]["fixed"]
            split = payload["preflight"]["historical_development_split"]
            protocol_ok = (
                config.get("dataset") == "dunnhumby"
                and config.get("seed") == 42
                and config.get("time_cutoff") == 690
                and config.get("evaluation_days") == 7
                and config.get("epochs") == 100
                and config.get("id_dim") == 64
                and config.get("n_layers") == 2
                and config.get("batch_size") == 8192
                and config.get("lr") == 5e-4
                and config.get("pref_reg") == 1e-3
                and split == expected_split
                and fixed.get("graph") == "binary"
                and fixed.get("negative_sampling") == "uniform"
                and fixed.get("sample_weighting") is False
                and fixed.get("validation_or_epoch_selection") is False
                and _canonical(payload.get("input_manifest"))
                == _canonical(current_manifest)
            )
            rows = [
                row
                for row in payload.get("absolute_rows", [])
                if row.get("model_id") == "m1_64"
                and row.get("seed") == 42
                and row.get("final_epoch") == 100
                and row.get("split") == "historical_development_days_684_690"
            ]
            if protocol_ok and len(rows) == 1:
                row = _normalise_metric_names(rows[0])
                row["source_result"] = str(path)
                compatible.append(row)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    if not compatible:
        raise RuntimeError(
            "호환되는 M1 historical baseline을 찾지 못했습니다. "
            "split·seed·100 epoch·입력 manifest가 모두 같아야 합니다."
        )
    metric_snapshots = {
        _canonical(
            {
                key: value
                for key, value in row.items()
                if key not in {"source_result"}
            }
        )
        for row in compatible
    }
    if len(metric_snapshots) != 1:
        raise RuntimeError("호환 baseline 파일끼리 결과가 달라 재사용할 수 없습니다")
    return compatible[-1]


def _config_hash(cfg: GateFreeLowDimConfig, input_hash: str, revision: str) -> str:
    payload = {
        "code_version": CODE_VERSION,
        "config": asdict(cfg),
        "model_id": MODEL_ID,
        "input_hash": input_hash,
        "source_revision": revision,
    }
    return hashlib.sha256(_canonical(payload).encode()).hexdigest()[:12]


def _prepare(cfg: GateFreeLowDimConfig) -> dict:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = moe.build_input_manifest(v3.SCHEMA[cfg.dataset])
    input_hash = moe.manifest_hash(manifest)
    revision = moe.source_revision()
    base_cfg = _base_config(cfg)
    data = v3.prepare_data(base_cfg, v3.DCFG)
    if set(data["splits"]) != {"test"}:
        raise RuntimeError(f"historical 개발분할 외 오염: {sorted(data['splits'])}")
    if float(data["train"].t.max()) != 683.0:
        raise RuntimeError(f"historical train 종료일 오류: {data['train'].t.max()}")
    if data.get("loss_w") is not None:
        raise RuntimeError("M2 screen에 M4 표본 가중치가 섞였습니다")
    data["loss_w"] = None
    snapshot = residual.build_final_snapshot(
        data["train"], data["n_users"], v3.DCFG["is_date"], cfg.input_days
    )
    axes = joint.build_user_axis_inputs(snapshot, data["n_users"])
    baseline = _load_compatible_baseline(cfg, manifest)
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
        "baseline": baseline,
        "meta": meta,
        "cache": cache,
    }
    prepared["config_hash"] = _config_hash(cfg, input_hash, revision)
    return prepared


def _build_model(prepared: dict, cfg: GateFreeLowDimConfig):
    data, axes = prepared["data"], prepared["axes"]
    v3.set_seed(cfg.seed)
    model = GateFreeLowDimNVLightGCN(
        n_users=data["n_users"],
        n_items=data["n_items"],
        user_activity=axes["activity"],
        user_value=axes["value"],
        user_activity_valid=axes["activity_valid"],
        user_value_valid=axes["value_valid"],
        q_n=axes["q_n"],
        q_v=axes["q_v"],
        adj=data["adj"],
        id_dim=cfg.id_dim,
        axis_dim=cfg.axis_dim,
        hidden_dim=cfg.hidden_dim,
        n_layers=cfg.n_layers,
        axis_budget=cfg.axis_budget,
        pref_reg=cfg.pref_reg,
    ).to(v3.DEVICE)
    return model, list(model.parameters())


def _arm_hash(prepared: dict, cfg: GateFreeLowDimConfig) -> str:
    return hashlib.sha256(
        _canonical(
            {
                "run": prepared["config_hash"],
                "model_id": MODEL_ID,
                "seed": cfg.seed,
            }
        ).encode()
    ).hexdigest()[:12]


def _run_model(prepared: dict, cfg: GateFreeLowDimConfig) -> dict:
    root = prepared["out_dir"] / "arms" / prepared["config_hash"]
    result_path = root / f"{MODEL_ID}_s{cfg.seed}.json"
    checkpoint_path = root / f"{MODEL_ID}_s{cfg.seed}.pt"
    if result_path.exists():
        print("  [cached] 새 M2 seed 42 완료 결과 재사용")
        return json.loads(result_path.read_text(encoding="utf-8"))
    model, params = _build_model(prepared, cfg)
    store = ProgressStore(
        prepared["out_dir"] / "progress" / prepared["config_hash"],
        RunIdentity(
            stage="historical_development_train",
            model_id=MODEL_ID,
            seed=cfg.seed,
            config_hash=_arm_hash(prepared, cfg),
            source_revision=prepared["revision"],
            input_hash=prepared["input_hash"],
        ),
    )
    training = test10._fixed_epoch_train(
        model, params, prepared, cfg, MODEL_ID, cfg.seed, store
    )
    model.eval()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint_path.with_suffix(".pt.tmp")
    torch.save(
        {
            "state": clone_state(model),
            "model_id": MODEL_ID,
            "config": asdict(cfg),
            "training": training,
            "source_revision": prepared["revision"],
            "input_hash": prepared["input_hash"],
        },
        temporary,
    )
    os.replace(temporary, checkpoint_path)
    metrics, _ = moe._flat_evaluation(
        model,
        0.0,
        prepared["cache"],
        prepared["meta"],
        prepared["data"],
        prepared["base_cfg"],
        per_user=False,
    )
    payload = {
        "model_id": MODEL_ID,
        "role": "model",
        "seed": cfg.seed,
        "split": "historical_development_days_684_690",
        "final_epoch": cfg.epochs,
        "metrics": test10._public_metrics(metrics),
        "diagnostics": model.representation_diagnostics(),
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


def _comparison(frame: pd.DataFrame) -> pd.DataFrame:
    indexed = frame.set_index("model_id")
    metadata = {
        "model_id",
        "role",
        "seed",
        "split",
        "final_epoch",
        "source_result",
        "axis_budget",
        "total_dim",
        "activity_user_coordinate_mean_abs",
        "value_user_coordinate_mean_abs",
        "mean_user_norm",
        "mean_item_norm",
    }
    rows = []
    for metric in frame.columns:
        if metric in metadata:
            continue
        baseline = indexed.at["m1_64", metric]
        model = indexed.at[MODEL_ID, metric]
        if not isinstance(baseline, (int, float, np.number)) or not isinstance(
            model, (int, float, np.number)
        ):
            continue
        rows.append(
            {
                "metric": metric,
                "m1_64": baseline,
                MODEL_ID: model,
                "absolute_delta": model - baseline,
                "relative_change_pct": (
                    100.0 * (model - baseline) / baseline if baseline != 0 else None
                ),
            }
        )
    return pd.DataFrame(rows)


def run_gatefree_lowdim_screen(
    cfg: GateFreeLowDimConfig | None = None,
) -> pd.DataFrame:
    cfg = validate_config(cfg or configure_gatefree_lowdim_run())
    print(json.dumps(preflight_summary(cfg), ensure_ascii=False, indent=2))
    prepared = _prepare(cfg)
    print("\n===== 새 M2만 학습 | seed 42 | fixed 100 epochs =====")
    arm = _run_model(prepared, cfg)
    baseline = dict(prepared["baseline"])
    baseline["role"] = "reused_baseline"
    rows = [baseline]
    rows.append(
        {
            "model_id": arm["model_id"],
            "role": arm["role"],
            "seed": arm["seed"],
            "split": arm["split"],
            "final_epoch": arm["final_epoch"],
            **arm["diagnostics"],
            **arm["metrics"],
        }
    )
    frame = pd.DataFrame(rows)
    comparison = _comparison(frame)
    stem = f"m2_gatefree_lowdim_{prepared['config_hash']}"
    paths = {
        "absolute_csv": prepared["out_dir"] / f"{stem}.csv",
        "comparison_csv": prepared["out_dir"] / f"{stem}_comparison.csv",
        "json": prepared["out_dir"] / f"{stem}.json",
    }
    test10._atomic_csv(paths["absolute_csv"], frame)
    test10._atomic_csv(paths["comparison_csv"], comparison)
    metric_index = comparison.set_index("metric")
    accuracy_names = [
        "recall@10",
        "ndcg@10",
        "recall@20",
        "ndcg@20",
        "recall@50",
        "ndcg@50",
    ]
    accuracy_ratios = {
        metric: float(metric_index.at[metric, MODEL_ID])
        / float(metric_index.at[metric, "m1_64"])
        for metric in accuracy_names
        if metric in metric_index.index
    }
    economic_metric = "price_purchase_amount_weighted_hit@10"
    reading = {
        "positive_screen": bool(
            economic_metric in metric_index.index
            and metric_index.at[economic_metric, "absolute_delta"] > 0
            and accuracy_ratios
            and min(accuracy_ratios.values()) >= 0.99
        ),
        "accuracy_ratios": accuracy_ratios,
        "next_if_positive": "run M1@72, shuffled-user, then 10 seeds",
        "statistical_note": "seed 42 exploratory screen; no significance claim",
    }
    payload = {
        "code_version": CODE_VERSION,
        "source_revision": prepared["revision"],
        "config": asdict(cfg),
        "preflight": preflight_summary(cfg),
        "input_manifest": prepared["manifest"],
        "reused_baseline_source": baseline["source_result"],
        "absolute_rows": frame.to_dict("records"),
        "comparison_rows": comparison.to_dict("records"),
        "screening_reading": reading,
        "result_paths": {key: str(value) for key, value in paths.items()},
    }
    test10._atomic_json(paths["json"], payload)
    frame.attrs["comparison"] = comparison
    frame.attrs["screening_reading"] = reading
    frame.attrs["result_paths"] = {key: str(value) for key, value in paths.items()}
    print("\n절대지표:")
    print(frame.to_string(index=False))
    print("\nM1 대비 변화:")
    print(comparison.to_string(index=False))
    print("\n판독:", reading)
    print("결과 파일:", frame.attrs["result_paths"])
    return frame


if __name__ == "__main__":
    print(
        json.dumps(
            preflight_summary(configure_gatefree_lowdim_run()),
            ensure_ascii=False,
            indent=2,
        )
    )
