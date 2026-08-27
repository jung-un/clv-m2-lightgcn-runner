"""Seed-42 historical screen for fixed F/L/V personal-history fit M2.

This runner changes one thing relative to the existing N/V personal-history
candidate-fit model: both within-user history shares receive a fixed time
decay based on the user's own mean transaction gap.  It does not add learned
attention, graph weights, sample weights, or a new loss term.
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

from clv_history_item_fit_model import (
    HistoryItemFitLightGCN,
    build_temporally_decayed_personal_history_weights,
)
from clv_run_state import ProgressStore, RunIdentity, clone_state, file_sha256
import lightgcn_clv_axis_specific_test10 as test10
import lightgcn_clv_gatefree_lowdim as gatefree
import lightgcn_clv_history_item_fit as current_runner
import lightgcn_clv_joint_nv as joint
import lightgcn_clv_moe as moe
import lightgcn_clv_residual as residual
import lightgcn_clv_v3 as v3


CODE_VERSION = "m2-flv-temporal-personal-history-fit-historical-screen-v1"
MODEL_ID = "m2_flv_temporal_personal_history_fit"
CURRENT_MODEL_ID = current_runner.MODEL_ID


@dataclass(frozen=True)
class TemporalHistoryItemFitConfig:
    dataset: str = "dunnhumby"
    seed: int = 42
    time_cutoff: int = 690
    evaluation_days: int = 7
    epochs: int = 100
    id_dim: int = 64
    axis_dim: int = 4
    rho: float = 0.05
    n_layers: int = 2
    batch_size: int = 8192
    lr: float = 5e-4
    pref_reg: float = 1e-3
    input_days: int = 365
    out_dir: str = ""
    baseline_result_dir: str = ""
    current_m2_result_dir: str = ""


def configure_temporal_history_item_fit_run(
    **overrides,
) -> TemporalHistoryItemFitConfig:
    root = v3.default_out_dir("dunnhumby")
    defaults = {
        "out_dir": (
            f"{root}_m2_flv_temporal_personal_history_fit_historical_screen_v1"
        ),
        "baseline_result_dir": (
            f"{root}_m2_repeatshare_historical_backtest_v1"
        ),
        "current_m2_result_dir": (
            f"{root}_m2_nv_personal_history_candidate_fit_historical_screen_v1"
        ),
    }
    return validate_config(
        TemporalHistoryItemFitConfig(**(defaults | overrides))
    )


def validate_config(
    cfg: TemporalHistoryItemFitConfig,
) -> TemporalHistoryItemFitConfig:
    required = {
        "dataset": "dunnhumby",
        "seed": 42,
        "time_cutoff": 690,
        "evaluation_days": 7,
        "epochs": 100,
        "id_dim": 64,
        "axis_dim": 4,
        "rho": 0.05,
        "n_layers": 2,
        "input_days": 365,
    }
    for key, expected in required.items():
        if getattr(cfg, key) != expected:
            raise ValueError(f"고정 F/L/V screen은 {key}={expected!r}이어야 합니다")
    if cfg.batch_size <= 0 or cfg.lr <= 0 or cfg.pref_reg < 0:
        raise ValueError("학습 설정이 잘못됐습니다")
    if not cfg.out_dir or not cfg.baseline_result_dir or not cfg.current_m2_result_dir:
        raise ValueError("out_dir·baseline_result_dir·current_m2_result_dir가 필요합니다")
    return cfg


def preflight_summary(cfg: TemporalHistoryItemFitConfig) -> dict:
    cfg = validate_config(cfg)
    return {
        "code_version": CODE_VERSION,
        "dataset": cfg.dataset,
        "seed": cfg.seed,
        "trained_models": [MODEL_ID],
        "reused_comparators": ["m1_64", CURRENT_MODEL_ID],
        "research_axis": "M2 representation intervention",
        "historical_development_split": {
            "train_end_inclusive": 683,
            "evaluation_start_inclusive": 684,
            "evaluation_end_inclusive": 690,
            "final_test_constructed": False,
            "holdout_constructed": False,
        },
        "m2": {
            "architecture": "LightGCN-ID(64)|history-FL-fit(4)|history-VL-fit(4)",
            "activity_history_weight": (
                "within-user distinct-basket share times fixed relationship-time decay"
            ),
            "value_history_weight": (
                "within-user purchase-amount share times fixed relationship-time decay"
            ),
            "relationship_time_decay": (
                "exp(-user-item recency / user mean distinct-basket gap)"
            ),
            "time_decay_learned": False,
            "invalid_gap_fallback": "original N/V history shares",
            "shares_renormalized_within_user": True,
            "learned_attention": False,
            "source_target_item_factors": "separate and jointly learned",
            "positive_item_excluded_from_training_history": True,
            "global_item_repeatshare_input": False,
            "global_item_popularity_input": False,
            "fixed_axis_scale": cfg.rho,
            "learned_global_axis_weight": False,
            "total_dim": cfg.id_dim + 2 * cfg.axis_dim,
            "historical_clv_note": "F, L, and V are computed only from train history",
        },
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
        "interpretation": (
            "seed-42 historical development screen; this tests the incremental "
            "relationship-time signal, not final-test generalization"
        ),
        "out_dir": cfg.out_dir,
        "baseline_result_dir": cfg.baseline_result_dir,
        "current_m2_result_dir": cfg.current_m2_result_dir,
    }


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def _base_config(cfg: TemporalHistoryItemFitConfig) -> dict:
    return gatefree._base_config(cfg)


def _config_hash(
    cfg: TemporalHistoryItemFitConfig, input_hash: str, revision: str
) -> str:
    payload = {
        "code_version": CODE_VERSION,
        "config": asdict(cfg),
        "input_hash": input_hash,
        "source_revision": revision,
    }
    return hashlib.sha256(_canonical(payload).encode()).hexdigest()[:12]


def _prepare(cfg: TemporalHistoryItemFitConfig) -> dict:
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
    history = build_temporally_decayed_personal_history_weights(
        data["train"],
        n_users=data["n_users"],
        n_items=data["n_items"],
        is_date=v3.DCFG["is_date"],
    )
    baseline = gatefree._load_compatible_baseline(cfg, manifest)
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
        "history": history,
        "baseline": baseline,
        "meta": meta,
        "cache": cache,
    }
    prepared["config_hash"] = _config_hash(cfg, input_hash, revision)
    return prepared


def _build_model(prepared: dict, cfg: TemporalHistoryItemFitConfig):
    data, axes = prepared["data"], prepared["axes"]
    v3.set_seed(cfg.seed)
    model = HistoryItemFitLightGCN(
        n_users=data["n_users"],
        n_items=data["n_items"],
        history=prepared["history"],
        q_n=axes["q_n"],
        q_v=axes["q_v"],
        activity_valid=axes["activity_valid"],
        value_valid=axes["value_valid"],
        adj=data["adj"],
        id_dim=cfg.id_dim,
        axis_dim=cfg.axis_dim,
        n_layers=cfg.n_layers,
        rho=cfg.rho,
        pref_reg=cfg.pref_reg,
    ).to(v3.DEVICE)
    return model, list(model.parameters())


def _arm_hash(prepared: dict, cfg: TemporalHistoryItemFitConfig) -> str:
    return hashlib.sha256(
        _canonical(
            {"run": prepared["config_hash"], "model_id": MODEL_ID, "seed": cfg.seed}
        ).encode()
    ).hexdigest()[:12]


def _run_model(prepared: dict, cfg: TemporalHistoryItemFitConfig) -> dict:
    root = prepared["out_dir"] / "arms" / prepared["config_hash"]
    result_path = root / f"{MODEL_ID}_s{cfg.seed}.json"
    checkpoint_path = root / f"{MODEL_ID}_s{cfg.seed}.pt"
    if result_path.exists():
        print("  [cached] F/L/V 시간감쇠 M2 완료 결과 재사용")
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
        "history_weight_diagnostics": prepared["history"].diagnostics,
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


def _load_current_m2_result(
    cfg: TemporalHistoryItemFitConfig, prepared: dict
) -> dict:
    root = Path(cfg.current_m2_result_dir)
    candidates = sorted(root.glob("m2_nv_personal_history_candidate_fit_*.json"))
    matches = []
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        config = payload.get("config", {})
        if payload.get("code_version") != current_runner.CODE_VERSION:
            continue
        if _canonical(payload.get("input_manifest")) != _canonical(prepared["manifest"]):
            continue
        expected = {
            "dataset": "dunnhumby",
            "seed": 42,
            "time_cutoff": 690,
            "evaluation_days": 7,
            "epochs": 100,
            "id_dim": 64,
            "axis_dim": 4,
            "rho": 0.05,
            "n_layers": 2,
            "batch_size": cfg.batch_size,
            "lr": cfg.lr,
            "pref_reg": cfg.pref_reg,
            "input_days": 365,
        }
        if any(config.get(key) != value for key, value in expected.items()):
            continue
        rows = [
            row
            for row in payload.get("absolute_rows", [])
            if row.get("model_id") == CURRENT_MODEL_ID
            and row.get("seed") == 42
            and row.get("final_epoch") == 100
            and row.get("split") == "historical_development_days_684_690"
        ]
        if len(rows) == 1:
            matches.append((path, rows[0]))
    if len(matches) != 1:
        raise RuntimeError(
            "동일 manifest·split·seed의 기존 N/V 이력 모형 결과가 정확히 "
            f"1개여야 합니다: dir={root}, matches={[str(p) for p, _ in matches]}"
        )
    path, row = matches[0]
    metrics = {
        key: row[key]
        for key in prepared["baseline"]
        if key in row and isinstance(row[key], (int, float, np.number))
    }
    return {
        "model_id": CURRENT_MODEL_ID,
        "role": "reused_current_m2",
        "seed": 42,
        "split": "historical_development_days_684_690",
        "final_epoch": 100,
        "metrics": metrics,
        "source_result": str(path),
    }


def _wide_comparison(baseline: dict, current: dict, temporal: dict) -> pd.DataFrame:
    rows = []
    for metric, m1_value in baseline.items():
        if metric not in current["metrics"] or metric not in temporal["metrics"]:
            continue
        current_value = current["metrics"][metric]
        temporal_value = temporal["metrics"][metric]
        if not all(
            isinstance(value, (int, float, np.number))
            for value in (m1_value, current_value, temporal_value)
        ):
            continue
        rows.append(
            {
                "metric": metric,
                "m1_64": float(m1_value),
                CURRENT_MODEL_ID: float(current_value),
                MODEL_ID: float(temporal_value),
                "temporal_minus_m1": float(temporal_value) - float(m1_value),
                "temporal_minus_current_m2": float(temporal_value)
                - float(current_value),
            }
        )
    return pd.DataFrame(rows)


def run_temporal_history_item_fit_screen(
    cfg: TemporalHistoryItemFitConfig | None = None,
) -> pd.DataFrame:
    cfg = validate_config(cfg or configure_temporal_history_item_fit_run())
    print(json.dumps(preflight_summary(cfg), ensure_ascii=False, indent=2))
    prepared = _prepare(cfg)
    current = _load_current_m2_result(cfg, prepared)
    print("\nF/L/V 개인 구매이력 가중치 진단:")
    print(json.dumps(prepared["history"].diagnostics, ensure_ascii=False, indent=2))
    print("\n===== F/L/V 시간감쇠 M2만 학습 | seed 42 | fixed 100 epochs =====")
    arm = _run_model(prepared, cfg)

    baseline = dict(prepared["baseline"])
    baseline["role"] = "reused_baseline"
    frame = pd.DataFrame(
        [
            baseline,
            {
                "model_id": current["model_id"],
                "role": current["role"],
                "seed": current["seed"],
                "split": current["split"],
                "final_epoch": current["final_epoch"],
                "source_result": current["source_result"],
                **current["metrics"],
            },
            {
                "model_id": arm["model_id"],
                "role": arm["role"],
                "seed": arm["seed"],
                "split": arm["split"],
                "final_epoch": arm["final_epoch"],
                **arm["history_weight_diagnostics"],
                **arm["diagnostics"],
                **arm["metrics"],
            },
        ]
    )
    comparison = _wide_comparison(baseline, current, arm)
    indexed = comparison.set_index("metric")
    accuracy_names = [
        "recall@10",
        "ndcg@10",
        "recall@20",
        "ndcg@20",
        "recall@50",
        "ndcg@50",
    ]
    accuracy_ratios = {
        metric: float(indexed.at[metric, MODEL_ID])
        / float(indexed.at[metric, "m1_64"])
        for metric in accuracy_names
        if metric in indexed.index
    }
    economic_metric = "price_purchase_amount_weighted_hit@10"
    reading = {
        "positive_screen_vs_m1": bool(
            economic_metric in indexed.index
            and indexed.at[economic_metric, "temporal_minus_m1"] > 0
            and accuracy_ratios
            and min(accuracy_ratios.values()) >= 0.99
        ),
        "incremental_time_signal_weighted_hit10_vs_current_m2": (
            float(indexed.at[economic_metric, "temporal_minus_current_m2"])
            if economic_metric in indexed.index
            else None
        ),
        "accuracy_ratios_vs_m1": accuracy_ratios,
        "next_if_positive": (
            "test one shared F/L/V profile before adding learned attention"
        ),
        "statistical_note": "seed 42 exploratory screen; no significance claim",
        "protocol_note": "final test and holdout were not constructed",
    }

    stem = f"m2_flv_temporal_personal_history_fit_{prepared['config_hash']}"
    paths = {
        "absolute_csv": prepared["out_dir"] / f"{stem}.csv",
        "comparison_csv": prepared["out_dir"] / f"{stem}_comparison.csv",
        "json": prepared["out_dir"] / f"{stem}.json",
    }
    test10._atomic_csv(paths["absolute_csv"], frame)
    test10._atomic_csv(paths["comparison_csv"], comparison)
    payload = {
        "code_version": CODE_VERSION,
        "source_revision": prepared["revision"],
        "config": asdict(cfg),
        "preflight": preflight_summary(cfg),
        "input_manifest": prepared["manifest"],
        "reused_baseline_source": baseline["source_result"],
        "reused_current_m2_source": current["source_result"],
        "history_weight_diagnostics": prepared["history"].diagnostics,
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
    print("\nM1 / 기존 N/V 이력 / F/L/V 시간감쇠 비교:")
    print(comparison.to_string(index=False))
    print("\n탐색 판독:", reading)
    print("결과 파일:", frame.attrs["result_paths"])
    return frame


if __name__ == "__main__":
    print(
        json.dumps(
            preflight_summary(configure_temporal_history_item_fit_run()),
            ensure_ascii=False,
            indent=2,
        )
    )
