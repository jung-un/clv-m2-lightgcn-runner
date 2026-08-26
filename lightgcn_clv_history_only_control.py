"""Matched history-only control for the purchase-history conditioned M2.

Only one new arm is trained: the exact same history-based LightGCN used by
the preceding M2, with ``rho=0``.  The completed M1@64 and full N/V-
conditioned M2 results are reused after strict split/input compatibility
checks.  This separates the cost of replacing the free user-ID table from the
incremental effect of the N/V-conditioned maps.
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

from clv_history_conditioned_lowrank_model import (
    CLVHistoryConditionedLowRankLightGCN,
)
from clv_run_state import ProgressStore, RunIdentity, clone_state, file_sha256
import lightgcn_clv_axis_specific_test10 as test10
import lightgcn_clv_history_conditioned_lowrank as full_runner
import lightgcn_clv_moe as moe
import lightgcn_clv_v3 as v3


CODE_VERSION = "m2-history-only-matched-control-v1"
MODEL_ID = "history_only_rho0"
FULL_MODEL_ID = full_runner.MODEL_ID


@dataclass(frozen=True)
class HistoryOnlyControlConfig:
    dataset: str = "dunnhumby"
    seed: int = 42
    time_cutoff: int = 690
    evaluation_days: int = 7
    epochs: int = 100
    embedding_dim: int = 64
    transform_rank: int = 4
    rho: float = 0.0
    n_layers: int = 2
    batch_size: int = 8192
    lr: float = 5e-4
    pref_reg: float = 1e-3
    input_days: int = 365
    out_dir: str = ""
    baseline_result_dir: str = ""
    full_m2_result_dir: str = ""

    @property
    def id_dim(self) -> int:
        return self.embedding_dim


def configure_history_only_control(**overrides) -> HistoryOnlyControlConfig:
    defaults = {
        "out_dir": (
            f"{v3.default_out_dir('dunnhumby')}"
            "_m2_history_only_matched_control_v1"
        ),
        "baseline_result_dir": (
            f"{v3.default_out_dir('dunnhumby')}"
            "_m2_repeatshare_historical_backtest_v1"
        ),
        "full_m2_result_dir": (
            f"{v3.default_out_dir('dunnhumby')}"
            "_m2_history_conditioned_lowrank_historical_screen_v1"
        ),
    }
    return validate_config(HistoryOnlyControlConfig(**(defaults | overrides)))


def validate_config(cfg: HistoryOnlyControlConfig) -> HistoryOnlyControlConfig:
    required = {
        "dataset": "dunnhumby",
        "seed": 42,
        "time_cutoff": 690,
        "evaluation_days": 7,
        "epochs": 100,
        "embedding_dim": 64,
        "transform_rank": 4,
        "rho": 0.0,
        "n_layers": 2,
        "input_days": 365,
    }
    for key, expected in required.items():
        if getattr(cfg, key) != expected:
            raise ValueError(f"matched history-only control은 {key}={expected!r}이어야 합니다")
    if cfg.batch_size <= 0 or cfg.lr <= 0 or cfg.pref_reg < 0:
        raise ValueError("학습 설정이 잘못됐습니다")
    if not cfg.out_dir or not cfg.baseline_result_dir or not cfg.full_m2_result_dir:
        raise ValueError("out_dir·baseline_result_dir·full_m2_result_dir가 필요합니다")
    return cfg


def preflight_summary(cfg: HistoryOnlyControlConfig) -> dict:
    cfg = validate_config(cfg)
    return {
        "code_version": CODE_VERSION,
        "dataset": cfg.dataset,
        "seed": cfg.seed,
        "trained_models": [MODEL_ID],
        "reused_comparators": ["m1_64", FULL_MODEL_ID],
        "historical_development_split": {
            "train_end_inclusive": 683,
            "evaluation_start_inclusive": 684,
            "evaluation_end_inclusive": 690,
            "original_validation_test_holdout_constructed": False,
        },
        "control": {
            "purpose": (
                "separate free-user-ID removal from the incremental N/V map effect"
            ),
            "user_base": "normalized_purchase_history",
            "free_user_id_embedding": False,
            "conditional_maps_present_for_matched_initialization": True,
            "rho": cfg.rho,
            "layer0_identity": "E_u^(0)=H_u exactly",
            "same_architecture_as_full_m2_except_rho": True,
        },
        "fixed": {
            "graph": "binary",
            "negative_sampling": "uniform",
            "sample_weighting": False,
            "loss": "one plain pairwise BPR plus the existing sampled L2",
            "new_loss_term": False,
            "one_training_loop_and_optimizer": True,
            "min_user_interactions": 1,
            "min_item_interactions": 1,
            "epochs": cfg.epochs,
            "validation_or_epoch_selection": False,
        },
        "interpretation": {
            "history_only_below_m1": (
                "removing the free user-ID table is a material source of loss"
            ),
            "history_only_near_m1_but_full_below_history": (
                "the N/V-conditioned maps are the material source of loss"
            ),
            "full_above_history_but_both_below_m1": (
                "N/V helps the history model but does not recover the user-ID loss"
            ),
            "statistical_note": "one-seed mechanism control; no significance claim",
        },
        "out_dir": cfg.out_dir,
        "baseline_result_dir": cfg.baseline_result_dir,
        "full_m2_result_dir": cfg.full_m2_result_dir,
    }


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def _config_hash(cfg: HistoryOnlyControlConfig, input_hash: str, revision: str) -> str:
    payload = {
        "code_version": CODE_VERSION,
        "config": asdict(cfg),
        "model_id": MODEL_ID,
        "input_hash": input_hash,
        "source_revision": revision,
    }
    return hashlib.sha256(_canonical(payload).encode()).hexdigest()[:12]


def _arm_hash(prepared: dict, cfg: HistoryOnlyControlConfig) -> str:
    return hashlib.sha256(
        _canonical(
            {
                "run": prepared["config_hash"],
                "model_id": MODEL_ID,
                "seed": cfg.seed,
            }
        ).encode()
    ).hexdigest()[:12]


def _base_config(cfg: HistoryOnlyControlConfig) -> dict:
    return full_runner._base_config(cfg)


def _prepare(cfg: HistoryOnlyControlConfig) -> dict:
    # The existing helper constructs the exact same historical split, axes,
    # baseline, metadata and evaluation cache.  Its public validation is not
    # called because this matched control deliberately fixes rho=0.
    prepared = full_runner._prepare(cfg)
    prepared["config_hash"] = _config_hash(
        cfg, prepared["input_hash"], prepared["revision"]
    )
    return prepared


def _build_model(prepared: dict, cfg: HistoryOnlyControlConfig):
    data, axes = prepared["data"], prepared["axes"]
    v3.set_seed(cfg.seed)
    model = CLVHistoryConditionedLowRankLightGCN(
        n_users=data["n_users"],
        n_items=data["n_items"],
        q_n=axes["q_n"],
        q_v=axes["q_v"],
        user_activity_valid=axes["activity_valid"],
        user_value_valid=axes["value_valid"],
        adj=data["adj"],
        embedding_dim=cfg.embedding_dim,
        transform_rank=cfg.transform_rank,
        rho=cfg.rho,
        n_layers=cfg.n_layers,
        pref_reg=cfg.pref_reg,
    ).to(v3.DEVICE)
    return model, list(model.parameters())


def _run_history_only(prepared: dict, cfg: HistoryOnlyControlConfig) -> dict:
    root = prepared["out_dir"] / "arms" / prepared["config_hash"]
    result_path = root / f"{MODEL_ID}_s{cfg.seed}.json"
    checkpoint_path = root / f"{MODEL_ID}_s{cfg.seed}.pt"
    if result_path.exists():
        print("  [cached] history-only rho=0 seed 42 완료 결과 재사용")
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
    diagnostics = model.representation_diagnostics()
    if diagnostics["mean_user_representation_change"] != 0.0:
        raise RuntimeError("rho=0 history-only가 H_u와 일치하지 않습니다")
    if (
        diagnostics["activity_effective_ratio_to_history"] != 0.0
        or diagnostics["value_effective_ratio_to_history"] != 0.0
    ):
        raise RuntimeError("rho=0에서 N/V 실효 개입량이 0이 아닙니다")

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
        "role": "matched_control",
        "seed": cfg.seed,
        "split": "historical_development_days_684_690",
        "final_epoch": cfg.epochs,
        "metrics": test10._public_metrics(metrics),
        "diagnostics": diagnostics,
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


def _load_full_m2_result(cfg: HistoryOnlyControlConfig, prepared: dict) -> dict:
    root = Path(cfg.full_m2_result_dir)
    candidates = sorted(root.glob("m2_history_conditioned_lowrank_*.json"))
    matches = []
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        config = payload.get("config", {})
        if payload.get("code_version") != full_runner.CODE_VERSION:
            continue
        if payload.get("input_manifest") != prepared["manifest"]:
            continue
        if any(
            config.get(key) != expected
            for key, expected in {
                "seed": 42,
                "time_cutoff": 690,
                "evaluation_days": 7,
                "epochs": 100,
                "embedding_dim": 64,
                "transform_rank": 4,
                "rho": 0.05,
                "n_layers": 2,
                "batch_size": cfg.batch_size,
                "lr": cfg.lr,
                "pref_reg": cfg.pref_reg,
                "input_days": 365,
            }.items()
        ):
            continue
        rows = [
            row
            for row in payload.get("absolute_rows", [])
            if row.get("model_id") == FULL_MODEL_ID
        ]
        if len(rows) == 1:
            matches.append((path, rows[0]))
    if len(matches) != 1:
        raise RuntimeError(
            "동일 manifest·split·seed의 완료된 full M2 결과가 정확히 1개여야 "
            f"합니다: dir={root}, matches={[str(path) for path, _ in matches]}"
        )
    path, row = matches[0]
    metrics = {
        key: row[key]
        for key in prepared["baseline"]
        if key in row and isinstance(row[key], (int, float, np.number))
    }
    return {
        "model_id": FULL_MODEL_ID,
        "role": "reused_full_m2",
        "seed": 42,
        "split": "historical_development_days_684_690",
        "final_epoch": 100,
        "metrics": metrics,
        "source_result": str(path),
    }


def _wide_comparison(
    baseline: dict, history_only: dict, full_m2: dict
) -> pd.DataFrame:
    rows = []
    for metric, m1_value in baseline.items():
        if metric not in history_only["metrics"] or metric not in full_m2["metrics"]:
            continue
        history_value = history_only["metrics"][metric]
        full_value = full_m2["metrics"][metric]
        if not all(
            isinstance(value, (int, float, np.number))
            for value in (m1_value, history_value, full_value)
        ):
            continue
        rows.append(
            {
                "metric": metric,
                "m1_64": float(m1_value),
                MODEL_ID: float(history_value),
                FULL_MODEL_ID: float(full_value),
                "history_only_minus_m1": float(history_value) - float(m1_value),
                "full_m2_minus_history_only": float(full_value)
                - float(history_value),
                "full_m2_minus_m1": float(full_value) - float(m1_value),
            }
        )
    return pd.DataFrame(rows)


def _mechanism_reading(comparison: pd.DataFrame) -> dict:
    indexed = comparison.set_index("metric")
    accuracy = [
        "recall@10",
        "ndcg@10",
        "recall@20",
        "ndcg@20",
        "recall@50",
        "ndcg@50",
    ]
    economic = "price_purchase_amount_weighted_hit@10"
    history_ratios = {
        metric: float(indexed.at[metric, MODEL_ID])
        / float(indexed.at[metric, "m1_64"])
        for metric in accuracy
    }
    full_vs_history_ratios = {
        metric: float(indexed.at[metric, FULL_MODEL_ID])
        / float(indexed.at[metric, MODEL_ID])
        for metric in accuracy
    }
    history_near_m1 = min(history_ratios.values()) >= 0.99
    full_economic_vs_history = float(indexed.at[economic, FULL_MODEL_ID]) - float(
        indexed.at[economic, MODEL_ID]
    )
    full_protects_history_accuracy = min(full_vs_history_ratios.values()) >= 0.99
    if history_near_m1 and (
        full_economic_vs_history < 0 or not full_protects_history_accuracy
    ):
        classification = "n_v_transform_is_primary_harm"
    elif (
        not history_near_m1
        and full_economic_vs_history > 0
        and full_protects_history_accuracy
    ):
        classification = "n_v_helps_history_base_but_user_id_loss_remains"
    elif not history_near_m1:
        classification = "free_user_id_removal_is_material_and_n_v_does_not_recover_it"
    else:
        classification = "no_single_dominant_source_on_seed42"
    return {
        "classification": classification,
        "history_only_accuracy_ratios_vs_m1": history_ratios,
        "full_m2_accuracy_ratios_vs_history_only": full_vs_history_ratios,
        "history_only_near_m1_99pct_rule": history_near_m1,
        "full_m2_weighted_hit10_delta_vs_history_only": full_economic_vs_history,
        "full_m2_protects_history_accuracy_99pct_rule": (
            full_protects_history_accuracy
        ),
        "statistical_note": "seed 42 mechanism screen; no significance claim",
    }


def run_history_only_control(
    cfg: HistoryOnlyControlConfig | None = None,
) -> pd.DataFrame:
    cfg = validate_config(cfg or configure_history_only_control())
    print(json.dumps(preflight_summary(cfg), ensure_ascii=False, indent=2))
    prepared = _prepare(cfg)
    full_m2 = _load_full_m2_result(cfg, prepared)
    print("\n===== matched history-only rho=0 | seed 42 | fixed 100 epochs =====")
    history_only = _run_history_only(prepared, cfg)
    baseline = dict(prepared["baseline"])
    baseline["role"] = "reused_baseline"

    frame = pd.DataFrame(
        [
            baseline,
            {
                "model_id": history_only["model_id"],
                "role": history_only["role"],
                "seed": history_only["seed"],
                "split": history_only["split"],
                "final_epoch": history_only["final_epoch"],
                **history_only["diagnostics"],
                **history_only["metrics"],
            },
            {
                "model_id": full_m2["model_id"],
                "role": full_m2["role"],
                "seed": full_m2["seed"],
                "split": full_m2["split"],
                "final_epoch": full_m2["final_epoch"],
                "source_result": full_m2["source_result"],
                **full_m2["metrics"],
            },
        ]
    )
    comparison = _wide_comparison(baseline, history_only, full_m2)
    reading = _mechanism_reading(comparison)
    stem = f"m2_history_only_control_{prepared['config_hash']}"
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
        "reused_full_m2_source": full_m2["source_result"],
        "absolute_rows": frame.to_dict("records"),
        "comparison_rows": comparison.to_dict("records"),
        "mechanism_reading": reading,
        "result_paths": {key: str(value) for key, value in paths.items()},
    }
    test10._atomic_json(paths["json"], payload)
    frame.attrs["comparison"] = comparison
    frame.attrs["mechanism_reading"] = reading
    frame.attrs["result_paths"] = {key: str(value) for key, value in paths.items()}

    print("\n절대지표:")
    print(frame.to_string(index=False))
    print("\nM1 / history-only / full M2 원인 분리표:")
    print(comparison.to_string(index=False))
    print("\n메커니즘 판독:", reading)
    print("결과 파일:", frame.attrs["result_paths"])
    return frame


if __name__ == "__main__":
    print(
        json.dumps(
            preflight_summary(configure_history_only_control()),
            ensure_ascii=False,
            indent=2,
        )
    )
