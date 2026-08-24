"""Seed-42 historical M2 screen with independent item axes and block dropout.

Only the new M2 arm is trained.  The compatible M1@64 historical result is
reused.  ID, activity, and transaction-value coordinates are learned in one
LightGCN with one optimizer and one plain BPR objective.  The two item-axis
tables are independent of the shared item-ID table.  During training, the N
and V score blocks are independently dropped; evaluation always uses both.
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
import lightgcn_clv_gatefree_lowdim_balanced_training as historical
import lightgcn_clv_moe as moe
import lightgcn_clv_v3 as v3


CODE_VERSION = "m2-gatefree-lowdim-independent-item-axis-block-dropout-v1"
MODEL_ID = "m2_gatefree_lowdim_independent_dropout"


@dataclass(frozen=True)
class IndependentDropoutConfig:
    dataset: str = "dunnhumby"
    seed: int = 42
    time_cutoff: int = 690
    evaluation_days: int = 7
    epochs: int = 100
    id_dim: int = 64
    axis_dim: int = 4
    hidden_dim: int = 8
    axis_budget: float = 0.1
    axis_keep_probability: float = 0.5
    n_layers: int = 2
    batch_size: int = 8192
    lr: float = 5e-4
    pref_reg: float = 1e-3
    input_days: int = 365
    out_dir: str = ""
    baseline_result_dir: str = ""


def configure_independent_dropout_run(**overrides) -> IndependentDropoutConfig:
    defaults = {
        "out_dir": (
            f"{v3.default_out_dir('dunnhumby')}"
            "_m2_gatefree_lowdim_independent_dropout_historical_screen_v1"
        ),
        "baseline_result_dir": (
            f"{v3.default_out_dir('dunnhumby')}"
            "_m2_repeatshare_historical_backtest_v1"
        ),
    }
    return validate_config(IndependentDropoutConfig(**(defaults | overrides)))


def validate_config(cfg: IndependentDropoutConfig) -> IndependentDropoutConfig:
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
    if cfg.axis_budget != 0.1:
        raise ValueError("이번 screen의 축별 고정 예산은 0.1이어야 합니다")
    if cfg.axis_keep_probability != 0.5:
        raise ValueError("이번 screen의 축 블록 유지확률은 0.5이어야 합니다")
    if cfg.batch_size <= 0 or cfg.lr <= 0 or cfg.pref_reg < 0:
        raise ValueError("학습 설정이 잘못됐습니다")
    if not cfg.out_dir or not cfg.baseline_result_dir:
        raise ValueError("out_dir와 baseline_result_dir가 필요합니다")
    return cfg


def preflight_summary(cfg: IndependentDropoutConfig) -> dict:
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
            "independent_item_axis_coordinates": True,
            "item_response": (
                "independent low-dimensional item coordinates learned by the "
                "same recommendation loss"
            ),
            "user_gate": False,
            "learned_axis_weight": False,
            "fixed_per_axis_budget": cfg.axis_budget,
            "axis_keep_probability": cfg.axis_keep_probability,
            "training_state_probabilities": {
                "ID_only": 0.25,
                "ID_plus_activity": 0.25,
                "ID_plus_transaction_value": 0.25,
                "full": 0.25,
            },
            "training_dropout_scaling": "inverted block dropout",
            "training_expected_score_formula": "S_ID + 0.1*S_N + 0.1*S_V",
            "evaluation_score_formula": "S_ID + 0.1*S_N + 0.1*S_V",
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


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def _config_hash(cfg: IndependentDropoutConfig, input_hash: str, revision: str) -> str:
    payload = {
        "code_version": CODE_VERSION,
        "config": asdict(cfg),
        "model_id": MODEL_ID,
        "input_hash": input_hash,
        "source_revision": revision,
    }
    return hashlib.sha256(_canonical(payload).encode()).hexdigest()[:12]


def _prepare(cfg: IndependentDropoutConfig) -> dict:
    # Reuse only the established split/data preparation and fail-closed M1
    # compatibility check.  The new arm receives its own run identity below.
    prepared = historical._prepare(cfg)
    prepared["config_hash"] = _config_hash(
        cfg, prepared["input_hash"], prepared["revision"]
    )
    return prepared


def _build_model(prepared: dict, cfg: IndependentDropoutConfig):
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
        training_axis_balance_delta=0.0,
        independent_item_axes=True,
        axis_keep_probability=cfg.axis_keep_probability,
        pref_reg=cfg.pref_reg,
    ).to(v3.DEVICE)
    return model, list(model.parameters())


def _arm_hash(prepared: dict, cfg: IndependentDropoutConfig) -> str:
    return hashlib.sha256(
        _canonical(
            {
                "run": prepared["config_hash"],
                "model_id": MODEL_ID,
                "seed": cfg.seed,
            }
        ).encode()
    ).hexdigest()[:12]


def _run_model(prepared: dict, cfg: IndependentDropoutConfig) -> dict:
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
        "training_axis_balance_delta",
        "independent_item_axes",
        "axis_keep_probability",
        "training_axis_block_dropout",
        "training_activity_multiplier_range",
        "training_value_multiplier_range",
        "evaluation_activity_multiplier",
        "evaluation_value_multiplier",
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


def run_independent_dropout_screen(
    cfg: IndependentDropoutConfig | None = None,
) -> pd.DataFrame:
    cfg = validate_config(cfg or configure_independent_dropout_run())
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
    stem = f"m2_gatefree_lowdim_independent_dropout_{prepared['config_hash']}"
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
            preflight_summary(configure_independent_dropout_run()),
            ensure_ascii=False,
            indent=2,
        )
    )
