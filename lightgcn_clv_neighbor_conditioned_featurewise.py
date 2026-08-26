"""Seed-42 historical screen for the simplified feature-wise M2 expression."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import torch

from clv_neighbor_conditioned_featurewise_model import (
    CLVNeighborConditionedFeaturewiseLightGCN,
)
from clv_run_state import ProgressStore, RunIdentity, clone_state, file_sha256
import lightgcn_clv_axis_specific_test10 as test10
import lightgcn_clv_neighbor_conditioned_id_transform as source
import lightgcn_clv_moe as moe
import lightgcn_clv_v3 as v3


CODE_VERSION = "m2-neighbor-conditioned-featurewise-historical-screen-v1"
MODEL_ID = "m2_neighbor_conditioned_featurewise"
MIN_EFFECTIVE_CORRECTION_RATIO = 1e-6
MIN_MEAN_USER_REPRESENTATION_CHANGE = 1e-8


@dataclass(frozen=True)
class NeighborConditionedFeaturewiseConfig:
    dataset: str = "dunnhumby"
    seed: int = 42
    time_cutoff: int = 690
    evaluation_days: int = 7
    epochs: int = 100
    embedding_dim: int = 64
    rho: float = 0.05
    n_layers: int = 2
    batch_size: int = 8192
    lr: float = 5e-4
    pref_reg: float = 1e-3
    input_days: int = 365
    out_dir: str = ""
    baseline_result_dir: str = ""

    @property
    def id_dim(self) -> int:
        return self.embedding_dim


def configure_neighbor_conditioned_featurewise_run(
    **overrides,
) -> NeighborConditionedFeaturewiseConfig:
    defaults = {
        "out_dir": (
            f"{v3.default_out_dir('dunnhumby')}"
            "_m2_neighbor_conditioned_featurewise_historical_screen_v1"
        ),
        "baseline_result_dir": (
            f"{v3.default_out_dir('dunnhumby')}"
            "_m2_repeatshare_historical_backtest_v1"
        ),
    }
    return validate_config(
        NeighborConditionedFeaturewiseConfig(**(defaults | overrides))
    )


def validate_config(
    cfg: NeighborConditionedFeaturewiseConfig,
) -> NeighborConditionedFeaturewiseConfig:
    required = {
        "dataset": "dunnhumby",
        "seed": 42,
        "time_cutoff": 690,
        "evaluation_days": 7,
        "epochs": 100,
        "embedding_dim": 64,
        "rho": 0.05,
        "n_layers": 2,
        "input_days": 365,
    }
    for key, expected in required.items():
        if getattr(cfg, key) != expected:
            raise ValueError(f"빠른 M2 screen은 {key}={expected!r}이어야 합니다")
    if cfg.batch_size <= 0 or cfg.lr <= 0 or cfg.pref_reg < 0:
        raise ValueError("학습 설정이 잘못됐습니다")
    if not cfg.out_dir or not cfg.baseline_result_dir:
        raise ValueError("out_dir와 baseline_result_dir가 필요합니다")
    return cfg


def preflight_summary(cfg: NeighborConditionedFeaturewiseConfig) -> dict:
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
            "original_validation_test_holdout_constructed": False,
        },
        "m2": {
            "architecture": (
                "64-d user ID plus feature-wise N/V conditioning of the "
                "one-hop learned-item-ID purchase-history expression"
            ),
            "formula": (
                "NormPreserve(E_u + rho[H_u*(c_N(u)w_N+c_V(u)w_V)]), "
                "c_k=m_k(q_k-mean_valid_train(q_k))"
            ),
            "embedding_dim": cfg.embedding_dim,
            "axis_parameter_count": 2 * cfg.embedding_dim,
            "rho": cfg.rho,
            "activity_condition": "train_percentile(repeat_transaction_rate)",
            "value_condition": "train_percentile(mean_transaction_value)",
            "condition_centring": "valid_train_user_mean",
            "both_axes_always_present": True,
            "separate_axis_score_space": False,
            "explicit_item_features": False,
            "item_transformation": False,
            "learned_global_axis_weight": False,
            "norm_preserving_user_transform": True,
            "zero_axis_vector_initialisation": True,
            "correction_source": "same_binary_graph_one_hop_item_ID_aggregate",
            "axis_regularization": "none",
            "epoch_liveness_diagnostics": True,
            "liveness_fail_closed": {
                "min_max_effective_correction_ratio": (
                    MIN_EFFECTIVE_CORRECTION_RATIO
                ),
                "min_mean_user_representation_change": (
                    MIN_MEAN_USER_REPRESENTATION_CHANGE
                ),
            },
        },
        "fixed": {
            "graph": "binary",
            "negative_sampling": "uniform",
            "sample_weighting": False,
            "loss": "one plain pairwise BPR plus the existing sampled-ID L2",
            "new_loss_term": False,
            "one_training_loop_and_optimizer": True,
            "min_user_interactions": 1,
            "min_item_interactions": 1,
            "epochs": cfg.epochs,
            "validation_or_epoch_selection": False,
        },
        "reading_rule": {
            "accuracy": "each Recall/NDCG@10/20/50 >= 99% of M1@64",
            "economic": "price_purchase_amount_weighted_hit@10 > M1@64",
            "significance": "not claimed from one exploratory seed",
        },
        "out_dir": cfg.out_dir,
        "baseline_result_dir": cfg.baseline_result_dir,
    }


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def _config_hash(
    cfg: NeighborConditionedFeaturewiseConfig, input_hash: str, revision: str
) -> str:
    payload = {
        "code_version": CODE_VERSION,
        "config": asdict(cfg),
        "model_id": MODEL_ID,
        "input_hash": input_hash,
        "source_revision": revision,
    }
    return hashlib.sha256(_canonical(payload).encode()).hexdigest()[:12]


def _base_config(cfg: NeighborConditionedFeaturewiseConfig) -> dict:
    return source._base_config(cfg)


def _prepare(cfg: NeighborConditionedFeaturewiseConfig) -> dict:
    # Use the same verified historical split, N/V construction, and M1 loader.
    prepared = source.previous._prepare(cfg)
    prepared["config_hash"] = _config_hash(
        cfg, prepared["input_hash"], prepared["revision"]
    )
    return prepared


def _build_model(prepared: dict, cfg: NeighborConditionedFeaturewiseConfig):
    data, axes = prepared["data"], prepared["axes"]
    v3.set_seed(cfg.seed)
    model = CLVNeighborConditionedFeaturewiseLightGCN(
        n_users=data["n_users"],
        n_items=data["n_items"],
        q_n=axes["q_n"],
        q_v=axes["q_v"],
        user_activity_valid=axes["activity_valid"],
        user_value_valid=axes["value_valid"],
        adj=data["adj"],
        embedding_dim=cfg.embedding_dim,
        rho=cfg.rho,
        n_layers=cfg.n_layers,
        pref_reg=cfg.pref_reg,
    ).to(v3.DEVICE)
    return model, list(model.parameters())


def _arm_hash(prepared: dict, cfg: NeighborConditionedFeaturewiseConfig) -> str:
    return hashlib.sha256(
        _canonical(
            {
                "run": prepared["config_hash"],
                "model_id": MODEL_ID,
                "seed": cfg.seed,
            }
        ).encode()
    ).hexdigest()[:12]


def _liveness_reading(diagnostics: dict) -> dict:
    activity_ratio = float(diagnostics["activity_effective_ratio_to_id"])
    value_ratio = float(diagnostics["value_effective_ratio_to_id"])
    representation_change = float(
        diagnostics["mean_user_representation_change"]
    )
    passed = bool(
        max(activity_ratio, value_ratio) >= MIN_EFFECTIVE_CORRECTION_RATIO
        and representation_change >= MIN_MEAN_USER_REPRESENTATION_CHANGE
    )
    return {
        "passed": passed,
        "status": "active_correction_path" if passed else "collapsed_correction_path",
        "activity_effective_ratio_to_id": activity_ratio,
        "value_effective_ratio_to_id": value_ratio,
        "mean_user_representation_change": representation_change,
        "min_max_effective_correction_ratio": MIN_EFFECTIVE_CORRECTION_RATIO,
        "min_mean_user_representation_change": (
            MIN_MEAN_USER_REPRESENTATION_CHANGE
        ),
    }


def _run_model(prepared: dict, cfg: NeighborConditionedFeaturewiseConfig) -> dict:
    root = prepared["out_dir"] / "arms" / prepared["config_hash"]
    result_path = root / f"{MODEL_ID}_s{cfg.seed}.json"
    checkpoint_path = root / f"{MODEL_ID}_s{cfg.seed}.pt"
    if result_path.exists():
        print("  [cached] 간략형 N/V 조건부 M2 seed 42 완료 결과 재사용")
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
    liveness = _liveness_reading(diagnostics)
    if not liveness["passed"]:
        raise RuntimeError(
            "N/V correction path collapsed below the predeclared numerical "
            f"liveness floor: {liveness}"
        )

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
        "diagnostics": diagnostics,
        "liveness": liveness,
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
        "rho",
        "axis_parameter_count",
        "total_dim",
        "explicit_item_features",
        "item_transformation",
        "correction_source",
        "condition_centring",
        "activity_condition_mean",
        "activity_condition_std",
        "value_condition_mean",
        "value_condition_std",
        "activity_valid_share",
        "value_valid_share",
        "purchase_neighbour_mean_norm",
        "activity_correction_mean_norm",
        "value_correction_mean_norm",
        "activity_effective_ratio_to_id",
        "value_effective_ratio_to_id",
        "activity_axis_vector_norm",
        "value_axis_vector_norm",
        "mean_user_norm",
        "mean_item_norm",
        "max_user_norm_change",
        "mean_user_representation_change",
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


def run_neighbor_conditioned_featurewise_screen(
    cfg: NeighborConditionedFeaturewiseConfig | None = None,
) -> pd.DataFrame:
    cfg = validate_config(cfg or configure_neighbor_conditioned_featurewise_run())
    print(json.dumps(preflight_summary(cfg), ensure_ascii=False, indent=2))
    prepared = _prepare(cfg)
    print("\n===== 간략형 N/V 조건부 M2만 학습 | seed 42 | fixed 100 epochs =====")
    arm = _run_model(prepared, cfg)
    baseline = dict(prepared["baseline"])
    baseline["role"] = "reused_baseline"
    frame = pd.DataFrame(
        [
            baseline,
            {
                "model_id": arm["model_id"],
                "role": arm["role"],
                "seed": arm["seed"],
                "split": arm["split"],
                "final_epoch": arm["final_epoch"],
                **arm["diagnostics"],
                **arm["metrics"],
            },
        ]
    )
    comparison = _comparison(frame)
    stem = f"m2_neighbor_conditioned_featurewise_{prepared['config_hash']}"
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
        "next_if_positive": (
            "compare directly with the rank-4 predecessor, then run controls"
        ),
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
    print("\nM1@64 대비 변화:")
    print(comparison.to_string(index=False))
    print("\n탐색 판독:", reading)
    print("결과 파일:", frame.attrs["result_paths"])
    return frame


if __name__ == "__main__":
    print(
        json.dumps(
            preflight_summary(configure_neighbor_conditioned_featurewise_run()),
            ensure_ascii=False,
            indent=2,
        )
    )
