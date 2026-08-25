"""Fast seed-42 historical screen of a purchase-history conditioned M2.

Only the new M2 arm is trained.  The already saved, protocol-compatible
M1@64 result is reused for a quick performance comparison.  This screen does
not establish attribution because the matched history-only and shuffled-N/V
controls are deliberately deferred until a positive result is observed.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import torch

from clv_history_conditioned_lowrank_model import (
    CLVHistoryConditionedLowRankLightGCN,
)
from clv_run_state import ProgressStore, RunIdentity, clone_state, file_sha256
import lightgcn_clv_axis_specific_test10 as test10
import lightgcn_clv_conditional_id_transform as shared
import lightgcn_clv_moe as moe
import lightgcn_clv_v3 as v3


CODE_VERSION = "m2-history-conditioned-lowrank-historical-screen-v1"
MODEL_ID = "m2_history_conditioned_lowrank_transform"


@dataclass(frozen=True)
class HistoryConditionedLowRankConfig:
    dataset: str = "dunnhumby"
    seed: int = 42
    time_cutoff: int = 690
    evaluation_days: int = 7
    epochs: int = 100
    embedding_dim: int = 64
    transform_rank: int = 4
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


def configure_history_conditioned_lowrank_run(
    **overrides,
) -> HistoryConditionedLowRankConfig:
    defaults = {
        "out_dir": (
            f"{v3.default_out_dir('dunnhumby')}"
            "_m2_history_conditioned_lowrank_historical_screen_v1"
        ),
        "baseline_result_dir": (
            f"{v3.default_out_dir('dunnhumby')}"
            "_m2_repeatshare_historical_backtest_v1"
        ),
    }
    return validate_config(
        HistoryConditionedLowRankConfig(**(defaults | overrides))
    )


def validate_config(
    cfg: HistoryConditionedLowRankConfig,
) -> HistoryConditionedLowRankConfig:
    required = {
        "dataset": "dunnhumby",
        "seed": 42,
        "time_cutoff": 690,
        "evaluation_days": 7,
        "epochs": 100,
        "embedding_dim": 64,
        "transform_rank": 4,
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


def preflight_summary(cfg: HistoryConditionedLowRankConfig) -> dict:
    cfg = validate_config(cfg)
    return {
        "code_version": CODE_VERSION,
        "dataset": cfg.dataset,
        "seed": cfg.seed,
        "trained_models": [MODEL_ID],
        "reused_comparator": "m1_64",
        "controls_trained": [],
        "historical_development_split": {
            "train_end_inclusive": 683,
            "evaluation_start_inclusive": 684,
            "evaluation_end_inclusive": 690,
            "original_validation_test_holdout_constructed": False,
        },
        "m2": {
            "architecture": (
                "purchase-history user representation with centred fixed "
                "N/V-conditioned rank-4 maps"
            ),
            "conditional_map": "[I + rho(c_N A_N + c_V A_V)] H_u",
            "user_base": "normalized_purchase_history",
            "free_user_id_embedding": False,
            "embedding_dim": cfg.embedding_dim,
            "transform_rank": cfg.transform_rank,
            "rho": cfg.rho,
            "activity_condition": (
                "train_percentile(repeat_transaction_rate) minus train mean"
            ),
            "value_condition": (
                "train_percentile(mean_transaction_value) minus train mean"
            ),
            "condition_statistics": "computed once on train users and frozen",
            "both_axes_present": True,
            "identity_base_map": True,
            "norm_preserving": True,
            "explicit_item_features": False,
            "separate_axis_scores": False,
            "external_reranking": False,
            "structural_property": "H_u=0 implies E_u^(0)=0",
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
        "reading_rule": {
            "accuracy": "each Recall/NDCG@10/20/50 >= 99% of M1@64",
            "economic": "price_purchase_amount_weighted_hit@10 > M1@64",
            "significance": "not claimed from one exploratory seed",
            "attribution": (
                "not claimed without a matched history-only and shuffled-N/V control"
            ),
        },
        "out_dir": cfg.out_dir,
        "baseline_result_dir": cfg.baseline_result_dir,
    }


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def _config_hash(
    cfg: HistoryConditionedLowRankConfig, input_hash: str, revision: str
) -> str:
    payload = {
        "code_version": CODE_VERSION,
        "config": asdict(cfg),
        "model_id": MODEL_ID,
        "input_hash": input_hash,
        "source_revision": revision,
    }
    return hashlib.sha256(_canonical(payload).encode()).hexdigest()[:12]


def _base_config(cfg: HistoryConditionedLowRankConfig) -> dict:
    return shared._base_config(cfg)


def _prepare(cfg: HistoryConditionedLowRankConfig) -> dict:
    prepared = shared._prepare(cfg)
    prepared["config_hash"] = _config_hash(
        cfg, prepared["input_hash"], prepared["revision"]
    )
    return prepared


def _build_model(prepared: dict, cfg: HistoryConditionedLowRankConfig):
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


def _arm_hash(prepared: dict, cfg: HistoryConditionedLowRankConfig) -> str:
    return hashlib.sha256(
        _canonical(
            {
                "run": prepared["config_hash"],
                "model_id": MODEL_ID,
                "seed": cfg.seed,
            }
        ).encode()
    ).hexdigest()[:12]


def _run_model(prepared: dict, cfg: HistoryConditionedLowRankConfig) -> dict:
    root = prepared["out_dir"] / "arms" / prepared["config_hash"]
    result_path = root / f"{MODEL_ID}_s{cfg.seed}.json"
    checkpoint_path = root / f"{MODEL_ID}_s{cfg.seed}.pt"
    if result_path.exists():
        print("  [cached] 구매이력 조건부 M2 seed 42 완료 결과 재사용")
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


def _comparison(baseline: dict, arm: dict) -> pd.DataFrame:
    rows = []
    for metric, model_value in arm["metrics"].items():
        if metric not in baseline:
            continue
        reference_value = baseline[metric]
        if not isinstance(reference_value, (int, float, np.number)) or not isinstance(
            model_value, (int, float, np.number)
        ):
            continue
        delta = float(model_value) - float(reference_value)
        rows.append(
            {
                "metric": metric,
                "m1_64": reference_value,
                MODEL_ID: model_value,
                "absolute_delta": delta,
                "relative_change_pct": (
                    100.0 * delta / float(reference_value)
                    if float(reference_value) != 0
                    else None
                ),
            }
        )
    return pd.DataFrame(rows)


def run_history_conditioned_lowrank_screen(
    cfg: HistoryConditionedLowRankConfig | None = None,
) -> pd.DataFrame:
    cfg = validate_config(cfg or configure_history_conditioned_lowrank_run())
    print(json.dumps(preflight_summary(cfg), ensure_ascii=False, indent=2))
    prepared = _prepare(cfg)
    print("\n===== 새 M2만 학습 | seed 42 | fixed 100 epochs =====")
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
    comparison = _comparison(baseline, arm)
    stem = f"m2_history_conditioned_lowrank_{prepared['config_hash']}"
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
            "run a matched history-only capacity control and shuffled N/V, then seeds"
        ),
        "statistical_note": "seed 42 exploratory screen; no significance claim",
        "attribution_note": (
            "comparison with reused M1 tests performance only, not CLV attribution"
        ),
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
            preflight_summary(configure_history_conditioned_lowrank_run()),
            ensure_ascii=False,
            indent=2,
        )
    )
