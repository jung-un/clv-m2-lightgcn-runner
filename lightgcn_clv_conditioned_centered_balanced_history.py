"""Seed-42 historical screen for centered, balanced CLV-conditioned history."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

from clv_conditioned_centered_balanced_history_model import (
    CenteredBalancedHistoryLightGCN,
)
from clv_run_state import ProgressStore, RunIdentity, clone_state, file_sha256
import lightgcn_clv_axis_specific_test10 as test10
import lightgcn_clv_conditioned_category_price_history as base
import lightgcn_clv_moe as moe
import lightgcn_clv_v3 as v3


CODE_VERSION = "m2-clv-conditioned-centered-balanced-history-historical-screen-v1"
MODEL_ID = "m2_clv_conditioned_centered_balanced_history"


@dataclass(frozen=True)
class CenteredBalancedHistoryConfig:
    dataset: str = "dunnhumby"
    seed: int = 42
    time_cutoff: int = 690
    evaluation_days: int = 7
    epochs: int = 100
    id_dim: int = 64
    category_dim: int = 4
    rho: float = 0.1
    warmup_epochs: int = 20
    n_layers: int = 2
    batch_size: int = 8192
    lr: float = 5e-4
    pref_reg: float = 1e-3
    input_days: int = 365
    out_dir: str = ""
    baseline_result_dir: str = ""


def configure_centered_balanced_history_run(
    **overrides,
) -> CenteredBalancedHistoryConfig:
    defaults = {
        "out_dir": (
            f"{v3.default_out_dir('dunnhumby')}"
            "_m2_clv_conditioned_centered_balanced_history_historical_screen_v1"
        ),
        "baseline_result_dir": (
            f"{v3.default_out_dir('dunnhumby')}"
            "_m2_repeatshare_historical_backtest_v1"
        ),
    }
    return validate_config(CenteredBalancedHistoryConfig(**(defaults | overrides)))


def validate_config(
    cfg: CenteredBalancedHistoryConfig,
) -> CenteredBalancedHistoryConfig:
    fixed = {
        "dataset": "dunnhumby",
        "seed": 42,
        "time_cutoff": 690,
        "evaluation_days": 7,
        "epochs": 100,
        "id_dim": 64,
        "category_dim": 4,
        "rho": 0.1,
        "warmup_epochs": 20,
        "n_layers": 2,
        "input_days": 365,
    }
    for key, expected in fixed.items():
        if getattr(cfg, key) != expected:
            raise ValueError(f"고정 M2 screen은 {key}={expected!r}이어야 합니다")
    if cfg.batch_size <= 0 or cfg.lr <= 0 or cfg.pref_reg < 0:
        raise ValueError("학습 설정이 잘못됐습니다")
    if not cfg.out_dir or not cfg.baseline_result_dir:
        raise ValueError("out_dir와 baseline_result_dir가 필요합니다")
    return cfg


def preflight_summary(cfg: CenteredBalancedHistoryConfig) -> dict:
    cfg = validate_config(cfg)
    return {
        "code_version": CODE_VERSION,
        "dataset": cfg.dataset,
        "seed": cfg.seed,
        "trained_models": [MODEL_ID],
        "reused_comparator": "m1_64",
        "research_axis": "M2 representation intervention",
        "historical_development_split": {
            "train_end_inclusive": 683,
            "evaluation_start_inclusive": 684,
            "evaluation_end_inclusive": 690,
            "final_test_constructed": False,
            "holdout_constructed": False,
        },
        "m2": {
            "architecture": "LightGCN ID(64)|centered category(4)|centered price(4)",
            "historical_clv_proxy": "N_hat times V_hat from train-only behavior composites",
            "condition": "[N_hat,V_hat,N_hat*V_hat,N_hat-V_hat]",
            "condition_mixer": (
                "category=0.25+0.5*sigmoid(w^T condition), price=1-category"
            ),
            "mixer_bounds": [0.25, 0.75],
            "history_normalization": (
                "valid-user population centering followed by separate row L2 normalization"
            ),
            "positive_item_leave_one_out": (
                "exact centered profile replacement under fixed two-layer linear propagation"
            ),
            "rho_max": cfg.rho,
            "rho_warmup_epochs": cfg.warmup_epochs,
            "learned_global_axis_weight": False,
            "raw_item_popularity_input": False,
            "raw_repeatshare_input": False,
            "total_dim": cfg.id_dim + 2 * cfg.category_dim,
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
            "historical development screen only; stop this category/price-history "
            "family if it does not beat the reused M1 comparator"
        ),
        "out_dir": cfg.out_dir,
        "baseline_result_dir": cfg.baseline_result_dir,
    }


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def _config_hash(cfg, input_hash: str, revision: str) -> str:
    payload = {
        "code_version": CODE_VERSION,
        "config": asdict(cfg),
        "input_hash": input_hash,
        "source_revision": revision,
    }
    return hashlib.sha256(_canonical(payload).encode()).hexdigest()[:12]


def _prepare(cfg: CenteredBalancedHistoryConfig) -> dict:
    prepared = base._prepare(cfg)
    prepared["config_hash"] = _config_hash(
        cfg, prepared["input_hash"], prepared["revision"]
    )
    return prepared


def _build_model(prepared: dict, cfg: CenteredBalancedHistoryConfig):
    data = prepared["data"]
    edge_key = data["pos_key"]
    edge_users = edge_key // data["n_items"]
    edge_items = edge_key % data["n_items"]
    v3.set_seed(cfg.seed)
    model = CenteredBalancedHistoryLightGCN(
        n_users=data["n_users"],
        n_items=data["n_items"],
        n_categories=data["n_cat"],
        features=prepared["features"],
        edge_users=edge_users,
        edge_items=edge_items,
        adj=data["adj"],
        id_dim=cfg.id_dim,
        category_dim=cfg.category_dim,
        n_layers=cfg.n_layers,
        rho=cfg.rho,
        warmup_epochs=cfg.warmup_epochs,
        pref_reg=cfg.pref_reg,
    ).to(v3.DEVICE)
    return model, list(model.parameters())


def _arm_hash(prepared: dict, cfg: CenteredBalancedHistoryConfig) -> str:
    return hashlib.sha256(
        _canonical(
            {"run": prepared["config_hash"], "model_id": MODEL_ID, "seed": cfg.seed}
        ).encode()
    ).hexdigest()[:12]


class _IDOnlyView(nn.Module):
    def __init__(self, model: CenteredBalancedHistoryLightGCN):
        super().__init__()
        self.model = model

    def embeddings(self, need_value: bool = True):
        user, item = self.model.id_only_embeddings()
        return (
            user,
            item,
            user.new_zeros((len(user), 1)),
            item.new_zeros((len(item), 1)),
        )


def _run_model(prepared: dict, cfg: CenteredBalancedHistoryConfig) -> dict:
    root = prepared["out_dir"] / "arms" / prepared["config_hash"]
    result_path = root / f"{MODEL_ID}_s{cfg.seed}.json"
    checkpoint_path = root / f"{MODEL_ID}_s{cfg.seed}.pt"
    if result_path.exists():
        print("  [cached] 중심화·균형 조건부 M2 완료 결과 재사용")
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
    model.set_training_epoch(cfg.epochs)
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
    id_only_metrics, _ = moe._flat_evaluation(
        _IDOnlyView(model),
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
        "id_only_metrics": test10._public_metrics(id_only_metrics),
        "diagnostics": model.representation_diagnostics(),
        "feature_diagnostics": prepared["features"].diagnostics,
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


def _comparison(reference: dict, arm: dict, label: str) -> pd.DataFrame:
    frame = base._comparison(reference, arm, reference_label=label)
    return frame.rename(columns={base.MODEL_ID: MODEL_ID})


def run_centered_balanced_history_screen(
    cfg: CenteredBalancedHistoryConfig | None = None,
) -> pd.DataFrame:
    cfg = validate_config(cfg or configure_centered_balanced_history_run())
    print(json.dumps(preflight_summary(cfg), ensure_ascii=False, indent=2))
    prepared = _prepare(cfg)
    print("\n===== 중심화·균형 조건부 M2 | seed 42 | fixed 100 epochs =====")
    arm = _run_model(prepared, cfg)

    baseline = dict(prepared["baseline"])
    baseline["role"] = "reused_baseline"
    model_row = {
        "model_id": arm["model_id"],
        "role": arm["role"],
        "seed": arm["seed"],
        "split": arm["split"],
        "final_epoch": arm["final_epoch"],
        **arm["diagnostics"],
        **arm["metrics"],
    }
    frame = pd.DataFrame([baseline, model_row])
    comparison = _comparison(baseline, arm, "m1_64")
    id_only_comparison = _comparison(
        arm["id_only_metrics"], arm, "jointly_trained_id_only"
    )
    stem = f"m2_clv_conditioned_centered_balanced_history_{prepared['config_hash']}"
    paths = {
        "absolute_csv": prepared["out_dir"] / f"{stem}.csv",
        "comparison_csv": prepared["out_dir"] / f"{stem}_comparison.csv",
        "id_only_comparison_csv": prepared["out_dir"]
        / f"{stem}_id_only_comparison.csv",
        "json": prepared["out_dir"] / f"{stem}.json",
    }
    test10._atomic_csv(paths["absolute_csv"], frame)
    test10._atomic_csv(paths["comparison_csv"], comparison)
    test10._atomic_csv(paths["id_only_comparison_csv"], id_only_comparison)

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
    economic_improved = bool(
        economic_metric in metric_index.index
        and metric_index.at[economic_metric, "absolute_delta"] > 0
    )
    reading = {
        "positive_screen": bool(
            economic_improved
            and accuracy_ratios
            and min(accuracy_ratios.values()) >= 0.99
        ),
        "accuracy_ratios": accuracy_ratios,
        "next_if_positive": "run matched-capacity and shuffled-condition controls",
        "next_if_negative": "stop this category/price-history representation family",
        "statistical_note": "seed 42 exploratory screen; no significance claim",
        "protocol_note": "final test and holdout were not constructed",
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
        "id_only_metrics": arm["id_only_metrics"],
        "id_only_comparison_rows": id_only_comparison.to_dict("records"),
        "screening_reading": reading,
        "result_paths": {key: str(value) for key, value in paths.items()},
    }
    test10._atomic_json(paths["json"], payload)
    frame.attrs["comparison"] = comparison
    frame.attrs["id_only_comparison"] = id_only_comparison
    frame.attrs["screening_reading"] = reading
    frame.attrs["result_paths"] = {key: str(value) for key, value in paths.items()}
    print("\n절대지표:")
    print(frame.to_string(index=False))
    print("\nM1@64 대비 변화:")
    print(comparison.to_string(index=False))
    print("\n공동학습 ID-only 대비 full 변화:")
    print(id_only_comparison.to_string(index=False))
    print("\n탐색 판독:", reading)
    print("결과 파일:", frame.attrs["result_paths"])
    return frame


if __name__ == "__main__":
    print(
        json.dumps(
            preflight_summary(configure_centered_balanced_history_run()),
            ensure_ascii=False,
            indent=2,
        )
    )
