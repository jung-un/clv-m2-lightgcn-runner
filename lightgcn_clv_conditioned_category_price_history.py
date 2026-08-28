"""Seed-42 historical screen for a CLV-conditioned category/price history M2."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from clv_conditioned_category_price_history_model import (
    ConditionedCategoryPriceHistoryLightGCN,
    build_conditioned_history_features,
)
from clv_run_state import ProgressStore, RunIdentity, clone_state, file_sha256
import lightgcn_clv_axis_specific_test10 as test10
import lightgcn_clv_gatefree_lowdim as gatefree
import lightgcn_clv_joint_nv as joint
import lightgcn_clv_moe as moe
import lightgcn_clv_residual as residual
import lightgcn_clv_v3 as v3


CODE_VERSION = "m2-clv-conditioned-category-price-history-historical-screen-v1"
MODEL_ID = "m2_clv_conditioned_category_price_history"


@dataclass(frozen=True)
class ConditionedHistoryConfig:
    dataset: str = "dunnhumby"
    seed: int = 42
    time_cutoff: int = 690
    evaluation_days: int = 7
    epochs: int = 100
    id_dim: int = 64
    category_dim: int = 4
    rho: float = 0.1
    n_layers: int = 2
    batch_size: int = 8192
    lr: float = 5e-4
    pref_reg: float = 1e-3
    input_days: int = 365
    out_dir: str = ""
    baseline_result_dir: str = ""


def configure_conditioned_history_run(**overrides) -> ConditionedHistoryConfig:
    defaults = {
        "out_dir": (
            f"{v3.default_out_dir('dunnhumby')}"
            "_m2_clv_conditioned_category_price_history_historical_screen_v1"
        ),
        "baseline_result_dir": (
            f"{v3.default_out_dir('dunnhumby')}"
            "_m2_repeatshare_historical_backtest_v1"
        ),
    }
    return validate_config(ConditionedHistoryConfig(**(defaults | overrides)))


def validate_config(cfg: ConditionedHistoryConfig) -> ConditionedHistoryConfig:
    required = {
        "dataset": "dunnhumby",
        "seed": 42,
        "time_cutoff": 690,
        "evaluation_days": 7,
        "epochs": 100,
        "id_dim": 64,
        "category_dim": 4,
        "rho": 0.1,
        "n_layers": 2,
        "input_days": 365,
    }
    for key, expected in required.items():
        if getattr(cfg, key) != expected:
            raise ValueError(f"고정 M2 screen은 {key}={expected!r}이어야 합니다")
    if cfg.batch_size <= 0 or cfg.lr <= 0 or cfg.pref_reg < 0:
        raise ValueError("학습 설정이 잘못됐습니다")
    if not cfg.out_dir or not cfg.baseline_result_dir:
        raise ValueError("out_dir와 baseline_result_dir가 필요합니다")
    return cfg


def preflight_summary(cfg: ConditionedHistoryConfig) -> dict:
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
            "architecture": "LightGCN ID(64)|one conditioned category-price history block(8)",
            "historical_clv_proxy": "N_hat times V_hat from train-only behavior composites",
            "n_behavior_inputs": [
                "observation-adjusted repeat transaction rate",
                "recent continuation",
                "inverse mean transaction gap",
            ],
            "v_behavior_inputs": [
                "mean transaction amount",
                "transaction-averaged within-category price position",
            ],
            "condition": "[N_hat,V_hat,N_hat*V_hat,N_hat-V_hat]",
            "condition_mixer": "one 2-way softmax with 10 trainable parameters",
            "history": "distinct-item category distribution and category-specific price position",
            "candidate": "category and train-only within-category price position",
            "free_auxiliary_item_embedding": False,
            "raw_item_popularity_input": False,
            "raw_repeatshare_input": False,
            "positive_item_leave_one_out": "exact under the fixed two-layer linear propagation",
            "one_item_user_auxiliary_disabled_train_and_eval": True,
            "rho": cfg.rho,
            "learned_global_axis_weight": False,
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
            "historical development screen only; a positive result still requires "
            "M1@72 and shuffled-user controls before CLV attribution"
        ),
        "out_dir": cfg.out_dir,
        "baseline_result_dir": cfg.baseline_result_dir,
    }


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def _base_config(cfg: ConditionedHistoryConfig) -> dict:
    return gatefree._base_config(cfg)


def _config_hash(cfg: ConditionedHistoryConfig, input_hash: str, revision: str) -> str:
    payload = {
        "code_version": CODE_VERSION,
        "config": asdict(cfg),
        "input_hash": input_hash,
        "source_revision": revision,
    }
    return hashlib.sha256(_canonical(payload).encode()).hexdigest()[:12]


def _prepare(cfg: ConditionedHistoryConfig) -> dict:
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

    features = build_conditioned_history_features(
        data["train"],
        n_users=data["n_users"],
        n_items=data["n_items"],
        n_categories=data["n_cat"],
        is_date=v3.DCFG["is_date"],
    )
    # Evaluation segments stay identical to the stored M1 comparator.  The new
    # behavior composite is a model input, not a post-hoc redefinition of M1.
    snapshot = residual.build_final_snapshot(
        data["train"], data["n_users"], v3.DCFG["is_date"], cfg.input_days
    )
    evaluation_axes = joint.build_user_axis_inputs(snapshot, data["n_users"])
    baseline = gatefree._load_compatible_baseline(cfg, manifest)
    meta = v3.item_meta(data["train"], data["n_items"])
    thresholds = v3.segment_thresholds(
        evaluation_axes["clv_proxy"], base_cfg["SEG_EDGES"]
    )
    cache = v3.EvalCache(
        *data["splits"]["test"],
        evaluation_axes["clv_proxy"],
        thresholds,
        data["n_items"],
    )
    prepared = {
        "out_dir": out_dir,
        "manifest": manifest,
        "input_hash": input_hash,
        "revision": revision,
        "base_cfg": base_cfg,
        "data": data,
        "features": features,
        "evaluation_axes": evaluation_axes,
        "baseline": baseline,
        "meta": meta,
        "cache": cache,
    }
    prepared["config_hash"] = _config_hash(cfg, input_hash, revision)
    return prepared


def _build_model(prepared: dict, cfg: ConditionedHistoryConfig):
    data = prepared["data"]
    edge_key = np.asarray(data["pos_key"], dtype=np.int64)
    edge_users = edge_key // data["n_items"]
    edge_items = edge_key % data["n_items"]
    v3.set_seed(cfg.seed)
    model = ConditionedCategoryPriceHistoryLightGCN(
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
        pref_reg=cfg.pref_reg,
    ).to(v3.DEVICE)
    return model, list(model.parameters())


def _arm_hash(prepared: dict, cfg: ConditionedHistoryConfig) -> str:
    return hashlib.sha256(
        _canonical(
            {"run": prepared["config_hash"], "model_id": MODEL_ID, "seed": cfg.seed}
        ).encode()
    ).hexdigest()[:12]


class _IDOnlyView(nn.Module):
    def __init__(self, model: ConditionedCategoryPriceHistoryLightGCN):
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


def _run_model(prepared: dict, cfg: ConditionedHistoryConfig) -> dict:
    root = prepared["out_dir"] / "arms" / prepared["config_hash"]
    result_path = root / f"{MODEL_ID}_s{cfg.seed}.json"
    checkpoint_path = root / f"{MODEL_ID}_s{cfg.seed}.pt"
    if result_path.exists():
        print("  [cached] CLV 조건부 상품군·가격 구매이력 M2 완료 결과 재사용")
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


def _comparison(
    baseline: dict,
    arm: dict,
    *,
    reference_label: str = "m1_64",
) -> pd.DataFrame:
    rows = []
    for metric, model_value in arm["metrics"].items():
        if metric not in baseline or metric.startswith(("저CLV_", "중CLV_", "고CLV_")):
            continue
        reference_value = baseline[metric]
        if not isinstance(reference_value, (int, float, np.number)) or not isinstance(
            model_value, (int, float, np.number)
        ):
            continue
        rows.append(
            {
                "metric": metric,
                reference_label: reference_value,
                MODEL_ID: model_value,
                "absolute_delta": model_value - reference_value,
                "relative_change_pct": (
                    100.0 * (model_value - reference_value) / reference_value
                    if reference_value != 0
                    else None
                ),
            }
        )
    return pd.DataFrame(rows)


def run_conditioned_history_screen(
    cfg: ConditionedHistoryConfig | None = None,
) -> pd.DataFrame:
    cfg = validate_config(cfg or configure_conditioned_history_run())
    print(json.dumps(preflight_summary(cfg), ensure_ascii=False, indent=2))
    prepared = _prepare(cfg)
    print("\ntrain-only N/V·상품군·가격 입력 진단:")
    print(json.dumps(prepared["features"].diagnostics, ensure_ascii=False, indent=2))
    print("\n===== 수정 M2만 학습 | seed 42 | fixed 100 epochs =====")
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
    comparison = _comparison(baseline, arm)
    id_only_comparison = _comparison(
        arm["id_only_metrics"], arm, reference_label="jointly_trained_id_only"
    )
    stem = f"m2_clv_conditioned_category_price_history_{prepared['config_hash']}"
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
    all_accuracy_improved = bool(
        accuracy_names
        and all(
            metric in metric_index.index
            and metric_index.at[metric, "absolute_delta"] > 0
            for metric in accuracy_names
        )
    )
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
        "strong_positive_screen": bool(economic_improved and all_accuracy_improved),
        "accuracy_ratios": accuracy_ratios,
        "next_if_positive": "run M1@72, shuffled-user, multi-seed, then H&M 2y",
        "statistical_note": "seed 42 exploratory screen; no significance claim",
        "protocol_note": "final test and holdout were not constructed",
    }
    payload = {
        "code_version": CODE_VERSION,
        "source_revision": prepared["revision"],
        "config": asdict(cfg),
        "preflight": preflight_summary(cfg),
        "input_manifest": prepared["manifest"],
        "data_stats": prepared["data"].get("data_stats", {}),
        "reused_baseline_source": baseline["source_result"],
        "feature_diagnostics": prepared["features"].diagnostics,
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
            preflight_summary(configure_conditioned_history_run()),
            ensure_ascii=False,
            indent=2,
        )
    )
