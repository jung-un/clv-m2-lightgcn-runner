"""Seed-42 screen changing only the item side of the constrained M2."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from clv_constrained_economic_embedding_model import (
    ConstrainedCLVEconomicLightGCN,
)
from clv_run_state import ProgressStore, RunIdentity, clone_state, file_sha256
import lightgcn_clv_axis_specific_test10 as test10
import lightgcn_clv_gradient_isolated_economic_interaction as helpers
import lightgcn_clv_joint_response_embedding as shared
import lightgcn_clv_moe as moe
import lightgcn_clv_v3 as v3


CODE_VERSION = "m2-hybrid-item-clv-economic-embedding-historical-screen-v2"
MATCHED_MODEL_ID = "m1_matched_rho0"
MODEL_ID = "m2_hybrid_item_clv_economic_embedding"
ID_ONLY_MODEL_ID = "m2_jointly_trained_id_only"
RELATION_ONLY_MODEL_ID = "m2_id_plus_item_relation_only"
PRICE_ONLY_MODEL_ID = "m2_id_plus_item_price_only"


@dataclass(frozen=True)
class ConstrainedEconomicConfig:
    dataset: str = "dunnhumby"
    seed: int = 42
    time_cutoff: int = 690
    evaluation_days: int = 7
    epochs: int = 100
    id_dim: int = 64
    clv_dim: int = 4
    rho: float = 0.05
    item_price_budget: float = 0.25
    n_layers: int = 2
    batch_size: int = 8192
    lr: float = 5e-4
    pref_reg: float = 1e-3
    input_days: int = 365
    diagnostic_max_k: int = 50
    out_dir: str = ""
    baseline_result_dir: str = ""


def configure_constrained_economic_run(**overrides) -> ConstrainedEconomicConfig:
    defaults = {
        "out_dir": (
            f"{v3.default_out_dir('dunnhumby')}"
            "_m2_hybrid_item_clv_economic_embedding_historical_screen_v2"
        ),
        "baseline_result_dir": (
            f"{v3.default_out_dir('dunnhumby')}"
            "_m2_repeatshare_historical_backtest_v1"
        ),
    }
    return validate_constrained_economic_config(
        ConstrainedEconomicConfig(**(defaults | overrides))
    )


def validate_constrained_economic_config(
    cfg: ConstrainedEconomicConfig,
) -> ConstrainedEconomicConfig:
    required = {
        "dataset": "dunnhumby",
        "seed": 42,
        "time_cutoff": 690,
        "evaluation_days": 7,
        "epochs": 100,
        "id_dim": 64,
        "clv_dim": 4,
        "rho": 0.05,
        "item_price_budget": 0.25,
        "n_layers": 2,
        "input_days": 365,
        "diagnostic_max_k": 50,
    }
    for key, expected in required.items():
        if getattr(cfg, key) != expected:
            raise ValueError(f"빠른 M2 screen은 {key}={expected!r}이어야 합니다")
    if cfg.batch_size <= 0 or cfg.lr <= 0 or cfg.pref_reg < 0:
        raise ValueError("학습 설정이 잘못됐습니다")
    if not cfg.out_dir or not cfg.baseline_result_dir:
        raise ValueError("out_dir와 baseline_result_dir가 필요합니다")
    return cfg


def preflight_summary(cfg: ConstrainedEconomicConfig) -> dict:
    cfg = validate_constrained_economic_config(cfg)
    return {
        "code_version": CODE_VERSION,
        "dataset": cfg.dataset,
        "seed": cfg.seed,
        "trained_models": [MATCHED_MODEL_ID, MODEL_ID],
        "reused_comparator": "m1_64 (display only)",
        "historical_development_split": {
            "train_end_inclusive": 683,
            "evaluation_start_inclusive": 684,
            "evaluation_end_inclusive": 690,
            "final_test_constructed": False,
            "holdout_constructed": False,
        },
        "m2": {
            "architecture": "ID(64)|one CLV-conditioned hybrid item block(4)",
            "total_dim": cfg.id_dim + cfg.clv_dim,
            "user_block": "q_C * unit(W_u[q_N,q_V])",
            "item_block": "[sqrt(1-beta)*unit(P_z E_i^ID)(2)|sqrt(beta)*centred prices(2)]",
            "user_tanh": False,
            "free_item_response_embedding": False,
            "item_inputs": [
                "existing item ID embedding projected to 2 dimensions",
                "overall price percentile",
                "within-category price percentile",
            ],
            "item_price_budget": cfg.item_price_budget,
            "joint_graph_propagation": True,
            "one_dot_score": True,
            "rho": cfg.rho,
            "symmetric_scale": "sqrt(rho) on user and item CLV blocks",
            "repeatshare_input": False,
            "item_popularity_input": False,
            "external_reranking": False,
        },
        "fixed": {
            "graph": "binary",
            "negative_sampling": "uniform",
            "sample_weighting": False,
            "loss": "one plain pairwise BPR plus existing sampled ID L2",
            "new_loss_term": False,
            "one_training_loop_and_optimizer": True,
            "min_user_interactions": 1,
            "min_item_interactions": 1,
            "epochs": cfg.epochs,
            "validation_or_epoch_selection": False,
        },
        "reading_rule": {
            "baseline_accuracy": "all Recall/NDCG@10/20/50 >= 99% of matched rho=0",
            "direct_clv": "full must beat jointly-trained ID-only on high-CLV Recall/NDCG@10 and weighted hit@10",
            "mechanism": "report ID-only, relation-only, price-only, full and Top-10 changes",
            "statistical_note": "seed 42 exploratory screen; no significance claim",
        },
        "out_dir": cfg.out_dir,
        "baseline_result_dir": cfg.baseline_result_dir,
    }


def _config_hash(
    cfg: ConstrainedEconomicConfig, input_hash: str, revision: str
) -> str:
    payload = {
        "code_version": CODE_VERSION,
        "config": asdict(cfg),
        "input_hash": input_hash,
        "source_revision": revision,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()[:12]


def _prepare(cfg: ConstrainedEconomicConfig) -> dict:
    prepared = shared._prepare(cfg)
    prepared["config_hash"] = _config_hash(
        cfg, prepared["input_hash"], prepared["revision"]
    )
    return prepared


def _build_model(prepared: dict, cfg: ConstrainedEconomicConfig, rho: float):
    data = prepared["data"]
    v3.set_seed(cfg.seed)
    model = ConstrainedCLVEconomicLightGCN(
        n_users=data["n_users"],
        n_items=data["n_items"],
        q_n=prepared["q_n"],
        q_v=prepared["q_v"],
        q_c=prepared["q_c"],
        user_clv_valid=prepared["clv_valid"],
        item_economic_features=prepared["item_economic"],
        item_economic_valid=prepared["item_economic_valid"],
        adj=data["adj"],
        id_dim=cfg.id_dim,
        clv_dim=cfg.clv_dim,
        rho=rho,
        item_price_budget=cfg.item_price_budget,
        n_layers=cfg.n_layers,
        pref_reg=cfg.pref_reg,
    ).to(v3.DEVICE)
    return model, list(model.parameters())


class _ComponentView(torch.nn.Module):
    def __init__(self, parent: ConstrainedCLVEconomicLightGCN, component: str):
        super().__init__()
        self.parent = parent
        self.component = component

    def embeddings(self, need_value: bool = True):
        user, item = self.parent.component_embeddings(self.component)
        zero_user = user.new_zeros((self.parent.n_users, 1))
        zero_item = item.new_zeros((self.parent.n_items, 1))
        return user, item, zero_user, zero_item


def _arm_paths(prepared: dict, model_id: str) -> dict[str, Path]:
    root = prepared["out_dir"] / "arms" / prepared["config_hash"]
    stem = f"{model_id}_s42"
    return {"result": root / f"{stem}.json", "checkpoint": root / f"{stem}.pt"}


def _arm_hash(prepared: dict, model_id: str, rho: float) -> str:
    payload = {
        "run": prepared["config_hash"],
        "model_id": model_id,
        "rho": rho,
        "seed": 42,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()[:12]


def _run_arm(
    prepared: dict,
    cfg: ConstrainedEconomicConfig,
    *,
    model_id: str,
    rho: float,
) -> tuple[dict, ConstrainedCLVEconomicLightGCN]:
    paths = _arm_paths(prepared, model_id)
    model, params = _build_model(prepared, cfg, rho)
    if paths["result"].exists() and paths["checkpoint"].exists():
        print(f"  [cached] {model_id} 완료 결과 재사용")
        payload = json.loads(paths["result"].read_text(encoding="utf-8"))
        checkpoint = helpers._load_state(paths["checkpoint"])
        if checkpoint.get("input_hash") != prepared["input_hash"]:
            raise RuntimeError("cached checkpoint와 현재 입력 hash가 다릅니다")
        model.load_state_dict(checkpoint["state"], strict=True)
        model.eval()
        return payload, model

    store = ProgressStore(
        prepared["out_dir"] / "progress" / prepared["config_hash"],
        RunIdentity(
            stage="historical_development_train",
            model_id=model_id,
            seed=cfg.seed,
            config_hash=_arm_hash(prepared, model_id, rho),
            source_revision=prepared["revision"],
            input_hash=prepared["input_hash"],
        ),
    )
    training = test10._fixed_epoch_train(
        model, params, prepared, cfg, model_id, cfg.seed, store
    )
    model.eval()
    paths["checkpoint"].parent.mkdir(parents=True, exist_ok=True)
    temporary = paths["checkpoint"].with_suffix(".pt.tmp")
    torch.save(
        {
            "state": clone_state(model),
            "model_id": model_id,
            "rho": rho,
            "config": asdict(cfg),
            "training": training,
            "source_revision": prepared["revision"],
            "input_hash": prepared["input_hash"],
        },
        temporary,
    )
    os.replace(temporary, paths["checkpoint"])
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
        "model_id": model_id,
        "role": "matched_control" if rho == 0.0 else "model",
        "seed": cfg.seed,
        "split": "historical_development_days_684_690",
        "final_epoch": cfg.epochs,
        "rho": rho,
        "metrics": test10._public_metrics(metrics),
        "diagnostics": model.representation_diagnostics(),
        "training": training,
        "checkpoint": str(paths["checkpoint"]),
        "checkpoint_sha256": file_sha256(paths["checkpoint"]),
    }
    test10._atomic_json(paths["result"], payload)
    store.mark_complete(
        epoch=cfg.epochs,
        max_epoch=cfg.epochs,
        selection="none",
        split="historical_development_days_684_690",
        checkpoint_path=str(paths["checkpoint"]),
        result_path=str(paths["result"]),
    )
    return payload, model


def run_constrained_economic_screen(
    cfg: ConstrainedEconomicConfig | None = None,
) -> pd.DataFrame:
    cfg = validate_constrained_economic_config(
        cfg or configure_constrained_economic_run()
    )
    print(json.dumps(preflight_summary(cfg), ensure_ascii=False, indent=2))
    prepared = _prepare(cfg)
    print("\n===== matched rho=0 | seed 42 | fixed 100 epochs =====")
    matched, matched_model = _run_arm(
        prepared, cfg, model_id=MATCHED_MODEL_ID, rho=0.0
    )
    print("\n===== constrained CLV-economic rho=0.05 | seed 42 =====")
    active, active_model = _run_arm(
        prepared, cfg, model_id=MODEL_ID, rho=cfg.rho
    )

    id_view = shared._IDOnlyView(active_model).to(v3.DEVICE)
    id_metrics_raw, _ = moe._flat_evaluation(
        id_view,
        0.0,
        prepared["cache"],
        prepared["meta"],
        prepared["data"],
        prepared["base_cfg"],
        per_user=False,
    )
    id_metrics = test10._public_metrics(id_metrics_raw)
    component_metrics = {}
    for component, model_id in (
        ("relation", RELATION_ONLY_MODEL_ID),
        ("price", PRICE_ONLY_MODEL_ID),
    ):
        view = _ComponentView(active_model, component).to(v3.DEVICE)
        raw_metrics, _ = moe._flat_evaluation(
            view,
            0.0,
            prepared["cache"],
            prepared["meta"],
            prepared["data"],
            prepared["base_cfg"],
            per_user=False,
        )
        component_metrics[model_id] = test10._public_metrics(raw_metrics)
    users, matched_top50 = helpers._masked_topk(
        matched_model, prepared, max_k=cfg.diagnostic_max_k
    )
    active_users, active_top50 = helpers._masked_topk(
        active_model, prepared, max_k=cfg.diagnostic_max_k
    )
    if not np.array_equal(users, active_users):
        raise RuntimeError("matched와 M2 평가 사용자 순서가 다릅니다")
    overlap = helpers.topk_overlap_summary(
        matched_top50, active_top50, prepared["cache"].seg, k=10
    )
    score_diagnostics = shared._score_diagnostics(
        active_model, users, active_top50, prepared
    )

    baseline = dict(prepared["baseline"])
    baseline["role"] = "reused_baseline_display_only"
    rows = [baseline]
    for arm in (matched, active):
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
    rows.append(
        {
            "model_id": ID_ONLY_MODEL_ID,
            "role": "joint_training_ablation",
            "seed": cfg.seed,
            "split": "historical_development_days_684_690",
            "final_epoch": cfg.epochs,
            **id_metrics,
        }
    )
    for model_id, metrics in component_metrics.items():
        rows.append(
            {
                "model_id": model_id,
                "role": "joint_training_ablation",
                "seed": cfg.seed,
                "split": "historical_development_days_684_690",
                "final_epoch": cfg.epochs,
                **metrics,
            }
        )
    frame = pd.DataFrame(rows)
    metric_rows = {
        "m1_64": {
            key: value
            for key, value in baseline.items()
            if "@" in key and isinstance(value, (int, float, np.number))
        },
        MATCHED_MODEL_ID: matched["metrics"],
        MODEL_ID: active["metrics"],
        ID_ONLY_MODEL_ID: id_metrics,
        **component_metrics,
    }
    comparison = helpers._metric_comparison(
        metric_rows, references=(MATCHED_MODEL_ID, "m1_64")
    )
    reading = shared.screening_reading(
        matched["metrics"],
        active["metrics"],
        overlap,
        matched["diagnostics"],
        id_only_metrics=id_metrics,
    )
    direct_metrics = (
        "고CLV_recall@10",
        "고CLV_ndcg@10",
        "price_purchase_amount_weighted_hit@10",
    )
    direct_deltas = {
        metric: float(active["metrics"][metric] - id_metrics[metric])
        for metric in direct_metrics
    }
    reading["direct_clv_deltas_vs_joint_id_only"] = direct_deltas
    reading["direct_clv_positive"] = all(
        delta > 0.0 for delta in direct_deltas.values()
    )
    reading["positive_screen"] = bool(
        reading["positive_screen"] and reading["direct_clv_positive"]
    )

    stem = f"m2_hybrid_item_clv_economic_embedding_{prepared['config_hash']}"
    paths = {
        "absolute_csv": prepared["out_dir"] / f"{stem}.csv",
        "comparison_csv": prepared["out_dir"] / f"{stem}_comparison.csv",
        "top10_overlap_csv": prepared["out_dir"] / f"{stem}_top10_overlap.csv",
        "score_diagnostics_csv": prepared["out_dir"] / f"{stem}_score_diagnostics.csv",
        "json": prepared["out_dir"] / f"{stem}.json",
    }
    test10._atomic_csv(paths["absolute_csv"], frame)
    test10._atomic_csv(paths["comparison_csv"], comparison)
    test10._atomic_csv(paths["top10_overlap_csv"], overlap)
    test10._atomic_csv(paths["score_diagnostics_csv"], score_diagnostics)
    payload = {
        "code_version": CODE_VERSION,
        "source_revision": prepared["revision"],
        "config": asdict(cfg),
        "preflight": preflight_summary(cfg),
        "input_manifest": prepared["manifest"],
        "absolute_rows": frame.to_dict("records"),
        "comparison_rows": comparison.to_dict("records"),
        "top10_overlap_rows": overlap.to_dict("records"),
        "score_diagnostic_rows": score_diagnostics.to_dict("records"),
        "screening_reading": reading,
        "result_paths": {key: str(value) for key, value in paths.items()},
    }
    test10._atomic_json(paths["json"], payload)
    frame.attrs["comparison"] = comparison.to_dict("records")
    frame.attrs["top10_overlap"] = overlap.to_dict("records")
    frame.attrs["score_diagnostics"] = score_diagnostics.to_dict("records")
    frame.attrs["screening_reading"] = reading
    frame.attrs["result_paths"] = {key: str(value) for key, value in paths.items()}

    key_metrics = (
        "recall@10",
        "ndcg@10",
        "recall@20",
        "ndcg@20",
        "recall@50",
        "ndcg@50",
        "price_purchase_amount_weighted_hit@10",
        "고CLV_recall@10",
        "고CLV_ndcg@10",
    )
    key_table = comparison[
        (comparison.reference == MATCHED_MODEL_ID)
        & (comparison.model_id == MODEL_ID)
        & comparison.metric.isin(key_metrics)
    ]
    print("\n절대지표:")
    print(frame.to_string(index=False))
    print("\n동일 초기화 rho=0 대비 핵심 변화:")
    print(key_table.to_string(index=False))
    print("\nTop-10 변경 진단:")
    print(overlap.to_string(index=False))
    print("\n점수 영향력 진단:")
    print(score_diagnostics.to_string(index=False))
    print("\n탐색 판독:", reading)
    print("결과 파일:", frame.attrs["result_paths"])
    return frame


if __name__ == "__main__":
    print(
        json.dumps(
            preflight_summary(configure_constrained_economic_run()),
            ensure_ascii=False,
            indent=2,
        )
    )
