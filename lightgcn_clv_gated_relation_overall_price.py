"""Seed-42 M2 screen with a gated relation block and overall price fit."""

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

from clv_gated_relation_overall_price_model import (
    GatedRelationOverallPriceLightGCN,
)
from clv_run_state import ProgressStore, RunIdentity, clone_state, file_sha256
import lightgcn_clv_axis_specific_test10 as test10
import lightgcn_clv_gatefree_lowdim as gatefree
import lightgcn_clv_gradient_isolated_economic_interaction as evaluation
import lightgcn_clv_joint_nv as joint
import lightgcn_clv_joint_response_embedding as shared
import lightgcn_clv_moe as moe
import lightgcn_clv_residual as residual
import lightgcn_clv_v3 as v3


CODE_VERSION = "m2-gated-relation-overall-price-embedding-historical-screen-v1"
MATCHED_MODEL_ID = "m1_matched_rho0"
MODEL_ID = "m2_gated_relation_overall_price_embedding"
ID_ONLY_MODEL_ID = "m2_jointly_trained_id_only"
RELATION_ONLY_MODEL_ID = "m2_id_plus_gated_relation_only"
PRICE_ONLY_MODEL_ID = "m2_id_plus_overall_price_only"


@dataclass(frozen=True)
class GatedRelationPriceConfig:
    dataset: str = "dunnhumby"
    seed: int = 42
    time_cutoff: int = 690
    evaluation_days: int = 7
    epochs: int = 100
    id_dim: int = 64
    auxiliary_dim: int = 3
    rho: float = 0.05
    price_budget: float = 0.25
    price_scale_delta: float = 0.5
    relation_gate_initial: float = 0.9
    relation_level_slope_initial: float = 0.1
    n_layers: int = 2
    batch_size: int = 8192
    lr: float = 5e-4
    pref_reg: float = 1e-3
    input_days: int = 365
    diagnostic_max_k: int = 50
    out_dir: str = ""
    baseline_result_dir: str = ""


def configure_gated_relation_price_run(**overrides) -> GatedRelationPriceConfig:
    defaults = {
        "out_dir": (
            f"{v3.default_out_dir('dunnhumby')}"
            "_m2_gated_relation_overall_price_embedding_historical_screen_v1"
        ),
        "baseline_result_dir": (
            f"{v3.default_out_dir('dunnhumby')}"
            "_m2_repeatshare_historical_backtest_v1"
        ),
    }
    return validate_gated_relation_price_config(
        GatedRelationPriceConfig(**(defaults | overrides))
    )


def validate_gated_relation_price_config(
    cfg: GatedRelationPriceConfig,
) -> GatedRelationPriceConfig:
    required = {
        "dataset": "dunnhumby",
        "seed": 42,
        "time_cutoff": 690,
        "evaluation_days": 7,
        "epochs": 100,
        "id_dim": 64,
        "auxiliary_dim": 3,
        "rho": 0.05,
        "price_budget": 0.25,
        "price_scale_delta": 0.5,
        "relation_gate_initial": 0.9,
        "relation_level_slope_initial": 0.1,
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


def preflight_summary(cfg: GatedRelationPriceConfig) -> dict:
    cfg = validate_gated_relation_price_config(cfg)
    return {
        "code_version": CODE_VERSION,
        "dataset": cfg.dataset,
        "seed": cfg.seed,
        "trained_models": [MATCHED_MODEL_ID, MODEL_ID],
        "reused_comparator": "m1_64 (display only)",
        "research_axis": "M2 representation intervention",
        "historical_development_split": {
            "train_end_inclusive": 683,
            "evaluation_start_inclusive": 684,
            "evaluation_end_inclusive": 690,
            "final_test_constructed": False,
            "holdout_constructed": False,
        },
        "m2": {
            "architecture": "ID(64)|gated CLV relation(2)|overall price fit(1)",
            "total_dim": cfg.id_dim + cfg.auxiliary_dim,
            "user_relation": (
                "q_C*g_R*unit([q_C,q_N-q_V]); "
                "g_R=sigmoid(a0+softplus(aC)*(q_C-.5)+aD*(q_N-q_V))"
            ),
            "item_relation": "unit(P_z E_i^ID)",
            "user_price": "q_C*(2*amount-weighted historical overall price percentile-1)",
            "item_price": "bounded-positive-scale*(2*overall item price percentile-1)",
            "within_category_price_input": False,
            "price_scale_range": [
                1.0 - cfg.price_scale_delta,
                1.0 + cfg.price_scale_delta,
            ],
            "rho": cfg.rho,
            "price_budget": cfg.price_budget,
            "symmetric_scale": "sqrt(rho) on user and item auxiliary blocks",
            "joint_graph_propagation": True,
            "one_dot_score": True,
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
            "accuracy": "all Recall/NDCG@10/20/50 >= 99% of matched rho=0",
            "high_clv": "high-CLV Recall@10 and NDCG@10 both improve",
            "economic": "price_purchase_amount_weighted_hit@10 increases",
            "direct_effect": (
                "full must beat jointly-trained ID-only on high-CLV "
                "Recall/NDCG@10 and weighted hit@10"
            ),
            "mechanism": "report ID-only, relation-only, price-only, and Top-10 changes",
            "statistical_note": "seed 42 exploratory screen; no significance claim",
        },
        "out_dir": cfg.out_dir,
        "baseline_result_dir": cfg.baseline_result_dir,
    }


def build_overall_price_fit_inputs(
    train: pd.DataFrame, *, n_users: int, n_items: int
) -> dict[str, np.ndarray]:
    """Train-only item price percentiles and users' amount-weighted positions."""

    required = {"u_idx", "i_idx", "up", "v"}
    missing = required - set(train.columns)
    if missing:
        raise KeyError(f"전체 가격 적합 입력 컬럼 누락: {sorted(missing)}")
    mean_price = (
        train.groupby("i_idx", sort=True)["up"]
        .mean()
        .reindex(np.arange(n_items))
        .to_numpy(np.float64)
    )
    item_valid = np.isfinite(mean_price)
    item_price = np.full(n_items, 0.5, dtype=np.float64)
    if item_valid.any():
        ranks = pd.Series(mean_price[item_valid]).rank(method="average").to_numpy()
        item_price[item_valid] = ranks / int(item_valid.sum())

    users = train["u_idx"].to_numpy(np.int64, copy=False)
    items = train["i_idx"].to_numpy(np.int64, copy=False)
    amount = np.maximum(train["v"].to_numpy(np.float64, copy=True), 0.0)
    valid_row = np.isfinite(amount) & item_valid[items]
    denominator = np.bincount(
        users[valid_row], weights=amount[valid_row], minlength=n_users
    )
    numerator = np.bincount(
        users[valid_row],
        weights=amount[valid_row] * item_price[items[valid_row]],
        minlength=n_users,
    )
    user_valid = denominator > 0.0
    user_price = np.divide(
        numerator,
        denominator,
        out=np.full(n_users, 0.5, dtype=np.float64),
        where=user_valid,
    )
    if not np.isfinite(item_price).all() or not np.isfinite(user_price).all():
        raise RuntimeError("전체 가격 적합 입력에 비유한 값이 있습니다")
    if ((item_price < 0.0) | (item_price > 1.0)).any():
        raise RuntimeError("상품 전체 가격 위치가 [0,1]을 벗어났습니다")
    if ((user_price < 0.0) | (user_price > 1.0)).any():
        raise RuntimeError("사용자 전체 가격 위치가 [0,1]을 벗어났습니다")
    return {
        "item_overall_price": item_price.astype(np.float32),
        "item_price_valid": item_valid,
        "user_overall_price": user_price.astype(np.float32),
        "user_price_valid": user_valid,
    }


def _config_hash(
    cfg: GatedRelationPriceConfig, input_hash: str, revision: str
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


def _prepare(cfg: GatedRelationPriceConfig) -> dict:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = moe.build_input_manifest(v3.SCHEMA[cfg.dataset])
    input_hash = moe.manifest_hash(manifest)
    revision = moe.source_revision()
    base_cfg = gatefree._base_config(cfg)
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
    q_n, q_v, q_c, clv_valid = evaluation.build_clv_inputs(axes)
    price = build_overall_price_fit_inputs(
        data["train"], n_users=data["n_users"], n_items=data["n_items"]
    )
    baseline = gatefree._load_compatible_baseline(cfg, manifest)
    meta = v3.item_meta(data["train"], data["n_items"])
    thresholds = v3.segment_thresholds(axes["clv_proxy"], base_cfg["SEG_EDGES"])
    cache = v3.EvalCache(
        *data["splits"]["test"], axes["clv_proxy"], thresholds, data["n_items"]
    )
    return {
        "out_dir": out_dir,
        "manifest": manifest,
        "input_hash": input_hash,
        "revision": revision,
        "base_cfg": base_cfg,
        "data": data,
        "axes": axes,
        "q_n": q_n,
        "q_v": q_v,
        "q_c": q_c,
        "clv_valid": clv_valid,
        **price,
        "baseline": baseline,
        "meta": meta,
        "thresholds": thresholds,
        "cache": cache,
        "config_hash": _config_hash(cfg, input_hash, revision),
    }


def _build_model(
    prepared: dict, cfg: GatedRelationPriceConfig, rho: float
) -> tuple[GatedRelationOverallPriceLightGCN, list[torch.nn.Parameter]]:
    data = prepared["data"]
    v3.set_seed(cfg.seed)
    model = GatedRelationOverallPriceLightGCN(
        n_users=data["n_users"],
        n_items=data["n_items"],
        q_n=prepared["q_n"],
        q_v=prepared["q_v"],
        q_c=prepared["q_c"],
        user_clv_valid=prepared["clv_valid"],
        user_overall_price=prepared["user_overall_price"],
        user_price_valid=prepared["user_price_valid"],
        item_overall_price=prepared["item_overall_price"],
        item_price_valid=prepared["item_price_valid"],
        adj=data["adj"],
        id_dim=cfg.id_dim,
        auxiliary_dim=cfg.auxiliary_dim,
        rho=rho,
        price_budget=cfg.price_budget,
        price_scale_delta=cfg.price_scale_delta,
        relation_gate_initial=cfg.relation_gate_initial,
        relation_level_slope_initial=cfg.relation_level_slope_initial,
        n_layers=cfg.n_layers,
        pref_reg=cfg.pref_reg,
    ).to(v3.DEVICE)
    return model, list(model.parameters())


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
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def _run_arm(
    prepared: dict,
    cfg: GatedRelationPriceConfig,
    *,
    model_id: str,
    rho: float,
) -> tuple[dict, GatedRelationOverallPriceLightGCN]:
    paths = _arm_paths(prepared, model_id)
    model, params = _build_model(prepared, cfg, rho)
    if paths["result"].exists() and paths["checkpoint"].exists():
        print(f"  [cached] {model_id} 완료 결과 재사용")
        payload = json.loads(paths["result"].read_text(encoding="utf-8"))
        checkpoint = evaluation._load_state(paths["checkpoint"])
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


class _IDOnlyView(nn.Module):
    def __init__(self, model: GatedRelationOverallPriceLightGCN):
        super().__init__()
        self.model = model

    def embeddings(self, need_value: bool = True):
        user, item = self.model.id_embeddings()
        return (
            user,
            item,
            user.new_zeros((len(user), 1)),
            item.new_zeros((len(item), 1)),
        )


class _ComponentView(nn.Module):
    def __init__(self, model: GatedRelationOverallPriceLightGCN, component: str):
        super().__init__()
        self.model = model
        self.component = component

    def embeddings(self, need_value: bool = True):
        user, item = self.model.component_embeddings(self.component)
        return (
            user,
            item,
            user.new_zeros((len(user), 1)),
            item.new_zeros((len(item), 1)),
        )


def run_gated_relation_price_screen(
    cfg: GatedRelationPriceConfig | None = None,
) -> pd.DataFrame:
    cfg = validate_gated_relation_price_config(
        cfg or configure_gated_relation_price_run()
    )
    print(json.dumps(preflight_summary(cfg), ensure_ascii=False, indent=2))
    prepared = _prepare(cfg)
    print("\n===== matched rho=0 | seed 42 | fixed 100 epochs =====")
    matched, matched_model = _run_arm(
        prepared, cfg, model_id=MATCHED_MODEL_ID, rho=0.0
    )
    print("\n===== gated relation + overall price rho=0.05 | seed 42 =====")
    active, active_model = _run_arm(
        prepared, cfg, model_id=MODEL_ID, rho=cfg.rho
    )

    view_metrics = {}
    for model_id, view in (
        (ID_ONLY_MODEL_ID, _IDOnlyView(active_model)),
        (RELATION_ONLY_MODEL_ID, _ComponentView(active_model, "relation")),
        (PRICE_ONLY_MODEL_ID, _ComponentView(active_model, "price")),
    ):
        raw, _ = moe._flat_evaluation(
            view.to(v3.DEVICE),
            0.0,
            prepared["cache"],
            prepared["meta"],
            prepared["data"],
            prepared["base_cfg"],
            per_user=False,
        )
        view_metrics[model_id] = test10._public_metrics(raw)

    users, matched_top50 = evaluation._masked_topk(
        matched_model, prepared, max_k=cfg.diagnostic_max_k
    )
    active_users, active_top50 = evaluation._masked_topk(
        active_model, prepared, max_k=cfg.diagnostic_max_k
    )
    if not np.array_equal(users, active_users):
        raise RuntimeError("matched와 M2 평가 사용자 순서가 다릅니다")
    overlap = evaluation.topk_overlap_summary(
        matched_top50, active_top50, prepared["cache"].seg, k=10
    )
    overlap.insert(0, "reference", MATCHED_MODEL_ID)
    overlap.insert(1, "model_id", MODEL_ID)
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
    for model_id, metrics in view_metrics.items():
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
        **view_metrics,
    }
    comparison = evaluation._metric_comparison(
        metric_rows, references=(MATCHED_MODEL_ID, "m1_64", ID_ONLY_MODEL_ID)
    )
    reading = shared.screening_reading(
        matched["metrics"],
        active["metrics"],
        overlap,
        matched["diagnostics"],
        id_only_metrics=view_metrics[ID_ONLY_MODEL_ID],
    )
    direct_metrics = (
        "고CLV_recall@10",
        "고CLV_ndcg@10",
        "price_purchase_amount_weighted_hit@10",
    )
    direct_deltas = {
        metric: float(active["metrics"][metric] - view_metrics[ID_ONLY_MODEL_ID][metric])
        for metric in direct_metrics
    }
    reading["direct_auxiliary_deltas_vs_joint_id_only"] = direct_deltas
    reading["direct_auxiliary_positive"] = all(
        delta > 0.0 for delta in direct_deltas.values()
    )
    reading["positive_screen"] = bool(
        reading["positive_screen"] and reading["direct_auxiliary_positive"]
    )
    reading["price_only_weighted_hit_delta_vs_id_only"] = float(
        view_metrics[PRICE_ONLY_MODEL_ID]["price_purchase_amount_weighted_hit@10"]
        - view_metrics[ID_ONLY_MODEL_ID]["price_purchase_amount_weighted_hit@10"]
    )

    stem = f"m2_gated_relation_overall_price_{prepared['config_hash']}"
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

    print("\n절대지표:")
    print(frame.to_string(index=False))
    print("\n대조군별 비교:")
    print(comparison.to_string(index=False))
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
            preflight_summary(configure_gated_relation_price_run()),
            ensure_ascii=False,
            indent=2,
        )
    )
