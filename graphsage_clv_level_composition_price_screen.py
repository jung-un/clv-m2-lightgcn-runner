"""Seed-42 GraphSAGE portability screen for the established M2 input."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from clv_run_state import ProgressStore, RunIdentity, clone_state, file_sha256
from graphsage_clv_level_composition_price_model import (
    GraphSAGECLVLevelCompositionPrice,
)
import lightgcn_clv_axis_specific_test10 as test10
import lightgcn_clv_constrained_economic_embedding as source
import lightgcn_clv_gradient_isolated_economic_interaction as helpers
import lightgcn_clv_joint_response_embedding as preparation
import lightgcn_clv_moe as moe
import lightgcn_clv_v3 as v3


CODE_VERSION = "m2-graphsage-clv-level-composition-price-historical-screen-v1"
GRAPHSAGE_M1_64 = "graphsage_m1_64"
GRAPHSAGE_M1_67 = "graphsage_m1_67"
GRAPHSAGE_M2 = "graphsage_m2_clv_level_composition_price"
GRAPHSAGE_SHUFFLE = "graphsage_m2_degree_matched_clv_shuffle"
ACCURACY_METRICS = (
    "recall@10",
    "ndcg@10",
    "recall@20",
    "ndcg@20",
    "recall@50",
    "ndcg@50",
)


@dataclass(frozen=True)
class GraphSAGECLVScreenConfig:
    dataset: str = "dunnhumby"
    seed: int = 42
    time_cutoff: int = 690
    evaluation_days: int = 7
    epochs: int = 100
    id_dim: int = 64
    clv_dim: int = 3
    rho: float = 0.05
    item_price_budget: float = 0.25
    n_layers: int = 2
    batch_size: int = 8192
    lr: float = 5e-4
    pref_reg: float = 1e-3
    input_days: int = 365
    diagnostic_max_k: int = 50
    shuffle_degree_bins: int = 10
    shuffle_seed: int = 42
    out_dir: str = ""
    baseline_result_dir: str = ""


def configure_graphsage_clv_screen(**overrides) -> GraphSAGECLVScreenConfig:
    defaults = {
        "out_dir": (
            f"{v3.default_out_dir('dunnhumby')}"
            "_m2_graphsage_clv_level_composition_price_historical_screen_v1"
        ),
        "baseline_result_dir": (
            f"{v3.default_out_dir('dunnhumby')}"
            "_m2_repeatshare_historical_backtest_v1"
        ),
    }
    return validate_graphsage_clv_config(
        GraphSAGECLVScreenConfig(**(defaults | overrides))
    )


def validate_graphsage_clv_config(
    cfg: GraphSAGECLVScreenConfig,
) -> GraphSAGECLVScreenConfig:
    required = {
        "dataset": "dunnhumby",
        "seed": 42,
        "time_cutoff": 690,
        "evaluation_days": 7,
        "epochs": 100,
        "id_dim": 64,
        "clv_dim": 3,
        "rho": 0.05,
        "item_price_budget": 0.25,
        "n_layers": 2,
        "input_days": 365,
        "diagnostic_max_k": 50,
        "shuffle_degree_bins": 10,
        "shuffle_seed": 42,
    }
    for key, expected in required.items():
        if getattr(cfg, key) != expected:
            raise ValueError(
                f"GraphSAGE 개발 screen은 {key}={expected!r}이어야 합니다"
            )
    if cfg.batch_size <= 0 or cfg.lr <= 0 or cfg.pref_reg < 0:
        raise ValueError("학습 설정이 잘못됐습니다")
    if not cfg.out_dir or not cfg.baseline_result_dir:
        raise ValueError("out_dir와 baseline_result_dir가 필요합니다")
    return cfg


def preflight_summary(cfg: GraphSAGECLVScreenConfig) -> dict:
    cfg = validate_graphsage_clv_config(cfg)
    return {
        "code_version": CODE_VERSION,
        "dataset": cfg.dataset,
        "seed": cfg.seed,
        "research_axis": "M2 representation intervention",
        "trained_models": [
            GRAPHSAGE_M1_64,
            GRAPHSAGE_M1_67,
            GRAPHSAGE_M2,
            GRAPHSAGE_SHUFFLE,
        ],
        "reused_reference": "LightGCN m1_64 (display only)",
        "historical_development_split": {
            "train_end_inclusive": 683,
            "evaluation_start_inclusive": 684,
            "evaluation_end_inclusive": 690,
            "new_item_task": True,
            "train_pairs_excluded_from_truth_and_ranking": True,
            "final_test_constructed": False,
            "holdout_constructed": False,
        },
        "m2": {
            "backbone": "two-layer mean-aggregating GraphSAGE",
            "layer0": "ID(64)|CLV level-and-composition relation(2)|price(1)",
            "graphsage_m1_dimensions": [64, 67],
            "m2_layer0_and_final_dim": 67,
            "propagation": (
                "Unit(ELU(W concat(self, binary-neighbor-row-mean)))"
            ),
            "layer_outputs": "mean(layer0, layer1, layer2)",
            "self_loop": False,
            "self_path": "explicit in GraphSAGE concatenation",
            "rho": cfg.rho,
            "rho_interpretation": "layer-0 input budget, not final-score bound",
            "item_price_budget": cfg.item_price_budget,
            "feature_dropout": 0.0,
            "degree_matched_clv_shuffle": True,
            "repeatshare_input": False,
            "raw_item_popularity_input": False,
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
            "accuracy_floor": (
                "all Recall/NDCG@10/20/50 >= 99% of GraphSAGE@67"
            ),
            "economic": (
                "weighted hit@10 must exceed GraphSAGE@67 and shuffled CLV"
            ),
            "semantic_attribution": (
                "observed CLV must beat degree-matched shuffle on both "
                "high-CLV Recall@10 and NDCG@10"
            ),
            "backbone_note": (
                "GraphSAGE portability is judged against GraphSAGE@67; "
                "LightGCN is display-only"
            ),
            "statistical_note": "seed 42 exploratory screen; no significance claim",
        },
        "out_dir": cfg.out_dir,
        "baseline_result_dir": cfg.baseline_result_dir,
    }


def _canonical(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)


def _config_hash(
    cfg: GraphSAGECLVScreenConfig, input_hash: str, revision: str
) -> str:
    payload = {
        "code_version": CODE_VERSION,
        "config": asdict(cfg),
        "input_hash": input_hash,
        "source_revision": revision,
    }
    return hashlib.sha256(_canonical(payload).encode()).hexdigest()[:12]


def _preparation_config(cfg: GraphSAGECLVScreenConfig):
    return preparation.JointResponseConfig(
        dataset=cfg.dataset,
        seed=cfg.seed,
        time_cutoff=cfg.time_cutoff,
        evaluation_days=cfg.evaluation_days,
        epochs=cfg.epochs,
        id_dim=cfg.id_dim,
        clv_dim=cfg.clv_dim,
        rho=cfg.rho,
        n_layers=cfg.n_layers,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        pref_reg=cfg.pref_reg,
        input_days=cfg.input_days,
        diagnostic_max_k=cfg.diagnostic_max_k,
        out_dir=cfg.out_dir,
        baseline_result_dir=cfg.baseline_result_dir,
    )


def _prepare(cfg: GraphSAGECLVScreenConfig) -> dict:
    prepared = preparation._prepare(_preparation_config(cfg))
    prepared["config_hash"] = _config_hash(
        cfg, prepared["input_hash"], prepared["revision"]
    )
    prepared["degree_matched_shuffle"] = source._degree_matched_clv_shuffle(
        prepared, cfg
    )
    if set(prepared["data"]["splits"]) != {"test"}:
        raise RuntimeError("개발평가 외 split이 구성됐습니다")
    if float(prepared["data"]["train"].t.max()) != 683.0:
        raise RuntimeError("학습 종료시점이 day 683이 아닙니다")
    if prepared["data"].get("loss_w") is not None:
        raise RuntimeError("M2에 M4 표본 가중치가 섞였습니다")
    return prepared


def _arm_spec(model_id: str) -> tuple[str, int, str]:
    if model_id == GRAPHSAGE_M1_64:
        return "id", 64, "none"
    if model_id == GRAPHSAGE_M1_67:
        return "id", 67, "none"
    if model_id == GRAPHSAGE_M2:
        return "clv", 64, "observed"
    if model_id == GRAPHSAGE_SHUFFLE:
        return "clv", 64, "degree_matched_shuffle"
    raise KeyError(model_id)


def _build_model(
    prepared: dict, cfg: GraphSAGECLVScreenConfig, model_id: str
):
    variant, id_dim, assignment_name = _arm_spec(model_id)
    assignment = (
        prepared["degree_matched_shuffle"]
        if assignment_name == "degree_matched_shuffle"
        else prepared
    )
    data = prepared["data"]
    v3.set_seed(cfg.seed)
    kwargs = {}
    if variant == "clv":
        kwargs = {
            "q_n": assignment["q_n"],
            "q_v": assignment["q_v"],
            "q_c": assignment["q_c"],
            "user_clv_valid": assignment["clv_valid"],
            "item_economic_features": prepared["item_economic"],
            "item_economic_valid": prepared["item_economic_valid"],
        }
    model = GraphSAGECLVLevelCompositionPrice(
        n_users=data["n_users"],
        n_items=data["n_items"],
        adj=data["adj"],
        id_dim=id_dim,
        variant=variant,
        rho=cfg.rho,
        item_price_budget=cfg.item_price_budget,
        n_layers=cfg.n_layers,
        pref_reg=cfg.pref_reg,
        **kwargs,
    ).to(v3.DEVICE)
    return model, list(model.parameters())


def _arm_hash(
    prepared: dict, cfg: GraphSAGECLVScreenConfig, model_id: str
) -> str:
    return hashlib.sha256(
        _canonical(
            {
                "run": prepared["config_hash"],
                "model_id": model_id,
                "model_spec": _arm_spec(model_id),
                "seed": cfg.seed,
                "epochs": cfg.epochs,
            }
        ).encode()
    ).hexdigest()[:12]


def _arm_paths(prepared: dict, model_id: str) -> dict[str, Path]:
    root = prepared["out_dir"] / "arms" / prepared["config_hash"]
    stem = f"{model_id}_s42"
    return {"result": root / f"{stem}.json", "checkpoint": root / f"{stem}.pt"}


def _run_arm(
    prepared: dict, cfg: GraphSAGECLVScreenConfig, model_id: str
) -> tuple[dict, GraphSAGECLVLevelCompositionPrice]:
    paths = _arm_paths(prepared, model_id)
    model, params = _build_model(prepared, cfg, model_id)
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
            config_hash=_arm_hash(prepared, cfg, model_id),
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
            "model_spec": _arm_spec(model_id),
            "config": asdict(cfg),
            "training": training,
            "source_revision": prepared["revision"],
            "input_hash": prepared["input_hash"],
        },
        temporary,
    )
    os.replace(temporary, paths["checkpoint"])
    metrics_raw, _ = moe._flat_evaluation(
        model,
        0.0,
        prepared["cache"],
        prepared["meta"],
        prepared["data"],
        prepared["base_cfg"],
        per_user=False,
    )
    variant, id_dim, assignment = _arm_spec(model_id)
    role = (
        "baseline"
        if model_id == GRAPHSAGE_M1_64
        else "matched_capacity_control"
        if model_id == GRAPHSAGE_M1_67
        else "attribution_control"
        if model_id == GRAPHSAGE_SHUFFLE
        else "model"
    )
    payload = {
        "model_id": model_id,
        "role": role,
        "seed": cfg.seed,
        "split": "historical_development_days_684_690",
        "final_epoch": cfg.epochs,
        "variant": variant,
        "id_dim": id_dim,
        "clv_assignment": assignment,
        "metrics": test10._public_metrics(metrics_raw),
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


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator != 0 else float("nan")


def screening_reading(metric_rows: dict[str, dict]) -> dict:
    baseline = metric_rows[GRAPHSAGE_M1_67]
    actual = metric_rows[GRAPHSAGE_M2]
    shuffled = metric_rows[GRAPHSAGE_SHUFFLE]
    ratios = {
        metric: _safe_ratio(actual[metric], baseline[metric])
        for metric in ACCURACY_METRICS
    }
    weighted = "price_purchase_amount_weighted_hit@10"
    high = ("고CLV_recall@10", "고CLV_ndcg@10")
    accuracy_pass = bool(min(ratios.values()) >= 0.99)
    economic_pass = bool(actual[weighted] > baseline[weighted])
    weighted_attribution = bool(actual[weighted] > shuffled[weighted])
    high_attribution = bool(all(actual[key] > shuffled[key] for key in high))
    return {
        "positive_screen": bool(
            accuracy_pass
            and economic_pass
            and weighted_attribution
            and high_attribution
        ),
        "accuracy_floor_vs_graphsage_m1_67": accuracy_pass,
        "accuracy_ratios_vs_graphsage_m1_67": ratios,
        "economic_gain_vs_graphsage_m1_67": economic_pass,
        "weighted_hit_at_10_delta_vs_graphsage_m1_67": float(
            actual[weighted] - baseline[weighted]
        ),
        "weighted_hit_at_10_delta_vs_degree_matched_shuffle": float(
            actual[weighted] - shuffled[weighted]
        ),
        "high_clv_attribution_pass": high_attribution,
        "high_clv_deltas_vs_degree_matched_shuffle": {
            key: float(actual[key] - shuffled[key]) for key in high
        },
        "next_if_positive": "paired development seeds, then H&M",
        "if_failed": "stop M2 backbone ports without tuning on days 684-690",
        "statistical_note": "seed 42 exploratory screen; no significance claim",
    }


def run_graphsage_clv_screen(
    cfg: GraphSAGECLVScreenConfig | None = None,
) -> pd.DataFrame:
    cfg = validate_graphsage_clv_config(
        cfg or configure_graphsage_clv_screen()
    )
    print(json.dumps(preflight_summary(cfg), ensure_ascii=False, indent=2))
    prepared = _prepare(cfg)
    results: dict[str, dict] = {}
    models: dict[str, GraphSAGECLVLevelCompositionPrice] = {}
    arms = (
        GRAPHSAGE_M1_64,
        GRAPHSAGE_M1_67,
        GRAPHSAGE_M2,
        GRAPHSAGE_SHUFFLE,
    )
    for model_id in arms:
        print(f"\n===== {model_id} | seed 42 | fixed 100 epochs =====")
        results[model_id], models[model_id] = _run_arm(
            prepared, cfg, model_id
        )

    metric_rows = {model_id: results[model_id]["metrics"] for model_id in arms}
    baseline_reference = dict(prepared["baseline"])
    baseline_reference["role"] = "LightGCN_reference_display_only"
    absolute_rows = [baseline_reference]
    for model_id in arms:
        result = results[model_id]
        last_training = result["training"]["history"][-1]
        absolute_rows.append(
            {
                "model_id": model_id,
                "role": result["role"],
                "seed": result["seed"],
                "split": result["split"],
                "final_epoch": result["final_epoch"],
                **result["diagnostics"],
                **{
                    f"last_epoch_{key}": value
                    for key, value in last_training.items()
                    if key != "epoch"
                },
                **result["metrics"],
            }
        )
    absolute = pd.DataFrame(absolute_rows)
    comparison = helpers._metric_comparison(
        metric_rows,
        references=(
            GRAPHSAGE_M1_64,
            GRAPHSAGE_M1_67,
            GRAPHSAGE_SHUFFLE,
        ),
    )

    base_users, base_top50 = helpers._masked_topk(
        models[GRAPHSAGE_M1_67], prepared, max_k=cfg.diagnostic_max_k
    )
    actual_users, actual_top50 = helpers._masked_topk(
        models[GRAPHSAGE_M2], prepared, max_k=cfg.diagnostic_max_k
    )
    shuffled_users, shuffled_top50 = helpers._masked_topk(
        models[GRAPHSAGE_SHUFFLE], prepared, max_k=cfg.diagnostic_max_k
    )
    if not (
        np.array_equal(base_users, actual_users)
        and np.array_equal(base_users, shuffled_users)
    ):
        raise RuntimeError("GraphSAGE arm의 평가 사용자 순서가 다릅니다")
    capacity_overlap = helpers.topk_overlap_summary(
        base_top50, actual_top50, prepared["cache"].seg, k=10
    )
    capacity_overlap.insert(0, "reference", GRAPHSAGE_M1_67)
    capacity_overlap.insert(1, "model_id", GRAPHSAGE_M2)
    shuffle_overlap = helpers.topk_overlap_summary(
        shuffled_top50, actual_top50, prepared["cache"].seg, k=10
    )
    shuffle_overlap.insert(0, "reference", GRAPHSAGE_SHUFFLE)
    shuffle_overlap.insert(1, "model_id", GRAPHSAGE_M2)
    overlap = pd.concat([capacity_overlap, shuffle_overlap], ignore_index=True)
    reading = screening_reading(metric_rows)
    reading["degree_matched_shuffle_changed_valid_user_share"] = prepared[
        "degree_matched_shuffle"
    ]["changed_valid_user_share"]

    stem = (
        "m2_graphsage_clv_level_composition_price_"
        f"{prepared['config_hash']}"
    )
    paths = {
        "absolute_csv": prepared["out_dir"] / f"{stem}.csv",
        "comparison_csv": prepared["out_dir"] / f"{stem}_comparison.csv",
        "top10_overlap_csv": prepared["out_dir"] / f"{stem}_top10_overlap.csv",
        "json": prepared["out_dir"] / f"{stem}.json",
    }
    test10._atomic_csv(paths["absolute_csv"], absolute)
    test10._atomic_csv(paths["comparison_csv"], comparison)
    test10._atomic_csv(paths["top10_overlap_csv"], overlap)
    payload = {
        "code_version": CODE_VERSION,
        "source_revision": prepared["revision"],
        "config": asdict(cfg),
        "preflight": preflight_summary(cfg),
        "input_manifest": prepared["manifest"],
        "absolute_rows": absolute.to_dict("records"),
        "comparison_rows": comparison.to_dict("records"),
        "top10_overlap_rows": overlap.to_dict("records"),
        "screening_reading": reading,
        "result_paths": {key: str(value) for key, value in paths.items()},
    }
    test10._atomic_json(paths["json"], payload)
    absolute.attrs["comparison"] = comparison.to_dict("records")
    absolute.attrs["top10_overlap"] = overlap.to_dict("records")
    absolute.attrs["screening_reading"] = reading
    absolute.attrs["result_paths"] = {
        key: str(value) for key, value in paths.items()
    }

    key_metrics = (
        *ACCURACY_METRICS,
        "price_purchase_amount_weighted_hit@10",
        "고CLV_recall@10",
        "고CLV_ndcg@10",
    )
    key_comparison = comparison[
        comparison.metric.isin(key_metrics)
        & comparison.reference.isin(
            [GRAPHSAGE_M1_67, GRAPHSAGE_SHUFFLE]
        )
        & (comparison.model_id == GRAPHSAGE_M2)
    ]
    print("\n1) 절대지표")
    print(absolute.to_string(index=False))
    print("\n2) GraphSAGE@67 및 degree-matched shuffle 대비 핵심 변화")
    print(key_comparison.to_string(index=False))
    print("\n3) Top-10 변경")
    print(overlap.to_string(index=False))
    print("\n4) 사전 판정")
    print(json.dumps(reading, ensure_ascii=False, indent=2))
    print("\n5) 저장 파일")
    print(json.dumps(absolute.attrs["result_paths"], ensure_ascii=False, indent=2))
    return absolute


if __name__ == "__main__":
    print(
        json.dumps(
            preflight_summary(configure_graphsage_clv_screen()),
            ensure_ascii=False,
            indent=2,
        )
    )
