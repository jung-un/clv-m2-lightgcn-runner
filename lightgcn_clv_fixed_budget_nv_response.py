"""Seed-42 M2 screen with a fixed CLV budget split across N and V axes."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from clv_fixed_budget_nv_response_model import FixedBudgetNVResponseLightGCN
from clv_run_state import ProgressStore, RunIdentity, clone_state, file_sha256
import lightgcn_clv_axis_specific_test10 as test10
import lightgcn_clv_constrained_economic_embedding as prior
import lightgcn_clv_gated_relation_overall_price as price_inputs
import lightgcn_clv_gradient_isolated_economic_interaction as helpers
import lightgcn_clv_joint_response_embedding as shared
import lightgcn_clv_moe as moe
import lightgcn_clv_v3 as v3


CODE_VERSION = "m2-fixed-budget-nv-response-embedding-historical-screen-v1"
MATCHED_MODEL_ID = "m1_matched_rho0"
MODEL_ID = "m2_fixed_budget_nv_response_embedding"
SHUFFLED_MODEL_ID = "m2_degree_matched_clv_shuffle"
ID_ONLY_MODEL_ID = "m2_jointly_trained_id_only"
N_ONLY_MODEL_ID = "m2_id_plus_n_response_only"
V_ONLY_MODEL_ID = "m2_id_plus_v_price_response_only"


@dataclass(frozen=True)
class FixedBudgetNVResponseConfig:
    dataset: str = "dunnhumby"
    seed: int = 42
    time_cutoff: int = 690
    evaluation_days: int = 7
    epochs: int = 100
    id_dim: int = 64
    rho: float = 0.05
    price_scale_initial: float = 0.9
    n_layers: int = 2
    batch_size: int = 8192
    lr: float = 5e-4
    pref_reg: float = 1e-3
    input_days: int = 365
    diagnostic_max_k: int = 50
    include_degree_matched_shuffle: bool = True
    shuffle_degree_bins: int = 10
    shuffle_seed: int = 42
    out_dir: str = ""
    baseline_result_dir: str = ""


def configure_fixed_budget_nv_response_run(
    **overrides,
) -> FixedBudgetNVResponseConfig:
    defaults = {
        "out_dir": (
            f"{v3.default_out_dir('dunnhumby')}"
            "_m2_fixed_budget_nv_response_embedding_historical_screen_v1"
        ),
        "baseline_result_dir": (
            f"{v3.default_out_dir('dunnhumby')}"
            "_m2_repeatshare_historical_backtest_v1"
        ),
    }
    return validate_fixed_budget_nv_response_config(
        FixedBudgetNVResponseConfig(**(defaults | overrides))
    )


def validate_fixed_budget_nv_response_config(
    cfg: FixedBudgetNVResponseConfig,
) -> FixedBudgetNVResponseConfig:
    required = {
        "dataset": "dunnhumby",
        "seed": 42,
        "time_cutoff": 690,
        "evaluation_days": 7,
        "epochs": 100,
        "id_dim": 64,
        "rho": 0.05,
        "price_scale_initial": 0.9,
        "n_layers": 2,
        "input_days": 365,
        "diagnostic_max_k": 50,
        "include_degree_matched_shuffle": True,
        "shuffle_degree_bins": 10,
        "shuffle_seed": 42,
    }
    for key, expected in required.items():
        if getattr(cfg, key) != expected:
            raise ValueError(f"빠른 M2 screen은 {key}={expected!r}이어야 합니다")
    if cfg.batch_size <= 0 or cfg.lr <= 0 or cfg.pref_reg < 0:
        raise ValueError("학습 설정이 잘못됐습니다")
    if not cfg.out_dir or not cfg.baseline_result_dir:
        raise ValueError("out_dir와 baseline_result_dir가 필요합니다")
    return cfg


def preflight_summary(cfg: FixedBudgetNVResponseConfig) -> dict:
    cfg = validate_fixed_budget_nv_response_config(cfg)
    return {
        "code_version": CODE_VERSION,
        "dataset": cfg.dataset,
        "seed": cfg.seed,
        "trained_models": [MATCHED_MODEL_ID, MODEL_ID, SHUFFLED_MODEL_ID],
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
            "architecture": (
                "ID(64)|fixed-budget N response(1)|"
                "fixed-budget V-price response(1)"
            ),
            "total_dim": cfg.id_dim + 2,
            "historical_clv": "q_C(u)=percentile(N_hat(u)*V_hat(u))",
            "clv_budget_identity": "b_N(u)+b_V(u)=q_C(u)",
            "user_coordinates": (
                "b_N=q_C*q_N/(q_N+q_V), "
                "b_V=q_C*q_V/(q_N+q_V)"
            ),
            "item_n_response": "tanh(w_N^T E_i^ID)",
            "item_v_response": (
                "sigmoid(a_V)*(2*overall_price_percentile_i-1)"
            ),
            "learned_user_projection": False,
            "free_item_response_embedding": False,
            "joint_graph_propagation": True,
            "one_dot_score": True,
            "rho": cfg.rho,
            "symmetric_scale": "sqrt(rho) on both user and item axes",
            "repeatshare_input": False,
            "item_popularity_input": False,
            "price_distance_feature": False,
            "external_reranking": False,
            "degree_matched_clv_shuffle": True,
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
            "baseline_accuracy": (
                "all Recall/NDCG@10/20/50 >= 99% of matched rho=0"
            ),
            "direct_clv": (
                "full beats jointly-trained ID-only on high-CLV "
                "Recall/NDCG@10 and weighted hit@10"
            ),
            "semantic_attribution": (
                "observed CLV beats degree-matched shuffle on six-metric "
                "accuracy geometric mean and weighted hit@10, and on either "
                "high-CLV Recall@10 or NDCG@10"
            ),
            "mechanism": "report ID-only, N-only, V-only and Top-10 changes",
            "statistical_note": "seed 42 exploratory screen; no significance claim",
        },
        "out_dir": cfg.out_dir,
        "baseline_result_dir": cfg.baseline_result_dir,
    }


def _config_hash(
    cfg: FixedBudgetNVResponseConfig, input_hash: str, revision: str
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


def _degree_matched_clv_shuffle(
    prepared: dict, cfg: FixedBudgetNVResponseConfig
) -> dict:
    return prior._degree_matched_clv_shuffle(prepared, cfg)


def _prepare(cfg: FixedBudgetNVResponseConfig) -> dict:
    # Reuse only the common data/CLV/price preparation; the model and run hash
    # are specific to this experiment.
    prepared = price_inputs._prepare(cfg)
    prepared["config_hash"] = _config_hash(
        cfg, prepared["input_hash"], prepared["revision"]
    )
    prepared["degree_matched_shuffle"] = _degree_matched_clv_shuffle(
        prepared, cfg
    )
    return prepared


def _build_model(
    prepared: dict,
    cfg: FixedBudgetNVResponseConfig,
    rho: float,
    *,
    clv_assignment: dict | None = None,
) -> tuple[FixedBudgetNVResponseLightGCN, list[torch.nn.Parameter]]:
    data = prepared["data"]
    assignment = clv_assignment or prepared
    v3.set_seed(cfg.seed)
    model = FixedBudgetNVResponseLightGCN(
        n_users=data["n_users"],
        n_items=data["n_items"],
        q_n=assignment["q_n"],
        q_v=assignment["q_v"],
        q_c=assignment["q_c"],
        user_clv_valid=assignment["clv_valid"],
        item_overall_price=prepared["item_overall_price"],
        item_price_valid=prepared["item_price_valid"],
        adj=data["adj"],
        id_dim=cfg.id_dim,
        rho=rho,
        n_layers=cfg.n_layers,
        pref_reg=cfg.pref_reg,
        price_scale_initial=cfg.price_scale_initial,
    ).to(v3.DEVICE)
    return model, list(model.parameters())


class _ComponentView(torch.nn.Module):
    def __init__(self, parent: FixedBudgetNVResponseLightGCN, component: str):
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


def _arm_hash(
    prepared: dict, model_id: str, rho: float, assignment_name: str
) -> str:
    payload = {
        "run": prepared["config_hash"],
        "model_id": model_id,
        "rho": rho,
        "clv_assignment": assignment_name,
        "seed": 42,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[
        :12
    ]


def _run_arm(
    prepared: dict,
    cfg: FixedBudgetNVResponseConfig,
    *,
    model_id: str,
    rho: float,
    clv_assignment: dict | None = None,
    assignment_name: str = "observed",
) -> tuple[dict, FixedBudgetNVResponseLightGCN]:
    paths = _arm_paths(prepared, model_id)
    model, params = _build_model(
        prepared, cfg, rho, clv_assignment=clv_assignment
    )
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
            config_hash=_arm_hash(prepared, model_id, rho, assignment_name),
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
            "clv_assignment": assignment_name,
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
        "role": (
            "matched_control"
            if rho == 0.0
            else "attribution_control"
            if assignment_name == "degree_matched_shuffle"
            else "model"
        ),
        "seed": cfg.seed,
        "split": "historical_development_days_684_690",
        "final_epoch": cfg.epochs,
        "rho": rho,
        "clv_assignment": assignment_name,
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


@torch.no_grad()
def _score_diagnostics(
    model: FixedBudgetNVResponseLightGCN,
    users: np.ndarray,
    top50: np.ndarray,
    prepared: dict,
) -> pd.DataFrame:
    width = top50.shape[1]
    pair_users = np.repeat(users.astype(np.int64), width)
    pair_items = top50.reshape(-1).astype(np.int64)
    collected = {key: [] for key in ("id", "n", "v", "clv", "full")}
    for start in range(0, len(pair_users), 65536):
        user_tensor = torch.as_tensor(
            pair_users[start : start + 65536], dtype=torch.long, device=v3.DEVICE
        )
        item_tensor = torch.as_tensor(
            pair_items[start : start + 65536], dtype=torch.long, device=v3.DEVICE
        )
        components = model.candidate_score_components(user_tensor, item_tensor)
        for key in collected:
            collected[key].append(components[key].cpu().numpy())
    values = {
        key: np.concatenate(chunks).astype(np.float64)
        for key, chunks in collected.items()
    }
    id_std = float(values["id"].std())
    clv_std = float(values["clv"].std())
    per_user_abs = np.abs(values["clv"]).reshape(len(users), width).mean(axis=1)
    q_n = prepared["q_n"][users]
    q_v = prepared["q_v"][users]
    denominator = q_n + q_v
    composition = np.divide(
        q_n - q_v,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    return pd.DataFrame(
        [
            {
                "candidate_pair_count": len(pair_users),
                "id_score_std": id_std,
                "n_score_std": float(values["n"].std()),
                "v_score_std": float(values["v"].std()),
                "clv_score_std": clv_std,
                "clv_score_std_ratio_to_id": (
                    clv_std / id_std if id_std > 0 else np.nan
                ),
                "clv_score_mean_abs": float(np.abs(values["clv"]).mean()),
                "per_user_clv_abs_q_c_spearman": float(
                    pd.Series(per_user_abs).corr(
                        pd.Series(prepared["q_c"][users]), method="spearman"
                    )
                ),
                "per_user_clv_abs_composition_spearman": float(
                    pd.Series(per_user_abs).corr(
                        pd.Series(np.abs(composition)), method="spearman"
                    )
                ),
                "max_full_decomposition_error": float(
                    np.max(
                        np.abs(
                            values["full"]
                            - values["id"]
                            - values["n"]
                            - values["v"]
                        )
                    )
                ),
            }
        ]
    )


def _screening_reading(
    matched: dict,
    active: dict,
    shuffled: dict,
    id_metrics: dict,
    overlap: pd.DataFrame,
    rho0_diagnostics: dict,
) -> dict:
    reading = shared.screening_reading(
        matched,
        active,
        overlap,
        rho0_diagnostics,
        id_only_metrics=id_metrics,
    )
    direct_metrics = (
        "고CLV_recall@10",
        "고CLV_ndcg@10",
        "price_purchase_amount_weighted_hit@10",
    )
    direct_deltas = {
        metric: float(active[metric] - id_metrics[metric])
        for metric in direct_metrics
    }
    direct_positive = all(delta > 0.0 for delta in direct_deltas.values())
    accuracy_metrics = (
        "recall@10",
        "ndcg@10",
        "recall@20",
        "ndcg@20",
        "recall@50",
        "ndcg@50",
    )
    ratios = {
        metric: float(active[metric] / shuffled[metric])
        for metric in accuracy_metrics
    }
    geomean = float(np.exp(np.mean(np.log(list(ratios.values())))))
    weighted_delta = float(
        active["price_purchase_amount_weighted_hit@10"]
        - shuffled["price_purchase_amount_weighted_hit@10"]
    )
    high_better = bool(
        active["고CLV_recall@10"] > shuffled["고CLV_recall@10"]
        or active["고CLV_ndcg@10"] > shuffled["고CLV_ndcg@10"]
    )
    attribution = bool(geomean > 1.0 and weighted_delta > 0.0 and high_better)
    reading.update(
        {
            "direct_clv_deltas_vs_joint_id_only": direct_deltas,
            "direct_clv_positive": direct_positive,
            "accuracy_ratios_vs_degree_matched_shuffle": ratios,
            "accuracy_geomean_ratio_vs_degree_matched_shuffle": geomean,
            "weighted_hit_at_10_delta_vs_degree_matched_shuffle": weighted_delta,
            "high_clv_accuracy_better_than_degree_matched_shuffle": high_better,
            "clv_attribution_supported": attribution,
        }
    )
    reading["positive_screen"] = bool(
        reading["positive_screen"] and direct_positive and attribution
    )
    return reading


def run_fixed_budget_nv_response_screen(
    cfg: FixedBudgetNVResponseConfig | None = None,
) -> pd.DataFrame:
    cfg = validate_fixed_budget_nv_response_config(
        cfg or configure_fixed_budget_nv_response_run()
    )
    print(json.dumps(preflight_summary(cfg), ensure_ascii=False, indent=2))
    prepared = _prepare(cfg)

    print("\n===== matched rho=0 | seed 42 | fixed 100 epochs =====")
    matched, matched_model = _run_arm(
        prepared, cfg, model_id=MATCHED_MODEL_ID, rho=0.0
    )
    print("\n===== fixed-budget N/V response rho=0.05 | seed 42 =====")
    active, active_model = _run_arm(
        prepared, cfg, model_id=MODEL_ID, rho=cfg.rho
    )
    print("\n===== degree-matched CLV shuffle rho=0.05 | seed 42 =====")
    shuffled, shuffled_model = _run_arm(
        prepared,
        cfg,
        model_id=SHUFFLED_MODEL_ID,
        rho=cfg.rho,
        clv_assignment=prepared["degree_matched_shuffle"],
        assignment_name="degree_matched_shuffle",
    )

    id_view = shared._IDOnlyView(active_model).to(v3.DEVICE)
    id_raw, _ = moe._flat_evaluation(
        id_view,
        0.0,
        prepared["cache"],
        prepared["meta"],
        prepared["data"],
        prepared["base_cfg"],
        per_user=False,
    )
    id_metrics = test10._public_metrics(id_raw)
    component_metrics = {}
    for component, model_id in (("n", N_ONLY_MODEL_ID), ("v", V_ONLY_MODEL_ID)):
        view = _ComponentView(active_model, component).to(v3.DEVICE)
        raw, _ = moe._flat_evaluation(
            view,
            0.0,
            prepared["cache"],
            prepared["meta"],
            prepared["data"],
            prepared["base_cfg"],
            per_user=False,
        )
        component_metrics[model_id] = test10._public_metrics(raw)

    users, matched_top50 = helpers._masked_topk(
        matched_model, prepared, max_k=cfg.diagnostic_max_k
    )
    active_users, active_top50 = helpers._masked_topk(
        active_model, prepared, max_k=cfg.diagnostic_max_k
    )
    shuffled_users, shuffled_top50 = helpers._masked_topk(
        shuffled_model, prepared, max_k=cfg.diagnostic_max_k
    )
    if not (
        np.array_equal(users, active_users)
        and np.array_equal(users, shuffled_users)
    ):
        raise RuntimeError("대조군과 M2 평가 사용자 순서가 다릅니다")
    overlap = helpers.topk_overlap_summary(
        matched_top50, active_top50, prepared["cache"].seg, k=10
    )
    overlap.insert(0, "reference", MATCHED_MODEL_ID)
    overlap.insert(1, "model_id", MODEL_ID)
    attribution_overlap = helpers.topk_overlap_summary(
        shuffled_top50, active_top50, prepared["cache"].seg, k=10
    )
    attribution_overlap.insert(0, "reference", SHUFFLED_MODEL_ID)
    attribution_overlap.insert(1, "model_id", MODEL_ID)
    score_diagnostics = _score_diagnostics(
        active_model, users, active_top50, prepared
    )

    baseline = dict(prepared["baseline"])
    baseline["role"] = "reused_baseline_display_only"
    rows = [baseline]
    for arm in (matched, active, shuffled):
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
        SHUFFLED_MODEL_ID: shuffled["metrics"],
        ID_ONLY_MODEL_ID: id_metrics,
        **component_metrics,
    }
    comparison = helpers._metric_comparison(
        metric_rows,
        references=(MATCHED_MODEL_ID, "m1_64", SHUFFLED_MODEL_ID, ID_ONLY_MODEL_ID),
    )
    reading = _screening_reading(
        matched["metrics"],
        active["metrics"],
        shuffled["metrics"],
        id_metrics,
        overlap,
        matched["diagnostics"],
    )
    reading["degree_matched_shuffle_changed_valid_user_share"] = prepared[
        "degree_matched_shuffle"
    ]["changed_valid_user_share"]

    stem = f"m2_fixed_budget_nv_response_{prepared['config_hash']}"
    paths = {
        "absolute_csv": prepared["out_dir"] / f"{stem}.csv",
        "comparison_csv": prepared["out_dir"] / f"{stem}_comparison.csv",
        "top10_overlap_csv": prepared["out_dir"] / f"{stem}_top10_overlap.csv",
        "attribution_overlap_csv": (
            prepared["out_dir"] / f"{stem}_attribution_overlap.csv"
        ),
        "score_diagnostics_csv": (
            prepared["out_dir"] / f"{stem}_score_diagnostics.csv"
        ),
        "json": prepared["out_dir"] / f"{stem}.json",
    }
    test10._atomic_csv(paths["absolute_csv"], frame)
    test10._atomic_csv(paths["comparison_csv"], comparison)
    test10._atomic_csv(paths["top10_overlap_csv"], overlap)
    test10._atomic_csv(paths["attribution_overlap_csv"], attribution_overlap)
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
        "attribution_overlap_rows": attribution_overlap.to_dict("records"),
        "score_diagnostic_rows": score_diagnostics.to_dict("records"),
        "screening_reading": reading,
        "result_paths": {key: str(value) for key, value in paths.items()},
    }
    test10._atomic_json(paths["json"], payload)
    frame.attrs["comparison"] = comparison.to_dict("records")
    frame.attrs["top10_overlap"] = overlap.to_dict("records")
    frame.attrs["attribution_overlap"] = attribution_overlap.to_dict("records")
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
    print("\n1) 절대지표: M1, rho=0, 실제 CLV, degree-matched shuffle, ID/N/V ablation")
    print(frame.to_string(index=False))
    print("\n2) 동일 초기화 rho=0 대비 핵심 변화")
    print(key_table.to_string(index=False))
    print("\n3) Top-10 변경")
    print(overlap.to_string(index=False))
    print("\n4) 실제 CLV 대 degree-matched shuffle Top-10 변경")
    print(attribution_overlap.to_string(index=False))
    print("\n5) 실제 점수 영향력")
    print(score_diagnostics.to_string(index=False))
    print("\n6) 사전 판정 규칙 결과")
    print(json.dumps(reading, ensure_ascii=False, indent=2))
    print("\n7) 저장 파일")
    print(json.dumps(frame.attrs["result_paths"], ensure_ascii=False, indent=2))
    return frame


if __name__ == "__main__":
    print(
        json.dumps(
            preflight_summary(configure_fixed_budget_nv_response_run()),
            ensure_ascii=False,
            indent=2,
        )
    )
