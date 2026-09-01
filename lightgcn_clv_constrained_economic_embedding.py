"""Seed-42 screen with explicit CLV composition and price coordinates."""

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


CODE_VERSION = "m2-clv-level-composition-price-embedding-historical-screen-v1"
RHO10_CODE_VERSION = (
    "m2-clv-level-composition-price-embedding-rho10-degree-shuffle-v1"
)
MATCHED_MODEL_ID = "m1_matched_rho0"
MODEL_ID = "m2_clv_level_composition_price_embedding"
SHUFFLED_MODEL_ID = "m2_degree_matched_clv_shuffle"
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
    clv_dim: int = 3
    rho: float = 0.05
    item_price_budget: float = 0.25
    n_layers: int = 2
    batch_size: int = 8192
    lr: float = 5e-4
    pref_reg: float = 1e-3
    input_days: int = 365
    diagnostic_max_k: int = 50
    include_degree_matched_shuffle: bool = False
    shuffle_degree_bins: int = 10
    shuffle_seed: int = 42
    out_dir: str = ""
    baseline_result_dir: str = ""


def configure_constrained_economic_run(**overrides) -> ConstrainedEconomicConfig:
    rho10_attribution = bool(
        overrides.get("include_degree_matched_shuffle", False)
    )
    defaults = {
        "out_dir": (
            f"{v3.default_out_dir('dunnhumby')}"
            + (
                "_m2_clv_level_composition_price_embedding_rho10_shuffle_v1"
                if rho10_attribution
                else "_m2_clv_level_composition_price_embedding_historical_screen_v1"
            )
        ),
        "baseline_result_dir": (
            f"{v3.default_out_dir('dunnhumby')}"
            "_m2_repeatshare_historical_backtest_v1"
        ),
    }
    return validate_constrained_economic_config(
        ConstrainedEconomicConfig(**(defaults | overrides))
    )


def configure_rho10_attribution_run(**overrides) -> ConstrainedEconomicConfig:
    """One bounded follow-up: rho=.10 plus a degree-matched CLV shuffle."""

    return configure_constrained_economic_run(
        rho=0.10,
        include_degree_matched_shuffle=True,
        shuffle_degree_bins=10,
        shuffle_seed=42,
        **overrides,
    )


def _code_version(cfg: ConstrainedEconomicConfig) -> str:
    return RHO10_CODE_VERSION if cfg.include_degree_matched_shuffle else CODE_VERSION


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
        "clv_dim": 3,
        "item_price_budget": 0.25,
        "n_layers": 2,
        "input_days": 365,
        "diagnostic_max_k": 50,
    }
    for key, expected in required.items():
        if getattr(cfg, key) != expected:
            raise ValueError(f"빠른 M2 screen은 {key}={expected!r}이어야 합니다")
    if cfg.include_degree_matched_shuffle:
        if cfg.rho != 0.10:
            raise ValueError("귀속 screen의 CLV 강도는 rho=0.10이어야 합니다")
        if cfg.shuffle_degree_bins != 10 or cfg.shuffle_seed != 42:
            raise ValueError("귀속 screen은 degree 10분위·shuffle seed 42를 사용합니다")
    elif cfg.rho != 0.05:
        raise ValueError("기존 screen의 CLV 강도는 rho=0.05이어야 합니다")
    if cfg.batch_size <= 0 or cfg.lr <= 0 or cfg.pref_reg < 0:
        raise ValueError("학습 설정이 잘못됐습니다")
    if not cfg.out_dir or not cfg.baseline_result_dir:
        raise ValueError("out_dir와 baseline_result_dir가 필요합니다")
    return cfg


def preflight_summary(cfg: ConstrainedEconomicConfig) -> dict:
    cfg = validate_constrained_economic_config(cfg)
    return {
        "code_version": _code_version(cfg),
        "dataset": cfg.dataset,
        "seed": cfg.seed,
        "trained_models": (
            [MATCHED_MODEL_ID, MODEL_ID, SHUFFLED_MODEL_ID]
            if cfg.include_degree_matched_shuffle
            else [MATCHED_MODEL_ID, MODEL_ID]
        ),
        "reused_comparator": "m1_64 (display only)",
        "historical_development_split": {
            "train_end_inclusive": 683,
            "evaluation_start_inclusive": 684,
            "evaluation_end_inclusive": 690,
            "final_test_constructed": False,
            "holdout_constructed": False,
        },
        "m2": {
            "architecture": "ID(64)|CLV relation(2)|explicit price fit(1)",
            "total_dim": cfg.id_dim + cfg.clv_dim,
            "user_block": "[sqrt(1-beta)*q_C*unit([q_C,q_N-q_V])(2)|sqrt(beta)*q_C*(2q_V-1)(1)]",
            "item_block": "[sqrt(1-beta)*unit(P_z E_i^ID)(2)|sqrt(beta)*positive_mix(centred prices)(1)]",
            "user_tanh": False,
            "learned_user_projection": False,
            "free_item_response_embedding": False,
            "item_inputs": [
                "existing item ID embedding projected to 2 dimensions",
                "overall price percentile",
                "within-category price percentile",
            ],
            "price_mixer": "two positive weights summing to one; learned by the same BPR",
            "item_price_budget": cfg.item_price_budget,
            "joint_graph_propagation": True,
            "one_dot_score": True,
            "rho": cfg.rho,
            "symmetric_scale": "sqrt(rho) on user and item CLV blocks",
            "repeatshare_input": False,
            "item_popularity_input": False,
            "external_reranking": False,
            "degree_matched_clv_shuffle": cfg.include_degree_matched_shuffle,
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
            "semantic_attribution": (
                "when enabled, observed CLV must beat the degree-matched shuffle "
                "on six-metric accuracy geometric mean and weighted hit@10, "
                "and on either high-CLV Recall@10 or NDCG@10"
            ),
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
        "code_version": _code_version(cfg),
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
    if cfg.include_degree_matched_shuffle:
        prepared["degree_matched_shuffle"] = _degree_matched_clv_shuffle(
            prepared, cfg
        )
    return prepared


def _degree_matched_clv_shuffle(
    prepared: dict, cfg: ConstrainedEconomicConfig
) -> dict:
    """Jointly permute N/V/CLV tuples inside binary user-degree deciles."""

    data = prepared["data"]
    train_edges = data["train"][["u_idx", "i_idx"]].drop_duplicates()
    user_degree = np.bincount(
        train_edges["u_idx"].to_numpy(np.int64), minlength=data["n_users"]
    )
    valid = np.asarray(prepared["clv_valid"], dtype=bool)
    valid_index = np.flatnonzero(valid & (user_degree > 0))
    if len(valid_index) < 2:
        raise RuntimeError("CLV 순열에 사용할 유효 고객이 부족합니다")

    ranks = pd.Series(user_degree[valid_index]).rank(method="average").to_numpy()
    strata_valid = np.floor(
        (ranks - 0.5) * cfg.shuffle_degree_bins / len(valid_index)
    ).astype(np.int16)
    strata_valid = np.minimum(strata_valid, cfg.shuffle_degree_bins - 1)
    source = np.arange(data["n_users"], dtype=np.int64)
    strata = np.full(data["n_users"], -1, dtype=np.int16)
    strata[valid_index] = strata_valid
    rng = np.random.default_rng(cfg.shuffle_seed)
    for stratum in np.unique(strata_valid):
        target = valid_index[strata_valid == stratum]
        if len(target) < 2:
            continue
        permuted = rng.permutation(target)
        if np.array_equal(permuted, target):
            permuted = np.roll(permuted, 1)
        source[target] = permuted

    changed = valid_index[source[valid_index] != valid_index]
    if not len(changed):
        raise RuntimeError("degree-matched CLV 순열이 고객 배정을 바꾸지 못했습니다")
    if np.any(strata[changed] != strata[source[changed]]):
        raise RuntimeError("CLV 순열이 사용자 degree 구간을 벗어났습니다")
    return {
        "q_n": np.asarray(prepared["q_n"], dtype=np.float32)[source],
        "q_v": np.asarray(prepared["q_v"], dtype=np.float32)[source],
        "q_c": np.asarray(prepared["q_c"], dtype=np.float32)[source],
        "clv_valid": valid[source],
        "source_user": source,
        "stratum": strata,
        "user_degree": user_degree,
        "changed_valid_user_share": float(len(changed) / len(valid_index)),
    }


def _build_model(
    prepared: dict,
    cfg: ConstrainedEconomicConfig,
    rho: float,
    *,
    clv_assignment: dict | None = None,
):
    data = prepared["data"]
    assignment = clv_assignment or prepared
    v3.set_seed(cfg.seed)
    model = ConstrainedCLVEconomicLightGCN(
        n_users=data["n_users"],
        n_items=data["n_items"],
        q_n=assignment["q_n"],
        q_v=assignment["q_v"],
        q_c=assignment["q_c"],
        user_clv_valid=assignment["clv_valid"],
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


def _arm_hash(
    prepared: dict, model_id: str, rho: float, assignment: str
) -> str:
    payload = {
        "run": prepared["config_hash"],
        "model_id": model_id,
        "rho": rho,
        "clv_assignment": assignment,
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
    clv_assignment: dict | None = None,
    assignment_name: str = "observed",
) -> tuple[dict, ConstrainedCLVEconomicLightGCN]:
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
    if rho == 0.0 and cfg.include_degree_matched_shuffle:
        legacy_root = Path(
            f"{v3.default_out_dir('dunnhumby')}"
            "_m2_clv_level_composition_price_embedding_historical_screen_v1"
        )
        legacy_results = sorted(
            legacy_root.glob(f"arms/*/{MATCHED_MODEL_ID}_s42.json")
        )
        for result_path in reversed(legacy_results):
            checkpoint_path = result_path.with_suffix(".pt")
            if not checkpoint_path.exists():
                continue
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            checkpoint = helpers._load_state(checkpoint_path)
            if (
                checkpoint.get("input_hash") != prepared["input_hash"]
                or float(checkpoint.get("rho", -1.0)) != 0.0
            ):
                continue
            model.load_state_dict(checkpoint["state"], strict=True)
            model.eval()
            payload["role"] = "reused_matched_control"
            payload["source_result"] = str(result_path)
            print("  [reused] 기존 rho=0 matched checkpoint 재사용")
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
    print(f"\n===== constrained CLV-economic rho={cfg.rho:.2f} | seed 42 =====")
    active, active_model = _run_arm(
        prepared, cfg, model_id=MODEL_ID, rho=cfg.rho
    )
    shuffled = shuffled_model = None
    if cfg.include_degree_matched_shuffle:
        print("\n===== degree-matched CLV shuffle rho=0.10 | seed 42 =====")
        shuffled, shuffled_model = _run_arm(
            prepared,
            cfg,
            model_id=SHUFFLED_MODEL_ID,
            rho=cfg.rho,
            clv_assignment=prepared["degree_matched_shuffle"],
            assignment_name="degree_matched_shuffle",
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
    overlap.insert(0, "reference", MATCHED_MODEL_ID)
    overlap.insert(1, "model_id", MODEL_ID)
    attribution_overlap = pd.DataFrame()
    if shuffled_model is not None:
        shuffled_users, shuffled_top50 = helpers._masked_topk(
            shuffled_model, prepared, max_k=cfg.diagnostic_max_k
        )
        if not np.array_equal(users, shuffled_users):
            raise RuntimeError("실제 CLV와 순열 CLV 평가 사용자 순서가 다릅니다")
        attribution_overlap = helpers.topk_overlap_summary(
            shuffled_top50, active_top50, prepared["cache"].seg, k=10
        )
        attribution_overlap.insert(0, "reference", SHUFFLED_MODEL_ID)
        attribution_overlap.insert(1, "model_id", MODEL_ID)
    score_diagnostics = shared._score_diagnostics(
        active_model, users, active_top50, prepared
    )

    baseline = dict(prepared["baseline"])
    baseline["role"] = "reused_baseline_display_only"
    rows = [baseline]
    trained_arms = [matched, active]
    if shuffled is not None:
        trained_arms.append(shuffled)
    for arm in trained_arms:
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
    if shuffled is not None:
        metric_rows[SHUFFLED_MODEL_ID] = shuffled["metrics"]
    references = [MATCHED_MODEL_ID, "m1_64"]
    if shuffled is not None:
        references.append(SHUFFLED_MODEL_ID)
    comparison = helpers._metric_comparison(
        metric_rows, references=tuple(references)
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
    if shuffled is not None:
        accuracy_metrics = (
            "recall@10",
            "ndcg@10",
            "recall@20",
            "ndcg@20",
            "recall@50",
            "ndcg@50",
        )
        accuracy_ratios_vs_shuffle = {
            metric: float(active["metrics"][metric] / shuffled["metrics"][metric])
            for metric in accuracy_metrics
        }
        accuracy_balance_vs_shuffle = float(
            np.exp(np.mean(np.log(list(accuracy_ratios_vs_shuffle.values()))))
        )
        weighted_delta_vs_shuffle = float(
            active["metrics"]["price_purchase_amount_weighted_hit@10"]
            - shuffled["metrics"]["price_purchase_amount_weighted_hit@10"]
        )
        high_clv_accuracy_vs_shuffle = bool(
            active["metrics"]["고CLV_recall@10"]
            > shuffled["metrics"]["고CLV_recall@10"]
            or active["metrics"]["고CLV_ndcg@10"]
            > shuffled["metrics"]["고CLV_ndcg@10"]
        )
        attribution_supported = bool(
            accuracy_balance_vs_shuffle > 1.0
            and weighted_delta_vs_shuffle > 0.0
            and high_clv_accuracy_vs_shuffle
        )
        reading.update(
            {
                "accuracy_ratios_vs_degree_matched_shuffle": (
                    accuracy_ratios_vs_shuffle
                ),
                "accuracy_geomean_ratio_vs_degree_matched_shuffle": (
                    accuracy_balance_vs_shuffle
                ),
                "weighted_hit_at_10_delta_vs_degree_matched_shuffle": (
                    weighted_delta_vs_shuffle
                ),
                "high_clv_accuracy_better_than_degree_matched_shuffle": (
                    high_clv_accuracy_vs_shuffle
                ),
                "clv_attribution_supported": attribution_supported,
                "degree_matched_shuffle_changed_valid_user_share": prepared[
                    "degree_matched_shuffle"
                ]["changed_valid_user_share"],
            }
        )
        reading["positive_screen"] = bool(
            reading["positive_screen"] and attribution_supported
        )

    stem = f"m2_clv_level_composition_price_embedding_{prepared['config_hash']}"
    paths = {
        "absolute_csv": prepared["out_dir"] / f"{stem}.csv",
        "comparison_csv": prepared["out_dir"] / f"{stem}_comparison.csv",
        "top10_overlap_csv": prepared["out_dir"] / f"{stem}_top10_overlap.csv",
        "attribution_overlap_csv": (
            prepared["out_dir"] / f"{stem}_attribution_overlap.csv"
        ),
        "score_diagnostics_csv": prepared["out_dir"] / f"{stem}_score_diagnostics.csv",
        "json": prepared["out_dir"] / f"{stem}.json",
    }
    test10._atomic_csv(paths["absolute_csv"], frame)
    test10._atomic_csv(paths["comparison_csv"], comparison)
    test10._atomic_csv(paths["top10_overlap_csv"], overlap)
    test10._atomic_csv(paths["attribution_overlap_csv"], attribution_overlap)
    test10._atomic_csv(paths["score_diagnostics_csv"], score_diagnostics)
    payload = {
        "code_version": _code_version(cfg),
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
    print("\n절대지표:")
    print(frame.to_string(index=False))
    print("\n동일 초기화 rho=0 대비 핵심 변화:")
    print(key_table.to_string(index=False))
    print("\nTop-10 변경 진단:")
    print(overlap.to_string(index=False))
    if not attribution_overlap.empty:
        print("\n실제 CLV 대 degree-matched shuffle Top-10 변경:")
        print(attribution_overlap.to_string(index=False))
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
