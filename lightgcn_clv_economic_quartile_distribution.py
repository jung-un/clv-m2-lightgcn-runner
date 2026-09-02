"""Seed-42 M2 screen using a CLV-conditioned economic-quartile profile."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from clv_economic_quartile_distribution_model import (
    CLVEconomicQuartileDistributionLightGCN,
)
from clv_run_state import ProgressStore, RunIdentity, clone_state, file_sha256
import lightgcn_clv_axis_specific_test10 as test10
import lightgcn_clv_constrained_economic_embedding as prior
import lightgcn_clv_gated_relation_overall_price as common
import lightgcn_clv_gradient_isolated_economic_interaction as helpers
import lightgcn_clv_joint_response_embedding as shared
import lightgcn_clv_moe as moe
import lightgcn_clv_v3 as v3


CODE_VERSION = "m2-clv-economic-quartile-distribution-historical-screen-v1"
MATCHED_MODEL_ID = "m1_matched_rho0_economic_quartile"
MODEL_ID = "m2_clv_economic_quartile_distribution"
SHUFFLED_MODEL_ID = "m2_degree_matched_clv_shuffle_economic_quartile"
ID_ONLY_MODEL_ID = "m2_jointly_trained_id_only_economic_quartile"


@dataclass(frozen=True)
class EconomicQuartileConfig:
    dataset: str = "dunnhumby"
    seed: int = 42
    time_cutoff: int = 690
    evaluation_days: int = 7
    epochs: int = 100
    id_dim: int = 64
    economic_bins: int = 4
    shrinkage_strength: float = 10.0
    rho: float = 0.05
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


def configure_economic_quartile_run(**overrides) -> EconomicQuartileConfig:
    defaults = {
        "out_dir": (
            f"{v3.default_out_dir('dunnhumby')}"
            "_m2_clv_economic_quartile_distribution_historical_screen_v1"
        ),
        "baseline_result_dir": (
            f"{v3.default_out_dir('dunnhumby')}"
            "_m2_repeatshare_historical_backtest_v1"
        ),
    }
    return validate_economic_quartile_config(
        EconomicQuartileConfig(**(defaults | overrides))
    )


def validate_economic_quartile_config(
    cfg: EconomicQuartileConfig,
) -> EconomicQuartileConfig:
    required = {
        "dataset": "dunnhumby",
        "seed": 42,
        "time_cutoff": 690,
        "evaluation_days": 7,
        "epochs": 100,
        "id_dim": 64,
        "economic_bins": 4,
        "shrinkage_strength": 10.0,
        "rho": 0.05,
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


def preflight_summary(cfg: EconomicQuartileConfig) -> dict:
    cfg = validate_economic_quartile_config(cfg)
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
            "architecture": "ID(64)|CLV-conditioned economic quartiles(4)",
            "total_dim": cfg.id_dim + cfg.economic_bins,
            "historical_clv": "q_C=training-user percentile of N_hat*V_hat",
            "item_economic_bins": (
                "four equal-item-count bins of train-only median unit price; "
                "stable item-index tie break"
            ),
            "user_economic_profile": (
                "purchase-amount share across four item-price bins"
            ),
            "population_centering": "user profile minus valid-user mean profile",
            "shrinkage": (
                "n_unique_items/(n_unique_items+kappa), kappa=10; "
                "no post-shrink unit normalization"
            ),
            "user_coordinate": "q_C * shrunk centered four-bin spend profile",
            "item_coordinate": "centered one-hot economic bin",
            "relative_bin_weight": (
                "four softmax weights learned by the same BPR; sum fixed to one"
            ),
            "learned_global_scale": False,
            "rho": cfg.rho,
            "symmetric_scale": "sqrt(rho) on both user and item blocks",
            "layer0_intervention": True,
            "joint_graph_propagation": True,
            "one_dot_score": True,
            "repeatshare_input": False,
            "item_degree_input": False,
            "category_input": False,
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
            "primary": (
                "price_purchase_amount_weighted_hit@10 must beat matched rho=0, "
                "jointly-trained ID-only, and degree-matched CLV shuffle"
            ),
            "high_clv": "high-CLV weighted hit@10 must beat matched rho=0",
            "accuracy_guardrail": (
                "all Recall/NDCG@10/20/50 >= 99% of matched rho=0"
            ),
            "liveness": (
                "economic score std / ID score std >= 0.001 and high-CLV "
                "Top-10 set changes"
            ),
            "statistical_note": "seed 42 exploratory screen; no significance claim",
        },
        "out_dir": cfg.out_dir,
        "baseline_result_dir": cfg.baseline_result_dir,
    }


def build_economic_quartile_inputs(
    train: pd.DataFrame,
    *,
    n_users: int,
    n_items: int,
    n_bins: int = 4,
    shrinkage_strength: float = 10.0,
) -> dict[str, np.ndarray | dict]:
    """Build train-only item bins and shrinkage-stabilised user spend profiles."""

    required = {"u_idx", "i_idx", "up", "v"}
    missing = required - set(train.columns)
    if missing:
        raise KeyError(f"경제구간 입력 컬럼 누락: {sorted(missing)}")
    if n_bins <= 1 or shrinkage_strength < 0:
        raise ValueError("n_bins와 shrinkage_strength 설정이 잘못됐습니다")

    item_price = (
        train.groupby("i_idx", sort=True)["up"]
        .median()
        .reindex(np.arange(n_items))
        .to_numpy(np.float64)
    )
    item_valid = np.isfinite(item_price) & (item_price >= 0.0)
    valid_items = np.flatnonzero(item_valid)
    if len(valid_items) < n_bins:
        raise RuntimeError("경제구간을 만들 유효 상품이 부족합니다")
    # Equal item counts are deterministic even when multiple products share a price.
    order = np.lexsort((valid_items, np.log1p(item_price[valid_items])))
    item_bin = np.full(n_items, -1, dtype=np.int16)
    assigned = np.floor(np.arange(len(valid_items)) * n_bins / len(valid_items))
    item_bin[valid_items[order]] = np.minimum(assigned, n_bins - 1).astype(np.int16)

    users = train["u_idx"].to_numpy(np.int64, copy=False)
    items = train["i_idx"].to_numpy(np.int64, copy=False)
    amount = np.maximum(train["v"].to_numpy(np.float64, copy=True), 0.0)
    valid_row = np.isfinite(amount) & (amount > 0.0) & item_valid[items]
    spend = np.zeros((n_users, n_bins), dtype=np.float64)
    np.add.at(spend, (users[valid_row], item_bin[items[valid_row]]), amount[valid_row])
    total = spend.sum(axis=1)
    profile_valid = total > 0.0
    raw_profile = np.divide(
        spend,
        total[:, None],
        out=np.zeros_like(spend),
        where=profile_valid[:, None],
    )
    population = raw_profile[profile_valid].mean(axis=0)

    unique_pairs = train.loc[valid_row, ["u_idx", "i_idx"]].drop_duplicates()
    observation_count = np.bincount(
        unique_pairs["u_idx"].to_numpy(np.int64), minlength=n_users
    ).astype(np.float64)
    reliability = observation_count / (observation_count + shrinkage_strength)
    normalizer = np.sqrt(2.0)
    user_profile = (
        reliability[:, None] * (raw_profile - population[None, :]) / normalizer
    )
    user_profile[~profile_valid] = 0.0
    item_basis = np.zeros((n_items, n_bins), dtype=np.float64)
    item_basis[valid_items] = -population[None, :]
    item_basis[valid_items, item_bin[valid_items]] += 1.0
    item_basis /= normalizer

    if not np.allclose(raw_profile[profile_valid].sum(axis=1), 1.0, atol=1e-7):
        raise RuntimeError("사용자 구간별 지출비중의 합이 1이 아닙니다")
    if not np.isfinite(user_profile).all() or not np.isfinite(item_basis).all():
        raise RuntimeError("경제구간 표현에 비유한 값이 있습니다")
    bin_counts = np.bincount(item_bin[valid_items], minlength=n_bins)
    diagnostics = {
        "economic_bin_count": int(n_bins),
        "valid_item_count": int(len(valid_items)),
        "valid_user_share": float(profile_valid.mean()),
        "median_unique_item_count": float(np.median(observation_count[profile_valid])),
        "mean_shrinkage_reliability": float(reliability[profile_valid].mean()),
        "population_spend_distribution": population.tolist(),
        "item_count_by_bin": bin_counts.tolist(),
        "item_count_bin_imbalance": int(bin_counts.max() - bin_counts.min()),
        "profile_row_sum_max_error": float(
            np.max(np.abs(raw_profile[profile_valid].sum(axis=1) - 1.0))
        ),
    }
    return {
        "user_economic_profile": user_profile.astype(np.float32),
        "user_profile_valid": profile_valid,
        "user_profile_reliability": reliability.astype(np.float32),
        "user_unique_item_count": observation_count.astype(np.int64),
        "item_economic_basis": item_basis.astype(np.float32),
        "item_economic_bin": item_bin,
        "item_economic_valid": item_valid,
        "economic_population_distribution": population.astype(np.float32),
        "economic_input_diagnostics": diagnostics,
    }


def _config_hash(
    cfg: EconomicQuartileConfig, input_hash: str, revision: str
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


def _prepare(cfg: EconomicQuartileConfig) -> dict:
    prepared = common._prepare(cfg)
    prepared.update(
        build_economic_quartile_inputs(
            prepared["data"]["train"],
            n_users=prepared["data"]["n_users"],
            n_items=prepared["data"]["n_items"],
            n_bins=cfg.economic_bins,
            shrinkage_strength=cfg.shrinkage_strength,
        )
    )
    prepared["config_hash"] = _config_hash(
        cfg, prepared["input_hash"], prepared["revision"]
    )
    prepared["degree_matched_shuffle"] = prior._degree_matched_clv_shuffle(
        prepared, cfg
    )
    return prepared


def _build_model(
    prepared: dict,
    cfg: EconomicQuartileConfig,
    rho: float,
    *,
    clv_assignment: dict | None = None,
) -> tuple[CLVEconomicQuartileDistributionLightGCN, list[torch.nn.Parameter]]:
    data = prepared["data"]
    assignment = clv_assignment or prepared
    v3.set_seed(cfg.seed)
    model = CLVEconomicQuartileDistributionLightGCN(
        n_users=data["n_users"],
        n_items=data["n_items"],
        q_c=assignment["q_c"],
        user_clv_valid=assignment["clv_valid"],
        user_economic_profile=prepared["user_economic_profile"],
        user_profile_valid=prepared["user_profile_valid"],
        item_economic_basis=prepared["item_economic_basis"],
        item_economic_valid=prepared["item_economic_valid"],
        adj=data["adj"],
        id_dim=cfg.id_dim,
        rho=rho,
        n_layers=cfg.n_layers,
        pref_reg=cfg.pref_reg,
    ).to(v3.DEVICE)
    return model, list(model.parameters())


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
    cfg: EconomicQuartileConfig,
    *,
    model_id: str,
    rho: float,
    clv_assignment: dict | None = None,
    assignment_name: str = "observed",
) -> tuple[dict, CLVEconomicQuartileDistributionLightGCN]:
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
        "diagnostics": (
            model.representation_diagnostics()
            | prepared["economic_input_diagnostics"]
        ),
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
    model: CLVEconomicQuartileDistributionLightGCN,
    users: np.ndarray,
    top50: np.ndarray,
    prepared: dict,
) -> pd.DataFrame:
    width = top50.shape[1]
    pair_users = np.repeat(users.astype(np.int64), width)
    pair_items = top50.reshape(-1).astype(np.int64)
    collected = {key: [] for key in ("id", "economic", "full")}
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
    economic_std = float(values["economic"].std())
    per_user_abs = np.abs(values["economic"]).reshape(len(users), width).mean(axis=1)
    weights = model.economic_bin_weights().detach().cpu().numpy()
    row = {
        "candidate_pair_count": len(pair_users),
        "id_score_std": id_std,
        "economic_score_std": economic_std,
        "economic_score_std_ratio_to_id": (
            economic_std / id_std if id_std > 0 else np.nan
        ),
        "economic_score_mean_abs": float(np.abs(values["economic"]).mean()),
        "per_user_economic_abs_q_c_spearman": float(
            pd.Series(per_user_abs).corr(
                pd.Series(prepared["q_c"][users]), method="spearman"
            )
        ),
        "per_user_economic_abs_reliability_spearman": float(
            pd.Series(per_user_abs).corr(
                pd.Series(prepared["user_profile_reliability"][users]),
                method="spearman",
            )
        ),
        "max_full_decomposition_error": float(
            np.max(np.abs(values["full"] - values["id"] - values["economic"]))
        ),
    }
    for index, value in enumerate(weights, start=1):
        row[f"economic_bin_{index}_weight"] = float(value)
    return pd.DataFrame([row])


def _screening_reading(
    matched: dict,
    active: dict,
    shuffled: dict,
    id_only: dict,
    overlap: pd.DataFrame,
    score_diagnostics: pd.DataFrame,
    rho0_diagnostics: dict,
) -> dict:
    accuracy_metrics = (
        "recall@10",
        "ndcg@10",
        "recall@20",
        "ndcg@20",
        "recall@50",
        "ndcg@50",
    )
    accuracy_ratios = {
        metric: float(active[metric] / matched[metric])
        for metric in accuracy_metrics
    }
    economic = "price_purchase_amount_weighted_hit@10"
    high_economic = "고CLV_revenue@10"
    weighted_deltas = {
        "vs_matched_rho0": float(active[economic] - matched[economic]),
        "vs_jointly_trained_id_only": float(active[economic] - id_only[economic]),
        "vs_degree_matched_clv_shuffle": float(active[economic] - shuffled[economic]),
    }
    high_weighted_delta = float(active[high_economic] - matched[high_economic])
    high_changed = float(
        overlap.set_index("group").at["고CLV", "top10_set_changed_user_share"]
    )
    score_ratio = float(score_diagnostics.iloc[0]["economic_score_std_ratio_to_id"])
    rho0_exact = float(rho0_diagnostics["rho_zero_auxiliary_max_abs"]) == 0.0
    positive = bool(
        min(accuracy_ratios.values()) >= 0.99
        and all(delta > 0.0 for delta in weighted_deltas.values())
        and high_weighted_delta > 0.0
        and score_ratio >= 0.001
        and high_changed > 0.0
        and rho0_exact
    )
    return {
        "positive_screen": positive,
        "accuracy_guardrail_pass": min(accuracy_ratios.values()) >= 0.99,
        "accuracy_ratios_vs_matched_rho0": accuracy_ratios,
        "weighted_hit_at_10_deltas": weighted_deltas,
        "high_clv_weighted_hit_at_10_delta_vs_matched": high_weighted_delta,
        "economic_score_std_ratio_to_id": score_ratio,
        "economic_score_liveness_pass": score_ratio >= 0.001,
        "high_clv_top10_changed_user_share": high_changed,
        "rho0_exact_nonintervention": rho0_exact,
        "next_if_positive": (
            "run several Dunnhumby development seeds, then a separately "
            "approved H&M shrinkage screen before final test"
        ),
        "statistical_note": "seed 42 exploratory screen; no significance claim",
    }


def run_economic_quartile_screen(
    cfg: EconomicQuartileConfig | None = None,
) -> pd.DataFrame:
    cfg = validate_economic_quartile_config(
        cfg or configure_economic_quartile_run()
    )
    print(json.dumps(preflight_summary(cfg), ensure_ascii=False, indent=2))
    prepared = _prepare(cfg)

    print("\n===== matched rho=0 | seed 42 | fixed 100 epochs =====")
    matched, matched_model = _run_arm(
        prepared, cfg, model_id=MATCHED_MODEL_ID, rho=0.0
    )
    print("\n===== CLV economic-quartile distribution rho=0.05 | seed 42 =====")
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
        score_diagnostics,
        matched["diagnostics"],
    )
    reading["degree_matched_shuffle_changed_valid_user_share"] = prepared[
        "degree_matched_shuffle"
    ]["changed_valid_user_share"]

    stem = f"m2_clv_economic_quartile_distribution_{prepared['config_hash']}"
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
        "economic_input_diagnostics": prepared["economic_input_diagnostics"],
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
    frame.attrs["economic_input_diagnostics"] = prepared[
        "economic_input_diagnostics"
    ]
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
        "고CLV_revenue@10",
    )
    key_table = comparison[
        (comparison.reference == MATCHED_MODEL_ID)
        & (comparison.model_id == MODEL_ID)
        & comparison.metric.isin(key_metrics)
    ]
    print("\n1) 절대지표: M1, rho=0, 실제 CLV, degree-matched shuffle, ID-only")
    print(frame.to_string(index=False))
    print("\n2) 동일 초기화 rho=0 대비 핵심 변화")
    print(key_table.to_string(index=False))
    print("\n3) rho=0 대비 CLV 구간별 Top-10 변경")
    print(overlap.to_string(index=False))
    print("\n4) 실제 CLV 대 degree-matched shuffle Top-10 변경")
    print(attribution_overlap.to_string(index=False))
    print("\n5) 경제구간 입력 진단")
    print(json.dumps(prepared["economic_input_diagnostics"], ensure_ascii=False, indent=2))
    print("\n6) 실제 점수 영향력")
    print(score_diagnostics.to_string(index=False))
    print("\n7) 사전 판정 규칙 결과")
    print(json.dumps(reading, ensure_ascii=False, indent=2))
    print("\n8) 저장 파일")
    print(json.dumps(frame.attrs["result_paths"], ensure_ascii=False, indent=2))
    return frame


if __name__ == "__main__":
    print(
        json.dumps(
            preflight_summary(configure_economic_quartile_run()),
            ensure_ascii=False,
            indent=2,
        )
    )
