"""Seed-42 development screen for a two-axis N/V M2 plus fixed M4."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from clv_m5_minimal_nv_model import M5MinimalNVEconomicLightGCN
import lightgcn_clv_gradient_isolated_economic_interaction as report_helpers
import lightgcn_clv_m5_economic_positive_weight as legacy
import lightgcn_clv_m5_nv_economic_positive_weight as nv
import lightgcn_clv_v3 as v3


CODE_VERSION = "m5-minimal-nv-personalized-positive-development-screen-v1"
M4_MODEL_ID = "m4_personalized_positive_weight_k5_minimal_nv_control"
M5_MODEL_ID = "m5_minimal_nv_personalized_positive_weight_k5"
M5_SHUFFLE_MODEL_ID = "m5_minimal_nv_representation_degree_shuffle"
MODEL_IDS = (M4_MODEL_ID, M5_MODEL_ID, M5_SHUFFLE_MODEL_ID)
PRIMARY_METRICS = (
    "price_purchase_amount_weighted_hit@10",
    "vndcg@10",
)
ACCURACY_METRICS = (
    "recall@10",
    "ndcg@10",
    "recall@20",
    "ndcg@20",
    "recall@50",
    "ndcg@50",
)


@dataclass(frozen=True)
class M5MinimalNVM4Config:
    dataset: str = "dunnhumby"
    seed: int = 42
    time_cutoff: int = 690
    evaluation_days: int = 7
    epochs: int = 100
    id_dim: int = 64
    economic_dim: int = 2
    economic_bins: int = 4
    shrinkage_strength: float = 10.0
    rho: float = 0.05
    scale_delta: float = 0.25
    positive_weight_lambda: float = 0.5
    n_layers: int = 2
    negative_count: int = 5
    batch_size: int = 8192
    lr: float = 5e-4
    pref_reg: float = 1e-3
    input_days: int = 365
    diagnostic_max_k: int = 50
    shuffle_degree_bins: int = 10
    shuffle_seed: int = 42
    out_dir: str = ""
    baseline_result_dir: str = ""


def configure_minimal_nv_m4_screen(**overrides) -> M5MinimalNVM4Config:
    defaults = {
        "out_dir": (
            f"{v3.default_out_dir('dunnhumby')}"
            "_m5_minimal_nv_personalized_positive_development_screen_v1"
        ),
        "baseline_result_dir": (
            f"{v3.default_out_dir('dunnhumby')}"
            "_m2_repeatshare_historical_backtest_v1"
        ),
    }
    return validate_config(M5MinimalNVM4Config(**(defaults | overrides)))


def validate_config(cfg: M5MinimalNVM4Config) -> M5MinimalNVM4Config:
    fixed = {
        "dataset": "dunnhumby",
        "seed": 42,
        "time_cutoff": 690,
        "evaluation_days": 7,
        "epochs": 100,
        "id_dim": 64,
        "economic_dim": 2,
        "economic_bins": 4,
        "shrinkage_strength": 10.0,
        "rho": 0.05,
        "scale_delta": 0.25,
        "positive_weight_lambda": 0.5,
        "n_layers": 2,
        "negative_count": 5,
        "input_days": 365,
        "diagnostic_max_k": 50,
        "shuffle_degree_bins": 10,
        "shuffle_seed": 42,
    }
    for key, expected in fixed.items():
        if getattr(cfg, key) != expected:
            raise ValueError(f"최소 N/V M5 screen은 {key}={expected!r}이어야 합니다")
    if cfg.batch_size <= 0 or cfg.lr <= 0 or cfg.pref_reg < 0:
        raise ValueError("학습 설정이 잘못됐습니다")
    if not cfg.out_dir or not cfg.baseline_result_dir:
        raise ValueError("out_dir와 baseline_result_dir가 필요합니다")
    return cfg


def preflight_summary(cfg: M5MinimalNVM4Config) -> dict:
    cfg = validate_config(cfg)
    return {
        "code_version": CODE_VERSION,
        "dataset": cfg.dataset,
        "seed": cfg.seed,
        "split": "historical_development_days_684_690",
        "trained_models": list(MODEL_IDS),
        "research_question": (
            "Does a minimal two-axis N/V representation improve the identical "
            "personalized CLV-positive M4 loss on new-item recommendation?"
        ),
        "m2": {
            "user_axes": "centered q_N frequency and centered q_V transaction value",
            "item_axes": (
                "train-only repeat-purchase propensity and item amount percentile"
            ),
            "economic_dim": cfg.economic_dim,
            "rho": cfg.rho,
            "learned_parameters": (
                "two positive bounded calibration scalars in [0.75,1.25]"
            ),
            "economic_graph_propagation": True,
            "joint_end_to_end_training": True,
            "q_c_used_in_m2": False,
        },
        "m4": {
            "formula": (
                "1 + 0.5*q_C*item_amount_percentile*clipped_user_bin_fit"
            ),
            "q_c": "percentile(n_u*v_u), not q_N*q_V",
            "uniform_negative_count": cfg.negative_count,
            "hard_negative": False,
            "same_in_all_arms": True,
        },
        "controls": {
            M4_MODEL_ID: "rho=0 nonintervention on the same model path",
            M5_MODEL_ID: "observed q_N and q_V",
            M5_SHUFFLE_MODEL_ID: (
                "only q_N and q_V jointly permuted within user-degree deciles; "
                "M4 remains observed"
            ),
        },
        "fixed": {
            "new_item_task": True,
            "train_pairs_excluded_from_evaluation": True,
            "min_item_interactions": 1,
            "graph": "binary",
            "negative_sampling": "uniform",
            "epochs": cfg.epochs,
            "validation_or_epoch_selection": False,
            "final_test_constructed": False,
            "holdout_constructed": False,
            "one_training_loop_and_optimizer": True,
            "external_reranking": False,
            "m3_edge_weight": False,
        },
        "reading_rule": {
            "primary_metrics": list(PRIMARY_METRICS),
            "m2_increment": "actual M5 > matched M4 on both primary metrics",
            "m2_assignment": "actual M5 > N/V shuffle on both primary metrics",
            "accuracy": {
                "strict_improvement": "six-metric geometric mean ratio >= 1.0",
                "economic_accuracy_tradeoff": "ratio in [0.995,1.0)",
                "reject": "ratio < 0.995",
            },
            "nonintervention": "economic-score std / ID-score std < 0.01",
            "statistical_note": (
                "one historical development seed; no significance or generalization claim"
            ),
        },
        "out_dir": cfg.out_dir,
    }


def _rank_percentile(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average", pct=True).to_numpy(np.float64)


def build_minimal_nv_inputs(
    train: pd.DataFrame,
    *,
    n_users: int,
    n_items: int,
    q_n: np.ndarray,
    q_v: np.ndarray,
    clv_valid: np.ndarray,
    item_price_percentile: np.ndarray,
    item_price_valid: np.ndarray,
) -> dict[str, np.ndarray | dict]:
    """Build the two fixed semantic axes using training rows only."""

    required = {"u_idx", "i_idx"}
    missing = required - set(train.columns)
    if missing:
        raise KeyError(f"최소 N/V 입력 컬럼 누락: {sorted(missing)}")
    q_n = np.asarray(q_n, dtype=np.float64)
    q_v = np.asarray(q_v, dtype=np.float64)
    clv_valid = np.asarray(clv_valid, dtype=bool)
    item_price = np.asarray(item_price_percentile, dtype=np.float64)
    price_valid = np.asarray(item_price_valid, dtype=bool)
    expected_user = (n_users,)
    expected_item = (n_items,)
    if any(values.shape != expected_user for values in (q_n, q_v, clv_valid)):
        raise ValueError("q_N·q_V·CLV valid shape이 n_users와 다릅니다")
    if item_price.shape != expected_item or price_valid.shape != expected_item:
        raise ValueError("상품 가격입력 shape이 n_items와 다릅니다")
    if not np.isfinite(q_n).all() or not np.isfinite(q_v).all():
        raise ValueError("q_N·q_V는 유한해야 합니다")
    if np.any((q_n < 0.0) | (q_n > 1.0)) or np.any(
        (q_v < 0.0) | (q_v > 1.0)
    ):
        raise ValueError("q_N·q_V 범위는 [0,1]이어야 합니다")
    if not np.isfinite(item_price).all() or np.any(
        (item_price < 0.0) | (item_price > 1.0)
    ):
        raise ValueError("상품 구매금액 백분위 범위는 [0,1]이어야 합니다")

    pairs = (
        train[["u_idx", "i_idx"]]
        .groupby(["u_idx", "i_idx"], sort=False)
        .size()
        .rename("row_count")
        .reset_index()
    )
    pair_items = pairs["i_idx"].to_numpy(np.int64, copy=False)
    if np.any((pair_items < 0) | (pair_items >= n_items)):
        raise ValueError("학습 상품 index가 n_items 범위를 벗어났습니다")
    buyer_count = np.bincount(pair_items, minlength=n_items).astype(np.float64)
    repeated = pairs.loc[pairs["row_count"] >= 2, "i_idx"].to_numpy(
        np.int64, copy=False
    )
    repeat_buyer_count = np.bincount(repeated, minlength=n_items).astype(
        np.float64
    )
    repeat_valid = buyer_count > 0.0
    repeat_rate = np.zeros(n_items, dtype=np.float64)
    repeat_rate[repeat_valid] = (
        repeat_buyer_count[repeat_valid] + 1.0
    ) / (buyer_count[repeat_valid] + 2.0)
    repeat_percentile = np.full(n_items, 0.5, dtype=np.float64)
    repeat_percentile[repeat_valid] = _rank_percentile(
        repeat_rate[repeat_valid]
    )
    item_valid = repeat_valid & price_valid

    user_n = 2.0 * q_n - 1.0
    user_v = 2.0 * q_v - 1.0
    user_n[~clv_valid] = 0.0
    user_v[~clv_valid] = 0.0
    item_repeat = 2.0 * repeat_percentile - 1.0
    item_price_centered = 2.0 * item_price - 1.0
    item_repeat[~item_valid] = 0.0
    item_price_centered[~item_valid] = 0.0

    diagnostics = {
        "minimal_nv_user_valid_share": float(clv_valid.mean()),
        "minimal_nv_item_valid_share": float(item_valid.mean()),
        "item_repeat_rate_mean": float(repeat_rate[repeat_valid].mean()),
        "item_repeat_rate_std": float(repeat_rate[repeat_valid].std()),
        "item_repeat_propensity_formula": (
            "(repeat purchasers + 1) / (purchasers + 2)"
        ),
        "profile_or_category_input_in_m2": False,
    }
    return {
        "user_n_centered": user_n.astype(np.float32),
        "user_v_centered": user_v.astype(np.float32),
        "clv_valid": clv_valid.copy(),
        "item_repeat_rate": repeat_rate.astype(np.float32),
        "item_repeat_centered": item_repeat.astype(np.float32),
        "item_price_centered": item_price_centered.astype(np.float32),
        "item_semantic_valid": item_valid,
        "minimal_nv_input_diagnostics": diagnostics,
    }


def representation_degree_matched_shuffle(
    prepared: dict, *, seed: int = 42, degree_bins: int = 10
) -> dict[str, np.ndarray]:
    """Jointly permute only the M2 q_N/q_V tuple inside degree bins."""

    bins = np.asarray(prepared["degree_bin"])
    if bins.ndim != 1 or bins.min(initial=0) < 0 or bins.max(initial=0) >= degree_bins:
        raise ValueError("degree_bin shape 또는 범위가 잘못됐습니다")
    rng = np.random.default_rng(seed)
    source = np.arange(len(bins), dtype=np.int64)
    for group in range(degree_bins):
        members = np.flatnonzero(bins == group)
        if len(members) > 1:
            order = rng.permutation(members)
            source[order] = np.roll(order, 1)
    return {
        "user_n_centered": np.asarray(prepared["user_n_centered"])[source].copy(),
        "user_v_centered": np.asarray(prepared["user_v_centered"])[source].copy(),
        "clv_valid": np.asarray(prepared["clv_valid"])[source].copy(),
        "source_user": source,
        "degree_bin": bins.copy(),
    }


def _config_hash(cfg: M5MinimalNVM4Config, input_hash: str, revision: str) -> str:
    payload = {
        "code_version": CODE_VERSION,
        "config": asdict(cfg),
        "input_hash": input_hash,
        "source_revision": revision,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()[:12]


def _prepare(cfg: M5MinimalNVM4Config) -> dict:
    prepared = legacy.common._prepare(legacy._common_config(cfg))
    m4_inputs = nv.build_nv_economic_inputs(
        prepared["data"]["train"],
        n_users=prepared["data"]["n_users"],
        n_items=prepared["data"]["n_items"],
        q_n=prepared["q_n"],
        q_v=prepared["q_v"],
        q_c=prepared["q_c"],
        clv_valid=prepared["clv_valid"],
        n_bins=cfg.economic_bins,
        shrinkage_strength=cfg.shrinkage_strength,
        degree_bins=cfg.shuffle_degree_bins,
    )
    prepared.update(m4_inputs)
    minimal = build_minimal_nv_inputs(
        prepared["data"]["train"],
        n_users=prepared["data"]["n_users"],
        n_items=prepared["data"]["n_items"],
        q_n=prepared["q_n"],
        q_v=prepared["q_v"],
        clv_valid=prepared["clv_valid"],
        item_price_percentile=prepared["item_amount_percentile"],
        item_price_valid=prepared["item_economic_valid"],
    )
    prepared.update(minimal)
    prepared["m2_actual"] = {
        "user_n_centered": prepared["user_n_centered"],
        "user_v_centered": prepared["user_v_centered"],
        "clv_valid": prepared["clv_valid"],
    }
    prepared["m2_shuffle"] = representation_degree_matched_shuffle(
        prepared, seed=cfg.shuffle_seed, degree_bins=cfg.shuffle_degree_bins
    )
    diagnostics = dict(prepared["economic_input_diagnostics"])
    diagnostics.update(prepared["minimal_nv_input_diagnostics"])
    prepared["economic_input_diagnostics"] = diagnostics
    prepared["config_hash"] = _config_hash(
        cfg, prepared["input_hash"], prepared["revision"]
    )
    return prepared


def arm_specifications(
    prepared: dict, cfg: M5MinimalNVM4Config
) -> list[dict]:
    return [
        {
            "model_id": M4_MODEL_ID,
            "role": "matched_m4_only_control",
            "rho": 0.0,
            "weighted": True,
            "assignment": prepared,
            "assignment_name": "observed_m4",
            "m2_assignment": prepared["m2_actual"],
            "m2_assignment_name": "nonintervention",
        },
        {
            "model_id": M5_MODEL_ID,
            "role": "actual_minimal_nv_m5",
            "rho": cfg.rho,
            "weighted": True,
            "assignment": prepared,
            "assignment_name": "observed_m4",
            "m2_assignment": prepared["m2_actual"],
            "m2_assignment_name": "observed_nv",
        },
        {
            "model_id": M5_SHUFFLE_MODEL_ID,
            "role": "m2_assignment_control",
            "rho": cfg.rho,
            "weighted": True,
            "assignment": prepared,
            "assignment_name": "observed_m4",
            "m2_assignment": prepared["m2_shuffle"],
            "m2_assignment_name": "degree_matched_nv_shuffle",
        },
    ]


def _build_model(prepared: dict, cfg: M5MinimalNVM4Config, spec: dict):
    data = prepared["data"]
    assignment = spec["m2_assignment"]
    v3.set_seed(cfg.seed)
    return M5MinimalNVEconomicLightGCN(
        n_users=data["n_users"],
        n_items=data["n_items"],
        user_n_centered=assignment["user_n_centered"],
        user_v_centered=assignment["user_v_centered"],
        user_clv_valid=assignment["clv_valid"],
        item_repeat_centered=prepared["item_repeat_centered"],
        item_price_centered=prepared["item_price_centered"],
        item_semantic_valid=prepared["item_semantic_valid"],
        adj=data["adj"],
        id_dim=cfg.id_dim,
        rho=spec["rho"],
        n_layers=cfg.n_layers,
        pref_reg=cfg.pref_reg,
        scale_delta=cfg.scale_delta,
    ).to(v3.DEVICE)


def screening_reading(metric_rows: dict[str, dict]) -> dict:
    m4 = metric_rows[M4_MODEL_ID]
    actual = metric_rows[M5_MODEL_ID]
    shuffled = metric_rows[M5_SHUFFLE_MODEL_ID]
    deltas_vs_m4 = {
        metric: float(actual[metric] - m4[metric]) for metric in PRIMARY_METRICS
    }
    deltas_vs_shuffle = {
        metric: float(actual[metric] - shuffled[metric])
        for metric in PRIMARY_METRICS
    }
    m2_increment = all(value > 0.0 for value in deltas_vs_m4.values())
    m2_assignment = all(value > 0.0 for value in deltas_vs_shuffle.values())
    ratios = [actual[metric] / m4[metric] for metric in ACCURACY_METRICS]
    accuracy_ratio = float(math.exp(np.log(ratios).mean()))
    if accuracy_ratio >= 1.0:
        accuracy_classification = "strict_improvement"
    elif accuracy_ratio >= 0.995:
        accuracy_classification = "economic_accuracy_tradeoff"
    else:
        accuracy_classification = "reject_below_0_995"
    accuracy_acceptable = accuracy_ratio >= 0.995
    return {
        "positive_screen": bool(
            m2_increment and m2_assignment and accuracy_acceptable
        ),
        "m2_increment_signal": m2_increment,
        "m2_assignment_signal": m2_assignment,
        "accuracy_acceptable": accuracy_acceptable,
        "accuracy_classification": accuracy_classification,
        "accuracy_geomean_ratio_vs_m4": accuracy_ratio,
        "primary_deltas_actual_minus_m4": deltas_vs_m4,
        "primary_deltas_actual_minus_nv_shuffle": deltas_vs_shuffle,
        "next_if_positive": (
            "freeze this structure, add the joint CLV assignment control, then "
            "run the predeclared test protocol"
        ),
        "next_if_nonpositive": (
            "stop this minimal M2 branch without tuning rho or adding proxies"
        ),
        "statistical_note": (
            "one historical development seed; no significance or generalization claim"
        ),
    }


def run_minimal_nv_m4_screen(
    cfg: M5MinimalNVM4Config | None = None,
) -> pd.DataFrame:
    cfg = validate_config(cfg or configure_minimal_nv_m4_screen())
    summary = preflight_summary(cfg)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    prepared = _prepare(cfg)
    arms: dict[str, dict] = {}
    models: dict[str, M5MinimalNVEconomicLightGCN] = {}
    for spec in arm_specifications(prepared, cfg):
        print(
            f"\n===== {spec['model_id']} | seed {cfg.seed} | "
            f"fixed {cfg.epochs} epochs ====="
        )
        with patch.object(legacy, "_build_model", _build_model):
            arm, model = legacy._run_arm(prepared, cfg, spec)
        arm["m2_assignment"] = spec["m2_assignment_name"]
        arms[spec["model_id"]] = arm
        models[spec["model_id"]] = model

    rows = []
    metric_rows = {}
    for model_id in MODEL_IDS:
        arm = arms[model_id]
        metric_rows[model_id] = arm["metrics"]
        rows.append(
            {
                "model_id": model_id,
                "role": arm["role"],
                "seed": arm["seed"],
                "split": arm["split"],
                "final_epoch": arm["final_epoch"],
                "rho": arm["rho"],
                "positive_weight_lambda": arm["positive_weight_lambda"],
                "m2_assignment": arm["m2_assignment"],
                "m4_assignment": arm["clv_assignment"],
                **arm["diagnostics"],
                **arm["training"].get("final_diagnostics", {}),
                **arm["metrics"],
            }
        )
    frame = pd.DataFrame(rows)
    comparison = report_helpers._metric_comparison(
        metric_rows, references=(M4_MODEL_ID, M5_SHUFFLE_MODEL_ID)
    )
    reading = screening_reading(metric_rows)

    users: np.ndarray | None = None
    topk: dict[str, np.ndarray] = {}
    score_rows = []
    for model_id in MODEL_IDS:
        arm_users, arm_top50 = report_helpers._masked_topk(
            models[model_id], prepared, max_k=cfg.diagnostic_max_k
        )
        if users is None:
            users = arm_users
        elif not np.array_equal(users, arm_users):
            raise RuntimeError("arm별 평가 사용자 순서가 다릅니다")
        topk[model_id] = arm_top50
        score_rows.append(
            legacy._score_diagnostics(
                models[model_id], arm_users, arm_top50, model_id=model_id
            )
        )
    score_frame = pd.DataFrame(score_rows)
    actual_score_ratio = float(
        score_frame.set_index("model_id").at[
            M5_MODEL_ID, "economic_score_std_ratio_to_id"
        ]
    )
    intervention_operational = actual_score_ratio >= 0.01
    reading["intervention_checks"] = {
        "economic_score_std_ratio_to_id": actual_score_ratio,
        "minimum_required": 0.01,
        "intervention_operational": intervention_operational,
        "actual_n_scale": float(
            frame.set_index("model_id").at[M5_MODEL_ID, "n_scale"]
        ),
        "actual_v_scale": float(
            frame.set_index("model_id").at[M5_MODEL_ID, "v_scale"]
        ),
    }
    reading["positive_screen"] = bool(
        reading["positive_screen"] and intervention_operational
    )

    assert users is not None
    overlap_frames = []
    for reference in (M4_MODEL_ID, M5_SHUFFLE_MODEL_ID):
        overlap = report_helpers.topk_overlap_summary(
            topk[reference], topk[M5_MODEL_ID], prepared["cache"].seg, k=10
        )
        overlap.insert(0, "reference", reference)
        overlap.insert(1, "model_id", M5_MODEL_ID)
        overlap_frames.append(overlap)
    overlap_frame = pd.concat(overlap_frames, ignore_index=True)

    out = Path(cfg.out_dir)
    stem = f"m5_minimal_nv_m4_{prepared['config_hash']}"
    paths = {
        "absolute_csv": out / f"{stem}.csv",
        "comparison_csv": out / f"{stem}_comparison.csv",
        "score_diagnostics_csv": out / f"{stem}_score_diagnostics.csv",
        "top10_overlap_csv": out / f"{stem}_top10_overlap.csv",
        "json": out / f"{stem}.json",
    }
    legacy.test10._atomic_csv(paths["absolute_csv"], frame)
    legacy.test10._atomic_csv(paths["comparison_csv"], comparison)
    legacy.test10._atomic_csv(paths["score_diagnostics_csv"], score_frame)
    legacy.test10._atomic_csv(paths["top10_overlap_csv"], overlap_frame)
    legacy.test10._atomic_json(
        paths["json"],
        {
            "code_version": CODE_VERSION,
            "source_revision": prepared["revision"],
            "config": asdict(cfg),
            "preflight": summary,
            "input_manifest": prepared["manifest"],
            "absolute_rows": frame.to_dict("records"),
            "comparison_rows": comparison.to_dict("records"),
            "score_diagnostic_rows": score_frame.to_dict("records"),
            "top10_overlap_rows": overlap_frame.to_dict("records"),
            "screening_reading": reading,
            "representation_shuffle": {
                "method": "joint q_N/q_V permutation within degree deciles",
                "m4_assignment_changed": False,
                "source_user": prepared["m2_shuffle"]["source_user"].tolist(),
            },
            "arms": arms,
            "result_paths": {key: str(value) for key, value in paths.items()},
        },
    )
    frame.attrs["comparison"] = comparison
    frame.attrs["score_diagnostics"] = score_frame
    frame.attrs["top10_overlap"] = overlap_frame
    frame.attrs["decision"] = reading
    frame.attrs["result_paths"] = {key: str(value) for key, value in paths.items()}
    return frame


if __name__ == "__main__":
    print(
        "Import this module from the dedicated Colab notebook and call "
        "run_minimal_nv_m4_screen()."
    )
