"""Seed-42 screen for q_N-gated three-basis q_V economic representation."""

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
from clv_m5_n_conditioned_value_basis_model import (
    M5NConditionedValueBasisLightGCN,
)
import lightgcn_clv_gradient_isolated_economic_interaction as report_helpers
import lightgcn_clv_m5_economic_positive_weight as legacy
import lightgcn_clv_m5_minimal_nv_m4_screen as minimal
import lightgcn_clv_m5_nv_economic_positive_weight as nv
import lightgcn_clv_v3 as v3


CODE_VERSION = "m5-n-conditioned-value-basis-development-screen-v1"
M1_MODEL_ID = "m1_multineg_mean_k5_value_basis_factorial"
M4_MODEL_ID = "m4_personalized_positive_weight_k5_value_basis_factorial"
SCALAR_M5_MODEL_ID = "m5_minimal_scalar_nv_personalized_positive_weight_k5"
BASIS_M5_MODEL_ID = "m5_n_conditioned_value_basis_personalized_positive_weight_k5"
BASIS_SHUFFLE_MODEL_ID = "m5_n_conditioned_value_basis_degree_nv_shuffle"
MODEL_IDS = (
    M1_MODEL_ID,
    M4_MODEL_ID,
    SCALAR_M5_MODEL_ID,
    BASIS_M5_MODEL_ID,
    BASIS_SHUFFLE_MODEL_ID,
)
ECONOMIC_METRICS = (
    "price_purchase_amount_weighted_hit@10",
    "vndcg@10",
)
TOP10_ACCURACY_METRICS = ("recall@10", "ndcg@10")
ACCURACY_METRICS = (
    "recall@10",
    "ndcg@10",
    "recall@20",
    "ndcg@20",
    "recall@50",
    "ndcg@50",
)


@dataclass(frozen=True)
class M5NConditionedValueBasisConfig:
    dataset: str = "dunnhumby"
    seed: int = 42
    time_cutoff: int = 690
    evaluation_days: int = 7
    epochs: int = 100
    id_dim: int = 64
    economic_dim: int = 3
    economic_bins: int = 4
    shrinkage_strength: float = 10.0
    rho: float = 0.05
    gate_delta: float = 0.25
    basis_bandwidth: float = 0.25
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


def configure_n_conditioned_value_basis_screen(
    **overrides,
) -> M5NConditionedValueBasisConfig:
    defaults = {
        "out_dir": (
            f"{v3.default_out_dir('dunnhumby')}"
            "_m5_n_conditioned_value_basis_development_screen_v1"
        ),
        "baseline_result_dir": (
            f"{v3.default_out_dir('dunnhumby')}"
            "_m2_repeatshare_historical_backtest_v1"
        ),
    }
    return validate_config(
        M5NConditionedValueBasisConfig(**(defaults | overrides))
    )


def validate_config(
    cfg: M5NConditionedValueBasisConfig,
) -> M5NConditionedValueBasisConfig:
    fixed = {
        "dataset": "dunnhumby",
        "seed": 42,
        "time_cutoff": 690,
        "evaluation_days": 7,
        "epochs": 100,
        "id_dim": 64,
        "economic_dim": 3,
        "economic_bins": 4,
        "shrinkage_strength": 10.0,
        "rho": 0.05,
        "gate_delta": 0.25,
        "basis_bandwidth": 0.25,
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
            raise ValueError(
                f"N-conditioned V-basis M5 screen은 {key}={expected!r}이어야 합니다"
            )
    if cfg.batch_size <= 0 or cfg.lr <= 0 or cfg.pref_reg < 0:
        raise ValueError("학습 설정이 잘못됐습니다")
    if not cfg.out_dir or not cfg.baseline_result_dir:
        raise ValueError("out_dir와 baseline_result_dir가 필요합니다")
    return cfg


def preflight_summary(cfg: M5NConditionedValueBasisConfig) -> dict:
    cfg = validate_config(cfg)
    return {
        "code_version": CODE_VERSION,
        "dataset": cfg.dataset,
        "seed": cfg.seed,
        "split": "historical_development_days_684_690",
        "trained_models": list(MODEL_IDS),
        "research_question": (
            "Does q_N-controlled use of a fixed q_V--item-price basis add "
            "new-item accuracy and economic hits beyond the same-run M1 and M4?"
        ),
        "c3_change_basis": (
            "The rejected item repeat-propensity N coordinate is removed. "
            "The previous scalar q_V signal showed small positive M4 increments "
            "but insufficient intervention; this new hypothesis expands only "
            "that fixed value-position relation without a learnable axis rotation."
        ),
        "m2": {
            "user_n": "q_N historical purchase-frequency percentile",
            "user_n_role": (
                "bounded gate in [0.75,1.25] over the user value-position block"
            ),
            "user_v": "q_V historical transaction-value percentile",
            "item_input": "train-only item amount percentile",
            "value_basis": (
                "fixed L2-normalized Gaussian RBF at low/mid/high centers "
                "[0.0,0.5,1.0]"
            ),
            "basis_bandwidth": cfg.basis_bandwidth,
            "economic_dim": cfg.economic_dim,
            "rho": cfg.rho,
            "learned_parameters": (
                "gate offset and q_N slope only; no item-N coordinate or axis rotation"
            ),
            "item_n_or_item_clv_input": False,
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
            "actual_assignment_in_all_weighted_arms": True,
        },
        "controls": {
            M1_MODEL_ID: "rho=0 and unweighted K=5 mean BPR",
            M4_MODEL_ID: "rho=0 with observed personalized M4 weighting",
            SCALAR_M5_MODEL_ID: (
                "previous one-dimensional N and V coordinates with observed M4"
            ),
            BASIS_M5_MODEL_ID: "observed q_N and q_V in the proposed M2 plus M4",
            BASIS_SHUFFLE_MODEL_ID: (
                "only q_N and q_V jointly permuted within degree deciles; "
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
            "top10_accuracy": "proposed actual Recall@10 and NDCG@10 > M1",
            "top10_economics": (
                "proposed actual weighted hit@10 and weighted NDCG@10 > M1 and M4"
            ),
            "assignment": (
                "proposed actual > q_N/q_V shuffle on all four top-10 metrics"
            ),
            "intervention": "economic-score std / ID-score std >= 0.01",
            "reported_not_gated": (
                "all @20/@50 accuracy, economic, exposure, and CLV-segment metrics"
            ),
            "statistical_note": (
                "one historical development seed; no significance or generalization claim"
            ),
        },
        "out_dir": cfg.out_dir,
    }


def representation_degree_matched_shuffle(
    prepared: dict, *, seed: int = 42, degree_bins: int = 10
) -> dict[str, np.ndarray]:
    """Jointly permute only raw q_N/q_V and validity inside degree bins."""

    bins = np.asarray(prepared["degree_bin"])
    q_n = np.asarray(prepared["q_n"])
    q_v = np.asarray(prepared["q_v"])
    valid = np.asarray(prepared["clv_valid"])
    if bins.ndim != 1 or any(len(values) != len(bins) for values in (q_n, q_v, valid)):
        raise ValueError("degree bin 또는 q_N/q_V shape이 잘못됐습니다")
    if bins.min(initial=0) < 0 or bins.max(initial=0) >= degree_bins:
        raise ValueError("degree_bin 범위가 잘못됐습니다")
    rng = np.random.default_rng(seed)
    source = np.arange(len(bins), dtype=np.int64)
    for group in range(degree_bins):
        members = np.flatnonzero(bins == group)
        if len(members) > 1:
            order = rng.permutation(members)
            source[order] = np.roll(order, 1)
    return {
        "q_n": q_n[source].copy(),
        "q_v": q_v[source].copy(),
        "clv_valid": valid[source].copy(),
        "source_user": source,
        "degree_bin": bins.copy(),
    }


def _config_hash(
    cfg: M5NConditionedValueBasisConfig, input_hash: str, revision: str
) -> str:
    payload = {
        "code_version": CODE_VERSION,
        "config": asdict(cfg),
        "input_hash": input_hash,
        "source_revision": revision,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()[:12]


def _prepare(cfg: M5NConditionedValueBasisConfig) -> dict:
    prepared = legacy.common._prepare(legacy._common_config(cfg))
    economic = nv.build_nv_economic_inputs(
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
    prepared.update(economic)
    scalar = minimal.build_minimal_nv_inputs(
        prepared["data"]["train"],
        n_users=prepared["data"]["n_users"],
        n_items=prepared["data"]["n_items"],
        q_n=prepared["q_n"],
        q_v=prepared["q_v"],
        clv_valid=prepared["clv_valid"],
        item_price_percentile=prepared["item_amount_percentile"],
        item_price_valid=prepared["item_economic_valid"],
    )
    prepared.update(scalar)
    prepared["m2_actual"] = {
        "q_n": prepared["q_n"],
        "q_v": prepared["q_v"],
        "clv_valid": prepared["clv_valid"],
        "user_n_centered": prepared["user_n_centered"],
        "user_v_centered": prepared["user_v_centered"],
    }
    prepared["m2_shuffle"] = representation_degree_matched_shuffle(
        prepared, seed=cfg.shuffle_seed, degree_bins=cfg.shuffle_degree_bins
    )
    prepared["scalar_input_diagnostics"] = prepared["minimal_nv_input_diagnostics"]
    prepared["config_hash"] = _config_hash(
        cfg, prepared["input_hash"], prepared["revision"]
    )
    return prepared


def arm_specifications(
    prepared: dict, cfg: M5NConditionedValueBasisConfig
) -> list[dict]:
    actual = prepared["m2_actual"]
    return [
        {
            "model_id": M1_MODEL_ID,
            "role": "same_run_m1",
            "architecture": "basis",
            "rho": 0.0,
            "weighted": False,
            "assignment": prepared,
            "assignment_name": "observed_m4",
            "m2_assignment": actual,
            "m2_assignment_name": "nonintervention",
        },
        {
            "model_id": M4_MODEL_ID,
            "role": "same_run_m4_only",
            "architecture": "basis",
            "rho": 0.0,
            "weighted": True,
            "assignment": prepared,
            "assignment_name": "observed_m4",
            "m2_assignment": actual,
            "m2_assignment_name": "nonintervention",
        },
        {
            "model_id": SCALAR_M5_MODEL_ID,
            "role": "previous_scalar_m2_plus_m4",
            "architecture": "scalar",
            "rho": cfg.rho,
            "weighted": True,
            "assignment": prepared,
            "assignment_name": "observed_m4",
            "m2_assignment": actual,
            "m2_assignment_name": "observed_nv",
        },
        {
            "model_id": BASIS_M5_MODEL_ID,
            "role": "actual_n_conditioned_value_basis_m5",
            "architecture": "basis",
            "rho": cfg.rho,
            "weighted": True,
            "assignment": prepared,
            "assignment_name": "observed_m4",
            "m2_assignment": actual,
            "m2_assignment_name": "observed_nv",
        },
        {
            "model_id": BASIS_SHUFFLE_MODEL_ID,
            "role": "m2_assignment_control",
            "architecture": "basis",
            "rho": cfg.rho,
            "weighted": True,
            "assignment": prepared,
            "assignment_name": "observed_m4",
            "m2_assignment": prepared["m2_shuffle"],
            "m2_assignment_name": "degree_matched_nv_shuffle",
        },
    ]


def _build_model(
    prepared: dict, cfg: M5NConditionedValueBasisConfig, spec: dict
):
    data = prepared["data"]
    assignment = spec["m2_assignment"]
    v3.set_seed(cfg.seed)
    if spec["architecture"] == "scalar":
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
            scale_delta=cfg.gate_delta,
        ).to(v3.DEVICE)
    if spec["architecture"] != "basis":
        raise ValueError(f"알 수 없는 M2 architecture: {spec['architecture']}")
    return M5NConditionedValueBasisLightGCN(
        n_users=data["n_users"],
        n_items=data["n_items"],
        user_q_n=assignment["q_n"],
        user_q_v=assignment["q_v"],
        user_clv_valid=assignment["clv_valid"],
        item_price_percentile=prepared["item_amount_percentile"],
        item_price_valid=prepared["item_economic_valid"],
        adj=data["adj"],
        id_dim=cfg.id_dim,
        rho=spec["rho"],
        n_layers=cfg.n_layers,
        pref_reg=cfg.pref_reg,
        gate_delta=cfg.gate_delta,
        basis_bandwidth=cfg.basis_bandwidth,
    ).to(v3.DEVICE)


def screening_reading(
    metric_rows: dict[str, dict], *, economic_score_ratio: float
) -> dict:
    m1 = metric_rows[M1_MODEL_ID]
    m4 = metric_rows[M4_MODEL_ID]
    actual = metric_rows[BASIS_M5_MODEL_ID]
    shuffled = metric_rows[BASIS_SHUFFLE_MODEL_ID]
    top10_accuracy_beats_m1 = all(
        actual[metric] > m1[metric] for metric in TOP10_ACCURACY_METRICS
    )
    top10_economics_beats_m1_and_m4 = all(
        actual[metric] > m1[metric] and actual[metric] > m4[metric]
        for metric in ECONOMIC_METRICS
    )
    assignment_metrics = TOP10_ACCURACY_METRICS + ECONOMIC_METRICS
    actual_beats_nv_shuffle = all(
        actual[metric] > shuffled[metric] for metric in assignment_metrics
    )
    intervention_operational = economic_score_ratio >= 0.01

    def deltas(reference: dict) -> dict[str, float]:
        return {
            metric: float(actual[metric] - reference[metric])
            for metric in assignment_metrics
        }

    def geomean_ratio(reference: dict) -> float:
        ratios = [actual[metric] / reference[metric] for metric in ACCURACY_METRICS]
        return float(math.exp(np.log(ratios).mean()))

    return {
        "positive_screen": bool(
            top10_accuracy_beats_m1
            and top10_economics_beats_m1_and_m4
            and actual_beats_nv_shuffle
            and intervention_operational
        ),
        "top10_accuracy_beats_m1": top10_accuracy_beats_m1,
        "top10_economics_beats_m1_and_m4": top10_economics_beats_m1_and_m4,
        "actual_beats_nv_shuffle": actual_beats_nv_shuffle,
        "intervention_operational": intervention_operational,
        "economic_score_std_ratio_to_id": float(economic_score_ratio),
        "minimum_intervention_ratio": 0.01,
        "accuracy_geomean_ratio_vs_m1": geomean_ratio(m1),
        "accuracy_geomean_ratio_vs_m4": geomean_ratio(m4),
        "top10_deltas_actual_minus_m1": deltas(m1),
        "top10_deltas_actual_minus_m4": deltas(m4),
        "top10_deltas_actual_minus_nv_shuffle": deltas(shuffled),
        "reported_not_gated": (
            "all @20/@50 accuracy, economic, exposure, and CLV-segment metrics"
        ),
        "next_if_positive": (
            "freeze the structure before any final test or multiseed expansion"
        ),
        "next_if_nonpositive": (
            "stop this candidate without tuning rho, bandwidth, or gate range "
            "on the same development interval"
        ),
        "statistical_note": (
            "one historical development seed; no significance or generalization claim"
        ),
    }


def run_n_conditioned_value_basis_screen(
    cfg: M5NConditionedValueBasisConfig | None = None,
) -> pd.DataFrame:
    cfg = validate_config(cfg or configure_n_conditioned_value_basis_screen())
    summary = preflight_summary(cfg)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    prepared = _prepare(cfg)
    arms: dict[str, dict] = {}
    models: dict[str, object] = {}
    for spec in arm_specifications(prepared, cfg):
        print(
            f"\n===== {spec['model_id']} | seed {cfg.seed} | "
            f"fixed {cfg.epochs} epochs ====="
        )
        with patch.object(legacy, "_build_model", _build_model):
            arm, model = legacy._run_arm(prepared, cfg, spec)
        arm["m2_assignment"] = spec["m2_assignment_name"]
        arm["m2_architecture"] = spec["architecture"]
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
                "m2_architecture": arm["m2_architecture"],
                "m2_assignment": arm["m2_assignment"],
                "m4_assignment": arm["clv_assignment"],
                **arm["diagnostics"],
                **arm["training"].get("final_diagnostics", {}),
                **arm["metrics"],
            }
        )
    frame = pd.DataFrame(rows)
    comparison = report_helpers._metric_comparison(
        metric_rows,
        references=(
            M1_MODEL_ID,
            M4_MODEL_ID,
            SCALAR_M5_MODEL_ID,
            BASIS_SHUFFLE_MODEL_ID,
        ),
    )

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
            BASIS_M5_MODEL_ID, "economic_score_std_ratio_to_id"
        ]
    )
    reading = screening_reading(
        metric_rows, economic_score_ratio=actual_score_ratio
    )

    assert users is not None
    overlap_frames = []
    for reference in (
        M1_MODEL_ID,
        M4_MODEL_ID,
        SCALAR_M5_MODEL_ID,
        BASIS_SHUFFLE_MODEL_ID,
    ):
        overlap = report_helpers.topk_overlap_summary(
            topk[reference], topk[BASIS_M5_MODEL_ID], prepared["cache"].seg, k=10
        )
        overlap.insert(0, "reference", reference)
        overlap.insert(1, "model_id", BASIS_M5_MODEL_ID)
        overlap_frames.append(overlap)
    overlap_frame = pd.concat(overlap_frames, ignore_index=True)

    out = Path(cfg.out_dir)
    stem = f"m5_n_conditioned_value_basis_{prepared['config_hash']}"
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
        "run_n_conditioned_value_basis_screen()."
    )
