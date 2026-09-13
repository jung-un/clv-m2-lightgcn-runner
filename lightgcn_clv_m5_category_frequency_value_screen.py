"""Minimal two-arm M5 screen for category-allocated historical N.

Both arms are trained from scratch in the same run.  They share the fixed q_V
value-position basis and the current personalized M4 positive-row weighting.
The only intervention difference is whether q_N is allocated over the user's
train-only basket-category composition and placed in the jointly propagated
M2 representation.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from clv_m5_category_frequency_value_model import (
    M5CategoryFrequencyValueLightGCN,
    build_category_frequency_features,
)
import lightgcn_clv_gradient_isolated_economic_interaction as report_helpers
import lightgcn_clv_m5_economic_positive_weight as legacy
import lightgcn_clv_m5_nv_economic_positive_weight as nv
import lightgcn_clv_v3 as v3


CODE_VERSION = "m5-category-frequency-n-value-basis-two-arm-development-screen-v1"
VALUE_ONLY_M5_ID = "m5_value_basis_personalized_positive_weight_category_n_off_k5"
CATEGORY_N_M5_ID = "m5_category_frequency_n_value_basis_personalized_positive_weight_k5"
MODEL_IDS = (VALUE_ONLY_M5_ID, CATEGORY_N_M5_ID)
ECONOMIC_METRICS = (
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
class M5CategoryFrequencyValueConfig:
    dataset: str = "dunnhumby"
    seed: int = 42
    time_cutoff: int = 690
    evaluation_days: int = 7
    epochs: int = 100
    id_dim: int = 64
    value_dim: int = 3
    category_dim: int = 4
    economic_bins: int = 4
    shrinkage_strength: float = 10.0
    rho_value: float = 0.05
    rho_category_n: float = 0.05
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


def configure_category_frequency_value_screen(
    **overrides,
) -> M5CategoryFrequencyValueConfig:
    data_root = v3.default_out_dir("dunnhumby")
    defaults = {
        "out_dir": (
            f"{data_root}"
            "_m5_category_frequency_n_value_basis_two_arm_development_screen_v1"
        ),
        "baseline_result_dir": (
            f"{data_root}_m2_repeatshare_historical_backtest_v1"
        ),
    }
    return validate_config(
        M5CategoryFrequencyValueConfig(**(defaults | overrides))
    )


def validate_config(
    cfg: M5CategoryFrequencyValueConfig,
) -> M5CategoryFrequencyValueConfig:
    fixed = {
        "dataset": "dunnhumby",
        "seed": 42,
        "time_cutoff": 690,
        "evaluation_days": 7,
        "epochs": 100,
        "id_dim": 64,
        "value_dim": 3,
        "category_dim": 4,
        "economic_bins": 4,
        "shrinkage_strength": 10.0,
        "rho_value": 0.05,
        "rho_category_n": 0.05,
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
                f"카테고리별 N 최소 M5 screen은 {key}={expected!r}이어야 합니다"
            )
    if cfg.batch_size <= 0 or cfg.lr <= 0 or cfg.pref_reg < 0:
        raise ValueError("학습 설정이 잘못됐습니다")
    if not cfg.out_dir or not cfg.baseline_result_dir:
        raise ValueError("out_dir와 baseline_result_dir가 필요합니다")
    return cfg


def preflight_summary(cfg: M5CategoryFrequencyValueConfig) -> dict:
    cfg = validate_config(cfg)
    return {
        "code_version": CODE_VERSION,
        "dataset": cfg.dataset,
        "seed": cfg.seed,
        "split": "historical_development_days_684_690",
        "trained_models": list(MODEL_IDS),
        "reused_models": [],
        "prior_result_file_required": False,
        "research_question": (
            "Within M5, does allocating the unchanged historical purchase-"
            "frequency component q_N over user-specific categories improve the "
            "existing q_V representation plus M4?"
        ),
        "c3_change_basis": (
            "This is not the rejected global category-repeatability or category-"
            "transition relation. It decomposes each user's own total q_N over "
            "basket categories, removes the population category share, has no "
            "free softmax, and does not encode exact-item repeat frequency."
        ),
        "arms": {
            VALUE_ONLY_M5_ID: "q_V value-position basis + unchanged M4; category N off",
            CATEGORY_N_M5_ID: (
                "category-allocated q_N + the same q_V basis + unchanged M4"
            ),
        },
        "m2": {
            "n_definition": (
                "q_N remains the train-history percentile of repeat transactions "
                "per customer age"
            ),
            "n_category_allocation": (
                "q_N times the shrunken user basket-category share minus the "
                "population basket-category share"
            ),
            "transaction_mass": (
                "one per basket, divided equally over distinct categories"
            ),
            "sparse_user_shrinkage": cfg.shrinkage_strength,
            "item_side": (
                "train-only category identity basis; no item CLV or item N"
            ),
            "exact_item_frequency": "not used; train pairs are evaluation-masked",
            "v_definition": "q_V user transaction-value percentile",
            "v_item_match": "fixed low/mid/high RBF versus item amount percentile",
            "rho_value": cfg.rho_value,
            "rho_category_n": cfg.rho_category_n,
            "category_dim": cfg.category_dim,
            "economic_graph_propagation": True,
            "joint_end_to_end_training": True,
        },
        "m4": {
            "formula": (
                "1 + 0.5*q_C*item_amount_percentile*clipped_user_bin_fit"
            ),
            "q_c": "percentile(n_u*v_u), not q_N*q_V",
            "same_assignment_in_both_arms": True,
            "uniform_negative_count": cfg.negative_count,
            "hard_negative": False,
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
            "one_training_loop_and_optimizer_per_arm": True,
            "external_reranking": False,
            "m3_edge_weight": False,
        },
        "reading_rule": {
            "directional_pass": (
                "category-N M5 > value-only M5 on both top-10 economic metrics"
            ),
            "accuracy": "all Recall/NDCG metrics are reported but are not gates",
            "attribution": (
                "not tested in this minimum two-arm run; a positive result only "
                "licenses category-only and degree-matched N-assignment controls"
            ),
            "statistical_note": (
                "one exposed historical development seed; no significance, "
                "stability, generalization, or final CLV-effect claim"
            ),
        },
        "out_dir": cfg.out_dir,
    }


def _config_hash(
    cfg: M5CategoryFrequencyValueConfig, input_hash: str, revision: str
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


def _prepare(cfg: M5CategoryFrequencyValueConfig) -> dict:
    # The shared loader normally looks up a completed M1 result.  This screen
    # trains and compares only its two arms, so no prior-result file is needed.
    with patch.object(
        legacy.common.gatefree, "_load_compatible_baseline", return_value=None
    ):
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
    prepared["category_frequency_features"] = build_category_frequency_features(
        prepared["data"]["train"],
        n_users=prepared["data"]["n_users"],
        n_items=prepared["data"]["n_items"],
        n_categories=prepared["data"]["n_cat"],
        q_n=prepared["q_n"],
        clv_valid=prepared["clv_valid"],
        shrinkage_strength=cfg.shrinkage_strength,
    )
    prepared["config_hash"] = _config_hash(
        cfg, prepared["input_hash"], prepared["revision"]
    )
    return prepared


def arm_specifications(
    prepared: dict, cfg: M5CategoryFrequencyValueConfig
) -> list[dict]:
    return [
        {
            "model_id": VALUE_ONLY_M5_ID,
            "role": "matched_value_basis_m4_category_n_off",
            "rho": cfg.rho_value,
            "rho_category_n": 0.0,
            "weighted": True,
            "assignment": prepared,
            "assignment_name": "observed_m4",
        },
        {
            "model_id": CATEGORY_N_M5_ID,
            "role": "category_frequency_n_plus_value_basis_m4",
            "rho": cfg.rho_value,
            "rho_category_n": cfg.rho_category_n,
            "weighted": True,
            "assignment": prepared,
            "assignment_name": "observed_m4",
        },
    ]


def _build_model(
    prepared: dict, cfg: M5CategoryFrequencyValueConfig, spec: dict
) -> M5CategoryFrequencyValueLightGCN:
    data = prepared["data"]
    v3.set_seed(cfg.seed)
    return M5CategoryFrequencyValueLightGCN(
        n_users=data["n_users"],
        n_items=data["n_items"],
        n_categories=data["n_cat"],
        user_q_v=prepared["q_v"],
        user_clv_valid=prepared["clv_valid"],
        item_price_percentile=prepared["item_amount_percentile"],
        item_price_valid=prepared["item_economic_valid"],
        category_features=prepared["category_frequency_features"],
        adj=data["adj"],
        id_dim=cfg.id_dim,
        category_dim=cfg.category_dim,
        rho_value=spec["rho"],
        rho_category_n=spec["rho_category_n"],
        n_layers=cfg.n_layers,
        pref_reg=cfg.pref_reg,
        basis_bandwidth=cfg.basis_bandwidth,
    ).to(v3.DEVICE)


@torch.no_grad()
def _score_diagnostics(
    model: M5CategoryFrequencyValueLightGCN,
    users: np.ndarray,
    top50: np.ndarray,
    *,
    model_id: str,
) -> dict[str, float | int | str]:
    width = top50.shape[1]
    pair_users = np.repeat(users.astype(np.int64), width)
    pair_items = top50.reshape(-1).astype(np.int64)
    names = ("id", "value", "category_n", "economic", "full")
    collected = {name: [] for name in names}
    for start in range(0, len(pair_users), 65536):
        user_tensor = torch.as_tensor(
            pair_users[start : start + 65536],
            dtype=torch.long,
            device=v3.DEVICE,
        )
        item_tensor = torch.as_tensor(
            pair_items[start : start + 65536],
            dtype=torch.long,
            device=v3.DEVICE,
        )
        components = model.candidate_score_components(user_tensor, item_tensor)
        for name in names:
            collected[name].append(components[name].cpu().numpy())
    values = {
        name: np.concatenate(parts).astype(np.float64)
        for name, parts in collected.items()
    }
    id_std = float(values["id"].std())
    category_std = float(values["category_n"].std())
    return {
        "model_id": model_id,
        "candidate_pair_count": int(len(pair_users)),
        "id_score_std": id_std,
        "value_score_std": float(values["value"].std()),
        "category_n_score_std": category_std,
        "category_n_score_std_ratio_to_id": (
            category_std / id_std if id_std > 0.0 else np.nan
        ),
        "economic_score_std": float(values["economic"].std()),
        "economic_score_std_ratio_to_id": (
            float(values["economic"].std()) / id_std
            if id_std > 0.0
            else np.nan
        ),
        "category_n_score_mean_abs": float(
            np.abs(values["category_n"]).mean()
        ),
        "max_full_decomposition_error": float(
            np.max(
                np.abs(
                    values["full"]
                    - values["id"]
                    - values["value"]
                    - values["category_n"]
                )
            )
        ),
    }


def screening_reading(metric_rows: dict[str, dict]) -> dict:
    reference = metric_rows[VALUE_ONLY_M5_ID]
    candidate = metric_rows[CATEGORY_N_M5_ID]
    economic_deltas = {
        metric: float(candidate[metric] - reference[metric])
        for metric in ECONOMIC_METRICS
    }
    accuracy_deltas = {
        metric: float(candidate[metric] - reference[metric])
        for metric in ACCURACY_METRICS
    }
    accuracy_ratios = {
        metric: float(candidate[metric] / reference[metric])
        for metric in ACCURACY_METRICS
    }
    return {
        "category_n_directional_pass": bool(
            all(delta > 0.0 for delta in economic_deltas.values())
        ),
        "top10_economic_deltas_category_n_minus_value_only": economic_deltas,
        "accuracy_deltas_reported_not_gated": accuracy_deltas,
        "accuracy_ratios_reported_not_gated": accuracy_ratios,
        "m1_or_m2_standalone_trained": False,
        "clv_assignment_tested": False,
        "final_candidate_decision_permitted": False,
        "interpretation_scope": (
            "whether category allocation of q_N deserves a controlled follow-up"
        ),
        "next_if_positive": (
            "add category-only capacity control and degree-matched q_N assignment "
            "control before any CLV attribution claim"
        ),
        "next_if_nonpositive": (
            "stop the category-specific N representation; do not tune it on the "
            "same exposed development interval"
        ),
        "statistical_note": (
            "one exposed historical development seed; no significance, stability, "
            "generalization, or final CLV-effect claim"
        ),
    }


def run_category_frequency_value_screen(
    cfg: M5CategoryFrequencyValueConfig | None = None,
) -> pd.DataFrame:
    cfg = validate_config(cfg or configure_category_frequency_value_screen())
    summary = preflight_summary(cfg)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    prepared = _prepare(cfg)
    print("\n[카테고리별 N 입력 진단]")
    print(
        json.dumps(
            prepared["category_frequency_features"].diagnostics,
            ensure_ascii=False,
            indent=2,
        )
    )

    arms: dict[str, dict] = {}
    models: dict[str, M5CategoryFrequencyValueLightGCN] = {}
    for spec in arm_specifications(prepared, cfg):
        print(
            f"\n===== {spec['model_id']} | seed {cfg.seed} | "
            f"fixed {cfg.epochs} epochs ====="
        )
        with patch.object(legacy, "_build_model", _build_model):
            arm, model = legacy._run_arm(prepared, cfg, spec)
        arm["rho_category_n"] = spec["rho_category_n"]
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
                "rho_value": arm["rho"],
                "rho_category_n": arm["rho_category_n"],
                "positive_weight_lambda": arm["positive_weight_lambda"],
                "m4_assignment": arm["clv_assignment"],
                **arm["diagnostics"],
                **arm["training"].get("final_diagnostics", {}),
                **arm["metrics"],
            }
        )
    frame = pd.DataFrame(rows)
    comparison = report_helpers._metric_comparison(
        metric_rows, references=(VALUE_ONLY_M5_ID,)
    )
    comparison = comparison.loc[
        comparison["model_id"] == CATEGORY_N_M5_ID
    ].reset_index(drop=True)

    topk: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    score_rows = []
    for model_id in MODEL_IDS:
        users, top50 = report_helpers._masked_topk(
            models[model_id], prepared, max_k=cfg.diagnostic_max_k
        )
        topk[model_id] = (users, top50)
        score_rows.append(
            _score_diagnostics(
                models[model_id], users, top50, model_id=model_id
            )
        )
    score_frame = pd.DataFrame(score_rows)

    reference_users, reference_top50 = topk[VALUE_ONLY_M5_ID]
    candidate_users, candidate_top50 = topk[CATEGORY_N_M5_ID]
    if not np.array_equal(reference_users, candidate_users):
        raise RuntimeError("두 arm의 평가 사용자 순서가 다릅니다")
    reference_top10 = reference_top50[:, :10]
    candidate_top10 = candidate_top50[:, :10]
    identical = np.all(reference_top10 == candidate_top10, axis=1)
    overlap = np.array(
        [
            len(set(left).intersection(right)) / 10.0
            for left, right in zip(reference_top10, candidate_top10)
        ],
        dtype=np.float64,
    )
    ranking_change = {
        "evaluation_user_count": int(len(reference_users)),
        "identical_ordered_top10_user_share": float(identical.mean()),
        "changed_ordered_top10_user_share": float(1.0 - identical.mean()),
        "mean_top10_set_overlap": float(overlap.mean()),
    }
    reading = screening_reading(metric_rows)
    category_score_ratio = float(
        score_frame.set_index("model_id").at[
            CATEGORY_N_M5_ID, "category_n_score_std_ratio_to_id"
        ]
    )
    reading["category_n_score_std_ratio_to_id"] = category_score_ratio
    reading["category_n_intervention_nonzero"] = bool(category_score_ratio > 0.0)
    reading["ranking_change"] = ranking_change

    out = Path(cfg.out_dir)
    stem = f"m5_category_frequency_n_value_{prepared['config_hash']}"
    paths = {
        "absolute_csv": out / f"{stem}.csv",
        "comparison_csv": out / f"{stem}_comparison.csv",
        "score_diagnostics_csv": out / f"{stem}_score_diagnostics.csv",
        "json": out / f"{stem}.json",
    }
    legacy.test10._atomic_csv(paths["absolute_csv"], frame)
    legacy.test10._atomic_csv(paths["comparison_csv"], comparison)
    legacy.test10._atomic_csv(paths["score_diagnostics_csv"], score_frame)
    legacy.test10._atomic_json(
        paths["json"],
        {
            "code_version": CODE_VERSION,
            "source_revision": prepared["revision"],
            "config": asdict(cfg),
            "preflight": summary,
            "input_manifest": prepared["manifest"],
            "category_frequency_diagnostics": (
                prepared["category_frequency_features"].diagnostics
            ),
            "absolute_rows": frame.to_dict("records"),
            "comparison_rows": comparison.to_dict("records"),
            "score_diagnostic_rows": score_frame.to_dict("records"),
            "ranking_change": ranking_change,
            "screening_reading": reading,
            "arms": arms,
            "result_paths": {key: str(value) for key, value in paths.items()},
        },
    )
    frame.attrs["comparison"] = comparison
    frame.attrs["score_diagnostics"] = score_frame
    frame.attrs["category_frequency_diagnostics"] = (
        prepared["category_frequency_features"].diagnostics
    )
    frame.attrs["ranking_change"] = ranking_change
    frame.attrs["decision"] = reading
    frame.attrs["preflight"] = summary
    frame.attrs["result_paths"] = {
        key: str(value) for key, value in paths.items()
    }

    print("\n1) q_V+M4 기준과 카테고리별 N+q_V+M4 절대지표")
    print(frame.to_string(index=False))
    print("\n2) 카테고리별 N 증분 전체 지표 비교")
    print(comparison.to_string(index=False))
    print("\n3) 점수 성분 진단")
    print(score_frame.to_string(index=False))
    print("\n4) Top-10 변경 진단")
    print(json.dumps(ranking_change, ensure_ascii=False, indent=2))
    print("\n5) 사전 판독")
    print(json.dumps(reading, ensure_ascii=False, indent=2))
    print("\n6) 저장 파일")
    print(json.dumps(frame.attrs["result_paths"], ensure_ascii=False, indent=2))
    return frame


if __name__ == "__main__":
    print(
        "Import this module from the dedicated Colab notebook and call "
        "run_category_frequency_value_screen()."
    )
