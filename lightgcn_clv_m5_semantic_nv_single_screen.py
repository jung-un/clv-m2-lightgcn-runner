"""One-arm seed-42 development screen for the semantic N/V M5."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from clv_m5_semantic_nv_model import M5SemanticNVEconomicLightGCN
import lightgcn_clv_gradient_isolated_economic_interaction as report_helpers
import lightgcn_clv_m5_economic_positive_weight as legacy
import lightgcn_clv_m5_nv_economic_positive_weight as nv
import lightgcn_clv_v3 as v3


CODE_VERSION = "m5-semantic-nv-personalized-positive-single-screen-v1"
M5_MODEL_ID = "m5_semantic_nv_personalized_positive_weight_k5"


@dataclass(frozen=True)
class M5SemanticNVSingleConfig:
    dataset: str = "dunnhumby"
    seed: int = 42
    time_cutoff: int = 690
    evaluation_days: int = 7
    epochs: int = 100
    id_dim: int = 64
    economic_dim: int = 5
    economic_bins: int = 4
    shrinkage_strength: float = 10.0
    rho: float = 0.15
    price_axis_budget: float = 0.25
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


def configure_semantic_nv_single_screen(**overrides) -> M5SemanticNVSingleConfig:
    defaults = {
        "out_dir": (
            f"{v3.default_out_dir('dunnhumby')}"
            "_m5_semantic_nv_personalized_positive_single_screen_v1"
        ),
        "baseline_result_dir": (
            f"{v3.default_out_dir('dunnhumby')}"
            "_m2_repeatshare_historical_backtest_v1"
        ),
    }
    return validate_config(M5SemanticNVSingleConfig(**(defaults | overrides)))


def validate_config(cfg: M5SemanticNVSingleConfig) -> M5SemanticNVSingleConfig:
    fixed = {
        "dataset": "dunnhumby",
        "seed": 42,
        "time_cutoff": 690,
        "evaluation_days": 7,
        "epochs": 100,
        "id_dim": 64,
        "economic_dim": 5,
        "economic_bins": 4,
        "shrinkage_strength": 10.0,
        "rho": 0.15,
        "price_axis_budget": 0.25,
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
            raise ValueError(f"단일 M5 screen은 {key}={expected!r}이어야 합니다")
    if cfg.batch_size <= 0 or cfg.lr <= 0 or cfg.pref_reg < 0:
        raise ValueError("학습 설정이 잘못됐습니다")
    if not cfg.out_dir or not cfg.baseline_result_dir:
        raise ValueError("out_dir와 baseline_result_dir가 필요합니다")
    return cfg


def preflight_summary(cfg: M5SemanticNVSingleConfig) -> dict:
    cfg = validate_config(cfg)
    return {
        "code_version": CODE_VERSION,
        "dataset": cfg.dataset,
        "seed": cfg.seed,
        "split": "historical_development_days_684_690",
        "trained_models": [M5_MODEL_ID],
        "controls_trained": False,
        "research_axis": "M5 partial combination: semantic N/V M2 plus current M4 loss",
        "m2": {
            "id": "64D LightGCN, binary graph, two layers",
            "economic_dim": cfg.economic_dim,
            "value_price_axis": "signed q_V times signed overall item-price percentile",
            "activity_profile_axes": (
                "q_N times shrunken centered four-bin spending profile "
                "matched to the centered item price-bin basis"
            ),
            "price_axis_budget": cfg.price_axis_budget,
            "rho": cfg.rho,
            "learned_parameters": (
                "two positive bounded calibration scalars in [0.75,1.25]"
            ),
            "semantic_axis_rotation": False,
            "economic_graph_propagation": False,
            "joint_end_to_end_training": True,
            "q_c_used_in_m2": False,
        },
        "m4": {
            "loss": "personalized CLV-conditioned positive-row weighted BPR",
            "formula": (
                "1 + lambda*q_C*item_amount_percentile*clipped_user_bin_fit"
            ),
            "lambda": cfg.positive_weight_lambda,
            "uniform_negative_count": cfg.negative_count,
            "hard_negative": False,
        },
        "fixed": {
            "new_item_task": True,
            "min_item_interactions": 1,
            "graph": "binary",
            "negative_sampling": "uniform",
            "epochs": cfg.epochs,
            "validation_or_epoch_selection": False,
            "final_test_constructed": False,
            "holdout_constructed": False,
            "one_training_loop_and_optimizer": True,
            "external_reranking": False,
        },
        "reading_rule": {
            "purpose": "runtime and directional pilot only",
            "success_or_failure_decision": False,
            "reason": "no same-run baseline, shuffle, or degree control is trained",
            "statistical_note": "one development seed; no significance or generalization claim",
        },
        "out_dir": cfg.out_dir,
    }


def _config_hash(cfg: M5SemanticNVSingleConfig, input_hash: str, revision: str) -> str:
    payload = {
        "code_version": CODE_VERSION,
        "config": asdict(cfg),
        "input_hash": input_hash,
        "source_revision": revision,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()[:12]


def _prepare(cfg: M5SemanticNVSingleConfig) -> dict:
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
    prepared["config_hash"] = _config_hash(
        cfg, prepared["input_hash"], prepared["revision"]
    )
    return prepared


def arm_specifications(
    prepared: dict, cfg: M5SemanticNVSingleConfig
) -> list[dict]:
    return [
        {
            "model_id": M5_MODEL_ID,
            "role": "single_actual_m5_pilot",
            "rho": cfg.rho,
            "weighted": True,
            "assignment": prepared,
            "assignment_name": "observed",
        }
    ]


def _build_model(prepared: dict, cfg: M5SemanticNVSingleConfig, spec: dict):
    data = prepared["data"]
    assignment = spec["assignment"]
    v3.set_seed(cfg.seed)
    return M5SemanticNVEconomicLightGCN(
        n_users=data["n_users"],
        n_items=data["n_items"],
        user_q_n=assignment["q_n"],
        user_q_v_centered=assignment["user_economic_input"][:, 0],
        user_centered_profile=assignment["user_economic_input"][:, 1:],
        user_economic_valid=assignment["user_economic_valid"],
        item_price_centered=prepared["item_economic_input"][:, 0],
        item_centered_bin=prepared["item_economic_input"][:, 1:],
        item_economic_valid=prepared["item_economic_valid"],
        adj=data["adj"],
        id_dim=cfg.id_dim,
        rho=spec["rho"],
        beta=cfg.price_axis_budget,
        n_layers=cfg.n_layers,
        pref_reg=cfg.pref_reg,
        scale_delta=cfg.scale_delta,
    ).to(v3.DEVICE)


def run_semantic_nv_single_screen(
    cfg: M5SemanticNVSingleConfig | None = None,
) -> pd.DataFrame:
    cfg = validate_config(cfg or configure_semantic_nv_single_screen())
    summary = preflight_summary(cfg)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    prepared = _prepare(cfg)
    spec = arm_specifications(prepared, cfg)[0]
    print(
        f"\n===== {spec['model_id']} | seed {cfg.seed} | "
        f"fixed {cfg.epochs} epochs | single actual arm ====="
    )
    with patch.object(legacy, "_build_model", _build_model):
        arm, model = legacy._run_arm(prepared, cfg, spec)

    row = {
        "model_id": arm["model_id"],
        "role": arm["role"],
        "seed": arm["seed"],
        "split": arm["split"],
        "final_epoch": arm["final_epoch"],
        "rho": arm["rho"],
        "positive_weight_lambda": arm["positive_weight_lambda"],
        "clv_assignment": arm["clv_assignment"],
        **arm["diagnostics"],
        **arm["training"].get("final_diagnostics", {}),
        **arm["metrics"],
    }
    frame = pd.DataFrame([row])
    users, top50 = report_helpers._masked_topk(
        model, prepared, max_k=cfg.diagnostic_max_k
    )
    score = pd.DataFrame(
        [legacy._score_diagnostics(model, users, top50, model_id=M5_MODEL_ID)]
    )

    out = Path(cfg.out_dir)
    stem = f"m5_semantic_nv_single_{prepared['config_hash']}"
    paths = {
        "absolute_csv": out / f"{stem}.csv",
        "score_diagnostics_csv": out / f"{stem}_score_diagnostics.csv",
        "json": out / f"{stem}.json",
    }
    legacy.test10._atomic_csv(paths["absolute_csv"], frame)
    legacy.test10._atomic_csv(paths["score_diagnostics_csv"], score)
    legacy.test10._atomic_json(
        paths["json"],
        {
            "code_version": CODE_VERSION,
            "source_revision": prepared["revision"],
            "config": asdict(cfg),
            "preflight": summary,
            "input_manifest": prepared["manifest"],
            "data_stats": prepared["data"].get("data_stats", {}),
            "absolute_rows": frame.to_dict("records"),
            "score_diagnostic_rows": score.to_dict("records"),
            "arm": arm,
            "result_paths": {key: str(value) for key, value in paths.items()},
        },
    )
    frame.attrs["score_diagnostics"] = score
    frame.attrs["preflight"] = summary
    frame.attrs["result_paths"] = {key: str(value) for key, value in paths.items()}
    print("\n1) 단일 actual M5 절대지표")
    print(frame.to_string(index=False))
    print("\n2) 실제 점수 영향력")
    print(score.to_string(index=False))
    print("\n대조군을 학습하지 않았으므로 이 실행만으로 성공·실패를 판정하지 않습니다.")
    print("결과 파일:", frame.attrs["result_paths"])
    return frame


if __name__ == "__main__":
    print(
        json.dumps(
            preflight_summary(configure_semantic_nv_single_screen()),
            ensure_ascii=False,
            indent=2,
        )
    )
