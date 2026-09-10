"""Matched M4-only control for the completed semantic N/V M5 pilot."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import lightgcn_clv_gradient_isolated_economic_interaction as report_helpers
import lightgcn_clv_m5_economic_positive_weight as legacy
import lightgcn_clv_m5_semantic_nv_single_screen as semantic
import lightgcn_clv_v3 as v3


CODE_VERSION = "m5-semantic-nv-m4-matched-control-screen-v1"
M4_MODEL_ID = "m4_personalized_positive_weight_k5_semantic_control"
M5_MODEL_ID = semantic.M5_MODEL_ID
PRIMARY_METRICS = (
    "vndcg@10",
    "price_purchase_amount_weighted_hit@10",
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
class M5SemanticNVM4ControlConfig:
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
    actual_m5_result_json: str = ""


def configure_semantic_nv_m4_control(
    **overrides,
) -> M5SemanticNVM4ControlConfig:
    actual_dir = (
        f"{v3.default_out_dir('dunnhumby')}"
        "_m5_semantic_nv_personalized_positive_single_screen_v1"
    )
    defaults = {
        "out_dir": (
            f"{v3.default_out_dir('dunnhumby')}"
            "_m5_semantic_nv_m4_matched_control_screen_v1"
        ),
        "baseline_result_dir": (
            f"{v3.default_out_dir('dunnhumby')}"
            "_m2_repeatshare_historical_backtest_v1"
        ),
        "actual_m5_result_json": (
            f"{actual_dir}/m5_semantic_nv_single_426685be4486.json"
        ),
    }
    return validate_config(M5SemanticNVM4ControlConfig(**(defaults | overrides)))


def validate_config(
    cfg: M5SemanticNVM4ControlConfig,
) -> M5SemanticNVM4ControlConfig:
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
            raise ValueError(f"M4 matched control은 {key}={expected!r}이어야 합니다")
    if cfg.batch_size <= 0 or cfg.lr <= 0 or cfg.pref_reg < 0:
        raise ValueError("학습 설정이 잘못됐습니다")
    if not cfg.out_dir or not cfg.baseline_result_dir or not cfg.actual_m5_result_json:
        raise ValueError("out_dir, baseline_result_dir, actual_m5_result_json이 필요합니다")
    return cfg


def preflight_summary(cfg: M5SemanticNVM4ControlConfig) -> dict:
    cfg = validate_config(cfg)
    return {
        "code_version": CODE_VERSION,
        "dataset": cfg.dataset,
        "seed": cfg.seed,
        "split": "historical_development_days_684_690",
        "trained_models": [M4_MODEL_ID],
        "reused_models": [M5_MODEL_ID],
        "research_question": (
            "Does the fixed-semantic N/V M2 add economic ranking value "
            "beyond the identical personalized CLV-positive M4 loss?"
        ),
        "matched_control": {
            "score": "64D ID-only two-layer binary LightGCN",
            "loss": (
                "1 + lambda*q_C*item_amount_percentile*clipped_user_bin_fit"
            ),
            "rho": 0.0,
            "uniform_negative_count": cfg.negative_count,
            "same_seed_and_negative_stream_as_actual_m5": True,
        },
        "fixed": {
            "new_item_task": True,
            "min_item_interactions": 1,
            "epochs": cfg.epochs,
            "validation_or_epoch_selection": False,
            "final_test_constructed": False,
            "holdout_constructed": False,
            "one_training_loop_and_optimizer": True,
            "external_reranking": False,
        },
        "reading_rule": {
            "primary_metrics": list(PRIMARY_METRICS),
            "m2_increment_signal": "actual M5 minus matched M4-only > 0 on both",
            "accuracy": "report all six metrics; no post-hoc hard pass line",
            "next_if_positive": (
                "run joint CLV shuffle and degree control before multiple seeds"
            ),
            "next_if_nonpositive": (
                "stop this semantic M2 branch; do not tune rho or beta"
            ),
            "statistical_note": (
                "one historical development seed; no significance or generalization claim"
            ),
        },
        "actual_m5_result_json": cfg.actual_m5_result_json,
        "out_dir": cfg.out_dir,
    }


def _config_hash(
    cfg: M5SemanticNVM4ControlConfig, input_hash: str, revision: str
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


def _prepare(cfg: M5SemanticNVM4ControlConfig) -> dict:
    prepared = semantic._prepare(cfg)
    prepared["config_hash"] = _config_hash(
        cfg, prepared["input_hash"], prepared["revision"]
    )
    return prepared


def arm_specification(prepared: dict, cfg: M5SemanticNVM4ControlConfig) -> dict:
    return {
        "model_id": M4_MODEL_ID,
        "role": "matched_m4_only_control",
        "rho": 0.0,
        "weighted": True,
        "assignment": prepared,
        "assignment_name": "observed",
    }


def _build_model(prepared: dict, cfg: M5SemanticNVM4ControlConfig, spec: dict):
    return semantic._build_model(prepared, cfg, spec)


def _validate_actual_payload(payload: dict, cfg, prepared: dict) -> dict:
    if payload.get("code_version") != semantic.CODE_VERSION:
        raise RuntimeError("actual M5 결과의 code_version이 다릅니다")
    actual_cfg = payload.get("config", {})
    for key in (
        "dataset",
        "seed",
        "time_cutoff",
        "evaluation_days",
        "epochs",
        "id_dim",
        "economic_dim",
        "economic_bins",
        "shrinkage_strength",
        "rho",
        "price_axis_budget",
        "scale_delta",
        "positive_weight_lambda",
        "n_layers",
        "negative_count",
        "batch_size",
        "lr",
        "pref_reg",
        "input_days",
    ):
        if actual_cfg.get(key) != getattr(cfg, key):
            raise RuntimeError(f"actual M5와 M4 control의 {key} 설정이 다릅니다")
    manifest = payload.get("input_manifest")
    if not manifest or legacy.moe.manifest_hash(manifest) != prepared["input_hash"]:
        raise RuntimeError("actual M5와 M4 control의 원천 입력이 다릅니다")
    arm = payload.get("arm", {})
    if (
        arm.get("model_id") != M5_MODEL_ID
        or arm.get("seed") != cfg.seed
        or arm.get("split") != "historical_development_days_684_690"
        or arm.get("final_epoch") != cfg.epochs
        or arm.get("clv_assignment") != "observed"
    ):
        raise RuntimeError("actual M5 결과행의 모형·seed·split 계약이 다릅니다")
    if not arm.get("metrics"):
        raise RuntimeError("actual M5 결과에 평가 지표가 없습니다")
    rows = payload.get("absolute_rows", [])
    if len(rows) != 1 or rows[0].get("model_id") != M5_MODEL_ID:
        raise RuntimeError("actual M5 절대지표 행이 정확히 하나가 아닙니다")
    return payload


def load_actual_m5(cfg, prepared: dict) -> dict:
    path = Path(cfg.actual_m5_result_json)
    if not path.exists():
        raise FileNotFoundError(
            "완료된 actual M5 JSON이 없습니다. 단일 actual M5 노트북을 먼저 "
            f"완료했는지 확인하세요: {path}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    return _validate_actual_payload(payload, cfg, prepared)


def incremental_reading(m4_metrics: dict, m5_metrics: dict) -> dict:
    primary_deltas = {
        metric: float(m5_metrics[metric] - m4_metrics[metric])
        for metric in PRIMARY_METRICS
    }
    accuracy_deltas = {
        metric: float(m5_metrics[metric] - m4_metrics[metric])
        for metric in ACCURACY_METRICS
    }
    ratios = [m5_metrics[metric] / m4_metrics[metric] for metric in ACCURACY_METRICS]
    accuracy_geomean_ratio = float(math.exp(np.log(ratios).mean()))
    positive = all(delta > 0.0 for delta in primary_deltas.values())
    return {
        "m2_increment_signal": positive,
        "primary_deltas_m5_minus_m4_only": primary_deltas,
        "accuracy_deltas_m5_minus_m4_only": accuracy_deltas,
        "accuracy_geomean_ratio_m5_vs_m4_only": accuracy_geomean_ratio,
        "decision_scope": (
            "single-seed development screen of incremental M2 contribution only"
        ),
        "clv_attribution_tested": False,
        "next_step": (
            "run joint CLV shuffle and degree control"
            if positive
            else "stop this semantic M2 branch without rho/beta tuning"
        ),
        "statistical_note": (
            "one historical development seed; no significance or generalization claim"
        ),
    }


def _row_from_arm(arm: dict) -> dict:
    return {
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


def run_semantic_nv_m4_control(
    cfg: M5SemanticNVM4ControlConfig | None = None,
) -> pd.DataFrame:
    cfg = validate_config(cfg or configure_semantic_nv_m4_control())
    summary = preflight_summary(cfg)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    prepared = _prepare(cfg)
    actual = load_actual_m5(cfg, prepared)
    spec = arm_specification(prepared, cfg)
    print(
        f"\n===== {spec['model_id']} | seed {cfg.seed} | "
        f"fixed {cfg.epochs} epochs | matched M4-only ====="
    )
    with patch.object(legacy, "_build_model", _build_model):
        m4_arm, _ = legacy._run_arm(prepared, cfg, spec)

    m5_arm = actual["arm"]
    rows = [_row_from_arm(m4_arm), actual["absolute_rows"][0]]
    frame = pd.DataFrame(rows)
    metric_rows = {
        M4_MODEL_ID: m4_arm["metrics"],
        M5_MODEL_ID: m5_arm["metrics"],
    }
    comparison = report_helpers._metric_comparison(
        metric_rows, references=(M4_MODEL_ID,)
    )
    reading = incremental_reading(m4_arm["metrics"], m5_arm["metrics"])

    out = Path(cfg.out_dir)
    stem = f"m5_semantic_nv_m4_control_{prepared['config_hash']}"
    paths = {
        "absolute_csv": out / f"{stem}.csv",
        "comparison_csv": out / f"{stem}_comparison.csv",
        "json": out / f"{stem}.json",
    }
    legacy.test10._atomic_csv(paths["absolute_csv"], frame)
    legacy.test10._atomic_csv(paths["comparison_csv"], comparison)
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
            "comparison_rows": comparison.to_dict("records"),
            "incremental_reading": reading,
            "m4_arm": m4_arm,
            "actual_m5_source": {
                "path": cfg.actual_m5_result_json,
                "code_version": actual["code_version"],
                "source_revision": actual.get("source_revision"),
            },
            "result_paths": {key: str(value) for key, value in paths.items()},
        },
    )
    frame.attrs["comparison"] = comparison
    frame.attrs["incremental_reading"] = reading
    frame.attrs["preflight"] = summary
    frame.attrs["result_paths"] = {key: str(value) for key, value in paths.items()}

    print("\n1) matched M4-only와 cached actual M5 절대지표")
    print(frame.to_string(index=False))
    print("\n2) M4-only 대비 M5 전체 지표 비교")
    print(comparison.to_string(index=False))
    print("\n3) M2 증분 판독")
    print(json.dumps(reading, ensure_ascii=False, indent=2))
    print("\n4) 저장 파일")
    print(json.dumps(frame.attrs["result_paths"], ensure_ascii=False, indent=2))
    return frame


if __name__ == "__main__":
    print(
        json.dumps(
            preflight_summary(configure_semantic_nv_m4_control()),
            ensure_ascii=False,
            indent=2,
        )
    )
