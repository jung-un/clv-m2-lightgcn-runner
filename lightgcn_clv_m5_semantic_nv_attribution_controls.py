"""Attribution controls for the completed semantic N/V M5 pilot."""

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
import lightgcn_clv_m5_nv_economic_positive_weight as nv
import lightgcn_clv_m5_semantic_nv_m4_control as m4_control
import lightgcn_clv_m5_semantic_nv_single_screen as semantic
import lightgcn_clv_v3 as v3


CODE_VERSION = "m5-semantic-nv-attribution-controls-screen-v1"
M4_MODEL_ID = m4_control.M4_MODEL_ID
M5_MODEL_ID = semantic.M5_MODEL_ID
JOINT_SHUFFLE_MODEL_ID = "m5_semantic_nv_degree_matched_joint_shuffle"
DEGREE_CONTROL_MODEL_ID = "m5_semantic_nv_degree_percentile_loss_gate"
PRIMARY_METRICS = m4_control.PRIMARY_METRICS
ACCURACY_METRICS = m4_control.ACCURACY_METRICS


@dataclass(frozen=True)
class M5SemanticNVAttributionConfig:
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
    m4_control_result_json: str = ""


def configure_semantic_nv_attribution_controls(
    **overrides,
) -> M5SemanticNVAttributionConfig:
    root = v3.default_out_dir("dunnhumby")
    defaults = {
        "out_dir": f"{root}_m5_semantic_nv_attribution_controls_screen_v1",
        "baseline_result_dir": f"{root}_m2_repeatshare_historical_backtest_v1",
        "actual_m5_result_json": (
            f"{root}_m5_semantic_nv_personalized_positive_single_screen_v1/"
            "m5_semantic_nv_single_426685be4486.json"
        ),
        "m4_control_result_json": (
            f"{root}_m5_semantic_nv_m4_matched_control_screen_v1/"
            "m5_semantic_nv_m4_control_cea1a4ef5860.json"
        ),
    }
    return validate_config(
        M5SemanticNVAttributionConfig(**(defaults | overrides))
    )


def validate_config(
    cfg: M5SemanticNVAttributionConfig,
) -> M5SemanticNVAttributionConfig:
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
            raise ValueError(f"귀속 대조실험은 {key}={expected!r}이어야 합니다")
    if cfg.batch_size <= 0 or cfg.lr <= 0 or cfg.pref_reg < 0:
        raise ValueError("학습 설정이 잘못됐습니다")
    required_paths = (
        cfg.out_dir,
        cfg.baseline_result_dir,
        cfg.actual_m5_result_json,
        cfg.m4_control_result_json,
    )
    if not all(required_paths):
        raise ValueError("결과·기준·캐시 JSON 경로가 필요합니다")
    return cfg


def preflight_summary(cfg: M5SemanticNVAttributionConfig) -> dict:
    cfg = validate_config(cfg)
    return {
        "code_version": CODE_VERSION,
        "dataset": cfg.dataset,
        "seed": cfg.seed,
        "split": "historical_development_days_684_690",
        "trained_models": [JOINT_SHUFFLE_MODEL_ID, DEGREE_CONTROL_MODEL_ID],
        "reused_models": [M4_MODEL_ID, M5_MODEL_ID],
        "research_question": (
            "Does the observed customer-to-CLV/N/V assignment explain the "
            "incremental Top-10 economic result beyond structure and degree?"
        ),
        "controls": {
            "joint_degree_matched_shuffle": {
                "description": (
                    "jointly permute q_N, q_V, q_C, profile, validity, and "
                    "positive-row fit within user-degree deciles"
                ),
                "same_tuple_in_m2_and_m4": True,
            },
            "degree_loss_gate": {
                "description": (
                    "keep observed semantic N/V M2; replace only M4 q_C with "
                    "the train user-degree percentile"
                ),
                "m2_representation_changed": False,
            },
        },
        "fixed": {
            "new_item_task": True,
            "min_item_interactions": 1,
            "binary_graph": True,
            "uniform_negative_count": cfg.negative_count,
            "epochs": cfg.epochs,
            "rho": cfg.rho,
            "beta": cfg.price_axis_budget,
            "positive_weight_lambda": cfg.positive_weight_lambda,
            "validation_or_epoch_selection": False,
            "final_test_constructed": False,
            "holdout_constructed": False,
            "one_training_loop_and_optimizer_per_arm": True,
            "external_reranking": False,
        },
        "reading_rule": {
            "primary_metrics": list(PRIMARY_METRICS),
            "joint_assignment_signal": (
                "observed M5 minus joint shuffle > 0 on both primary metrics"
            ),
            "degree_control_signal": (
                "observed M5 minus degree loss-gate control > 0 on both "
                "primary metrics"
            ),
            "positive_attribution_screen": (
                "both attribution signals are true; accuracy is reported but "
                "not used as a post-hoc gate"
            ),
            "statistical_note": (
                "one historical development seed; no significance or "
                "generalization claim"
            ),
        },
        "actual_m5_result_json": cfg.actual_m5_result_json,
        "m4_control_result_json": cfg.m4_control_result_json,
        "out_dir": cfg.out_dir,
    }


def _config_hash(
    cfg: M5SemanticNVAttributionConfig, input_hash: str, revision: str
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


def attach_control_assignments(
    prepared: dict, cfg: M5SemanticNVAttributionConfig
) -> dict:
    prepared["joint_shuffle"] = nv.joint_degree_matched_shuffle(
        prepared,
        seed=cfg.shuffle_seed,
        degree_bins=cfg.shuffle_degree_bins,
    )
    prepared["degree_control"] = {
        "q_n": np.asarray(prepared["q_n"]).copy(),
        "q_v": np.asarray(prepared["q_v"]).copy(),
        "q_c": np.asarray(prepared["degree_percentile"]).copy(),
        "clv_valid": np.asarray(prepared["clv_valid"]).copy(),
        "user_activity_gate": np.asarray(prepared["user_activity_gate"]).copy(),
        "user_economic_input": np.asarray(
            prepared["user_economic_input"]
        ).copy(),
        "user_economic_valid": np.asarray(
            prepared["user_economic_valid"]
        ).copy(),
        "user_bin_fit": np.asarray(prepared["user_bin_fit"]).copy(),
    }
    return prepared


def _prepare(cfg: M5SemanticNVAttributionConfig) -> dict:
    prepared = semantic._prepare(cfg)
    attach_control_assignments(prepared, cfg)
    prepared["config_hash"] = _config_hash(
        cfg, prepared["input_hash"], prepared["revision"]
    )
    return prepared


def arm_specifications(
    prepared: dict, cfg: M5SemanticNVAttributionConfig
) -> list[dict]:
    return [
        {
            "model_id": JOINT_SHUFFLE_MODEL_ID,
            "role": "joint_assignment_control",
            "rho": cfg.rho,
            "weighted": True,
            "assignment": prepared["joint_shuffle"],
            "assignment_name": "degree_matched_joint_semantic_nv_shuffle",
        },
        {
            "model_id": DEGREE_CONTROL_MODEL_ID,
            "role": "degree_loss_gate_control",
            "rho": cfg.rho,
            "weighted": True,
            "assignment": prepared["degree_control"],
            "assignment_name": "degree_percentile_loss_gate",
        },
    ]


def _build_model(prepared: dict, cfg, spec: dict):
    return semantic._build_model(prepared, cfg, spec)


def _matching_config(payload: dict, cfg) -> None:
    cached = payload.get("config", {})
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
        if cached.get(key) != getattr(cfg, key):
            raise RuntimeError(f"캐시 결과와 귀속 대조실험의 {key}가 다릅니다")


def load_completed_results(cfg, prepared: dict) -> tuple[dict, dict]:
    actual = m4_control.load_actual_m5(cfg, prepared)
    path = Path(cfg.m4_control_result_json)
    if not path.exists():
        raise FileNotFoundError(f"완료된 M4-only 대조실험 JSON이 없습니다: {path}")
    control = json.loads(path.read_text(encoding="utf-8"))
    if control.get("code_version") != m4_control.CODE_VERSION:
        raise RuntimeError("M4-only 대조실험 code_version이 다릅니다")
    _matching_config(control, cfg)
    manifest = control.get("input_manifest")
    if not manifest or legacy.moe.manifest_hash(manifest) != prepared["input_hash"]:
        raise RuntimeError("M4-only 대조실험과 현재 원천 입력이 다릅니다")
    if not control.get("incremental_reading", {}).get("m2_increment_signal"):
        raise RuntimeError("M4-only 대비 M2 추가기여 screen이 양수가 아닙니다")
    m4_arm = control.get("m4_arm", {})
    if m4_arm.get("model_id") != M4_MODEL_ID or not m4_arm.get("metrics"):
        raise RuntimeError("M4-only 캐시 arm이 없거나 잘못됐습니다")
    cached_actual_rows = [
        row
        for row in control.get("absolute_rows", [])
        if row.get("model_id") == M5_MODEL_ID
    ]
    if len(cached_actual_rows) != 1:
        raise RuntimeError("M4-only 결과의 actual M5 행이 잘못됐습니다")
    for metric in PRIMARY_METRICS + ACCURACY_METRICS:
        if not math.isclose(
            float(cached_actual_rows[0][metric]),
            float(actual["absolute_rows"][0][metric]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError("actual M5 캐시와 M4-only 비교행이 다릅니다")
    return actual, control


def attribution_reading(metric_rows: dict[str, dict]) -> dict:
    actual = metric_rows[M5_MODEL_ID]
    m4 = metric_rows[M4_MODEL_ID]
    shuffled = metric_rows[JOINT_SHUFFLE_MODEL_ID]
    degree = metric_rows[DEGREE_CONTROL_MODEL_ID]

    primary_deltas = {
        "vs_m4_only": {
            metric: float(actual[metric] - m4[metric])
            for metric in PRIMARY_METRICS
        },
        "vs_joint_shuffle": {
            metric: float(actual[metric] - shuffled[metric])
            for metric in PRIMARY_METRICS
        },
        "vs_degree_control": {
            metric: float(actual[metric] - degree[metric])
            for metric in PRIMARY_METRICS
        },
    }
    joint_signal = all(
        value > 0.0 for value in primary_deltas["vs_joint_shuffle"].values()
    )
    degree_signal = all(
        value > 0.0 for value in primary_deltas["vs_degree_control"].values()
    )
    accuracy_deltas = {
        reference: {
            metric: float(actual[metric] - values[metric])
            for metric in ACCURACY_METRICS
        }
        for reference, values in (
            ("vs_m4_only", m4),
            ("vs_joint_shuffle", shuffled),
            ("vs_degree_control", degree),
        )
    }
    accuracy_geomean_ratios = {
        reference: float(
            math.exp(
                np.log(
                    [actual[metric] / values[metric] for metric in ACCURACY_METRICS]
                ).mean()
            )
        )
        for reference, values in (
            ("vs_m4_only", m4),
            ("vs_joint_shuffle", shuffled),
            ("vs_degree_control", degree),
        )
    }
    return {
        "positive_attribution_screen": bool(joint_signal and degree_signal),
        "m2_increment_signal_vs_m4_only": all(
            value > 0.0 for value in primary_deltas["vs_m4_only"].values()
        ),
        "joint_assignment_signal": joint_signal,
        "degree_control_signal": degree_signal,
        "primary_metrics": list(PRIMARY_METRICS),
        "primary_deltas_actual_m5_minus_reference": primary_deltas,
        "accuracy_deltas_actual_m5_minus_reference": accuracy_deltas,
        "accuracy_geomean_ratios": accuracy_geomean_ratios,
        "accuracy_guard_required": False,
        "decision_scope": "single-seed development attribution screen",
        "next_step": (
            "freeze this architecture and repeat development seeds"
            if joint_signal and degree_signal
            else "report the failed attribution control without tuning this split"
        ),
        "statistical_note": (
            "one historical development seed; no significance or "
            "generalization claim"
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


def run_semantic_nv_attribution_controls(
    cfg: M5SemanticNVAttributionConfig | None = None,
) -> pd.DataFrame:
    cfg = validate_config(cfg or configure_semantic_nv_attribution_controls())
    summary = preflight_summary(cfg)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    prepared = _prepare(cfg)
    actual_payload, m4_payload = load_completed_results(cfg, prepared)

    trained: list[dict] = []
    with patch.object(legacy, "_build_model", _build_model):
        for spec in arm_specifications(prepared, cfg):
            print(
                f"\n===== {spec['model_id']} | seed {cfg.seed} | "
                f"fixed {cfg.epochs} epochs | attribution control ====="
            )
            arm, _ = legacy._run_arm(prepared, cfg, spec)
            trained.append(arm)

    m4_arm = m4_payload["m4_arm"]
    actual_arm = actual_payload["arm"]
    rows = [
        _row_from_arm(m4_arm),
        actual_payload["absolute_rows"][0],
        *[_row_from_arm(arm) for arm in trained],
    ]
    frame = pd.DataFrame(rows)
    metric_rows = {
        M4_MODEL_ID: m4_arm["metrics"],
        M5_MODEL_ID: actual_arm["metrics"],
        **{arm["model_id"]: arm["metrics"] for arm in trained},
    }
    comparison = report_helpers._metric_comparison(
        metric_rows,
        references=(M4_MODEL_ID, JOINT_SHUFFLE_MODEL_ID, DEGREE_CONTROL_MODEL_ID),
    )
    focused = comparison[comparison["model_id"] == M5_MODEL_ID].reset_index(
        drop=True
    )
    reading = attribution_reading(metric_rows)

    out = Path(cfg.out_dir)
    stem = f"m5_semantic_nv_attribution_{prepared['config_hash']}"
    paths = {
        "absolute_csv": out / f"{stem}.csv",
        "comparison_csv": out / f"{stem}_comparison.csv",
        "focused_comparison_csv": out / f"{stem}_focused_comparison.csv",
        "json": out / f"{stem}.json",
    }
    legacy.test10._atomic_csv(paths["absolute_csv"], frame)
    legacy.test10._atomic_csv(paths["comparison_csv"], comparison)
    legacy.test10._atomic_csv(paths["focused_comparison_csv"], focused)
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
            "focused_comparison_rows": focused.to_dict("records"),
            "attribution_reading": reading,
            "trained_control_arms": trained,
            "reused_sources": {
                "actual_m5": cfg.actual_m5_result_json,
                "m4_control": cfg.m4_control_result_json,
            },
            "result_paths": {key: str(value) for key, value in paths.items()},
        },
    )
    frame.attrs["comparison"] = comparison
    frame.attrs["focused_comparison"] = focused
    frame.attrs["attribution_reading"] = reading
    frame.attrs["preflight"] = summary
    frame.attrs["result_paths"] = {
        key: str(value) for key, value in paths.items()
    }

    print("\n1) M4-only·actual M5·귀속 대조군 절대지표")
    print(frame.to_string(index=False))
    print("\n2) 각 대조군 대비 actual M5 지표")
    print(focused.to_string(index=False))
    print("\n3) CLV/N/V 귀속 판독")
    print(json.dumps(reading, ensure_ascii=False, indent=2))
    print("\n4) 저장 파일")
    print(json.dumps(frame.attrs["result_paths"], ensure_ascii=False, indent=2))
    return frame


if __name__ == "__main__":
    print(
        json.dumps(
            preflight_summary(configure_semantic_nv_attribution_controls()),
            ensure_ascii=False,
            indent=2,
        )
    )
