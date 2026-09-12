"""Two-arm screen for q_N-gated three-basis q_V representation.

The unchanged M1 and personalized-positive M4 controls are loaded from the two
completed runs whose matching conditions are recorded in RESEARCH_STATUS.md.
Only the changed M2 and M5 arms are trained here.
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

from clv_m5_n_conditioned_value_basis_model import (
    M5NConditionedValueBasisLightGCN,
)
import lightgcn_clv_gradient_isolated_economic_interaction as report_helpers
import lightgcn_clv_m5_economic_positive_weight as legacy
import lightgcn_clv_m5_nv_economic_positive_weight as nv
import lightgcn_clv_v3 as v3


CODE_VERSION = "m5-n-conditioned-value-basis-reuse-controls-development-screen-v2"
M1_MODEL_ID = "m1_multineg_mean_k5"
M4_MODEL_ID = "m4_personalized_positive_weight_k5_minimal_nv_control"
BASIS_M2_MODEL_ID = "m2_n_conditioned_value_basis_multineg_mean_k5"
BASIS_M5_MODEL_ID = "m5_n_conditioned_value_basis_personalized_positive_weight_k5"
REFERENCE_MODEL_IDS = (
    M1_MODEL_ID,
    M4_MODEL_ID,
)
TRAINED_MODEL_IDS = (
    BASIS_M2_MODEL_ID,
    BASIS_M5_MODEL_ID,
)
MODEL_IDS = REFERENCE_MODEL_IDS + TRAINED_MODEL_IDS
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
    m1_reference_json: str = ""
    m4_reference_json: str = ""


def configure_n_conditioned_value_basis_screen(
    **overrides,
) -> M5NConditionedValueBasisConfig:
    data_root = v3.default_out_dir("dunnhumby")
    defaults = {
        "out_dir": (
            f"{data_root}"
            "_m5_n_conditioned_value_basis_reuse_controls_development_screen_v2"
        ),
        "baseline_result_dir": (
            f"{data_root}_m2_repeatshare_historical_backtest_v1"
        ),
        "m1_reference_json": (
            f"{data_root}_m5_m2_m4_joint_historical_screen_v1/"
            "m5_m2_m4_joint_7cd818302fb1.json"
        ),
        "m4_reference_json": (
            f"{data_root}_m5_minimal_nv_personalized_positive_development_screen_v1/"
            "m5_minimal_nv_m4_d01883236772.json"
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
    if not all(
        (
            cfg.out_dir,
            cfg.baseline_result_dir,
            cfg.m1_reference_json,
            cfg.m4_reference_json,
        )
    ):
        raise ValueError(
            "out_dir, baseline_result_dir, M1·M4 reference JSON이 필요합니다"
        )
    return cfg


def preflight_summary(cfg: M5NConditionedValueBasisConfig) -> dict:
    cfg = validate_config(cfg)
    return {
        "code_version": CODE_VERSION,
        "dataset": cfg.dataset,
        "seed": cfg.seed,
        "split": "historical_development_days_684_690",
        "trained_models": list(TRAINED_MODEL_IDS),
        "reused_models": list(REFERENCE_MODEL_IDS),
        "research_question": (
            "Does q_N-controlled use of a fixed q_V--item-price basis add "
            "new-item accuracy and economic hits beyond compatible completed "
            "M1 and M4 references?"
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
            "actual_assignment_in_new_m5": True,
        },
        "execution": {
            "trained_now": {
                BASIS_M2_MODEL_ID: "new M2 with K=5 mean BPR",
                BASIS_M5_MODEL_ID: "the same new M2 plus observed M4",
            },
            "reused_completed_results": {
                M1_MODEL_ID: cfg.m1_reference_json,
                M4_MODEL_ID: cfg.m4_reference_json,
            },
            "condition_match_source": "RESEARCH_STATUS.md manual verification",
            "different_run_disclosure": True,
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
            "main_directional_check": (
                "new M5 > reused compatible M4 on both top-10 economic metrics"
            ),
            "top10_accuracy": "reported for M2-vs-M1 and M5-vs-M4; not a gate",
            "m2_standalone": "reported against reused M1; not required for M5 pass",
            "intervention": "economic-score std / ID-score std >= 0.01",
            "clv_attribution": (
                "not tested because no shuffle arm is trained in this fast run"
            ),
            "reported_not_gated": (
                "all @20/@50 accuracy, economic, exposure, and CLV-segment metrics"
            ),
            "statistical_note": (
                "one historical development seed; no significance or generalization claim"
            ),
        },
        "reference_sources": {
            M1_MODEL_ID: cfg.m1_reference_json,
            M4_MODEL_ID: cfg.m4_reference_json,
        },
        "out_dir": cfg.out_dir,
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
    prepared["m2_actual"] = {
        "q_n": prepared["q_n"],
        "q_v": prepared["q_v"],
        "clv_valid": prepared["clv_valid"],
    }
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
            "model_id": BASIS_M2_MODEL_ID,
            "role": "new_n_conditioned_value_basis_m2",
            "architecture": "basis",
            "rho": cfg.rho,
            "weighted": False,
            "assignment": prepared,
            "assignment_name": "unweighted",
            "m2_assignment": actual,
            "m2_assignment_name": "observed_nv",
        },
        {
            "model_id": BASIS_M5_MODEL_ID,
            "role": "new_n_conditioned_value_basis_m5",
            "architecture": "basis",
            "rho": cfg.rho,
            "weighted": True,
            "assignment": prepared,
            "assignment_name": "observed_m4",
            "m2_assignment": actual,
            "m2_assignment_name": "observed_nv",
        },
    ]


def _build_model(
    prepared: dict, cfg: M5NConditionedValueBasisConfig, spec: dict
):
    data = prepared["data"]
    assignment = spec["m2_assignment"]
    v3.set_seed(cfg.seed)
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


def _load_reused_reference(
    path: str,
    *,
    expected_model_id: str,
) -> tuple[dict, dict]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(
            f"재사용할 {expected_model_id} 결과 JSON이 없습니다: {source}"
        )
    payload = json.loads(source.read_text(encoding="utf-8"))
    rows = [
        row
        for row in payload.get("absolute_rows", [])
        if row.get("model_id") == expected_model_id
    ]
    if len(rows) != 1:
        raise RuntimeError(
            f"{expected_model_id} 절대지표 행이 정확히 하나가 아닙니다"
        )
    row = dict(rows[0])
    arms = payload.get("arms", {})
    arm = arms.get(expected_model_id)
    if not arm or not arm.get("metrics"):
        raise RuntimeError(f"{expected_model_id} arm 또는 전체 지표가 없습니다")

    row.update(
        {
            "execution_source": "reused_prior_run",
            "source_result_path": str(source),
            "source_code_version": payload["code_version"],
            "source_revision": payload.get("source_revision"),
            "reference_contract": "manually_verified_in_RESEARCH_STATUS.md",
        }
    )
    provenance = {
        "reused_without_retraining": True,
        "model_id": expected_model_id,
        "path": str(source),
        "code_version": payload["code_version"],
        "source_revision": payload.get("source_revision"),
        "condition_match_source": "RESEARCH_STATUS.md",
    }
    return {"row": row, "metrics": arm["metrics"], "arm": arm}, provenance


def load_reused_references(
    cfg: M5NConditionedValueBasisConfig,
) -> tuple[dict[str, dict], dict[str, dict]]:
    specifications = (
        (M1_MODEL_ID, cfg.m1_reference_json),
        (M4_MODEL_ID, cfg.m4_reference_json),
    )
    references = {}
    provenance = {}
    for model_id, path in specifications:
        reference, source = _load_reused_reference(
            path,
            expected_model_id=model_id,
        )
        references[model_id] = reference
        provenance[model_id] = source
    return references, provenance


def screening_reading(
    metric_rows: dict[str, dict], *, economic_score_ratios: dict[str, float]
) -> dict:
    m1 = metric_rows[M1_MODEL_ID]
    m4 = metric_rows[M4_MODEL_ID]
    m2 = metric_rows[BASIS_M2_MODEL_ID]
    m5 = metric_rows[BASIS_M5_MODEL_ID]
    m2_top10_accuracy_beats_reused_m1 = all(
        m2[metric] > m1[metric] for metric in TOP10_ACCURACY_METRICS
    )
    m5_top10_accuracy_beats_reused_m4 = all(
        m5[metric] > m4[metric] for metric in TOP10_ACCURACY_METRICS
    )
    m2_top10_economics_beats_reused_m1 = all(
        m2[metric] > m1[metric] for metric in ECONOMIC_METRICS
    )
    m5_top10_economics_beats_reused_m4 = all(
        m5[metric] > m4[metric] for metric in ECONOMIC_METRICS
    )
    m5_score_ratio = float(economic_score_ratios[BASIS_M5_MODEL_ID])
    intervention_operational = m5_score_ratio >= 0.01
    reported_metrics = TOP10_ACCURACY_METRICS + ECONOMIC_METRICS

    def deltas(model: dict, reference: dict) -> dict[str, float]:
        return {
            metric: float(model[metric] - reference[metric])
            for metric in reported_metrics
        }

    def geomean_ratio(model: dict, reference: dict) -> float:
        ratios = [model[metric] / reference[metric] for metric in ACCURACY_METRICS]
        return float(math.exp(np.log(ratios).mean()))

    return {
        "directional_screen_pass": bool(
            m5_top10_economics_beats_reused_m4 and intervention_operational
        ),
        "m2_top10_accuracy_beats_reused_m1": (
            m2_top10_accuracy_beats_reused_m1
        ),
        "m2_top10_economics_beats_reused_m1": (
            m2_top10_economics_beats_reused_m1
        ),
        "m5_top10_accuracy_beats_reused_m4": (
            m5_top10_accuracy_beats_reused_m4
        ),
        "m5_top10_economics_beats_reused_m4": (
            m5_top10_economics_beats_reused_m4
        ),
        "intervention_operational": intervention_operational,
        "economic_score_std_ratio_to_id": {
            key: float(value) for key, value in economic_score_ratios.items()
        },
        "minimum_intervention_ratio": 0.01,
        "accuracy_geomean_ratio_m2_vs_reused_m1": geomean_ratio(m2, m1),
        "accuracy_geomean_ratio_m5_vs_reused_m4": geomean_ratio(m5, m4),
        "top10_deltas_m2_minus_reused_m1": deltas(m2, m1),
        "top10_deltas_m5_minus_reused_m4": deltas(m5, m4),
        "m2_standalone_success_required": False,
        "clv_assignment_tested": False,
        "positive_screen_or_attribution_decision_permitted": False,
        "comparison_scope": (
            "directional development comparison using contract-verified controls "
            "from separate completed runs"
        ),
        "reported_not_gated": (
            "all @20/@50 accuracy, economic, exposure, and CLV-segment metrics"
        ),
        "next_if_directional_pass": (
            "train the q_N/q_V assignment control before any attribution claim"
        ),
        "next_if_directional_nonpass": (
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
    references, provenance = load_reused_references(cfg)
    print("\n[재사용] RESEARCH_STATUS.md에서 조건을 확인한 M1·M4입니다.")
    for model_id in REFERENCE_MODEL_IDS:
        print(f"  - {model_id}: {provenance[model_id]['path']}")
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

    rows = [references[model_id]["row"] for model_id in REFERENCE_MODEL_IDS]
    metric_rows = {
        model_id: references[model_id]["metrics"]
        for model_id in REFERENCE_MODEL_IDS
    }
    for model_id in TRAINED_MODEL_IDS:
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
                "execution_source": "current_two_arm_run",
                "source_result_path": None,
                "source_code_version": CODE_VERSION,
                "source_revision": prepared["revision"],
                "reference_contract": None,
                **arm["diagnostics"],
                **arm["training"].get("final_diagnostics", {}),
                **arm["metrics"],
            }
        )
    frame = pd.DataFrame(rows)
    full_comparison = report_helpers._metric_comparison(
        metric_rows,
        references=REFERENCE_MODEL_IDS,
    )
    comparison = full_comparison.loc[
        (
            (full_comparison["reference"] == M1_MODEL_ID)
            & (full_comparison["model_id"] == BASIS_M2_MODEL_ID)
        )
        | (
            (full_comparison["reference"] == M4_MODEL_ID)
            & (full_comparison["model_id"] == BASIS_M5_MODEL_ID)
        )
    ].reset_index(drop=True)

    score_rows = []
    for model_id in TRAINED_MODEL_IDS:
        arm_users, arm_top50 = report_helpers._masked_topk(
            models[model_id], prepared, max_k=cfg.diagnostic_max_k
        )
        score_rows.append(
            legacy._score_diagnostics(
                models[model_id], arm_users, arm_top50, model_id=model_id
            )
        )
    score_frame = pd.DataFrame(score_rows)
    score_ratios = {
        model_id: float(
            score_frame.set_index("model_id").at[
                model_id, "economic_score_std_ratio_to_id"
            ]
        )
        for model_id in TRAINED_MODEL_IDS
    }
    reading = screening_reading(
        metric_rows, economic_score_ratios=score_ratios
    )

    out = Path(cfg.out_dir)
    stem = f"m5_n_conditioned_value_basis_{prepared['config_hash']}"
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
            "absolute_rows": frame.to_dict("records"),
            "comparison_rows": comparison.to_dict("records"),
            "score_diagnostic_rows": score_frame.to_dict("records"),
            "screening_reading": reading,
            "reused_references": {
                model_id: {
                    "provenance": provenance[model_id],
                    "arm": references[model_id]["arm"],
                }
                for model_id in REFERENCE_MODEL_IDS
            },
            "arms": arms,
            "result_paths": {key: str(value) for key, value in paths.items()},
        },
    )
    frame.attrs["comparison"] = comparison
    frame.attrs["score_diagnostics"] = score_frame
    frame.attrs["decision"] = reading
    frame.attrs["reference_provenance"] = provenance
    frame.attrs["preflight"] = summary
    frame.attrs["result_paths"] = {key: str(value) for key, value in paths.items()}
    print("\n1) 재사용 M1·M4와 새로 학습한 M2·M5 절대지표")
    print(frame.to_string(index=False))
    print("\n2) M2-M1, M5-M4 전체 지표 비교")
    print(comparison.to_string(index=False))
    print("\n3) 새 M2·M5 ID 점수 대비 경제점수 영향력")
    print(score_frame.to_string(index=False))
    print("\n4) 방향성 판독")
    print(json.dumps(reading, ensure_ascii=False, indent=2))
    print("\n5) 재사용 결과 출처")
    print(json.dumps(provenance, ensure_ascii=False, indent=2))
    print("\n6) 저장 파일")
    print(json.dumps(frame.attrs["result_paths"], ensure_ascii=False, indent=2))
    return frame


if __name__ == "__main__":
    print(
        "Import this module from the dedicated Colab notebook and call "
        "run_n_conditioned_value_basis_screen()."
    )
