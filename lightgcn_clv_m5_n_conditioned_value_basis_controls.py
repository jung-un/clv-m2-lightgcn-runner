"""Two-arm mechanism controls for the N-conditioned value-basis M5.

The completed observed M5 and its matched M4 reference are reused.  This
runner trains only (1) a degree-matched joint q_N/q_V assignment control and
(2) a q_V-only control with the same maximum constant gate as the observed
model.  M4 always keeps its observed q_C and user-bin fit.
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
import lightgcn_clv_m5_n_conditioned_value_basis_screen as base
import lightgcn_clv_v3 as v3


CODE_VERSION = "m5-n-conditioned-value-basis-controls-development-screen-v1"
EXPECTED_ACTUAL_CODE_VERSION = base.CODE_VERSION
EXPECTED_ACTUAL_SOURCE_REVISION = "5eb1bcbd8e6ab75d2e77a45b1cf6881c010f207d"
M4_MODEL_ID = base.M4_MODEL_ID
ACTUAL_M5_MODEL_ID = base.BASIS_M5_MODEL_ID
SHUFFLED_M5_MODEL_ID = "m5_n_conditioned_value_basis_degree_matched_nv_shuffle"
V_ONLY_M5_MODEL_ID = "m5_value_basis_constant_gate_personalized_positive_weight_k5"
REUSED_MODEL_IDS = (M4_MODEL_ID, ACTUAL_M5_MODEL_ID)
TRAINED_MODEL_IDS = (SHUFFLED_M5_MODEL_ID, V_ONLY_M5_MODEL_ID)
MODEL_IDS = REUSED_MODEL_IDS + TRAINED_MODEL_IDS
ECONOMIC_METRICS = base.ECONOMIC_METRICS
ACCURACY_METRICS = base.ACCURACY_METRICS


@dataclass(frozen=True)
class M5NConditionedValueBasisControlsConfig(base.M5NConditionedValueBasisConfig):
    actual_result_json: str = ""
    constant_gate: float = 1.25


def configure_value_basis_controls(
    **overrides,
) -> M5NConditionedValueBasisControlsConfig:
    data_root = v3.default_out_dir("dunnhumby")
    base_cfg = base.configure_n_conditioned_value_basis_screen()
    defaults = asdict(base_cfg) | {
        "out_dir": (
            f"{data_root}"
            "_m5_n_conditioned_value_basis_controls_development_screen_v1"
        ),
        "actual_result_json": (
            f"{data_root}"
            "_m5_n_conditioned_value_basis_reuse_controls_development_screen_v2/"
            "m5_n_conditioned_value_basis_bea6e1d8bf5b.json"
        ),
        "constant_gate": 1.25,
    }
    return validate_config(
        M5NConditionedValueBasisControlsConfig(**(defaults | overrides))
    )


def validate_config(
    cfg: M5NConditionedValueBasisControlsConfig,
) -> M5NConditionedValueBasisControlsConfig:
    base.validate_config(cfg)
    if not cfg.actual_result_json:
        raise ValueError("완료 actual M5 결과 JSON이 필요합니다")
    if cfg.constant_gate != 1.25:
        raise ValueError("V-only 대조군은 사전 고정 constant_gate=1.25여야 합니다")
    return cfg


def preflight_summary(cfg: M5NConditionedValueBasisControlsConfig) -> dict:
    cfg = validate_config(cfg)
    return {
        "code_version": CODE_VERSION,
        "dataset": cfg.dataset,
        "seed": cfg.seed,
        "split": "historical_development_days_684_690",
        "trained_models": list(TRAINED_MODEL_IDS),
        "reused_models": list(REUSED_MODEL_IDS),
        "research_question": (
            "Did the completed M5 benefit from correctly assigned q_N/q_V, "
            "and did q_N add anything beyond the observed q_V basis at the same "
            "effective gate ceiling?"
        ),
        "controls": {
            SHUFFLED_M5_MODEL_ID: (
                "jointly permute q_N, q_V, and their validity inside binary user-"
                "degree deciles; keep observed M4 q_C and user-bin fit"
            ),
            V_ONLY_M5_MODEL_ID: (
                "keep observed q_V and replace the learned q_N gate with the "
                "fixed design ceiling 1.25; keep observed M4"
            ),
        },
        "fixed": {
            "new_item_task": True,
            "train_pairs_excluded_from_evaluation": True,
            "min_item_interactions": 1,
            "graph": "binary",
            "negative_sampling": "uniform",
            "negative_count": cfg.negative_count,
            "epochs": cfg.epochs,
            "rho": cfg.rho,
            "basis_bandwidth": cfg.basis_bandwidth,
            "positive_weight_lambda": cfg.positive_weight_lambda,
            "m4_assignment": "observed q_C and observed user-bin fit in every arm",
            "validation_or_epoch_selection": False,
            "final_test_constructed": False,
            "holdout_constructed": False,
            "one_training_loop_and_optimizer_per_arm": True,
            "external_reranking": False,
            "m3_edge_weight": False,
        },
        "reading_rule": {
            "assignment_signal": (
                "actual M5 > degree-matched q_N/q_V shuffle on both top-10 "
                "economic metrics"
            ),
            "n_increment_signal": (
                "actual M5 > q_V-only constant-gate M5 on both top-10 "
                "economic metrics"
            ),
            "mechanism_screen_pass": (
                "assignment_signal and n_increment_signal; accuracy, wider cutoffs, "
                "exposure, and segments are reported but not gates"
            ),
            "statistical_note": (
                "one exposed historical development seed; no significance, "
                "stability, generalization, or final CLV-effect claim"
            ),
        },
        "actual_result_json": cfg.actual_result_json,
        "out_dir": cfg.out_dir,
    }


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _config_hash(
    cfg: M5NConditionedValueBasisControlsConfig,
    input_hash: str,
    revision: str,
) -> str:
    payload = {
        "code_version": CODE_VERSION,
        "config": asdict(cfg),
        "input_hash": input_hash,
        "source_revision": revision,
    }
    return hashlib.sha256(_canonical(payload).encode()).hexdigest()[:12]


def degree_matched_nv_shuffle(
    prepared: dict,
    *,
    seed: int = 42,
    degree_bins: int = 10,
) -> dict[str, np.ndarray]:
    """Jointly reassign q_N/q_V/valid inside binary-degree strata."""

    bins = np.asarray(prepared["degree_bin"], dtype=np.int64)
    q_n = np.asarray(prepared["q_n"], dtype=np.float32)
    q_v = np.asarray(prepared["q_v"], dtype=np.float32)
    valid = np.asarray(prepared["clv_valid"], dtype=bool)
    if any(values.shape != bins.shape for values in (q_n, q_v, valid)):
        raise ValueError("degree·q_N·q_V·valid shape이 일치해야 합니다")
    if bins.ndim != 1 or np.any((bins < 0) | (bins >= degree_bins)):
        raise ValueError("degree_bin shape 또는 범위가 잘못됐습니다")

    rng = np.random.default_rng(seed)
    source = np.arange(len(bins), dtype=np.int64)
    for group in range(degree_bins):
        members = np.flatnonzero(bins == group)
        if len(members) > 1:
            order = rng.permutation(members)
            source[order] = np.roll(order, 1)
    if np.any(bins[source] != bins):
        raise RuntimeError("N/V 순열이 degree 층 밖으로 이동했습니다")
    return {
        "q_n": q_n[source].copy(),
        "q_v": q_v[source].copy(),
        "clv_valid": valid[source].copy(),
        "source_user": source,
        "degree_bin": bins.copy(),
    }


def _prepare(cfg: M5NConditionedValueBasisControlsConfig) -> dict:
    prepared = base._prepare(cfg)
    prepared["m2_shuffle"] = degree_matched_nv_shuffle(
        prepared,
        seed=cfg.shuffle_seed,
        degree_bins=cfg.shuffle_degree_bins,
    )
    prepared["m2_v_only"] = {
        "q_n": np.asarray(prepared["q_n"]).copy(),
        "q_v": np.asarray(prepared["q_v"]).copy(),
        "clv_valid": np.asarray(prepared["clv_valid"]).copy(),
    }
    prepared["config_hash"] = _config_hash(
        cfg, prepared["input_hash"], prepared["revision"]
    )
    source = prepared["m2_shuffle"]["source_user"]
    prepared["control_diagnostics"] = {
        "n_users": int(len(source)),
        "valid_users": int(np.asarray(prepared["clv_valid"]).sum()),
        "shuffle_moved_user_share": float(np.mean(source != np.arange(len(source)))),
        "shuffle_same_degree_bin": bool(
            np.all(prepared["degree_bin"][source] == prepared["degree_bin"])
        ),
        "shuffle_q_n_multiset_preserved": bool(
            np.array_equal(
                np.sort(prepared["m2_shuffle"]["q_n"]),
                np.sort(np.asarray(prepared["q_n"])),
            )
        ),
        "shuffle_q_v_multiset_preserved": bool(
            np.array_equal(
                np.sort(prepared["m2_shuffle"]["q_v"]),
                np.sort(np.asarray(prepared["q_v"])),
            )
        ),
        "shuffle_valid_multiset_preserved": bool(
            np.array_equal(
                np.sort(prepared["m2_shuffle"]["clv_valid"]),
                np.sort(np.asarray(prepared["clv_valid"])),
            )
        ),
        "v_only_constant_gate": cfg.constant_gate,
    }
    if not all(
        prepared["control_diagnostics"][key]
        for key in (
            "shuffle_same_degree_bin",
            "shuffle_q_n_multiset_preserved",
            "shuffle_q_v_multiset_preserved",
            "shuffle_valid_multiset_preserved",
        )
    ):
        raise RuntimeError("N/V 순열 불변식이 성립하지 않습니다")
    return prepared


def arm_specifications(
    prepared: dict,
    cfg: M5NConditionedValueBasisControlsConfig,
) -> list[dict]:
    return [
        {
            "model_id": SHUFFLED_M5_MODEL_ID,
            "role": "degree_matched_nv_assignment_control",
            "architecture": "q_n_conditioned_basis",
            "rho": cfg.rho,
            "weighted": True,
            "assignment": prepared,
            "assignment_name": "observed_m4",
            "m2_assignment": prepared["m2_shuffle"],
            "m2_assignment_name": "degree_matched_joint_nv_shuffle",
            "constant_gate": None,
        },
        {
            "model_id": V_ONLY_M5_MODEL_ID,
            "role": "q_v_only_constant_gate_control",
            "architecture": "q_v_only_fixed_basis",
            "rho": cfg.rho,
            "weighted": True,
            "assignment": prepared,
            "assignment_name": "observed_m4",
            "m2_assignment": prepared["m2_v_only"],
            "m2_assignment_name": "observed_q_v_without_q_n_conditioning",
            "constant_gate": cfg.constant_gate,
        },
    ]


def _build_model(
    prepared: dict,
    cfg: M5NConditionedValueBasisControlsConfig,
    spec: dict,
):
    data = prepared["data"]
    assignment = spec["m2_assignment"]
    v3.set_seed(cfg.seed)
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
        constant_gate=spec["constant_gate"],
    ).to(v3.DEVICE)


def _find_one(rows: list[dict], model_id: str, label: str) -> dict:
    selected = [row for row in rows if row.get("model_id") == model_id]
    if len(selected) != 1:
        raise RuntimeError(f"{label}의 {model_id} 행이 정확히 하나가 아닙니다")
    return dict(selected[0])


def load_completed_references(
    cfg: M5NConditionedValueBasisControlsConfig,
    prepared: dict,
) -> tuple[dict[str, dict], dict]:
    source = Path(cfg.actual_result_json)
    if not source.exists():
        raise FileNotFoundError(f"완료 actual M5 결과 JSON이 없습니다: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("code_version") != EXPECTED_ACTUAL_CODE_VERSION:
        raise RuntimeError("완료 actual M5 code_version이 고정값과 다릅니다")
    if payload.get("source_revision") != EXPECTED_ACTUAL_SOURCE_REVISION:
        raise RuntimeError("완료 actual M5 source revision이 고정값과 다릅니다")
    if payload.get("input_manifest") != prepared["manifest"]:
        raise RuntimeError("완료 actual M5와 현재 입력 manifest가 다릅니다")
    if not payload.get("screening_reading", {}).get("directional_screen_pass"):
        raise RuntimeError("완료 actual M5가 사전 방향성 screen을 통과하지 않았습니다")

    expected_config = {
        key: getattr(cfg, key)
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
            "gate_delta",
            "basis_bandwidth",
            "positive_weight_lambda",
            "n_layers",
            "negative_count",
            "batch_size",
            "lr",
            "pref_reg",
            "input_days",
        )
    }
    source_config = payload.get("config", {})
    mismatches = {
        key: (source_config.get(key), expected)
        for key, expected in expected_config.items()
        if source_config.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"완료 actual M5 설정 불일치: {mismatches}")

    actual_row = _find_one(
        payload.get("absolute_rows", []), ACTUAL_M5_MODEL_ID, "absolute_rows"
    )
    m4_row = _find_one(
        payload.get("absolute_rows", []), M4_MODEL_ID, "absolute_rows"
    )
    actual_arm = payload.get("arms", {}).get(ACTUAL_M5_MODEL_ID)
    m4_arm = (
        payload.get("reused_references", {})
        .get(M4_MODEL_ID, {})
        .get("arm")
    )
    if not actual_arm or not actual_arm.get("metrics"):
        raise RuntimeError("완료 actual M5 arm 또는 지표가 없습니다")
    if not m4_arm or not m4_arm.get("metrics"):
        raise RuntimeError("완료 M4 reference arm 또는 지표가 없습니다")

    actual_row.update(
        {
            "execution_source": "reused_completed_directional_run",
            "source_result_path": str(source),
            "source_code_version": payload["code_version"],
            "source_revision": payload["source_revision"],
        }
    )
    m4_row.update(
        {
            "execution_source": "reused_through_completed_directional_run",
            "source_result_path": str(source),
            "source_code_version": payload["code_version"],
            "source_revision": payload["source_revision"],
        }
    )
    references = {
        M4_MODEL_ID: {"row": m4_row, "metrics": m4_arm["metrics"], "arm": m4_arm},
        ACTUAL_M5_MODEL_ID: {
            "row": actual_row,
            "metrics": actual_arm["metrics"],
            "arm": actual_arm,
        },
    }
    provenance = {
        "reused_without_retraining": True,
        "path": str(source),
        "code_version": payload["code_version"],
        "source_revision": payload["source_revision"],
        "input_manifest_match": True,
        "config_match": True,
        "directional_screen_pass": True,
        "actual_n_gate_mean": actual_row.get("n_gate_mean"),
        "actual_n_gate_std": actual_row.get("n_gate_std"),
    }
    return references, provenance


def mechanism_reading(metric_rows: dict[str, dict]) -> dict:
    m4 = metric_rows[M4_MODEL_ID]
    actual = metric_rows[ACTUAL_M5_MODEL_ID]
    shuffled = metric_rows[SHUFFLED_M5_MODEL_ID]
    v_only = metric_rows[V_ONLY_M5_MODEL_ID]

    def beats(reference: dict) -> bool:
        return all(actual[metric] > reference[metric] for metric in ECONOMIC_METRICS)

    def deltas(reference: dict) -> dict[str, float]:
        metrics = ACCURACY_METRICS + ECONOMIC_METRICS
        return {
            metric: float(actual[metric] - reference[metric]) for metric in metrics
        }

    def accuracy_geomean(reference: dict) -> float:
        ratios = [actual[metric] / reference[metric] for metric in ACCURACY_METRICS]
        return float(math.exp(np.log(ratios).mean()))

    assignment_signal = beats(shuffled)
    n_increment_signal = beats(v_only)
    actual_beats_m4 = beats(m4)
    if assignment_signal and n_increment_signal:
        classification = "n_and_v_assignment_candidate"
    elif assignment_signal:
        classification = "value_assignment_without_n_increment"
    elif n_increment_signal:
        classification = "n_increment_without_assignment_attribution"
    else:
        classification = "no_assignment_or_n_increment_signal"
    return {
        "mechanism_screen_pass": bool(assignment_signal and n_increment_signal),
        "actual_beats_m4_on_both_top10_economic_metrics": actual_beats_m4,
        "actual_beats_degree_matched_nv_shuffle": assignment_signal,
        "actual_beats_qv_only_constant_gate": n_increment_signal,
        "classification": classification,
        "actual_minus_m4": deltas(m4),
        "actual_minus_degree_matched_nv_shuffle": deltas(shuffled),
        "actual_minus_qv_only_constant_gate": deltas(v_only),
        "accuracy_geomean_ratio_actual_vs_m4": accuracy_geomean(m4),
        "accuracy_geomean_ratio_actual_vs_nv_shuffle": accuracy_geomean(shuffled),
        "accuracy_geomean_ratio_actual_vs_qv_only": accuracy_geomean(v_only),
        "decision_scope": (
            "single-seed exposed development mechanism screen; not a final model "
            "selection or population claim"
        ),
        "next_if_pass": (
            "freeze the complete M5 specification before the one-time multiseed test"
        ),
        "next_if_value_only": (
            "do not claim q_N contribution; retain q_V+M4 only as a separate finding "
            "or redesign N on a new predeclared development period"
        ),
        "next_if_nonpass": (
            "stop this N-conditioned value-basis branch without tuning it on the same "
            "development interval"
        ),
        "statistical_note": (
            "one historical development seed; no significance, stability, "
            "generalization, or final CLV attribution claim"
        ),
    }


def run_value_basis_controls(
    cfg: M5NConditionedValueBasisControlsConfig | None = None,
) -> pd.DataFrame:
    cfg = validate_config(cfg or configure_value_basis_controls())
    summary = preflight_summary(cfg)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    prepared = _prepare(cfg)
    references, provenance = load_completed_references(cfg, prepared)
    print("\n[재사용] 완료 M4와 actual M5를 재학습하지 않습니다.")
    print(f"  - {provenance['path']}")

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

    rows = [references[model_id]["row"] for model_id in REUSED_MODEL_IDS]
    metric_rows = {
        model_id: references[model_id]["metrics"] for model_id in REUSED_MODEL_IDS
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
                "execution_source": "current_two_control_run",
                "source_result_path": None,
                "source_code_version": CODE_VERSION,
                "source_revision": prepared["revision"],
                **arm["diagnostics"],
                **arm["training"].get("final_diagnostics", {}),
                **arm["metrics"],
            }
        )
    frame = pd.DataFrame(rows)

    full_comparison = report_helpers._metric_comparison(
        metric_rows,
        references=(M4_MODEL_ID, SHUFFLED_M5_MODEL_ID, V_ONLY_M5_MODEL_ID),
    )
    comparison = full_comparison.loc[
        full_comparison["model_id"] == ACTUAL_M5_MODEL_ID
    ].reset_index(drop=True)

    score_rows = []
    completed_scores = json.loads(
        Path(cfg.actual_result_json).read_text(encoding="utf-8")
    ).get("score_diagnostic_rows", [])
    score_rows.append(
        _find_one(completed_scores, ACTUAL_M5_MODEL_ID, "score_diagnostic_rows")
    )
    for model_id in TRAINED_MODEL_IDS:
        users, top50 = report_helpers._masked_topk(
            models[model_id], prepared, max_k=cfg.diagnostic_max_k
        )
        score_rows.append(
            legacy._score_diagnostics(
                models[model_id], users, top50, model_id=model_id
            )
        )
    score_frame = pd.DataFrame(score_rows)
    reading = mechanism_reading(metric_rows)

    out = Path(cfg.out_dir)
    stem = f"m5_n_conditioned_value_basis_controls_{prepared['config_hash']}"
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
            "control_diagnostics": prepared["control_diagnostics"],
            "absolute_rows": frame.to_dict("records"),
            "comparison_rows": comparison.to_dict("records"),
            "score_diagnostic_rows": score_frame.to_dict("records"),
            "mechanism_reading": reading,
            "reused_reference": provenance,
            "arms": arms,
            "result_paths": {key: str(value) for key, value in paths.items()},
        },
    )
    frame.attrs["comparison"] = comparison
    frame.attrs["score_diagnostics"] = score_frame
    frame.attrs["decision"] = reading
    frame.attrs["control_diagnostics"] = prepared["control_diagnostics"]
    frame.attrs["reference_provenance"] = provenance
    frame.attrs["preflight"] = summary
    frame.attrs["result_paths"] = {key: str(value) for key, value in paths.items()}

    print("\n1) 재사용 M4·actual M5와 새 대조군 절대지표")
    print(frame.to_string(index=False))
    print("\n2) M4·N/V 순열·V-only 대비 actual M5 전체 지표")
    print(comparison.to_string(index=False))
    print("\n3) actual·N/V 순열·V-only 경제점수 영향력")
    print(score_frame.to_string(index=False))
    print("\n4) 순열 불변식")
    print(json.dumps(prepared["control_diagnostics"], ensure_ascii=False, indent=2))
    print("\n5) 기제 판독")
    print(json.dumps(reading, ensure_ascii=False, indent=2))
    print("\n6) 저장 파일")
    print(json.dumps(frame.attrs["result_paths"], ensure_ascii=False, indent=2))
    return frame


if __name__ == "__main__":
    print(
        "Import this module from the dedicated Colab notebook and call "
        "run_value_basis_controls()."
    )
