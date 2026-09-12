"""Standalone mechanism controls for the N-conditioned value-basis M5.

This runner trains the observed M5, a degree-matched joint q_N/q_V assignment
control, and a q_V-only constant-gate control in the same run.  It has no
dependency on a previously saved result.  M4 always keeps its observed q_C
and user-bin fit.
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


CODE_VERSION = "m5-n-conditioned-value-basis-controls-development-screen-v2"
ACTUAL_M5_MODEL_ID = base.BASIS_M5_MODEL_ID
SHUFFLED_M5_MODEL_ID = "m5_n_conditioned_value_basis_degree_matched_nv_shuffle"
V_ONLY_M5_MODEL_ID = "m5_value_basis_constant_gate_personalized_positive_weight_k5"
REUSED_MODEL_IDS: tuple[str, ...] = ()
TRAINED_MODEL_IDS = (
    ACTUAL_M5_MODEL_ID,
    SHUFFLED_M5_MODEL_ID,
    V_ONLY_M5_MODEL_ID,
)
MODEL_IDS = REUSED_MODEL_IDS + TRAINED_MODEL_IDS
ECONOMIC_METRICS = base.ECONOMIC_METRICS
ACCURACY_METRICS = base.ACCURACY_METRICS


@dataclass(frozen=True)
class M5NConditionedValueBasisControlsConfig(base.M5NConditionedValueBasisConfig):
    constant_gate: float = 1.25


def configure_value_basis_controls(
    **overrides,
) -> M5NConditionedValueBasisControlsConfig:
    data_root = v3.default_out_dir("dunnhumby")
    base_cfg = base.configure_n_conditioned_value_basis_screen()
    defaults = asdict(base_cfg) | {
        "out_dir": (
            f"{data_root}"
            "_m5_n_conditioned_value_basis_controls_development_screen_v2"
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
        "reused_models": [],
        "research_question": (
            "Did the observed M5 benefit from correctly assigned q_N/q_V, "
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
            "model_id": ACTUAL_M5_MODEL_ID,
            "role": "observed_nv_actual_m5",
            "architecture": "q_n_conditioned_basis",
            "rho": cfg.rho,
            "weighted": True,
            "assignment": prepared,
            "assignment_name": "observed_m4",
            "m2_assignment": prepared["m2_actual"],
            "m2_assignment_name": "observed_nv",
            "constant_gate": None,
        },
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


def mechanism_reading(metric_rows: dict[str, dict]) -> dict:
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
        "actual_beats_degree_matched_nv_shuffle": assignment_signal,
        "actual_beats_qv_only_constant_gate": n_increment_signal,
        "classification": classification,
        "actual_minus_degree_matched_nv_shuffle": deltas(shuffled),
        "actual_minus_qv_only_constant_gate": deltas(v_only),
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
                "execution_source": "current_standalone_three_arm_run",
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
        references=(SHUFFLED_M5_MODEL_ID, V_ONLY_M5_MODEL_ID),
    )
    comparison = full_comparison.loc[
        full_comparison["model_id"] == ACTUAL_M5_MODEL_ID
    ].reset_index(drop=True)

    score_rows = []
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
            "arms": arms,
            "result_paths": {key: str(value) for key, value in paths.items()},
        },
    )
    frame.attrs["comparison"] = comparison
    frame.attrs["score_diagnostics"] = score_frame
    frame.attrs["decision"] = reading
    frame.attrs["control_diagnostics"] = prepared["control_diagnostics"]
    frame.attrs["preflight"] = summary
    frame.attrs["result_paths"] = {key: str(value) for key, value in paths.items()}

    print("\n1) 같은 실행의 actual M5·N/V 순열·V-only 절대지표")
    print(frame.to_string(index=False))
    print("\n2) N/V 순열·V-only 대비 actual M5 전체 지표")
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
