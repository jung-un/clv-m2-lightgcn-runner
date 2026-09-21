"""H&M development screen for the K=1 personalized positive-row M4.

The three arms use the same H&M two-year development split, initialization,
binary graph, one-uniform-negative BPR, optimizer, and fixed 100 epochs.  The
only difference between the two M4 arms is whether the historical-CLV
percentile ``q_C`` is assigned to its observed user or permuted among valid
users in the same binary-degree decile.

Every completed epoch is saved through :class:`clv_run_state.ProgressStore`.
Re-running the same Colab cell therefore resumes the active arm at the next
epoch, while fully completed arms are loaded from their final checkpoints.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import lightgcn_clv_axis_specific_test10 as test10
import lightgcn_clv_gradient_isolated_economic_interaction as report_helpers
import lightgcn_clv_m4_k1_assignment_control_screen as control_helpers
import lightgcn_clv_m5_k1_m4_improvement_screen as training
import lightgcn_clv_m5_nv_economic_positive_weight as economic_helpers
import lightgcn_clv_m5_value_basis_hm2y_screen as hm_base
import lightgcn_clv_v3 as v3


CODE_VERSION = "m4-personalized-positive-weight-k1-assignment-control-hm2y-development-screen-v1"
SPLIT_LABEL = "hm2y_validation_2020-09-02_08"
M1_MODEL_ID = "m1_bpr_k1_hm2y_m4_assignment_control"
M4_ACTUAL_MODEL_ID = "m4_personalized_positive_weight_actual_qc_bpr_k1_hm2y"
M4_SHUFFLED_MODEL_ID = (
    "m4_personalized_positive_weight_degree_matched_qc_shuffle_bpr_k1_hm2y"
)
MODEL_IDS = (M1_MODEL_ID, M4_ACTUAL_MODEL_ID, M4_SHUFFLED_MODEL_ID)
ECONOMIC_METRICS = hm_base.ECONOMIC_METRICS
ACCURACY_METRICS = hm_base.ACCURACY_METRICS
ACCURACY_GUARD = 0.99
REPLICATION_SEEDS = (42, 43, 44)


@dataclass(frozen=True)
class M4K1AssignmentHm2yConfig(hm_base.M5ValueBasisHm2yConfig):
    economic_bins: int = 4
    shrinkage_strength: float = 10.0
    positive_weight_lambda: float = 0.5


def configure_hm2y_m4_assignment_screen(**overrides) -> M4K1AssignmentHm2yConfig:
    defaults = {
        "out_dir": (
            f"{v3.default_out_dir('hm')}"
            "_m4_k1_assignment_control_hm2y_development_screen_v1"
        )
    }
    return validate_config(M4K1AssignmentHm2yConfig(**(defaults | overrides)))


def validate_config(cfg: M4K1AssignmentHm2yConfig) -> M4K1AssignmentHm2yConfig:
    fixed = {
        "dataset": "hm",
        "epochs": 100,
        "id_dim": 64,
        "n_layers": 2,
        "negative_count": 1,
        "batch_size": 131_072,
        "input_days": 365,
        "economic_bins": 4,
        "shrinkage_strength": 10.0,
        "positive_weight_lambda": 0.5,
        "shuffle_degree_bins": 10,
        "include_shuffle": True,
    }
    for key, expected in fixed.items():
        if getattr(cfg, key) != expected:
            raise ValueError(f"H&M M4 K=1 screen은 {key}={expected!r}이어야 합니다")
    if cfg.seed not in REPLICATION_SEEDS:
        raise ValueError(f"H&M M4 K=1 replication seed는 {REPLICATION_SEEDS} 중 하나여야 합니다")
    if cfg.shuffle_seed != cfg.seed:
        raise ValueError("shuffle_seed는 해당 학습 seed와 같아야 합니다")
    if cfg.lr <= 0 or cfg.pref_reg < 0 or not cfg.out_dir:
        raise ValueError("H&M M4 K=1 screen의 학습 설정 또는 out_dir가 잘못됐습니다")
    return cfg


def preflight_summary(cfg: M4K1AssignmentHm2yConfig) -> dict:
    cfg = validate_config(cfg)
    return {
        "code_version": CODE_VERSION,
        "dataset": cfg.dataset,
        "seed": cfg.seed,
        "split": SPLIT_LABEL,
        "trained_models": list(MODEL_IDS),
        "reused_models": [],
        "research_question": (
            "Does the K=1 personalized positive-row M4 improve H&M new-item "
            "recommendation, and is any gain due to the observed q_C assignment?"
        ),
        "m4": {
            "historical_clv_proxy": "q_C=percentile(n_u*v_u)",
            "raw_positive_row_weight": (
                "1 + 0.5*q_C*item_amount_percentile*clipped_user_bin_fit"
            ),
            "normalization": "divide by the mean raw weight over all train rows",
            "positive_weight_lambda": cfg.positive_weight_lambda,
        },
        "control": {
            "permuted": "q_C only",
            "stratum": (
                f"{cfg.shuffle_degree_bins} binary user-degree rank strata, "
                "CLV-valid users only"
            ),
            "held_fixed": [
                "item amount percentile",
                "observed user economic-bin fit",
                "binary graph",
                "one uniform negative",
                "initialization, BPR loss and optimizer",
            ],
        },
        "reading_rule": {
            "primary_metrics": list(ECONOMIC_METRICS),
            "actual_beats_baseline": "actual M4 > M1 on both primary metrics",
            "actual_beats_shuffle": "actual M4 > q_C shuffle on both primary metrics",
            "accuracy_guard": "all six accuracy metrics of actual M4 >= 99% of M1",
            "evaluable": "actual and shuffled M4 change at least one Top-10 set",
            "intervention": "both M4 row-weight coefficients of variation are positive",
        },
        "staged_replication": {
            "seeds": list(REPLICATION_SEEDS),
            "completed_seed": 42,
            "new_parallel_seeds": [43, 44],
            "pass_rule": (
                "at least 2 of 3 seeds pass the frozen single-seed attribution rule"
            ),
            "next_if_pass": "extend the unchanged H&M protocol to five total seeds",
            "next_if_nonpass": "stop without extending to five or ten seeds",
        },
        "fixed": {
            "new_item_task": True,
            "train_pairs_excluded_from_truth_and_candidates": True,
            "min_item_interactions": 1,
            "graph": "binary",
            "negative_count": 1,
            "negative_sampling": "one uniformly sampled unseen item",
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "validation_or_epoch_selection": False,
            "final_test_constructed": False,
            "holdout_constructed": False,
            "one_training_loop_and_optimizer_per_arm": True,
            "external_reranking": False,
        },
        "checkpointing": {
            "save_after_each_completed_epoch": True,
            "atomic_replace": True,
            "resume": "next epoch after the latest completed epoch",
            "completed_arm_cache": True,
        },
        "statistical_note": (
            "one H&M development seed; no significance, stability, "
            "generalization or final CLV-attribution claim"
        ),
        "out_dir": cfg.out_dir,
    }


def _config_hash(
    cfg: M4K1AssignmentHm2yConfig, input_hash: str, revision: str
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


def _prepare(cfg: M4K1AssignmentHm2yConfig) -> dict:
    prepared = hm_base._prepare(cfg)
    economic = economic_helpers.build_nv_economic_inputs(
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
        "q_n": np.asarray(prepared["q_n"], dtype=np.float32),
        "q_v": np.asarray(prepared["q_v"], dtype=np.float32),
        "q_c": np.asarray(prepared["q_c"], dtype=np.float32),
        "clv_valid": np.asarray(prepared["clv_valid"], dtype=bool),
    }
    prepared["q_c_shuffle"] = control_helpers.degree_matched_q_c_shuffle(
        prepared, cfg
    )
    prepared["config_hash"] = _config_hash(
        cfg, prepared["input_hash"], prepared["revision"]
    )
    return prepared


def arm_specifications(prepared: dict) -> list[dict]:
    return [
        {
            "model_id": M1_MODEL_ID,
            "role": "hm2y_k1_m1",
            "rho": 0.0,
            "improvement": None,
            "m4_assignment": "inactive",
            "q_c": prepared["q_c"],
            "split": SPLIT_LABEL,
        },
        {
            "model_id": M4_ACTUAL_MODEL_ID,
            "role": "hm2y_k1_m4_actual_q_c",
            "rho": 0.0,
            "improvement": "original",
            "m4_assignment": "observed_q_c",
            "q_c": prepared["q_c"],
            "split": SPLIT_LABEL,
        },
        {
            "model_id": M4_SHUFFLED_MODEL_ID,
            "role": "hm2y_k1_m4_degree_matched_q_c_shuffle",
            "rho": 0.0,
            "improvement": "original",
            "m4_assignment": "degree_matched_q_c_shuffle",
            "q_c": prepared["q_c_shuffle"]["q_c"],
            "split": SPLIT_LABEL,
        },
    ]


def arm_prepared(prepared: dict, spec: dict) -> dict:
    """Replace only the q_C consumed by the M4 row-weight path."""

    arm = dict(prepared)
    arm["q_c"] = np.asarray(spec["q_c"], dtype=np.float32)
    m2_actual = dict(prepared["m2_actual"])
    m2_actual["q_c"] = arm["q_c"]
    arm["m2_actual"] = m2_actual
    return arm


def attribution_reading(
    metric_rows: dict[str, dict],
    *,
    actual_vs_shuffle_top10_change_share: float,
    row_weight_cvs: dict[str, float],
) -> dict:
    m1 = metric_rows[M1_MODEL_ID]
    actual = metric_rows[M4_ACTUAL_MODEL_ID]
    shuffled = metric_rows[M4_SHUFFLED_MODEL_ID]
    actual_beats_m1 = all(actual[m] > m1[m] for m in ECONOMIC_METRICS)
    actual_beats_shuffle = all(actual[m] > shuffled[m] for m in ECONOMIC_METRICS)
    accuracy_guard = all(
        actual[metric] >= ACCURACY_GUARD * m1[metric]
        for metric in ACCURACY_METRICS
    )
    evaluable = actual_vs_shuffle_top10_change_share > 0.0
    intervention = all(
        row_weight_cvs[model_id] > 0.0
        for model_id in (M4_ACTUAL_MODEL_ID, M4_SHUFFLED_MODEL_ID)
    )
    passed = bool(
        actual_beats_m1
        and actual_beats_shuffle
        and accuracy_guard
        and evaluable
        and intervention
    )

    def deltas(model: dict, reference: dict) -> dict[str, float]:
        return {
            metric: float(model[metric] - reference[metric])
            for metric in ACCURACY_METRICS + ECONOMIC_METRICS
        }

    ratios = [actual[metric] / m1[metric] for metric in ACCURACY_METRICS]
    return {
        "classification": (
            "hm_q_c_assignment_supported"
            if passed
            else "hm_q_c_assignment_not_supported"
        ),
        "attribution_pass": passed,
        "actual_beats_m1_on_both_economic_metrics": bool(actual_beats_m1),
        "actual_beats_shuffle_on_both_economic_metrics": bool(
            actual_beats_shuffle
        ),
        "six_accuracy_metrics_at_least_99pct_of_m1": bool(accuracy_guard),
        "actual_vs_shuffle_top10_changed": bool(evaluable),
        "actual_vs_shuffle_top10_set_changed_user_share": float(
            actual_vs_shuffle_top10_change_share
        ),
        "row_weight_intervention_operational": bool(intervention),
        "row_weight_cv": {
            key: float(value) for key, value in row_weight_cvs.items()
        },
        "accuracy_geomean_ratio_actual_vs_m1": float(
            math.exp(np.log(ratios).mean())
        ),
        "deltas_actual_minus_m1": deltas(actual, m1),
        "deltas_actual_minus_shuffle": deltas(actual, shuffled),
        "next_if_pass": "repeat the same frozen H&M screen over multiple seeds",
        "next_if_nonpass": "do not claim H&M q_C-assignment portability",
        "statistical_note": (
            "one H&M development seed; no significance, stability, "
            "generalization or final CLV-attribution claim"
        ),
    }


def run_hm2y_m4_assignment_screen(
    cfg: M4K1AssignmentHm2yConfig | None = None,
) -> pd.DataFrame:
    cfg = validate_config(cfg or configure_hm2y_m4_assignment_screen())
    summary = preflight_summary(cfg)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    prepared = _prepare(cfg)

    arms: dict[str, dict] = {}
    models: dict[str, object] = {}
    for spec in arm_specifications(prepared):
        print(
            f"\n===== {spec['model_id']} | seed {cfg.seed} | "
            f"K={cfg.negative_count} | fixed {cfg.epochs} epochs ====="
        )
        arm, model = training._run_arm(arm_prepared(prepared, spec), cfg, spec)
        arm["m4_assignment"] = spec["m4_assignment"]
        arms[spec["model_id"]] = arm
        models[spec["model_id"]] = model

    metric_rows = {model_id: arms[model_id]["metrics"] for model_id in MODEL_IDS}
    frame = pd.DataFrame(
        [
            {
                "model_id": model_id,
                "role": arms[model_id]["role"],
                "seed": arms[model_id]["seed"],
                "split": arms[model_id]["split"],
                "final_epoch": arms[model_id]["final_epoch"],
                "m4_assignment": arms[model_id]["m4_assignment"],
                "positive_weight_lambda": (
                    cfg.positive_weight_lambda if model_id != M1_MODEL_ID else 0.0
                ),
                **arms[model_id]["weight_diagnostics"],
                **arms[model_id]["metrics"],
            }
            for model_id in MODEL_IDS
        ]
    )
    comparison = report_helpers._metric_comparison(
        metric_rows, references=(M1_MODEL_ID, M4_SHUFFLED_MODEL_ID)
    )

    topk = {
        model_id: report_helpers._masked_topk(
            models[model_id], prepared, max_k=cfg.diagnostic_max_k
        )
        for model_id in MODEL_IDS
    }
    users = topk[M1_MODEL_ID][0]
    if not all(np.array_equal(users, topk[mid][0]) for mid in MODEL_IDS):
        raise RuntimeError("arm별 평가 사용자 순서가 다릅니다")
    overlap = pd.concat(
        [
            report_helpers.topk_overlap_summary(
                topk[reference][1], topk[model_id][1], prepared["cache"].seg
            ).assign(reference=reference, model_id=model_id)
            for reference, model_id in (
                (M1_MODEL_ID, M4_ACTUAL_MODEL_ID),
                (M1_MODEL_ID, M4_SHUFFLED_MODEL_ID),
                (M4_SHUFFLED_MODEL_ID, M4_ACTUAL_MODEL_ID),
            )
        ],
        ignore_index=True,
    )
    overall = overlap[overlap.group.eq("전체")].set_index(
        ["reference", "model_id"]
    )
    change_share = float(
        overall.at[
            (M4_SHUFFLED_MODEL_ID, M4_ACTUAL_MODEL_ID),
            "top10_set_changed_user_share",
        ]
    )
    row_weight_cvs = {
        model_id: float(arms[model_id]["weight_diagnostics"]["row_weight_cv"])
        for model_id in (M4_ACTUAL_MODEL_ID, M4_SHUFFLED_MODEL_ID)
    }
    reading = attribution_reading(
        metric_rows,
        actual_vs_shuffle_top10_change_share=change_share,
        row_weight_cvs=row_weight_cvs,
    )

    control = prepared["q_c_shuffle"]
    control_diagnostics = {
        key: value
        for key, value in control.items()
        if key not in {"q_clv", "q_c", "source_user", "stratum"}
    }
    out = Path(cfg.out_dir)
    stem = f"m4_k1_assignment_control_hm2y_{prepared['config_hash']}"
    paths = {
        "absolute_csv": out / f"{stem}.csv",
        "comparison_csv": out / f"{stem}_comparison.csv",
        "top10_overlap_csv": out / f"{stem}_top10_overlap.csv",
        "json": out / f"{stem}.json",
    }
    test10._atomic_csv(paths["absolute_csv"], frame)
    test10._atomic_csv(paths["comparison_csv"], comparison)
    test10._atomic_csv(paths["top10_overlap_csv"], overlap)
    test10._atomic_json(
        paths["json"],
        {
            "code_version": CODE_VERSION,
            "source_revision": prepared["revision"],
            "config": asdict(cfg),
            "preflight": summary,
            "input_manifest": prepared["manifest"],
            "absolute_rows": frame.to_dict("records"),
            "comparison_rows": comparison.to_dict("records"),
            "top10_overlap_rows": overlap.to_dict("records"),
            "control_diagnostics": control_diagnostics,
            "screening_reading": reading,
            "arms": arms,
            "result_paths": {key: str(value) for key, value in paths.items()},
        },
    )
    frame.attrs.update(
        comparison=comparison,
        top10_overlap=overlap,
        control_diagnostics=control_diagnostics,
        decision=reading,
        result_paths={key: str(value) for key, value in paths.items()},
    )
    print("\n1) 판독")
    print(json.dumps(reading, ensure_ascii=False, indent=2))
    print("\n결과 파일:", frame.attrs["result_paths"])
    return frame


def run_hm2y_m4_assignment_arm(
    cfg: M4K1AssignmentHm2yConfig,
    selected_model_id: str,
) -> dict:
    """Train or resume exactly one arm without changing its checkpoint identity.

    The selected arm uses the same config hash and arm path as the original
    three-arm runner.  This makes it safe to run the three arms in separate
    Colab runtimes and aggregate them later with
    :func:`run_hm2y_m4_assignment_screen`.
    """

    cfg = validate_config(cfg)
    if selected_model_id not in MODEL_IDS:
        raise ValueError(
            f"selected_model_id는 {MODEL_IDS} 중 하나여야 합니다"
        )
    summary = preflight_summary(cfg)
    summary["parallel_execution"] = {
        "selected_model_id": selected_model_id,
        "checkpoint_identity_unchanged": True,
        "aggregate_after_all_arms": "run_hm2y_m4_assignment_screen(cfg)",
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    prepared = _prepare(cfg)
    spec = next(
        spec
        for spec in arm_specifications(prepared)
        if spec["model_id"] == selected_model_id
    )
    print(
        f"\n===== {spec['model_id']} | seed {cfg.seed} | "
        f"K={cfg.negative_count} | fixed {cfg.epochs} epochs | "
        "parallel single-arm ====="
    )
    arm, _ = training._run_arm(arm_prepared(prepared, spec), cfg, spec)
    arm["m4_assignment"] = spec["m4_assignment"]
    print(
        "\n선택 arm 완료. 세 arm이 모두 완료되면 "
        "run_hm2y_m4_assignment_screen(cfg)로 집계하세요."
    )
    return arm


if __name__ == "__main__":
    print(
        json.dumps(
            preflight_summary(configure_hm2y_m4_assignment_screen()),
            ensure_ascii=False,
            indent=2,
        )
    )
