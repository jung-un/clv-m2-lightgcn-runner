"""K=1 attribution screen for the personalized positive-row M4.

The three arms share the same initialization, one-uniform-negative BPR,
binary graph, optimizer, and fixed 100-epoch schedule.  The only difference
between the two M4 arms is whether the historical-CLV percentile ``q_C`` is
assigned to its observed user or permuted among CLV-valid users in the same
binary user-degree decile.  Item amount percentile and the user's observed
economic-bin fit are never shuffled.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import lightgcn_clv_gradient_isolated_economic_interaction as report_helpers
import lightgcn_clv_m4_clv_hard_negative_multiseed as shuffle_helpers
import lightgcn_clv_m5_economic_positive_weight as legacy
import lightgcn_clv_m5_k1_m4_improvement_screen as training
import lightgcn_clv_m5_clv_scaled_value_basis_k1_screen as screen
import lightgcn_clv_v3 as v3


CODE_VERSION = "m4-personalized-positive-weight-k1-assignment-control-development-screen-v1"
M1_MODEL_ID = "m1_bpr_k1_m4_assignment_control"
M4_ACTUAL_MODEL_ID = "m4_personalized_positive_weight_actual_qc_bpr_k1"
M4_SHUFFLED_MODEL_ID = (
    "m4_personalized_positive_weight_degree_matched_qc_shuffle_bpr_k1"
)
MODEL_IDS = (M1_MODEL_ID, M4_ACTUAL_MODEL_ID, M4_SHUFFLED_MODEL_ID)
ECONOMIC_METRICS = screen.ECONOMIC_METRICS
ACCURACY_METRICS = screen.ACCURACY_METRICS
ACCURACY_GUARD = 0.99

M4K1AssignmentControlConfig = training.M5K1ImprovementConfig


def configure_m4_k1_assignment_control_screen(
    **overrides,
) -> M4K1AssignmentControlConfig:
    data_root = v3.default_out_dir("dunnhumby")
    defaults = {
        "out_dir": f"{data_root}_m4_k1_assignment_control_development_screen_v1",
        "baseline_result_dir": f"{data_root}_m2_repeatshare_historical_backtest_v1",
    }
    return validate_config(M4K1AssignmentControlConfig(**(defaults | overrides)))


def validate_config(
    cfg: M4K1AssignmentControlConfig,
) -> M4K1AssignmentControlConfig:
    training.validate_config(cfg)
    if cfg.negative_count != 1:
        raise ValueError("M4 attribution screen은 양성당 균등 음성 1개여야 합니다")
    if cfg.seed != 42 or cfg.time_cutoff != 690 or cfg.evaluation_days != 7:
        raise ValueError("M4 attribution screen은 DAY 1~683 -> 684~690 seed 42입니다")
    return cfg


def preflight_summary(cfg: M4K1AssignmentControlConfig) -> dict:
    cfg = validate_config(cfg)
    return {
        "code_version": CODE_VERSION,
        "dataset": cfg.dataset,
        "seed": cfg.seed,
        "split": "historical_development_days_684_690",
        "trained_models": list(MODEL_IDS),
        "reused_models": [],
        "research_question": (
            "Does the K=1 personalized positive-row M4 improve new-item "
            "recommendation because the observed historical q_C is assigned "
            "to the correct customer?"
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
                "uniform negative stream",
                "BPR loss and optimizer",
            ],
        },
        "loss": {
            "bpr": "row_weight*softplus(s(u,j)-s(u,i+)) plus sampled layer-0 L2",
            "negative_count": cfg.negative_count,
            "negative_sampling": "one uniformly sampled unseen item",
            "hard_negative": False,
        },
        "reading_rule": {
            "primary_metrics": list(ECONOMIC_METRICS),
            "actual_beats_baseline": "actual M4 > M1 on both primary metrics",
            "actual_beats_shuffle": (
                "actual M4 > degree-matched q_C shuffle on both primary metrics"
            ),
            "accuracy_guard": (
                "each Recall/NDCG@10,@20,@50 of actual M4 >= 99% of M1"
            ),
            "evaluable": "actual and shuffled M4 change at least one Top-10 set",
            "intervention": "both M4 row-weight coefficients of variation are positive",
            "attribution_pass": (
                "all baseline, shuffle, accuracy, evaluability and intervention "
                "conditions hold"
            ),
        },
        "fixed": {
            "new_item_task": True,
            "train_pairs_excluded_from_truth_and_candidates": True,
            "min_item_interactions": 1,
            "graph": "binary",
            "epochs": cfg.epochs,
            "validation_or_epoch_selection": False,
            "final_test_constructed": False,
            "holdout_constructed": False,
            "one_training_loop_and_optimizer_per_arm": True,
            "external_reranking": False,
        },
        "next_if_pass": "run the same M1/actual/shuffle comparison over seeds 42--51",
        "next_if_nonpass": (
            "stop attributing the current M4 gain to q_C assignment; test the "
            "separately preregistered first-purchase-row M4 as a new hypothesis"
        ),
        "statistical_note": (
            "one exposed historical development seed; no significance, stability "
            "or generalization claim"
        ),
        "out_dir": cfg.out_dir,
    }


def _config_hash(
    cfg: M4K1AssignmentControlConfig, input_hash: str, revision: str
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


def degree_matched_q_c_shuffle(
    prepared: dict, cfg: M4K1AssignmentControlConfig
) -> dict:
    """Shuffle q_C only among valid users in binary-degree rank strata."""

    result = shuffle_helpers.degree_matched_q_clv_shuffle(
        prepared["q_c"],
        prepared["clv_valid"],
        prepared["degree"],
        n_bins=cfg.shuffle_degree_bins,
        seed=cfg.shuffle_seed,
    )
    actual = np.asarray(prepared["q_c"], dtype=np.float32)
    shuffled = np.asarray(result["q_clv"], dtype=np.float32)
    valid = np.asarray(prepared["clv_valid"], dtype=bool)
    source = np.asarray(result["source_user"], dtype=np.int64)
    strata = np.asarray(result["stratum"], dtype=np.int16)

    if not np.array_equal(np.sort(actual[valid]), np.sort(shuffled[valid])):
        raise RuntimeError("q_C 순열이 유효 사용자 값의 multiset을 보존하지 못했습니다")
    if np.any(shuffled[~valid] != 0.0):
        raise RuntimeError("무효 사용자의 순열 q_C는 0이어야 합니다")
    changed = np.flatnonzero(valid & (source != np.arange(len(source))))
    if not len(changed) or np.any(strata[changed] != strata[source[changed]]):
        raise RuntimeError("q_C 순열이 degree 층 안에서 사용자 배정을 바꾸지 못했습니다")

    result["q_c"] = shuffled
    result["valid_user_count"] = int(valid.sum())
    result["invalid_user_count"] = int((~valid).sum())
    result["q_c_multiset_preserved"] = True
    result["invalid_q_c_zero_preserved"] = True
    result["degree_stratum_preserved"] = True
    return result


def _prepare(cfg: M4K1AssignmentControlConfig) -> dict:
    prepared = training._prepare(cfg)
    prepared["config_hash"] = _config_hash(
        cfg, prepared["input_hash"], prepared["revision"]
    )
    prepared["q_c_shuffle"] = degree_matched_q_c_shuffle(prepared, cfg)
    return prepared


def arm_specifications(prepared: dict) -> list[dict]:
    return [
        {
            "model_id": M1_MODEL_ID,
            "role": "k1_m1",
            "rho": 0.0,
            "improvement": None,
            "m4_assignment": "inactive",
            "q_c": prepared["q_c"],
        },
        {
            "model_id": M4_ACTUAL_MODEL_ID,
            "role": "k1_m4_actual_q_c",
            "rho": 0.0,
            "improvement": "original",
            "m4_assignment": "observed_q_c",
            "q_c": prepared["q_c"],
        },
        {
            "model_id": M4_SHUFFLED_MODEL_ID,
            "role": "k1_m4_degree_matched_q_c_shuffle",
            "rho": 0.0,
            "improvement": "original",
            "m4_assignment": "degree_matched_q_c_shuffle",
            "q_c": prepared["q_c_shuffle"]["q_c"],
        },
    ]


def _arm_prepared(prepared: dict, spec: dict) -> dict:
    """Replace only the q_C consumed by the M4 row-weight path."""

    arm_prepared = dict(prepared)
    arm_prepared["q_c"] = np.asarray(spec["q_c"], dtype=np.float32)
    m2_actual = dict(prepared["m2_actual"])
    m2_actual["q_c"] = arm_prepared["q_c"]
    arm_prepared["m2_actual"] = m2_actual
    return arm_prepared


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
    intervention = all(row_weight_cvs[model_id] > 0.0 for model_id in (
        M4_ACTUAL_MODEL_ID,
        M4_SHUFFLED_MODEL_ID,
    ))
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
            "q_c_assignment_supported" if passed else "q_c_assignment_not_supported"
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
        "next_if_pass": (
            "freeze this M4 and repeat M1/actual/shuffle over seeds 42--51"
        ),
        "next_if_nonpass": (
            "do not attribute the current M4 gain to q_C assignment; move to "
            "the separately preregistered first-purchase-row M4"
        ),
        "statistical_note": (
            "one exposed historical development seed; no significance, stability "
            "or generalization claim"
        ),
    }


def run_m4_k1_assignment_control_screen(
    cfg: M4K1AssignmentControlConfig | None = None,
) -> pd.DataFrame:
    cfg = validate_config(cfg or configure_m4_k1_assignment_control_screen())
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
        arm, model = training._run_arm(_arm_prepared(prepared, spec), cfg, spec)
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
    control_diagnostics.update(
        q_c_multiset_preserved=control["q_c_multiset_preserved"],
        invalid_q_c_zero_preserved=control["invalid_q_c_zero_preserved"],
        degree_stratum_preserved=control["degree_stratum_preserved"],
    )

    out = Path(cfg.out_dir)
    stem = f"m4_k1_assignment_control_{prepared['config_hash']}"
    paths = {
        "absolute_csv": out / f"{stem}.csv",
        "comparison_csv": out / f"{stem}_comparison.csv",
        "top10_overlap_csv": out / f"{stem}_top10_overlap.csv",
        "json": out / f"{stem}.json",
    }
    legacy.test10._atomic_csv(paths["absolute_csv"], frame)
    legacy.test10._atomic_csv(paths["comparison_csv"], comparison)
    legacy.test10._atomic_csv(paths["top10_overlap_csv"], overlap)
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


if __name__ == "__main__":
    print(
        json.dumps(
            preflight_summary(configure_m4_k1_assignment_control_screen()),
            ensure_ascii=False,
            indent=2,
        )
    )
