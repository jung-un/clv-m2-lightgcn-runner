"""Four-arm single-negative screen for the CLV-scaled value-basis M5.

Plan A of 2026-09-15. M1, M2, M4 and M5 are trained in the same run with the
original LightGCN BPR (one uniform unseen negative per positive row). No arm is
reused from another run.

* M2 scales a fixed q_V price-position RBF basis by q_C (and the bounded q_N
  gate) with rho=0.25, and appends that block after ID propagation so each
  user's own CLV position is not averaged with neighbours.
* M4 keeps the personalized positive weight
  ``1 + 0.5*q_C*item_amount_percentile*clipped_user_bin_fit``.

The 2026-09-15 price-band diagnostic found that 68-75% of M1's missed-truth /
false-positive pairs lie in different price bands and that q_V separates those
pairs for fixed high-CLV users in both datasets. This screen asks whether a
block strong enough to act on them changes Top-10 lists and improves the
economic metrics beyond M4 without breaking accuracy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from clv_m5_n_conditioned_value_basis_model import M5NConditionedValueBasisLightGCN
import lightgcn_clv_gradient_isolated_economic_interaction as report_helpers
import lightgcn_clv_m5_economic_positive_weight as legacy
import lightgcn_clv_m5_n_conditioned_value_basis_screen as base
import lightgcn_clv_v3 as v3


CODE_VERSION = "m5-clv-scaled-value-basis-k1-four-arm-screen-v1"
M1_MODEL_ID = "m1_bpr_k1"
M2_MODEL_ID = "m2_clv_scaled_value_basis_bpr_k1"
M4_MODEL_ID = "m4_personalized_positive_weight_bpr_k1"
M5_MODEL_ID = "m5_clv_scaled_value_basis_positive_weight_bpr_k1"
MODEL_IDS = (M1_MODEL_ID, M2_MODEL_ID, M4_MODEL_ID, M5_MODEL_ID)
ECONOMIC_METRICS = base.ECONOMIC_METRICS
TOP10_ACCURACY_METRICS = base.TOP10_ACCURACY_METRICS
ACCURACY_METRICS = base.ACCURACY_METRICS
ACCURACY_GUARD = 0.99


@dataclass(frozen=True)
class M5ValueBasisK1Config(base.M5NConditionedValueBasisConfig):
    rho: float = 0.25
    negative_count: int = 1


def configure_value_basis_k1_screen(**overrides) -> M5ValueBasisK1Config:
    data_root = v3.default_out_dir("dunnhumby")
    defaults = {
        "out_dir": f"{data_root}_m5_clv_scaled_value_basis_k1_screen_v1",
        "baseline_result_dir": f"{data_root}_m2_repeatshare_historical_backtest_v1",
    }
    return validate_config(M5ValueBasisK1Config(**(defaults | overrides)))


def validate_config(cfg: M5ValueBasisK1Config) -> M5ValueBasisK1Config:
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
        "rho": 0.25,
        "gate_delta": 0.25,
        "basis_bandwidth": 0.25,
        "positive_weight_lambda": 0.5,
        "n_layers": 2,
        "negative_count": 1,
        "input_days": 365,
        "diagnostic_max_k": 50,
        "shuffle_degree_bins": 10,
        "shuffle_seed": 42,
    }
    for key, expected in fixed.items():
        if getattr(cfg, key) != expected:
            raise ValueError(f"K=1 가치기저 M5 screen은 {key}={expected!r}이어야 합니다")
    if cfg.batch_size <= 0 or cfg.lr <= 0 or cfg.pref_reg < 0:
        raise ValueError("학습 설정이 잘못됐습니다")
    if not cfg.out_dir or not cfg.baseline_result_dir:
        raise ValueError("out_dir와 baseline_result_dir가 필요합니다")
    return cfg


def preflight_summary(cfg: M5ValueBasisK1Config) -> dict:
    cfg = validate_config(cfg)
    return {
        "code_version": CODE_VERSION,
        "dataset": cfg.dataset,
        "seed": cfg.seed,
        "split": "historical_development_days_684_690",
        "trained_models": list(MODEL_IDS),
        "reused_models": [],
        "research_question": (
            "With the original single-negative BPR, does a q_C-scaled q_V price "
            "basis strong enough to change Top-10 lists add economic hits "
            "beyond M4 without losing accuracy?"
        ),
        "loss": {
            "bpr": "mean softplus(s(u,j) - s(u,i+)) + batch layer-0 L2",
            "negative_count": cfg.negative_count,
            "negative_sampling": "uniform over unseen items",
            "m4_weight": "1 + 0.5*q_C*item_amount_percentile*clipped_user_bin_fit, train-row mean 1",
        },
        "m2": {
            "user_block": "sqrt(rho) * q_C * g_N(q_N) * RBF(q_V)",
            "item_block": "sqrt(rho) * RBF(item_amount_percentile)",
            "rho": cfg.rho,
            "basis_bandwidth": cfg.basis_bandwidth,
            "q_n_gate_range": [1.0 - cfg.gate_delta, 1.0 + cfg.gate_delta],
            "economic_graph_propagation": False,
            "q_c_used_in_m2": True,
            "item_n_or_item_clv_input": False,
        },
        "arms": {
            M1_MODEL_ID: "rho=0, unweighted BPR",
            M2_MODEL_ID: f"rho={cfg.rho}, unweighted BPR",
            M4_MODEL_ID: "rho=0, M4 positive weight",
            M5_MODEL_ID: f"rho={cfg.rho}, M4 positive weight",
        },
        "c2_basis": (
            "rho=0.05 changed no Top-10 metric under CLV shuffling "
            "(c9ee4ef4d331); the economic score was 4.86% of the ID score and "
            "the learned strength saturated twice. rho=0.25 is fixed before "
            "this run and is not tuned afterwards."
        ),
        "reading_rule": {
            "evaluable": "M5 changes the Top-10 set of at least one user versus M4",
            "m5_beats_m4": "M5 > M4 on both economic metrics",
            "m5_beats_m1": "M5 > M1 on both economic metrics",
            "accuracy_guard": f"M5 Recall@10 and NDCG@10 >= {ACCURACY_GUARD} x M1",
            "directional_screen_pass": "evaluable and m5_beats_m4 and m5_beats_m1 and accuracy_guard",
            "reported_not_gated": (
                "interaction (M5-M4)-(M2-M1), Top-10 change shares, same/cross "
                "price-band error pairs, economic score ratio, @20/@50, segments"
            ),
            "clv_attribution": "not tested; a CLV assignment shuffle follows only after a pass",
        },
        "fixed": {
            "new_item_task": True,
            "min_item_interactions": 1,
            "graph": "binary",
            "epochs": cfg.epochs,
            "validation_or_epoch_selection": False,
            "final_test_constructed": False,
            "holdout_constructed": False,
            "one_training_loop_and_optimizer_per_arm": True,
            "external_reranking": False,
        },
        "statistical_note": "one historical development seed; no significance or generalization claim",
        "out_dir": cfg.out_dir,
    }


def _config_hash(cfg: M5ValueBasisK1Config, input_hash: str, revision: str) -> str:
    payload = {
        "code_version": CODE_VERSION,
        "config": asdict(cfg),
        "input_hash": input_hash,
        "source_revision": revision,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()[:12]


def _prepare(cfg: M5ValueBasisK1Config) -> dict:
    prepared = base._prepare(cfg)
    prepared["config_hash"] = _config_hash(
        cfg, prepared["input_hash"], prepared["revision"]
    )
    return prepared


def arm_specifications(prepared: dict, cfg: M5ValueBasisK1Config) -> list[dict]:
    def arm(model_id: str, role: str, rho: float, weighted: bool) -> dict:
        return {
            "model_id": model_id,
            "role": role,
            "rho": rho,
            "weighted": weighted,
            "assignment": prepared,
            "assignment_name": "observed_m4" if weighted else "unweighted",
            "m2_assignment": prepared["m2_actual"],
            "m2_assignment_name": "observed_nvc" if rho > 0 else "inactive",
        }

    return [
        arm(M1_MODEL_ID, "k1_m1", 0.0, False),
        arm(M2_MODEL_ID, "k1_m2_clv_scaled_value_basis", cfg.rho, False),
        arm(M4_MODEL_ID, "k1_m4_personalized_positive_weight", 0.0, True),
        arm(M5_MODEL_ID, "k1_m5_value_basis_plus_m4", cfg.rho, True),
    ]


def _build_model(prepared: dict, cfg: M5ValueBasisK1Config, spec: dict):
    data = prepared["data"]
    assignment = spec["m2_assignment"]
    v3.set_seed(cfg.seed)
    return M5NConditionedValueBasisLightGCN(
        n_users=data["n_users"],
        n_items=data["n_items"],
        user_q_n=assignment["q_n"],
        user_q_v=assignment["q_v"],
        user_q_c=assignment["q_c"],
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
        economic_propagation=False,
    ).to(v3.DEVICE)


def band_error_pairs(
    users: np.ndarray,
    topk: np.ndarray,
    truth: dict,
    item_bin: np.ndarray,
    *,
    k: int = 10,
) -> dict[str, int]:
    """Count (missed truth, Top-k false positive) pairs by price-band match."""

    same = cross = 0
    item_bin = np.asarray(item_bin)
    for row, user in enumerate(np.asarray(users, dtype=np.int64)):
        ranked = np.asarray(topk[row, :k], dtype=np.int64)
        truth_items = np.asarray(truth[int(user)], dtype=np.int64)
        missed = truth_items[~np.isin(truth_items, ranked)]
        false_positive = ranked[~np.isin(ranked, truth_items)]
        if not len(missed) or not len(false_positive):
            continue
        equal = item_bin[missed][:, None] == item_bin[false_positive][None, :]
        same += int(equal.sum())
        cross += int(equal.size - equal.sum())
    return {"same_band_error_pairs": same, "cross_band_error_pairs": cross}


def screening_reading(
    metric_rows: dict[str, dict],
    *,
    top10_change_shares: dict[str, float],
    economic_score_ratios: dict[str, float],
    band_pairs: dict[str, dict[str, int]],
) -> dict:
    m1, m2, m4, m5 = (metric_rows[model_id] for model_id in MODEL_IDS)

    def beats(model: dict, reference: dict) -> bool:
        return all(model[metric] > reference[metric] for metric in ECONOMIC_METRICS)

    def geomean_ratio(model: dict, reference: dict) -> float:
        ratios = [model[metric] / reference[metric] for metric in ACCURACY_METRICS]
        return float(math.exp(np.log(ratios).mean()))

    reported = TOP10_ACCURACY_METRICS + ECONOMIC_METRICS
    evaluable = top10_change_shares["m5_vs_m4"] > 0.0
    m5_beats_m4 = beats(m5, m4)
    m5_beats_m1 = beats(m5, m1)
    accuracy_guard = all(
        m5[metric] >= ACCURACY_GUARD * m1[metric] for metric in TOP10_ACCURACY_METRICS
    )
    passed = bool(evaluable and m5_beats_m4 and m5_beats_m1 and accuracy_guard)
    if not evaluable:
        classification = "not_evaluable_no_top10_change"
    elif passed:
        classification = "directional_pass"
    else:
        classification = "directional_nonpass"
    return {
        "classification": classification,
        "directional_screen_pass": passed,
        "evaluable": bool(evaluable),
        "m5_beats_m4_on_both_economic_metrics": m5_beats_m4,
        "m5_beats_m1_on_both_economic_metrics": m5_beats_m1,
        "m5_accuracy_guard_vs_m1": bool(accuracy_guard),
        "top10_set_changed_user_share": dict(top10_change_shares),
        "economic_score_std_ratio_to_id": dict(economic_score_ratios),
        "interaction_m5_minus_m4_minus_m2_minus_m1": {
            metric: float((m5[metric] - m4[metric]) - (m2[metric] - m1[metric]))
            for metric in reported
        },
        "deltas_m2_minus_m1": {metric: float(m2[metric] - m1[metric]) for metric in reported},
        "deltas_m4_minus_m1": {metric: float(m4[metric] - m1[metric]) for metric in reported},
        "deltas_m5_minus_m4": {metric: float(m5[metric] - m4[metric]) for metric in reported},
        "deltas_m5_minus_m1": {metric: float(m5[metric] - m1[metric]) for metric in reported},
        "accuracy_geomean_ratio": {
            "m2_vs_m1": geomean_ratio(m2, m1),
            "m4_vs_m1": geomean_ratio(m4, m1),
            "m5_vs_m4": geomean_ratio(m5, m4),
            "m5_vs_m1": geomean_ratio(m5, m1),
        },
        "price_band_error_pairs": band_pairs,
        "clv_attribution_tested": False,
        "next_if_pass": "train the degree-matched CLV assignment shuffle before any attribution claim",
        "next_if_nonpass": "stop this candidate without tuning rho, bandwidth or gate on this interval",
        "next_if_not_evaluable": "report that even rho=0.25 does not change Top-10 lists",
        "statistical_note": "one historical development seed; no significance or generalization claim",
    }


def run_value_basis_k1_screen(cfg: M5ValueBasisK1Config | None = None) -> pd.DataFrame:
    cfg = validate_config(cfg or configure_value_basis_k1_screen())
    summary = preflight_summary(cfg)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    prepared = _prepare(cfg)

    arms: dict[str, dict] = {}
    models: dict[str, object] = {}
    for spec in arm_specifications(prepared, cfg):
        print(
            f"\n===== {spec['model_id']} | seed {cfg.seed} | K={cfg.negative_count} | "
            f"fixed {cfg.epochs} epochs ====="
        )
        with patch.object(legacy, "_build_model", _build_model):
            arm, model = legacy._run_arm(prepared, cfg, spec)
        arms[spec["model_id"]] = arm
        models[spec["model_id"]] = model

    metric_rows = {model_id: arms[model_id]["metrics"] for model_id in MODEL_IDS}
    rows = []
    for model_id in MODEL_IDS:
        arm = arms[model_id]
        rows.append(
            {
                "model_id": model_id,
                "role": arm["role"],
                "seed": arm["seed"],
                "split": arm["split"],
                "final_epoch": arm["final_epoch"],
                "rho": arm["rho"],
                "positive_weight_lambda": arm["positive_weight_lambda"],
                "m4_assignment": arm["clv_assignment"],
                **arm["diagnostics"],
                **arm["training"].get("final_diagnostics", {}),
                **arm["metrics"],
            }
        )
    frame = pd.DataFrame(rows)
    comparison = report_helpers._metric_comparison(
        metric_rows, references=(M1_MODEL_ID, M4_MODEL_ID)
    )

    topk = {
        model_id: report_helpers._masked_topk(
            models[model_id], prepared, max_k=cfg.diagnostic_max_k
        )
        for model_id in MODEL_IDS
    }
    users = topk[M1_MODEL_ID][0]
    if not all(np.array_equal(users, topk[model_id][0]) for model_id in MODEL_IDS):
        raise RuntimeError("arm별 평가 사용자 순서가 다릅니다")
    segments = prepared["cache"].seg
    overlap = pd.concat(
        [
            report_helpers.topk_overlap_summary(
                topk[reference][1], topk[model_id][1], segments
            ).assign(reference=reference, model_id=model_id)
            for reference, model_id in (
                (M1_MODEL_ID, M2_MODEL_ID),
                (M1_MODEL_ID, M4_MODEL_ID),
                (M4_MODEL_ID, M5_MODEL_ID),
                (M1_MODEL_ID, M5_MODEL_ID),
            )
        ],
        ignore_index=True,
    )
    overall = overlap[overlap.group.eq("전체")].set_index(["reference", "model_id"])
    change_shares = {
        "m2_vs_m1": float(overall.at[(M1_MODEL_ID, M2_MODEL_ID), "top10_set_changed_user_share"]),
        "m4_vs_m1": float(overall.at[(M1_MODEL_ID, M4_MODEL_ID), "top10_set_changed_user_share"]),
        "m5_vs_m4": float(overall.at[(M4_MODEL_ID, M5_MODEL_ID), "top10_set_changed_user_share"]),
        "m5_vs_m1": float(overall.at[(M1_MODEL_ID, M5_MODEL_ID), "top10_set_changed_user_share"]),
    }
    band_pairs = {
        model_id: band_error_pairs(
            users, topk[model_id][1], prepared["cache"].gt, prepared["item_bin"]
        )
        for model_id in MODEL_IDS
    }
    score_frame = pd.DataFrame(
        [
            legacy._score_diagnostics(
                models[model_id], users, topk[model_id][1], model_id=model_id
            )
            for model_id in (M2_MODEL_ID, M5_MODEL_ID)
        ]
    )
    score_ratios = {
        model_id: float(
            score_frame.set_index("model_id").at[model_id, "economic_score_std_ratio_to_id"]
        )
        for model_id in (M2_MODEL_ID, M5_MODEL_ID)
    }
    reading = screening_reading(
        metric_rows,
        top10_change_shares=change_shares,
        economic_score_ratios=score_ratios,
        band_pairs=band_pairs,
    )

    out = Path(cfg.out_dir)
    stem = f"m5_value_basis_k1_{prepared['config_hash']}"
    paths = {
        "absolute_csv": out / f"{stem}.csv",
        "comparison_csv": out / f"{stem}_comparison.csv",
        "top10_overlap_csv": out / f"{stem}_top10_overlap.csv",
        "score_diagnostics_csv": out / f"{stem}_score_diagnostics.csv",
        "json": out / f"{stem}.json",
    }
    legacy.test10._atomic_csv(paths["absolute_csv"], frame)
    legacy.test10._atomic_csv(paths["comparison_csv"], comparison)
    legacy.test10._atomic_csv(paths["top10_overlap_csv"], overlap)
    legacy.test10._atomic_csv(paths["score_diagnostics_csv"], score_frame)
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
            "top10_overlap_rows": overlap.to_dict("records"),
            "score_diagnostic_rows": score_frame.to_dict("records"),
            "screening_reading": reading,
            "arms": arms,
            "result_paths": {key: str(value) for key, value in paths.items()},
        },
    )
    frame.attrs.update(
        comparison=comparison,
        top10_overlap=overlap,
        score_diagnostics=score_frame,
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
            preflight_summary(configure_value_basis_k1_screen()),
            ensure_ascii=False,
            indent=2,
        )
    )
