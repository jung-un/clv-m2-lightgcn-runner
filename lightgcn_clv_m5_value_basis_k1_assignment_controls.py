"""CLV assignment controls for the single-negative value-basis M2 and M5.

The 2026-09-16 four-arm screen passed its directional rule: with the original
single-negative BPR, the q_C-scaled q_V value basis changed 18.6% of Top-10
lists and raised the price/purchase-amount weighted hit@10 by 2.32% over M1.
That gain is not yet attributable to the customer-CLV assignment, because a
value block built from any plausible price signal could do the same.

This run trains the actual and the degree-matched permuted assignment of the
same model in one execution:

* ``m2_actual`` / ``m2_shuffled``  - unweighted BPR, value basis only;
* ``m5_actual`` / ``m5_shuffled``  - the same basis plus the M4 positive weight.

Only the M2 value block is permuted. The M4 weight keeps the observed q_C in
every arm, so the comparison isolates the assignment used by the representation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import lightgcn_clv_gradient_isolated_economic_interaction as report_helpers
import lightgcn_clv_m5_clv_scaled_value_basis_k1_screen as screen
import lightgcn_clv_m5_economic_positive_weight as legacy
import lightgcn_clv_m5_n_conditioned_value_basis_controls as controls
import lightgcn_clv_v3 as v3


CODE_VERSION = "m5-value-basis-k1-clv-assignment-controls-v1"
M2_ACTUAL_MODEL_ID = "m2_clv_scaled_value_basis_bpr_k1_actual"
M2_SHUFFLED_MODEL_ID = "m2_clv_scaled_value_basis_bpr_k1_degree_matched_shuffle"
M5_ACTUAL_MODEL_ID = "m5_clv_scaled_value_basis_positive_weight_bpr_k1_actual"
M5_SHUFFLED_MODEL_ID = (
    "m5_clv_scaled_value_basis_positive_weight_bpr_k1_degree_matched_shuffle"
)
MODEL_IDS = (
    M2_ACTUAL_MODEL_ID,
    M2_SHUFFLED_MODEL_ID,
    M5_ACTUAL_MODEL_ID,
    M5_SHUFFLED_MODEL_ID,
)
ACTUAL_TO_SHUFFLED = {
    M2_ACTUAL_MODEL_ID: M2_SHUFFLED_MODEL_ID,
    M5_ACTUAL_MODEL_ID: M5_SHUFFLED_MODEL_ID,
}
ECONOMIC_METRICS = screen.ECONOMIC_METRICS
TOP10_ACCURACY_METRICS = screen.TOP10_ACCURACY_METRICS


@dataclass(frozen=True)
class M5ValueBasisK1ControlsConfig(screen.M5ValueBasisK1Config):
    pass


def configure_assignment_controls(**overrides) -> M5ValueBasisK1ControlsConfig:
    data_root = v3.default_out_dir("dunnhumby")
    defaults = {
        "out_dir": f"{data_root}_m5_value_basis_k1_assignment_controls_v1",
        "baseline_result_dir": f"{data_root}_m2_repeatshare_historical_backtest_v1",
    }
    return validate_config(M5ValueBasisK1ControlsConfig(**(defaults | overrides)))


def validate_config(cfg: M5ValueBasisK1ControlsConfig) -> M5ValueBasisK1ControlsConfig:
    screen.validate_config(cfg)
    return cfg


def preflight_summary(cfg: M5ValueBasisK1ControlsConfig) -> dict:
    cfg = validate_config(cfg)
    return {
        "code_version": CODE_VERSION,
        "dataset": cfg.dataset,
        "seed": cfg.seed,
        "split": "historical_development_days_684_690",
        "trained_models": list(MODEL_IDS),
        "reused_models": [],
        "research_question": (
            "Does the single-negative value-basis gain come from the correct "
            "customer-CLV assignment rather than from any price signal of the "
            "same shape?"
        ),
        "control": {
            "permutation": (
                "q_N, q_V, q_C and the CLV validity flag move together to another "
                "user inside the same binary user-degree decile"
            ),
            "degree_bins": cfg.shuffle_degree_bins,
            "shuffle_seed": cfg.shuffle_seed,
            "m4_weight_assignment": "observed q_C in every arm",
            "permuted_component": "M2 value block only",
        },
        "m2": {
            "user_block": "sqrt(rho) * q_C * g_N(q_N) * RBF(q_V)",
            "rho": cfg.rho,
            "economic_graph_propagation": False,
        },
        "loss": {
            "bpr": "mean softplus(s(u,j) - s(u,i+)) + batch layer-0 L2",
            "negative_count": cfg.negative_count,
        },
        "reading_rule": {
            "evaluable": "actual and shuffled differ in at least one user's Top-10 set",
            "m2_assignment_signal": "m2_actual > m2_shuffled on both economic metrics",
            "m5_assignment_signal": "m5_actual > m5_shuffled on both economic metrics",
            "clv_assignment_supported": "evaluable and both signals",
            "partial": "evaluable and exactly one signal",
            "reported_not_gated": (
                "accuracy, @20/@50, CLV segments, exposure, price alignment, "
                "price-band error pairs"
            ),
        },
        "fixed": {
            "new_item_task": True,
            "min_item_interactions": 1,
            "graph": "binary",
            "epochs": cfg.epochs,
            "final_test_constructed": False,
            "holdout_constructed": False,
            "one_training_loop_and_optimizer_per_arm": True,
        },
        "statistical_note": (
            "one historical development seed; no significance, stability or "
            "generalization claim"
        ),
        "out_dir": cfg.out_dir,
    }


def _config_hash(cfg: M5ValueBasisK1ControlsConfig, input_hash: str, revision: str) -> str:
    payload = {
        "code_version": CODE_VERSION,
        "config": asdict(cfg),
        "input_hash": input_hash,
        "source_revision": revision,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()[:12]


def _prepare(cfg: M5ValueBasisK1ControlsConfig) -> dict:
    prepared = screen._prepare(cfg)
    prepared["m2_shuffle"] = controls.degree_matched_nv_shuffle(
        prepared, seed=cfg.shuffle_seed, degree_bins=cfg.shuffle_degree_bins
    )
    source = prepared["m2_shuffle"]["source_user"]
    diagnostics = {
        "n_users": int(len(source)),
        "valid_users": int(np.asarray(prepared["clv_valid"]).sum()),
        "moved_user_share": float(np.mean(source != np.arange(len(source)))),
        "same_degree_bin": bool(
            np.all(prepared["degree_bin"][source] == prepared["degree_bin"])
        ),
    }
    for key in ("q_n", "q_v", "q_c", "clv_valid"):
        diagnostics[f"{key}_multiset_preserved"] = bool(
            np.array_equal(
                np.sort(np.asarray(prepared["m2_shuffle"][key])),
                np.sort(np.asarray(prepared[key])),
            )
        )
    if not all(value for key, value in diagnostics.items() if key.endswith(("preserved", "bin"))):
        raise RuntimeError("CLV 순열 불변식이 성립하지 않습니다")
    prepared["control_diagnostics"] = diagnostics
    prepared["config_hash"] = _config_hash(
        cfg, prepared["input_hash"], prepared["revision"]
    )
    return prepared


def arm_specifications(
    prepared: dict, cfg: M5ValueBasisK1ControlsConfig
) -> list[dict]:
    def arm(model_id: str, role: str, weighted: bool, shuffled: bool) -> dict:
        return {
            "model_id": model_id,
            "role": role,
            "rho": cfg.rho,
            "weighted": weighted,
            "assignment": prepared,
            "assignment_name": "observed_m4" if weighted else "unweighted",
            "m2_assignment": (
                prepared["m2_shuffle"] if shuffled else prepared["m2_actual"]
            ),
            "m2_assignment_name": (
                "degree_matched_clv_shuffle" if shuffled else "observed_clv"
            ),
        }

    return [
        arm(M2_ACTUAL_MODEL_ID, "m2_observed_clv", False, False),
        arm(M2_SHUFFLED_MODEL_ID, "m2_assignment_control", False, True),
        arm(M5_ACTUAL_MODEL_ID, "m5_observed_clv", True, False),
        arm(M5_SHUFFLED_MODEL_ID, "m5_assignment_control", True, True),
    ]


def attribution_reading(
    metric_rows: dict[str, dict],
    *,
    top10_change_shares: dict[str, float],
    economic_score_ratios: dict[str, float],
    band_pairs: dict[str, dict[str, int]],
) -> dict:
    def beats(actual: str, shuffled: str) -> bool:
        return all(
            metric_rows[actual][metric] > metric_rows[shuffled][metric]
            for metric in ECONOMIC_METRICS
        )

    def deltas(actual: str, shuffled: str) -> dict[str, float]:
        return {
            metric: float(metric_rows[actual][metric] - metric_rows[shuffled][metric])
            for metric in TOP10_ACCURACY_METRICS + ECONOMIC_METRICS
        }

    evaluable = all(share > 0.0 for share in top10_change_shares.values())
    m2_signal = beats(M2_ACTUAL_MODEL_ID, M2_SHUFFLED_MODEL_ID)
    m5_signal = beats(M5_ACTUAL_MODEL_ID, M5_SHUFFLED_MODEL_ID)
    if not evaluable:
        classification = "not_evaluable_no_top10_change"
    elif m2_signal and m5_signal:
        classification = "clv_assignment_supported"
    elif m2_signal or m5_signal:
        classification = "partial_clv_assignment_signal"
    else:
        classification = "no_clv_assignment_signal"
    return {
        "classification": classification,
        "clv_assignment_supported": bool(evaluable and m2_signal and m5_signal),
        "evaluable": bool(evaluable),
        "m2_assignment_signal": bool(m2_signal),
        "m5_assignment_signal": bool(m5_signal),
        "top10_set_changed_user_share": dict(top10_change_shares),
        "economic_score_std_ratio_to_id": dict(economic_score_ratios),
        "deltas_m2_actual_minus_shuffled": deltas(
            M2_ACTUAL_MODEL_ID, M2_SHUFFLED_MODEL_ID
        ),
        "deltas_m5_actual_minus_shuffled": deltas(
            M5_ACTUAL_MODEL_ID, M5_SHUFFLED_MODEL_ID
        ),
        "price_band_error_pairs": band_pairs,
        "next_if_supported": (
            "freeze this specification and run the ten-seed test, then H&M"
        ),
        "next_if_not_supported": (
            "report the gain as a price-signal effect that the CLV assignment "
            "does not explain, and do not claim CLV attribution"
        ),
        "statistical_note": (
            "one historical development seed; no significance, stability or "
            "generalization claim"
        ),
    }


def run_assignment_controls(
    cfg: M5ValueBasisK1ControlsConfig | None = None,
) -> pd.DataFrame:
    cfg = validate_config(cfg or configure_assignment_controls())
    summary = preflight_summary(cfg)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    prepared = _prepare(cfg)
    print("\n[순열 불변식]")
    print(json.dumps(prepared["control_diagnostics"], ensure_ascii=False, indent=2))

    arms: dict[str, dict] = {}
    models: dict[str, object] = {}
    for spec in arm_specifications(prepared, cfg):
        print(
            f"\n===== {spec['model_id']} | seed {cfg.seed} | K={cfg.negative_count} | "
            f"fixed {cfg.epochs} epochs ====="
        )
        with patch.object(legacy, "_build_model", screen._build_model):
            arm, model = legacy._run_arm(prepared, cfg, spec)
        arm["m2_assignment"] = spec["m2_assignment_name"]
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
                "rho": arms[model_id]["rho"],
                "positive_weight_lambda": arms[model_id]["positive_weight_lambda"],
                "m2_assignment": arms[model_id]["m2_assignment"],
                "m4_assignment": arms[model_id]["clv_assignment"],
                **arms[model_id]["diagnostics"],
                **arms[model_id]["training"].get("final_diagnostics", {}),
                **arms[model_id]["metrics"],
            }
            for model_id in MODEL_IDS
        ]
    )
    comparison = report_helpers._metric_comparison(
        metric_rows, references=(M2_SHUFFLED_MODEL_ID, M5_SHUFFLED_MODEL_ID)
    )

    topk = {
        model_id: report_helpers._masked_topk(
            models[model_id], prepared, max_k=cfg.diagnostic_max_k
        )
        for model_id in MODEL_IDS
    }
    users = topk[M2_ACTUAL_MODEL_ID][0]
    if not all(np.array_equal(users, topk[model_id][0]) for model_id in MODEL_IDS):
        raise RuntimeError("arm별 평가 사용자 순서가 다릅니다")
    segments = prepared["cache"].seg
    overlap = pd.concat(
        [
            report_helpers.topk_overlap_summary(
                topk[shuffled][1], topk[actual][1], segments
            ).assign(reference=shuffled, model_id=actual)
            for actual, shuffled in ACTUAL_TO_SHUFFLED.items()
        ],
        ignore_index=True,
    )
    overall = overlap[overlap.group.eq("전체")].set_index("model_id")
    change_shares = {
        f"{actual}_vs_shuffle": float(
            overall.at[actual, "top10_set_changed_user_share"]
        )
        for actual in ACTUAL_TO_SHUFFLED
    }
    band_pairs = {
        model_id: screen.band_error_pairs(
            users, topk[model_id][1], prepared["cache"].gt, prepared["item_bin"]
        )
        for model_id in MODEL_IDS
    }
    score_frame = pd.DataFrame(
        [
            legacy._score_diagnostics(
                models[model_id], users, topk[model_id][1], model_id=model_id
            )
            for model_id in MODEL_IDS
        ]
    )
    score_ratios = {
        model_id: float(
            score_frame.set_index("model_id").at[
                model_id, "economic_score_std_ratio_to_id"
            ]
        )
        for model_id in MODEL_IDS
    }
    reading = attribution_reading(
        metric_rows,
        top10_change_shares=change_shares,
        economic_score_ratios=score_ratios,
        band_pairs=band_pairs,
    )

    out = Path(cfg.out_dir)
    stem = f"m5_value_basis_k1_controls_{prepared['config_hash']}"
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
            "control_diagnostics": prepared["control_diagnostics"],
            "absolute_rows": frame.to_dict("records"),
            "comparison_rows": comparison.to_dict("records"),
            "top10_overlap_rows": overlap.to_dict("records"),
            "score_diagnostic_rows": score_frame.to_dict("records"),
            "attribution_reading": reading,
            "arms": arms,
            "result_paths": {key: str(value) for key, value in paths.items()},
        },
    )
    frame.attrs.update(
        comparison=comparison,
        top10_overlap=overlap,
        score_diagnostics=score_frame,
        decision=reading,
        control_diagnostics=prepared["control_diagnostics"],
        result_paths={key: str(value) for key, value in paths.items()},
    )
    print("\n1) 판독")
    print(json.dumps(reading, ensure_ascii=False, indent=2))
    print("\n결과 파일:", frame.attrs["result_paths"])
    return frame


if __name__ == "__main__":
    print(
        json.dumps(
            preflight_summary(configure_assignment_controls()),
            ensure_ascii=False,
            indent=2,
        )
    )
