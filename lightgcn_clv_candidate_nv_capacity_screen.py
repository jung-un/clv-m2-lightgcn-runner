"""Matched-capacity M1/M2/M3 screen for the candidate-specific N/V design.

This runner tests the professor's underfitting hypothesis without changing the
M2 representation or M3 edge formula.  At each ID width (64, 128, 256), M1,
M2 and M3 use the same width and training budget.  The historical development
split is used; final test and holdout are never constructed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from clv_candidate_nv_fit_model import CandidateNVFitLightGCN
import lightgcn_clv_candidate_nv_fit_factorial as base
import lightgcn_clv_gradient_isolated_economic_interaction as report_helpers
import lightgcn_clv_m5_economic_positive_weight as legacy
import lightgcn_clv_v3 as v3


CODE_VERSION = "candidate-nv-m1-m2-m3-capacity-development-screen-v1"
CAPACITY_DIMENSIONS = (64, 128, 256)
AXES = ("m1", "m2", "m3")
TOP10_METRICS = (
    "recall@10",
    "ndcg@10",
    "price_purchase_amount_weighted_hit@10",
    "vndcg@10",
)


def model_id(axis: str, dimension: int) -> str:
    if axis not in AXES or dimension not in CAPACITY_DIMENSIONS:
        raise ValueError("axis나 dimension이 사전 설정과 다릅니다")
    return f"{axis}_candidate_nv_capacity_d{dimension}_bpr_k1"


MODEL_IDS = tuple(
    model_id(axis, dimension)
    for dimension in CAPACITY_DIMENSIONS
    for axis in AXES
)


@dataclass(frozen=True)
class CapacityScreenConfig(base.CandidateNVFitConfig):
    capacity_dimensions: tuple[int, ...] = CAPACITY_DIMENSIONS
    margin_sample_size: int = 100_000


def configure_capacity_screen(**overrides) -> CapacityScreenConfig:
    root = v3.default_out_dir("dunnhumby")
    defaults = {
        "out_dir": f"{root}_candidate_nv_m1_m2_m3_capacity_development_screen_v1",
        "baseline_result_dir": f"{root}_m2_repeatshare_historical_backtest_v1",
    }
    return validate_config(CapacityScreenConfig(**(defaults | overrides)))


def validate_config(cfg: CapacityScreenConfig) -> CapacityScreenConfig:
    fixed = {
        "dataset": "dunnhumby",
        "seed": 42,
        "time_cutoff": 690,
        "evaluation_days": 7,
        "epochs": 100,
        "id_dim": 64,
        "economic_dim": 2,
        "rho": 0.15,
        "beta_m3": 0.15,
        "n_layers": 2,
        "negative_count": 1,
        "input_days": 365,
        "diagnostic_max_k": 50,
        "capacity_dimensions": CAPACITY_DIMENSIONS,
        "margin_sample_size": 100_000,
    }
    for key, expected in fixed.items():
        if getattr(cfg, key) != expected:
            raise ValueError(f"용량 screen은 {key}={expected!r}이어야 합니다")
    if cfg.batch_size <= 0 or cfg.lr <= 0.0 or cfg.pref_reg < 0.0:
        raise ValueError("학습 설정이 잘못됐습니다")
    if not cfg.out_dir or not cfg.baseline_result_dir:
        raise ValueError("out_dir와 baseline_result_dir가 필요합니다")
    return cfg


def preflight_summary(cfg: CapacityScreenConfig) -> dict:
    cfg = validate_config(cfg)
    return {
        "code_version": CODE_VERSION,
        "dataset": cfg.dataset,
        "seed": cfg.seed,
        "split": "historical_development_days_684_690",
        "trained_models": list(MODEL_IDS),
        "reused_models": [],
        "research_question": (
            "Were prior M2/M3 losses caused by insufficient LightGCN ID "
            "capacity rather than by the candidate-specific CLV signal itself?"
        ),
        "capacity_test": {
            "id_dimensions": list(cfg.capacity_dimensions),
            "matched_within_dimension": True,
            "m2_formula_changed": False,
            "m3_formula_changed": False,
            "rho": cfg.rho,
            "beta_m3": cfg.beta_m3,
            "layers": cfg.n_layers,
            "epochs": cfg.epochs,
        },
        "arms": {
            model_id(axis, dimension): {
                "axis": axis,
                "id_dimension": dimension,
                "m2_expression": axis == "m2",
                "m3_edge_weight": axis == "m3",
            }
            for dimension in cfg.capacity_dimensions
            for axis in AXES
        },
        "reading_rule": {
            "primary_comparison": (
                "compare M2 and M3 only with the M1 of the same ID dimension"
            ),
            "underfitting_signal": (
                "an axis that does not beat matched M1 at d64 newly beats matched "
                "M1 at d128 or d256 on all four Top-10 metrics"
            ),
            "not_underfitting_proof": (
                "a nonpass does not prove all CLV models are impossible; it only "
                "fails to support this capacity explanation"
            ),
            "selection": "do not tune rho, beta, layers or epochs in this run",
        },
        "fixed": {
            "new_item_task": True,
            "train_pairs_excluded_from_evaluation": True,
            "min_item_interactions": 1,
            "graph": "binary except the frozen M3 edge coefficients",
            "negative_sampling": "one uniform unseen item",
            "validation_or_epoch_selection": False,
            "final_test_constructed": False,
            "holdout_constructed": False,
            "one_training_loop_and_optimizer_per_arm": True,
            "external_reranking": False,
        },
        "statistical_note": (
            "one exposed historical development seed; no significance, stability, "
            "generalization or CLV-attribution claim"
        ),
        "out_dir": cfg.out_dir,
    }


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def _config_hash(cfg: CapacityScreenConfig, input_hash: str, revision: str) -> str:
    payload = {
        "code_version": CODE_VERSION,
        "config": asdict(cfg),
        "input_hash": input_hash,
        "source_revision": revision,
    }
    return hashlib.sha256(_canonical(payload).encode()).hexdigest()[:12]


def arm_specifications(prepared: dict, cfg: CapacityScreenConfig) -> list[dict]:
    specs = []
    for dimension in cfg.capacity_dimensions:
        for axis in AXES:
            specs.append(
                {
                    "model_id": model_id(axis, dimension),
                    "role": f"capacity_{axis}_d{dimension}",
                    "id_dim": dimension,
                    "rho": cfg.rho if axis == "m2" else 0.0,
                    "m2": axis == "m2",
                    "m3": axis == "m3",
                    "weighted": False,
                    "assignment": prepared,
                    "assignment_name": "unweighted",
                }
            )
    return specs


def _build_model(prepared: dict, cfg: CapacityScreenConfig, spec: dict):
    data = prepared["data"]
    v3.set_seed(cfg.seed)
    return CandidateNVFitLightGCN(
        n_users=data["n_users"],
        n_items=data["n_items"],
        user_q_n=prepared["q_n"],
        user_q_v=prepared["q_v"],
        user_q_c=prepared["q_c"],
        user_clv_valid=prepared["clv_valid"],
        item_buyer_q_n=prepared["item_buyer_q_n"],
        item_amount_percentile=prepared["item_amount_percentile"],
        item_context_valid=prepared["item_context_valid"],
        adj=prepared["m3_adj"] if spec["m3"] else data["adj"],
        id_dim=spec["id_dim"],
        rho=spec["rho"],
        n_layers=cfg.n_layers,
        pref_reg=cfg.pref_reg,
    ).to(v3.DEVICE)


@torch.no_grad()
def sampled_training_margin(
    model,
    prepared: dict,
    *,
    sample_size: int,
) -> dict:
    """Measure a fixed train-pair margin without changing model selection."""

    data = prepared["data"]
    rng = np.random.default_rng(20260921)
    count = min(sample_size, len(data["tr_u"]))
    selected = rng.choice(len(data["tr_u"]), size=count, replace=False)
    users_np = data["tr_u"][selected]
    positives_np = data["tr_i"][selected]
    negatives_np = legacy.m4_helpers.sample_uniform_negative_matrix(
        users_np,
        positives_np,
        data["n_items"],
        data["pos_key"],
        rng,
        k=1,
    )[:, 0]
    users = torch.as_tensor(users_np, dtype=torch.long, device=v3.DEVICE)
    positives = torch.as_tensor(
        positives_np, dtype=torch.long, device=v3.DEVICE
    )
    negatives = torch.as_tensor(
        negatives_np, dtype=torch.long, device=v3.DEVICE
    )
    user_z, item_z = model.propagated_embeddings()
    margins = (
        (user_z[users] * item_z[positives]).sum(dim=1)
        - (user_z[users] * item_z[negatives]).sum(dim=1)
    ).detach().cpu().numpy()
    return {
        "train_margin_sample_size": int(count),
        "train_margin_mean": float(margins.mean()),
        "train_margin_std": float(margins.std()),
        "train_margin_median": float(np.median(margins)),
        "train_margin_p10": float(np.quantile(margins, 0.10)),
        "train_margin_p90": float(np.quantile(margins, 0.90)),
        "train_margin_positive_share": float(np.mean(margins > 0.0)),
    }


def capacity_reading(metric_rows: dict[str, dict]) -> dict:
    rows = {}
    for dimension in CAPACITY_DIMENSIONS:
        baseline = metric_rows[model_id("m1", dimension)]
        rows[str(dimension)] = {}
        for axis in ("m2", "m3"):
            candidate = metric_rows[model_id(axis, dimension)]
            deltas = {
                metric: float(candidate[metric] - baseline[metric])
                for metric in TOP10_METRICS
            }
            rows[str(dimension)][axis] = {
                "beats_matched_m1_on_all_four_top10_metrics": all(
                    value > 0.0 for value in deltas.values()
                ),
                "deltas_vs_matched_m1": deltas,
            }
    newly_positive = {
        axis: [
            dimension
            for dimension in CAPACITY_DIMENSIONS[1:]
            if rows[str(dimension)][axis][
                "beats_matched_m1_on_all_four_top10_metrics"
            ]
            and not rows["64"][axis][
                "beats_matched_m1_on_all_four_top10_metrics"
            ]
        ]
        for axis in ("m2", "m3")
    }
    return {
        "per_capacity": rows,
        "newly_positive_larger_capacity": newly_positive,
        "capacity_underfitting_signal": any(newly_positive.values()),
        "clv_attribution_tested": False,
        "success_or_failure_final_decision_permitted": False,
        "statistical_note": (
            "one exposed historical development seed; inspect all absolute, "
            "comparison, training and mechanism outputs before interpretation"
        ),
    }


def run_capacity_screen(
    cfg: CapacityScreenConfig | None = None,
) -> pd.DataFrame:
    cfg = validate_config(cfg or configure_capacity_screen())
    summary = preflight_summary(cfg)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    prepared = base._prepare(cfg)
    prepared["config_hash"] = _config_hash(
        cfg, prepared["input_hash"], prepared["revision"]
    )

    arms: dict[str, dict] = {}
    models: dict[str, object] = {}
    specs = arm_specifications(prepared, cfg)
    for spec in specs:
        print(
            f"\n===== {spec['model_id']} | seed {cfg.seed} | "
            f"ID d={spec['id_dim']} | K=1 | {cfg.epochs} epochs ====="
        )
        with (
            patch.object(legacy, "_build_model", _build_model),
            patch.object(legacy, "_train_arm", base._train_arm),
        ):
            arm, model = legacy._run_arm(prepared, cfg, spec)
        arms[spec["model_id"]] = arm
        models[spec["model_id"]] = model

    metric_rows = {key: arms[key]["metrics"] for key in MODEL_IDS}
    rows = []
    margin_rows = []
    for spec in specs:
        arm = arms[spec["model_id"]]
        margin = sampled_training_margin(
            models[spec["model_id"]],
            prepared,
            sample_size=cfg.margin_sample_size,
        )
        margin_rows.append({"model_id": spec["model_id"], **margin})
        rows.append(
            {
                "model_id": spec["model_id"],
                "axis": spec["role"].split("_")[1],
                "id_dim": spec["id_dim"],
                "trainable_parameter_count": int(
                    sum(
                        parameter.numel()
                        for parameter in models[spec["model_id"]].parameters()
                        if parameter.requires_grad
                    )
                ),
                "m2_expression": spec["m2"],
                "m3_edge_weight": spec["m3"],
                "final_epoch": arm["final_epoch"],
                **arm["training"].get("final_diagnostics", {}),
                **margin,
                **arm["metrics"],
            }
        )
    frame = pd.DataFrame(rows)

    comparison_rows = []
    overlap_rows = []
    topk = {
        key: report_helpers._masked_topk(
            models[key], prepared, max_k=cfg.diagnostic_max_k
        )
        for key in MODEL_IDS
    }
    for dimension in cfg.capacity_dimensions:
        baseline_id = model_id("m1", dimension)
        baseline = metric_rows[baseline_id]
        for axis in ("m2", "m3"):
            candidate_id = model_id(axis, dimension)
            for metric, value in metric_rows[candidate_id].items():
                if metric not in baseline or not np.isscalar(value):
                    continue
                comparison_rows.append(
                    {
                        "id_dim": dimension,
                        "reference": baseline_id,
                        "model_id": candidate_id,
                        "metric": metric,
                        "reference_value": float(baseline[metric]),
                        "model_value": float(value),
                        "absolute_delta": float(value - baseline[metric]),
                        "relative_change_pct": (
                            float(100.0 * (value - baseline[metric]) / baseline[metric])
                            if baseline[metric] != 0.0
                            else float("nan")
                        ),
                    }
                )
            overlap_rows.extend(
                report_helpers.topk_overlap_summary(
                    topk[baseline_id][1],
                    topk[candidate_id][1],
                    prepared["cache"].seg,
                )
                .assign(
                    id_dim=dimension,
                    reference=baseline_id,
                    model_id=candidate_id,
                )
                .to_dict("records")
            )
    comparison = pd.DataFrame(comparison_rows)
    overlap = pd.DataFrame(overlap_rows)
    margins = pd.DataFrame(margin_rows)
    training = pd.DataFrame(
        [
            {"model_id": key, **record}
            for key in MODEL_IDS
            for record in arms[key]["training"].get("history", [])
        ]
    )
    reading = capacity_reading(metric_rows)
    mechanism = {
        "candidate_fit": base.candidate_fit_diagnostics(
            prepared, sample_size=cfg.candidate_diagnostic_sample_size
        ),
        "m3": prepared["m3_diagnostics"],
        "item_context": prepared["economic_input_diagnostics"],
    }

    out = Path(cfg.out_dir)
    stem = f"candidate_nv_capacity_{prepared['config_hash']}"
    paths = {
        "absolute_csv": out / f"{stem}.csv",
        "comparison_csv": out / f"{stem}_comparison.csv",
        "training_history_csv": out / f"{stem}_training_history.csv",
        "training_margin_csv": out / f"{stem}_training_margin.csv",
        "top10_overlap_csv": out / f"{stem}_top10_overlap.csv",
        "json": out / f"{stem}.json",
    }
    legacy.test10._atomic_csv(paths["absolute_csv"], frame)
    legacy.test10._atomic_csv(paths["comparison_csv"], comparison)
    legacy.test10._atomic_csv(paths["training_history_csv"], training)
    legacy.test10._atomic_csv(paths["training_margin_csv"], margins)
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
            "training_history_rows": training.to_dict("records"),
            "training_margin_rows": margins.to_dict("records"),
            "top10_overlap_rows": overlap.to_dict("records"),
            "mechanism_diagnostics": mechanism,
            "capacity_reading": reading,
            "arms": arms,
            "result_paths": {key: str(value) for key, value in paths.items()},
        },
    )
    frame.attrs.update(
        comparison=comparison.to_dict("records"),
        training_history=training.to_dict("records"),
        training_margin=margins.to_dict("records"),
        top10_overlap=overlap.to_dict("records"),
        mechanism_diagnostics=mechanism,
        decision=reading,
        result_paths={key: str(value) for key, value in paths.items()},
    )
    print("\n1) M1·M2·M3 용량별 절대지표")
    print(frame)
    print("\n2) 동일 용량 M1 대비")
    print(comparison)
    print("\n3) 용량부족 가설 판독")
    print(json.dumps(reading, ensure_ascii=False, indent=2))
    print("\n결과 파일:", frame.attrs["result_paths"])
    return frame


if __name__ == "__main__":
    print(
        json.dumps(
            preflight_summary(configure_capacity_screen()),
            ensure_ascii=False,
            indent=2,
        )
    )
