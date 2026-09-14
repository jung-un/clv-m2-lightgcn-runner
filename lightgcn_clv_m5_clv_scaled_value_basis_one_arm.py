"""One-arm seed-42 screen for a q_C-scaled value-basis M5.

Only the new M5 is trained. Its result is descriptively compared with the
verified, condition-matched M1 and M4 results recorded in RESEARCH_STATUS.md.
Those references came from different completed runs and are not loaded from
Drive, so a missing historical artifact cannot block this screen.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from clv_m5_n_conditioned_value_basis_model import (
    M5NConditionedValueBasisLightGCN,
)
import lightgcn_clv_axis_specific_test10 as test10
import lightgcn_clv_gated_relation_overall_price as common
import lightgcn_clv_gradient_isolated_economic_interaction as report_helpers
import lightgcn_clv_joint_nv as joint
import lightgcn_clv_m5_economic_positive_weight as legacy
import lightgcn_clv_m5_nv_economic_positive_weight as nv
import lightgcn_clv_moe as moe
import lightgcn_clv_residual as residual
import lightgcn_clv_v3 as v3


CODE_VERSION = "m5-clv-scaled-value-basis-one-arm-development-screen-v1"
MODEL_ID = "m5_clv_scaled_value_basis_personalized_positive_weight_k5"
M1_REFERENCE_ID = "7cd818302fb1"
M4_REFERENCE_ID = "d01883236772"
REFERENCE_METRICS = {
    "m1_multineg_mean_k5": {
        "recall@10": 0.015949,
        "ndcg@10": 0.019283,
        "recall@20": 0.025010,
        "ndcg@20": 0.021400,
        "recall@50": 0.044861,
        "ndcg@50": 0.028012,
        "price_purchase_amount_weighted_hit@10": 0.381488,
        "vndcg@10": 0.010665,
        "coverage@10": 0.003610,
    },
    "m4_personalized_positive_weight_k5_minimal_nv_control": {
        "recall@10": 0.01576683048,
        "ndcg@10": 0.01931665202,
        "recall@20": 0.02484586784,
        "ndcg@20": 0.02131666610,
        "recall@50": 0.04421511714,
        "ndcg@50": 0.02794741202,
        "price_purchase_amount_weighted_hit@10": 0.3863577262,
        "vndcg@10": 0.01086602226,
        "coverage@10": 0.003598874936,
    },
}
REFERENCE_PROVENANCE = {
    "m1_multineg_mean_k5": {
        "result_id": M1_REFERENCE_ID,
        "source": "verified original output recorded in RESEARCH_STATUS.md",
        "precision": "published output rounded to six decimals",
    },
    "m4_personalized_positive_weight_k5_minimal_nv_control": {
        "result_id": M4_REFERENCE_ID,
        "source": "verified original CSV m5_minimal_nv_m4_d01883236772.csv",
        "precision": "original CSV values",
    },
}
CORE_METRICS = (
    "recall@10",
    "ndcg@10",
    "recall@20",
    "ndcg@20",
    "recall@50",
    "ndcg@50",
    "price_purchase_amount_weighted_hit@10",
    "vndcg@10",
    "coverage@10",
)
TOP10_DIRECTION_METRICS = (
    "recall@10",
    "ndcg@10",
    "price_purchase_amount_weighted_hit@10",
    "vndcg@10",
)


@dataclass(frozen=True)
class M5CLVScaledValueBasisConfig:
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
    out_dir: str = ""


def configure_clv_scaled_value_basis_one_arm(
    **overrides,
) -> M5CLVScaledValueBasisConfig:
    defaults = {
        "out_dir": (
            f"{v3.default_out_dir('dunnhumby')}"
            "_m5_clv_scaled_value_basis_one_arm_development_screen_v1"
        )
    }
    return validate_config(M5CLVScaledValueBasisConfig(**(defaults | overrides)))


def validate_config(
    cfg: M5CLVScaledValueBasisConfig,
) -> M5CLVScaledValueBasisConfig:
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
    }
    for key, expected in fixed.items():
        if getattr(cfg, key) != expected:
            raise ValueError(f"q_C×q_V M5 screen은 {key}={expected!r}이어야 합니다")
    if cfg.batch_size <= 0 or cfg.lr <= 0 or cfg.pref_reg < 0 or not cfg.out_dir:
        raise ValueError("학습 설정 또는 출력 경로가 잘못됐습니다")
    return cfg


def preflight_summary(cfg: M5CLVScaledValueBasisConfig) -> dict:
    cfg = validate_config(cfg)
    return {
        "code_version": CODE_VERSION,
        "dataset": cfg.dataset,
        "seed": cfg.seed,
        "split": "historical_development_days_684_690",
        "trained_models": [MODEL_ID],
        "reused_models": [],
        "research_question": (
            "Does a q_C-scaled q_V--item-price representation combined with the "
            "unchanged personalized M4 outperform the prior M1 and M4 directions?"
        ),
        "m2": {
            "q_c": "midrank percentile of raw n_u*v_u",
            "q_c_role": "direct strength of the complete user value-position block",
            "q_v_role": "position in the fixed low/mid/high value basis",
            "separate_q_n_gate": False,
            "user_coordinates": "sqrt(rho)*q_C(u)*b(q_V(u))",
            "item_coordinates": "sqrt(rho)*b(item amount percentile)",
            "basis": "fixed L2-normalized Gaussian RBF at [0,0.5,1]",
            "economic_graph_propagation": True,
            "joint_end_to_end_training": True,
            "external_reranking": False,
        },
        "m4": {
            "formula": "1 + 0.5*q_C*item_amount_percentile*clipped_user_bin_fit",
            "normalization": "divide by the mean over all positive training rows",
            "uniform_negative_count": cfg.negative_count,
            "hard_negative": False,
            "changed_from_prior_m4": False,
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
        "comparison": {
            "mode": "descriptive references from separate completed runs",
            "m1_result_id": M1_REFERENCE_ID,
            "m4_result_id": M4_REFERENCE_ID,
            "missing_reference_file_can_block_run": False,
            "official_factorial_or_attribution_claim": False,
        },
        "reading_rule": {
            "directional_candidate": (
                "new M5 is strictly above both prior M1 and prior M4 on Recall@10, "
                "NDCG@10, weighted hit@10, and weighted NDCG@10"
            ),
            "rank_intervention": (
                "report within-user candidate economic-score dispersion and the "
                "share of Top-10 lists changed versus the trained ID-only score"
            ),
            "clv_attribution_tested": False,
            "next_if_directional_candidate": (
                "train one degree-matched q_C/q_V assignment control before any "
                "CLV attribution claim"
            ),
            "statistical_note": (
                "one exposed historical development seed; no significance, "
                "stability, generalization, or final model claim"
            ),
        },
        "out_dir": cfg.out_dir,
    }


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _config_hash(
    cfg: M5CLVScaledValueBasisConfig, input_hash: str, revision: str
) -> str:
    payload = {
        "code_version": CODE_VERSION,
        "config": asdict(cfg),
        "input_hash": input_hash,
        "source_revision": revision,
    }
    return hashlib.sha256(_canonical(payload).encode()).hexdigest()[:12]


def _base_config(cfg: M5CLVScaledValueBasisConfig) -> dict:
    adapter = common.GatedRelationPriceConfig(
        dataset=cfg.dataset,
        seed=cfg.seed,
        time_cutoff=cfg.time_cutoff,
        evaluation_days=cfg.evaluation_days,
        epochs=cfg.epochs,
        id_dim=cfg.id_dim,
        n_layers=cfg.n_layers,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        pref_reg=cfg.pref_reg,
        input_days=cfg.input_days,
        diagnostic_max_k=cfg.diagnostic_max_k,
        out_dir=cfg.out_dir,
        baseline_result_dir="unused",
    )
    return common.gatefree._base_config(adapter)


def _prepare(cfg: M5CLVScaledValueBasisConfig) -> dict:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = moe.build_input_manifest(v3.SCHEMA[cfg.dataset])
    input_hash = moe.manifest_hash(manifest)
    revision = moe.source_revision()
    base_cfg = _base_config(cfg)
    data = v3.prepare_data(base_cfg, v3.DCFG)
    if set(data["splits"]) != {"test"}:
        raise RuntimeError(f"개발평가 분할 오염: {sorted(data['splits'])}")
    if float(data["train"].t.max()) != 683.0:
        raise RuntimeError(f"학습 종료일 오류: {data['train'].t.max()}")
    if data.get("loss_w") is not None:
        raise RuntimeError("외부 표본 가중치가 섞였습니다")
    data["loss_w"] = None

    snapshot = residual.build_final_snapshot(
        data["train"], data["n_users"], v3.DCFG["is_date"], cfg.input_days
    )
    axes = joint.build_user_axis_inputs(snapshot, data["n_users"])
    q_n, q_v, q_c, clv_valid = report_helpers.build_clv_inputs(axes)
    economic = nv.build_nv_economic_inputs(
        data["train"],
        n_users=data["n_users"],
        n_items=data["n_items"],
        q_n=q_n,
        q_v=q_v,
        q_c=q_c,
        clv_valid=clv_valid,
        n_bins=cfg.economic_bins,
        shrinkage_strength=cfg.shrinkage_strength,
    )
    economic["economic_input_diagnostics"] = dict(
        economic["economic_input_diagnostics"]
    ) | {
        "explicit_q_n_input": False,
        "explicit_q_v_input": True,
        "q_c_excluded_from_m2_input": False,
        "q_n_role": "used only inside q_C=percentile(n_u*v_u)",
        "v_input": "q_C-scaled q_V fixed value-position basis",
        "item_input": "overall amount-percentile fixed value-position basis",
        "user_economic_input_dim": cfg.economic_dim,
        "item_economic_input_dim": cfg.economic_dim,
    }
    meta = v3.item_meta(data["train"], data["n_items"])
    thresholds = v3.segment_thresholds(axes["clv_proxy"], base_cfg["SEG_EDGES"])
    cache = v3.EvalCache(
        *data["splits"]["test"], axes["clv_proxy"], thresholds, data["n_items"]
    )
    prepared = {
        "out_dir": out_dir,
        "manifest": manifest,
        "input_hash": input_hash,
        "revision": revision,
        "base_cfg": base_cfg,
        "data": data,
        "axes": axes,
        "q_n": q_n,
        "q_v": q_v,
        "q_c": q_c,
        "clv_valid": clv_valid,
        "meta": meta,
        "thresholds": thresholds,
        "cache": cache,
        **economic,
    }
    prepared["config_hash"] = _config_hash(cfg, input_hash, revision)
    return prepared


def arm_specification(prepared: dict, cfg: M5CLVScaledValueBasisConfig) -> dict:
    return {
        "model_id": MODEL_ID,
        "role": "q_c_scaled_value_basis_plus_personalized_m4",
        "architecture": "q_c_scaled_fixed_value_basis",
        "rho": cfg.rho,
        "weighted": True,
        "assignment": prepared,
        "assignment_name": "observed_m4",
        "m2_assignment_name": "observed_q_c_and_q_v",
    }


def _build_model(
    prepared: dict, cfg: M5CLVScaledValueBasisConfig, spec: dict
) -> M5NConditionedValueBasisLightGCN:
    data = prepared["data"]
    v3.set_seed(cfg.seed)
    return M5NConditionedValueBasisLightGCN(
        n_users=data["n_users"],
        n_items=data["n_items"],
        user_q_n=prepared["q_n"],
        user_q_v=prepared["q_v"],
        user_q_c=prepared["q_c"],
        user_clv_valid=prepared["clv_valid"],
        item_price_percentile=prepared["item_amount_percentile"],
        item_price_valid=prepared["item_economic_valid"],
        adj=data["adj"],
        id_dim=cfg.id_dim,
        rho=spec["rho"],
        n_layers=cfg.n_layers,
        pref_reg=cfg.pref_reg,
        gate_delta=cfg.gate_delta,
        basis_bandwidth=cfg.basis_bandwidth,
        constant_gate=1.0,
    ).to(v3.DEVICE)


class _IDOnlyView(nn.Module):
    def __init__(self, model: M5NConditionedValueBasisLightGCN):
        super().__init__()
        self.model = model

    def embeddings(self, need_value: bool = True):
        user, item = self.model.id_embeddings()
        zero_user = user.new_zeros((len(user), 1))
        zero_item = item.new_zeros((len(item), 1))
        return user, item, zero_user, zero_item


@torch.no_grad()
def _rank_intervention_diagnostics(
    model: M5NConditionedValueBasisLightGCN,
    prepared: dict,
    *,
    max_k: int,
) -> tuple[dict, pd.DataFrame]:
    users, full_topk = report_helpers._masked_topk(model, prepared, max_k=max_k)
    id_users, id_topk = report_helpers._masked_topk(
        _IDOnlyView(model).to(v3.DEVICE), prepared, max_k=max_k
    )
    if not np.array_equal(users, id_users):
        raise RuntimeError("full·ID-only 평가 사용자 순서가 다릅니다")

    width = full_topk.shape[1]
    pair_users = np.repeat(users.astype(np.int64), width)
    pair_items = full_topk.reshape(-1).astype(np.int64)
    collected = {key: [] for key in ("id", "economic")}
    for start in range(0, len(pair_users), 65536):
        user_tensor = torch.as_tensor(
            pair_users[start : start + 65536], dtype=torch.long, device=v3.DEVICE
        )
        item_tensor = torch.as_tensor(
            pair_items[start : start + 65536], dtype=torch.long, device=v3.DEVICE
        )
        components = model.candidate_score_components(user_tensor, item_tensor)
        for key in collected:
            collected[key].append(components[key].cpu().numpy())
    values = {
        key: np.concatenate(parts).astype(np.float64).reshape(len(users), width)
        for key, parts in collected.items()
    }
    id_within_std = values["id"].std(axis=1)
    economic_within_std = values["economic"].std(axis=1)
    economic_within_range = np.ptp(values["economic"], axis=1)
    overlap = report_helpers.topk_overlap_summary(
        id_topk, full_topk, prepared["cache"].seg, k=10
    )
    overall = overlap.loc[overlap["group"] == "전체"].iloc[0]
    diagnostics = {
        "model_id": MODEL_ID,
        "candidate_width": int(width),
        "mean_within_user_id_score_std": float(id_within_std.mean()),
        "mean_within_user_economic_score_std": float(economic_within_std.mean()),
        "mean_within_user_economic_to_id_std_ratio": float(
            economic_within_std.mean() / id_within_std.mean()
        ),
        "median_within_user_economic_score_std": float(
            np.median(economic_within_std)
        ),
        "mean_within_user_economic_score_range": float(
            economic_within_range.mean()
        ),
        "users_with_economic_std_above_1e_6_share": float(
            np.mean(economic_within_std > 1e-6)
        ),
        "top10_set_changed_user_share_vs_trained_id_only": float(
            overall["top10_set_changed_user_share"]
        ),
        "top10_order_changed_user_share_vs_trained_id_only": float(
            overall["top10_order_changed_user_share"]
        ),
        "top10_mean_jaccard_vs_trained_id_only": float(
            overall["top10_mean_jaccard"]
        ),
    }
    return diagnostics, overlap


def _reference_comparison(metrics: dict) -> pd.DataFrame:
    rows = []
    for reference, reference_metrics in REFERENCE_METRICS.items():
        for metric in CORE_METRICS:
            reference_value = float(reference_metrics[metric])
            model_value = float(metrics[metric])
            rows.append(
                {
                    "comparison_scope": "different_run_descriptive_reference",
                    "reference": reference,
                    "reference_result_id": REFERENCE_PROVENANCE[reference][
                        "result_id"
                    ],
                    "model_id": MODEL_ID,
                    "metric": metric,
                    "reference_value": reference_value,
                    "model_value": model_value,
                    "absolute_delta": model_value - reference_value,
                    "relative_change_pct": 100.0
                    * (model_value - reference_value)
                    / reference_value,
                }
            )
    return pd.DataFrame(rows)


def _directional_reading(metrics: dict, rank_diagnostics: dict) -> dict:
    comparisons = {
        reference: {
            metric: float(metrics[metric] - reference_metrics[metric])
            for metric in TOP10_DIRECTION_METRICS
        }
        for reference, reference_metrics in REFERENCE_METRICS.items()
    }
    beats = {
        reference: all(delta > 0.0 for delta in deltas.values())
        for reference, deltas in comparisons.items()
    }
    return {
        "directional_candidate": bool(all(beats.values())),
        "m5_beats_prior_m1_on_all_four_top10_metrics": beats[
            "m1_multineg_mean_k5"
        ],
        "m5_beats_prior_m4_on_all_four_top10_metrics": beats[
            "m4_personalized_positive_weight_k5_minimal_nv_control"
        ],
        "top10_deltas": comparisons,
        "rank_intervention_observed": bool(
            rank_diagnostics["top10_order_changed_user_share_vs_trained_id_only"]
            > 0.0
        ),
        "official_factorial_or_clv_attribution_decision_permitted": False,
        "comparison_scope": (
            "single-seed development direction using M1 and M4 values from "
            "separate completed runs"
        ),
        "next_if_directional_candidate": (
            "train one degree-matched q_C/q_V assignment control before any "
            "CLV-attribution claim"
        ),
        "next_if_nonpass": (
            "do not tune rho or M4 on the same exposed development interval"
        ),
        "statistical_note": (
            "one exposed historical development seed; no significance, stability, "
            "generalization, or final model claim"
        ),
    }


def run_clv_scaled_value_basis_one_arm(
    cfg: M5CLVScaledValueBasisConfig | None = None,
) -> pd.DataFrame:
    cfg = validate_config(cfg or configure_clv_scaled_value_basis_one_arm())
    summary = preflight_summary(cfg)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    prepared = _prepare(cfg)
    spec = arm_specification(prepared, cfg)
    print(f"\n===== {MODEL_ID} | seed {cfg.seed} | fixed {cfg.epochs} epochs =====")
    with patch.object(legacy, "_build_model", _build_model):
        arm, model = legacy._run_arm(prepared, cfg, spec)
    arm["m2_assignment"] = spec["m2_assignment_name"]
    arm["m2_architecture"] = spec["architecture"]

    row = {
        "model_id": MODEL_ID,
        "role": arm["role"],
        "seed": arm["seed"],
        "split": arm["split"],
        "final_epoch": arm["final_epoch"],
        "rho": arm["rho"],
        "positive_weight_lambda": arm["positive_weight_lambda"],
        "m2_architecture": arm["m2_architecture"],
        "m2_assignment": arm["m2_assignment"],
        "m4_assignment": arm["clv_assignment"],
        **arm["diagnostics"],
        **arm["training"].get("final_diagnostics", {}),
        **arm["metrics"],
    }
    frame = pd.DataFrame([row])
    users, top50 = report_helpers._masked_topk(
        model, prepared, max_k=cfg.diagnostic_max_k
    )
    score_frame = pd.DataFrame(
        [legacy._score_diagnostics(model, users, top50, model_id=MODEL_ID)]
    )
    rank_diagnostics, overlap = _rank_intervention_diagnostics(
        model, prepared, max_k=cfg.diagnostic_max_k
    )
    comparison = _reference_comparison(arm["metrics"])
    reading = _directional_reading(arm["metrics"], rank_diagnostics)

    out = Path(cfg.out_dir)
    stem = f"m5_clv_scaled_value_basis_{prepared['config_hash']}"
    paths = {
        "absolute_csv": out / f"{stem}.csv",
        "reference_comparison_csv": out / f"{stem}_reference_comparison.csv",
        "score_diagnostics_csv": out / f"{stem}_score_diagnostics.csv",
        "top10_overlap_csv": out / f"{stem}_top10_overlap.csv",
        "json": out / f"{stem}.json",
    }
    test10._atomic_csv(paths["absolute_csv"], frame)
    test10._atomic_csv(paths["reference_comparison_csv"], comparison)
    test10._atomic_csv(paths["score_diagnostics_csv"], score_frame)
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
            "reference_comparison_rows": comparison.to_dict("records"),
            "reference_provenance": REFERENCE_PROVENANCE,
            "score_diagnostic_rows": score_frame.to_dict("records"),
            "rank_intervention_diagnostics": rank_diagnostics,
            "top10_overlap_rows": overlap.to_dict("records"),
            "directional_reading": reading,
            "arm": arm,
            "result_paths": {key: str(value) for key, value in paths.items()},
        },
    )
    frame.attrs["reference_comparison"] = comparison
    frame.attrs["score_diagnostics"] = score_frame
    frame.attrs["rank_intervention_diagnostics"] = rank_diagnostics
    frame.attrs["top10_overlap"] = overlap
    frame.attrs["decision"] = reading
    frame.attrs["preflight"] = summary
    frame.attrs["result_paths"] = {key: str(value) for key, value in paths.items()}

    print("\n1) 새 M5 전체 절대지표")
    print(frame.to_string(index=False))
    print("\n2) 이전 M1·M4 참고값과 핵심지표 비교(서로 다른 실행)")
    print(comparison.to_string(index=False))
    print("\n3) ID 점수 대비 경제점수 영향력")
    print(score_frame.to_string(index=False))
    print("\n4) 사용자 내 후보 구분·Top-10 변경 진단")
    print(json.dumps(rank_diagnostics, ensure_ascii=False, indent=2))
    print(overlap.to_string(index=False))
    print("\n5) 방향성 판독")
    print(json.dumps(reading, ensure_ascii=False, indent=2))
    print("\n6) 저장 파일")
    print(json.dumps(frame.attrs["result_paths"], ensure_ascii=False, indent=2))
    return frame


if __name__ == "__main__":
    print(
        "Import this module from its dedicated Colab notebook and call "
        "run_clv_scaled_value_basis_one_arm()."
    )
