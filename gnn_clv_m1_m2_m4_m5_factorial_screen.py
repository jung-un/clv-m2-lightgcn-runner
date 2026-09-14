"""Seed-42 M1/M2/M4/M5 factorial screen for NGCF, GAT and GraphSAGE."""

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

from clv_scaled_value_basis_gnn_model import BACKBONES, CLVScaledValueBasisGNN
import lightgcn_clv_axis_specific_test10 as test10
import lightgcn_clv_gradient_isolated_economic_interaction as report_helpers
import lightgcn_clv_m5_clv_scaled_value_basis_one_arm as value_basis_source
import lightgcn_clv_m5_economic_positive_weight as legacy
import lightgcn_clv_v3 as v3


CODE_VERSION = "gnn-clv-m1-m2-m4-m5-factorial-development-screen-v1"
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
TOP10_METRICS = (
    "recall@10",
    "ndcg@10",
    "price_purchase_amount_weighted_hit@10",
    "vndcg@10",
)


@dataclass(frozen=True)
class GNNCLVFactorialConfig:
    backbone: str = "ngcf"
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


def model_ids(backbone: str) -> dict[str, str]:
    if backbone not in BACKBONES:
        raise ValueError(f"backbone은 {BACKBONES} 중 하나여야 합니다")
    return {
        "m1": f"{backbone}_m1_id67_multineg_mean_k5",
        "m2": f"{backbone}_m2_clv_scaled_value_basis_multineg_mean_k5",
        "m4": f"{backbone}_m4_personalized_positive_weight_id67_k5",
        "m5": f"{backbone}_m5_clv_scaled_value_basis_personalized_positive_weight_k5",
    }


def configure_gnn_clv_factorial_screen(
    backbone: str = "ngcf", **overrides
) -> GNNCLVFactorialConfig:
    if backbone not in BACKBONES:
        raise ValueError(f"backbone은 {BACKBONES} 중 하나여야 합니다")
    defaults = {
        "backbone": backbone,
        "out_dir": (
            f"{v3.default_out_dir('dunnhumby')}"
            f"_{backbone}_clv_m1_m2_m4_m5_factorial_development_screen_v1"
        ),
    }
    return validate_config(GNNCLVFactorialConfig(**(defaults | overrides)))


def validate_config(cfg: GNNCLVFactorialConfig) -> GNNCLVFactorialConfig:
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
    if cfg.backbone not in BACKBONES:
        raise ValueError(f"backbone은 {BACKBONES} 중 하나여야 합니다")
    for key, expected in fixed.items():
        if getattr(cfg, key) != expected:
            raise ValueError(f"GNN factorial screen은 {key}={expected!r}이어야 합니다")
    if cfg.batch_size <= 0 or cfg.lr <= 0 or cfg.pref_reg < 0 or not cfg.out_dir:
        raise ValueError("학습 설정 또는 출력 경로가 잘못됐습니다")
    return cfg


def preflight_summary(cfg: GNNCLVFactorialConfig) -> dict:
    cfg = validate_config(cfg)
    ids = model_ids(cfg.backbone)
    return {
        "code_version": CODE_VERSION,
        "dataset": cfg.dataset,
        "backbone": cfg.backbone,
        "seed": cfg.seed,
        "split": "historical_development_days_684_690",
        "trained_models": list(ids.values()),
        "reused_models": [],
        "research_question": (
            "Within one non-LightGCN backbone, what are the separate and combined "
            "effects of the q_C-scaled q_V value-position representation (M2) and "
            "the unchanged personalized historical-CLV positive weighting (M4)?"
        ),
        "factorial_arms": {
            "M1": "matched 67-dimensional ID representation; unweighted BPR",
            "M2": "64 ID + 3 fixed q_C-scaled q_V/item-price RBF coordinates; unweighted BPR",
            "M4": "matched 67-dimensional ID representation; personalized positive weighting",
            "M5": "M2 representation plus M4 weighting",
        },
        "m2": {
            "q_c": "midrank percentile of raw n_u*v_u",
            "q_c_role": "strength of the complete user value-position basis",
            "q_v_role": "position in a fixed low/mid/high value basis",
            "user_coordinates": "sqrt(rho)*q_C(u)*b(q_V(u))",
            "item_coordinates": "sqrt(rho)*b(item amount percentile)",
            "basis": "fixed L2-normalized Gaussian RBF at [0,0.5,1]",
            "separate_q_n_gate": False,
            "same_recommendation_gradient": True,
            "external_reranking": False,
        },
        "m4": {
            "formula": "1 + 0.5*q_C*item_amount_percentile*clipped_user_bin_fit",
            "normalization": "divide by mean over all positive training rows",
            "changed_from_prior_personalized_m4": False,
        },
        "capacity_matching": {
            "propagation_input_width": 67,
            "m1_m4": "67 trainable ID coordinates",
            "m2_m5": "64 trainable ID coordinates plus 3 fixed semantic coordinates",
        },
        "fixed": {
            "new_item_task": True,
            "train_pairs_excluded_from_evaluation": True,
            "min_item_interactions": 1,
            "graph": "binary",
            "negative_sampling": "uniform",
            "negative_count": cfg.negative_count,
            "epochs": cfg.epochs,
            "validation_or_epoch_selection": False,
            "final_test_constructed": False,
            "holdout_constructed": False,
            "one_training_loop_and_optimizer_per_arm": True,
            "external_reranking": False,
            "m3_edge_weight": False,
        },
        "reading_rule": {
            "m2_increment": "M2 > M1 on each reported metric",
            "m4_increment": "M4 > M1 on each reported metric",
            "m5_vs_baseline": "M5 > M1 on all four Top-10 accuracy/economic metrics",
            "m5_vs_m4": "M5 > M4 on all four Top-10 accuracy/economic metrics",
            "factorial_interaction": "(M5-M4)-(M2-M1), reported per metric; positive synergy is not required",
            "clv_attribution_tested": False,
            "statistical_note": (
                "one exposed historical development seed; no significance, "
                "stability, generalization, or final model claim"
            ),
        },
        "c3_new_evidence": (
            "Earlier GNN screens used the rejected level/composition/price M2 alone. "
            "This screen tests a different fixed q_C-scaled q_V basis together with "
            "the later personalized M4 in a predeclared 2x2 factorial design."
        ),
        "out_dir": cfg.out_dir,
    }


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _config_hash(cfg: GNNCLVFactorialConfig, prepared: dict) -> str:
    payload = {
        "code_version": CODE_VERSION,
        "config": asdict(cfg),
        "input_hash": prepared["input_hash"],
        "source_revision": prepared["revision"],
    }
    return hashlib.sha256(_canonical(payload).encode()).hexdigest()[:12]


def _prepare(cfg: GNNCLVFactorialConfig) -> dict:
    prepared = value_basis_source._prepare(cfg)
    prepared["config_hash"] = _config_hash(cfg, prepared)
    # The reused legacy trainer merges this dictionary into every arm's model
    # diagnostics.  Arm-specific representation facts come from the model.
    prepared["economic_input_diagnostics"] = {}
    return prepared


def arm_specifications(prepared: dict, cfg: GNNCLVFactorialConfig) -> list[dict]:
    ids = model_ids(cfg.backbone)
    return [
        {
            "model_id": ids["m1"],
            "role": "factorial_m1",
            "rho": 0.0,
            "m2_active": False,
            "weighted": False,
            "assignment": prepared,
            "assignment_name": "no_clv_assignment",
        },
        {
            "model_id": ids["m2"],
            "role": "factorial_m2",
            "rho": cfg.rho,
            "m2_active": True,
            "weighted": False,
            "assignment": prepared,
            "assignment_name": "observed_q_c_q_v",
        },
        {
            "model_id": ids["m4"],
            "role": "factorial_m4",
            "rho": 0.0,
            "m2_active": False,
            "weighted": True,
            "assignment": prepared,
            "assignment_name": "observed_personalized_m4",
        },
        {
            "model_id": ids["m5"],
            "role": "factorial_m5",
            "rho": cfg.rho,
            "m2_active": True,
            "weighted": True,
            "assignment": prepared,
            "assignment_name": "observed_q_c_q_v_and_personalized_m4",
        },
    ]


def _build_model(
    prepared: dict, cfg: GNNCLVFactorialConfig, spec: dict
) -> CLVScaledValueBasisGNN:
    data = prepared["data"]
    v3.set_seed(cfg.seed)
    return CLVScaledValueBasisGNN(
        backbone=cfg.backbone,
        m2_active=spec["m2_active"],
        n_users=data["n_users"],
        n_items=data["n_items"],
        user_q_v=prepared["q_v"],
        user_q_c=prepared["q_c"],
        user_clv_valid=prepared["clv_valid"],
        item_price_percentile=prepared["item_amount_percentile"],
        item_price_valid=prepared["item_economic_valid"],
        adj=data["adj"],
        base_id_dim=cfg.id_dim,
        economic_dim=cfg.economic_dim,
        rho=spec["rho"],
        basis_bandwidth=cfg.basis_bandwidth,
        n_layers=cfg.n_layers,
        pref_reg=cfg.pref_reg,
    ).to(v3.DEVICE)


class _M2OffCounterfactualView(nn.Module):
    """Evaluation-only view of one trained M2/M5 with its fixed input zeroed."""

    def __init__(self, model: CLVScaledValueBasisGNN):
        super().__init__()
        if not model.m2_active:
            raise ValueError("M2·M5에서만 counterfactual view를 만들 수 있습니다")
        self.model = model

    def embeddings(self, need_value: bool = True):
        user, item = self.model.propagated_embeddings(m2_active=False)
        zero_user = user.new_zeros((len(user), 1))
        zero_item = item.new_zeros((len(item), 1))
        return user, item, zero_user, zero_item


@torch.no_grad()
def _counterfactual_m2_diagnostics(
    model: CLVScaledValueBasisGNN,
    prepared: dict,
    *,
    model_id: str,
    max_k: int,
) -> tuple[dict, pd.DataFrame]:
    users, full_topk = report_helpers._masked_topk(model, prepared, max_k=max_k)
    off_view = _M2OffCounterfactualView(model).to(v3.DEVICE)
    off_users, off_topk = report_helpers._masked_topk(
        off_view, prepared, max_k=max_k
    )
    if not np.array_equal(users, off_users):
        raise RuntimeError("M2 on/off 평가 사용자 순서가 다릅니다")

    full_user, full_item = model.propagated_embeddings()
    off_user, off_item = model.propagated_embeddings(m2_active=False)
    width = full_topk.shape[1]
    pair_users = np.repeat(users.astype(np.int64), width)
    pair_items = full_topk.reshape(-1).astype(np.int64)
    full_scores = []
    off_scores = []
    for start in range(0, len(pair_users), 65536):
        u = torch.as_tensor(
            pair_users[start : start + 65536], dtype=torch.long, device=v3.DEVICE
        )
        i = torch.as_tensor(
            pair_items[start : start + 65536], dtype=torch.long, device=v3.DEVICE
        )
        full_scores.append((full_user[u] * full_item[i]).sum(dim=1).cpu().numpy())
        off_scores.append((off_user[u] * off_item[i]).sum(dim=1).cpu().numpy())
    full_values = np.concatenate(full_scores).reshape(len(users), width).astype(float)
    off_values = np.concatenate(off_scores).reshape(len(users), width).astype(float)
    delta = full_values - off_values
    overlap = report_helpers.topk_overlap_summary(
        off_topk, full_topk, prepared["cache"].seg, k=10
    )
    overall = overlap.loc[overlap["group"] == "전체"].iloc[0]
    diagnostics = {
        "model_id": model_id,
        "diagnostic": "same-trained-model fixed M2 input on versus zeroed",
        "candidate_width": int(width),
        "counterfactual_score_delta_mean_abs": float(np.abs(delta).mean()),
        "counterfactual_score_delta_std": float(delta.std()),
        "mean_within_user_counterfactual_delta_std": float(
            delta.std(axis=1).mean()
        ),
        "users_with_within_delta_std_above_1e_6_share": float(
            np.mean(delta.std(axis=1) > 1e-6)
        ),
        "top10_set_changed_user_share": float(
            overall["top10_set_changed_user_share"]
        ),
        "top10_order_changed_user_share": float(
            overall["top10_order_changed_user_share"]
        ),
        "top10_mean_jaccard": float(overall["top10_mean_jaccard"]),
        "additive_score_decomposition_claimed": False,
    }
    overlap.insert(0, "model_id", model_id)
    overlap.insert(1, "reference", "same_trained_model_with_m2_input_zeroed")
    return diagnostics, overlap


def interaction_rows(metric_rows: dict[str, dict], ids: dict[str, str]) -> pd.DataFrame:
    m1, m2 = metric_rows[ids["m1"]], metric_rows[ids["m2"]]
    m4, m5 = metric_rows[ids["m4"]], metric_rows[ids["m5"]]
    rows = []
    for metric in CORE_METRICS:
        if not all(metric in values for values in (m1, m2, m4, m5)):
            continue
        rows.append(
            {
                "metric": metric,
                "m2_minus_m1": float(m2[metric] - m1[metric]),
                "m4_minus_m1": float(m4[metric] - m1[metric]),
                "m5_minus_m1": float(m5[metric] - m1[metric]),
                "m5_minus_m4": float(m5[metric] - m4[metric]),
                "interaction": float(
                    (m5[metric] - m4[metric]) - (m2[metric] - m1[metric])
                ),
            }
        )
    return pd.DataFrame(rows)


def screening_reading(metric_rows: dict[str, dict], ids: dict[str, str]) -> dict:
    m1, m2 = metric_rows[ids["m1"]], metric_rows[ids["m2"]]
    m4, m5 = metric_rows[ids["m4"]], metric_rows[ids["m5"]]
    by_metric = {}
    for metric in TOP10_METRICS:
        by_metric[metric] = {
            "m2_minus_m1": float(m2[metric] - m1[metric]),
            "m4_minus_m1": float(m4[metric] - m1[metric]),
            "m5_minus_m1": float(m5[metric] - m1[metric]),
            "m5_minus_m4": float(m5[metric] - m4[metric]),
            "interaction": float(
                (m5[metric] - m4[metric]) - (m2[metric] - m1[metric])
            ),
        }
    return {
        "top10_deltas": by_metric,
        "m5_beats_m1_all_four_top10_metrics": all(
            by_metric[metric]["m5_minus_m1"] > 0.0 for metric in TOP10_METRICS
        ),
        "m5_beats_m4_all_four_top10_metrics": all(
            by_metric[metric]["m5_minus_m4"] > 0.0 for metric in TOP10_METRICS
        ),
        "positive_interaction_all_four_top10_metrics": all(
            by_metric[metric]["interaction"] > 0.0 for metric in TOP10_METRICS
        ),
        "clv_attribution_tested": False,
        "decision_scope": "single-seed historical-development backbone portability screen",
        "statistical_note": (
            "one exposed historical development seed; no significance, stability, "
            "generalization, or final model claim"
        ),
        "next_step_rule": (
            "Only a backbone whose M5 beats both its matched M1 and M4 on all four "
            "Top-10 metrics is eligible for an assignment-control run."
        ),
    }


def run_gnn_clv_factorial_screen(
    cfg: GNNCLVFactorialConfig | None = None,
) -> pd.DataFrame:
    cfg = validate_config(cfg or configure_gnn_clv_factorial_screen())
    summary = preflight_summary(cfg)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    prepared = _prepare(cfg)
    ids = model_ids(cfg.backbone)

    arms: dict[str, dict] = {}
    models: dict[str, CLVScaledValueBasisGNN] = {}
    for spec in arm_specifications(prepared, cfg):
        print(
            f"\n===== {spec['model_id']} | seed {cfg.seed} | "
            f"fixed {cfg.epochs} epochs ====="
        )
        with patch.object(legacy, "_build_model", _build_model):
            arm, model = legacy._run_arm(prepared, cfg, spec)
        arm["diagnostics"] = model.representation_diagnostics()
        arms[spec["model_id"]] = arm
        models[spec["model_id"]] = model

    rows = []
    metric_rows = {}
    for model_id in ids.values():
        arm = arms[model_id]
        metric_rows[model_id] = arm["metrics"]
        rows.append(
            {
                "model_id": model_id,
                "role": arm["role"],
                "backbone": cfg.backbone,
                "seed": arm["seed"],
                "split": arm["split"],
                "final_epoch": arm["final_epoch"],
                "rho": arm["rho"],
                "positive_weight_lambda": arm["positive_weight_lambda"],
                **arm["diagnostics"],
                **arm["training"].get("final_diagnostics", {}),
                **arm["metrics"],
            }
        )
    frame = pd.DataFrame(rows)
    comparison = report_helpers._metric_comparison(
        metric_rows, references=(ids["m1"], ids["m4"])
    )
    interactions = interaction_rows(metric_rows, ids)
    reading = screening_reading(metric_rows, ids)

    counterfactual_rows = []
    overlap_frames = []
    for key in ("m2", "m5"):
        diagnostics, overlap = _counterfactual_m2_diagnostics(
            models[ids[key]],
            prepared,
            model_id=ids[key],
            max_k=cfg.diagnostic_max_k,
        )
        counterfactual_rows.append(diagnostics)
        overlap_frames.append(overlap)
    counterfactual_frame = pd.DataFrame(counterfactual_rows)
    overlap_frame = pd.concat(overlap_frames, ignore_index=True)

    out = Path(cfg.out_dir)
    stem = f"{cfg.backbone}_clv_factorial_{prepared['config_hash']}"
    paths = {
        "absolute_csv": out / f"{stem}.csv",
        "comparison_csv": out / f"{stem}_comparison.csv",
        "interaction_csv": out / f"{stem}_interaction.csv",
        "counterfactual_m2_csv": out / f"{stem}_counterfactual_m2.csv",
        "top10_overlap_csv": out / f"{stem}_top10_overlap.csv",
        "json": out / f"{stem}.json",
    }
    test10._atomic_csv(paths["absolute_csv"], frame)
    test10._atomic_csv(paths["comparison_csv"], comparison)
    test10._atomic_csv(paths["interaction_csv"], interactions)
    test10._atomic_csv(paths["counterfactual_m2_csv"], counterfactual_frame)
    test10._atomic_csv(paths["top10_overlap_csv"], overlap_frame)
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
            "interaction_rows": interactions.to_dict("records"),
            "counterfactual_m2_rows": counterfactual_frame.to_dict("records"),
            "top10_overlap_rows": overlap_frame.to_dict("records"),
            "screening_reading": reading,
            "arms": arms,
            "result_paths": {key: str(value) for key, value in paths.items()},
        },
    )
    frame.attrs["comparison"] = comparison
    frame.attrs["interaction"] = interactions
    frame.attrs["counterfactual_m2"] = counterfactual_frame
    frame.attrs["top10_overlap"] = overlap_frame
    frame.attrs["decision"] = reading
    frame.attrs["preflight"] = summary
    frame.attrs["result_paths"] = {key: str(value) for key, value in paths.items()}

    print("\n1) M1·M2·M4·M5 전체 절대지표")
    print(frame.to_string(index=False))
    print("\n2) M1·M4 대비 전체 지표 비교")
    print(comparison.to_string(index=False))
    print("\n3) 2x2 요인효과와 상호작용")
    print(interactions.to_string(index=False))
    print("\n4) 동일 학습모형에서 M2 입력 on/off 진단")
    print(counterfactual_frame.to_string(index=False))
    print(overlap_frame.to_string(index=False))
    print("\n5) 사전 고정 판독")
    print(json.dumps(reading, ensure_ascii=False, indent=2))
    print("\n6) 저장 파일")
    print(json.dumps(frame.attrs["result_paths"], ensure_ascii=False, indent=2))
    return frame


if __name__ == "__main__":
    print(
        "Import this module from one of the three dedicated Colab notebooks and "
        "call run_gnn_clv_factorial_screen()."
    )
