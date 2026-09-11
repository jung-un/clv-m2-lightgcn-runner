"""Seed-42 development screen for semantic N/V M2 + isolated M3 + M4."""

from __future__ import annotations

import gc
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from clv_m3_directional_value_graph import match_first_hop_gates
from clv_m5_semantic_nv_m3_model import (
    M5SemanticNVIsolatedFirstHopLightGCN,
)
import lightgcn_clv_gradient_isolated_economic_interaction as report_helpers
import lightgcn_clv_m5_economic_positive_weight as legacy
import lightgcn_clv_m5_nv_economic_positive_weight as nv
import lightgcn_clv_m5_semantic_nv_single_screen as semantic
import lightgcn_clv_v3 as v3


CODE_VERSION = "m5-semantic-nv-isolated-first-hop-screen-v1"
M3_OFF_MODEL_ID = "m5_semantic_nv_m3off_isolated"
ACTUAL_MODEL_ID = "m5_semantic_nv_m3_actual_clv"
SHUFFLE_MODEL_ID = "m5_semantic_nv_m3_degree_shuffled_clv"
RELATION_MODEL_ID = "m5_semantic_nv_m3_relation_only"
MODEL_IDS = (
    M3_OFF_MODEL_ID,
    ACTUAL_MODEL_ID,
    SHUFFLE_MODEL_ID,
    RELATION_MODEL_ID,
)
PRIMARY_METRICS = (
    "vndcg@10",
    "price_purchase_amount_weighted_hit@10",
)
ACCURACY_METRICS = (
    "recall@10",
    "ndcg@10",
    "recall@20",
    "ndcg@20",
    "recall@50",
    "ndcg@50",
)


@dataclass(frozen=True)
class M5SemanticNVM3ScreenConfig:
    dataset: str = "dunnhumby"
    seed: int = 42
    time_cutoff: int = 690
    evaluation_days: int = 7
    epochs: int = 100
    id_dim: int = 64
    economic_dim: int = 5
    economic_bins: int = 4
    shrinkage_strength: float = 10.0
    rho: float = 0.15
    price_axis_budget: float = 0.25
    scale_delta: float = 0.25
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
    m3_target_strength: float = 0.075
    m3_beta_cap: float = 20.0
    out_dir: str = ""
    baseline_result_dir: str = ""
    cached_lineage_b_result_json: str = ""


def configure_semantic_nv_m3_screen(**overrides) -> M5SemanticNVM3ScreenConfig:
    root = v3.default_out_dir("dunnhumby")
    defaults = {
        "out_dir": f"{root}_m5_semantic_nv_isolated_m3_screen_v1",
        "baseline_result_dir": f"{root}_m2_repeatshare_historical_backtest_v1",
        "cached_lineage_b_result_json": (
            f"{root}_m5_semantic_nv_personalized_positive_single_screen_v1/"
            "m5_semantic_nv_single_426685be4486.json"
        ),
    }
    return validate_config(M5SemanticNVM3ScreenConfig(**(defaults | overrides)))


def validate_config(cfg: M5SemanticNVM3ScreenConfig) -> M5SemanticNVM3ScreenConfig:
    fixed = {
        "dataset": "dunnhumby",
        "seed": 42,
        "time_cutoff": 690,
        "evaluation_days": 7,
        "epochs": 100,
        "id_dim": 64,
        "economic_dim": 5,
        "economic_bins": 4,
        "shrinkage_strength": 10.0,
        "rho": 0.15,
        "price_axis_budget": 0.25,
        "scale_delta": 0.25,
        "positive_weight_lambda": 0.5,
        "n_layers": 2,
        "negative_count": 5,
        "input_days": 365,
        "diagnostic_max_k": 50,
        "shuffle_degree_bins": 10,
        "shuffle_seed": 42,
        "m3_target_strength": 0.075,
        "m3_beta_cap": 20.0,
    }
    for key, expected in fixed.items():
        if getattr(cfg, key) != expected:
            raise ValueError(f"격리 M3 screen은 {key}={expected!r}이어야 합니다")
    if cfg.batch_size <= 0 or cfg.lr <= 0 or cfg.pref_reg < 0:
        raise ValueError("학습 설정이 잘못됐습니다")
    if not cfg.out_dir or not cfg.baseline_result_dir:
        raise ValueError("out_dir와 baseline_result_dir가 필요합니다")
    return cfg


def preflight_summary(cfg: M5SemanticNVM3ScreenConfig) -> dict:
    cfg = validate_config(cfg)
    return {
        "code_version": CODE_VERSION,
        "dataset": cfg.dataset,
        "seed": cfg.seed,
        "split": "historical_development_days_684_690",
        "trained_models": list(MODEL_IDS),
        "research_question": (
            "Does train-only directional first-hop propagation add value to the "
            "fixed semantic N/V M2 plus personalized positive-weight M4, and is "
            "that increment attributable to the observed historical CLV assignment?"
        ),
        "m2_frozen_definition": {
            "formula": (
                "c_V*beta*(2q_V-1)*(2price_pct-1) + "
                "c_N*(1-beta)*q_N*dot(shrunken_profile, centered_item_bin)"
            ),
            "rho": cfg.rho,
            "beta": cfg.price_axis_budget,
            "joint_end_to_end_training": True,
        },
        "m3": {
            "relation": (
                "within-user centered midrank of mean item share of basket value"
            ),
            "shared_actual_gate": "the exact q_C array used by M4",
            "changed_path": "user layer-1 item message only",
            "binary_item_and_two_hop_paths_preserved": True,
            "per_user_first_hop_mass_preserved": True,
            "target_log_coefficient_ratio_std": cfg.m3_target_strength,
            "beta_cap": cfg.m3_beta_cap,
            "prior_rejection_disclosure": (
                "the same relation family failed on 2026-09-01; this run tests "
                "the pre-specified combined M5 context with an isolated path"
            ),
        },
        "m4_frozen_definition": {
            "row_weight": (
                "1 + 0.5*q_C*item_amount_percentile*personalized_bin_fit"
            ),
            "normalization": "mean raw weight over every positive training row",
            "loss": "mean of K=5 uniform-negative BPR losses, then row weighted",
        },
        "arms": {
            M3_OFF_MODEL_ID: "same isolated code path; active U<-I equals base U<-I",
            ACTUAL_MODEL_ID: "M3 gate = observed q_C",
            SHUFFLE_MODEL_ID: (
                "M3 gate only = degree-bin joint-shuffle source q_C; M2/M4 observed"
            ),
            RELATION_MODEL_ID: "M3 gate = 1 for CLV-valid users and 0 otherwise",
        },
        "fixed": {
            "new_item_task": True,
            "min_item_interactions": 1,
            "binary_base_graph": True,
            "uniform_negative_sampling": True,
            "epochs": cfg.epochs,
            "validation_or_epoch_selection": False,
            "test_constructed": False,
            "holdout_constructed": False,
            "one_training_loop_and_optimizer_per_arm": True,
            "pretraining_or_freezing": False,
            "external_reranking": False,
        },
        "reading_rule": {
            "primary_metrics": list(PRIMARY_METRICS),
            "conditional_complementarity": (
                "actual M3 beats M3-off, M3-shuffle, and relation-only on both metrics"
            ),
            "accuracy_and_exposure": "reported in full but not used as a post-hoc gate",
            "statistical_note": (
                "one historical development seed; no significance or generalization claim"
            ),
        },
        "out_dir": cfg.out_dir,
    }


def _config_hash(cfg: M5SemanticNVM3ScreenConfig, input_hash: str, revision: str) -> str:
    payload = {
        "code_version": CODE_VERSION,
        "config": asdict(cfg),
        "input_hash": input_hash,
        "source_revision": revision,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()[:12]


def _extract_base_first_hop(data: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_users, n_items = data["n_users"], data["n_items"]
    adjacency = data["adj"].coalesce()
    indices = adjacency.indices().detach().cpu().numpy()
    values = adjacency.values().detach().cpu().numpy()
    mask = (indices[0] < n_users) & (indices[1] >= n_users)
    users = indices[0, mask].astype(np.int64)
    items = (indices[1, mask] - n_users).astype(np.int64)
    coefficients = values[mask].astype(np.float64)
    order = np.argsort(users * n_items + items, kind="stable")
    users, items, coefficients = users[order], items[order], coefficients[order]
    expected = np.asarray(data["pos_key"], dtype=np.int64)
    actual = users * n_items + items
    if not np.array_equal(actual, expected):
        raise RuntimeError("actual LightGCN adjacency and train edge set differ")
    return users, items, coefficients


def _sparse_ui(
    users: np.ndarray,
    items: np.ndarray,
    values: np.ndarray,
    n_users: int,
    n_items: int,
) -> torch.Tensor:
    indices = torch.from_numpy(np.stack([users, items]))
    tensor_values = torch.from_numpy(np.asarray(values, dtype=np.float32))
    return torch.sparse_coo_tensor(
        indices,
        tensor_values,
        size=(n_users, n_items),
        check_invariants=False,
    ).coalesce().to(v3.DEVICE)


def attach_m3_inputs(prepared: dict, cfg: M5SemanticNVM3ScreenConfig) -> dict:
    users, items, base = _extract_base_first_hop(prepared["data"])
    shuffled = nv.joint_degree_matched_shuffle(
        prepared, seed=cfg.shuffle_seed, degree_bins=cfg.shuffle_degree_bins
    )
    source = np.asarray(shuffled["source_user"], dtype=np.int64)
    q_c = np.asarray(prepared["q_c"])
    valid = np.asarray(prepared["clv_valid"], dtype=bool)
    gates = {
        "actual": q_c,
        "shuffle": q_c[source],
        "relation_only": np.where(valid, 1.0, 0.0),
    }
    matched = match_first_hop_gates(
        prepared["data"]["train"],
        users,
        items,
        base,
        gates,
        n_users=prepared["data"]["n_users"],
        target_strength=cfg.m3_target_strength,
        beta_cap=cfg.m3_beta_cap,
    )
    base_ui = _sparse_ui(
        users, items, base, prepared["data"]["n_users"], prepared["data"]["n_items"]
    )
    prepared["m3_matched"] = matched
    prepared["m3_source_user"] = source
    prepared["m3_operators"] = {
        "off": base_ui,
        "actual": _sparse_ui(
            users,
            items,
            matched.user_from_item_coefficients["actual"],
            prepared["data"]["n_users"],
            prepared["data"]["n_items"],
        ),
        "shuffle": _sparse_ui(
            users,
            items,
            matched.user_from_item_coefficients["shuffle"],
            prepared["data"]["n_users"],
            prepared["data"]["n_items"],
        ),
        "relation_only": _sparse_ui(
            users,
            items,
            matched.user_from_item_coefficients["relation_only"],
            prepared["data"]["n_users"],
            prepared["data"]["n_items"],
        ),
    }
    prepared["m3_base_ui"] = base_ui
    prepared["m3_base_iu"] = base_ui.transpose(0, 1).coalesce()
    prepared["m3_gate_preflight"] = _gate_preflight(prepared)
    return prepared


def _gate_preflight(prepared: dict) -> dict:
    q_c = np.asarray(prepared["q_c"])
    valid = np.asarray(prepared["clv_valid"], dtype=bool)
    source = np.asarray(prepared["m3_source_user"], dtype=np.int64)
    degree_bin = np.asarray(prepared["degree_bin"])
    matched = prepared["m3_matched"]
    actual = np.asarray(matched.user_gates["actual"])
    shuffled = np.asarray(matched.user_gates["shuffle"])
    relation = np.asarray(matched.user_gates["relation_only"])
    checks = {
        "actual_is_exact_prepared_q_c": bool(np.array_equal(actual, q_c)),
        "source_is_permutation": bool(np.array_equal(np.sort(source), np.arange(len(source)))),
        "source_stays_in_degree_bin": bool(np.array_equal(degree_bin, degree_bin[source])),
        "shuffle_preserves_gate_multiset": bool(
            np.array_equal(np.sort(shuffled), np.sort(actual))
        ),
        "shuffle_preserves_zero_gate_count": bool(
            np.count_nonzero(shuffled == 0) == np.count_nonzero(actual == 0)
        ),
        "actual_active_equals_valid": bool(np.array_equal(actual > 0, valid)),
        "relation_active_equals_valid": bool(np.array_equal(relation > 0, valid)),
    }
    return {
        "n_users": int(len(q_c)),
        "valid_user_count": int(valid.sum()),
        "invalid_user_count": int((~valid).sum()),
        "actual_active_user_count": int(np.count_nonzero(actual > 0)),
        "shuffle_active_user_count": int(np.count_nonzero(shuffled > 0)),
        "relation_only_active_user_count": int(np.count_nonzero(relation > 0)),
        "checks": checks,
        "all_checks_pass": bool(all(checks.values())),
    }


def _prepare(cfg: M5SemanticNVM3ScreenConfig) -> dict:
    prepared = semantic._prepare(cfg)
    attach_m3_inputs(prepared, cfg)
    prepared["config_hash"] = _config_hash(
        cfg, prepared["input_hash"], prepared["revision"]
    )
    return prepared


def arm_specifications(prepared: dict, cfg: M5SemanticNVM3ScreenConfig) -> list[dict]:
    common = {
        "rho": cfg.rho,
        "weighted": True,
        "assignment": prepared,
        "assignment_name": "observed_m2_and_m4",
    }
    return [
        {"model_id": M3_OFF_MODEL_ID, "role": "m3_off_isolated_control", "m3_arm": "off", **common},
        {"model_id": ACTUAL_MODEL_ID, "role": "actual_shared_q_c_m3", "m3_arm": "actual", **common},
        {"model_id": SHUFFLE_MODEL_ID, "role": "m3_gate_assignment_control", "m3_arm": "shuffle", **common},
        {"model_id": RELATION_MODEL_ID, "role": "m3_relation_only_control", "m3_arm": "relation_only", **common},
    ]


def _build_model(prepared: dict, cfg: M5SemanticNVM3ScreenConfig, spec: dict):
    data = prepared["data"]
    assignment = spec["assignment"]
    v3.set_seed(cfg.seed)
    return M5SemanticNVIsolatedFirstHopLightGCN(
        n_users=data["n_users"],
        n_items=data["n_items"],
        user_q_n=assignment["q_n"],
        user_q_v_centered=assignment["user_economic_input"][:, 0],
        user_centered_profile=assignment["user_economic_input"][:, 1:],
        user_economic_valid=assignment["user_economic_valid"],
        item_price_centered=prepared["item_economic_input"][:, 0],
        item_centered_bin=prepared["item_economic_input"][:, 1:],
        item_economic_valid=prepared["item_economic_valid"],
        adj=data["adj"],
        base_user_from_item=prepared["m3_base_ui"],
        base_item_from_user=prepared["m3_base_iu"],
        active_user_from_item=prepared["m3_operators"][spec["m3_arm"]],
        id_dim=cfg.id_dim,
        rho=spec["rho"],
        beta=cfg.price_axis_budget,
        n_layers=cfg.n_layers,
        pref_reg=cfg.pref_reg,
        scale_delta=cfg.scale_delta,
    ).to(v3.DEVICE)


def _weighted_batch_loss(model, prepared: dict, cfg, users, positives, negatives):
    user_z, item_z = model.propagated_embeddings()
    positive_scores = (user_z[users] * item_z[positives]).sum(dim=1)
    negative_scores = (user_z[users, None] * item_z[negatives]).sum(dim=2)
    normalizer = legacy._train_weight_normalizer(
        prepared, prepared, cfg.positive_weight_lambda
    )
    q_c = torch.as_tensor(prepared["q_c"], device=v3.DEVICE)
    amount = torch.as_tensor(prepared["item_amount_percentile"], device=v3.DEVICE)
    fit = torch.as_tensor(prepared["user_bin_fit"], device=v3.DEVICE)
    item_bin = torch.as_tensor(
        prepared["item_bin"], dtype=torch.long, device=v3.DEVICE
    )
    weights = legacy.personalized_positive_row_weights(
        q_c[users],
        amount[positives],
        fit[users, item_bin[positives]],
        train_mean_raw_weight=normalizer,
        lambda_=cfg.positive_weight_lambda,
    )
    bpr, _ = legacy.weighted_multi_negative_bpr(
        positive_scores, negative_scores, weights
    )
    return bpr + model.sampled_l2(users, positives, negatives)


def identity_preflight(prepared: dict, cfg: M5SemanticNVM3ScreenConfig) -> dict:
    spec = arm_specifications(prepared, cfg)[0]
    reference = semantic._build_model(prepared, cfg, spec)
    isolated = _build_model(prepared, cfg, spec)
    isolated.load_state_dict(reference.state_dict(), strict=True)
    tolerance = 1e-5 if v3.DEVICE.type == "cuda" else 1e-6
    with torch.no_grad():
        reference_id = reference.id_embeddings()
        isolated_id = isolated.id_embeddings()
        reference_economic = reference.economic_coordinates()
        isolated_economic = isolated.economic_coordinates()
        reference_full = reference.propagated_embeddings()
        isolated_full = isolated.propagated_embeddings()

    def max_error(left, right):
        return max(float((a - b).abs().max()) for a, b in zip(left, right, strict=True))

    errors = {
        "id_embedding_max_abs_error": max_error(reference_id, isolated_id),
        "economic_embedding_max_abs_error": max_error(
            reference_economic, isolated_economic
        ),
        "full_embedding_max_abs_error": max_error(reference_full, isolated_full),
    }
    count = min(32, len(prepared["data"]["tr_u"]))
    batch_index = np.arange(count)
    users_np = prepared["data"]["tr_u"][batch_index]
    positives_np = prepared["data"]["tr_i"][batch_index]
    rng = np.random.default_rng(cfg.seed)
    negatives_np = legacy.m4_helpers.sample_uniform_negative_matrix(
        users_np,
        positives_np,
        prepared["data"]["n_items"],
        prepared["data"]["pos_key"],
        rng,
        k=cfg.negative_count,
    )
    users = torch.as_tensor(users_np, dtype=torch.long, device=v3.DEVICE)
    positives = torch.as_tensor(positives_np, dtype=torch.long, device=v3.DEVICE)
    negatives = torch.as_tensor(negatives_np, dtype=torch.long, device=v3.DEVICE)
    gradient_values = {}
    losses = {}
    for name, model in (("reference", reference), ("isolated", isolated)):
        model.zero_grad(set_to_none=True)
        loss = _weighted_batch_loss(
            model, prepared, cfg, users, positives, negatives
        )
        loss.backward()
        losses[name] = float(loss.detach())
        gradient_values[name] = {
            "id_user": model.E_u.weight.grad.detach().clone(),
            "id_item": model.E_i.weight.grad.detach().clone(),
            "c_v": model.value_scale_parameter.grad.detach().clone(),
            "c_n": model.profile_scale_parameter.grad.detach().clone(),
        }
    gradient_errors = {
        name: float(
            (gradient_values["reference"][name] - gradient_values["isolated"][name])
            .abs()
            .max()
        )
        for name in gradient_values["reference"]
    }
    gradient_norms = {
        name: float(value.norm()) for name, value in gradient_values["isolated"].items()
    }
    passed = bool(
        all(value <= tolerance for value in errors.values())
        and abs(losses["reference"] - losses["isolated"]) <= tolerance
        and all(value <= tolerance for value in gradient_errors.values())
        and all(value > 0.0 for value in gradient_norms.values())
    )
    result = {
        "device": str(v3.DEVICE),
        "tolerance": tolerance,
        **errors,
        "weighted_batch_loss_abs_error": abs(
            losses["reference"] - losses["isolated"]
        ),
        "gradient_max_abs_errors": gradient_errors,
        "isolated_gradient_norms": gradient_norms,
        "passed": passed,
    }
    del reference, isolated
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def operational_preflight(prepared: dict, cfg: M5SemanticNVM3ScreenConfig) -> dict:
    matched = prepared["m3_matched"]
    arms = matched.diagnostics["arms"]
    result = {
        "gate_preflight": prepared["m3_gate_preflight"],
        "m3_arms": arms,
        "all_target_reached": bool(
            all(values["target_reached"] for values in arms.values())
        ),
        "all_mass_errors_below_1e_8": bool(
            all(values["max_user_mass_abs_error"] < 1e-8 for values in arms.values())
        ),
    }
    result["identity"] = identity_preflight(prepared, cfg)
    result["passed"] = bool(
        result["gate_preflight"]["all_checks_pass"]
        and result["all_target_reached"]
        and result["all_mass_errors_below_1e_8"]
        and result["identity"]["passed"]
    )
    return result


def _row_from_arm(arm: dict) -> dict:
    return {
        "model_id": arm["model_id"],
        "role": arm["role"],
        "seed": arm["seed"],
        "split": arm["split"],
        "final_epoch": arm["final_epoch"],
        "rho": arm["rho"],
        "positive_weight_lambda": arm["positive_weight_lambda"],
        "clv_assignment": arm["clv_assignment"],
        **arm["diagnostics"],
        **arm["training"].get("final_diagnostics", {}),
        **arm["metrics"],
    }


def _all_better(left: dict, right: dict) -> bool:
    return all(float(left[metric]) > float(right[metric]) for metric in PRIMARY_METRICS)


def outcome_reading(metric_rows: dict[str, dict], operational: bool) -> dict:
    off = metric_rows[M3_OFF_MODEL_ID]
    actual = metric_rows[ACTUAL_MODEL_ID]
    shuffled = metric_rows[SHUFFLE_MODEL_ID]
    relation = metric_rows[RELATION_MODEL_ID]
    decisions = {
        "actual_beats_m3_off": _all_better(actual, off),
        "actual_beats_shuffle": _all_better(actual, shuffled),
        "actual_beats_relation_only": _all_better(actual, relation),
        "shuffle_beats_m3_off": _all_better(shuffled, off),
        "relation_only_beats_m3_off": _all_better(relation, off),
    }
    if not operational:
        outcome = "not_evaluable"
    elif not decisions["actual_beats_m3_off"]:
        outcome = "no_reason_to_add_m3"
    elif not decisions["actual_beats_relation_only"]:
        outcome = "relation_increment_without_clv_modulation_support"
    elif not decisions["actual_beats_shuffle"]:
        outcome = "m3_increment_without_clv_assignment_support"
    else:
        outcome = "conditional_complementarity"
    deltas = {
        reference: {
            metric: float(actual[metric] - values[metric])
            for metric in PRIMARY_METRICS
        }
        for reference, values in (
            ("vs_m3_off", off),
            ("vs_shuffle", shuffled),
            ("vs_relation_only", relation),
        )
    }
    accuracy = {
        reference: {
            metric: float(actual[metric] - values[metric])
            for metric in ACCURACY_METRICS
        }
        for reference, values in (
            ("vs_m3_off", off),
            ("vs_shuffle", shuffled),
            ("vs_relation_only", relation),
        )
    }
    return {
        "outcome": outcome,
        "conditional_complementarity": outcome == "conditional_complementarity",
        **decisions,
        "primary_deltas_actual_minus_reference": deltas,
        "accuracy_deltas_actual_minus_reference_reported_not_gated": accuracy,
        "statistical_note": (
            "one historical development seed; no significance or generalization claim"
        ),
    }


def _truth_flow_rows(
    reference_topk: np.ndarray,
    model_topk: np.ndarray,
    users: np.ndarray,
    prepared: dict,
    *,
    reference_id: str,
    model_id: str,
) -> list[dict]:
    entered_count = np.zeros(len(users), dtype=np.int64)
    exited_count = np.zeros(len(users), dtype=np.int64)
    entered_value = np.zeros(len(users), dtype=np.float64)
    exited_value = np.zeros(len(users), dtype=np.float64)
    for row, user in enumerate(users):
        truth_items = prepared["cache"].gt[int(user)]
        truth_values = prepared["cache"].rev[int(user)]
        truth = {int(item): float(value) for item, value in zip(truth_items, truth_values)}
        reference = set(map(int, reference_topk[row, :10]))
        model = set(map(int, model_topk[row, :10]))
        entered = (model - reference) & truth.keys()
        exited = (reference - model) & truth.keys()
        entered_count[row] = len(entered)
        exited_count[row] = len(exited)
        entered_value[row] = sum(truth[item] for item in entered)
        exited_value[row] = sum(truth[item] for item in exited)
    rows = []
    segments = np.asarray(prepared["cache"].seg, dtype=object)
    for group in ("전체", "저CLV", "중CLV", "고CLV"):
        mask = np.ones(len(users), dtype=bool) if group == "전체" else segments == group
        rows.append(
            {
                "reference": reference_id,
                "model_id": model_id,
                "group": group,
                "n_users": int(mask.sum()),
                "new_truth_entry_count@10": int(entered_count[mask].sum()),
                "lost_truth_exit_count@10": int(exited_count[mask].sum()),
                "new_truth_entry_purchase_amount@10": float(entered_value[mask].sum()),
                "lost_truth_exit_purchase_amount@10": float(exited_value[mask].sum()),
            }
        )
    return rows


def run_semantic_nv_m3_screen(
    cfg: M5SemanticNVM3ScreenConfig | None = None,
) -> pd.DataFrame:
    cfg = validate_config(cfg or configure_semantic_nv_m3_screen())
    summary = preflight_summary(cfg)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    prepared = _prepare(cfg)
    operational = operational_preflight(prepared, cfg)
    print("\n학습 전 불변식 점검")
    print(json.dumps(operational, ensure_ascii=False, indent=2))
    if not operational["passed"]:
        raise RuntimeError("M3 강도·총량·gate·동일성 사전점검 실패로 학습을 중단합니다")

    arms: dict[str, dict] = {}
    models: dict[str, M5SemanticNVIsolatedFirstHopLightGCN] = {}
    with patch.object(legacy, "_build_model", _build_model):
        for spec in arm_specifications(prepared, cfg):
            print(
                f"\n===== {spec['model_id']} | seed {cfg.seed} | "
                f"fixed {cfg.epochs} epochs ====="
            )
            arms[spec["model_id"]], models[spec["model_id"]] = legacy._run_arm(
                prepared, cfg, spec
            )

    frame = pd.DataFrame([_row_from_arm(arms[model_id]) for model_id in MODEL_IDS])
    metric_rows = {model_id: arms[model_id]["metrics"] for model_id in MODEL_IDS}
    comparison = report_helpers._metric_comparison(
        metric_rows,
        references=(M3_OFF_MODEL_ID, SHUFFLE_MODEL_ID, RELATION_MODEL_ID),
    )

    users = None
    topk = {}
    score_rows = []
    for model_id in MODEL_IDS:
        arm_users, arm_top50 = report_helpers._masked_topk(
            models[model_id], prepared, max_k=cfg.diagnostic_max_k
        )
        if users is None:
            users = arm_users
        elif not np.array_equal(users, arm_users):
            raise RuntimeError("arm별 평가 사용자 배열이 다릅니다")
        topk[model_id] = arm_top50
        score_rows.append(
            legacy._score_diagnostics(
                models[model_id], arm_users, arm_top50, model_id=model_id
            )
        )
    assert users is not None

    overlap_frames = []
    truth_rows = []
    for model_id in (ACTUAL_MODEL_ID, SHUFFLE_MODEL_ID, RELATION_MODEL_ID):
        overlap = report_helpers.topk_overlap_summary(
            topk[M3_OFF_MODEL_ID],
            topk[model_id],
            prepared["cache"].seg,
            k=10,
        )
        overlap.insert(0, "reference", M3_OFF_MODEL_ID)
        overlap.insert(1, "model_id", model_id)
        overlap_frames.append(overlap)
        truth_rows.extend(
            _truth_flow_rows(
                topk[M3_OFF_MODEL_ID],
                topk[model_id],
                users,
                prepared,
                reference_id=M3_OFF_MODEL_ID,
                model_id=model_id,
            )
        )
    overlap_frame = pd.concat(overlap_frames, ignore_index=True)
    truth_frame = pd.DataFrame(truth_rows)
    score_frame = pd.DataFrame(score_rows)
    actual_changed = float(
        overlap_frame.loc[
            (overlap_frame["model_id"] == ACTUAL_MODEL_ID)
            & (overlap_frame["group"] == "전체"),
            "top10_set_changed_user_share",
        ].iloc[0]
    )
    operational["actual_top10_set_changed_user_share_vs_m3_off"] = actual_changed
    operational["actual_top10_intervention_nonzero"] = actual_changed > 0.0
    operational["passed_after_training"] = bool(
        operational["passed"] and actual_changed > 0.0
    )
    reading = outcome_reading(metric_rows, operational["passed_after_training"])

    cached_diagnostic = None
    cached_path = Path(cfg.cached_lineage_b_result_json)
    if cfg.cached_lineage_b_result_json and cached_path.exists():
        cached = json.loads(cached_path.read_text(encoding="utf-8"))
        cached_diagnostic = {
            "path": str(cached_path),
            "code_version": cached.get("code_version"),
            "same_run_comparison": False,
            "note": "diagnostic only; all formal increments use the newly trained M3-off arm",
        }

    out = Path(cfg.out_dir)
    stem = f"m5_semantic_nv_isolated_m3_{prepared['config_hash']}"
    paths = {
        "absolute_csv": out / f"{stem}.csv",
        "comparison_csv": out / f"{stem}_comparison.csv",
        "score_diagnostics_csv": out / f"{stem}_score_diagnostics.csv",
        "top10_overlap_csv": out / f"{stem}_top10_overlap.csv",
        "truth_flow_csv": out / f"{stem}_truth_flow.csv",
        "json": out / f"{stem}.json",
    }
    legacy.test10._atomic_csv(paths["absolute_csv"], frame)
    legacy.test10._atomic_csv(paths["comparison_csv"], comparison)
    legacy.test10._atomic_csv(paths["score_diagnostics_csv"], score_frame)
    legacy.test10._atomic_csv(paths["top10_overlap_csv"], overlap_frame)
    legacy.test10._atomic_csv(paths["truth_flow_csv"], truth_frame)
    legacy.test10._atomic_json(
        paths["json"],
        {
            "code_version": CODE_VERSION,
            "source_revision": prepared["revision"],
            "config": asdict(cfg),
            "preflight": summary,
            "input_manifest": prepared["manifest"],
            "data_stats": prepared["data"].get("data_stats", {}),
            "gate_preflight": prepared["m3_gate_preflight"],
            "m3_graph_diagnostics": prepared["m3_matched"].diagnostics,
            "operational_checks": operational,
            "absolute_rows": frame.to_dict("records"),
            "comparison_rows": comparison.to_dict("records"),
            "score_diagnostic_rows": score_frame.to_dict("records"),
            "top10_overlap_rows": overlap_frame.to_dict("records"),
            "truth_flow_rows": truth_frame.to_dict("records"),
            "outcome_reading": reading,
            "arms": arms,
            "m3_shuffle_source_user": prepared["m3_source_user"].tolist(),
            "cached_lineage_b_diagnostic": cached_diagnostic,
            "result_paths": {key: str(value) for key, value in paths.items()},
        },
    )
    frame.attrs["comparison"] = comparison
    frame.attrs["score_diagnostics"] = score_frame
    frame.attrs["top10_overlap"] = overlap_frame
    frame.attrs["truth_flow"] = truth_frame
    frame.attrs["operational_checks"] = operational
    frame.attrs["outcome_reading"] = reading
    frame.attrs["result_paths"] = {key: str(value) for key, value in paths.items()}

    print("\n1) B′·actual·M3 순열·관계-only 절대지표")
    print(frame.to_string(index=False))
    print("\n2) 대조군별 전체 지표 비교")
    print(comparison.to_string(index=False))
    print("\n3) Top-10 변화 및 정답 진입·이탈")
    print(overlap_frame.to_string(index=False))
    print(truth_frame.to_string(index=False))
    print("\n4) 사전 고정 판독")
    print(json.dumps(reading, ensure_ascii=False, indent=2))
    print("\n5) 저장 파일")
    print(json.dumps(frame.attrs["result_paths"], ensure_ascii=False, indent=2))
    return frame


if __name__ == "__main__":
    print(
        json.dumps(
            preflight_summary(configure_semantic_nv_m3_screen()),
            ensure_ascii=False,
            indent=2,
        )
    )
