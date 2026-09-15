"""Post-hoc seed-42 GraphSAGE screen for M2 attribution and a simpler M4.

The completed GraphSAGE factorial run made M2 the only promising standalone
intervention, while the item-directed M4 pushed recommendations toward a
globally higher price level.  This runner therefore keeps the GraphSAGE M2
unchanged and replaces M4 with a user-only historical-CLV row weight.

This is an exposed historical-development screen, not a final test.  It first
tries to reuse the strictly matched M1/M2 rows from run ``2d01d43c6d43``.  If
that JSON is not available in the mounted Drive, it trains M1 and M2 in the
same run instead of failing with a missing-file error.
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

from clv_scaled_value_basis_gnn_model import CLVScaledValueBasisGNN
import gnn_clv_m1_m2_m4_m5_factorial_screen as factorial
import lightgcn_clv_axis_specific_test10 as test10
import lightgcn_clv_gradient_isolated_economic_interaction as report_helpers
import lightgcn_clv_m5_clv_scaled_value_basis_one_arm as value_basis_source
import lightgcn_clv_m5_economic_positive_weight as legacy
import lightgcn_clv_moe as moe
import lightgcn_clv_v3 as v3


CODE_VERSION = "graphsage-clv-m2-user-clv-weight-development-screen-v1"
REFERENCE_CODE_VERSION = factorial.CODE_VERSION
REFERENCE_RESULT_ID = "2d01d43c6d43"
CORE_METRICS = factorial.CORE_METRICS
TOP10_METRICS = factorial.TOP10_METRICS


def model_ids() -> dict[str, str]:
    return {
        "m1": "graphsage_m1_id67_multineg_mean_k5",
        "m2": "graphsage_m2_clv_scaled_value_basis_multineg_mean_k5",
        "m2_shuffle": (
            "graphsage_m2_clv_scaled_value_basis_degree_matched_assignment_control_k5"
        ),
        "m4": "graphsage_m4_user_clv_positive_weight_id67_k5",
        "m5": "graphsage_m5_clv_scaled_value_basis_user_clv_positive_weight_k5",
    }


@dataclass(frozen=True)
class GraphSAGEM2UserCLVWeightConfig:
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
    shuffle_degree_bins: int = 10
    shuffle_seed: int = 42
    out_dir: str = ""
    reference_result_json: str = ""
    reference_search_root: str = "/content/drive/MyDrive/논문/data"
    retrain_references_if_missing: bool = True


def configure_graphsage_m2_user_clv_weight_screen(
    **overrides,
) -> GraphSAGEM2UserCLVWeightConfig:
    root = v3.default_out_dir("dunnhumby")
    defaults = {
        "out_dir": f"{root}_graphsage_m2_user_clv_weight_development_screen_v1",
        "reference_result_json": (
            f"{root}_graphsage_clv_m1_m2_m4_m5_factorial_development_screen_v1/"
            f"graphsage_clv_factorial_{REFERENCE_RESULT_ID}.json"
        ),
    }
    return validate_config(
        GraphSAGEM2UserCLVWeightConfig(**(defaults | overrides))
    )


def validate_config(
    cfg: GraphSAGEM2UserCLVWeightConfig,
) -> GraphSAGEM2UserCLVWeightConfig:
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
        "shuffle_degree_bins": 10,
        "shuffle_seed": 42,
        "retrain_references_if_missing": True,
    }
    for key, expected in fixed.items():
        if getattr(cfg, key) != expected:
            raise ValueError(f"GraphSAGE 후속 screen은 {key}={expected!r}이어야 합니다")
    if cfg.batch_size <= 0 or cfg.lr <= 0 or cfg.pref_reg < 0:
        raise ValueError("학습 설정이 잘못됐습니다")
    if not cfg.out_dir or not cfg.reference_result_json:
        raise ValueError("출력 또는 참조 경로가 비어 있습니다")
    return cfg


def preflight_summary(cfg: GraphSAGEM2UserCLVWeightConfig) -> dict:
    cfg = validate_config(cfg)
    ids = model_ids()
    return {
        "code_version": CODE_VERSION,
        "dataset": cfg.dataset,
        "backbone": "graphsage",
        "seed": cfg.seed,
        "split": "historical_development_days_684_690",
        "trained_models": [ids["m2_shuffle"], ids["m4"], ids["m5"]],
        "conditionally_retrained_if_reference_missing": [ids["m1"], ids["m2"]],
        "reference_result_id": REFERENCE_RESULT_ID,
        "research_question": (
            "Does the promising GraphSAGE M2 depend on the correct user assignment, "
            "and can a user-only historical-CLV loss weight preserve that M2 signal "
            "better than the rejected item-directed M4?"
        ),
        "m2_unchanged": {
            "user_coordinates": "sqrt(0.05)*q_C(u)*b(q_V(u))",
            "item_coordinates": "sqrt(0.05)*b(item amount percentile)",
            "q_c": "midrank percentile of raw n_u*v_u",
            "basis": "fixed L2-normalized Gaussian RBF at [0,0.5,1]",
            "propagation": "inside both GraphSAGE layers",
            "same_recommendation_gradient": True,
            "external_reranking": False,
        },
        "m2_assignment_control": {
            "operation": "jointly permute q_C, q_V and validity within binary user-degree deciles",
            "item_economic_attributes": "observed and fixed",
            "purpose": "separate correct CLV assignment from fixed-coordinate/capacity effects",
        },
        "modified_m4": {
            "raw_weight": "1 + 0.5*q_C(u)",
            "normalization": "divide by mean over all positive training rows",
            "item_price_or_bin_fit_in_weight": False,
            "purpose": "prioritize errors of higher historical-CLV users without naming an item direction",
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
            "m2_assignment_signal": "observed M2 > shuffled M2 on all four Top-10 metrics",
            "modified_m4_signal": "modified M4 > M1 on all four Top-10 metrics",
            "m5_vs_m2": "modified M5 > observed M2 on all four Top-10 metrics",
            "m5_vs_m4": "modified M5 > modified M4 on all four Top-10 metrics",
            "m5_vs_baseline": "modified M5 > M1 on all four Top-10 metrics",
            "screen_pass": "m2_assignment_signal and m5_vs_m2 and m5_vs_baseline",
            "statistical_note": (
                "post-hoc one-seed exposed historical-development screen; no significance, "
                "stability, generalization, attribution confirmation, or final model claim"
            ),
        },
        "rule_change_disclosure": (
            "The original cross-backbone rule allowed assignment controls only after an M5 "
            "passed. All three original M5s failed. This explicitly post-result follow-up "
            "instead investigates the uniquely positive GraphSAGE M2 and a new M4 formula."
        ),
        "out_dir": cfg.out_dir,
    }


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _config_hash(cfg: GraphSAGEM2UserCLVWeightConfig, prepared: dict) -> str:
    payload = {
        "code_version": CODE_VERSION,
        "config": asdict(cfg),
        "input_hash": prepared["input_hash"],
        "source_revision": prepared["revision"],
    }
    return hashlib.sha256(_canonical(payload).encode()).hexdigest()[:12]


def degree_matched_joint_assignment(
    prepared: dict, *, seed: int, n_bins: int
) -> dict:
    """Jointly permute q_C/q_V tuples inside binary user-degree deciles."""

    data = prepared["data"]
    train_edges = data["train"][["u_idx", "i_idx"]].drop_duplicates()
    user_degree = np.bincount(
        train_edges["u_idx"].to_numpy(np.int64), minlength=data["n_users"]
    )
    valid = np.asarray(prepared["clv_valid"], dtype=bool)
    valid_index = np.flatnonzero(valid & (user_degree > 0))
    if len(valid_index) < 2:
        raise RuntimeError("CLV 순열에 사용할 유효 고객이 부족합니다")

    ranks = pd.Series(user_degree[valid_index]).rank(method="average").to_numpy()
    strata_valid = np.floor((ranks - 0.5) * n_bins / len(valid_index)).astype(
        np.int16
    )
    strata_valid = np.minimum(strata_valid, n_bins - 1)
    strata = np.full(data["n_users"], -1, dtype=np.int16)
    strata[valid_index] = strata_valid
    source = np.arange(data["n_users"], dtype=np.int64)
    rng = np.random.default_rng(seed)
    for stratum in np.unique(strata_valid):
        target = valid_index[strata_valid == stratum]
        if len(target) < 2:
            continue
        permuted = rng.permutation(target)
        if np.array_equal(permuted, target):
            permuted = np.roll(permuted, 1)
        source[target] = permuted

    changed = valid_index[source[valid_index] != valid_index]
    if not len(changed):
        raise RuntimeError("degree-matched CLV 순열이 고객 배정을 바꾸지 못했습니다")
    if np.any(strata[changed] != strata[source[changed]]):
        raise RuntimeError("CLV 순열이 user-degree 구간을 벗어났습니다")

    assignment = {
        "q_v": np.asarray(prepared["q_v"], dtype=np.float32)[source].copy(),
        "q_c": np.asarray(prepared["q_c"], dtype=np.float32)[source].copy(),
        "clv_valid": valid[source].copy(),
        "source_user": source,
        "stratum": strata,
        "user_degree": user_degree,
        "changed_valid_user_share": float(len(changed) / len(valid_index)),
    }
    if not np.array_equal(
        np.sort(assignment["q_c"]), np.sort(np.asarray(prepared["q_c"]))
    ):
        raise RuntimeError("q_C 순열이 값의 multiset을 보존하지 못했습니다")
    if not np.array_equal(
        np.sort(assignment["q_v"]), np.sort(np.asarray(prepared["q_v"]))
    ):
        raise RuntimeError("q_V 순열이 값의 multiset을 보존하지 못했습니다")
    return assignment


def _prepare(cfg: GraphSAGEM2UserCLVWeightConfig) -> dict:
    prepared = value_basis_source._prepare(cfg)
    prepared["config_hash"] = _config_hash(cfg, prepared)
    prepared["economic_input_diagnostics"] = {}
    prepared["degree_matched_assignment"] = degree_matched_joint_assignment(
        prepared, seed=cfg.shuffle_seed, n_bins=cfg.shuffle_degree_bins
    )
    return prepared


def arm_specifications(
    prepared: dict, cfg: GraphSAGEM2UserCLVWeightConfig
) -> list[dict]:
    ids = model_ids()
    observed = {
        "q_v": prepared["q_v"],
        "q_c": prepared["q_c"],
        "clv_valid": prepared["clv_valid"],
    }
    shuffled = prepared["degree_matched_assignment"]
    return [
        {
            "key": "m1",
            "model_id": ids["m1"],
            "role": "reference_m1",
            "rho": 0.0,
            "m2_active": False,
            "weighted": False,
            "assignment": observed,
            "assignment_name": "no_clv_intervention",
        },
        {
            "key": "m2",
            "model_id": ids["m2"],
            "role": "reference_observed_m2",
            "rho": cfg.rho,
            "m2_active": True,
            "weighted": False,
            "assignment": observed,
            "assignment_name": "observed_q_c_q_v",
        },
        {
            "key": "m2_shuffle",
            "model_id": ids["m2_shuffle"],
            "role": "m2_degree_matched_assignment_control",
            "rho": cfg.rho,
            "m2_active": True,
            "weighted": False,
            "assignment": shuffled,
            "assignment_name": "degree_matched_joint_q_c_q_v_shuffle",
        },
        {
            "key": "m4",
            "model_id": ids["m4"],
            "role": "modified_m4_user_clv_weight",
            "rho": 0.0,
            "m2_active": False,
            "weighted": True,
            "assignment": observed,
            "assignment_name": "observed_user_q_c_weight",
        },
        {
            "key": "m5",
            "model_id": ids["m5"],
            "role": "observed_m2_plus_modified_m4",
            "rho": cfg.rho,
            "m2_active": True,
            "weighted": True,
            "assignment": observed,
            "assignment_name": "observed_q_c_q_v_and_user_q_c_weight",
        },
    ]


def _build_model(
    prepared: dict, cfg: GraphSAGEM2UserCLVWeightConfig, spec: dict
) -> CLVScaledValueBasisGNN:
    data = prepared["data"]
    assignment = spec["assignment"]
    v3.set_seed(cfg.seed)
    return CLVScaledValueBasisGNN(
        backbone="graphsage",
        m2_active=spec["m2_active"],
        n_users=data["n_users"],
        n_items=data["n_items"],
        user_q_v=assignment["q_v"],
        user_q_c=assignment["q_c"],
        user_clv_valid=assignment["clv_valid"],
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


def user_clv_train_weight_normalizer(
    prepared: dict, assignment: dict, lambda_: float
) -> float:
    train_users = np.asarray(prepared["data"]["tr_u"], dtype=np.int64)
    q_c = np.asarray(assignment["q_c"], dtype=np.float32)[train_users]
    raw = 1.0 + lambda_ * q_c
    mean = float(raw.mean())
    if not np.isfinite(mean) or mean <= 0.0:
        raise RuntimeError("사용자 CLV 양성 가중치 정규화값이 잘못됐습니다")
    return mean


def user_clv_positive_row_weights(
    q_c: torch.Tensor,
    _item_amount: torch.Tensor,
    *,
    train_mean_raw_weight: float,
    lambda_: float,
) -> torch.Tensor:
    """Normalized positive-row weights with no item-direction term."""

    if train_mean_raw_weight <= 0.0:
        raise ValueError("train_mean_raw_weight는 양수여야 합니다")
    return (1.0 + lambda_ * q_c) / train_mean_raw_weight


def _candidate_reference_paths(cfg: GraphSAGEM2UserCLVWeightConfig) -> list[Path]:
    exact = Path(cfg.reference_result_json)
    paths = [exact]
    root = Path(cfg.reference_search_root)
    if not exact.exists() and root.exists():
        paths.extend(
            sorted(root.rglob(f"graphsage_clv_factorial_{REFERENCE_RESULT_ID}.json"))
        )
    unique = []
    seen = set()
    for path in paths:
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def _validated_reference_rows(
    cfg: GraphSAGEM2UserCLVWeightConfig, prepared: dict
) -> tuple[dict[str, dict], dict, str] | None:
    ids = model_ids()
    for path in _candidate_reference_paths(cfg):
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("code_version") != REFERENCE_CODE_VERSION:
                raise ValueError("code_version mismatch")
            old_cfg = payload.get("config", {})
            required = {
                "backbone": "graphsage",
                "dataset": cfg.dataset,
                "seed": cfg.seed,
                "time_cutoff": cfg.time_cutoff,
                "evaluation_days": cfg.evaluation_days,
                "epochs": cfg.epochs,
                "id_dim": cfg.id_dim,
                "economic_dim": cfg.economic_dim,
                "rho": cfg.rho,
                "n_layers": cfg.n_layers,
                "negative_count": cfg.negative_count,
            }
            if any(old_cfg.get(key) != value for key, value in required.items()):
                raise ValueError("training configuration mismatch")
            old_manifest = payload.get("input_manifest")
            if old_manifest is None or moe.manifest_hash(old_manifest) != prepared["input_hash"]:
                raise ValueError("input manifest mismatch")
            indexed = payload.get("arms", {})
            rows = {key: indexed[ids[key]] for key in ("m1", "m2")}
            context = {
                key: indexed[ids[key]]
                for key in ("m4", "m5")
                if ids.get(key) in indexed
            }
            return rows, context, str(path)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            print(f"  [reference skip] {path}: {error}")
    return None


def _run_with_user_weight(
    prepared: dict, cfg: GraphSAGEM2UserCLVWeightConfig, spec: dict
) -> tuple[dict, CLVScaledValueBasisGNN]:
    with (
        patch.object(legacy, "_build_model", _build_model),
        patch.object(
            legacy, "_train_weight_normalizer", user_clv_train_weight_normalizer
        ),
        patch.object(legacy, "positive_row_weights", user_clv_positive_row_weights),
    ):
        return legacy._run_arm(prepared, cfg, spec)


def _result_row(arm: dict, *, origin: str) -> dict:
    return {
        "model_id": arm["model_id"],
        "role": arm["role"],
        "training_origin": origin,
        "backbone": "graphsage",
        "seed": arm["seed"],
        "split": arm["split"],
        "final_epoch": arm["final_epoch"],
        "rho": arm["rho"],
        "positive_weight_lambda": arm["positive_weight_lambda"],
        **arm.get("diagnostics", {}),
        **arm.get("training", {}).get("final_diagnostics", {}),
        **arm["metrics"],
    }


def screening_reading(metric_rows: dict[str, dict]) -> dict:
    ids = model_ids()

    def deltas(model: str, reference: str) -> dict[str, float]:
        return {
            metric: float(
                metric_rows[ids[model]][metric] - metric_rows[ids[reference]][metric]
            )
            for metric in TOP10_METRICS
        }

    m2_vs_shuffle = deltas("m2", "m2_shuffle")
    m4_vs_m1 = deltas("m4", "m1")
    m5_vs_m1 = deltas("m5", "m1")
    m5_vs_m2 = deltas("m5", "m2")
    m5_vs_m4 = deltas("m5", "m4")
    positive = lambda values: all(value > 0.0 for value in values.values())
    flags = {
        "m2_assignment_signal": positive(m2_vs_shuffle),
        "modified_m4_beats_m1_all_four_top10_metrics": positive(m4_vs_m1),
        "modified_m5_beats_m1_all_four_top10_metrics": positive(m5_vs_m1),
        "modified_m5_beats_observed_m2_all_four_top10_metrics": positive(m5_vs_m2),
        "modified_m5_beats_modified_m4_all_four_top10_metrics": positive(m5_vs_m4),
    }
    flags["mechanism_screen_pass"] = bool(
        flags["m2_assignment_signal"]
        and flags["modified_m5_beats_m1_all_four_top10_metrics"]
        and flags["modified_m5_beats_observed_m2_all_four_top10_metrics"]
    )
    return {
        "top10_deltas": {
            "observed_m2_minus_shuffle": m2_vs_shuffle,
            "modified_m4_minus_m1": m4_vs_m1,
            "modified_m5_minus_m1": m5_vs_m1,
            "modified_m5_minus_observed_m2": m5_vs_m2,
            "modified_m5_minus_modified_m4": m5_vs_m4,
        },
        **flags,
        "decision_scope": "post-hoc single-seed historical-development mechanism screen",
        "statistical_note": (
            "one exposed historical development seed; no significance, stability, "
            "generalization, confirmed attribution, or final model claim"
        ),
    }


def run_graphsage_m2_user_clv_weight_screen(
    cfg: GraphSAGEM2UserCLVWeightConfig | None = None,
) -> pd.DataFrame:
    cfg = validate_config(cfg or configure_graphsage_m2_user_clv_weight_screen())
    summary = preflight_summary(cfg)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    prepared = _prepare(cfg)
    ids = model_ids()
    specs = {spec["key"]: spec for spec in arm_specifications(prepared, cfg)}

    arms: dict[str, dict] = {}
    models: dict[str, CLVScaledValueBasisGNN] = {}
    prior_context: dict[str, dict] = {}
    reference = _validated_reference_rows(cfg, prepared)
    if reference is not None:
        reference_rows, prior_context, reference_path = reference
        print(f"\n[reference] matched M1·M2 재사용: {reference_path}")
        for key, row in reference_rows.items():
            arms[ids[key]] = dict(row) | {
                "role": specs[key]["role"],
                "reference_path": reference_path,
            }
    elif cfg.retrain_references_if_missing:
        print(
            "\n[reference] Drive에 완료 M1·M2 JSON이 없어 중단하지 않고 "
            "두 arm을 동일 실행에서 새로 학습합니다."
        )
        for key in ("m1", "m2"):
            spec = specs[key]
            print(f"\n===== {spec['model_id']} | seed {cfg.seed} | fixed 100 epochs =====")
            arm, model = _run_with_user_weight(prepared, cfg, spec)
            arms[spec["model_id"]] = arm
            models[spec["model_id"]] = model
    else:
        raise FileNotFoundError("완료 GraphSAGE M1·M2 참조 결과를 찾지 못했습니다")

    for key in ("m2_shuffle", "m4", "m5"):
        spec = specs[key]
        print(f"\n===== {spec['model_id']} | seed {cfg.seed} | fixed 100 epochs =====")
        arm, model = _run_with_user_weight(prepared, cfg, spec)
        arms[spec["model_id"]] = arm
        models[spec["model_id"]] = model

    rows = []
    metric_rows = {}
    for key, model_id in ids.items():
        arm = arms[model_id]
        metric_rows[model_id] = arm["metrics"]
        origin = (
            "reused_matched_run_2d01d43c6d43"
            if key in {"m1", "m2"} and model_id not in models
            else "trained_in_current_run"
        )
        rows.append(_result_row(arm, origin=origin))
    frame = pd.DataFrame(rows)
    comparison = report_helpers._metric_comparison(
        metric_rows, references=(ids["m1"], ids["m2"], ids["m4"])
    )
    interactions = factorial.interaction_rows(metric_rows, ids)
    reading = screening_reading(metric_rows)

    counterfactual_rows = []
    overlap_frames = []
    for key in ("m2", "m2_shuffle", "m5"):
        model_id = ids[key]
        if model_id not in models:
            continue
        diagnostics, overlap = factorial._counterfactual_m2_diagnostics(
            models[model_id],
            prepared,
            model_id=model_id,
            max_k=cfg.diagnostic_max_k,
        )
        counterfactual_rows.append(diagnostics)
        overlap_frames.append(overlap)
    counterfactual = pd.DataFrame(counterfactual_rows)
    overlap = (
        pd.concat(overlap_frames, ignore_index=True)
        if overlap_frames
        else pd.DataFrame()
    )

    out = Path(cfg.out_dir)
    stem = f"graphsage_m2_user_clv_weight_{prepared['config_hash']}"
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
    test10._atomic_csv(paths["counterfactual_m2_csv"], counterfactual)
    test10._atomic_csv(paths["top10_overlap_csv"], overlap)
    test10._atomic_json(
        paths["json"],
        {
            "code_version": CODE_VERSION,
            "source_revision": prepared["revision"],
            "config": asdict(cfg),
            "preflight": summary,
            "input_manifest": prepared["manifest"],
            "shuffle_diagnostics": {
                key: value
                for key, value in prepared["degree_matched_assignment"].items()
                if key in {"changed_valid_user_share"}
            },
            "absolute_rows": frame.to_dict("records"),
            "comparison_rows": comparison.to_dict("records"),
            "interaction_rows": interactions.to_dict("records"),
            "counterfactual_m2_rows": counterfactual.to_dict("records"),
            "top10_overlap_rows": overlap.to_dict("records"),
            "screening_reading": reading,
            "prior_item_directed_m4_m5_context": prior_context,
            "arms": arms,
            "result_paths": {key: str(value) for key, value in paths.items()},
        },
    )
    frame.attrs["comparison"] = comparison
    frame.attrs["interaction"] = interactions
    frame.attrs["counterfactual_m2"] = counterfactual
    frame.attrs["top10_overlap"] = overlap
    frame.attrs["decision"] = reading
    frame.attrs["preflight"] = summary
    frame.attrs["result_paths"] = {key: str(value) for key, value in paths.items()}

    print("\n1) GraphSAGE M1·M2·M2 순열·수정 M4·수정 M5 핵심 절대지표")
    columns = ["model_id", "role", *CORE_METRICS]
    print(frame[[column for column in columns if column in frame]].to_string(index=False))
    print("\n2) M1·M2·수정 M4 대비 전체 지표 비교")
    print(comparison.to_string(index=False))
    print("\n3) 수정 M4 기준 2x2 상호작용")
    print(interactions.to_string(index=False))
    print("\n4) 학습된 새 M2 경로의 입력 on/off 진단")
    print(counterfactual.to_string(index=False))
    print(overlap.to_string(index=False))
    print("\n5) 사전 고정 판독")
    print(json.dumps(reading, ensure_ascii=False, indent=2))
    print("\n6) 저장 파일")
    print(json.dumps(frame.attrs["result_paths"], ensure_ascii=False, indent=2))
    return frame


if __name__ == "__main__":
    print(
        "Import this module from the dedicated Colab notebook and call "
        "run_graphsage_m2_user_clv_weight_screen()."
    )
