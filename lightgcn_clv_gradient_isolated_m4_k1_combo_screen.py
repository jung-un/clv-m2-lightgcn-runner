"""Seed-43 M2-GI + K=1 personalized-positive-row M4 combination screen.

Only the two previously untrained arms are fitted here.  Exact seed-43 M1 and
M4 metrics are read from the completed K=1 M4 multi-seed development run.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from clv_gradient_isolated_economic_interaction_model import (
    GradientIsolatedCLVEconomicInteractionLightGCN,
)
from clv_run_state import ProgressStore, RunIdentity, clone_state, file_sha256
import lightgcn_clv_axis_specific_test10 as test10
import lightgcn_clv_gradient_isolated_economic_interaction as gi
import lightgcn_clv_m4_k1_assignment_control_multiseed as m4_multi
import lightgcn_clv_m4_k1_assignment_control_screen as m4_single
import lightgcn_clv_m5_economic_positive_weight as economic
import lightgcn_clv_m5_k1_m4_improvement_screen as weighted_training
import lightgcn_clv_m5_nv_economic_positive_weight as nv_economic
import lightgcn_clv_moe as moe
import lightgcn_clv_v3 as v3


CODE_VERSION = "m5-gradient-isolated-m2-personalized-m4-k1-seed43-screen-v1"
M1_MODEL_ID = "m1_bpr_k1_reused_seed43"
M2_MODEL_ID = "m2_gradient_isolated_clv_economic_interaction_bpr_k1"
M4_MODEL_ID = "m4_personalized_positive_weight_actual_qc_bpr_k1_reused_seed43"
M5_MODEL_ID = "m5_gradient_isolated_m2_personalized_m4_bpr_k1"
MODEL_IDS = (M1_MODEL_ID, M2_MODEL_ID, M4_MODEL_ID, M5_MODEL_ID)
ACCURACY_METRICS = (
    "recall@10", "ndcg@10", "recall@20", "ndcg@20", "recall@50", "ndcg@50"
)
ECONOMIC_METRICS = (
    "price_purchase_amount_weighted_hit@10",
    "vndcg@10",
)


@dataclass(frozen=True)
class GradientIsolatedM4ComboConfig(gi.GradientIsolatedConfig):
    seed: int = 43
    positive_weight_lambda: float = 0.5
    economic_bins: int = 4
    shrinkage_strength: float = 10.0
    negative_count: int = 1
    reference_result_dir: str = ""


def configure_combo_screen(**overrides) -> GradientIsolatedM4ComboConfig:
    root = v3.default_out_dir("dunnhumby")
    defaults = {
        "out_dir": f"{root}_m5_gradient_isolated_m4_k1_seed43_screen_v1",
        "baseline_result_dir": f"{root}_m2_repeatshare_historical_backtest_v1",
        "reference_result_dir": f"{root}_m4_k1_assignment_control_development_multiseed_v1",
    }
    return validate_config(GradientIsolatedM4ComboConfig(**(defaults | overrides)))


def validate_config(cfg: GradientIsolatedM4ComboConfig) -> GradientIsolatedM4ComboConfig:
    fixed = {
        "dataset": "dunnhumby", "seed": 43, "time_cutoff": 690,
        "evaluation_days": 7, "epochs": 100, "id_dim": 64,
        "relation_dim": 3, "rho": 0.05, "beta": 0.25, "delta": 0.25,
        "eta": 0.5, "price_epsilon": 0.5, "n_layers": 2,
        "input_days": 365, "diagnostic_max_k": 50,
        "positive_weight_lambda": 0.5, "economic_bins": 4,
        "shrinkage_strength": 10.0, "negative_count": 1,
    }
    for name, expected in fixed.items():
        if getattr(cfg, name) != expected:
            raise ValueError(f"GI-M2+M4 screen은 {name}={expected!r}이어야 합니다")
    if cfg.batch_size <= 0 or cfg.lr <= 0 or cfg.pref_reg < 0:
        raise ValueError("학습 설정이 잘못됐습니다")
    if not cfg.out_dir or not cfg.baseline_result_dir or not cfg.reference_result_dir:
        raise ValueError("결과·기준 경로가 모두 필요합니다")
    return cfg


def preflight_summary(cfg: GradientIsolatedM4ComboConfig) -> dict:
    cfg = validate_config(cfg)
    return {
        "code_version": CODE_VERSION,
        "dataset": cfg.dataset,
        "seed": cfg.seed,
        "split": "historical_development_days_684_690",
        "reused_models": [M1_MODEL_ID, M4_MODEL_ID],
        "trained_models": [M2_MODEL_ID, M5_MODEL_ID],
        "research_question": (
            "Does the unchanged gradient-isolated CLV/economic M2 replicate at seed 43, "
            "and does the unchanged K=1 personalized positive-row M4 add value to it?"
        ),
        "m2": {
            "conditions": "[q_N,q_V,q_C] from train-only historical CLV inputs",
            "score": "ID dot + bounded relation dot + fixed-sign value-price dot",
            "gradient_isolation": "auxiliary paths read detached final ID embeddings",
            "rho": cfg.rho, "beta": cfg.beta, "delta": cfg.delta, "eta": cfg.eta,
        },
        "m4": {
            "raw_positive_row_weight": (
                "1 + 0.5*q_C*item_amount_percentile*clipped_user_bin_fit"
            ),
            "normalization": "divide by the mean raw weight over all train rows",
        },
        "fixed": {
            "new_item_task": True,
            "train_pairs_excluded_from_truth_and_candidates": True,
            "min_item_interactions": 1,
            "graph": "binary",
            "negative_sampling": "one uniformly sampled unseen item",
            "epochs": cfg.epochs,
            "validation_or_epoch_selection": False,
            "final_test_constructed": False,
            "holdout_constructed": False,
            "external_reranking": False,
            "one_training_loop_and_optimizer_per_trained_arm": True,
        },
        "reading_rule": {
            "m2_replication": "M2 > M1 on both top-10 economic metrics",
            "combination": "M5 > max(M2,M4) on both top-10 economic metrics",
            "accuracy": "M5 six-metric accuracy geometric mean >= max(M2,M4)",
            "guard": "each M5 accuracy metric >= 99% of M1",
            "attribution": "not tested in this directional screen",
        },
        "statistical_note": "one exposed development seed; no significance or generalization claim",
        "out_dir": cfg.out_dir,
    }


class GradientIsolatedComboModel(GradientIsolatedCLVEconomicInteractionLightGCN):
    """GI model with aliases used by the common fixed-row-weight trainer."""

    def propagated_embeddings(self):
        return self.component_embeddings()

    def sampled_l2(self, users, positives, negatives):
        return self.batch_l2(users, positives, negatives)


def _config_hash(cfg, input_hash: str, revision: str) -> str:
    payload = {
        "code_version": CODE_VERSION,
        "config": asdict(cfg),
        "input_hash": input_hash,
        "source_revision": revision,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:12]


def _prepare(cfg: GradientIsolatedM4ComboConfig) -> dict:
    prepared = gi._prepare(cfg)
    # Reuse the exact train-only economic-input construction of the validated
    # personalized M4.  The lower-level legacy builder does not create
    # ``user_bin_fit``, which is required by the positive-row weights.
    inputs = nv_economic.build_nv_economic_inputs(
        prepared["data"]["train"],
        n_users=prepared["data"]["n_users"],
        n_items=prepared["data"]["n_items"],
        q_n=prepared["q_n"],
        q_v=prepared["q_v"],
        q_c=prepared["q_c"],
        clv_valid=prepared["clv_valid"],
        n_bins=cfg.economic_bins,
        shrinkage_strength=cfg.shrinkage_strength,
    )
    prepared.update(inputs)
    prepared["config_hash"] = _config_hash(cfg, prepared["input_hash"], prepared["revision"])
    return prepared


def _build_model(prepared: dict, cfg: GradientIsolatedM4ComboConfig):
    data = prepared["data"]
    v3.set_seed(cfg.seed)
    return GradientIsolatedComboModel(
        n_users=data["n_users"], n_items=data["n_items"],
        q_n=prepared["q_n"], q_v=prepared["q_v"], q_c=prepared["q_c"],
        user_clv_valid=prepared["clv_valid"],
        item_price_percentile=prepared["item_price"],
        item_price_valid=prepared["item_price_valid"], adj=data["adj"],
        id_dim=cfg.id_dim, relation_dim=cfg.relation_dim, rho=cfg.rho,
        beta=cfg.beta, delta=cfg.delta, eta=cfg.eta,
        price_epsilon=cfg.price_epsilon, n_layers=cfg.n_layers,
        pref_reg=cfg.pref_reg,
    ).to(v3.DEVICE)


def _weights(prepared: dict, cfg, weighted: bool) -> tuple[np.ndarray, dict]:
    if not weighted:
        return np.ones(len(prepared["data"]["tr_u"]), np.float64), {
            "weight_mode": "unweighted", "row_weight_cv": 0.0
        }
    return weighted_training.row_weights(prepared, cfg, "original")


def _arm_paths(prepared: dict, model_id: str, seed: int) -> dict[str, Path]:
    root = prepared["out_dir"] / "arms" / prepared["config_hash"]
    root.mkdir(parents=True, exist_ok=True)
    stem = f"{model_id}_s{seed}"
    return {"checkpoint": root / f"{stem}.pt", "result": root / f"{stem}.json"}


def _load_compatible_completed_arm(prepared: dict, cfg, *, model_id: str, weighted: bool):
    """Reuse a completed arm across this bug-fix revision when its model is unchanged."""

    candidates = []
    pattern = f"arms/*/{model_id}_s{cfg.seed}.json"
    for result_path in sorted(prepared["out_dir"].glob(pattern)):
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            checkpoint_path = Path(payload["checkpoint"])
            checkpoint = economic.load_checkpoint_or_discard(checkpoint_path)
        except (KeyError, OSError, json.JSONDecodeError):
            continue
        if checkpoint is None:
            continue
        if (
            payload.get("model_id") != model_id
            or payload.get("seed") != cfg.seed
            or payload.get("final_epoch") != cfg.epochs
            or bool(payload.get("m4_weighted")) != bool(weighted)
            or checkpoint.get("model_id") != model_id
            or checkpoint.get("input_hash") != prepared["input_hash"]
            or checkpoint.get("config") != asdict(cfg)
        ):
            continue
        candidates.append((result_path, payload, checkpoint))
    if not candidates:
        return None
    hashes = {candidate[1].get("checkpoint_sha256") for candidate in candidates}
    if len(candidates) > 1 and len(hashes) > 1:
        raise RuntimeError(f"서로 다른 호환 완료 결과가 여러 개입니다: {pattern}")
    return candidates[0]


def _run_trained_arm(prepared: dict, cfg, *, model_id: str, role: str, weighted: bool):
    paths = _arm_paths(prepared, model_id, cfg.seed)
    model = _build_model(prepared, cfg)
    if paths["result"].exists() and paths["checkpoint"].exists():
        checkpoint = economic.load_checkpoint_or_discard(paths["checkpoint"])
        if checkpoint is None or checkpoint.get("input_hash") != prepared["input_hash"]:
            raise RuntimeError("cached checkpoint와 현재 입력이 다릅니다")
        model.load_state_dict(checkpoint["state"], strict=True)
        model.eval()
        return json.loads(paths["result"].read_text(encoding="utf-8")), model

    compatible = _load_compatible_completed_arm(
        prepared, cfg, model_id=model_id, weighted=weighted
    )
    if compatible is not None:
        result_path, payload, checkpoint = compatible
        model.load_state_dict(checkpoint["state"], strict=True)
        model.eval()
        print(f"[reused compatible completed arm] {result_path}")
        return payload, model

    spec = {"model_id": model_id, "role": role, "rho": cfg.rho,
            "improvement": "original" if weighted else None}
    weights, weight_diag = _weights(prepared, cfg, weighted)
    store = ProgressStore(
        prepared["out_dir"] / "progress" / prepared["config_hash"],
        RunIdentity(
            stage="historical_development_train", model_id=model_id, seed=cfg.seed,
            config_hash=prepared["config_hash"], source_revision=prepared["revision"],
            input_hash=prepared["input_hash"],
        ),
    )
    training = weighted_training._train_arm(model, prepared, cfg, spec, weights, store)
    model.eval()
    temporary = paths["checkpoint"].with_suffix(".pt.tmp")
    torch.save({
        "state": clone_state(model), "model_id": model_id, "config": asdict(cfg),
        "source_revision": prepared["revision"], "input_hash": prepared["input_hash"],
    }, temporary)
    os.replace(temporary, paths["checkpoint"])
    raw, _ = moe._flat_evaluation(
        model, 0.0, prepared["cache"], prepared["meta"], prepared["data"],
        prepared["base_cfg"], per_user=False,
    )
    payload = {
        "model_id": model_id, "role": role, "seed": cfg.seed,
        "split": "historical_development_days_684_690", "final_epoch": cfg.epochs,
        "rho": cfg.rho, "m4_weighted": weighted,
        "weight_diagnostics": weight_diag,
        "diagnostics": model.representation_diagnostics(),
        "metrics": test10._public_metrics(raw), "training": training,
        "checkpoint": str(paths["checkpoint"]),
        "checkpoint_sha256": file_sha256(paths["checkpoint"]),
    }
    test10._atomic_json(paths["result"], payload)
    store.mark_complete(
        epoch=cfg.epochs, max_epoch=cfg.epochs, selection="none",
        split=payload["split"], checkpoint_path=str(paths["checkpoint"]),
        result_path=str(paths["result"]),
    )
    return payload, model


def _load_references(prepared: dict, cfg) -> tuple[dict, dict, str]:
    candidates = sorted(Path(cfg.reference_result_dir).glob("m4_k1_assignment_control_multiseed_*.json"))
    valid = []
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("code_version") != m4_multi.CODE_VERSION:
            continue
        if payload.get("input_manifest") != prepared["manifest"]:
            continue
        rows = [row for row in payload.get("absolute_rows", []) if row.get("seed") == cfg.seed]
        by_id = {row.get("model_id"): row for row in rows}
        if {m4_multi.M1_MODEL_ID, m4_multi.M4_ACTUAL_MODEL_ID} <= set(by_id):
            valid.append((path, by_id))
    if len(valid) != 1:
        raise FileNotFoundError(
            f"seed {cfg.seed} M1·M4를 포함한 정확히 일치하는 10-seed 결과가 "
            f"1개여야 합니다: found={len(valid)}, root={cfg.reference_result_dir}"
        )
    path, rows = valid[0]
    return rows[m4_multi.M1_MODEL_ID], rows[m4_multi.M4_ACTUAL_MODEL_ID], str(path)


def _metric_dict(row: dict) -> dict:
    return {k: v for k, v in row.items() if "@" in k or k == "user_value_tendency_recommended_price_alignment"}


def decision(metric_rows: dict[str, dict]) -> dict:
    m1, m2, m4, m5 = (metric_rows[mid] for mid in MODEL_IDS)
    m2_replication = all(m2[m] > m1[m] for m in ECONOMIC_METRICS)
    combination = all(m5[m] > max(m2[m], m4[m]) for m in ECONOMIC_METRICS)

    def geomean_ratio(left, right):
        return float(np.exp(np.mean([np.log(left[m] / right[m]) for m in ACCURACY_METRICS])))

    best_part = m2 if np.prod([m2[m] for m in ACCURACY_METRICS]) >= np.prod([m4[m] for m in ACCURACY_METRICS]) else m4
    accuracy_addition = geomean_ratio(m5, best_part) >= 1.0
    guard = all(m5[m] >= 0.99 * m1[m] for m in ACCURACY_METRICS)
    passed = bool(m2_replication and combination and accuracy_addition and guard)
    reported = ACCURACY_METRICS + ECONOMIC_METRICS
    return {
        "classification": "directional_pass" if passed else "directional_nonpass",
        "directional_screen_pass": passed,
        "m2_replication_on_both_economic_metrics": bool(m2_replication),
        "m5_beats_both_parts_on_both_economic_metrics": bool(combination),
        "m5_accuracy_geomean_at_least_best_part": bool(accuracy_addition),
        "m5_each_accuracy_metric_at_least_99pct_m1": bool(guard),
        "deltas_m2_minus_m1": {m: float(m2[m] - m1[m]) for m in reported},
        "deltas_m5_minus_m4": {m: float(m5[m] - m4[m]) for m in reported},
        "deltas_m5_minus_m2": {m: float(m5[m] - m2[m]) for m in reported},
        "interaction": {m: float((m5[m] - m4[m]) - (m2[m] - m1[m])) for m in reported},
        "clv_attribution_tested": False,
        "statistical_note": "one exposed development seed; no significance or generalization claim",
    }


def run_combo_screen(cfg: GradientIsolatedM4ComboConfig | None = None) -> pd.DataFrame:
    cfg = validate_config(cfg or configure_combo_screen())
    print(json.dumps(preflight_summary(cfg), ensure_ascii=False, indent=2))
    prepared = _prepare(cfg)
    m1_ref, m4_ref, reference_path = _load_references(prepared, cfg)
    print(f"[reused] seed {cfg.seed} M1·M4: {reference_path}")

    trained = {}
    models = {}
    for model_id, role, weighted in (
        (M2_MODEL_ID, "gradient_isolated_m2", False),
        (M5_MODEL_ID, "gradient_isolated_m2_plus_personalized_m4", True),
    ):
        print(f"\n===== {model_id} | seed {cfg.seed} | K=1 | fixed 100 epochs =====")
        trained[model_id], models[model_id] = _run_trained_arm(
            prepared, cfg, model_id=model_id, role=role, weighted=weighted
        )

    rows = [
        {**m1_ref, "model_id": M1_MODEL_ID, "training_origin": "reused_exact_seed43"},
        {"model_id": M2_MODEL_ID, "training_origin": "trained_here", **trained[M2_MODEL_ID]["diagnostics"], **trained[M2_MODEL_ID]["weight_diagnostics"], **trained[M2_MODEL_ID]["metrics"]},
        {**m4_ref, "model_id": M4_MODEL_ID, "training_origin": "reused_exact_seed43"},
        {"model_id": M5_MODEL_ID, "training_origin": "trained_here", **trained[M5_MODEL_ID]["diagnostics"], **trained[M5_MODEL_ID]["weight_diagnostics"], **trained[M5_MODEL_ID]["metrics"]},
    ]
    frame = pd.DataFrame(rows)
    metric_rows = {
        M1_MODEL_ID: _metric_dict(m1_ref), M2_MODEL_ID: trained[M2_MODEL_ID]["metrics"],
        M4_MODEL_ID: _metric_dict(m4_ref), M5_MODEL_ID: trained[M5_MODEL_ID]["metrics"],
    }
    comparison = gi._metric_comparison(metric_rows, references=(M1_MODEL_ID, M2_MODEL_ID, M4_MODEL_ID))
    reading = decision(metric_rows)
    out = Path(cfg.out_dir)
    stem = f"m5_gradient_isolated_m4_k1_{prepared['config_hash']}"
    paths = {"absolute_csv": out / f"{stem}.csv", "comparison_csv": out / f"{stem}_comparison.csv", "json": out / f"{stem}.json"}
    test10._atomic_csv(paths["absolute_csv"], frame)
    test10._atomic_csv(paths["comparison_csv"], comparison)
    test10._atomic_json(paths["json"], {
        "code_version": CODE_VERSION, "source_revision": prepared["revision"],
        "config": asdict(cfg), "preflight": preflight_summary(cfg),
        "input_manifest": prepared["manifest"], "reference_result": reference_path,
        "absolute_rows": frame.to_dict("records"),
        "comparison_rows": comparison.to_dict("records"),
        "screening_reading": reading, "trained_arms": trained,
        "result_paths": {k: str(v) for k, v in paths.items()},
    })
    frame.attrs.update(comparison=comparison, decision=reading, result_paths={k: str(v) for k, v in paths.items()})
    print("\n1) 판독")
    print(json.dumps(reading, ensure_ascii=False, indent=2))
    print("\n2) 전체 절대지표")
    print(frame.to_string(index=False))
    print("\n결과 파일:", frame.attrs["result_paths"])
    return frame


if __name__ == "__main__":
    print(json.dumps(preflight_summary(configure_combo_screen()), ensure_ascii=False, indent=2))
