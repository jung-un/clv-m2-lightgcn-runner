"""H&M development-only transfer screen for Dunnhumby's attributed M5-C.

This intentionally does not open H&M's final last week.  It reuses the exact
seed-43 M1/M3 development curves and trains only observed-assignment M5-C and
its degree-matched joint N/V/q_C permutation for fixed 300 epochs.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import clv_m5_m3_m4_user_centered_attribution as attribution
import clv_m5_m3_m4_user_centered_screen as m4c
import lightgcn_clv_axis_specific_test10 as io
import lightgcn_clv_hm2y_seed42_common as common
import lightgcn_clv_m2_training_budget_hm2y as budget
import lightgcn_clv_m3_centered_value_graph_hm2y as hm_graph
import lightgcn_clv_m5_nv_economic_positive_weight as economics
import lightgcn_clv_v3 as v3


CODE_VERSION = "clv-m5-user-centered-hm2y-transfer-screen-v1"
M1 = hm_graph.M1_MODEL_ID
M3 = hm_graph.ARM_VALUE_ACTIVITY
M5_C = "m5_m3nv_user_centered_m4_bpr_k1_hm2y_s43"
M5_C_SHUFFLED = "m5_m3nv_user_centered_m4_tuple_shuffle_bpr_k1_hm2y_s43"
ACCURACY = ("recall@10", "ndcg@10", "recall@20", "ndcg@20", "recall@50", "ndcg@50")
ECONOMIC = ("price_purchase_amount_weighted_hit@10", "vndcg@10")
NUMERIC_TOLERANCE = 1e-8
SHUFFLE_SEED = 20261008 + 43


def configure(**overrides):
    root = v3.default_out_dir("hm")
    defaults = {
        "seed": 43, "epochs": 300, "evaluation_epochs": (100, 200, 300),
        "reported_epochs": (100, 300), "arms": ("value_and_activity",),
        "out_dir": root + "_m5c_user_centered_hm2y_transfer_s43_v1",
        "m1_result_dir": root + "_clv_m2_training_budget_seed43_v1",
        "allow_baseline_training": False, "eval_test": False, "eval_holdout": False,
    }
    cfg = hm_graph.configure_centered_graph_hm2y(**(defaults | overrides))
    if (cfg.seed != 43 or cfg.epochs != 300
            or tuple(cfg.evaluation_epochs) != (100, 200, 300)
            or tuple(cfg.arms) != ("value_and_activity",)
            or cfg.eval_test or cfg.eval_holdout or cfg.allow_baseline_training):
        raise ValueError("H&M 개발분할 seed43·고정300epoch·M1/M3 재사용만 허용합니다")
    return cfg


def models_for(cfg) -> tuple[str, ...]:
    configure(**asdict(cfg))
    return M1, M3, M5_C, M5_C_SHUFFLED


def preflight_summary(cfg) -> dict:
    configure(**asdict(cfg))
    return {
        "code_version": CODE_VERSION,
        "question": "Does the unchanged attributed M5-C transfer to H&M development data?",
        "split": hm_graph.SPLIT,
        "seed": cfg.seed,
        "fixed_epochs": list(cfg.evaluation_epochs),
        "reused": [M1, M3],
        "new_fits": [M5_C, M5_C_SHUFFLED],
        "same_formula_as_dunnhumby": True,
        "decision_epoch": 300,
        "numerical_tolerance": NUMERIC_TOLERANCE,
        "fixed": {
            "new_item_task": True, "train_pairs_excluded": True,
            "min_item_interactions": 1, "binary_base_graph": True,
            "uniform_negative_k": 1, "validation_only": True,
            "test_constructed": False, "holdout_constructed": False,
            "external_reranking": False,
        },
        "limits": "single repeatedly exposed development seed; no success, significance or generalization claim",
    }


def _load_m3_curve(prepared: dict, cfg, expected_beta: float) -> pd.DataFrame:
    root = Path(v3.default_out_dir("hm") + "_clv_m3_centered_value_graph_s43_v1")
    path = root / "arms" / prepared["run_hash"] / f"{M3}.json"
    if not path.is_file():
        raise RuntimeError(f"exact completed H&M M3 reference is required: {path}")
    payload = json.loads(path.read_text())
    if (payload.get("seed") != 43 or payload.get("split") != hm_graph.SPLIT
            or not np.isclose(payload.get("beta"), expected_beta, rtol=0, atol=1e-12)):
        raise RuntimeError("H&M M3 reference identity/beta mismatch")
    curve = hm_graph.curve_table([payload], [])
    if set(curve.epoch) != set(cfg.evaluation_epochs):
        raise RuntimeError("H&M M3 reference curve is incomplete")
    curve["seed"] = 43
    return curve


def prepare(cfg):
    cfg = configure(**asdict(cfg))
    baseline = hm_graph.load_baseline_curve(cfg)
    prepared = hm_graph._prepare(cfg)
    econ = economics.build_nv_economic_inputs(
        prepared["data"]["train"], n_users=prepared["data"]["n_users"],
        n_items=prepared["data"]["n_items"], q_n=prepared["q_n"], q_v=prepared["q_v"],
        q_c=prepared["q_c"], clv_valid=prepared["clv_valid"], n_bins=4,
        shrinkage_strength=10.0, degree_bins=10,
    )
    prepared.update({key: econ[key] for key in (
        "item_amount_percentile", "item_economic_valid", "user_economic_valid",
        "user_bin_fit", "item_bin",
    )})
    spec = hm_graph.arm_specifications(cfg)[0]
    observed_graph = hm_graph.build_arm_graph(prepared, cfg, spec)
    observed_weights, observed_weight_audit = m4c.row_weights(prepared)
    assignment, shuffle_audit = attribution.degree_matched_tuple_shuffle(
        prepared, seed=SHUFFLE_SEED,
    )
    shuffled = attribution._prepared_with_assignment(prepared, assignment)
    shuffled_graph = hm_graph.build_arm_graph(shuffled, cfg, spec)
    shuffled_weights, shuffled_weight_audit = m4c.row_weights(shuffled)

    baseline_curve = hm_graph.curve_table([], baseline)
    baseline_curve["seed"] = 43
    m3_curve = _load_m3_curve(prepared, cfg, observed_graph["beta"])
    identity = {
        "code_version": CODE_VERSION, "source_revision": prepared["revision"],
        "input_hash": prepared["input_hash"], "config": asdict(cfg),
        "observed_beta": observed_graph["beta"], "shuffled_beta": shuffled_graph["beta"],
        "observed_weight": observed_weight_audit["sha256"],
        "shuffled_weight": shuffled_weight_audit["sha256"],
    }
    run_hash = hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest()[:12]
    run_dir = Path(cfg.out_dir) / run_hash
    protocol = preflight_summary(cfg) | {
        "config": asdict(cfg), "input_hash": prepared["input_hash"],
        "source_revision": prepared["revision"],
        "observed_m3_audit": observed_graph["audit"],
        "shuffled_m3_audit": shuffled_graph["audit"],
        "observed_m4_c_audit": observed_weight_audit,
        "shuffled_m4_c_audit": shuffled_weight_audit,
        "shuffle_audit": shuffle_audit,
    }
    common.atomic_json(run_dir / "protocol.json", protocol)
    shuffled["config_hash"] = run_hash
    prepared.update(run_dir=run_dir, config_hash=run_hash, protocol=protocol,
                    reference_curve=pd.concat([baseline_curve, m3_curve], ignore_index=True),
                    observed_graph=observed_graph, observed_weights=observed_weights,
                    shuffled_prepared=shuffled, shuffled_graph=shuffled_graph,
                    shuffled_weights=shuffled_weights)
    return prepared


def _run_arm(prepared: dict, cfg, model_id: str) -> dict:
    if model_id == M5_C:
        arm_prepared, graph, weights = prepared, prepared["observed_graph"], prepared["observed_weights"]
    elif model_id == M5_C_SHUFFLED:
        arm_prepared, graph, weights = (prepared["shuffled_prepared"],
                                        prepared["shuffled_graph"],
                                        prepared["shuffled_weights"])
    else:
        raise ValueError(model_id)
    path = prepared["run_dir"] / "arms" / f"{model_id}.json"
    if path.is_file():
        payload = json.loads(path.read_text())
        if payload.get("run_hash") != prepared["config_hash"]:
            raise RuntimeError("completed H&M M5-C arm identity mismatch")
        return payload
    spec = {"model_id": model_id, "arm": "value_and_activity", "gamma": 1.0,
            "question": "observed CLV assignment" if model_id == M5_C else "degree-matched CLV null"}
    model = hm_graph._build_model(arm_prepared, cfg, graph)
    store = common.progress_store(arm_prepared, cfg, model_id, prepared["config_hash"])
    curve = budget._train_curve(model, arm_prepared, cfg, spec, store, row_weights=weights)
    payload = {**spec, "seed": 43, "split": hm_graph.SPLIT,
               "code_version": CODE_VERSION, "source_revision": prepared["revision"],
               "run_hash": prepared["config_hash"], "beta": graph["beta"],
               "edge_audit": graph["audit"], "curve": curve}
    common.atomic_json(path, payload)
    store.mark_complete(epoch=cfg.epochs, max_epoch=cfg.epochs, selection="none",
                        checkpoint_path="", result_path=str(path))
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return payload


def _validate_curve(curve: pd.DataFrame, cfg) -> list[str]:
    expected = {(model, 43, epoch) for model in models_for(cfg) for epoch in cfg.evaluation_epochs}
    found = set(zip(curve.model_id, curve.seed, curve.epoch))
    if len(curve) != len(expected) or found != expected or curve.duplicated(["model_id", "seed", "epoch"]).any():
        raise RuntimeError("four H&M models at all three fixed epochs are required")
    metrics = [column for column in curve.columns if "@" in column]
    if not metrics or not np.isfinite(curve[metrics].to_numpy(float)).all():
        raise RuntimeError("missing or non-finite H&M metrics")
    return metrics


def reading(curve: pd.DataFrame, cfg) -> dict:
    _validate_curve(curve, cfg)
    at = curve[curve.epoch.eq(300)].set_index("model_id")
    accuracy_guard = all(
        float(at.at[M5_C, metric]) >= .99 * float(at.at[reference, metric])
        for reference in (M1, M3) for metric in ACCURACY
    )
    economic_up = all(
        float(at.at[M5_C, metric]) > float(at.at[reference, metric]) + NUMERIC_TOLERANCE
        for reference in (M1, M3) for metric in ECONOMIC
    )
    attribution_up = all(
        float(at.at[M5_C, metric]) > float(at.at[M5_C_SHUFFLED, metric]) + NUMERIC_TOLERANCE
        for metric in ECONOMIC
    )
    eligible = accuracy_guard and economic_up and attribution_up
    return {
        "complete": True, "development_transfer_screen_only": True,
        "seed_count": 1, "decision_epoch": 300, "final_test_evaluated": False,
        "m5_c_accuracy_guard_99pct_vs_m1_and_m3": bool(accuracy_guard),
        "m5_c_both_economic_at10_above_m1_and_m3": bool(economic_up),
        "observed_assignment_beats_shuffle_on_both_economic_at10": bool(attribution_up),
        "eligible_for_future_final_evaluation": bool(eligible),
        "success_claim": False, "significance_claim": False,
    }


def _comparison(curve: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    index = curve.set_index(["model_id", "epoch"])
    rows = []
    for epoch in (100, 200, 300):
        for model, reference in ((M3, M1), (M5_C, M1), (M5_C, M3),
                                 (M5_C, M5_C_SHUFFLED)):
            for metric in metrics:
                value, base = float(index.at[(model, epoch), metric]), float(index.at[(reference, epoch), metric])
                rows.append({"model_id": model, "reference": reference, "seed": 43,
                             "epoch": epoch, "metric": metric, "value": value,
                             "reference_value": base, "delta": value - base,
                             "relative_change_pct": 100 * (value / base - 1) if base else np.nan})
    return pd.DataFrame(rows)


def report(prepared: dict, cfg, curve: pd.DataFrame, diagnostics: pd.DataFrame) -> dict:
    metrics = _validate_curve(curve, cfg)
    comparison = _comparison(curve, metrics)
    decision = reading(curve, cfg)
    tables = {"absolute": curve, "comparison": comparison, "diagnostics": diagnostics}
    root = Path(prepared["run_dir"]) / "reports"
    paths = {key: str(root / f"{key}.csv") for key in tables}
    for key, table in tables.items():
        io._atomic_csv(Path(paths[key]), table)
    paths["json"] = str(root / "result.json")
    common.atomic_json(Path(paths["json"]), {
        "protocol": prepared["protocol"], "reading": decision, "result_paths": paths,
    })
    return {**tables, "reading": decision, "paths": paths}


def run(cfg=None, prepared=None) -> dict:
    cfg = configure() if cfg is None else configure(**asdict(cfg))
    prepared = prepare(cfg) if prepared is None else prepared
    observed = _run_arm(prepared, cfg, M5_C)
    shuffled = _run_arm(prepared, cfg, M5_C_SHUFFLED)
    new_curve = hm_graph.curve_table([observed, shuffled], {})
    new_curve["seed"] = 43
    curve = pd.concat([prepared["reference_curve"], new_curve], ignore_index=True)
    diagnostics = pd.DataFrame([
        {"model_id": M5_C, "seed": 43, "beta": prepared["observed_graph"]["beta"]},
        {"model_id": M5_C_SHUFFLED, "seed": 43, "beta": prepared["shuffled_graph"]["beta"]},
    ])
    return report(prepared, cfg, curve, diagnostics)


if __name__ == "__main__":
    print(json.dumps(preflight_summary(configure()), ensure_ascii=False, indent=2))
