"""Two-seed 2x2 gate for fixed history-M2 on the surviving M3/M4-C path.

This development-only runner keeps the accepted components unchanged:

* M2: personal-history N/V representation, axis_dim=4 and rho=.05.
* M3: the two-axis centred value/activity propagation graph.
* M4-C: customer-mass-preserving positive-row BPR weights.

Existing M3 and M3+M4-C curves are reused after identity checks. Only
M2+M3 and M2+M3+M4-C are trained, from random initialisation, with one model
and one optimiser. No component is pretrained or frozen and no score is
added after training.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from clv_history_item_fit_model import HistoryItemFitLightGCN
from clv_run_state import ProgressStore, RunIdentity
import clv_m5_m3_m4_split_nv_screen as prior
import clv_m5_m3_m4_user_centered_attribution as attribution
import clv_m5_m3_m4_user_centered_screen as observed
import lightgcn_clv_axis_specific_test10 as io
import lightgcn_clv_m2_capacity_search as capacity
import lightgcn_clv_m3_centered_value_graph as m3
import lightgcn_clv_v3 as v3


CODE_VERSION = "clv-m5-history-m2-m3-m4-user-centered-dev-v1"
SEEDS = (42, 44)
FIXED_EPOCH = 300
DIAGNOSTIC_EPOCH = 100
M2_AXIS_DIM = 4
M2_RHO = 0.05
M2_M3_MODEL_ID = "m5_history_m2_m3nv_graph_bpr_k1"
FULL_MODEL_ID = "m5_history_m2_m3nv_graph_user_centered_m4_bpr_k1"
ACCURACY = observed.ACCURACY
ECONOMIC = observed.ECONOMIC


def configure(seed: int, **overrides):
    root = v3.default_out_dir("dunnhumby")
    defaults = {
        "out_dir": f"{root}_clv_m5_history_m2_m3_m4_user_centered_v1",
    }
    cfg = observed.configure(seed=seed, **(defaults | overrides))
    if cfg.seeds != (seed,) or seed not in SEEDS:
        raise ValueError(f"사전 고정한 개발 시드 {SEEDS}만 허용합니다")
    return cfg


def preflight_summary() -> dict:
    return {
        "code_version": CODE_VERSION,
        "question": (
            "Does fixed personal-history N/V M2 add to the current M3+M4-C "
            "candidate, rather than merely beating M1?"
        ),
        "dataset": "dunnhumby",
        "split": "historical_development_days_684_690",
        "seeds": list(SEEDS),
        "fixed_epoch": FIXED_EPOCH,
        "reused_without_training": [m3.ARM_VALUE_ACTIVITY, observed.MODEL_ID],
        "new_training_per_seed": [M2_M3_MODEL_ID, FULL_MODEL_ID],
        "new_fit_count": 2 * len(SEEDS),
        "fixed_m2": {"axis_dim": M2_AXIS_DIM, "rho": M2_RHO},
        "fixed_m3": {
            "model_id": m3.ARM_VALUE_ACTIVITY,
            "target_cv": 0.20,
            "axes": ["q_N", "q_V"],
        },
        "fixed_m4_c": {"lambda": observed.LAMBDA, "customer_mass_preserved": True},
        "training": {
            "joint_from_initialisation": True,
            "one_optimizer": True,
            "pretraining_or_freezing": False,
            "posthoc_score_addition_or_reranking": False,
            "negative_sampling": "uniform_k1",
            "objective": "plain_bpr for M2+M3; existing row-weighted BPR for full",
        },
        "primary_increment": f"{FULL_MODEL_ID} - {observed.MODEL_ID}",
        "gate": {
            "both_economic_at10_mean_positive": list(ECONOMIC),
            "six_accuracy_metrics_at_least_99pct_of_m3_m4_c": list(ACCURACY),
            "strict_replication": "both economic @10 deltas positive in both seeds",
        },
        "selection": "fixed epoch 300 only; epoch 100 is diagnostic",
        "final_test_constructed": False,
        "holdout_constructed": False,
        "limits": (
            "two seeds on a repeatedly exposed development split; no significance, "
            "generalization, final-model, or causal interaction claim"
        ),
    }


def _weighted_adjacency(prepared: dict, graph: dict) -> torch.Tensor:
    data = prepared["data"]
    return v3.build_adj(
        prepared["signals"]["edge_users"],
        prepared["signals"]["edge_items"],
        np.asarray(graph["weights"], dtype=np.float32),
        data["n_users"],
        data["n_items"],
    )


def _build_history_model(prepared: dict, cfg, graph: dict, seed: int):
    data = prepared["data"]
    valid = np.asarray(prepared["clv_valid"], dtype=bool)
    v3.set_seed(seed)
    model = HistoryItemFitLightGCN(
        n_users=data["n_users"],
        n_items=data["n_items"],
        history=prepared["history"],
        q_n=np.where(valid, prepared["q_n"], 0.0).astype(np.float32),
        q_v=np.where(valid, prepared["q_v"], 0.0).astype(np.float32),
        activity_valid=valid,
        value_valid=valid,
        adj=_weighted_adjacency(prepared, graph),
        id_dim=cfg.id_dim,
        axis_dim=M2_AXIS_DIM,
        n_layers=cfg.n_layers,
        rho=M2_RHO,
        pref_reg=cfg.pref_reg,
    )
    return model.to(v3.DEVICE)


def _arm_path(prepared: dict, model_id: str, seed: int) -> Path:
    root = Path(prepared["out_dir"]) / "arms" / prepared["config_hash"]
    return root / f"{model_id}_s{seed}.json"


def _run_history_arm(
    prepared: dict,
    cfg,
    graph: dict,
    *,
    model_id: str,
    seed: int,
    row_weights: np.ndarray | None,
) -> dict:
    result_path = _arm_path(prepared, model_id, seed)
    if result_path.exists():
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        required = {
            "code_version": CODE_VERSION,
            "source_revision": prepared["revision"],
            "input_hash": prepared["input_hash"],
            "model_id": model_id,
            "seed": seed,
            "axis_dim": M2_AXIS_DIM,
            "rho": M2_RHO,
        }
        if any(payload.get(key) != value for key, value in required.items()):
            raise RuntimeError(f"완료 캐시 신원이 현재 실행과 다릅니다: {result_path}")
        print(f"  [cached] {model_id} s{seed} 곡선 재사용", flush=True)
        return payload

    model = _build_history_model(prepared, cfg, graph, seed)
    stage = "m2_m3_m4_c_joint_dev" if row_weights is not None else "m2_m3_joint_dev"
    store = ProgressStore(
        Path(prepared["out_dir"]) / "progress" / prepared["config_hash"],
        RunIdentity(
            stage=stage,
            model_id=model_id,
            seed=seed,
            config_hash=prepared["config_hash"],
            source_revision=prepared["revision"],
            input_hash=prepared["input_hash"],
        ),
    )
    spec = {
        "model_id": model_id,
        "arm": "history_m2_value_and_activity_m3",
        "question": (
            "Does fixed history-M2 add to M3+M4-C?"
            if row_weights is not None
            else "Does fixed history-M2 add to M3 without M4-C?"
        ),
    }
    curve = capacity._train_curve(
        model,
        prepared,
        cfg,
        spec,
        seed,
        store,
        row_weights=row_weights,
    )
    payload = {
        **spec,
        "seed": seed,
        "axis_dim": M2_AXIS_DIM,
        "rho": M2_RHO,
        "m3_beta": float(graph["beta"]),
        "m3_edge_weight_sha256": graph["audit"].get("sha256"),
        "m4_c_row_weight_sha256": (
            None
            if row_weights is None
            else hashlib.sha256(np.asarray(row_weights, np.float32).tobytes()).hexdigest()
        ),
        "objective": "plain_bpr" if row_weights is None else "row_weighted_bpr",
        "code_version": CODE_VERSION,
        "source_revision": prepared["revision"],
        "input_hash": prepared["input_hash"],
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "curve": curve,
    }
    io._atomic_json(result_path, payload)
    store.mark_complete(
        epoch=cfg.epochs,
        max_epoch=cfg.epochs,
        selection="none",
        checkpoint_path="",
        result_path=str(result_path),
    )
    return payload


def _history_curve(arms: list[dict]) -> pd.DataFrame:
    rows = []
    for arm in arms:
        for record in arm["curve"]:
            if "metrics" not in record:
                continue
            rows.append({
                "model_id": arm["model_id"],
                "arm": arm["arm"],
                "seed": arm["seed"],
                "epoch": record["epoch"],
                "loss": record["loss"],
                "p_correct": record["p_correct"],
                "axis_dim": arm["axis_dim"],
                "rho": arm["rho"],
                **{
                    key: value
                    for key, value in record.get("score_split", {}).items()
                    if key != "clv_score_measured_on"
                },
                **record.get("gradient_diagnostics", {}),
                **record["metrics"],
            })
    return pd.DataFrame(rows)


def _metric_columns(curve: pd.DataFrame) -> list[str]:
    return [
        column
        for column in curve.columns
        if "@" in column
        or column == "user_value_tendency_recommended_price_alignment"
    ]


def comparison_table(curve: pd.DataFrame) -> pd.DataFrame:
    index = curve.set_index(["model_id", "seed", "epoch"])
    pairs = (
        (M2_M3_MODEL_ID, m3.ARM_VALUE_ACTIVITY, "M2+M3_minus_M3"),
        (FULL_MODEL_ID, observed.MODEL_ID, "FULL_minus_M3+M4-C"),
        (FULL_MODEL_ID, M2_M3_MODEL_ID, "FULL_minus_M2+M3"),
        (FULL_MODEL_ID, m3.ARM_VALUE_ACTIVITY, "FULL_minus_M3"),
        (FULL_MODEL_ID, m3.M1_MODEL_ID, "FULL_minus_M1"),
    )
    rows = []
    for seed in SEEDS:
        for epoch in (DIAGNOSTIC_EPOCH, FIXED_EPOCH):
            for model_id, reference, contrast in pairs:
                candidate = index.loc[(model_id, seed, epoch)]
                baseline = index.loc[(reference, seed, epoch)]
                for metric in _metric_columns(curve):
                    base = float(baseline[metric])
                    value = float(candidate[metric])
                    rows.append({
                        "seed": seed,
                        "epoch": epoch,
                        "contrast": contrast,
                        "model_id": model_id,
                        "reference": reference,
                        "metric": metric,
                        "reference_value": base,
                        "candidate_value": value,
                        "delta": value - base,
                        "ratio": value / base if base else np.nan,
                    })
    return pd.DataFrame(rows)


def factorial_table(curve: pd.DataFrame) -> pd.DataFrame:
    at = curve[curve.epoch.eq(FIXED_EPOCH)].set_index(["seed", "model_id"])
    rows = []
    for seed in SEEDS:
        graph = at.loc[(seed, m3.ARM_VALUE_ACTIVITY)]
        graph_m2 = at.loc[(seed, M2_M3_MODEL_ID)]
        graph_m4 = at.loc[(seed, observed.MODEL_ID)]
        full = at.loc[(seed, FULL_MODEL_ID)]
        for metric in _metric_columns(curve):
            rows.append({
                "seed": seed,
                "epoch": FIXED_EPOCH,
                "metric": metric,
                "m2_on_m3": float(graph_m2[metric] - graph[metric]),
                "m4_c_on_m3": float(graph_m4[metric] - graph[metric]),
                "m2_on_m3_m4_c": float(full[metric] - graph_m4[metric]),
                "m4_c_on_m2_m3": float(full[metric] - graph_m2[metric]),
                "interaction_conditional_on_m3": float(
                    full[metric] - graph_m2[metric] - graph_m4[metric] + graph[metric]
                ),
            })
    return pd.DataFrame(rows)


def summary_table(curve: pd.DataFrame) -> pd.DataFrame:
    metrics = _metric_columns(curve)
    at = curve[curve.epoch.eq(FIXED_EPOCH)][["model_id", "seed", *metrics]]
    return at.groupby("model_id", sort=False)[metrics].agg(["mean", "std"]).reset_index()


def reading(curve: pd.DataFrame, comparison: pd.DataFrame) -> dict:
    metrics = _metric_columns(curve)
    at = curve[curve.epoch.eq(FIXED_EPOCH)]
    means = at.groupby("model_id")[metrics].mean()
    full = means.loc[FULL_MODEL_ID]
    reference = means.loc[observed.MODEL_ID]
    accuracy_guard = all(full[metric] >= 0.99 * reference[metric] for metric in ACCURACY)
    economic_mean = all(full[metric] > reference[metric] for metric in ECONOMIC)
    paired = comparison[
        comparison.epoch.eq(FIXED_EPOCH)
        & comparison.contrast.eq("FULL_minus_M3+M4-C")
        & comparison.metric.isin(ECONOMIC)
    ]
    positive_seed_count = {
        metric: int((paired.loc[paired.metric.eq(metric), "delta"] > 0).sum())
        for metric in ECONOMIC
    }
    strict_replication = all(count == len(SEEDS) for count in positive_seed_count.values())
    return {
        "development_gate_only": True,
        "repeatedly_exposed_split": True,
        "fixed_epoch": FIXED_EPOCH,
        "seeds": list(SEEDS),
        "primary_reference": observed.MODEL_ID,
        "full_accuracy_guard_99pct_vs_m3_m4_c_mean": bool(accuracy_guard),
        "full_both_economic_at10_above_m3_m4_c_mean": bool(economic_mean),
        "full_minus_m3_m4_c_positive_seed_count": positive_seed_count,
        "mean_screen_condition_met": bool(accuracy_guard and economic_mean),
        "strict_two_seed_condition_met": bool(
            accuracy_guard and economic_mean and strict_replication
        ),
        "m2_redesigned_or_tuned": False,
        "test_or_holdout_evaluated": False,
        "significance_claim": False,
        "generalization_claim": False,
        "causal_interaction_claim": False,
    }


def run(seeds: tuple[int, ...] = SEEDS, **overrides) -> dict:
    if tuple(seeds) != SEEDS:
        raise ValueError(f"사전 고정한 개발 시드는 {SEEDS}만 허용합니다")
    all_curves = []
    audits = {}
    references = {}
    new_fits = []

    for seed in seeds:
        cfg = configure(seed, **overrides)
        prepared = m3._prepare(cfg)
        graph_spec = next(
            spec
            for spec in m3.arm_specifications()
            if spec["model_id"] == m3.ARM_VALUE_ACTIVITY
        )
        graph = m3.build_arm_graph(prepared, cfg, graph_spec)
        old_curve, old_reference = prior._original_curves(cfg, graph["beta"])
        weights, weight_audit = observed.row_weights(prepared)
        observed_curve, observed_reference = attribution._observed_m5_curve(
            cfg,
            prepared["input_hash"],
            graph["beta"],
            weight_audit["sha256"],
        )

        base_hash = prepared["config_hash"]
        prepared["config_hash"] = hashlib.sha256(
            (
                f"{CODE_VERSION}:{SEEDS}:{M2_AXIS_DIM}:{M2_RHO}:"
                f"{observed.LAMBDA}:{base_hash}:{weight_audit['sha256']}"
            ).encode()
        ).hexdigest()[:12]
        print(json.dumps({
            "scope": "Dunnhumby historical development days 684-690",
            "seed": seed,
            "fixed_epoch": FIXED_EPOCH,
            "new_fits": [M2_M3_MODEL_ID, FULL_MODEL_ID],
            "reused": [m3.M1_MODEL_ID, m3.ARM_VALUE_ACTIVITY, observed.MODEL_ID],
            "fixed_m2": {"axis_dim": M2_AXIS_DIM, "rho": M2_RHO},
            "m3_beta": graph["beta"],
            "m4_c_weight_sha256": weight_audit["sha256"],
        }, ensure_ascii=False, indent=2), flush=True)

        m2_m3 = _run_history_arm(
            prepared,
            cfg,
            graph,
            model_id=M2_M3_MODEL_ID,
            seed=seed,
            row_weights=None,
        )
        full = _run_history_arm(
            prepared,
            cfg,
            graph,
            model_id=FULL_MODEL_ID,
            seed=seed,
            row_weights=weights,
        )
        new_fits.extend([f"{seed}:{M2_M3_MODEL_ID}", f"{seed}:{FULL_MODEL_ID}"])
        all_curves.append(pd.concat([
            old_curve,
            observed_curve,
            _history_curve([m2_m3, full]),
        ], ignore_index=True))
        audits[str(seed)] = {
            "m3": graph["audit"],
            "m3_beta": graph["beta"],
            "m4_c": weight_audit,
            "m2": {"axis_dim": M2_AXIS_DIM, "rho": M2_RHO},
        }
        references[str(seed)] = {
            "m1_m3": old_reference,
            "m3_m4_c": observed_reference,
            "input_hash": prepared["input_hash"],
            "source_revision": prepared["revision"],
            "config": asdict(cfg),
        }

    curve = pd.concat(all_curves, ignore_index=True)
    if curve.duplicated(["model_id", "seed", "epoch"]).any():
        raise RuntimeError("전체 비교곡선에 model/seed/epoch 중복이 있습니다")
    required_models = {
        m3.M1_MODEL_ID,
        m3.ARM_VALUE_ACTIVITY,
        M2_M3_MODEL_ID,
        observed.MODEL_ID,
        FULL_MODEL_ID,
    }
    if set(curve.model_id) != required_models:
        raise RuntimeError(f"비교 모형 집합이 다릅니다: {sorted(set(curve.model_id))}")

    comparison = comparison_table(curve)
    factorial = factorial_table(curve)
    summary = summary_table(curve)
    final_reading = reading(curve, comparison)

    out = Path(configure(SEEDS[0], **overrides).out_dir)
    run_hash = hashlib.sha256(
        f"{CODE_VERSION}:{SEEDS}:{FIXED_EPOCH}:{M2_AXIS_DIM}:{M2_RHO}".encode()
    ).hexdigest()[:12]
    stem = f"{CODE_VERSION}_{run_hash}"
    paths = {
        "absolute_csv": out / f"{stem}_absolute.csv",
        "comparison_csv": out / f"{stem}_comparison.csv",
        "factorial_csv": out / f"{stem}_factorial.csv",
        "summary_csv": out / f"{stem}_summary.csv",
        "diagnostics_json": out / f"{stem}_diagnostics.json",
        "json": out / f"{stem}.json",
    }
    io._atomic_csv(paths["absolute_csv"], curve)
    io._atomic_csv(paths["comparison_csv"], comparison)
    io._atomic_csv(paths["factorial_csv"], factorial)
    io._atomic_csv(paths["summary_csv"], summary)
    io._atomic_json(paths["diagnostics_json"], audits)
    io._atomic_json(paths["json"], {
        "code_version": CODE_VERSION,
        "split": "historical_development_days_684_690",
        "final_test": False,
        "holdout": False,
        "seeds": list(SEEDS),
        "fixed_epoch": FIXED_EPOCH,
        "m2": {
            "model": "personal-history N/V source/target representation",
            "axis_dim": M2_AXIS_DIM,
            "rho": M2_RHO,
            "redesigned_or_tuned": False,
        },
        "m3_model_id": m3.ARM_VALUE_ACTIVITY,
        "m4_c_lambda": observed.LAMBDA,
        "joint_training": True,
        "pretraining_or_freezing": False,
        "posthoc_score_addition_or_reranking": False,
        "new_fits": new_fits,
        "new_fit_count": len(new_fits),
        "reused_models": [m3.M1_MODEL_ID, m3.ARM_VALUE_ACTIVITY, observed.MODEL_ID],
        "references": references,
        "audits": audits,
        "reading": final_reading,
        "limits": (
            "two repeatedly exposed development seeds; no significance, generalization, "
            "final-model, or causal interaction claim"
        ),
        "result_paths": {key: str(value) for key, value in paths.items()},
    })
    print(json.dumps({
        "reading": final_reading,
        "paths": {key: str(value) for key, value in paths.items()},
    }, ensure_ascii=False, indent=2), flush=True)
    return {
        "absolute": curve,
        "comparison": comparison,
        "factorial": factorial,
        "summary": summary,
        "reading": final_reading,
        "paths": paths,
    }


if __name__ == "__main__":
    print(json.dumps(preflight_summary(), ensure_ascii=False, indent=2))
