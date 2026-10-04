"""Two new sequential fits: reliability-gated N/V M2 and full M2+M3+M4-B.

Exact seed-43 M1 and M5-B development curves are reused. New fits keep the
same 64-D ID capacity and fixed 300-epoch protocol. M2 uses the binary graph
and plain BPR; M5-full uses the existing centered two-axis M3 graph and the
existing split-N/V M4-B row weights. No test or holdout is opened.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from clv_reliability_orthogonal_nv_model import (
    ReliabilityOrthogonalNVLightGCN,
    build_features,
    self_test as model_self_test,
)
from clv_run_state import ProgressStore, RunIdentity
import clv_m5_m3_m4_split_nv_screen as m5
import lightgcn_clv_axis_specific_test10 as io
import lightgcn_clv_m2_capacity_search as capacity
import lightgcn_clv_m3_centered_value_graph as m3
import lightgcn_clv_v3 as v3


CODE_VERSION = "clv-reliability-orthogonal-nv-m2-m5-dev-v1"
SEED = 43
FIXED_EPOCH = 300
RHO = 0.10
SHARED_L2 = 1e-6
BANDWIDTH = 0.25
RELIABILITY_FLOOR = 0.05
M2 = "m2_reliability_orthogonal_nv_bpr_k1"
M5_FULL = "m5_m2_orthogonal_nv_m3nv_graph_split_nv_m4_bpr_k1"
M5_BASE = m5.ARM_B
ACCURACY = m5.ACCURACY
ECONOMIC = m5.ECONOMIC


def specs() -> list[dict]:
    return [
        {
            "model_id": M2,
            "arm": "binary_graph_plain_bpr",
            "role": "M2",
            "question": "Does reliability-gated N/V matching improve M1 overall accuracy?",
            "weighted": False,
        },
        {
            "model_id": M5_FULL,
            "arm": "m2_representation_m3_graph_m4_loss",
            "role": "M5-full",
            "question": "Does the same M2 add accuracy to the existing M5-B without losing economics?",
            "weighted": True,
        },
    ]


def configure(**overrides):
    defaults = {
        "seed": SEED,
        "epochs": FIXED_EPOCH,
        "eval_every": 25,
        "out_dir": (
            f"{v3.default_out_dir('dunnhumby')}"
            "_clv_reliability_orthogonal_nv_m2_m5_s43_v1"
        ),
    }
    cfg = m5.configure(**(defaults | overrides))
    if cfg.seeds != (SEED,) or cfg.epochs != FIXED_EPOCH or cfg.negative_count != 1:
        raise ValueError("seed43·300epoch·uniform K=1만 허용합니다")
    return cfg


def _validate(cfg):
    if cfg.seeds != (SEED,) or cfg.epochs != FIXED_EPOCH or cfg.negative_count != 1:
        raise ValueError("seed43·300epoch·uniform K=1만 허용합니다")
    return cfg


def feature_hash(features: dict) -> str:
    digest = hashlib.sha256()
    for key in sorted(key for key in features if key != "diagnostics"):
        values = np.ascontiguousarray(features[key])
        digest.update(f"{key}:{values.dtype}:{values.shape}".encode())
        digest.update(values.tobytes())
    return digest.hexdigest()


def prepare(cfg=None) -> tuple[object, dict]:
    cfg = configure() if cfg is None else _validate(cfg)
    prepared = m3._prepare(cfg)
    old_curve, old_reference = m5._existing_m5_b_curves(cfg, prepared["input_hash"])

    graph_spec = next(
        spec for spec in m3.arm_specifications() if spec["model_id"] == m3.ARM_VALUE_ACTIVITY
    )
    graph = m3.build_arm_graph(prepared, cfg, graph_spec)
    weights, weight_audit = m5.row_weights(prepared)
    m5.check_original_m4(weight_audit)
    data = prepared["data"]
    features = build_features(
        data["train"], n_users=data["n_users"], n_items=data["n_items"],
        q_n=prepared["q_n"], q_v=prepared["q_v"], valid=prepared["clv_valid"],
        bandwidth=BANDWIDTH, reliability_floor=RELIABILITY_FLOOR,
    )
    if not np.array_equal(features["keys"], np.asarray(data["pos_key"], np.int64)):
        raise RuntimeError("M2 상품 프로필과 M1 binary graph의 TRAIN pair가 다릅니다")
    settings = {
        "rho": RHO, "shared_l2": SHARED_L2, "bandwidth": BANDWIDTH,
        "reliability_floor": RELIABILITY_FLOOR,
    }
    digest = hashlib.sha256(
        json.dumps(
            {
                "version": CODE_VERSION,
                "old_config_hash": prepared["config_hash"],
                "feature_hash": feature_hash(features),
                "settings": settings,
            }, sort_keys=True,
        ).encode()
    ).hexdigest()[:12]
    prepared.update(
        out_dir=Path(cfg.out_dir), config_hash=digest, features=features,
        features_sha256=feature_hash(features), feature_settings=settings,
        m3_graph=graph, m4_weights=weights[m5.ARM_B],
        m4_weight_audit=weight_audit[m5.ARM_B], old_curve=old_curve,
        old_reference=old_reference,
    )
    io._atomic_json(Path(cfg.out_dir) / "feature_diagnostic.json", features["diagnostics"])
    print(json.dumps({
        "code_version": CODE_VERSION, "new_fits_sequential": [M2, M5_FULL],
        "reused": [m3.M1_MODEL_ID, M5_BASE], "seed": SEED,
        "epochs": FIXED_EPOCH, "test": False, "holdout": False,
        "feature_settings": settings, "feature_diagnostics": features["diagnostics"],
        "m3_beta": graph["beta"], "m3_graph_audit": graph["audit"],
        "m4_weight_audit": weight_audit[m5.ARM_B],
    }, ensure_ascii=False, indent=2), flush=True)
    print("준비 완료(학습 없음). 한 GPU에서 M2 완료 후 M5-full을 순차 실행합니다.", flush=True)
    return cfg, prepared


def _adjacency(prepared: dict, spec: dict):
    if spec["role"] == "M2":
        return prepared["data"]["adj"]
    graph = prepared["m3_graph"]
    return v3.build_adj(
        prepared["signals"]["edge_users"], prepared["signals"]["edge_items"],
        graph["weights"].astype(np.float32), prepared["data"]["n_users"],
        prepared["data"]["n_items"],
    )


def _build(prepared: dict, cfg, spec: dict):
    v3.set_seed(SEED)
    data = prepared["data"]
    return ReliabilityOrthogonalNVLightGCN(
        n_users=data["n_users"], n_items=data["n_items"],
        features=prepared["features"], adj=_adjacency(prepared, spec),
        id_dim=cfg.id_dim, n_layers=cfg.n_layers, pref_reg=cfg.pref_reg,
        shared_l2=SHARED_L2, rho=RHO,
    ).to(v3.DEVICE)


def _arm_path(prepared: dict, spec: dict) -> Path:
    return prepared["out_dir"] / "arms" / prepared["config_hash"] / f"{spec['model_id']}.json"


def _run_arm(prepared: dict, cfg, spec: dict) -> dict:
    path = _arm_path(prepared, spec)
    if path.is_file():
        payload = json.loads(path.read_text())
        if (payload.get("code_version") != CODE_VERSION
                or payload.get("input_hash") != prepared["input_hash"]
                or payload.get("config_hash") != prepared["config_hash"]):
            raise RuntimeError(f"완료 캐시 신원이 다릅니다: {path}")
        print(f"[cached] {spec['model_id']} 완료 결과 재사용", flush=True)
        return payload

    model = _build(prepared, cfg, spec)
    store = ProgressStore(
        prepared["out_dir"] / "progress" / prepared["config_hash"],
        RunIdentity(
            stage="reliability_orthogonal_nv_dev", model_id=spec["model_id"], seed=SEED,
            config_hash=prepared["config_hash"], source_revision=prepared["revision"],
            input_hash=prepared["input_hash"],
        ),
    )
    row_weights = prepared["m4_weights"] if spec["weighted"] else None
    curve = capacity._train_curve(model, prepared, cfg, spec, SEED, store, row_weights=row_weights)
    payload = {
        **spec, "seed": SEED, "code_version": CODE_VERSION,
        "source_revision": prepared["revision"], "input_hash": prepared["input_hash"],
        "config_hash": prepared["config_hash"], "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "feature_settings": prepared["feature_settings"], "curve": curve,
    }
    io._atomic_json(path, payload)
    store.mark_complete(epoch=cfg.epochs, max_epoch=cfg.epochs, selection="none",
                        checkpoint_path="", result_path=str(path))
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return payload


def _new_curve(arms: list[dict]) -> pd.DataFrame:
    rows = []
    for arm in arms:
        for record in arm["curve"]:
            if "metrics" not in record:
                continue
            rows.append({
                "model_id": arm["model_id"], "role": arm["role"], "seed": SEED,
                "epoch": record["epoch"], "loss": record["loss"],
                "p_correct": record["p_correct"],
                **{key: value for key, value in record.get("score_split", {}).items()
                   if key != "clv_score_measured_on"},
                **record.get("gradient_diagnostics", {}), **record["metrics"],
            })
    return pd.DataFrame(rows)


def _comparison(curve: pd.DataFrame) -> pd.DataFrame:
    index = curve.set_index(["model_id", "seed", "epoch"])
    metric_columns = [column for column in curve.columns if "@" in column]
    pairs = ((M2, m3.M1_MODEL_ID), (M5_FULL, M5_BASE), (M5_FULL, m3.M1_MODEL_ID))
    rows = []
    for epoch in (100, FIXED_EPOCH):
        for model_id, reference in pairs:
            candidate = index.loc[(model_id, SEED, epoch)]
            baseline = index.loc[(reference, SEED, epoch)]
            for metric in metric_columns:
                value, ref = float(candidate[metric]), float(baseline[metric])
                rows.append({
                    "seed": SEED, "epoch": epoch, "model_id": model_id,
                    "reference": reference, "metric": metric,
                    "value": value, "reference_value": ref, "delta": value - ref,
                    "ratio": value / ref if ref else np.nan,
                })
    return pd.DataFrame(rows)


def _geometric_ratio(candidate, reference) -> float:
    ratios = np.array([float(candidate[key]) / float(reference[key]) for key in ACCURACY])
    return float(np.exp(np.log(ratios).mean()))


def reading(curve: pd.DataFrame) -> dict:
    at = curve[curve.seed.eq(SEED) & curve.epoch.eq(FIXED_EPOCH)].set_index("model_id")
    required = (m3.M1_MODEL_ID, M5_BASE, M2, M5_FULL)
    missing = set(required).difference(at.index)
    if missing:
        raise RuntimeError(f"판정에 필요한 모형이 없습니다: {sorted(missing)}")

    def judge(candidate_id, reference_id):
        candidate, reference = at.loc[candidate_id], at.loc[reference_id]
        accuracy_ratios = {key: float(candidate[key] / reference[key]) for key in ACCURACY}
        economic_ratios = {key: float(candidate[key] / reference[key]) for key in ECONOMIC}
        return {
            "reference": reference_id,
            "accuracy_geometric_mean_ratio": _geometric_ratio(candidate, reference),
            "accuracy_ratios": accuracy_ratios,
            "economic_at10_ratios": economic_ratios,
            "top10_accuracy_above_reference": bool(
                candidate["recall@10"] > reference["recall@10"]
                and candidate["ndcg@10"] > reference["ndcg@10"]
            ),
            "six_accuracy_guard_99pct": bool(min(accuracy_ratios.values()) >= 0.99),
            "economic_at10_guard_99pct": bool(min(economic_ratios.values()) >= 0.99),
        }

    m2_judgment = judge(M2, m3.M1_MODEL_ID)
    m2_judgment["candidate"] = bool(
        m2_judgment["accuracy_geometric_mean_ratio"] > 1
        and m2_judgment["top10_accuracy_above_reference"]
        and m2_judgment["six_accuracy_guard_99pct"]
        and m2_judgment["economic_at10_guard_99pct"]
    )
    m5_judgment = judge(M5_FULL, M5_BASE)
    m5_judgment["candidate"] = bool(
        m5_judgment["accuracy_geometric_mean_ratio"] > 1
        and m5_judgment["top10_accuracy_above_reference"]
        and m5_judgment["six_accuracy_guard_99pct"]
        and m5_judgment["economic_at10_guard_99pct"]
    )
    interaction = {
        key: float((at.loc[M5_FULL, key] - at.loc[M5_BASE, key])
                   - (at.loc[M2, key] - at.loc[m3.M1_MODEL_ID, key]))
        for key in (*ACCURACY, *ECONOMIC)
    }
    return {
        "development_screen_only": True, "fixed_epoch": FIXED_EPOCH,
        "significance_claim": False, "clv_attribution_claim": False,
        M2: m2_judgment, M5_FULL: m5_judgment,
        "m2_conditional_interaction_absolute": interaction,
        "next_if_positive": "train degree-stratified N/V permutation only for passing arms",
    }


def save(cfg, prepared: dict, arms: list[dict]) -> dict:
    old = prepared["old_curve"]
    old = old[old.model_id.isin((m3.M1_MODEL_ID, M5_BASE))].copy()
    curve = pd.concat([old, _new_curve(arms)], ignore_index=True, sort=False)
    if curve.duplicated(["model_id", "seed", "epoch"]).any():
        raise RuntimeError("결과 곡선에 중복 model/seed/epoch가 있습니다")
    comparison = _comparison(curve)
    result_reading = reading(curve)
    root = Path(cfg.out_dir) / "reports"
    paths = {
        "absolute": root / "absolute.csv",
        "comparison": root / "comparison.csv",
        "json": root / "result.json",
    }
    io._atomic_csv(paths["absolute"], curve)
    io._atomic_csv(paths["comparison"], comparison)
    io._atomic_json(paths["json"], {
        "code_version": CODE_VERSION, "config": asdict(cfg), "seed": SEED,
        "source_revision": prepared["revision"], "input_hash": prepared["input_hash"],
        "split": "historical_development_days_684_690", "final_test": False,
        "holdout": False, "new_fit_count": 2, "new_arms": specs(),
        "feature_settings": prepared["feature_settings"],
        "feature_diagnostics": prepared["features"]["diagnostics"],
        "representation": (
            "64-D propagated ID plus support-gated N/V residual orthogonal to the same ID; "
            "user q_N/q_V matches leave-one-buyer-out item buyer q_N/q_V and train price"
        ),
        "m3_graph_audit": prepared["m3_graph"]["audit"],
        "m4_weight_audit": prepared["m4_weight_audit"],
        "reused_reference": prepared["old_reference"], "reading": result_reading,
        "arms": arms,
        "limits": (
            "one repeatedly exposed development seed; no significance, generalization, "
            "causal interaction or CLV attribution claim; no permutation arm in this screen"
        ),
        "paths": {key: str(value) for key, value in paths.items()},
    })
    return {"absolute": curve, "comparison": comparison, "reading": result_reading,
            "paths": paths}


def run(cfg, prepared: dict) -> dict:
    _validate(cfg)
    if feature_hash(prepared["features"]) != prepared["features_sha256"]:
        raise ValueError("준비된 N/V 특징이 바뀌었습니다")
    arms = []
    for spec in specs():
        print(f"\n===== {spec['model_id']} | seed {SEED} | {FIXED_EPOCH} epoch =====", flush=True)
        arms.append(_run_arm(prepared, cfg, spec))
    return save(cfg, prepared, arms)


def self_test() -> None:
    model_self_test()
    metrics = {key: 1.0 for key in (*ACCURACY, *ECONOMIC)}
    rows = []
    for model_id, factor in ((m3.M1_MODEL_ID, 1.0), (M5_BASE, 1.01),
                             (M2, 1.02), (M5_FULL, 1.04)):
        rows.append({"model_id": model_id, "seed": SEED, "epoch": FIXED_EPOCH,
                     **{key: value * factor for key, value in metrics.items()}})
    result = reading(pd.DataFrame(rows))
    assert result[M2]["candidate"] and result[M5_FULL]["candidate"]


if __name__ == "__main__":
    self_test()
    print("reliability orthogonal N/V screen self-test ok")
