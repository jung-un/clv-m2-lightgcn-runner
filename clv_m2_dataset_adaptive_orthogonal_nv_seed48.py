"""One-arm Dunnhumby development screen for dataset-adaptive M2.

The completed seed-48 M1 and fixed-composition M2 curves are immutable
references.  Only the dataset-adaptive M2 is freshly trained.  This is a
development comparison on days 684--690, not the last-week final test.
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

from clv_dataset_adaptive_orthogonal_nv_model import (
    DatasetAdaptiveOrthogonalNVLightGCN,
    dataset_parameter_initialization,
)
from clv_reliability_orthogonal_nv_model import build_features
from clv_run_state import ProgressStore, RunIdentity
import clv_reliability_orthogonal_nv_screen as original
import lightgcn_clv_axis_specific_test10 as io
import lightgcn_clv_m2_capacity_search as capacity
import lightgcn_clv_m3_centered_value_graph as m3
import lightgcn_clv_v3 as v3


CODE_VERSION = "clv-m2-dataset-adaptive-orthogonal-nv-seed48-v1"
REFERENCE_CODE_VERSION = "clv-reliability-orthogonal-nv-seed48-replication-v1"
REFERENCE_REVISION = "b541ba7816143bb38da9461163ea470665eb5de8"
SEED = 48
FIXED_EPOCH = 300
M1 = m3.M1_MODEL_ID
M2_FIXED = original.M2
M2_ADAPTIVE = "m2_dataset_adaptive_orthogonal_nv_bpr_k1"
ACCURACY = original.ACCURACY
ECONOMIC = original.ECONOMIC
AT10 = ("recall@10", "ndcg@10", *ECONOMIC)


def specs() -> list[dict]:
    return [
        {
            "model_id": M2_ADAPTIVE,
            "arm": "binary_graph_plain_bpr",
            "role": "M2",
            "id_dim": 64,
            "pref_reg": 1e-3,
        }
    ]


def configure(**overrides):
    defaults = {
        "seeds": (SEED,),
        "epochs": FIXED_EPOCH,
        "eval_every": 25,
        "out_dir": (
            f"{v3.default_out_dir('dunnhumby')}"
            "_clv_m2_dataset_adaptive_orthogonal_nv_seed48_v1"
        ),
    }
    cfg = m3.configure_centered_graph(**(defaults | overrides))
    if cfg.seeds != (SEED,) or cfg.epochs != FIXED_EPOCH or cfg.negative_count != 1:
        raise ValueError("seed48·300epoch·uniform K=1만 허용합니다")
    return cfg


def _feature_hash(features: dict) -> str:
    digest = hashlib.sha256()
    for key in sorted(key for key in features if key != "diagnostics"):
        values = np.ascontiguousarray(features[key])
        digest.update(f"{key}:{values.dtype}:{values.shape}".encode())
        digest.update(values.tobytes())
    return digest.hexdigest()


def _reference_result_path() -> Path:
    return Path(
        f"{v3.default_out_dir('dunnhumby')}"
        "_clv_reliability_orthogonal_nv_seed48_replication_v1/reports/result.json"
    )


def _validate_reference(payload: dict, prepared: dict, cfg) -> list[dict]:
    expected_config = {
        "seeds": [SEED],
        "epochs": FIXED_EPOCH,
        "eval_every": 25,
        "batch_size": cfg.batch_size,
        "lr": cfg.lr,
        "n_layers": cfg.n_layers,
        "id_dim": cfg.id_dim,
        "pref_reg": cfg.pref_reg,
        "negative_count": 1,
    }
    if payload.get("code_version") != REFERENCE_CODE_VERSION:
        raise RuntimeError("기준 실행 code_version이 다릅니다")
    if payload.get("source_revision") != REFERENCE_REVISION:
        raise RuntimeError("기준 실행 source revision이 다릅니다")
    if payload.get("input_hash") != prepared["input_hash"]:
        raise RuntimeError("기준 실행과 새 M2의 학습·평가 입력이 다릅니다")
    if (
        payload.get("seed") != SEED
        or payload.get("split") != "historical_development_days_684_690"
        or payload.get("final_test") is not False
        or payload.get("holdout") is not False
    ):
        raise RuntimeError("기준 실행의 seed/split 정체성이 다릅니다")
    old_config = payload.get("config", {})
    if any(old_config.get(key) != value for key, value in expected_config.items()):
        raise RuntimeError("기준 M1/M2와 새 M2의 학습 설정이 다릅니다")

    arms = {arm.get("model_id"): arm for arm in payload.get("arms", [])}
    if set(arms) != {M1, M2_FIXED}:
        raise RuntimeError("기준 실행에 M1·고정계수 M2 두 arm이 필요합니다")
    for model_id, arm in arms.items():
        epochs = {row.get("epoch") for row in arm.get("curve", []) if "metrics" in row}
        if not {100, FIXED_EPOCH}.issubset(epochs):
            raise RuntimeError(f"기준 {model_id} 곡선에 100/300 epoch 평가가 없습니다")
        arm["reused_reference"] = True
    return [arms[M1], arms[M2_FIXED]]


def prepare(cfg=None) -> tuple[object, dict]:
    cfg = configure() if cfg is None else cfg
    prepared = m3._prepare(cfg)
    data = prepared["data"]
    features = build_features(
        data["train"],
        n_users=data["n_users"],
        n_items=data["n_items"],
        q_n=prepared["q_n"],
        q_v=prepared["q_v"],
        valid=prepared["clv_valid"],
        bandwidth=original.BANDWIDTH,
        reliability_floor=original.RELIABILITY_FLOOR,
    )
    if not np.array_equal(features["keys"], np.asarray(data["pos_key"], np.int64)):
        raise RuntimeError("M2 상품 프로필과 M1 TRAIN pair가 다릅니다")
    feature_sha = _feature_hash(features)
    initialization = dataset_parameter_initialization(features)
    reference_path = _reference_result_path()
    if not reference_path.is_file():
        raise FileNotFoundError(
            f"완료된 seed48 M1·고정계수 M2 결과가 필요합니다: {reference_path}"
        )
    references = _validate_reference(json.loads(reference_path.read_text()), prepared, cfg)
    settings = {
        "shared_l2": original.SHARED_L2,
        "bandwidth": original.BANDWIDTH,
        "reliability_floor": original.RELIABILITY_FLOOR,
        "axis_floor": 0.15,
        "rho_range": [0.03, 0.15],
        "dataset_initialization": initialization,
    }
    config_hash = hashlib.sha256(
        json.dumps(
            {
                "version": CODE_VERSION,
                "input_hash": prepared["input_hash"],
                "feature_hash": feature_sha,
                "settings": settings,
                "seed": SEED,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:12]
    prepared.update(
        out_dir=Path(cfg.out_dir),
        config_hash=config_hash,
        features=features,
        features_sha256=feature_sha,
        feature_settings=settings,
        reference_arms=references,
        reference_result=str(reference_path),
    )
    io._atomic_json(Path(cfg.out_dir) / "feature_diagnostic.json", {
        **features["diagnostics"], "dataset_initialization": initialization,
    })
    print(json.dumps({
        "code_version": CODE_VERSION,
        "fresh_fits": [M2_ADAPTIVE],
        "reused": [M1, M2_FIXED],
        "seed": SEED,
        "epochs": FIXED_EPOCH,
        "split": "historical_development_days_684_690",
        "test": False,
        "holdout": False,
        "dataset_initialization": initialization,
    }, ensure_ascii=False, indent=2), flush=True)
    return cfg, prepared


def _build(prepared: dict, cfg):
    v3.set_seed(SEED)
    data = prepared["data"]
    return DatasetAdaptiveOrthogonalNVLightGCN(
        n_users=data["n_users"],
        n_items=data["n_items"],
        features=prepared["features"],
        adj=data["adj"],
        id_dim=cfg.id_dim,
        n_layers=cfg.n_layers,
        pref_reg=cfg.pref_reg,
        shared_l2=original.SHARED_L2,
    ).to(v3.DEVICE)


def _run_arm(prepared: dict, cfg, spec: dict) -> dict:
    path = prepared["out_dir"] / "arms" / prepared["config_hash"] / f"{M2_ADAPTIVE}_s{SEED}.json"
    if path.is_file():
        payload = json.loads(path.read_text())
        if (
            payload.get("code_version") != CODE_VERSION
            or payload.get("input_hash") != prepared["input_hash"]
            or payload.get("config_hash") != prepared["config_hash"]
        ):
            raise RuntimeError(f"완료 캐시 신원이 다릅니다: {path}")
        print(f"[cached] {M2_ADAPTIVE} seed {SEED}", flush=True)
        return payload

    model = _build(prepared, cfg)
    store = ProgressStore(
        prepared["out_dir"] / "progress" / prepared["config_hash"],
        RunIdentity(
            stage="m2_dataset_adaptive_orthogonal_nv_seed48",
            model_id=M2_ADAPTIVE,
            seed=SEED,
            config_hash=prepared["config_hash"],
            source_revision=prepared["revision"],
            input_hash=prepared["input_hash"],
        ),
    )
    curve = capacity._train_curve(model, prepared, cfg, spec, SEED, store)
    payload = {
        **spec,
        "seed": SEED,
        "code_version": CODE_VERSION,
        "source_revision": prepared["revision"],
        "input_hash": prepared["input_hash"],
        "config_hash": prepared["config_hash"],
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "initialization": model.dataset_initialization,
        "final_representation_diagnostics": model.representation_diagnostics(),
        "curve": curve,
    }
    io._atomic_json(path, payload)
    store.mark_complete(epoch=cfg.epochs, max_epoch=cfg.epochs, selection="none",
                        checkpoint_path="", result_path=str(path))
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return payload


def _curve(arms: list[dict]) -> pd.DataFrame:
    rows = []
    for arm in arms:
        for record in arm["curve"]:
            if "metrics" not in record:
                continue
            rows.append({
                "model_id": arm["model_id"],
                "role": arm["role"],
                "seed": SEED,
                "epoch": record["epoch"],
                "loss": record["loss"],
                "p_correct": record["p_correct"],
                **record.get("gradient_diagnostics", {}),
                **record["metrics"],
            })
    return pd.DataFrame(rows)


def _comparison(curve: pd.DataFrame) -> pd.DataFrame:
    index = curve.set_index(["model_id", "seed", "epoch"])
    metrics = [column for column in curve.columns if "@" in column]
    rows = []
    for epoch in (100, FIXED_EPOCH):
        candidate = index.loc[(M2_ADAPTIVE, SEED, epoch)]
        for reference_id in (M1, M2_FIXED):
            reference = index.loc[(reference_id, SEED, epoch)]
            for metric in metrics:
                value, baseline = float(candidate[metric]), float(reference[metric])
                rows.append({
                    "seed": SEED, "epoch": epoch, "model_id": M2_ADAPTIVE,
                    "reference": reference_id, "metric": metric, "value": value,
                    "reference_value": baseline, "delta": value - baseline,
                    "ratio": value / baseline if baseline else np.nan,
                })
    return pd.DataFrame(rows)


def reading(curve: pd.DataFrame) -> dict:
    at = curve[curve.epoch.eq(FIXED_EPOCH)].set_index("model_id")
    candidate, m1, fixed = at.loc[M2_ADAPTIVE], at.loc[M1], at.loc[M2_FIXED]
    accuracy = {key: float(candidate[key] / m1[key]) for key in ACCURACY}
    economic = {key: float(candidate[key] / m1[key]) for key in ECONOMIC}
    geometric = float(np.exp(np.log(np.asarray(list(accuracy.values()))).mean()))
    candidate_pass = bool(
        geometric > 1
        and candidate["recall@10"] > m1["recall@10"]
        and candidate["ndcg@10"] > m1["ndcg@10"]
        and min(accuracy.values()) >= 0.99
        and min(economic.values()) >= 0.99
    )
    return {
        "development_screen_only": True,
        "fixed_epoch": FIXED_EPOCH,
        "seed": SEED,
        "accuracy_geometric_mean_ratio_vs_m1": geometric,
        "accuracy_ratios_vs_m1": accuracy,
        "economic_at10_ratios_vs_m1": economic,
        "candidate_vs_m1": candidate_pass,
        "all_at10_above_fixed_m2": bool(all(candidate[key] > fixed[key] for key in AT10)),
        "significance_claim": False,
        "clv_attribution_claim": False,
    }


def run(cfg, prepared: dict) -> dict:
    if _feature_hash(prepared["features"]) != prepared["features_sha256"]:
        raise ValueError("준비된 N/V 특징이 바뀌었습니다")
    print(f"\n===== {M2_ADAPTIVE} | seed {SEED} | {FIXED_EPOCH} epoch =====", flush=True)
    candidate = _run_arm(prepared, cfg, specs()[0])
    arms = [*prepared["reference_arms"], candidate]
    curve = _curve(arms)
    if curve.duplicated(["model_id", "seed", "epoch"]).any():
        raise RuntimeError("결과 곡선에 중복이 있습니다")
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
        "code_version": CODE_VERSION,
        "config": asdict(cfg),
        "seed": SEED,
        "source_revision": prepared["revision"],
        "input_hash": prepared["input_hash"],
        "split": "historical_development_days_684_690",
        "final_test": False,
        "holdout": False,
        "fresh_fit_count": 1,
        "reused_reference": prepared["reference_result"],
        "reading": result_reading,
        "feature_settings": prepared["feature_settings"],
        "feature_diagnostics": prepared["features"]["diagnostics"],
        "arms": arms,
        "paths": {key: str(value) for key, value in paths.items()},
        "limits": (
            "one repeatedly exposed development seed; no significance, generalization, "
            "CLV attribution, epoch selection, final-test or holdout claim"
        ),
    })
    return {"absolute": curve, "comparison": comparison,
            "reading": result_reading, "paths": paths}


def self_test() -> None:
    rows = []
    for model_id, factor in ((M1, 1.0), (M2_FIXED, 1.005), (M2_ADAPTIVE, 1.02)):
        rows.append({"model_id": model_id, "seed": SEED, "epoch": FIXED_EPOCH,
                     **{metric: factor for metric in (*ACCURACY, *ECONOMIC)}})
    result = reading(pd.DataFrame(rows))
    assert result["candidate_vs_m1"] and result["all_at10_above_fixed_m2"]


if __name__ == "__main__":
    self_test()
    print("dataset-adaptive orthogonal N/V seed48 self-test ok")
