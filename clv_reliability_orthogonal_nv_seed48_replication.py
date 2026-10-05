"""Fresh-seed replication of M1 versus reliability-gated orthogonal N/V M2.

Both arms are trained from scratch on Dunnhumby development days 684--690
with seed 48 and a fixed 300-epoch budget.  No M3/M4/M5, permutation arm,
test split, or holdout is opened.
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

from clv_reliability_orthogonal_nv_model import ReliabilityOrthogonalNVLightGCN, build_features
from clv_run_state import ProgressStore, RunIdentity
import clv_reliability_orthogonal_nv_screen as original
import lightgcn_clv_axis_specific_test10 as io
import lightgcn_clv_m2_capacity_search as capacity
import lightgcn_clv_m3_centered_value_graph as m3
import lightgcn_clv_v3 as v3


CODE_VERSION = "clv-reliability-orthogonal-nv-seed48-replication-v1"
SEED = 48
FIXED_EPOCH = 300
M1 = m3.M1_MODEL_ID
M2 = original.M2
ACCURACY = original.ACCURACY
ECONOMIC = original.ECONOMIC


def specs() -> list[dict]:
    return [
        {
            "model_id": M1,
            "condition": "baseline",
            "role": "M1",
            "id_dim": 64,
            "pref_reg": 1e-3,
        },
        {
            "model_id": M2,
            "arm": "binary_graph_plain_bpr",
            "role": "M2",
            "id_dim": 64,
            "pref_reg": 1e-3,
        },
    ]


def configure(**overrides):
    defaults = {
        "seeds": (SEED,),
        "epochs": FIXED_EPOCH,
        "eval_every": 25,
        "out_dir": (
            f"{v3.default_out_dir('dunnhumby')}"
            "_clv_reliability_orthogonal_nv_seed48_replication_v1"
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


def prepare(cfg=None) -> tuple[object, dict]:
    cfg = configure() if cfg is None else cfg
    if cfg.seeds != (SEED,) or cfg.epochs != FIXED_EPOCH:
        raise ValueError("seed48·300epoch 설정이 필요합니다")
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
    settings = {
        "rho": original.RHO,
        "shared_l2": original.SHARED_L2,
        "bandwidth": original.BANDWIDTH,
        "reliability_floor": original.RELIABILITY_FLOOR,
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
    )
    io._atomic_json(Path(cfg.out_dir) / "feature_diagnostic.json", features["diagnostics"])
    print(
        json.dumps(
            {
                "code_version": CODE_VERSION,
                "fresh_fits": [M1, M2],
                "seed": SEED,
                "epochs": FIXED_EPOCH,
                "split": "historical_development_days_684_690",
                "test": False,
                "holdout": False,
                "feature_settings": settings,
                "feature_diagnostics": features["diagnostics"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return cfg, prepared


def _build(prepared: dict, cfg, spec: dict):
    v3.set_seed(SEED)
    if spec["model_id"] == M1:
        return capacity._build_model(prepared, cfg, spec, SEED)
    data = prepared["data"]
    return ReliabilityOrthogonalNVLightGCN(
        n_users=data["n_users"],
        n_items=data["n_items"],
        features=prepared["features"],
        adj=data["adj"],
        id_dim=cfg.id_dim,
        n_layers=cfg.n_layers,
        pref_reg=cfg.pref_reg,
        shared_l2=original.SHARED_L2,
        rho=original.RHO,
    ).to(v3.DEVICE)


def _run_arm(prepared: dict, cfg, spec: dict) -> dict:
    path = (
        prepared["out_dir"]
        / "arms"
        / prepared["config_hash"]
        / f"{spec['model_id']}_s{SEED}.json"
    )
    if path.is_file():
        payload = json.loads(path.read_text())
        if (
            payload.get("code_version") != CODE_VERSION
            or payload.get("input_hash") != prepared["input_hash"]
            or payload.get("config_hash") != prepared["config_hash"]
        ):
            raise RuntimeError(f"완료 캐시 신원이 다릅니다: {path}")
        print(f"[cached] {spec['model_id']} seed {SEED}", flush=True)
        return payload

    model = _build(prepared, cfg, spec)
    store = ProgressStore(
        prepared["out_dir"] / "progress" / prepared["config_hash"],
        RunIdentity(
            stage="reliability_orthogonal_nv_seed48_replication",
            model_id=spec["model_id"],
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
        "curve": curve,
    }
    io._atomic_json(path, payload)
    store.mark_complete(
        epoch=cfg.epochs,
        max_epoch=cfg.epochs,
        selection="none",
        checkpoint_path="",
        result_path=str(path),
    )
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
            rows.append(
                {
                    "model_id": arm["model_id"],
                    "role": arm["role"],
                    "seed": SEED,
                    "epoch": record["epoch"],
                    "loss": record["loss"],
                    "p_correct": record["p_correct"],
                    **{
                        key: value
                        for key, value in record.get("score_split", {}).items()
                        if key != "clv_score_measured_on"
                    },
                    **record.get("gradient_diagnostics", {}),
                    **record["metrics"],
                }
            )
    return pd.DataFrame(rows)


def _comparison(curve: pd.DataFrame) -> pd.DataFrame:
    index = curve.set_index(["model_id", "seed", "epoch"])
    metrics = [column for column in curve.columns if "@" in column]
    rows = []
    for epoch in (100, FIXED_EPOCH):
        candidate = index.loc[(M2, SEED, epoch)]
        baseline = index.loc[(M1, SEED, epoch)]
        for metric in metrics:
            value, reference = float(candidate[metric]), float(baseline[metric])
            rows.append(
                {
                    "seed": SEED,
                    "epoch": epoch,
                    "model_id": M2,
                    "reference": M1,
                    "metric": metric,
                    "value": value,
                    "reference_value": reference,
                    "delta": value - reference,
                    "ratio": value / reference if reference else np.nan,
                }
            )
    return pd.DataFrame(rows)


def reading(curve: pd.DataFrame) -> dict:
    at = curve[curve.epoch.eq(FIXED_EPOCH)].set_index("model_id")
    candidate, reference = at.loc[M2], at.loc[M1]
    accuracy = {key: float(candidate[key] / reference[key]) for key in ACCURACY}
    economic = {key: float(candidate[key] / reference[key]) for key in ECONOMIC}
    geometric = float(np.exp(np.log(np.asarray(list(accuracy.values()))).mean()))
    candidate_pass = bool(
        geometric > 1
        and candidate["recall@10"] > reference["recall@10"]
        and candidate["ndcg@10"] > reference["ndcg@10"]
        and min(accuracy.values()) >= 0.99
        and min(economic.values()) >= 0.99
    )
    return {
        "development_replication_only": True,
        "fixed_epoch": FIXED_EPOCH,
        "seed": SEED,
        "accuracy_geometric_mean_ratio": geometric,
        "accuracy_ratios": accuracy,
        "economic_at10_ratios": economic,
        "candidate": candidate_pass,
        "significance_claim": False,
        "clv_attribution_claim": False,
    }


def run(cfg, prepared: dict) -> dict:
    if _feature_hash(prepared["features"]) != prepared["features_sha256"]:
        raise ValueError("준비된 N/V 특징이 바뀌었습니다")
    arms = []
    for spec in specs():
        print(f"\n===== {spec['model_id']} | seed {SEED} | {FIXED_EPOCH} epoch =====", flush=True)
        arms.append(_run_arm(prepared, cfg, spec))
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
    io._atomic_json(
        paths["json"],
        {
            "code_version": CODE_VERSION,
            "config": asdict(cfg),
            "seed": SEED,
            "source_revision": prepared["revision"],
            "input_hash": prepared["input_hash"],
            "split": "historical_development_days_684_690",
            "final_test": False,
            "holdout": False,
            "fresh_fit_count": 2,
            "reading": result_reading,
            "feature_settings": prepared["feature_settings"],
            "feature_diagnostics": prepared["features"]["diagnostics"],
            "arms": arms,
            "paths": {key: str(value) for key, value in paths.items()},
            "limits": (
                "one fresh development seed; no significance, generalization, "
                "CLV attribution, epoch selection, test or holdout claim"
            ),
        },
    )
    return {
        "absolute": curve,
        "comparison": comparison,
        "reading": result_reading,
        "paths": paths,
    }


def self_test() -> None:
    rows = []
    for model_id, factor in ((M1, 1.0), (M2, 1.01)):
        rows.append(
            {
                "model_id": model_id,
                "seed": SEED,
                "epoch": FIXED_EPOCH,
                **{metric: factor for metric in (*ACCURACY, *ECONOMIC)},
            }
        )
    result = reading(pd.DataFrame(rows))
    assert result["candidate"] and result["seed"] == SEED


if __name__ == "__main__":
    self_test()
    print("seed48 M1/M2 replication self-test ok")
