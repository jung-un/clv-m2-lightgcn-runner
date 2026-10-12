"""Frozen standalone adaptive M2 on each dataset's final week.

The model, epoch budget and seeds are fixed before test. Each completed seed
is scored once at epoch 300; partial seed batches are only for resumption.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from clv_dataset_adaptive_orthogonal_nv_model import DatasetAdaptiveOrthogonalNVLightGCN, dataset_parameter_initialization
from clv_reliability_orthogonal_nv_model import build_features
from clv_run_state import ProgressStore, RunIdentity
import clv_reliability_orthogonal_nv_screen as development
import clv_m2_m5_lastweek_test10 as lastweek
import lightgcn_clv_m3_dunnhumby_test1 as fixed


v3, io = fixed.v3, fixed.test10
CODE_VERSION = "clv-m2-adaptive-lastweek-final10-v1"
MODEL_ID = "m2_dataset_adaptive_orthogonal_nv_bpr_k1_lastweek_final10"
SEEDS = tuple(range(42, 52))


@dataclass(frozen=True)
class Config:
    dataset: str
    out_dir: str
    seeds: tuple[int, ...] = SEEDS
    epochs: int = 300
    id_dim: int = 64
    n_layers: int = 2
    batch_size: int = 8192
    lr: float = 5e-4
    pref_reg: float = 1e-3
    negative_count: int = 1
    input_days: int = 365
    bandwidth: float = development.BANDWIDTH
    reliability_floor: float = development.RELIABILITY_FLOOR
    shared_l2: float = development.SHARED_L2


def configure(dataset: str, *, out_dir: str | None = None) -> Config:
    if dataset not in ("dunnhumby", "hm"):
        raise ValueError("dataset은 dunnhumby 또는 hm입니다")
    return Config(
        dataset=dataset,
        out_dir=out_dir or v3.default_out_dir(dataset) + "_m2_adaptive_lastweek_final10_v1",
        seeds=SEEDS if dataset == "dunnhumby" else (49,),
        batch_size=8192 if dataset == "dunnhumby" else 131072,
    )


def _validate_config(cfg: Config) -> None:
    if cfg != configure(cfg.dataset, out_dir=cfg.out_dir):
        raise ValueError("사전 고정한 Dunnhumby 10시드/H&M seed49·300epoch·M2 설정은 변경할 수 없습니다")


def base_config(cfg: Config) -> dict:
    _validate_config(cfg)
    return lastweek.base_config(cfg)


def preflight(cfg: Config) -> dict:
    _validate_config(cfg)
    return {
        "code_version": CODE_VERSION,
        "question": "Does dataset-adaptive N/V M2 alone improve final-week new-item recommendation versus same-condition M1?",
        "dataset": cfg.dataset,
        "split": "train DAY 1-704; test DAY 705-711" if cfg.dataset == "dunnhumby"
                 else "train through 2020-09-15; test 2020-09-16--22",
        "model_id": MODEL_ID,
        "compute_seeds": list(cfg.seeds),
        "final_reporting_seeds": list(SEEDS),
        "fixed_epoch": 300,
        "graph": "binary",
        "negative_sampling": "uniform K=1",
        "loss": "plain BPR with unchanged ID/shared L2; no row weights or extra objective",
        "clv_input": "last 365 training days; q_N/q_V fixed observations; N/V and item-economic encoders jointly optimized",
        "axis_weights": "positive, sum=3, each >=0.15; train-only dataset initialization",
        "rho_range": [0.03, 0.15],
        "test_evaluations_per_fit": 1,
        "validation": False,
        "holdout": False,
        "early_stopping": False,
        "minimum_item_interactions": 1,
        "new_item_only": True,
        "partial_seed_results": "H&M seed49 is a first final-week computation batch only; no retuning or final success decision",
        "final_decision": "descriptive paired 10-seed mean/std versus verified same-split M1; report all 142 metrics and adverse outcomes",
        "significance_claim": False,
        "unseen_confirmation_claim": False,
    }


def _feature_hash(features: dict) -> str:
    digest = hashlib.sha256()
    for key in sorted(key for key in features if key != "diagnostics"):
        values = np.ascontiguousarray(features[key])
        digest.update(f"{key}:{values.dtype}:{values.shape}".encode())
        digest.update(values.tobytes())
    return digest.hexdigest()


def prepare(cfg: Config) -> dict:
    _validate_config(cfg)
    manifest = fixed.moe.build_input_manifest(v3.SCHEMA[cfg.dataset])
    input_hash, revision = fixed.moe.manifest_hash(manifest), fixed.moe.source_revision()
    base = base_config(cfg)
    data = v3.prepare_data(base, v3.DCFG)
    lastweek.validate_split(data, cfg)
    if data.get("loss_w") is not None:
        raise RuntimeError("M2는 BPR 행 가중치를 사용할 수 없습니다")
    snapshot = fixed.residual.build_final_snapshot(
        data["train"], data["n_users"], v3.DCFG["is_date"], cfg.input_days,
    )
    axes = fixed.joint.build_user_axis_inputs(snapshot, data["n_users"])
    q_n, q_v, _, valid = fixed.evaluation.build_clv_inputs(axes)
    q_n, q_v = (np.where(valid, q, 0).astype(np.float32) for q in (q_n, q_v))
    features = build_features(
        data["train"], n_users=data["n_users"], n_items=data["n_items"],
        q_n=q_n, q_v=q_v, valid=valid, bandwidth=cfg.bandwidth,
        reliability_floor=cfg.reliability_floor,
    )
    if not np.array_equal(features["keys"], np.asarray(data["pos_key"], np.int64)):
        raise RuntimeError("M2 특징의 TRAIN pair가 이진 그래프와 다릅니다")
    feature_sha = _feature_hash(features)
    initialization = dataset_parameter_initialization(features)
    identity = {"code_version": CODE_VERSION, "config": asdict(cfg),
                "input_hash": input_hash, "source_revision": revision,
                "feature_sha256": feature_sha}
    config_hash = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
    run_dir = Path(cfg.out_dir) / config_hash
    protocol = preflight(cfg) | {
        "config": asdict(cfg), "input_hash": input_hash,
        "source_revision": revision, "feature_sha256": feature_sha,
        "feature_diagnostics": features["diagnostics"],
        "dataset_initialization": initialization, "data_stats": data["data_stats"],
    }
    protocol_path = run_dir / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != json.loads(json.dumps(protocol)):
        raise RuntimeError("기존 실행 경로의 고정 protocol이 다릅니다")
    io._atomic_json(protocol_path, protocol)
    thresholds = v3.segment_thresholds(axes["clv_proxy"], base["SEG_EDGES"])
    return {
        "data": data, "base_cfg": base, "axes": axes, "features": features,
        "feature_sha256": feature_sha, "input_hash": input_hash,
        "revision": revision, "config_hash": config_hash, "run_dir": run_dir,
        "protocol": protocol, "meta": v3.item_meta(data["train"], data["n_items"]),
        "cache": v3.EvalCache(*data["splits"]["test"], axes["clv_proxy"], thresholds,
                              data["n_items"]),
    }


def _arm_path(prepared: dict, seed: int) -> Path:
    return prepared["run_dir"] / "arms" / f"{MODEL_ID}_s{seed}.json"


def _load_completed(prepared: dict, seed: int) -> dict | None:
    path = _arm_path(prepared, seed)
    if not path.exists():
        return None
    row = json.loads(path.read_text())
    identity = RunIdentity(CODE_VERSION, MODEL_ID, seed, prepared["config_hash"],
                           prepared["revision"], prepared["input_hash"])
    if row.get("identity") != asdict(identity) or row.get("epochs") != 300:
        raise RuntimeError(f"완료 결과의 신원이 다릅니다: {path}")
    return row


def run_seed(prepared: dict, cfg: Config, seed: int) -> dict:
    _validate_config(cfg)
    if seed not in SEEDS:
        raise ValueError("사전 지정한 seed42~51만 허용합니다")
    completed = _load_completed(prepared, seed)
    if completed is not None:
        print(f"[재사용] {cfg.dataset} M2 seed{seed}", flush=True)
        return completed
    if _feature_hash(prepared["features"]) != prepared["feature_sha256"]:
        raise RuntimeError("준비된 N/V 특징이 바뀌었습니다")
    data = prepared["data"]
    v3.set_seed(seed)
    model = DatasetAdaptiveOrthogonalNVLightGCN(
        n_users=data["n_users"], n_items=data["n_items"],
        features=prepared["features"], adj=data["adj"], id_dim=cfg.id_dim,
        n_layers=cfg.n_layers, pref_reg=cfg.pref_reg, shared_l2=cfg.shared_l2,
    ).to(v3.DEVICE)
    identity = RunIdentity(CODE_VERSION, MODEL_ID, seed, prepared["config_hash"],
                           prepared["revision"], prepared["input_hash"])
    store = ProgressStore(prepared["run_dir"] / "progress", identity)
    history = fixed._train(model, prepared, cfg, MODEL_ID, seed, store, row_weights=None)
    if len(history) != 300 or history[-1]["epoch"] != 300:
        raise RuntimeError("300epoch 미완료: test를 평가하지 않습니다")
    metrics = fixed._evaluate(model, prepared)
    if len(metrics) != 142 or not np.isfinite(list(metrics.values())).all():
        raise RuntimeError("최종 142지표 누락 또는 비유한값")
    row = {
        "identity": asdict(identity), "model_id": MODEL_ID, "seed": seed,
        "epochs": 300, "metrics": metrics, "training_history": history,
        "diagnostics": {**model.representation_diagnostics(),
                        **model.training_gradient_diagnostics(),
                        "graph": "binary", "row_weighted": False,
                        "feature_sha256": prepared["feature_sha256"]},
    }
    path = _arm_path(prepared, seed)
    io._atomic_json(path, row)
    store.mark_complete(epoch=300, max_epoch=300, selection="none",
                        checkpoint_path=str(store.latest_checkpoint), result_path=str(path))
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return row


def completed_rows(prepared: dict) -> list[dict]:
    seeds = tuple(prepared["protocol"]["compute_seeds"])
    return [row for seed in seeds if (row := _load_completed(prepared, seed)) is not None]


def report(prepared: dict) -> dict:
    rows = completed_rows(prepared)
    absolute = pd.DataFrame([{"model_id": MODEL_ID, "seed": row["seed"], **row["metrics"]}
                             for row in rows])
    diagnostics = pd.DataFrame([{"seed": row["seed"], **row["diagnostics"]}
                                for row in rows])
    complete = {row["seed"] for row in rows} == set(SEEDS)
    root = prepared["run_dir"] / ("reports" if complete else "partial_reports")
    paths = {key: str(root / f"{key}.csv") for key in ("absolute", "diagnostics")}
    io._atomic_csv(Path(paths["absolute"]), absolute)
    io._atomic_csv(Path(paths["diagnostics"]), diagnostics)
    result = {"absolute": absolute, "diagnostics": diagnostics,
              "complete": complete, "completed_seeds": [row["seed"] for row in rows]}
    if complete:
        if len(absolute) != 10 or len([c for c in absolute if c not in ("model_id", "seed")]) != 142:
            raise RuntimeError("10시드×142지표가 완성되지 않았습니다")
        summary = absolute.melt(id_vars=["model_id", "seed"], var_name="metric", value_name="value")
        summary = summary.groupby(["model_id", "metric"]).value.agg(["mean", "std", "count"]).reset_index()
        paths["summary"] = str(root / "summary.csv")
        io._atomic_csv(Path(paths["summary"]), summary)
        result["summary"] = summary
    paths["json"] = str(root / "result.json")
    io._atomic_json(Path(paths["json"]), {
        "protocol": prepared["protocol"],
        "complete": complete, "completed_seeds": result["completed_seeds"],
        "final_comparison_to_m1_pending_verified_reference": True,
        "retuning_from_partial_results_allowed": False,
        "significance_claim": False, "result_paths": paths,
    })
    result["paths"] = paths
    return result


def run_batch(prepared: dict, cfg: Config, *, seeds: tuple[int, ...] | None = None) -> dict:
    _validate_config(cfg)
    seeds = cfg.seeds if seeds is None else tuple(seeds)
    if not seeds or len(set(seeds)) != len(seeds) or not set(seeds).issubset(cfg.seeds):
        raise ValueError("사전 지정한 계산 시드의 중복 없는 부분집합이어야 합니다")
    for seed in seeds:
        print(f"[학습] {cfg.dataset} 적응형 M2 seed{seed}/300epoch", flush=True)
        run_seed(prepared, cfg, seed)
    return report(prepared)
