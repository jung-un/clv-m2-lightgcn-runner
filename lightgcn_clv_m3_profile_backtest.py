"""One-seed historical pilot for a CLV user-profile relation graph.

This exploratory run trains on Dunnhumby DAY 1--676 and evaluates new-item
recommendations on DAY 677--683.  The already-inspected final test, holdout,
and the previously used DAY 684--690 development interval are not constructed.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import torch

from clv_m3_profile_graph import CLVProfileLightGCN, build_clv_profile_graph
from clv_run_state import ProgressStore, RunIdentity, clone_state, file_sha256
import lightgcn_clv_axis_specific_test10 as test10
import lightgcn_clv_moe as moe
import lightgcn_clv_v3 as v3


CODE_VERSION = "m3-clv-profile-relation-historical-pilot-v1"
MODEL_IDS = ("m1_baseline", "m3_clv_profile")
HISTORICAL_SPLIT = "historical_development_days_677_683"
ACCURACY_METRICS = (
    "recall@10",
    "ndcg@10",
    "recall@20",
    "ndcg@20",
    "recall@50",
    "ndcg@50",
)


@dataclass(frozen=True)
class M3CLVProfileBacktestConfig:
    dataset: str = "dunnhumby"
    seed: int = 42
    time_cutoff: int = 683
    evaluation_days: int = 7
    epochs: int = 100
    dim: int = 64
    n_layers: int = 2
    n_profile_bins: int = 10
    profile_alpha_init: float = 0.1
    batch_size: int = 8192
    lr: float = 5e-4
    pref_reg: float = 1e-3
    out_dir: str = ""


def _default_out_dir() -> str:
    if v3.IN_COLAB:
        return (
            "/content/drive/MyDrive/논문/data/"
            "results_m3_clv_profile_historical_dunnhumby"
        )
    return f"{v3.default_out_dir('dunnhumby')}_m3_clv_profile_historical"


def configure_m3_clv_profile_backtest(**overrides) -> M3CLVProfileBacktestConfig:
    return validate_config(
        M3CLVProfileBacktestConfig(
            **({"out_dir": _default_out_dir()} | overrides)
        )
    )


def validate_config(cfg: M3CLVProfileBacktestConfig) -> M3CLVProfileBacktestConfig:
    fixed = {
        "dataset": "dunnhumby",
        "seed": 42,
        "time_cutoff": 683,
        "evaluation_days": 7,
        "epochs": 100,
        "dim": 64,
        "n_layers": 2,
        "n_profile_bins": 10,
        "profile_alpha_init": 0.1,
    }
    for key, expected in fixed.items():
        if getattr(cfg, key) != expected:
            raise ValueError(f"historical CLV-profile pilot requires {key}={expected!r}")
    if cfg.batch_size <= 0 or cfg.lr <= 0 or cfg.pref_reg < 0:
        raise ValueError("invalid fixed-training setting")
    if not cfg.out_dir:
        raise ValueError("out_dir is required")
    return cfg


def preflight_summary(cfg: M3CLVProfileBacktestConfig) -> dict:
    cfg = validate_config(cfg)
    return {
        "code_version": CODE_VERSION,
        "dataset": cfg.dataset,
        "models": list(MODEL_IDS),
        "seed": cfg.seed,
        "epochs": cfg.epochs,
        "historical_development_split": {
            "train_end_inclusive": 676,
            "evaluation_start_inclusive": 677,
            "evaluation_end_inclusive": 683,
            "previous_days_684_690_constructed": False,
            "final_test_constructed": False,
            "holdout_constructed": False,
        },
        "validation_or_epoch_selection": False,
        "early_stopping": False,
        "new_item_task": (
            "all DAY 1--676 train pairs are excluded from evaluation truth and Top-K"
        ),
        "graph_intervention": {
            "purchase_graph": "unchanged binary M1 graph and normalization",
            "historical_clv_proxy": "N_hat * V_hat",
            "profile_nodes": "10 train-only mid-rank CLV percentile bins",
            "profile_path": "user -> shared CLV-profile node -> other users",
            "profile_message": "mean M1 representation of same-bin peers; self excluded",
            "fusion": "joint internal residual with one global learned sigmoid coefficient",
            "profile_alpha_initial": cfg.profile_alpha_init,
        },
        "fixed": {
            "min_user_interactions": 1,
            "min_item_interactions": 1,
            "loss": "plain pairwise BPR",
            "negative_sampling": "uniform",
            "sample_weighting": False,
            "m2_embedding": False,
            "m4_loss_weight": False,
            "single_optimizer": True,
        },
        "interpretation": (
            "exploratory one-seed historical pilot; no variance, interval, "
            "significance, generalization, or final-test claim"
        ),
        "out_dir": cfg.out_dir,
    }


def _base_config(cfg: M3CLVProfileBacktestConfig) -> dict:
    previous_cfg = dict(v3.CFG)
    previous_dcfg = v3.DCFG
    try:
        base = dict(
            v3.configure_run(
                cfg.dataset,
                out_dir=cfg.out_dir,
                ARCH="pref_only",
                SEED_LIST=[cfg.seed],
                WINDOW_DAYS=None,
                TIME_CUTOFF=cfg.time_cutoff,
                TRAIN_ON_VAL=True,
                VAL_DAYS=7,
                TEST_DAYS=cfg.evaluation_days,
                HOLDOUT_DAYS=0,
                EVAL_TEST=True,
                EVAL_HOLDOUT=False,
                GRAPH_MODE="binary",
                LOSS_MODE="plain",
                NEG_MODE="uniform",
                GATE_MODE="none",
                MIN_USER_INTER=1,
                MIN_ITEM_INTER=1,
                DIM=cfg.dim,
                N_LAYERS=cfg.n_layers,
                BATCH_SIZE=cfg.batch_size,
                LR=cfg.lr,
                PREF_REG=cfg.pref_reg,
                EPOCHS=cfg.epochs,
                EARLY_STOP=cfg.epochs,
                REPORT_LEGACY_VALUE_FEATURES=False,
            )
        )
    finally:
        v3.CFG.clear()
        v3.CFG.update(previous_cfg)
        v3.DCFG = previous_dcfg
    required = {
        "TIME_CUTOFF": 683,
        "TRAIN_ON_VAL": True,
        "TEST_DAYS": 7,
        "HOLDOUT_DAYS": 0,
        "EVAL_TEST": True,
        "EVAL_HOLDOUT": False,
        "GRAPH_MODE": "binary",
        "LOSS_MODE": "plain",
        "NEG_MODE": "uniform",
        "MIN_USER_INTER": 1,
        "MIN_ITEM_INTER": 1,
        "EPOCHS": 100,
        "EARLY_STOP": 100,
    }
    for key, expected in required.items():
        if base.get(key) != expected:
            raise RuntimeError(f"historical CLV-profile contamination: {key}")
    return base


def _config_hash(
    cfg: M3CLVProfileBacktestConfig, input_hash: str, revision: str
) -> str:
    payload = {
        "code_version": CODE_VERSION,
        "config": asdict(cfg),
        "models": MODEL_IDS,
        "input_hash": input_hash,
        "source_revision": revision,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()[:12]


def _prepare(cfg: M3CLVProfileBacktestConfig) -> dict:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = moe.build_input_manifest(v3.SCHEMA[cfg.dataset])
    input_hash = moe.manifest_hash(manifest)
    revision = moe.source_revision()
    base_cfg = _base_config(cfg)
    data = v3.prepare_data(base_cfg, v3.SCHEMA[cfg.dataset])
    if set(data["splits"]) != {"test"}:
        raise RuntimeError(f"historical evaluation contamination: {data['splits']}")
    if float(data["train"]["t"].max()) != 676.0:
        raise RuntimeError("historical train must end at DAY 676")
    if data.get("loss_w") is not None:
        raise RuntimeError("M4 loss weighting contaminated the M3 pilot")
    data["loss_w"] = None
    stats = data.get("data_stats", {})
    if stats.get("split_evaluation_status", {}).get("holdout") != "not_constructed":
        raise RuntimeError("holdout must not be constructed")
    if float(stats.get("source", {}).get("time_max", -1)) != 683.0:
        raise RuntimeError("rows after DAY 683 must not enter the pilot")

    profile = build_clv_profile_graph(
        data["train"], data["n_users"], n_profile_bins=cfg.n_profile_bins
    )
    stats["m3_clv_profile"] = profile.diagnostics
    thresholds = v3.segment_thresholds(profile.clv_proxy, base_cfg["SEG_EDGES"])
    cache = v3.EvalCache(
        *data["splits"]["test"],
        profile.clv_proxy,
        thresholds,
        data["n_items"],
    )
    meta = v3.item_meta(data["train"], data["n_items"])
    x_item, item_cat = v3.item_value_features(
        data["train"], data["n_items"], report=False
    )
    prepared = {
        "out_dir": out_dir,
        "manifest": manifest,
        "input_hash": input_hash,
        "revision": revision,
        "base_cfg": base_cfg,
        "data": data,
        "profile": profile,
        "cache": cache,
        "meta": meta,
        "x_item": x_item,
        "item_cat": item_cat,
    }
    prepared["config_hash"] = _config_hash(cfg, input_hash, revision)
    return prepared


def _build_model(
    prepared: dict, cfg: M3CLVProfileBacktestConfig, model_id: str
):
    data = prepared["data"]
    if model_id not in MODEL_IDS:
        raise KeyError(model_id)
    v3.set_seed(cfg.seed)
    if model_id == "m1_baseline":
        model = v3.build_model(
            data,
            data["x_val_u"],
            prepared["x_item"],
            prepared["item_cat"],
            prepared["base_cfg"],
        )
    else:
        model = CLVProfileLightGCN(
            data["n_users"],
            data["n_items"],
            data["n_cat"],
            data["x_val_u"],
            prepared["x_item"],
            prepared["item_cat"],
            prepared["base_cfg"],
            data["adj"],
            prepared["profile"],
            alpha_init=cfg.profile_alpha_init,
        ).to(v3.DEVICE)
    return model, list(model.pref_params())


def _arm_hash(
    prepared: dict, cfg: M3CLVProfileBacktestConfig, model_id: str
) -> str:
    payload = {
        "config_hash": prepared["config_hash"],
        "model_id": model_id,
        "seed": cfg.seed,
        "epochs": cfg.epochs,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()[:12]


def _arm_paths(prepared: dict, cfg: M3CLVProfileBacktestConfig, model_id: str):
    root = prepared["out_dir"] / "arms" / prepared["config_hash"]
    stem = f"{model_id}_s{cfg.seed}"
    return {"result": root / f"{stem}.json", "checkpoint": root / f"{stem}.pt"}


def _run_arm(
    prepared: dict, cfg: M3CLVProfileBacktestConfig, model_id: str
) -> dict:
    paths = _arm_paths(prepared, cfg, model_id)
    if paths["result"].exists():
        print(f"  [cached] {model_id} completed historical result reused")
        return json.loads(paths["result"].read_text(encoding="utf-8"))

    model, params = _build_model(prepared, cfg, model_id)
    store = ProgressStore(
        prepared["out_dir"] / "progress" / prepared["config_hash"],
        RunIdentity(
            stage="historical_development_train",
            model_id=model_id,
            seed=cfg.seed,
            config_hash=_arm_hash(prepared, cfg, model_id),
            source_revision=prepared["revision"],
            input_hash=prepared["input_hash"],
        ),
    )
    training = test10._fixed_epoch_train(
        model, params, prepared, cfg, model_id, cfg.seed, store
    )
    model.eval()
    paths["checkpoint"].parent.mkdir(parents=True, exist_ok=True)
    temporary = paths["checkpoint"].with_suffix(".pt.tmp")
    torch.save(
        {
            "state": clone_state(model),
            "model_id": model_id,
            "seed": cfg.seed,
            "config": asdict(cfg),
            "training": training,
            "source_revision": prepared["revision"],
            "input_hash": prepared["input_hash"],
        },
        temporary,
    )
    os.replace(temporary, paths["checkpoint"])
    metrics, _ = moe._flat_evaluation(
        model,
        0.0,
        prepared["cache"],
        prepared["meta"],
        prepared["data"],
        prepared["base_cfg"],
        per_user=False,
    )
    graph_diagnostics = None
    if model_id == "m3_clv_profile":
        graph_diagnostics = {
            **prepared["profile"].diagnostics,
            "model": model.profile_diagnostics(),
        }
    payload = {
        "model_id": model_id,
        "role": "baseline" if model_id == "m1_baseline" else "model",
        "seed": cfg.seed,
        "split": HISTORICAL_SPLIT,
        "final_epoch": cfg.epochs,
        "validation_selection": False,
        "evaluation_count": 1,
        "metrics": test10._public_metrics(metrics),
        "graph_diagnostics": graph_diagnostics,
        "training": training,
        "checkpoint": str(paths["checkpoint"]),
        "checkpoint_sha256": file_sha256(paths["checkpoint"]),
    }
    test10._atomic_json(paths["result"], payload)
    store.mark_complete(
        epoch=cfg.epochs,
        max_epoch=cfg.epochs,
        selection="none",
        split=HISTORICAL_SPLIT,
        evaluation_count=1,
        checkpoint_path=str(paths["checkpoint"]),
        result_path=str(paths["result"]),
    )
    return payload


def _pilot_decision(frame: pd.DataFrame) -> dict:
    rows = frame.set_index("model_id")
    base = rows.loc["m1_baseline"]
    model = rows.loc["m3_clv_profile"]
    accuracy = {
        metric: float(model[metric] / base[metric]) for metric in ACCURACY_METRICS
    }
    price_ratio = float(
        model["mean_recommended_price_percentile@10"]
        / base["mean_recommended_price_percentile@10"]
    )
    catalog_ratio = float(model["n_distinct@10"] / base["n_distinct@10"])
    exposure_delta = float(model["top10_share@10"] - base["top10_share@10"])
    checks = {
        "accuracy_guard": all(value >= 0.99 for value in accuracy.values()),
        "weighted_hit_improved": bool(
            model["price_purchase_amount_weighted_hit@10"]
            > base["price_purchase_amount_weighted_hit@10"]
        ),
        "price_guard": 0.97 <= price_ratio <= 1.03,
        "catalog_guard": catalog_ratio >= 0.95,
        "exposure_guard": exposure_delta <= 0.01,
    }
    return {
        "passes_pilot": all(checks.values()),
        "checks": checks,
        "accuracy_ratios": accuracy,
        "weighted_hit_delta": float(
            model["price_purchase_amount_weighted_hit@10"]
            - base["price_purchase_amount_weighted_hit@10"]
        ),
        "price_ratio": price_ratio,
        "catalog_ratio": catalog_ratio,
        "top10_share_absolute_delta": exposure_delta,
        "single_seed_limitation": (
            "no variance, interval, significance, or generalization claim"
        ),
    }


def _comparison(frame: pd.DataFrame) -> pd.DataFrame:
    metadata = {"model_id", "role", "seed", "split", "final_epoch"}
    metrics = [column for column in frame.columns if column not in metadata]
    rows = frame.set_index("model_id")
    base = rows.loc["m1_baseline"]
    model = rows.loc["m3_clv_profile"]
    output = []
    for metric in metrics:
        baseline_value = base[metric]
        model_value = model[metric]
        output.append(
            {
                "model_id": "m3_clv_profile",
                "reference": "m1_baseline",
                "metric": metric,
                "reference_value": baseline_value,
                "model_value": model_value,
                "absolute_delta": model_value - baseline_value,
                "relative_change_pct": (
                    100.0 * (model_value - baseline_value) / baseline_value
                    if baseline_value != 0
                    else None
                ),
            }
        )
    return pd.DataFrame(output)


def run_m3_clv_profile_backtest(
    cfg: M3CLVProfileBacktestConfig | None = None,
) -> pd.DataFrame:
    cfg = validate_config(cfg or configure_m3_clv_profile_backtest())
    print(json.dumps(preflight_summary(cfg), ensure_ascii=False, indent=2))
    prepared = _prepare(cfg)
    arms = []
    for model_id in MODEL_IDS:
        print(f"\n===== {model_id} | seed 42 | fixed 100 epochs =====")
        arms.append(_run_arm(prepared, cfg, model_id))
    frame = pd.DataFrame(
        [
            {
                "model_id": arm["model_id"],
                "role": arm["role"],
                "seed": arm["seed"],
                "split": arm["split"],
                "final_epoch": arm["final_epoch"],
                **arm["metrics"],
            }
            for arm in arms
        ]
    )
    comparison = _comparison(frame)
    decision = _pilot_decision(frame)
    stem = f"m3_clv_profile_backtest_{prepared['config_hash']}"
    paths = {
        "absolute_csv": prepared["out_dir"] / f"{stem}.csv",
        "comparison_csv": prepared["out_dir"] / f"{stem}_comparison.csv",
        "json": prepared["out_dir"] / f"{stem}.json",
    }
    test10._atomic_csv(paths["absolute_csv"], frame)
    test10._atomic_csv(paths["comparison_csv"], comparison)
    test10._atomic_json(
        paths["json"],
        {
            "code_version": CODE_VERSION,
            "source_revision": prepared["revision"],
            "config": asdict(cfg),
            "preflight": preflight_summary(cfg),
            "input_manifest": prepared["manifest"],
            "data_stats": prepared["data"].get("data_stats", {}),
            "graph_diagnostics": prepared["profile"].diagnostics,
            "arms": arms,
            "absolute_rows": frame.to_dict("records"),
            "comparison_rows": comparison.to_dict("records"),
            "pilot_decision": decision,
            "result_paths": {key: str(value) for key, value in paths.items()},
            "interpretation": (
                "exploratory one-seed historical pilot; passing only opens "
                "the shuffled-profile, N-only/V-only, multi-seed, or external-data stage"
            ),
        },
    )
    frame.attrs["comparison"] = comparison
    frame.attrs["pilot_decision"] = decision
    frame.attrs["result_paths"] = {key: str(value) for key, value in paths.items()}
    frame.attrs["profile_diagnostics"] = prepared["profile"].diagnostics
    print("\nHistorical absolute metrics:")
    print(frame.to_string(index=False))
    print("\nM3 - M1:")
    print(comparison.to_string(index=False))
    print("\nPilot decision:")
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    print("Result files:", frame.attrs["result_paths"])
    return frame


if __name__ == "__main__":
    print(
        json.dumps(
            preflight_summary(configure_m3_clv_profile_backtest()),
            ensure_ascii=False,
            indent=2,
        )
    )
