"""One-seed historical pilot for a CLV-profile-to-item relation graph."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass

import pandas as pd
import torch

from clv_m3_profile_item_graph import (
    CLVProfileItemLightGCN,
    build_clv_profile_item_graph,
)
from clv_run_state import ProgressStore, RunIdentity, clone_state, file_sha256
import lightgcn_clv_axis_specific_test10 as test10
import lightgcn_clv_m3_profile_backtest as previous
import lightgcn_clv_moe as moe
import lightgcn_clv_v3 as v3


CODE_VERSION = "m3-clv-profile-item-ppmi-historical-pilot-v1"
MODEL_IDS = ("m1_baseline", "m3_clv_profile_item")
HISTORICAL_SPLIT = "historical_development_days_677_683"
ACCURACY_METRICS = previous.ACCURACY_METRICS


@dataclass(frozen=True)
class M3CLVProfileItemBacktestConfig(previous.M3CLVProfileBacktestConfig):
    pass


def _default_out_dir() -> str:
    if v3.IN_COLAB:
        return (
            "/content/drive/MyDrive/논문/data/"
            "results_m3_clv_profile_item_historical_dunnhumby"
        )
    return f"{v3.default_out_dir('dunnhumby')}_m3_clv_profile_item_historical"


def configure_m3_clv_profile_item_backtest(
    **overrides,
) -> M3CLVProfileItemBacktestConfig:
    return validate_config(
        M3CLVProfileItemBacktestConfig(
            **({"out_dir": _default_out_dir()} | overrides)
        )
    )


def validate_config(
    cfg: M3CLVProfileItemBacktestConfig,
) -> M3CLVProfileItemBacktestConfig:
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
            raise ValueError(
                f"historical CLV profile-item pilot requires {key}={expected!r}"
            )
    if cfg.batch_size <= 0 or cfg.lr <= 0 or cfg.pref_reg < 0:
        raise ValueError("invalid fixed-training setting")
    if not cfg.out_dir:
        raise ValueError("out_dir is required")
    return cfg


def preflight_summary(cfg: M3CLVProfileItemBacktestConfig) -> dict:
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
            "profile_item_weight": (
                "row-normalized positive pointwise mutual information from "
                "train-only unique user-item pairs"
            ),
            "profile_path": "user -> CLV profile -> profile-selective items",
            "nonselective_item_relation": "zero",
            "item_price_used": False,
            "fusion": (
                "joint internal user residual with one global learned sigmoid coefficient"
            ),
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


def _config_hash(cfg, input_hash: str, revision: str) -> str:
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


def _prepare(cfg: M3CLVProfileItemBacktestConfig) -> dict:
    prepared = previous._prepare(cfg)
    data = prepared["data"]
    profile_item = build_clv_profile_item_graph(
        data["train"],
        data["n_users"],
        data["n_items"],
        n_profile_bins=cfg.n_profile_bins,
    )
    stats = data.get("data_stats", {})
    stats.pop("m3_clv_profile", None)
    stats["m3_clv_profile_item"] = profile_item.diagnostics
    prepared["profile_item"] = profile_item
    prepared.pop("profile", None)
    prepared["config_hash"] = _config_hash(
        cfg, prepared["input_hash"], prepared["revision"]
    )
    return prepared


def _build_model(prepared: dict, cfg, model_id: str):
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
        model = CLVProfileItemLightGCN(
            data["n_users"],
            data["n_items"],
            data["n_cat"],
            data["x_val_u"],
            prepared["x_item"],
            prepared["item_cat"],
            prepared["base_cfg"],
            data["adj"],
            prepared["profile_item"],
            alpha_init=cfg.profile_alpha_init,
        ).to(v3.DEVICE)
    return model, list(model.pref_params())


def _arm_hash(prepared: dict, cfg, model_id: str) -> str:
    payload = {
        "config_hash": prepared["config_hash"],
        "model_id": model_id,
        "seed": cfg.seed,
        "epochs": cfg.epochs,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()[:12]


def _arm_paths(prepared: dict, cfg, model_id: str):
    root = prepared["out_dir"] / "arms" / prepared["config_hash"]
    stem = f"{model_id}_s{cfg.seed}"
    return {"result": root / f"{stem}.json", "checkpoint": root / f"{stem}.pt"}


def _run_arm(prepared: dict, cfg, model_id: str) -> dict:
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
    if model_id == "m3_clv_profile_item":
        graph_diagnostics = {
            **prepared["profile_item"].diagnostics,
            "model": model.profile_item_diagnostics(),
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
    model = rows.loc["m3_clv_profile_item"]
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
    rows = frame.set_index("model_id")
    base = rows.loc["m1_baseline"]
    model = rows.loc["m3_clv_profile_item"]
    output = []
    for metric in (column for column in frame.columns if column not in metadata):
        baseline_value = base[metric]
        model_value = model[metric]
        output.append(
            {
                "model_id": "m3_clv_profile_item",
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


def run_m3_clv_profile_item_backtest(
    cfg: M3CLVProfileItemBacktestConfig | None = None,
) -> pd.DataFrame:
    cfg = validate_config(cfg or configure_m3_clv_profile_item_backtest())
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
    stem = f"m3_clv_profile_item_backtest_{prepared['config_hash']}"
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
            "graph_diagnostics": prepared["profile_item"].diagnostics,
            "arms": arms,
            "absolute_rows": frame.to_dict("records"),
            "comparison_rows": comparison.to_dict("records"),
            "pilot_decision": decision,
            "result_paths": {key: str(value) for key, value in paths.items()},
            "interpretation": (
                "exploratory one-seed historical pilot; passing only opens "
                "the multi-seed or external-data stage"
            ),
        },
    )
    frame.attrs["comparison"] = comparison
    frame.attrs["pilot_decision"] = decision
    frame.attrs["result_paths"] = {key: str(value) for key, value in paths.items()}
    frame.attrs["profile_item_diagnostics"] = {
        **prepared["profile_item"].diagnostics,
        "model": arms[1]["graph_diagnostics"]["model"],
    }
    return frame


if __name__ == "__main__":
    print(
        json.dumps(
            preflight_summary(configure_m3_clv_profile_item_backtest()),
            ensure_ascii=False,
            indent=2,
        )
    )
