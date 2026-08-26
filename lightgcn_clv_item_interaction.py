"""One-seed historical screen for a minimal CLV-conditioned item interaction."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
import torch

from clv_item_interaction_model import CLVItemInteractionLightGCN
from clv_run_state import ProgressStore, RunIdentity, clone_state, file_sha256
from lightgcn_clv_information_ceiling_diagnostic import (
    ACCURACY_METRICS,
    _balanced_accuracy_index,
)
import lightgcn_clv_axis_specific_test10 as test10
import lightgcn_clv_gatefree_lowdim as previous_screen
import lightgcn_clv_moe as moe
import lightgcn_clv_v3 as v3


CODE_VERSION = "m2-clv-item-interaction-historical-screen-v1"
MODEL_IDS = (
    "m2_clv_item_interaction",
    "m2_clv_item_interaction_shuffle",
)
HISTORICAL_SPLIT = "historical_development_days_684_690"


@dataclass(frozen=True)
class CLVItemInteractionConfig:
    dataset: str = "dunnhumby"
    seed: int = 42
    shuffle_seed: int = 20260826
    time_cutoff: int = 690
    evaluation_days: int = 7
    epochs: int = 100
    embedding_dim: int = 64
    n_layers: int = 2
    batch_size: int = 8192
    lr: float = 5e-4
    pref_reg: float = 1e-3
    out_dir: str = ""
    baseline_result_dir: str = ""

    @property
    def id_dim(self) -> int:
        return self.embedding_dim


def _defaults() -> dict:
    base = v3.default_out_dir("dunnhumby")
    return {
        "out_dir": f"{base}_m2_clv_item_interaction_historical_screen_v1",
        "baseline_result_dir": f"{base}_m2_repeatshare_historical_backtest_v1",
    }


def configure_clv_item_interaction_run(**overrides) -> CLVItemInteractionConfig:
    return validate_config(CLVItemInteractionConfig(**(_defaults() | overrides)))


def validate_config(cfg: CLVItemInteractionConfig) -> CLVItemInteractionConfig:
    fixed = {
        "dataset": "dunnhumby",
        "seed": 42,
        "shuffle_seed": 20260826,
        "time_cutoff": 690,
        "evaluation_days": 7,
        "epochs": 100,
        "embedding_dim": 64,
        "n_layers": 2,
        "batch_size": 8192,
        "lr": 5e-4,
        "pref_reg": 1e-3,
    }
    for key, expected in fixed.items():
        if getattr(cfg, key) != expected:
            raise ValueError(f"빠른 CLV-item screen은 {key}={expected!r}이어야 합니다")
    if not cfg.out_dir or not cfg.baseline_result_dir:
        raise ValueError("out_dir와 baseline_result_dir가 필요합니다")
    return cfg


def preflight_summary(cfg: CLVItemInteractionConfig) -> dict:
    cfg = validate_config(cfg)
    return {
        "code_version": CODE_VERSION,
        "dataset": cfg.dataset,
        "seed": cfg.seed,
        "trained_models": list(MODEL_IDS),
        "reused_comparator": "m1_64",
        "historical_development_split": {
            "train_end_inclusive": 683,
            "evaluation_start_inclusive": 684,
            "evaluation_end_inclusive": 690,
            "final_test_constructed": False,
            "holdout_constructed": False,
        },
        "m2": {
            "historical_clv_proxy": "N_hat * V_hat",
            "score_formula": "LightGCN(u,i) + (q_CLV(u)-mean(q_CLV))*a_i",
            "item_specific_parameter": "one scalar a_i per item, zero initialised",
            "actual_assignment": "train-only CLV midrank percentile",
            "shuffle_control": (
                "same centered CLV coordinates permuted within N_hat midrank deciles"
            ),
            "n_v_routing": False,
            "item_price_input": False,
            "external_reranking": False,
        },
        "fixed": {
            "graph": "unchanged binary M1 graph",
            "negative_sampling": "uniform",
            "sample_weighting": False,
            "loss": "one plain pairwise BPR; no added loss",
            "one_training_loop_and_optimizer": True,
            "min_user_interactions": 1,
            "min_item_interactions": 1,
            "epochs": cfg.epochs,
            "validation_or_epoch_selection": False,
        },
        "primary_reading": (
            "six-metric geometric balance index vs M1 and correct CLV vs shuffle"
        ),
        "interpretation": (
            "exploratory post-hoc historical screen; no significance or generalization claim"
        ),
        "out_dir": cfg.out_dir,
    }


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def _base_config(cfg: CLVItemInteractionConfig) -> dict:
    return previous_screen._base_config(cfg)


def _midrank_bins(values: np.ndarray, n_bins: int) -> np.ndarray:
    q = (rankdata(values, method="average") - 0.5) / len(values)
    return np.minimum(np.floor(q * n_bins).astype(np.int8), n_bins - 1)


def _build_clv_coordinates(
    train: pd.DataFrame,
    *,
    n_users: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    basket_column = "b_raw" if "b_raw" in train else "t"
    baskets = (
        train.groupby(["u_idx", basket_column], sort=False)["v"]
        .sum()
        .rename("basket_value")
        .reset_index()
    )
    grouped = baskets.groupby("u_idx", sort=False).basket_value
    n_hat = np.zeros(n_users, dtype=np.float64)
    v_hat = np.zeros(n_users, dtype=np.float64)
    counts = grouped.size()
    means = grouped.mean()
    users = counts.index.to_numpy(int)
    n_hat[users] = counts.to_numpy(float)
    v_hat[users] = means.reindex(counts.index).to_numpy(float)
    if np.any(n_hat <= 0):
        raise ValueError("all indexed users need a construction basket")
    clv_proxy = n_hat * v_hat
    q_clv = (rankdata(clv_proxy, method="average") - 0.5) / n_users
    coordinate = (q_clv - q_clv.mean()).astype(np.float32)
    n_decile = _midrank_bins(n_hat, 10)
    shuffled = coordinate.copy()
    rng = np.random.default_rng(seed)
    for decile in np.unique(n_decile):
        indices = np.flatnonzero(n_decile == decile)
        shuffled[indices] = coordinate[rng.permutation(indices)]
    diagnostics = {
        "n_hat": n_hat,
        "v_hat": v_hat,
        "clv_proxy": clv_proxy,
        "q_clv": q_clv,
        "n_decile": n_decile,
        "shuffle_seed": seed,
        "coordinate_mean": float(coordinate.mean()),
        "coordinate_std": float(coordinate.std()),
    }
    return coordinate, shuffled, diagnostics


def _config_hash(cfg, input_hash, revision) -> str:
    return hashlib.sha256(
        _canonical(
            {
                "code_version": CODE_VERSION,
                "config": asdict(cfg),
                "models": MODEL_IDS,
                "input_hash": input_hash,
                "source_revision": revision,
            }
        ).encode()
    ).hexdigest()[:12]


def _prepare(cfg: CLVItemInteractionConfig) -> dict:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = moe.build_input_manifest(v3.SCHEMA[cfg.dataset])
    input_hash = moe.manifest_hash(manifest)
    revision = moe.source_revision()
    base_cfg = _base_config(cfg)
    data = v3.prepare_data(base_cfg, v3.DCFG)
    if set(data["splits"]) != {"test"} or float(data["train"].t.max()) != 683.0:
        raise RuntimeError("historical DAY 1--683 / 684--690 split contamination")
    if data.get("loss_w") is not None:
        raise RuntimeError("M4 sample weighting contaminated this M2 screen")
    data["loss_w"] = None
    actual, shuffled, coordinate_diagnostics = _build_clv_coordinates(
        data["train"], n_users=data["n_users"], seed=cfg.shuffle_seed
    )
    baseline = previous_screen._load_compatible_baseline(cfg, manifest)
    meta = v3.item_meta(data["train"], data["n_items"])
    thresholds = v3.segment_thresholds(
        coordinate_diagnostics["clv_proxy"], base_cfg["SEG_EDGES"]
    )
    cache = v3.EvalCache(
        *data["splits"]["test"],
        coordinate_diagnostics["clv_proxy"],
        thresholds,
        data["n_items"],
    )
    prepared = {
        "out_dir": out_dir,
        "manifest": manifest,
        "input_hash": input_hash,
        "revision": revision,
        "base_cfg": base_cfg,
        "data": data,
        "baseline": baseline,
        "meta": meta,
        "cache": cache,
        "coordinates": {
            MODEL_IDS[0]: actual,
            MODEL_IDS[1]: shuffled,
        },
        "coordinate_diagnostics": coordinate_diagnostics,
    }
    prepared["config_hash"] = _config_hash(cfg, input_hash, revision)
    return prepared


def _build_model(prepared, cfg, model_id):
    data = prepared["data"]
    v3.set_seed(cfg.seed)
    model = CLVItemInteractionLightGCN(
        n_users=data["n_users"],
        n_items=data["n_items"],
        clv_coordinate=prepared["coordinates"][model_id],
        adj=data["adj"],
        embedding_dim=cfg.embedding_dim,
        n_layers=cfg.n_layers,
        pref_reg=cfg.pref_reg,
    ).to(v3.DEVICE)
    return model, list(model.parameters())


def _arm_hash(prepared, cfg, model_id):
    return hashlib.sha256(
        _canonical(
            {
                "run": prepared["config_hash"],
                "model_id": model_id,
                "seed": cfg.seed,
            }
        ).encode()
    ).hexdigest()[:12]


def _run_arm(prepared, cfg, model_id):
    root = prepared["out_dir"] / "arms" / prepared["config_hash"]
    result_path = root / f"{model_id}_s{cfg.seed}.json"
    checkpoint_path = root / f"{model_id}_s{cfg.seed}.pt"
    if result_path.exists():
        print(f"  [cached] {model_id} 완료 결과 재사용")
        return json.loads(result_path.read_text(encoding="utf-8"))
    model, parameters = _build_model(prepared, cfg, model_id)
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
        model, parameters, prepared, cfg, model_id, cfg.seed, store
    )
    model.eval()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint_path.with_suffix(".pt.tmp")
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
    os.replace(temporary, checkpoint_path)
    metrics, _ = moe._flat_evaluation(
        model,
        0.0,
        prepared["cache"],
        prepared["meta"],
        prepared["data"],
        prepared["base_cfg"],
        per_user=False,
    )
    payload = {
        "model_id": model_id,
        "role": "model" if model_id == MODEL_IDS[0] else "control",
        "seed": cfg.seed,
        "split": HISTORICAL_SPLIT,
        "final_epoch": cfg.epochs,
        "validation_selection": False,
        "evaluation_count": 1,
        "metrics": test10._public_metrics(metrics),
        "interaction_diagnostics": model.interaction_diagnostics(),
        "training": training,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
    }
    test10._atomic_json(result_path, payload)
    store.mark_complete(
        epoch=cfg.epochs,
        max_epoch=cfg.epochs,
        selection="none",
        split=HISTORICAL_SPLIT,
        evaluation_count=1,
        checkpoint_path=str(checkpoint_path),
        result_path=str(result_path),
    )
    return payload


def _baseline_row(prepared):
    baseline = prepared["baseline"]
    metadata = {
        "model_id",
        "role",
        "seed",
        "split",
        "final_epoch",
        "source_result",
    }
    return {
        "model_id": "m1_baseline",
        "role": "baseline",
        "seed": 42,
        "split": HISTORICAL_SPLIT,
        "final_epoch": 100,
        **{key: value for key, value in baseline.items() if key not in metadata},
    }


def _decision(frame):
    rows = frame.set_index("model_id")
    m1 = rows.loc["m1_baseline"].to_dict()
    actual = rows.loc[MODEL_IDS[0]].to_dict()
    shuffled = rows.loc[MODEL_IDS[1]].to_dict()
    actual_vs_m1 = _balanced_accuracy_index(actual, m1)
    shuffle_vs_m1 = _balanced_accuracy_index(shuffled, m1)
    actual_vs_shuffle = _balanced_accuracy_index(actual, shuffled)
    return {
        "meaningful_balanced_improvement": bool(
            actual_vs_m1 > 1.0 and actual_vs_shuffle > 1.0
        ),
        "balanced_index_clv_vs_m1": actual_vs_m1,
        "balanced_index_shuffle_vs_m1": shuffle_vs_m1,
        "balanced_index_clv_vs_shuffle": actual_vs_shuffle,
        "metric_ratios_clv_vs_m1": {
            metric: float(actual[metric] / m1[metric])
            for metric in ACCURACY_METRICS
        },
        "single_seed_limitation": (
            "exploratory post-hoc historical screen; no interval, significance, "
            "or generalization claim"
        ),
    }


def run_clv_item_interaction(
    cfg: CLVItemInteractionConfig | None = None,
) -> pd.DataFrame:
    cfg = validate_config(cfg or configure_clv_item_interaction_run())
    print(json.dumps(preflight_summary(cfg), ensure_ascii=False, indent=2))
    prepared = _prepare(cfg)
    arms = []
    for model_id in MODEL_IDS:
        print(f"\n===== {model_id} | seed 42 | fixed 100 epochs =====")
        arms.append(_run_arm(prepared, cfg, model_id))
    frame = pd.DataFrame(
        [
            _baseline_row(prepared),
            *[
                {
                    "model_id": arm["model_id"],
                    "role": arm["role"],
                    "seed": arm["seed"],
                    "split": arm["split"],
                    "final_epoch": arm["final_epoch"],
                    **arm["metrics"],
                }
                for arm in arms
            ],
        ]
    )
    decision = _decision(frame)
    stem = f"m2_clv_item_interaction_{prepared['config_hash']}"
    paths = {
        "absolute_csv": prepared["out_dir"] / f"{stem}.csv",
        "json": prepared["out_dir"] / f"{stem}.json",
    }
    test10._atomic_csv(paths["absolute_csv"], frame)
    test10._atomic_json(
        paths["json"],
        {
            "code_version": CODE_VERSION,
            "source_revision": prepared["revision"],
            "config": asdict(cfg),
            "preflight": preflight_summary(cfg),
            "input_manifest": prepared["manifest"],
            "data_stats": prepared["data"].get("data_stats", {}),
            "coordinate_diagnostics": {
                key: value
                for key, value in prepared["coordinate_diagnostics"].items()
                if key not in {"n_hat", "v_hat", "clv_proxy", "q_clv", "n_decile"}
            },
            "arms": arms,
            "absolute_rows": frame.to_dict("records"),
            "decision": decision,
            "result_paths": {key: str(value) for key, value in paths.items()},
        },
    )
    frame.attrs["decision"] = decision
    frame.attrs["result_paths"] = {key: str(value) for key, value in paths.items()}
    return frame


if __name__ == "__main__":
    result = run_clv_item_interaction()
    print(result.to_string(index=False))
    print(json.dumps(result.attrs["decision"], ensure_ascii=False, indent=2))
