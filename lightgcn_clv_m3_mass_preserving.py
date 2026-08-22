"""Test-only runner for direct CLV influence in LightGCN propagation.

Former train and validation intervals are merged. Every arm trains for a fixed
100 epochs without validation or early stopping, then evaluates the fixed
DAY 698--704 test interval exactly once. Later rows are ignored and never
evaluated. The default is the requested seed-42 pilot; ``FULL_SEEDS`` can be
supplied later unchanged.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
import torch

from clv_m3_mass_preserving_graph import (
    DEFAULT_SHUFFLE_SEED,
    build_directional_torch_adj,
    build_mass_preserving_clv_graph,
)
from clv_run_state import ProgressStore, RunIdentity, clone_state, file_sha256
import lightgcn_clv_axis_specific_test10 as fixed_train
import lightgcn_clv_moe as moe
import lightgcn_clv_v3 as v3


CODE_VERSION = "m3-direct-clv-item-message-test-only-v2"
PILOT_SEEDS = (42,)
FULL_SEEDS = tuple(range(42, 52))
MODEL_IDS = {
    "m1": "m1_baseline",
    "n_only": "m3_n_only_influence",
    "v_only": "m3_v_only_influence",
    "clv": "m3_clv_influence",
    "clv_shuffle": "m3_clv_influence_shuffle",
}
MODEL_ORDER = tuple(MODEL_IDS.values())


@dataclass(frozen=True)
class M3TestConfig:
    dataset: str = "dunnhumby"
    seeds: tuple[int, ...] = PILOT_SEEDS
    epochs: int = 100
    dim: int = 64
    n_layers: int = 2
    batch_size: int = 8192
    lr: float = 5e-4
    pref_reg: float = 1e-3
    test_days: int = 7
    out_dir: str = ""


def _default_out_dir() -> str:
    if v3.IN_COLAB:
        return "/content/drive/MyDrive/논문/data/results_m3_clv_influence_test_dunnhumby"
    return str(
        Path(v3.default_out_dir("dunnhumby")).with_name(
            "results_m3_clv_influence_test_dunnhumby"
        )
    )


def configure_m3_clv_influence_test_run(**overrides) -> M3TestConfig:
    return validate_test_config(
        M3TestConfig(**({"out_dir": _default_out_dir()} | overrides))
    )


def validate_test_config(cfg: M3TestConfig) -> M3TestConfig:
    required = {
        "dataset": "dunnhumby",
        "epochs": 100,
        "dim": 64,
        "n_layers": 2,
        "test_days": 7,
    }
    for key, expected in required.items():
        if getattr(cfg, key) != expected:
            raise ValueError(f"M3 test-only setting requires {key}={expected!r}")
    if not cfg.seeds or len(set(cfg.seeds)) != len(cfg.seeds):
        raise ValueError("seeds must be non-empty and unique")
    if tuple(sorted(cfg.seeds)) != cfg.seeds:
        raise ValueError("seeds must be sorted")
    if not set(cfg.seeds).issubset(FULL_SEEDS):
        raise ValueError(f"seeds must be a subset of {FULL_SEEDS}")
    if cfg.batch_size <= 0 or cfg.lr <= 0 or cfg.pref_reg < 0:
        raise ValueError("invalid training setting")
    if not cfg.out_dir or "dunnhumby" not in cfg.out_dir:
        raise ValueError("out_dir must identify the Dunnhumby M3 test run")
    return cfg


def preflight_summary(cfg: M3TestConfig) -> dict:
    cfg = validate_test_config(cfg)
    return {
        "code_version": CODE_VERSION,
        "dataset": cfg.dataset,
        "seeds": list(cfg.seeds),
        "current_scope": (
            "single-seed protocol check"
            if cfg.seeds == PILOT_SEEDS
            else "multi-seed run"
        ),
        "planned_full_seeds": list(FULL_SEEDS),
        "models": list(MODEL_ORDER),
        "training_data": "former train + validation",
        "test_data": "fixed DAY 698--704 after merged training through DAY 697",
        "validation_constructed": False,
        "validation_selection": False,
        "early_stopping": False,
        "epochs": cfg.epochs,
        "test_evaluation": "one final-checkpoint evaluation per seed/model",
        "post_test_rows": "ignored; no truth, metric, or result is constructed",
        "holdout_evaluation": False,
        "new_item_task": (
            "all user-item pairs in merged train+validation are excluded from test truth"
        ),
        "customer_value": {
            "n_hat": "number of merged-train transactions/baskets",
            "v_hat": "mean merged-train transaction/basket value",
            "clv_proxy": "n_hat * v_hat",
            "factor": "mean-one 0.5 + merged-train user percentile; no alpha/beta",
        },
        "propagation": {
            "user_from_item": "unchanged M1 symmetric-normalized coefficients",
            "item_from_user": "fixed item mass redistributed by the user factor",
            "identity": "factor == 1 gives the exact M1 operator",
        },
        "fixed_boundaries": {
            "edge_set": "same unique merged-train user-item pairs as M1",
            "loss": "plain pairwise BPR; no sample weights or added loss",
            "negative_sampling": "uniform",
            "min_user_interactions": 1,
            "min_item_interactions": 1,
            "m2_embedding": False,
            "m4_loss_weight": False,
        },
        "reporting": (
            "descriptive test results only; with 10 seeds, report means and "
            "same-seed paired differences with 95% t intervals"
        ),
        "test_use_prohibition": (
            "test must not select or modify formula, model, epoch, or hyperparameter"
        ),
        "out_dir": cfg.out_dir,
    }


def _base_config(cfg: M3TestConfig) -> dict:
    base = dict(
        v3.configure_run(
            cfg.dataset,
            out_dir=cfg.out_dir,
            ARCH="pref_only",
            SEED_LIST=list(cfg.seeds),
            WINDOW_DAYS=None,
            VAL_DAYS=7,
            TEST_DAYS=cfg.test_days,
            # Preserve the already-fixed DAY 698--704 test boundary. The final
            # seven days are ignored rather than re-labelled as a new test.
            HOLDOUT_DAYS=7,
            TRAIN_ON_VAL=True,
            EVAL_TEST=True,
            EVAL_HOLDOUT=False,
            GRAPH_MODE="binary",
            GRAPH_ALPHA=1.0,
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
    required = {
        "TRAIN_ON_VAL": True,
        "EVAL_TEST": True,
        "EVAL_HOLDOUT": False,
        "HOLDOUT_DAYS": 7,
        "GRAPH_MODE": "binary",
        "LOSS_MODE": "plain",
        "NEG_MODE": "uniform",
        "MIN_USER_INTER": 1,
        "MIN_ITEM_INTER": 1,
        "EPOCHS": 100,
    }
    for key, expected in required.items():
        if base.get(key) != expected:
            raise RuntimeError(
                f"test-only configuration contamination: {key}={base.get(key)!r}"
            )
    return base


def _method_hash(cfg: M3TestConfig, input_hash: str, revision: str) -> str:
    payload = {
        key: value
        for key, value in asdict(cfg).items()
        if key not in {"seeds", "out_dir"}
    }
    payload.update(
        code_version=CODE_VERSION,
        models=MODEL_ORDER,
        shuffle_seed=DEFAULT_SHUFFLE_SEED,
        input_hash=input_hash,
        source_revision=revision,
    )
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()[:12]


def _prepare(cfg: M3TestConfig) -> dict:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = moe.build_input_manifest(v3.SCHEMA[cfg.dataset])
    input_hash = moe.manifest_hash(manifest)
    revision = moe.source_revision()
    base_cfg = _base_config(cfg)
    data = v3.prepare_data(base_cfg, v3.DCFG)
    if set(data["splits"]) != {"test"}:
        raise RuntimeError(f"test-only split contamination: {sorted(data['splits'])}")
    split_rows = data["data_stats"].get("split_rows", {})
    if split_rows.get("val") != 0:
        raise RuntimeError(f"validation rows must be zero: {split_rows}")
    split_status = data["data_stats"].get("split_evaluation_status", {})
    if split_status.get("holdout") != "not_constructed":
        raise RuntimeError(f"post-test rows must not be evaluated: {split_status}")
    data["loss_w"] = None

    graph = build_mass_preserving_clv_graph(
        data["train"], data["n_users"], data["n_items"]
    )
    expected_users = (data["pos_key"] // data["n_items"]).astype(np.int64)
    expected_items = (data["pos_key"] % data["n_items"]).astype(np.int64)
    if not (
        np.array_equal(graph.edge_users, expected_users)
        and np.array_equal(graph.edge_items, expected_items)
    ):
        raise RuntimeError("M3 graph edge order differs from merged-train M1")
    data["clv"] = graph.clv_proxy
    data["vhat"] = graph.v_hat
    data["data_stats"]["m3_mass_preserving_graph"] = graph.diagnostics
    thresholds = v3.segment_thresholds(graph.clv_proxy, base_cfg["SEG_EDGES"])
    cache = v3.EvalCache(
        *data["splits"]["test"], graph.clv_proxy, thresholds, data["n_items"]
    )
    meta = v3.item_meta(data["train"], data["n_items"])
    x_item, item_cat = v3.item_value_features(data["train"], data["n_items"])
    return {
        "out_dir": out_dir,
        "manifest": manifest,
        "input_hash": input_hash,
        "revision": revision,
        "method_hash": _method_hash(cfg, input_hash, revision),
        "base_cfg": base_cfg,
        "data": data,
        "graph": graph,
        "cache": cache,
        "meta": meta,
        "x_item": x_item,
        "item_cat": item_cat,
    }


def _mode_for_model(model_id: str) -> str | None:
    for mode, candidate in MODEL_IDS.items():
        if candidate == model_id:
            return None if mode == "m1" else mode
    raise KeyError(model_id)


def _build_model(prepared: dict, cfg: M3TestConfig, model_id: str, seed: int):
    data = prepared["data"]
    mode = _mode_for_model(model_id)
    model_data = data
    if mode is not None:
        model_data = {
            **data,
            "adj": build_directional_torch_adj(
                prepared["graph"],
                mode,
                data["n_users"],
                data["n_items"],
                v3.DEVICE,
            ),
        }
    v3.set_seed(seed)
    model = v3.build_model(
        model_data,
        data["x_val_u"],
        prepared["x_item"],
        prepared["item_cat"],
        prepared["base_cfg"],
    )
    return model, list(model.pref_params())


def _arm_hash(prepared: dict, cfg: M3TestConfig, model_id: str, seed: int) -> str:
    payload = {
        "method": prepared["method_hash"],
        "model_id": model_id,
        "seed": seed,
        "epochs": cfg.epochs,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()[:12]


def _arm_paths(prepared: dict, model_id: str, seed: int) -> dict[str, Path]:
    root = prepared["out_dir"] / "arms" / prepared["method_hash"]
    stem = f"{model_id}_s{seed}"
    return {
        "result": root / f"{stem}.json",
        "per_user": root / f"{stem}_per_user.npz",
        "checkpoint": root / f"{stem}.pt",
    }


def _progress_store(
    prepared: dict, cfg: M3TestConfig, model_id: str, seed: int
) -> ProgressStore:
    return ProgressStore(
        prepared["out_dir"] / "progress" / prepared["method_hash"],
        RunIdentity(
            stage="final_train_test",
            model_id=model_id,
            seed=seed,
            config_hash=_arm_hash(prepared, cfg, model_id, seed),
            source_revision=prepared["revision"],
            input_hash=prepared["input_hash"],
        ),
    )


def _load_cached_arm(paths: dict[str, Path]) -> dict | None:
    if not (paths["result"].exists() and paths["per_user"].exists()):
        return None
    payload = json.loads(paths["result"].read_text(encoding="utf-8"))
    arrays = np.load(paths["per_user"])
    payload["per_user"] = {key: arrays[key] for key in arrays.files}
    print(
        f"  [cached] {payload['model_id']} seed {payload['seed']} reused; "
        "no repeated test evaluation"
    )
    return payload


def _run_arm(
    prepared: dict, cfg: M3TestConfig, model_id: str, seed: int
) -> dict:
    paths = _arm_paths(prepared, model_id, seed)
    cached = _load_cached_arm(paths)
    if cached is not None:
        return cached
    model, params = _build_model(prepared, cfg, model_id, seed)
    store = _progress_store(prepared, cfg, model_id, seed)
    training = fixed_train._fixed_epoch_train(
        model, params, prepared, cfg, model_id, seed, store
    )
    model.eval()
    checkpoint_payload = {
        "state": clone_state(model),
        "model_id": model_id,
        "seed": seed,
        "training": training,
        "config": asdict(cfg),
        "source_revision": prepared["revision"],
        "input_hash": prepared["input_hash"],
    }
    paths["checkpoint"].parent.mkdir(parents=True, exist_ok=True)
    temporary = paths["checkpoint"].with_suffix(".pt.tmp")
    torch.save(checkpoint_payload, temporary)
    os.replace(temporary, paths["checkpoint"])

    metrics, per_user = moe._flat_evaluation(
        model,
        0.0,
        prepared["cache"],
        prepared["meta"],
        prepared["data"],
        prepared["base_cfg"],
        per_user=True,
    )
    public_metrics = fixed_train._public_metrics(metrics)
    public_per_user = fixed_train._public_per_user(per_user)
    fixed_train._atomic_npz(paths["per_user"], public_per_user)
    mode = _mode_for_model(model_id)
    payload = {
        "model_id": model_id,
        "role": "baseline" if mode is None else (
            "model" if mode == "clv" else "control"
        ),
        "seed": seed,
        "split": "test",
        "final_epoch": cfg.epochs,
        "validation_selection": False,
        "test_evaluation_count": 1,
        "test_evaluated_at": datetime.now(timezone.utc).isoformat(),
        "metrics": public_metrics,
        "graph_diagnostics": (
            None if mode is None else prepared["graph"].diagnostics["modes"][mode]
        ),
        "training": training,
        "checkpoint": str(paths["checkpoint"]),
        "checkpoint_sha256": file_sha256(paths["checkpoint"]),
        "per_user_path": str(paths["per_user"]),
    }
    fixed_train._atomic_json(paths["result"], payload)
    payload["per_user"] = public_per_user
    store.mark_complete(
        epoch=cfg.epochs,
        max_epoch=cfg.epochs,
        selection="none",
        split="test",
        test_evaluation_count=1,
        checkpoint_path=str(paths["checkpoint"]),
        result_path=str(paths["result"]),
    )
    return payload


def _absolute_rows(arms: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "seed": arm["seed"],
                "model_id": arm["model_id"],
                "role": arm["role"],
                "split": "test",
                "epoch": arm["final_epoch"],
                **arm["metrics"],
            }
            for arm in arms
        ]
    ).sort_values(["seed", "model_id"]).reset_index(drop=True)


def _mean_ci(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    mean = float(values.mean())
    if n < 2:
        return {
            "n_seeds": n,
            "mean": mean,
            "sd": np.nan,
            "lo": np.nan,
            "hi": np.nan,
        }
    sd = float(values.std(ddof=1))
    half = float(student_t.ppf(0.975, n - 1)) * sd / math.sqrt(n)
    return {
        "n_seeds": n,
        "mean": mean,
        "sd": sd,
        "lo": mean - half,
        "hi": mean + half,
    }


def _summary_tables(
    absolute: pd.DataFrame, arms: list[dict], seeds: tuple[int, ...]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_columns = [
        column
        for column in absolute.columns
        if "@" in column
        or column == "user_value_tendency_recommended_price_alignment"
    ]
    absolute_summary = []
    for model_id, group in absolute.groupby("model_id", sort=False):
        for metric in metric_columns:
            absolute_summary.append(
                {"model_id": model_id, "metric": metric, **_mean_ci(group[metric])}
            )
    arm_map = {(arm["seed"], arm["model_id"]): arm for arm in arms}
    paired_rows = []
    for seed in seeds:
        baseline = arm_map[(seed, MODEL_IDS["m1"])]
        for model_id in MODEL_ORDER[1:]:
            compared = arm_map[(seed, model_id)]
            for metric in metric_columns:
                paired_rows.append(
                    {
                        "seed": seed,
                        "model_id": model_id,
                        "reference": MODEL_IDS["m1"],
                        "metric": metric,
                        "delta": float(
                            compared["metrics"][metric]
                            - baseline["metrics"][metric]
                        ),
                    }
                )
    paired_seed = pd.DataFrame(paired_rows)
    paired_summary = []
    for (model_id, metric), group in paired_seed.groupby(
        ["model_id", "metric"], sort=False
    ):
        paired_summary.append(
            {
                "model_id": model_id,
                "reference": MODEL_IDS["m1"],
                "metric": metric,
                **_mean_ci(group["delta"].to_numpy()),
                "positive_seed_count": int((group["delta"] > 0).sum()),
            }
        )
    return pd.DataFrame(absolute_summary), paired_seed, pd.DataFrame(paired_summary)


def _persist(
    prepared: dict, cfg: M3TestConfig, arms: list[dict]
) -> pd.DataFrame:
    absolute = _absolute_rows(arms)
    absolute_summary, paired_seed, paired_summary = _summary_tables(
        absolute, arms, cfg.seeds
    )
    run_hash = hashlib.sha256(
        json.dumps(
            {"method": prepared["method_hash"], "seeds": cfg.seeds},
            sort_keys=True,
        ).encode()
    ).hexdigest()[:12]
    stem = f"m3_clv_influence_test_{run_hash}"
    paths = {
        "absolute_csv": prepared["out_dir"] / f"{stem}.csv",
        "absolute_summary_csv": prepared["out_dir"] / f"{stem}_mean.csv",
        "paired_seed_csv": prepared["out_dir"] / f"{stem}_paired_seed.csv",
        "paired_summary_csv": prepared["out_dir"] / f"{stem}_paired_mean.csv",
        "json": prepared["out_dir"] / f"{stem}.json",
    }
    fixed_train._atomic_csv(paths["absolute_csv"], absolute)
    fixed_train._atomic_csv(paths["absolute_summary_csv"], absolute_summary)
    fixed_train._atomic_csv(paths["paired_seed_csv"], paired_seed)
    fixed_train._atomic_csv(paths["paired_summary_csv"], paired_summary)
    payload = {
        "code_version": CODE_VERSION,
        "source_revision": prepared["revision"],
        "input_manifest": prepared["manifest"],
        "config": asdict(cfg),
        "preflight": preflight_summary(cfg),
        "data_stats": prepared["data"].get("data_stats", {}),
        "graph_diagnostics": prepared["graph"].diagnostics,
        "absolute_rows": absolute.to_dict("records"),
        "absolute_seed_summary": absolute_summary.to_dict("records"),
        "same_seed_differences": paired_seed.to_dict("records"),
        "same_seed_summary": paired_summary.to_dict("records"),
        "result_paths": {name: str(path) for name, path in paths.items()},
        "interpretation": {
            "selection": "none; validation is absent and test is not used for selection",
            "single_seed": (
                "one seed is a protocol check only; dispersion and significance are not estimable"
                if len(cfg.seeds) == 1
                else None
            ),
            "weighted_hit": (
                "price/purchase-amount weighted recommendation hit; not actual incremental revenue"
            ),
        },
    }
    fixed_train._atomic_json(paths["json"], payload)
    absolute.attrs["absolute_summary"] = absolute_summary
    absolute.attrs["paired_seed"] = paired_seed
    absolute.attrs["paired_summary"] = paired_summary
    absolute.attrs["result_paths"] = {
        name: str(path) for name, path in paths.items()
    }
    return absolute


def run_test(cfg: M3TestConfig | None = None) -> pd.DataFrame:
    cfg = validate_test_config(cfg or configure_m3_clv_influence_test_run())
    print(json.dumps(preflight_summary(cfg), ensure_ascii=False, indent=2))
    prepared = _prepare(cfg)
    arms = []
    for seed in cfg.seeds:
        for model_id in MODEL_ORDER:
            print(f"\n===== seed {seed} | {model_id} | fixed 100-epoch train =====")
            arms.append(_run_arm(prepared, cfg, model_id, seed))
    frame = _persist(prepared, cfg, arms)
    print("\nTest absolute metrics:")
    print(frame.to_string(index=False))
    print("\nSame-seed differences from M1:")
    print(frame.attrs["paired_summary"].to_string(index=False))
    if len(cfg.seeds) == 1:
        print("\nOne seed only: no variance, confidence interval, or significance claim.")
    print("Result files:", frame.attrs["result_paths"])
    return frame


if __name__ == "__main__":
    print(
        json.dumps(
            preflight_summary(configure_m3_clv_influence_test_run()),
            ensure_ascii=False,
            indent=2,
        )
    )
