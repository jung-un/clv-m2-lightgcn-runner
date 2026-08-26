"""Final Dunnhumby 10-seed test runner for the frozen neighbour-conditioned M2.

The former train and validation intervals are merged.  For every paired seed,
M1@64 and the frozen rank-4 neighbour-conditioned M2 are trained for exactly
100 epochs and evaluated once on test.  There is no validation, early stopping,
holdout evaluation, or test-driven checkpoint selection.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from clv_neighbor_conditioned_id_transform_model import (
    CLVNeighborConditionedIDTransformLightGCN,
)
from clv_run_state import ProgressStore, RunIdentity, clone_state, file_sha256
import lightgcn_clv_axis_specific_test10 as shared
import lightgcn_clv_joint_nv as joint
import lightgcn_clv_moe as moe
import lightgcn_clv_residual as residual
import lightgcn_clv_v3 as v3


CODE_VERSION = "m2-neighbor-conditioned-id-transform-test10-v1"
SEEDS = tuple(range(42, 52))
MODELS = ("m1_64", "m2_neighbor_conditioned_id_transform")


@dataclass(frozen=True)
class NeighborConditionedTest10Config:
    dataset: str = "dunnhumby"
    seeds: tuple[int, ...] = SEEDS
    epochs: int = 100
    embedding_dim: int = 64
    transform_rank: int = 4
    rho: float = 0.05
    n_layers: int = 2
    batch_size: int = 8192
    lr: float = 5e-4
    pref_reg: float = 1e-3
    input_days: int = 365
    out_dir: str = ""

    @property
    def id_dim(self) -> int:
        return self.embedding_dim


def configure_neighbor_conditioned_test10_run(
    **overrides,
) -> NeighborConditionedTest10Config:
    defaults = {
        "out_dir": (
            f"{v3.default_out_dir('dunnhumby')}"
            "_m2_neighbor_conditioned_id_transform_test10_v1"
        )
    }
    return validate_config(
        NeighborConditionedTest10Config(**(defaults | overrides))
    )


def validate_config(
    cfg: NeighborConditionedTest10Config,
) -> NeighborConditionedTest10Config:
    """Fail closed if the predeclared final-test protocol is altered."""

    required = {
        "dataset": "dunnhumby",
        "seeds": SEEDS,
        "epochs": 100,
        "embedding_dim": 64,
        "transform_rank": 4,
        "rho": 0.05,
        "n_layers": 2,
        "input_days": 365,
    }
    for key, expected in required.items():
        if getattr(cfg, key) != expected:
            raise ValueError(f"최종 test 설정은 {key}={expected!r}이어야 합니다")
    if cfg.batch_size <= 0 or cfg.lr <= 0 or cfg.pref_reg < 0:
        raise ValueError("학습 설정이 잘못됐습니다")
    if not cfg.out_dir:
        raise ValueError("out_dir가 필요합니다")
    return cfg


def preflight_summary(cfg: NeighborConditionedTest10Config) -> dict:
    cfg = validate_config(cfg)
    return {
        "code_version": CODE_VERSION,
        "dataset": cfg.dataset,
        "seeds": list(cfg.seeds),
        "trained_models": list(MODELS),
        "training_data": "former train + validation",
        "new_item_task": (
            "all user-item pairs in merged train+validation are removed from "
            "test truth"
        ),
        "epochs": cfg.epochs,
        "validation_selection": False,
        "early_stopping": False,
        "test_evaluation": "one final checkpoint per seed/model",
        "holdout_evaluation": False,
        "automatic_epoch_resume": True,
        "m2": {
            "formula": (
                "NormPreserve(E_u + rho[D_N(u)+D_V(u)]), "
                "D_k=centre_valid(q_k A_k Norm(sum_i Ahat_ui E_i))"
            ),
            "embedding_dim": cfg.embedding_dim,
            "transform_rank": cfg.transform_rank,
            "rho": cfg.rho,
            "activity_condition": "train percentile of repeat transaction rate",
            "value_condition": "train percentile of mean transaction value",
            "explicit_item_features": False,
            "learned_global_axis_weight": False,
            "one_model_one_optimizer": True,
        },
        "fixed_boundaries": {
            "graph": "binary",
            "negative_sampling": "uniform",
            "sample_weighting": False,
            "loss": "one plain pairwise BPR plus existing sampled ID L2",
            "new_loss_term": False,
            "min_user_interactions": 1,
            "min_item_interactions": 1,
        },
        "reporting": {
            "per_seed": True,
            "mean_sd_95pct_t_interval": True,
            "same_seed_paired_delta": True,
            "positive_seed_count": True,
            "test_not_used_for_selection": True,
        },
        "out_dir": cfg.out_dir,
    }


def _base_config(cfg: NeighborConditionedTest10Config) -> dict:
    configured = v3.configure_run(
        cfg.dataset,
        out_dir=cfg.out_dir,
        ARCH="pref_only",
        SEED_LIST=list(cfg.seeds),
        WINDOW_DAYS=None,
        TRAIN_ON_VAL=True,
        EVAL_TEST=True,
        EVAL_HOLDOUT=False,
        GRAPH_MODE="binary",
        LOSS_MODE="plain",
        NEG_MODE="uniform",
        MIN_USER_INTER=1,
        MIN_ITEM_INTER=1,
        DIM=cfg.embedding_dim,
        N_LAYERS=cfg.n_layers,
        BATCH_SIZE=cfg.batch_size,
        LR=cfg.lr,
        PREF_REG=cfg.pref_reg,
        EPOCHS=cfg.epochs,
        EARLY_STOP=cfg.epochs,
        REPORT_LEGACY_VALUE_FEATURES=False,
    )
    base = dict(configured)
    required = {
        "TRAIN_ON_VAL": True,
        "EVAL_TEST": True,
        "EVAL_HOLDOUT": False,
        "GRAPH_MODE": "binary",
        "LOSS_MODE": "plain",
        "NEG_MODE": "uniform",
        "MIN_USER_INTER": 1,
        "MIN_ITEM_INTER": 1,
        "EPOCHS": 100,
    }
    for key, expected in required.items():
        if base[key] != expected:
            raise RuntimeError(f"최종 test 설정 오염: {key}={base[key]!r}")
    return base


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def _config_hash(
    cfg: NeighborConditionedTest10Config, input_hash: str, revision: str
) -> str:
    payload = {
        "code_version": CODE_VERSION,
        "config": asdict(cfg),
        "models": MODELS,
        "input_hash": input_hash,
        "source_revision": revision,
    }
    return hashlib.sha256(_canonical(payload).encode()).hexdigest()[:12]


def _prepare(cfg: NeighborConditionedTest10Config) -> dict:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = moe.build_input_manifest(v3.SCHEMA[cfg.dataset])
    input_hash = moe.manifest_hash(manifest)
    revision = moe.source_revision()
    base_cfg = _base_config(cfg)
    data = v3.prepare_data(base_cfg, v3.DCFG)
    if set(data["splits"]) != {"test"}:
        raise RuntimeError(
            f"test-only runner에 보호 split 오염: {sorted(data['splits'])}"
        )
    if data.get("loss_w") is not None:
        raise RuntimeError("M2 test에 M4 표본 가중치가 섞였습니다")
    data["loss_w"] = None

    snapshot = residual.build_final_snapshot(
        data["train"], data["n_users"], v3.DCFG["is_date"], cfg.input_days
    )
    axes = joint.build_user_axis_inputs(snapshot, data["n_users"])
    meta = v3.item_meta(data["train"], data["n_items"])
    thresholds = v3.segment_thresholds(axes["clv_proxy"], base_cfg["SEG_EDGES"])
    cache = v3.EvalCache(
        *data["splits"]["test"],
        axes["clv_proxy"],
        thresholds,
        data["n_items"],
    )
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
        "axes": axes,
        "meta": meta,
        "cache": cache,
        "x_item": x_item,
        "item_cat": item_cat,
        "feature_schema": {
            "activity_condition": "q_n: repeat transaction rate percentile",
            "value_condition": "q_v: mean transaction value percentile",
            "item_explicit_features_in_m2": [],
        },
    }
    prepared["config_hash"] = _config_hash(cfg, input_hash, revision)
    return prepared


def _build_model(
    prepared: dict,
    cfg: NeighborConditionedTest10Config,
    model_id: str,
    seed: int,
):
    data, axes = prepared["data"], prepared["axes"]
    v3.set_seed(seed)
    if model_id == "m1_64":
        model_cfg = {**prepared["base_cfg"], "DIM": cfg.embedding_dim}
        model = v3.build_model(
            data,
            data["x_val_u"],
            prepared["x_item"],
            prepared["item_cat"],
            model_cfg,
        )
        return model, list(model.pref_params())
    if model_id != "m2_neighbor_conditioned_id_transform":
        raise KeyError(model_id)
    model = CLVNeighborConditionedIDTransformLightGCN(
        n_users=data["n_users"],
        n_items=data["n_items"],
        q_n=axes["q_n"],
        q_v=axes["q_v"],
        user_activity_valid=axes["activity_valid"],
        user_value_valid=axes["value_valid"],
        adj=data["adj"],
        embedding_dim=cfg.embedding_dim,
        transform_rank=cfg.transform_rank,
        rho=cfg.rho,
        n_layers=cfg.n_layers,
        pref_reg=cfg.pref_reg,
    ).to(v3.DEVICE)
    return model, list(model.parameters())


def _arm_hash(
    prepared: dict,
    cfg: NeighborConditionedTest10Config,
    model_id: str,
    seed: int,
) -> str:
    payload = {
        "run": prepared["config_hash"],
        "model_id": model_id,
        "seed": seed,
        "epochs": cfg.epochs,
    }
    return hashlib.sha256(_canonical(payload).encode()).hexdigest()[:12]


def _arm_paths(prepared: dict, model_id: str, seed: int) -> dict[str, Path]:
    root = prepared["out_dir"] / "arms" / prepared["config_hash"]
    stem = f"{model_id}_s{seed}"
    return {
        "result": root / f"{stem}.json",
        "per_user": root / f"{stem}_per_user.npz",
        "checkpoint": root / f"{stem}.pt",
    }


def _load_cached_arm(paths: dict[str, Path]) -> dict | None:
    if not (paths["result"].exists() and paths["per_user"].exists()):
        return None
    payload = json.loads(paths["result"].read_text(encoding="utf-8"))
    with np.load(paths["per_user"]) as arrays:
        payload["per_user"] = {key: arrays[key] for key in arrays.files}
    print(
        f"  [cached] {payload['model_id']} s{payload['seed']} 완료 결과 재사용"
        "(test 재평가 없음)"
    )
    return payload


def _progress_store(
    prepared: dict,
    cfg: NeighborConditionedTest10Config,
    model_id: str,
    seed: int,
) -> ProgressStore:
    return ProgressStore(
        prepared["out_dir"] / "progress" / prepared["config_hash"],
        RunIdentity(
            stage="final_train_test",
            model_id=model_id,
            seed=seed,
            config_hash=_arm_hash(prepared, cfg, model_id, seed),
            source_revision=prepared["revision"],
            input_hash=prepared["input_hash"],
        ),
    )


def _diagnostics(model) -> dict:
    method = getattr(model, "representation_diagnostics", None)
    if method is None:
        return {}
    diagnostics = shared._optional_model_diagnostics(
        model, "representation_diagnostics"
    )
    if isinstance(model, CLVNeighborConditionedIDTransformLightGCN):
        activity = float(diagnostics["activity_effective_ratio_to_id"])
        value = float(diagnostics["value_effective_ratio_to_id"])
        change = float(diagnostics["mean_user_representation_change"])
        if max(activity, value) < 1e-6 or change < 1e-8:
            raise RuntimeError(
                "N/V correction path collapsed below the frozen numerical "
                f"liveness floor: N={activity}, V={value}, change={change}"
            )
    return diagnostics


def _run_arm(
    prepared: dict,
    cfg: NeighborConditionedTest10Config,
    model_id: str,
    seed: int,
) -> dict:
    paths = _arm_paths(prepared, model_id, seed)
    cached = _load_cached_arm(paths)
    if cached is not None:
        return cached

    model, params = _build_model(prepared, cfg, model_id, seed)
    store = _progress_store(prepared, cfg, model_id, seed)
    training = shared._fixed_epoch_train(
        model, params, prepared, cfg, model_id, seed, store
    )
    model.eval()
    diagnostics = _diagnostics(model)

    paths["checkpoint"].parent.mkdir(parents=True, exist_ok=True)
    temporary_checkpoint = paths["checkpoint"].with_suffix(".pt.tmp")
    torch.save(
        {
            "state": clone_state(model),
            "model_id": model_id,
            "seed": seed,
            "training": training,
            "config": asdict(cfg),
            "source_revision": prepared["revision"],
            "input_hash": prepared["input_hash"],
        },
        temporary_checkpoint,
    )
    os.replace(temporary_checkpoint, paths["checkpoint"])

    # This is the arm's only protected-test evaluation.  Reconnects use the
    # atomic cached result above instead of opening test a second time.
    metrics, per_user = moe._flat_evaluation(
        model,
        0.0,
        prepared["cache"],
        prepared["meta"],
        prepared["data"],
        prepared["base_cfg"],
        per_user=True,
    )
    public_metrics = shared._public_metrics(metrics)
    public_per_user = shared._public_per_user(per_user)
    shared._atomic_npz(paths["per_user"], public_per_user)
    payload = {
        "model_id": model_id,
        "role": "baseline" if model_id == "m1_64" else "model",
        "seed": seed,
        "split": "test",
        "final_epoch": cfg.epochs,
        "validation_selection": False,
        "test_evaluation_count": 1,
        "test_evaluated_at": datetime.now(timezone.utc).isoformat(),
        "metrics": public_metrics,
        "diagnostics": diagnostics,
        "training": training,
        "checkpoint": str(paths["checkpoint"]),
        "checkpoint_sha256": file_sha256(paths["checkpoint"]),
        "per_user_path": str(paths["per_user"]),
    }
    shared._atomic_json(paths["result"], payload)
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
    rows = [
        {
            "seed": arm["seed"],
            "model_id": arm["model_id"],
            "role": arm["role"],
            "split": "test",
            "epoch": arm["final_epoch"],
            **arm["diagnostics"],
            **arm["metrics"],
        }
        for arm in arms
    ]
    return pd.DataFrame(rows).sort_values(["seed", "model_id"]).reset_index(
        drop=True
    )


def _summary_tables(
    absolute: pd.DataFrame, arms: list[dict], cfg: NeighborConditionedTest10Config
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
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            if len(values) != len(cfg.seeds):
                continue
            absolute_summary.append(
                {
                    "model_id": model_id,
                    "metric": metric,
                    **shared._mean_ci(values.to_numpy()),
                }
            )

    arm_map = {(arm["seed"], arm["model_id"]): arm for arm in arms}
    paired_rows = []
    model_id = "m2_neighbor_conditioned_id_transform"
    for seed in cfg.seeds:
        baseline = arm_map[(seed, "m1_64")]
        compared = arm_map[(seed, model_id)]
        for metric in metric_columns:
            paired_rows.append(
                {
                    "seed": seed,
                    "model_id": model_id,
                    "reference": "m1_64",
                    "metric": metric,
                    "delta": float(
                        compared["metrics"][metric]
                        - baseline["metrics"][metric]
                    ),
                }
            )
    paired_seed = pd.DataFrame(paired_rows)
    paired_summary = []
    for metric, group in paired_seed.groupby("metric", sort=False):
        paired_summary.append(
            {
                "model_id": model_id,
                "reference": "m1_64",
                "metric": metric,
                **shared._mean_ci(group["delta"].to_numpy()),
                "positive_seed_count": int((group["delta"] > 0).sum()),
            }
        )
    return (
        pd.DataFrame(absolute_summary),
        paired_seed,
        pd.DataFrame(paired_summary),
    )


def _descriptive_reading(
    absolute_summary: pd.DataFrame, paired_summary: pd.DataFrame
) -> dict:
    mean_table = absolute_summary.pivot(
        index="metric", columns="model_id", values="mean"
    )
    accuracy = [
        "recall@10",
        "ndcg@10",
        "recall@20",
        "ndcg@20",
        "recall@50",
        "ndcg@50",
    ]
    ratios = {
        metric: float(
            mean_table.at[metric, "m2_neighbor_conditioned_id_transform"]
            / mean_table.at[metric, "m1_64"]
        )
        for metric in accuracy
        if metric in mean_table.index
    }
    paired = paired_summary.set_index("metric")
    economic = "price_purchase_amount_weighted_hit@10"
    return {
        "accuracy_mean_ratios": ratios,
        "all_six_accuracy_means_at_least_99pct_of_m1": bool(
            ratios and min(ratios.values()) >= 0.99
        ),
        "weighted_hit_at_10_mean_delta": (
            float(paired.at[economic, "mean"])
            if economic in paired.index
            else None
        ),
        "weighted_hit_at_10_95pct_interval": (
            [float(paired.at[economic, "lo"]), float(paired.at[economic, "hi"])]
            if economic in paired.index
            else None
        ),
        "test_used_for_selection": False,
        "note": (
            "사전 고정한 test 결과의 기술적 판독이며, 이 결과로 같은 test에 "
            "맞춰 구조·rho·rank·epoch를 다시 선택하지 않습니다."
        ),
    }


def _persist(
    prepared: dict,
    cfg: NeighborConditionedTest10Config,
    arms: list[dict],
) -> pd.DataFrame:
    absolute = _absolute_rows(arms)
    absolute_summary, paired_seed, paired_summary = _summary_tables(
        absolute, arms, cfg
    )
    stem = f"m2_neighbor_conditioned_test10_{prepared['config_hash']}"
    paths = {
        "absolute_csv": prepared["out_dir"] / f"{stem}.csv",
        "absolute_summary_csv": prepared["out_dir"] / f"{stem}_mean.csv",
        "paired_seed_csv": prepared["out_dir"] / f"{stem}_paired_seed.csv",
        "paired_summary_csv": prepared["out_dir"] / f"{stem}_paired_mean.csv",
        "json": prepared["out_dir"] / f"{stem}.json",
    }
    seed_dir = prepared["out_dir"] / "seeds" / prepared["config_hash"]
    seed_paths = {
        int(seed): seed_dir / f"seed_{int(seed)}.csv" for seed in cfg.seeds
    }
    shared._atomic_csv(paths["absolute_csv"], absolute)
    shared._atomic_csv(paths["absolute_summary_csv"], absolute_summary)
    shared._atomic_csv(paths["paired_seed_csv"], paired_seed)
    shared._atomic_csv(paths["paired_summary_csv"], paired_summary)
    for seed, path in seed_paths.items():
        shared._atomic_csv(path, absolute[absolute["seed"].eq(seed)].copy())

    reading = _descriptive_reading(absolute_summary, paired_summary)
    payload = {
        "code_version": CODE_VERSION,
        "source_revision": prepared["revision"],
        "input_manifest": prepared["manifest"],
        "config": asdict(cfg),
        "preflight": preflight_summary(cfg),
        "data_stats": prepared["data"].get("data_stats", {}),
        "feature_schema": prepared["feature_schema"],
        "absolute_rows": absolute.to_dict("records"),
        "absolute_10seed_summary": absolute_summary.to_dict("records"),
        "same_seed_differences": paired_seed.to_dict("records"),
        "same_seed_10seed_summary": paired_summary.to_dict("records"),
        "descriptive_reading": reading,
        "result_paths": {name: str(path) for name, path in paths.items()},
        "per_seed_csv": {
            str(seed): str(path) for seed, path in seed_paths.items()
        },
        "interpretation": {
            "clv": "historical N and V proxy components condition M2 internally",
            "weighted_hit": (
                "price/purchase-amount weighted recommendation hit; not actual "
                "incremental revenue"
            ),
            "selection": "none; test was not used for model or epoch selection",
        },
    }
    shared._atomic_json(paths["json"], payload)
    absolute.attrs["absolute_summary"] = absolute_summary
    absolute.attrs["paired_seed"] = paired_seed
    absolute.attrs["paired_summary"] = paired_summary
    absolute.attrs["descriptive_reading"] = reading
    absolute.attrs["result_paths"] = {
        **{name: str(path) for name, path in paths.items()},
        **{f"seed_{seed}_csv": str(path) for seed, path in seed_paths.items()},
    }
    return absolute


def run_neighbor_conditioned_test10(
    cfg: NeighborConditionedTest10Config | None = None,
) -> pd.DataFrame:
    cfg = validate_config(cfg or configure_neighbor_conditioned_test10_run())
    print(json.dumps(preflight_summary(cfg), ensure_ascii=False, indent=2))
    prepared = _prepare(cfg)
    arms = []
    for seed in cfg.seeds:
        for model_id in MODELS:
            print(f"\n===== seed {seed} | {model_id} | fixed 100 epochs =====")
            arms.append(_run_arm(prepared, cfg, model_id, seed))
    frame = _persist(prepared, cfg, arms)
    print("\n10시드 test 절대지표:")
    print(frame.to_string(index=False))
    print("\n동일 seed M1@64 대비 10시드 평균 차이:")
    print(frame.attrs["paired_summary"].to_string(index=False))
    print("\n판독:", frame.attrs["descriptive_reading"])
    print("결과 파일:", frame.attrs["result_paths"])
    return frame


if __name__ == "__main__":
    print(
        json.dumps(
            preflight_summary(configure_neighbor_conditioned_test10_run()),
            ensure_ascii=False,
            indent=2,
        )
    )
