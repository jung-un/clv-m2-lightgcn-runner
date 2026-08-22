"""Fast exploratory Dunnhumby M2 test run with no validation or holdout.

The former validation interval is merged into training. M1 and the proposed
M2 are each trained for exactly 100 epochs with seed 42, then the test split
is evaluated once at the final checkpoint. The test split has already been
exposed in earlier research, so this runner labels its evidence exploratory.
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

from clv_dual_axis_model import build_dual_item_profiles
from clv_postprop_gate_model import PostPropagationGatedJointNVLightGCN
from clv_run_state import ProgressStore, RunIdentity, clone_state, file_sha256
import lightgcn_clv_axis_specific_test10 as final10
import lightgcn_clv_joint_nv as joint
import lightgcn_clv_moe as moe
import lightgcn_clv_residual as residual
import lightgcn_clv_v3 as v3


CODE_VERSION = "m2-postprop-axis-gate-test1-v1"
SEED = 42
MODELS = ("m1_64", "m2_postprop_axis_gate")
FEATURE_SCHEMA = final10.ACCEPTED_M2_FEATURE_SCHEMA


@dataclass(frozen=True)
class Test1Config:
    dataset: str = "dunnhumby"
    seed: int = SEED
    epochs: int = 100
    id_dim: int = 64
    axis_dim: int = 16
    hidden_dim: int = 32
    n_layers: int = 2
    batch_size: int = 8192
    lr: float = 5e-4
    pref_reg: float = 1e-3
    gate_shape: str = "axis_positive"
    input_days: int = 365
    out_dir: str = ""


def configure_test1_run(**overrides) -> Test1Config:
    defaults = {
        "out_dir": (
            f"{v3.default_out_dir('dunnhumby')}"
            "_m2_postprop_axis_gate_test1_v1"
        )
    }
    return validate_test1_config(Test1Config(**(defaults | overrides)))


def validate_test1_config(cfg: Test1Config) -> Test1Config:
    required = {
        "dataset": "dunnhumby",
        "seed": 42,
        "epochs": 100,
        "id_dim": 64,
        "axis_dim": 16,
        "hidden_dim": 32,
        "n_layers": 2,
        "gate_shape": "axis_positive",
        "input_days": 365,
    }
    for key, expected in required.items():
        if getattr(cfg, key) != expected:
            raise ValueError(f"이번 단일시드 설정은 {key}={expected!r}이어야 합니다")
    if cfg.batch_size <= 0 or cfg.lr <= 0 or cfg.pref_reg < 0:
        raise ValueError("학습 설정이 잘못됐습니다")
    if not cfg.out_dir:
        raise ValueError("out_dir가 필요합니다")
    return cfg


def preflight_summary(cfg: Test1Config) -> dict:
    cfg = validate_test1_config(cfg)
    return {
        "code_version": CODE_VERSION,
        "evidence_status": (
            "exploratory test check because this test split was exposed earlier"
        ),
        "dataset": cfg.dataset,
        "seed": cfg.seed,
        "models": list(MODELS),
        "training_data": "former train + validation",
        "validation": "not constructed, not evaluated, not used for selection",
        "test": "evaluated once per model at the fixed epoch-100 checkpoint",
        "holdout": "disabled and not constructed",
        "epochs": cfg.epochs,
        "early_stopping": False,
        "automatic_epoch_resume": True,
        "new_item_task": (
            "all user-item pairs seen in merged train+validation are excluded "
            "from test truth"
        ),
        "m2_feature_schema": FEATURE_SCHEMA,
        "m2_representation": {
            "layer0": "ID|N|V, with no learned global N/V scalar",
            "propagation": "one shared binary LightGCN",
            "after_propagation": (
                "L2-normalize N/V blocks, then apply fixed positive "
                "user-specific N/V gates"
            ),
            "score": "one dot product inside the same BPR training graph",
        },
        "fixed_boundaries": {
            "graph": "binary",
            "negative_sampling": "uniform",
            "sample_weighting": False,
            "min_user_interactions": 1,
            "min_item_interactions": 1,
        },
        "out_dir": cfg.out_dir,
    }


def _base_config(cfg: Test1Config) -> dict:
    configured = v3.configure_run(
        cfg.dataset,
        out_dir=cfg.out_dir,
        ARCH="pref_only",
        SEED_LIST=[cfg.seed],
        WINDOW_DAYS=None,
        TRAIN_ON_VAL=True,
        EVAL_TEST=True,
        EVAL_HOLDOUT=False,
        GRAPH_MODE="binary",
        LOSS_MODE="plain",
        NEG_MODE="uniform",
        MIN_USER_INTER=1,
        MIN_ITEM_INTER=1,
        DIM=cfg.id_dim,
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
            raise RuntimeError(f"test 설정 오염: {key}={base[key]!r}")
    return base


def _config_hash(cfg: Test1Config, input_hash: str, revision: str) -> str:
    payload = {
        "code_version": CODE_VERSION,
        "config": asdict(cfg),
        "models": MODELS,
        "input_hash": input_hash,
        "source_revision": revision,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()[:12]


def _prepare(cfg: Test1Config) -> dict:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = moe.build_input_manifest(v3.SCHEMA[cfg.dataset])
    input_hash = moe.manifest_hash(manifest)
    revision = moe.source_revision()
    config_hash = _config_hash(cfg, input_hash, revision)
    base_cfg = _base_config(cfg)
    data = v3.prepare_data(base_cfg, v3.DCFG)
    if set(data["splits"]) != {"test"}:
        raise RuntimeError(
            f"test-only runner에 split 오염: {sorted(data['splits'])}"
        )
    if data.get("loss_w") is not None:
        raise RuntimeError("M2에 M4 표본 가중치가 섞였습니다")
    data["loss_w"] = None

    snapshot = residual.build_final_snapshot(
        data["train"], data["n_users"], v3.DCFG["is_date"], cfg.input_days
    )
    axes = joint.build_user_axis_inputs(snapshot, data["n_users"])
    item_profile = build_dual_item_profiles(
        data["train"], data["n_items"], v3.DCFG["is_date"]
    )
    actual_schema = {
        "user_activity": list(axes["activity_names"]),
        "user_value": list(axes["value_names"]),
        "item_activity": list(item_profile.activity_names),
        "item_value": list(item_profile.value_names),
    }
    if actual_schema != FEATURE_SCHEMA:
        raise RuntimeError(
            "승인된 M2 입력과 실제 입력이 다릅니다: "
            f"expected={FEATURE_SCHEMA}, actual={actual_schema}"
        )
    print("  실제 M2 입력(일반 v3의 legacy 안내문과 별개):")
    for axis, names in actual_schema.items():
        print(f"    {axis}: {names}")

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
    return {
        "out_dir": out_dir,
        "manifest": manifest,
        "input_hash": input_hash,
        "revision": revision,
        "config_hash": config_hash,
        "base_cfg": base_cfg,
        "data": data,
        "axes": axes,
        "item_profile": item_profile,
        "feature_schema": actual_schema,
        "meta": meta,
        "cache": cache,
        "x_item": x_item,
        "item_cat": item_cat,
    }


def _build_model(prepared: dict, cfg: Test1Config, model_id: str):
    data = prepared["data"]
    v3.set_seed(cfg.seed)
    if model_id == "m1_64":
        model_cfg = {**prepared["base_cfg"], "DIM": cfg.id_dim}
        model = v3.build_model(
            data,
            data["x_val_u"],
            prepared["x_item"],
            prepared["item_cat"],
            model_cfg,
        )
        return model, list(model.pref_params())
    if model_id != "m2_postprop_axis_gate":
        raise KeyError(model_id)
    model = PostPropagationGatedJointNVLightGCN(
        n_users=data["n_users"],
        n_items=data["n_items"],
        user_activity=prepared["axes"]["activity"],
        user_value=prepared["axes"]["value"],
        user_activity_valid=prepared["axes"]["activity_valid"],
        user_value_valid=prepared["axes"]["value_valid"],
        item_profile=prepared["item_profile"],
        q_n=prepared["axes"]["q_n"],
        q_v=prepared["axes"]["q_v"],
        adj=data["adj"],
        id_dim=cfg.id_dim,
        axis_dim=cfg.axis_dim,
        hidden_dim=cfg.hidden_dim,
        n_layers=cfg.n_layers,
        variant="joint_nv",
        gate_shape=cfg.gate_shape,
        shuffle_seed=cfg.seed,
        pref_reg=cfg.pref_reg,
        anchor_weight=0.0,
        preference_preserving=True,
    ).to(v3.DEVICE)
    if any("gamma" in name for name, _ in model.named_parameters()):
        raise RuntimeError("학습형 전체 N/V 가중치가 제거되지 않았습니다")
    return model, list(model.parameters())


def _progress_store(prepared: dict, cfg: Test1Config, model_id: str):
    arm_hash = hashlib.sha256(
        json.dumps(
            {
                "run": prepared["config_hash"],
                "model_id": model_id,
                "seed": cfg.seed,
                "epochs": cfg.epochs,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:12]
    return ProgressStore(
        prepared["out_dir"] / "progress" / prepared["config_hash"],
        RunIdentity(
            stage="exploratory_train_test",
            model_id=model_id,
            seed=cfg.seed,
            config_hash=arm_hash,
            source_revision=prepared["revision"],
            input_hash=prepared["input_hash"],
        ),
    )


def _paths(prepared: dict, model_id: str) -> dict[str, Path]:
    root = prepared["out_dir"] / "arms" / prepared["config_hash"]
    stem = f"{model_id}_s{SEED}"
    return {
        "result": root / f"{stem}.json",
        "per_user": root / f"{stem}_per_user.npz",
        "checkpoint": root / f"{stem}.pt",
    }


def _run_arm(prepared: dict, cfg: Test1Config, model_id: str) -> dict:
    paths = _paths(prepared, model_id)
    cached = final10._load_cached_arm(paths)
    if cached is not None:
        return cached
    model, params = _build_model(prepared, cfg, model_id)
    store = _progress_store(prepared, cfg, model_id)
    training = final10._fixed_epoch_train(
        model, params, prepared, cfg, model_id, cfg.seed, store
    )
    model.eval()
    checkpoint_payload = {
        "state": clone_state(model),
        "model_id": model_id,
        "seed": cfg.seed,
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
    public_metrics = final10._public_metrics(metrics)
    public_per_user = final10._public_per_user(per_user)
    final10._atomic_npz(paths["per_user"], public_per_user)
    diagnostics = (
        model.score_diagnostics(seed=cfg.seed)
        if isinstance(model, PostPropagationGatedJointNVLightGCN)
        else {
            "learned_global_axis_weights": None,
            "gate_application": None,
            "axis_normalization": None,
        }
    )
    payload = {
        "model_id": model_id,
        "role": "baseline" if model_id == "m1_64" else "model",
        "seed": cfg.seed,
        "split": "test",
        "evidence_status": "exploratory_test_after_prior_test_exposure",
        "final_epoch": cfg.epochs,
        "validation_selection": False,
        "holdout_evaluation": False,
        "test_evaluation_count": 1,
        "test_evaluated_at": datetime.now(timezone.utc).isoformat(),
        "metrics": public_metrics,
        "diagnostics": diagnostics,
        "training": training,
        "checkpoint": str(paths["checkpoint"]),
        "checkpoint_sha256": file_sha256(paths["checkpoint"]),
        "per_user_path": str(paths["per_user"]),
    }
    final10._atomic_json(paths["result"], payload)
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


def _comparison(absolute: pd.DataFrame) -> pd.DataFrame:
    baseline = absolute.loc[absolute["model_id"].eq("m1_64")].iloc[0]
    model = absolute.loc[
        absolute["model_id"].eq("m2_postprop_axis_gate")
    ].iloc[0]
    metrics = [
        column
        for column in absolute.columns
        if "@" in column
        or column == "user_value_tendency_recommended_price_alignment"
    ]
    rows = []
    for metric in metrics:
        base_value = float(baseline[metric])
        model_value = float(model[metric])
        rows.append(
            {
                "metric": metric,
                "m1_64": base_value,
                "m2_postprop_axis_gate": model_value,
                "absolute_delta": model_value - base_value,
                "relative_change_pct": (
                    100.0 * (model_value - base_value) / base_value
                    if base_value != 0
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def _attach_result_metadata(
    absolute: pd.DataFrame,
    comparison: pd.DataFrame,
    paths: dict[str, str | Path],
) -> None:
    """Attach only scalar containers so wide DataFrames remain display-safe."""
    absolute.attrs["comparison"] = comparison.to_dict("records")
    absolute.attrs["result_paths"] = {
        name: str(path) for name, path in paths.items()
    }


def run_test1(cfg: Test1Config | None = None) -> pd.DataFrame:
    cfg = validate_test1_config(cfg or configure_test1_run())
    print(json.dumps(preflight_summary(cfg), ensure_ascii=False, indent=2))
    prepared = _prepare(cfg)
    arms = []
    for model_id in MODELS:
        print(f"\n===== seed {cfg.seed} | {model_id} | fixed 100 epochs =====")
        arms.append(_run_arm(prepared, cfg, model_id))

    absolute = final10._absolute_rows(arms)
    comparison = _comparison(absolute)
    stem = f"m2_postprop_axis_gate_test1_{prepared['config_hash']}"
    paths = {
        "absolute_csv": prepared["out_dir"] / f"{stem}.csv",
        "comparison_csv": prepared["out_dir"] / f"{stem}_comparison.csv",
        "json": prepared["out_dir"] / f"{stem}.json",
    }
    final10._atomic_csv(paths["absolute_csv"], absolute)
    final10._atomic_csv(paths["comparison_csv"], comparison)
    final10._atomic_json(
        paths["json"],
        {
            "code_version": CODE_VERSION,
            "source_revision": prepared["revision"],
            "evidence_status": "exploratory_test_after_prior_test_exposure",
            "config": asdict(cfg),
            "preflight": preflight_summary(cfg),
            "input_manifest": prepared["manifest"],
            "feature_schema": prepared["feature_schema"],
            "absolute_rows": absolute.to_dict("records"),
            "comparison": comparison.to_dict("records"),
            "result_paths": {name: str(path) for name, path in paths.items()},
            "interpretation": {
                "selection": "none; both models use the fixed epoch-100 checkpoint",
                "significance": "one seed; statistical significance is not claimed",
                "weighted_hit": (
                    "price/purchase-amount weighted hit, not actual incremental revenue"
                ),
            },
        },
    )
    _attach_result_metadata(absolute, comparison, paths)
    print("\n절대지표:")
    print(absolute.to_string(index=False))
    print("\nM1 대비 변화:")
    print(comparison.to_string(index=False))
    print("결과 파일:", absolute.attrs["result_paths"])
    return absolute


if __name__ == "__main__":
    print(
        json.dumps(
            preflight_summary(configure_test1_run()),
            ensure_ascii=False,
            indent=2,
        )
    )
