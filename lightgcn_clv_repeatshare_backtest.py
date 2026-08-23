"""Historical development backtest for popularity-controlled RepeatShare.

This is an exploratory M2 diagnosis, not a second look at the already-inspected
test split.  Dunnhumby transactions through day 683 form the training period
and days 684--690 form a previously unused historical development period.

Only one input changes between the two M2 arms:
  raw RepeatShare_i
  RepeatShare_i residual after controlling log(1 + train unique buyers_i)

Graph, negative sampling, loss, dimensions, gates, optimizer, epoch budget, and
all other user/item features are identical.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import torch

from clv_dual_axis_model import build_dual_item_profiles
from clv_joint_nv_model import JointNVLightGCN
from clv_run_state import ProgressStore, RunIdentity, clone_state, file_sha256
import lightgcn_clv_axis_specific_test10 as test10
import lightgcn_clv_joint_nv as joint
import lightgcn_clv_moe as moe
import lightgcn_clv_residual as residual
import lightgcn_clv_v3 as v3


CODE_VERSION = "m2-popularity-controlled-repeatshare-backtest-v1"
MODELS = (
    "m1_64",
    "m2_raw_repeatshare",
    "m2_popularity_controlled_repeatshare",
)


@dataclass(frozen=True)
class RepeatShareBacktestConfig:
    dataset: str = "dunnhumby"
    seed: int = 42
    time_cutoff: int = 690
    evaluation_days: int = 7
    epochs: int = 100
    id_dim: int = 64
    axis_dim: int = 16
    hidden_dim: int = 32
    n_layers: int = 2
    batch_size: int = 8192
    lr: float = 5e-4
    pref_reg: float = 1e-3
    gamma_init: float = 0.1
    gate_shape: str = "axis_positive"
    input_days: int = 365
    out_dir: str = ""


def configure_repeatshare_backtest(**overrides) -> RepeatShareBacktestConfig:
    defaults = {
        "out_dir": (
            f"{v3.default_out_dir('dunnhumby')}"
            "_m2_repeatshare_historical_backtest_v1"
        )
    }
    return validate_repeatshare_backtest_config(
        RepeatShareBacktestConfig(**(defaults | overrides))
    )


def validate_repeatshare_backtest_config(
    cfg: RepeatShareBacktestConfig,
) -> RepeatShareBacktestConfig:
    required = {
        "dataset": "dunnhumby",
        "seed": 42,
        "time_cutoff": 690,
        "evaluation_days": 7,
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
            raise ValueError(
                f"역사적 RepeatShare 백테스트는 {key}={expected!r}이어야 합니다"
            )
    if cfg.batch_size <= 0 or cfg.lr <= 0 or cfg.pref_reg < 0:
        raise ValueError("학습 설정이 잘못됐습니다")
    if not cfg.out_dir:
        raise ValueError("out_dir가 필요합니다")
    return cfg


def preflight_summary(cfg: RepeatShareBacktestConfig) -> dict:
    cfg = validate_repeatshare_backtest_config(cfg)
    return {
        "code_version": CODE_VERSION,
        "dataset": cfg.dataset,
        "seed": cfg.seed,
        "models": list(MODELS),
        "historical_development_split": {
            "train_end_inclusive": cfg.time_cutoff - cfg.evaluation_days,
            "evaluation_start_inclusive": (
                cfg.time_cutoff - cfg.evaluation_days + 1
            ),
            "evaluation_end_inclusive": cfg.time_cutoff,
            "original_validation_test_holdout_constructed": False,
        },
        "single_changed_input": (
            "item RepeatShare residual after OLS control for "
            "log(1 + train unique buyers)"
        ),
        "fixed": {
            "graph": "binary",
            "negative_sampling": "uniform",
            "sample_weighting": False,
            "loss": "plain pairwise BPR; no added loss",
            "min_user_interactions": 1,
            "min_item_interactions": 1,
            "epochs": cfg.epochs,
            "validation_or_epoch_selection": False,
        },
        "interpretation": (
            "exploratory historical backtest; it does not replace an untouched "
            "confirmatory evaluation"
        ),
        "out_dir": cfg.out_dir,
    }


def _base_config(cfg: RepeatShareBacktestConfig) -> dict:
    configured = v3.configure_run(
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
        "TIME_CUTOFF": 690,
        "TRAIN_ON_VAL": True,
        "EVAL_TEST": True,
        "EVAL_HOLDOUT": False,
        "HOLDOUT_DAYS": 0,
        "GRAPH_MODE": "binary",
        "LOSS_MODE": "plain",
        "NEG_MODE": "uniform",
        "MIN_USER_INTER": 1,
        "MIN_ITEM_INTER": 1,
        "EPOCHS": 100,
    }
    for key, expected in required.items():
        if base[key] != expected:
            raise RuntimeError(f"역사적 백테스트 설정 오염: {key}={base[key]!r}")
    return base


def _config_hash(
    cfg: RepeatShareBacktestConfig, input_hash: str, revision: str
) -> str:
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


def _prepare(cfg: RepeatShareBacktestConfig) -> dict:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = moe.build_input_manifest(v3.SCHEMA[cfg.dataset])
    input_hash = moe.manifest_hash(manifest)
    revision = moe.source_revision()
    base_cfg = _base_config(cfg)
    data = v3.prepare_data(base_cfg, v3.DCFG)
    if set(data["splits"]) != {"test"}:
        raise RuntimeError(
            f"역사적 개발분할 외 평가구간 오염: {sorted(data['splits'])}"
        )
    expected_train_end = cfg.time_cutoff - cfg.evaluation_days
    if float(data["train"].t.max()) != float(expected_train_end):
        raise RuntimeError(
            f"역사적 train 종료일 오류: {data['train'].t.max()}"
        )
    if data.get("loss_w") is not None:
        raise RuntimeError("M2 백테스트에 M4 표본 가중치가 섞였습니다")
    data["loss_w"] = None

    snapshot = residual.build_final_snapshot(
        data["train"], data["n_users"], v3.DCFG["is_date"], cfg.input_days
    )
    axes = joint.build_user_axis_inputs(snapshot, data["n_users"])
    raw_profile = build_dual_item_profiles(
        data["train"],
        data["n_items"],
        v3.DCFG["is_date"],
        repeat_share_mode="raw",
    )
    controlled_profile = build_dual_item_profiles(
        data["train"],
        data["n_items"],
        v3.DCFG["is_date"],
        repeat_share_mode="popularity_controlled",
    )
    # Fail closed: only the first item-activity column may change.
    if not (
        torch.equal(
            torch.from_numpy(raw_profile.activity[:, 1:]),
            torch.from_numpy(controlled_profile.activity[:, 1:]),
        )
        and torch.equal(
            torch.from_numpy(raw_profile.value),
            torch.from_numpy(controlled_profile.value),
        )
        and torch.equal(
            torch.from_numpy(raw_profile.valid_item),
            torch.from_numpy(controlled_profile.valid_item),
        )
    ):
        raise RuntimeError("RepeatShare 외 아이템 입력까지 달라졌습니다")

    print("  RepeatShare 인기도 통제 진단:")
    print(
        json.dumps(
            controlled_profile.repeat_share_diagnostics,
            ensure_ascii=False,
            indent=2,
        )
    )
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
        "profiles": {
            "m2_raw_repeatshare": raw_profile,
            "m2_popularity_controlled_repeatshare": controlled_profile,
        },
        "meta": meta,
        "cache": cache,
        "x_item": x_item,
        "item_cat": item_cat,
    }
    prepared["config_hash"] = _config_hash(cfg, input_hash, revision)
    return prepared


def _build_model(prepared: dict, cfg: RepeatShareBacktestConfig, model_id: str):
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
    model = JointNVLightGCN(
        n_users=data["n_users"],
        n_items=data["n_items"],
        user_activity=prepared["axes"]["activity"],
        user_value=prepared["axes"]["value"],
        user_activity_valid=prepared["axes"]["activity_valid"],
        user_value_valid=prepared["axes"]["value_valid"],
        item_profile=prepared["profiles"][model_id],
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
        gamma_init=cfg.gamma_init,
        anchor_weight=0.0,
        preference_preserving=True,
    ).to(v3.DEVICE)
    return model, list(model.parameters())


def _arm_hash(prepared: dict, model_id: str) -> str:
    return hashlib.sha256(
        json.dumps(
            {
                "run": prepared["config_hash"],
                "model_id": model_id,
                "seed": 42,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:12]


def _run_arm(
    prepared: dict, cfg: RepeatShareBacktestConfig, model_id: str
) -> dict:
    root = prepared["out_dir"] / "arms" / prepared["config_hash"]
    result_path = root / f"{model_id}_s{cfg.seed}.json"
    checkpoint_path = root / f"{model_id}_s{cfg.seed}.pt"
    if result_path.exists():
        print(f"  [cached] {model_id} 완료 결과 재사용")
        return json.loads(result_path.read_text(encoding="utf-8"))

    model, params = _build_model(prepared, cfg, model_id)
    store = ProgressStore(
        prepared["out_dir"] / "progress" / prepared["config_hash"],
        RunIdentity(
            stage="historical_development_train",
            model_id=model_id,
            seed=cfg.seed,
            config_hash=_arm_hash(prepared, model_id),
            source_revision=prepared["revision"],
            input_hash=prepared["input_hash"],
        ),
    )
    training = test10._fixed_epoch_train(
        model, params, prepared, cfg, model_id, cfg.seed, store
    )
    model.eval()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint_path.with_suffix(".pt.tmp")
    torch.save(
        {
            "state": clone_state(model),
            "model_id": model_id,
            "config": asdict(cfg),
            "training": training,
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
        "role": "baseline" if model_id == "m1_64" else "model",
        "seed": cfg.seed,
        "split": "historical_development_days_684_690",
        "final_epoch": cfg.epochs,
        "metrics": test10._public_metrics(metrics),
        "diagnostics": test10._model_diagnostics(model),
        "training": training,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
    }
    test10._atomic_json(result_path, payload)
    store.mark_complete(
        epoch=cfg.epochs,
        max_epoch=cfg.epochs,
        selection="none",
        split="historical_development_days_684_690",
        checkpoint_path=str(checkpoint_path),
        result_path=str(result_path),
    )
    return payload


def _comparison(frame: pd.DataFrame) -> pd.DataFrame:
    metadata = {
        "model_id",
        "role",
        "seed",
        "split",
        "final_epoch",
        "activity_axis_weight",
        "transaction_value_axis_weight",
        "activity_gate_mean",
        "activity_gate_std",
        "transaction_value_gate_mean",
        "transaction_value_gate_std",
    }
    metrics = [column for column in frame.columns if column not in metadata]
    indexed = frame.set_index("model_id")
    rows = []
    pairs = (
        ("m2_raw_repeatshare", "m1_64"),
        ("m2_popularity_controlled_repeatshare", "m1_64"),
        (
            "m2_popularity_controlled_repeatshare",
            "m2_raw_repeatshare",
        ),
    )
    for model_id, reference in pairs:
        for metric in metrics:
            base = indexed.at[reference, metric]
            value = indexed.at[model_id, metric]
            rows.append(
                {
                    "model_id": model_id,
                    "reference": reference,
                    "metric": metric,
                    "reference_value": base,
                    "model_value": value,
                    "absolute_delta": value - base,
                    "relative_change_pct": (
                        100.0 * (value - base) / base if base != 0 else None
                    ),
                }
            )
    return pd.DataFrame(rows)


def run_repeatshare_backtest(
    cfg: RepeatShareBacktestConfig | None = None,
) -> pd.DataFrame:
    cfg = validate_repeatshare_backtest_config(
        cfg or configure_repeatshare_backtest()
    )
    print(json.dumps(preflight_summary(cfg), ensure_ascii=False, indent=2))
    prepared = _prepare(cfg)
    arms = []
    for model_id in MODELS:
        print(f"\n===== {model_id} | historical development fixed 100 epochs =====")
        arms.append(_run_arm(prepared, cfg, model_id))
    rows = []
    for arm in arms:
        rows.append(
            {
                "model_id": arm["model_id"],
                "role": arm["role"],
                "seed": arm["seed"],
                "split": arm["split"],
                "final_epoch": arm["final_epoch"],
                **arm["diagnostics"],
                **arm["metrics"],
            }
        )
    frame = pd.DataFrame(rows)
    comparison = _comparison(frame)
    stem = f"m2_repeatshare_backtest_{prepared['config_hash']}"
    paths = {
        "absolute_csv": prepared["out_dir"] / f"{stem}.csv",
        "comparison_csv": prepared["out_dir"] / f"{stem}_comparison.csv",
        "json": prepared["out_dir"] / f"{stem}.json",
    }
    test10._atomic_csv(paths["absolute_csv"], frame)
    test10._atomic_csv(paths["comparison_csv"], comparison)
    payload = {
        "code_version": CODE_VERSION,
        "source_revision": prepared["revision"],
        "config": asdict(cfg),
        "preflight": preflight_summary(cfg),
        "input_manifest": prepared["manifest"],
        "data_stats": prepared["data"].get("data_stats", {}),
        "repeat_share_diagnostics": prepared["profiles"][
            "m2_popularity_controlled_repeatshare"
        ].repeat_share_diagnostics,
        "absolute_rows": frame.to_dict("records"),
        "comparison_rows": comparison.to_dict("records"),
        "result_paths": {key: str(value) for key, value in paths.items()},
        "reading_rule": {
            "supports_popularity_shortcut_hypothesis_if": (
                "the controlled arm improves over the raw arm while reducing "
                "recommendation concentration and recovering ranking metrics"
            ),
            "otherwise": (
                "reject popularity-confounded RepeatShare as the main cause"
            ),
            "status": "exploratory historical backtest; no significance claim",
        },
    }
    test10._atomic_json(paths["json"], payload)
    frame.attrs["comparison"] = comparison
    frame.attrs["result_paths"] = {
        key: str(value) for key, value in paths.items()
    }
    print("\n역사적 개발분할 절대지표:")
    print(frame.to_string(index=False))
    print("\n비교표:")
    print(comparison.to_string(index=False))
    print("결과 파일:", frame.attrs["result_paths"])
    return frame


if __name__ == "__main__":
    print(
        json.dumps(
            preflight_summary(configure_repeatshare_backtest()),
            ensure_ascii=False,
            indent=2,
        )
    )
