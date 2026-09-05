"""Test-only M5 runner with a seed-42 pilot and frozen 10-seed extension.

The former training and validation intervals are merged through DAY 697.
Every arm is trained for exactly 100 epochs and evaluated once on the fixed
DAY 698--704 test interval.  DAY 705--711 is ignored and never evaluated.
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

from clv_run_state import ProgressStore, RunIdentity, clone_state, file_sha256
import lightgcn_clv_axis_specific_test10 as result_helpers
import lightgcn_clv_gradient_isolated_economic_interaction as evaluation
import lightgcn_clv_joint_nv as joint
import lightgcn_clv_m5_economic_positive_weight as screen
import lightgcn_clv_moe as moe
import lightgcn_clv_residual as residual
import lightgcn_clv_v3 as v3


CODE_VERSION = "m5-clv-economic-positive-weighting-test-only-v1"
PILOT_SEEDS = (42,)
FULL_SEEDS = tuple(range(42, 52))
MODEL_IDS = screen.MODEL_IDS


@dataclass(frozen=True)
class M5EconomicPositiveTestConfig:
    dataset: str = "dunnhumby"
    seeds: tuple[int, ...] = PILOT_SEEDS
    epochs: int = 100
    id_dim: int = 64
    economic_dim: int = 4
    economic_bins: int = 4
    shrinkage_strength: float = 10.0
    rho: float = 0.15
    positive_weight_lambda: float = 0.5
    n_layers: int = 2
    negative_count: int = 5
    batch_size: int = 8192
    lr: float = 5e-4
    pref_reg: float = 1e-3
    input_days: int = 365
    shuffle_degree_bins: int = 10
    out_dir: str = ""


def configure_m5_economic_positive_test_run(
    **overrides,
) -> M5EconomicPositiveTestConfig:
    defaults = {
        "out_dir": (
            f"{v3.default_out_dir('dunnhumby')}"
            "_m5_economic_positive_weighting_test_seed42_v1"
        )
    }
    return validate_test_config(
        M5EconomicPositiveTestConfig(**(defaults | overrides))
    )


def validate_test_config(
    cfg: M5EconomicPositiveTestConfig,
) -> M5EconomicPositiveTestConfig:
    required = {
        "dataset": "dunnhumby",
        "epochs": 100,
        "id_dim": 64,
        "economic_dim": 4,
        "economic_bins": 4,
        "shrinkage_strength": 10.0,
        "rho": 0.15,
        "positive_weight_lambda": 0.5,
        "n_layers": 2,
        "negative_count": 5,
        "input_days": 365,
        "shuffle_degree_bins": 10,
    }
    for key, expected in required.items():
        if getattr(cfg, key) != expected:
            raise ValueError(f"M5 test-only 설정은 {key}={expected!r}이어야 합니다")
    if cfg.seeds not in {PILOT_SEEDS, FULL_SEEDS}:
        raise ValueError(
            "M5 test-only seeds는 seed-42 파일럿 또는 고정 seed 42~51이어야 합니다"
        )
    if cfg.batch_size <= 0 or cfg.lr <= 0 or cfg.pref_reg < 0:
        raise ValueError("M5 test-only 학습 설정이 잘못됐습니다")
    if not cfg.out_dir:
        raise ValueError("M5 test-only out_dir가 필요합니다")
    return cfg


def preflight_summary(cfg: M5EconomicPositiveTestConfig) -> dict:
    cfg = validate_test_config(cfg)
    return {
        "code_version": CODE_VERSION,
        "dataset": cfg.dataset,
        "seeds": list(cfg.seeds),
        "current_scope": (
            "single-seed protocol run"
            if cfg.seeds == PILOT_SEEDS
            else "frozen ten-seed final run"
        ),
        "planned_full_seeds": list(FULL_SEEDS),
        "models": list(MODEL_IDS),
        "training_data": "DAY 1--697 (former train + validation)",
        "test_data": "DAY 698--704",
        "validation_constructed": False,
        "validation_selection": False,
        "early_stopping": False,
        "post_test_rows": "DAY 705--711 ignored",
        "holdout_evaluation": False,
        "test_evaluation": (
            "one final-checkpoint evaluation per seed/model; completed results are cached"
        ),
        "new_item_task": (
            "every user-item pair in merged training is excluded from test truth"
        ),
        "m2": {
            "architecture": "ID64 plus one jointly propagated 4D economic block",
            "economic_bins": cfg.economic_bins,
            "shrinkage_strength": cfg.shrinkage_strength,
            "rho": cfg.rho,
        },
        "m4_prime": {
            "loss": "positive-row weighted mean of per-negative BPR losses",
            "negative_count": cfg.negative_count,
            "lambda": cfg.positive_weight_lambda,
            "hard_negative": False,
        },
        "fixed": {
            "graph": "binary",
            "negative_sampling": "uniform",
            "min_user_interactions": 1,
            "min_item_interactions": 1,
            "epochs": cfg.epochs,
            "test_use_prohibition": (
                "test cannot select or modify the model, epoch, formula, or hyperparameter"
            ),
        },
        "reporting": (
            "the current seed-42 result is descriptive; the final report uses "
            "means and same-seed differences over seeds 42--51"
        ),
        "out_dir": cfg.out_dir,
    }


def _base_config(cfg: M5EconomicPositiveTestConfig) -> dict:
    base = dict(
        v3.configure_run(
            cfg.dataset,
            out_dir=cfg.out_dir,
            ARCH="pref_only",
            SEED_LIST=list(cfg.seeds),
            WINDOW_DAYS=None,
            TIME_CUTOFF=None,
            VAL_DAYS=7,
            TEST_DAYS=7,
            HOLDOUT_DAYS=7,
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
    )
    required = {
        "TIME_CUTOFF": None,
        "TRAIN_ON_VAL": True,
        "EVAL_TEST": True,
        "EVAL_HOLDOUT": False,
        "VAL_DAYS": 7,
        "TEST_DAYS": 7,
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
                f"M5 test-only 설정 오염: {key}={base.get(key)!r}"
            )
    return base


def _as_float(value) -> float:
    return float(value) if value is not None else math.nan


def validate_final_test_data(data: dict) -> None:
    if set(data.get("splits", {})) != {"test"}:
        raise RuntimeError(
            f"M5 test-only runner에는 test split만 있어야 합니다: "
            f"{sorted(data.get('splits', {}))}"
        )
    stats = data.get("data_stats", {})
    boundaries = stats.get("split_boundaries", {})
    status = stats.get("split_evaluation_status", {})
    train_end = _as_float(boundaries.get("train", {}).get("end_inclusive"))
    test_start = _as_float(boundaries.get("test", {}).get("start_exclusive"))
    test_end = _as_float(boundaries.get("test", {}).get("end_inclusive"))
    post_start = _as_float(boundaries.get("holdout", {}).get("start_exclusive"))
    post_end = _as_float(boundaries.get("holdout", {}).get("end_inclusive"))
    if (train_end, test_start, test_end) != (697.0, 697.0, 704.0):
        raise RuntimeError(
            "고정 test는 DAY 1--697 학습 후 DAY 698--704여야 합니다"
        )
    if (post_start, post_end) != (704.0, 711.0):
        raise RuntimeError("DAY 705--711 보호 구간 경계가 달라졌습니다")
    expected_status = {
        "val": "merged_into_train",
        "test": "constructed",
        "holdout": "not_constructed",
    }
    if status != expected_status:
        raise RuntimeError(f"최종평가 split 상태가 오염됐습니다: {status}")


def _config_hash(
    cfg: M5EconomicPositiveTestConfig, input_hash: str, revision: str
) -> str:
    payload = {
        "code_version": CODE_VERSION,
        "config": asdict(cfg),
        "input_hash": input_hash,
        "source_revision": revision,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()[:12]


def _prepare(cfg: M5EconomicPositiveTestConfig) -> dict:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = moe.build_input_manifest(v3.SCHEMA[cfg.dataset])
    input_hash = moe.manifest_hash(manifest)
    revision = moe.source_revision()
    base_cfg = _base_config(cfg)
    data = v3.prepare_data(base_cfg, v3.DCFG)
    validate_final_test_data(data)
    if data.get("loss_w") is not None:
        raise RuntimeError("M5 자체 구현 외의 표본 가중치가 섞였습니다")
    data["loss_w"] = None

    snapshot = residual.build_final_snapshot(
        data["train"], data["n_users"], v3.DCFG["is_date"], cfg.input_days
    )
    axes = joint.build_user_axis_inputs(snapshot, data["n_users"])
    q_n, q_v, q_c, clv_valid = evaluation.build_clv_inputs(axes)
    economic = screen.build_economic_inputs(
        data["train"],
        n_users=data["n_users"],
        n_items=data["n_items"],
        q_v=q_v,
        q_c=q_c,
        clv_valid=clv_valid,
        n_bins=cfg.economic_bins,
        shrinkage_strength=cfg.shrinkage_strength,
        degree_bins=cfg.shuffle_degree_bins,
    )
    meta = v3.item_meta(data["train"], data["n_items"])
    thresholds = v3.segment_thresholds(axes["clv_proxy"], base_cfg["SEG_EDGES"])
    cache = v3.EvalCache(
        *data["splits"]["test"],
        axes["clv_proxy"],
        thresholds,
        data["n_items"],
    )
    return {
        "out_dir": out_dir,
        "manifest": manifest,
        "input_hash": input_hash,
        "revision": revision,
        "config_hash": _config_hash(cfg, input_hash, revision),
        "base_cfg": base_cfg,
        "data": data,
        "axes": axes,
        "q_n": q_n,
        "q_v": q_v,
        "q_c": q_c,
        "clv_valid": clv_valid,
        "meta": meta,
        "thresholds": thresholds,
        "cache": cache,
        **economic,
    }


def _screen_config(
    cfg: M5EconomicPositiveTestConfig, seed: int
) -> screen.M5EconomicPositiveConfig:
    return screen.M5EconomicPositiveConfig(
        dataset=cfg.dataset,
        seed=seed,
        time_cutoff=711,
        evaluation_days=7,
        epochs=cfg.epochs,
        id_dim=cfg.id_dim,
        economic_dim=cfg.economic_dim,
        economic_bins=cfg.economic_bins,
        shrinkage_strength=cfg.shrinkage_strength,
        rho=cfg.rho,
        positive_weight_lambda=cfg.positive_weight_lambda,
        n_layers=cfg.n_layers,
        negative_count=cfg.negative_count,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        pref_reg=cfg.pref_reg,
        input_days=cfg.input_days,
        diagnostic_max_k=50,
        shuffle_degree_bins=cfg.shuffle_degree_bins,
        shuffle_seed=seed,
        out_dir=cfg.out_dir,
        baseline_result_dir="not-used-in-test-only-runner",
    )


def _prepare_seed_assignments(prepared: dict, seed: int, degree_bins: int) -> None:
    prepared["joint_shuffle"] = screen.joint_degree_matched_shuffle(
        prepared, seed=seed, degree_bins=degree_bins
    )
    prepared["degree_gate"] = {
        "q_v": prepared["q_v"],
        "q_c": prepared["degree_percentile"],
        "clv_valid": prepared["clv_valid"],
        "user_economic_input": prepared["user_economic_input"],
        "user_economic_valid": prepared["user_economic_valid"],
    }


def _run_arm(
    prepared: dict,
    cfg: M5EconomicPositiveTestConfig,
    run_cfg: screen.M5EconomicPositiveConfig,
    spec: dict,
) -> dict:
    paths = screen._arm_paths(prepared, run_cfg, spec["model_id"])
    if paths["result"].exists() and paths["checkpoint"].exists():
        payload = json.loads(paths["result"].read_text(encoding="utf-8"))
        if payload.get("split") != "test" or payload.get("test_evaluation_count") != 1:
            raise RuntimeError("cached arm이 final test 결과가 아닙니다")
        print(
            f"  [cached] {spec['model_id']} s{run_cfg.seed} 완료 결과 재사용"
            "(test 재평가 없음)"
        )
        return payload

    model = screen._build_model(prepared, run_cfg, spec)
    store = ProgressStore(
        prepared["out_dir"] / "progress" / prepared["config_hash"],
        RunIdentity(
            stage="final_train_test",
            model_id=spec["model_id"],
            seed=run_cfg.seed,
            config_hash=screen._arm_hash(prepared, run_cfg, spec),
            source_revision=prepared["revision"],
            input_hash=prepared["input_hash"],
        ),
    )
    training = screen._train_arm(model, prepared, run_cfg, spec, store)
    model.eval()
    paths["checkpoint"].parent.mkdir(parents=True, exist_ok=True)
    temporary = paths["checkpoint"].with_suffix(".pt.tmp")
    torch.save(
        {
            "state": clone_state(model),
            "model_id": spec["model_id"],
            "seed": run_cfg.seed,
            "role": spec["role"],
            "rho": spec["rho"],
            "positive_weighted": spec["weighted"],
            "assignment": spec["assignment_name"],
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
    payload = {
        "model_id": spec["model_id"],
        "role": spec["role"],
        "seed": run_cfg.seed,
        "split": "test",
        "final_epoch": cfg.epochs,
        "validation_selection": False,
        "test_evaluation_count": 1,
        "test_evaluated_at": datetime.now(timezone.utc).isoformat(),
        "rho": spec["rho"],
        "positive_weight_lambda": (
            cfg.positive_weight_lambda if spec["weighted"] else 0.0
        ),
        "negative_count": cfg.negative_count,
        "hard_negative": False,
        "clv_assignment": spec["assignment_name"],
        "metrics": result_helpers._public_metrics(metrics),
        "diagnostics": (
            model.representation_diagnostics()
            | prepared["economic_input_diagnostics"]
        ),
        "training": training,
        "checkpoint": str(paths["checkpoint"]),
        "checkpoint_sha256": file_sha256(paths["checkpoint"]),
    }
    result_helpers._atomic_json(paths["result"], payload)
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


def _mean_summary(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    mean = float(values.mean())
    if n == 1:
        return {"n_seeds": 1, "mean": mean, "sd": np.nan, "lo": np.nan, "hi": np.nan}
    sd = float(values.std(ddof=1))
    half = float(student_t.ppf(0.975, n - 1)) * sd / math.sqrt(n)
    return {
        "n_seeds": n,
        "mean": mean,
        "sd": sd,
        "lo": mean - half,
        "hi": mean + half,
    }


def _persist(
    prepared: dict,
    cfg: M5EconomicPositiveTestConfig,
    arms: list[dict],
) -> pd.DataFrame:
    absolute_rows = []
    metric_columns = sorted(arms[0]["metrics"])
    for arm in arms:
        absolute_rows.append(
            {
                "seed": arm["seed"],
                "model_id": arm["model_id"],
                "role": arm["role"],
                "split": "test",
                "final_epoch": arm["final_epoch"],
                "rho": arm["rho"],
                "positive_weight_lambda": arm["positive_weight_lambda"],
                "clv_assignment": arm["clv_assignment"],
                **arm["diagnostics"],
                **arm["training"].get("final_diagnostics", {}),
                **arm["metrics"],
            }
        )
    absolute = pd.DataFrame(absolute_rows).sort_values(
        ["seed", "model_id"]
    ).reset_index(drop=True)
    arm_map = {(arm["seed"], arm["model_id"]): arm for arm in arms}

    mean_rows = []
    for model_id in MODEL_IDS:
        group = absolute[absolute["model_id"].eq(model_id)]
        for metric in metric_columns:
            mean_rows.append(
                {
                    "model_id": model_id,
                    "metric": metric,
                    **_mean_summary(group[metric].to_numpy()),
                }
            )
    mean_frame = pd.DataFrame(mean_rows)

    comparison_rows = []
    references = (
        screen.M1_MODEL_ID,
        screen.M4P_MODEL_ID,
        screen.M5_SHUFFLED_MODEL_ID,
        screen.M5_DEGREE_GATE_MODEL_ID,
    )
    for seed in cfg.seeds:
        for reference in references:
            for model_id in MODEL_IDS:
                if model_id == reference:
                    continue
                for metric in metric_columns:
                    reference_value = float(arm_map[(seed, reference)]["metrics"][metric])
                    model_value = float(arm_map[(seed, model_id)]["metrics"][metric])
                    comparison_rows.append(
                        {
                            "seed": seed,
                            "reference": reference,
                            "model_id": model_id,
                            "metric": metric,
                            "reference_value": reference_value,
                            "model_value": model_value,
                            "absolute_delta": model_value - reference_value,
                            "relative_change_pct": (
                                100.0 * (model_value - reference_value) / reference_value
                                if reference_value != 0.0
                                else np.nan
                            ),
                        }
                    )
    comparison = pd.DataFrame(comparison_rows)
    paired_mean_rows = []
    for (reference, model_id, metric), group in comparison.groupby(
        ["reference", "model_id", "metric"], sort=False
    ):
        paired_mean_rows.append(
            {
                "reference": reference,
                "model_id": model_id,
                "metric": metric,
                **_mean_summary(group["absolute_delta"].to_numpy()),
                "positive_seed_count": int((group["absolute_delta"] > 0).sum()),
            }
        )
    paired_mean = pd.DataFrame(paired_mean_rows)

    interaction_frames = []
    reading_by_seed = {}
    for seed in cfg.seeds:
        metrics = {
            model_id: arm_map[(seed, model_id)]["metrics"]
            for model_id in MODEL_IDS
        }
        interaction = screen.interaction_rows(metrics)
        interaction.insert(0, "seed", seed)
        interaction_frames.append(interaction)
        reading_by_seed[str(seed)] = screen.screening_reading(metrics)
    interactions = pd.concat(interaction_frames, ignore_index=True)

    stem = f"m5_economic_positive_weight_test_{prepared['config_hash']}"
    paths = {
        "absolute_csv": prepared["out_dir"] / f"{stem}.csv",
        "mean_csv": prepared["out_dir"] / f"{stem}_mean.csv",
        "comparison_csv": prepared["out_dir"] / f"{stem}_comparison.csv",
        "paired_mean_csv": prepared["out_dir"] / f"{stem}_paired_mean.csv",
        "interaction_csv": prepared["out_dir"] / f"{stem}_interaction.csv",
        "json": prepared["out_dir"] / f"{stem}.json",
    }
    result_helpers._atomic_csv(paths["absolute_csv"], absolute)
    result_helpers._atomic_csv(paths["mean_csv"], mean_frame)
    result_helpers._atomic_csv(paths["comparison_csv"], comparison)
    result_helpers._atomic_csv(paths["paired_mean_csv"], paired_mean)
    result_helpers._atomic_csv(paths["interaction_csv"], interactions)
    result_helpers._atomic_json(
        paths["json"],
        {
            "code_version": CODE_VERSION,
            "source_revision": prepared["revision"],
            "config": asdict(cfg),
            "preflight": preflight_summary(cfg),
            "input_manifest": prepared["manifest"],
            "data_stats": prepared["data"].get("data_stats", {}),
            "absolute_rows": absolute.to_dict("records"),
            "seed_mean_rows": mean_frame.to_dict("records"),
            "same_seed_comparison_rows": comparison.to_dict("records"),
            "same_seed_mean_rows": paired_mean.to_dict("records"),
            "interaction_rows": interactions.to_dict("records"),
            "descriptive_pre_registered_reading_by_seed": reading_by_seed,
            "interpretation": {
                "selection": "none; test is not used for model, epoch, or hyperparameter selection",
                "single_seed": "seed 42 is descriptive and does not establish stability or significance",
                "final_reporting": "use the frozen seeds 42--51 mean after the protocol check",
            },
            "arms": arms,
            "result_paths": {key: str(path) for key, path in paths.items()},
        },
    )
    absolute.attrs["mean"] = mean_frame
    absolute.attrs["comparison"] = comparison
    absolute.attrs["paired_mean"] = paired_mean
    absolute.attrs["interaction"] = interactions
    absolute.attrs["descriptive_reading"] = reading_by_seed
    absolute.attrs["result_paths"] = {key: str(path) for key, path in paths.items()}
    return absolute


def run_m5_economic_positive_test(
    cfg: M5EconomicPositiveTestConfig | None = None,
) -> pd.DataFrame:
    cfg = validate_test_config(cfg or configure_m5_economic_positive_test_run())
    print(json.dumps(preflight_summary(cfg), ensure_ascii=False, indent=2))
    prepared = _prepare(cfg)
    arms = []
    for seed in cfg.seeds:
        _prepare_seed_assignments(prepared, seed, cfg.shuffle_degree_bins)
        run_cfg = _screen_config(cfg, seed)
        for spec in screen.arm_specifications(prepared, run_cfg):
            print(
                f"\n===== seed {seed} | {spec['model_id']} | "
                f"fixed {cfg.epochs}-epoch final train ====="
            )
            arms.append(_run_arm(prepared, cfg, run_cfg, spec))
    frame = _persist(prepared, cfg, arms)
    print("\nTest 절대지표:")
    print(frame.to_string(index=False))
    print("\n동일 seed 대조군 비교:")
    print(frame.attrs["comparison"].to_string(index=False))
    print("\nM2×M4' 상호작용:")
    print(frame.attrs["interaction"].to_string(index=False))
    print("결과 파일:", frame.attrs["result_paths"])
    return frame


if __name__ == "__main__":
    print(
        json.dumps(
            preflight_summary(configure_m5_economic_positive_test_run()),
            ensure_ascii=False,
            indent=2,
        )
    )
