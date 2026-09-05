"""Seed-42 test-only run for the frozen CLV level/composition/price M2.

The former training and validation intervals are merged through DAY 697.
Each arm is trained for exactly 100 epochs and evaluated once on the fixed
DAY 698--704 test interval. DAY 705--711 is ignored and never evaluated.
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
import torch

from clv_run_state import ProgressStore, RunIdentity, clone_state, file_sha256
import lightgcn_clv_axis_specific_test10 as result_helpers
import lightgcn_clv_constrained_economic_embedding as selected
import lightgcn_clv_gradient_isolated_economic_interaction as evaluation
import lightgcn_clv_joint_nv as joint
import lightgcn_clv_joint_response_embedding as shared
import lightgcn_clv_moe as moe
import lightgcn_clv_residual as residual
import lightgcn_clv_v3 as v3


CODE_VERSION = "m2-clv-level-composition-price-test-only-seed42-v1"
SEED = 42
MATCHED_MODEL_ID = selected.MATCHED_MODEL_ID
MODEL_ID = selected.MODEL_ID
SHUFFLED_MODEL_ID = selected.SHUFFLED_MODEL_ID
ID_ONLY_MODEL_ID = selected.ID_ONLY_MODEL_ID
TRAINED_MODELS = (MATCHED_MODEL_ID, MODEL_ID, SHUFFLED_MODEL_ID)
REPORTED_MODELS = TRAINED_MODELS + (ID_ONLY_MODEL_ID,)
ACCURACY_METRICS = (
    "recall@10",
    "ndcg@10",
    "recall@20",
    "ndcg@20",
    "recall@50",
    "ndcg@50",
)


@dataclass(frozen=True)
class M2LevelCompositionPriceTestConfig:
    dataset: str = "dunnhumby"
    seed: int = SEED
    epochs: int = 100
    id_dim: int = 64
    clv_dim: int = 3
    rho: float = 0.05
    item_price_budget: float = 0.25
    n_layers: int = 2
    batch_size: int = 8192
    lr: float = 5e-4
    pref_reg: float = 1e-3
    input_days: int = 365
    shuffle_degree_bins: int = 10
    shuffle_seed: int = 1042
    out_dir: str = ""


def configure_m2_level_composition_price_test_run(
    **overrides,
) -> M2LevelCompositionPriceTestConfig:
    defaults = {
        "out_dir": (
            f"{v3.default_out_dir('dunnhumby')}"
            "_m2_level_composition_price_test_seed42_v1"
        )
    }
    return validate_test_config(
        M2LevelCompositionPriceTestConfig(**(defaults | overrides))
    )


def validate_test_config(
    cfg: M2LevelCompositionPriceTestConfig,
) -> M2LevelCompositionPriceTestConfig:
    required = {
        "dataset": "dunnhumby",
        "seed": SEED,
        "epochs": 100,
        "id_dim": 64,
        "clv_dim": 3,
        "rho": 0.05,
        "item_price_budget": 0.25,
        "n_layers": 2,
        "batch_size": 8192,
        "lr": 5e-4,
        "pref_reg": 1e-3,
        "input_days": 365,
        "shuffle_degree_bins": 10,
        "shuffle_seed": 1042,
    }
    for key, expected in required.items():
        if getattr(cfg, key) != expected:
            raise ValueError(
                f"M2 seed-42 test-only 설정은 {key}={expected!r}이어야 합니다"
            )
    if not cfg.out_dir:
        raise ValueError("M2 seed-42 test-only out_dir가 필요합니다")
    return cfg


def preflight_summary(cfg: M2LevelCompositionPriceTestConfig) -> dict:
    cfg = validate_test_config(cfg)
    return {
        "code_version": CODE_VERSION,
        "dataset": cfg.dataset,
        "seed": cfg.seed,
        "scope": "single-seed descriptive test-protocol run",
        "trained_models": list(TRAINED_MODELS),
        "reported_models": list(REPORTED_MODELS),
        "training_data": "DAY 1--697 (former train + validation)",
        "test_data": "DAY 698--704",
        "validation_constructed": False,
        "validation_selection": False,
        "early_stopping": False,
        "post_test_rows": "DAY 705--711 ignored",
        "holdout_constructed": False,
        "test_evaluation": (
            "one final-checkpoint evaluation per reported model; completed results are cached"
        ),
        "new_item_task": (
            "every user-item pair in merged training is excluded from test truth"
        ),
        "m2": {
            "architecture": "ID(64)|CLV level/composition relation(2)|explicit price fit(1)",
            "total_dim": cfg.id_dim + cfg.clv_dim,
            "rho": cfg.rho,
            "item_price_budget": cfg.item_price_budget,
            "joint_binary_lightgcn_propagation": True,
            "one_training_loop_and_optimizer": True,
            "external_reranking": False,
        },
        "fixed": {
            "graph": "binary",
            "negative_sampling": "uniform",
            "sample_weighting": False,
            "loss": "one plain pairwise BPR plus existing sampled ID L2",
            "new_loss_term": False,
            "min_user_interactions": 1,
            "min_item_interactions": 1,
            "layers": cfg.n_layers,
            "epochs": cfg.epochs,
        },
        "paired_controls": {
            "matched_rho0": MATCHED_MODEL_ID,
            "degree_matched_clv_assignment": SHUFFLED_MODEL_ID,
            "jointly_trained_id_only": ID_ONLY_MODEL_ID,
            "same_seed_initialization_batches_and_negatives": True,
        },
        "interpretation": {
            "single_seed": (
                "descriptive only; no stability, significance, or generalization claim"
            ),
            "test_use_prohibition": (
                "the result cannot select or modify the model, epoch, formula, or hyperparameter"
            ),
            "already_exposed_test": (
                "DAY 698--704 has been exposed by prior project runs, so this is a protocol "
                "comparison rather than a new confirmatory test"
            ),
        },
        "out_dir": cfg.out_dir,
    }


def _base_config(cfg: M2LevelCompositionPriceTestConfig) -> dict:
    base = dict(
        v3.configure_run(
            cfg.dataset,
            out_dir=cfg.out_dir,
            ARCH="pref_only",
            SEED_LIST=[cfg.seed],
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
                f"M2 seed-42 test-only 설정 오염: {key}={base.get(key)!r}"
            )
    return base


def _as_float(value) -> float:
    return float(value) if value is not None else math.nan


def validate_final_test_data(data: dict) -> None:
    if set(data.get("splits", {})) != {"test"}:
        raise RuntimeError(
            "M2 test-only runner에는 test split만 있어야 합니다: "
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
        raise RuntimeError("DAY 705--711 미사용 구간 경계가 달라졌습니다")
    expected_status = {
        "val": "merged_into_train",
        "test": "constructed",
        "holdout": "not_constructed",
    }
    if status != expected_status:
        raise RuntimeError(f"test-only split 상태가 오염됐습니다: {status}")


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def _config_hash(
    cfg: M2LevelCompositionPriceTestConfig, input_hash: str, revision: str
) -> str:
    payload = {
        "code_version": CODE_VERSION,
        "config": asdict(cfg),
        "models": REPORTED_MODELS,
        "input_hash": input_hash,
        "source_revision": revision,
    }
    return hashlib.sha256(_canonical(payload).encode()).hexdigest()[:12]


def _prepare(cfg: M2LevelCompositionPriceTestConfig) -> dict:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = moe.build_input_manifest(v3.SCHEMA[cfg.dataset])
    input_hash = moe.manifest_hash(manifest)
    revision = moe.source_revision()
    base_cfg = _base_config(cfg)
    data = v3.prepare_data(base_cfg, v3.DCFG)
    validate_final_test_data(data)
    if data.get("loss_w") is not None:
        raise RuntimeError("M2 test-only runner에 M4 표본 가중치가 섞였습니다")
    data["loss_w"] = None

    snapshot = residual.build_final_snapshot(
        data["train"], data["n_users"], v3.DCFG["is_date"], cfg.input_days
    )
    axes = joint.build_user_axis_inputs(snapshot, data["n_users"])
    q_n, q_v, q_c, clv_valid = evaluation.build_clv_inputs(axes)
    item_economic, item_economic_valid = shared.build_item_economic_inputs(
        data["train"], data["n_items"]
    )
    meta = v3.item_meta(data["train"], data["n_items"])
    thresholds = v3.segment_thresholds(axes["clv_proxy"], base_cfg["SEG_EDGES"])
    cache = v3.EvalCache(
        *data["splits"]["test"], axes["clv_proxy"], thresholds, data["n_items"]
    )
    prepared = {
        "out_dir": out_dir,
        "manifest": manifest,
        "input_hash": input_hash,
        "revision": revision,
        "base_cfg": base_cfg,
        "data": data,
        "axes": axes,
        "q_n": q_n,
        "q_v": q_v,
        "q_c": q_c,
        "clv_valid": clv_valid,
        "item_economic": item_economic,
        "item_economic_valid": item_economic_valid,
        "meta": meta,
        "thresholds": thresholds,
        "cache": cache,
    }
    prepared["config_hash"] = _config_hash(cfg, input_hash, revision)
    shuffle_cfg = selected.ConstrainedEconomicConfig(
        dataset=cfg.dataset,
        seed=cfg.seed,
        time_cutoff=704,
        evaluation_days=7,
        epochs=cfg.epochs,
        id_dim=cfg.id_dim,
        clv_dim=cfg.clv_dim,
        rho=cfg.rho,
        item_price_budget=cfg.item_price_budget,
        n_layers=cfg.n_layers,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        pref_reg=cfg.pref_reg,
        input_days=cfg.input_days,
        diagnostic_max_k=50,
        include_degree_matched_shuffle=False,
        shuffle_degree_bins=cfg.shuffle_degree_bins,
        shuffle_seed=cfg.shuffle_seed,
        out_dir=cfg.out_dir,
        baseline_result_dir="not-used-in-test-only-runner",
    )
    prepared["degree_matched_shuffle"] = selected._degree_matched_clv_shuffle(
        prepared, shuffle_cfg
    )
    return prepared


def _model_config(
    cfg: M2LevelCompositionPriceTestConfig,
) -> selected.ConstrainedEconomicConfig:
    return selected.ConstrainedEconomicConfig(
        dataset=cfg.dataset,
        seed=cfg.seed,
        time_cutoff=704,
        evaluation_days=7,
        epochs=cfg.epochs,
        id_dim=cfg.id_dim,
        clv_dim=cfg.clv_dim,
        rho=cfg.rho,
        item_price_budget=cfg.item_price_budget,
        n_layers=cfg.n_layers,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        pref_reg=cfg.pref_reg,
        input_days=cfg.input_days,
        diagnostic_max_k=50,
        include_degree_matched_shuffle=False,
        shuffle_degree_bins=cfg.shuffle_degree_bins,
        shuffle_seed=cfg.shuffle_seed,
        out_dir=cfg.out_dir,
        baseline_result_dir="not-used-in-test-only-runner",
    )


def _arm_paths(prepared: dict, model_id: str) -> dict[str, Path]:
    root = prepared["out_dir"] / "arms" / prepared["config_hash"]
    stem = f"{model_id}_s{SEED}_test"
    return {"result": root / f"{stem}.json", "checkpoint": root / f"{stem}.pt"}


def _arm_hash(
    prepared: dict, *, model_id: str, rho: float, assignment: str
) -> str:
    payload = {
        "run": prepared["config_hash"],
        "model_id": model_id,
        "seed": SEED,
        "rho": rho,
        "assignment": assignment,
        "split": "test",
    }
    return hashlib.sha256(_canonical(payload).encode()).hexdigest()[:12]


def _validate_cached_payload(payload: dict, model_id: str) -> None:
    if payload.get("model_id") != model_id:
        raise RuntimeError("cached test 모형 ID가 다릅니다")
    if payload.get("split") != "test" or payload.get("test_evaluation_count") != 1:
        raise RuntimeError("cached 결과가 1회 test 평가 결과가 아닙니다")
    if payload.get("validation_selection") is not False:
        raise RuntimeError("cached test 결과에 validation 선택이 섞였습니다")


def _run_trained_arm(
    prepared: dict,
    cfg: M2LevelCompositionPriceTestConfig,
    *,
    model_id: str,
    rho: float,
    assignment: dict | None = None,
    assignment_name: str = "observed",
) -> tuple[dict, torch.nn.Module]:
    model_cfg = _model_config(cfg)
    model, params = selected._build_model(
        prepared, model_cfg, rho, clv_assignment=assignment
    )
    paths = _arm_paths(prepared, model_id)
    if paths["result"].exists() and paths["checkpoint"].exists():
        payload = json.loads(paths["result"].read_text(encoding="utf-8"))
        _validate_cached_payload(payload, model_id)
        checkpoint = evaluation._load_state(paths["checkpoint"])
        if checkpoint.get("input_hash") != prepared["input_hash"]:
            raise RuntimeError("cached checkpoint와 현재 입력 hash가 다릅니다")
        model.load_state_dict(checkpoint["state"], strict=True)
        model.eval()
        print(f"  [cached] {model_id} 완료 결과 재사용(test 재평가 없음)")
        return payload, model

    store = ProgressStore(
        prepared["out_dir"] / "progress" / prepared["config_hash"],
        RunIdentity(
            stage="final_train_test",
            model_id=model_id,
            seed=cfg.seed,
            config_hash=_arm_hash(
                prepared,
                model_id=model_id,
                rho=rho,
                assignment=assignment_name,
            ),
            source_revision=prepared["revision"],
            input_hash=prepared["input_hash"],
        ),
    )
    training = result_helpers._fixed_epoch_train(
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
            "rho": rho,
            "clv_assignment": assignment_name,
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
        "model_id": model_id,
        "role": {
            MATCHED_MODEL_ID: "matched_control",
            MODEL_ID: "model",
            SHUFFLED_MODEL_ID: "assignment_control",
        }[model_id],
        "seed": cfg.seed,
        "split": "test",
        "final_epoch": cfg.epochs,
        "validation_selection": False,
        "test_evaluation_count": 1,
        "test_evaluated_at": datetime.now(timezone.utc).isoformat(),
        "rho": rho,
        "clv_assignment": assignment_name,
        "metrics": result_helpers._public_metrics(metrics),
        "diagnostics": model.representation_diagnostics(),
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
    return payload, model


def _id_only_payload(
    active_model: torch.nn.Module,
    active_payload: dict,
    prepared: dict,
    cfg: M2LevelCompositionPriceTestConfig,
) -> dict:
    paths = _arm_paths(prepared, ID_ONLY_MODEL_ID)
    if paths["result"].exists():
        payload = json.loads(paths["result"].read_text(encoding="utf-8"))
        _validate_cached_payload(payload, ID_ONLY_MODEL_ID)
        if payload.get("source_checkpoint_sha256") != active_payload.get(
            "checkpoint_sha256"
        ):
            raise RuntimeError("cached ID-only view의 원본 checkpoint가 다릅니다")
        print("  [cached] jointly-trained ID-only test 결과 재사용")
        return payload

    view = shared._IDOnlyView(active_model).to(v3.DEVICE)
    metrics, _ = moe._flat_evaluation(
        view,
        0.0,
        prepared["cache"],
        prepared["meta"],
        prepared["data"],
        prepared["base_cfg"],
        per_user=False,
    )
    payload = {
        "model_id": ID_ONLY_MODEL_ID,
        "role": "joint_training_ablation",
        "seed": cfg.seed,
        "split": "test",
        "final_epoch": cfg.epochs,
        "validation_selection": False,
        "test_evaluation_count": 1,
        "test_evaluated_at": datetime.now(timezone.utc).isoformat(),
        "rho": 0.0,
        "clv_assignment": "posthoc_id_view_of_observed_m2",
        "metrics": result_helpers._public_metrics(metrics),
        "diagnostics": {},
        "training": {"additional_training": False},
        "source_checkpoint": active_payload["checkpoint"],
        "source_checkpoint_sha256": active_payload["checkpoint_sha256"],
    }
    result_helpers._atomic_json(paths["result"], payload)
    return payload


def _geometric_mean(metrics: dict, names: tuple[str, ...]) -> float:
    values = np.asarray([metrics[name] for name in names], dtype=np.float64)
    if (values <= 0).any():
        return 0.0
    return float(np.exp(np.log(values).mean()))


def descriptive_reading(arm_map: dict[str, dict]) -> dict:
    matched = arm_map[MATCHED_MODEL_ID]["metrics"]
    actual = arm_map[MODEL_ID]["metrics"]
    shuffled = arm_map[SHUFFLED_MODEL_ID]["metrics"]
    id_only = arm_map[ID_ONLY_MODEL_ID]["metrics"]
    accuracy_ratios = {
        metric: actual[metric] / matched[metric] for metric in ACCURACY_METRICS
    }
    key = "price_purchase_amount_weighted_hit@10"
    return {
        "descriptive_only": True,
        "single_seed_final_decision_permitted": False,
        "accuracy_ratios_vs_matched_rho0": accuracy_ratios,
        "all_six_accuracy_metrics_within_99pct_of_matched": all(
            ratio >= 0.99 for ratio in accuracy_ratios.values()
        ),
        "six_metric_geomean_ratio_vs_degree_matched_shuffle": (
            _geometric_mean(actual, ACCURACY_METRICS)
            / _geometric_mean(shuffled, ACCURACY_METRICS)
        ),
        "weighted_hit_at_10_deltas": {
            "vs_matched_rho0": actual[key] - matched[key],
            "vs_jointly_trained_id_only": actual[key] - id_only[key],
            "vs_degree_matched_clv_shuffle": actual[key] - shuffled[key],
        },
        "high_clv_deltas": {
            "recall@10_vs_matched_rho0": (
                actual["고CLV_recall@10"] - matched["고CLV_recall@10"]
            ),
            "ndcg@10_vs_matched_rho0": (
                actual["고CLV_ndcg@10"] - matched["고CLV_ndcg@10"]
            ),
            "recall@10_vs_shuffle": (
                actual["고CLV_recall@10"] - shuffled["고CLV_recall@10"]
            ),
            "ndcg@10_vs_shuffle": (
                actual["고CLV_ndcg@10"] - shuffled["고CLV_ndcg@10"]
            ),
        },
        "statistical_note": (
            "seed 42 test-only descriptive comparison; no significance, stability, "
            "or generalization claim"
        ),
    }


def _persist(
    prepared: dict,
    cfg: M2LevelCompositionPriceTestConfig,
    arms: list[dict],
) -> pd.DataFrame:
    rows = []
    for arm in arms:
        rows.append(
            {
                "model_id": arm["model_id"],
                "role": arm["role"],
                "seed": arm["seed"],
                "split": arm["split"],
                "final_epoch": arm["final_epoch"],
                "rho": arm["rho"],
                "clv_assignment": arm["clv_assignment"],
                **arm.get("diagnostics", {}),
                **arm.get("training", {}).get("final_diagnostics", {}),
                **arm["metrics"],
            }
        )
    absolute = pd.DataFrame(rows)
    order = {model_id: index for index, model_id in enumerate(REPORTED_MODELS)}
    absolute["_order"] = absolute["model_id"].map(order)
    absolute = absolute.sort_values("_order").drop(columns="_order").reset_index(drop=True)

    arm_map = {arm["model_id"]: arm for arm in arms}
    metric_rows = {model_id: arm["metrics"] for model_id, arm in arm_map.items()}
    comparison = evaluation._metric_comparison(
        metric_rows,
        references=(MATCHED_MODEL_ID, ID_ONLY_MODEL_ID, SHUFFLED_MODEL_ID),
    )
    reading = descriptive_reading(arm_map)
    stem = f"m2_level_composition_price_test_{prepared['config_hash']}"
    paths = {
        "absolute_csv": prepared["out_dir"] / f"{stem}.csv",
        "comparison_csv": prepared["out_dir"] / f"{stem}_comparison.csv",
        "json": prepared["out_dir"] / f"{stem}.json",
    }
    result_helpers._atomic_csv(paths["absolute_csv"], absolute)
    result_helpers._atomic_csv(paths["comparison_csv"], comparison)
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
            "comparison_rows": comparison.to_dict("records"),
            "descriptive_reading": reading,
            "interpretation": {
                "selection": (
                    "none; test is not used for model, epoch, formula, or hyperparameter selection"
                ),
                "confirmation_limit": (
                    "DAY 698--704 was exposed previously, so this run cannot restore "
                    "confirmatory independence"
                ),
                "single_seed": (
                    "seed 42 alone cannot establish stability or statistical significance"
                ),
            },
            "arms": arms,
            "result_paths": {key: str(path) for key, path in paths.items()},
        },
    )
    absolute.attrs["comparison"] = comparison
    absolute.attrs["descriptive_reading"] = reading
    absolute.attrs["result_paths"] = {key: str(path) for key, path in paths.items()}
    return absolute


def run_m2_level_composition_price_test(
    cfg: M2LevelCompositionPriceTestConfig | None = None,
) -> pd.DataFrame:
    cfg = validate_test_config(
        cfg or configure_m2_level_composition_price_test_run()
    )
    print(json.dumps(preflight_summary(cfg), ensure_ascii=False, indent=2))
    prepared = _prepare(cfg)
    print("\n===== matched rho=0 | seed 42 | fixed 100-epoch final train =====")
    matched, _ = _run_trained_arm(
        prepared, cfg, model_id=MATCHED_MODEL_ID, rho=0.0
    )
    print("\n===== actual CLV M2 rho=.05 | seed 42 | fixed 100-epoch final train =====")
    active, active_model = _run_trained_arm(
        prepared, cfg, model_id=MODEL_ID, rho=cfg.rho
    )
    print("\n===== degree-matched CLV shuffle rho=.05 | seed 42 =====")
    shuffled, _ = _run_trained_arm(
        prepared,
        cfg,
        model_id=SHUFFLED_MODEL_ID,
        rho=cfg.rho,
        assignment=prepared["degree_matched_shuffle"],
        assignment_name="degree_matched_shuffle",
    )
    print("\n===== jointly-trained ID-only view | no additional training =====")
    id_only = _id_only_payload(
        active_model, active, prepared, cfg
    )
    frame = _persist(prepared, cfg, [matched, active, shuffled, id_only])
    print("\nTest 절대지표:")
    print(frame.to_string(index=False))
    print("\n동일 seed 대조군 비교:")
    print(frame.attrs["comparison"].to_string(index=False))
    print("\n서술적 판독:")
    print(json.dumps(frame.attrs["descriptive_reading"], ensure_ascii=False, indent=2))
    print("결과 파일:", frame.attrs["result_paths"])
    return frame


if __name__ == "__main__":
    print(
        json.dumps(
            preflight_summary(configure_m2_level_composition_price_test_run()),
            ensure_ascii=False,
            indent=2,
        )
    )
