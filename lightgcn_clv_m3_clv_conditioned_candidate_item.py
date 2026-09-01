"""One-seed screen for a CLV-conditioned candidate-item relation M3."""

from __future__ import annotations

import gc
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from clv_m3_clv_conditioned_candidate_item_graph import (
    ACTIVE_ARMS,
    ARM_ACTUAL,
    ARM_GENERAL,
    ARM_SHUFFLE,
    DEFAULT_ITEM_KAPPA,
    DEFAULT_ITEM_MIN_SUPPORT_USERS,
    DEFAULT_MAX_CANDIDATE_ITEMS,
    DEFAULT_MAX_TARGET_CATEGORIES,
    build_clv_conditioned_candidate_item_graph,
)
from clv_m3_clv_conditioned_category_transition_graph import (
    DEFAULT_CROSS_FIT_FOLDS,
    DEFAULT_KAPPA,
    DEFAULT_MIN_SUPPORT_USERS,
    DEFAULT_SHUFFLE_DEGREE_BINS,
    DEFAULT_SHUFFLE_SEED,
)
from clv_m3_clv_conditioned_candidate_item_model import (
    CLVCandidateItemLightGCN,
    build_binary_directional_blocks,
)
from clv_run_state import ProgressStore, RunIdentity, clone_state, file_sha256
import lightgcn_clv_axis_specific_test10 as fixed_train
import lightgcn_clv_gatefree_lowdim as baseline_support
import lightgcn_clv_moe as moe
import lightgcn_clv_v3 as v3


CODE_VERSION = "m3-clv-conditioned-candidate-item-historical-screen-v1"
M1_ID = "m1_baseline"
GENERAL_ID = "m3_general_candidate_item_relation_control"
ACTUAL_ID = "m3_clv_conditioned_candidate_item_relation"
SHUFFLE_ID = "m3_clv_conditioned_candidate_item_relation_shuffle"
ARM_MODEL_IDS = {
    ARM_GENERAL: GENERAL_ID,
    ARM_ACTUAL: ACTUAL_ID,
    ARM_SHUFFLE: SHUFFLE_ID,
}
ACCURACY_METRICS = (
    "recall@10",
    "ndcg@10",
    "recall@20",
    "ndcg@20",
    "recall@50",
    "ndcg@50",
)


@dataclass(frozen=True)
class CLVCandidateItemConfig:
    dataset: str = "dunnhumby"
    seed: int = 42
    time_cutoff: int = 690
    evaluation_days: int = 7
    epochs: int = 100
    id_dim: int = 64
    n_layers: int = 2
    batch_size: int = 8192
    lr: float = 5e-4
    pref_reg: float = 1e-3
    gamma: float = 0.075
    category_kappa: float = DEFAULT_KAPPA
    category_min_support_users: int = DEFAULT_MIN_SUPPORT_USERS
    item_kappa: float = DEFAULT_ITEM_KAPPA
    item_min_support_users: int = DEFAULT_ITEM_MIN_SUPPORT_USERS
    shuffle_seed: int = DEFAULT_SHUFFLE_SEED
    shuffle_degree_bins: int = DEFAULT_SHUFFLE_DEGREE_BINS
    cross_fit_folds: int = DEFAULT_CROSS_FIT_FOLDS
    max_target_categories: int = DEFAULT_MAX_TARGET_CATEGORIES
    max_candidate_items: int = DEFAULT_MAX_CANDIDATE_ITEMS
    out_dir: str = ""
    baseline_result_dir: str = ""


def configure_clv_candidate_item_run(**overrides) -> CLVCandidateItemConfig:
    defaults = {
        "out_dir": (
            f"{v3.default_out_dir('dunnhumby')}"
            "_m3_clv_candidate_item_historical_screen_v1"
        ),
        "baseline_result_dir": (
            f"{v3.default_out_dir('dunnhumby')}"
            "_m2_repeatshare_historical_backtest_v1"
        ),
    }
    return validate_clv_candidate_item_config(
        CLVCandidateItemConfig(**(defaults | overrides))
    )


def validate_clv_candidate_item_config(
    cfg: CLVCandidateItemConfig,
) -> CLVCandidateItemConfig:
    required = {
        "dataset": "dunnhumby",
        "seed": 42,
        "time_cutoff": 690,
        "evaluation_days": 7,
        "epochs": 100,
        "id_dim": 64,
        "n_layers": 2,
        "gamma": 0.075,
        "category_kappa": 20.0,
        "category_min_support_users": 5,
        "item_kappa": 20.0,
        "item_min_support_users": 5,
        "shuffle_seed": 42,
        "shuffle_degree_bins": 10,
        "cross_fit_folds": 5,
        "max_target_categories": 20,
        "max_candidate_items": 100,
    }
    for key, expected in required.items():
        actual = getattr(cfg, key)
        matches = (
            bool(np.isclose(actual, expected))
            if isinstance(expected, (int, float))
            else actual == expected
        )
        if not matches:
            raise ValueError(
                f"빠른 M3 candidate-item screen은 {key}={expected!r}이어야 합니다"
            )
    if cfg.batch_size <= 0 or cfg.lr <= 0 or cfg.pref_reg < 0:
        raise ValueError("학습 설정이 잘못됐습니다")
    if not cfg.out_dir or not cfg.baseline_result_dir:
        raise ValueError("out_dir와 baseline_result_dir가 필요합니다")
    return cfg


def preflight_summary(cfg: CLVCandidateItemConfig) -> dict:
    cfg = validate_clv_candidate_item_config(cfg)
    return {
        "code_version": CODE_VERSION,
        "dataset": cfg.dataset,
        "seed": cfg.seed,
        "trained_models": [GENERAL_ID, ACTUAL_ID, SHUFFLE_ID],
        "reused_comparator": M1_ID,
        "historical_development_split": {
            "train_end_inclusive": 683,
            "evaluation_start_inclusive": 684,
            "evaluation_end_inclusive": 690,
            "final_test_constructed": False,
            "holdout_constructed": False,
        },
        "research_question": (
            "whether historical CLV can distinguish new-to-user candidate items "
            "inside the graph beyond pooled transitions and a degree-matched shuffle"
        ),
        "m3": {
            "research_axis": "graph structure and propagation",
            "status": (
                "exploratory candidate-edge extension after category-message "
                "resolution failure"
            ),
            "historical_clv_proxy": "N_hat * V_hat",
            "n_hat": "number of distinct train baskets",
            "v_hat": "mean train basket value",
            "category_direction": (
                "CLV-conditioned first-acquisition next-category probability"
            ),
            "within_category_candidate_allocation": (
                "CLV-conditioned first-acquisition item probability"
            ),
            "actual_candidate_edge": (
                "positive absolute probability excess over the pooled "
                "candidate-item distribution"
            ),
            "general_relation_control": "pooled candidate-item probability",
            "shuffle_control": (
                "CLV values permuted within binary user-degree deciles in both "
                "relation estimation and user conditioning"
            ),
            "candidate_train_pairs_excluded": True,
            "cross_fit_folds": cfg.cross_fit_folds,
            "category_kappa": cfg.category_kappa,
            "category_minimum_distinct_user_support": (
                cfg.category_min_support_users
            ),
            "item_kappa": cfg.item_kappa,
            "item_minimum_distinct_user_support": cfg.item_min_support_users,
            "max_target_categories_per_user": cfg.max_target_categories,
            "max_candidate_items_per_user": cfg.max_candidate_items,
            "row_mass_normalized": True,
            "gamma": cfg.gamma,
            "item_price_input": False,
            "external_reranking": False,
        },
        "fixed": {
            "binary_m1_graph_preserved": True,
            "negative_sampling": "uniform",
            "sample_weighting": False,
            "loss": "one plain pairwise BPR plus existing sampled ID L2",
            "new_loss_term": False,
            "one_training_loop_and_optimizer": True,
            "min_user_interactions": 1,
            "min_item_interactions": 1,
            "epochs": cfg.epochs,
            "validation_or_epoch_selection": False,
        },
        "reading_rule": {
            "primary": (
                "actual six-metric geometric balance > 1 against M1, pooled "
                "candidate relation, and degree-matched CLV shuffle"
            ),
            "accuracy_metrics": list(ACCURACY_METRICS),
            "accuracy_guardrails": False,
            "economic_and_exposure_metrics": "descriptive diagnostics only",
            "statistical_note": (
                "single-seed exploratory screen; no significance or generalization claim"
            ),
        },
        "out_dir": cfg.out_dir,
        "baseline_result_dir": cfg.baseline_result_dir,
    }


def _base_config(cfg: CLVCandidateItemConfig) -> dict:
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
        GATE_MODE="none",
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
        "TEST_DAYS": 7,
        "HOLDOUT_DAYS": 0,
        "EVAL_TEST": True,
        "EVAL_HOLDOUT": False,
        "GRAPH_MODE": "binary",
        "LOSS_MODE": "plain",
        "NEG_MODE": "uniform",
        "MIN_USER_INTER": 1,
        "MIN_ITEM_INTER": 1,
        "N_LAYERS": 2,
        "EPOCHS": 100,
    }
    for key, expected in required.items():
        if base[key] != expected:
            raise RuntimeError(f"M3 candidate-item 설정 오염: {key}={base[key]!r}")
    return base


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def _config_hash(cfg, input_hash: str, revision: str) -> str:
    payload = {
        "code_version": CODE_VERSION,
        "config": asdict(cfg),
        "input_hash": input_hash,
        "source_revision": revision,
    }
    return hashlib.sha256(_canonical(payload).encode()).hexdigest()[:12]


def _prepare(cfg: CLVCandidateItemConfig) -> dict:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = moe.build_input_manifest(v3.SCHEMA[cfg.dataset])
    input_hash = moe.manifest_hash(manifest)
    revision = moe.source_revision()
    base_cfg = _base_config(cfg)
    data = v3.prepare_data(base_cfg, v3.DCFG)
    if set(data["splits"]) != {"test"}:
        raise RuntimeError(f"historical 개발분할 외 오염: {sorted(data['splits'])}")
    if float(data["train"].t.max()) != 683.0:
        raise RuntimeError(f"historical train 종료일 오류: {data['train'].t.max()}")
    if data.get("loss_w") is not None:
        raise RuntimeError("M3 candidate-item screen에 M4 표본 가중치가 섞였습니다")
    data["loss_w"] = None

    graph = build_clv_conditioned_candidate_item_graph(
        data["train"],
        data["n_users"],
        data["n_items"],
        data["n_cat"],
        category_kappa=cfg.category_kappa,
        category_min_support_users=cfg.category_min_support_users,
        item_kappa=cfg.item_kappa,
        item_min_support_users=cfg.item_min_support_users,
        shuffle_seed=cfg.shuffle_seed,
        shuffle_degree_bins=cfg.shuffle_degree_bins,
        cross_fit_folds=cfg.cross_fit_folds,
        max_target_categories=cfg.max_target_categories,
        max_candidate_items=cfg.max_candidate_items,
    )
    baseline = baseline_support._load_compatible_baseline(cfg, manifest)
    meta = v3.item_meta(data["train"], data["n_items"])
    thresholds = v3.segment_thresholds(graph.clv_proxy, base_cfg["SEG_EDGES"])
    cache = v3.EvalCache(
        *data["splits"]["test"],
        graph.clv_proxy,
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
        "graph": graph,
        "baseline": baseline,
        "meta": meta,
        "thresholds": thresholds,
        "cache": cache,
    }
    prepared["config_hash"] = _config_hash(cfg, input_hash, revision)
    return prepared


def _build_model(
    prepared: dict,
    cfg: CLVCandidateItemConfig,
    arm: str,
) -> CLVCandidateItemLightGCN:
    if arm not in ACTIVE_ARMS:
        raise KeyError(arm)
    data = prepared["data"]
    v3.set_seed(cfg.seed)
    edge_users = data["pos_key"] // data["n_items"]
    edge_items = data["pos_key"] % data["n_items"]
    user_item, item_user = build_binary_directional_blocks(
        edge_users,
        edge_items,
        data["n_users"],
        data["n_items"],
        v3.DEVICE,
    )
    return CLVCandidateItemLightGCN(
        n_users=data["n_users"],
        n_items=data["n_items"],
        base_user_from_item=user_item,
        base_item_from_user=item_user,
        user_candidate_item=(
            prepared["graph"].user_item_operators[arm].to(v3.DEVICE)
        ),
        gamma=cfg.gamma,
        dim=cfg.id_dim,
        n_layers=cfg.n_layers,
        pref_reg=cfg.pref_reg,
    ).to(v3.DEVICE)


def _arm_paths(prepared: dict, model_id: str) -> dict[str, Path]:
    root = prepared["out_dir"] / "arms" / prepared["config_hash"]
    stem = f"{model_id}_s42"
    return {
        "result": root / f"{stem}.json",
        "checkpoint": root / f"{stem}.pt",
    }


def _arm_hash(prepared: dict, model_id: str, arm: str) -> str:
    return hashlib.sha256(
        _canonical(
            {
                "run": prepared["config_hash"],
                "model_id": model_id,
                "graph_arm": arm,
                "seed": 42,
            }
        ).encode()
    ).hexdigest()[:12]


def _run_arm(
    prepared: dict,
    cfg: CLVCandidateItemConfig,
    *,
    arm: str,
    model_id: str,
) -> dict:
    paths = _arm_paths(prepared, model_id)
    if paths["result"].exists() and paths["checkpoint"].exists():
        payload = json.loads(paths["result"].read_text(encoding="utf-8"))
        if payload.get("input_hash") != prepared["input_hash"]:
            raise RuntimeError("cached result와 현재 입력 hash가 다릅니다")
        print(f"  [cached] {model_id} 완료 결과 재사용")
        return payload

    model = _build_model(prepared, cfg, arm)
    store = ProgressStore(
        prepared["out_dir"] / "progress" / prepared["config_hash"],
        RunIdentity(
            stage="historical_development_train",
            model_id=model_id,
            seed=cfg.seed,
            config_hash=_arm_hash(prepared, model_id, arm),
            source_revision=prepared["revision"],
            input_hash=prepared["input_hash"],
        ),
    )
    training = fixed_train._fixed_epoch_train(
        model,
        list(model.parameters()),
        prepared,
        cfg,
        model_id,
        cfg.seed,
        store,
    )
    model.eval()
    metrics, _ = moe._flat_evaluation(
        model,
        0.0,
        prepared["cache"],
        prepared["meta"],
        prepared["data"],
        prepared["base_cfg"],
        per_user=False,
    )
    model_diagnostics = model.representation_diagnostics()

    paths["checkpoint"].parent.mkdir(parents=True, exist_ok=True)
    temporary = paths["checkpoint"].with_suffix(".pt.tmp")
    torch.save(
        {
            "state": clone_state(model),
            "model_id": model_id,
            "graph_arm": arm,
            "config": asdict(cfg),
            "training": training,
            "source_revision": prepared["revision"],
            "input_hash": prepared["input_hash"],
        },
        temporary,
    )
    os.replace(temporary, paths["checkpoint"])
    payload = {
        "model_id": model_id,
        "role": "model" if arm == ARM_ACTUAL else "control",
        "graph_arm": arm,
        "seed": cfg.seed,
        "split": "historical_development_days_684_690",
        "final_epoch": cfg.epochs,
        "input_hash": prepared["input_hash"],
        "metrics": fixed_train._public_metrics(metrics),
        "model_diagnostics": model_diagnostics,
        "graph_diagnostics": prepared["graph"].diagnostics["arms"][arm],
        "training": training,
        "checkpoint": str(paths["checkpoint"]),
        "checkpoint_sha256": file_sha256(paths["checkpoint"]),
    }
    fixed_train._atomic_json(paths["result"], payload)
    store.mark_complete(
        epoch=cfg.epochs,
        max_epoch=cfg.epochs,
        selection="none",
        split="historical_development_days_684_690",
        checkpoint_path=str(paths["checkpoint"]),
        result_path=str(paths["result"]),
    )
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return payload


def _row(frame: pd.DataFrame, model_id: str) -> pd.Series:
    selected = frame.loc[frame["model_id"].eq(model_id)]
    if len(selected) != 1:
        raise ValueError(f"expected exactly one row for {model_id}, got {len(selected)}")
    return selected.iloc[0]


def _six_metric_balance(actual: pd.Series, reference: pd.Series) -> float:
    ratios = np.asarray(
        [float(actual[name]) / float(reference[name]) for name in ACCURACY_METRICS],
        dtype=np.float64,
    )
    if not np.isfinite(ratios).all() or np.any(ratios <= 0):
        raise ValueError("six accuracy metrics must be finite and positive")
    return float(np.exp(np.log(ratios).mean()))


def attribution_reading(frame: pd.DataFrame) -> dict:
    actual = _row(frame, ACTUAL_ID)
    references = {
        "m1": _row(frame, M1_ID),
        "general_candidate_relation": _row(frame, GENERAL_ID),
        "shuffle": _row(frame, SHUFFLE_ID),
    }
    balances = {
        name: _six_metric_balance(actual, reference)
        for name, reference in references.items()
    }
    ratios = {
        name: {
            metric: float(actual[metric]) / float(reference[metric])
            for metric in ACCURACY_METRICS
        }
        for name, reference in references.items()
    }
    weighted_hit = "price_purchase_amount_weighted_hit@10"
    recommended_price = "mean_recommended_price_percentile@10"
    return {
        "clv_attribution_supported": bool(
            all(value > 1.0 for value in balances.values())
        ),
        "primary_rule": (
            "actual six-metric geometric balance > 1 against M1, pooled "
            "candidate relation, and degree-matched CLV shuffle"
        ),
        "six_metric_balance_actual_vs_m1": balances["m1"],
        "six_metric_balance_actual_vs_general_candidate_relation": balances[
            "general_candidate_relation"
        ],
        "six_metric_balance_actual_vs_shuffle": balances["shuffle"],
        "accuracy_ratios": ratios,
        "price_purchase_amount_weighted_hit@10_deltas": {
            name: float(actual[weighted_hit]) - float(reference[weighted_hit])
            for name, reference in references.items()
        },
        "mean_recommended_price_percentile@10_deltas": {
            name: float(actual[recommended_price])
            - float(reference[recommended_price])
            for name, reference in references.items()
        },
        "descriptive_diagnostics_are_not_success_guards": True,
        "single_seed_limitation": (
            "no variance, interval, significance, or generalization claim"
        ),
    }


def _comparison(frame: pd.DataFrame) -> pd.DataFrame:
    actual = _row(frame, ACTUAL_ID)
    rows = []
    for reference_id in (M1_ID, GENERAL_ID, SHUFFLE_ID):
        reference = _row(frame, reference_id)
        for metric in frame.columns:
            if "@" not in metric:
                continue
            left, right = actual[metric], reference[metric]
            if not isinstance(left, (int, float, np.number)) or not isinstance(
                right, (int, float, np.number)
            ):
                continue
            rows.append(
                {
                    "model_id": ACTUAL_ID,
                    "reference": reference_id,
                    "metric": metric,
                    "reference_value": float(right),
                    "model_value": float(left),
                    "absolute_delta": float(left - right),
                    "relative_change_pct": (
                        float(100.0 * (left - right) / right)
                        if right != 0
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def run_clv_candidate_item_screen(
    cfg: CLVCandidateItemConfig | None = None,
) -> pd.DataFrame:
    cfg = validate_clv_candidate_item_config(
        cfg or configure_clv_candidate_item_run()
    )
    preflight = preflight_summary(cfg)
    print(json.dumps(preflight, ensure_ascii=False, indent=2))
    prepared = _prepare(cfg)

    arms = []
    for graph_arm in ACTIVE_ARMS:
        model_id = ARM_MODEL_IDS[graph_arm]
        print(
            f"\n===== {model_id} | seed {cfg.seed} | "
            f"fixed {cfg.epochs} epochs ====="
        )
        arms.append(
            _run_arm(prepared, cfg, arm=graph_arm, model_id=model_id)
        )

    source_baseline = dict(prepared["baseline"])
    baseline = {
        **source_baseline,
        "model_id": M1_ID,
        "role": "reused_baseline",
        "source_model_id": source_baseline.get("model_id"),
    }
    rows = [baseline]
    for arm in arms:
        rows.append(
            {
                "model_id": arm["model_id"],
                "role": arm["role"],
                "graph_arm": arm["graph_arm"],
                "seed": arm["seed"],
                "split": arm["split"],
                "final_epoch": arm["final_epoch"],
                **arm["model_diagnostics"],
                **arm["graph_diagnostics"],
                **arm["metrics"],
            }
        )
    frame = pd.DataFrame(rows)
    comparison = _comparison(frame)
    reading = attribution_reading(frame)

    stem = f"m3_clv_candidate_item_{prepared['config_hash']}"
    paths = {
        "absolute_csv": prepared["out_dir"] / f"{stem}.csv",
        "comparison_csv": prepared["out_dir"] / f"{stem}_comparison.csv",
        "json": prepared["out_dir"] / f"{stem}.json",
    }
    fixed_train._atomic_csv(paths["absolute_csv"], frame)
    fixed_train._atomic_csv(paths["comparison_csv"], comparison)
    payload = {
        "code_version": CODE_VERSION,
        "source_revision": prepared["revision"],
        "config": asdict(cfg),
        "preflight": preflight,
        "input_manifest": prepared["manifest"],
        "reused_baseline_source": source_baseline.get("source_result"),
        "graph_diagnostics": prepared["graph"].diagnostics,
        "absolute_rows": frame.to_dict("records"),
        "comparison_rows": comparison.to_dict("records"),
        "attribution_reading": reading,
        "result_paths": {key: str(value) for key, value in paths.items()},
    }
    fixed_train._atomic_json(paths["json"], payload)
    frame.attrs["comparison"] = comparison.to_dict("records")
    frame.attrs["attribution_reading"] = reading
    frame.attrs["graph_diagnostics"] = prepared["graph"].diagnostics
    frame.attrs["result_paths"] = {
        key: str(value) for key, value in paths.items()
    }

    print("\nM3 CLV 귀속 판정:")
    print(json.dumps(reading, ensure_ascii=False, indent=2))
    print("결과 파일:", frame.attrs["result_paths"])
    return frame


if __name__ == "__main__":
    run_clv_candidate_item_screen()
