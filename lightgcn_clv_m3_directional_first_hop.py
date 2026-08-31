"""Seed-42 historical screen for a CLV-conditioned directional M3 graph.

This is an exploratory historical-development run, not the final test.  It
trains three graph arms with the same plain BPR loop and compares them with a
protocol-compatible M1 result: a CLV-free relationship control, the actual
historical-CLV assignment, and a binary-degree-stratified CLV shuffle.
"""

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

from clv_m3_directional_first_hop_model import DirectionalFirstHopLightGCN
from clv_m3_directional_value_graph import (
    ARM_ACTUAL,
    ARM_RELATION_ONLY,
    ARM_SHUFFLE,
    DEFAULT_BETA_CAP,
    DEFAULT_SHUFFLE_DEGREE_BINS,
    DEFAULT_SHUFFLE_SEED,
    DEFAULT_TARGET_STRENGTH,
    build_directional_operators,
    build_directional_value_graph,
)
from clv_run_state import ProgressStore, RunIdentity, clone_state, file_sha256
import lightgcn_clv_axis_specific_test10 as fixed_train
import lightgcn_clv_gatefree_lowdim as baseline_support
import lightgcn_clv_gradient_isolated_economic_interaction as shared
import lightgcn_clv_moe as moe
import lightgcn_clv_v3 as v3


CODE_VERSION = "m3-clv-directional-first-hop-historical-screen-v1"
M1_ID = "m1_baseline"
RELATION_ONLY_ID = "m3_value_relation_first_hop_control"
ACTUAL_ID = "m3_clv_directional_first_hop"
SHUFFLE_ID = "m3_clv_directional_first_hop_shuffle"
ARM_MODEL_IDS = {
    ARM_RELATION_ONLY: RELATION_ONLY_ID,
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
class DirectionalFirstHopConfig:
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
    target_strength: float = DEFAULT_TARGET_STRENGTH
    beta_cap: float = DEFAULT_BETA_CAP
    shuffle_seed: int = DEFAULT_SHUFFLE_SEED
    shuffle_degree_bins: int = DEFAULT_SHUFFLE_DEGREE_BINS
    out_dir: str = ""
    baseline_result_dir: str = ""


def configure_directional_first_hop_run(
    **overrides,
) -> DirectionalFirstHopConfig:
    defaults = {
        "out_dir": (
            f"{v3.default_out_dir('dunnhumby')}"
            "_m3_clv_directional_first_hop_historical_screen_v1"
        ),
        "baseline_result_dir": (
            f"{v3.default_out_dir('dunnhumby')}"
            "_m2_repeatshare_historical_backtest_v1"
        ),
    }
    return validate_directional_first_hop_config(
        DirectionalFirstHopConfig(**(defaults | overrides))
    )


def validate_directional_first_hop_config(
    cfg: DirectionalFirstHopConfig,
) -> DirectionalFirstHopConfig:
    required = {
        "dataset": "dunnhumby",
        "seed": 42,
        "time_cutoff": 690,
        "evaluation_days": 7,
        "epochs": 100,
        "id_dim": 64,
        "n_layers": 2,
        "target_strength": 0.075,
        "beta_cap": 20.0,
        "shuffle_seed": 42,
        "shuffle_degree_bins": 10,
    }
    for key, expected in required.items():
        if getattr(cfg, key) != expected:
            raise ValueError(f"빠른 M3 screen은 {key}={expected!r}이어야 합니다")
    if cfg.batch_size <= 0 or cfg.lr <= 0 or cfg.pref_reg < 0:
        raise ValueError("학습 설정이 잘못됐습니다")
    if not cfg.out_dir or not cfg.baseline_result_dir:
        raise ValueError("out_dir와 baseline_result_dir가 필요합니다")
    return cfg


def preflight_summary(cfg: DirectionalFirstHopConfig) -> dict:
    cfg = validate_directional_first_hop_config(cfg)
    return {
        "code_version": CODE_VERSION,
        "dataset": cfg.dataset,
        "seed": cfg.seed,
        "trained_models": [RELATION_ONLY_ID, ACTUAL_ID, SHUFFLE_ID],
        "reused_comparator": M1_ID,
        "historical_development_split": {
            "train_end_inclusive": 683,
            "evaluation_start_inclusive": 684,
            "evaluation_end_inclusive": 690,
            "final_test_constructed": False,
            "holdout_constructed": False,
        },
        "m3": {
            "research_axis": "graph structure and propagation",
            "historical_clv_proxy": "N_hat * V_hat",
            "n_hat": "number of distinct train baskets",
            "v_hat": "mean train basket value",
            "edge_relationship": (
                "mean item share of a user's basket value, ranked within user"
            ),
            "actual_multiplier": "exp(beta * q_CLV(user) * z(user,item))",
            "mass_preservation": (
                "each user's selected first-hop coefficient sum equals M1"
            ),
            "changed_term": "user first-hop only",
            "preserved_terms": [
                "binary M1 layer-0 embeddings",
                "binary M1 item first-hop",
                "binary M1 user/item two-hop",
                "binary M1 final item representation",
            ],
            "target_first_hop_strength": cfg.target_strength,
            "relationship_control": "q_CLV(user) == 1 at matched strength",
            "shuffle_control": (
                "q_CLV values permuted within binary user-degree deciles"
            ),
            "item_price_input": False,
            "external_reranking": False,
        },
        "fixed": {
            "edge_set": "same binary unique user-item pairs as M1",
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
                "actual six-metric geometric balance > 1 against M1, "
                "relationship-only, and degree-matched CLV shuffle"
            ),
            "accuracy_metrics": list(ACCURACY_METRICS),
            "accuracy_guardrails": False,
            "economic_and_exposure_metrics": "descriptive diagnostics only",
            "statistical_note": (
                "single-seed exploratory screen; no significance or "
                "generalization claim"
            ),
        },
        "next_if_positive": (
            "freeze the design, then run additional seeds or H&M before final test"
        ),
        "out_dir": cfg.out_dir,
        "baseline_result_dir": cfg.baseline_result_dir,
    }


def _base_config(cfg: DirectionalFirstHopConfig) -> dict:
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
            raise RuntimeError(f"M3 historical screen 설정 오염: {key}={base[key]!r}")
    return base


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def _config_hash(
    cfg: DirectionalFirstHopConfig, input_hash: str, revision: str
) -> str:
    payload = {
        "code_version": CODE_VERSION,
        "config": asdict(cfg),
        "input_hash": input_hash,
        "source_revision": revision,
    }
    return hashlib.sha256(_canonical(payload).encode()).hexdigest()[:12]


def _prepare(cfg: DirectionalFirstHopConfig) -> dict:
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
        raise RuntimeError("M3 screen에 M4 표본 가중치가 섞였습니다")
    data["loss_w"] = None

    graph = build_directional_value_graph(
        data["train"],
        data["n_users"],
        data["n_items"],
        target_strength=cfg.target_strength,
        beta_cap=cfg.beta_cap,
        shuffle_seed=cfg.shuffle_seed,
        shuffle_degree_bins=cfg.shuffle_degree_bins,
    )
    missed = [
        arm
        for arm in (ARM_RELATION_ONLY, ARM_ACTUAL, ARM_SHUFFLE)
        if not graph.diagnostics["arms"][arm]["target_reached"]
    ]
    if missed:
        raise RuntimeError(f"matched first-hop strength에 도달하지 못한 arm: {missed}")
    baseline = baseline_support._load_compatible_baseline(cfg, manifest)
    meta = v3.item_meta(data["train"], data["n_items"])
    thresholds = v3.segment_thresholds(
        graph.clv_proxy, base_cfg["SEG_EDGES"]
    )
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
    prepared: dict, cfg: DirectionalFirstHopConfig, arm: str
) -> DirectionalFirstHopLightGCN:
    data = prepared["data"]
    v3.set_seed(cfg.seed)
    base_u_i, base_i_u, active_u_i = build_directional_operators(
        prepared["graph"],
        arm,
        data["n_users"],
        data["n_items"],
        v3.DEVICE,
    )
    return DirectionalFirstHopLightGCN(
        n_users=data["n_users"],
        n_items=data["n_items"],
        base_user_from_item=base_u_i,
        base_item_from_user=base_i_u,
        active_user_from_item=active_u_i,
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
    cfg: DirectionalFirstHopConfig,
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
        "model_diagnostics": model.representation_diagnostics(),
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
        "relation_only": _row(frame, RELATION_ONLY_ID),
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
            "actual six-metric geometric balance > 1 against M1, "
            "relationship-only, and degree-matched CLV shuffle"
        ),
        "six_metric_balance_actual_vs_m1": balances["m1"],
        "six_metric_balance_actual_vs_relation_only": balances["relation_only"],
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
    for reference_id in (M1_ID, RELATION_ONLY_ID, SHUFFLE_ID):
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


def run_directional_first_hop_screen(
    cfg: DirectionalFirstHopConfig | None = None,
) -> pd.DataFrame:
    cfg = validate_directional_first_hop_config(
        cfg or configure_directional_first_hop_run()
    )
    preflight = preflight_summary(cfg)
    print(json.dumps(preflight, ensure_ascii=False, indent=2))
    prepared = _prepare(cfg)

    arms = []
    for graph_arm in (ARM_RELATION_ONLY, ARM_ACTUAL, ARM_SHUFFLE):
        model_id = ARM_MODEL_IDS[graph_arm]
        print(
            f"\n===== {model_id} | seed {cfg.seed} | "
            f"fixed {cfg.epochs} epochs ====="
        )
        arms.append(
            _run_arm(
                prepared,
                cfg,
                arm=graph_arm,
                model_id=model_id,
            )
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

    stem = f"m3_clv_directional_first_hop_{prepared['config_hash']}"
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
    frame.attrs["result_paths"] = {key: str(value) for key, value in paths.items()}

    print("\nM3 CLV 귀속 판정:")
    print(json.dumps(reading, ensure_ascii=False, indent=2))
    print("결과 파일:", frame.attrs["result_paths"])
    return frame


if __name__ == "__main__":
    run_directional_first_hop_screen()

