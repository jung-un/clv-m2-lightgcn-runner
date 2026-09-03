"""Seed-42 M3 screen for a historical-CLV-gated taste-neighbor residual."""

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

from clv_m3_clv_conditioned_candidate_item_model import (
    build_binary_directional_blocks,
)
from clv_m3_tfidf_neighbor_graph import (
    build_historical_clv_gates,
    build_tfidf_neighbor_operator,
)
from clv_m3_tfidf_neighbor_residual_model import (
    TFIDFNeighborResidualLightGCN,
)
from clv_run_state import ProgressStore, RunIdentity, clone_state, file_sha256
import lightgcn_clv_axis_specific_test10 as fixed_train
import lightgcn_clv_gatefree_lowdim as baseline_support
import lightgcn_clv_moe as moe
import lightgcn_clv_v3 as v3


CODE_VERSION = "m3-clv-tfidf-neighbor-residual-historical-screen-v1"
M1_ID = "m1_baseline"
RELATION_ID = "m3_tfidf_neighbor_residual_constant_gate_control"
ACTUAL_ID = "m3_clv_tfidf_neighbor_residual"
SHUFFLE_ID = "m3_clv_tfidf_neighbor_residual_shuffle"
DEGREE_ID = "m3_tfidf_neighbor_residual_degree_gate_control"
ARM_ORDER = (RELATION_ID, ACTUAL_ID, SHUFFLE_ID, DEGREE_ID)
ACCURACY_METRICS = (
    "recall@10",
    "ndcg@10",
    "recall@20",
    "ndcg@20",
    "recall@50",
    "ndcg@50",
)


@dataclass(frozen=True)
class TFIDFNeighborResidualConfig:
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
    top_k_neighbors: int = 20
    degree_bins: int = 10
    shuffle_seed: int = 42
    rho: float = 0.075
    budget_warning_relative_difference: float = 0.10
    out_dir: str = ""
    baseline_result_dir: str = ""


def configure_tfidf_neighbor_residual_run(
    **overrides,
) -> TFIDFNeighborResidualConfig:
    defaults = {
        "out_dir": (
            f"{v3.default_out_dir('dunnhumby')}"
            "_m3_clv_tfidf_neighbor_residual_historical_screen_v1"
        ),
        "baseline_result_dir": (
            f"{v3.default_out_dir('dunnhumby')}"
            "_m2_repeatshare_historical_backtest_v1"
        ),
    }
    return validate_tfidf_neighbor_residual_config(
        TFIDFNeighborResidualConfig(**(defaults | overrides))
    )


def validate_tfidf_neighbor_residual_config(
    cfg: TFIDFNeighborResidualConfig,
) -> TFIDFNeighborResidualConfig:
    fixed = {
        "dataset": "dunnhumby",
        "seed": 42,
        "time_cutoff": 690,
        "evaluation_days": 7,
        "epochs": 100,
        "id_dim": 64,
        "n_layers": 2,
        "top_k_neighbors": 20,
        "degree_bins": 10,
        "shuffle_seed": 42,
        "rho": 0.075,
        "budget_warning_relative_difference": 0.10,
    }
    for name, expected in fixed.items():
        if getattr(cfg, name) != expected:
            raise ValueError(f"빠른 M3 screen은 {name}={expected!r}이어야 합니다")
    if cfg.batch_size <= 0 or cfg.lr <= 0 or cfg.pref_reg < 0:
        raise ValueError("학습 설정이 잘못됐습니다")
    if not cfg.out_dir or not cfg.baseline_result_dir:
        raise ValueError("out_dir와 baseline_result_dir가 필요합니다")
    return cfg


def preflight_summary(cfg: TFIDFNeighborResidualConfig) -> dict:
    cfg = validate_tfidf_neighbor_residual_config(cfg)
    return {
        "code_version": CODE_VERSION,
        "dataset": cfg.dataset,
        "seed": cfg.seed,
        "all_models": [M1_ID, *ARM_ORDER],
        "trained_models": list(ARM_ORDER),
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
            "taste_relation": (
                "binary-purchase TF-IDF cosine Top-20 user neighbors"
            ),
            "neighbor_message": "row-normalized neighbor mean of M1 user layer-1",
            "residual": (
                "component orthogonal to each current M1 user representation; "
                "scaled by ||z_M1||/||neighbor_message||"
            ),
            "actual_gate": "historical CLV percentile",
            "rho": cfg.rho,
            "item_representation_changed": False,
            "external_reranking": False,
        },
        "controls": {
            RELATION_ID: "constant mean historical-CLV gate",
            SHUFFLE_ID: "historical CLV permuted within binary-degree deciles",
            DEGREE_ID: "binary user-degree percentile gate",
        },
        "fixed": {
            "binary_m1_graph_preserved": True,
            "negative_sampling": "uniform",
            "sample_weighting": False,
            "loss": "one plain pairwise BPR plus existing sampled ID L2",
            "new_loss_term": False,
            "one_training_loop_and_optimizer": True,
            "pretraining_or_freezing": False,
            "min_user_interactions": 1,
            "min_item_interactions": 1,
            "epochs": cfg.epochs,
            "validation_or_epoch_selection": False,
        },
        "training_gate": (
            "train-only five-anchor TF-IDF mechanism diagnostic must pass"
        ),
        "reading_rule": (
            "actual six-metric geometric balance > 1 against M1, constant-gate, "
            "degree-matched CLV shuffle, and degree-gate control"
        ),
        "budget_warning_rule": (
            "do not interpret as direction-only attribution when actual and shuffle "
            "eligible effective budgets differ by more than 10%"
        ),
        "statistical_note": (
            "single-seed exploratory historical screen; no significance or "
            "generalization claim"
        ),
        "out_dir": cfg.out_dir,
        "baseline_result_dir": cfg.baseline_result_dir,
    }


def _base_config(cfg: TFIDFNeighborResidualConfig) -> dict:
    base = dict(
        v3.configure_run(
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
    )
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
    for name, expected in required.items():
        if base[name] != expected:
            raise RuntimeError(f"M3 historical screen 설정 오염: {name}={base[name]!r}")
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


def _scipy_to_torch(matrix, device: torch.device) -> torch.Tensor:
    coo = matrix.tocoo()
    indices = torch.as_tensor(
        np.stack([coo.row, coo.col]), dtype=torch.long, device=device
    )
    values = torch.as_tensor(coo.data, dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(
        indices, values, size=coo.shape, device=device
    ).coalesce()


def _prepare(cfg: TFIDFNeighborResidualConfig) -> dict:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = moe.build_input_manifest(v3.SCHEMA[cfg.dataset])
    input_hash = moe.manifest_hash(manifest)
    revision = moe.source_revision()
    base_cfg = _base_config(cfg)
    data = v3.prepare_data(base_cfg, v3.DCFG)
    if set(data["splits"]) != {"test"}:
        raise RuntimeError("historical screen must construct exactly one test split")
    if float(data["train"]["t"].max()) != 683.0:
        raise RuntimeError("historical train boundary is not DAY 683")
    if data.get("loss_w") is not None:
        raise RuntimeError("M4 sample weighting entered the M3 screen")
    data["loss_w"] = None
    relation, relation_diagnostics = build_tfidf_neighbor_operator(
        data["train"],
        data["n_users"],
        data["n_items"],
        top_k=cfg.top_k_neighbors,
    )
    gates = build_historical_clv_gates(
        data["train"],
        data["n_users"],
        shuffle_degree_bins=cfg.degree_bins,
        shuffle_seed=cfg.shuffle_seed,
    )
    baseline = baseline_support._load_compatible_baseline(cfg, manifest)
    meta = v3.item_meta(data["train"], data["n_items"])
    thresholds = v3.segment_thresholds(gates.clv_proxy, base_cfg["SEG_EDGES"])
    cache = v3.EvalCache(
        *data["splits"]["test"], gates.clv_proxy, thresholds, data["n_items"]
    )
    prepared = {
        "out_dir": out_dir,
        "manifest": manifest,
        "input_hash": input_hash,
        "revision": revision,
        "base_cfg": base_cfg,
        "data": data,
        "relation": relation,
        "relation_diagnostics": relation_diagnostics,
        "gates": gates,
        "baseline": baseline,
        "meta": meta,
        "thresholds": thresholds,
        "cache": cache,
    }
    prepared["config_hash"] = _config_hash(cfg, input_hash, revision)
    return prepared


def _arm_gate(prepared: dict, model_id: str) -> np.ndarray:
    gates = prepared["gates"]
    if model_id == RELATION_ID:
        return gates.constant_gate
    if model_id == ACTUAL_ID:
        return gates.clv_percentile
    if model_id == SHUFFLE_ID:
        return gates.clv_shuffle_percentile
    if model_id == DEGREE_ID:
        return gates.degree_percentile
    raise ValueError(f"unknown M3 arm: {model_id}")


def _build_model(
    prepared: dict, cfg: TFIDFNeighborResidualConfig, model_id: str
) -> TFIDFNeighborResidualLightGCN:
    data = prepared["data"]
    v3.set_seed(cfg.seed)
    edge_key = np.asarray(data["pos_key"], dtype=np.int64)
    edge_users = edge_key // data["n_items"]
    edge_items = edge_key % data["n_items"]
    user_item, item_user = build_binary_directional_blocks(
        edge_users,
        edge_items,
        data["n_users"],
        data["n_items"],
        v3.DEVICE,
    )
    neighbor = _scipy_to_torch(prepared["relation"], v3.DEVICE)
    gate = torch.as_tensor(
        _arm_gate(prepared, model_id), dtype=torch.float32, device=v3.DEVICE
    )
    return TFIDFNeighborResidualLightGCN(
        n_users=data["n_users"],
        n_items=data["n_items"],
        base_user_from_item=user_item,
        base_item_from_user=item_user,
        user_neighbor_operator=neighbor,
        gate=gate,
        rho=cfg.rho,
        dim=cfg.id_dim,
        n_layers=cfg.n_layers,
        pref_reg=cfg.pref_reg,
    ).to(v3.DEVICE)


def _arm_paths(prepared: dict, model_id: str) -> dict[str, Path]:
    root = prepared["out_dir"] / "arms" / prepared["config_hash"]
    stem = f"{model_id}_s42"
    return {"result": root / f"{stem}.json", "checkpoint": root / f"{stem}.pt"}


def _arm_hash(prepared: dict, model_id: str) -> str:
    return hashlib.sha256(
        _canonical(
            {
                "run": prepared["config_hash"],
                "model_id": model_id,
                "seed": 42,
            }
        ).encode()
    ).hexdigest()[:12]


def _run_arm(
    prepared: dict, cfg: TFIDFNeighborResidualConfig, model_id: str
) -> dict:
    paths = _arm_paths(prepared, model_id)
    if paths["result"].exists() and paths["checkpoint"].exists():
        payload = json.loads(paths["result"].read_text(encoding="utf-8"))
        if payload.get("input_hash") != prepared["input_hash"]:
            raise RuntimeError("cached result and current input hashes differ")
        print(f"  [cached] {model_id} 완료 결과 재사용")
        return payload
    model = _build_model(prepared, cfg, model_id)
    initial = model.intervention_diagnostics()
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
    final_diagnostics = model.representation_diagnostics()
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
            "config": asdict(cfg),
            "initial_diagnostics": initial,
            "final_diagnostics": final_diagnostics,
            "training": training,
            "source_revision": prepared["revision"],
            "input_hash": prepared["input_hash"],
        },
        temporary,
    )
    os.replace(temporary, paths["checkpoint"])
    payload = {
        "model_id": model_id,
        "role": "model" if model_id == ACTUAL_ID else "control",
        "seed": cfg.seed,
        "split": "historical_development_days_684_690",
        "final_epoch": cfg.epochs,
        "input_hash": prepared["input_hash"],
        "metrics": fixed_train._public_metrics(metrics),
        "initial_diagnostics": initial,
        "model_diagnostics": final_diagnostics,
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
        raise ValueError(f"expected one row for {model_id}, got {len(selected)}")
    return selected.iloc[0]


def _six_metric_balance(actual: pd.Series, reference: pd.Series) -> float:
    ratios = np.asarray(
        [float(actual[name]) / float(reference[name]) for name in ACCURACY_METRICS]
    )
    if not np.isfinite(ratios).all() or np.any(ratios <= 0):
        raise ValueError("all six accuracy metrics must be finite and positive")
    return float(np.exp(np.log(ratios).mean()))


def attribution_reading(frame: pd.DataFrame) -> dict:
    actual = _row(frame, ACTUAL_ID)
    references = {
        "m1": _row(frame, M1_ID),
        "relation_constant": _row(frame, RELATION_ID),
        "shuffle": _row(frame, SHUFFLE_ID),
        "degree_gate": _row(frame, DEGREE_ID),
    }
    balances = {
        name: _six_metric_balance(actual, reference)
        for name, reference in references.items()
    }
    actual_budget = float(actual["effective_budget_eligible"])
    shuffle_budget = float(references["shuffle"]["effective_budget_eligible"])
    budget_relative_difference = abs(actual_budget - shuffle_budget) / max(
        abs(shuffle_budget), 1e-12
    )
    weighted = "price_purchase_amount_weighted_hit@10"
    price = "mean_recommended_price_percentile@10"
    return {
        "clv_attribution_supported": bool(
            all(value > 1.0 for value in balances.values())
        ),
        "primary_rule": (
            "actual six-metric geometric balance > 1 against M1, constant-gate, "
            "degree-matched CLV shuffle, and degree-gate control"
        ),
        **{
            f"six_metric_balance_actual_vs_{name}": value
            for name, value in balances.items()
        },
        "weighted_hit@10_deltas": {
            name: float(actual[weighted]) - float(reference[weighted])
            for name, reference in references.items()
        },
        "recommended_price_percentile@10_deltas": {
            name: float(actual[price]) - float(reference[price])
            for name, reference in references.items()
        },
        "effective_budget_eligible": {
            "actual": actual_budget,
            "shuffle": shuffle_budget,
            "relative_difference": budget_relative_difference,
        },
        "budget_direction_only_warning": bool(budget_relative_difference > 0.10),
        "single_seed_limitation": (
            "no variance, interval, significance, or generalization claim"
        ),
    }


def _comparison(frame: pd.DataFrame) -> pd.DataFrame:
    actual = _row(frame, ACTUAL_ID)
    rows = []
    for reference_id in (M1_ID, RELATION_ID, SHUFFLE_ID, DEGREE_ID):
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
                        float(100 * (left - right) / right) if right != 0 else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def run_tfidf_neighbor_residual_screen(
    cfg: TFIDFNeighborResidualConfig | None = None,
    *,
    mechanism_reading: dict | None = None,
) -> pd.DataFrame:
    cfg = validate_tfidf_neighbor_residual_config(
        cfg or configure_tfidf_neighbor_residual_run()
    )
    if not mechanism_reading or mechanism_reading.get("precheck_passed") is not True:
        raise RuntimeError(
            "train-only TF-IDF 관계 진단이 통과되지 않아 고비용 M3 학습을 중단합니다"
        )
    preflight = preflight_summary(cfg)
    print(json.dumps(preflight, ensure_ascii=False, indent=2))
    prepared = _prepare(cfg)
    arms = []
    for model_id in ARM_ORDER:
        print(f"\n===== {model_id} | seed 42 | fixed 100 epochs =====")
        arms.append(_run_arm(prepared, cfg, model_id))

    source_baseline = dict(prepared["baseline"])
    rows = [
        {
            **source_baseline,
            "model_id": M1_ID,
            "role": "reused_baseline",
            "effective_budget_all": 0.0,
            "effective_budget_eligible": 0.0,
        }
    ]
    for arm in arms:
        rows.append(
            {
                "model_id": arm["model_id"],
                "role": arm["role"],
                "seed": arm["seed"],
                "split": arm["split"],
                "final_epoch": arm["final_epoch"],
                **arm["model_diagnostics"],
                **arm["metrics"],
            }
        )
    frame = pd.DataFrame(rows)
    comparison = _comparison(frame)
    reading = attribution_reading(frame)
    stem = f"m3_clv_tfidf_neighbor_residual_{prepared['config_hash']}"
    paths = {
        "absolute_csv": prepared["out_dir"] / f"{stem}.csv",
        "comparison_csv": prepared["out_dir"] / f"{stem}_comparison.csv",
        "json": prepared["out_dir"] / f"{stem}.json",
    }
    fixed_train._atomic_csv(paths["absolute_csv"], frame)
    fixed_train._atomic_csv(paths["comparison_csv"], comparison)
    fixed_train._atomic_json(
        paths["json"],
        {
            "code_version": CODE_VERSION,
            "source_revision": prepared["revision"],
            "config": asdict(cfg),
            "preflight": preflight,
            "mechanism_reading": mechanism_reading,
            "input_manifest": prepared["manifest"],
            "reused_baseline_source": source_baseline.get("source_result"),
            "relation_diagnostics": prepared["relation_diagnostics"],
            "gate_diagnostics": prepared["gates"].diagnostics,
            "absolute_rows": frame.to_dict("records"),
            "comparison_rows": comparison.to_dict("records"),
            "attribution_reading": reading,
            "result_paths": {key: str(value) for key, value in paths.items()},
        },
    )
    frame.attrs["attribution_reading"] = reading
    frame.attrs["comparison"] = comparison.to_dict("records")
    frame.attrs["result_paths"] = {key: str(value) for key, value in paths.items()}
    print("\nM3 CLV 귀속 판정:")
    print(json.dumps(reading, ensure_ascii=False, indent=2))
    print("결과 파일:", frame.attrs["result_paths"])
    return frame


if __name__ == "__main__":
    raise SystemExit("Run the train-only mechanism diagnostic before training.")
