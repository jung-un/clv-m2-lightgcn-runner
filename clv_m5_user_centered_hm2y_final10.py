"""Frozen H&M last-week evaluation of M1, M3, M4-C and M5-C.

The protocol plans seeds 42--51 on H&M's final week (2020-09-16--22),
using every earlier transaction for training.  Computation may be split into
batches, but the model formula, seeds, epoch budget and final decision remain
fixed.  Partial batches are operational readouts only and never create a final
success decision.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import clv_m2_m5_lastweek_test10 as lastweek
import clv_m5_m3_m4_user_centered_attribution as attribution
import clv_m5_m3_m4_user_centered_screen as m4c
import lightgcn_clv_m3_dunnhumby_test1 as fixed
import lightgcn_clv_m3_centered_value_graph_hm2y as hm_graph
import lightgcn_clv_m5_nv_economic_positive_weight as economics
from clv_run_state import ProgressStore, RunIdentity


v3, io = fixed.v3, fixed.test10
CODE_VERSION = "clv-m5-user-centered-hm2y-final10-v1"
SEEDS = tuple(range(42, 52))
INITIAL_SEEDS = (49, 50)

M1 = "m1_bpr_k1_hm2y_lastweek_final10"
M3 = "m3_centered_nv_bpr_k1_hm2y_lastweek_final10"
M4_C = "m4_user_centered_bpr_k1_hm2y_lastweek_final10"
M5_C = "m5_m3nv_user_centered_m4_bpr_k1_hm2y_lastweek_final10"
M5_C_SHUFFLED = "m5_m3nv_user_centered_m4_tuple_shuffle_bpr_k1_hm2y_lastweek_final10"
CORE_MODELS = (M1, M3, M5_C)
ALL_MODELS = (M1, M3, M4_C, M5_C, M5_C_SHUFFLED)

ACCURACY = ("recall@10", "ndcg@10", "recall@20", "ndcg@20", "recall@50", "ndcg@50")
ECONOMIC = ("price_purchase_amount_weighted_hit@10", "vndcg@10")
NUMERIC_TOLERANCE = 1e-8
SHUFFLE_SEED_BASE = 20261010


@dataclass(frozen=True)
class Config:
    dataset: str = "hm"
    seeds: tuple[int, ...] = SEEDS
    epochs: int = 300
    id_dim: int = 64
    n_layers: int = 2
    batch_size: int = 131072
    lr: float = 5e-4
    pref_reg: float = 1e-3
    negative_count: int = 1
    input_days: int = 365
    target_cv: float = .20
    max_degree_correlation: float = .05
    max_price_correlation: float = .20
    max_popularity_correlation: float = .20
    out_dir: str = ""


def configure(**overrides) -> Config:
    defaults = {"out_dir": v3.default_out_dir("hm") + "_m5c_lastweek_final10_v1"}
    cfg = Config(**(defaults | overrides))
    if cfg.dataset != "hm" or cfg.seeds != SEEDS or cfg.epochs != 300:
        raise ValueError("H&M 마지막1주·seed42~51·고정300epoch만 허용합니다")
    if (cfg.id_dim, cfg.n_layers, cfg.batch_size, cfg.negative_count) != (64, 2, 131072, 1):
        raise ValueError("사전 고정한 용량·배치·uniform K=1 설정과 다릅니다")
    if not cfg.out_dir:
        raise ValueError("out_dir가 필요합니다")
    return cfg


def models_for(cfg: Config) -> tuple[str, ...]:
    configure(**asdict(cfg))
    return ALL_MODELS


def base_config(cfg: Config) -> dict:
    return dict(v3.configure_run(
        "hm", out_dir=cfg.out_dir, ARCH="pref_only", SEED_LIST=list(cfg.seeds),
        WINDOW_DAYS=None, TIME_CUTOFF=None, VAL_DAYS=0, TEST_DAYS=7, HOLDOUT_DAYS=0,
        TRAIN_ON_VAL=True, EVAL_TEST=True, EVAL_HOLDOUT=False,
        GRAPH_MODE="binary", LOSS_MODE="plain", NEG_MODE="uniform",
        MIN_USER_INTER=1, MIN_ITEM_INTER=1, DIM=cfg.id_dim, N_LAYERS=cfg.n_layers,
        BATCH_SIZE=cfg.batch_size, LR=cfg.lr, PREF_REG=cfg.pref_reg,
        EPOCHS=cfg.epochs, EARLY_STOP=cfg.epochs, REPORT_LEGACY_VALUE_FEATURES=False,
    ))


def validate_split(data: dict) -> None:
    stats = data["data_stats"]
    bounds = stats["split_boundaries"]
    timestamp = pd.Timestamp
    expected_start = timestamp("2018-09-20")
    expected_train_end = timestamp("2020-09-15")
    expected_test_end = timestamp("2020-09-22")
    if (set(data["splits"]) != {"test"}
            or stats["split_rows"]["val"] != 0
            or stats["split_rows"]["holdout"] != 0
            or timestamp(stats["source"]["time_min"]) != expected_start
            or timestamp(stats["source"]["time_max"]) != expected_test_end
            or timestamp(bounds["train"]["start_inclusive"]) != expected_start
            or timestamp(bounds["train"]["end_inclusive"]) != expected_train_end
            or timestamp(bounds["test"]["start_exclusive"]) != expected_train_end
            or timestamp(bounds["test"]["end_inclusive"]) != expected_test_end
            or timestamp(data["train"].t.max()) > expected_train_end
            or data.get("loss_w") is not None):
        raise RuntimeError("H&M 마지막7일 test/앞기간 전체학습/검증·holdout 없음 조건 불일치")


def preflight_summary(cfg: Config) -> dict:
    configure(**asdict(cfg))
    return {
        "code_version": CODE_VERSION,
        "question": "Does frozen M5-C improve H&M's final last-week recommendation over M1 and M3?",
        "split": "train 2018-09-20--2020-09-15; test 2020-09-16--22",
        "planned_models": list(ALL_MODELS),
        "planned_seeds": list(SEEDS),
        "initial_compute_batch": {"seeds": list(INITIAL_SEEDS), "models": list(CORE_MODELS)},
        "fixed_epoch": cfg.epochs,
        "m3": "exp(beta*[q_V*z_V+q_N*z_N]); beta calibrated to edge-weight CV=.20",
        "m4_c": "exp(.5*q_C*(amount*fit-customer mean)); valid-row customer mean one",
        "control": "joint q_N/q_V/q_C/valid permutation within binary-degree deciles",
        "fixed": {
            "new_item_truth_and_candidates_exclude_train_pairs": True,
            "min_item_interactions": 1,
            "validation": False,
            "holdout": False,
            "early_stopping": False,
            "test_evaluations_per_fit": 1,
            "one_optimizer_per_arm": True,
            "external_reranking": False,
        },
        "partial_batch_rule": (
            "partial seed batches are compute staging only; no success decision or retuning"
        ),
        "statistical_note": "final mean/std after all ten seeds; no significance claim",
    }


def prepare(cfg: Config) -> dict:
    cfg = configure(**asdict(cfg))
    manifest = fixed.moe.build_input_manifest(v3.SCHEMA["hm"])
    input_hash, revision = fixed.moe.manifest_hash(manifest), fixed.moe.source_revision()
    base = base_config(cfg)
    data = v3.prepare_data(base, v3.DCFG)
    validate_split(data)
    data["loss_w"] = None
    train = data["train"]
    snapshot = fixed.residual.build_final_snapshot(
        train, data["n_users"], v3.DCFG["is_date"], cfg.input_days,
    )
    axes = fixed.joint.build_user_axis_inputs(snapshot, data["n_users"])
    q_n, q_v, q_c, valid = fixed.evaluation.build_clv_inputs(axes)
    q_n, q_v, q_c = (np.where(valid, q, 0).astype(np.float32) for q in (q_n, q_v, q_c))
    econ = economics.build_nv_economic_inputs(
        train, n_users=data["n_users"], n_items=data["n_items"],
        q_n=q_n, q_v=q_v, q_c=q_c, clv_valid=valid, n_bins=4,
        shrinkage_strength=10.0, degree_bins=10,
    )
    keyed_train = hm_graph._purchase_keyed(train)
    signals = fixed.graph.centered_edge_signals(
        keyed_train, data["n_users"], data["n_items"],
    )
    edge_keys = signals["edge_users"] * data["n_items"] + signals["edge_items"]
    if not np.array_equal(edge_keys, data["pos_key"]):
        raise RuntimeError("H&M M3 edge order differs from the binary graph")
    thresholds = v3.segment_thresholds(axes["clv_proxy"], base["SEG_EDGES"])
    prepared = {
        "out_dir": Path(cfg.out_dir), "manifest": manifest, "input_hash": input_hash,
        "revision": revision, "base_cfg": base, "data": data, "axes": axes,
        "q_n": q_n, "q_v": q_v, "q_c": q_c, "clv_valid": np.asarray(valid, bool),
        "signals": signals, "q_value": q_v.astype(float), "q_activity": q_n.astype(float),
        "meta": v3.item_meta(train, data["n_items"]),
        "cache": v3.EvalCache(
            *data["splits"]["test"], axes["clv_proxy"], thresholds, data["n_items"],
        ),
    }
    prepared.update({key: econ[key] for key in (
        "item_amount_percentile", "item_economic_valid", "user_economic_valid",
        "user_bin_fit", "item_bin",
    )})
    observed_graph = fixed.build_graph(prepared, cfg)
    observed_weights, observed_weight_audit = m4c.row_weights(prepared)
    observed_adj = v3.build_adj(
        signals["edge_users"], signals["edge_items"],
        observed_graph["weights"].astype(np.float32), data["n_users"], data["n_items"],
    )
    identity = {
        "code_version": CODE_VERSION, "config": asdict(cfg), "input_hash": input_hash,
        "revision": revision, "m3_beta": observed_graph["beta"],
        "m4_weight_sha256": observed_weight_audit["sha256"],
    }
    config_hash = hashlib.sha256(
        json.dumps(identity, sort_keys=True, default=str).encode()
    ).hexdigest()[:12]
    run_dir = Path(cfg.out_dir) / config_hash
    protocol = preflight_summary(cfg) | {
        "config": asdict(cfg), "input_hash": input_hash, "source_revision": revision,
        "m3_audit": observed_graph["audit"],
        "m4_c_weight_audit": observed_weight_audit,
        "data_stats": data["data_stats"],
    }
    protocol_path = run_dir / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != json.loads(json.dumps(protocol)):
        raise RuntimeError("same H&M final run path contains a different protocol")
    io._atomic_json(protocol_path, protocol)
    prepared.update(
        config_hash=config_hash, run_dir=run_dir, protocol=protocol,
        observed_graph=observed_graph, observed_adj=observed_adj,
        observed_weights=observed_weights, observed_weight_audit=observed_weight_audit,
    )
    return prepared


def _shuffled_inputs(prepared: dict, cfg: Config, seed: int) -> dict:
    assignment, shuffle_audit = attribution.degree_matched_tuple_shuffle(
        prepared, seed=SHUFFLE_SEED_BASE + seed,
    )
    shuffled = attribution._prepared_with_assignment(prepared, assignment)
    graph = fixed.build_graph(shuffled, cfg)
    weights, weight_audit = m4c.row_weights(shuffled)
    adjacency = v3.build_adj(
        shuffled["signals"]["edge_users"], shuffled["signals"]["edge_items"],
        graph["weights"].astype(np.float32),
        shuffled["data"]["n_users"], shuffled["data"]["n_items"],
    )
    return {
        "prepared": shuffled, "adjacency": adjacency, "row_weights": weights,
        "graph_audit": graph["audit"], "weight_audit": weight_audit,
        "shuffle_audit": shuffle_audit,
    }


def _arm_inputs(prepared: dict, cfg: Config, model_id: str, seed: int) -> dict:
    if model_id == M1:
        return {
            "prepared": prepared, "adjacency": prepared["data"]["adj"],
            "row_weights": None, "graph_audit": None, "weight_audit": None,
            "shuffle_audit": None,
        }
    if model_id == M3:
        return {
            "prepared": prepared, "adjacency": prepared["observed_adj"],
            "row_weights": None, "graph_audit": prepared["observed_graph"]["audit"],
            "weight_audit": None, "shuffle_audit": None,
        }
    if model_id == M4_C:
        return {
            "prepared": prepared, "adjacency": prepared["data"]["adj"],
            "row_weights": prepared["observed_weights"], "graph_audit": None,
            "weight_audit": prepared["observed_weight_audit"], "shuffle_audit": None,
        }
    if model_id == M5_C:
        return {
            "prepared": prepared, "adjacency": prepared["observed_adj"],
            "row_weights": prepared["observed_weights"],
            "graph_audit": prepared["observed_graph"]["audit"],
            "weight_audit": prepared["observed_weight_audit"], "shuffle_audit": None,
        }
    if model_id == M5_C_SHUFFLED:
        return _shuffled_inputs(prepared, cfg, seed)
    raise ValueError(model_id)


def _build_model(arm: dict, cfg: Config, seed: int):
    prepared, data = arm["prepared"], arm["prepared"]["data"]
    v3.set_seed(seed)
    return fixed.M5NConditionedValueBasisLightGCN(
        n_users=data["n_users"], n_items=data["n_items"],
        user_q_n=prepared["q_n"], user_q_v=prepared["q_v"], user_q_c=prepared["q_c"],
        user_clv_valid=prepared["clv_valid"],
        item_price_percentile=prepared["item_amount_percentile"],
        item_price_valid=prepared["item_economic_valid"], adj=arm["adjacency"],
        id_dim=cfg.id_dim, rho=0.0, n_layers=cfg.n_layers, pref_reg=cfg.pref_reg,
        constant_gate=1.0, economic_propagation=False,
    ).to(v3.DEVICE)


def _arm_path(prepared: dict, model_id: str, seed: int) -> Path:
    return prepared["run_dir"] / "arms" / f"{model_id}_s{seed}.json"


def _load_completed_arm(prepared: dict, model_id: str, seed: int) -> dict | None:
    path = _arm_path(prepared, model_id, seed)
    if not path.exists():
        return None
    row = json.loads(path.read_text())
    identity = RunIdentity(
        CODE_VERSION, model_id, seed, prepared["config_hash"],
        prepared["revision"], prepared["input_hash"],
    )
    if row.get("identity") != asdict(identity) or row.get("epochs") != 300:
        raise RuntimeError("completed H&M final arm identity mismatch")
    return row


def run_arm(prepared: dict, cfg: Config, model_id: str, seed: int) -> dict:
    completed = _load_completed_arm(prepared, model_id, seed)
    if completed is not None:
        print(f"[reused] seed{seed} {model_id}", flush=True)
        return completed
    arm = _arm_inputs(prepared, cfg, model_id, seed)
    model = _build_model(arm, cfg, seed)
    identity = RunIdentity(
        CODE_VERSION, model_id, seed, prepared["config_hash"],
        prepared["revision"], prepared["input_hash"],
    )
    store = ProgressStore(prepared["run_dir"] / "progress", identity)
    history = fixed._train(
        model, arm["prepared"], cfg, model_id, seed, store,
        row_weights=arm["row_weights"],
    )
    if len(history) != cfg.epochs or history[-1]["epoch"] != cfg.epochs:
        raise RuntimeError("fixed epoch training incomplete; test was not evaluated")
    metrics = fixed._evaluate(model, arm["prepared"])
    if not np.isfinite(list(metrics.values())).all():
        raise RuntimeError("non-finite H&M final metric")
    row = {
        "identity": asdict(identity), "model_id": model_id, "seed": seed,
        "epochs": cfg.epochs, "metrics": metrics, "training_history": history,
        "diagnostics": {
            "graph": "binary" if model_id in (M1, M4_C) else "centered_nv",
            "row_weighted": arm["row_weights"] is not None,
            "graph_audit": arm["graph_audit"], "weight_audit": arm["weight_audit"],
            "shuffle_audit": arm["shuffle_audit"],
        },
    }
    path = _arm_path(prepared, model_id, seed)
    io._atomic_json(path, row)
    store.mark_complete(
        epoch=cfg.epochs, max_epoch=cfg.epochs, selection="none",
        checkpoint_path=str(store.latest_checkpoint), result_path=str(path),
    )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return row


def _validate_batch(seeds: tuple[int, ...], model_ids: tuple[str, ...]) -> None:
    if (not seeds or len(set(seeds)) != len(seeds) or not set(seeds).issubset(SEEDS)):
        raise ValueError("batch seeds must be a unique non-empty subset of 42--51")
    if (not model_ids or len(set(model_ids)) != len(model_ids)
            or not set(model_ids).issubset(ALL_MODELS)):
        raise ValueError("batch models must be a unique non-empty subset of the frozen five models")


def completed_rows(prepared: dict) -> list[dict]:
    rows = []
    for seed in SEEDS:
        for model_id in ALL_MODELS:
            row = _load_completed_arm(prepared, model_id, seed)
            if row is not None:
                rows.append(row)
    return rows


def _tables(rows: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    absolute = pd.DataFrame([
        {"model_id": row["model_id"], "seed": row["seed"], **row["metrics"]}
        for row in rows
    ])
    diagnostics = pd.DataFrame([
        {"model_id": row["model_id"], "seed": row["seed"], **row["diagnostics"]}
        for row in rows
    ])
    return absolute, diagnostics


def partial_status(rows: list[dict]) -> dict:
    completed = {(row["model_id"], int(row["seed"])) for row in rows}
    expected = {(model, seed) for seed in SEEDS for model in ALL_MODELS}
    initial_expected = {(model, seed) for seed in INITIAL_SEEDS for model in CORE_MODELS}
    return {
        "complete": completed == expected,
        "test_results_partial": completed != expected,
        "completed_arm_count": len(completed),
        "planned_arm_count": len(expected),
        "initial_batch_complete": initial_expected.issubset(completed),
        "missing": [f"seed{seed}:{model}" for model, seed in sorted(expected - completed, key=lambda x: (x[1], x[0]))],
        "final_decision_available": completed == expected,
        "settings_frozen": True,
        "retuning_from_partial_results_allowed": False,
        "significance_claim": False,
    }


def partial_report(prepared: dict, rows: list[dict]) -> dict:
    absolute, diagnostics = _tables(rows)
    status = partial_status(rows)
    root = prepared["run_dir"] / "partial_reports"
    paths = {
        "absolute": str(root / "partial_absolute.csv"),
        "diagnostics": str(root / "partial_diagnostics.csv"),
        "json": str(root / "partial_status.json"),
    }
    io._atomic_csv(Path(paths["absolute"]), absolute)
    io._atomic_csv(Path(paths["diagnostics"]), diagnostics)
    io._atomic_json(Path(paths["json"]), {
        "protocol": prepared["protocol"], "status": status, "result_paths": paths,
    })
    return {"absolute": absolute, "diagnostics": diagnostics, "status": status, "paths": paths}


def run_batch(prepared: dict, cfg: Config, *, seeds: tuple[int, ...],
              model_ids: tuple[str, ...]) -> dict:
    cfg = configure(**asdict(cfg))
    _validate_batch(tuple(seeds), tuple(model_ids))
    total = len(seeds) * len(model_ids)
    done = 0
    for seed in seeds:
        for model_id in model_ids:
            done += 1
            print(f"[{done}/{total}] H&M seed{seed} {model_id}", flush=True)
            run_arm(prepared, cfg, model_id, seed)
    return partial_report(prepared, completed_rows(prepared))


def _validate_absolute(absolute: pd.DataFrame, cfg: Config) -> list[str]:
    expected = {(model, seed) for seed in cfg.seeds for model in ALL_MODELS}
    found = set(zip(absolute.model_id, absolute.seed))
    if (len(absolute) != len(expected) or found != expected
            or absolute.duplicated(["model_id", "seed"]).any()):
        raise RuntimeError("all fifty H&M final arms are required before final reading")
    metrics = [column for column in absolute.columns if "@" in column]
    if not metrics or not np.isfinite(absolute[metrics].to_numpy(float)).all():
        raise RuntimeError("missing or non-finite H&M final metrics")
    return metrics


def reading(absolute: pd.DataFrame, cfg: Config) -> dict:
    _validate_absolute(absolute, cfg)
    means = absolute.groupby("model_id")[[*ACCURACY, *ECONOMIC]].mean()
    accuracy_guard = all(
        means.at[M5_C, metric] >= .99 * means.at[reference, metric]
        for reference in (M1, M3) for metric in ACCURACY
    )
    economics_up = all(
        means.at[M5_C, metric] > means.at[reference, metric] + NUMERIC_TOLERANCE
        for reference in (M1, M3) for metric in ECONOMIC
    )
    attribution_up = all(
        means.at[M5_C, metric] > means.at[M5_C_SHUFFLED, metric] + NUMERIC_TOLERANCE
        for metric in ECONOMIC
    )
    return {
        "complete": True, "seed_count": len(cfg.seeds), "final_ten_seed_report": True,
        "fixed_epoch": cfg.epochs, "numerical_tolerance": NUMERIC_TOLERANCE,
        "m5_c_accuracy_guard_99pct_vs_m1_and_m3_mean": bool(accuracy_guard),
        "m5_c_both_economic_at10_above_m1_and_m3_mean": bool(economics_up),
        "observed_assignment_beats_shuffle_on_both_economic_at10_mean": bool(attribution_up),
        "final_condition_met": bool(accuracy_guard and economics_up and attribution_up),
        "significance_claim": False, "unseen_confirmation_claim": False,
        "note": "all adverse metrics and seed-wise paired differences must be reported",
    }


def _comparison(absolute: pd.DataFrame, cfg: Config, metrics: list[str]) -> pd.DataFrame:
    rows = []
    index = absolute.set_index(["model_id", "seed"])
    pairs = (
        (M3, M1), (M4_C, M1), (M5_C, M1), (M5_C, M3),
        (M5_C, M4_C), (M5_C, M5_C_SHUFFLED), (M5_C_SHUFFLED, M1),
    )
    for model, reference in pairs:
        for metric in metrics:
            for seed in cfg.seeds:
                value = float(index.at[(model, seed), metric])
                base = float(index.at[(reference, seed), metric])
                rows.append({
                    "model_id": model, "reference": reference, "metric": metric,
                    "seed": seed, "value": value, "reference_value": base,
                    "delta": value - base,
                })
    return pd.DataFrame(rows)


def report(prepared: dict, cfg: Config, absolute: pd.DataFrame,
           diagnostics: pd.DataFrame) -> dict:
    metrics = _validate_absolute(absolute, cfg)
    long = absolute.melt(
        id_vars=["model_id", "seed"], value_vars=metrics,
        var_name="metric", value_name="value",
    )
    summary = long.groupby(["model_id", "metric"]).value.agg(["mean", "std", "count"]).reset_index()
    comparison = _comparison(absolute, cfg, metrics)
    decision = reading(absolute, cfg)
    tables = {
        "absolute": absolute, "summary": summary, "comparison": comparison,
        "diagnostics": diagnostics,
    }
    root = prepared["run_dir"] / "reports"
    paths = {key: str(root / f"{key}.csv") for key in tables}
    for key, table in tables.items():
        io._atomic_csv(Path(paths[key]), table)
    paths["json"] = str(root / "result.json")
    io._atomic_json(Path(paths["json"]), {
        "protocol": prepared["protocol"], "reading": decision, "result_paths": paths,
    })
    return {**tables, "reading": decision, "paths": paths}


def run_final(cfg: Config | None = None, prepared: dict | None = None) -> dict:
    cfg = configure() if cfg is None else configure(**asdict(cfg))
    prepared = prepare(cfg) if prepared is None else prepared
    for seed in cfg.seeds:
        for model_id in ALL_MODELS:
            run_arm(prepared, cfg, model_id, seed)
    rows = completed_rows(prepared)
    absolute, diagnostics = _tables(rows)
    return report(prepared, cfg, absolute, diagnostics)


if __name__ == "__main__":
    print(json.dumps(preflight_summary(configure()), ensure_ascii=False, indent=2))
