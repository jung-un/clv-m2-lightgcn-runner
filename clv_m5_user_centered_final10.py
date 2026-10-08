"""Frozen Dunnhumby last-week evaluation of M1, M3 and attributed M5-C.

The final window is DAY 705--711 and every earlier row is training data.  The
observed M5-C and its degree-matched joint N/V/q_C permutation are trained from
scratch under the same seed.  Test metrics are computed exactly once, after the
fixed epoch 300 checkpoint.  No validation, early stopping, checkpoint
selection, external score addition or reranking is used.
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
CODE_VERSION = "clv-m5-user-centered-final10-v1"
SEEDS = tuple(range(42, 52))
M1 = "m1_bpr_k1_lastweek_final10"
M3 = "m3_centered_nv_bpr_k1_lastweek_final10"
M5_C = "m5_m3nv_user_centered_m4_bpr_k1_lastweek_final10"
M5_C_SHUFFLED = "m5_m3nv_user_centered_m4_tuple_shuffle_bpr_k1_lastweek_final10"
ACCURACY = ("recall@10", "ndcg@10", "recall@20", "ndcg@20", "recall@50", "ndcg@50")
ECONOMIC = ("price_purchase_amount_weighted_hit@10", "vndcg@10")
NUMERIC_TOLERANCE = 1e-8
SHUFFLE_SEED_BASE = 20261008


@dataclass(frozen=True)
class Config:
    dataset: str = "dunnhumby"
    seeds: tuple[int, ...] = SEEDS
    epochs: int = 300
    id_dim: int = 64
    n_layers: int = 2
    batch_size: int = 8192
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
    defaults = {"out_dir": v3.default_out_dir("dunnhumby") + "_m5c_lastweek_final10_v1"}
    cfg = Config(**(defaults | overrides))
    if cfg.dataset != "dunnhumby" or cfg.seeds != SEEDS or cfg.epochs != 300:
        raise ValueError("Dunnhumby 마지막1주·seed42~51·고정300epoch만 허용합니다")
    if (cfg.id_dim, cfg.n_layers, cfg.batch_size, cfg.negative_count) != (64, 2, 8192, 1):
        raise ValueError("사전 고정한 용량·배치·uniform K=1 설정과 다릅니다")
    if not cfg.out_dir:
        raise ValueError("out_dir가 필요합니다")
    return cfg


def models_for(cfg: Config) -> tuple[str, ...]:
    configure(**asdict(cfg))
    return M1, M3, M5_C, M5_C_SHUFFLED


def base_config(cfg: Config) -> dict:
    return dict(v3.configure_run(
        "dunnhumby", out_dir=cfg.out_dir, ARCH="pref_only", SEED_LIST=list(cfg.seeds),
        WINDOW_DAYS=None, TIME_CUTOFF=None, VAL_DAYS=0, TEST_DAYS=7, HOLDOUT_DAYS=0,
        TRAIN_ON_VAL=True, EVAL_TEST=True, EVAL_HOLDOUT=False,
        GRAPH_MODE="binary", LOSS_MODE="plain", NEG_MODE="uniform",
        MIN_USER_INTER=1, MIN_ITEM_INTER=1, DIM=cfg.id_dim, N_LAYERS=cfg.n_layers,
        BATCH_SIZE=cfg.batch_size, LR=cfg.lr, PREF_REG=cfg.pref_reg,
        EPOCHS=cfg.epochs, EARLY_STOP=cfg.epochs, REPORT_LEGACY_VALUE_FEATURES=False,
    ))


def preflight_summary(cfg: Config) -> dict:
    configure(**asdict(cfg))
    return {
        "code_version": CODE_VERSION,
        "question": "Does observed-assignment M5-C beat M1, M3 and its degree-matched CLV null?",
        "split": "train DAY 1-704; test DAY 705-711",
        "models": list(models_for(cfg)),
        "seeds": list(cfg.seeds),
        "fixed_epoch": cfg.epochs,
        "m3": "exp(beta*[q_V*z_V+q_N*z_N]); beta calibrated to edge-weight CV=.20",
        "m4_c": "exp(.5*q_C*(amount*fit-customer mean)); valid-row customer mean one",
        "control": "joint q_N/q_V/q_C/valid permutation within binary-degree deciles",
        "decision": {
            "accuracy": "M5-C mean of all Recall/NDCG@10/20/50 >=99% of both M1 and M3",
            "economics": "both economic@10 means exceed M1 and M3",
            "attribution": f"both observed-shuffle mean deltas > {NUMERIC_TOLERANCE}",
        },
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
        "statistical_note": "mean/std and seed-wise paired differences; no significance claim",
    }


def prepare(cfg: Config) -> dict:
    cfg = configure(**asdict(cfg))
    manifest = fixed.moe.build_input_manifest(v3.SCHEMA["dunnhumby"])
    input_hash, revision = fixed.moe.manifest_hash(manifest), fixed.moe.source_revision()
    base = base_config(cfg)
    data = v3.prepare_data(base, v3.DCFG)
    lastweek.validate_split(data, lastweek.Config(dataset="dunnhumby", seeds=cfg.seeds,
                                                  epochs=cfg.epochs, out_dir=cfg.out_dir))
    data["loss_w"] = None
    snapshot = fixed.residual.build_final_snapshot(data["train"], data["n_users"],
                                                   v3.DCFG["is_date"], cfg.input_days)
    axes = fixed.joint.build_user_axis_inputs(snapshot, data["n_users"])
    q_n, q_v, q_c, valid = fixed.evaluation.build_clv_inputs(axes)
    q_n, q_v, q_c = (np.where(valid, q, 0).astype(np.float32) for q in (q_n, q_v, q_c))
    econ = economics.build_nv_economic_inputs(
        data["train"], n_users=data["n_users"], n_items=data["n_items"],
        q_n=q_n, q_v=q_v, q_c=q_c, clv_valid=valid, n_bins=4,
        shrinkage_strength=10.0, degree_bins=10,
    )
    signals = fixed.graph.centered_edge_signals(data["train"], data["n_users"], data["n_items"])
    if not np.array_equal(signals["edge_users"] * data["n_items"] + signals["edge_items"], data["pos_key"]):
        raise RuntimeError("M3 edge order differs from the binary graph")
    thresholds = v3.segment_thresholds(axes["clv_proxy"], base["SEG_EDGES"])
    prepared = {
        "out_dir": Path(cfg.out_dir), "manifest": manifest, "input_hash": input_hash,
        "revision": revision, "base_cfg": base, "data": data, "axes": axes,
        "q_n": q_n, "q_v": q_v, "q_c": q_c, "clv_valid": np.asarray(valid, bool),
        "signals": signals, "q_value": q_v.astype(float), "q_activity": q_n.astype(float),
        "meta": v3.item_meta(data["train"], data["n_items"]),
        "cache": v3.EvalCache(*data["splits"]["test"], axes["clv_proxy"], thresholds, data["n_items"]),
    }
    prepared.update({key: econ[key] for key in (
        "item_amount_percentile", "item_economic_valid", "user_economic_valid",
        "user_bin_fit", "item_bin",
    )})
    observed_graph = fixed.build_graph(prepared, cfg)
    observed_weights, observed_weight_audit = m4c.row_weights(prepared)
    observed_adj = v3.build_adj(signals["edge_users"], signals["edge_items"],
                                observed_graph["weights"].astype(np.float32),
                                data["n_users"], data["n_items"])
    identity = {
        "code_version": CODE_VERSION, "config": asdict(cfg), "input_hash": input_hash,
        "revision": revision, "m3_beta": observed_graph["beta"],
        "m4_weight_sha256": observed_weight_audit["sha256"],
    }
    config_hash = hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest()[:12]
    run_dir = Path(cfg.out_dir) / config_hash
    protocol = preflight_summary(cfg) | {
        "config": asdict(cfg), "input_hash": input_hash, "source_revision": revision,
        "m3_audit": observed_graph["audit"], "m4_c_weight_audit": observed_weight_audit,
        "data_stats": data["data_stats"],
    }
    io._atomic_json(run_dir / "protocol.json", protocol)
    prepared.update(config_hash=config_hash, run_dir=run_dir, protocol=protocol,
                    observed_graph=observed_graph, observed_adj=observed_adj,
                    observed_weights=observed_weights,
                    observed_weight_audit=observed_weight_audit)
    return prepared


def _arm_inputs(prepared: dict, cfg: Config, model_id: str, seed: int) -> dict:
    if model_id in (M1, M3, M5_C):
        return {
            "prepared": prepared,
            "adjacency": prepared["data"]["adj"] if model_id == M1 else prepared["observed_adj"],
            "row_weights": prepared["observed_weights"] if model_id == M5_C else None,
            "graph_audit": None if model_id == M1 else prepared["observed_graph"]["audit"],
            "weight_audit": prepared["observed_weight_audit"] if model_id == M5_C else None,
            "shuffle_audit": None,
        }
    if model_id != M5_C_SHUFFLED:
        raise ValueError(model_id)
    assignment, shuffle_audit = attribution.degree_matched_tuple_shuffle(
        prepared, seed=SHUFFLE_SEED_BASE + seed,
    )
    shuffled = attribution._prepared_with_assignment(prepared, assignment)
    graph = fixed.build_graph(shuffled, cfg)
    weights, weight_audit = m4c.row_weights(shuffled)
    adjacency = v3.build_adj(shuffled["signals"]["edge_users"], shuffled["signals"]["edge_items"],
                             graph["weights"].astype(np.float32),
                             shuffled["data"]["n_users"], shuffled["data"]["n_items"])
    return {"prepared": shuffled, "adjacency": adjacency, "row_weights": weights,
            "graph_audit": graph["audit"], "weight_audit": weight_audit,
            "shuffle_audit": shuffle_audit}


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


def run_arm(prepared: dict, cfg: Config, model_id: str, seed: int) -> dict:
    path = prepared["run_dir"] / "arms" / f"{model_id}_s{seed}.json"
    identity = RunIdentity(CODE_VERSION, model_id, seed, prepared["config_hash"],
                           prepared["revision"], prepared["input_hash"])
    if path.exists():
        row = json.loads(path.read_text())
        if row.get("identity") != asdict(identity):
            raise RuntimeError("completed-arm identity mismatch")
        return row
    arm = _arm_inputs(prepared, cfg, model_id, seed)
    model = _build_model(arm, cfg, seed)
    store = ProgressStore(prepared["run_dir"] / "progress", identity)
    history = fixed._train(model, arm["prepared"], cfg, model_id, seed, store,
                           row_weights=arm["row_weights"])
    if len(history) != cfg.epochs or history[-1]["epoch"] != cfg.epochs:
        raise RuntimeError("fixed epoch training incomplete; test was not evaluated")
    metrics = fixed._evaluate(model, arm["prepared"])
    if not np.isfinite(list(metrics.values())).all():
        raise RuntimeError("non-finite final metric")
    row = {
        "identity": asdict(identity), "model_id": model_id, "seed": seed,
        "epochs": cfg.epochs, "metrics": metrics, "training_history": history,
        "diagnostics": {
            "graph": "binary" if model_id == M1 else "centered_nv",
            "row_weighted": arm["row_weights"] is not None,
            "graph_audit": arm["graph_audit"], "weight_audit": arm["weight_audit"],
            "shuffle_audit": arm["shuffle_audit"],
        },
    }
    io._atomic_json(path, row)
    store.mark_complete(epoch=cfg.epochs, max_epoch=cfg.epochs, selection="none",
                        checkpoint_path=str(store.latest_checkpoint), result_path=str(path))
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return row


def _validate_absolute(absolute: pd.DataFrame, cfg: Config) -> list[str]:
    expected = {(model, seed) for model in models_for(cfg) for seed in cfg.seeds}
    found = set(zip(absolute.model_id, absolute.seed))
    if len(absolute) != len(expected) or found != expected or absolute.duplicated(["model_id", "seed"]).any():
        raise RuntimeError("all forty final arms are required before reading")
    metrics = [column for column in absolute.columns if "@" in column]
    if not metrics or not np.isfinite(absolute[metrics].to_numpy(float)).all():
        raise RuntimeError("missing or non-finite metrics")
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
        "note": "the last week was exposed during earlier research; report all adverse metrics",
    }


def _comparison(absolute: pd.DataFrame, cfg: Config, metrics: list[str]) -> pd.DataFrame:
    rows = []
    index = absolute.set_index(["model_id", "seed"])
    for model, reference in ((M3, M1), (M5_C, M1), (M5_C, M3),
                             (M5_C, M5_C_SHUFFLED), (M5_C_SHUFFLED, M1)):
        for metric in metrics:
            delta = []
            for seed in cfg.seeds:
                value, base = float(index.at[(model, seed), metric]), float(index.at[(reference, seed), metric])
                delta.append(value - base)
                rows.append({"model_id": model, "reference": reference, "metric": metric,
                             "seed": seed, "value": value, "reference_value": base,
                             "delta": value - base})
    return pd.DataFrame(rows)


def report(prepared: dict, cfg: Config, absolute: pd.DataFrame,
           diagnostics: pd.DataFrame) -> dict:
    metrics = _validate_absolute(absolute, cfg)
    long = absolute.melt(id_vars=["model_id", "seed"], value_vars=metrics,
                         var_name="metric", value_name="value")
    summary = long.groupby(["model_id", "metric"]).value.agg(["mean", "std", "count"]).reset_index()
    comparison = _comparison(absolute, cfg, metrics)
    decision = reading(absolute, cfg)
    tables = {"absolute": absolute, "summary": summary, "comparison": comparison,
              "diagnostics": diagnostics}
    root = Path(prepared["run_dir"]) / "reports"
    paths = {key: str(root / f"{key}.csv") for key in tables}
    for key, table in tables.items():
        io._atomic_csv(Path(paths[key]), table)
    paths["json"] = str(root / "result.json")
    io._atomic_json(Path(paths["json"]), {
        "protocol": prepared["protocol"], "reading": decision,
        "result_paths": paths,
    })
    return {**tables, "reading": decision, "paths": paths}


def run(cfg: Config | None = None, prepared: dict | None = None) -> dict:
    cfg = configure() if cfg is None else configure(**asdict(cfg))
    prepared = prepare(cfg) if prepared is None else prepared
    rows = []
    for seed in cfg.seeds:
        for model_id in models_for(cfg):
            print(f"[{len(rows)+1}/40] seed{seed} {model_id}", flush=True)
            rows.append(run_arm(prepared, cfg, model_id, seed))
    absolute = pd.DataFrame({"model_id": row["model_id"], "seed": row["seed"],
                             **row["metrics"]} for row in rows)
    diagnostics = pd.DataFrame({"model_id": row["model_id"], "seed": row["seed"],
                                **row["diagnostics"]} for row in rows)
    return report(prepared, cfg, absolute, diagnostics)


if __name__ == "__main__":
    print(json.dumps(preflight_summary(configure()), ensure_ascii=False, indent=2))
