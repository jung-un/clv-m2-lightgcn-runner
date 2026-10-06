"""Dunnhumby test-only run of the centered value graph M3, seeds averaged.

The development backtest cut the data at DAY 690, so its "holdout" is the very
window the design was built on - there is no unseen Dunnhumby data inside it.
This runner leaves the backtest: the former training and validation intervals
are merged through DAY 697, each arm trains for exactly 300 epochs, and the
fixed DAY 698-704 test interval is scored once at the end. DAY 705-711 is never
touched.

Following the supervisor's instruction, nothing is selected on a validation
split and no epoch is chosen: the number of epochs is fixed in advance and the
result is the mean over seeds, reported with its spread and with how many seeds
moved each metric in the same direction.

The arm is the two-axis centered value graph - both CLV axes, the one that met
the pre-registered condition on the development split. Its strength beta is
calibrated on this training window the same way, and the same gates refuse to
train when the weights become a price, popularity or degree proxy.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from clv_m5_n_conditioned_value_basis_model import M5NConditionedValueBasisLightGCN
from clv_run_state import ProgressStore, RunIdentity
import lightgcn_clv_axis_specific_test10 as test10
import lightgcn_clv_component_recheck as recheck
import lightgcn_clv_gradient_isolated_economic_interaction as evaluation
import lightgcn_clv_joint_nv as joint
import lightgcn_clv_joint_response_embedding as shared
import lightgcn_clv_level_composition_price_test1 as test1
import lightgcn_clv_m3_centered_value_graph as graph
import lightgcn_clv_m4_clv_hard_negative as m4_helpers
import lightgcn_clv_moe as moe
import lightgcn_clv_residual as residual
import lightgcn_clv_v3 as v3


CODE_VERSION = "clv-m3-centered-value-graph-dunnhumby-test1-v1"
M1_MODEL_ID = "m1_bpr_k1_dunnhumby_test1"
M3_MODEL_ID = "m3_centered_value_activity_graph_bpr_k1_dunnhumby_test1"
SPLIT = "dunnhumby_test_day_698_704"
ECONOMIC_METRICS = ("price_purchase_amount_weighted_hit@10", "vndcg@10")
ACCURACY_METRICS = ("recall@10", "ndcg@10", "recall@20", "ndcg@20",
                    "recall@50", "ndcg@50")
ACCURACY_GUARD = 0.99


@dataclass(frozen=True)
class DunnhumbyTestConfig:
    dataset: str = "dunnhumby"
    seeds: tuple[int, ...] = (42,)
    epochs: int = 300
    id_dim: int = 64
    n_layers: int = 2
    batch_size: int = 8192
    lr: float = 5e-4
    pref_reg: float = 1e-3
    negative_count: int = 1
    input_days: int = 365
    target_cv: float = 0.20
    max_degree_correlation: float = 0.05
    max_price_correlation: float = 0.20
    max_popularity_correlation: float = 0.20
    out_dir: str = ""


def configure_dunnhumby_test1(**overrides) -> DunnhumbyTestConfig:
    defaults = {
        "out_dir": (
            f"{v3.default_out_dir('dunnhumby')}_clv_m3_centered_value_graph_test1_v1"
        )
    }
    return validate_config(DunnhumbyTestConfig(**(defaults | overrides)))


def validate_config(cfg: DunnhumbyTestConfig) -> DunnhumbyTestConfig:
    if cfg.dataset != "dunnhumby":
        raise ValueError("이 러너는 Dunnhumby 전용입니다")
    if cfg.negative_count != 1:
        raise ValueError("기준 손실은 원 LightGCN BPR(음성 1개)입니다")
    if cfg.epochs != 300:
        raise ValueError("개발분할에서 조건을 충족한 설정은 300 epoch입니다")
    if not cfg.seeds or len(set(cfg.seeds)) != len(cfg.seeds):
        raise ValueError(f"시드가 비었거나 중복입니다: {cfg.seeds}")
    if not cfg.out_dir:
        raise ValueError("out_dir가 필요합니다")
    return cfg


def preflight_summary(cfg: DunnhumbyTestConfig) -> dict:
    cfg = validate_config(cfg)
    return {
        "code_version": CODE_VERSION,
        "dataset": "dunnhumby",
        "split": SPLIT,
        "training_interval": "train and validation merged through DAY 697",
        "never_touched": "DAY 705-711",
        "backtest_cutoff_removed": True,
        "seeds": list(cfg.seeds),
        "epochs_fixed": cfg.epochs,
        "no_validation_selection": (
            "no epoch, arm or checkpoint is selected; the test interval is scored "
            "once after the last epoch"
        ),
        "arms": [M1_MODEL_ID, M3_MODEL_ID],
        "arm_definition": (
            "two-axis centered value graph: "
            "exp(beta * [q_V(u)*z_V(u,i) + q_N(u)*z_N(u,i)]), both margins averaged "
            "to one; beta calibrated to a weight coefficient of variation of "
            f"{cfg.target_cv} on this training window"
        ),
        "gates_before_training": {
            "coefficient_of_variation": [0.15, 0.35],
            "customer_weight_vs_degree": cfg.max_degree_correlation,
            "item_weight_vs_price": cfg.max_price_correlation,
            "item_weight_vs_popularity": cfg.max_popularity_correlation,
        },
        "primary_metrics": list(ECONOMIC_METRICS),
        "reading": (
            "mean over seeds for every metric, with the per-seed win count; the "
            "pre-registered condition is both primary metrics improved on the mean "
            f"and all six accuracy metrics at least {ACCURACY_GUARD} of M1 on the mean"
        ),
        "limits": (
            "one evaluation of one protected interval; no confidence interval and no "
            "significance claim. Reported whatever it shows."
        ),
    }


# --------------------------------------------------------------------------
# preparation
# --------------------------------------------------------------------------


def _base_config(cfg: DunnhumbyTestConfig, seed: int) -> dict:
    return dict(v3.configure_run(
        cfg.dataset, out_dir=cfg.out_dir, ARCH="pref_only", SEED_LIST=[seed],
        WINDOW_DAYS=None, TIME_CUTOFF=None, VAL_DAYS=7, TEST_DAYS=7, HOLDOUT_DAYS=7,
        TRAIN_ON_VAL=True, EVAL_TEST=True, EVAL_HOLDOUT=False,
        GRAPH_MODE="binary", LOSS_MODE="plain", NEG_MODE="uniform",
        MIN_USER_INTER=1, MIN_ITEM_INTER=1, DIM=cfg.id_dim, N_LAYERS=cfg.n_layers,
        BATCH_SIZE=cfg.batch_size, LR=cfg.lr, PREF_REG=cfg.pref_reg,
        EPOCHS=cfg.epochs, EARLY_STOP=cfg.epochs,
        REPORT_LEGACY_VALUE_FEATURES=False,
    ))


def _config_hash(cfg: DunnhumbyTestConfig, input_hash: str, revision: str) -> str:
    payload = {
        "code_version": CODE_VERSION,
        "protocol": {field: getattr(cfg, field) for field in asdict(cfg)
                     if field not in {"out_dir", "seeds"}},
        "input_hash": input_hash, "source_revision": revision,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()[:12]


def _prepare(cfg: DunnhumbyTestConfig, seed: int) -> dict:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = moe.build_input_manifest(v3.SCHEMA[cfg.dataset])
    input_hash = moe.manifest_hash(manifest)
    revision = moe.source_revision()
    base_cfg = _base_config(cfg, seed)
    data = v3.prepare_data(base_cfg, v3.DCFG)
    test1.validate_final_test_data(data)        # 분할 구조가 최종 test 형태인지
    data["loss_w"] = None

    snapshot = residual.build_final_snapshot(
        data["train"], data["n_users"], v3.DCFG["is_date"], cfg.input_days
    )
    axes = joint.build_user_axis_inputs(snapshot, data["n_users"])
    q_n, q_v, q_c, clv_valid = evaluation.build_clv_inputs(axes)
    _, item_valid = shared.build_item_economic_inputs(data["train"], data["n_items"])
    price_features, _ = v3.item_value_features(data["train"], data["n_items"], report=False)
    thresholds = v3.segment_thresholds(axes["clv_proxy"], base_cfg["SEG_EDGES"])

    signals = graph.centered_edge_signals(data["train"], data["n_users"], data["n_items"])
    if not np.array_equal(
        signals["edge_users"] * data["n_items"] + signals["edge_items"],
        np.asarray(data["pos_key"], np.int64),
    ):
        raise RuntimeError("엣지 순서가 M1 이진 그래프와 다릅니다")

    valid = np.asarray(clv_valid, bool)
    prepared = {
        "out_dir": out_dir, "manifest": manifest, "input_hash": input_hash,
        "revision": revision, "base_cfg": base_cfg, "data": data, "axes": axes,
        "q_n": np.asarray(q_n, np.float32), "q_v": np.asarray(q_v, np.float32),
        "q_c": np.asarray(q_c, np.float32), "clv_valid": valid,
        "item_amount_percentile": np.asarray(price_features, float)[:, 0],
        "item_economic_valid": item_valid,
        "meta": v3.item_meta(data["train"], data["n_items"]),
        "thresholds": thresholds,
        "cache": v3.EvalCache(
            *data["splits"]["test"], axes["clv_proxy"], thresholds, data["n_items"]
        ),
        "signals": signals,
        "q_value": np.where(valid, q_v, 0.0).astype(float),
        "q_activity": np.where(valid, q_n, 0.0).astype(float),
    }
    prepared["config_hash"] = _config_hash(cfg, input_hash, revision)
    return prepared


def build_graph(prepared: dict, cfg: DunnhumbyTestConfig) -> dict:
    """Calibrate beta on this training window and audit before anything trains."""

    beta = graph.calibrate_beta(
        prepared["signals"], prepared["q_value"], prepared["q_activity"],
        1.0, cfg.target_cv,
    )
    weights = graph.edge_weights(
        prepared["signals"], prepared["q_value"], prepared["q_activity"], 1.0, beta
    )
    audit = graph.audit_weights(weights, prepared["signals"], prepared)
    graph.check_gates(audit, cfg, M3_MODEL_ID)
    return {"beta": beta, "weights": weights, "audit": audit}


def _build_model(prepared: dict, cfg: DunnhumbyTestConfig, seed: int, adjacency):
    data = prepared["data"]
    v3.set_seed(seed)
    model = M5NConditionedValueBasisLightGCN(
        n_users=data["n_users"], n_items=data["n_items"],
        user_q_n=prepared["q_n"], user_q_v=prepared["q_v"], user_q_c=prepared["q_c"],
        user_clv_valid=prepared["clv_valid"],
        item_price_percentile=prepared["item_amount_percentile"],
        item_price_valid=prepared["item_economic_valid"],
        adj=adjacency, id_dim=cfg.id_dim, rho=0.0, n_layers=cfg.n_layers,
        pref_reg=cfg.pref_reg, economic_propagation=False,
    )
    return model.to(v3.DEVICE)


# --------------------------------------------------------------------------
# training and the single evaluation
# --------------------------------------------------------------------------


def _train(model, prepared: dict, cfg: DunnhumbyTestConfig, model_id: str,
           seed: int, store: ProgressStore, row_weights=None) -> list[dict]:
    """Train to a fixed number of epochs. The test interval is not scored here."""

    data = prepared["data"]
    tr_u, tr_i, positive_keys = data["tr_u"], data["tr_i"], data["pos_key"]
    if row_weights is not None:
        row_weights = np.asarray(row_weights)
        if (row_weights.shape != (len(tr_u),) or not np.isfinite(row_weights).all()
                or np.any(row_weights <= 0)):
            raise ValueError("BPR 행 가중치 길이/유한성/양수 조건 불일치")
    n_batches = math.ceil(len(tr_u) / cfg.batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    rng = np.random.default_rng(seed)

    restored = store.restore_epoch(model, optimizer, rng)
    start = 1 if restored is None else int(restored["next_epoch"])
    history = list(restored.get("history", [])) if restored else []
    if restored is not None:
        print(f"  [{model_id} s{seed}] epoch {start - 1}에서 재개")
    started = time.time()
    for epoch in range(start, cfg.epochs + 1):
        model.train()
        permutation = rng.permutation(len(tr_u))
        totals = {"loss": 0.0, "p_correct": 0.0}
        for batch in range(n_batches):
            index = permutation[batch * cfg.batch_size:(batch + 1) * cfg.batch_size]
            users_np, positives_np = tr_u[index], tr_i[index]
            negatives_np = m4_helpers.sample_uniform_negative_matrix(
                users_np, positives_np, data["n_items"], positive_keys, rng,
                k=cfg.negative_count,
            )
            tensors = [torch.as_tensor(value, dtype=torch.long, device=v3.DEVICE)
                       for value in (users_np, positives_np, negatives_np)]
            batch_weights = None if row_weights is None else torch.as_tensor(
                row_weights[index], dtype=torch.float32, device=v3.DEVICE)
            loss, _, correct = recheck._batch_loss(model, *tensors, batch_weights)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            totals["loss"] += float(loss.detach())
            totals["p_correct"] += correct
            store.heartbeat(epoch=epoch, max_epoch=cfg.epochs, batch=batch + 1,
                            batches=n_batches, loss=totals["loss"] / (batch + 1),
                            selection="none")
        history.append({"epoch": epoch, "loss": totals["loss"] / n_batches,
                        "p_correct": totals["p_correct"] / n_batches})
        if cfg.dataset == "hm" or epoch % 25 == 0 or epoch == cfg.epochs:
            print(f"  [{model_id} s{seed}] ep {epoch:3d}/{cfg.epochs} | "
                  f"loss {history[-1]['loss']:.4f} | "
                  f"P(pos>neg) {history[-1]['p_correct']:.3f}", flush=True)
        store.save_epoch(model, optimizer, rng, epoch=epoch, history=history,
                         wall_clock_sec=time.time() - started, selection="none")
    return history


@torch.no_grad()
def _evaluate(model, prepared: dict) -> dict:
    model.eval()
    flat, _ = moe._flat_evaluation(
        model, 0.0, prepared["cache"], prepared["meta"], prepared["data"],
        prepared["base_cfg"], per_user=False,
    )
    return test10._public_metrics(flat)


def _arm_path(prepared: dict, model_id: str, seed: int) -> Path:
    root = Path(prepared["out_dir"]) / "arms" / prepared["config_hash"]
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{model_id}_s{seed}.json"


def _run_arm(prepared: dict, cfg: DunnhumbyTestConfig, model_id: str, seed: int,
             adjacency, arm_graph: dict | None) -> dict:
    path = _arm_path(prepared, model_id, seed)
    if path.exists():
        print(f"  [cached] {model_id} s{seed} 재사용")
        return json.loads(path.read_text(encoding="utf-8"))

    model = _build_model(prepared, cfg, seed, adjacency)
    store = ProgressStore(
        Path(prepared["out_dir"]) / "progress" / prepared["config_hash"],
        RunIdentity(stage="dunnhumby_test1", model_id=model_id, seed=seed,
                    config_hash=prepared["config_hash"],
                    source_revision=prepared["revision"],
                    input_hash=prepared["input_hash"]),
    )
    history = _train(model, prepared, cfg, model_id, seed, store)
    payload = {
        "model_id": model_id, "seed": seed, "split": SPLIT,
        "epochs": cfg.epochs, "code_version": CODE_VERSION,
        "source_revision": prepared["revision"],
        "beta": arm_graph["beta"] if arm_graph else None,
        "edge_audit": arm_graph["audit"] if arm_graph else None,
        "final_loss": history[-1]["loss"],
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "metrics": _evaluate(model, prepared),
    }
    test10._atomic_json(path, payload)
    store.mark_complete(epoch=cfg.epochs, max_epoch=cfg.epochs, selection="none",
                        checkpoint_path="", result_path=str(path))
    return payload


# --------------------------------------------------------------------------
# reading
# --------------------------------------------------------------------------


def seed_table(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{"seed": row["seed"], "model_id": row["model_id"],
                          **row["metrics"]} for row in rows])


def averaged_reading(table: pd.DataFrame, cfg: DunnhumbyTestConfig) -> dict:
    metrics = list(ACCURACY_METRICS) + list(ECONOMIC_METRICS)
    m1 = table[table.model_id.eq(M1_MODEL_ID)].set_index("seed")
    m3 = table[table.model_id.eq(M3_MODEL_ID)].set_index("seed")
    shared_seeds = sorted(set(m1.index) & set(m3.index))
    if not shared_seeds:
        raise RuntimeError("두 arm이 모두 끝난 시드가 없습니다")

    summary = {}
    for metric in metrics:
        base, arm = m1.loc[shared_seeds, metric], m3.loc[shared_seeds, metric]
        summary[metric] = {
            "m1_mean": float(base.mean()), "m3_mean": float(arm.mean()),
            "relative_change_pct": float((arm.mean() / base.mean() - 1) * 100),
            "paired_diff_mean": float((arm - base).mean()),
            "paired_diff_sd": float((arm - base).std(ddof=1)) if len(shared_seeds) > 1
            else float("nan"),
            "seeds_improved": int((arm > base).sum()),
        }
    economic_up = all(summary[m]["paired_diff_mean"] > 0 for m in ECONOMIC_METRICS)
    guard = all(
        summary[m]["m3_mean"] >= ACCURACY_GUARD * summary[m]["m1_mean"]
        for m in ACCURACY_METRICS
    )
    return {
        "split": SPLIT, "seeds": shared_seeds, "seed_count": len(shared_seeds),
        "both_economic_metrics_improved_on_the_mean": bool(economic_up),
        "six_accuracy_metrics_at_least_99pct_of_m1_on_the_mean": bool(guard),
        "condition_met": bool(economic_up and guard),
        "worst_accuracy_ratio_on_the_mean": min(
            summary[m]["m3_mean"] / summary[m]["m1_mean"] for m in ACCURACY_METRICS
        ),
        "per_metric": summary,
        "significance_claimed": False,
        "selected_after_seeing": False,
    }


def run_dunnhumby_test1(cfg: DunnhumbyTestConfig | None = None) -> pd.DataFrame:
    cfg = validate_config(cfg or configure_dunnhumby_test1())
    print(json.dumps(preflight_summary(cfg), ensure_ascii=False, indent=2))

    rows = []
    for seed in cfg.seeds:
        print(f"\n===== seed {seed} =====")
        prepared = _prepare(cfg, seed)
        arm_graph = build_graph(prepared, cfg)      # 게이트 실패 시 예외
        print(f"  beta {arm_graph['beta']:.4f} | "
              f"{json.dumps(arm_graph['audit'], ensure_ascii=False)}")
        adjacency = v3.build_adj(
            prepared["signals"]["edge_users"], prepared["signals"]["edge_items"],
            arm_graph["weights"].astype(np.float32),
            prepared["data"]["n_users"], prepared["data"]["n_items"],
        )
        rows.append(_run_arm(prepared, cfg, M1_MODEL_ID, seed,
                             prepared["data"]["adj"], None))
        rows.append(_run_arm(prepared, cfg, M3_MODEL_ID, seed, adjacency, arm_graph))

    table = seed_table(rows)
    reading = averaged_reading(table, cfg)

    out = Path(cfg.out_dir)
    stem = f"clv_m3_dunnhumby_test1_{'_'.join(str(s) for s in cfg.seeds)}"
    paths = {"seed_csv": out / f"{stem}_seeds.csv", "json": out / f"{stem}.json"}
    test10._atomic_csv(paths["seed_csv"], table)
    test10._atomic_json(paths["json"], {
        "code_version": CODE_VERSION, "config": asdict(cfg),
        "preflight": preflight_summary(cfg),
        "seed_rows": table.to_dict("records"), "reading": reading,
        "result_paths": {k: str(v) for k, v in paths.items()},
    })
    print("\n판독:", json.dumps(reading, ensure_ascii=False, indent=2))
    print("저장:", json.dumps({k: str(v) for k, v in paths.items()}, ensure_ascii=False))
    table.attrs.update(reading=reading,
                       result_paths={k: str(v) for k, v in paths.items()})
    return table


if __name__ == "__main__":
    print(json.dumps(preflight_summary(configure_dunnhumby_test1()),
                     ensure_ascii=False, indent=2))
