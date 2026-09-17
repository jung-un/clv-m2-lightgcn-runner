"""Five-seed development recheck of three CLV components before combining them.

The combination plan assigns one CLV component to each intervention point:

* ``m3_v_contribution``  - graph: customer V percentile x the item's share of
  that customer's basket value reweights the binary edges (08-18 design).
* ``m2_nv_history_fit``  - representation: q_N / q_V weight the user's own
  purchase history against separate candidate-item N/V tables (08-27 design).
* ``m4_complementary``   - loss: ``1 + lambda*q_C*(1 - <RBF(q_V), RBF(price)>)``
  positive weights (09-16 design).

Every earlier result came from a different pipeline, negative count, split or
CLV definition, so none of them can be combined as-is.  This runner retrains
all three next to M1 under one protocol: Dunnhumby ``DAY 1~683`` training and
``684~690`` development evaluation, the current historical CLV proxy
(``q_N``, ``q_V``, ``q_C`` from the last 365 training days), original LightGCN
BPR with one uniform unseen negative, 100 fixed epochs and seeds 42-46.

Each component is judged only on the role it is meant to play in the
combination.  Five seeds cannot establish significance; the screen decides
which components keep a consistent direction and may enter the combination.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import torch

from clv_history_item_fit_model import (
    HistoryItemFitLightGCN,
    build_personal_history_weights,
)
from clv_m3_transfer_graph import build_m3_transfer_graphs
from clv_m5_n_conditioned_value_basis_model import M5NConditionedValueBasisLightGCN
from clv_run_state import ProgressStore, RunIdentity, clone_state, file_sha256
import lightgcn_clv_axis_specific_test10 as test10
import lightgcn_clv_gatefree_lowdim as gatefree
import lightgcn_clv_m4_clv_hard_negative as m4_helpers
import lightgcn_clv_m5_k1_m4_improvement_screen as improvement
import lightgcn_clv_moe as moe
import lightgcn_clv_v3 as v3
import lightgcn_clv_value_basis_test10 as paired


CODE_VERSION = "clv-component-recheck-dev5-v1"
SEEDS = (42, 43, 44, 45, 46)

M1_MODEL_ID = "m1_bpr_k1"
M3_MODEL_ID = "m3_v_contribution_graph_bpr_k1"
M2_MODEL_ID = "m2_nv_history_fit_bpr_k1"
M4_MODEL_ID = "m4_complementary_weight_bpr_k1"
MODEL_IDS = (M1_MODEL_ID, M3_MODEL_ID, M2_MODEL_ID, M4_MODEL_ID)

WEIGHTED_HIT_10 = "price_purchase_amount_weighted_hit@10"
WEIGHTED_HIT_50 = "price_purchase_amount_weighted_hit@50"


@dataclass(frozen=True)
class ComponentRecheckConfig:
    seeds: tuple[int, ...] = SEEDS
    epochs: int = 100
    batch_size: int = 8192
    lr: float = 5e-4
    pref_reg: float = 1e-3
    id_dim: int = 64
    n_layers: int = 2
    negative_count: int = 1
    m3_beta_cap: float = 0.25
    m3_category_prior_strength: float = 20.0
    history_axis_dim: int = 4
    history_rho: float = 0.05
    m4_lambda: float = 0.5
    basis_bandwidth: float = 0.25
    accuracy_guard: float = 0.98
    min_positive_seeds: int = 4
    out_dir: str = ""


PROTOCOL_FIELDS = tuple(
    field for field in ComponentRecheckConfig.__dataclass_fields__ if field != "out_dir"
)


def configure_component_recheck(**overrides) -> ComponentRecheckConfig:
    defaults = {
        "out_dir": f"{v3.default_out_dir('dunnhumby')}_clv_component_recheck_dev5_v1"
    }
    return validate_config(ComponentRecheckConfig(**(defaults | overrides)))


def validate_config(cfg: ComponentRecheckConfig) -> ComponentRecheckConfig:
    """Fail closed if the shared protocol or any component setting drifts."""

    required = {
        "seeds": SEEDS,
        "epochs": 100,
        "id_dim": 64,
        "n_layers": 2,
        "negative_count": 1,
        "m3_beta_cap": 0.25,
        "m3_category_prior_strength": 20.0,
        "history_axis_dim": 4,
        "history_rho": 0.05,
        "m4_lambda": 0.5,
        "basis_bandwidth": 0.25,
        "accuracy_guard": 0.98,
        "min_positive_seeds": 4,
    }
    for key, expected in required.items():
        if getattr(cfg, key) != expected:
            raise ValueError(f"구성요소 재확인 설정은 {key}={expected!r}이어야 합니다")
    if cfg.batch_size <= 0 or cfg.lr <= 0 or cfg.pref_reg < 0:
        raise ValueError("학습 설정이 잘못됐습니다")
    if not cfg.out_dir:
        raise ValueError("out_dir가 필요합니다")
    return cfg


def arm_specifications() -> list[dict]:
    return [
        {"model_id": M1_MODEL_ID, "role": "baseline", "kind": "lightgcn",
         "graph": "binary", "weighted": False},
        {"model_id": M3_MODEL_ID, "role": "graph_value_without_accuracy_loss",
         "kind": "lightgcn", "graph": "m3_v_contribution", "weighted": False},
        {"model_id": M2_MODEL_ID, "role": "representation_candidate_expansion",
         "kind": "history_fit", "graph": "binary", "weighted": False},
        {"model_id": M4_MODEL_ID, "role": "loss_top_rank_correction",
         "kind": "lightgcn", "graph": "binary", "weighted": True},
    ]


def preflight_summary(cfg: ComponentRecheckConfig) -> dict:
    cfg = validate_config(cfg)
    return {
        "code_version": CODE_VERSION,
        "dataset": "dunnhumby",
        "seeds": list(cfg.seeds),
        "split": "historical_development_days_684_690",
        "trained_models": list(MODEL_IDS),
        "clv_definition": (
            "current historical proxy from the last 365 training days: "
            "q_N=percentile(repeat transaction rate), q_V=percentile(mean "
            "basket value), q_C=percentile(N*V); shared by every component"
        ),
        "loss": {
            "bpr": "mean softplus(s(u,j) - s(u,i+)) + batch L2",
            "negative_count": cfg.negative_count,
            "negative_sampling": "uniform over unseen items",
            "hard_negative": False,
        },
        "components": {
            M3_MODEL_ID: (
                "binary LightGCN whose edges are reweighted by "
                "q_V(u) x mean item share of the user's basket value "
                f"(beta cap {cfg.m3_beta_cap}, strength matched to the N relation)"
            ),
            M2_MODEL_ID: (
                "LightGCN ID plus q_N / q_V weighted personal-history to "
                f"candidate fit blocks (axis dim {cfg.history_axis_dim}, "
                f"rho {cfg.history_rho}, leave-one-out training profiles)"
            ),
            M4_MODEL_ID: (
                f"binary LightGCN with positive weights 1 + {cfg.m4_lambda}*q_C*"
                "(1 - <RBF(q_V), RBF(item amount percentile)>), mean one"
            ),
        },
        "decision_rule": {
            M3_MODEL_ID: (
                f"{WEIGHTED_HIT_10} paired mean > 0 with at least "
                f"{cfg.min_positive_seeds}/5 positive seeds, and mean NDCG@10 "
                f">= {cfg.accuracy_guard} x M1"
            ),
            M2_MODEL_ID: (
                f"recall@50 and {WEIGHTED_HIT_50} each paired mean > 0 with at "
                f"least {cfg.min_positive_seeds}/5 positive seeds, and mean "
                f"NDCG@10 >= {cfg.accuracy_guard} x M1"
            ),
            M4_MODEL_ID: (
                "recall@10 or ndcg@10 paired mean > 0 with at least "
                f"{cfg.min_positive_seeds}/5 positive seeds, and mean "
                f"{WEIGHTED_HIT_10} >= {cfg.accuracy_guard} x M1"
            ),
            "outcome": "components that pass enter the combination screen",
        },
        "limits": (
            "five development seeds on a repeatedly exposed split; intervals are "
            "reported but no significance, CLV attribution or generalization is claimed"
        ),
    }


def _config_hash(cfg: ComponentRecheckConfig, input_hash: str) -> str:
    payload = {
        "code_version": CODE_VERSION,
        "protocol": {field: getattr(cfg, field) for field in PROTOCOL_FIELDS},
        "input_hash": input_hash,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()[:12]


def _prepare(cfg: ComponentRecheckConfig) -> dict:
    improvement_cfg = improvement.configure_improvement_screen(out_dir=cfg.out_dir)
    # M1 is retrained here under the same seeds, so the shared preparation's
    # display-only lookup of an old seed-42 M1 result must not gate the run.
    with mock.patch.object(gatefree, "_load_compatible_baseline", return_value=None):
        prepared = improvement._prepare(improvement_cfg)
    data = prepared["data"]
    train = data["train"]
    missing = {"u_idx", "i_idx", "b_raw", "cat_idx", "t", "v"}.difference(train.columns)
    if missing:
        raise RuntimeError(f"구성요소 재확인 train 열 누락: {sorted(missing)}")
    n_users, n_items = data["n_users"], data["n_items"]
    q_n = np.where(prepared["clv_valid"], prepared["q_n"], 0.0)
    q_v = np.where(prepared["clv_valid"], prepared["q_v"], 0.0)

    graph = build_m3_transfer_graphs(
        train,
        n_users,
        n_items,
        beta_cap=cfg.m3_beta_cap,
        category_prior_strength=cfg.m3_category_prior_strength,
        user_q_n=q_n,
        user_q_v=q_v,
    )
    edge_keys = graph.edge_users * n_items + graph.edge_items
    if not np.array_equal(edge_keys, np.asarray(data["pos_key"], dtype=np.int64)):
        raise RuntimeError("M3 가중 엣지 순서가 M1 이진 그래프와 다릅니다")
    prepared["m3_adj"] = v3.build_adj(
        graph.edge_users,
        graph.edge_items,
        graph.v_weights.astype(np.float32),
        n_users,
        n_items,
    )
    prepared["m3_diagnostics"] = {
        **graph.diagnostics,
        "v_weight_min": float(graph.v_weights.min()),
        "v_weight_max": float(graph.v_weights.max()),
        "v_weight_cv": float(graph.v_weights.std() / graph.v_weights.mean()),
    }
    prepared["history"] = build_personal_history_weights(
        train, n_users=n_users, n_items=n_items
    )
    weights, weight_diagnostics = improvement.row_weights(
        prepared, improvement_cfg, "complementary"
    )
    prepared["m4_weights"] = weights
    prepared["m4_diagnostics"] = weight_diagnostics
    prepared["out_dir"] = Path(cfg.out_dir)
    prepared["config_hash"] = _config_hash(cfg, prepared["input_hash"])
    return prepared


def _build_model(prepared: dict, cfg: ComponentRecheckConfig, spec: dict, seed: int):
    data = prepared["data"]
    valid = np.asarray(prepared["clv_valid"], dtype=bool)
    v3.set_seed(seed)
    if spec["kind"] == "history_fit":
        model = HistoryItemFitLightGCN(
            n_users=data["n_users"],
            n_items=data["n_items"],
            history=prepared["history"],
            q_n=np.where(valid, prepared["q_n"], 0.0).astype(np.float32),
            q_v=np.where(valid, prepared["q_v"], 0.0).astype(np.float32),
            activity_valid=valid,
            value_valid=valid,
            adj=data["adj"],
            id_dim=cfg.id_dim,
            axis_dim=cfg.history_axis_dim,
            n_layers=cfg.n_layers,
            rho=cfg.history_rho,
            pref_reg=cfg.pref_reg,
        )
    else:
        adj = prepared["m3_adj"] if spec["graph"] == "m3_v_contribution" else data["adj"]
        model = M5NConditionedValueBasisLightGCN(
            n_users=data["n_users"],
            n_items=data["n_items"],
            user_q_n=prepared["q_n"],
            user_q_v=prepared["q_v"],
            user_q_c=prepared["q_c"],
            user_clv_valid=valid,
            item_price_percentile=prepared["item_amount_percentile"],
            item_price_valid=prepared["item_economic_valid"],
            adj=adj,
            id_dim=cfg.id_dim,
            rho=0.0,
            n_layers=cfg.n_layers,
            pref_reg=cfg.pref_reg,
            basis_bandwidth=cfg.basis_bandwidth,
            economic_propagation=False,
        )
    return model.to(v3.DEVICE)


def _arm_hash(prepared: dict, cfg: ComponentRecheckConfig, spec: dict, seed: int) -> str:
    payload = {"run": prepared["config_hash"], **spec, "seed": seed, "epochs": cfg.epochs}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()[:12]


def _arm_paths(prepared: dict, model_id: str, seed: int) -> dict[str, Path]:
    root = prepared["out_dir"] / "arms" / prepared["config_hash"]
    stem = f"{model_id}_s{seed}"
    return {
        "result": root / f"{stem}.json",
        "per_user": root / f"{stem}_per_user.npz",
        "checkpoint": root / f"{stem}.pt",
    }


def _batch_loss(model, users, positives, negatives, batch_weights):
    """Plain BPR for every arm; history-fit models use their own leave-one-out loss."""

    if batch_weights is None and hasattr(model, "bpr_loss"):
        loss, diagnostics = model.bpr_loss(users, positives, negatives[:, 0])
        return loss, float(diagnostics["bpr"]), float(diagnostics["p_correct"])
    user_z, item_z = model.propagated_embeddings()
    positive_scores = (user_z[users] * item_z[positives]).sum(dim=1)
    negative_scores = (user_z[users, None, :] * item_z[negatives]).sum(dim=2)
    per_row = torch.nn.functional.softplus(
        negative_scores - positive_scores[:, None]
    ).mean(dim=1)
    bpr = per_row.mean() if batch_weights is None else (batch_weights * per_row).mean()
    loss = bpr + model.sampled_l2(users, positives, negatives)
    correct = (positive_scores[:, None] > negative_scores).float().mean()
    return loss, float(bpr.detach()), float(correct.detach())


def _train_arm(
    model,
    prepared: dict,
    cfg: ComponentRecheckConfig,
    spec: dict,
    seed: int,
    store: ProgressStore,
) -> dict:
    data = prepared["data"]
    tr_u, tr_i, positive_keys = data["tr_u"], data["tr_i"], data["pos_key"]
    n_train = len(tr_u)
    n_batches = math.ceil(n_train / cfg.batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    rng = np.random.default_rng(seed)
    weight_tensor = (
        torch.as_tensor(prepared["m4_weights"], dtype=torch.float32, device=v3.DEVICE)
        if spec["weighted"]
        else None
    )

    restored = store.restore_epoch(model, optimizer, rng)
    start_epoch = 1 if restored is None else int(restored["next_epoch"])
    history = list(restored.get("history", [])) if restored else []
    previous_wall = float(restored.get("wall_clock_sec", 0.0)) if restored else 0.0
    if restored is not None:
        print(f"  [{spec['model_id']} s{seed}] epoch {start_epoch - 1}에서 자동 재개")
    store.mark_stage("running", epoch=start_epoch - 1, max_epoch=cfg.epochs, selection="none")

    started = time.time()
    last_epoch = start_epoch - 1
    for epoch in range(start_epoch, cfg.epochs + 1):
        last_epoch = epoch
        model.train()
        epoch_started = time.time()
        permutation = rng.permutation(n_train)
        totals = {"loss": 0.0, "bpr": 0.0, "p_correct": 0.0}
        for batch in range(n_batches):
            index = permutation[batch * cfg.batch_size : (batch + 1) * cfg.batch_size]
            users_np, positives_np = tr_u[index], tr_i[index]
            negatives_np = m4_helpers.sample_uniform_negative_matrix(
                users_np, positives_np, data["n_items"], positive_keys, rng,
                k=cfg.negative_count,
            )
            users = torch.as_tensor(users_np, dtype=torch.long, device=v3.DEVICE)
            positives = torch.as_tensor(positives_np, dtype=torch.long, device=v3.DEVICE)
            negatives = torch.as_tensor(negatives_np, dtype=torch.long, device=v3.DEVICE)
            batch_weights = (
                None
                if weight_tensor is None
                else weight_tensor[torch.as_tensor(index, dtype=torch.long, device=v3.DEVICE)]
            )
            loss, bpr, correct = _batch_loss(model, users, positives, negatives, batch_weights)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            totals["loss"] += float(loss.detach())
            totals["bpr"] += bpr
            totals["p_correct"] += correct
            store.heartbeat(
                epoch=epoch, max_epoch=cfg.epochs, batch=batch + 1,
                batches=n_batches, loss=totals["loss"] / (batch + 1), selection="none",
            )
        record = {
            "epoch": epoch,
            "loss": totals["loss"] / n_batches,
            "bpr": totals["bpr"] / n_batches,
            "p_correct": totals["p_correct"] / n_batches,
            "epoch_sec": time.time() - epoch_started,
        }
        history.append(record)
        store.save_epoch(
            model, optimizer, rng, epoch=epoch, history=history,
            wall_clock_sec=previous_wall + time.time() - started, selection="none",
        )
        print(
            f"  [{spec['model_id']} s{seed}] ep {epoch:3d}/{cfg.epochs} | "
            f"loss {record['loss']:.4f} | P(pos>neg) {record['p_correct']:.3f} | "
            f"{record['epoch_sec']:.0f}s"
        )
    if last_epoch != cfg.epochs:
        raise RuntimeError(f"고정 {cfg.epochs} epoch 미완료: {last_epoch}")
    return {
        "epochs_run": cfg.epochs,
        "selection": "none",
        "wall_clock_sec": previous_wall + time.time() - started,
        "resumed_from_epoch": start_epoch - 1,
        "history": history,
    }


def _run_arm(prepared: dict, cfg: ComponentRecheckConfig, spec: dict, seed: int) -> dict:
    paths = _arm_paths(prepared, spec["model_id"], seed)
    cached = test10._load_cached_arm(paths)
    if cached is not None:
        return cached

    model = _build_model(prepared, cfg, spec, seed)
    store = ProgressStore(
        prepared["out_dir"] / "progress" / prepared["config_hash"],
        RunIdentity(
            stage="component_recheck_dev",
            model_id=spec["model_id"],
            seed=seed,
            config_hash=_arm_hash(prepared, cfg, spec, seed),
            source_revision=prepared["revision"],
            input_hash=prepared["input_hash"],
        ),
    )
    training = _train_arm(model, prepared, cfg, spec, seed, store)
    model.eval()
    paths["checkpoint"].parent.mkdir(parents=True, exist_ok=True)
    temporary = paths["checkpoint"].with_suffix(".pt.tmp")
    torch.save(
        {"state": clone_state(model), "model_id": spec["model_id"], "seed": seed,
         "training": training, "config": asdict(cfg),
         "source_revision": prepared["revision"], "input_hash": prepared["input_hash"]},
        temporary,
    )
    os.replace(temporary, paths["checkpoint"])
    metrics, per_user = moe._flat_evaluation(
        model, 0.0, prepared["cache"], prepared["meta"], prepared["data"],
        prepared["base_cfg"], per_user=True,
    )
    public_per_user = test10._public_per_user(per_user)
    test10._atomic_npz(paths["per_user"], public_per_user)
    diagnostics = (
        model.representation_diagnostics()
        if hasattr(model, "representation_diagnostics")
        else {}
    )
    payload = {
        "model_id": spec["model_id"],
        "role": spec["role"],
        "seed": seed,
        "split": "historical_development_days_684_690",
        "final_epoch": cfg.epochs,
        "metrics": test10._public_metrics(metrics),
        "diagnostics": diagnostics,
        "training": training,
        "code_version": CODE_VERSION,
        "source_revision": prepared["revision"],
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(paths["checkpoint"]),
        "checkpoint_sha256": file_sha256(paths["checkpoint"]),
    }
    test10._atomic_json(paths["result"], payload)
    payload["per_user"] = public_per_user
    store.mark_complete(
        epoch=cfg.epochs, max_epoch=cfg.epochs, selection="none",
        checkpoint_path=str(paths["checkpoint"]), result_path=str(paths["result"]),
    )
    return payload


def _absolute_rows(arms: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"model_id": arm["model_id"], "role": arm["role"], "seed": arm["seed"],
          **arm["metrics"]} for arm in arms]
    )


def component_reading(
    absolute_summary: pd.DataFrame,
    paired_summary: pd.DataFrame,
    cfg: ComponentRecheckConfig,
) -> dict:
    """Apply each component's role-specific pre-registered rule."""

    def paired_stat(model_id: str, metric: str) -> dict:
        row = paired_summary[
            paired_summary.model_id.eq(model_id)
            & paired_summary.reference.eq(M1_MODEL_ID)
            & paired_summary.metric.eq(metric)
        ]
        if len(row) != 1:
            raise KeyError(f"{model_id}의 {metric} 대응 요약이 없습니다")
        return row.iloc[0].to_dict()

    def ratio(model_id: str, metric: str) -> float:
        def mean(model: str) -> float:
            row = absolute_summary[
                absolute_summary.model_id.eq(model) & absolute_summary.metric.eq(metric)
            ]
            return float(row.iloc[0]["mean"])

        return mean(model_id) / mean(M1_MODEL_ID)

    def consistent_gain(model_id: str, metric: str) -> bool:
        stat = paired_stat(model_id, metric)
        return bool(stat["mean"] > 0 and stat["positive_seed_count"] >= cfg.min_positive_seeds)

    m3 = {
        "value_gain": consistent_gain(M3_MODEL_ID, WEIGHTED_HIT_10),
        "accuracy_guard": ratio(M3_MODEL_ID, "ndcg@10") >= cfg.accuracy_guard,
    }
    m2 = {
        "recall50_gain": consistent_gain(M2_MODEL_ID, "recall@50"),
        "weighted_hit50_gain": consistent_gain(M2_MODEL_ID, WEIGHTED_HIT_50),
        "accuracy_guard": ratio(M2_MODEL_ID, "ndcg@10") >= cfg.accuracy_guard,
    }
    m4 = {
        "top_rank_gain": consistent_gain(M4_MODEL_ID, "recall@10")
        or consistent_gain(M4_MODEL_ID, "ndcg@10"),
        "value_guard": ratio(M4_MODEL_ID, WEIGHTED_HIT_10) >= cfg.accuracy_guard,
    }
    components = {M3_MODEL_ID: m3, M2_MODEL_ID: m2, M4_MODEL_ID: m4}
    for checks in components.values():
        checks["retained"] = all(value for value in checks.values())
    return {
        "components": components,
        "combination_candidates": [
            model_id for model_id, checks in components.items() if checks["retained"]
        ],
        "significance_claimed": False,
        "clv_attribution_tested": False,
    }


def _persist(prepared: dict, cfg: ComponentRecheckConfig, arms: list[dict]) -> pd.DataFrame:
    absolute = _absolute_rows(arms)
    metric_columns = paired._metric_columns(absolute)
    absolute_summary = pd.DataFrame(
        [{"model_id": model_id, "metric": metric, **test10._mean_ci(group[metric].to_numpy())}
         for model_id, group in absolute.groupby("model_id", sort=False)
         for metric in metric_columns]
    )
    paired_seed, paired_summary = paired.paired_tables(
        absolute, arms, [(model_id, M1_MODEL_ID) for model_id in MODEL_IDS[1:]]
    )
    reading = component_reading(absolute_summary, paired_summary, cfg)

    stem = f"clv_component_recheck_{prepared['config_hash']}"
    out = prepared["out_dir"]
    paths = {
        "absolute_csv": out / f"{stem}.csv",
        "absolute_summary_csv": out / f"{stem}_mean.csv",
        "paired_seed_csv": out / f"{stem}_paired_seed.csv",
        "paired_summary_csv": out / f"{stem}_paired_mean.csv",
        "json": out / f"{stem}.json",
    }
    test10._atomic_csv(paths["absolute_csv"], absolute)
    test10._atomic_csv(paths["absolute_summary_csv"], absolute_summary)
    test10._atomic_csv(paths["paired_seed_csv"], paired_seed)
    test10._atomic_csv(paths["paired_summary_csv"], paired_summary)
    test10._atomic_json(
        paths["json"],
        {
            "code_version": CODE_VERSION,
            "source_revision": prepared["revision"],
            "config": asdict(cfg),
            "preflight": preflight_summary(cfg),
            "m3_graph_diagnostics": prepared["m3_diagnostics"],
            "m4_weight_diagnostics": prepared["m4_diagnostics"],
            "history_diagnostics": prepared["history"].diagnostics,
            "absolute_rows": absolute.to_dict("records"),
            "absolute_summary": absolute_summary.to_dict("records"),
            "paired_seed": paired_seed.to_dict("records"),
            "paired_summary": paired_summary.to_dict("records"),
            "reading": reading,
            "result_paths": {name: str(path) for name, path in paths.items()},
        },
    )
    absolute.attrs.update(
        absolute_summary=absolute_summary,
        paired_summary=paired_summary,
        reading=reading,
        result_paths={name: str(path) for name, path in paths.items()},
    )
    return absolute


def run_component_recheck(cfg: ComponentRecheckConfig | None = None) -> pd.DataFrame:
    cfg = validate_config(cfg or configure_component_recheck())
    print(json.dumps(preflight_summary(cfg), ensure_ascii=False, indent=2))
    prepared = _prepare(cfg)
    print("M3 V 기여 엣지:", json.dumps(prepared["m3_diagnostics"], default=str)[:600])
    print("M4 보완 가중치:", json.dumps(prepared["m4_diagnostics"], default=str))
    arms = []
    for seed in cfg.seeds:
        for spec in arm_specifications():
            print(f"\n===== seed {seed} | {spec['model_id']} | K=1 | 100 epoch =====")
            arms.append(_run_arm(prepared, cfg, spec, seed))
    frame = _persist(prepared, cfg, arms)
    print("\nM1 대비 대응 차이 (5시드):")
    print(frame.attrs["paired_summary"].to_string(index=False))
    print("\n사전 판정:", json.dumps(frame.attrs["reading"], ensure_ascii=False, indent=2))
    print("결과 파일:", frame.attrs["result_paths"])
    return frame


if __name__ == "__main__":
    print(json.dumps(preflight_summary(configure_component_recheck()), ensure_ascii=False, indent=2))
