"""Last-week pilot for the smallest direct N/V M2 representation.

ID embeddings alone use the binary LightGCN graph. Two fixed, centered N/V
coordinates are appended after propagation and trained jointly through two
global scalars. No q_C, RBF, matrix, gate, edge weight, row weight, freeze, or
external reranking is used.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

import clv_m2_m5_lastweek_test10 as common
import lightgcn_clv_candidate_nv_fit_factorial as candidate
from clv_m5_n_conditioned_value_basis_model import M5NConditionedValueBasisLightGCN
from clv_run_state import ProgressStore, RunIdentity


v3, fixed, io = common.v3, common.fixed, common.io
CODE_VERSION = "clv-m2-direct-nv-lastweek-seed49-v1"
M1 = "m1_bpr_k1"
M2 = "m2_direct_nv_postprop_bpr_k1"
MODELS = (M1, M2)
SEEDS = common.SEEDS
ACCURACY, ECONOMIC = common.ACCURACY, common.ECONOMIC


@dataclass(frozen=True)
class Config:
    dataset: str = "dunnhumby"
    seeds: tuple[int, ...] = (49,)
    epochs: int = 300
    id_dim: int = 64
    n_layers: int = 2
    batch_size: int = 8192
    lr: float = 5e-4
    pref_reg: float = 1e-3
    negative_count: int = 1
    input_days: int = 365
    out_dir: str = ""


def configure(*, seeds=(49,), out_dir=None) -> Config:
    seeds = tuple(seeds)
    if seeds not in ((49,), SEEDS):
        raise ValueError("Dunnhumby seed49 예비 실행 또는 고정 10시드만 허용합니다")
    suffix = "_m2_direct_nv_lastweek_seed49_pilot_v1" if len(seeds) == 1 else "_m2_direct_nv_lastweek_test10_v1"
    return Config(seeds=seeds, out_dir=out_dir or v3.default_out_dir("dunnhumby") + suffix)


def _base_config(cfg: Config) -> dict:
    return common.base_config(common.Config(
        dataset="dunnhumby", seeds=cfg.seeds, epochs=cfg.epochs,
        id_dim=cfg.id_dim, n_layers=cfg.n_layers, batch_size=cfg.batch_size,
        lr=cfg.lr, pref_reg=cfg.pref_reg, negative_count=cfg.negative_count,
        input_days=cfg.input_days, out_dir=cfg.out_dir,
    ))


def _item_n_totals(train: pd.DataFrame, q_n: np.ndarray, valid: np.ndarray, n_items: int):
    pairs = train[["u_idx", "i_idx"]].drop_duplicates()
    users = pairs.u_idx.to_numpy(np.int64, copy=False)
    items = pairs.i_idx.to_numpy(np.int64, copy=False)
    usable = valid[users]
    return (
        np.bincount(items[usable], weights=q_n[users[usable]], minlength=n_items).astype(np.float32),
        np.bincount(items[usable], minlength=n_items).astype(np.float32),
    )


class DirectNVM2(M5NConditionedValueBasisLightGCN):
    """ID-only propagation plus two unpropagated, directly interpretable axes."""

    def __init__(self, *, user_q_n, user_q_v, user_clv_valid,
                 item_buyer_q_n, item_amount_percentile, item_n_sum,
                 item_n_count, active=True, **kwargs):
        valid = np.asarray(user_clv_valid, dtype=bool)
        item_n = np.asarray(item_buyer_q_n, dtype=np.float32)
        item_v = np.asarray(item_amount_percentile, dtype=np.float32)
        item_valid = np.isfinite(item_n) & np.isfinite(item_v)
        super().__init__(
            user_q_n=user_q_n, user_q_v=user_q_v,
            user_q_c=valid.astype(np.float32), user_clv_valid=valid,
            item_price_percentile=item_v, item_price_valid=item_valid,
            rho=1.0 if active else 0.0, basis_bandwidth=.25,
            constant_gate=1.0, economic_propagation=False, **kwargs,
        )
        q_n = np.asarray(user_q_n, dtype=np.float32)
        q_v = np.asarray(user_q_v, dtype=np.float32)
        item_n_sum = np.asarray(item_n_sum, dtype=np.float32)
        item_n_count = np.asarray(item_n_count, dtype=np.float32)
        if any(x.shape != (self.n_users,) for x in (q_n, q_v, valid)):
            raise ValueError("사용자 N/V 입력 shape이 잘못됐습니다")
        if any(x.shape != (self.n_items,) for x in (item_n, item_v, item_n_sum, item_n_count)):
            raise ValueError("상품 N/V 입력 shape이 잘못됐습니다")
        self.active = bool(active)
        self.economic_dim = 2
        self.gamma = nn.Parameter(torch.ones(2))
        user = np.column_stack((2 * q_n - 1, 2 * q_v - 1)).astype(np.float32)
        user[~valid] = 0.0
        item = np.column_stack((2 * item_n - 1, 2 * item_v - 1)).astype(np.float32)
        item[item_n_count <= 0, 0] = 0.0
        self.register_buffer("direct_user", torch.from_numpy(user), persistent=False)
        self.register_buffer("direct_item", torch.from_numpy(item), persistent=False)
        self.register_buffer("item_n_sum", torch.from_numpy(item_n_sum), persistent=False)
        self.register_buffer("item_n_count", torch.from_numpy(item_n_count), persistent=False)

    def economic_coordinates(self):
        if not self.active:
            return torch.zeros_like(self.direct_user), torch.zeros_like(self.direct_item)
        return self.direct_user * self.gamma[None, :], self.direct_item

    def bpr_loss(self, users, positives, negatives):
        user_id, item_id = self.id_embeddings()
        positive = (user_id[users] * item_id[positives]).sum(1)
        negative = (user_id[users] * item_id[negatives]).sum(1)
        if self.active:
            user = self.direct_user[users]
            remaining = self.item_n_count[positives] - self.user_clv_valid[users]
            loo_mean = torch.where(
                remaining > 0,
                (self.item_n_sum[positives] - self.user_clv_valid[users]
                 * (self.user_q_n_centered[users] + 1) / 2) / remaining.clamp_min(1),
                torch.full_like(remaining, .5),
            )
            positive = positive + self.gamma[0] * user[:, 0] * (2 * loo_mean - 1)
            positive = positive + self.gamma[1] * user[:, 1] * self.direct_item[positives, 1]
            negative = negative + (self.gamma * user * self.direct_item[negatives]).sum(1)
        row = F.softplus(negative - positive)
        bpr = row.mean()
        loss = bpr + self.sampled_l2(users, positives, negatives[:, None])
        return loss, {"bpr": float(bpr.detach()),
                      "p_correct": float((positive > negative).float().mean().detach())}

    @torch.no_grad()
    def representation_diagnostics(self):
        return {
            "id_dim": self.id_dim, "economic_dim": 2,
            "gamma_n": float(self.gamma[0]), "gamma_v": float(self.gamma[1]),
            "gamma_gradient_norm": 0.0 if self.gamma.grad is None else float(self.gamma.grad.norm()),
            "q_c_in_m2": False, "rbf": False, "matrix": False, "gate": False,
            "economic_graph_propagation": False, "joint_end_to_end_training": True,
            "positive_item_n_leave_one_out": True, "external_reranking": False,
            "n_role": "centered user q_N versus distinct-buyer mean q_N",
            "v_role": "centered user q_V versus item purchase-amount percentile",
        }


def prepare(cfg=None):
    cfg = cfg or configure()
    if cfg != configure(seeds=cfg.seeds, out_dir=cfg.out_dir):
        raise ValueError("사전 고정된 실행설정을 변경할 수 없습니다")
    manifest = fixed.moe.build_input_manifest(v3.SCHEMA["dunnhumby"])
    input_hash = fixed.moe.manifest_hash(manifest)
    revision = fixed.moe.source_revision()
    base = _base_config(cfg)
    data = v3.prepare_data(base, v3.DCFG)
    split_cfg = common.Config(dataset="dunnhumby", seeds=cfg.seeds,
                              epochs=cfg.epochs, id_dim=cfg.id_dim,
                              n_layers=cfg.n_layers, batch_size=cfg.batch_size,
                              lr=cfg.lr, pref_reg=cfg.pref_reg,
                              negative_count=cfg.negative_count,
                              input_days=cfg.input_days, out_dir=cfg.out_dir)
    common.validate_split(data, split_cfg)
    snapshot = fixed.residual.build_final_snapshot(data["train"], data["n_users"],
                                                    v3.DCFG["is_date"], cfg.input_days)
    axes = fixed.joint.build_user_axis_inputs(snapshot, data["n_users"])
    q_n, q_v, q_c, valid = fixed.evaluation.build_clv_inputs(axes)
    q_n, q_v, q_c = (np.where(valid, q, 0).astype(np.float32) for q in (q_n, q_v, q_c))
    features = candidate.build_candidate_nv_inputs(
        data["train"], n_users=data["n_users"], n_items=data["n_items"],
        q_n=q_n, q_v=q_v, q_c=q_c, clv_valid=valid,
    )
    item_n_sum, item_n_count = _item_n_totals(data["train"], q_n, valid, data["n_items"])
    thresholds = v3.segment_thresholds(axes["clv_proxy"], base["SEG_EDGES"])
    protocol = {
        "code_version": CODE_VERSION, "config": asdict(cfg), "models": list(MODELS),
        "split": "last_seven_days_test", "interval": common.INTERVALS["dunnhumby"],
        "validation": False, "holdout": False, "pilot_only": len(cfg.seeds) == 1,
        "previously_exposed_test": True, "post_hoc_exploratory_model": True,
        "unseen_confirmation_claim": False, "no_early_stopping": True,
        "test_checkpoint": "fixed epoch300 only; one evaluation per fit",
        "new_item_task": True, "min_item_interactions": 1,
        "graph": "binary ID-only propagation", "negative_sampling": "uniform K=1",
        "loss": "plain BPR + existing sampled ID L2", "q_c_in_m2": False,
        "m2": "post-propagation [centered q_N, centered q_V] x [buyer-q_N context, item amount percentile] with two learned scalars",
        "positive_item_n_leave_one_out": True, "economic_graph_propagation": False,
        "selection": "none; no formula/weight/epoch adjustment after this test",
        "candidate_rule": "both economic@10 > M1 and six Recall/NDCG >= .99*M1",
        "significance_claim": False, "clv_attribution_claim": False,
        "source_revision": revision, "input_hash": input_hash,
        "economic_input_diagnostics": features["economic_input_diagnostics"],
        "data_stats": data["data_stats"],
    }
    identity = {"version": CODE_VERSION, "config": asdict(cfg),
                "source_revision": revision, "input_hash": input_hash}
    config_hash = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
    run_dir = Path(cfg.out_dir) / config_hash
    io._atomic_json(run_dir / "protocol.json", protocol)
    prep = dict(data=data, base_cfg=base, axes=axes, q_n=q_n, q_v=q_v,
                clv_valid=valid, item_buyer_q_n=features["item_buyer_q_n"],
                item_amount_percentile=features["item_amount_percentile"],
                item_n_sum=item_n_sum, item_n_count=item_n_count,
                meta=v3.item_meta(data["train"], data["n_items"]),
                cache=v3.EvalCache(*data["splits"]["test"], axes["clv_proxy"],
                                   thresholds, data["n_items"]),
                out_dir=Path(cfg.out_dir), run_dir=run_dir, protocol=protocol,
                config_hash=config_hash, revision=revision, input_hash=input_hash)
    print(json.dumps({k: v for k, v in protocol.items() if k != "data_stats"}, ensure_ascii=False, indent=2))
    return cfg, prep


def build_model(prep, cfg, model_id, seed):
    if model_id not in MODELS or seed not in cfg.seeds:
        raise ValueError("사전등록 모형/시드가 아닙니다")
    v3.set_seed(seed)
    data = prep["data"]
    return DirectNVM2(
        n_users=data["n_users"], n_items=data["n_items"],
        user_q_n=prep["q_n"], user_q_v=prep["q_v"], user_clv_valid=prep["clv_valid"],
        item_buyer_q_n=prep["item_buyer_q_n"],
        item_amount_percentile=prep["item_amount_percentile"],
        item_n_sum=prep["item_n_sum"], item_n_count=prep["item_n_count"],
        active=model_id == M2, adj=data["adj"], id_dim=cfg.id_dim,
        n_layers=cfg.n_layers, pref_reg=cfg.pref_reg,
    ).to(v3.DEVICE)


def run_arm(prep, cfg, model_id, seed):
    identity = RunIdentity(CODE_VERSION, model_id, seed, prep["config_hash"],
                           prep["revision"], prep["input_hash"])
    path = prep["run_dir"] / "arms" / f"{model_id}_s{seed}.json"
    if path.exists():
        row = json.loads(path.read_text())
        if row.get("identity") != asdict(identity) or row.get("epochs") != cfg.epochs:
            raise RuntimeError("완료 결과 신원 불일치")
        return row
    model = build_model(prep, cfg, model_id, seed)
    store = ProgressStore(prep["run_dir"] / "progress", identity)
    history = fixed._train(model, prep, cfg, model_id, seed, store)
    if len(history) != cfg.epochs or history[-1]["epoch"] != cfg.epochs:
        raise RuntimeError("고정 epoch 학습을 완료하지 않아 test를 평가하지 않습니다")
    metrics = fixed._evaluate(model, prep)
    row = {"identity": asdict(identity), "model_id": model_id, "seed": seed,
           "epochs": cfg.epochs, "metrics": metrics, "training_history": history,
           "diagnostics": model.representation_diagnostics()}
    io._atomic_json(path, row)
    store.mark_complete(epoch=cfg.epochs, max_epoch=cfg.epochs, selection="none",
                        checkpoint_path=str(store.latest_checkpoint), result_path=str(path))
    return row


def report(prep, cfg, rows):
    absolute = pd.DataFrame([{"model_id": r["model_id"], "seed": r["seed"], **r["metrics"]} for r in rows])
    expected = {(model, seed) for model in MODELS for seed in cfg.seeds}
    if set(zip(absolute.model_id, absolute.seed)) != expected or len(absolute) != len(expected):
        raise RuntimeError("예정된 M1·M2 실행이 모두 끝나지 않았습니다")
    metrics = list(rows[0]["metrics"])
    if not np.isfinite(absolute[metrics].to_numpy(float)).all():
        raise RuntimeError("집계 지표 누락/비유한값")
    means = absolute.groupby("model_id")[metrics].mean()
    paired = []
    for metric in metrics:
        base = absolute[absolute.model_id.eq(M1)].set_index("seed")[metric]
        arm = absolute[absolute.model_id.eq(M2)].set_index("seed")[metric]
        for seed in cfg.seeds:
            paired.append({"metric": metric, "seed": seed, "m1": float(base[seed]),
                           "m2": float(arm[seed]), "delta": float(arm[seed] - base[seed])})
    guard = all(means.at[M2, metric] >= .99 * means.at[M1, metric] for metric in ACCURACY)
    economic = all(means.at[M2, metric] > means.at[M1, metric] for metric in ECONOMIC)
    reading = {"complete": True, "seed_count": len(cfg.seeds),
               "pilot_only": len(cfg.seeds) == 1, "final_ten_seed_report": len(cfg.seeds) == 10,
               "accuracy_mean_guard": bool(guard), "both_economic_means_above_m1": bool(economic),
               "candidate_condition_met": bool(guard and economic),
               "previously_exposed_test": True, "post_hoc_exploratory_model": True,
               "significance_claim": False, "clv_attribution_claim": False}
    tables = {
        "absolute": absolute,
        "summary": absolute.melt(id_vars=["model_id", "seed"], var_name="metric", value_name="value")
                           .groupby(["model_id", "metric"]).value.agg(["mean", "std", "count"]).reset_index(),
        "paired_comparison": pd.DataFrame(paired),
        "diagnostics": pd.DataFrame([{"model_id": r["model_id"], "seed": r["seed"],
                                      **r["diagnostics"]} for r in rows]),
    }
    root = prep["run_dir"] / "reports"
    paths = {name: str(root / f"{name}.csv") for name in tables}
    for name, table in tables.items():
        io._atomic_csv(Path(paths[name]), table)
    paths["json"] = str(root / "result.json")
    io._atomic_json(Path(paths["json"]), {"protocol": prep["protocol"], "reading": reading,
                                          "rows": rows, "paths": paths})
    return {**tables, "reading": reading, "paths": paths}


def run(cfg, prep):
    rows = [run_arm(prep, cfg, model, seed) for seed in cfg.seeds for model in MODELS]
    return report(prep, cfg, rows)
