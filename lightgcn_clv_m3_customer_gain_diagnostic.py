"""Where does the centered value graph M3 help, customer by customer? No training.

The question is whether a customer-level M4 weight has a role left to play beside
M3. M3 reweights the edges inside each customer, so it changes which items that
customer is pulled toward; a customer-level loss weight changes how much each
customer's errors matter. Those are different levers only if M3's gains are not
already concentrated on the customers such a weight would favour.

The 2026-09-26 axis diagnosis found that q_N predicts whether a customer
transacts (Spearman +0.519) while q_V predicts what they buy (+0.296), and the
ranking evaluation conditions the first away. If that reading is right, M3's
per-customer gains should track q_V rather than q_N, leaving the activity axis
for a customer weight. This runner measures that instead of assuming it.

Reads the finished epoch-300 checkpoints of M1, arm A and arm B for the three
seeds and scores every evaluation customer. Nothing is trained and no new
protocol is introduced; the split, the candidate masking and the metrics are the
ones the M3 run used.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

import lightgcn_clv_axis_specific_test10 as test10
import lightgcn_clv_component_recheck as recheck
import lightgcn_clv_m2_capacity_search as capacity
import lightgcn_clv_m3_centered_value_graph as graph
import lightgcn_clv_v3 as v3


CODE_VERSION = "clv-m3-customer-gain-diagnostic-v1"
M1_MODEL_ID = graph.M1_MODEL_ID
ARM_VALUE, ARM_VALUE_ACTIVITY = graph.ARM_VALUE, graph.ARM_VALUE_ACTIVITY
CHECKPOINT_EPOCH = 300          # 저장된 재개 파일은 마지막 epoch 상태뿐이다

# 결과를 보기 전에 고정한 판독 기준.
ROOM_THRESHOLD = 0.10           # |rho| < 0.10 → 고객 가중의 자리가 남아 있다
REDUNDANT_THRESHOLD = 0.30      # rho > +0.30 → M3가 이미 그 고객들을 챙겼다 (중복)
AXES = ("q_n", "q_v", "q_c")
PER_USER_METRICS = ("recall", "revenue")   # revenue = 가격·구매금액 가중 적중값


@dataclass(frozen=True)
class CustomerGainConfig:
    seeds: tuple[int, ...] = (42, 43, 44)
    epoch: int = CHECKPOINT_EPOCH
    id_dim: int = 64
    n_layers: int = 2
    pref_reg: float = 1e-3
    ks: tuple[int, ...] = (10, 50)
    deciles: int = 10
    m3_result_dir: str = ""
    m1_result_dir: str = ""
    out_dir: str = ""


def configure_customer_gain(**overrides) -> CustomerGainConfig:
    root = v3.default_out_dir("dunnhumby")
    defaults = {
        "out_dir": f"{root}_clv_m3_customer_gain_diagnostic_v1",
        "m3_result_dir": f"{root}_clv_m3_centered_value_graph_v1",
        "m1_result_dir": f"{root}_clv_m2_capacity_search_v1",
    }
    return CustomerGainConfig(**(defaults | overrides))


def preflight_summary(cfg: CustomerGainConfig) -> dict:
    return {
        "code_version": CODE_VERSION,
        "question": (
            "does the M3 gain per customer track q_N, which a customer-level M4 "
            "weight would favour, or q_V, which M3 itself acts on?"
        ),
        "trains_nothing": True,
        "checkpoints": f"finished epoch {cfg.epoch} states of M1, arm A and arm B",
        "seeds": list(cfg.seeds),
        "per_customer_metrics": [f"{m}@{k}" for m in PER_USER_METRICS for k in cfg.ks],
        "reading_fixed_before_results": {
            "room_for_a_customer_weight": f"|rho| < {ROOM_THRESHOLD} on all seeds",
            "redundant_with_m3": f"rho > +{REDUNDANT_THRESHOLD} on all seeds",
            "between": "no direction is claimed",
        },
        "limits": (
            "epoch 300 only, because the resume file keeps the last epoch alone; the "
            "pre-registered candidate for arm A was epoch 100. A rank correlation over "
            "one development split is not an effect of a weight that was never trained"
        ),
    }


# --------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------


def _checkpoint(root: Path, stage: str, model_id: str, seed: int) -> Path:
    matches = sorted(root.glob(
        f"progress/*/resume/{stage}_{model_id}_s{seed}_latest.pt"))
    if not matches:
        raise RuntimeError(
            f"{root}에서 {stage}_{model_id}_s{seed} 체크포인트를 찾지 못했습니다")
    if len(matches) > 1:
        raise RuntimeError(f"체크포인트가 여러 개입니다: {[str(m) for m in matches]}")
    return matches[0]


def load_state(path: Path, cfg: CustomerGainConfig) -> dict:
    """Read a finished checkpoint for scoring only, and report what it is.

    restore_epoch would refuse a checkpoint written by another revision, which is
    right when resuming training. Here nothing is trained, so the identity is
    printed and the epoch is checked instead of being silently accepted.
    """

    payload = torch.load(path, map_location="cpu", weights_only=False)
    identity = payload.get("identity", {})
    epoch = int(payload.get("epoch", -1))
    if epoch != cfg.epoch:
        raise RuntimeError(f"{path.name}: epoch {epoch}, 기대 {cfg.epoch}")
    print(f"  {path.name} | epoch {epoch} | 커밋 {str(identity.get('source_revision'))[:12]}"
          f" | config {identity.get('config_hash')}")
    return {"model_state": payload["model_state"], "identity": identity, "epoch": epoch}


def _model(prepared: dict, cfg: CustomerGainConfig, adjacency):
    model = graph.M5NConditionedValueBasisLightGCN(
        n_users=prepared["data"]["n_users"], n_items=prepared["data"]["n_items"],
        user_q_n=prepared["q_n"], user_q_v=prepared["q_v"], user_q_c=prepared["q_c"],
        user_clv_valid=np.asarray(prepared["clv_valid"], bool),
        item_price_percentile=prepared["item_amount_percentile"],
        item_price_valid=prepared["item_economic_valid"],
        adj=adjacency, id_dim=cfg.id_dim, rho=0.0, n_layers=cfg.n_layers,
        pref_reg=cfg.pref_reg, economic_propagation=False,
    )
    return model.to(v3.DEVICE)


# --------------------------------------------------------------------------
# per-customer scoring
# --------------------------------------------------------------------------


@torch.no_grad()
def per_customer(model, prepared: dict, cfg: CustomerGainConfig) -> pd.DataFrame:
    """One row per evaluation customer, the metrics the M3 run reported."""

    model.eval()
    data, base = prepared["data"], prepared["base_cfg"]
    gate = torch.ones(data["n_users"], dtype=torch.float32, device=v3.DEVICE)
    frame = pd.DataFrame({"u_idx": np.asarray(prepared["cache"].users, np.int64)})
    for k in cfg.ks:
        result = v3.evaluate(
            model, 0.0, gate, prepared["cache"], prepared["meta"], [k],
            data["csr_ptr"], data["csr_items"], base, per_user=True,
        )
        for metric in PER_USER_METRICS:
            frame[f"{metric}@{k}"] = np.asarray(result["per_user"][metric], float)
    return frame


def customer_table(prepared: dict, cfg: CustomerGainConfig) -> pd.DataFrame:
    """M1 level and per-arm gain for every evaluation customer and seed."""

    signals = prepared["signals"]
    weighted = {}
    for spec in graph.arm_specifications():
        arm = graph.build_arm_graph(prepared, _gate_cfg(cfg), spec)
        weighted[spec["model_id"]] = v3.build_adj(
            signals["edge_users"], signals["edge_items"],
            arm["weights"].astype(np.float32),
            prepared["data"]["n_users"], prepared["data"]["n_items"],
        )

    rows = []
    for seed in cfg.seeds:
        print(f"\n[seed {seed}] 체크포인트")
        sources = {
            M1_MODEL_ID: (Path(cfg.m1_result_dir), "capacity_search_dev",
                          f"baseline_{M1_MODEL_ID}", prepared["data"]["adj"]),
            ARM_VALUE: (Path(cfg.m3_result_dir), "centered_graph_dev",
                        ARM_VALUE, weighted[ARM_VALUE]),
            ARM_VALUE_ACTIVITY: (Path(cfg.m3_result_dir), "centered_graph_dev",
                                 ARM_VALUE_ACTIVITY, weighted[ARM_VALUE_ACTIVITY]),
        }
        scored = {}
        for model_id, (root, stage, stem, adjacency) in sources.items():
            state = load_state(_checkpoint(root, stage, stem, seed), cfg)
            model = _model(prepared, cfg, adjacency)
            model.load_state_dict(state["model_state"])
            scored[model_id] = per_customer(model, prepared, cfg)

        base = scored[M1_MODEL_ID]
        for model_id in (ARM_VALUE, ARM_VALUE_ACTIVITY):
            frame = base[["u_idx"]].copy()
            frame["seed"], frame["model_id"] = seed, model_id
            for column in base.columns.drop("u_idx"):
                frame[f"m1_{column}"] = base[column].to_numpy()
                frame[f"gain_{column}"] = (scored[model_id][column].to_numpy()
                                           - base[column].to_numpy())
            rows.append(frame)
    table = pd.concat(rows, ignore_index=True)
    for axis in AXES:
        table[axis] = np.asarray(prepared[axis], float)[table.u_idx.to_numpy()]
    table["clv_valid"] = np.asarray(prepared["clv_valid"], bool)[table.u_idx.to_numpy()]
    return table


def _gate_cfg(cfg: CustomerGainConfig):
    """The gate limits belong to the M3 run; reuse them rather than restate them."""

    return graph.configure_centered_graph(
        seeds=cfg.seeds, id_dim=cfg.id_dim, n_layers=cfg.n_layers,
        pref_reg=cfg.pref_reg, out_dir=cfg.out_dir, m1_result_dir=cfg.m1_result_dir,
    )


# --------------------------------------------------------------------------
# reading
# --------------------------------------------------------------------------


def correlations(table: pd.DataFrame, cfg: CustomerGainConfig) -> pd.DataFrame:
    metrics = [f"{m}@{k}" for m in PER_USER_METRICS for k in cfg.ks]
    rows = []
    for (model_id, seed), part in table.groupby(["model_id", "seed"], sort=False):
        for metric in metrics:
            for axis in AXES:
                for kind in ("m1", "gain"):
                    column = f"{kind}_{metric}"
                    rho = spearmanr(part[column], part[axis]).statistic
                    rows.append({"model_id": model_id, "seed": seed, "metric": metric,
                                 "axis": axis, "quantity": kind, "spearman": float(rho)})
    return pd.DataFrame(rows)


def decile_table(table: pd.DataFrame, cfg: CustomerGainConfig, axis: str = "q_n") -> pd.DataFrame:
    metrics = [f"{m}@{k}" for m in PER_USER_METRICS for k in cfg.ks]
    rows = []
    for (model_id, seed), part in table.groupby(["model_id", "seed"], sort=False):
        bins = pd.qcut(part[axis].rank(method="first"), cfg.deciles, labels=False)
        for decile, group in part.groupby(bins):
            row = {"model_id": model_id, "seed": seed, f"{axis}_decile": int(decile) + 1,
                   "customers": int(len(group)), f"{axis}_mean": float(group[axis].mean())}
            for metric in metrics:
                row[f"m1_{metric}"] = float(group[f"m1_{metric}"].mean())
                row[f"gain_{metric}"] = float(group[f"gain_{metric}"].mean())
                row[f"improved_share_{metric}"] = float((group[f"gain_{metric}"] > 0).mean())
            rows.append(row)
    return pd.DataFrame(rows)


def gain_reading(correlation: pd.DataFrame, cfg: CustomerGainConfig) -> dict:
    """Apply the fixed thresholds; say nothing when the seeds disagree."""

    reading = {"thresholds": {"room": ROOM_THRESHOLD, "redundant": REDUNDANT_THRESHOLD},
               "seeds": list(cfg.seeds), "trains_nothing": True}
    for model_id in (ARM_VALUE, ARM_VALUE_ACTIVITY):
        for metric in [f"{m}@{k}" for m in PER_USER_METRICS for k in cfg.ks]:
            for axis in AXES:
                part = correlation[
                    correlation.model_id.eq(model_id) & correlation.metric.eq(metric)
                    & correlation.axis.eq(axis) & correlation.quantity.eq("gain")]
                values = part.spearman.to_numpy()
                if len(values) != len(cfg.seeds):
                    raise RuntimeError(f"{model_id}/{metric}/{axis} 시드 수 {len(values)}")
                if all(abs(v) < ROOM_THRESHOLD for v in values):
                    verdict = "room_for_a_customer_weight"
                elif all(v > REDUNDANT_THRESHOLD for v in values):
                    verdict = "redundant_with_m3"
                else:
                    verdict = "no_direction_claimed"
                reading[f"{model_id}|{metric}|{axis}"] = {
                    "spearman_per_seed": [round(float(v), 4) for v in values],
                    "mean": round(float(values.mean()), 4), "verdict": verdict,
                }
    return reading


def run_customer_gain(cfg: CustomerGainConfig | None = None) -> pd.DataFrame:
    cfg = cfg or configure_customer_gain()
    print(json.dumps(preflight_summary(cfg), ensure_ascii=False, indent=2))
    prepared = graph._prepare(_gate_cfg(cfg))
    table = customer_table(prepared, cfg)
    correlation = correlations(table, cfg)
    deciles = pd.concat([decile_table(table, cfg, axis) for axis in ("q_n", "q_v")],
                        ignore_index=True, sort=False)
    reading = gain_reading(correlation, cfg)

    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"clv_m3_customer_gain_{prepared['config_hash']}"
    paths = {"customer_csv": out / f"{stem}_customer.csv",
             "correlation_csv": out / f"{stem}_correlation.csv",
             "decile_csv": out / f"{stem}_decile.csv",
             "json": out / f"{stem}.json"}
    test10._atomic_csv(paths["customer_csv"], table)
    test10._atomic_csv(paths["correlation_csv"], correlation)
    test10._atomic_csv(paths["decile_csv"], deciles)
    test10._atomic_json(paths["json"], {
        "code_version": CODE_VERSION, "config": asdict(cfg),
        "preflight": preflight_summary(cfg), "source_revision": prepared["revision"],
        "correlation": correlation.to_dict("records"),
        "reading": reading,
        "result_paths": {name: str(path) for name, path in paths.items()},
    })
    print("\n판독:", json.dumps(reading, ensure_ascii=False, indent=2))
    print("저장:", json.dumps({k: str(v) for k, v in paths.items()}, ensure_ascii=False))
    correlation.attrs.update(reading=reading, decile_records=deciles.to_dict("records"),
                             result_paths={k: str(v) for k, v in paths.items()})
    return correlation


def decile_frame(correlation: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(correlation.attrs["decile_records"])


if __name__ == "__main__":
    print(json.dumps(preflight_summary(configure_customer_gain()),
                     ensure_ascii=False, indent=2))
