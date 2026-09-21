"""Is M2 behind M1 because it is under-trained, or because of its design?

The advisor asked whether the M2 and M3 deficits are an underfitting artifact:
the protocol stops at a fixed 100 epochs, and the training loss was still
falling there (M1 seed 42: 0.1435 at epoch 50, 0.1127 at epoch 100, no
plateau).  Nothing in the project ever evaluated an intermediate epoch, so we
cannot say whether development accuracy was still rising, already flat, or
past its peak.

This runner trains to 300 epochs and evaluates every ``eval_every`` epochs, so
one run yields the whole learning curve.  Four conditions separate the knobs
the advisor named, one change at a time, with M1 retrained under every shared
setting so each comparison stays paired:

* ``baseline``       - the protocol as run so far (64 dims, L2 1e-3, rho 0.05)
* ``wide``           - capacity: 128 dims
* ``light_l2``       - regularization: L2 1e-4
* ``strong_signal``  - intervention strength: rho 0.10 (the model's own cap)

The epoch axis is free inside a run, so it is never a grid dimension.  No
condition is selected here: seed 42 screens the curves, and any shortlisted
condition goes to five seeds before a judgment is made (rule D1).
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

from clv_history_item_fit_model import HistoryItemFitLightGCN
from clv_m5_n_conditioned_value_basis_model import M5NConditionedValueBasisLightGCN
from clv_run_state import ProgressStore, RunIdentity
import lightgcn_clv_axis_specific_test10 as test10
import lightgcn_clv_component_recheck as recheck
import lightgcn_clv_m4_clv_hard_negative as m4_helpers
import lightgcn_clv_moe as moe
import lightgcn_clv_v3 as v3


CODE_VERSION = "clv-m2-capacity-search-dev-v1"
PROTOCOL_EPOCH = 100          # where every earlier comparison was taken
M1_MODEL_ID = "m1_bpr_k1"
M2_MODEL_ID = "m2_nv_history_fit_bpr_k1"


@dataclass(frozen=True)
class Condition:
    name: str
    hypothesis: str
    id_dim: int
    pref_reg: float
    rho: float
    axis_dim: int

    @property
    def shared_key(self) -> str:
        """M1 depends only on the settings it shares; it has no CLV block."""

        return f"dim{self.id_dim}_l2{self.pref_reg:g}"


CONDITIONS = (
    Condition("baseline", "training_budget", 64, 1e-3, 0.05, 4),
    Condition("wide", "id_capacity", 128, 1e-3, 0.05, 4),
    Condition("axis_wide", "clv_block_capacity", 64, 1e-3, 0.05, 8),
    Condition("light_l2", "regularization", 64, 1e-4, 0.05, 4),
    Condition("strong_signal", "intervention_strength", 64, 1e-3, 0.10, 4),
)


@dataclass(frozen=True)
class CapacitySearchConfig:
    conditions: tuple[str, ...] = tuple(c.name for c in CONDITIONS)
    seeds: tuple[int, ...] = (42,)
    epochs: int = 300
    eval_every: int = 25
    batch_size: int = 8192
    lr: float = 5e-4
    n_layers: int = 2
    negative_count: int = 1
    out_dir: str = ""


def configure_capacity_search(**overrides) -> CapacitySearchConfig:
    defaults = {
        "out_dir": f"{v3.default_out_dir('dunnhumby')}_clv_m2_capacity_search_v1"
    }
    return validate_config(CapacitySearchConfig(**(defaults | overrides)))


def validate_config(cfg: CapacitySearchConfig) -> CapacitySearchConfig:
    if cfg.epochs < PROTOCOL_EPOCH * 2:
        raise ValueError("학습 예산 가설을 보려면 기존 100 epoch의 두 배 이상이어야 합니다")
    if PROTOCOL_EPOCH % cfg.eval_every:
        raise ValueError("기존 100 epoch 지점이 평가 격자에 포함돼야 비교가 됩니다")
    if cfg.negative_count != 1:
        raise ValueError("K=1 기준 음성표본을 바꾸지 않습니다")
    known = {c.name for c in CONDITIONS}
    if not cfg.conditions or not set(cfg.conditions).issubset(known):
        raise ValueError(f"조건 이름은 {sorted(known)} 안에서 골라야 합니다")
    if CONDITIONS[0].name not in cfg.conditions:
        raise ValueError("기준 조건이 빠지면 다른 조건을 비교할 대상이 없습니다")
    if not cfg.out_dir:
        raise ValueError("out_dir가 필요합니다")
    return cfg


def evaluation_epochs(cfg: CapacitySearchConfig) -> list[int]:
    grid = list(range(cfg.eval_every, cfg.epochs + 1, cfg.eval_every))
    if cfg.epochs not in grid:
        grid.append(cfg.epochs)
    return grid


def arm_specifications(cfg: CapacitySearchConfig) -> list[dict]:
    """One M2 arm per condition, and one M1 arm per distinct shared setting."""

    arms, seen = [], set()
    for condition in (c for c in CONDITIONS if c.name in cfg.conditions):
        if condition.shared_key not in seen:
            seen.add(condition.shared_key)
            arms.append(
                {
                    "model_id": M1_MODEL_ID,
                    "condition": condition.name,
                    "hypothesis": condition.hypothesis,
                    "shared_key": condition.shared_key,
                    "id_dim": condition.id_dim,
                    "pref_reg": condition.pref_reg,
                    "rho": 0.0,
                    "axis_dim": 0,
                }
            )
        arms.append(
            {
                "model_id": M2_MODEL_ID,
                "condition": condition.name,
                "hypothesis": condition.hypothesis,
                "shared_key": condition.shared_key,
                "id_dim": condition.id_dim,
                "pref_reg": condition.pref_reg,
                "rho": condition.rho,
                "axis_dim": condition.axis_dim,
            }
        )
    return arms


def shared_baseline(spec: dict) -> str:
    return spec["shared_key"]


def preflight_summary(cfg: CapacitySearchConfig) -> dict:
    cfg = validate_config(cfg)
    return {
        "code_version": CODE_VERSION,
        "question": (
            "is the M2 deficit against M1 an artifact of stopping at a fixed "
            "100 epochs, of too little capacity, of too much regularization, "
            "or of too weak an intervention?"
        ),
        "dataset": "dunnhumby",
        "split": "historical_development_days_684_690",
        "seeds": list(cfg.seeds),
        "epochs": cfg.epochs,
        "evaluated_at_epochs": evaluation_epochs(cfg),
        "protocol_epoch": PROTOCOL_EPOCH,
        "conditions": {c.name: asdict(c) for c in CONDITIONS if c.name in cfg.conditions},
        "hypotheses": sorted(
            {c.hypothesis for c in CONDITIONS if c.name in cfg.conditions}
        ),
        "not_an_underfitting_test": [
            c.name
            for c in CONDITIONS
            if c.name in cfg.conditions and c.hypothesis == "intervention_strength"
        ],
        "clv_entry_point": (
            "q_N(u) and q_V(u), the customer's two CLV axis percentiles from the "
            "last 365 training days, scale that customer's own purchase-history "
            "block; the block enters the score during training and evaluation "
            "alike, never as a post-hoc correction. q_C is not used (approved "
            "2026-09-16). Training uses leave-one-out history profiles so the "
            "positive item cannot see its own share; evaluation uses the full "
            "history."
        ),
        "clv_score_share_recorded": True,
        "m1_retrained_per_shared_setting": True,
        "loss": {"negative_count": cfg.negative_count, "hard_negative": False},
        "reading": (
            "report the curve; a condition is only shortlisted here and must be "
            "confirmed on five seeds before any judgment (rule D1)"
        ),
        "limits": (
            "one screening seed on a repeatedly exposed development split; no "
            "epoch and no condition is selected, no significance is claimed"
        ),
    }


def _config_hash(cfg: CapacitySearchConfig, input_hash: str) -> str:
    # The condition list and the seed list stay out of the hash so that running
    # one condition first and adding the rest later reuses the finished curves.
    payload = {
        "code_version": CODE_VERSION,
        "protocol": {
            field: getattr(cfg, field)
            for field in asdict(cfg)
            if field not in {"out_dir", "seeds", "conditions"}
        },
        "input_hash": input_hash,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()[:12]


def _prepare(cfg: CapacitySearchConfig) -> dict:
    prepared = recheck._prepare(recheck.configure_component_recheck())
    prepared["out_dir"] = Path(cfg.out_dir)
    prepared["config_hash"] = _config_hash(cfg, prepared["input_hash"])
    return prepared


def _build_model(prepared: dict, cfg: CapacitySearchConfig, spec: dict, seed: int):
    data = prepared["data"]
    valid = np.asarray(prepared["clv_valid"], dtype=bool)
    v3.set_seed(seed)
    if spec["model_id"] == M2_MODEL_ID:
        model = HistoryItemFitLightGCN(
            n_users=data["n_users"],
            n_items=data["n_items"],
            history=prepared["history"],
            q_n=np.where(valid, prepared["q_n"], 0.0).astype(np.float32),
            q_v=np.where(valid, prepared["q_v"], 0.0).astype(np.float32),
            activity_valid=valid,
            value_valid=valid,
            adj=data["adj"],
            id_dim=spec["id_dim"],
            axis_dim=spec["axis_dim"],
            n_layers=cfg.n_layers,
            rho=spec["rho"],
            pref_reg=spec["pref_reg"],
        )
    else:
        model = M5NConditionedValueBasisLightGCN(
            n_users=data["n_users"],
            n_items=data["n_items"],
            user_q_n=prepared["q_n"],
            user_q_v=prepared["q_v"],
            user_q_c=prepared["q_c"],
            user_clv_valid=valid,
            item_price_percentile=prepared["item_amount_percentile"],
            item_price_valid=prepared["item_economic_valid"],
            adj=data["adj"],
            id_dim=spec["id_dim"],
            rho=0.0,
            n_layers=cfg.n_layers,
            pref_reg=spec["pref_reg"],
            economic_propagation=False,
        )
    return model.to(v3.DEVICE)


def _arm_paths(prepared: dict, spec: dict, seed: int) -> dict[str, Path]:
    root = prepared["out_dir"] / "arms" / prepared["config_hash"]
    stem = f"{spec['condition']}_{spec['model_id']}_s{seed}"
    return {"result": root / f"{stem}.json"}


@torch.no_grad()
def _clv_score_share(model, prepared: dict, cfg: CapacitySearchConfig) -> dict:
    """How much of the score the CLV-scaled block carries where it decides.

    Measured on the items the model actually recommends: evaluation customers,
    their own training items masked out exactly as in evaluation, the top ten
    taken from the full catalogue.  Scoring training positives instead would
    flatter the block, because a customer's history profile contains that very
    item and would meet its own target vector.
    """

    cache, data = prepared["cache"], prepared["data"]
    csr_ptr, csr_items = data["csr_ptr"], data["csr_items"]
    rng = np.random.default_rng(0)
    sample = cache.users[
        rng.choice(len(cache.users), size=min(256, len(cache.users)), replace=False)
    ]
    user_vectors, item_vectors, *_ = model.embeddings()
    cut = model.id_dim
    rows = torch.as_tensor(sample, dtype=torch.long, device=v3.DEVICE)
    scores = user_vectors[rows] @ item_vectors.T
    for offset, user in enumerate(sample):
        lo, hi = csr_ptr[user], csr_ptr[user + 1]
        if hi > lo:
            scores[offset, csr_items[lo:hi]] = -1e9
    top = scores.topk(10, dim=1).indices
    picked_users = rows[:, None].expand_as(top).reshape(-1)
    picked_items = top.reshape(-1)
    id_part = (
        user_vectors[picked_users, :cut] * item_vectors[picked_items, :cut]
    ).sum(dim=1)
    clv_part = (
        user_vectors[picked_users, cut:] * item_vectors[picked_items, cut:]
    ).sum(dim=1)
    return {
        "id_score_mean_abs": float(id_part.abs().mean()),
        "clv_score_mean_abs": float(clv_part.abs().mean()),
        "clv_score_share": float(
            clv_part.abs().mean() / (id_part.abs().mean() + clv_part.abs().mean() + 1e-12)
        ),
        "clv_score_measured_on": "top10_of_evaluation_users",
    }


@torch.no_grad()
def _evaluate(model, prepared: dict) -> dict:
    model.eval()
    metrics, _ = moe._flat_evaluation(
        model, 0.0, prepared["cache"], prepared["meta"], prepared["data"],
        prepared["base_cfg"], per_user=False,
    )
    model.train()
    return test10._public_metrics(metrics)


def _train_curve(
    model, prepared: dict, cfg: CapacitySearchConfig, spec: dict, seed: int,
    store: ProgressStore,
) -> list[dict]:
    """Train to cfg.epochs, recording development metrics along the way."""

    data = prepared["data"]
    tr_u, tr_i, positive_keys = data["tr_u"], data["tr_i"], data["pos_key"]
    n_batches = math.ceil(len(tr_u) / cfg.batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    rng = np.random.default_rng(seed)
    checkpoints = set(evaluation_epochs(cfg))

    restored = store.restore_epoch(model, optimizer, rng)
    start_epoch = 1 if restored is None else int(restored["next_epoch"])
    curve = list(restored.get("history", [])) if restored else []
    if restored is not None:
        print(f"  [{spec['condition']}/{spec['model_id']} s{seed}] epoch {start_epoch - 1} 재개")
    store.mark_stage("running", epoch=start_epoch - 1, max_epoch=cfg.epochs, selection="none")

    started = time.time()
    for epoch in range(start_epoch, cfg.epochs + 1):
        permutation = rng.permutation(len(tr_u))
        totals = {"loss": 0.0, "p_correct": 0.0}
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
            loss, _, correct = recheck._batch_loss(model, users, positives, negatives, None)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            totals["loss"] += float(loss.detach())
            totals["p_correct"] += correct
            store.heartbeat(epoch=epoch, max_epoch=cfg.epochs, batch=batch + 1,
                            batches=n_batches, loss=totals["loss"] / (batch + 1),
                            selection="none")
        record = {
            "epoch": epoch,
            "loss": totals["loss"] / n_batches,
            "p_correct": totals["p_correct"] / n_batches,
        }
        if epoch in checkpoints:
            record["metrics"] = _evaluate(model, prepared)
            record["score_split"] = _clv_score_share(model, prepared, cfg)
            print(
                f"  [{spec['condition']}/{spec['model_id']} s{seed}] ep {epoch:3d} | "
                f"loss {record['loss']:.4f} | recall@10 {record['metrics']['recall@10']:.6f} | "
                f"ndcg@10 {record['metrics']['ndcg@10']:.6f}"
            )
        curve.append(record)
        store.save_epoch(model, optimizer, rng, epoch=epoch, history=curve,
                         wall_clock_sec=time.time() - started, selection="none")
    return curve


def _run_arm(prepared: dict, cfg: CapacitySearchConfig, spec: dict, seed: int) -> dict:
    paths = _arm_paths(prepared, spec, seed)
    if paths["result"].exists():
        payload = json.loads(paths["result"].read_text(encoding="utf-8"))
        print(f"  [cached] {spec['condition']}/{spec['model_id']} s{seed} 곡선 재사용")
        return payload

    model = _build_model(prepared, cfg, spec, seed)
    store = ProgressStore(
        prepared["out_dir"] / "progress" / prepared["config_hash"],
        RunIdentity(
            stage="capacity_search_dev",
            model_id=f"{spec['condition']}_{spec['model_id']}",
            seed=seed,
            config_hash=prepared["config_hash"],
            source_revision=prepared["revision"],
            input_hash=prepared["input_hash"],
        ),
    )
    curve = _train_curve(model, prepared, cfg, spec, seed, store)
    payload = {
        **spec,
        "seed": seed,
        "code_version": CODE_VERSION,
        "source_revision": prepared["revision"],
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "curve": curve,
    }
    test10._atomic_json(paths["result"], payload)
    store.mark_complete(epoch=cfg.epochs, max_epoch=cfg.epochs, selection="none",
                        checkpoint_path="", result_path=str(paths["result"]))
    return payload


def curve_table(arms: list[dict]) -> pd.DataFrame:
    rows = []
    for arm in arms:
        for record in arm["curve"]:
            if "metrics" not in record:
                continue
            rows.append(
                {
                    "condition": arm["condition"],
                    "hypothesis": arm["hypothesis"],
                    "shared_key": arm["shared_key"],
                    "model_id": arm["model_id"],
                    "id_dim": arm["id_dim"],
                    "axis_dim": arm["axis_dim"],
                    "pref_reg": arm["pref_reg"],
                    "rho": arm["rho"],
                    "seed": arm["seed"],
                    "epoch": record["epoch"],
                    "loss": record["loss"],
                    "p_correct": record["p_correct"],
                    **{k: v for k, v in record.get("score_split", {}).items()
                       if k != "clv_score_measured_on"},
                    **record["metrics"],
                }
            )
    return pd.DataFrame(rows)


def gap_table(curve: pd.DataFrame) -> pd.DataFrame:
    """M2 minus the M1 trained under the same shared setting, at every epoch.

    M1 is trained once per shared setting, not once per condition, so the pair
    is found by ``shared_key``.  Matching on the condition name instead would
    silently drop every condition that only changes an M2-side knob.
    """

    metrics = [c for c in curve.columns if "@" in c]
    baselines = curve[curve.model_id.eq(M1_MODEL_ID)].set_index(
        ["shared_key", "seed", "epoch"]
    )
    rows = []
    for _, arm in curve[curve.model_id.eq(M2_MODEL_ID)].iterrows():
        key = (arm["shared_key"], arm["seed"], arm["epoch"])
        if key not in baselines.index:
            raise KeyError(
                f"{arm['condition']}에 대응하는 M1({arm['shared_key']}, seed {arm['seed']}, "
                f"epoch {arm['epoch']}) 결과가 없습니다"
            )
        reference = baselines.loc[key]
        if isinstance(reference, pd.DataFrame):
            reference = reference.iloc[0]
        row = {
            "condition": arm["condition"],
            "hypothesis": arm["hypothesis"],
            "shared_key": arm["shared_key"],
            "seed": arm["seed"],
            "epoch": arm["epoch"],
        }
        for metric in metrics:
            base, value = float(reference[metric]), float(arm[metric])
            row[metric] = value - base
            row[f"{metric}_ratio"] = value / base if base else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def search_reading(curve: pd.DataFrame, gap: pd.DataFrame, cfg: CapacitySearchConfig) -> dict:
    def at(frame: pd.DataFrame, epoch: int, metric: str) -> float:
        row = frame[frame.epoch.eq(epoch)]
        return float(row.iloc[0][metric]) if len(row) else float("nan")

    baseline_key = CONDITIONS[0].shared_key
    baseline_m1 = curve[curve.shared_key.eq(baseline_key) & curve.model_id.eq(M1_MODEL_ID)]
    peak_epoch = int(baseline_m1.loc[baseline_m1["recall@10"].idxmax(), "epoch"])
    baseline_gap = gap[gap.condition.eq("baseline")]
    reading = {
        "m1_peak_epoch": peak_epoch,
        "m1_still_improving_after_protocol_epoch": peak_epoch > PROTOCOL_EPOCH,
        "m1_recall10_at_protocol_epoch": at(baseline_m1, PROTOCOL_EPOCH, "recall@10"),
        "m1_recall10_at_end": at(baseline_m1, cfg.epochs, "recall@10"),
        "m2_gap_at_protocol_epoch": at(baseline_gap, PROTOCOL_EPOCH, "recall@10"),
        "m2_gap_at_end": at(baseline_gap, cfg.epochs, "recall@10"),
        "conditions_closing_the_gap": sorted(
            {
                str(condition)
                for condition in gap.condition.unique()
                if at(gap[gap.condition.eq(condition)], cfg.epochs, "recall@10")
                > at(baseline_gap, PROTOCOL_EPOCH, "recall@10")
            }
        ),
        "condition_selected": False,
        "significance_claimed": False,
        "seeds_used": sorted(curve.seed.unique().tolist()),
    }
    reading["gap_closes_with_training"] = (
        reading["m2_gap_at_end"] > reading["m2_gap_at_protocol_epoch"]
    )
    return reading


def run_capacity_search(cfg: CapacitySearchConfig | None = None) -> pd.DataFrame:
    cfg = validate_config(cfg or configure_capacity_search())
    print(json.dumps(preflight_summary(cfg), ensure_ascii=False, indent=2))
    prepared = _prepare(cfg)
    arms = []
    for seed in cfg.seeds:
        for spec in arm_specifications(cfg):
            print(f"\n===== {spec['condition']} | {spec['model_id']} | seed {seed} "
                  f"| dim {spec['id_dim']} | L2 {spec['pref_reg']} | rho {spec['rho']} =====")
            arms.append(_run_arm(prepared, cfg, spec, seed))
    curve = curve_table(arms)
    gap = gap_table(curve)
    reading = search_reading(curve, gap, cfg)

    out = Path(cfg.out_dir)
    stem = f"clv_m2_capacity_search_{prepared['config_hash']}"
    paths = {
        "curve_csv": out / f"{stem}_curve.csv",
        "gap_csv": out / f"{stem}_gap.csv",
        "json": out / f"{stem}.json",
    }
    test10._atomic_csv(paths["curve_csv"], curve)
    test10._atomic_csv(paths["gap_csv"], gap)
    test10._atomic_json(
        paths["json"],
        {
            "code_version": CODE_VERSION,
            "config": asdict(cfg),
            "preflight": preflight_summary(cfg),
            "source_revision": prepared["revision"],
            "curve": curve.to_dict("records"),
            "gap": gap.to_dict("records"),
            "reading": reading,
            "result_paths": {name: str(path) for name, path in paths.items()},
        },
    )
    print("\n학습 곡선 (평가 지점):")
    print(curve[["condition", "model_id", "seed", "epoch", "loss", "recall@10", "ndcg@10"]]
          .to_string(index=False))
    print("\nM2 - M1 (같은 조건, 같은 시드):")
    print(gap[["condition", "seed", "epoch", "recall@10", "ndcg@10"]].to_string(index=False))
    print("\n판독:", json.dumps(reading, ensure_ascii=False, indent=2))
    curve.attrs.update(gap=gap, reading=reading,
                       result_paths={k: str(v) for k, v in paths.items()})
    return curve


if __name__ == "__main__":
    print(json.dumps(preflight_summary(configure_capacity_search()), ensure_ascii=False, indent=2))
