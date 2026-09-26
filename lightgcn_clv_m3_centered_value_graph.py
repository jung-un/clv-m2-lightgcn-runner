"""A graph-side CLV intervention built from what every earlier M3 got wrong.

Three measurements shaped this design, all taken before any training:

* The previous value graph weighted an edge by the item's share of that
  customer's spend.  Expensive items take a large share of *everyone's* spend,
  so the item-side mean weight correlated with price at ``+0.881``: the
  propagation pushed every customer toward expensive items and the alignment
  between a customer's own price tendency and what they were shown fell 52%.
  Centering the signal within the item as well as within the customer drops
  that correlation to ``+0.095``.
* The candidate-specific M3 changed 99.86% of the edges with a coefficient
  standard deviation of ``0.0119`` - it touched everything by nothing.  Here
  the single strength is calibrated so the weights reach a fixed coefficient
  of variation.
* Weighting by the repeat-transaction axis raised the edge mass of the most
  active tenth of customers from 27.15% to 30.15%, undoing part of the degree
  normalisation LightGCN relies on.  Normalising each customer's weights to
  mean one keeps their weighted degree exactly equal to the binary degree,
  because ``build_adj`` normalises by weighted degree.

Two arms separate the two CLV axes.  Neither axis is rescaled to a common
size: the value axis and the activity axis enter with their own dispersion, so
a dataset where customers rarely rebuy the same item contributes almost no
activity signal without anyone setting a per-dataset coefficient.  On
Dunnhumby the activity axis has 45% of the value axis's per-customer spread;
on H&M it has 5%, and a third of customers have none at all.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from clv_m5_n_conditioned_value_basis_model import M5NConditionedValueBasisLightGCN
from clv_run_state import ProgressStore, RunIdentity
import lightgcn_clv_axis_specific_test10 as test10
import lightgcn_clv_component_recheck as recheck
import lightgcn_clv_m2_capacity_search as capacity
import lightgcn_clv_v3 as v3


CODE_VERSION = "clv-m3-centered-value-graph-dev-v1"
PROTOCOL_EPOCH = capacity.PROTOCOL_EPOCH          # 100, where every earlier comparison was taken
BALANCE_ROUNDS = 100                              # 양쪽 여백이 동시에 1로 수렴하기까지의 반복 횟수
M1_MODEL_ID = capacity.M1_MODEL_ID
ARM_VALUE = "m3_centered_value_graph_bpr_k1"
ARM_VALUE_ACTIVITY = "m3_centered_value_activity_graph_bpr_k1"


@dataclass(frozen=True)
class CenteredGraphConfig:
    seeds: tuple[int, ...] = (42, 43, 44)
    epochs: int = 300
    eval_every: int = 25
    batch_size: int = 8192
    lr: float = 5e-4
    n_layers: int = 2
    id_dim: int = 64
    pref_reg: float = 1e-3
    negative_count: int = 1
    target_cv: float = 0.20
    max_degree_correlation: float = 0.05
    max_price_correlation: float = 0.20
    max_popularity_correlation: float = 0.20
    m1_result_dir: str = ""
    allow_baseline_training: tuple[str, ...] = ()
    out_dir: str = ""


def configure_centered_graph(**overrides) -> CenteredGraphConfig:
    root = v3.default_out_dir("dunnhumby")
    defaults = {
        "out_dir": f"{root}_clv_m3_centered_value_graph_v1",
        "m1_result_dir": f"{root}_clv_m2_capacity_search_v1",
    }
    return validate_config(CenteredGraphConfig(**(defaults | overrides)))


def validate_config(cfg: CenteredGraphConfig) -> CenteredGraphConfig:
    if cfg.epochs < PROTOCOL_EPOCH * 2 or PROTOCOL_EPOCH % cfg.eval_every:
        raise ValueError("100 epoch 지점이 평가 격자에 있고 그 두 배 이상 학습해야 합니다")
    if cfg.negative_count != 1:
        raise ValueError("K=1 기준 음성표본을 바꾸지 않습니다")
    if not 0.05 <= cfg.target_cv <= 0.5:
        raise ValueError("목표 변동계수가 비현실적입니다")
    if not cfg.out_dir or not cfg.m1_result_dir:
        raise ValueError("out_dir와 m1_result_dir가 필요합니다")
    return cfg


def arm_specifications() -> list[dict]:
    return [
        {"model_id": ARM_VALUE, "arm": "value_only", "gamma": 0.0,
         "question": "가치축만으로 전파를 바꾸면 M1을 넘는가"},
        {"model_id": ARM_VALUE_ACTIVITY, "arm": "value_and_activity", "gamma": 1.0,
         "question": "반복거래축을 함께 넣으면 가치축만일 때보다 나아지는가"},
    ]


def preflight_summary(cfg: CenteredGraphConfig) -> dict:
    cfg = validate_config(cfg)
    return {
        "code_version": CODE_VERSION,
        "dataset": "dunnhumby",
        "split": "historical_development_days_684_690",
        "seeds": list(cfg.seeds),
        "epochs": cfg.epochs,
        "evaluated_at_epochs": capacity.evaluation_epochs(cfg),
        "reported_epochs": [PROTOCOL_EPOCH, cfg.epochs],
        "edge_weight": (
            "exp(beta * [q_V(u)*z_V(u,i) + gamma * q_N(u)*z_N(u,i)]), normalised to "
            "mean one on both the customer and the item side. z is the log share of that customer's "
            "spend (value axis) or baskets (activity axis), centered within the "
            "customer and then within the item."
        ),
        "axes_not_rescaled_to_each_other": True,
        "single_strength_rule": (
            f"beta is calibrated so the edge weights reach a coefficient of "
            f"variation of {cfg.target_cv}; it is never chosen from results"
        ),
        "gates_before_training": {
            "coefficient_of_variation": [0.15, 0.35],
            "customer_weight_vs_degree": cfg.max_degree_correlation,
            "item_weight_vs_price": cfg.max_price_correlation,
            "item_weight_vs_popularity": cfg.max_popularity_correlation,
        },
        "arms": arm_specifications(),
        "m1": "reused from the capacity-search curves; never retrained silently",
        "reading": (
            "arm - M1 at the two fixed epochs, and arm B - arm A for the activity "
            "axis; no epoch and no arm is selected after seeing results"
        ),
        "limits": (
            "three seeds on a repeatedly exposed development split; no significance, "
            "no CLV attribution and no generalization is claimed"
        ),
    }


# --------------------------------------------------------------------------
# edge signals and weights
# --------------------------------------------------------------------------


def _double_center(mass: np.ndarray, users: np.ndarray, items: np.ndarray,
                   n_users: int, n_items: int) -> np.ndarray:
    """log share within the customer, centered within customer then within item."""

    user_total = np.bincount(users, weights=mass, minlength=n_users)
    share = mass / np.maximum(user_total[users], 1e-12)
    x = np.log(share + 1e-9)
    user_count = np.maximum(np.bincount(users, minlength=n_users), 1)
    x = x - (np.bincount(users, weights=x, minlength=n_users) / user_count)[users]
    item_count = np.maximum(np.bincount(items, minlength=n_items), 1)
    return x - (np.bincount(items, weights=x, minlength=n_items) / item_count)[items]


def centered_edge_signals(train: pd.DataFrame, n_users: int, n_items: int) -> dict:
    missing = {"u_idx", "i_idx", "b_raw", "v"}.difference(train.columns)
    if missing:
        raise ValueError(f"엣지 신호 입력 열 누락: {sorted(missing)}")
    pair = (
        train.assign(positive_value=train["v"].clip(lower=0.0))
        .groupby(["u_idx", "i_idx"], sort=True)
        .agg(value=("positive_value", "sum"), baskets=("b_raw", "nunique"))
        .reset_index()
    )
    users = pair.u_idx.to_numpy(np.int64)
    items = pair.i_idx.to_numpy(np.int64)
    return {
        "edge_users": users,
        "edge_items": items,
        "n_users": n_users,
        "n_items": n_items,
        "z_value": _double_center(pair.value.to_numpy(float), users, items, n_users, n_items),
        "z_activity": _double_center(
            pair.baskets.to_numpy(float), users, items, n_users, n_items
        ),
    }


def _balance_margins(weights: np.ndarray, signals: dict,
                     rounds: int = BALANCE_ROUNDS) -> np.ndarray:
    """Scale customers and items in turn until both margins average one.

    Normalising only inside each customer keeps their weighted degree at the
    binary one, but it leaves the item side tilted: when a customer holds few
    edges their scale factor tracks what they happen to buy, which on H&M put
    the item weight-price correlation at +0.77. Alternating the two margins
    (Sinkhorn) removes both tilts at once, and by the last round the customer
    mass is preserved as well.
    """

    users, items = signals["edge_users"], signals["edge_items"]
    n_users, n_items = signals["n_users"], signals["n_items"]
    user_count = np.maximum(np.bincount(users, minlength=n_users), 1)
    item_count = np.maximum(np.bincount(items, minlength=n_items), 1)
    for _ in range(rounds):
        mean = np.bincount(users, weights=weights, minlength=n_users) / user_count
        weights = weights / np.maximum(mean[users], 1e-12)
        mean = np.bincount(items, weights=weights, minlength=n_items) / item_count
        weights = weights / np.maximum(mean[items], 1e-12)
    return weights


def edge_weights(signals: dict, q_value: np.ndarray, q_activity: np.ndarray,
                 gamma: float, beta: float) -> np.ndarray:
    users = signals["edge_users"]
    exponent = beta * (
        q_value[users] * signals["z_value"] + gamma * q_activity[users] * signals["z_activity"]
    )
    return _balance_margins(np.exp(exponent), signals)


def calibrate_beta(signals: dict, q_value: np.ndarray, q_activity: np.ndarray,
                   gamma: float, target_cv: float) -> float:
    """The one knob, fixed by a data statistic instead of by a result."""

    beta = 1.0
    for _ in range(40):
        weights = edge_weights(signals, q_value, q_activity, gamma, beta)
        cv = float(weights.std() / max(weights.mean(), 1e-12))
        if not np.isfinite(cv) or cv <= 0:
            raise RuntimeError("엣지 가중치 변동계수를 계산할 수 없습니다")
        if abs(cv - target_cv) < 1e-4:
            break
        beta *= target_cv / cv
    return float(beta)


def audit_weights(weights: np.ndarray, signals: dict, prepared: dict) -> dict:
    """The checks the earlier M3 designs would have failed before training."""

    users, items = signals["edge_users"], signals["edge_items"]
    data = prepared["data"]
    n_users, n_items = data["n_users"], data["n_items"]
    price = np.asarray(prepared["item_amount_percentile"], float)
    user_count = np.bincount(users, minlength=n_users)
    item_count = np.bincount(items, minlength=n_items)
    user_mean = np.bincount(users, weights=weights, minlength=n_users) / np.maximum(user_count, 1)
    item_mean = np.bincount(items, weights=weights, minlength=n_items) / np.maximum(item_count, 1)
    seen_u, seen_i = user_count > 0, item_count > 0
    user_mass = np.bincount(users, weights=weights, minlength=n_users)
    return {
        "coefficient_of_variation": float(weights.std() / max(weights.mean(), 1e-12)),
        "customer_weight_vs_degree": float(
            spearmanr(user_mean[seen_u], user_count[seen_u]).statistic
        ),
        "item_weight_vs_price": float(spearmanr(item_mean[seen_i], price[seen_i]).statistic),
        "item_weight_vs_popularity": float(
            spearmanr(item_mean[seen_i], item_count[seen_i]).statistic
        ),
        "max_customer_mass_error": float(
            (np.abs(user_mass[seen_u] - user_count[seen_u]) / user_count[seen_u]).max()
        ),
        "weight_min": float(weights.min()),
        "weight_max": float(weights.max()),
    }


def check_gates(audit: dict, cfg: CenteredGraphConfig, arm: str) -> None:
    cv = audit["coefficient_of_variation"]
    if not 0.15 <= cv <= 0.35:
        raise RuntimeError(f"{arm}: 개입 강도 CV {cv:.3f}가 허용 범위 밖입니다")
    for key, limit in (
        ("customer_weight_vs_degree", cfg.max_degree_correlation),
        ("item_weight_vs_price", cfg.max_price_correlation),
        ("item_weight_vs_popularity", cfg.max_popularity_correlation),
    ):
        if abs(audit[key]) > limit:
            raise RuntimeError(
                f"{arm}: {key} {audit[key]:+.3f}가 한계 {limit}를 넘습니다 — 학습하지 않습니다"
            )
    if audit["max_customer_mass_error"] > 1e-3:
        raise RuntimeError(
            f"{arm}: 고객별 엣지 질량 상대오차 {audit['max_customer_mass_error']:.5f}"
            f" — 양쪽 여백 맞추기가 수렴하지 않았습니다"
        )


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------


def _config_hash(cfg: CenteredGraphConfig, input_hash: str) -> str:
    payload = {
        "code_version": CODE_VERSION,
        "protocol": {
            field: getattr(cfg, field)
            for field in asdict(cfg)
            if field not in {"out_dir", "seeds", "m1_result_dir", "allow_baseline_training"}
        },
        "input_hash": input_hash,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()[:12]


def _prepare(cfg: CenteredGraphConfig) -> dict:
    prepared = recheck._prepare(recheck.configure_component_recheck())
    data = prepared["data"]
    valid = np.asarray(prepared["clv_valid"], bool)
    signals = centered_edge_signals(data["train"], data["n_users"], data["n_items"])
    if not np.array_equal(
        signals["edge_users"] * data["n_items"] + signals["edge_items"],
        np.asarray(data["pos_key"], np.int64),
    ):
        raise RuntimeError("엣지 순서가 M1 이진 그래프와 다릅니다")
    prepared["signals"] = signals
    prepared["q_value"] = np.where(valid, prepared["q_v"], 0.0).astype(float)
    prepared["q_activity"] = np.where(valid, prepared["q_n"], 0.0).astype(float)
    prepared["out_dir"] = Path(cfg.out_dir)
    prepared["config_hash"] = _config_hash(cfg, prepared["input_hash"])
    return prepared


def build_arm_graph(prepared: dict, cfg: CenteredGraphConfig, spec: dict) -> dict:
    beta = calibrate_beta(
        prepared["signals"], prepared["q_value"], prepared["q_activity"],
        spec["gamma"], cfg.target_cv,
    )
    weights = edge_weights(
        prepared["signals"], prepared["q_value"], prepared["q_activity"],
        spec["gamma"], beta,
    )
    audit = audit_weights(weights, prepared["signals"], prepared)
    check_gates(audit, cfg, spec["model_id"])
    return {"beta": beta, "weights": weights, "audit": audit}


def _build_model(prepared: dict, cfg: CenteredGraphConfig, graph: dict, seed: int):
    data = prepared["data"]
    v3.set_seed(seed)
    adjacency = v3.build_adj(
        prepared["signals"]["edge_users"], prepared["signals"]["edge_items"],
        graph["weights"].astype(np.float32), data["n_users"], data["n_items"],
    )
    model = M5NConditionedValueBasisLightGCN(
        n_users=data["n_users"], n_items=data["n_items"],
        user_q_n=prepared["q_n"], user_q_v=prepared["q_v"], user_q_c=prepared["q_c"],
        user_clv_valid=np.asarray(prepared["clv_valid"], bool),
        item_price_percentile=prepared["item_amount_percentile"],
        item_price_valid=prepared["item_economic_valid"],
        adj=adjacency, id_dim=cfg.id_dim, rho=0.0, n_layers=cfg.n_layers,
        pref_reg=cfg.pref_reg, economic_propagation=False,
    )
    return model.to(v3.DEVICE)


def load_baseline_curves(cfg: CenteredGraphConfig) -> dict[int, list[dict]]:
    """Reuse the finished M1 curves; never retrain a baseline without permission."""

    root = Path(cfg.m1_result_dir)
    grid = capacity.evaluation_epochs(cfg)
    curves, missing = {}, []
    for seed in cfg.seeds:
        matches = sorted(root.glob(f"arms/*/baseline_{M1_MODEL_ID}_s{seed}.json"))
        payload = None
        for path in matches:
            candidate = json.loads(path.read_text(encoding="utf-8"))
            evaluated = [r["epoch"] for r in candidate.get("curve", []) if "metrics" in r]
            if (
                candidate.get("id_dim") == cfg.id_dim
                and candidate.get("pref_reg") == cfg.pref_reg
                and evaluated == grid
            ):
                payload = candidate
                break
        if payload is None:
            missing.append(seed)
        else:
            curves[seed] = payload["curve"]
    if missing and not set(f"{seed}:m1" for seed in missing).issubset(
        set(cfg.allow_baseline_training)
    ):
        raise RuntimeError(
            f"seed {missing}의 M1 곡선을 {root}에서 찾지 못했습니다. "
            f"새로 학습하려면 allow_baseline_training에 "
            f"{[f'{s}:m1' for s in missing]}를 명시하세요"
        )
    return curves


def _arm_paths(prepared: dict, model_id: str, seed: int) -> dict[str, Path]:
    root = prepared["out_dir"] / "arms" / prepared["config_hash"]
    return {"result": root / f"{model_id}_s{seed}.json"}


def _run_arm(prepared: dict, cfg: CenteredGraphConfig, spec: dict, graph: dict,
             seed: int) -> dict:
    paths = _arm_paths(prepared, spec["model_id"], seed)
    if paths["result"].exists():
        print(f"  [cached] {spec['model_id']} s{seed} 곡선 재사용")
        return json.loads(paths["result"].read_text(encoding="utf-8"))

    model = _build_model(prepared, cfg, graph, seed)
    store = ProgressStore(
        prepared["out_dir"] / "progress" / prepared["config_hash"],
        RunIdentity(
            stage="centered_graph_dev", model_id=spec["model_id"], seed=seed,
            config_hash=prepared["config_hash"], source_revision=prepared["revision"],
            input_hash=prepared["input_hash"],
        ),
    )
    curve = capacity._train_curve(model, prepared, cfg, spec, seed, store)
    payload = {
        **{key: spec[key] for key in ("model_id", "arm", "gamma", "question")},
        "seed": seed, "beta": graph["beta"], "edge_audit": graph["audit"],
        "id_dim": cfg.id_dim, "pref_reg": cfg.pref_reg,
        "code_version": CODE_VERSION, "source_revision": prepared["revision"],
        "evaluated_at": datetime.now(timezone.utc).isoformat(), "curve": curve,
    }
    test10._atomic_json(paths["result"], payload)
    store.mark_complete(epoch=cfg.epochs, max_epoch=cfg.epochs, selection="none",
                        checkpoint_path="", result_path=str(paths["result"]))
    return payload


def curve_table(arms: list[dict], baselines: dict[int, list[dict]]) -> pd.DataFrame:
    rows = []
    for seed, curve in baselines.items():
        for record in curve:
            if "metrics" in record:
                rows.append({"model_id": M1_MODEL_ID, "arm": "baseline", "gamma": np.nan,
                             "seed": seed, "epoch": record["epoch"], "loss": record["loss"],
                             **record["metrics"]})
    for arm in arms:
        for record in arm["curve"]:
            if "metrics" in record:
                rows.append({"model_id": arm["model_id"], "arm": arm["arm"],
                             "gamma": arm["gamma"], "seed": arm["seed"],
                             "epoch": record["epoch"], "loss": record["loss"],
                             **record["metrics"]})
    return pd.DataFrame(rows)


def difference_table(curve: pd.DataFrame, reported_epochs: list[int]) -> pd.DataFrame:
    """arm - M1, and the activity axis's own contribution (arm B - arm A)."""

    metrics = [column for column in curve.columns if "@" in column]
    index = curve.set_index(["model_id", "seed", "epoch"])
    rows = []
    pairs = [(ARM_VALUE, M1_MODEL_ID), (ARM_VALUE_ACTIVITY, M1_MODEL_ID),
             (ARM_VALUE_ACTIVITY, ARM_VALUE)]
    for model_id, reference in pairs:
        for seed in sorted(curve.seed.unique()):
            for epoch in reported_epochs:
                if (model_id, seed, epoch) not in index.index:
                    continue
                if (reference, seed, epoch) not in index.index:
                    # 참조가 없으면 조용히 빠져 seed 수가 줄어든다 — 용량탐색에서 같은
                    # 형태의 누락이 비교표를 왜곡했으므로 여기서는 멈춘다
                    raise KeyError(
                        f"{model_id} seed {seed} epoch {epoch}의 참조 {reference} 결과가 없습니다"
                    )
                left, right = index.loc[(model_id, seed, epoch)], index.loc[(reference, seed, epoch)]
                row = {"model_id": model_id, "reference": reference, "seed": seed, "epoch": epoch}
                for metric in metrics:
                    base = float(right[metric])
                    row[metric] = float(left[metric]) - base
                    row[f"{metric}_ratio"] = float(left[metric]) / base if base else np.nan
                rows.append(row)
    return pd.DataFrame(rows)


def centered_graph_reading(difference: pd.DataFrame, cfg: CenteredGraphConfig) -> dict:
    def consistent(model_id: str, reference: str, metric: str, epoch: int) -> dict:
        part = difference[
            difference.model_id.eq(model_id) & difference.reference.eq(reference)
            & difference.epoch.eq(epoch)
        ]
        positive = int((part[metric] > 0).sum())
        return {"seeds": int(len(part)), "positive_seeds": positive,
                "mean": float(part[metric].mean()) if len(part) else float("nan")}

    def guard(model_id: str, epoch: int) -> bool:
        part = difference[
            difference.model_id.eq(model_id) & difference.reference.eq(M1_MODEL_ID)
            & difference.epoch.eq(epoch)
        ]
        columns = [f"{m}@{k}_ratio" for m in ("recall", "ndcg") for k in (10, 20, 50)]
        return bool(len(part)) and all(
            float(part[column].min()) >= 0.99 for column in columns if column in part
        )

    reading = {"reported_epochs": [PROTOCOL_EPOCH, cfg.epochs], "arm_selected": False,
               "significance_claimed": False}
    for model_id in (ARM_VALUE, ARM_VALUE_ACTIVITY):
        for epoch in (PROTOCOL_EPOCH, cfg.epochs):
            deep = consistent(model_id, M1_MODEL_ID,
                              "price_purchase_amount_weighted_hit@50", epoch)
            high = consistent(model_id, M1_MODEL_ID, "고CLV_recall@50", epoch)
            reading[f"{model_id}@{epoch}"] = {
                "weighted_hit_50": deep, "high_clv_recall_50": high,
                "accuracy_guard_99pct": guard(model_id, epoch),
                "candidate": bool(
                    guard(model_id, epoch)
                    and deep["positive_seeds"] * 3 >= deep["seeds"] * 2
                    and high["positive_seeds"] * 3 >= high["seeds"] * 2
                ),
            }
    reading["activity_axis_contribution"] = {
        f"epoch_{epoch}": consistent(ARM_VALUE_ACTIVITY, ARM_VALUE,
                                     "price_purchase_amount_weighted_hit@50", epoch)
        for epoch in (PROTOCOL_EPOCH, cfg.epochs)
    }
    return reading


def run_centered_graph(cfg: CenteredGraphConfig | None = None) -> pd.DataFrame:
    cfg = validate_config(cfg or configure_centered_graph())
    print(json.dumps(preflight_summary(cfg), ensure_ascii=False, indent=2))
    baselines = load_baseline_curves(cfg)
    prepared = _prepare(cfg)

    graphs = {}
    for spec in arm_specifications():
        graph = build_arm_graph(prepared, cfg, spec)
        graphs[spec["model_id"]] = graph
        print(f"\n[{spec['model_id']}] beta {graph['beta']:.4f} | "
              f"{json.dumps(graph['audit'], ensure_ascii=False)}")

    arms = []
    for seed in cfg.seeds:
        for spec in arm_specifications():
            print(f"\n===== {spec['model_id']} | seed {seed} | {cfg.epochs} epoch =====")
            arms.append(_run_arm(prepared, cfg, spec, graphs[spec["model_id"]], seed))

    curve = curve_table(arms, baselines)
    difference = difference_table(curve, [PROTOCOL_EPOCH, cfg.epochs])
    reading = centered_graph_reading(difference, cfg)

    out = Path(cfg.out_dir)
    stem = f"clv_m3_centered_value_graph_{prepared['config_hash']}"
    paths = {"curve_csv": out / f"{stem}_curve.csv",
             "difference_csv": out / f"{stem}_difference.csv",
             "json": out / f"{stem}.json"}
    test10._atomic_csv(paths["curve_csv"], curve)
    test10._atomic_csv(paths["difference_csv"], difference)
    test10._atomic_json(paths["json"], {
        "code_version": CODE_VERSION, "config": asdict(cfg),
        "preflight": preflight_summary(cfg), "source_revision": prepared["revision"],
        "edge_audits": {key: value["audit"] for key, value in graphs.items()},
        "betas": {key: value["beta"] for key, value in graphs.items()},
        "curve": curve.to_dict("records"), "difference": difference.to_dict("records"),
        "reading": reading,
        "result_paths": {name: str(path) for name, path in paths.items()},
    })
    print("\n판독:", json.dumps(reading, ensure_ascii=False, indent=2))
    print("저장:", json.dumps({k: str(v) for k, v in paths.items()}, ensure_ascii=False))
    curve.attrs.update(difference_records=difference.to_dict("records"), reading=reading,
                       result_paths={k: str(v) for k, v in paths.items()})
    return curve


def difference_frame(curve: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(curve.attrs["difference_records"])


if __name__ == "__main__":
    print(json.dumps(preflight_summary(configure_centered_graph()), ensure_ascii=False, indent=2))
