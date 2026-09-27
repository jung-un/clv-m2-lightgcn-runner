"""H&M two-year seed-43 run of the centered value graph M3 (value axis, +activity axis).

The design, the gates and the reading rule are the Dunnhumby ones — only the data
set changes, which is the point: the professor asked whether the same structure can
carry different per-data-set strengths, and beta here is derived from H&M's own
weight spread rather than copied from Dunnhumby.

H&M keeps its own conventions: the full two-year graph, MIN_ITEM_INTER=1, batch
131,072, and evaluation only at epochs 100/200/300 because a full evaluation over
one million customers is expensive. M1 is reused from the seed-43 learning-budget
run, which trained the same model under the same protocol.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import lightgcn_clv_axis_specific_test10 as test10
import lightgcn_clv_hm2y_seed42_common as common
import lightgcn_clv_m2_training_budget_hm2y as budget
import lightgcn_clv_m3_centered_value_graph as graph
import lightgcn_clv_v3 as v3
from clv_m5_n_conditioned_value_basis_model import M5NConditionedValueBasisLightGCN


CODE_VERSION = "clv-m3-centered-value-graph-hm2y-s43-v1"
M1_MODEL_ID = budget.M1_MODEL_ID                  # 예산 실행에서 재사용하는 기준모형
ARM_VALUE = "m3_centered_value_graph_bpr_k1_hm2y_s43"
ARM_VALUE_ACTIVITY = "m3_centered_value_activity_graph_bpr_k1_hm2y_s43"
SPLIT = "hm2y_validation_2020-09-02_08"


@dataclass(frozen=True)
class CenteredGraphHMConfig:
    dataset: str = "hm"
    seed: int = 43
    window_days: None = None
    input_days: int = 365
    epochs: int = 300
    evaluation_epochs: tuple[int, ...] = (100, 200, 300)
    reported_epochs: tuple[int, ...] = (100, 300)
    id_dim: int = 64
    n_layers: int = 2
    batch_size: int = common.DEFAULT_BATCH_SIZE
    lr: float = 5e-4
    pref_reg: float = 1e-3
    negative_count: int = 1
    target_cv: float = 0.20
    max_degree_correlation: float = 0.05
    max_price_correlation: float = 0.20
    max_popularity_correlation: float = 0.20
    # 한 arm에 20시간이 걸리므로 나눠 실행할 수 있게 둔다. arm 목록은 run hash에서
    # 제외하므로 나중에 나머지 arm을 더해도 끝난 arm은 그대로 재사용된다.
    # 논문의 주 모형은 CLV 두 축을 함께 쓰는 value_and_activity이므로 그것을 먼저 둔다.
    arms: tuple[str, ...] = ("value_and_activity", "value_only")
    m1_result_dir: str = ""
    allow_baseline_training: bool = False
    eval_test: bool = False
    eval_holdout: bool = False
    out_dir: str = ""


def configure_centered_graph_hm2y(**overrides) -> CenteredGraphHMConfig:
    root = v3.default_out_dir("hm")
    defaults = {
        "out_dir": f"{root}_clv_m3_centered_value_graph_s43_v1",
        "m1_result_dir": f"{root}_clv_m2_training_budget_seed43_v1",
    }
    return validate_config(CenteredGraphHMConfig(**(defaults | overrides)))


def validate_config(cfg: CenteredGraphHMConfig) -> CenteredGraphHMConfig:
    if cfg.dataset != "hm" or cfg.window_days is not None:
        raise ValueError("이 러너는 H&M 2년 전체 구간 전용입니다")
    if cfg.negative_count != 1:
        raise ValueError("기준 손실은 원 LightGCN BPR(음성 1개)입니다")
    missing = set(cfg.reported_epochs) - set(cfg.evaluation_epochs)
    if missing or cfg.epochs != max(cfg.evaluation_epochs):
        raise ValueError("보고할 epoch이 평가 격자에 없습니다")
    if cfg.eval_test or cfg.eval_holdout:
        raise ValueError("개발 단계에서 test/holdout을 열지 않습니다")
    unknown = set(cfg.arms) - {"value_only", "value_and_activity"}
    if unknown or not cfg.arms:
        raise ValueError(f"알 수 없는 arm: {sorted(unknown)}")
    return cfg


def arm_specifications(cfg: CenteredGraphHMConfig | None = None) -> list[dict]:
    chosen = set(cfg.arms) if cfg is not None else {"value_only", "value_and_activity"}
    return [spec for spec in _ALL_ARMS if spec["arm"] in chosen]


_ALL_ARMS = [
    {"model_id": ARM_VALUE_ACTIVITY, "arm": "value_and_activity", "gamma": 1.0,
     "kind": "m3",
     "question": "CLV 두 축(N·V)을 함께 넣어 전파를 바꾸면 H&M에서 M1을 넘는가"},
    {"model_id": ARM_VALUE, "arm": "value_only", "gamma": 0.0, "kind": "m3",
     "question": "그 성과가 가치축에서 오는지 반복거래축에서 오는지 나누기 위한 분해"},
]


def preflight_summary(cfg: CenteredGraphHMConfig) -> dict:
    cfg = validate_config(cfg)
    return {
        "code_version": CODE_VERSION,
        "dataset": "hm",
        "split": SPLIT,
        "graph_window": "two years, MIN_ITEM_INTER=1",
        "clv_input_days": cfg.input_days,
        "seeds": [cfg.seed],
        "epochs": cfg.epochs,
        "evaluated_at_epochs": list(cfg.evaluation_epochs),
        "reported_epochs": list(cfg.reported_epochs),
        "edge_weight": (
            "exp(beta * [q_V(u)*z_V(u,i) + gamma * q_N(u)*z_N(u,i)]), scaled until "
            "both the customer and the item margin average one. Same structure as the "
            "Dunnhumby run; beta comes from H&M's own weight spread"
        ),
        "axes_not_rescaled_to_each_other": True,
        "single_strength_rule": (
            f"beta is calibrated to a weight coefficient of variation of "
            f"{cfg.target_cv}; it is never chosen from results"
        ),
        "gates_before_training": {
            "coefficient_of_variation": [0.15, 0.35],
            "customer_weight_vs_degree": cfg.max_degree_correlation,
            "item_weight_vs_price": cfg.max_price_correlation,
            "item_weight_vs_popularity": cfg.max_popularity_correlation,
        },
        "arms": arm_specifications(cfg),
        "arms_not_in_run_hash": "한 arm씩 나눠 실행해도 끝난 arm은 재사용된다",
        "m1": (
            "reused from the seed-43 learning-budget run after checking the protocol; "
            "never retrained unless allow_baseline_training is set"
        ),
        "reading": (
            "arm - M1 at the reported epochs, and arm B - arm A for the activity axis; "
            "no epoch and no arm is selected after seeing results"
        ),
        "limits": (
            "one seed on a development split; no significance, no CLV attribution and "
            "no generalization is claimed. Dunnhumby used a different split and "
            "evaluation grid, so the two data sets are not compared number to number"
        ),
    }


# --------------------------------------------------------------------------
# preparation and edge weights
# --------------------------------------------------------------------------


def _config_hash(cfg: CenteredGraphHMConfig, input_hash: str) -> str:
    payload = {
        "code_version": CODE_VERSION,
        "protocol": {field: getattr(cfg, field) for field in asdict(cfg)
                     if field not in {"out_dir", "m1_result_dir", "arms",
                                      "allow_baseline_training"}},
        "input_hash": input_hash,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()[:12]


def _purchase_keyed(train: pd.DataFrame) -> pd.DataFrame:
    """Give H&M the purchase identifier the activity axis counts.

    Dunnhumby carries BASKET_ID, but H&M has no order id and its time resolution is
    the date, so the schema treats (customer, date) as one purchase. Inside a
    (customer, item) group the number of distinct dates is that pair's repeat count,
    which is exactly what the activity axis needs.
    """

    if "b_raw" in train.columns:
        return train
    if "t" not in train.columns:
        raise RuntimeError("구매 1건을 식별할 열(b_raw 또는 t)이 없습니다")
    return train.assign(b_raw=train["t"])


def _prepare(cfg: CenteredGraphHMConfig) -> dict:
    prepared = common.prepare_hm2y(cfg, code_version=CODE_VERSION)
    data = prepared["data"]
    signals = graph.centered_edge_signals(
        _purchase_keyed(data["train"]), data["n_users"], data["n_items"]
    )
    if not np.array_equal(
        signals["edge_users"] * data["n_items"] + signals["edge_items"],
        np.asarray(data["pos_key"], np.int64),
    ):
        raise RuntimeError("엣지 순서가 M1 이진 그래프와 다릅니다")
    valid = np.asarray(prepared["clv_valid"], bool)
    prepared["signals"] = signals
    prepared["q_value"] = np.where(valid, prepared["q_v"], 0.0).astype(float)
    prepared["q_activity"] = np.where(valid, prepared["q_n"], 0.0).astype(float)
    prepared["item_amount_percentile"] = budget._item_price_percentile(prepared)
    prepared["run_hash"] = _config_hash(cfg, prepared["input_hash"])
    return prepared


def build_arm_graph(prepared: dict, cfg: CenteredGraphHMConfig, spec: dict) -> dict:
    beta = graph.calibrate_beta(
        prepared["signals"], prepared["q_value"], prepared["q_activity"],
        spec["gamma"], cfg.target_cv,
    )
    weights = graph.edge_weights(
        prepared["signals"], prepared["q_value"], prepared["q_activity"],
        spec["gamma"], beta,
    )
    audit = graph.audit_weights(weights, prepared["signals"], prepared)
    graph.check_gates(audit, cfg, spec["model_id"])
    return {"beta": beta, "weights": weights, "audit": audit}


def _build_model(prepared: dict, cfg: CenteredGraphHMConfig, arm_graph: dict):
    data = prepared["data"]
    v3.set_seed(cfg.seed)
    adjacency = v3.build_adj(
        prepared["signals"]["edge_users"], prepared["signals"]["edge_items"],
        arm_graph["weights"].astype(np.float32), data["n_users"], data["n_items"],
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


# --------------------------------------------------------------------------
# baseline reuse
# --------------------------------------------------------------------------

BASELINE_PROTOCOL = ("seed", "id_dim", "pref_reg", "batch_size", "lr",
                     "negative_count", "epochs", "window_days", "input_days")
ALIGNMENT_METRIC = "user_value_tendency_recommended_price_alignment"


def _baseline_record(record: dict) -> dict | None:
    """Normalise one M1 curve record.

    The budget run's top-level file stores its comparison table, so the metrics are
    flattened next to the identifiers and the diagnostics; its per-arm files keep
    them nested under "metrics". Accept both and keep the reported metrics only —
    everything scored at a cut-off, plus the value-tendency alignment.
    """

    if "metrics" in record:
        metrics = dict(record["metrics"])
    else:
        metrics = {key: value for key, value in record.items()
                   if "@" in key or key == ALIGNMENT_METRIC}
    if not metrics:
        return None
    return {"epoch": record["epoch"], "loss": record.get("loss"), "metrics": metrics}


def load_baseline_curve(cfg: CenteredGraphHMConfig) -> list[dict]:
    """Take the M1 curve from the seed-43 budget run, after checking the protocol."""

    root = Path(cfg.m1_result_dir)
    for path in sorted(root.glob("clv_m2_training_budget_hm2y_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        config = payload.get("config", {})
        if any(config.get(field) != getattr(cfg, field) for field in BASELINE_PROTOCOL):
            continue
        if tuple(config.get("evaluation_epochs") or ()) != tuple(cfg.evaluation_epochs):
            continue
        curve = [normalised for normalised in
                 (_baseline_record(record) for record in payload.get("curve", [])
                  if record.get("model_id") == M1_MODEL_ID)
                 if normalised is not None]
        evaluated = sorted(record["epoch"] for record in curve)
        if evaluated != sorted(cfg.evaluation_epochs):
            continue
        print(f"  M1 재사용: {path.name} (커밋 {str(payload.get('source_revision'))[:12]})")
        return curve
    if not cfg.allow_baseline_training:
        raise RuntimeError(
            f"{root}에서 프로토콜이 일치하는 M1 곡선을 찾지 못했습니다. "
            f"새로 학습하려면 allow_baseline_training=True를 명시하세요 "
            f"(학습 1회가 추가됩니다)"
        )
    return []


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------


def _arm_path(prepared: dict, model_id: str) -> Path:
    root = Path(prepared["out_dir"]) / "arms" / prepared["run_hash"]
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{model_id}.json"


def _run_arm(prepared: dict, cfg: CenteredGraphHMConfig, spec: dict,
             arm_graph: dict | None) -> dict:
    path = _arm_path(prepared, spec["model_id"])
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        other = payload.get("source_revision") != prepared["revision"]
        print(f"  [cached] {spec['model_id']} 곡선 재사용"
              + (f" — 다른 커밋 {str(payload.get('source_revision'))[:12]} 결과다" if other else ""))
        return payload

    if arm_graph is None:                      # 허용된 경우의 M1 학습
        model = budget._build_model(prepared, cfg, {"kind": "m1"})
    else:
        model = _build_model(prepared, cfg, arm_graph)
    store = common.progress_store(prepared, cfg, spec["model_id"], prepared["run_hash"])
    curve = budget._train_curve(model, prepared, cfg, spec, store)
    payload = {
        **{key: spec[key] for key in ("model_id", "arm", "gamma", "question")},
        "seed": cfg.seed, "split": SPLIT,
        "beta": arm_graph["beta"] if arm_graph else None,
        "edge_audit": arm_graph["audit"] if arm_graph else None,
        "id_dim": cfg.id_dim, "pref_reg": cfg.pref_reg,
        "code_version": CODE_VERSION, "source_revision": prepared["revision"],
        "evaluated_at": datetime.now(timezone.utc).isoformat(), "curve": curve,
    }
    common.atomic_json(path, payload)
    store.mark_complete(epoch=cfg.epochs, max_epoch=cfg.epochs, selection="none",
                        checkpoint_path="", result_path=str(path))
    return payload


def curve_table(arms: list[dict], baseline: list[dict]) -> pd.DataFrame:
    rows = []
    for record in baseline:
        if "metrics" in record:
            rows.append({"model_id": M1_MODEL_ID, "arm": "baseline", "gamma": np.nan,
                         "epoch": record["epoch"], "loss": record.get("loss"),
                         **record["metrics"]})
    for arm in arms:
        for record in arm["curve"]:
            if "metrics" in record:
                rows.append({"model_id": arm["model_id"], "arm": arm["arm"],
                             "gamma": arm["gamma"], "epoch": record["epoch"],
                             "loss": record.get("loss"), **record["metrics"]})
    return pd.DataFrame(rows)


def difference_table(curve: pd.DataFrame, reported_epochs) -> pd.DataFrame:
    metrics = [column for column in curve.columns if "@" in column]
    index = curve.set_index(["model_id", "epoch"])
    rows = []
    for model_id, reference in ((ARM_VALUE, M1_MODEL_ID),
                                (ARM_VALUE_ACTIVITY, M1_MODEL_ID),
                                (ARM_VALUE_ACTIVITY, ARM_VALUE)):
        for epoch in reported_epochs:
            if (model_id, epoch) not in index.index:
                continue
            if reference == ARM_VALUE and (reference, epoch) not in index.index:
                continue          # arm A를 아직 돌리지 않은 단계 실행
            if (reference, epoch) not in index.index:
                raise KeyError(f"{model_id} epoch {epoch}의 참조 {reference} 결과가 없습니다")
            left, right = index.loc[(model_id, epoch)], index.loc[(reference, epoch)]
            row = {"model_id": model_id, "reference": reference, "epoch": epoch}
            for metric in metrics:
                base = float(right[metric])
                row[metric] = float(left[metric]) - base
                row[f"{metric}_ratio"] = float(left[metric]) / base if base else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def centered_graph_hm_reading(difference: pd.DataFrame,
                              cfg: CenteredGraphHMConfig) -> dict:
    """One seed, so this records direction only — never a pass or a selection."""

    def at(model_id, reference, epoch):
        part = difference[difference.model_id.eq(model_id)
                         & difference.reference.eq(reference)
                         & difference.epoch.eq(epoch)]
        return part.iloc[0] if len(part) else None

    reading = {"seeds": 1, "reported_epochs": list(cfg.reported_epochs),
               "arm_selected": False, "significance_claimed": False,
               "candidate_decided": False}
    guard_columns = [f"{name}@{k}_ratio" for name in ("recall", "ndcg") for k in (10, 20, 50)]
    for model_id in (ARM_VALUE, ARM_VALUE_ACTIVITY):
        for epoch in cfg.reported_epochs:
            row = at(model_id, M1_MODEL_ID, epoch)
            if row is None:
                continue
            ratios = [float(row[column]) for column in guard_columns if column in row]
            reading[f"{model_id}@{epoch}"] = {
                "weighted_hit_10": float(row["price_purchase_amount_weighted_hit@10"]),
                "weighted_hit_50": float(row["price_purchase_amount_weighted_hit@50"]),
                "vndcg_10": float(row["vndcg@10"]),
                "high_clv_recall_50": float(row.get("고CLV_recall@50", np.nan)),
                "accuracy_guard_99pct": bool(ratios and min(ratios) >= 0.99),
                "worst_accuracy_ratio": min(ratios) if ratios else None,
            }
    reading["activity_axis_contribution"] = {
        f"epoch_{epoch}": (
            float(at(ARM_VALUE_ACTIVITY, ARM_VALUE, epoch)
                  ["price_purchase_amount_weighted_hit@50"])
            if at(ARM_VALUE_ACTIVITY, ARM_VALUE, epoch) is not None else None
        )
        for epoch in cfg.reported_epochs
    }
    return reading


def run_centered_graph_hm2y(cfg: CenteredGraphHMConfig | None = None) -> pd.DataFrame:
    cfg = validate_config(cfg or configure_centered_graph_hm2y())
    print(json.dumps(preflight_summary(cfg), ensure_ascii=False, indent=2))
    baseline = load_baseline_curve(cfg)
    prepared = _prepare(cfg)

    graphs = {}
    for spec in arm_specifications(cfg):
        arm_graph = build_arm_graph(prepared, cfg, spec)   # 게이트 실패 시 예외
        graphs[spec["model_id"]] = arm_graph
        print(f"\n[{spec['model_id']}] beta {arm_graph['beta']:.4f} | "
              f"{json.dumps(arm_graph['audit'], ensure_ascii=False)}")

    if not baseline:
        print(f"\n===== {M1_MODEL_ID} | seed {cfg.seed} | {cfg.epochs} epoch "
              f"(허용된 baseline 학습) =====")
        baseline = _run_arm(prepared, cfg,
                            {"model_id": M1_MODEL_ID, "arm": "baseline", "gamma": 0.0,
                             "question": "기준모형"}, None)["curve"]

    arms = []
    for spec in arm_specifications(cfg):
        print(f"\n===== {spec['model_id']} | seed {cfg.seed} | {cfg.epochs} epoch =====")
        arms.append(_run_arm(prepared, cfg, spec, graphs[spec["model_id"]]))
    for spec in _ALL_ARMS:                     # 이전 실행에서 끝난 arm도 표에 올린다
        path = _arm_path(prepared, spec["model_id"])
        if spec["arm"] not in cfg.arms and path.exists():
            print(f"  [cached] {spec['model_id']} 이전 실행 결과를 표에 포함")
            arms.append(json.loads(path.read_text(encoding="utf-8")))

    curve = curve_table(arms, baseline)
    difference = difference_table(curve, cfg.reported_epochs)
    reading = centered_graph_hm_reading(difference, cfg)

    out = Path(prepared["out_dir"])
    stem = f"clv_m3_centered_value_graph_hm2y_{prepared['run_hash']}"
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
    print("\n판독(방향만, 1 seed):", json.dumps(reading, ensure_ascii=False, indent=2))
    print("저장:", json.dumps({k: str(v) for k, v in paths.items()}, ensure_ascii=False))
    curve.attrs.update(difference_records=difference.to_dict("records"), reading=reading,
                       result_paths={k: str(v) for k, v in paths.items()})
    return curve


def difference_frame(curve: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(curve.attrs["difference_records"])


if __name__ == "__main__":
    print(json.dumps(preflight_summary(configure_centered_graph_hm2y()),
                     ensure_ascii=False, indent=2))
