"""One exploratory M5 fit; reuse the verified seed48 M1, never train controls."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import clv_m2_nv_conditional_basis_screen as baseline
import clv_m5_m3_m4_user_centered_screen as m4
from clv_m5_shared_attributes_model import JointAttributeLightGCN, build_signals
from clv_run_state import ProgressStore, RunIdentity
import lightgcn_clv_axis_specific_test10 as io
import lightgcn_clv_component_recheck as recheck
import lightgcn_clv_m2_capacity_search as capacity
import lightgcn_clv_v3 as v3

CODE_VERSION = "clv-m5-shared-attributes-seed48-dev-v1"
MODEL_ID = "m5_shared_attributes_nv_learned_graph_user_centered_bpr_k1"
SPLIT = baseline.SPLIT


@dataclass(frozen=True)
class Config:
    seeds: tuple[int, ...] = (48,)
    epochs: int = 300
    eval_every: int = 25
    batch_size: int = 8192
    lr: float = 5e-4
    n_layers: int = 2
    id_dim: int = 64
    pref_reg: float = .001
    negative_count: int = 1
    eta: float = .1
    epsilon: float = .1
    kappa: float = .2
    out_dir: str = baseline.ROOT + "_m5_shared_attributes_seed48_dev_v1"
    baseline_json: str = baseline.Config().baseline_json


def configure(**paths):
    if set(paths) - {"out_dir", "baseline_json"}:
        raise ValueError("실행 전 고정값은 변경하지 않습니다. 경로만 지정하세요")
    return validate_config(Config(**paths))


def validate_config(cfg):
    for key, value in asdict(Config()).items():
        if key not in ("out_dir", "baseline_json") and getattr(cfg, key) != value:
            raise ValueError(f"사전 고정값 불일치: {key}")
    if not cfg.out_dir or not cfg.baseline_json:
        raise ValueError("결과·기준 경로 누락")
    return cfg


def validate_prepared(prepared):
    data = prepared["data"]
    if set(data["splits"]) != {"test"} or data["train"].t.max() > 683:
        raise RuntimeError("DAY684~690 개발분할 외 평가자료")
    required = {"TIME_CUTOFF": 690, "EVAL_HOLDOUT": False, "HOLDOUT_DAYS": 0,
                "GRAPH_MODE": "binary", "LOSS_MODE": "plain", "NEG_MODE": "uniform",
                "MIN_ITEM_INTER": 1, "TRAIN_ON_VAL": True, "TEST_DAYS": 7}
    if any(prepared["base_cfg"].get(k) != v for k, v in required.items()):
        raise RuntimeError("분할·카탈로그·sampling 설정 불일치")
    if data.get("loss_w") is not None:
        raise RuntimeError("외부 기본 가중치와 M4-C가 중복됩니다")
    train_keys = np.asarray(data["pos_key"], dtype=np.int64)
    truths, _ = data["splits"]["test"]
    keys = np.concatenate([int(u) * data["n_items"] + np.asarray(items, np.int64)
                           for u, items in truths.items()]) if truths else np.empty(0, np.int64)
    if np.isin(keys, train_keys).any():
        raise RuntimeError("학습쌍이 신규상품 정답에 포함됩니다")


def prepare(cfg=None):
    cfg = validate_config(cfg or configure())
    if not Path(cfg.baseline_json).is_file():
        raise FileNotFoundError(f"기존 seed48 M1 JSON이 필요합니다. 새 M1 학습 없음: {cfg.baseline_json}")
    prepared = recheck._prepare(recheck.configure_component_recheck(out_dir=cfg.out_dir))
    validate_prepared(prepared)
    old, provenance = baseline.load_baseline(cfg, prepared["input_hash"])
    products = pd.read_csv(v3.DCFG["item_meta_path"],
                           usecols=["PRODUCT_ID", "COMMODITY_DESC", "SUB_COMMODITY_DESC"])
    data = prepared["data"]
    signals = build_signals(data["train"], products, data["n_users"], data["n_items"])
    if not np.array_equal(signals["edge_users"] * data["n_items"] + signals["edge_items"], data["pos_key"]):
        raise RuntimeError("학습형 M3와 binary graph의 엣지집합이 다릅니다")
    weights, weight_audit = m4.row_weights(prepared)
    attributes_hash = hashlib.sha256(products.to_csv(index=False).encode()).hexdigest()
    fingerprint = {"version": CODE_VERSION, "config": asdict(cfg), "source": prepared["revision"],
                   "input": prepared["input_hash"], "baseline_sha": provenance["sha256"],
                   "metadata_sha": attributes_hash}
    preflight = {
        "code_version": CODE_VERSION, "config": asdict(cfg), "split": SPLIT,
        "new_fit_count": 1, "new_model": MODEL_ID, "m1": "existing seed48 reuse only",
        "seed_selection": "existing compatible M1 artifact; not selected by metric; previously exposed seed",
        "train_days": [1, 683], "development_days": [684, 690],
        "final_test": False, "holdout": False, "selection": "none; fixed epoch300",
        "same_forward_one_optimizer": True, "pretraining_or_freeze": False,
        "external_score_addition_or_reranking": False, "min_item_interactions": 1,
        "new_item_task": True, "negative": "uniform unseen K=1",
        "m2": "shared subtype8+price8; N/V-conditioned per-feature history modulation; positive item LOO",
        "m3": "learned differentiable bounded edge weights; same train pairs; degree normalization",
        "m4": "existing customer-mass-preserving M4-C lambda=.5",
        "objective": "M4-weighted BPR + unchanged sampled ID L2; no extra shared-parameter L2",
        "clv": "original last365day observed N,V,C=N*V midranks; not future CLV",
        "price_caveat": "basket value V is not unit-price preference; price interaction is a hypothesis",
        "attribute_caveat": "shared attributes and capacity are additional explanations; no CLV attribution from M1-only comparison",
        "screen_rule": "six recall/ndcg ratios>=.99 and both economic@10 strictly>M1 at300",
        "limits": "one exposed development seed; no significance or final success claim",
        "signals": signals["audit"], "metadata_sha256": attributes_hash, "m4_audit": weight_audit}
    prepared["base_cfg"].update(EPOCHS=300, SEED_LIST=[48])
    prepared.update(out_dir=Path(cfg.out_dir), baseline=old, baseline_provenance=provenance,
                    signals=signals, row_weights=weights, preflight=preflight,
                    config_hash=hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()[:12])
    io._atomic_json(Path(cfg.out_dir) / "preflight.json", preflight)
    print(json.dumps(preflight, ensure_ascii=False, indent=2), flush=True)
    return cfg, prepared


def build_model(cfg, prepared):
    v3.set_seed(48)
    data = prepared["data"]
    return JointAttributeLightGCN(n_users=data["n_users"], n_items=data["n_items"],
        signals=prepared["signals"], q_n=prepared["q_n"], q_v=prepared["q_v"],
        valid=prepared["clv_valid"], id_dim=cfg.id_dim, n_layers=cfg.n_layers,
        pref_reg=cfg.pref_reg, eta=cfg.eta, epsilon=cfg.epsilon, kappa=cfg.kappa).to(v3.DEVICE)


def run_arm(cfg, prepared):
    identity = RunIdentity(CODE_VERSION, MODEL_ID, 48, prepared["config_hash"],
                           prepared["revision"], prepared["input_hash"])
    path = prepared["out_dir"] / "arms" / prepared["config_hash"] / "m5_s48.json"
    if path.is_file():
        payload = json.loads(path.read_text())
        if payload.get("identity") != asdict(identity) or payload["curve"][-1]["epoch"] != 300:
            raise RuntimeError("완료 결과 신원/epoch 불일치")
        return payload
    model = build_model(cfg, prepared)
    initial = model.representation_diagnostics()
    store = ProgressStore(prepared["out_dir"] / "progress" / prepared["config_hash"], identity)
    print("[학습] M5 seed48 1회만 실행. 학습형 그래프는 고정그래프보다 느릴 수 있습니다.", flush=True)
    curve = capacity._train_curve(model, prepared, cfg,
        {"model_id": MODEL_ID, "condition": "joint_learned_graph_attributes"}, 48,
        store, row_weights=prepared["row_weights"])
    for record in curve:
        if "score_split" in record:
            # Helper's legacy CLV labels refer to the entire attribute block,
            # which is NOT an isolated measurement of the N/V contribution.
            record["attribute_score_split"] = {
                k.replace("clv", "attribute_branch"): v for k, v in record.pop("score_split").items()}
            record["attribute_score_split"]["not_clv_attribution"] = True
    payload = {"identity": asdict(identity), "model_id": MODEL_ID, "seed": 48,
               "curve": curve, "initial_diagnostics": initial,
               "final_diagnostics": model.representation_diagnostics()}
    io._atomic_json(path, payload)
    store.mark_complete(epoch=300, max_epoch=300, selection="none",
                        checkpoint_path=str(store.latest_checkpoint), result_path=str(path))
    return payload


def report(cfg, prepared, arm):
    old = prepared["baseline"]
    metrics = list(old["curve"][-1]["metrics"])
    rows, diagnostics = [], []
    for current in (old, arm):
        for r in current["curve"]:
            if "metrics" not in r:
                continue
            if set(r["metrics"]) != set(metrics):
                raise RuntimeError("M1/M5 전체 metric key가 다릅니다")
            meta = {"model_id": current["model_id"], "seed": 48, "epoch": r["epoch"]}
            rows.append({**meta, **r["metrics"]})
            diagnostics.append({**meta, "loss": r["loss"], "p_correct": r["p_correct"],
                                **r.get("gradient_diagnostics", {}),
                                **r.get("attribute_score_split", {})})
    absolute = pd.DataFrame(rows)
    expected = {(m, ep) for m in (baseline.M1, MODEL_ID) for ep in capacity.evaluation_epochs(cfg)}
    if (absolute.duplicated(["model_id", "epoch"]).any()
            or set(zip(absolute.model_id, absolute.epoch)) != expected
            or not np.isfinite(absolute[metrics].to_numpy(float)).all()):
        raise RuntimeError("평가점 중복/누락/비유한 지표")
    comparisons = []
    for epoch, group in absolute.groupby("epoch"):
        at = group.set_index("model_id")
        for metric in metrics:
            value, reference = float(at.at[MODEL_ID, metric]), float(at.at[baseline.M1, metric])
            comparisons.append({"seed": 48, "epoch": int(epoch), "metric": metric,
                "m1": reference, "m5": value, "difference": value - reference,
                "relative_percent": 100 * (value / reference - 1) if reference else None})
    final = absolute[absolute.epoch.eq(300)].set_index("model_id")
    accuracy = {m: float(final.at[MODEL_ID, m] / final.at[baseline.M1, m]) for m in baseline.ACCURACY}
    economic = {m: float(final.at[MODEL_ID, m] / final.at[baseline.M1, m]) for m in baseline.ECONOMIC}
    reading = {"accuracy_ratios": accuracy, "economic_ratios": economic,
               "exploratory_screen_rule_passed": min(accuracy.values()) >= .99 and min(economic.values()) > 1,
               "significance_claim": False, "clv_attribution_claim": False, "final_success_claim": False,
               "requires_full_raw_results_and_diagnostics_review": True}
    tables = {"absolute": absolute, "comparison": pd.DataFrame(comparisons),
              "diagnostics": pd.DataFrame(diagnostics)}
    root = Path(cfg.out_dir) / "reports"
    paths = {k: str(root / f"{k}.csv") for k in tables}
    for key, table in tables.items():
        io._atomic_csv(Path(paths[key]), table)
    paths["json"] = str(root / "result.json")
    paths["preflight"] = str(Path(cfg.out_dir) / "preflight.json")
    io._atomic_json(Path(paths["json"]), {**prepared["preflight"],
        "source_revision": prepared["revision"], "input_hash": prepared["input_hash"],
        "config_hash": prepared["config_hash"], "baseline_provenance": prepared["baseline_provenance"],
        "reading": reading, "arms": [old, arm], "paths": paths})
    return {**tables, "reading": reading, "paths": paths}


def run(cfg, prepared):
    validate_config(cfg)
    validate_prepared(prepared)
    if prepared["preflight"]["config"] != asdict(cfg):
        raise ValueError("prepare 이후 설정 변경")
    _, provenance = baseline.load_baseline(cfg, prepared["input_hash"])
    if provenance != prepared["baseline_provenance"]:
        raise ValueError("prepare 이후 M1 JSON 변경")
    return report(cfg, prepared, run_arm(cfg, prepared))
