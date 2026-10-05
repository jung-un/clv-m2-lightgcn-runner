"""Seed44 development: M2 alone, then M2+fixed M3+M4-B; reuse M1/M5-B.

The existing N-conditioned mapping and shared training loop are unchanged.
Standalone failure never skips the combination. No protected test is opened.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import clv_m2_nv_conditional_basis_screen as basis
import clv_m5_m3_m4_split_nv_screen as m5
import lightgcn_clv_m3_centered_value_graph as m3
from clv_run_state import ProgressStore, RunIdentity

io, capacity, v3 = basis.io, basis.capacity, basis.v3
CODE_VERSION = "clv-m2-nv-conditional-basis-m5-seed44-dev-v1"
SEED, RHO, BANDWIDTH = 44, .25, .25
M1, M2, M5_BASE = basis.M1, basis.NEW, m5.ARM_B
M5_FULL = "m5_nv_conditional_basis_m3nv_split_nv_bpr_k1"
ACCURACY, ECONOMIC = basis.ACCURACY, basis.ECONOMIC


def configure(*, out_dir=None):
    return m5.configure(seed=SEED, out_dir=out_dir or
                        basis.ROOT + "_m2_nv_conditional_basis_m5_seed44_v1")


def validate_config(cfg):
    if asdict(cfg) != asdict(configure(out_dir=cfg.out_dir)):
        raise ValueError("seed44·300epoch 및 사전 고정 설정을 변경할 수 없습니다")


def metric_columns(table):
    return [c for c in table if "@" in c or c ==
            "user_value_tendency_recommended_price_alignment"]


def validate_curve(table, models, cfg):
    if table.duplicated(["model_id", "seed", "epoch"]).any():
        raise ValueError("중복 평가행")
    expected = capacity.evaluation_epochs(cfg)
    for model_id in models:
        part = table[table.model_id.eq(model_id)]
        if not part.seed.eq(SEED).all() or sorted(part.epoch.tolist()) != expected:
            raise ValueError(f"평가 seed/격자 불일치: {model_id}")
    metrics = metric_columns(table)
    if set((*ACCURACY, *ECONOMIC)) - set(metrics):
        raise ValueError("필수 전체 지표 누락")
    if not np.isfinite(table[metrics].to_numpy(float)).all():
        raise ValueError("평가지표 결측 또는 비유한값")


def validate_reference(old, cfg, input_hash, graph, weight_audit):
    expected = dict(code_version=m5.CODE_VERSION, seed=SEED,
                    input_hash=input_hash, split=basis.SPLIT,
                    source_revision="9814a7bae305a1d6257fd36be4410afcbe854372",
                    final_test=False, holdout=False)
    for key, value in expected.items():
        if old.get(key) != value:
            raise ValueError(f"기존 M5-B 신원 불일치: {key}")
    for key, value in asdict(cfg).items():
        if key not in ("out_dir", "m1_result_dir"):
            stored = old["config"].get(key)
            if stored != (list(value) if isinstance(value, tuple) else value):
                raise ValueError(f"기존 M5-B 학습/그래프 설정 불일치: {key}")
    if old.get("lambda") != m5.LAMBDA:
        raise ValueError("기존 M4 강도 불일치")
    if not np.isclose(graph["beta"], old["beta"], rtol=0, atol=1e-6):
        raise ValueError("기존 M3 beta 불일치")
    if weight_audit["sha256"] != old["weight_audit"][M5_BASE]["sha256"]:
        raise ValueError("기존 M4-B 행 가중치 배열 불일치")


def prepare(cfg=None):
    cfg = cfg or configure()
    validate_config(cfg)
    prepared = m3._prepare(cfg)
    data = prepared["data"]
    required = dict(TIME_CUTOFF=690, EVAL_HOLDOUT=False, HOLDOUT_DAYS=0,
                    GRAPH_MODE="binary", LOSS_MODE="plain", NEG_MODE="uniform",
                    MIN_ITEM_INTER=1, TRAIN_ON_VAL=True, TEST_DAYS=7)
    if (any(prepared["base_cfg"].get(k) != v for k, v in required.items())
            or set(data["splits"]) != {"test"} or data["train"].t.max() > 683
            or data.get("loss_w") is not None):
        raise RuntimeError("개발684~690 분할 또는 M2 개입경계 불일치")
    old_curve, provenance = m5._existing_m5_b_curves(cfg, prepared["input_hash"])
    old_curve = old_curve[old_curve.model_id.isin((M1, M5_BASE))].copy()
    validate_curve(old_curve, (M1, M5_BASE), cfg)
    graph_spec = next(s for s in m3.arm_specifications()
                      if s["model_id"] == m3.ARM_VALUE_ACTIVITY)
    graph = m3.build_arm_graph(prepared, cfg, graph_spec)
    weights, audit = m5.row_weights(prepared)
    m5.check_original_m4(audit)
    old = json.loads(Path(provenance["summary_path"]).read_text())
    validate_reference(old, cfg, prepared["input_hash"], graph, audit[M5_BASE])
    prepared["base_cfg"].update(EPOCHS=300, SEED_LIST=[SEED])
    preflight = dict(
        code_version=CODE_VERSION, config=asdict(cfg), seed=SEED, split=basis.SPLIT,
        final_test=False, holdout=False, new_fits_in_order=[M2, M5_FULL],
        reused=[M1, M5_BASE], rho=RHO, basis_bandwidth=BANDWIDTH,
        model="e_u=q_C*[T0+(2*q_N-1)*TN]*b(q_V); e_i=b(item amount percentile)",
        initialization="T0=I, TN=0; separate from-scratch joint fits, no freeze",
        m2="binary graph, plain BPR + existing sampled ID L2",
        m5="unchanged centered N/V M3 graph + split N/V M4-B weights",
        economic_propagation=False, matrix_l2=0.0, external_reranking=False,
        new_item_task=True, negative="uniform unseen K1", min_item_interactions=1,
        primary="300epoch: M5-full both whole-user economic@10 > M5-B and M1",
        guard="six whole-user Recall/NDCG ratios vs M1 >= .99",
        standalone_is_diagnostic_only=True, early_stopping=False,
        selection="none; earlier epochs descriptive, no protected test/holdout",
        data_adaptation="train percentiles and learned T0/TN; fixed rho/bandwidth; M3 target CV=.20",
        limits="already exposed single development seed; no significance/CLV attribution/generalization claim",
        graph_beta=graph["beta"], graph_audit=graph["audit"],
        m4_weight_audit=audit[M5_BASE], baseline_provenance=provenance,
    )
    fingerprint = dict(preflight=preflight, source=prepared["revision"],
                       input=prepared["input_hash"])
    prepared.update(old_curve=old_curve, baseline_provenance=provenance,
                    m3_graph=graph, m4_weights=weights[M5_BASE], preflight=preflight,
                    config_hash=hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()[:12])
    io._atomic_json(Path(cfg.out_dir) / "preflight.json", preflight)
    print(json.dumps(preflight, ensure_ascii=False, indent=2), flush=True)
    return cfg, prepared


def build_model(prepared, cfg, model_id):
    if model_id not in (M2, M5_FULL):
        raise ValueError("새 M2와 M5-full만 학습합니다")
    data = prepared["data"]
    adj = data["adj"]
    if model_id == M5_FULL:
        adj = v3.build_adj(prepared["signals"]["edge_users"],
                           prepared["signals"]["edge_items"],
                           prepared["m3_graph"]["weights"].astype(np.float32),
                           data["n_users"], data["n_items"])
    v3.set_seed(SEED)
    return basis.ConditionalBasisLightGCN(
        n_users=data["n_users"], n_items=data["n_items"],
        user_q_n=prepared["q_n"], user_q_v=prepared["q_v"], user_q_c=prepared["q_c"],
        user_clv_valid=prepared["clv_valid"], item_price_percentile=prepared["item_amount_percentile"],
        item_price_valid=prepared["item_economic_valid"], adj=adj,
        id_dim=cfg.id_dim, n_layers=cfg.n_layers, pref_reg=cfg.pref_reg,
        rho=RHO, basis_bandwidth=BANDWIDTH).to(v3.DEVICE)


def run_arm(cfg, prepared, model_id):
    identity = RunIdentity(CODE_VERSION, model_id, SEED, prepared["config_hash"],
                           prepared["revision"], prepared["input_hash"])
    path = Path(cfg.out_dir) / "arms" / prepared["config_hash"] / f"{model_id}_s{SEED}.json"
    if path.is_file():
        payload = json.loads(path.read_text())
        if payload.get("identity") != asdict(identity) or payload["curve"][-1]["epoch"] != cfg.epochs:
            raise RuntimeError("완료 캐시 신원/epoch 불일치")
        print(f"[cached] {model_id}", flush=True)
        return payload
    model = build_model(prepared, cfg, model_id)
    initial = model.representation_diagnostics()
    store = ProgressStore(Path(cfg.out_dir) / "progress" / prepared["config_hash"], identity)
    spec = dict(model_id=model_id, condition="binary_plain" if model_id == M2 else "m3_m4_b")
    row_weights = None if model_id == M2 else prepared["m4_weights"]
    print(f"[학습] {model_id}: 300epoch, 매epoch 저장/재개", flush=True)
    curve = capacity._train_curve(model, prepared, cfg, spec, SEED, store, row_weights=row_weights)
    payload = dict(identity=asdict(identity), model_id=model_id, seed=SEED, curve=curve,
                   initial_diagnostics=initial, final_diagnostics=model.representation_diagnostics())
    io._atomic_json(path, payload)
    store.mark_complete(epoch=cfg.epochs, max_epoch=cfg.epochs, selection="none",
                        checkpoint_path=str(store.latest_checkpoint), result_path=str(path))
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return payload


def reading(absolute):
    at = absolute[absolute.epoch.eq(300)].set_index("model_id")
    def ratios(model, reference, metrics):
        return {m: float(at.at[model, m] / at.at[reference, m]) for m in metrics}
    accuracy = ratios(M5_FULL, M1, ACCURACY)
    economics = {r: ratios(M5_FULL, r, ECONOMIC) for r in (M1, M5_BASE)}
    return dict(
        fixed_epoch=300, significance_claim=False, clv_attribution_claim=False,
        m2_standalone_diagnostic=dict(accuracy=ratios(M2, M1, ACCURACY),
                                      economics=ratios(M2, M1, ECONOMIC)),
        m5_accuracy_vs_m1=accuracy, m5_accuracy_vs_m5_base=ratios(M5_FULL, M5_BASE, ACCURACY),
        m5_economics=economics,
        combination_candidate=bool(min(accuracy.values()) >= .99
                                   and all(min(r.values()) > 1 for r in economics.values())),
        conditional_interaction_absolute={m: float((at.at[M5_FULL, m] - at.at[M5_BASE, m])
                                                   - (at.at[M2, m] - at.at[M1, m]))
                                          for m in (*ACCURACY, *ECONOMIC)},
        limits="single development seed; descriptive interaction, not attribution",
    )


def report(cfg, prepared, arms):
    rows, diagnostics = [], []
    for arm in arms:
        for r in arm["curve"]:
            if "metrics" not in r:
                continue
            meta = dict(model_id=arm["model_id"], seed=SEED, epoch=r["epoch"])
            rows.append({**meta, **r["metrics"]})
            diagnostics.append({**meta, "loss": r["loss"], "p_correct": r["p_correct"],
                                **r.get("score_split", {}), **r.get("gradient_diagnostics", {})})
    old = prepared["old_curve"]
    metrics = metric_columns(old)
    absolute = pd.concat([old[["model_id", "seed", "epoch", *metrics]], pd.DataFrame(rows)], ignore_index=True)
    validate_curve(absolute, (M1, M5_BASE, M2, M5_FULL), cfg)
    comparisons = []
    for epoch, group in absolute.groupby("epoch"):
        at = group.set_index("model_id")
        for model, reference in ((M2, M1), (M5_BASE, M1), (M5_FULL, M5_BASE), (M5_FULL, M1)):
            for metric in metrics:
                value, base = float(at.at[model, metric]), float(at.at[reference, metric])
                comparisons.append(dict(seed=SEED, epoch=int(epoch), model_id=model,
                                         reference=reference, metric=metric, value=value,
                                         reference_value=base, delta=value-base,
                                         ratio=value/base if base else None))
    tables = dict(absolute=absolute, comparison=pd.DataFrame(comparisons),
                  diagnostics=pd.DataFrame(diagnostics))
    root = Path(cfg.out_dir) / "reports"
    paths = {k: str(root / f"{k}.csv") for k in tables}
    for key, table in tables.items():
        io._atomic_csv(Path(paths[key]), table)
    decision = reading(absolute)
    paths["json"] = str(root / "result.json")
    io._atomic_json(Path(paths["json"]), {**prepared["preflight"],
                    "source_revision": prepared["revision"], "input_hash": prepared["input_hash"],
                    "config_hash": prepared["config_hash"], "arms": arms,
                    "reading": decision, "paths": paths})
    return {**tables, "reading": decision, "paths": paths}


def run(cfg, prepared):
    validate_config(cfg)
    if prepared["preflight"]["config"] != asdict(cfg):
        raise ValueError("prepare 이후 설정 변경")
    _, current = m5._existing_m5_b_curves(cfg, prepared["input_hash"])
    if current != prepared["baseline_provenance"]:
        raise ValueError("prepare 이후 기준 결과 파일 변경")
    # Deliberately no standalone-performance gate between the two fits.
    arms = [run_arm(cfg, prepared, model_id) for model_id in (M2, M5_FULL)]
    return report(cfg, prepared, arms)
