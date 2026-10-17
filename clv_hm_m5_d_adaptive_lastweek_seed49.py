"""H&M seed49: fit only fixed adaptive M2 + centered N/V M3, M4 weighted BPR."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import pandas as pd
import torch

import clv_m5_ab_adaptive_lastweek_seed49 as base

VERSION = "clv-hm-m5-d-adaptive-lastweek-seed49-v1"
MODEL_ID = "m5_d_adaptive_m2_centered_nv_m3_user_centered_m4_bpr_k1_hm2y_lastweek_s49"
LABELS = {base.hm.M1: "M1", base.m2.MODEL_ID: "M2", base.hm.M3: "M3",
          base.hm.M4_C: "M4", MODEL_ID: "M5_D"}
CORE = tuple(f"{m}@{k}" for k in (10, 20, 50) for m in
             ("recall", "ndcg", "price_purchase_amount_weighted_hit", "vndcg"))


def configure(*, out_dir=None, m2_reference=None, m4_reference=None):
    return base.configure(dataset="hm", seed=49,
        out_dir=out_dir or base.v3.default_out_dir("hm") + "_m5_d_adaptive_lastweek_seed49_v1",
        m2_reference=m2_reference, m4_reference=m4_reference)


def validate_config(cfg):
    base._validate_config(cfg)
    if cfg != configure(out_dir=cfg.out_dir, m2_reference=cfg.m2_reference, m4_reference=cfg.m4_reference):
        raise ValueError("H&M seed49·고정300epoch 설정만 허용합니다")


def prepare(cfg):
    validate_config(cfg)
    references = base.load_references(cfg)  # Four same-seed rows; never retrain missing references.
    p2, p3 = (references["sources"][key]["protocol"] for key in ("m2", "m4"))
    gc = base.hm.configure(out_dir=str(Path(cfg.out_dir) / "inputs"))
    for key in ("target_cv", "max_degree_correlation", "max_price_correlation", "max_popularity_correlation"):
        if p3["config"][key] != getattr(gc, key):
            raise RuntimeError(f"기존 M3 그래프 설정 불일치: {key}")
    prepared = base._prepare_hm_inputs(cfg)
    if prepared["observed_weight_audit"]["sha256"] != p3["m4_c_weight_audit"]["sha256"]:
        raise RuntimeError("기존 M4와 손실 가중치 불일치")
    data = prepared["data"]
    base.hm.validate_split(data)
    if prepared["input_hash"] != p2["input_hash"] or data["data_stats"] != p2["data_stats"]:
        raise RuntimeError("기존 참조와 입력/마지막주 분할 불일치")
    if (prepared["feature_sha256"] != p2["feature_sha256"] or
            not np.array_equal(prepared["features"]["keys"], data["pos_key"])):
        raise RuntimeError("기존 M2와 특징/구매쌍 순서 불일치")
    qn, qv, _, valid = base.fixed.evaluation.build_clv_inputs(prepared["axes"])
    qn, qv = (np.where(valid, q, 0).astype(np.float32) for q in (qn, qv))
    keyed = base.hm.hm_graph._purchase_keyed(data["train"])
    signals = base.fixed.graph.centered_edge_signals(keyed, data["n_users"], data["n_items"])
    del keyed
    if not np.array_equal(signals["edge_users"] * data["n_items"] + signals["edge_items"], data["pos_key"]):
        raise RuntimeError("M3 엣지와 기존 구매쌍 순서 불일치")
    prepared.update(signals=signals, q_value=qv.astype(float), q_activity=qn.astype(float),
                    item_amount_percentile=prepared["item_amount_percentile"].astype(float))
    graph = base.fixed.build_graph(prepared, gc)  # Same beta calibration and gates as H&M M3.
    expected_audit = p3["m3_audit"]
    if set(graph["audit"]) != set(expected_audit) or any(
            not np.isclose(graph["audit"][k], v, rtol=1e-8, atol=1e-10) for k, v in expected_audit.items()):
        raise RuntimeError("기존 H&M M3와 그래프 진단 불일치: 학습하지 않습니다")
    prepared["observed_adj"] = base.v3.build_adj(signals["edge_users"], signals["edge_items"],
        graph["weights"].astype(np.float32), data["n_users"], data["n_items"])
    protocol = {"code_version": VERSION, "config": asdict(cfg),
        "source_revision": prepared["revision"], "input_hash": prepared["input_hash"],
        "split": p3["split"], "data_stats": data["data_stats"],
        "feature_sha256": prepared["feature_sha256"], "dataset_initialization": prepared["protocol"]["dataset_initialization"],
        "m3_audit": graph["audit"], "m3_beta": graph["beta"],
        "m3_weight_sha256": hashlib.sha256(np.ascontiguousarray(graph["weights"]).tobytes()).hexdigest(),
        "m3_config": {k: getattr(gc, k) for k in
            ("target_cv", "max_degree_correlation", "max_price_correlation", "max_popularity_correlation")},
        "reference_audit": references["audit"], "labels": LABELS,
        "new_fit_models": [MODEL_ID], "reused_models": list(references["absolute"].model_id),
        "seeds": [49], "fixed_epoch": 300, "fixed": p3["fixed"],
        "model": "M5_D: adaptive M2 representation + centered N/V M3 graph + M4 weighted BPR",
        "row_weights": "user_centered", "m4_intervention": True,
        "m4_weight_audit": prepared["observed_weight_audit"],
        "initialization": "fresh joint initialization; one optimizer; no reference checkpoints loaded",
        "decision": "all142 metrics against same-seed M1/M2/M3/M4; no test-based retuning",
        "single_seed_only": True, "significance_claim": False,
        "clv_attribution_claim": False, "unseen_confirmation_claim": False}
    config_hash = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()[:12]
    run_dir = Path(cfg.out_dir) / config_hash
    base.io._atomic_json(run_dir / "protocol.json", protocol)
    base.io._atomic_json(run_dir / "reference_audit.json", references["audit"])
    prepared.update(cfg=cfg, references=references, protocol=protocol, config_hash=config_hash, run_dir=run_dir)
    # The training/evaluation path no longer needs these large construction arrays.
    for key in ("signals", "q_value", "q_activity", "item_amount_percentile"):
        prepared.pop(key, None)
    print("확인: H&M seed49 M1/M2/M3/M4 4개 재사용 / 새 학습 M5_D 1개 / M2+M3+M4 공동 학습 / 별도 순열 학습 없음", flush=True)
    report(prepared)
    return prepared


def identity(prepared):
    return base.RunIdentity(VERSION, MODEL_ID, 49, prepared["config_hash"],
                            prepared["revision"], prepared["input_hash"])


def completed(prepared):
    path = prepared["run_dir"] / "arms" / f"{MODEL_ID}.json"
    if not path.exists():
        return None
    row = json.loads(path.read_text())
    if (row.get("identity") != asdict(identity(prepared)) or row.get("epochs") != 300
            or row.get("seed") != 49 or row.get("model_id") != MODEL_ID
            or set(row.get("metrics", {})) != set(prepared["references"]["metrics"])
            or not np.isfinite(list(row["metrics"].values())).all()):
        raise RuntimeError("완료 결과의 신원/300epoch/142지표 불일치")
    return row


def run(prepared):
    cfg = prepared["cfg"]
    validate_config(cfg)
    if completed(prepared) is not None:
        print("[재사용] M5_D seed49 — 재학습·재평가 없음", flush=True)
        return report(prepared)
    if prepared["data"].get("loss_w") is not None:
        raise RuntimeError("M4 가중치 중복 적용 금지")
    weights = prepared.get("observed_weights")
    expected = prepared["protocol"]["m4_weight_audit"]["sha256"]
    if weights is None or hashlib.sha256(np.asarray(weights, np.float32).tobytes()).hexdigest() != expected:
        raise RuntimeError("M4 가중치 누락/변경: 학습하지 않습니다")
    if base.m2._feature_hash(prepared["features"]) != prepared["feature_sha256"]:
        raise RuntimeError("준비 후 M2 특징 변경")
    data = prepared["data"]
    base.v3.set_seed(49)
    model = base.m2.DatasetAdaptiveOrthogonalNVLightGCN(
        n_users=data["n_users"], n_items=data["n_items"], features=prepared["features"],
        adj=prepared["observed_adj"], id_dim=cfg.id_dim, n_layers=cfg.n_layers,
        pref_reg=cfg.pref_reg, shared_l2=cfg.shared_l2).to(base.v3.DEVICE)
    store = base.ProgressStore(prepared["run_dir"] / "progress", identity(prepared))
    print("[1/1] H&M seed49 M5_D — M2+M3+M4 / M4 weighted BPR", flush=True)
    history = base.fixed._train(model, prepared, cfg, MODEL_ID, 49, store, row_weights=weights)
    if [r["epoch"] for r in history] != list(range(1, 301)):
        raise RuntimeError("300epoch 미완료: test 평가 금지")
    metrics = base.fixed._evaluate(model, prepared)
    if set(metrics) != set(prepared["references"]["metrics"]) or not np.isfinite(list(metrics.values())).all():
        raise RuntimeError("최종142지표 불완전")
    row = {"identity": asdict(identity(prepared)), "model_id": MODEL_ID, "seed": 49,
        "epochs": 300, "metrics": metrics, "training_history": history,
        "diagnostics": {**model.representation_diagnostics(), **model.training_gradient_diagnostics(),
            "graph": "centered_nv", "row_weighted": True, "feature_sha256": prepared["feature_sha256"]}}
    path = prepared["run_dir"] / "arms" / f"{MODEL_ID}.json"
    base.io._atomic_json(path, row)
    store.mark_complete(epoch=300, max_epoch=300, selection="none",
                        checkpoint_path=str(store.latest_checkpoint), result_path=str(path))
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return report(prepared)


def report(prepared):
    refs, row = prepared["references"], completed(prepared)
    frames = [refs["absolute"]]
    diagnostics = [refs["diagnostics"]]
    comparisons = []
    if row is not None:
        frames.append(pd.DataFrame([{"model_id": MODEL_ID, "seed": 49, **row["metrics"]}]))
        diagnostics.append(pd.DataFrame([{"model_id": MODEL_ID, "seed": 49, **row["diagnostics"]}]))
        for ref in refs["absolute"].to_dict("records"):
            for metric in refs["metrics"]:
                x, y = float(row["metrics"][metric]), float(ref[metric])
                comparisons.append({"seed": 49, "model": "M5_D", "reference": LABELS[ref["model_id"]],
                    "metric": metric, "candidate_value": x, "reference_value": y, "delta": x-y,
                    "relative_change_pct": 100*(x/y-1) if y else None})
    absolute = pd.concat(frames, ignore_index=True)
    absolute.insert(0, "model", absolute.model_id.map(LABELS))
    absolute["source"] = np.where(absolute.model_id == MODEL_ID, "new_fit", "reused_seed49")
    reading = {"complete": row is not None, "missing": [] if row else ["M5_D"],
        "seed": 49, "seed_count": 1, "new_fit_count": int(row is not None), "reused_count": 4,
        "fixed_epoch": 300, "significance_claim": False, "clv_attribution_claim": False,
        "unseen_confirmation_claim": False, "note": "1시드 기술적 비교; CI/표준편차 없음; test 재튜닝 금지"}
    tables = {"absolute": absolute, "diagnostics": pd.concat(diagnostics, ignore_index=True),
        "comparison": pd.DataFrame(comparisons, columns=["seed", "model", "reference", "metric",
            "candidate_value", "reference_value", "delta", "relative_change_pct"])}
    paths = {}
    for name, table in tables.items():
        paths[name] = str(prepared["run_dir"] / "reports" / f"{name}.csv")
        base.io._atomic_csv(Path(paths[name]), table)
    paths["json"] = str(prepared["run_dir"] / "reports" / "result.json")
    base.io._atomic_json(Path(paths["json"]), {"protocol": prepared["protocol"], "reading": reading,
                                             "result_paths": paths})
    return {"complete": row is not None, "reading": reading, "paths": paths, **tables}


def export_zip(prepared, destination):
    result = report(prepared)
    if not result["complete"]:
        raise RuntimeError("M5_D 300epoch·최종평가 완료 후 ZIP을 생성하세요")
    with ZipFile(destination, "w", ZIP_DEFLATED) as archive:
        for path in result["paths"].values():
            archive.write(path, Path(path).name)
        for name in ("protocol.json", "reference_audit.json"):
            archive.write(prepared["run_dir"] / name, name)
        archive.write(prepared["run_dir"] / "arms" / f"{MODEL_ID}.json", f"arms/{MODEL_ID}.json")
    return str(destination)

