"""Two new joint fits only: current M5_A=M2+M3 and M5_B=M2+M4.

Reuse immutable final-week seed49 metrics, never reference checkpoints. This
is a descriptive comparison on an already exposed test, not fresh confirmation.
"""
from dataclasses import asdict, dataclass
from io import BytesIO
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import pandas as pd
import torch

import clv_m2_adaptive_lastweek_final10 as m2
import clv_m4_user_centered_final10 as m4
import clv_m5_user_centered_final10 as prior
from clv_run_state import ProgressStore, RunIdentity

fixed, io, v3 = m2.fixed, m2.io, m2.v3
CODE_VERSION = "clv-m5-ab-adaptive-lastweek-seed49-v1"
M5_A = "m5_a_adaptive_m2_centered_nv_m3_bpr_k1_lastweek_s49"
M5_B = "m5_b_adaptive_m2_user_centered_m4_bpr_k1_lastweek_s49"
MODELS = (M5_A, M5_B)
LABELS = {prior.M1: "M1", m2.MODEL_ID: "M2", prior.M3: "M3",
          m4.M4_C: "M4", prior.M5_C: "M5_C", M5_A: "M5_A", M5_B: "M5_B"}
# Exact original report bytes: accepts either their Drive folders or original ZIPs.
REFERENCE_SHA = {
    "m2": {"result.json": "8b40f3b48e1c65d1ee1108db9b896725590512864d89c2bc082f35cd43980da9",
           "absolute.csv": "9c87d9923ccb9c29c7a602815f1e209a8c4529a728d198b83edce7bacb2477f7",
           "diagnostics.csv": "0805e2ee36b1f7afb6bff6b11352ca648356a33762f37e8a1c29b38feaf0b6f4"},
    "m4": {"result.json": "f42111cefd336e759fa2b5077926e1bc1f09116fc7f3b185e0e15f2169fc3023",
           "absolute.csv": "edbc7acb80f8220ab64b12438be505faaccef5617399e902e835567197057f3c",
           "diagnostics.csv": "d8056b28195c21321817fabdbcd8aef57632e0057ce88647b7a9a96514f5564e"},
}


@dataclass(frozen=True)
class Config:
    out_dir: str
    m2_reference: str
    m4_reference: str
    dataset: str = "dunnhumby"
    seeds: tuple = (49,)
    epochs: int = 300
    id_dim: int = 64
    n_layers: int = 2
    batch_size: int = 8192
    lr: float = 5e-4
    pref_reg: float = 1e-3
    negative_count: int = 1
    input_days: int = 365
    bandwidth: float = .25
    reliability_floor: float = .05
    shared_l2: float = 1e-6


def configure(*, out_dir=None, m2_reference=None, m4_reference=None):
    root = v3.default_out_dir("dunnhumby")
    return Config(out_dir or root + "_m5_ab_adaptive_lastweek_seed49_v1",
                  m2_reference or root + "_m2_adaptive_lastweek_final10_v1",
                  m4_reference or root + "_m4c_lastweek_final10_v1")


def _validate_config(cfg):
    if cfg != configure(out_dir=cfg.out_dir, m2_reference=cfg.m2_reference,
                        m4_reference=cfg.m4_reference):
        raise ValueError("고정 seed49·300epoch·기존 구성요소 설정을 바꿀 수 없습니다")
    if any(Path(cfg.out_dir).resolve() == Path(p).resolve()
           for p in (cfg.m2_reference, cfg.m4_reference)):
        raise ValueError("새 결과는 참조 결과와 별도 폴더에 저장해야 합니다")


def _read_reference(source, kind):
    source = Path(source)
    expected = REFERENCE_SHA[kind]
    matches = []
    if source.suffix.lower() == ".zip" and source.is_file():
        with ZipFile(source) as archive:
            matches.append((str(source), {name: archive.read(name) for name in expected}))
    elif source.is_dir():
        candidates = {source / "result.json", source / "reports/result.json"}
        candidates.update(source.glob("*/reports/result.json"))
        for path in sorted(candidates):
            if path.is_file() and all((path.parent / n).is_file() for n in expected):
                matches.append((str(path.parent), {n: (path.parent / n).read_bytes() for n in expected}))
    verified = [(location, blobs) for location, blobs in matches if all(
        hashlib.sha256(blobs[n]).hexdigest() == sha for n, sha in expected.items())]
    if not verified:
        raise RuntimeError(f"{kind} 원본과 일치하는 완료 결과가 없습니다: {source}\n"
                           "기존 원본 ZIP 또는 완료 reports 폴더를 지정하세요. 참조를 재학습하지 않습니다.")
    location, blobs = verified[0]  # Any duplicate has identical verified bytes.
    raw = json.loads(blobs["result.json"])
    absolute = pd.read_csv(BytesIO(blobs["absolute.csv"]))
    diagnostics = pd.read_csv(BytesIO(blobs["diagnostics.csv"]))
    return {"protocol": raw["protocol"], "absolute": absolute,
            "diagnostics": diagnostics, "audit": {"location": location, "sha256": expected}}


def load_references(cfg):
    _validate_config(cfg)
    refs = {kind: _read_reference(getattr(cfg, kind + "_reference"), kind) for kind in ("m2", "m4")}
    for ref in refs.values():
        p = ref["protocol"]
        if p["split"] != "train DAY 1-704; test DAY 705-711":
            raise RuntimeError("참조 평가 구간이 다릅니다")
        for key in ("dataset", "epochs", "id_dim", "n_layers", "batch_size", "lr",
                    "pref_reg", "negative_count", "input_days"):
            if p["config"][key] != getattr(cfg, key):
                raise RuntimeError(f"참조 학습 설정 불일치: {key}")
    for key in ("input_hash", "data_stats"):
        if refs["m2"]["protocol"][key] != refs["m4"]["protocol"][key]:
            raise RuntimeError(f"M2/M4 원본의 {key} 불일치")
    absolute = pd.concat([r["absolute"].query("seed == 49") for r in refs.values()], ignore_index=True)
    if len(absolute) != 5 or set(absolute.model_id) != set(LABELS) - set(MODELS):
        raise RuntimeError("seed49 참조 5개 모형이 모두 필요합니다")
    metrics = [c for c in absolute if c not in ("model_id", "seed")]
    if len(metrics) != 142 or not np.isfinite(absolute[metrics].to_numpy(float)).all():
        raise RuntimeError("참조의 전체142지표가 불완전합니다")
    return {"sources": refs, "absolute": absolute, "metrics": metrics,
            "diagnostics": pd.concat([r["diagnostics"].query("seed == 49").assign(
                model_id=m2.MODEL_ID) if k == "m2" else r["diagnostics"].query("seed == 49")
                for k, r in refs.items()], ignore_index=True),
            "audit": {k: r["audit"] for k, r in refs.items()}}


def prepare(cfg):
    references = load_references(cfg)  # Fail before expensive data preparation or fitting.
    p2, p4 = (references["sources"][k]["protocol"] for k in ("m2", "m4"))
    for name, sha in p4["frozen_code_sha256"].items():
        if hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() != sha:
            raise RuntimeError(f"확정 M3/M4 구현이 변경되었습니다: {name}")
    # The original preparation computes graph/weights only, never trains any arm.
    prepared = prior.prepare(prior.configure(out_dir=str(Path(cfg.out_dir) / "inputs")))
    prepared["base_cfg"]["SEED_LIST"] = [49]
    if prepared["input_hash"] != p2["input_hash"] or prepared["data"]["data_stats"] != p2["data_stats"]:
        raise RuntimeError("학습 입력 또는 마지막주 분할이 기존 결과와 다릅니다")
    if prepared["observed_weight_audit"]["sha256"] != p4["m4_c_weight_audit"]["sha256"]:
        raise RuntimeError("M4 가중치가 기존 구현 결과와 다릅니다")
    data = prepared["data"]
    if "b_raw" not in data["train"]:
        raise RuntimeError("Dunnhumby BASKET_ID 기반 구매단위가 필요합니다")
    features = m2.build_features(data["train"], n_users=data["n_users"], n_items=data["n_items"],
        q_n=prepared["q_n"], q_v=prepared["q_v"], valid=prepared["clv_valid"],
        bandwidth=cfg.bandwidth, reliability_floor=cfg.reliability_floor)
    feature_sha = m2._feature_hash(features)
    if feature_sha != p2["feature_sha256"] or not np.array_equal(features["keys"], data["pos_key"]):
        raise RuntimeError("M2 N/V·상품 특징 또는 양성쌍 순서가 기존 결과와 다릅니다")
    protocol = {
        "code_version": CODE_VERSION, "config": asdict(cfg),
        "source_revision": prepared["revision"], "input_hash": prepared["input_hash"],
        "split": p2["split"], "data_stats": data["data_stats"],
        "feature_sha256": feature_sha, "m3_audit": prepared["observed_graph"]["audit"],
        "m4_weight_audit": prepared["observed_weight_audit"],
        "reference_audit": references["audit"], "labels": LABELS,
        "new_fit_models": list(MODELS), "reused_models": list(references["absolute"].model_id),
        "seeds": [49], "fixed_epoch": 300, "fixed": p4["fixed"],
        "m5_a": "adaptive M2 representation + centered N/V graph + plain BPR",
        "m5_b": "adaptive M2 representation + binary graph + user-centered M4 BPR weights",
        "initialization": "fresh joint initialization per arm; no reference checkpoints loaded",
        "decision": "all142 metrics paired against seed49 references; no best-arm/test-based tuning",
        "significance_claim": False, "unseen_confirmation_claim": False,
        "clv_attribution_claim": False, "single_seed_only": True,
    }
    config_hash = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()[:12]
    run_dir = Path(cfg.out_dir) / config_hash
    io._atomic_json(run_dir / "protocol.json", protocol)
    io._atomic_json(run_dir / "reference_audit.json", references["audit"])
    prepared.update(features=features, feature_sha256=feature_sha, config_hash=config_hash,
                    run_dir=run_dir, protocol=protocol, references=references, cfg=cfg)
    print("확인: seed49 참조 5개 재사용 / 새 학습 M5_A → M5_B / 각300epoch", flush=True)
    return prepared


def arm_inputs(prepared, model_id):
    if model_id == M5_A:
        return prepared["observed_adj"], None
    if model_id == M5_B:
        return prepared["data"]["adj"], prepared["observed_weights"]
    raise ValueError("새로 학습할 수 있는 모형은 M5_A와 M5_B뿐입니다")


def _identity(prepared, model_id):
    arm_inputs(prepared, model_id)
    return RunIdentity(CODE_VERSION, model_id, 49, prepared["config_hash"],
                       prepared["revision"], prepared["input_hash"])


def _completed(prepared, model_id):
    path = prepared["run_dir"] / "arms" / f"{model_id}.json"
    if not path.exists():
        return None
    row = json.loads(path.read_text())
    if (row.get("identity") != asdict(_identity(prepared, model_id)) or row.get("epochs") != 300
            or set(row["metrics"]) != set(prepared["references"]["metrics"])
            or not np.isfinite(list(row["metrics"].values())).all()):
        raise RuntimeError("완료 결과의 신원 또는142지표가 다릅니다")
    return row


def run_arm(prepared, model_id):
    cfg = prepared["cfg"]
    _validate_config(cfg)
    completed = _completed(prepared, model_id)
    if completed is not None:
        print(f"[재사용] {LABELS[model_id]} seed49 — 재학습·재평가 없음", flush=True)
        return completed
    if m2._feature_hash(prepared["features"]) != prepared["feature_sha256"]:
        raise RuntimeError("준비 후 M2 특징 변경")
    adjacency, weights = arm_inputs(prepared, model_id)
    data = prepared["data"]
    v3.set_seed(49)
    model = m2.DatasetAdaptiveOrthogonalNVLightGCN(
        n_users=data["n_users"], n_items=data["n_items"], features=prepared["features"],
        adj=adjacency, id_dim=cfg.id_dim, n_layers=cfg.n_layers,
        pref_reg=cfg.pref_reg, shared_l2=cfg.shared_l2).to(v3.DEVICE)
    identity = _identity(prepared, model_id)
    store = ProgressStore(prepared["run_dir"] / "progress", identity)
    history = fixed._train(model, prepared, cfg, model_id, 49, store, row_weights=weights)
    if len(history) != 300 or history[-1]["epoch"] != 300:
        raise RuntimeError("300epoch 미완료: test 평가 금지")
    metrics = fixed._evaluate(model, prepared)
    if set(metrics) != set(prepared["references"]["metrics"]) or not np.isfinite(list(metrics.values())).all():
        raise RuntimeError("최종142지표 불완전")
    row = {"identity": asdict(identity), "model_id": model_id, "seed": 49,
           "epochs": 300, "metrics": metrics, "training_history": history,
           "diagnostics": {**model.representation_diagnostics(), **model.training_gradient_diagnostics(),
               "graph": "centered_nv" if model_id == M5_A else "binary",
               "row_weighted": weights is not None, "feature_sha256": prepared["feature_sha256"]}}
    path = prepared["run_dir"] / "arms" / f"{model_id}.json"
    io._atomic_json(path, row)
    store.mark_complete(epoch=300, max_epoch=300, selection="none",
                        checkpoint_path=str(store.latest_checkpoint), result_path=str(path))
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return row


def report(prepared):
    rows = [r for model in MODELS if (r := _completed(prepared, model)) is not None]
    refs = prepared["references"]
    absolute = pd.concat([refs["absolute"], pd.DataFrame([
        {"model_id": r["model_id"], "seed": 49, **r["metrics"]} for r in rows])], ignore_index=True)
    absolute.insert(0, "model", absolute.model_id.map(LABELS))
    absolute["source"] = np.where(absolute.model_id.isin(MODELS), "new_fit", "reused_seed49")
    comparisons = []
    for candidate in rows:
        for reference in refs["absolute"].to_dict("records"):
            for metric in refs["metrics"]:
                old, new = float(reference[metric]), float(candidate["metrics"][metric])
                comparisons.append({"seed": 49, "model": LABELS[candidate["model_id"]],
                    "reference": LABELS[reference["model_id"]], "metric": metric,
                    "reference_value": old, "candidate_value": new, "delta": new - old,
                    "relative_change_pct": 100 * (new / old - 1) if old != 0 else None})
    comparison = pd.DataFrame(comparisons, columns=["seed", "model", "reference", "metric",
        "reference_value", "candidate_value", "delta", "relative_change_pct"])
    diagnostics = pd.concat([refs["diagnostics"], pd.DataFrame([
        {"model_id": r["model_id"], "seed": 49, **r["diagnostics"]} for r in rows])], ignore_index=True)
    missing = [LABELS[m] for m in MODELS if m not in {r["model_id"] for r in rows}]
    reading = {"complete": not missing, "missing": missing, "seed": 49, "seed_count": 1,
        "fixed_epoch": 300, "new_fit_count": len(rows), "reused_count": 5,
        "significance_claim": False, "clv_attribution_claim": False,
        "unseen_confirmation_claim": False, "note": "1시드 기술적 비교; 모든 불리한 지표 포함, test 재튜닝 금지"}
    root = prepared["run_dir"] / "reports"
    paths = {}
    for name, frame in (("absolute", absolute), ("comparison", comparison), ("diagnostics", diagnostics)):
        paths[name] = str(root / f"{name}.csv")
        io._atomic_csv(Path(paths[name]), frame)
    paths["json"] = str(root / "result.json")
    io._atomic_json(Path(paths["json"]), {"protocol": prepared["protocol"], "reading": reading, "result_paths": paths})
    return {"complete": not missing, "missing": missing, "reading": reading,
            "absolute": absolute, "comparison": comparison, "diagnostics": diagnostics, "paths": paths}


def run(prepared):
    for index, model_id in enumerate(MODELS, 1):
        print(f"[{index}/2] {LABELS[model_id]} seed49", flush=True)
        run_arm(prepared, model_id)
        result = report(prepared)  # Save completed-arm metrics before starting the next fit.
    return result


def export_zip(prepared, destination):
    result = report(prepared)
    if not result["complete"]:
        raise RuntimeError(f"두 결합모형 완료 후 ZIP 생성: 미완료 {result['missing']}")
    with ZipFile(destination, "w", ZIP_DEFLATED) as archive:
        for path in result["paths"].values():
            archive.write(path, Path(path).name)
        for name in ("protocol.json", "reference_audit.json"):
            archive.write(prepared["run_dir"] / name, name)
        for model_id in MODELS:
            archive.write(prepared["run_dir"] / "arms" / f"{model_id}.json", f"arms/{model_id}.json")
    return str(destination)
