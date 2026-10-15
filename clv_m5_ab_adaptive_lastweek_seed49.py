"""Fixed combination batches: DH A/B seeds42--51, HM B seed49 only.

Reuse immutable final-week same-seed metrics, never reference checkpoints. This
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
import clv_m5_user_centered_hm2y_final10 as hm
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
    "hm_m2": {"result.json": "99c2a72fc7e7d64258185839315a33a7a6922e96a0a38291c2f3edfcb2a24aae",
              "absolute.csv": "ffc227becd163278e933f29b2668ae65d5377237705db015fd6be558a00c5e03",
              "diagnostics.csv": "3e0d57beb265ecb5c4c30f8f8924558b6fb765f5acb8a23e6001e0d6366f3f28"},
    "hm_m4": {"result.json": "d87807e619ca8bf89fb8c276f94cebec752fdcb623dff79a50615b0613377a2f",
              "absolute.csv": "2a6a010a1a4e7a7fd9120203be9129ce863481da913c79686e81da716fb95310",
              "diagnostics.csv": "00d8f24b5feaf17b911fa477d29b8d9997b77d14c98e4d17f3b08ec5794cf93e"},
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


def configure(*, dataset="dunnhumby", seed=49, out_dir=None, m2_reference=None, m4_reference=None):
    if not ((dataset == "dunnhumby" and seed in range(42, 52)) or (dataset, seed) == ("hm", 49)):
        raise ValueError("승인된 DH seed42~51 A·B 또는 H&M seed49 B만 허용합니다")
    root = v3.default_out_dir(dataset)
    suffix = f"_m5_ab_adaptive_lastweek_seed{seed}_v1" if dataset == "dunnhumby" else "_m5_b_adaptive_lastweek_seed49_v1"
    return Config(out_dir or root + suffix,
                  m2_reference or root + "_m2_adaptive_lastweek_final10_v1",
                  m4_reference or root + ("_m4c_lastweek_final10_v1" if dataset == "dunnhumby" else "_m5c_lastweek_final10_v1"),
                  dataset=dataset, seeds=(seed,), batch_size=8192 if dataset == "dunnhumby" else 131072)


def models_for(cfg):
    if cfg.dataset == "hm":
        return (M5_B.replace("_lastweek_s49", "_hm2y_lastweek_s49"),)
    return tuple(m.replace("_s49", f"_s{cfg.seeds[0]}") for m in MODELS)


def labels_for(cfg):
    if cfg.dataset == "hm":
        return {hm.M1: "M1", m2.MODEL_ID: "M2", hm.M3: "M3", hm.M4_C: "M4",
                models_for(cfg)[0]: "M5_B"}
    return {**{k: v for k, v in LABELS.items() if k not in MODELS},
            **dict(zip(models_for(cfg), ("M5_A", "M5_B")))}


def _validate_config(cfg):
    if len(cfg.seeds) != 1 or cfg != configure(dataset=cfg.dataset, seed=cfg.seeds[0], out_dir=cfg.out_dir, m2_reference=cfg.m2_reference,
                        m4_reference=cfg.m4_reference):
        raise ValueError("고정 실행 배치·300epoch·기존 구성요소 설정을 바꿀 수 없습니다")
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
        for folder in ("reports", "partial_reports", "m4c_seed49_export"):
            candidates.add(source / folder / "result.json")
            candidates.update(source.glob(f"*/{folder}/result.json"))
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
    seed = cfg.seeds[0]
    labels = labels_for(cfg)
    refs = {kind: _read_reference(getattr(cfg, kind + "_reference"),
             ("hm_" if cfg.dataset == "hm" else "") + kind) for kind in ("m2", "m4")}
    for ref in refs.values():
        p = ref["protocol"]
        if cfg.dataset == "dunnhumby" and p["split"] != "train DAY 1-704; test DAY 705-711":
            raise RuntimeError("참조 평가 구간이 다릅니다")
        for key in ("dataset", "epochs", "id_dim", "n_layers", "batch_size", "lr",
                    "pref_reg", "negative_count", "input_days"):
            if p["config"][key] != getattr(cfg, key):
                raise RuntimeError(f"참조 학습 설정 불일치: {key}")
    for key in ("input_hash", "data_stats"):
        if refs["m2"]["protocol"][key] != refs["m4"]["protocol"][key]:
            raise RuntimeError(f"M2/M4 원본의 {key} 불일치")
    absolute = pd.concat([r["absolute"].loc[r["absolute"].seed == seed] for r in refs.values()], ignore_index=True)
    expected_models = set(labels) - set(models_for(cfg))
    if len(absolute) != len(expected_models) or set(absolute.model_id) != expected_models:
        raise RuntimeError(f"seed{seed} 참조 {len(expected_models)}개 모형이 모두 필요합니다")
    metrics = [c for c in absolute if c not in ("model_id", "seed")]
    if len(metrics) != 142 or not np.isfinite(absolute[metrics].to_numpy(float)).all():
        raise RuntimeError("참조의 전체142지표가 불완전합니다")
    return {"sources": refs, "absolute": absolute, "metrics": metrics,
            "diagnostics": pd.concat([r["diagnostics"].loc[r["diagnostics"].seed == seed].assign(
                model_id=m2.MODEL_ID) if k == "m2" else r["diagnostics"].loc[r["diagnostics"].seed == seed]
                for k, r in refs.items()], ignore_index=True),
            "audit": {k: r["audit"] for k, r in refs.items()}}


def _prepare_hm_inputs(cfg):
    # Binary M2 preparation only: no M3 graph, no reference fitting/evaluation.
    prepared = m2.prepare(m2.configure("hm", out_dir=str(Path(cfg.out_dir) / "inputs")))
    hm.validate_split(prepared["data"])
    data = prepared["data"]
    q_n, q_v, q_c, valid = fixed.evaluation.build_clv_inputs(prepared["axes"])
    q_n, q_v, q_c = (np.where(valid, q, 0).astype(np.float32) for q in (q_n, q_v, q_c))
    econ = hm.economics.build_nv_economic_inputs(
        data["train"], n_users=data["n_users"], n_items=data["n_items"],
        q_n=q_n, q_v=q_v, q_c=q_c, clv_valid=valid,
        n_bins=4, shrinkage_strength=10.0, degree_bins=10)
    prepared.update(q_n=q_n, q_v=q_v, q_c=q_c, clv_valid=np.asarray(valid, bool))
    prepared.update({key: econ[key] for key in ("item_amount_percentile", "item_economic_valid",
        "user_economic_valid", "user_bin_fit", "item_bin")})
    weights, audit = hm.m4c.row_weights(prepared)
    prepared.update(observed_weights=weights, observed_weight_audit=audit)
    return prepared


def prepare(cfg):
    references = load_references(cfg)  # Fail before expensive data preparation or fitting.
    p2, p4 = (references["sources"][k]["protocol"] for k in ("m2", "m4"))
    for name, sha in p4.get("frozen_code_sha256", {}).items():
        if hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() != sha:
            raise RuntimeError(f"확정 M3/M4 구현이 변경되었습니다: {name}")
    # The original preparation computes graph/weights only, never trains any arm.
    prepared = (_prepare_hm_inputs(cfg) if cfg.dataset == "hm" else
                prior.prepare(prior.configure(out_dir=str(Path(cfg.out_dir) / "inputs"))))
    prepared["base_cfg"]["SEED_LIST"] = list(cfg.seeds)
    if prepared["input_hash"] != p2["input_hash"] or prepared["data"]["data_stats"] != p2["data_stats"]:
        raise RuntimeError("학습 입력 또는 마지막주 분할이 기존 결과와 다릅니다")
    if prepared["observed_weight_audit"]["sha256"] != p4["m4_c_weight_audit"]["sha256"]:
        raise RuntimeError("M4 가중치가 기존 구현 결과와 다릅니다")
    data = prepared["data"]
    if cfg.dataset == "dunnhumby" and "b_raw" not in data["train"]:
        raise RuntimeError("Dunnhumby BASKET_ID 기반 구매단위가 필요합니다")
    features = prepared.get("features")
    if features is None:
        features = m2.build_features(data["train"], n_users=data["n_users"], n_items=data["n_items"],
        q_n=prepared["q_n"], q_v=prepared["q_v"], valid=prepared["clv_valid"],
        bandwidth=cfg.bandwidth, reliability_floor=cfg.reliability_floor)
    feature_sha = m2._feature_hash(features)
    if feature_sha != p2["feature_sha256"] or not np.array_equal(features["keys"], data["pos_key"]):
        raise RuntimeError("M2 N/V·상품 특징 또는 양성쌍 순서가 기존 결과와 다릅니다")
    protocol = {
        "code_version": CODE_VERSION if (cfg.dataset, cfg.seeds) == ("dunnhumby", (49,)) else
                        f"clv-m5-adaptive-fixed-{cfg.dataset}-seed{cfg.seeds[0]}-v1",
        "config": asdict(cfg),
        "source_revision": prepared["revision"], "input_hash": prepared["input_hash"],
        "split": p2["split"], "data_stats": data["data_stats"],
        "feature_sha256": feature_sha, "m3_audit": prepared.get("observed_graph", {}).get("audit"),
        "m4_weight_audit": prepared["observed_weight_audit"],
        "reference_audit": references["audit"], "labels": labels_for(cfg),
        "new_fit_models": list(models_for(cfg)), "reused_models": list(references["absolute"].model_id),
        "seeds": list(cfg.seeds), "fixed_epoch": 300, "fixed": p4["fixed"],
        "m5_a": "adaptive M2 representation + centered N/V graph + plain BPR",
        "m5_b": "adaptive M2 representation + binary graph + user-centered M4 BPR weights",
        "initialization": "fresh joint initialization per arm; no reference checkpoints loaded",
        "decision": f"all142 metrics paired against seed{cfg.seeds[0]} references; no best-arm/test-based tuning",
        "significance_claim": False, "unseen_confirmation_claim": False,
        "clv_attribution_claim": False, "single_seed_only": True,
    }
    config_hash = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()[:12]
    run_dir = Path(cfg.out_dir) / config_hash
    io._atomic_json(run_dir / "protocol.json", protocol)
    io._atomic_json(run_dir / "reference_audit.json", references["audit"])
    prepared.update(features=features, feature_sha256=feature_sha, config_hash=config_hash,
                    run_dir=run_dir, protocol=protocol, references=references, cfg=cfg)
    print(f"확인: {cfg.dataset} seed{cfg.seeds[0]} 참조 {len(references['absolute'])}개 재사용 / "
          f"새 학습 {[labels_for(cfg)[m] for m in models_for(cfg)]} / 각300epoch", flush=True)
    return prepared


def arm_inputs(prepared, model_id):
    cfg = prepared["cfg"]
    if model_id not in models_for(cfg):
        raise ValueError("승인된 결합모형 외에는 학습할 수 없습니다")
    if labels_for(cfg)[model_id] == "M5_A":
        return prepared["observed_adj"], None
    if labels_for(cfg)[model_id] == "M5_B":
        return prepared["data"]["adj"], prepared["observed_weights"]
    raise ValueError("새로 학습할 수 있는 모형은 M5_A와 M5_B뿐입니다")


def _identity(prepared, model_id):
    arm_inputs(prepared, model_id)
    return RunIdentity(prepared["protocol"].get("code_version", CODE_VERSION), model_id,
                       prepared["cfg"].seeds[0], prepared["config_hash"],
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
    seed, labels = cfg.seeds[0], labels_for(cfg)
    _validate_config(cfg)
    completed = _completed(prepared, model_id)
    if completed is not None:
        print(f"[재사용] {labels[model_id]} seed{seed} — 재학습·재평가 없음", flush=True)
        return completed
    if m2._feature_hash(prepared["features"]) != prepared["feature_sha256"]:
        raise RuntimeError("준비 후 M2 특징 변경")
    adjacency, weights = arm_inputs(prepared, model_id)
    data = prepared["data"]
    v3.set_seed(seed)
    model = m2.DatasetAdaptiveOrthogonalNVLightGCN(
        n_users=data["n_users"], n_items=data["n_items"], features=prepared["features"],
        adj=adjacency, id_dim=cfg.id_dim, n_layers=cfg.n_layers,
        pref_reg=cfg.pref_reg, shared_l2=cfg.shared_l2).to(v3.DEVICE)
    identity = _identity(prepared, model_id)
    store = ProgressStore(prepared["run_dir"] / "progress", identity)
    history = fixed._train(model, prepared, cfg, model_id, seed, store, row_weights=weights)
    if len(history) != 300 or history[-1]["epoch"] != 300:
        raise RuntimeError("300epoch 미완료: test 평가 금지")
    metrics = fixed._evaluate(model, prepared)
    if set(metrics) != set(prepared["references"]["metrics"]) or not np.isfinite(list(metrics.values())).all():
        raise RuntimeError("최종142지표 불완전")
    row = {"identity": asdict(identity), "model_id": model_id, "seed": seed,
           "epochs": 300, "metrics": metrics, "training_history": history,
           "diagnostics": {**model.representation_diagnostics(), **model.training_gradient_diagnostics(),
               "graph": "centered_nv" if labels[model_id] == "M5_A" else "binary",
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
    cfg = prepared["cfg"]
    seed, models, labels = cfg.seeds[0], models_for(cfg), labels_for(cfg)
    rows = [r for model in models if (r := _completed(prepared, model)) is not None]
    refs = prepared["references"]
    absolute = pd.concat([refs["absolute"], pd.DataFrame([
        {"model_id": r["model_id"], "seed": seed, **r["metrics"]} for r in rows])], ignore_index=True)
    absolute.insert(0, "model", absolute.model_id.map(labels))
    absolute["source"] = np.where(absolute.model_id.isin(models), "new_fit", f"reused_seed{seed}")
    comparisons = []
    for candidate in rows:
        for reference in refs["absolute"].to_dict("records"):
            for metric in refs["metrics"]:
                old, new = float(reference[metric]), float(candidate["metrics"][metric])
                comparisons.append({"seed": seed, "model": labels[candidate["model_id"]],
                    "reference": labels[reference["model_id"]], "metric": metric,
                    "reference_value": old, "candidate_value": new, "delta": new - old,
                    "relative_change_pct": 100 * (new / old - 1) if old != 0 else None})
    comparison = pd.DataFrame(comparisons, columns=["seed", "model", "reference", "metric",
        "reference_value", "candidate_value", "delta", "relative_change_pct"])
    diagnostics = pd.concat([refs["diagnostics"], pd.DataFrame([
        {"model_id": r["model_id"], "seed": seed, **r["diagnostics"]} for r in rows])], ignore_index=True)
    missing = [labels[m] for m in models if m not in {r["model_id"] for r in rows}]
    reading = {"complete": not missing, "missing": missing, "seed": seed, "seed_count": 1,
        "fixed_epoch": 300, "new_fit_count": len(rows), "reused_count": len(refs["absolute"]),
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
    cfg = prepared["cfg"]
    models, labels = models_for(cfg), labels_for(cfg)
    for index, model_id in enumerate(models, 1):
        print(f"[{index}/{len(models)}] {labels[model_id]} seed{cfg.seeds[0]}", flush=True)
        run_arm(prepared, model_id)
        result = report(prepared)  # Save completed-arm metrics before starting the next fit.
    return result


def export_zip(prepared, destination):
    result = report(prepared)
    if not result["complete"]:
        raise RuntimeError(f"이번 배치 완료 후 ZIP 생성: 미완료 {result['missing']}")
    with ZipFile(destination, "w", ZIP_DEFLATED) as archive:
        for path in result["paths"].values():
            archive.write(path, Path(path).name)
        for name in ("protocol.json", "reference_audit.json"):
            archive.write(prepared["run_dir"] / name, name)
        for model_id in models_for(prepared["cfg"]):
            archive.write(prepared["run_dir"] / "arms" / f"{model_id}.json", f"arms/{model_id}.json")
    return str(destination)
