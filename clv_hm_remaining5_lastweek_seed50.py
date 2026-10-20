"""Fixed H&M seed50 complement to the independent M1/M3/M5_C batch.

Only M2, M4, M5_A, M5_B, M5_D may train. Seed49 reports establish the
unchanged input/design fingerprints, never substitute for seed50 metrics.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import pandas as pd
import torch

import clv_m5_ab_adaptive_lastweek_seed49 as base

VERSION = "clv-hm-remaining5-lastweek-seed50-v1"
SEED = 50
M2, M4 = base.m2.MODEL_ID, base.hm.M4_C
M5_A = "m5_a_adaptive_m2_centered_nv_m3_bpr_k1_hm2y_lastweek_s50"
M5_B = "m5_b_adaptive_m2_user_centered_m4_bpr_k1_hm2y_lastweek_s50"
M5_D = "m5_d_adaptive_m2_centered_nv_m3_user_centered_m4_bpr_k1_hm2y_lastweek_s50"
MODELS = (M2, M4, M5_A, M5_B, M5_D)
LABELS = {base.hm.M1: "M1", M2: "M2", base.hm.M3: "M3", M4: "M4",
          M5_A: "M5_A", M5_B: "M5_B", base.hm.M5_C: "M5_C", M5_D: "M5_D"}
CORE = tuple(f"{m}@{k}" for k in (10, 20, 50) for m in
             ("recall", "ndcg", "price_purchase_amount_weighted_hit", "vndcg"))
GRAPH_KEYS = ("target_cv", "max_degree_correlation", "max_price_correlation", "max_popularity_correlation")
TRAIN_KEYS = ("dataset", "epochs", "id_dim", "n_layers", "batch_size", "lr",
              "pref_reg", "negative_count", "input_days")


class ProgressStore(base.ProgressStore):
    """Keep epoch saving unchanged; fail closed instead of deleting corrupt saves."""

    def restore_epoch(self, model, optimizer, rng):
        if not self.latest_checkpoint.exists():
            return None
        try:
            payload = torch.load(self.latest_checkpoint, map_location="cpu", weights_only=False)
        except Exception as error:
            raise RuntimeError("checkpoint를 읽을 수 없습니다. 원본을 보존하고 중단합니다; 자동 재학습하지 않습니다") from error
        self._validate_identity(payload.get("identity", {}))
        epoch = payload.get("epoch")
        if not isinstance(epoch, int) or not 1 <= epoch <= 300 or [r["epoch"] for r in payload.get("history", [])] != list(range(1, epoch + 1)):
            raise RuntimeError("checkpoint epoch/이력 불일치: 원본 보존 후 중단")
        model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        rng.bit_generator.state = payload["numpy_rng_state"]
        torch.set_rng_state(payload["torch_rng_state"])
        if torch.cuda.is_available() and payload.get("cuda_rng_state"):
            torch.cuda.set_rng_state_all(payload["cuda_rng_state"])
        return {"next_epoch": epoch + 1, "history": payload["history"]}


@dataclass(frozen=True)
class Config(base.Config):
    core_reference: str = ""


def configure(*, out_dir=None, m2_reference=None, m4_reference=None, core_reference=None):
    root = base.v3.default_out_dir("hm")
    old = base.configure(dataset="hm", out_dir=out_dir or root + "_remaining5_lastweek_seed50_v1",
                         m2_reference=m2_reference, m4_reference=m4_reference)
    return Config(**(asdict(old) | {"seeds": (SEED,),
        "core_reference": core_reference or root + "_m5c_lastweek_final10_v1"}))


def validate_config(cfg):
    if cfg != configure(**{k: getattr(cfg, k) for k in
                          ("out_dir", "m2_reference", "m4_reference", "core_reference")}):
        raise ValueError("H&M seed50·고정300epoch·기존 설계만 허용합니다")
    output = Path(cfg.out_dir).resolve()
    for source in (cfg.m2_reference, cfg.m4_reference, cfg.core_reference):
        ref = Path(source).resolve()
        if output == ref or output in ref.parents or ref in output.parents:
            raise ValueError("새 출력은 참조 폴더와 겹칠 수 없습니다")


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _check_protocol(protocol, cfg, anchor):
    for key in TRAIN_KEYS:
        if protocol["config"][key] != getattr(cfg, key):
            raise RuntimeError(f"참조 학습 설정 불일치: {key}")
    if any(protocol[k] != anchor[k] for k in ("input_hash", "data_stats")):
        raise RuntimeError("참조 입력/마지막주 분할 불일치")


def _core_row(run_dir, protocol, model_id, metrics):
    path = run_dir / "arms" / f"{model_id}_s{SEED}.json"
    if not path.is_file():
        return None
    row = json.loads(path.read_text())
    identity = base.RunIdentity(base.hm.CODE_VERSION, model_id, SEED, run_dir.name,
                               protocol["source_revision"], protocol["input_hash"])
    if (row.get("identity") != asdict(identity) or row.get("seed") != SEED
            or row.get("model_id") != model_id or row.get("epochs") != 300
            or set(row.get("metrics", {})) != set(metrics)
            or not np.isfinite(list(row["metrics"].values())).all()):
        raise RuntimeError(f"seed50 완료 결과 신원/142지표 불일치: {path}")
    return row


def load_references(cfg):
    """Read-only preflight; missing M1 stops before costly data preparation."""
    validate_config(cfg)
    design = {k: base._read_reference(getattr(cfg, k + "_reference"), "hm_" + k)
              for k in ("m2", "m4")}
    p2, p3 = (design[k]["protocol"] for k in ("m2", "m4"))
    for p in (p2, p3):
        _check_protocol(p, cfg, p2)
    for key in ("bandwidth", "reliability_floor", "shared_l2"):
        if p2["config"][key] != getattr(cfg, key):
            raise RuntimeError(f"M2 고정 설정 불일치: {key}")
    gc = base.hm.configure()
    for key in GRAPH_KEYS:
        if p3["config"][key] != getattr(gc, key):
            raise RuntimeError(f"M3 고정 설정 불일치: {key}")
    metrics = [c for c in design["m2"]["absolute"] if c not in ("model_id", "seed")]
    if len(metrics) != 142:
        raise RuntimeError("설계 참조의 전체142지표 누락")
    source = Path(cfg.core_reference)
    paths = [source / "protocol.json", *sorted(source.glob("*/protocol.json"))]
    found = []
    for path in paths:
        if not path.is_file():
            continue
        protocol = json.loads(path.read_text())
        if protocol.get("code_version") != base.hm.CODE_VERSION:
            continue
        arm_path = path.parent / "arms" / f"{base.hm.M1}_s{SEED}.json"
        if not arm_path.is_file():
            continue
        _check_protocol(protocol, cfg, p2)
        if (any(protocol["config"][k] != p3["config"][k] for k in GRAPH_KEYS)
                or protocol["m3_audit"] != p3["m3_audit"]
                or protocol["m4_c_weight_audit"]["sha256"] != p3["m4_c_weight_audit"]["sha256"]):
            raise RuntimeError("참조 M3/M4 설계 불일치")
        row = _core_row(path.parent, protocol, base.hm.M1, metrics)
        found.append((path.parent, protocol, row, arm_path))
    if not found:
        raise RuntimeError("같은 조건의 H&M M1 seed50 완료 JSON이 없습니다. 기존 Colab에서 완료 후 재시도하세요. M1을 새로 학습하지 않습니다.")
    if any(x[2]["metrics"] != found[0][2]["metrics"] for x in found[1:]):
        raise RuntimeError("서로 다른 M1 seed50 결과가 여러 개입니다. core_reference에 정확한 실행 폴더를 지정하세요")
    run_dir, protocol, row, path = found[0]
    return {"design": design, "metrics": metrics, "core_dir": run_dir, "core_protocol": protocol,
            "m1": row, "audit": {"design_seed49_only": {k: v["audit"] for k, v in design.items()},
                "m1_seed50": {"path": str(path), "sha256": _digest(path),
                              "protocol_sha256": _digest(run_dir / "protocol.json")}}}


def prepare(cfg):
    started = time.monotonic()
    refs = load_references(cfg)
    p2, p3 = (refs["design"][k]["protocol"] for k in ("m2", "m4"))
    print("사전 확인: M1 seed50 재사용. seed49 파일은 설계/입력 검증용이며 성능표에 섞지 않습니다.", flush=True)
    prepared = base._prepare_hm_inputs(cfg)  # One shared preparation; no model training.
    data = prepared["data"]
    base.hm.validate_split(data)
    if prepared["input_hash"] != p2["input_hash"] or data["data_stats"] != p2["data_stats"]:
        raise RuntimeError("실제 입력/분할이 참조와 다릅니다")
    if (prepared["feature_sha256"] != p2["feature_sha256"] or
            not np.array_equal(prepared["features"]["keys"], data["pos_key"])):
        raise RuntimeError("M2 특징/구매쌍 순서 불일치")
    if prepared["observed_weight_audit"]["sha256"] != p3["m4_c_weight_audit"]["sha256"]:
        raise RuntimeError("M4 고정 손실 가중치 불일치")
    print(f"입력·M2 특징·M4 가중치 준비: {time.monotonic()-started:.1f}초", flush=True)
    graph_start = time.monotonic()
    keyed = base.hm.hm_graph._purchase_keyed(data["train"])
    signals = base.fixed.graph.centered_edge_signals(keyed, data["n_users"], data["n_items"])
    del keyed
    if not np.array_equal(signals["edge_users"] * data["n_items"] + signals["edge_items"], data["pos_key"]):
        raise RuntimeError("M3 엣지/구매쌍 순서 불일치")
    prepared.update(signals=signals, q_value=prepared["q_v"].astype(float),
                    q_activity=prepared["q_n"].astype(float))
    amount = prepared["item_amount_percentile"]
    prepared["item_amount_percentile"] = amount.astype(float)
    gc = base.hm.configure(out_dir=str(Path(cfg.out_dir) / "inputs"))
    graph = base.fixed.build_graph(prepared, gc)
    expected = p3["m3_audit"]
    if set(graph["audit"]) != set(expected) or any(
            not np.isclose(graph["audit"][k], v, rtol=1e-8, atol=1e-10) for k, v in expected.items()):
        raise RuntimeError("M3 고정 그래프 진단 불일치")
    prepared["observed_adj"] = base.v3.build_adj(signals["edge_users"], signals["edge_items"],
        graph["weights"].astype(np.float32), data["n_users"], data["n_items"])
    prepared["item_amount_percentile"] = amount
    protocol = {"code_version": VERSION, "config": asdict(cfg), "seeds": [SEED],
        "source_revision": prepared["revision"], "input_hash": prepared["input_hash"],
        "split": p2["split"], "data_stats": data["data_stats"],
        "feature_sha256": prepared["feature_sha256"],
        "dataset_initialization": prepared["protocol"]["dataset_initialization"],
        "m3_config": {k: getattr(gc, k) for k in GRAPH_KEYS},
        "m3_audit": graph["audit"], "m3_beta": graph["beta"],
        "m4_weight_audit": prepared["observed_weight_audit"], "reference_audit": refs["audit"],
        "new_fit_models": list(MODELS), "labels": LABELS, "fixed_epoch": 300,
        "fixed": p3["fixed"], "single_seed_only": True, "significance_claim": False,
        "unseen_confirmation_claim": False, "clv_attribution_claim": False,
        "initialization": "fresh seed50 initialization per model; one joint optimizer; no frozen/pretrained components",
        "reuse": "M1 seed50 required; M3/M5_C seed50 optional read-only; seed49 used only for design fingerprints"}
    config_hash = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()[:12]
    run_dir = Path(cfg.out_dir) / config_hash
    path = run_dir / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != json.loads(json.dumps(protocol)):
        raise RuntimeError("동일 출력 경로의 protocol 불일치")
    base.io._atomic_json(path, protocol)
    base.io._atomic_json(run_dir / "reference_audit.json", refs["audit"])
    prepared["base_cfg"]["SEED_LIST"] = [SEED]
    prepared.update(cfg=cfg, references=refs, protocol=protocol, config_hash=config_hash, run_dir=run_dir)
    for key in ("signals", "q_value", "q_activity"):
        prepared.pop(key, None)
    print(f"M3 그래프 준비: {time.monotonic()-graph_start:.1f}초 / 전체 준비 {time.monotonic()-started:.1f}초", flush=True)
    report(prepared)
    return prepared


def arm_inputs(prepared, model_id):
    if model_id not in MODELS:
        raise ValueError("M1/M3/M5_C 및 다른 seed는 이 Colab에서 학습하지 않습니다")
    adjacency = prepared["observed_adj"] if model_id in (M5_A, M5_D) else prepared["data"]["adj"]
    weights = prepared["observed_weights"] if model_id in (M4, M5_B, M5_D) else None
    if model_id in (M4, M5_B, M5_D) and weights is None:
        raise RuntimeError("M4 가중치 누락")
    return adjacency, weights


def identity(prepared, model_id):
    arm_inputs(prepared, model_id)
    return base.RunIdentity(VERSION, model_id, SEED, prepared["config_hash"],
                            prepared["revision"], prepared["input_hash"])


def arm_path(prepared, model_id):
    return prepared["run_dir"] / "arms" / f"{model_id}_s{SEED}.json"


def completed(prepared, model_id):
    path = arm_path(prepared, model_id)
    if not path.exists():
        return None
    row = json.loads(path.read_text())
    if (row.get("identity") != asdict(identity(prepared, model_id)) or row.get("epochs") != 300
            or row.get("seed") != SEED or row.get("model_id") != model_id
            or set(row.get("metrics", {})) != set(prepared["references"]["metrics"])
            or not np.isfinite(list(row["metrics"].values())).all()):
        raise RuntimeError("완료 결과 신원/142지표 불일치")
    return row


def run_arm(prepared, model_id):
    cfg = prepared["cfg"]
    validate_config(cfg)
    old = completed(prepared, model_id)
    if old is not None:
        print(f"[재사용] seed50 {LABELS[model_id]} — 재학습·재평가 없음", flush=True)
        return old
    if prepared["data"].get("loss_w") is not None:
        raise RuntimeError("M4 중복 가중 금지")
    if base.m2._feature_hash(prepared["features"]) != prepared["feature_sha256"]:
        raise RuntimeError("M2 특징 변경")
    adj, weights = arm_inputs(prepared, model_id)
    if weights is not None and hashlib.sha256(np.asarray(weights, np.float32).tobytes()).hexdigest() != prepared["protocol"]["m4_weight_audit"]["sha256"]:
        raise RuntimeError("M4 가중치 변경")
    data = prepared["data"]
    base.v3.set_seed(SEED)
    if model_id == M4:
        model = base.hm._build_model({"prepared": prepared, "adjacency": adj}, cfg, SEED)
    else:
        model = base.m2.DatasetAdaptiveOrthogonalNVLightGCN(
            n_users=data["n_users"], n_items=data["n_items"], features=prepared["features"],
            adj=adj, id_dim=cfg.id_dim, n_layers=cfg.n_layers,
            pref_reg=cfg.pref_reg, shared_l2=cfg.shared_l2).to(base.v3.DEVICE)
    store = ProgressStore(prepared["run_dir"] / "progress", identity(prepared, model_id))
    started = time.monotonic()
    history = base.fixed._train(model, prepared, cfg, model_id, SEED, store, row_weights=weights)
    if [r["epoch"] for r in history] != list(range(1, 301)):
        raise RuntimeError("300epoch 미완료: test 평가 금지")
    print(f"{LABELS[model_id]} 학습/저장 완료. 고정300epoch test 평가 중…", flush=True)
    metrics = base.fixed._evaluate(model, prepared)
    if set(metrics) != set(prepared["references"]["metrics"]) or not np.isfinite(list(metrics.values())).all():
        raise RuntimeError("최종142지표 불완전")
    diagnostics = {"graph": "centered_nv" if model_id in (M5_A, M5_D) else "binary",
                   "row_weighted": weights is not None, "m2_intervention": model_id != M4,
                   "feature_sha256": prepared["feature_sha256"] if model_id != M4 else None}
    if model_id != M4:
        diagnostics.update(model.representation_diagnostics())
        diagnostics.update(model.training_gradient_diagnostics())
    row = {"identity": asdict(identity(prepared, model_id)), "model_id": model_id,
           "seed": SEED, "epochs": 300, "metrics": metrics, "training_history": history,
           "diagnostics": diagnostics}
    path = arm_path(prepared, model_id)
    base.io._atomic_json(path, row)
    store.mark_complete(epoch=300, max_epoch=300, selection="none",
                        checkpoint_path=str(store.latest_checkpoint), result_path=str(path))
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"{LABELS[model_id]} 완료 결과 저장 / 이번 실행 구간 {time.monotonic()-started:.1f}초", flush=True)
    return row


def report(prepared):
    refs = prepared["references"]
    # Optional core results can appear while the other GPU is training. Never fit or wait for them.
    core = [r for mid in base.hm.CORE_MODELS
            if (r := _core_row(refs["core_dir"], refs["core_protocol"], mid, refs["metrics"])) is not None]
    m1 = next((r for r in core if r["model_id"] == base.hm.M1), None)
    if m1 is None or m1 != refs["m1"]:
        raise RuntimeError("준비 후 M1 seed50 참조 누락/변경")
    new = [r for mid in MODELS if (r := completed(prepared, mid)) is not None]
    rows = core + new
    absolute = pd.DataFrame([{"model": LABELS[r["model_id"]], "model_id": r["model_id"],
        "seed": SEED, "source": "new_fit" if r["model_id"] in MODELS else "reused_seed50",
        **r["metrics"]} for r in rows])
    absolute = absolute.sort_values("model").reset_index(drop=True)
    comparisons = []
    for candidate in rows:
        for reference in rows:
            if candidate["model_id"] == reference["model_id"]:
                continue
            for metric in refs["metrics"]:
                x, y = float(candidate["metrics"][metric]), float(reference["metrics"][metric])
                comparisons.append({"seed": SEED, "model": LABELS[candidate["model_id"]],
                    "reference": LABELS[reference["model_id"]], "metric": metric,
                    "candidate_value": x, "reference_value": y, "delta": x-y,
                    "relative_change_pct": 100*(x/y-1) if y else None})
    missing = [LABELS[mid] for mid in MODELS if not any(r["model_id"] == mid for r in new)]
    reading = {"complete": not missing, "missing": missing, "seed": SEED, "seed_count": 1,
        "new_fit_count": len(new), "reused_count": len(core), "fixed_epoch": 300,
        "eight_models_complete": len(rows) == 8,
        "missing_other_gpu": [LABELS[mid] for mid in base.hm.CORE_MODELS if not any(r["model_id"] == mid for r in core)],
        "significance_claim": False, "clv_attribution_claim": False, "unseen_confirmation_claim": False,
        "note": "seed50 1시드 기술적 비교; 3/10시드 평균·SD·CI 아님; test 재튜닝 금지"}
    tables = {"absolute": absolute, "comparison": pd.DataFrame(comparisons, columns=[
        "seed", "model", "reference", "metric", "candidate_value", "reference_value", "delta", "relative_change_pct"]),
        "diagnostics": pd.DataFrame([{"model_id": r["model_id"], "seed": SEED, **r.get("diagnostics", {})} for r in rows])}
    paths = {}
    for name, table in tables.items():
        paths[name] = str(prepared["run_dir"] / "reports" / f"{name}.csv")
        base.io._atomic_csv(Path(paths[name]), table)
    paths["json"] = str(prepared["run_dir"] / "reports/result.json")
    base.io._atomic_json(Path(paths["json"]), {"protocol": prepared["protocol"], "reading": reading,
                                             "result_paths": paths})
    return {"complete": not missing, "reading": reading, "paths": paths, **tables}


def run(prepared):
    for index, mid in enumerate(MODELS, 1):
        print(f"[{index}/5] H&M seed50 {LABELS[mid]}", flush=True)
        run_arm(prepared, mid)
        report(prepared)  # Save a partial report after each completed model.
    return report(prepared)


def export_zip(prepared, destination, *, allow_partial=False):
    result = report(prepared)
    if not result["complete"] and not allow_partial:
        raise RuntimeError("5모형 미완료. 완료건만 회수하려면 allow_partial=True를 지정하세요")
    with ZipFile(destination, "w", ZIP_DEFLATED) as archive:
        for path in result["paths"].values():
            archive.write(path, Path(path).name)
        for name in ("protocol.json", "reference_audit.json"):
            archive.write(prepared["run_dir"] / name, name)
        for mid in MODELS:
            path = arm_path(prepared, mid)
            if path.is_file():
                archive.write(path, "arms/" + path.name)
        archive.write(prepared["references"]["core_dir"] / "protocol.json", "references/core_protocol.json")
        for mid in base.hm.CORE_MODELS:
            path = prepared["references"]["core_dir"] / "arms" / f"{mid}_s{SEED}.json"
            if LABELS[mid] in set(result["absolute"].model):
                archive.write(path, "references/" + path.name)
    return str(destination)
