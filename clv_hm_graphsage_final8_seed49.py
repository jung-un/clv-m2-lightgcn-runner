"""Eight fixed GraphSAGE intervention arms, H&M last week, seed49 only.

Uses unchanged M2/M3/M4 input preparation and verifies against frozen LightGCN
reference artifacts. Those reference metrics are NOT GraphSAGE baseline results.
No validation, interim test, tuning, shuffled arm or automatic extra seed.
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
from clv_graphsage_adaptive_model import AdaptiveNVGraphSAGE
import clv_hm_graphsage_training_io as execution
import clv_hm_remaining5_lastweek_seed50 as hm_schema
from clv_hm_ngcf_final8_seed49 import compare_m3_audit

VERSION = "clv-hm-graphsage-final8-seed49-v2"
# Bits: M2 representation, M3 graph, M4 loss. Order is fixed before test.
ARMS = {"M1": (False, False, False), "M2": (True, False, False),
        "M3": (False, True, False), "M4": (False, False, True),
        "M5_A": (True, True, False), "M5_B": (True, False, True),
        "M5_C": (False, True, True), "M5_D": (True, True, True)}
ORDER = tuple(ARMS)
CORE = tuple(f"{m}@{k}" for k in (10, 20, 50) for m in
             ("recall", "ndcg", "price_purchase_amount_weighted_hit", "vndcg"))


@dataclass(frozen=True)
class Config:
    out_dir: str
    m2_reference: str
    m4_reference: str
    dataset: str = "hm"
    seeds: tuple = (49,)
    epochs: int = 300
    id_dim: int = 64
    n_layers: int = 2
    batch_size: int = 131072
    lr: float = .0005
    pref_reg: float = .001
    negative_count: int = 1
    input_days: int = 365
    bandwidth: float = .25
    reliability_floor: float = .05
    shared_l2: float = 1e-6
    backbone_l2: float = 1e-6


def configure(*, out_dir=None, m2_reference=None, m4_reference=None):
    root = base.v3.default_out_dir("hm")
    return Config(out_dir or root + "_graphsage_final8_seed49_v1",
                  m2_reference or root + "_m2_adaptive_lastweek_final10_v1",
                  m4_reference or root + "_m5c_lastweek_final10_v1")


def validate_config(cfg):
    if cfg != configure(out_dir=cfg.out_dir, m2_reference=cfg.m2_reference,
                        m4_reference=cfg.m4_reference):
        raise ValueError("GraphSAGE 8모형/seed49/300epoch 및 사전 고정 설정만 허용합니다")
    for path in (cfg.m2_reference, cfg.m4_reference):
        if (Path(cfg.out_dir).resolve() == Path(path).resolve()
                or Path(cfg.out_dir).resolve() in Path(path).resolve().parents
                or Path(path).resolve() in Path(cfg.out_dir).resolve().parents):
            raise ValueError("새 출력과 참조 결과는 분리해야 합니다")


def verify_input_files(expected_hash):
    """Fail before expensive preprocessing; file paths do not affect identity."""
    manifest = base.fixed.moe.build_input_manifest(base.v3.SCHEMA['hm'])
    actual = base.fixed.moe.manifest_hash(manifest)
    if actual != expected_hash:
        raise RuntimeError(
            f'H&M 원본 파일 불일치: actual={actual}, expected={expected_hash}. '
            '학교 Drive raw/hm 원본을 확인하세요. CSV 재변환/검증 우회 금지.')
    return manifest


def prepare(cfg):
    preparation_started = time.perf_counter()
    validate_config(cfg)
    # This helper does no fitting: it checks frozen reference bytes and the
    # historical CLV/M2 feature/M3/M4 computations against the existing design.
    reference_cfg = base.configure(dataset="hm", seed=49, out_dir=cfg.out_dir,
        m2_reference=cfg.m2_reference, m4_reference=cfg.m4_reference)
    references = base.load_references(reference_cfg)
    p2, p3 = (references["sources"][k]["protocol"] for k in ("m2", "m4"))
    for key in hm_schema.GRAPH_KEYS:
        if p3["config"][key] != getattr(base.hm.configure(), key):
            raise RuntimeError("H&M M3 고정 설정 불일치: " + key)
    verify_input_files(p2["input_hash"])
    prepared = base._prepare_hm_inputs(cfg)
    data = prepared['data']
    base.hm.validate_split(data)
    if set(data['splits']) != {'test'} or data.get('loss_w') is not None:
        raise RuntimeError('test 단일 분할/기본 무가중 BPR 필요')
    if prepared['input_hash'] != p2['input_hash'] or data['data_stats'] != p2['data_stats']:
        raise RuntimeError('실제 입력/마지막주 분할 불일치')
    if prepared['feature_sha256'] != p2['feature_sha256'] or not np.array_equal(prepared['features']['keys'], data['pos_key']):
        raise RuntimeError('M2 특징/구매쌍 순서 불일치')
    if prepared['observed_weight_audit']['sha256'] != p3['m4_c_weight_audit']['sha256']:
        raise RuntimeError('M4 고정 가중치 불일치')
    inputs_seconds = time.perf_counter() - preparation_started
    print(f'[준비] 입력·CLV·M2·M4 {inputs_seconds:.1f}초', flush=True)
    keyed = base.hm.hm_graph._purchase_keyed(data['train'])
    signals = base.fixed.graph.centered_edge_signals(keyed, data['n_users'], data['n_items'])
    del keyed
    if not np.array_equal(signals['edge_users'] * data['n_items'] + signals['edge_items'], data['pos_key']):
        raise RuntimeError('M3 엣지/구매쌍 순서 불일치')
    prepared.update(signals=signals, q_value=prepared['q_v'].astype(float), q_activity=prepared['q_n'].astype(float))
    amount = prepared['item_amount_percentile']
    prepared['item_amount_percentile'] = amount.astype(float)
    gc = base.hm.configure(out_dir=str(Path(cfg.out_dir) / 'inputs'))
    graph = base.fixed.build_graph(prepared, gc)
    expected = p3['m3_audit']
    graph_comparison = compare_m3_audit(expected, graph['audit'], gc)
    prepared["item_amount_percentile"] = amount
    prepared.update(observed_graph=graph, references=references)
    prepared["base_cfg"]["SEED_LIST"] = [49]
    old = {"split": p2["split"], "m3_audit": graph["audit"],
           "m4_weight_audit": prepared["observed_weight_audit"], "fixed": p3["fixed"]}
    prepared.update(prepare_graphsage_graphs(prepared))
    frozen = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in (Path(__file__).name, "clv_graphsage_adaptive_model.py",
                     "clv_dataset_adaptive_orthogonal_nv_model.py",
                     "clv_reliability_orthogonal_nv_model.py", "clv_run_state.py",
                     "clv_ngcf_training_io.py", "clv_hm_graphsage_training_io.py",
                     "clv_dh_graphsage_final8_seed49.py")}
    protocol = {"code_version": VERSION, "config": asdict(cfg),
        "source_revision": prepared["revision"], "code_sha256": frozen,
        "input_hash": prepared["input_hash"], "data_stats": data["data_stats"],
        "split": old["split"], "feature_sha256": prepared["feature_sha256"],
        "m3_audit": old["m3_audit"], "m4_weight_audit": old["m4_weight_audit"],
        "m3_reference_comparison": graph_comparison,
        "reference_audit": prepared["references"]["audit"],
        "dataset_initialization": base.m2.dataset_parameter_initialization(prepared["features"]),
        "fixed_intervention_settings": old["fixed"],
        "models": {m: {"M2": b[0], "M3": b[1], "M4": b[2]} for m, b in ARMS.items()},
        "backbone": "GraphSAGE mean: ID64, two64-D layers; mean layers0/1/2 output64",
        "propagation": "P=D^-1 A_w; Hnext=L2normalize(ELU(concat(H,P H) W+b)); Z=mean(E,H1,H2)",
        "adjacency": "same binary support; raw M3 weights or ones; weighted row normalization; no self-loop edges; separate self concatenation; full neighbors without sampling",
        "normalization_precision": "raw float32 weights; float64 degree sums and validation; normalized adjacency cast to float32 for unchanged training precision",
        "regularization": "project sampled raw-ID L2=.001; M2 shared L2=1e-6; GraphSAGE weights/bias L2=1e-6",
        "implementation_scope": "project GraphSAGE mean preset: trainable IDs, BPR, ELU/L2 each layer, dropout0, layers0/1/2 mean; not original benchmark reproduction or unseen-node inductive generalization",
        "paper": "https://papers.nips.cc/paper/6703-inductive-representation-learning-on-large-graphs.pdf",
        "m2_adaptation": "same fixed N/V inputs, positive leave-one-user-out, adaptive orthogonal mix; encoder output remains64",
        "initialization": "normal(.1) ID; Xavier GraphSAGE matrices, zero biases; identical backbone initialization; no pretrained checkpoints",
        "new_fit_count": 8, "reused_graphsage_results": 0,
        "lightgcn_references": "input audit only; NOT GraphSAGE performance baselines",
        "fixed_epoch": 300, "test_evaluations_per_completed_fit": 1,
        "execution_optimization": "same uniform K1 draws/retries/RNG; recheck rejected negatives only; prevalidated CPU positive-pair positions; unchanged floating arithmetic; heartbeat300sec; synchronous atomic Drive checkpoint every epoch",
        "checkpoint_storage": "full model/Adam/RNG each active epoch; after epoch300 result and completion persist, atomically replace with exact final weights/RNG/history, omit Adam; completed arms skipped",
        "timing": "first5 batches of first3 session epochs sampled with CUDA synchronization; every epoch training/save split; no test used for profiling",
        "validation": False, "holdout": False, "early_stopping": False,
        "single_seed_only": True, "significance_claim": False,
        "clv_attribution_claim": False, "unseen_confirmation_claim": False,
        "m3_adaptation": "same fixed centered N/V raw weights; weighted neighbor mean divides by row weight sum. Common row factors cancel; raw two-sided mass audit does not imply preservation after GraphSAGE row normalization. Self path identical across arms",
        "decision": "descriptive all142 metrics; no best-model/epoch/seed selection or test retuning"}
    config_hash = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()[:12]
    run_dir = Path(cfg.out_dir) / config_hash
    path = run_dir / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != json.loads(json.dumps(protocol)):
        raise RuntimeError("기존 protocol 불일치: 덮어쓰지 않습니다")
    prepared.update(cfg=cfg, run_dir=run_dir, config_hash=config_hash, protocol=protocol)
    base.io._atomic_json(path, protocol)
    base.io._atomic_json(run_dir / "reference_audit.json", prepared["references"]["audit"])
    preparation_timing = {"input_preparation_seconds": time.perf_counter() - preparation_started,
                          "includes": "reference validation, data/CLV/features/graphs/cache preparation; no training"}
    base.io._atomic_json(run_dir / "preparation_timing.json", preparation_timing)
    print(f"[준비 계측] 자료·CLV·그래프·참조 검증 {preparation_timing['input_preparation_seconds']:.1f}초", flush=True)
    print("GraphSAGE 입력 검증 완료: 8모형 순차 새 학습 / seed49 / 마지막 주 / 300epoch", flush=True)
    for key in ("signals", "q_value", "q_activity", "observed_graph"):
        prepared.pop(key, None)
    report(prepared)
    return prepared


from clv_dh_graphsage_final8_seed49 import prepare_graphsage_graphs


def identity(prepared, label):
    if label not in ARMS:
        raise ValueError("허용하지 않은 모형")
    return base.RunIdentity(VERSION, f"graphsage_{label.lower()}_lastweek", 49,
                            prepared["config_hash"], prepared["revision"], prepared["input_hash"])


def arm_path(prepared, label):
    return prepared["run_dir"] / "arms" / f"{identity(prepared, label).model_id}_s49.json"


def completed(prepared, label):
    path = arm_path(prepared, label)
    if not path.exists():
        return None
    row = json.loads(path.read_text())
    if (row.get("identity") != asdict(identity(prepared, label)) or row.get("epochs") != 300
            or row.get("model") != label or row.get("seed") != 49
            or set(row.get("metrics", {})) != set(prepared["references"]["metrics"])
            or not np.isfinite(list(row["metrics"].values())).all()):
        raise RuntimeError(f"{label} 완료 결과 신원/300epoch/전체지표 불일치")
    return row


class SafeProgressStore(base.ProgressStore):
    """Do not silently delete corrupt checkpoints or restart costly training."""

    def __init__(self, root, identity, *, heartbeat_interval_sec=300):
        super().__init__(root, identity, heartbeat_interval_sec=heartbeat_interval_sec)

    def restore_epoch(self, model, optimizer, rng):
        if not self.latest_checkpoint.exists():
            if self.stage_path.exists():
                prior = self.read_progress()
                if int(prior.get("epoch", 0)) > 1 or prior.get("checkpoint_sha256"):
                    raise RuntimeError("진행 기록은 있으나 checkpoint가 없습니다. Drive 연결/파일을 확인하세요")
            return None
        try:
            payload = torch.load(self.latest_checkpoint, map_location="cpu", weights_only=False)
        except Exception as exc:
            raise RuntimeError("checkpoint를 읽지 못했습니다. 파일은 보존하며 자동 처음부터 학습하지 않습니다. Drive 연결을 확인하세요") from exc
        self._validate_identity(payload.get("identity", {}))
        epoch = payload.get("epoch")
        if not isinstance(epoch, int) or not 1 <= epoch <= 300 or [r.get("epoch") for r in
                payload.get("history", [])] != list(range(1, epoch + 1)):
            raise RuntimeError("checkpoint epoch/이력 불일치")
        if payload.get("completed_weights_only"):
            raise RuntimeError("완료 모형의 최종 가중치입니다. 완료 결과를 확인하고 재학습하지 마세요")
        model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        rng.bit_generator.state = payload["numpy_rng_state"]
        torch.set_rng_state(payload["torch_rng_state"])
        if torch.cuda.is_available() and payload.get("cuda_rng_state"):
            torch.cuda.set_rng_state_all(payload["cuda_rng_state"])
        return {"next_epoch": epoch + 1, "history": payload["history"]}

    def save_epoch(self, model, optimizer, rng, **state):
        return execution.save_epoch_local_then_drive(self, model, optimizer, rng, state)


def build_model(prepared, label):
    m2, m3, m4 = ARMS[label]
    cfg, data = prepared["cfg"], prepared["data"]
    adj = prepared["graphsage_m3_adj"] if m3 else prepared["graphsage_binary_adj"]
    model = AdaptiveNVGraphSAGE(m2_active=m2, n_users=data["n_users"], n_items=data["n_items"],
        features=prepared["features"], adj=adj, id_dim=cfg.id_dim, n_layers=cfg.n_layers,
        pref_reg=cfg.pref_reg, shared_l2=cfg.shared_l2,
        backbone_l2=cfg.backbone_l2).to(base.v3.DEVICE)
    return model, prepared["observed_weights"] if m4 else None


def sample_uniform_negative_matrix(users, positives, n_items, positive_keys, rng, *, k):
    return execution.sample_uniform_negative_matrix(
        users, positives, n_items, positive_keys, rng, k=k)


def run_arm(prepared, label):
    cfg = prepared["cfg"]
    validate_config(cfg)
    old = completed(prepared, label)
    if old is not None:
        execution.compact_completed_checkpoint(SafeProgressStore(
            prepared["run_dir"] / "progress", identity(prepared, label)))
        print(f"[재사용] GraphSAGE {label} seed49 — 재학습·재평가 없음", flush=True)
        return old
    if base.m2._feature_hash(prepared["features"]) != prepared["feature_sha256"]:
        raise RuntimeError("준비 후 M2 특징 변경")
    ident = identity(prepared, label)
    store = SafeProgressStore(prepared["run_dir"] / "progress", ident)
    base.v3.set_seed(49)
    model, weights = build_model(prepared, label)
    # All arms start from the same RNG state; no pretrained components.
    base.v3.set_seed(49)
    started = time.monotonic()
    try:
        history = execution.train(model, prepared, cfg, ident.model_id, 49, store,
                                  row_weights=weights, negative_sampler=sample_uniform_negative_matrix)
        if [r["epoch"] for r in history] != list(range(1, 301)):
            raise RuntimeError("300epoch 미완료: test 평가 금지")
        print(f"[평가] GraphSAGE {label} — 고정300epoch 마지막점", flush=True)
        metrics = base.fixed._evaluate(model, prepared)
        if set(metrics) != set(prepared["references"]["metrics"]) or not np.isfinite(list(metrics.values())).all():
            raise RuntimeError("전체142지표 누락/비유한 값")
        row = {"identity": asdict(ident), "model": label, "model_id": ident.model_id,
            "seed": 49, "epochs": 300, "metrics": metrics, "training_history": history,
            "this_session_seconds": time.monotonic() - started,
            "diagnostics": model.representation_diagnostics() | model.training_gradient_diagnostics() |
                {"graph": "centered_nv" if ARMS[label][1] else "binary",
                 "row_weighted": ARMS[label][2], "feature_sha256": prepared["feature_sha256"]}}
        path = arm_path(prepared, label)
        base.io._atomic_json(path, row)
        store.mark_complete(epoch=300, max_epoch=300, selection="none", result_path=str(path))
        execution.compact_completed_checkpoint(store)
        return row
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def report(prepared):
    """Read saved completed-arm results only; never trains or evaluates."""
    rows = [r for label in ORDER if (r := completed(prepared, label)) is not None]
    metrics = prepared["references"]["metrics"]
    absolute = pd.DataFrame([{"model": r["model"], "model_id": r["model_id"], "seed": 49,
                              **r["metrics"]} for r in rows], columns=["model", "model_id", "seed", *metrics])
    diagnostics = pd.DataFrame([{"model": r["model"], "seed": 49, **r["diagnostics"]} for r in rows])
    comparisons = []
    for candidate in rows:
        for reference in rows:
            if candidate["model"] == reference["model"]:
                continue
            for metric in metrics:
                x, y = candidate["metrics"][metric], reference["metrics"][metric]
                comparisons.append({"model": candidate["model"], "reference": reference["model"],
                    "seed": 49, "metric": metric, "candidate_value": x, "reference_value": y,
                    "delta": x-y, "relative_change_pct": 100*(x/y-1) if y != 0 else np.nan})
    comparison = pd.DataFrame(comparisons, columns=["model", "reference", "seed", "metric",
        "candidate_value", "reference_value", "delta", "relative_change_pct"])
    missing = [m for m in ORDER if m not in set(absolute.model)]
    reading = {"complete": not missing, "completed": len(rows), "planned": 8, "missing": missing,
        "seed": 49, "backbone": "GraphSAGE", "fixed_epoch": 300, "single_seed_only": True,
        "std_ci_available": False, "significance_claim": False, "clv_attribution_claim": False,
        "unseen_confirmation_claim": False, "retuning_allowed": False}
    paths = {}
    for name, frame in {"absolute": absolute, "comparison": comparison, "diagnostics": diagnostics}.items():
        paths[name] = str(prepared["run_dir"] / "reports" / f"{name}.csv")
        base.io._atomic_csv(Path(paths[name]), frame)
    paths["json"] = str(prepared["run_dir"] / "reports" / "result.json")
    base.io._atomic_json(Path(paths["json"]), {"protocol": prepared["protocol"], "reading": reading,
                                             "result_paths": paths})
    return {"complete": not missing, "reading": reading, "paths": paths,
            "absolute": absolute, "comparison": comparison, "diagnostics": diagnostics}


def run(prepared):
    for i, label in enumerate(ORDER, 1):
        print(f"[{i}/8] H&M GraphSAGE {label} seed49", flush=True)
        run_arm(prepared, label)
        result = report(prepared)
        print(f"[보고서 저장] {result['reading']['completed']}/8 완료", flush=True)
    return result


def export_zip(prepared, destination):
    result = report(prepared)
    path = Path(destination)
    if not result["complete"] and not path.stem.endswith("_partial"):
        path = path.with_name(path.stem + "_partial" + path.suffix)
    with ZipFile(path, "w", ZIP_DEFLATED) as archive:
        for file in result["paths"].values():
            archive.write(file, Path(file).name)
        for name in ("protocol.json", "reference_audit.json"):
            archive.write(prepared["run_dir"] / name, name)
        prep_timing = prepared["run_dir"] / "preparation_timing.json"
        if prep_timing.exists():
            archive.write(prep_timing, prep_timing.name)
        for label in ORDER:
            file = arm_path(prepared, label)
            if file.exists():
                archive.write(file, "arms/" + file.name)
        for file in sorted((prepared["run_dir"] / "progress" / "timing").glob("*.csv")):
            archive.write(file, "timing/" + file.name)
    return str(path)
