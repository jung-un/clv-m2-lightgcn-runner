"""Fixed M3/M4-C, then add the unchanged shared-attribute M2 at seed48.

Development only. Existing C seed44 is diagnosed before two fresh seed48
fits. All learned parameters start together; fixed train-derived graph
weights are not a pretrained frozen encoder. No final-week data is opened.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import gc
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import torch

import clv_m1_m5_error_diagnostic as features
import clv_m5_nv_rank_diagnostic as ranks
import clv_m5_shared_attributes_seed48_screen as shared
import lightgcn_clv_m3_centered_value_graph as m3
from clv_m5_shared_attributes_model import JointAttributeLightGCN
from clv_run_state import RunIdentity, file_sha256

io, capacity, v3, m4 = shared.io, shared.capacity, shared.v3, shared.m4
CODE_VERSION = "clv-m5-shared-attributes-fixed-m3-m4c-seed48-dev-v1"
REFERENCE = m4.MODEL_ID
MODEL_ID = "m5_shared_attributes_nv_fixed_centered_graph_user_centered_bpr_k1"
ACCURACY, ECONOMIC = m4.ACCURACY, m4.ECONOMIC


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
    target_cv: float = .2
    out_dir: str = shared.baseline.ROOT + "_m5_shared_attributes_fixed_m3_m4c_seed48_dev_v1"
    baseline_json: str = shared.Config().baseline_json
    c_reference_dir: str = shared.baseline.ROOT + "_clv_m5_m3_m4_user_centered_s44_v1"


def configure(**paths):
    if set(paths) - {"out_dir", "baseline_json", "c_reference_dir"}:
        raise ValueError("seed48·300epoch·표현·M3·M4 강도는 고정입니다. 경로만 지정하세요")
    return validate_config(Config(**paths))


def validate_config(cfg):
    for key, value in asdict(Config()).items():
        if key not in ("out_dir", "baseline_json", "c_reference_dir") and getattr(cfg, key) != value:
            raise ValueError(f"사전 고정 설정 불일치: {key}")
    if not all(getattr(cfg, k) for k in ("out_dir", "baseline_json", "c_reference_dir")):
        raise ValueError("입력·결과 경로 누락")
    return cfg


class FixedGraphAttributeLightGCN(JointAttributeLightGCN):
    """Reuse M2 verbatim; replace only the differentiable graph path."""
    def __init__(self, *, adjacency, graph_weights, **kwargs):
        super().__init__(**kwargs)
        del self.graph_net
        self.register_buffer("fixed_adj", adjacency.detach().coalesce())
        self.register_buffer("fixed_weights", torch.as_tensor(graph_weights, dtype=torch.float32))

    def graph_weights(self):
        return self.fixed_weights

    def weighted_adjacency(self):
        return self.fixed_adj

    def id_vectors(self):
        current = torch.cat([self.E_u.weight, self.E_i.weight])
        layers = [current]
        for _ in range(self.n_layers):
            current = torch.sparse.mm(self.fixed_adj, current)
            layers.append(current)
        return torch.stack(layers).mean(0).split([self.n_users, self.n_items])

    @torch.no_grad()
    def training_gradient_diagnostics(self):
        blocks = {"id_user": self.E_u, "id_item": self.E_i,
                  "shared_type": self.type_encoder, "shared_price": self.price_encoder,
                  "nv_feature": self.feature_net}
        return {name + "_gradient_norm": float(sum(
            p.grad.square().sum() for p in module.parameters() if p.grad is not None).sqrt())
            if any(p.grad is not None for p in module.parameters()) else 0.
            for name, module in blocks.items()}

    @torch.no_grad()
    def representation_diagnostics(self):
        attrs = self.attributes()
        profiles, _ = self.history_vectors(attrs)
        count = min(1024, len(self.edge_users))
        users = self.edge_users[:count]
        inputs = torch.cat([attrs[self.edge_items[:count]], self.context[users],
                            self.relations[:count, 2:3]], 1)
        changed = inputs.clone()
        changed[:, 16:18] = 1 - changed[:, 16:18]
        return {"fixed_graph": True, "graph_parameters": 0,
                "attribute_norm_mean": float(attrs.norm(dim=1).mean()),
                "history_norm_mean": float(profiles.norm(dim=1).mean()),
                "nv_feature_local_sensitivity": float((self.feature_net(inputs)
                    - self.feature_net(changed)).abs().mean()),
                "not_clv_attribution": True, **self.training_gradient_diagnostics()}


def _c_source(cfg):
    root = Path(cfg.c_reference_dir)
    matches = sorted(root.glob(f"{m4.CODE_VERSION}_*.json"))
    if len(matches) != 1:
        raise FileNotFoundError(f"기존 C형 seed44 완료 JSON 한 개가 필요합니다: {root}")
    path = matches[0]
    report = json.loads(path.read_text())
    config_hash = path.stem[len(m4.CODE_VERSION) + 1:]
    checkpoint = root / "progress" / config_hash / "resume" / (
        f"m5_m3_m4_user_centered_dev_{REFERENCE}_s44_latest.pt")
    absolute = Path(report["result_paths"]["absolute_csv"])
    if not checkpoint.is_file() or not absolute.is_file():
        raise FileNotFoundError(f"C형 checkpoint·absolute가 필요합니다. 새 학습 없음: {root}")
    fixed = ("epochs", "eval_every", "batch_size", "lr", "n_layers", "id_dim",
             "pref_reg", "negative_count", "target_cv")
    if (report.get("seed") != 44 or report.get("split") != shared.SPLIT
            or report.get("final_test") is not False or report.get("holdout") is not False
            or any(report["config"][k] != getattr(cfg, k) for k in fixed)):
        raise RuntimeError("기존 C형의 분할·설정이 이번 개발진단과 다릅니다")
    return {"root": root, "path": path, "report": report, "config_hash": config_hash,
            "checkpoint": checkpoint, "absolute": absolute,
            "report_sha256": file_sha256(path), "absolute_sha256": file_sha256(absolute)}


def prepare(cfg=None):
    cfg = validate_config(cfg or configure())
    source = _c_source(cfg)  # fail before costly preparation if old C is absent
    _, prepared = shared.prepare(shared.configure(out_dir=cfg.out_dir, baseline_json=cfg.baseline_json))
    data = prepared["data"]
    graph_signals = m3.centered_edge_signals(data["train"], data["n_users"], data["n_items"])
    if not np.array_equal(graph_signals["edge_users"] * data["n_items"] + graph_signals["edge_items"], data["pos_key"]):
        raise RuntimeError("고정 M3와 M2의 학습엣지가 다릅니다")
    valid = np.asarray(prepared["clv_valid"], bool)
    graph_prepared = dict(prepared, signals=graph_signals,
        q_value=np.where(valid, prepared["q_v"], 0), q_activity=np.where(valid, prepared["q_n"], 0))
    graph_cfg = m3.configure_centered_graph(seeds=(48,), out_dir=cfg.out_dir)
    spec = next(s for s in m3.arm_specifications() if s["model_id"] == m3.ARM_VALUE_ACTIVITY)
    graph = m3.build_arm_graph(graph_prepared, graph_cfg, spec)
    graph["weight_sha256"] = hashlib.sha256(graph["weights"].astype(np.float32).tobytes()).hexdigest()
    old = source["report"]
    if (old["input_hash"] != prepared["input_hash"]
            or not np.isclose(old["m3_beta"], graph["beta"], rtol=0, atol=1e-12)
            or old["m4_c_weight_audit"]["sha256"] != prepared["preflight"]["m4_audit"]["sha256"]):
        raise RuntimeError("기존 C형과 이번 데이터·M3·M4 신원 불일치")
    preflight = dict(prepared["preflight"])
    for key in ("new_model", "propagation_backward"):
        preflight.pop(key, None)
    preflight.update(code_version=CODE_VERSION, config=asdict(cfg), new_fit_count=2,
        new_models=[REFERENCE, MODEL_ID], primary_reference=REFERENCE,
        m3="existing fixed centered q_N/q_V graph; target CV=.20; no learned graph parameters",
        m3_audit=graph["audit"], m3_beta=graph["beta"], m3_weight_sha256=graph["weight_sha256"],
        c_diagnostic_seed=44, c_report_sha256=source["report_sha256"],
        screen_rule="at300 vs matched M3+M4-C: six accuracy>=.99, both economic@10>=.99, recall@20 and @50 strictly higher",
        attribute_caveat="added representation effect conditional on M3/M4; not isolated CLV attribution")
    fingerprint = {"preflight": preflight, "source": prepared["revision"],
                   "input": prepared["input_hash"], "m1_sha": prepared["baseline_provenance"]["sha256"]}
    prepared.update(preflight=preflight, graph=graph, graph_prepared=graph_prepared,
        graph_cfg=graph_cfg, c_source=source,
        config_hash=hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()[:12])
    io._atomic_json(Path(cfg.out_dir) / "preflight.json", preflight)
    return cfg, prepared


def build_model(cfg, prepared, model_id):
    if model_id == REFERENCE:
        return m3._build_model(prepared["graph_prepared"], prepared["graph_cfg"], prepared["graph"], 48)
    if model_id != MODEL_ID:
        raise ValueError("허용되지 않은 실험군")
    v3.set_seed(48)
    data, signals = prepared["data"], prepared["signals"]
    adjacency = v3.build_adj(signals["edge_users"], signals["edge_items"],
        prepared["graph"]["weights"].astype(np.float32), data["n_users"], data["n_items"])
    return FixedGraphAttributeLightGCN(adjacency=adjacency, graph_weights=prepared["graph"]["weights"],
        n_users=data["n_users"], n_items=data["n_items"], signals=signals,
        q_n=prepared["q_n"], q_v=prepared["q_v"], valid=prepared["clv_valid"],
        id_dim=cfg.id_dim, n_layers=cfg.n_layers, pref_reg=cfg.pref_reg,
        eta=cfg.eta, epsilon=cfg.epsilon).to(v3.DEVICE)


def _release():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _population(prepared, seed):
    cache, data = prepared["cache"], prepared["data"]
    degree = np.diff(data["csr_ptr"])
    population = pd.DataFrame(dict(seed=seed, user=np.asarray(cache.users, np.int64), segment=cache.seg,
        truth_count=[len(cache.gt[u]) for u in cache.users], degree=degree[cache.users],
        q_n=np.asarray(prepared["q_n"])[cache.users], q_v=np.asarray(prepared["q_v"])[cache.users],
        clv_valid=np.asarray(prepared["clv_valid"], bool)[cache.users]))
    for axis, column in (("q_n", "n_bin"), ("q_v", "v_bin")):
        population[column], _ = ranks.train_bins(prepared[axis], prepared["clv_valid"], cache.users)
    population["degree_bin"], _ = ranks.train_bins(degree, degree > 0, cache.users)
    return population


def _readback(recorded, measured, label):
    if set(recorded) != set(measured):
        raise RuntimeError("checkpoint의 전체 지표 키가 원본과 다릅니다")
    rows = []
    for key in recorded:
        # Alignment has a known CPU/CUDA tiny floating-point difference.
        atol = 1e-5 if key == "user_value_tendency_recommended_price_alignment" else 1e-7
        rows.append(dict(model_id=label, metric=key, recorded=float(recorded[key]),
            readback=float(measured[key]), atol=atol, rtol=1e-5,
            passed=bool(np.isclose(recorded[key], measured[key], atol=atol, rtol=1e-5))))
    return rows


def _save_pair(prepared, reference, candidate, recommendations, seed, folder, labels):
    population = _population(prepared, seed)
    truth, users = ranks.movement_tables(reference, candidate, population)
    summary, correlations, transitions = ranks.summaries(truth, users)
    tables = dict(truth_movements=truth, user_movements=users, summary=summary,
                  axis_correlations=correlations, rank_transitions=transitions)
    tables.update(features.enrich_tables(truth, users, pd.DataFrame(recommendations), prepared))
    support = tables["truth_features"].copy()
    support["buyer_support"] = pd.cut(support.item_buyers, [0, 1, 5, 20, 100, np.inf],
        labels=["1", "2~5", "6~20", "21~100", ">100"])
    support["price_bin"] = pd.cut(support.item_amount_percentile, [0, .2, .4, .6, .8, 1], include_lowest=True)
    parts = []
    for k in (10, 20, 50):
        part = support.assign(k=k, status=support[f"status@{k}"])
        part = pd.concat([part, part.assign(segment="all")], ignore_index=True)
        for axis in ("buyer_support", "price_bin"):
            keys = ["segment", axis, "status", "k"]
            counts = part.groupby(keys, observed=True, dropna=False).agg(truth_pairs=("item", "size"), users=("user", "nunique")).reset_index()
            parts.append(counts.rename(columns={axis: "group"}).assign(axis=axis))
    tables["support_price_movements"] = pd.concat(parts, ignore_index=True)
    root = prepared["out_dir"] / folder
    paths = {k: str(root / f"{k}.csv") for k in tables}
    for key, table in tables.items():
        io._atomic_csv(Path(paths[key]), table)
    io._atomic_json(root / "labels.json", {"seed": seed, "reference": labels[0], "candidate": labels[1],
        "legacy_rank_m3_means_reference": True, "legacy_rank_m5_means_candidate": True,
        "rank_nan_means": "outside top100, not absent truth", "feature_source": "train only"})
    return paths


def diagnose_existing_c(cfg, prepared):
    """Read seed44 M1/C before any fresh fit; verify all original metrics."""
    shared.validate_prepared(prepared)
    source = prepared["c_source"]
    if (file_sha256(source["path"]) != source["report_sha256"]
            or file_sha256(source["absolute"]) != source["absolute_sha256"]):
        raise RuntimeError("prepare 이후 C형 원본 변경")
    old = source["report"]
    absolute = pd.read_csv(source["absolute"])
    io._atomic_json(prepared["out_dir"] / "existing_c_diagnostic" / "source_report.json", old)
    io._atomic_csv(prepared["out_dir"] / "existing_c_diagnostic" / "source_absolute.csv", absolute)
    m1_location, m1_meta = features.baseline_source(Path(cfg.c_reference_dir).parent, 44, cfg, prepared)
    entries = [("m1", m3.M1_MODEL_ID, m1_location[2], m1_location[3], m1_location[4], m1_location[5]),
               ("m5", REFERENCE, source["root"], "m5_m3_m4_user_centered_dev", source["config_hash"], old["source_revision"])]
    truth, recommendations, audits, identities = {}, [], [], []
    for label, model_id, root, stage, config_hash, revision in entries:
        state, provenance = ranks._checkpoint(root, stage,
            "baseline_" + model_id if label == "m1" else model_id,
            44, prepared["input_hash"], config_hash, revision)
        graph = prepared["graph"] if label == "m5" else {"adjacency": prepared["data"]["adj"], "beta": 0}
        model = m3._build_model(prepared["graph_prepared"], prepared["graph_cfg"], graph, 44)
        model.load_state_dict(state, strict=True)
        expected = absolute[absolute.model_id.eq(model_id) & absolute.seed.eq(44) & absolute.epoch.eq(300)]
        metrics = prepared["baseline"]["curve"][-1]["metrics"]
        if len(expected) != 1:
            raise RuntimeError("기존 C형 absolute에 M1/C@300 단일 행이 없습니다")
        audits.extend(_readback(expected.iloc[0][list(metrics)].to_dict(), capacity._evaluate(model, prepared), model_id))
        io._atomic_csv(prepared["out_dir"] / "existing_c_diagnostic" / "readback.csv", pd.DataFrame(audits))
        if not all(r["passed"] for r in audits):
            raise RuntimeError("C형 checkpoint 재현 차이: readback.csv 확인. 새 학습 없음")
        recs = []
        truth[label] = ranks._score_truth(model, prepared, 44, recs)
        recommendations.extend(dict(r, model=label) for r in recs)
        identities.append(provenance)
        del model, state
        _release()
    paths = _save_pair(prepared, truth["m1"], truth["m5"], recommendations, 44,
                       "existing_c_diagnostic", (m3.M1_MODEL_ID, REFERENCE))
    diagnostic = {"source_report_sha256": source["report_sha256"], "m1_source": m1_meta,
        "checkpoints": identities, "all_metric_readback_passed": True, "paths": paths,
        "new_training": 0, "causal_or_clv_attribution_claim": False}
    io._atomic_json(prepared["out_dir"] / "existing_c_diagnostic" / "result.json", diagnostic)
    prepared["existing_c_diagnostic"] = diagnostic
    print("[기존 C형 진단 완료] seed44 M1/C 전체지표 재현·정답순위·세그먼트·상품지지도·가격대", flush=True)
    return diagnostic


def _runtime_root(prepared):
    base = Path("/content") if Path("/content").is_dir() else Path(tempfile.gettempdir())
    return base / ".clv_m5_fixed_shared_runtime" / prepared["config_hash"]


def run_arm(cfg, prepared, model_id):
    identity = RunIdentity(CODE_VERSION, model_id, 48, prepared["config_hash"],
                           prepared["revision"], prepared["input_hash"])
    path = prepared["out_dir"] / "arms" / prepared["config_hash"] / f"{model_id}_s48.json"
    model = build_model(cfg, prepared, model_id)
    store = shared.LocalFirstProgressStore(_runtime_root(prepared),
        prepared["out_dir"] / "progress" / prepared["config_hash"], identity, max_epoch=300)
    if path.is_file():
        payload = json.loads(path.read_text())
        if payload.get("identity") != asdict(identity) or payload["curve"][-1]["epoch"] != 300:
            raise RuntimeError("완료 결과의 identity/epoch 불일치")
        checkpoint = store._load_exact(store.latest_checkpoint)
        if checkpoint.get("identity") != asdict(identity) or checkpoint["epoch"] != 300:
            raise RuntimeError("완료 결과에 대응하는300epoch checkpoint가 없습니다")
        model.load_state_dict(checkpoint["model_state"], strict=True)
    else:
        print(f"[학습] seed48 {model_id} 고정300epoch", flush=True)
        curve = capacity._train_curve(model, prepared, cfg,
            {"model_id": model_id, "condition": "fixed_m3_m4c_shared_m2"}, 48,
            store, row_weights=prepared["row_weights"])
        for record in curve:
            if "score_split" in record:
                record["attribute_score_split"] = {
                    k.replace("clv", "attribute_branch"): v for k, v in record.pop("score_split").items()}
        payload = {"identity": asdict(identity), "model_id": model_id, "seed": 48,
                   "curve": curve, "m3_weight_sha256": prepared["graph"]["weight_sha256"],
                   "m4_weight_sha256": prepared["preflight"]["m4_audit"]["sha256"],
                   "final_diagnostics": model.representation_diagnostics() if model_id == MODEL_ID else {"m2_absent": True}}
        io._atomic_json(path, payload)
        store.mark_complete(epoch=300, max_epoch=300, selection="none", result_path=str(path))
    audit = _readback(payload["curve"][-1]["metrics"], capacity._evaluate(model, prepared), model_id)
    io._atomic_csv(prepared["out_dir"] / "reports" / f"{model_id}_readback.csv", pd.DataFrame(audit))
    if not all(r["passed"] for r in audit):
        raise RuntimeError("새 checkpoint와 기록 지표에 차이가 있습니다")
    recs = []
    truth = ranks._score_truth(model, prepared, 48, recs)
    del model
    _release()
    return payload, truth, recs


def report(cfg, prepared, arms):
    old = prepared["baseline"]
    metrics = list(old["curve"][-1]["metrics"])
    rows, diagnostics = [], []
    for arm in [old, *arms]:
        for record in arm["curve"]:
            if "metrics" not in record:
                continue
            if set(record["metrics"]) != set(metrics):
                raise RuntimeError("모형별 전체 metric key 불일치")
            meta = dict(model_id=arm["model_id"], seed=48, epoch=record["epoch"])
            rows.append(meta | record["metrics"])
            diagnostics.append(meta | {"loss": record["loss"], "p_correct": record["p_correct"]}
                | record.get("gradient_diagnostics", {}) | record.get("attribute_score_split", {}))
    absolute = pd.DataFrame(rows)
    expected = {(m, ep) for m in (shared.baseline.M1, REFERENCE, MODEL_ID) for ep in capacity.evaluation_epochs(cfg)}
    if (absolute.duplicated(["model_id", "epoch"]).any()
            or set(zip(absolute.model_id, absolute.epoch)) != expected
            or not np.isfinite(absolute[metrics].to_numpy(float)).all()):
        raise RuntimeError("평가점 중복·누락·비유한 지표")
    comparisons = []
    for epoch, group in absolute.groupby("epoch"):
        at = group.set_index("model_id")
        for candidate, reference in ((MODEL_ID, REFERENCE), (MODEL_ID, shared.baseline.M1), (REFERENCE, shared.baseline.M1)):
            for metric in metrics:
                value, base = float(at.at[candidate, metric]), float(at.at[reference, metric])
                comparisons.append(dict(seed=48, epoch=int(epoch), model_id=candidate, reference=reference,
                    metric=metric, reference_value=base, candidate_value=value, delta=value - base,
                    relative_change_pct=100 * (value / base - 1) if base else None))
    comparison = pd.DataFrame(comparisons)
    at = comparison[comparison.epoch.eq(300) & comparison.model_id.eq(MODEL_ID) & comparison.reference.eq(REFERENCE)].set_index("metric")
    guards = {key: bool(at.at[key, "candidate_value"] >= .99 * at.at[key, "reference_value"])
              for key in (*ACCURACY, *ECONOMIC)}
    recovery = {key: bool(at.at[key, "delta"] > 0) for key in ("recall@20", "recall@50")}
    reading = dict(primary_reference=REFERENCE, accuracy_and_economic_guards=guards,
        recall_recovery=recovery, exploratory_screen_rule_passed=all(guards.values()) and all(recovery.values()),
        significance_claim=False, clv_attribution_claim=False, final_success_claim=False,
        requires_full_raw_results_and_diagnostics_review=True)
    tables = dict(absolute=absolute, comparison=comparison, diagnostics=pd.DataFrame(diagnostics))
    root = prepared["out_dir"] / "reports"
    paths = {key: str(root / f"{key}.csv") for key in tables}
    for key, table in tables.items():
        io._atomic_csv(Path(paths[key]), table)
    paths["json"] = str(root / "result.json")
    io._atomic_json(Path(paths["json"]), prepared["preflight"] | dict(source_revision=prepared["revision"],
        config_hash=prepared["config_hash"], input_hash=prepared["input_hash"],
        baseline_provenance=prepared["baseline_provenance"], reading=reading, arms=[old, *arms],
        existing_c_diagnostic=prepared["existing_c_diagnostic"],
        new_pair_diagnostic=prepared["new_pair_diagnostic"], paths=paths))
    return tables | {"reading": reading, "paths": paths}


def run(cfg, prepared):
    validate_config(cfg)
    shared.validate_prepared(prepared)
    if prepared["preflight"]["config"] != asdict(cfg):
        raise RuntimeError("prepare 이후 설정 변경")
    _, provenance = shared.baseline.load_baseline(cfg, prepared["input_hash"])
    if provenance != prepared["baseline_provenance"]:
        raise RuntimeError("prepare 이후 M1 원본 변경")
    # Must run or rerun the read-only C audit before training; no stale boolean.
    diagnose_existing_c(cfg, prepared)
    reference, rtruth, rrecs = run_arm(cfg, prepared, REFERENCE)
    candidate, ctruth, crecs = run_arm(cfg, prepared, MODEL_ID)
    recs = [dict(r, model="m1") for r in rrecs] + [dict(r, model="m5") for r in crecs]
    prepared["new_pair_diagnostic"] = _save_pair(prepared, rtruth, ctruth, recs, 48,
        "new_pair_diagnostic", (REFERENCE, MODEL_ID))
    return report(cfg, prepared, [reference, candidate])
