"""Fixed final-week M5_D panel: ten joint fits, fifty verified reference rows."""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import pandas as pd
import torch

import clv_m5_ab_adaptive_lastweek_seed49 as base

VERSION = "clv-dh-m5-d-adaptive-final10-v1"
SEEDS = tuple(range(42, 52))
ORDER = ("M1", "M2", "M3", "M4", "M5_C", "M5_D")
CORE = tuple(f"{m}@{k}" for k in (10, 20, 50) for m in
             ("recall", "ndcg", "price_purchase_amount_weighted_hit", "vndcg"))


@dataclass(frozen=True)
class Config:
    out_dir: str
    m2_reference: str
    m4_reference: str


def configure(**paths):
    root = base.v3.default_out_dir("dunnhumby")
    return Config(**({"out_dir": root + "_m5_d_adaptive_lastweek_final10_v1",
        "m2_reference": root + "_m2_adaptive_lastweek_final10_v1",
        "m4_reference": root + "_m4c_lastweek_final10_v1"} | paths))


def seed_config(cfg, seed):
    if seed not in SEEDS:
        raise ValueError("사전 고정 시드는42~51입니다")
    return base.configure(seed=seed, out_dir=cfg.out_dir,
                          m2_reference=cfg.m2_reference, m4_reference=cfg.m4_reference)


def model_id(seed):
    if seed not in SEEDS:
        raise ValueError("사전 고정 시드는42~51입니다")
    return f"m5_d_adaptive_m2_centered_nv_m3_user_centered_m4_bpr_k1_lastweek_s{seed}"


def load_references(cfg):
    for source in (cfg.m2_reference, cfg.m4_reference):
        if Path(cfg.out_dir).resolve() == Path(source).resolve():
            raise ValueError("출력과 참조 경로는 분리해야 합니다")
    return {seed: base.load_references(seed_config(cfg, seed)) for seed in SEEDS}


def prepare(cfg):
    refs = load_references(cfg)  # Reject incomplete/mismatched references before any fit.
    sc = seed_config(cfg, 42)
    p2, p4 = (refs[42]["sources"][k]["protocol"] for k in ("m2", "m4"))
    for name, sha in p4.get("frozen_code_sha256", {}).items():
        if hashlib.sha256(Path(base.__file__).with_name(name).read_bytes()).hexdigest() != sha:
            raise RuntimeError(f"확정 M3/M4 구현이 변경되었습니다: {name}")
    # This helper only prepares train data, the centered N/V graph and M4 weights.
    shared = base.prior.prepare(base.prior.configure(out_dir=str(Path(cfg.out_dir) / "inputs")))
    data = shared["data"]
    if shared["input_hash"] != p2["input_hash"] or data["data_stats"] != p2["data_stats"]:
        raise RuntimeError("기존 참조와 입력/마지막주 분할 불일치")
    if set(data["splits"]) != {"test"} or "b_raw" not in data["train"]:
        raise RuntimeError("test 단일 분할·BASKET_ID 구매단위가 필요합니다")
    if shared["observed_weight_audit"]["sha256"] != p4["m4_c_weight_audit"]["sha256"]:
        raise RuntimeError("기존 M4와 손실 가중치 불일치")
    features = base.m2.build_features(data["train"], n_users=data["n_users"], n_items=data["n_items"],
        q_n=shared["q_n"], q_v=shared["q_v"], valid=shared["clv_valid"],
        bandwidth=sc.bandwidth, reliability_floor=sc.reliability_floor)
    feature_sha = base.m2._feature_hash(features)
    if feature_sha != p2["feature_sha256"] or not np.array_equal(features["keys"], data["pos_key"]):
        raise RuntimeError("기존 M2와 특징/구매쌍 순서 불일치")
    shared.update(features=features, feature_sha256=feature_sha)
    shared["base_cfg"]["SEED_LIST"] = list(SEEDS)
    protocol = {"code_version": VERSION, "source_revision": shared["revision"],
        "config": {**asdict(sc), "seeds": list(SEEDS)}, "source_paths": asdict(cfg),
        "input_hash": shared["input_hash"], "data_stats": data["data_stats"], "split": p2["split"],
        "feature_sha256": feature_sha, "m3_audit": shared["observed_graph"]["audit"],
        "m4_weight_audit": shared["observed_weight_audit"], "fixed": p4["fixed"],
        "reference_audit": refs[42]["audit"], "seeds": list(SEEDS), "fixed_epoch": 300,
        "new_fit_models": [model_id(s) for s in SEEDS], "new_fit_count": 10,
        "reused_models": list(refs[42]["absolute"].model_id), "reused_reference_count": 50,
        "model": "M5_D: adaptive M2 representation + centered N/V M3 graph + user-centered M4 BPR weights",
        "initialization": "fresh joint initialization; single optimizer; no reference checkpoints loaded",
        "aggregation": "same seeds42--51; arithmetic mean; sample std ddof=1; all142 metrics",
        "decision": "descriptive paired comparisons; no best-seed/epoch selection or test retuning",
        "validation": False, "holdout": False, "early_stopping": False,
        "test_evaluations_per_fit": 1, "single_seed_only": False,
        "significance_claim": False, "clv_attribution_claim": False, "unseen_confirmation_claim": False}
    config_hash = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()[:12]
    run_dir = Path(cfg.out_dir) / config_hash
    shared.update(config_hash=config_hash, run_dir=run_dir, protocol=protocol)
    prepared = {"cfg": cfg, "shared": shared, "references": refs, "run_dir": run_dir, "protocol": protocol}
    base.io._atomic_json(run_dir / "protocol.json", protocol)
    base.io._atomic_json(run_dir / "reference_audit.json", refs[42]["audit"])
    print("확인: 기존5모형×10시드=50개 재사용 / 새 학습 M5_D만10개 / 입력 준비1회", flush=True)
    report(prepared)
    return prepared


def context(prepared, seed):
    return {**prepared["shared"], "cfg": seed_config(prepared["cfg"], seed),
            "references": prepared["references"][seed]}


def identity(ctx):
    seed = ctx["cfg"].seeds[0]
    return base.RunIdentity(VERSION, model_id(seed), seed, ctx["config_hash"],
                            ctx["revision"], ctx["input_hash"])


def completed(ctx):
    ident = identity(ctx)
    path = ctx["run_dir"] / "arms" / f"{ident.model_id}.json"
    if not path.exists():
        return None
    row = json.loads(path.read_text())
    if (row.get("identity") != asdict(ident) or row.get("epochs") != 300
            or row.get("model_id") != ident.model_id or row.get("seed") != ident.seed
            or set(row.get("metrics", {})) != set(ctx["references"]["metrics"])
            or not np.isfinite(list(row["metrics"].values())).all()):
        raise RuntimeError("완료 M5_D의 신원/300epoch/142지표 불일치")
    return row


def run_arm(ctx):
    cfg, data = ctx["cfg"], ctx["data"]
    base._validate_config(cfg)
    old = completed(ctx)
    if old is not None:
        print(f"[재사용] M5_D seed{cfg.seeds[0]} — 재학습·재평가 없음", flush=True)
        return old
    if base.m2._feature_hash(ctx["features"]) != ctx["feature_sha256"]:
        raise RuntimeError("준비 후 M2 특징 변경")
    ident = identity(ctx)
    base.v3.set_seed(ident.seed)
    model = base.m2.DatasetAdaptiveOrthogonalNVLightGCN(
        n_users=data["n_users"], n_items=data["n_items"], features=ctx["features"],
        adj=ctx["observed_adj"], id_dim=cfg.id_dim, n_layers=cfg.n_layers,
        pref_reg=cfg.pref_reg, shared_l2=cfg.shared_l2).to(base.v3.DEVICE)
    store = base.ProgressStore(ctx["run_dir"] / "progress", ident)
    history = base.fixed._train(model, ctx, cfg, ident.model_id, ident.seed, store,
                                row_weights=ctx["observed_weights"])
    if [r["epoch"] for r in history] != list(range(1, 301)):
        raise RuntimeError("300epoch 미완료: test 평가 금지")
    metrics = base.fixed._evaluate(model, ctx)
    if set(metrics) != set(ctx["references"]["metrics"]) or not np.isfinite(list(metrics.values())).all():
        raise RuntimeError("최종142지표 불완전")
    row = {"identity": asdict(ident), "model_id": ident.model_id, "seed": ident.seed,
        "epochs": 300, "metrics": metrics, "training_history": history,
        "diagnostics": {**model.representation_diagnostics(), **model.training_gradient_diagnostics(),
            "graph": "centered_nv", "row_weighted": True, "feature_sha256": ctx["feature_sha256"]}}
    path = ctx["run_dir"] / "arms" / f"{ident.model_id}.json"
    base.io._atomic_json(path, row)
    store.mark_complete(epoch=300, max_epoch=300, selection="none",
                        checkpoint_path=str(store.latest_checkpoint), result_path=str(path))
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return row


def aggregate(a, metrics):
    expected = {(m, s) for m in ORDER for s in SEEDS}
    found = set(zip(a.model, a.seed))
    if len(a) != len(found) or not found.issubset(expected):
        raise RuntimeError("중복 또는 허용하지 않은 모델/시드")
    if len(metrics) != 142 or not np.isfinite(a[metrics].to_numpy(float)).all():
        raise RuntimeError("전체142지표 중 누락 또는 비유한 값")
    if found != expected:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    summary, comparisons, paired = [], [], []
    panels = {m: a[a.model == m].set_index("seed").loc[list(SEEDS), metrics] for m in ORDER}
    for m, frame in panels.items():
        for metric in metrics:
            summary.append({"model": m, "metric": metric, "n_seeds": 10,
                "mean": frame[metric].mean(), "std": frame[metric].std(ddof=1),
                "min": frame[metric].min(), "max": frame[metric].max()})
    for ref in ORDER[:-1]:
        for metric in metrics:
            x, y = panels["M5_D"][metric].to_numpy(float), panels[ref][metric].to_numpy(float)
            delta = x-y
            pct = np.divide(100*delta, y, out=np.full(10, np.nan), where=y != 0)
            for seed, cv, rv, dv, pv in zip(SEEDS, x, y, delta, pct):
                paired.append({"model": "M5_D", "reference": ref, "seed": seed, "metric": metric,
                    "candidate_value": cv, "reference_value": rv, "delta": dv, "relative_change_pct": pv})
            comparisons.append({"model": "M5_D", "reference": ref, "metric": metric, "n_seeds": 10,
                "candidate_mean": x.mean(), "reference_mean": y.mean(),
                "ratio_of_means_pct": 100*(x.mean()/y.mean()-1) if y.mean() else np.nan,
                "paired_delta_mean": delta.mean(), "paired_delta_std": delta.std(ddof=1),
                "paired_pct_mean": pct.mean(), "paired_pct_std": pct.std(ddof=1),
                "positive_delta_seeds": int((delta>0).sum()), "negative_delta_seeds": int((delta<0).sum()),
                "tie_seeds": int((delta==0).sum())})
    return pd.DataFrame(summary), pd.DataFrame(comparisons), pd.DataFrame(paired)


def report(prepared):
    frames, diagnostics, missing = [], [], []
    metrics = prepared["references"][42]["metrics"]
    for seed in SEEDS:
        ctx = context(prepared, seed)
        refs = ctx["references"]
        ref = refs["absolute"].copy()
        ref.insert(0, "model", ref.model_id.map(base.labels_for(ctx["cfg"])))
        frames.append(ref.assign(source="reused_reference"))
        diagnostics.append(refs["diagnostics"].assign(source="reused_reference"))
        row = completed(ctx)
        if row is None:
            missing.append(seed)
            continue
        frames.append(pd.DataFrame([{"model": "M5_D", "model_id": row["model_id"], "seed": seed,
                                    "source": "new_fit", **row["metrics"]}]))
        diagnostics.append(pd.DataFrame([{"model_id": row["model_id"], "seed": seed,
                                         "source": "new_fit", **row["diagnostics"]}]))
    absolute = pd.concat(frames, ignore_index=True).sort_values(["seed", "model"])
    summary, comparison, paired = aggregate(absolute, metrics)
    reading = {"complete": not missing, "planned_seeds": list(SEEDS), "missing_seeds": missing,
        "completed_new_fit_count": 10-len(missing), "planned_new_fit_count": 10,
        "reused_reference_count": 50, "fixed_epoch": 300, "sample_std_ddof": 1,
        "completed_seed_counts": absolute.groupby("model").seed.nunique().to_dict(),
        "final_summary_available": not missing, "significance_claim": False,
        "clv_attribution_claim": False, "unseen_confirmation_claim": False}
    tables = {"absolute": absolute, "diagnostics": pd.concat(diagnostics, ignore_index=True)}
    if not missing:
        tables.update(summary=summary, comparison=comparison, paired_comparison=paired)
        pretty = summary.assign(value=summary.apply(lambda r: f"{r['mean']:.8f} ± {r['std']:.8f}", axis=1))
        tables["core_mean_std"] = pretty[pretty.metric.isin(CORE)].pivot(
            index="metric", columns="model", values="value").reindex(index=CORE, columns=ORDER).reset_index()
    paths = {}
    for name, table in tables.items():
        paths[name] = str(prepared["run_dir"] / "reports" / f"{name}.csv")
        base.io._atomic_csv(Path(paths[name]), table)
    paths["json"] = str(prepared["run_dir"] / "reports" / "result.json")
    base.io._atomic_json(Path(paths["json"]), {"protocol": prepared["protocol"], "reading": reading,
                                             "result_paths": paths})
    return {"complete": not missing, "reading": reading, "paths": paths, **tables}


def run(prepared):
    for index, seed in enumerate(SEEDS, 1):
        print(f"[{index}/10] Dunnhumby M5_D seed{seed}", flush=True)
        run_arm(context(prepared, seed))
        report(prepared)
    return report(prepared)


def export_zip(prepared, destination, *, allow_partial=False):
    result = report(prepared)
    if not result["complete"] and not allow_partial:
        raise RuntimeError("10시드 미완료: 부분 보관은 allow_partial=True")
    with ZipFile(destination, "w", ZIP_DEFLATED) as archive:
        for path in result["paths"].values():
            archive.write(path, Path(path).name)
        for name in ("protocol.json", "reference_audit.json"):
            archive.write(prepared["run_dir"] / name, name)
        for seed in SEEDS:
            path = prepared["run_dir"] / "arms" / f"{model_id(seed)}.json"
            if path.exists():
                archive.write(path, f"arms/{path.name}")
    return str(destination)
