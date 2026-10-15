"""Complete the fixed 42--51 panel; reuse 49/50, fit only 16 missing A/B arms."""
from dataclasses import asdict, dataclass
from io import BytesIO
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import pandas as pd

import clv_m5_ab_adaptive_lastweek_seed49 as base

VERSION = "clv-dh-m5-ab-adaptive-remaining8-v1"
SEEDS = tuple(range(42, 52))
REUSED_SEEDS = (49, 50)
NEW_SEEDS = (42, 43, 44, 45, 46, 47, 48, 51)
ORDER = ("M1", "M2", "M3", "M4", "M5_A", "M5_B", "M5_C")
CORE = tuple(f"{m}@{k}" for k in (10, 20, 50) for m in
             ("recall", "ndcg", "price_purchase_amount_weighted_hit", "vndcg"))
OLD_SHA = {
    49: {"absolute.csv": "369812a6ca1a652bd5c433490884328eaf1b92a9f822376ff0a259ad386c90b3",
         "diagnostics.csv": "d7c080c70d3907565d68caa802bc0734c794f6862b7ce6d041d8d3720c6e540b",
         "result.json": "493ac91af32546effa877a5e0996edb22238208b348b74184434a17ee304398d",
         "protocol.json": "99e667b05b9b9b385f9bd1ea4c142e798c444087bcd459a87b48ad3672483dae",
         "arms/m5_a_adaptive_m2_centered_nv_m3_bpr_k1_lastweek_s49.json": "6d89de416f91358de49728a3840ae9429f318c7870b18e7a60d3f44e93fd8e70",
         "arms/m5_b_adaptive_m2_user_centered_m4_bpr_k1_lastweek_s49.json": "9f5d9d9e634d448d1f9e3b7af6e8d5681c174a6cd949b9545ff4872587ebdbcf"},
    50: {"absolute.csv": "555c38869f614188f79d474d53d4d388ada5453c4379b6b3792a4e8368155333",
         "diagnostics.csv": "24464f66409c8a29217f4d6986abf150eb567ed59d7b636a725e41f4be14cf40",
         "result.json": "b67a4677ffa8eb95f11b2acdead5b954a1ac48235ee5493d8d240eedc2f17313",
         "protocol.json": "fa3ddf8dc5e8aa7b47ea4544c870842f86e7b2c0c343705ce5ad9a1fb0b04d40",
         "arms/m5_a_adaptive_m2_centered_nv_m3_bpr_k1_lastweek_s50.json": "b18751dabf29bf175a0547e43eff3e7a93661500a3bc64f700ee9bd79953f0c5",
         "arms/m5_b_adaptive_m2_user_centered_m4_bpr_k1_lastweek_s50.json": "290f00ffc19b83b9bb5be8a291c58c6c2e1c92d0a1eaf3c3c822e11a2a4066a8"},
}


@dataclass(frozen=True)
class Config:
    out_dir: str
    m2_reference: str
    m4_reference: str
    seed49_reference: str
    seed50_reference: str


def configure(**paths):
    root = base.v3.default_out_dir("dunnhumby")
    return Config(**({"out_dir": root + "_m5_ab_adaptive_lastweek_final10_v1",
        "m2_reference": root + "_m2_adaptive_lastweek_final10_v1",
        "m4_reference": root + "_m4c_lastweek_final10_v1",
        "seed49_reference": root + "_m5_ab_adaptive_lastweek_seed49_v1",
        "seed50_reference": root + "_m5_ab_adaptive_lastweek_seed50_v1"} | paths))


def seed_config(cfg, seed):
    return base.configure(seed=seed, out_dir=cfg.out_dir,
                          m2_reference=cfg.m2_reference, m4_reference=cfg.m4_reference)


def _previous(source, seed):
    source, expected = Path(source), OLD_SHA[seed]
    candidates = []
    if source.is_file() and source.suffix.lower() == ".zip":
        with ZipFile(source) as z:
            candidates.append((str(source), {n: z.read(n) for n in expected}))
    elif source.is_dir():
        roots = {source, source / "reports"} | {p.parent for p in source.glob("*/reports/result.json")}
        for report_dir in sorted(roots):
            run_dir = report_dir.parent if report_dir.name == "reports" else report_dir
            files = {n: (run_dir if n.startswith("arms/") or n == "protocol.json" else report_dir) / n for n in expected}
            if all(p.is_file() for p in files.values()):
                candidates.append((str(report_dir), {n: p.read_bytes() for n, p in files.items()}))
    for location, blobs in candidates:
        if all(hashlib.sha256(blobs[n]).hexdigest() == sha for n, sha in expected.items()):
            result = json.loads(blobs["result.json"])
            if not result["reading"]["complete"] or result["reading"]["seed"] != seed:
                raise RuntimeError("기존 seed49/50 완료 상태 불일치")
            return {"protocol": json.loads(blobs["protocol.json"]),
                    "absolute": pd.read_csv(BytesIO(blobs["absolute.csv"])),
                    "diagnostics": pd.read_csv(BytesIO(blobs["diagnostics.csv"])),
                    "blobs": blobs, "audit": {"location": location, "sha256": expected}}
    raise RuntimeError(f"seed{seed} 완료 원본을 찾지 못했습니다: {source}. 원본 ZIP을 지정하세요. 재학습하지 않습니다.")


def load_sources(cfg):
    for name, value in asdict(cfg).items():
        if name != "out_dir" and Path(value).resolve() == Path(cfg.out_dir).resolve():
            raise ValueError("새 출력과 참조 폴더는 분리해야 합니다")
    refs = {seed: base.load_references(seed_config(cfg, seed)) for seed in SEEDS}
    old = {seed: _previous(getattr(cfg, f"seed{seed}_reference"), seed) for seed in REUSED_SEEDS}
    p2 = refs[42]["sources"]["m2"]["protocol"]
    for seed, item in old.items():
        p = item["protocol"]
        for key in ("input_hash", "split", "data_stats", "feature_sha256"):
            if p[key] != p2[key]:
                raise RuntimeError(f"seed{seed} 기존 결과 조건 불일치: {key}")
        for key, value in asdict(seed_config(cfg, seed)).items():
            if key not in ("out_dir", "m2_reference", "m4_reference"):
                old_value = p["config"][key]
                if (tuple(old_value) if key == "seeds" else old_value) != value:
                    raise RuntimeError(f"기존 학습설정 불일치: {key}")
        a = item["absolute"].set_index("model_id")
        for row in refs[seed]["absolute"].to_dict("records"):
            for metric in refs[seed]["metrics"]:
                if not np.isclose(a.loc[row["model_id"], metric], row[metric], rtol=0, atol=1e-12):
                    raise RuntimeError("기존 A/B 파일의 참조값이 원본과 다릅니다")
    return refs, old


def prepare(cfg):
    refs, old = load_sources(cfg)  # All 54 reused rows verified before data preparation.
    shared = base.prepare(seed_config(cfg, 42))  # Input, features and graph once; no fit.
    for seed in REUSED_SEEDS:
        p = old[seed]["protocol"]
        if (p["m3_audit"] != shared["protocol"]["m3_audit"] or
                p["m4_weight_audit"] != shared["protocol"]["m4_weight_audit"]):
            raise RuntimeError("기존49/50과 새 공통 그래프/가중치가 다릅니다")
    protocol = {**shared["protocol"], "code_version": VERSION,
        "config": {**shared["protocol"]["config"], "seeds": list(SEEDS)},
        "seeds": list(SEEDS), "new_fit_seeds": list(NEW_SEEDS), "reused_ab_seeds": list(REUSED_SEEDS),
        "new_fit_models": [m for seed in NEW_SEEDS for m in base.models_for(seed_config(cfg, seed))],
        "labels": {m: label for seed in SEEDS for m, label in base.labels_for(seed_config(cfg, seed)).items()},
        "source_paths": asdict(cfg), "previous_ab_audit": {str(s): old[s]["audit"] for s in REUSED_SEEDS},
        "aggregation": "arithmetic seed mean; sample std ddof=1; all seven models use seeds42--51",
        "decision": "fixed descriptive comparison; no best-seed/epoch selection or test retuning",
        "single_seed_only": False, "significance_claim": False, "clv_attribution_claim": False,
        "unseen_confirmation_claim": False}
    config_hash = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()[:12]
    run_dir = Path(cfg.out_dir) / config_hash
    base.io._atomic_json(run_dir / "protocol.json", protocol)
    for seed, item in old.items():
        for name, blob in item["blobs"].items():
            target = run_dir / "reused" / f"seed{seed}" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.read_bytes() != blob:
                raise RuntimeError(f"재사용 사본 내용 변경: {target}")
            if not target.exists():
                target.write_bytes(blob)
    shared.update(config_hash=config_hash, run_dir=run_dir, protocol=protocol)
    prepared = {"cfg": cfg, "shared": shared, "references": refs, "previous": old,
                "run_dir": run_dir, "protocol": protocol}
    print("확인: 참조50개 + A/B seed49·50 4개 재사용. 새 학습16개, 입력 준비1회.", flush=True)
    report(prepared)
    return prepared


def context(prepared, seed):
    if seed not in NEW_SEEDS:
        raise ValueError("49·50은 원본만 재사용하며 학습할 수 없습니다")
    cfg = seed_config(prepared["cfg"], seed)
    return {**prepared["shared"], "cfg": cfg, "references": prepared["references"][seed]}


def _validate_absolute(a, metrics):
    expected = {(m, s) for m in ORDER for s in SEEDS}
    found = set(zip(a.model, a.seed))
    if len(a) != len(found) or not found.issubset(expected):
        raise RuntimeError("중복 또는 허용하지 않은 모델/시드")
    if not np.isfinite(a[metrics].to_numpy(float)).all():
        raise RuntimeError("전체142지표 중 누락 또는 비유한 값")
    return expected - found


def aggregate(a, metrics):
    missing = _validate_absolute(a, metrics)
    summary_rows, comparisons, paired_rows = [], [], []
    # Final tables exist only for the complete balanced 70-row panel.
    if missing:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    for model in ORDER:
        frame = a[a.model == model].set_index("seed").loc[list(SEEDS), metrics]
        for metric in metrics:
            summary_rows.append({"model": model, "metric": metric, "n_seeds": 10,
                "mean": float(frame[metric].mean()), "std": float(frame[metric].std(ddof=1)),
                "min": float(frame[metric].min()), "max": float(frame[metric].max())})
    for candidate in ("M5_A", "M5_B"):
        for reference in ORDER:
            if reference == candidate:
                continue
            ca = a[a.model == candidate].set_index("seed").loc[list(SEEDS), metrics]
            re = a[a.model == reference].set_index("seed").loc[list(SEEDS), metrics]
            for metric in metrics:
                x, y = ca[metric].to_numpy(float), re[metric].to_numpy(float)
                delta = x - y
                pct = np.divide(delta * 100, y, out=np.full(10, np.nan), where=y != 0)
                for seed, cv, rv, dv, pv in zip(SEEDS, x, y, delta, pct):
                    paired_rows.append({"model": candidate, "reference": reference, "seed": seed,
                        "metric": metric, "candidate_value": cv, "reference_value": rv,
                        "delta": dv, "relative_change_pct": pv})
                comparisons.append({"model": candidate, "reference": reference, "metric": metric,
                    "n_seeds": 10, "candidate_mean": x.mean(), "reference_mean": y.mean(),
                    "ratio_of_means_pct": (x.mean() / y.mean() - 1) * 100 if y.mean() else np.nan,
                    "paired_delta_mean": delta.mean(), "paired_delta_std": delta.std(ddof=1),
                    "paired_pct_mean": pct.mean(), "paired_pct_std": pct.std(ddof=1),
                    "positive_delta_seeds": int((delta > 0).sum()),
                    "negative_delta_seeds": int((delta < 0).sum()), "tie_seeds": int((delta == 0).sum())})
    return pd.DataFrame(summary_rows), pd.DataFrame(comparisons), pd.DataFrame(paired_rows)


def report(prepared):
    frames, diagnostics, missing_new = [], [], []
    metrics = prepared["references"][42]["metrics"]
    for seed in SEEDS:
        refs = prepared["references"][seed]
        labels = base.labels_for(seed_config(prepared["cfg"], seed))
        ref = refs["absolute"].copy()
        ref.insert(0, "model", ref.model_id.map(labels))
        ref["source"] = "reused_reference"
        frames.append(ref)
        diagnostics.append(refs["diagnostics"].assign(source="reused_reference"))
        if seed in REUSED_SEEDS:
            old = prepared["previous"][seed]
            frames.append(old["absolute"].loc[old["absolute"].model.isin(("M5_A", "M5_B"))].assign(source="reused_ab"))
            mids = base.models_for(seed_config(prepared["cfg"], seed))
            diagnostics.append(old["diagnostics"].loc[old["diagnostics"].model_id.isin(mids)].assign(source="reused_ab"))
        else:
            ctx = context(prepared, seed)
            for mid in base.models_for(ctx["cfg"]):
                row = base._completed(ctx, mid)
                if row is None:
                    missing_new.append(f"seed{seed}:{labels[mid]}")
                    continue
                frames.append(pd.DataFrame([{"model": labels[mid], "model_id": mid, "seed": seed,
                                            "source": "new_fit", **row["metrics"]}]))
                diagnostics.append(pd.DataFrame([{"model_id": mid, "seed": seed, "source": "new_fit", **row["diagnostics"]}]))
    absolute = pd.concat(frames, ignore_index=True).sort_values(["seed", "model"])
    missing = _validate_absolute(absolute, metrics)
    summary, comparisons, paired = aggregate(absolute, metrics)
    diag = pd.concat(diagnostics, ignore_index=True)
    reading = {"complete": not missing, "planned_seeds": list(SEEDS),
        "completed_new_fit_count": 16-len(missing_new), "planned_new_fit_count": 16,
        "reused_reference_count": 50, "reused_ab_count": 4, "missing": missing_new,
        "completed_seed_counts": absolute.groupby("model").seed.nunique().to_dict(),
        "sample_std_ddof": 1, "final_summary_available": not missing,
        "significance_claim": False, "clv_attribution_claim": False, "unseen_confirmation_claim": False}
    root = prepared["run_dir"] / "reports"
    tables = {"absolute": absolute, "diagnostics": diag}
    if not missing:
        tables.update(summary=summary, comparison=comparisons, paired_comparison=paired)
        pretty = summary.assign(value=summary.apply(lambda r: f"{r['mean']:.8f} ± {r['std']:.8f}", axis=1))
        tables["core_mean_std"] = pretty[pretty.metric.isin(CORE)].pivot(index="metric", columns="model", values="value").reindex(index=CORE, columns=ORDER).reset_index()
    paths = {}
    for name, table in tables.items():
        paths[name] = str(root / f"{name}.csv")
        base.io._atomic_csv(Path(paths[name]), table)
    paths["json"] = str(root / "result.json")
    base.io._atomic_json(Path(paths["json"]), {"protocol": prepared["protocol"], "reading": reading, "result_paths": paths})
    return {"complete": not missing, "reading": reading, "paths": paths, **tables}


def run(prepared):
    for i, seed in enumerate(NEW_SEEDS):
        ctx = context(prepared, seed)
        for j, mid in enumerate(base.models_for(ctx["cfg"])):
            print(f"[{2*i+j+1}/16] seed{seed} {base.labels_for(ctx['cfg'])[mid]}", flush=True)
            base.run_arm(ctx, mid)
            report(prepared)  # Durable partial results after every completed arm.
    return report(prepared)


def export_zip(prepared, destination, *, allow_partial=False):
    result = report(prepared)
    if not result["complete"] and not allow_partial:
        raise RuntimeError("10시드 미완료: 최종 ZIP은 만들지 않습니다. 부분 보관은 allow_partial=True")
    with ZipFile(destination, "w", ZIP_DEFLATED) as z:
        for path in result["paths"].values():
            z.write(path, Path(path).name)
        z.write(prepared["run_dir"] / "protocol.json", "protocol.json")
        for seed in NEW_SEEDS:
            ctx = context(prepared, seed)
            for mid in base.models_for(ctx["cfg"]):
                path = prepared["run_dir"] / "arms" / f"{mid}.json"
                if path.exists():
                    z.write(path, f"arms/{path.name}")
        for seed, old in prepared["previous"].items():
            for name, blob in old["blobs"].items():
                z.writestr(f"reused/seed{seed}/{name}", blob)
    return str(destination)
