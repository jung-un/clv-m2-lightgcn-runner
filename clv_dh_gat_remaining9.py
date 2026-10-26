"""Fixed GAT 42--51 panel: reuse audited seed49; fit the other 72 arms.

No changes to GAT/M2/M3/M4, sampling, loss, Adam or every-epoch checkpoint.
Partial reports never masquerade as balanced final ten-seed averages.
"""
from dataclasses import asdict, dataclass
from io import BytesIO
import hashlib
import json
from pathlib import Path
import time
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import pandas as pd
import torch

import clv_dh_gat_final8_seed49 as original

base = original.base
VERSION = "clv-dh-gat-remaining9-v1"
SEEDS = tuple(range(42, 52))
NEW_SEEDS = tuple(s for s in SEEDS if s != 49)
ORDER, ARMS, CORE = original.ORDER, original.ARMS, original.CORE
OLD_SHA = {'absolute.csv': '20d868855631395bf14406a28655b7049f253db656261a5f7a1c9bd8a467a82a', 'comparison.csv': 'ede3f765ddb9b9e28cbacbd483c028c0ba6d2781ef3c78f2a69605536b3d453f', 'diagnostics.csv': '0d48f6217c06bff5aefe9fdb9987dd8d2cf4d993ac97306e1db06435c3b8978a', 'result.json': 'ec3a4ff9967e1cb5f42a1f42f055ab4a688ff28216ab37b8b9793d5fdf6d0905', 'protocol.json': '60658c7875900de2c612dd685d8ef96f7b1bf1d5ae868d9643c03e805a297da1', 'arms/gat_m1_lastweek_s49.json': 'cf07611aba922d2797cc52d4af62f9d73d0aabbaf37de776634f9e31912f906d', 'arms/gat_m2_lastweek_s49.json': '0806e98c824357ef5e732a44f93faced2784fcea7f722a9c5bad9c57b0cf5d89', 'arms/gat_m3_lastweek_s49.json': 'bbf9b44c387b55e36d7a1be102ffb5292a76ec3f2785a547390309558392e945', 'arms/gat_m4_lastweek_s49.json': 'fd1f35bad2bbbeaf1f880547231b4a1f6009b7bc5e70135f96f7af30a229157d', 'arms/gat_m5_a_lastweek_s49.json': '205d60b42c704842a0cd3430f1bab10ddc5814f77f25f5bc7a82048515a50336', 'arms/gat_m5_b_lastweek_s49.json': 'ef1497c06cc4b441d57842cb9c7c32a24084a1dec7e040aaa18bd8fc3689729a', 'arms/gat_m5_c_lastweek_s49.json': '8add4c7873aa1553917878639d07d05597c77c63b31c6c4b57f09c4d83408e87', 'arms/gat_m5_d_lastweek_s49.json': '08d252df6f68c9b767d24fe85eef266c89164aa89017e2daff85f9db93487fe3'}

# The sole dependency change adds optional loss injection.
# Keep the original GAT uniform sampler and default loss; no loss override.
IO_SHA = "f0d2e679ae466a616272ca07d53c9db0e58882826d9532cf751cb7fc892953bb"


@dataclass(frozen=True)
class Config(original.Config):
    seeds: tuple = SEEDS
    seed49_reference: str = ""


def configure(*, out_dir=None, m2_reference=None, m4_reference=None, seed49_reference=None):
    old = original.configure(m2_reference=m2_reference, m4_reference=m4_reference)
    return Config(**{**asdict(old),
        "out_dir": out_dir or base.v3.default_out_dir("dunnhumby") + "_gat_final8_final10_v1",
        "seeds": SEEDS, "seed49_reference": seed49_reference or old.out_dir})


def validate_config(cfg):
    paths = {k: getattr(cfg, k) for k in ("out_dir", "m2_reference", "m4_reference", "seed49_reference")}
    if cfg != configure(**paths):
        raise ValueError("GAT seed42~51/고정300epoch/기존 설정을 변경할 수 없습니다")
    out = Path(cfg.out_dir).resolve()
    for value in (cfg.m2_reference, cfg.m4_reference, cfg.seed49_reference):
        source = Path(value).resolve()
        if out == source or out in source.parents or source in out.parents:
            raise ValueError("출력과 참조 경로는 서로 겹치지 않아야 합니다")


def _validate_row(row, label, seed, metrics, ident=None):
    if (row.get("model") != label or row.get("seed") != seed or row.get("epochs") != 300
            or row.get("model_id") != f"gat_{label.lower()}_lastweek"
            or set(row.get("metrics", {})) != set(metrics)
            or not np.isfinite(list(row["metrics"].values())).all()
            or [r.get("epoch") for r in row.get("training_history", [])] != list(range(1, 301))):
        raise RuntimeError(f"{label} seed{seed} 완료 결과 신원/300epoch/전체지표 불일치")
    if ident is not None and row.get("identity") != ident:
        raise RuntimeError(f"{label} seed{seed} 완료 결과 신원 불일치")
    diag = row.get("diagnostics", {})
    if (diag.get("graph") != ("centered_nv" if ARMS[label][1] else "binary")
            or diag.get("row_weighted") != ARMS[label][2]):
        raise RuntimeError("모형 개입 경로 진단 불일치")


def load_seed49(source):
    """Read-only exact original bytes from ZIP, run root, reports or parent root."""
    source = Path(source)
    candidates = []
    if source.is_file() and source.suffix.lower() == ".zip":
        with ZipFile(source) as z:
            if all(n in z.namelist() for n in OLD_SHA):
                candidates.append((str(source), {n: z.read(n) for n in OLD_SHA}))
    elif source.is_dir():
        root = source.parent if source.name == "reports" else source
        roots = {root} | {p.parent.parent for p in root.glob("*/reports/result.json")}
        for run in sorted(roots):
            paths = {n: run / n if n == "protocol.json" or n.startswith("arms/")
                     else run / "reports" / n for n in OLD_SHA}
            if all(p.is_file() for p in paths.values()):
                candidates.append((str(run), {n: p.read_bytes() for n, p in paths.items()}))
    match = next(((loc, blobs) for loc, blobs in candidates if all(
        hashlib.sha256(blobs[n]).hexdigest() == h for n, h in OLD_SHA.items())), None)
    if match is None:
        raise RuntimeError(f"seed49 GAT 완료 원본을 찾지 못했습니다: {source}. "
            "dh_gat_final8_seed49_results.zip의 Drive 경로를 지정하세요. seed49는 재학습하지 않습니다.")
    location, blobs = match
    protocol, result = (json.loads(blobs[n]) for n in ("protocol.json", "result.json"))
    absolute = pd.read_csv(BytesIO(blobs["absolute.csv"]))
    metrics = [c for c in absolute if c not in ("model", "model_id", "seed")]
    if (len(absolute) != 8 or set(absolute.model) != set(ORDER) or set(absolute.seed) != {49}
            or len(metrics) != 142 or not np.isfinite(absolute[metrics].to_numpy()).all()
            or result["protocol"] != protocol or not result["reading"]["complete"]):
        raise RuntimeError("seed49 원본의 전체8모형/142지표/완료 상태 불일치")
    rows = {}
    for label in ORDER:
        row = json.loads(blobs[f"arms/gat_{label.lower()}_lastweek_s49.json"])
        _validate_row(row, label, 49, metrics)
        if (row["identity"]["source_revision"] != protocol["source_revision"]
                or row["identity"]["input_hash"] != protocol["input_hash"]
                or row["diagnostics"]["feature_sha256"] != protocol["feature_sha256"]):
            raise RuntimeError("seed49 원본 신원/입력 지문 불일치")
        saved = absolute.set_index("model").loc[label, metrics].to_numpy(float)
        # CSV is rounded by the existing exporter; arm JSON retains full precision.
        if not np.allclose(saved, [row["metrics"][m] for m in metrics], rtol=1e-8, atol=1e-12):
            raise RuntimeError("seed49 CSV와 arm 원본 지표가 다릅니다")
        rows[label] = row
    return {"protocol": protocol, "rows": rows, "metrics": metrics, "blobs": blobs,
            "audit": {"location": location, "sha256": OLD_SHA}}


def check_source_compatibility(previous):
    fingerprints = previous["protocol"]["code_sha256"]
    current = {n: hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest()
               for n in fingerprints}
    for name, sha in current.items():
        expected = IO_SHA if name == "clv_ngcf_training_io.py" else fingerprints[name]
        if sha != expected:
            raise RuntimeError(f"seed49와 비교하는 고정 코드가 변경되었습니다: {name}")
    return {"seed49_sha256": fingerprints, "current_sha256": current,
            "difference": "training_io only: optional loss injection unused; original GAT sampler and default loss preserved"}


def compare_audits(old, now):
    # Rank correlations of group means near ties can differ across BLAS builds.
    # They remain subject to the original graph gates in original.prepare().
    rank_keys = {"customer_weight_vs_degree", "item_weight_vs_price", "item_weight_vs_popularity"}
    differences = {}
    for section in ("m3_audit", "m4_weight_audit"):
        a, b = old[section], now[section]
        if set(a) != set(b):
            raise RuntimeError(f"{section} 진단 항목 불일치")
        for key in a:
            x, y = a[key], b[key]
            if section == "m3_audit" and key in rank_keys:
                if not np.isfinite([x, y]).all():
                    raise RuntimeError("M3 상관 진단 비유한값")
            elif isinstance(x, float):
                if not np.isclose(x, y, rtol=1e-10, atol=1e-12):
                    raise RuntimeError(f"{section}.{key} 값 불일치")
            elif x != y:
                raise RuntimeError(f"{section}.{key} 불일치")
            if x != y:
                differences[f"{section}.{key}"] = {"seed49": x, "current": y}
    return {"differences": differences,
            "rule": "input/feature/M4 hash and frozen M3 code exact; numeric audits rtol1e-10 atol1e-12; M3 rank correlations retain original gates and are recorded, not required bitwise equal across platforms"}


def prepare(cfg):
    validate_config(cfg)
    previous = load_seed49(cfg.seed49_reference)  # Before costly input preparation.
    compatibility = check_source_compatibility(previous)
    old = previous["protocol"]
    for key, value in asdict(cfg).items():
        if key not in ("out_dir", "m2_reference", "m4_reference", "seed49_reference", "seeds"):
            if old["config"][key] != value:
                raise RuntimeError(f"seed49 학습설정 불일치: {key}")
    shared = original.prepare(original.configure(out_dir=str(Path(cfg.out_dir) / "input_audit"),
        m2_reference=cfg.m2_reference, m4_reference=cfg.m4_reference))
    now = shared["protocol"]
    for key in ("input_hash", "data_stats", "split", "feature_sha256",
                "dataset_initialization", "fixed_intervention_settings", "models", "backbone", "propagation",
                "adjacency", "regularization", "m2_adaptation", "initialization"):
        if now[key] != old[key]:
            raise RuntimeError(f"seed49와 준비된 입력/설계가 다릅니다: {key}")
    audit_comparison = compare_audits(old, now)
    if set(shared["references"]["metrics"]) != set(previous["metrics"]):
        raise RuntimeError("seed49와 전체 지표 목록 불일치")
    protocol = {**now, "code_version": VERSION, "config": asdict(cfg),
        "code_sha256": {**now["code_sha256"], Path(__file__).name:
            hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "new_fit_count": 72, "reused_gat_results": 8, "new_fit_seeds": NEW_SEEDS,
        "reused_seeds": [49], "single_seed_only": False,
        "seed49_audit": previous["audit"], "source_compatibility": compatibility,
        "cross_platform_audit": audit_comparison,
        "aggregation": "8 models x seeds42--51; full precision arm metrics; arithmetic mean and sample SD ddof=1; final tables only with all80 rows",
        "seed49_checkpoint_reused_for_initialization": False,
        "decision": "fixed 72 fits regardless of interim scores; no seed/epoch/model selection, test retuning or significance claim"}
    protocol = json.loads(json.dumps(protocol))
    config_hash = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()[:12]
    run_dir = Path(cfg.out_dir) / config_hash
    path = run_dir / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise RuntimeError("기존 protocol 불일치: 덮어쓰지 않습니다")
    base.io._atomic_json(path, protocol)
    for name, blob in previous["blobs"].items():
        target = run_dir / "reused" / "seed49" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != blob:
            raise RuntimeError(f"seed49 보관 사본 변경: {target}")
        if not target.exists():
            temporary = target.with_suffix(target.suffix + ".tmp")
            temporary.write_bytes(blob)
            temporary.replace(target)
    shared.update(cfg=cfg, previous=previous, config_hash=config_hash, run_dir=run_dir, protocol=protocol)
    base.io._atomic_json(run_dir / "reference_audit.json", {
        "GAT_seed49": previous["audit"], "LightGCN_inputs_only": shared["references"]["audit"]})
    print("GAT 계획: seed49 8개 원본 재사용 / 새 학습72개 / 입력 준비1회 / 각300epoch", flush=True)
    report(shared)
    return shared


def identity(prepared, label, seed):
    if label not in ORDER or seed not in NEW_SEEDS:
        raise ValueError("새 학습은 8모형×seed42~48·50·51만 허용합니다. seed49는 재사용 전용입니다")
    return base.RunIdentity(VERSION, f"gat_{label.lower()}_lastweek", seed,
                           prepared["config_hash"], prepared["revision"], prepared["input_hash"])


def arm_path(prepared, label, seed):
    ident = identity(prepared, label, seed)
    return prepared["run_dir"] / "arms" / f"{ident.model_id}_s{seed}.json"


def completed(prepared, label, seed):
    ident = identity(prepared, label, seed)
    path = arm_path(prepared, label, seed)
    if not path.exists():
        return None
    row = json.loads(path.read_text())
    _validate_row(row, label, seed, prepared["references"]["metrics"], asdict(ident))
    if row["diagnostics"].get("feature_sha256") != prepared["feature_sha256"]:
        raise RuntimeError("완료 결과 특징 지문 불일치")
    return row


def run_arm(prepared, label, seed):
    validate_config(prepared["cfg"])
    ident = identity(prepared, label, seed)  # Blocks seed49 before model creation.
    old = completed(prepared, label, seed)
    if old is not None:
        print(f"[재사용] GAT {label} seed{seed} — 재학습·재평가 없음", flush=True)
        return old
    if base.m2._feature_hash(prepared["features"]) != prepared["feature_sha256"]:
        raise RuntimeError("준비 후 M2 특징 변경")
    store = original.SafeProgressStore(prepared["run_dir"] / "progress", ident)
    base.v3.set_seed(seed)
    model, weights = original.build_model(prepared, label)
    base.v3.set_seed(seed)  # Identical dropout RNG after extra M2 initialization.
    started = time.monotonic()
    try:
        history = original.execution.train(model, prepared, prepared["cfg"], ident.model_id,
                                           seed, store, row_weights=weights,
                                           negative_sampler=original.sample_uniform_negative_matrix)
        if [r["epoch"] for r in history] != list(range(1, 301)):
            raise RuntimeError("300epoch 미완료: test 평가 금지")
        print(f"[평가] GAT {label} seed{seed} 고정300epoch 마지막점", flush=True)
        metrics = base.fixed._evaluate(model, prepared)
        row = {"identity": asdict(ident), "model": label, "model_id": ident.model_id,
            "seed": seed, "epochs": 300, "metrics": metrics, "training_history": history,
            "this_session_seconds": time.monotonic() - started,
            "diagnostics": model.representation_diagnostics() | model.training_gradient_diagnostics() |
                {"graph": "centered_nv" if ARMS[label][1] else "binary",
                 "row_weighted": ARMS[label][2], "feature_sha256": prepared["feature_sha256"]}}
        _validate_row(row, label, seed, prepared["references"]["metrics"], asdict(ident))
        path = arm_path(prepared, label, seed)
        base.io._atomic_json(path, row)
        store.mark_complete(epoch=300, max_epoch=300, selection="none", result_path=str(path))
        return row
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def aggregate(absolute, metrics):
    expected = {(m, s) for m in ORDER for s in SEEDS}
    found = set(zip(absolute.model, absolute.seed))
    if len(absolute) != len(found) or not found.issubset(expected):
        raise RuntimeError("중복 또는 허용하지 않은 모델/시드")
    if len(metrics) != 142 or not np.isfinite(absolute[metrics].to_numpy(float)).all():
        raise RuntimeError("전체142지표 누락/비유한 값")
    if found != expected:
        return {}
    summary, comparison, paired = [], [], []
    frames = {m: absolute[absolute.model == m].set_index("seed").loc[list(SEEDS), metrics] for m in ORDER}
    for model, frame in frames.items():
        for metric in metrics:
            x = frame[metric].to_numpy(float)
            summary.append({"model": model, "metric": metric, "n_seeds": 10,
                            "mean": x.mean(), "std": x.std(ddof=1), "min": x.min(), "max": x.max()})
            if model == "M1":
                continue
            y = frames["M1"][metric].to_numpy(float)
            delta = x - y
            pct = np.divide(100 * delta, y, out=np.full(10, np.nan), where=y != 0)
            comparison.append({"model": model, "reference": "M1", "metric": metric, "n_seeds": 10,
                "candidate_mean": x.mean(), "reference_mean": y.mean(),
                "relative_change_pct": 100 * (x.mean()/y.mean()-1) if y.mean() else np.nan,
                "paired_delta_mean": delta.mean(), "paired_delta_std": delta.std(ddof=1),
                "positive_delta_seeds": int((delta > 0).sum()), "negative_delta_seeds": int((delta < 0).sum())})
            paired.extend({"model": model, "reference": "M1", "seed": s, "metric": metric,
                "candidate_value": a, "reference_value": b, "delta": d, "relative_change_pct": p}
                for s, a, b, d, p in zip(SEEDS, x, y, delta, pct))
    summary = pd.DataFrame(summary)
    pretty = summary.assign(value=summary.apply(lambda r: f"{r['mean']:.8f} ± {r['std']:.8f}", axis=1))
    core = pretty[pretty.metric.isin(CORE)].pivot(index="metric", columns="model", values="value")
    return {"summary": summary, "comparison_vs_m1": pd.DataFrame(comparison),
        "paired_vs_m1": pd.DataFrame(paired),
        "core_mean_std": core.reindex(index=CORE, columns=ORDER).reset_index()}


def report(prepared):
    """Only saved results. No training, re-evaluation or unbalanced final means."""
    rows = [prepared["previous"]["rows"][m] for m in ORDER]
    missing = []
    for seed in NEW_SEEDS:
        for label in ORDER:
            row = completed(prepared, label, seed)
            if row is None:
                missing.append(f"seed{seed}:{label}")
            else:
                rows.append(row)
    absolute = pd.DataFrame([{"model": r["model"], "model_id": r["model_id"], "seed": r["seed"],
        "source": "reused_seed49" if r["seed"] == 49 else "new_fit", **r["metrics"]} for r in rows])
    absolute = absolute.sort_values(["seed", "model"]).reset_index(drop=True)
    tables = {"absolute": absolute, "diagnostics": pd.DataFrame([
        {"model": r["model"], "seed": r["seed"], **r["diagnostics"]} for r in rows])}
    tables.update(aggregate(absolute, prepared["references"]["metrics"]))
    reading = {"complete": not missing, "backbone": "GAT", "planned_seeds": list(SEEDS),
        "completed_new_fit_count": 72-len(missing), "planned_new_fit_count": 72,
        "reused_seed49_count": 8, "completed_total": len(rows), "planned_total": 80,
        "missing": missing, "completed_seed_counts": absolute.groupby("model").seed.nunique().to_dict(),
        "sample_std_ddof": 1, "final_summary_available": not missing, "fixed_epoch": 300,
        "significance_claim": False, "clv_attribution_claim": False, "unseen_confirmation_claim": False,
        "retuning_allowed": False, "note": "SD is seed variability, not CI or significance; final tables require all80 results"}
    paths = {}
    for name, frame in tables.items():
        paths[name] = str(prepared["run_dir"] / "reports" / f"{name}.csv")
        base.io._atomic_csv(Path(paths[name]), frame)
    paths["json"] = str(prepared["run_dir"] / "reports" / "result.json")
    base.io._atomic_json(Path(paths["json"]), {"protocol": prepared["protocol"], "reading": reading, "result_paths": paths})
    return {"complete": not missing, "reading": reading, "paths": paths, **tables}


def run(prepared):
    validate_config(prepared["cfg"])
    for i, seed in enumerate(NEW_SEEDS):
        for j, label in enumerate(ORDER):
            print(f"[{8*i+j+1}/72] Dunnhumby GAT seed{seed} {label}", flush=True)
            run_arm(prepared, label, seed)
            result = report(prepared)
            print(f"[저장] 새 학습 {result['reading']['completed_new_fit_count']}/72 완료 + seed49 재사용8개", flush=True)
    return result


def export_zip(prepared, destination, *, allow_partial=False):
    result = report(prepared)
    path = Path(destination)
    if not result["complete"]:
        if not allow_partial:
            raise RuntimeError("10시드 미완료: 부분 보관만 allow_partial=True로 가능합니다")
        if not path.stem.endswith("_partial"):
            path = path.with_name(path.stem + "_partial" + path.suffix)
    path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(path, "w", ZIP_DEFLATED) as z:
        for file in result["paths"].values():
            z.write(file, Path(file).name)
        for name in ("protocol.json", "reference_audit.json"):
            z.write(prepared["run_dir"] / name, name)
        for seed in NEW_SEEDS:
            for label in ORDER:
                file = arm_path(prepared, label, seed)
                if file.exists():
                    z.write(file, "arms/" + file.name)
        for name, blob in prepared["previous"]["blobs"].items():
            z.writestr("reused/seed49/" + name, blob)
        for file in sorted((prepared["run_dir"] / "progress" / "timing").glob("*.csv")):
            z.write(file, "timing/" + file.name)
    return str(path)
