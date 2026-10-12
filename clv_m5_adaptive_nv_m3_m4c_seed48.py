"""One new joint adaptive M2+fixed M3+M4-C fit; reuse completed seed48 C.

Historical development DAY684--690 only. The completed reference supplies
both M1 and M3+M4-C curves. Neither reference is trained or scored again.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import pandas as pd
import torch

import clv_m2_dataset_adaptive_orthogonal_nv_seed48 as adaptive
import clv_m5_m3_m4_user_centered_screen as m4c
from clv_m5_shared_attributes_seed48_screen import LocalFirstProgressStore, validate_prepared
from clv_run_state import RunIdentity


m3, capacity, v3, io = adaptive.m3, adaptive.capacity, adaptive.v3, adaptive.io
CODE_VERSION = "clv-m5-adaptive-nv-m3-m4c-seed48-dev-v1"
REFERENCE_VERSION = "clv-m5-shared-attributes-fixed-m3-m4c-seed48-dev-v1"
REFERENCE_REVISION = "883ffd084b9b2cec81a0aacc4ad3031fa183ad8b"
M2_REVISION = "a17035f97c23380968399d17d1f3d6847be06f88"
ROOT = v3.default_out_dir("dunnhumby")
SEED = 48
M1, REFERENCE = adaptive.M1, m4c.MODEL_ID
MODEL_ID = "m5_adaptive_orthogonal_nv_centered_graph_user_centered_bpr_k1"
SPLIT = "historical_development_days_684_690"
NUMERIC_TOLERANCE = 1e-8


@dataclass(frozen=True)
class Config(m3.CenteredGraphConfig):
    seeds: tuple[int, ...] = (SEED,)
    out_dir: str = ROOT + "_m5_adaptive_nv_m3_m4c_seed48_dev_v1"
    m1_result_dir: str = ROOT + "_clv_m2_capacity_search_v1"
    reference_json: str = ROOT + "_m5_shared_attributes_fixed_m3_m4c_seed48_dev_v1/reports/result.json"


def configure(**paths) -> Config:
    if set(paths) - {"out_dir", "reference_json"}:
        raise ValueError("seed48·300epoch·M2/M3/M4 설정은 고정이며 경로만 지정할 수 있습니다")
    return validate_config(Config(**paths))


def validate_config(cfg: Config) -> Config:
    for key, value in asdict(Config()).items():
        if key not in ("out_dir", "reference_json") and getattr(cfg, key) != value:
            raise ValueError(f"사전 고정 설정 불일치: {key}")
    if not cfg.out_dir or not cfg.reference_json:
        raise ValueError("결과·완료 참조 경로가 필요합니다")
    if Path(cfg.out_dir).resolve() == Path(cfg.reference_json).resolve().parents[1]:
        raise ValueError("기존 결과를 덮어쓰지 않도록 별도 out_dir를 사용하세요")
    return m3.validate_config(cfg)


def load_reference(cfg: Config) -> dict:
    path = Path(cfg.reference_json)
    if not path.is_file():
        raise FileNotFoundError(f"완료 seed48 M3+M4-C 결과가 필요합니다. 재학습 없음: {path}")
    old = json.loads(path.read_text())
    fields = ("epochs", "eval_every", "batch_size", "lr", "n_layers", "id_dim",
              "pref_reg", "negative_count", "target_cv")
    if (old.get("code_version") != REFERENCE_VERSION
            or old.get("source_revision") != REFERENCE_REVISION
            or old.get("split") != SPLIT or old.get("final_test") is not False
            or old.get("holdout") is not False or old.get("config", {}).get("seeds") != [SEED]
            or old.get("train_days") != [1, 683] or old.get("development_days") != [684, 690]
            or any(old["config"].get(key) != getattr(cfg, key) for key in fields)):
        raise RuntimeError("기존 C형의 source·seed·개발분할·고정학습설정 불일치")
    arms = {a["model_id"]: a for a in old.get("arms", [])}
    if not {M1, REFERENCE}.issubset(arms):
        raise RuntimeError("참조 JSON에 완료 M1/C형 arm이 없습니다")
    absolute_path = path.parent / "absolute.csv"
    absolute = pd.read_csv(absolute_path)
    curves, metrics = [], None
    for model_id in (M1, REFERENCE):
        arm = arms[model_id]
        if arm.get("seed") != SEED:
            raise RuntimeError("참조 arm seed 불일치")
        if model_id == M1 and (
                arm.get("source_revision") != adaptive.REFERENCE_REVISION
                or arm.get("code_version") != adaptive.REFERENCE_CODE_VERSION
                or arm.get("input_hash") != old.get("input_hash")):
            raise RuntimeError("재사용 M1의 source·입력 신원 불일치")
        if model_id == REFERENCE:
            identity = arm.get("identity", {})
            expected = asdict(RunIdentity(REFERENCE_VERSION, REFERENCE, SEED,
                                         old.get("config_hash"), REFERENCE_REVISION, old.get("input_hash")))
            if (identity != expected
                    or arm.get("m3_weight_sha256") != old.get("m3_weight_sha256")
                    or arm.get("m4_weight_sha256") != old.get("m4_audit", {}).get("sha256")):
                raise RuntimeError("C형 완료 arm과 보고서의 신원·가중치가 다릅니다")
        curve = [r for r in arm.get("curve", []) if "metrics" in r]
        if ([r["epoch"] for r in curve] != list(range(25, 301, 25))
                or arm["curve"][-1]["epoch"] != 300):
            raise RuntimeError("참조는 고정300epoch까지 완료되어야 합니다")
        for record in curve:
            metrics = sorted(record["metrics"]) if metrics is None else metrics
            if set(record["metrics"]) != set(metrics):
                raise RuntimeError("참조 전체 지표 키가 다릅니다")
            matching = absolute[absolute.model_id.eq(model_id) & absolute.seed.eq(SEED)
                                & absolute.epoch.eq(record["epoch"])]
            values = np.asarray([record["metrics"][key] for key in metrics], float)
            if (len(matching) != 1 or not np.isfinite(values).all()
                    or not np.allclose(matching.iloc[0][metrics].to_numpy(float), values,
                                       rtol=0, atol=1e-7)):
                raise RuntimeError("참조 absolute 원본과 JSON 곡선 불일치")
            curves.append({"model_id": model_id, "seed": SEED, "epoch": record["epoch"],
                           **record["metrics"]})
    return {"payload": old, "absolute": pd.DataFrame(curves), "metrics": metrics,
            "audit": {"result_path": str(path), "result_sha256": _sha_file(path),
                      "absolute_path": str(absolute_path), "absolute_sha256": _sha_file(absolute_path),
                      "source_revision": REFERENCE_REVISION}}


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_source() -> dict:
    root = Path(__file__).resolve().parent
    originals = {
        "clv_dataset_adaptive_orthogonal_nv_model.py": M2_REVISION,
        "clv_reliability_orthogonal_nv_model.py": M2_REVISION,
        "clv_reliability_orthogonal_nv_screen.py": M2_REVISION,
        "lightgcn_clv_m3_centered_value_graph.py": REFERENCE_REVISION,
        "clv_m5_m3_m4_user_centered_screen.py": REFERENCE_REVISION,
    }
    for name, revision in originals.items():
        frozen = subprocess.check_output(["git", "show", f"{revision}:{name}"], cwd=root)
        if (root / name).read_bytes() != frozen:
            raise RuntimeError(f"사전 구조를 바꾸지 않는 실험인데 구현이 변경됐습니다: {name}")
    return {name: _sha_file(root / name) for name in originals}


def prepare(cfg: Config | None = None) -> tuple[Config, dict]:
    cfg = configure() if cfg is None else validate_config(cfg)
    reference = load_reference(cfg)
    sources = _verify_source()
    prepared = m3._prepare(cfg)
    validate_prepared(prepared)
    graph_spec = next(s for s in m3.arm_specifications() if s["model_id"] == m3.ARM_VALUE_ACTIVITY)
    graph = m3.build_arm_graph(prepared, cfg, graph_spec)
    graph["sha256"] = hashlib.sha256(graph["weights"].astype(np.float32).tobytes()).hexdigest()
    weights, weight_audit = m4c.row_weights(prepared)
    old = reference["payload"]
    if (old["input_hash"] != prepared["input_hash"]
            or graph["sha256"] != old["m3_weight_sha256"]
            or not np.isclose(graph["beta"], old["m3_beta"], rtol=0, atol=1e-12)
            or weight_audit["sha256"] != old["m4_audit"]["sha256"]):
        raise RuntimeError("기존 C형과 새 결합의 입력·M3그래프·M4-C가 다릅니다. 학습 중단")
    data = prepared["data"]
    features = adaptive.build_features(
        data["train"], n_users=data["n_users"], n_items=data["n_items"],
        q_n=prepared["q_n"], q_v=prepared["q_v"], valid=prepared["clv_valid"],
        bandwidth=adaptive.original.BANDWIDTH,
        reliability_floor=adaptive.original.RELIABILITY_FLOOR,
    )
    if not np.array_equal(features["keys"], np.asarray(data["pos_key"], np.int64)):
        raise RuntimeError("M2 프로필과 M3 TRAIN pair 순서 불일치")
    initialization = adaptive.dataset_parameter_initialization(features)
    preflight = {
        "code_version": CODE_VERSION, "config": asdict(cfg), "seed": SEED,
        "split": SPLIT, "train_days": [1, 683], "development_days": [684, 690],
        "final_test": False, "holdout": False, "new_fit_count": 1,
        "new_models": [MODEL_ID], "reused_models": [M1, REFERENCE],
        "source_revision": prepared["revision"], "input_hash": prepared["input_hash"],
        "frozen_source_sha256": sources, "reference_audit": reference["audit"],
        "features_sha256": adaptive._feature_hash(features),
        "m2": "unchanged adaptive orthogonal N/V representation; alpha floor .15; rho [.03,.15]",
        "m2_shared_l2": adaptive.original.SHARED_L2,
        "dataset_initialization": initialization, "feature_diagnostics": features["diagnostics"],
        "m3": "fixed centered q_N/q_V graph, targetCV .20, gamma 1",
        "m3_beta": graph["beta"], "m3_weight_sha256": graph["sha256"], "m3_audit": graph["audit"],
        "m4_c": "unchanged customer-mass-preserving row weights, lambda .5", "m4_audit": weight_audit,
        "primary_reference": REFERENCE, "selection": "none; fixed epoch300",
        "same_forward_one_optimizer": True, "pretraining_or_freeze": False,
        "new_item_task": True, "min_item_interactions": 1, "negative": "uniform unseen K=1",
        "external_score_addition_or_reranking": False,
        "screen_rule": "at300 vs C: all six accuracy>=99%; both economic@10 strictly higher by>1e-8",
        "checkpoint_io": "epoch-local; Drive recovery every25 epochs",
        "limits": "one exposed development seed; no significance, final success or CLV attribution claim",
    }
    config_hash = hashlib.sha256(json.dumps(preflight, sort_keys=True).encode()).hexdigest()[:12]
    prepared.update(out_dir=Path(cfg.out_dir), config_hash=config_hash, reference=reference,
                    features=features, features_sha256=preflight["features_sha256"],
                    m3_graph=graph, row_weights=weights, preflight=preflight)
    prepared["base_cfg"].update(EPOCHS=300, SEED_LIST=[SEED])
    prepared.pop("m3_adj", None)
    io._atomic_json(Path(cfg.out_dir) / "preflight.json", preflight)
    print(json.dumps(preflight, ensure_ascii=False, indent=2), flush=True)
    print("준비 완료 — M1/C형 재학습·재평가 없음; 새 완전결합1개만 학습", flush=True)
    return cfg, prepared


def build_model(cfg: Config, prepared: dict):
    v3.set_seed(SEED)
    data, graph = prepared["data"], prepared["m3_graph"]
    adjacency = v3.build_adj(prepared["signals"]["edge_users"], prepared["signals"]["edge_items"],
                             graph["weights"].astype(np.float32), data["n_users"], data["n_items"])
    return adaptive.DatasetAdaptiveOrthogonalNVLightGCN(
        n_users=data["n_users"], n_items=data["n_items"], features=prepared["features"],
        adj=adjacency, id_dim=cfg.id_dim, n_layers=cfg.n_layers,
        pref_reg=cfg.pref_reg, shared_l2=adaptive.original.SHARED_L2,
    ).to(v3.DEVICE)


def _identity(prepared: dict) -> RunIdentity:
    return RunIdentity(CODE_VERSION, MODEL_ID, SEED, prepared["config_hash"],
                       prepared["revision"], prepared["input_hash"])


def _run_arm(cfg: Config, prepared: dict) -> dict:
    identity = _identity(prepared)
    path = prepared["out_dir"] / "arms" / prepared["config_hash"] / f"{MODEL_ID}_s{SEED}.json"
    if path.is_file():
        arm = json.loads(path.read_text())
        if (arm.get("identity") != asdict(identity) or arm.get("model_id") != MODEL_ID
                or arm.get("seed") != SEED or arm["curve"][-1]["epoch"] != 300
                or arm.get("m3_weight_sha256") != prepared["m3_graph"]["sha256"]
                or arm.get("m4_weight_sha256") != prepared["preflight"]["m4_audit"]["sha256"]):
            raise RuntimeError("완료 결합 캐시 신원 불일치")
        print("[재사용] 완전결합 seed48 300epoch 완료 — 추가학습·재평가 없음", flush=True)
        return arm
    model = build_model(cfg, prepared)
    local = Path("/content") if Path("/content").is_dir() else Path(tempfile.gettempdir())
    store = LocalFirstProgressStore(local / ".clv_m5_adaptive_nv_runtime" / prepared["config_hash"],
                                    prepared["out_dir"] / "progress" / prepared["config_hash"],
                                    identity, max_epoch=300)
    print("[학습1/1] 적응형 M2+M3+M4-C seed48 고정300epoch", flush=True)
    curve = capacity._train_curve(model, prepared, cfg,
        {"model_id": MODEL_ID, "condition": "adaptive_m2_fixed_m3_m4c"}, SEED,
        store, row_weights=prepared["row_weights"])
    if len(curve) != 300 or curve[-1]["epoch"] != 300:
        raise RuntimeError("고정300epoch 학습이 미완료입니다")
    arm = {"identity": asdict(identity), "model_id": MODEL_ID, "seed": SEED, "curve": curve,
           "m3_weight_sha256": prepared["m3_graph"]["sha256"],
           "m4_weight_sha256": prepared["preflight"]["m4_audit"]["sha256"],
           "final_representation_diagnostics": model.representation_diagnostics()}
    io._atomic_json(path, arm)
    store.mark_complete(epoch=300, max_epoch=300, selection="none", result_path=str(path))
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return arm


def report(cfg: Config, prepared: dict, candidate: dict) -> dict:
    metrics = prepared["reference"]["metrics"]
    records = [r for r in candidate["curve"] if "metrics" in r]
    if ([r["epoch"] for r in records] != list(range(25, 301, 25))
            or any(set(r["metrics"]) != set(metrics) for r in records)):
        raise RuntimeError("새 곡선의 평가점·전체 지표가 불완전합니다")
    new = pd.DataFrame({"model_id": MODEL_ID, "seed": SEED, "epoch": r["epoch"], **r["metrics"]}
                       for r in records)
    absolute = pd.concat([prepared["reference"]["absolute"], new], ignore_index=True)
    if (absolute.duplicated(["model_id", "seed", "epoch"]).any()
            or not np.isfinite(absolute[metrics].to_numpy(float)).all()):
        raise RuntimeError("전체 결과의 중복·비유한 지표")
    comparisons = []
    for epoch, group in absolute.groupby("epoch"):
        at = group.set_index("model_id")
        for reference in (REFERENCE, M1):
            for metric in metrics:
                value, base = float(at.at[MODEL_ID, metric]), float(at.at[reference, metric])
                comparisons.append({"model_id": MODEL_ID, "reference": reference, "seed": SEED,
                    "epoch": int(epoch), "metric": metric, "candidate_value": value,
                    "reference_value": base, "delta": value - base,
                    "relative_change_pct": 100 * (value / base - 1) if base else None})
    comparison = pd.DataFrame(comparisons)
    at = absolute[absolute.epoch.eq(300)].set_index("model_id")
    guards = {k: bool(at.at[MODEL_ID, k] >= .99 * at.at[REFERENCE, k]) for k in adaptive.ACCURACY}
    economic = {k: bool(at.at[MODEL_ID, k] > at.at[REFERENCE, k] + NUMERIC_TOLERANCE)
                for k in adaptive.ECONOMIC}
    reading = {"complete": True, "primary_reference": REFERENCE, "seed": SEED, "fixed_epoch": 300,
               "development_screen_only": True, "accuracy_guards_99pct_vs_c": guards,
               "economic_at10_strictly_above_c": economic,
               "exploratory_screen_rule_passed": all(guards.values()) and all(economic.values()),
               "significance_claim": False, "clv_attribution_claim": False, "final_success_claim": False}
    diagnostics = pd.DataFrame({"model_id": MODEL_ID, "seed": SEED, "epoch": r["epoch"],
                               "loss": r["loss"], "p_correct": r["p_correct"],
                               **r.get("score_split", {}),
                               **r.get("gradient_diagnostics", {})} for r in records)
    tables = {"absolute": absolute, "comparison": comparison, "diagnostics": diagnostics}
    root = prepared["out_dir"] / "reports"
    paths = {key: str(root / f"{key}.csv") for key in tables}
    for key, table in tables.items():
        io._atomic_csv(Path(paths[key]), table)
    paths["json"] = str(root / "result.json")
    paths["preflight"] = str(prepared["out_dir"] / "preflight.json")
    io._atomic_json(Path(paths["json"]), prepared["preflight"] | {
        "config_hash": prepared["config_hash"], "reading": reading,
        "arms": [candidate], "paths": paths,
        "reference_metrics_reused": True, "reference_checkpoint_used_for_initialization": False})
    return tables | {"reading": reading, "paths": paths}


def run(cfg: Config, prepared: dict) -> dict:
    validate_config(cfg)
    validate_prepared(prepared)
    current = load_reference(cfg)
    if (current["audit"] != prepared["reference"]["audit"]
            or prepared["preflight"]["config"] != asdict(cfg)
            or adaptive._feature_hash(prepared["features"]) != prepared["features_sha256"]
            or hashlib.sha256(prepared["row_weights"].astype(np.float32).tobytes()).hexdigest()
            != prepared["preflight"]["m4_audit"]["sha256"]
            or hashlib.sha256(prepared["m3_graph"]["weights"].astype(np.float32).tobytes()).hexdigest()
            != prepared["m3_graph"]["sha256"]):
        raise RuntimeError("준비 후 참조·설정·M2특징·M3·M4가 바뀌었습니다")
    return report(cfg, prepared, _run_arm(cfg, prepared))
