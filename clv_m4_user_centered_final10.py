"""Complete the frozen Dunnhumby 2x2 comparison with ten M4-C fits only.

M1/M3/M5-C are read from the completed last-week run, never retrained or
reevaluated. Development-week M4-C results cannot enter this comparison.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import torch

import clv_m5_user_centered_final10 as prior
from clv_run_state import ProgressStore, RunIdentity


CODE_VERSION = "clv-m4-user-centered-final10-v1"
REFERENCE_REVISION = "58ff14e7017b67d09ade8be140ffb2d46321fa8a"
M4_C = "m4_user_centered_bpr_k1_lastweek_final10"
REFERENCES = (prior.M1, prior.M3, prior.M5_C)
MODELS = (*REFERENCES, M4_C)
# These existing implementation paths are frozen to the completed reference run.
FROZEN_FILES = (
    "clv_m5_user_centered_final10.py", "clv_m5_m3_m4_user_centered_screen.py",
    "lightgcn_clv_m3_dunnhumby_test1.py", "lightgcn_clv_v3.py",
    "lightgcn_clv_m5_nv_economic_positive_weight.py",
    "clv_m5_n_conditioned_value_basis_model.py", "clv_run_state.py",
    "lightgcn_clv_m3_centered_value_graph.py", "lightgcn_clv_joint_nv.py",
    "lightgcn_clv_residual.py", "lightgcn_clv_component_recheck.py",
    "lightgcn_clv_m4_clv_hard_negative.py", "lightgcn_clv_moe.py",
    "lightgcn_clv_gradient_isolated_economic_interaction.py",
    "lightgcn_clv_axis_specific_test10.py",
)


@dataclass(frozen=True)
class Config:
    reference_dir: str
    out_dir: str


def configure(**overrides) -> Config:
    root = prior.v3.default_out_dir("dunnhumby")
    cfg = Config(**({"reference_dir": root + "_m5c_lastweek_final10_v1",
                     "out_dir": root + "_m4c_lastweek_final10_v1"} | overrides))
    if not cfg.reference_dir or not cfg.out_dir:
        raise ValueError("완료 참조 결과 경로와 새 결과 경로가 필요합니다")
    if Path(cfg.reference_dir).resolve() == Path(cfg.out_dir).resolve():
        raise ValueError("기존 M5 결과와 새 M4 결과는 별도 경로에 저장합니다")
    return cfg


def load_references(cfg: Config) -> dict:
    matches = sorted(Path(cfg.reference_dir).glob("*/reports/result.json"))
    eligible = []
    for path in matches:
        raw = json.loads(path.read_text())
        protocol = raw.get("protocol", {})
        if protocol.get("code_version") != prior.CODE_VERSION:
            continue
        expected = asdict(prior.configure())
        actual = protocol.get("config", {})
        if (protocol.get("source_revision") != REFERENCE_REVISION
                or protocol.get("split") != "train DAY 1-704; test DAY 705-711"
                or any(actual.get(k) != json.loads(json.dumps(v))
                       for k, v in expected.items() if k != "out_dir")
                or protocol.get("fixed") != prior.preflight_summary(prior.configure())["fixed"]
                or raw.get("reading", {}).get("complete") is not True):
            raise RuntimeError(f"완료 참조의 분할·학습설정·source 신원이 다릅니다: {path}")
        absolute_path, diag_path = path.parent / "absolute.csv", path.parent / "diagnostics.csv"
        absolute = pd.read_csv(absolute_path)
        prior._validate_absolute(absolute, prior.configure())
        # Include non-@ scalar evaluation metrics, including price alignment.
        metrics = [c for c in absolute if c not in ("model_id", "seed")]
        if not np.isfinite(absolute[metrics].to_numpy(float)).all():
            raise RuntimeError("참조 원본의 전체 지표에 비유한 값이 있습니다")
        diagnostics = pd.read_csv(diag_path)
        if (set(zip(diagnostics.model_id, diagnostics.seed))
                != set(zip(absolute.model_id, absolute.seed))
                or diagnostics.duplicated(["model_id", "seed"]).any()):
            raise RuntimeError("참조 진단표의 모형·시드가 absolute와 다릅니다")
        paths = (path, absolute_path, diag_path)
        eligible.append({
            "protocol": protocol, "metrics": metrics,
            "absolute": absolute[absolute.model_id.isin(REFERENCES)].copy(),
            "diagnostics": diagnostics[diagnostics.model_id.isin(REFERENCES)].copy(),
            "audit": {"source_revision": protocol["source_revision"],
                      "files": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}},
        })
    if len(eligible) != 1:
        raise RuntimeError(f"동일 마지막 주 10시드 완료 참조가 정확히 하나 필요합니다: {cfg.reference_dir}")
    return eligible[0]


def _verify_frozen_code() -> dict:
    root = Path(__file__).resolve().parent
    hashes = {}
    for name in FROZEN_FILES:
        frozen = subprocess.check_output(["git", "show", f"{REFERENCE_REVISION}:{name}"], cwd=root)
        current = (root / name).read_bytes()
        if current != frozen:
            raise RuntimeError(f"기존 결과와 비교할 공통 구현이 변경됐습니다: {name}")
        hashes[name] = hashlib.sha256(current).hexdigest()
    return hashes


def prepare(cfg: Config) -> dict:
    cfg = configure(**asdict(cfg))
    refs = load_references(cfg)  # Stop before data preparation if references are absent.
    code_hashes = _verify_frozen_code()
    training_cfg = prior.configure(out_dir=cfg.out_dir)
    prepared = prior.prepare(training_cfg)
    old = refs["protocol"]
    if (old["input_hash"] != prepared["input_hash"]
            or old["data_stats"] != prepared["protocol"]["data_stats"]
            or old["m4_c_weight_audit"]["sha256"] != prepared["observed_weight_audit"]["sha256"]):
        raise RuntimeError("기존 최종 결과와 이번 입력·분할·M4-C 가중치가 다릅니다")
    protocol = {
        "code_version": CODE_VERSION, "config": asdict(training_cfg),
        "split": old["split"], "input_hash": prepared["input_hash"],
        "source_revision": prepared["revision"], "fixed": old["fixed"],
        "reference_audit": refs["audit"], "frozen_code_sha256": code_hashes,
        "m4_c": old["m4_c"], "m4_c_weight_audit": prepared["observed_weight_audit"],
        "data_stats": prepared["protocol"]["data_stats"],
        "question": "M4-C independent effect and (M5-C-M3)-(M4-C-M1)",
        "new_fit_models": [M4_C], "reused_models": list(REFERENCES),
        "planned_seeds": list(prior.SEEDS), "fixed_epoch": 300,
        "decision": "descriptive 2x2; M4-C standalone victory is not required for M5-C",
        "significance_claim": False, "clv_attribution_claim": False,
        "unseen_confirmation_claim": False,
    }
    config_hash = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()[:12]
    run_dir = Path(cfg.out_dir) / config_hash
    prior.io._atomic_json(run_dir / "protocol.json", protocol)
    prepared.update(config_hash=config_hash, run_dir=run_dir, protocol=protocol,
                    references=refs, training_cfg=training_cfg)
    # M3 is a reused result here; its extra sparse GPU adjacency is unnecessary.
    del prepared["observed_adj"]
    return prepared


def _identity(prepared: dict, seed: int) -> RunIdentity:
    return RunIdentity(CODE_VERSION, M4_C, seed, prepared["config_hash"],
                       prepared["revision"], prepared["input_hash"])


def _completed(prepared: dict, seed: int) -> dict | None:
    path = prepared["run_dir"] / "arms" / f"{M4_C}_s{seed}.json"
    if not path.exists():
        return None
    row = json.loads(path.read_text())
    if (row.get("identity") != asdict(_identity(prepared, seed))
            or row.get("model_id") != M4_C or row.get("seed") != seed
            or row.get("epochs") != 300
            or set(row.get("metrics", {})) != set(prepared["references"]["metrics"])
            or not np.isfinite(list(row["metrics"].values())).all()
            or row.get("diagnostics", {}).get("graph") != "binary"
            or row["diagnostics"].get("row_weighted") is not True
            or row["diagnostics"].get("weight_audit", {}).get("sha256")
            != prepared["observed_weight_audit"]["sha256"]):
        raise RuntimeError(f"완료 M4-C 결과 신원이 다릅니다: {path}")
    return row


def execution_plan(prepared: dict) -> pd.DataFrame:
    rows = []
    for seed in prior.SEEDS:
        if _completed(prepared, seed) is not None:
            action, epoch = "reuse_completed", 300
        else:
            identity = _identity(prepared, seed)
            safe = f"{identity.stage}_{identity.model_id}_s{seed}"
            progress = prepared["run_dir"] / "progress"
            checkpoint = progress / "resume" / f"{safe}_latest.pt"
            stage = progress / "stages" / f"{safe}.json"
            epoch = 0
            if checkpoint.is_file() and stage.is_file():
                state = json.loads(stage.read_text())
                if any(state.get(k) != v for k, v in asdict(identity).items()):
                    raise RuntimeError("M4-C 재개 신원이 다릅니다")
                epoch = int(state.get("epoch", 0))
                if not 0 <= epoch <= 300:
                    raise RuntimeError("재개 checkpoint epoch 범위가 잘못됐습니다")
            action = "resume_checkpoint" if checkpoint.is_file() else "train_missing"
        rows.append({"seed": seed, "model_id": M4_C, "action": action,
                     "saved_epoch": epoch, "max_remaining_epochs": 300 - epoch})
    return pd.DataFrame(rows)


def run_arm(prepared: dict, seed: int) -> dict:
    completed = _completed(prepared, seed)
    if completed is not None:
        print(f"[reused] M4-C seed{seed} — 추가학습 없음", flush=True)
        return completed
    cfg = prepared["training_cfg"]
    arm = {"prepared": prepared, "adjacency": prepared["data"]["adj"]}
    model = prior._build_model(arm, cfg, seed)
    store = ProgressStore(prepared["run_dir"] / "progress", _identity(prepared, seed))
    history = prior.fixed._train(model, prepared, cfg, M4_C, seed, store,
                                 row_weights=prepared["observed_weights"])
    if len(history) != 300 or history[-1]["epoch"] != 300:
        raise RuntimeError("300epoch 미완료: test를 평가하지 않습니다")
    metrics = prior.fixed._evaluate(model, prepared)
    if (set(metrics) != set(prepared["references"]["metrics"])
            or not np.isfinite(list(metrics.values())).all()):
        raise RuntimeError("최종 전체 지표의 누락·비유한 값")
    row = {
        "identity": asdict(_identity(prepared, seed)), "model_id": M4_C,
        "seed": seed, "epochs": 300, "metrics": metrics, "training_history": history,
        "diagnostics": {"graph": "binary", "row_weighted": True,
                        "weight_audit": prepared["observed_weight_audit"]},
    }
    path = prepared["run_dir"] / "arms" / f"{M4_C}_s{seed}.json"
    prior.io._atomic_json(path, row)
    store.mark_complete(epoch=300, max_epoch=300, selection="none",
                        checkpoint_path=str(store.latest_checkpoint), result_path=str(path))
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return row


def report(prepared: dict, rows: list[dict]) -> dict:
    refs, metrics = prepared["references"], prepared["references"]["metrics"]
    new = pd.DataFrame({"model_id": r["model_id"], "seed": r["seed"], **r["metrics"]} for r in rows)
    absolute = pd.concat([refs["absolute"], new], ignore_index=True)
    expected = {(model, seed) for model in MODELS for seed in prior.SEEDS}
    if (len(absolute) != 40 or set(zip(absolute.model_id, absolute.seed)) != expected
            or absolute.duplicated(["model_id", "seed"]).any()
            or not np.isfinite(absolute[metrics].to_numpy(float)).all()):
        raise RuntimeError("M1/M3/M4-C/M5-C 각10시드 전체 결과가 필요합니다")
    long = absolute.melt(id_vars=["model_id", "seed"], value_vars=metrics,
                         var_name="metric", value_name="value")
    summary = long.groupby(["model_id", "metric"]).value.agg(["mean", "std", "count"]).reset_index()
    index = absolute.set_index(["model_id", "seed"])
    comparisons, interactions = [], []
    for seed in prior.SEEDS:
        for metric in metrics:
            for model, reference in ((M4_C, prior.M1), (prior.M5_C, M4_C),
                                     (prior.M3, prior.M1), (prior.M5_C, prior.M3)):
                value, base = float(index.at[(model, seed), metric]), float(index.at[(reference, seed), metric])
                comparisons.append({"model_id": model, "reference": reference, "seed": seed,
                                    "metric": metric, "value": value, "reference_value": base,
                                    "delta": value - base, "ratio": value / base if base else np.nan})
            standalone = index.at[(M4_C, seed), metric] - index.at[(prior.M1, seed), metric]
            combined = index.at[(prior.M5_C, seed), metric] - index.at[(prior.M3, seed), metric]
            interactions.append({"seed": seed, "metric": metric, "m4_minus_m1": standalone,
                                 "m5_minus_m3": combined, "interaction": combined - standalone})
    diagnostics = pd.concat([refs["diagnostics"], pd.DataFrame(
        {"model_id": r["model_id"], "seed": r["seed"], **r["diagnostics"]} for r in rows
    )], ignore_index=True)
    reading = {"complete": True, "seed_count": 10, "fixed_epoch": 300,
               "reused_reference_arm_count": 30, "m4_arm_count": 10,
               "m4_standalone_victory_required_for_m5": False,
               "significance_claim": False, "clv_attribution_claim": False,
               "note": "descriptive independent/combined effects; no test-based retuning"}
    tables = {"absolute": absolute, "summary": summary, "comparison": pd.DataFrame(comparisons),
              "factorial": pd.DataFrame(interactions), "diagnostics": diagnostics}
    root = prepared["run_dir"] / "reports"
    paths = {k: str(root / f"{k}.csv") for k in tables}
    for key, table in tables.items():
        prior.io._atomic_csv(Path(paths[key]), table)
    paths["json"] = str(root / "result.json")
    prior.io._atomic_json(Path(paths["json"]), {"protocol": prepared["protocol"],
                                             "reading": reading, "result_paths": paths})
    return {**tables, "reading": reading, "paths": paths}


def run(prepared: dict) -> dict:
    print("기존 마지막주 M1/M3/M5-C 30개 완료 결과 재사용 — 추가학습·재평가 없음", flush=True)
    rows = []
    for seed in prior.SEEDS:
        print(f"[{len(rows) + 1}/10] M4-C seed{seed}", flush=True)
        rows.append(run_arm(prepared, seed))
    return report(prepared, rows)
