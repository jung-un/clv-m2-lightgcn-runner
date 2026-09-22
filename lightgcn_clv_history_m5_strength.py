"""Development-only personal-history M2 / fixed M4 strength factorial.

M4 mode must be chosen explicitly: complementary and original are different
published-in-project hypotheses, never interchangeable cached baselines.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from clv_history_item_fit_model import build_personal_history_weights
from clv_run_state import ProgressStore, RunIdentity, file_sha256
import lightgcn_clv_component_recheck as components
import lightgcn_clv_m2_capacity_search as capacity
import lightgcn_clv_m2_training_budget_hm2y as hm_budget
import lightgcn_clv_hm2y_seed42_common as hm_common
import lightgcn_clv_gatefree_lowdim as gatefree
import lightgcn_clv_m5_k1_m4_improvement_screen as weights_module
import lightgcn_clv_m5_nv_economic_positive_weight as economic
import lightgcn_clv_m4_k1_assignment_control_screen as assignment
import lightgcn_clv_v3 as v3

CODE_VERSION = "history-m2-fixed-m4-strength-development-v1"
ACCURACY = tuple(f"{m}@{k}" for k in (10, 20, 50) for m in ("recall", "ndcg"))
PRIMARY = "고CLV_revenue@10"  # legacy key: price/purchase-amount weighted hit


@dataclass(frozen=True)
class StrengthConfig:
    m4_mode: str = ""  # Explicit choice required; never silently change the M4.
    dataset: str = "dunnhumby"
    seeds: tuple[int, ...] = (42, 43, 44)
    rhos: tuple[float, ...] = (0.025, 0.05, 0.10)
    epochs: int = 100
    id_dim: int = 64
    history_axis_dim: int = 4
    n_layers: int = 2
    batch_size: int = 8192
    lr: float = 5e-4
    pref_reg: float = 1e-3
    negative_count: int = 1
    input_days: int = 365
    window_days: None = None
    positive_weight_lambda: float = 0.5
    basis_bandwidth: float = 0.25
    economic_bins: int = 4
    shrinkage_strength: float = 10.0
    out_dir: str = ""
    reuse_dirs: tuple[str, ...] = ()


def configure_strength(**overrides):
    dataset = overrides.get("dataset", "dunnhumby")
    defaults = {"out_dir": f"{v3.default_out_dir(dataset)}_history_m5_strength_v1"}
    if dataset == "hm":
        defaults["batch_size"] = hm_common.DEFAULT_BATCH_SIZE
    else:
        defaults["reuse_dirs"] = (
            f"{v3.default_out_dir(dataset)}_clv_component_recheck_dev5_v1",
            f"{v3.default_out_dir(dataset)}_clv_component_recheck_dev_v1",
            f"{v3.default_out_dir(dataset)}_clv_m2_capacity_search_v1",
            f"{v3.default_out_dir(dataset)}_m4_k1_assignment_control_development_multiseed_v1",
            *(f"{v3.default_out_dir(dataset)}_clv_m2_training_budget_seed{s}_v1" for s in (42, 43, 44)),
        )
    return validate_config(StrengthConfig(**(defaults | overrides)))


def validate_config(cfg):
    if cfg.m4_mode not in {"original", "complementary"}:
        raise ValueError("M4 식을 명시하세요: complementary(최근 결합) 또는 original(경제구간 적합도)")
    if cfg.dataset not in {"dunnhumby", "hm"}:
        raise ValueError("지원 데이터: dunnhumby, hm")
    fixed = {"epochs": 100, "negative_count": 1, "id_dim": 64,
             "history_axis_dim": 4, "n_layers": 2, "input_days": 365,
             "window_days": None, "positive_weight_lambda": 0.5,
             "basis_bandwidth": 0.25, "economic_bins": 4,
             "shrinkage_strength": 10.0, "pref_reg": 1e-3, "lr": 5e-4}
    for key, value in fixed.items():
        if getattr(cfg, key) != value:
            raise ValueError(f"이번 강도 진단의 고정 설정: {key}={value}")
    if (not cfg.rhos or len(set(cfg.rhos)) != len(cfg.rhos)
            or any(r not in (0.025, 0.05, 0.1) for r in cfg.rhos)):
        raise ValueError("rho 후보는 0.025/0.05/0.10의 중복 없는 부분집합입니다")
    if not cfg.seeds or len(set(cfg.seeds)) != len(cfg.seeds) or any(s not in (42, 43, 44) for s in cfg.seeds):
        raise ValueError("seed는 42/43/44의 중복 없는 부분집합입니다")
    if not cfg.out_dir or cfg.batch_size <= 0:
        raise ValueError("결과 경로와 양수 batch_size가 필요합니다")
    if cfg.dataset == "dunnhumby" and cfg.batch_size != 8192:
        raise ValueError("Dunnhumby matched batch_size=8192")
    if cfg.dataset == "hm" and cfg.batch_size not in hm_common.BATCH_CANDIDATES:
        raise ValueError("H&M batch_size는 131072/65536/32768")
    return cfg


def arm_specifications(cfg):
    arms = [dict(model_id="m1", role="M1", kind="lightgcn", graph="binary", weighted=False, rho=0.0),
            dict(model_id="m4", role="M4", kind="lightgcn", graph="binary", weighted=True, rho=0.0)]
    for rho in cfg.rhos:
        for role in ("M2", "M5"):
            arms.append(dict(model_id=f"{role.lower()}_rho{rho:g}", role=role,
                             kind="history_fit", graph="binary", weighted=role == "M5", rho=rho))
    return arms


def preflight_summary(cfg):
    validate_config(cfg)
    return {"code_version": CODE_VERSION, "config": asdict(cfg),
            "question": "Does M2 strength improve M5 over fixed M4, within a 1% M1 accuracy guard?",
            "m2": "personal-history q_N/q_V, leave-one-out training, rho linear in score; no q_C in M2",
            "m4": ("1+0.5*q_C*(1-RBF_value_fit)" if cfg.m4_mode == "complementary"
                   else "1+0.5*q_C*item_amount_percentile*clipped_user_bin_fit"),
            "weight_normalization": "mean raw weight across all train rows",
            "split": split_name(cfg), "final_test": False, "holdout": False,
            "new_item_task": True, "min_item_interactions": 1,
            "graph": "binary", "negative": "K=1 uniform unseen", "external_reranking": False,
            "primary": "고CLV price/purchase-amount weighted hit@10: M5 > M4",
            "guard": "each of six overall Recall/NDCG >= 99% of matched M1",
            "reading": "seed-wise and mean separately; exploratory only, no significance/CLV attribution",
            "selection": "100 epochs fixed; candidate shortlist only, no final-test selection",
            "arms_per_seed_before_reuse": len(arm_specifications(cfg))}


def split_name(cfg):
    return ("historical_development_days_684_690" if cfg.dataset == "dunnhumby"
            else "hm2y_validation_2020-09-02_08")


def _prepare(cfg):
    if cfg.dataset == "dunnhumby":
        prep_cfg = weights_module.configure_improvement_screen(out_dir=cfg.out_dir)
        with patch.object(gatefree, "_load_compatible_baseline", return_value=None):
            prepared = weights_module._prepare(prep_cfg)
    else:
        prepared = hm_common.prepare_hm2y(cfg, code_version=CODE_VERSION)
        data = prepared["data"]
        prepared.update(economic.build_nv_economic_inputs(
            data["train"], n_users=data["n_users"], n_items=data["n_items"],
            q_n=prepared["q_n"], q_v=prepared["q_v"], q_c=prepared["q_c"],
            clv_valid=prepared["clv_valid"], n_bins=cfg.economic_bins,
            shrinkage_strength=cfg.shrinkage_strength))
    data = prepared["data"]
    if prepared["base_cfg"].get("EVAL_HOLDOUT") or prepared["base_cfg"].get("MIN_ITEM_INTER") != 1:
        raise RuntimeError("보호 split 또는 전체 카탈로그 설정 위반")
    expected = {"test"} if cfg.dataset == "dunnhumby" else {"val"}
    if set(data["splits"]) != expected:
        raise RuntimeError("개발분할 외 평가 차단")
    # 'test' is a legacy name for the already-exposed 684..690 development split.
    if cfg.dataset == "dunnhumby" and float(data["train"].t.max()) != 683.0:
        raise RuntimeError("Dunnhumby train 종료일 불일치")
    prepared["history"] = build_personal_history_weights(data["train"], n_users=data["n_users"], n_items=data["n_items"])
    prepared["m4_weights"], prepared["m4_diagnostics"] = weights_module.row_weights(prepared, cfg, cfg.m4_mode)
    history = prepared["history"]
    valid = np.asarray(prepared["clv_valid"], bool)
    prepared["data_diagnostics"] = {
        "user_unique_item_degree_median": float(np.median(np.bincount(history.users, minlength=data["n_users"]))),
        "item_unique_user_degree_median": float(np.median(np.bincount(history.items, minlength=data["n_items"]))),
        "clv_valid_user_share": float(valid.mean()),
        "q_n_unique_valid": int(np.unique(prepared["q_n"][valid]).size),
        "q_v_unique_valid": int(np.unique(prepared["q_v"][valid]).size),
        **history.diagnostics,
    }
    capacity.test10._atomic_json(Path(cfg.out_dir) / f"data_diagnostics_{cfg.dataset}_{cfg.m4_mode}.json",
                               prepared["data_diagnostics"])
    prepared["out_dir"] = Path(cfg.out_dir)
    return prepared


def _identity(prepared, cfg, spec, seed):
    protocol = asdict(cfg)
    for key in ("out_dir", "reuse_dirs", "seeds", "rhos"):
        protocol.pop(key)
    # Hash source content: unrelated commits don't invalidate valid resume files.
    names = (Path(__file__).name, "clv_history_item_fit_model.py", "clv_run_state.py",
             "lightgcn_clv_component_recheck.py", "lightgcn_clv_m5_k1_m4_improvement_screen.py",
             "lightgcn_clv_v3.py", "clv_m5_n_conditioned_value_basis_model.py",
             "lightgcn_clv_m5_nv_economic_positive_weight.py", "lightgcn_clv_hm2y_seed42_common.py",
             "lightgcn_clv_m2_capacity_search.py", "lightgcn_clv_m2_training_budget_hm2y.py",
             "lightgcn_clv_m4_clv_hard_negative.py", "lightgcn_clv_moe.py")
    return {"version": CODE_VERSION, "protocol": protocol, "spec": spec, "seed": seed,
            "input_hash": prepared["input_hash"], "split": split_name(cfg),
            "source_hashes": {n: file_sha256(Path(__file__).with_name(n)) for n in names}}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _build_model(prepared, cfg, spec, seed):
    # Same initialization and leave-one-out path as the completed component recheck.
    model_cfg = SimpleNamespace(**asdict(cfg), history_rho=spec["rho"])
    return components._build_model(prepared, model_cfg, spec, seed)


def _reuse_component(prepared, cfg, spec, seed):
    """Only reuse fully identified completed component checkpoints, not rounded CSVs."""
    if cfg.dataset != "dunnhumby" or (spec["weighted"] and cfg.m4_mode != "complementary"):
        return None
    if spec["kind"] == "history_fit" and spec["rho"] != 0.05:
        return None
    ids = {"M1": components.M1_MODEL_ID, "M2": components.M2_MODEL_ID,
           "M4": components.M4_MODEL_ID, "M5": components.M5_MODEL_ID}
    old_id = ids[spec["role"]]
    expected = asdict(components.configure_component_recheck())
    for field in ("batch_size", "lr", "pref_reg", "epochs", "id_dim", "n_layers", "negative_count"):
        expected[field] = getattr(cfg, field)
    matches = []
    for root in cfg.reuse_dirs:
        for path in sorted(Path(root).glob(f"arms/*/{old_id}_s{seed}.json")):
            payload = json.loads(path.read_text())
            checkpoint = Path(payload.get("checkpoint", ""))
            if not checkpoint.is_file():
                checkpoint = path.with_suffix(".pt")
            if not checkpoint.is_file() or file_sha256(checkpoint) != payload.get("checkpoint_sha256"):
                continue
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            stored = state.get("config", {})
            if (state.get("input_hash") != prepared["input_hash"] or state.get("seed") != seed
                    or state.get("model_id") != old_id or payload.get("final_epoch") != cfg.epochs
                    or payload.get("split") != split_name(cfg)
                    or payload.get("code_version") != components.CODE_VERSION):
                continue
            if any(stored.get(k) != v for k, v in expected.items() if k not in {"out_dir", "seeds"}):
                continue
            matches.append((path, payload))
    if not matches:
        return None
    if len({p[1]["checkpoint_sha256"] for p in matches}) > 1:
        raise RuntimeError(f"서로 다른 호환 checkpoint가 여러 개입니다: {old_id}, seed {seed}")
    path, payload = matches[0]
    return {**spec, "seed": seed, "metrics": payload["metrics"],
            "origin": "reused_component", "source_result": str(path),
            "diagnostics": payload.get("diagnostics", {}),
            "training": payload.get("training", {})}


def _reuse_capacity(prepared, cfg, spec, seed):
    """Reuse the fixed 100-epoch record, never the best point of a 300-epoch curve."""
    if cfg.dataset != "dunnhumby" or spec["weighted"]:
        return None
    if spec["role"] == "M2" and spec["rho"] != 0.05:
        return None
    old_id = capacity.M1_MODEL_ID if spec["role"] == "M1" else capacity.M2_MODEL_ID
    candidates = []
    for root in cfg.reuse_dirs:
        for summary_path in sorted(Path(root).glob("clv_m2_capacity_search_*.json")):
            summary = json.loads(summary_path.read_text())
            stored = summary.get("config", {})
            if summary.get("code_version") != capacity.CODE_VERSION:
                continue
            if any(stored.get(k) != getattr(cfg, k) for k in ("batch_size", "lr", "n_layers", "negative_count")):
                continue
            if seed not in stored.get("seeds", []) or "baseline" not in stored.get("conditions", []):
                continue
            old_cfg = capacity.CapacitySearchConfig(**stored)
            old_hash = capacity._config_hash(old_cfg, prepared["input_hash"])
            if summary_path.stem != f"clv_m2_capacity_search_{old_hash}":
                continue
            arm_path = Path(root) / "arms" / old_hash / f"baseline_{old_id}_s{seed}.json"
            if not arm_path.exists():
                continue
            arm = json.loads(arm_path.read_text())
            if (arm.get("seed") != seed or arm.get("model_id") != old_id
                    or arm.get("id_dim") != cfg.id_dim or arm.get("pref_reg") != cfg.pref_reg
                    or arm.get("rho") != spec["rho"]):
                continue
            records = [r for r in arm["curve"] if r["epoch"] == cfg.epochs and "metrics" in r]
            if len(records) != 1:
                continue
            record = records[0]
            candidates.append({**spec, "seed": seed, "metrics": record["metrics"],
                "origin": "reused_capacity_epoch100", "source_result": str(arm_path),
                "diagnostics": {**record.get("score_split", {}), **record.get("gradient_diagnostics", {})},
                "training": {"epochs_run": 100, "source_total_epochs": stored["epochs"]}})
    if len({_digest(c["metrics"]) for c in candidates}) > 1:
        raise RuntimeError(f"서로 다른 100 epoch 결과: {old_id} seed {seed}")
    return candidates[0] if candidates else None


def _reuse_assignment(prepared, cfg, spec, seed):
    if cfg.dataset != "dunnhumby" or spec["role"] not in {"M1", "M4"}:
        return None
    if spec["role"] == "M4" and cfg.m4_mode != "original":
        return None
    old_id = assignment.M1_MODEL_ID if spec["role"] == "M1" else assignment.M4_ACTUAL_MODEL_ID
    expected = asdict(weights_module.configure_improvement_screen())
    expected.update(seed=seed, batch_size=cfg.batch_size, lr=cfg.lr, pref_reg=cfg.pref_reg)
    found = []
    for root in cfg.reuse_dirs:
        for path in sorted(Path(root).glob(f"arms/*/{old_id}_s{seed}.json")):
            payload = json.loads(path.read_text())
            cp = path.with_suffix(".pt")
            if not cp.is_file() or file_sha256(cp) != payload.get("checkpoint_sha256"):
                continue
            checkpoint = torch.load(cp, map_location="cpu", weights_only=False)
            if checkpoint.get("input_hash") != prepared["input_hash"] or checkpoint.get("model_id") != old_id:
                continue
            ignored = {"out_dir", "baseline_result_dir", "m1_reference_json", "m4_reference_json"}
            if any(checkpoint.get("config", {}).get(k) != v for k, v in expected.items() if k not in ignored):
                continue
            if (payload.get("seed") != seed or payload.get("final_epoch") != 100
                    or payload.get("split") != split_name(cfg) or payload.get("rho") != 0.0
                    or payload.get("improvement") != ("original" if spec["weighted"] else None)):
                continue
            found.append({**spec, "seed": seed, "metrics": payload["metrics"],
                          "origin": "reused_assignment", "source_result": str(path),
                          "diagnostics": payload.get("diagnostics", {}),
                          "training": payload.get("training", {})})
    if len({_digest(p["metrics"]) for p in found}) > 1:
        raise RuntimeError(f"서로 다른 M4 대조군 결과: {old_id} seed {seed}")
    return found[0] if found else None


def _run_arm(prepared, cfg, spec, seed):
    identity = _identity(prepared, cfg, spec, seed)
    key = _digest(identity)
    root = Path(cfg.out_dir) / "arms" / key
    result_path = root / "result.json"
    if result_path.exists():
        payload = json.loads(result_path.read_text())
        if payload["identity"] != identity:
            raise RuntimeError("완료 캐시의 실험 신분 불일치")
        print(f"[cached] seed {seed} {spec['model_id']}", flush=True)
        return payload
    payload = (_reuse_component(prepared, cfg, spec, seed)
               or _reuse_capacity(prepared, cfg, spec, seed)
               or _reuse_assignment(prepared, cfg, spec, seed))
    if payload is None:
        model = _build_model(prepared, cfg, spec, seed)
        store = ProgressStore(root / "progress", RunIdentity(
            stage=CODE_VERSION, model_id=spec["model_id"], seed=seed, config_hash=key,
            source_revision="content:" + _digest(identity["source_hashes"]), input_hash=prepared["input_hash"]))
        training = components._train_arm(model, prepared, cfg, spec, seed, store)
        metrics = capacity._evaluate(model, prepared)
        payload = {**spec, "seed": seed, "metrics": metrics, "origin": "trained",
                   "diagnostics": {**model.representation_diagnostics(),
                                   **model.training_gradient_diagnostics(),
                                   **hm_budget._score_share(model, prepared)},
                   "training": training}
        store.mark_complete(epoch=cfg.epochs, max_epoch=cfg.epochs, selection="none",
                            result_path=str(result_path), checkpoint_path="")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        print(f"[reused] seed {seed} {spec['model_id']}: {payload['source_result']}", flush=True)
    payload.update(identity=identity, data_diagnostics=prepared.get("data_diagnostics", {}),
                   m4_diagnostics=prepared["m4_diagnostics"] if spec["weighted"] else {})
    capacity.test10._atomic_json(result_path, payload)
    return payload


def result_tables(arms, expected_seeds):
    absolute = pd.DataFrame([{**{k: a[k] for k in ("model_id", "role", "rho", "seed", "origin")},
                              **a["metrics"]} for a in arms])
    metrics = sorted(set().union(*(a["metrics"].keys() for a in arms)))
    comparisons = []
    for seed, part in absolute.groupby("seed"):
        refs = part.set_index("model_id")
        for _, row in part.iterrows():
            references = ["m1"] if row.role != "M1" else []
            if row.role == "M5":
                references += ["m4", f"m2_rho{row.rho:g}"]
            for ref in references:
                if ref not in refs.index:
                    continue
                for metric in metrics:
                    base, value = refs.loc[ref, metric], row[metric]
                    if not isinstance(base, (int, float, np.number)) or not isinstance(value, (int, float, np.number)):
                        continue
                    comparisons.append(dict(seed=seed, model_id=row.model_id, reference=ref,
                                            rho=row.rho, metric=metric, reference_value=base, value=value,
                                            delta=value-base,
                                            relative_change_pct=100*(value/base-1) if base else np.nan))
    comparison = pd.DataFrame(comparisons)
    if comparison.empty:
        return absolute, comparison, pd.DataFrame(), pd.DataFrame()
    summary = comparison.groupby(["model_id", "reference", "rho", "metric"], as_index=False).agg(
        seeds=("seed", "nunique"), mean_reference=("reference_value", "mean"),
        mean_value=("value", "mean"), mean_delta=("delta", "mean"),
        positive_seeds=("delta", lambda x: int((x > 0).sum())), sd_delta=("delta", "std"))
    summary["relative_change_pct"] = 100*(summary.mean_value/summary.mean_reference.replace(0, np.nan)-1)
    decisions = []
    for model_id in absolute.loc[absolute.role.eq("M5"), "model_id"].unique():
        frame = comparison[comparison.model_id.eq(model_id)]
        for label, part in list(frame.groupby("seed")) + [("mean", frame)]:
            guard = part[part.reference.eq("m1") & part.metric.isin(ACCURACY)].groupby("metric")[["reference_value", "value"]].mean()
            primary = part[part.reference.eq("m4") & part.metric.eq(PRIMARY)]
            complete = set(guard.index) == set(ACCURACY) and not primary.empty
            if label == "mean":
                complete = complete and set(part.seed) == set(expected_seeds)
                complete = complete and all(
                    set(part[(part.reference == ref) & (part.metric == metric)].seed) == set(expected_seeds)
                    for ref, metric in [("m1", m) for m in ACCURACY] + [("m4", PRIMARY)])
            accuracy_ok = bool(complete and (guard.value >= .99*guard.reference_value).all())
            economic_ok = bool(complete and primary.delta.mean() > 0)
            decisions.append(dict(model_id=model_id, seed=label, complete=complete,
                                  accuracy_guard=accuracy_ok, high_clv_gain=economic_ok,
                                  directional_pass=bool(accuracy_ok and economic_ok),
                                  significance_claim=False))
    return absolute, comparison, summary, pd.DataFrame(decisions)


def save_report(arms, cfg):
    tables = result_tables(arms, cfg.seeds)
    report_id = _digest({"config": asdict(cfg), "version": CODE_VERSION})
    root = Path(cfg.out_dir) / "reports" / report_id
    paths = {}
    for name, table in zip(("absolute", "comparison", "summary", "reading"), tables):
        path = root / f"{name}.csv"
        capacity.test10._atomic_csv(path, table)
        paths[name] = str(path)
    capacity.test10._atomic_json(root / "result.json", {
        "preflight": preflight_summary(cfg), "arms": arms,
        "reading": tables[-1].to_dict("records"), "paths": paths})
    tables[0].attrs.update(paths=paths, comparison_records=tables[1].to_dict("records"),
                          summary_records=tables[2].to_dict("records"), reading_records=tables[3].to_dict("records"))
    return tables[0]


def run_strength(cfg):
    validate_config(cfg)
    print(json.dumps(preflight_summary(cfg), ensure_ascii=False, indent=2), flush=True)
    prepared = _prepare(cfg)
    arms = []
    for seed in cfg.seeds:
        for spec in arm_specifications(cfg):
            print(f"=== {cfg.dataset} seed {seed} {spec['model_id']} M4={cfg.m4_mode} ===", flush=True)
            arms.append(_run_arm(prepared, cfg, spec, seed))
            save_report(arms, cfg)  # Read partial results while later arms train.
    return save_report(arms, cfg)
