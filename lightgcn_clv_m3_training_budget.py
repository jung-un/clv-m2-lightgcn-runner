"""300-epoch diagnostic for the frozen candidate-specific N/V M3."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from clv_candidate_nv_fit_model import CandidateNVFitLightGCN
from clv_run_state import ProgressStore, RunIdentity
import lightgcn_clv_candidate_nv_fit_factorial as candidate
import lightgcn_clv_m2_capacity_search as curves
import lightgcn_clv_v3 as v3

CODE_VERSION = "clv-m3-candidate-nv-training-budget-dev-v1"
MODEL_ID = "m3_candidate_nv_edge_weight_bpr_k1_e300"


@dataclass(frozen=True)
class M3BudgetConfig:
    reference_curve_csv: str
    out_dir: str
    seed: int = 42
    epochs: int = 300
    eval_every: int = 25
    id_dim: int = 64
    pref_reg: float = 1e-3
    beta_m3: float = 0.15
    batch_size: int = 8192
    lr: float = 5e-4
    n_layers: int = 2
    negative_count: int = 1


def configure_m3_budget(**overrides) -> M3BudgetConfig:
    root = v3.default_out_dir("dunnhumby")
    defaults = {
        "reference_curve_csv": f"{root}_clv_m2_capacity_search_v1/clv_m2_capacity_search_5f4b9c6e45d2_curve.csv",
        "out_dir": f"{root}_clv_m3_candidate_nv_training_budget_v1",
    }
    return validate_config(M3BudgetConfig(**(defaults | overrides)))


def validate_config(cfg: M3BudgetConfig) -> M3BudgetConfig:
    fixed = {"seed": 42, "epochs": 300, "eval_every": 25, "id_dim": 64,
             "pref_reg": 1e-3, "beta_m3": 0.15, "n_layers": 2,
             "negative_count": 1}
    for key, expected in fixed.items():
        if getattr(cfg, key) != expected:
            raise ValueError(f"M3 학습예산 진단은 {key}={expected!r}이어야 합니다")
    if not cfg.reference_curve_csv or not cfg.out_dir or cfg.batch_size <= 0 or cfg.lr <= 0:
        raise ValueError("경로 또는 학습 설정이 잘못됐습니다")
    return cfg


def _curve_cfg(cfg: M3BudgetConfig) -> curves.CapacitySearchConfig:
    return curves.CapacitySearchConfig(
        conditions=("baseline",), seeds=(cfg.seed,), epochs=cfg.epochs,
        eval_every=cfg.eval_every, batch_size=cfg.batch_size, lr=cfg.lr,
        n_layers=cfg.n_layers, negative_count=cfg.negative_count,
        out_dir=cfg.out_dir,
    )


def preflight_summary(cfg: M3BudgetConfig) -> dict:
    cfg = validate_config(cfg)
    return {
        "code_version": CODE_VERSION, "dataset": "dunnhumby",
        "split": "historical_development_days_684_690",
        "research_question": "Does frozen candidate-specific N/V M3 need more than 100 epochs?",
        "m3": {
            "F": "1-(abs(q_N-b_N)+abs(q_V-p_V))/2",
            "edge_weight": "exp(beta*q_C*(F-user_mean_F)), user mean normalized to one",
            "beta": cfg.beta_m3, "formula_changed": False,
        },
        "reference": {"curve_csv": cfg.reference_curve_csv,
                      "reuse_only_after_identity_check": True},
        "fixed": {"new_item_task": True, "train_pairs_excluded": True,
                  "min_item_interactions": 1,
                  "graph": "binary edge set with frozen M3 coefficients",
                  "negative_sampling": "one uniform unseen item",
                  "epochs": cfg.epochs,
                  "evaluated_at": curves.evaluation_epochs(_curve_cfg(cfg)),
                  "final_test": False, "holdout": False,
                  "external_reranking": False},
        "limits": "one exposed development seed; no selection, significance or attribution claim",
        "out_dir": cfg.out_dir,
    }


def load_reference_curve(cfg: M3BudgetConfig) -> pd.DataFrame:
    path = Path(cfg.reference_curve_csv)
    if not path.exists():
        raise FileNotFoundError(f"완료된 M1 curve CSV가 없습니다: {path}")
    frame = pd.read_csv(path)
    required = {"condition", "model_id", "seed", "epoch", "id_dim", "pref_reg", "rho"}
    if not required.issubset(frame.columns):
        raise ValueError(f"M1 curve 필수 열이 없습니다: {sorted(required - set(frame.columns))}")
    ref = frame[frame.condition.eq("baseline") & frame.model_id.eq(curves.M1_MODEL_ID)].copy()
    checks = {
        "seed": set(ref.seed.astype(int)) == {cfg.seed},
        "id_dim": set(ref.id_dim.astype(int)) == {cfg.id_dim},
        "pref_reg": np.allclose(ref.pref_reg.astype(float), cfg.pref_reg),
        "rho": np.allclose(ref.rho.astype(float), 0.0),
        "epochs": sorted(ref.epoch.astype(int).tolist()) == curves.evaluation_epochs(_curve_cfg(cfg)),
    }
    failed = [key for key, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"M1 재사용 신분이 일치하지 않습니다: {failed}")
    return ref.sort_values("epoch").reset_index(drop=True)


def _candidate_cfg(cfg: M3BudgetConfig) -> candidate.CandidateNVFitConfig:
    return candidate.CandidateNVFitConfig(
        out_dir=cfg.out_dir, baseline_result_dir=cfg.out_dir, seed=cfg.seed,
        epochs=cfg.epochs, id_dim=cfg.id_dim, pref_reg=cfg.pref_reg,
        beta_m3=cfg.beta_m3, batch_size=cfg.batch_size, lr=cfg.lr,
        n_layers=cfg.n_layers, negative_count=cfg.negative_count,
    )


def _build_model(prepared: dict, cfg: M3BudgetConfig):
    data = prepared["data"]
    v3.set_seed(cfg.seed)
    return CandidateNVFitLightGCN(
        n_users=data["n_users"], n_items=data["n_items"],
        user_q_n=prepared["q_n"], user_q_v=prepared["q_v"],
        user_q_c=prepared["q_c"], user_clv_valid=prepared["clv_valid"],
        item_buyer_q_n=prepared["item_buyer_q_n"],
        item_amount_percentile=prepared["item_amount_percentile"],
        item_context_valid=prepared["item_context_valid"], adj=prepared["m3_adj"],
        id_dim=cfg.id_dim, rho=0.0, n_layers=cfg.n_layers,
        pref_reg=cfg.pref_reg,
    ).to(v3.DEVICE)


def _frame(payload: dict) -> pd.DataFrame:
    rows = []
    for record in payload["curve"]:
        if "metrics" in record:
            rows.append({"condition": "m3_training_budget", "model_id": MODEL_ID,
                         "seed": payload["seed"], "epoch": record["epoch"],
                         "loss": record["loss"], "p_correct": record["p_correct"],
                         **record.get("gradient_diagnostics", {}), **record["metrics"]})
    return pd.DataFrame(rows)


def paired_gap(reference: pd.DataFrame, m3: pd.DataFrame) -> pd.DataFrame:
    metrics = sorted(col for col in reference.columns if "@" in col and col in m3.columns)
    merged = reference[["epoch", *metrics]].merge(
        m3[["epoch", *metrics]], on="epoch", suffixes=("_m1", "_m3"), validate="one_to_one")
    rows = []
    for _, row in merged.iterrows():
        output = {"epoch": int(row.epoch)}
        for metric in metrics:
            base, value = float(row[f"{metric}_m1"]), float(row[f"{metric}_m3"])
            output[metric] = value - base
            output[f"{metric}_ratio"] = value / base if base else np.nan
        rows.append(output)
    return pd.DataFrame(rows)


def run_m3_budget(cfg: M3BudgetConfig | None = None) -> pd.DataFrame:
    cfg = validate_config(cfg or configure_m3_budget())
    print(json.dumps(preflight_summary(cfg), ensure_ascii=False, indent=2))
    reference = load_reference_curve(cfg)
    prepared = candidate._prepare(_candidate_cfg(cfg))
    identity = {"code_version": CODE_VERSION, "config": asdict(cfg),
                "input_hash": prepared["input_hash"], "revision": prepared["revision"]}
    prepared["out_dir"] = Path(cfg.out_dir)
    prepared["config_hash"] = hashlib.sha256(
        json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest()[:12]
    spec = {"condition": "m3_training_budget", "hypothesis": "training_budget",
            "shared_key": "dim64_l20.001", "model_id": MODEL_ID,
            "id_dim": 64, "axis_dim": 0, "pref_reg": 1e-3, "rho": 0.0}
    result_path = prepared["out_dir"] / f"m3_training_budget_{prepared['config_hash']}.json"
    if result_path.exists():
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        print("[cached] 완료된 M3 곡선을 재사용합니다")
    else:
        model = _build_model(prepared, cfg)
        store = ProgressStore(
            prepared["out_dir"] / "progress" / prepared["config_hash"],
            RunIdentity(stage="m3_training_budget_dev", model_id=MODEL_ID,
                        seed=cfg.seed, config_hash=prepared["config_hash"],
                        source_revision=prepared["revision"],
                        input_hash=prepared["input_hash"]),
        )
        history = curves._train_curve(model, prepared, _curve_cfg(cfg), spec, cfg.seed, store)
        payload = {**spec, "seed": cfg.seed, "curve": history,
                   "m3_diagnostics": prepared["m3_diagnostics"],
                   "evaluated_at": datetime.now(timezone.utc).isoformat()}
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        store.mark_complete(epoch=cfg.epochs, max_epoch=cfg.epochs,
                            checkpoint_path="", result_path=str(result_path))
    frame = _frame(payload)
    gap = paired_gap(reference, frame)
    curve_csv = Path(cfg.out_dir) / "m3_training_budget_curve.csv"
    gap_csv = Path(cfg.out_dir) / "m3_vs_m1_gap.csv"
    frame.to_csv(curve_csv, index=False)
    gap.to_csv(gap_csv, index=False)
    print("\nM3 학습곡선:")
    print(frame[["epoch", "loss", "recall@10", "ndcg@10"]].to_string(index=False))
    print("\nM3 - 재사용 M1:")
    print(gap[["epoch", "recall@10", "ndcg@10"]].to_string(index=False))
    frame.attrs.update(gap=gap, reference=reference, result_paths={
        "curve_csv": str(curve_csv), "gap_csv": str(gap_csv), "json": str(result_path)})
    return frame


if __name__ == "__main__":
    print(json.dumps(preflight_summary(configure_m3_budget()), ensure_ascii=False, indent=2))
