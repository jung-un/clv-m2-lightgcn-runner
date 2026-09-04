"""H&M two-year seed-42 validation of the selected M2 embedding."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from clv_constrained_economic_embedding_model import ConstrainedCLVEconomicLightGCN
import lightgcn_clv_constrained_economic_embedding as selected
import lightgcn_clv_hm2y_seed42_common as common
import lightgcn_clv_joint_response_embedding as views
import lightgcn_clv_v3 as v3


CODE_VERSION = "m2-clv-level-composition-price-embedding-hm2y-seed42-v1"
MATCHED_MODEL_ID = selected.MATCHED_MODEL_ID
MODEL_ID = selected.MODEL_ID
SHUFFLED_MODEL_ID = selected.SHUFFLED_MODEL_ID
ID_ONLY_MODEL_ID = selected.ID_ONLY_MODEL_ID
MODELS = (MATCHED_MODEL_ID, MODEL_ID, SHUFFLED_MODEL_ID, ID_ONLY_MODEL_ID)
ACCURACY_METRICS = common.ACCURACY_METRICS
PRIMARY_METRICS = (
    "고CLV_recall@10",
    "고CLV_ndcg@10",
    "price_purchase_amount_weighted_hit@10",
)


@dataclass(frozen=True)
class HMM2Seed42Config:
    dataset: str = "hm"
    seed: int = 42
    window_days: None = None
    input_days: int = 365
    epochs: int = 100
    id_dim: int = 64
    clv_dim: int = 3
    rho: float = 0.05
    item_price_budget: float = 0.25
    n_layers: int = 2
    batch_size: int = common.DEFAULT_BATCH_SIZE
    lr: float = 5e-4
    pref_reg: float = 1e-3
    shuffle_degree_bins: int = 10
    shuffle_seed: int = 1042
    eval_test: bool = False
    eval_holdout: bool = False
    out_dir: str = ""


def configure_hm2y_seed42_run(**overrides) -> HMM2Seed42Config:
    defaults = {
        "out_dir": (
            f"{v3.default_out_dir('hm')}"
            "_m2_level_composition_price_hm2y_seed42_v1"
        )
    }
    return validate_hm2y_seed42_config(
        HMM2Seed42Config(**(defaults | overrides))
    )


def validate_hm2y_seed42_config(cfg: HMM2Seed42Config) -> HMM2Seed42Config:
    required = {
        "dataset": "hm",
        "seed": 42,
        "window_days": None,
        "input_days": 365,
        "epochs": 100,
        "id_dim": 64,
        "clv_dim": 3,
        "rho": 0.05,
        "item_price_budget": 0.25,
        "n_layers": 2,
        "shuffle_degree_bins": 10,
        "shuffle_seed": 1042,
        "eval_test": False,
        "eval_holdout": False,
    }
    for key, expected in required.items():
        if getattr(cfg, key) != expected:
            raise ValueError(f"H&M M2 seed-42 실행은 {key}={expected!r}이어야 합니다")
    if cfg.batch_size not in common.BATCH_CANDIDATES:
        raise ValueError("batch_size는 131072/65536/32768 중 하나여야 합니다")
    if cfg.lr <= 0 or cfg.pref_reg < 0 or not cfg.out_dir:
        raise ValueError("H&M M2 학습 설정이 잘못됐습니다")
    return cfg


def preflight_summary(cfg: HMM2Seed42Config) -> dict:
    cfg = validate_hm2y_seed42_config(cfg)
    return {
        "code_version": CODE_VERSION,
        "dataset": "hm",
        "period": "full_history_about_2_years",
        "seed": 42,
        "split": "hm2y_validation",
        "models": list(MODELS),
        "m2": {
            "architecture": "ID(64)|CLV relation(2)|explicit price fit(1)",
            "rho": cfg.rho,
            "item_price_budget": cfg.item_price_budget,
            "changed_from_dunnhumby_ten_seed_model": False,
            "degree_matched_clv_shuffle": True,
            "id_only_is_posthoc_view": True,
        },
        "fixed": {
            "task": "new-item recommendation",
            "graph": "binary",
            "negative_sampling": "uniform",
            "sample_weighting": False,
            "loss": "one plain BPR plus sampled ID L2",
            "one_optimizer": True,
            "min_item_interactions": 1,
            "epochs": 100,
            "epoch_selection": False,
            "test_constructed": False,
            "holdout_constructed": False,
        },
        "decision": (
            "seed-42 exploratory attribution screen: actual CLV must preserve "
            "all six accuracy metrics within 99% of rho=0 and beat rho=0, "
            "joint ID-only, and degree-matched shuffle on all three primary metrics"
        ),
        "statistical_note": (
            "H&M 2-year seed 42 validation only; no significance or generalization claim"
        ),
        "automatic_epoch_resume": True,
        "compact_parameter_only_checkpoint": True,
        "out_dir": cfg.out_dir,
    }


def _arm_hash(prepared: dict, model_id: str, rho: float, assignment: str) -> str:
    payload = {
        "run": prepared["config_hash"],
        "model_id": model_id,
        "seed": 42,
        "rho": rho,
        "assignment": assignment,
    }
    return hashlib.sha256(common.canonical(payload).encode()).hexdigest()[:12]


def _arm_paths(prepared: dict, model_id: str) -> dict[str, Path]:
    root = prepared["out_dir"] / "arms" / prepared["config_hash"]
    return {
        "result": root / f"{model_id}_s42.json",
        "checkpoint": root / f"{model_id}_s42.pt",
    }


def _build_model(prepared: dict, cfg: HMM2Seed42Config, rho: float, assignment):
    v3.set_seed(42)
    model = ConstrainedCLVEconomicLightGCN(
        n_users=prepared["data"]["n_users"],
        n_items=prepared["data"]["n_items"],
        q_n=assignment["q_n"],
        q_v=assignment["q_v"],
        q_c=assignment["q_c"],
        user_clv_valid=assignment["clv_valid"],
        item_economic_features=prepared["item_economic"],
        item_economic_valid=prepared["item_economic_valid"],
        adj=prepared["data"]["adj"],
        id_dim=cfg.id_dim,
        clv_dim=cfg.clv_dim,
        rho=rho,
        item_price_budget=cfg.item_price_budget,
        n_layers=cfg.n_layers,
        pref_reg=cfg.pref_reg,
    ).to(v3.DEVICE)
    return model


def _run_arm(prepared, cfg, *, model_id, rho, assignment, assignment_name):
    paths = _arm_paths(prepared, model_id)
    model = _build_model(prepared, cfg, rho, assignment)
    if paths["result"].exists() and paths["checkpoint"].exists():
        payload = json.loads(paths["result"].read_text(encoding="utf-8"))
        checkpoint = torch.load(
            paths["checkpoint"], map_location="cpu", weights_only=False
        )
        if checkpoint.get("input_hash") != prepared["input_hash"]:
            raise RuntimeError("cached M2 checkpoint와 H&M 입력 hash가 다릅니다")
        common.load_parameter_state(model, checkpoint["parameter_state"])
        model.eval()
        print(f"  [cached] {model_id}")
        return payload, model
    store = common.progress_store(
        prepared,
        cfg,
        model_id,
        _arm_hash(prepared, model_id, rho, assignment_name),
    )
    training = common.train_plain_bpr(model, prepared, cfg, model_id, store)
    model.eval()
    paths["checkpoint"].parent.mkdir(parents=True, exist_ok=True)
    common.final_parameter_checkpoint(
        paths["checkpoint"], model, prepared, cfg, model_id, training
    )
    metrics = common.evaluate(model, prepared)
    payload = {
        "model_id": model_id,
        "role": {
            MATCHED_MODEL_ID: "matched_control",
            MODEL_ID: "model",
            SHUFFLED_MODEL_ID: "assignment_control",
        }[model_id],
        "seed": 42,
        "split": "hm2y_validation",
        "final_epoch": 100,
        "rho": rho,
        "clv_assignment": assignment_name,
        "metrics": metrics,
        "diagnostics": model.representation_diagnostics(),
        "training": training,
        "checkpoint": str(paths["checkpoint"]),
        "checkpoint_sha256": common.checkpoint_sha256(paths["checkpoint"]),
        "input_hash": prepared["input_hash"],
    }
    common.atomic_json(paths["result"], payload)
    store.mark_complete(
        epoch=100,
        max_epoch=100,
        selection="none",
        split="hm2y_validation",
        checkpoint_path=str(paths["checkpoint"]),
        result_path=str(paths["result"]),
    )
    return payload, model


def _id_only_payload(model, prepared, cfg):
    metrics = common.evaluate(views._IDOnlyView(model).to(v3.DEVICE), prepared)
    return {
        "model_id": ID_ONLY_MODEL_ID,
        "role": "joint_training_ablation",
        "seed": 42,
        "split": "hm2y_validation",
        "final_epoch": 100,
        "metrics": metrics,
        "diagnostics": {},
        "training": {"additional_training": False},
    }


def _absolute_rows(arms: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model_id": arm["model_id"],
                "role": arm["role"],
                "seed": 42,
                "split": "hm2y_validation",
                "final_epoch": 100,
                **arm.get("diagnostics", {}),
                **arm["metrics"],
            }
            for arm in arms
        ]
    )


def seed42_decision(absolute: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    indexed = absolute.set_index("model_id")
    missing = set(MODELS) - set(indexed.index)
    if missing or len(absolute) != len(MODELS):
        raise ValueError(f"M2 seed-42 비교 view 누락: {sorted(missing)}")
    ratios = {
        metric: float(indexed.loc[MODEL_ID, metric])
        / max(float(indexed.loc[MATCHED_MODEL_ID, metric]), 1e-12)
        for metric in ACCURACY_METRICS
    }
    paired_rows = []
    for reference in (MATCHED_MODEL_ID, ID_ONLY_MODEL_ID, SHUFFLED_MODEL_ID):
        for metric in PRIMARY_METRICS:
            delta = float(
                indexed.loc[MODEL_ID, metric] - indexed.loc[reference, metric]
            )
            paired_rows.append(
                {
                    "reference": reference,
                    "metric": metric,
                    "delta": delta,
                    "passes": delta > 0.0,
                }
            )
    paired = pd.DataFrame(paired_rows)
    guard = all(value >= 0.99 for value in ratios.values())
    return {
        "positive_screen": bool(guard and paired["passes"].all()),
        "accuracy_guard_pass": bool(guard),
        "all_primary_control_comparisons_pass": bool(paired["passes"].all()),
        "accuracy_ratios_vs_matched_rho0": ratios,
        "statistical_note": (
            "H&M 2-year seed 42 validation only; no significance or generalization claim"
        ),
    }, paired


def run_hm2y_seed42(cfg: HMM2Seed42Config | None = None) -> pd.DataFrame:
    cfg = validate_hm2y_seed42_config(cfg or configure_hm2y_seed42_run())
    preflight = preflight_summary(cfg)
    print(json.dumps(preflight, ensure_ascii=False, indent=2))
    prepared = common.prepare_hm2y(cfg, code_version=CODE_VERSION)
    observed = {
        key: prepared[key] for key in ("q_n", "q_v", "q_c", "clv_valid")
    }
    shuffle_meta = common.degree_matched_sources(
        prepared["clv_valid"],
        prepared["binary_user_degree"],
        n_bins=cfg.shuffle_degree_bins,
        seed=cfg.shuffle_seed,
    )
    source = shuffle_meta["source_user"]
    shuffled = {
        "q_n": prepared["q_n"][source],
        "q_v": prepared["q_v"][source],
        "q_c": prepared["q_c"][source],
        "clv_valid": prepared["clv_valid"][source],
    }
    matched, _ = _run_arm(
        prepared,
        cfg,
        model_id=MATCHED_MODEL_ID,
        rho=0.0,
        assignment=observed,
        assignment_name="observed_nonintervention",
    )
    actual, actual_model = _run_arm(
        prepared,
        cfg,
        model_id=MODEL_ID,
        rho=cfg.rho,
        assignment=observed,
        assignment_name="observed",
    )
    shuffled_arm, _ = _run_arm(
        prepared,
        cfg,
        model_id=SHUFFLED_MODEL_ID,
        rho=cfg.rho,
        assignment=shuffled,
        assignment_name="degree_matched_shuffle",
    )
    arms = [matched, actual, shuffled_arm, _id_only_payload(actual_model, prepared, cfg)]
    absolute = _absolute_rows(arms)
    decision, paired = seed42_decision(absolute)
    out = prepared["out_dir"]
    stem = f"m2_level_composition_price_hm2y_seed42_{prepared['config_hash']}"
    paths = {
        "absolute_csv": out / f"{stem}.csv",
        "paired_csv": out / f"{stem}_paired.csv",
        "json": out / f"{stem}.json",
    }
    absolute.to_csv(paths["absolute_csv"], index=False)
    paired.to_csv(paths["paired_csv"], index=False)
    common.atomic_json(
        paths["json"],
        {
            "code_version": CODE_VERSION,
            "config": asdict(cfg),
            "preflight": preflight,
            "input_manifest": prepared["manifest"],
            "absolute_rows": absolute.to_dict("records"),
            "paired_control_rows": paired.to_dict("records"),
            "decision": decision,
            "shuffle_diagnostics": {
                key: value
                for key, value in shuffle_meta.items()
                if key not in {"source_user", "stratum"}
            },
            "arms": {arm["model_id"]: arm for arm in arms},
        },
    )
    absolute.attrs["decision"] = decision
    absolute.attrs["paired_control"] = paired.to_dict("records")
    absolute.attrs["result_paths"] = {key: str(value) for key, value in paths.items()}
    print("H&M 2년 M2 seed-42 validation 판정:")
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    print("결과 파일:", absolute.attrs["result_paths"])
    return absolute


def read_progress(out_dir: str | Path) -> dict:
    return common.read_progress(out_dir)


if __name__ == "__main__":
    print(json.dumps(preflight_summary(configure_hm2y_seed42_run()), ensure_ascii=False, indent=2))
