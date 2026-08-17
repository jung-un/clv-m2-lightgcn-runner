"""Fast Dunnhumby validation runner for CLV-conditioned modulation M2."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from clv_conditioned_modulation_model import CLVConditionedModulationLightGCN
import lightgcn_clv_joint_nv as joint
import lightgcn_clv_v3 as v3


CODE_VERSION = "m2-clv-conditioned-modulation-v1"
MODEL_ID = "m2_clv_modulation"
ACCURACY_METRICS = (
    "recall@10",
    "ndcg@10",
    "recall@20",
    "ndcg@20",
    "recall@50",
    "ndcg@50",
)


@dataclass(frozen=True)
class ModulationConfig:
    dataset: str = "dunnhumby"
    seed: int = 42
    window_days: int | None = None
    input_days: int = 365
    id_dim: int = 64
    modulation_rank: int = 4
    tau: float = 0.10
    n_layers: int = 2
    batch_size: int = 8192
    lr: float = 5e-4
    pref_reg: float = 1e-3
    compute_variable_validity: bool = False
    max_epochs: int = 100
    early_stop: int = 20
    eval_test: bool = False
    eval_holdout: bool = False
    out_dir: str = ""
    m1_checkpoint_dir: str = ""


def configure_modulation_dunnhumby_run(**overrides) -> ModulationConfig:
    defaults = {
        "out_dir": (
            f"{v3.default_out_dir('dunnhumby')}"
            "_m2_clv_conditioned_modulation_v1"
        ),
        "m1_checkpoint_dir": v3.default_out_dir("dunnhumby"),
    }
    return validate_modulation_config(ModulationConfig(**(defaults | overrides)))


def validate_modulation_config(cfg: ModulationConfig) -> ModulationConfig:
    if cfg.eval_test or cfg.eval_holdout:
        raise ValueError("M2 modulation screening is validation-only")
    if cfg.dataset != "dunnhumby" or cfg.seed != 42:
        raise ValueError("first screening requires Dunnhumby seed 42")
    if cfg.window_days is not None or cfg.input_days != 365:
        raise ValueError("Dunnhumby screening uses the full window and 365-day input")
    if cfg.id_dim != 64:
        raise ValueError("M1 and M2 final embedding dimension must both be 64")
    if cfg.modulation_rank != 4 or cfg.tau != 0.10:
        raise ValueError("first screen fixes modulation_rank=4 and tau=0.10")
    if cfg.n_layers < 0 or min(
        cfg.batch_size, cfg.max_epochs, cfg.early_stop
    ) <= 0:
        raise ValueError("invalid training configuration")
    if cfg.compute_variable_validity:
        raise ValueError("fast screen excludes the already completed validity analysis")
    return cfg


def preflight_summary(cfg: ModulationConfig) -> dict:
    cfg = validate_modulation_config(cfg)
    return {
        "code_version": CODE_VERSION,
        "dataset": cfg.dataset,
        "seed": cfg.seed,
        "models": ["m1", MODEL_ID],
        "architecture": (
            "CLV N/V-conditioned modulation -> one 64d LightGCN -> one dot score"
        ),
        "feature_role": (
            "historical N/V behaviour conditions the scale of ID embeddings; "
            "it does not create an independent score"
        ),
        "final_embedding_dim": cfg.id_dim,
        "modulation_rank": cfg.modulation_rank,
        "tau": cfg.tau,
        "initial_state": "zero output projections; exactly ordinary LightGCN",
        "graph_mode": "binary",
        "negative_sampling": "uniform",
        "loss": "plain_bpr",
        "separate_encoder": False,
        "external_m1_or_post_score_residual": False,
        "lambda_or_gamma": False,
        "eval_test": cfg.eval_test,
        "eval_holdout": cfg.eval_holdout,
        "out_dir": cfg.out_dir,
    }


def _prepare(cfg: ModulationConfig) -> dict:
    return joint._prepare(cfg)


def _train_m1(prepared: dict, cfg: ModulationConfig):
    return joint._train_m1(prepared, cfg)


def _build_model(prepared: dict, cfg: ModulationConfig):
    data = prepared["data"]
    axes = prepared["axes"]
    return CLVConditionedModulationLightGCN(
        n_users=data["n_users"],
        n_items=data["n_items"],
        user_activity=axes["activity"],
        user_value=axes["value"],
        user_activity_valid=axes["activity_valid"],
        user_value_valid=axes["value_valid"],
        item_profile=prepared["item_profile"],
        adj=data["adj"],
        embedding_dim=cfg.id_dim,
        modulation_rank=cfg.modulation_rank,
        tau=cfg.tau,
        n_layers=cfg.n_layers,
        pref_reg=cfg.pref_reg,
    ).to(v3.DEVICE)


def _evaluate(model, prepared, per_user=True):
    return joint._evaluate(model, prepared, per_user=per_user)


def _config_hash(cfg: ModulationConfig, input_hash: str, revision: str) -> str:
    payload = {"config": asdict(cfg), "input_hash": input_hash, "source": revision}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()[:12]


def _train_modulation(prepared: dict, cfg: ModulationConfig) -> dict:
    # _prepare uses the joint helper's hash; recompute under this runner's
    # explicit config so progress/checkpoint identity cannot collide with v1.5.
    config_hash = _config_hash(
        cfg, prepared["input_hash"], prepared["revision"]
    )
    prepared["config_hash"] = config_hash
    v3.set_seed(cfg.seed)
    model = _build_model(prepared, cfg)
    store = joint._progress_store(
        prepared["out_dir"],
        MODEL_ID,
        cfg,
        config_hash,
        prepared["input_hash"],
        prepared["revision"],
    )
    gate = torch.ones(prepared["data"]["n_users"], device=v3.DEVICE)
    training = v3.train_phase(
        model,
        list(model.parameters()),
        prepared["data"],
        gate,
        0.0,
        prepared["base_cfg"],
        cfg.seed,
        MODEL_ID,
        prepared["cache"],
        prepared["meta"],
        progress_store=store,
    )
    model.eval()
    metrics, per_user = _evaluate(model, prepared)
    diagnostics = model.modulation_diagnostics()
    checkpoint = prepared["out_dir"] / (
        f"{MODEL_ID}_{cfg.dataset}_s{cfg.seed}_{config_hash}.pt"
    )
    torch.save(
        {
            "state": model.state_dict(),
            "training": training,
            "diagnostics": diagnostics,
            "config": asdict(cfg),
            "source_revision": prepared["revision"],
            "input_hash": prepared["input_hash"],
        },
        checkpoint,
    )
    store.mark_complete(
        best_metric=float(metrics["recall@10"]), checkpoint_path=str(checkpoint)
    )
    return {
        "model": model,
        "training": training,
        "metrics": metrics,
        "per_user": per_user,
        "diagnostics": diagnostics,
        "checkpoint": str(checkpoint),
    }


def screening_decision(model: dict, baseline: dict) -> dict:
    accuracy_ratios = {
        metric: float(model[metric] / max(float(baseline[metric]), 1e-12))
        for metric in ACCURACY_METRICS
    }
    economic_gain = float(model["revenue@10"] - baseline["revenue@10"])
    return {
        "success": bool(
            economic_gain > 0 and min(accuracy_ratios.values()) >= 0.99
        ),
        "economic_improved_vs_m1": bool(economic_gain > 0),
        "revenue@10_delta": economic_gain,
        "accuracy_ratios_vs_m1": accuracy_ratios,
        "note": (
            "This is a post-hoc validation reading rule, not a training "
            "constraint and not a mechanism that forces improvement."
        ),
    }


def _with_public_metric_names(metrics: dict) -> dict:
    normalized = dict(metrics)
    for k in (10, 20, 50):
        source = f"entropy@{k}"
        target = f"exposure_entropy@{k}"
        if source in normalized and target not in normalized:
            normalized[target] = normalized[source]
    return normalized


def _mask_summary(axes: dict) -> dict:
    activity = np.asarray(axes["activity_valid"], bool)
    value = np.asarray(axes["value_valid"], bool)
    both = activity & value
    return {
        "n_users": int(len(activity)),
        "activity_valid_count": int(activity.sum()),
        "activity_valid_share": float(activity.mean()),
        "value_valid_count": int(value.sum()),
        "value_valid_share": float(value.mean()),
        "both_valid_count": int(both.sum()),
        "both_valid_share": float(both.mean()),
    }


def _persist(
    prepared: dict,
    cfg: ModulationConfig,
    rows: list[dict],
    baseline_per_user: dict,
    run: dict,
    decision: dict,
) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    delta_rows = []
    for metric in ("recall", "ndcg", "revenue", "arp"):
        difference = run["per_user"][metric] - baseline_per_user[metric]
        delta_rows.append(
            {
                "model_id": MODEL_ID,
                "split": "val",
                "metric": metric,
                **v3.paired_bootstrap(
                    [difference], prepared["base_cfg"]["N_BOOT"]
                ),
            }
        )

    stem = f"m2_clv_modulation_{cfg.dataset}_{prepared['config_hash']}"
    csv_path = prepared["out_dir"] / f"{stem}.csv"
    delta_path = prepared["out_dir"] / f"{stem}_delta.csv"
    json_path = prepared["out_dir"] / f"{stem}.json"
    frame.to_csv(csv_path, index=False, float_format="%.8f")
    pd.DataFrame(delta_rows).to_csv(delta_path, index=False)
    mask_summary = _mask_summary(prepared["axes"])
    payload = {
        "code_version": CODE_VERSION,
        "source_revision": prepared["revision"],
        "input_manifest": prepared["manifest"],
        "config": asdict(cfg),
        "preflight": preflight_summary(cfg),
        "feature_schema": {
            "user_activity": list(prepared["axes"]["activity_names"]),
            "user_value": list(prepared["axes"]["value_names"]),
            "item_activity": list(prepared["item_profile"].activity_names),
            "item_value": list(prepared["item_profile"].value_names),
        },
        "user_axis_mask_summary": mask_summary,
        "decision": decision,
        "training": run["training"],
        "diagnostics": run["diagnostics"],
        "checkpoint": run["checkpoint"],
        "absolute_rows": frame.to_dict("records"),
        "paired_delta": delta_rows,
        "interpretation": {
            "clv": (
                "historical transaction-activity N and transaction-value V "
                "behaviour condition the learned representation"
            ),
            "revenue": (
                "price/purchase-amount weighted recommendation hit, not CLV "
                "or incremental revenue"
            ),
            "item": "item activity/economic attributes, not item CLV",
        },
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    paths = {
        "csv": str(csv_path),
        "delta_csv": str(delta_path),
        "json": str(json_path),
        "checkpoint": run["checkpoint"],
    }
    frame.attrs["screening_decision"] = decision
    frame.attrs["user_axis_mask_summary"] = mask_summary
    frame.attrs["result_paths"] = paths
    frame.attrs["out_dir"] = str(prepared["out_dir"])
    return frame


def run_experiment(cfg: ModulationConfig | None = None) -> pd.DataFrame:
    cfg = validate_modulation_config(
        cfg or configure_modulation_dunnhumby_run()
    )
    print(json.dumps(preflight_summary(cfg), ensure_ascii=False, indent=2))
    prepared = _prepare(cfg)
    _, m1_training, baseline, baseline_per_user = _train_m1(prepared, cfg)
    run = _train_modulation(prepared, cfg)
    baseline = _with_public_metric_names(baseline)
    model_metrics = _with_public_metric_names(run["metrics"])
    rows = [
        joint.result_row("m1", "baseline", "none", cfg.seed, baseline),
        joint.result_row(
            MODEL_ID,
            "model",
            "none",
            cfg.seed,
            model_metrics,
            run["diagnostics"],
        ),
    ]
    decision = screening_decision(model_metrics, baseline)
    frame = _persist(
        prepared, cfg, rows, baseline_per_user, run, decision
    )
    frame.attrs["m1_training"] = m1_training
    print("M2 modulation diagnostics:", run["diagnostics"])
    print("validation decision:", decision)
    print("result paths:", frame.attrs["result_paths"])
    return frame


if __name__ == "__main__":
    print(json.dumps(preflight_summary(configure_modulation_dunnhumby_run()), indent=2))
