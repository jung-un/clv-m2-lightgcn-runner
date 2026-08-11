"""Validation-only runner for identifying single-adapter CLV information effects."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import lightgcn_clv_moe as moe
import lightgcn_clv_v3 as v3
from clv_moe_model import CLVMixtureEmbeddingModel


PRIMARY_MODEL_ID = "single_full"
REQUIRED_CONTROLS = (
    "single_zero_user",
    "single_shuffled_user",
    "single_base_only",
)
MECHANISM_CONTROLS = ("single_zero_item",)
ALL_SINGLE_MODELS = (PRIMARY_MODEL_ID, *REQUIRED_CONTROLS, *MECHANISM_CONTROLS)
CODE_VERSION = "clv-single-identification-v1.0"
LAMBDA_GRID = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0)
REUSE_CONFIG_KEYS = (
    "dataset",
    "seed_list",
    "input_days",
    "target_days",
    "anchor_offsets",
    "encoder_epochs",
    "encoder_patience",
    "encoder_batch_size",
    "encoder_lr",
    "expert_count",
    "expert_hidden_dim",
    "expert_dim",
    "category_dim",
    "frozen_epochs",
    "max_epochs",
    "patience",
    "adapter_lr",
    "base_lr",
    "lambda_train",
    "lambda_eval",
    "accuracy_tolerance",
)


@dataclass(frozen=True)
class ReusableSingleFull:
    model: CLVMixtureEmbeddingModel
    rows: tuple[dict, ...]
    training: dict
    diagnostics: dict
    result_json_sha256: str
    legacy_source_revision: str
    legacy_checkpoint: str


def array_sha256(values) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    payload = array.dtype.str.encode() + str(array.shape).encode() + array.tobytes()
    return hashlib.sha256(payload).hexdigest()


def configure_single_run(dataset: str, **overrides) -> moe.MoEConfig:
    defaults = {
        "seed_list": (42,),
        "eval_test": False,
        "eval_holdout": False,
        "lambda_eval": LAMBDA_GRID,
        "run_controls_after_success": True,
        "out_dir": f"{v3.default_out_dir(dataset)}_clv_single",
    }
    return validate_single_config(
        moe.configure_moe_run(dataset, **(defaults | overrides))
    )


def validate_single_config(cfg: moe.MoEConfig) -> moe.MoEConfig:
    if cfg.eval_test or cfg.eval_holdout:
        raise ValueError(
            "single-adapter screening-only runner cannot open test/holdout"
        )
    cfg = moe.validate_moe_config(cfg)
    if tuple(cfg.seed_list) != (42,):
        raise ValueError("single-adapter screening-only runner requires seed 42")
    if tuple(cfg.lambda_eval) != LAMBDA_GRID:
        raise ValueError("single-adapter lambda grid is frozen by the approved design")
    return cfg


def preflight_summary(cfg: moe.MoEConfig) -> dict:
    cfg = validate_single_config(cfg)
    summary = moe.preflight_summary(cfg)
    summary.update(
        {
            "code_version": CODE_VERSION,
            "primary_model_id": PRIMARY_MODEL_ID,
            "required_controls": list(REQUIRED_CONTROLS),
            "mechanism_controls": list(MECHANISM_CONTROLS),
            "models": ["m1", *ALL_SINGLE_MODELS, "pref_continue"],
            "config": asdict(cfg),
        }
    )
    return summary


def _selected_revenue(
    table: pd.DataFrame, selected: dict[str, float], model_id: str
) -> float | None:
    if model_id not in selected:
        return None
    subset = table[
        table["model_id"].eq(model_id)
        & table["split"].eq("val")
        & table["seed"].eq(42)
        & np.isclose(
            table["lambda"].to_numpy(dtype=float), float(selected[model_id])
        )
    ]
    if subset.empty:
        return None
    return float(subset["revenue@10"].iloc[0])


def single_screening_decision(
    rows: list[dict],
    selected: dict[str, float],
    selection_success: dict[str, bool],
) -> dict:
    table = pd.DataFrame(rows)
    main = _selected_revenue(table, selected, PRIMARY_MODEL_ID)
    required = {
        control: _selected_revenue(table, selected, control)
        for control in REQUIRED_CONTROLS
    }
    mechanism = {
        control: _selected_revenue(table, selected, control)
        for control in MECHANISM_CONTROLS
    }
    failed = [
        control
        for control, revenue in required.items()
        if main is None or revenue is None or not main > revenue
    ]
    main_passed = bool(selection_success.get(PRIMARY_MODEL_ID, False))
    success = main_passed and not failed
    return {
        "success": success,
        "reason": (
            "single_full economically outperformed M1 and required controls"
            if success
            else "single_full improvement is absent or explained by a required control"
        ),
        "main_revenue@10": main,
        "required_control_revenue@10": required,
        "mechanism_comparison": mechanism,
        "failed_controls": failed,
    }


def _same_json_value(left, right) -> bool:
    return json.dumps(left, sort_keys=True, default=str) == json.dumps(
        right, sort_keys=True, default=str
    )


def _require_reuse_payload(payload: dict) -> None:
    required = {
        "source_revision",
        "input_manifest",
        "config",
        "baseline_state_hashes",
        "feature_schema",
        "checkpoint_paths",
        "absolute_rows",
        "training",
        "moe_diagnostics",
    }
    missing = required.difference(payload)
    if missing:
        raise RuntimeError(f"saved result JSON is missing fields: {sorted(missing)}")


def load_reusable_single_full(
    result_json: str | Path,
    *,
    current_manifest: dict,
    baseline_state_hash: str,
    cfg: moe.MoEConfig,
    base_cfg: dict,
    context: dict,
    data: dict,
) -> ReusableSingleFull:
    """Reuse a legacy single adapter only after exact provenance round-trip checks."""
    cfg = validate_single_config(cfg)
    result_path = Path(result_json)
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"saved result JSON cannot be read: {result_path}") from error
    if not isinstance(payload, dict):
        raise RuntimeError("saved result JSON must contain an object")
    _require_reuse_payload(payload)

    if payload["input_manifest"] != current_manifest:
        raise RuntimeError("input manifest mismatch; refusing single-adapter reuse")
    if payload["baseline_state_hashes"].get("42") != baseline_state_hash:
        raise RuntimeError("M1 state mismatch; refusing single-adapter reuse")

    current_config = asdict(cfg)
    saved_config = payload["config"]
    for key in REUSE_CONFIG_KEYS:
        if key not in saved_config or not _same_json_value(
            saved_config[key], current_config[key]
        ):
            raise RuntimeError(f"saved config mismatch for {key}")

    expected_schema = {
        "user": list(context["user_profile"].feature_names),
        "item_numeric": list(context["item_profile"].numeric_names),
    }
    if payload["feature_schema"] != expected_schema:
        raise RuntimeError("feature schema mismatch; refusing single-adapter reuse")

    checkpoint_value = payload["checkpoint_paths"].get("single_adapter_s42")
    if not checkpoint_value:
        raise RuntimeError("single_adapter seed-42 checkpoint is missing")
    checkpoint_path = Path(checkpoint_value)
    if not checkpoint_path.is_file():
        raise RuntimeError(f"single_adapter checkpoint does not exist: {checkpoint_path}")
    try:
        checkpoint_payload = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
    except (OSError, RuntimeError) as error:
        raise RuntimeError("single_adapter checkpoint cannot be read") from error
    if "ev_all" not in checkpoint_payload or array_sha256(
        checkpoint_payload["ev_all"]
    ) != array_sha256(context["artifact"].ev_all):
        raise RuntimeError("encoder ev_all hash mismatch")

    base_model = context.get("base_model", context.get("external_m1"))
    model = moe.load_moe_checkpoint(
        checkpoint_path,
        base_model,
        cfg,
        control="single_adapter",
        device=v3.DEVICE,
    )

    saved_rows = [
        row
        for row in payload["absolute_rows"]
        if row.get("seed") == 42
        and row.get("split") == "val"
        and row.get("model_id") == "single_adapter"
    ]
    by_lambda = {float(row["lambda"]): row for row in saved_rows}
    if len(saved_rows) != len(cfg.lambda_eval) or set(by_lambda) != set(
        map(float, cfg.lambda_eval)
    ):
        raise RuntimeError("saved single_adapter lambda curve is incomplete")

    relabeled_rows = []
    for lam in cfg.lambda_eval:
        flat, _ = moe._flat_evaluation(
            model,
            float(lam),
            context["caches"]["val"],
            context.get("meta"),
            data,
            base_cfg,
            per_user=False,
        )
        saved = by_lambda[float(lam)]
        for key, value in flat.items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                if key not in saved or not np.isclose(
                    float(saved[key]), float(value), rtol=0.0, atol=5e-8
                ):
                    raise RuntimeError(
                        f"metric round-trip mismatch for lambda={lam}, metric={key}"
                    )
        relabeled_rows.append(saved | {"model_id": PRIMARY_MODEL_ID, "role": "model"})

    result_sha = hashlib.sha256(result_path.read_bytes()).hexdigest()
    key = "single_adapter_s42"
    return ReusableSingleFull(
        model=model,
        rows=tuple(relabeled_rows),
        training=dict(payload["training"].get(key, {})),
        diagnostics=dict(payload["moe_diagnostics"].get(key, {})),
        result_json_sha256=result_sha,
        legacy_source_revision=str(payload["source_revision"]),
        legacy_checkpoint=str(checkpoint_path),
    )
