"""Validation-only runner for identifying single-adapter CLV information effects."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd

import lightgcn_clv_moe as moe
import lightgcn_clv_v3 as v3


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
