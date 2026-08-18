"""Fast Dunnhumby validation runner for the full-CLV compositional M3 graph."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import lightgcn_clv_v3 as v3


CODE_VERSION = "m3-clv-composition-graph-v1"
MODEL_ID = "m3_clv_composition"
ACCURACY_METRICS = (
    "recall@10",
    "ndcg@10",
    "recall@20",
    "ndcg@20",
    "recall@50",
    "ndcg@50",
)


def _default_out_dir() -> str:
    if v3.IN_COLAB:
        return "/content/drive/MyDrive/논문/data/results_m3_clv_composition_dunnhumby"
    return str(
        Path(v3.default_out_dir("dunnhumby")).with_name(
            "results_m3_clv_composition_dunnhumby"
        )
    )


def validate_screening_config(cfg: dict) -> None:
    expected = {
        "DATASET": "dunnhumby",
        "SEED_LIST": [42],
        "ARCH": "pref_only",
        "GRAPH_MODE": "clv_composition",
        "LOSS_MODE": "plain",
        "NEG_MODE": "uniform",
        "WINDOW_DAYS": None,
        "MIN_USER_INTER": 1,
        "MIN_ITEM_INTER": 1,
        "EVAL_TEST": False,
        "EVAL_HOLDOUT": False,
    }
    for key, wanted in expected.items():
        if cfg.get(key) != wanted:
            raise ValueError(f"screening config requires {key}={wanted!r}")
    if not cfg.get("OUT_DIR") or "dunnhumby" not in str(cfg["OUT_DIR"]):
        raise ValueError("OUT_DIR must identify the Dunnhumby M3 run")


def configure_m3_clv_composition_dunnhumby_run(
    *, out_dir: str | None = None, **overrides
) -> dict:
    """Create the fixed M1/full-CLV seed-42 validation comparison."""
    settings = {
        "ARCH": "pref_only",
        "SEED_LIST": [42],
        "WINDOW_DAYS": None,
        "MIN_USER_INTER": 1,
        "MIN_ITEM_INTER": 1,
        "GRAPH_MODE": "clv_composition",
        "GRAPH_ALPHA": 1.0,
        "LOSS_MODE": "plain",
        "NEG_MODE": "uniform",
        "EVAL_TEST": False,
        "EVAL_HOLDOUT": False,
    }
    settings.update(overrides)
    cfg = dict(v3.CFG)
    cfg.update(settings)
    cfg["DATASET"] = "dunnhumby"
    cfg["OUT_DIR"] = out_dir or _default_out_dir()
    validate_screening_config(cfg)
    return cfg


def preflight_summary(cfg: dict) -> dict:
    validate_screening_config(cfg)
    return {
        "code_version": CODE_VERSION,
        "dataset": cfg["DATASET"],
        "seed": cfg["SEED_LIST"][0],
        "split": "validation only",
        "models": ["m1_baseline", MODEL_ID],
        "clv_definition": "historical CLV proxy magnitude = N_u * V_u",
        "composition": (
            "pi_N=q_N/(q_N+q_V), pi_V=1-pi_N; "
            "q_CLV*(pi_N*z_N_transfer + pi_V*z_V_contribution)"
        ),
        "edge_weight": (
            "positive exponential weight; mean one inside each user's train "
            "neighborhood; train-only effective propagation strength matched"
        ),
        "common_conditions": {
            "edge_set": "same unique train user-item edges as binary M1",
            "loss": "plain BPR",
            "negative_sampling": "uniform",
            "min_user_item_interactions": 1,
        },
        "eval_test": cfg["EVAL_TEST"],
        "eval_holdout": cfg["EVAL_HOLDOUT"],
        "out_dir": cfg["OUT_DIR"],
    }


def normalize_result_schema(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    for k in (10, 20, 50):
        source = f"entropy@{k}"
        target = f"exposure_entropy@{k}"
        if source in normalized.columns and target not in normalized.columns:
            normalized[target] = normalized[source]
    return normalized


def screening_decision(frame: pd.DataFrame) -> dict:
    val = frame[frame["split"].eq("val")]
    model = val[val["model_id"].eq(MODEL_ID)]
    baseline = val[val["model_id"].eq("m1_baseline")]
    if model.empty or baseline.empty:
        raise ValueError(
            "validation result must contain m3_clv_composition and m1_baseline"
        )
    model_mean = model[list(ACCURACY_METRICS) + ["revenue@10"]].mean()
    base_mean = baseline[list(ACCURACY_METRICS) + ["revenue@10"]].mean()
    ratios = {
        metric: float(model_mean[metric] / base_mean[metric])
        for metric in ACCURACY_METRICS
    }
    economic_gain = float(model_mean["revenue@10"] - base_mean["revenue@10"])
    accuracy_pass = all(value >= 0.99 for value in ratios.values())
    return {
        "success": bool(accuracy_pass and economic_gain > 0.0),
        "economic_improved_vs_m1": bool(economic_gain > 0.0),
        "revenue@10_delta": economic_gain,
        "accuracy_ratios_vs_m1": ratios,
        "note": (
            "This is a post-hoc validation reading rule, not a training "
            "constraint. revenue is a purchase-value-weighted hit metric."
        ),
    }


def _activate(cfg: dict) -> None:
    validate_screening_config(cfg)
    settings = {
        key: value for key, value in cfg.items() if key not in {"DATASET", "OUT_DIR"}
    }
    v3.configure_run(cfg["DATASET"], out_dir=cfg["OUT_DIR"], **settings)


def run_experiment(cfg: dict | None = None) -> pd.DataFrame:
    cfg = (
        configure_m3_clv_composition_dunnhumby_run()
        if cfg is None
        else dict(cfg)
    )
    _activate(cfg)
    summary = preflight_summary(cfg)
    print("M3 full-CLV composition graph preflight:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    frame = normalize_result_schema(v3.main())
    decision = screening_decision(frame)
    frame.attrs["screening_decision"] = decision
    frame.attrs["preflight"] = summary
    frame.attrs["out_dir"] = cfg["OUT_DIR"]
    print("M3 full-CLV composition screening decision:")
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    return frame


if __name__ == "__main__":
    print(
        json.dumps(
            preflight_summary(configure_m3_clv_composition_dunnhumby_run()),
            ensure_ascii=False,
            indent=2,
        )
    )
