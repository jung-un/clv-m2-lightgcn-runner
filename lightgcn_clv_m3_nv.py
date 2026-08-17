"""Fast Dunnhumby validation runner for the M3 CLV-NV value graph."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import lightgcn_clv_v3 as v3


CODE_VERSION = "m3-clv-nv-graph-v1"
MODEL_ID = "m3_clv_nv"
ACCURACY_METRICS = (
    "recall@10", "ndcg@10", "recall@20",
    "ndcg@20", "recall@50", "ndcg@50",
)


def _default_out_dir() -> str:
    if v3.IN_COLAB:
        return "/content/drive/MyDrive/논문/data/results_m3_clv_nv_dunnhumby"
    return str(Path(v3.default_out_dir("dunnhumby")).with_name(
        "results_m3_clv_nv_dunnhumby"
    ))


def validate_screening_config(cfg: dict) -> None:
    expected = {
        "DATASET": "dunnhumby",
        "SEED_LIST": [42],
        "ARCH": "pref_only",
        "GRAPH_MODE": "clv_nv",
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


def configure_m3_clv_nv_dunnhumby_run(
    *, out_dir: str | None = None, **overrides
) -> dict:
    """Create the fixed seed-42 validation screen; no test/holdout is exposed."""
    settings = {
        "ARCH": "pref_only",
        "SEED_LIST": [42],
        "WINDOW_DAYS": None,
        "MIN_USER_INTER": 1,
        "MIN_ITEM_INTER": 1,
        "GRAPH_MODE": "clv_nv",
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
        "models": ["m1", MODEL_ID],
        "architecture": (
            "same unique user-item edges; M3 changes only train-edge propagation weights"
        ),
        "edge_weight": (
            "historical user N/V percentile x within-user relation N/V percentile; "
            "component mean normalization; weight clip [0.25, 4.0]"
        ),
        "graph_mode": cfg["GRAPH_MODE"],
        "loss_mode": cfg["LOSS_MODE"],
        "negative_sampling": cfg["NEG_MODE"],
        "eval_test": cfg["EVAL_TEST"],
        "eval_holdout": cfg["EVAL_HOLDOUT"],
        "out_dir": cfg["OUT_DIR"],
    }


def screening_decision(frame: pd.DataFrame) -> dict:
    val = frame[frame["split"].eq("val")]
    model = val[val["model_id"].eq(MODEL_ID)]
    baseline = val[val["model_id"].eq("m1_baseline")]
    if model.empty or baseline.empty:
        raise ValueError("validation result must contain m3_clv_nv and m1_baseline rows")
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
        "note": "This is a post-hoc validation decision, not a training constraint.",
    }


def normalize_result_schema(frame: pd.DataFrame) -> pd.DataFrame:
    """Expose the explicit public name used by the M3 result notebook.

    The shared v3 evaluator historically stores Shannon entropy as
    ``entropy@K``.  M3 reports label it ``exposure_entropy@K`` so it cannot be
    confused with another entropy measure.  Keep the historical column too so
    saved v3 results and downstream analysis remain backwards compatible.
    """
    normalized = frame.copy()
    for k in (10, 20, 50):
        source = f"entropy@{k}"
        target = f"exposure_entropy@{k}"
        if source in normalized.columns and target not in normalized.columns:
            normalized[target] = normalized[source]
    return normalized


def _activate(cfg: dict) -> None:
    validate_screening_config(cfg)
    settings = {key: value for key, value in cfg.items()
                if key not in {"DATASET", "OUT_DIR"}}
    v3.configure_run(cfg["DATASET"], out_dir=cfg["OUT_DIR"], **settings)


def run_experiment(cfg: dict | None = None) -> pd.DataFrame:
    cfg = configure_m3_clv_nv_dunnhumby_run() if cfg is None else dict(cfg)
    _activate(cfg)
    print("M3-CLV-NV screening preflight:")
    print(preflight_summary(cfg))
    frame = normalize_result_schema(v3.main())
    decision = screening_decision(frame)
    frame.attrs["screening_decision"] = decision
    frame.attrs["preflight"] = preflight_summary(cfg)
    frame.attrs["out_dir"] = cfg["OUT_DIR"]
    print("M3-CLV-NV screening decision:", decision)
    return frame


if __name__ == "__main__":
    cfg = configure_m3_clv_nv_dunnhumby_run()
    print(preflight_summary(cfg))
