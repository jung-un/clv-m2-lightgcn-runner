"""Fast M1/V-only/full-CLV axis-adaptive M3 screening on Dunnhumby."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

import lightgcn_clv_v3 as v3


CODE_VERSION = "m3-clv-axis-adaptive-graph-v1"
MODEL_ID = "m3_clv_axis_adaptive"
V_ONLY_ID = "m3_clv_axis_adaptive_v_only"
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
        return "/content/drive/MyDrive/논문/data/results_m3_clv_axis_adaptive_dunnhumby"
    return str(
        Path(v3.default_out_dir("dunnhumby")).with_name(
            "results_m3_clv_axis_adaptive_dunnhumby"
        )
    )


def validate_screening_config(cfg: dict) -> None:
    expected = {
        "DATASET": "dunnhumby",
        "SEED_LIST": [42],
        "ARCH": "pref_only",
        "GRAPH_MODE": "clv_axis_adaptive",
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


def configure_m3_axis_adaptive_dunnhumby_run(
    *, out_dir: str | None = None, **overrides
) -> dict:
    """Create the fixed seed-42 validation comparison for M1, V-only and M3."""
    settings = {
        "ARCH": "pref_only",
        "SEED_LIST": [42],
        "WINDOW_DAYS": None,
        "MIN_USER_INTER": 1,
        "MIN_ITEM_INTER": 1,
        "GRAPH_MODE": "clv_axis_adaptive",
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
        "dataset": "dunnhumby",
        "seed": 42,
        "split": "validation only",
        "models": ["m1_baseline", V_ONLY_ID, MODEL_ID],
        "architecture": (
            "same binary M1 edge set; train-only CLV N/V factor-adaptive "
            "mean-one propagation weights"
        ),
        "n_user_gate": "percentile(log(1 + transaction activity))",
        "v_user_gate": "percentile(log(1 + mean basket value))",
        "n_relation": (
            "item-level estimate of 7-day next transaction x next-basket novel "
            "item share, leave-one-basket user-residualized and category-shrunk"
        ),
        "v_relation": "mean item share of the user's basket value",
        "training": {
            "loss": "plain BPR",
            "negative_sampling": "uniform",
            "min_user_item_interactions": 1,
            "no_m2_embedding": True,
            "no_m4_loss_weight": True,
        },
        "train_only_audit": (
            "pair/item observation counts, N relation-popularity Spearman, "
            "N relation variance/unique values; descriptive and non-blocking"
        ),
        "required_control": (
            "same-setting V-only (beta_N=0); distinguishes full CLV gain from "
            "a value-axis-only explanation"
        ),
        "eval_test": False,
        "eval_holdout": False,
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
    v_only = val[val["model_id"].eq(V_ONLY_ID)]
    if model.empty or baseline.empty or v_only.empty:
        raise ValueError(
            "validation result must contain M1, V-only, and full M3 rows"
        )
    model_mean = model[list(ACCURACY_METRICS) + ["revenue@10"]].mean()
    base_mean = baseline[list(ACCURACY_METRICS) + ["revenue@10"]].mean()
    ratios = {
        metric: float(model_mean[metric] / base_mean[metric])
        for metric in ACCURACY_METRICS
    }
    economic_gain = float(model_mean["revenue@10"] - base_mean["revenue@10"])
    v_only_revenue = float(v_only["revenue@10"].mean())
    full_beats_v_only = float(model_mean["revenue@10"] - v_only_revenue)
    baseline_screen = bool(
        all(value >= 0.99 for value in ratios.values()) and economic_gain > 0
    )
    return {
        "success": bool(baseline_screen and full_beats_v_only > 0),
        "baseline_screen_success": baseline_screen,
        "full_clv_beats_v_only": bool(full_beats_v_only > 0),
        "economic_improved_vs_m1": bool(economic_gain > 0),
        "revenue@10_delta": economic_gain,
        "revenue@10_delta_vs_v_only": full_beats_v_only,
        "accuracy_ratios_vs_m1": ratios,
        "note": (
            "Post-hoc validation reading only. revenue is a purchase-value-weighted "
            "hit metric, not actual revenue. Additional controls run only after "
            "a positive screen."
        ),
    }


def _run_mode(cfg: dict, graph_mode: str) -> pd.DataFrame:
    current = dict(cfg)
    current["GRAPH_MODE"] = graph_mode
    settings = {
        key: value
        for key, value in current.items()
        if key not in {"DATASET", "OUT_DIR"}
    }
    v3.configure_run(current["DATASET"], out_dir=current["OUT_DIR"], **settings)
    return normalize_result_schema(v3.main())


def _native_result_paths() -> dict:
    stem = (
        f"result_pref_only_{v3.CFG['DATASET']}_"
        f"{v3.result_hash(v3.CFG, v3.DCFG, 'pref_only')}"
    )
    out = Path(v3.CFG["OUT_DIR"])
    return {
        "json": str(out / f"{stem}.json"),
        "val_csv": str(out / f"{stem}_val.csv"),
        "delta_csv": str(out / f"{stem}_delta.csv"),
    }


def _result_fingerprint(cfg: dict) -> str:
    payload = {
        "code_version": CODE_VERSION,
        "v3_code_version": v3.CODE_VERSION,
        "dataset": cfg["DATASET"],
        "seed_list": cfg["SEED_LIST"],
        "graph_modes": ["clv_axis_adaptive_v_only", "clv_axis_adaptive"],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()[:12]


def run_experiment(cfg: dict | None = None) -> pd.DataFrame:
    cfg = (
        configure_m3_axis_adaptive_dunnhumby_run()
        if cfg is None
        else dict(cfg)
    )
    summary = preflight_summary(cfg)
    print("M3 CLV axis-adaptive graph preflight:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    full = _run_mode(cfg, "clv_axis_adaptive")
    full_paths = _native_result_paths()
    v_only = _run_mode(cfg, "clv_axis_adaptive_v_only")
    v_only_paths = _native_result_paths()
    v_only = v_only[v_only["model_id"].eq(V_ONLY_ID)].copy()
    baseline = full[full["model_id"].eq("m1_baseline")]
    full_model = full[full["model_id"].eq(MODEL_ID)]
    frame = pd.concat([baseline, v_only, full_model], ignore_index=True)
    decision = screening_decision(frame)
    out = Path(cfg["OUT_DIR"])
    out.mkdir(parents=True, exist_ok=True)
    fingerprint = _result_fingerprint(cfg)
    csv_path = out / f"m3_axis_adaptive_comparison_{fingerprint}.csv"
    json_path = out / f"m3_axis_adaptive_comparison_{fingerprint}.json"
    frame.to_csv(csv_path, index=False, float_format="%.8f")
    native_paths = {"full": full_paths, "v_only": v_only_paths}
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "preflight": summary,
                "screening_decision": decision,
                "absolute_rows": frame.to_dict("records"),
                "native_result_paths": native_paths,
            },
            handle,
            ensure_ascii=False,
            indent=2,
            default=float,
        )
    frame.attrs["screening_decision"] = decision
    frame.attrs["preflight"] = summary
    frame.attrs["out_dir"] = cfg["OUT_DIR"]
    frame.attrs["result_paths"] = {
        "csv": str(csv_path),
        "json": str(json_path),
        "native": native_paths,
    }
    print("M3 CLV axis-adaptive screening decision:")
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    print("result files:", frame.attrs["result_paths"])
    return frame


if __name__ == "__main__":
    print(
        json.dumps(
            preflight_summary(configure_m3_axis_adaptive_dunnhumby_run()),
            ensure_ascii=False,
            indent=2,
        )
    )
