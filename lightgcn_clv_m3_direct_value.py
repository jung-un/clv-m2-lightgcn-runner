"""Dunnhumby seed-42 validation screen for direct CLV graph weights."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

import lightgcn_clv_v3 as v3


CODE_VERSION = "m3-direct-clv-value-graph-v1"
USER_ID = "m3_clv_direct_user"
SPEND_CONTROL_ID = "m3_clv_direct_spend_control"
CLV_SPEND_ID = "m3_clv_direct_user_spend"
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
        return "/content/drive/MyDrive/논문/data/results_m3_direct_clv_value_dunnhumby"
    return str(
        Path(v3.default_out_dir("dunnhumby")).with_name(
            "results_m3_direct_clv_value_dunnhumby"
        )
    )


def validate_screening_config(cfg: dict) -> None:
    expected = {
        "DATASET": "dunnhumby",
        "SEED_LIST": [42],
        "ARCH": "pref_only",
        "GRAPH_MODE": "clv_direct_user_spend",
        "GRAPH_ALPHA": 1.0,
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


def configure_m3_direct_clv_dunnhumby_run(
    *, out_dir: str | None = None, **overrides
) -> dict:
    settings = {
        "ARCH": "pref_only",
        "SEED_LIST": [42],
        "WINDOW_DAYS": None,
        "MIN_USER_INTER": 1,
        "MIN_ITEM_INTER": 1,
        "GRAPH_MODE": "clv_direct_user_spend",
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
        "models": ["m1_baseline", USER_ID, SPEND_CONTROL_ID, CLV_SPEND_ID],
        "historical_clv": (
            "existing train-only CLV proxy used as one scalar; no N/V graph split"
        ),
        "edge_weights": {
            USER_ID: "mean-one(1 + alpha * g(CLV_u))",
            SPEND_CONTROL_ID: "mean-one(1 + alpha * log1p(spend_ui / mean_unit_price))",
            CLV_SPEND_ID: (
                "mean-one(1 + alpha * g(CLV_u) * "
                "log1p(spend_ui / mean_unit_price))"
            ),
        },
        "alpha": 1.0,
        "shared_invariants": {
            "edge_set": "same unique train user-item pairs as M1",
            "loss": "plain BPR without sample weights",
            "negative_sampling": "uniform",
            "min_user_inter": 1,
            "min_item_inter": 1,
            "no_m2_embedding": True,
            "no_m4_loss_weight": True,
        },
        "interpretation": (
            "spend-only is an identification control, not the CLV proposal; "
            "CLV x spend must beat both M1 and spend-only to identify added CLV value"
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


def _arm_decision(model: pd.Series, baseline: pd.Series) -> dict:
    ratios = {
        metric: float(model[metric] / baseline[metric])
        for metric in ACCURACY_METRICS
    }
    economic_gain = float(model["revenue@10"] - baseline["revenue@10"])
    return {
        "passes_m1_screen": bool(
            all(value >= 0.99 for value in ratios.values()) and economic_gain > 0
        ),
        "economic_improved_vs_m1": bool(economic_gain > 0),
        "revenue@10_delta_vs_m1": economic_gain,
        "accuracy_ratios_vs_m1": ratios,
    }


def screening_decision(frame: pd.DataFrame) -> dict:
    val = frame[frame["split"].eq("val")]

    def row(model_id: str) -> pd.Series:
        selected = val[val["model_id"].eq(model_id)]
        if selected.empty:
            raise ValueError(f"validation result is missing {model_id}")
        return selected[list(ACCURACY_METRICS) + ["revenue@10"]].mean()

    baseline = row("m1_baseline")
    user = row(USER_ID)
    spend = row(SPEND_CONTROL_ID)
    joint = row(CLV_SPEND_ID)
    user_result = _arm_decision(user, baseline)
    joint_result = _arm_decision(joint, baseline)
    joint_delta_vs_spend = float(joint["revenue@10"] - spend["revenue@10"])
    joint_result["revenue@10_delta_vs_spend_control"] = joint_delta_vs_spend
    joint_result["beats_spend_control"] = bool(joint_delta_vs_spend > 0)
    joint_result["success"] = bool(
        joint_result["passes_m1_screen"] and joint_delta_vs_spend > 0
    )
    return {
        "success": bool(user_result["passes_m1_screen"] or joint_result["success"]),
        "user_clv_only": user_result,
        "clv_x_edge_spend": joint_result,
        "spend_control_revenue@10_delta_vs_m1": float(
            spend["revenue@10"] - baseline["revenue@10"]
        ),
        "note": (
            "Validation screening only. revenue is a purchase-value-weighted hit "
            "metric, not actual revenue. No population significance is claimed "
            "without the paired intervals saved by the native runs."
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
        "graph_alpha": cfg["GRAPH_ALPHA"],
        "graph_modes": [
            "clv_direct_user",
            "clv_direct_spend_control",
            "clv_direct_user_spend",
        ],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()[:12]


def run_experiment(cfg: dict | None = None) -> pd.DataFrame:
    cfg = configure_m3_direct_clv_dunnhumby_run() if cfg is None else dict(cfg)
    summary = preflight_summary(cfg)
    print("M3 direct CLV value graph preflight:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    joint = _run_mode(cfg, "clv_direct_user_spend")
    joint_paths = _native_result_paths()
    user = _run_mode(cfg, "clv_direct_user")
    user_paths = _native_result_paths()
    spend = _run_mode(cfg, "clv_direct_spend_control")
    spend_paths = _native_result_paths()

    frame = pd.concat(
        [
            joint[joint["model_id"].eq("m1_baseline")],
            user[user["model_id"].eq(USER_ID)],
            spend[spend["model_id"].eq(SPEND_CONTROL_ID)],
            joint[joint["model_id"].eq(CLV_SPEND_ID)],
        ],
        ignore_index=True,
    )
    decision = screening_decision(frame)
    out = Path(cfg["OUT_DIR"])
    out.mkdir(parents=True, exist_ok=True)
    fingerprint = _result_fingerprint(cfg)
    csv_path = out / f"m3_direct_clv_value_comparison_{fingerprint}.csv"
    json_path = out / f"m3_direct_clv_value_comparison_{fingerprint}.json"
    frame.to_csv(csv_path, index=False, float_format="%.8f")
    native_paths = {
        "user_clv_only": user_paths,
        "spend_control": spend_paths,
        "clv_x_edge_spend": joint_paths,
    }
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
    print("M3 direct CLV value graph screening decision:")
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    print("result files:", frame.attrs["result_paths"])
    return frame


if __name__ == "__main__":
    print(
        json.dumps(
            preflight_summary(configure_m3_direct_clv_dunnhumby_run()),
            ensure_ascii=False,
            indent=2,
        )
    )
