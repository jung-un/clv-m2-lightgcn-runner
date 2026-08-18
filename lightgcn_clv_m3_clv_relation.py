"""Dunnhumby seed-42 validation screen for two scalar-CLV M3 graphs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import lightgcn_clv_v3 as v3


CODE_VERSION = "m3-scalar-clv-relation-mixture-v1"
RELATION_CONTROL_ID = "m3_relation_only"
GATE_ID = "m3_clv_relation_gate"
ALLOCATED_CONTROL_ID = "m3_allocated_relation_only"
ALLOCATED_GATE_ID = "m3_clv_allocated_relation_gate"
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
        return "/content/drive/MyDrive/논문/data/results_m3_clv_relation_dunnhumby"
    return str(
        Path(v3.default_out_dir("dunnhumby")).with_name(
            "results_m3_clv_relation_dunnhumby"
        )
    )


def validate_screening_config(cfg: dict) -> None:
    expected = {
        "DATASET": "dunnhumby",
        "SEED_LIST": [42],
        "ARCH": "pref_only",
        "GRAPH_MODE": "clv_allocated_relation_gate",
        "GRAPH_ALPHA": 0.075,
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


def configure_m3_clv_relation_dunnhumby_run(
    *, out_dir: str | None = None, **overrides
) -> dict:
    settings = {
        "ARCH": "pref_only",
        "SEED_LIST": [42],
        "WINDOW_DAYS": None,
        "MIN_USER_INTER": 1,
        "MIN_ITEM_INTER": 1,
        "GRAPH_MODE": "clv_allocated_relation_gate",
        "GRAPH_ALPHA": 0.075,
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
        "models": [
            "m1_baseline",
            RELATION_CONTROL_ID,
            GATE_ID,
            ALLOCATED_CONTROL_ID,
            ALLOCATED_GATE_ID,
        ],
        "historical_clv": "single train-only N_hat * V_hat scalar; no N/V graph split",
        "edge_weights": {
            RELATION_CONTROL_ID: "q_ui from within-user rank of smoothed log observed/expected interaction count",
            GATE_ID: "1 + percentile(CLV_u) * (q_ui - 1)",
            ALLOCATED_CONTROL_ID: "q0_ui from item-adjusted within-user interaction share",
            ALLOCATED_GATE_ID: (
                "1 + percentile(CLV_u) * (qCLV_ui - 1), where qCLV uses "
                "item-adjusted CLV_u * within-user interaction share"
            ),
        },
        "target_propagation_strength": 0.075,
        "shared_invariants": {
            "edge_set": "same unique train user-item pairs as M1",
            "loss": "plain BPR without sample weights",
            "negative_sampling": "uniform",
            "min_user_inter": 1,
            "min_item_inter": 1,
            "no_m2_embedding": True,
            "no_m4_loss_weight": True,
        },
        "screen": {
            "all_accuracy_ratios_vs_m1": ">= 0.99",
            "purchase_value_weighted_hit_at_10": "> M1 and > matching CLV-free control",
            "recommended_mean_price_percentile_ratio": "0.97 to 1.03 vs M1",
            "distinct_items_at_10_ratio": ">= 0.95 vs M1",
            "top10_exposure_share_increase": "<= 0.01 absolute vs M1",
        },
        "additional_controls": "degree-stratified CLV permutation only after a positive screen",
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
        hits = k * normalized[f"precision@{k}"]
        normalized[f"mean_hits@{k}"] = hits
        normalized[f"hit_value@{k}"] = np.where(
            hits > 0, normalized[f"revenue@{k}"] / hits, np.nan
        )
    return normalized


def _proposal_decision(
    model: pd.Series, baseline: pd.Series, control: pd.Series
) -> dict:
    ratios = {
        metric: float(model[metric] / baseline[metric])
        for metric in ACCURACY_METRICS
    }
    weighted_hit_delta = float(model["revenue@10"] - baseline["revenue@10"])
    control_delta = float(model["revenue@10"] - control["revenue@10"])
    price_ratio = float(model["arp@10"] / baseline["arp@10"])
    distinct_ratio = float(model["n_distinct@10"] / baseline["n_distinct@10"])
    top10_delta = float(model["top10_share@10"] - baseline["top10_share@10"])
    guards = {
        "accuracy": bool(all(value >= 0.99 for value in ratios.values())),
        "weighted_hit_vs_m1": bool(weighted_hit_delta > 0),
        "weighted_hit_vs_clv_free_control": bool(control_delta > 0),
        "recommended_price_percentile": bool(0.97 <= price_ratio <= 1.03),
        "distinct_items": bool(distinct_ratio >= 0.95),
        "top10_exposure_share": bool(top10_delta <= 0.01),
    }
    return {
        "passes_screen": bool(all(guards.values())),
        "guards": guards,
        "accuracy_ratios_vs_m1": ratios,
        "weighted_hit_at_10_delta_vs_m1": weighted_hit_delta,
        "weighted_hit_at_10_delta_vs_clv_free_control": control_delta,
        "recommended_mean_price_percentile_ratio_vs_m1": price_ratio,
        "distinct_items_at_10_ratio_vs_m1": distinct_ratio,
        "top10_exposure_share_delta_vs_m1": top10_delta,
        "mean_hits_at_10_delta_vs_m1": float(
            model["mean_hits@10"] - baseline["mean_hits@10"]
        ),
        "hit_value_at_10_delta_vs_m1": float(
            model["hit_value@10"] - baseline["hit_value@10"]
        ),
    }


def screening_decision(frame: pd.DataFrame) -> dict:
    val = frame[frame["split"].eq("val")]

    def row(model_id: str) -> pd.Series:
        selected = val[val["model_id"].eq(model_id)]
        if selected.empty:
            raise ValueError(f"validation result is missing {model_id}")
        return selected.select_dtypes(include=[np.number]).mean()

    baseline = row("m1_baseline")
    relation_control = row(RELATION_CONTROL_ID)
    gate = _proposal_decision(row(GATE_ID), baseline, relation_control)
    allocated_control = row(ALLOCATED_CONTROL_ID)
    allocated = _proposal_decision(
        row(ALLOCATED_GATE_ID), baseline, allocated_control
    )
    return {
        "success": bool(gate["passes_screen"] or allocated["passes_screen"]),
        "clv_as_mixture_gate": gate,
        "clv_in_edge_and_gate": allocated,
        "note": (
            "Validation screening only. The code field revenue is a purchase-value-"
            "weighted hit metric, not actual revenue. hit_value uses the exact identity "
            "revenue@K = (K * precision@K) * hit_value@K. No population significance "
            "is claimed without the paired intervals saved by the native runs."
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


def _fingerprint(cfg: dict) -> str:
    payload = {
        "code_version": CODE_VERSION,
        "v3_code_version": v3.CODE_VERSION,
        "dataset": cfg["DATASET"],
        "seed_list": cfg["SEED_LIST"],
        "target_propagation_strength": cfg["GRAPH_ALPHA"],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()[:12]


def run_experiment(cfg: dict | None = None) -> pd.DataFrame:
    cfg = configure_m3_clv_relation_dunnhumby_run() if cfg is None else dict(cfg)
    summary = preflight_summary(cfg)
    print("M3 scalar-CLV relation graph preflight:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    modes = (
        "relation_only",
        "clv_relation_gate",
        "allocated_relation_only",
        "clv_allocated_relation_gate",
    )
    frames, paths = {}, {}
    for mode in modes:
        frames[mode] = _run_mode(cfg, mode)
        paths[mode] = _native_result_paths()

    frame = pd.concat(
        [
            frames[modes[0]][frames[modes[0]]["model_id"].eq("m1_baseline")],
            *[
                frames[mode][frames[mode]["model_id"].eq(f"m3_{mode}")]
                for mode in modes
            ],
        ],
        ignore_index=True,
    )
    decision = screening_decision(frame)
    out = Path(cfg["OUT_DIR"])
    out.mkdir(parents=True, exist_ok=True)
    fingerprint = _fingerprint(cfg)
    csv_path = out / f"m3_clv_relation_comparison_{fingerprint}.csv"
    json_path = out / f"m3_clv_relation_comparison_{fingerprint}.json"
    frame.to_csv(csv_path, index=False, float_format="%.8f")
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "preflight": summary,
                "screening_decision": decision,
                "absolute_rows": frame.to_dict("records"),
                "native_result_paths": paths,
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
        "native": paths,
    }
    print("M3 scalar-CLV relation graph screening decision:")
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    print("result files:", frame.attrs["result_paths"])
    return frame


if __name__ == "__main__":
    print(
        json.dumps(
            preflight_summary(configure_m3_clv_relation_dunnhumby_run()),
            ensure_ascii=False,
            indent=2,
        )
    )
