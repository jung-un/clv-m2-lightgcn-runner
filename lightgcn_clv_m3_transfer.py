"""Fast Dunnhumby validation runner for revised CLV-informed M3 graphs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

import lightgcn_clv_v3 as v3


CODE_VERSION = "m3-clv-transfer-graph-v1"
GRAPH_MODES = ("n_transfer", "v_contribution")
MODEL_IDS = tuple(f"m3_{mode}" for mode in GRAPH_MODES)
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
        return "/content/drive/MyDrive/논문/data/results_m3_transfer_dunnhumby"
    return str(
        Path(v3.default_out_dir("dunnhumby")).with_name(
            "results_m3_transfer_dunnhumby"
        )
    )


def validate_screening_config(cfg: dict) -> None:
    expected = {
        "DATASET": "dunnhumby",
        "SEED_LIST": [42],
        "ARCH": "pref_only",
        "GRAPH_MODES": GRAPH_MODES,
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


def configure_m3_transfer_dunnhumby_run(
    *, out_dir: str | None = None, **overrides
) -> dict:
    """Create the fixed seed-42, validation-only M1/N/V comparison."""
    settings = {
        "ARCH": "pref_only",
        "SEED_LIST": [42],
        "WINDOW_DAYS": None,
        "MIN_USER_INTER": 1,
        "MIN_ITEM_INTER": 1,
        "GRAPH_MODES": GRAPH_MODES,
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
        "models": ["m1_baseline", *MODEL_IDS],
        "common_conditions": {
            "unique_edge_set": "same train user-item edges as binary M1",
            "graph": "only normalized propagation weights differ",
            "loss": "plain BPR",
            "negative_sampling": "uniform",
        },
        "n_transfer": (
            "customer activity percentile x smoothed category repeatability"
        ),
        "v_contribution": (
            "customer mean-basket-value percentile x mean item basket-value share"
        ),
        "weighting": (
            "rank signal -> exp(beta*z)/mean; beta<=0.25; N/V effective "
            "normalized-propagation strength matched using train only"
        ),
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
    baseline = val[val["model_id"].eq("m1_baseline")]
    if baseline.empty:
        raise ValueError("validation result must contain m1_baseline")
    base = baseline[list(ACCURACY_METRICS) + ["revenue@10"]].mean()
    decisions = {}
    for model_id in MODEL_IDS:
        model = val[val["model_id"].eq(model_id)]
        if model.empty:
            raise ValueError(f"validation result must contain {model_id}")
        current = model[list(ACCURACY_METRICS) + ["revenue@10"]].mean()
        ratios = {
            metric: float(current[metric] / base[metric])
            for metric in ACCURACY_METRICS
        }
        economic_delta = float(current["revenue@10"] - base["revenue@10"])
        accuracy_pass = all(value >= 0.99 for value in ratios.values())
        decisions[model_id] = {
            "success": bool(accuracy_pass and economic_delta > 0.0),
            "economic_improved_vs_m1": bool(economic_delta > 0.0),
            "revenue@10_delta": economic_delta,
            "accuracy_ratios_vs_m1": ratios,
        }
    return {
        "any_axis_success": any(row["success"] for row in decisions.values()),
        "arms": decisions,
        "next_step": (
            "combine N and V only if an individual arm passes; otherwise stop M3 "
            "screening and retain the negative result"
        ),
        "note": "This is a post-hoc validation reading rule, not a training constraint.",
    }


def _activate(cfg: dict, graph_mode: str) -> None:
    validate_screening_config(cfg)
    if graph_mode not in GRAPH_MODES:
        raise ValueError(f"unsupported graph mode: {graph_mode}")
    settings = {
        key: value
        for key, value in cfg.items()
        if key not in {"DATASET", "OUT_DIR", "GRAPH_MODES"}
    }
    settings["GRAPH_MODE"] = graph_mode
    v3.configure_run(cfg["DATASET"], out_dir=cfg["OUT_DIR"], **settings)


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
        "dataset": cfg["DATASET"],
        "seed_list": cfg["SEED_LIST"],
        "graph_modes": cfg["GRAPH_MODES"],
        "out_dir": cfg["OUT_DIR"],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()[:12]


def run_experiment(cfg: dict | None = None) -> pd.DataFrame:
    cfg = configure_m3_transfer_dunnhumby_run() if cfg is None else dict(cfg)
    validate_screening_config(cfg)
    summary = preflight_summary(cfg)
    print("M3 transfer-graph screening preflight:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    frames = []
    native_paths = {}
    for graph_mode in GRAPH_MODES:
        print(f"\n{'=' * 84}\nM3 {graph_mode} validation\n{'=' * 84}")
        _activate(cfg, graph_mode)
        current = normalize_result_schema(v3.main())
        current = current[current["split"].eq("val")].copy()
        frames.append(current[current["model_id"].eq(f"m3_{graph_mode}")])
        if graph_mode == GRAPH_MODES[0]:
            frames.append(current[current["model_id"].eq("m1_baseline")])
        native_paths[graph_mode] = _native_result_paths()

    frame = pd.concat(frames, ignore_index=True)
    frame = frame.drop_duplicates(["seed", "model_id", "split", "lambda"])
    decision = screening_decision(frame)
    out = Path(cfg["OUT_DIR"])
    out.mkdir(parents=True, exist_ok=True)
    fingerprint = _result_fingerprint(cfg)
    csv_path = out / f"m3_transfer_comparison_{fingerprint}.csv"
    json_path = out / f"m3_transfer_comparison_{fingerprint}.json"
    frame.to_csv(csv_path, index=False, float_format="%.8f")
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
    frame.attrs["result_paths"] = {
        "csv": str(csv_path),
        "json": str(json_path),
        "native": native_paths,
    }
    print("\nM3 transfer-graph screening decision:")
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    print("result files:", frame.attrs["result_paths"])
    return frame


if __name__ == "__main__":
    print(json.dumps(preflight_summary(configure_m3_transfer_dunnhumby_run()),
                     ensure_ascii=False, indent=2))
