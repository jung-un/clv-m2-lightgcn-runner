"""Dunnhumby seed-42 validation screen for direct CLV message redistribution."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import lightgcn_clv_v3 as v3
from clv_m3_mass_preserving_graph import (
    DEFAULT_SHUFFLE_SEED,
    MODES,
    build_directional_torch_adj,
    build_mass_preserving_clv_graph,
)


CODE_VERSION = "m3-direct-clv-item-message-redistribution-v1"
MODEL_IDS = {
    "n_only": "m3_n_only_influence",
    "v_only": "m3_v_only_influence",
    "clv": "m3_clv_influence",
    "clv_shuffle": "m3_clv_influence_shuffle",
}
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
        return "/content/drive/MyDrive/논문/data/results_m3_clv_influence_dunnhumby"
    return str(
        Path(v3.default_out_dir("dunnhumby")).with_name(
            "results_m3_clv_influence_dunnhumby"
        )
    )


def validate_screening_config(cfg: dict) -> None:
    expected = {
        "DATASET": "dunnhumby",
        "SEED_LIST": [42],
        "ARCH": "pref_only",
        "GRAPH_MODE": "clv",
        "LOSS_MODE": "plain",
        "NEG_MODE": "uniform",
        "WINDOW_DAYS": None,
        "MIN_USER_INTER": 1,
        "MIN_ITEM_INTER": 1,
        "GATE_MODE": "none",
        "EVAL_TEST": False,
        "EVAL_HOLDOUT": False,
    }
    for key, wanted in expected.items():
        if cfg.get(key) != wanted:
            raise ValueError(f"screening config requires {key}={wanted!r}")
    if not cfg.get("OUT_DIR") or "dunnhumby" not in str(cfg["OUT_DIR"]):
        raise ValueError("OUT_DIR must identify the Dunnhumby M3 run")


def configure_m3_clv_influence_dunnhumby_run(
    *, out_dir: str | None = None, **overrides
) -> dict:
    settings = {
        "ARCH": "pref_only",
        "SEED_LIST": [42],
        "WINDOW_DAYS": None,
        "MIN_USER_INTER": 1,
        "MIN_ITEM_INTER": 1,
        "GRAPH_MODE": "clv",
        "GRAPH_ALPHA": 1.0,  # runner compatibility only; the method has no alpha
        "LOSS_MODE": "plain",
        "NEG_MODE": "uniform",
        "GATE_MODE": "none",
        "REPORT_LEGACY_VALUE_FEATURES": False,
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
        "models": ["m1_baseline", *MODEL_IDS.values()],
        "customer_value": {
            "n_hat": "number of train transactions/baskets",
            "v_hat": "mean train transaction/basket value",
            "clv_proxy": "n_hat * v_hat",
            "factor": "mean-one 0.5 + train-user percentile; no alpha/beta",
        },
        "propagation": {
            "user_from_item": "unchanged M1 symmetric-normalized coefficients",
            "item_from_user": (
                "M1 coefficients redistributed within each item by the user factor"
            ),
            "identity": "factor == 1 gives the exact M1 operator",
            "mass_constraint": "each item's incoming coefficient sum equals M1",
        },
        "controls": {
            MODEL_IDS["n_only"]: "same operator with percentile(n_hat)",
            MODEL_IDS["v_only"]: "same operator with percentile(v_hat)",
            MODEL_IDS["clv_shuffle"]: (
                f"same CLV factors permuted across train users; seed={DEFAULT_SHUFFLE_SEED}"
            ),
        },
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
            "purchase_value_weighted_hit_at_10": (
                "> M1, N-only, V-only, and shuffled-CLV control"
            ),
            "recommended_mean_price_percentile_ratio": "0.97 to 1.03 vs M1",
            "distinct_items_at_10_ratio": ">= 0.95 vs M1",
            "top10_exposure_share_increase": "<= 0.01 absolute vs M1",
        },
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


def screening_decision(frame: pd.DataFrame) -> dict:
    val = frame[frame["split"].eq("val")]

    def row(model_id: str) -> pd.Series:
        selected = val[val["model_id"].eq(model_id)]
        if selected.empty:
            raise ValueError(f"validation result is missing {model_id}")
        return selected.select_dtypes(include=[np.number]).mean()

    baseline = row("m1_baseline")
    proposed = row(MODEL_IDS["clv"])
    controls = {
        mode: row(MODEL_IDS[mode]) for mode in ("n_only", "v_only", "clv_shuffle")
    }
    ratios = {
        metric: float(proposed[metric] / baseline[metric])
        for metric in ACCURACY_METRICS
    }
    control_deltas = {
        mode: float(proposed["revenue@10"] - control["revenue@10"])
        for mode, control in controls.items()
    }
    price_ratio = float(proposed["arp@10"] / baseline["arp@10"])
    distinct_ratio = float(
        proposed["n_distinct@10"] / baseline["n_distinct@10"]
    )
    top10_delta = float(proposed["top10_share@10"] - baseline["top10_share@10"])
    guards = {
        "accuracy": bool(all(value >= 0.99 for value in ratios.values())),
        "weighted_hit_vs_m1": bool(proposed["revenue@10"] > baseline["revenue@10"]),
        "weighted_hit_vs_n_only": bool(control_deltas["n_only"] > 0),
        "weighted_hit_vs_v_only": bool(control_deltas["v_only"] > 0),
        "weighted_hit_vs_shuffled_clv": bool(control_deltas["clv_shuffle"] > 0),
        "recommended_price_percentile": bool(0.97 <= price_ratio <= 1.03),
        "distinct_items": bool(distinct_ratio >= 0.95),
        "top10_exposure_share": bool(top10_delta <= 0.01),
    }
    return {
        "success": bool(all(guards.values())),
        "guards": guards,
        "accuracy_ratios_vs_m1": ratios,
        "weighted_hit_at_10_delta_vs_m1": float(
            proposed["revenue@10"] - baseline["revenue@10"]
        ),
        "weighted_hit_at_10_delta_vs_controls": control_deltas,
        "recommended_mean_price_percentile_ratio_vs_m1": price_ratio,
        "distinct_items_at_10_ratio_vs_m1": distinct_ratio,
        "top10_exposure_share_delta_vs_m1": top10_delta,
        "mean_hits_at_10_delta_vs_m1": float(
            proposed["mean_hits@10"] - baseline["mean_hits@10"]
        ),
        "hit_value_at_10_delta_vs_m1": float(
            proposed["hit_value@10"] - baseline["hit_value@10"]
        ),
        "note": (
            "Validation screening only. The code field revenue is a purchase-value-"
            "weighted hit metric, not actual revenue. No population significance is "
            "claimed from this one-seed screen."
        ),
    }


def _prepare_with_redistribution(original_prepare, mode: str):
    def wrapped(cfg: dict, dcfg: dict) -> dict:
        binary_cfg = dict(cfg)
        binary_cfg["GRAPH_MODE"] = "binary"
        data = original_prepare(binary_cfg, dcfg)
        graph = build_mass_preserving_clv_graph(
            data["train"], data["n_users"], data["n_items"]
        )
        expected_users = (data["pos_key"] // data["n_items"]).astype(np.int64)
        expected_items = (data["pos_key"] % data["n_items"]).astype(np.int64)
        if not (
            np.array_equal(graph.edge_users, expected_users)
            and np.array_equal(graph.edge_items, expected_items)
        ):
            raise RuntimeError("CLV redistribution edge order differs from M1")
        data["adj"] = build_directional_torch_adj(
            graph, mode, data["n_users"], data["n_items"], v3.DEVICE
        )
        data["w_edge"] = graph.user_factors[mode][graph.edge_users]
        data["clv"] = graph.clv_proxy
        data["vhat"] = graph.v_hat
        data["m3_mass_preserving_graph"] = graph
        data["data_stats"]["m3_mass_preserving_graph"] = {
            **graph.diagnostics,
            "selected_mode": mode,
        }
        mode_diag = graph.diagnostics["modes"][mode]
        print(
            f"  M3 direct CLV influence ({mode}): edges {len(graph.edge_users):,} | "
            f"factor std {mode_diag['user_factor']['std']:.4f} | "
            "item factor Kish median "
            f"{mode_diag['item_kish_ratio_from_user_factor_median']:.4f} | "
            f"max item-mass error {mode_diag['max_item_mass_abs_error']:.3e}"
        )
        return data

    return wrapped


def _run_mode(cfg: dict, mode: str) -> pd.DataFrame:
    current = dict(cfg)
    current["GRAPH_MODE"] = mode
    settings = {
        key: value for key, value in current.items() if key not in {"DATASET", "OUT_DIR"}
    }
    v3.configure_run(current["DATASET"], out_dir=current["OUT_DIR"], **settings)
    original_prepare = v3.prepare_data
    v3.prepare_data = _prepare_with_redistribution(original_prepare, mode)
    try:
        return normalize_result_schema(v3.main())
    finally:
        v3.prepare_data = original_prepare


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
        "shuffle_seed": DEFAULT_SHUFFLE_SEED,
        "modes": MODES,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()[:12]


def run_experiment(cfg: dict | None = None) -> pd.DataFrame:
    cfg = configure_m3_clv_influence_dunnhumby_run() if cfg is None else dict(cfg)
    summary = preflight_summary(cfg)
    print("M3 direct CLV influence preflight:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    frames, paths = {}, {}
    for mode in MODES:
        frames[mode] = _run_mode(cfg, mode)
        paths[mode] = _native_result_paths()

    frame = pd.concat(
        [
            frames[MODES[0]][frames[MODES[0]]["model_id"].eq("m1_baseline")],
            *[
                frames[mode][frames[mode]["model_id"].eq(f"m3_{mode}")].assign(
                    model_id=MODEL_IDS[mode]
                )
                for mode in MODES
            ],
        ],
        ignore_index=True,
    )
    decision = screening_decision(frame)
    out = Path(cfg["OUT_DIR"])
    out.mkdir(parents=True, exist_ok=True)
    fingerprint = _fingerprint(cfg)
    csv_path = out / f"m3_clv_influence_comparison_{fingerprint}.csv"
    json_path = out / f"m3_clv_influence_comparison_{fingerprint}.json"
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
    print("M3 direct CLV influence screening decision:")
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    print("result files:", frame.attrs["result_paths"])
    return frame


if __name__ == "__main__":
    print(
        json.dumps(
            preflight_summary(configure_m3_clv_influence_dunnhumby_run()),
            ensure_ascii=False,
            indent=2,
        )
    )
