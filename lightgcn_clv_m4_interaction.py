"""Dunnhumby validation runner for the CLV-conditioned M4 experiment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

import lightgcn_clv_v3 as v3


CODE_VERSION = "m4-clv-conditioned-interaction-v1"
LOSS_MODES = ("user", "pair_contribution", "clv_pair")
LOSS_STRENGTHS = {
    "user": 1.0,
    "pair_contribution": 0.25,
    "clv_pair": 0.25,
}
MODEL_IDS = tuple(f"m4_{mode}" for mode in LOSS_MODES)
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
        return "/content/drive/MyDrive/논문/data/results_m4_interaction_dunnhumby"
    return str(
        Path(v3.default_out_dir("dunnhumby")).with_name(
            "results_m4_interaction_dunnhumby"
        )
    )


def validate_screening_config(cfg: dict) -> None:
    expected = {
        "DATASET": "dunnhumby",
        "SEED_LIST": [42],
        "ARCH": "pref_only",
        "GRAPH_MODE": "binary",
        "LOSS_MODE": "plain",
        "LOSS_MODES": LOSS_MODES,
        "LOSS_STRENGTHS": LOSS_STRENGTHS,
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
        raise ValueError("OUT_DIR must identify the Dunnhumby M4 run")


def configure_m4_interaction_dunnhumby_run(
    *, out_dir: str | None = None, **overrides
) -> dict:
    settings = {
        "ARCH": "pref_only",
        "SEED_LIST": [42],
        "WINDOW_DAYS": None,
        "MIN_USER_INTER": 1,
        "MIN_ITEM_INTER": 1,
        "GRAPH_MODE": "binary",
        "LOSS_MODE": "plain",
        "LOSS_MODES": LOSS_MODES,
        "LOSS_STRENGTHS": dict(LOSS_STRENGTHS),
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
            "task": "new-item recommendation; train pairs excluded",
            "graph": "binary M1 graph",
            "architecture": "pref_only LightGCN",
            "negative_sampling": "uniform",
            "only_intervention": "positive BPR row weight",
        },
        "arms": {
            "m4_user": "historical CLV user weight; lambda=1.0",
            "m4_pair_contribution": (
                "within-user rank of mean positive-item basket-value share; "
                "beta=0.25; no CLV"
            ),
            "m4_clv_pair": (
                "historical CLV percentile x within-user positive-item "
                "basket-value contribution rank; beta=0.25"
            ),
        },
        "fixed_guards": {
            "accuracy": "all Recall/NDCG @10/20/50 >= 99% of M1",
            "weighted_hit": "price/purchase-amount weighted hit@10 > M1",
            "distinct_and_effective_catalog": ">= 95% of M1",
            "top10_exposure_share": "<= 105% of M1",
            "clv_identification": "m4_clv_pair weighted hit@10 > pair control",
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
    baseline = val[val["model_id"].eq("m1_baseline")]
    if baseline.empty:
        raise ValueError("validation result must contain m1_baseline")
    required = list(ACCURACY_METRICS) + [
        "revenue@10",
        "n_distinct@10",
        "eff_catalog@10",
        "top10_share@10",
    ]
    base = baseline[required].mean()
    decisions = {}
    for model_id in MODEL_IDS:
        model = val[val["model_id"].eq(model_id)]
        if model.empty:
            raise ValueError(f"validation result must contain {model_id}")
        current = model[required].mean()
        accuracy_ratios = {
            metric: float(current[metric] / base[metric])
            for metric in ACCURACY_METRICS
        }
        distinct_ratio = float(current["n_distinct@10"] / base["n_distinct@10"])
        effective_ratio = float(current["eff_catalog@10"] / base["eff_catalog@10"])
        top10_ratio = float(current["top10_share@10"] / base["top10_share@10"])
        economic_delta = float(current["revenue@10"] - base["revenue@10"])
        accuracy_pass = all(value >= 0.99 for value in accuracy_ratios.values())
        exposure_pass = (
            distinct_ratio >= 0.95
            and effective_ratio >= 0.95
            and top10_ratio <= 1.05
        )
        decisions[model_id] = {
            "passes_m1_screen": bool(
                accuracy_pass and exposure_pass and economic_delta > 0.0
            ),
            "accuracy_pass": bool(accuracy_pass),
            "exposure_pass": bool(exposure_pass),
            "weighted_hit@10_improved_vs_m1": bool(economic_delta > 0.0),
            "revenue@10_delta_vs_m1": economic_delta,
            "accuracy_ratios_vs_m1": accuracy_ratios,
            "n_distinct@10_ratio_vs_m1": distinct_ratio,
            "eff_catalog@10_ratio_vs_m1": effective_ratio,
            "top10_share@10_ratio_vs_m1": top10_ratio,
        }
    main = val[val["model_id"].eq("m4_clv_pair")]["revenue@10"].mean()
    control = val[val["model_id"].eq("m4_pair_contribution")]["revenue@10"].mean()
    control_delta = float(main - control)
    clv_specific = bool(
        decisions["m4_clv_pair"]["passes_m1_screen"] and control_delta > 0.0
    )
    return {
        "clv_specific_candidate": clv_specific,
        "main_revenue@10_delta_vs_pair_control": control_delta,
        "arms": decisions,
        "next_step": (
            "run the fixed shuffled-user control only if clv_specific_candidate "
            "is true; otherwise retain the negative validation result"
        ),
        "note": (
            "revenue@10 is a price/purchase-amount weighted recommendation hit, "
            "not actual or incremental revenue"
        ),
    }


def _activate(cfg: dict, loss_mode: str) -> None:
    validate_screening_config(cfg)
    if loss_mode not in LOSS_MODES:
        raise ValueError(f"unsupported loss mode: {loss_mode}")
    settings = {
        key: value
        for key, value in cfg.items()
        if key not in {"DATASET", "OUT_DIR", "LOSS_MODES", "LOSS_STRENGTHS"}
    }
    settings["LOSS_MODE"] = loss_mode
    settings["LOSS_LAMBDA"] = cfg["LOSS_STRENGTHS"][loss_mode]
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
        "loss_modes": cfg["LOSS_MODES"],
        "loss_strengths": cfg["LOSS_STRENGTHS"],
        "out_dir": cfg["OUT_DIR"],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()[:12]


def run_experiment(cfg: dict | None = None) -> pd.DataFrame:
    cfg = configure_m4_interaction_dunnhumby_run() if cfg is None else dict(cfg)
    validate_screening_config(cfg)
    summary = preflight_summary(cfg)
    print("M4 CLV-conditioned interaction screening preflight:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    frames = []
    native_paths = {}
    weight_diagnostics = {}
    for loss_mode in LOSS_MODES:
        print(f"\n{'=' * 84}\nM4 {loss_mode} validation\n{'=' * 84}")
        _activate(cfg, loss_mode)
        current = normalize_result_schema(v3.main())
        current = current[current["split"].eq("val")].copy()
        model_id = f"m4_{loss_mode}"
        frames.append(current[current["model_id"].eq(model_id)])
        if loss_mode == LOSS_MODES[0]:
            frames.append(current[current["model_id"].eq("m1_baseline")])
        paths = _native_result_paths()
        native_paths[loss_mode] = paths
        with Path(paths["json"]).open(encoding="utf-8") as handle:
            weight_diagnostics[model_id] = json.load(handle).get(
                "loss_weight_diagnostics", {}
            )

    frame = pd.concat(frames, ignore_index=True)
    frame = frame.drop_duplicates(["seed", "model_id", "split", "lambda"])
    decision = screening_decision(frame)
    out = Path(cfg["OUT_DIR"])
    out.mkdir(parents=True, exist_ok=True)
    fingerprint = _result_fingerprint(cfg)
    csv_path = out / f"m4_interaction_comparison_{fingerprint}.csv"
    json_path = out / f"m4_interaction_comparison_{fingerprint}.json"
    frame.to_csv(csv_path, index=False, float_format="%.8f")
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "preflight": summary,
                "screening_decision": decision,
                "loss_weight_diagnostics": weight_diagnostics,
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
    frame.attrs["loss_weight_diagnostics"] = weight_diagnostics
    frame.attrs["result_paths"] = {
        "csv": str(csv_path),
        "json": str(json_path),
        "native": native_paths,
    }
    print("\nM4 screening decision:")
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    print("result files:", frame.attrs["result_paths"])
    return frame


if __name__ == "__main__":
    print(
        json.dumps(
            preflight_summary(configure_m4_interaction_dunnhumby_run()),
            ensure_ascii=False,
            indent=2,
        )
    )
