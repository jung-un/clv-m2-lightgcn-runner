"""Frozen seed-43/44 validation for the dual-axis CLV M2 model."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

import numpy as np
import pandas as pd

import lightgcn_clv_dual as dual
import lightgcn_clv_dual_checkpoint_diagnostic as checkpoint_diagnostic
import lightgcn_clv_moe as moe
import lightgcn_clv_v3 as v3


CODE_VERSION = "clv-dual-multiseed-validation-v1.0"
ACCURACY_METRICS = tuple(
    f"{metric}@{k}" for metric in ("recall", "ndcg") for k in (10, 20, 50)
)


@dataclass(frozen=True)
class MultiSeedValidationConfig:
    dataset: str
    seed42_result_json: str
    window_days: int | None
    gate_shape: str
    fixed_lambda: float
    new_seeds: tuple[int, ...] = (43, 44)
    model_ids: tuple[str, ...] = ("m1", "dual_clv_fixed")
    eval_test: bool = False
    eval_holdout: bool = False
    out_dir: str | None = None


def configure_multiseed_validation(
    dataset: str,
    seed42_result_json: str | Path,
    *,
    short_hm: bool = False,
    out_dir: str | Path | None = None,
) -> MultiSeedValidationConfig:
    dataset = dataset.lower()
    if dataset == "dunnhumby":
        if short_hm:
            raise ValueError("short_hm은 H&M에서만 사용합니다")
        window_days, gate_shape, fixed_lambda = None, "equal", 2.0
    elif dataset == "hm":
        if not short_hm:
            raise ValueError("이 runner는 H&M 60일 validation만 허용합니다")
        window_days, gate_shape, fixed_lambda = 60, "high", 1.0
    else:
        raise ValueError("dataset은 dunnhumby 또는 hm이어야 합니다")
    default_out = Path(seed42_result_json).parent / "multiseed_validation"
    return validate_multiseed_config(
        MultiSeedValidationConfig(
            dataset=dataset,
            seed42_result_json=str(seed42_result_json),
            window_days=window_days,
            gate_shape=gate_shape,
            fixed_lambda=fixed_lambda,
            out_dir=str(out_dir or default_out),
        )
    )


def validate_multiseed_config(
    cfg: MultiSeedValidationConfig,
) -> MultiSeedValidationConfig:
    if tuple(cfg.new_seeds) != (43, 44):
        raise ValueError("추가 validation seed는 정확히 43, 44입니다")
    if cfg.eval_test or cfg.eval_holdout:
        raise ValueError("multiseed runner는 validation-only입니다")
    if tuple(cfg.model_ids) != ("m1", "dual_clv_fixed"):
        raise ValueError("이번 단계는 M1과 dual_clv_fixed 두 모형만 허용합니다")
    if cfg.dataset == "dunnhumby":
        if cfg.window_days is not None:
            raise ValueError("Dunnhumby는 전체 관찰기간 설정입니다")
        if cfg.gate_shape != "equal" or not np.isclose(cfg.fixed_lambda, 2.0):
            raise ValueError("Dunnhumby 동결 운영점은 equal, lambda=2.0입니다")
    elif cfg.dataset == "hm":
        if cfg.window_days != 60:
            raise ValueError("이 runner의 H&M 범위는 60일뿐입니다")
        if cfg.gate_shape != "high" or not np.isclose(cfg.fixed_lambda, 1.0):
            raise ValueError("H&M 60일 동결 운영점은 high, lambda=1.0입니다")
    else:
        raise ValueError("dataset은 dunnhumby 또는 hm이어야 합니다")
    return cfg


def reproducibility_decision(absolute_rows: pd.DataFrame) -> dict:
    table = pd.DataFrame(absolute_rows).copy()
    if set(table.seed) != {42, 43, 44}:
        raise ValueError("재현성 판정에는 seed 42, 43, 44가 모두 필요합니다")
    baseline = table[table.model_id.eq("m1")].set_index("seed")
    model = table[table.model_id.eq("dual_clv_fixed")].set_index("seed")
    if set(baseline.index) != {42, 43, 44} or set(model.index) != {42, 43, 44}:
        raise ValueError("각 seed에 M1과 dual_clv_fixed가 하나씩 필요합니다")
    revenue_delta = model["revenue@10"] - baseline["revenue@10"]
    mean_revenue_delta = float(revenue_delta.mean())
    positive_count = int((revenue_delta > 0).sum())
    accuracy_ratios = {
        metric: float(model[metric].mean() / baseline[metric].mean())
        for metric in ACCURACY_METRICS
    }
    conditions = {
        "mean_revenue_delta_positive": mean_revenue_delta > 0,
        "positive_revenue_in_at_least_two_seeds": positive_count >= 2,
        "six_accuracy_mean_ratios_at_least_0.99": all(
            ratio >= 0.99 for ratio in accuracy_ratios.values()
        ),
    }
    return {
        "success": all(conditions.values()),
        "conditions": conditions,
        "failed_conditions": [
            name for name, passed in conditions.items() if not passed
        ],
        "mean_revenue_delta": mean_revenue_delta,
        "positive_revenue_seed_count": positive_count,
        "seed_revenue_delta": {
            str(seed): float(value) for seed, value in revenue_delta.items()
        },
        "accuracy_mean_ratios": accuracy_ratios,
    }


def _load_and_validate_seed42_payload(cfg: MultiSeedValidationConfig) -> dict:
    path = Path(cfg.seed42_result_json)
    if not path.is_file():
        raise FileNotFoundError(f"seed 42 원본 결과가 없습니다: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    checkpoint_diagnostic._validate_payload(payload)
    source_cfg = payload["config"]
    expected_window = cfg.window_days
    selected = payload.get("selected_operating_point", {})
    if source_cfg.get("dataset") != cfg.dataset:
        raise RuntimeError("seed 42 결과의 dataset이 재현성 preset과 다릅니다")
    if source_cfg.get("window_days") != expected_window:
        raise RuntimeError("seed 42 결과의 관찰기간이 재현성 preset과 다릅니다")
    if selected.get("gate_shape") != cfg.gate_shape or not np.isclose(
        float(selected.get("lambda", -1)), cfg.fixed_lambda
    ):
        raise RuntimeError("seed 42 결과의 선택 운영점이 동결 preset과 다릅니다")
    checkpoint_diagnostic._verify_checkpoints(payload)
    return payload


def _seed42_context(cfg, payload):
    paths = checkpoint_diagnostic._verify_checkpoints(payload)
    base_cfg = checkpoint_diagnostic._configure_base(payload, paths["m1_s42"])
    manifest = checkpoint_diagnostic._validate_inputs(payload)
    data = v3.prepare_data(base_cfg, v3.DCFG)
    x_item, item_cat = v3.item_value_features(data["train"], data["n_items"])
    meta = v3.item_meta(data["train"], data["n_items"])
    base, model, clv_proxy, baseline_hash = checkpoint_diagnostic._load_models(
        payload, paths, data, base_cfg, x_item, item_cat
    )
    thresholds = v3.segment_thresholds(clv_proxy, base_cfg["SEG_EDGES"])
    cache = v3.EvalCache(
        *data["splits"]["val"], clv_proxy, thresholds, data["n_items"]
    )
    return {
        "paths": paths,
        "manifest": manifest,
        "base_cfg": base_cfg,
        "data": data,
        "meta": meta,
        "cache": cache,
        "base": base,
        "model": model,
        "baseline_hash": baseline_hash,
    }


def _load_seed42_evaluation(cfg, payload):
    context = _seed42_context(cfg, payload)
    baseline_flat, baseline_per_user = moe._flat_evaluation(
        context["base"],
        0.0,
        context["cache"],
        context["meta"],
        context["data"],
        context["base_cfg"],
        per_user=True,
    )
    model = context["model"]
    model.set_eval_axes("n_plus_v")
    model.set_gate_shape(cfg.gate_shape)
    model_flat, model_per_user = moe._flat_evaluation(
        model,
        cfg.fixed_lambda,
        context["cache"],
        context["meta"],
        context["data"],
        context["base_cfg"],
        per_user=True,
    )
    rows = [
        {
            "seed": 42,
            "model_id": "m1",
            "split": "val",
            "gate_shape": "none",
            "lambda": 0.0,
            **baseline_flat,
        },
        {
            "seed": 42,
            "model_id": dual.PRIMARY_MODEL,
            "split": "val",
            "gate_shape": cfg.gate_shape,
            "lambda": cfg.fixed_lambda,
            **model_flat,
        },
    ]
    return {
        "rows": rows,
        "baseline_per_user": baseline_per_user,
        "model_per_user": model_per_user,
        "eval_users": context["cache"].users,
        "checkpoints": {
            name: str(path) for name, path in context["paths"].items()
        },
        "checkpoint_sha256": payload["checkpoint_sha256"],
        "training": payload.get("training", {}),
        "manifest": context["manifest"],
    }


def _run_new_seed(cfg, payload, seed):
    short_hm = cfg.dataset == "hm"
    source_cfg = payload["config"]
    seed_out = Path(cfg.out_dir) / f"seed_{seed}"
    reusable = {
        field.name: source_cfg[field.name]
        for field in fields(moe.MoEConfig)
        if field.name in source_cfg
        and field.name
        not in {
            "dataset",
            "seed_list",
            "window_days",
            "eval_test",
            "eval_holdout",
            "lambda_eval",
            "out_dir",
        }
    }
    screening_cfg = dual.configure_dual_run(
        cfg.dataset,
        short_hm=short_hm,
        out_dir=str(seed_out),
        **reusable,
    )
    run_cfg = replace(
        screening_cfg,
        seed_list=(seed,),
        lambda_eval=(cfg.fixed_lambda,),
        eval_test=False,
        eval_holdout=False,
    )
    prepared = dual._prepare(run_cfg, seed=seed)
    baseline_row = {
        "seed": seed,
        "model_id": "m1",
        "split": "val",
        "gate_shape": "none",
        "lambda": 0.0,
        **prepared["baseline_flat"],
    }
    run = dual._train_variant(
        dual.PRIMARY_MODEL,
        dual._fresh_base(prepared, seed=seed),
        prepared,
        run_cfg,
        seed=seed,
        gate_shapes=(cfg.gate_shape,),
        lambda_eval=(cfg.fixed_lambda,),
    )
    checkpoints = {
        f"m1_s{seed}": prepared["m1_checkpoint"],
        f"encoder_s{seed}": prepared["encoder_checkpoint"],
        f"dual_clv_fixed_s{seed}": run["checkpoint"],
    }
    return {
        "rows": [baseline_row, run["rows"][0]],
        "baseline_per_user": prepared["baseline_per_user"],
        "model_per_user": run["per_user"][(cfg.gate_shape, cfg.fixed_lambda)],
        "eval_users": prepared["cache"].users,
        "checkpoints": checkpoints,
        "checkpoint_sha256": {
            name: moe.file_sha256(path) for name, path in checkpoints.items()
        },
        "training": {dual.PRIMARY_MODEL: run["training"]},
        "manifest": prepared["manifest"],
    }


def _delta_records(results, n_boot=2000):
    records = []
    diffs = {metric: [] for metric in ("recall", "ndcg", "revenue", "arp")}
    expected_users = None
    for result in results:
        users = np.asarray(result["eval_users"])
        if expected_users is None:
            expected_users = users
        elif not np.array_equal(expected_users, users):
            raise RuntimeError("seed 간 validation 사용자 순서가 다릅니다")
        seed = int(result["rows"][0]["seed"])
        for metric in diffs:
            diff = np.asarray(result["model_per_user"][metric], float) - np.asarray(
                result["baseline_per_user"][metric], float
            )
            diffs[metric].append(diff)
            records.append(
                {
                    "scope": "seed",
                    "seed": seed,
                    "metric": metric,
                    **v3.paired_bootstrap([diff], n_boot),
                }
            )
    for metric, metric_diffs in diffs.items():
        records.append(
            {
                "scope": "three_seed",
                "seed": "all",
                "metric": metric,
                **v3.paired_bootstrap(metric_diffs, n_boot),
            }
        )
    return records


def _persist(cfg, payload, results, decision):
    output = Path(cfg.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([row for result in results for row in result["rows"]])
    delta_records = _delta_records(results)
    decision_rows = [
        {"condition": name, "passed": passed}
        for name, passed in decision["conditions"].items()
    ]
    stem = f"clv_dual_multiseed_{cfg.dataset}"
    csv_path = output / f"{stem}.csv"
    delta_path = output / f"{stem}_delta.csv"
    decision_path = output / f"{stem}_decision.csv"
    json_path = output / f"{stem}.json"
    frame.to_csv(csv_path, index=False, float_format="%.8f")
    pd.DataFrame(delta_records).to_csv(delta_path, index=False)
    pd.DataFrame(decision_rows).to_csv(decision_path, index=False)
    checkpoints = {
        name: path for result in results for name, path in result["checkpoints"].items()
    }
    checkpoint_hashes = {
        name: value
        for result in results
        for name, value in result["checkpoint_sha256"].items()
    }
    report = {
        "code_version": CODE_VERSION,
        "source_revision": moe.source_revision(),
        "config": asdict(cfg),
        "original_seed42_result_json": cfg.seed42_result_json,
        "original_seed42_result_fingerprint": payload["result_fingerprint"],
        "original_source_revision": payload.get("source_revision"),
        "input_manifest": payload.get("input_manifest"),
        "checkpoints": checkpoints,
        "checkpoint_sha256": checkpoint_hashes,
        "training": {
            str(result["rows"][0]["seed"]): result["training"]
            for result in results
        },
        "absolute_rows": frame.to_dict("records"),
        "delta_rows": delta_records,
        "reproducibility_decision": decision,
        "interpretation": {
            "validation_only": True,
            "hm_two_year_executed": False,
            "test_executed": False,
            "holdout_executed": False,
            "controls_executed": False,
            "revenue": "price/purchase-amount weighted hit, not incremental revenue",
        },
    }
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    frame.attrs["reproducibility_decision"] = decision
    frame.attrs["result_paths"] = {
        "csv": str(csv_path),
        "delta_csv": str(delta_path),
        "decision_csv": str(decision_path),
        "json": str(json_path),
    }
    return frame


def run_multiseed_validation(
    cfg: MultiSeedValidationConfig,
) -> pd.DataFrame:
    cfg = validate_multiseed_config(cfg)
    payload = _load_and_validate_seed42_payload(cfg)
    results = [_load_seed42_evaluation(cfg, payload)]
    for seed in cfg.new_seeds:
        results.append(_run_new_seed(cfg, payload, seed))
    absolute = pd.DataFrame(
        [row for result in results for row in result["rows"]]
    )
    decision = reproducibility_decision(absolute)
    return _persist(cfg, payload, results, decision)


__all__ = [
    "MultiSeedValidationConfig",
    "configure_multiseed_validation",
    "reproducibility_decision",
    "run_multiseed_validation",
    "validate_multiseed_config",
]
