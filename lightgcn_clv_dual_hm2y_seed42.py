"""H&M full-period seed-42 validation at the frozen normalized M2 point."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

import lightgcn_clv_dual as dual
import lightgcn_clv_moe as moe
import lightgcn_clv_v3 as v3


CODE_VERSION = "clv-dual-hm2y-seed42-v1.0"
GATE_SHAPE = "high"
TARGET_RHO = 0.2
ACCURACY_METRICS = tuple(
    f"{metric}@{k}" for metric in ("recall", "ndcg") for k in (10, 20, 50)
)


def configure_hm2y_seed42(**overrides) -> moe.MoEConfig:
    return validate_hm2y_config(
        dual.configure_dual_run("hm", short_hm=False, **overrides)
    )


def validate_hm2y_config(cfg: moe.MoEConfig) -> moe.MoEConfig:
    if cfg.dataset != "hm" or cfg.window_days is not None:
        raise ValueError("이 runner는 H&M 전체기간만 허용합니다")
    if tuple(cfg.seed_list) != (42,):
        raise ValueError("이번 단계는 seed 42만 허용합니다")
    if cfg.eval_test or cfg.eval_holdout:
        raise ValueError("이번 단계는 validation-only입니다")
    return dual.validate_dual_config(cfg)


def operating_point(raw_effective_ratio: float) -> dict:
    ratio = float(raw_effective_ratio)
    if not np.isfinite(ratio) or ratio <= 0:
        raise ValueError("raw_effective_ratio는 양의 유한값이어야 합니다")
    lam = TARGET_RHO / ratio
    return {
        "gate_shape": GATE_SHAPE,
        "rho": TARGET_RHO,
        "raw_effective_ratio": ratio,
        "lambda": lam,
        "effective_strength": lam * ratio,
    }


def seed42_decision(baseline: dict, model: dict) -> dict:
    accuracy_ratios = {
        metric: float(model[metric]) / float(baseline[metric])
        for metric in ACCURACY_METRICS
    }
    conditions = {
        "six_accuracy_ratios_at_least_0.99": all(
            ratio >= 0.99 for ratio in accuracy_ratios.values()
        ),
        "revenue@10_above_m1": float(model["revenue@10"])
        > float(baseline["revenue@10"]),
    }
    return {
        "success": all(conditions.values()),
        "conditions": conditions,
        "failed_conditions": [
            name for name, passed in conditions.items() if not passed
        ],
        "accuracy_ratios": accuracy_ratios,
        "revenue@10_delta": float(model["revenue@10"])
        - float(baseline["revenue@10"]),
    }


def preflight_summary(cfg: moe.MoEConfig) -> dict:
    cfg = validate_hm2y_config(cfg)
    return {
        "code_version": CODE_VERSION,
        "dataset": "hm",
        "period": "full",
        "window_days": None,
        "seed_list": [42],
        "models": ["m1", dual.PRIMARY_MODEL],
        "gate_shape": GATE_SHAPE,
        "target_rho": TARGET_RHO,
        "lambda_rule": "target_rho / raw_effective_ratio",
        "eval_test": False,
        "eval_holdout": False,
        "out_dir": None if cfg.out_dir is None else str(cfg.out_dir),
        "m1_checkpoint_dir": (
            None
            if cfg.m1_checkpoint_dir is None
            else str(cfg.m1_checkpoint_dir)
        ),
    }


def _persist(cfg, prepared, run, point, baseline, model_flat, model_per_user):
    rows = [
        {
            "seed": 42,
            "model_id": "m1",
            "split": "val",
            "gate_shape": "none",
            "rho": 0.0,
            "lambda": 0.0,
            "effective_strength": 0.0,
            **baseline,
        },
        {
            "seed": 42,
            "model_id": dual.PRIMARY_MODEL,
            "split": "val",
            **point,
            **model_flat,
        },
    ]
    frame = pd.DataFrame(rows)
    delta_rows = []
    for metric in ("recall", "ndcg", "revenue", "arp"):
        diff = (
            model_per_user[metric] - prepared["baseline_per_user"][metric]
        )
        delta_rows.append(
            {
                "model_id": dual.PRIMARY_MODEL,
                "split": "val",
                "gate_shape": GATE_SHAPE,
                "rho": TARGET_RHO,
                "lambda": point["lambda"],
                "metric": metric,
                **v3.paired_bootstrap([diff], prepared["base_cfg"]["N_BOOT"]),
            }
        )
    decision = seed42_decision(baseline, model_flat)
    stem = f"clv_dual_hm2y_seed42_{prepared['fingerprint']}"
    out_dir = Path(prepared["out_dir"])
    csv_path = out_dir / f"{stem}.csv"
    delta_path = out_dir / f"{stem}_delta.csv"
    json_path = out_dir / f"{stem}.json"
    frame.to_csv(csv_path, index=False, float_format="%.8f")
    pd.DataFrame(delta_rows).to_csv(delta_path, index=False)
    payload = {
        "code_version": CODE_VERSION,
        "source_revision": prepared["revision"],
        "result_fingerprint": prepared["fingerprint"],
        "input_manifest": prepared["manifest"],
        "config": asdict(cfg),
        "operating_point": point,
        "decision": decision,
        "absolute_rows": rows,
        "delta": delta_rows,
        "training": run["training"],
        "diagnostics": run["diagnostics"],
        "checkpoint_paths": {
            "m1_s42": prepared["m1_checkpoint"],
            "encoder_s42": prepared["encoder_checkpoint"],
            "dual_clv_fixed_s42": run["checkpoint"],
        },
    }
    payload["checkpoint_sha256"] = {
        name: moe.file_sha256(path)
        for name, path in payload["checkpoint_paths"].items()
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    frame.attrs["operating_point"] = point
    frame.attrs["decision"] = decision
    frame.attrs["result_paths"] = {
        "csv": str(csv_path),
        "delta_csv": str(delta_path),
        "json": str(json_path),
    }
    return frame


def run_hm2y_seed42(cfg: moe.MoEConfig | None = None) -> pd.DataFrame:
    cfg = validate_hm2y_config(cfg or configure_hm2y_seed42())
    print(json.dumps(preflight_summary(cfg), ensure_ascii=False, indent=2))
    prepared = dual._prepare(cfg, seed=42)
    run = dual._train_variant(
        dual.PRIMARY_MODEL,
        dual._fresh_base(prepared, seed=42),
        prepared,
        cfg,
        seed=42,
        gate_shapes=(GATE_SHAPE,),
        lambda_eval=(),
    )
    ratio = run["diagnostics"]["gate_shape_diagnostics"][GATE_SHAPE][
        "effective_total_ratio"
    ]
    point = operating_point(ratio)
    model = run["model"]
    model.set_eval_axes("n_plus_v")
    model.set_gate_shape(GATE_SHAPE)
    model_flat, model_per_user = moe._flat_evaluation(
        model,
        point["lambda"],
        prepared["cache"],
        prepared["meta"],
        prepared["data"],
        prepared["base_cfg"],
        per_user=True,
    )
    frame = _persist(
        cfg,
        prepared,
        run,
        point,
        prepared["baseline_flat"],
        model_flat,
        model_per_user,
    )
    print("H&M 2년 seed42 판정:", frame.attrs["decision"])
    print("결과 파일:", frame.attrs["result_paths"])
    return frame


if __name__ == "__main__":
    print(json.dumps(preflight_summary(configure_hm2y_seed42()), ensure_ascii=False, indent=2))
    print("학습은 Colab에서만 시작하세요.")
