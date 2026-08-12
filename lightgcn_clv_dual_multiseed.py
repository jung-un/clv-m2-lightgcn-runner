"""Frozen seed-43/44 validation for the dual-axis CLV M2 model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


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


__all__ = [
    "MultiSeedValidationConfig",
    "configure_multiseed_validation",
    "reproducibility_decision",
    "validate_multiseed_config",
]
