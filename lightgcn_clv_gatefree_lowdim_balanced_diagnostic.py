"""Checkpoint-only diagnostics for the balanced-training gate-free M2.

This module never trains or selects a model.  It reads the already-completed
seed-42 historical-development M1 and balanced M2 checkpoints and answers:

1. Which propagated ID/N/V block changes recommendation metrics?
2. Where do held-out truth items move, with the exact evaluation purchase-
   amount weight attached to each truth item?
3. How large are the realised ID/N/V candidate-score contributions?

All metrics reuse the project's existing evaluator.  The word ``revenue`` is
not introduced here: the economic diagnostic is the project's
price/purchase-amount weighted hit value, not actual incremental revenue.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from clv_run_state import file_sha256
import lightgcn_clv_axis_specific_test10 as test10
import lightgcn_clv_gatefree_lowdim_balanced_training as balanced
import lightgcn_clv_gatefree_lowdim_diagnostic as base_diagnostic
import lightgcn_clv_v3 as v3


CODE_VERSION = "m2-gatefree-lowdim-balanced-checkpoint-diagnostic-v1"
VIEW_MODES = base_diagnostic.VIEW_MODES
RANK_BUCKETS = base_diagnostic.RANK_BUCKETS
SEGMENT_ORDER = base_diagnostic.SEGMENT_ORDER
BUCKET_POSITION = {name: index for index, name in enumerate(RANK_BUCKETS)}


@dataclass(frozen=True)
class BalancedCheckpointDiagnosticConfig:
    out_dir: str = ""
    baseline_result_dir: str = ""
    m2_checkpoint: str = ""
    m1_checkpoint: str = ""
    eval_batch_size: int = 32


def configure_balanced_checkpoint_diagnostic(
    **overrides,
) -> BalancedCheckpointDiagnosticConfig:
    defaults = balanced.configure_balanced_training_run()
    values = {
        "out_dir": defaults.out_dir,
        "baseline_result_dir": defaults.baseline_result_dir,
    }
    values.update(overrides)
    cfg = BalancedCheckpointDiagnosticConfig(**values)
    if not cfg.out_dir or not cfg.baseline_result_dir:
        raise ValueError("out_dir와 baseline_result_dir가 필요합니다")
    if cfg.eval_batch_size <= 0:
        raise ValueError("eval_batch_size는 양수여야 합니다")
    return cfg


def preflight_summary(cfg: BalancedCheckpointDiagnosticConfig) -> dict:
    return {
        "code_version": CODE_VERSION,
        "training": False,
        "model_selection": False,
        "scope": "existing balanced M2 and M1 checkpoints only",
        "split": "historical_development_days_684_690",
        "seed": 42,
        "views": list(VIEW_MODES),
        "rank_buckets": list(RANK_BUCKETS),
        "segments": list(SEGMENT_ORDER),
        "truth_weight_source": (
            "the exact per-truth-item purchase-amount weights already stored "
            "in EvalCache.rev"
        ),
        "candidate_score_scope": "all unseen evaluation candidates",
        "statistical_note": (
            "single-seed descriptive mechanism diagnostic; no significance "
            "or generalisation claim"
        ),
        "out_dir": cfg.out_dir,
        "baseline_result_dir": cfg.baseline_result_dir,
    }


def value_weighted_rank_transition_table(
    *,
    users: np.ndarray,
    segments: np.ndarray,
    truth: dict[int, np.ndarray],
    truth_weights: dict[int, np.ndarray],
    reference_ranks: dict[int, dict[int, int]],
    model_ranks: dict[int, dict[int, int]],
) -> pd.DataFrame:
    """Aggregate truth rank-bucket movements with exact evaluation weights."""
    aggregates: dict[tuple[str, str, str], dict[str, float]] = {}
    segment_counts: dict[str, int] = {}
    segment_weights: dict[str, float] = {}
    for user, segment in zip(users.tolist(), segments.tolist(), strict=True):
        user_id = int(user)
        truth_items = np.asarray(truth[user_id], dtype=np.int64)
        weights = np.asarray(truth_weights[user_id], dtype=np.float64)
        if truth_items.shape != weights.shape:
            raise RuntimeError(
                f"정답 아이템과 가중치 길이가 다릅니다: user={user_id}, "
                f"items={truth_items.shape}, weights={weights.shape}"
            )
        if not np.isfinite(weights).all() or (weights < 0).any():
            raise RuntimeError(f"정답 가중치에 잘못된 값이 있습니다: user={user_id}")
        segment = str(segment)
        segment_counts[segment] = segment_counts.get(segment, 0) + len(truth_items)
        segment_weights[segment] = segment_weights.get(segment, 0.0) + float(weights.sum())
        for item, weight in zip(truth_items.tolist(), weights.tolist(), strict=True):
            reference_bucket = base_diagnostic._rank_bucket(
                int(reference_ranks.get(user_id, {}).get(int(item), 51))
            )
            model_bucket = base_diagnostic._rank_bucket(
                int(model_ranks.get(user_id, {}).get(int(item), 51))
            )
            key = (segment, reference_bucket, model_bucket)
            aggregate = aggregates.setdefault(
                key, {"truth_item_count": 0.0, "truth_weight_sum": 0.0}
            )
            aggregate["truth_item_count"] += 1
            aggregate["truth_weight_sum"] += float(weight)

    rows = []
    for segment in SEGMENT_ORDER:
        for reference_bucket in RANK_BUCKETS:
            for model_bucket in RANK_BUCKETS:
                aggregate = aggregates.get((segment, reference_bucket, model_bucket))
                if aggregate is None:
                    continue
                count = int(aggregate["truth_item_count"])
                weight_sum = float(aggregate["truth_weight_sum"])
                total_weight = segment_weights[segment]
                rows.append(
                    {
                        "segment": segment,
                        "reference_bucket": reference_bucket,
                        "model_bucket": model_bucket,
                        "bucket_movement": _bucket_movement(
                            reference_bucket, model_bucket
                        ),
                        "truth_item_count": count,
                        "truth_item_share_within_segment": (
                            count / segment_counts[segment]
                        ),
                        "truth_weight_sum": weight_sum,
                        "truth_weight_share_within_segment": (
                            weight_sum / total_weight if total_weight > 0 else np.nan
                        ),
                        "mean_truth_weight": weight_sum / count,
                    }
                )
    return pd.DataFrame(rows)


def _bucket_movement(reference_bucket: str, model_bucket: str) -> str:
    reference_position = BUCKET_POSITION[reference_bucket]
    model_position = BUCKET_POSITION[model_bucket]
    if model_position < reference_position:
        return "promoted"
    if model_position > reference_position:
        return "demoted"
    return "same_bucket"


def weighted_movement_summary(transition: pd.DataFrame) -> pd.DataFrame:
    if transition.empty:
        return pd.DataFrame()
    grouped = (
        transition.groupby(["segment", "bucket_movement"], as_index=False)
        .agg(
            truth_item_count=("truth_item_count", "sum"),
            truth_weight_sum=("truth_weight_sum", "sum"),
        )
    )
    segment_totals = grouped.groupby("segment").agg(
        segment_truth_item_count=("truth_item_count", "sum"),
        segment_truth_weight_sum=("truth_weight_sum", "sum"),
    )
    grouped = grouped.join(segment_totals, on="segment")
    grouped["truth_item_share_within_segment"] = (
        grouped["truth_item_count"] / grouped["segment_truth_item_count"]
    )
    grouped["truth_weight_share_within_segment"] = (
        grouped["truth_weight_sum"] / grouped["segment_truth_weight_sum"]
    )
    return grouped.drop(
        columns=["segment_truth_item_count", "segment_truth_weight_sum"]
    )


def axis_attribution_table(view_metrics: pd.DataFrame) -> pd.DataFrame:
    """Compare each checkpoint ablation against its jointly trained ID block."""
    indexed = view_metrics.set_index("view")
    if "id_only" not in indexed.index:
        raise ValueError("id_only 성과가 없습니다")
    metrics = [
        f"{name}@{cutoff}"
        for cutoff in (10, 20, 50)
        for name in (
            "recall",
            "ndcg",
            "price_purchase_amount_weighted_hit",
        )
    ]
    rows = []
    for view in ("id_n", "id_v", "full"):
        for metric in metrics:
            if metric not in indexed.columns:
                continue
            reference = float(indexed.at["id_only", metric])
            value = float(indexed.at[view, metric])
            rows.append(
                {
                    "view": view,
                    "reference_view": "id_only",
                    "metric": metric,
                    "reference_value": reference,
                    "view_value": value,
                    "absolute_delta": value - reference,
                    "relative_change_pct": (
                        100.0 * (value - reference) / reference
                        if reference != 0
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def _load_balanced_m2(prepared: dict, runner_cfg, diagnostic_cfg):
    if diagnostic_cfg.m2_checkpoint:
        checkpoint = Path(diagnostic_cfg.m2_checkpoint)
        record = {}
    else:
        checkpoint, record = base_diagnostic._checkpoint_record(
            Path(diagnostic_cfg.out_dir), balanced.MODEL_ID
        )
    payload = torch.load(checkpoint, map_location=v3.DEVICE, weights_only=False)
    expected = asdict(runner_cfg)
    recorded = payload.get("config", {})
    identity_keys = (
        "dataset",
        "seed",
        "time_cutoff",
        "evaluation_days",
        "id_dim",
        "axis_dim",
        "hidden_dim",
        "axis_budget",
        "training_axis_balance_delta",
        "n_layers",
        "input_days",
    )
    mismatch = {
        key: {"expected": expected[key], "actual": recorded.get(key)}
        for key in identity_keys
        if recorded.get(key) != expected[key]
    }
    if payload.get("model_id") != balanced.MODEL_ID:
        mismatch["model_id"] = {
            "expected": balanced.MODEL_ID,
            "actual": payload.get("model_id"),
        }
    if payload.get("input_hash") != prepared["input_hash"]:
        mismatch["input_hash"] = {
            "expected": prepared["input_hash"],
            "actual": payload.get("input_hash"),
        }
    if mismatch:
        raise RuntimeError(f"balanced M2 checkpoint identity mismatch: {mismatch}")
    model, _ = balanced._build_model(prepared, runner_cfg)
    model.load_state_dict(payload["state"], strict=True)
    model.eval()
    return model, checkpoint, record


def _assert_full_metrics_match_record(
    view_metrics: pd.DataFrame, record: dict, *, atol: float = 5e-7
) -> None:
    if not record:
        return
    full = view_metrics.set_index("view").loc["full"]
    for metric, expected in record.get("metrics", {}).items():
        if metric not in full.index or not isinstance(expected, (int, float)):
            continue
        if not np.isclose(float(full[metric]), float(expected), rtol=0.0, atol=atol):
            raise RuntimeError(
                f"재평가 성과가 저장된 checkpoint 성과와 다릅니다: "
                f"{metric}, reevaluated={full[metric]}, recorded={expected}"
            )


def _assert_transition_totals(transition: pd.DataFrame, cache) -> None:
    expected_count = 0
    expected_weight = 0.0
    for user in cache.users.tolist():
        user_id = int(user)
        expected_count += len(cache.gt[user_id])
        expected_weight += float(np.asarray(cache.rev[user_id], dtype=float).sum())
    actual_count = int(transition["truth_item_count"].sum())
    actual_weight = float(transition["truth_weight_sum"].sum())
    if actual_count != expected_count:
        raise RuntimeError(
            f"순위이동 정답 수가 평가 cache와 다릅니다: "
            f"actual={actual_count}, expected={expected_count}"
        )
    if not np.isclose(actual_weight, expected_weight, rtol=0.0, atol=1e-8):
        raise RuntimeError(
            f"순위이동 정답 가중치 합이 평가 cache와 다릅니다: "
            f"actual={actual_weight}, expected={expected_weight}"
        )


def _persist(report: dict, cfg: BalancedCheckpointDiagnosticConfig) -> dict[str, str]:
    root = Path(cfg.out_dir) / "checkpoint_diagnostics_balanced"
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "rank_transition_csv": root / "m2_balanced_value_weighted_rank_transition.csv",
        "movement_summary_csv": root / "m2_balanced_value_weighted_movement_summary.csv",
        "view_metrics_csv": root / "m2_balanced_axis_view_metrics.csv",
        "axis_attribution_csv": root / "m2_balanced_axis_attribution.csv",
        "score_strength_csv": root / "m2_balanced_score_strength.csv",
        "json": root / "m2_balanced_checkpoint_diagnostic.json",
    }
    for key, report_key in (
        ("rank_transition_csv", "rank_transition"),
        ("movement_summary_csv", "movement_summary"),
        ("view_metrics_csv", "view_metrics"),
        ("axis_attribution_csv", "axis_attribution"),
        ("score_strength_csv", "score_strength"),
    ):
        test10._atomic_csv(paths[key], report[report_key])
    payload = {
        "code_version": CODE_VERSION,
        "scope": "existing checkpoints only; no training or model selection",
        "config": asdict(cfg),
        "preflight": preflight_summary(cfg),
        "checkpoints": report["checkpoints"],
        "rank_transition": report["rank_transition"].to_dict("records"),
        "movement_summary": report["movement_summary"].to_dict("records"),
        "view_metrics": report["view_metrics"].to_dict("records"),
        "axis_attribution": report["axis_attribution"].to_dict("records"),
        "score_strength": report["score_strength"].to_dict("records"),
        "interpretation_limits": [
            "single-seed descriptive mechanism diagnostic only",
            "ID-only is an internal jointly-trained ablation, not external M1",
            "price/purchase-amount weighted hit is not actual incremental revenue",
            "no significance or generalisation claim",
        ],
    }
    test10._atomic_json(paths["json"], payload)
    return {name: str(path) for name, path in paths.items()}


def run_balanced_checkpoint_diagnostic(
    cfg: BalancedCheckpointDiagnosticConfig | None = None,
) -> dict[str, pd.DataFrame]:
    cfg = cfg or configure_balanced_checkpoint_diagnostic()
    print(json.dumps(preflight_summary(cfg), ensure_ascii=False, indent=2))
    runner_cfg = balanced.configure_balanced_training_run(
        out_dir=cfg.out_dir,
        baseline_result_dir=cfg.baseline_result_dir,
    )
    prepared = balanced._prepare(runner_cfg)
    m2_model, m2_checkpoint, m2_record = _load_balanced_m2(
        prepared, runner_cfg, cfg
    )
    m1_model, m1_checkpoint, _ = base_diagnostic._load_m1(prepared, cfg)
    with torch.no_grad():
        m2_user, m2_item = m2_model.propagate()
        m1_user, m1_item = m1_model.propagate_pref()

    views = base_diagnostic.axis_views(
        m2_user,
        m2_item,
        id_dim=runner_cfg.id_dim,
        axis_dim=runner_cfg.axis_dim,
    )
    view_metrics = base_diagnostic._view_metrics(views, prepared)
    _assert_full_metrics_match_record(view_metrics, m2_record)
    axis_attribution = axis_attribution_table(view_metrics)

    m1_ranks = base_diagnostic._top50_ranks(
        m1_user, m1_item, prepared, batch_size=cfg.eval_batch_size
    )
    m2_ranks = base_diagnostic._top50_ranks(
        *views["full"], prepared, batch_size=cfg.eval_batch_size
    )
    rank_transition = value_weighted_rank_transition_table(
        users=prepared["cache"].users,
        segments=prepared["cache"].seg,
        truth=prepared["cache"].gt,
        truth_weights=prepared["cache"].rev,
        reference_ranks=m1_ranks,
        model_ranks=m2_ranks,
    )
    _assert_transition_totals(rank_transition, prepared["cache"])
    movement_summary = weighted_movement_summary(rank_transition)

    score_strength = base_diagnostic._score_strength(
        m2_user,
        m2_item,
        prepared,
        id_dim=runner_cfg.id_dim,
        axis_dim=runner_cfg.axis_dim,
        batch_size=cfg.eval_batch_size,
    )
    decomposition_error = float(
        score_strength["max_full_decomposition_error"].max()
    )
    if decomposition_error > 1e-4:
        raise RuntimeError(
            f"ID+N+V 점수 분해 오차가 너무 큽니다: {decomposition_error}"
        )

    report = {
        "rank_transition": rank_transition,
        "movement_summary": movement_summary,
        "view_metrics": view_metrics,
        "axis_attribution": axis_attribution,
        "score_strength": score_strength,
        "checkpoints": {
            "m1": {"path": str(m1_checkpoint), "sha256": file_sha256(m1_checkpoint)},
            "m2_balanced": {
                "path": str(m2_checkpoint),
                "sha256": file_sha256(m2_checkpoint),
            },
        },
    }
    paths = _persist(report, cfg)
    report["paths"] = paths

    print("\n===== ID/N/V 블록별 성과 =====")
    print(view_metrics.to_string(index=False))
    print("\n===== ID-only 대비 축별 증분 =====")
    print(axis_attribution.to_string(index=False))
    print("\n===== CLV 구간별 가중 정답상품 순위이동 =====")
    print(movement_summary.to_string(index=False))
    print("\n===== 실제 후보점수 영향력 =====")
    print(score_strength.to_string(index=False))
    print("\n저장 파일:", paths)
    return report


if __name__ == "__main__":
    print(
        "No training is started automatically. "
        "Call run_balanced_checkpoint_diagnostic()."
    )
