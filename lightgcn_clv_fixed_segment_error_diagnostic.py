"""Checkpoint-only M1 error analysis by fixed historical-CLV segment.

This module does not train a model or select a checkpoint.  It reuses the
seed-42 M1 checkpoint from the historical-development split (train through day
683, evaluation on days 684--690), divides users by the fixed train-history
N×V proxy, and compares held-out truth items with M1 Top-10 errors.

The output is diagnostic evidence for deciding whether a CLV-segment-
conditioned representation is justified.  It must not be described as a new
confirmatory result.
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
import lightgcn_clv_gatefree_lowdim as gatefree
import lightgcn_clv_gatefree_lowdim_diagnostic as common
import lightgcn_clv_history_item_fit_diagnostic as item_fit
import lightgcn_clv_v3 as v3


CODE_VERSION = "m1-fixed-clv-segment-error-diagnostic-v1.4"
SEGMENT_ORDER = tuple(v3.SEG_NAMES)
NV_QUADRANT_ORDER = (
    "저N·저V",
    "고N·저V",
    "저N·고V",
    "고N·고V",
)
HIGH_CLV_COMPOSITION_ORDER = (
    "V우세 고CLV",
    "균형 고CLV",
    "N우세 고CLV",
)
ROLE_ORDER = (
    "truth_all",
    "truth_hit_top10",
    "truth_miss_top10",
    "truth_rank_11_20",
    "truth_rank_21_50",
    "truth_rank_over_50",
    "recommended_top10",
    "false_positive_top10",
)
TRAITS = (
    "price_percentile",
    "train_user_count",
    "repeat_purchase_share",
    "history_category_overlap",
    "history_embedding_cosine",
)


@dataclass(frozen=True)
class FixedSegmentErrorDiagnosticConfig:
    out_dir: str = ""
    baseline_result_dir: str = ""
    m1_checkpoint: str = ""
    eval_batch_size: int = 32
    top_examples: int = 20


def configure_fixed_segment_error_diagnostic(
    **overrides,
) -> FixedSegmentErrorDiagnosticConfig:
    defaults = gatefree.configure_gatefree_lowdim_run()
    values = {
        "out_dir": (
            f"{v3.default_out_dir('dunnhumby')}"
            "_m1_fixed_clv_segment_error_diagnostic_v1"
        ),
        "baseline_result_dir": defaults.baseline_result_dir,
    }
    values.update(overrides)
    cfg = FixedSegmentErrorDiagnosticConfig(**values)
    if not cfg.out_dir or not cfg.baseline_result_dir:
        raise ValueError("out_dir와 baseline_result_dir가 필요합니다")
    if cfg.eval_batch_size <= 0 or cfg.top_examples <= 0:
        raise ValueError("배치 크기와 산출 예시 수는 양수여야 합니다")
    return cfg


def preflight_summary(cfg: FixedSegmentErrorDiagnosticConfig) -> dict:
    return {
        "code_version": CODE_VERSION,
        "training": False,
        "checkpoint_selection": False,
        "model": "existing seed-42 M1@64 checkpoint",
        "split": "historical_development_days_684_690",
        "fixed_clv_source": "train-history N×V proxy at day 683",
        "segments": list(SEGMENT_ORDER),
        "nv_quadrants": list(NV_QUADRANT_ORDER),
        "high_clv_compositions": list(HIGH_CLV_COMPOSITION_ORDER),
        "nv_group_definition": (
            "q_N and q_V are train-history percentiles; low < 0.5 and "
            "high >= 0.5"
        ),
        "high_clv_composition_definition": (
            "within train-history high-CLV users, tertiles of q_N-q_V "
            "define V-dominant, balanced, and N-dominant composition"
        ),
        "roles": list(ROLE_ORDER),
        "comparison": (
            "held-out truth, Top-10 hits, Top-10 misses, and Top-10 false "
            "positives by fixed historical-CLV segment"
        ),
        "interpretation": (
            "descriptive development diagnostic only; implement a segment-"
            "conditioned M2 only if error profiles differ across segments"
        ),
        "statistical_note": "single-seed descriptive diagnostic; no significance claim",
        "out_dir": cfg.out_dir,
        "baseline_result_dir": cfg.baseline_result_dir,
    }


def _rank_bucket(rank: int) -> str:
    if rank <= 10:
        return "1-10"
    if rank <= 20:
        return "11-20"
    if rank <= 50:
        return "21-50"
    return ">50"


def segments_for_users(cache, users: np.ndarray) -> np.ndarray:
    """Return segment labels for global user IDs in the requested order.

    ``EvalCache.seg`` is parallel to ``EvalCache.users``; it is not indexed by
    the original global user ID.  Evaluation users can therefore contain IDs
    larger than ``len(cache.seg) - 1``.
    """
    cache_users = np.asarray(cache.users, dtype=np.int64)
    cache_segments = np.asarray(cache.seg)
    if len(cache_users) != len(cache_segments):
        raise ValueError("evaluation users and segment labels must be parallel")
    position_by_user = {
        int(user): position for position, user in enumerate(cache_users)
    }
    requested = np.asarray(users, dtype=np.int64)
    try:
        positions = np.fromiter(
            (position_by_user[int(user)] for user in requested),
            dtype=np.int64,
            count=len(requested),
        )
    except KeyError as error:
        raise KeyError(f"unknown evaluation user: {int(error.args[0])}") from error
    return cache_segments[positions]


def build_user_value_groups(
    axes: dict,
    *,
    clv_thresholds: tuple[float, float],
    evaluation_users: np.ndarray,
) -> tuple[pd.DataFrame, dict]:
    """Build fixed train-only N/V groups without using evaluation outcomes."""
    q_n = np.asarray(axes["q_n"], dtype=np.float64)
    q_v = np.asarray(axes["q_v"], dtype=np.float64)
    n_score = np.asarray(axes["n_behavior_score"], dtype=np.float64)
    v_score = np.asarray(axes["v_behavior_score"], dtype=np.float64)
    clv = np.asarray(axes["clv_proxy"], dtype=np.float64)
    valid = (
        np.asarray(axes["valid_user"], dtype=bool)
        & np.asarray(axes["activity_valid"], dtype=bool)
        & np.asarray(axes["value_valid"], dtype=bool)
        & np.isfinite(q_n)
        & np.isfinite(q_v)
        & np.isfinite(clv)
    )
    lengths = {len(q_n), len(q_v), len(n_score), len(v_score), len(clv), len(valid)}
    if len(lengths) != 1:
        raise ValueError("N/V 사용자 입력의 길이가 서로 다릅니다")

    low_clv, high_clv = map(float, clv_thresholds)
    fixed_segment = np.where(
        clv <= low_clv,
        SEGMENT_ORDER[0],
        np.where(clv >= high_clv, SEGMENT_ORDER[2], SEGMENT_ORDER[1]),
    )
    nv_quadrant = np.full(len(clv), "계산불가", dtype=object)
    nv_quadrant[valid & (q_n < 0.5) & (q_v < 0.5)] = NV_QUADRANT_ORDER[0]
    nv_quadrant[valid & (q_n >= 0.5) & (q_v < 0.5)] = NV_QUADRANT_ORDER[1]
    nv_quadrant[valid & (q_n < 0.5) & (q_v >= 0.5)] = NV_QUADRANT_ORDER[2]
    nv_quadrant[valid & (q_n >= 0.5) & (q_v >= 0.5)] = NV_QUADRANT_ORDER[3]

    difference = q_n - q_v
    high_valid = valid & fixed_segment.astype(str).__eq__(SEGMENT_ORDER[2])
    if int(high_valid.sum()) < 3:
        raise ValueError("고CLV 유효 학습고객이 3명보다 적어 구성 3분위를 만들 수 없습니다")
    lower, upper = np.quantile(difference[high_valid], [1.0 / 3.0, 2.0 / 3.0])
    composition = np.full(len(clv), "비고CLV", dtype=object)
    composition[high_valid & (difference <= lower)] = HIGH_CLV_COMPOSITION_ORDER[0]
    composition[high_valid & (difference > lower) & (difference < upper)] = (
        HIGH_CLV_COMPOSITION_ORDER[1]
    )
    composition[high_valid & (difference >= upper)] = HIGH_CLV_COMPOSITION_ORDER[2]

    evaluation_mask = np.zeros(len(clv), dtype=bool)
    requested_users = np.asarray(evaluation_users, dtype=np.int64)
    if len(requested_users) and (
        requested_users.min() < 0 or requested_users.max() >= len(clv)
    ):
        raise ValueError("평가 사용자 ID가 사용자 입력 범위를 벗어났습니다")
    evaluation_mask[requested_users] = True
    membership = pd.DataFrame(
        {
            "user_idx": np.arange(len(clv), dtype=np.int64),
            "fixed_clv_segment": fixed_segment,
            "nv_quadrant": nv_quadrant,
            "high_clv_composition": composition,
            "n_behavior_score": n_score,
            "v_behavior_score": v_score,
            "historical_clv_proxy": clv,
            "q_n": q_n,
            "q_v": q_v,
            "q_n_minus_q_v": difference,
            "axis_valid": valid,
            "is_evaluation_user": evaluation_mask,
        }
    )
    thresholds = {
        "fixed_clv_low_max": low_clv,
        "fixed_clv_high_min": high_clv,
        "nv_low_high_cutoff": 0.5,
        "high_clv_qn_minus_qv_lower_tertile": float(lower),
        "high_clv_qn_minus_qv_upper_tertile": float(upper),
        "high_clv_train_user_count": int(high_valid.sum()),
        "axis_valid_train_user_count": int(valid.sum()),
    }
    return membership, thresholds


def summarize_user_groups(
    membership: pd.DataFrame,
    *,
    group_column: str,
    group_order: tuple[str, ...],
) -> pd.DataFrame:
    selected = membership[membership[group_column].isin(group_order)].copy()
    selected[group_column] = pd.Categorical(
        selected[group_column], categories=group_order, ordered=True
    )
    return (
        selected.groupby(group_column, observed=False, sort=True)
        .agg(
            n_train_users=("user_idx", "size"),
            n_evaluation_users=("is_evaluation_user", "sum"),
            mean_n_behavior_score=("n_behavior_score", "mean"),
            mean_v_behavior_score=("v_behavior_score", "mean"),
            mean_historical_clv_proxy=("historical_clv_proxy", "mean"),
            mean_q_n=("q_n", "mean"),
            mean_q_v=("q_v", "mean"),
            mean_q_n_minus_q_v=("q_n_minus_q_v", "mean"),
        )
        .reset_index()
    )


def _raw_item_traits(train: pd.DataFrame, n_items: int) -> pd.DataFrame:
    """Build the exact train-only item columns consumed by this diagnostic."""
    def modal(series: pd.Series):
        mode = series.mode(dropna=True)
        return mode.iat[0] if len(mode) else "UNKNOWN"

    item = train.groupby("i_idx", sort=True).agg(
        item_id=("i_raw", "first"),
        category=("cat_raw", modal),
        train_user_count=("u_idx", "nunique"),
        mean_unit_price=("up", "mean"),
    )
    pair_counts = train.groupby(["u_idx", "i_idx"], sort=False).size()
    item["repeat_purchase_share"] = (
        pair_counts.gt(1).groupby(level="i_idx").mean()
    )
    item["price_percentile"] = item["mean_unit_price"].rank(
        pct=True, method="average"
    )
    result = pd.DataFrame({"item_idx": np.arange(n_items, dtype=np.int64)})
    return result.merge(
        item.rename_axis("item_idx").reset_index(), on="item_idx", how="left"
    )


def item_role_occurrences(
    *,
    users: np.ndarray,
    segments: np.ndarray,
    truth: dict[int, np.ndarray],
    top50: np.ndarray,
    truth_amount: dict[int, np.ndarray],
) -> pd.DataFrame:
    """Expand M1 truth/hit/miss/false-positive roles without retraining."""
    rows: list[dict] = []
    for row_index, (user, segment) in enumerate(
        zip(users.tolist(), segments.tolist(), strict=True)
    ):
        user = int(user)
        ranked = np.asarray(top50[row_index], dtype=np.int64)
        rank = {int(item): position for position, item in enumerate(ranked, 1)}
        truth_items = np.asarray(truth[user], dtype=np.int64)
        amounts = np.asarray(truth_amount[user], dtype=float)
        amount_by_item = {
            int(item): float(amount)
            for item, amount in zip(truth_items, amounts, strict=True)
        }
        truth_set = set(amount_by_item)
        top10 = ranked[:10]
        roles: dict[str, np.ndarray] = {
            "truth_all": truth_items,
            "truth_hit_top10": np.asarray(
                [item for item in truth_items if rank.get(int(item), 51) <= 10],
                dtype=np.int64,
            ),
            "truth_miss_top10": np.asarray(
                [item for item in truth_items if rank.get(int(item), 51) > 10],
                dtype=np.int64,
            ),
            "truth_rank_11_20": np.asarray(
                [item for item in truth_items if 11 <= rank.get(int(item), 51) <= 20],
                dtype=np.int64,
            ),
            "truth_rank_21_50": np.asarray(
                [item for item in truth_items if 21 <= rank.get(int(item), 51) <= 50],
                dtype=np.int64,
            ),
            "truth_rank_over_50": np.asarray(
                [item for item in truth_items if rank.get(int(item), 51) > 50],
                dtype=np.int64,
            ),
            "recommended_top10": top10,
            "false_positive_top10": np.asarray(
                [item for item in top10 if int(item) not in truth_set],
                dtype=np.int64,
            ),
        }
        for role, items in roles.items():
            for item in items:
                item = int(item)
                item_rank = rank.get(item, 51)
                rows.append(
                    {
                        "user_idx": user,
                        "segment": str(segment),
                        "role": role,
                        "item_idx": item,
                        "m1_rank_capped_51": item_rank,
                        "rank_bucket": _rank_bucket(item_rank),
                        "is_truth_item": item in truth_set,
                        "evaluation_purchase_amount": amount_by_item.get(item, 0.0),
                    }
                )
    return pd.DataFrame(rows)


def attach_history_relations(
    occurrences: pd.DataFrame,
    *,
    train: pd.DataFrame,
    item_embedding: np.ndarray,
    n_users: int,
) -> pd.DataFrame:
    """Attach candidate-to-own-history relations for descriptive comparison."""
    output = occurrences.copy()
    categories = (
        train.groupby("u_idx", sort=False)["cat_raw"]
        .agg(lambda values: set(values.dropna().astype(str)))
        .to_dict()
    )
    output["history_category_overlap"] = [
        float(str(category) in categories.get(int(user), set()))
        for user, category in zip(
            output.user_idx, output.category, strict=True
        )
    ]

    embedding = np.asarray(item_embedding, dtype=np.float64)
    norms = np.linalg.norm(embedding, axis=1, keepdims=True)
    unit_items = np.divide(
        embedding,
        norms,
        out=np.zeros_like(embedding),
        where=norms > 0,
    )
    centroids = np.zeros((n_users, embedding.shape[1]), dtype=np.float64)
    for user, items in train.groupby("u_idx", sort=False)["i_idx"]:
        unique_items = np.unique(items.to_numpy(dtype=np.int64))
        if len(unique_items):
            centroids[int(user)] = unit_items[unique_items].mean(axis=0)
    centroid_norm = np.linalg.norm(centroids, axis=1, keepdims=True)
    centroids = np.divide(
        centroids,
        centroid_norm,
        out=np.zeros_like(centroids),
        where=centroid_norm > 0,
    )
    user_index = output.user_idx.to_numpy(dtype=np.int64)
    item_index = output.item_idx.to_numpy(dtype=np.int64)
    output["history_embedding_cosine"] = (
        centroids[user_index] * unit_items[item_index]
    ).sum(axis=1)
    return output


def summarize_item_roles_by(
    occurrences: pd.DataFrame, *, group_column: str
) -> pd.DataFrame:
    available = [trait for trait in TRAITS if trait in occurrences]
    aggregations = {
        "item_occurrence_count": ("item_idx", "size"),
        "distinct_item_count": ("item_idx", "nunique"),
        **{f"mean_{trait}": (trait, "mean") for trait in available},
    }
    return (
        occurrences.groupby([group_column, "role"], sort=False, dropna=False)
        .agg(**aggregations)
        .reset_index()
    )


def summarize_segment_item_roles(occurrences: pd.DataFrame) -> pd.DataFrame:
    return summarize_item_roles_by(occurrences, group_column="segment")


def contrasts_by_group(
    summary: pd.DataFrame,
    *,
    group_column: str,
    group_order: tuple[str, ...],
) -> pd.DataFrame:
    rows = []
    for group in group_order:
        selected = summary[summary[group_column].eq(group)].set_index("role")
        if not {"truth_miss_top10", "false_positive_top10"}.issubset(selected.index):
            continue
        for trait in TRAITS:
            column = f"mean_{trait}"
            if column not in selected:
                continue
            missed = float(selected.at["truth_miss_top10", column])
            false_positive = float(selected.at["false_positive_top10", column])
            rows.append(
                {
                    group_column: group,
                    "trait": trait,
                    "truth_miss_mean": missed,
                    "false_positive_mean": false_positive,
                    "miss_minus_false_positive": missed - false_positive,
                }
            )
    return pd.DataFrame(rows)


def miss_false_positive_contrasts(summary: pd.DataFrame) -> pd.DataFrame:
    return contrasts_by_group(
        summary, group_column="segment", group_order=SEGMENT_ORDER
    )


def _per_user_metrics(
    *, users: np.ndarray, top50: np.ndarray, prepared: dict
) -> pd.DataFrame:
    cache, meta, data = prepared["cache"], prepared["meta"], prepared["data"]
    novelty = -np.log2(meta["pop_prob"] + 1e-12)
    scored = v3.score_topk(
        top50,
        users,
        [10, 20, 50],
        cache.pos_key,
        cache.pos_rev,
        data["n_items"],
        cache.P_arr,
        meta["price_pct"],
        novelty,
        meta["cat"],
        cache.ideal,
    )
    frame = pd.DataFrame(
        {
            "user_idx": users,
            "truth_item_count": cache.P_arr[users],
        }
    )
    aliases = {
        "revenue": "price_purchase_amount_weighted_hit",
        "arp": "mean_recommended_price_percentile",
    }
    for k, metrics in scored.items():
        for metric in ("recall", "precision", "ndcg", "hr", "map", "revenue", "arp"):
            frame[f"{aliases.get(metric, metric)}@{k}"] = metrics[metric]
    return frame


def summarize_metrics_by_group(
    per_user: pd.DataFrame,
    membership: pd.DataFrame,
    *,
    group_column: str,
    group_order: tuple[str, ...],
) -> pd.DataFrame:
    user_columns = [
        "user_idx",
        group_column,
        "n_behavior_score",
        "v_behavior_score",
        "historical_clv_proxy",
        "q_n",
        "q_v",
        "q_n_minus_q_v",
    ]
    frame = per_user.merge(
        membership[user_columns], on="user_idx", how="left", validate="one_to_one"
    )
    frame = frame[frame[group_column].isin(group_order)].copy()
    frame[group_column] = pd.Categorical(
        frame[group_column], categories=group_order, ordered=True
    )
    metric_columns = [column for column in frame if "@" in column]
    return (
        frame.groupby(group_column, observed=False, sort=True)
        .agg(
            n_users=("user_idx", "size"),
            mean_truth_items=("truth_item_count", "mean"),
            mean_n_behavior_score=("n_behavior_score", "mean"),
            mean_v_behavior_score=("v_behavior_score", "mean"),
            mean_historical_clv_proxy=("historical_clv_proxy", "mean"),
            mean_q_n=("q_n", "mean"),
            mean_q_v=("q_v", "mean"),
            mean_q_n_minus_q_v=("q_n_minus_q_v", "mean"),
            **{column: (column, "mean") for column in metric_columns},
        )
        .reset_index()
    )


def _segment_metrics(
    *, users: np.ndarray, top50: np.ndarray, prepared: dict
) -> pd.DataFrame:
    frame = _per_user_metrics(users=users, top50=top50, prepared=prepared)
    frame["segment"] = segments_for_users(prepared["cache"], users)
    metric_columns = [column for column in frame if "@" in column]
    return (
        frame.groupby("segment", sort=False)
        .agg(
            n_users=("user_idx", "size"),
            mean_truth_items=("truth_item_count", "mean"),
            **{column: (column, "mean") for column in metric_columns},
        )
        .reset_index()
    )


def _category_summary(
    occurrences: pd.DataFrame, top_n: int, *, group_column: str = "segment"
) -> pd.DataFrame:
    focused = occurrences[
        occurrences.role.isin(
            ["truth_hit_top10", "truth_miss_top10", "false_positive_top10"]
        )
    ]
    if focused.empty:
        return focused.copy()
    grouped = (
        focused.groupby([group_column, "role", "category"], dropna=False)
        .size()
        .rename("item_occurrence_count")
        .reset_index()
    )
    grouped["share_within_segment_role"] = grouped.groupby(
        [group_column, "role"]
    ).item_occurrence_count.transform(lambda values: values / values.sum())
    return (
        grouped.sort_values(
            [group_column, "role", "item_occurrence_count"],
            ascending=[True, True, False],
        )
        .groupby([group_column, "role"], sort=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def _examples(
    occurrences: pd.DataFrame, top_n: int, *, group_column: str = "segment"
) -> pd.DataFrame:
    focused = occurrences[
        occurrences.role.isin(["truth_miss_top10", "false_positive_top10"])
    ]
    if focused.empty:
        return focused.copy()
    grouped = (
        focused.groupby([group_column, "role", "item_idx"], dropna=False)
        .agg(
            occurrence_count=("item_idx", "size"),
            affected_user_count=("user_idx", "nunique"),
            mean_evaluation_purchase_amount=("evaluation_purchase_amount", "mean"),
            item_id=("item_id", "first"),
            category=("category", "first"),
            train_user_count=("train_user_count", "first"),
            price_percentile=("price_percentile", "first"),
            repeat_purchase_share=("repeat_purchase_share", "first"),
            mean_history_category_overlap=("history_category_overlap", "mean"),
            mean_history_embedding_cosine=("history_embedding_cosine", "mean"),
        )
        .reset_index()
    )
    return (
        grouped.sort_values(
            [group_column, "role", "occurrence_count", "mean_evaluation_purchase_amount"],
            ascending=[True, True, False, False],
        )
        .groupby([group_column, "role"], sort=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def _persist(report: dict, cfg: FixedSegmentErrorDiagnosticConfig) -> dict[str, str]:
    root = Path(cfg.out_dir) / "checkpoint_diagnostics"
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "segment_metrics_csv": root / "m1_fixed_clv_segment_metrics.csv",
        "user_value_groups_csv": root / "m1_fixed_clv_nv_user_groups.csv",
        "nv_quadrant_population_csv": root / "m1_fixed_clv_nv_quadrant_population.csv",
        "nv_quadrant_metrics_csv": root / "m1_fixed_clv_nv_quadrant_metrics.csv",
        "nv_quadrant_item_role_summary_csv": root / "m1_fixed_clv_nv_quadrant_item_role_summary.csv",
        "nv_quadrant_contrasts_csv": root / "m1_fixed_clv_nv_quadrant_contrasts.csv",
        "nv_quadrant_category_summary_csv": root / "m1_fixed_clv_nv_quadrant_category_summary.csv",
        "nv_quadrant_examples_csv": root / "m1_fixed_clv_nv_quadrant_examples.csv",
        "high_clv_composition_population_csv": root / "m1_fixed_clv_high_composition_population.csv",
        "high_clv_composition_metrics_csv": root / "m1_fixed_clv_high_composition_metrics.csv",
        "high_clv_composition_item_role_summary_csv": root / "m1_fixed_clv_high_composition_item_role_summary.csv",
        "high_clv_composition_contrasts_csv": root / "m1_fixed_clv_high_composition_contrasts.csv",
        "high_clv_composition_category_summary_csv": root / "m1_fixed_clv_high_composition_category_summary.csv",
        "high_clv_composition_examples_csv": root / "m1_fixed_clv_high_composition_examples.csv",
        "item_role_occurrences_csv": root / "m1_fixed_clv_item_role_occurrences.csv",
        "item_role_summary_csv": root / "m1_fixed_clv_item_role_summary.csv",
        "miss_false_positive_contrasts_csv": root / "m1_fixed_clv_miss_false_positive_contrasts.csv",
        "category_summary_csv": root / "m1_fixed_clv_category_summary.csv",
        "examples_csv": root / "m1_fixed_clv_error_examples.csv",
        "json": root / "m1_fixed_clv_segment_error_diagnostic.json",
    }
    frame_mapping = {
        "segment_metrics_csv": "segment_metrics",
        "user_value_groups_csv": "user_value_groups",
        "nv_quadrant_population_csv": "nv_quadrant_population",
        "nv_quadrant_metrics_csv": "nv_quadrant_metrics",
        "nv_quadrant_item_role_summary_csv": "nv_quadrant_item_role_summary",
        "nv_quadrant_contrasts_csv": "nv_quadrant_contrasts",
        "nv_quadrant_category_summary_csv": "nv_quadrant_category_summary",
        "nv_quadrant_examples_csv": "nv_quadrant_examples",
        "high_clv_composition_population_csv": "high_clv_composition_population",
        "high_clv_composition_metrics_csv": "high_clv_composition_metrics",
        "high_clv_composition_item_role_summary_csv": "high_clv_composition_item_role_summary",
        "high_clv_composition_contrasts_csv": "high_clv_composition_contrasts",
        "high_clv_composition_category_summary_csv": "high_clv_composition_category_summary",
        "high_clv_composition_examples_csv": "high_clv_composition_examples",
        "item_role_occurrences_csv": "item_role_occurrences",
        "item_role_summary_csv": "item_role_summary",
        "miss_false_positive_contrasts_csv": "contrasts",
        "category_summary_csv": "category_summary",
        "examples_csv": "examples",
    }
    for path_key, report_key in frame_mapping.items():
        test10._atomic_csv(paths[path_key], report[report_key])
    payload = {
        "code_version": CODE_VERSION,
        "config": asdict(cfg),
        "preflight": preflight_summary(cfg),
        "checkpoint": report["checkpoint"],
        "group_thresholds": report["group_thresholds"],
        "segment_metrics": report["segment_metrics"].to_dict("records"),
        "nv_quadrant_population": report["nv_quadrant_population"].to_dict("records"),
        "nv_quadrant_metrics": report["nv_quadrant_metrics"].to_dict("records"),
        "nv_quadrant_item_role_summary": report[
            "nv_quadrant_item_role_summary"
        ].to_dict("records"),
        "nv_quadrant_contrasts": report["nv_quadrant_contrasts"].to_dict("records"),
        "high_clv_composition_population": report[
            "high_clv_composition_population"
        ].to_dict("records"),
        "high_clv_composition_metrics": report[
            "high_clv_composition_metrics"
        ].to_dict("records"),
        "high_clv_composition_item_role_summary": report[
            "high_clv_composition_item_role_summary"
        ].to_dict("records"),
        "high_clv_composition_contrasts": report[
            "high_clv_composition_contrasts"
        ].to_dict("records"),
        "item_role_summary": report["item_role_summary"].to_dict("records"),
        "miss_false_positive_contrasts": report["contrasts"].to_dict("records"),
        "result_paths": {name: str(path) for name, path in paths.items()},
        "reading_rule": {
            "supports_segment_conditioning_if": (
                "the truth-miss versus false-positive trait gaps differ "
                "systematically across low/mid/high fixed-CLV segments"
            ),
            "otherwise": (
                "do not implement CLV-segment embeddings; the fixed-CLV "
                "partition does not explain M1 ranking errors"
            ),
            "supports_nv_conditioning_if": (
                "high-N/low-V and low-N/high-V users, or N-dominant and "
                "V-dominant high-CLV users, have meaningfully different "
                "truth and error profiles"
            ),
            "otherwise_for_nv": (
                "use one continuous fixed historical-CLV level rather than "
                "separate N/V routing"
            ),
        },
        "interpretation_limits": [
            "historical-development diagnostic only",
            "no training or checkpoint selection",
            "no statistical significance claim",
            "price/purchase-amount weighted hit is not actual incremental revenue",
        ],
    }
    test10._atomic_json(paths["json"], payload)
    return {name: str(path) for name, path in paths.items()}


def run_fixed_segment_error_diagnostic(
    cfg: FixedSegmentErrorDiagnosticConfig | None = None,
) -> dict:
    cfg = cfg or configure_fixed_segment_error_diagnostic()
    print(json.dumps(preflight_summary(cfg), ensure_ascii=False, indent=2))
    runner_cfg = gatefree.configure_gatefree_lowdim_run(
        out_dir=cfg.out_dir,
        baseline_result_dir=cfg.baseline_result_dir,
    )
    prepared = gatefree._prepare(runner_cfg)
    m1_model, checkpoint, _ = common._load_m1(prepared, cfg)
    with torch.no_grad():
        user_embedding, item_embedding, *_ = m1_model.embeddings(need_value=False)
    users, top50 = item_fit._masked_topk(
        user_embedding,
        item_embedding,
        prepared,
        max_k=50,
        batch_size=cfg.eval_batch_size,
    )
    clv_values = np.asarray(prepared["axes"]["clv_proxy"], dtype=np.float64)
    clv_edges = prepared["base_cfg"]["SEG_EDGES"]
    non_missing_clv = clv_values[~np.isnan(clv_values)]
    clv_thresholds = tuple(
        float(value)
        for value in np.quantile(non_missing_clv, [clv_edges[0], clv_edges[1]])
    )
    user_value_groups, group_thresholds = build_user_value_groups(
        prepared["axes"],
        clv_thresholds=clv_thresholds,
        evaluation_users=users,
    )
    expected_segments = segments_for_users(prepared["cache"], users)
    actual_segments = user_value_groups.set_index("user_idx").loc[
        users, "fixed_clv_segment"
    ].to_numpy()
    if not np.array_equal(expected_segments, actual_segments):
        raise RuntimeError("기존 저·중·고 CLV 구간과 N/V 분석 구간이 일치하지 않습니다")
    occurrences = item_role_occurrences(
        users=users,
        segments=expected_segments,
        truth=prepared["cache"].gt,
        top50=top50,
        truth_amount=prepared["cache"].rev,
    )
    item_traits = _raw_item_traits(
        prepared["data"]["train"], prepared["data"]["n_items"]
    )
    occurrences = occurrences.merge(item_traits, on="item_idx", how="left")
    occurrences = attach_history_relations(
        occurrences,
        train=prepared["data"]["train"],
        item_embedding=item_embedding.detach().cpu().numpy(),
        n_users=prepared["data"]["n_users"],
    )
    occurrences = occurrences.merge(
        user_value_groups[
            ["user_idx", "nv_quadrant", "high_clv_composition"]
        ],
        on="user_idx",
        how="left",
        validate="many_to_one",
    )
    per_user_metrics = _per_user_metrics(
        users=users, top50=top50, prepared=prepared
    )
    segment_metrics = _segment_metrics(
        users=users, top50=top50, prepared=prepared
    )
    nv_quadrant_population = summarize_user_groups(
        user_value_groups,
        group_column="nv_quadrant",
        group_order=NV_QUADRANT_ORDER,
    )
    high_clv_composition_population = summarize_user_groups(
        user_value_groups,
        group_column="high_clv_composition",
        group_order=HIGH_CLV_COMPOSITION_ORDER,
    )
    nv_quadrant_metrics = summarize_metrics_by_group(
        per_user_metrics,
        user_value_groups,
        group_column="nv_quadrant",
        group_order=NV_QUADRANT_ORDER,
    )
    high_clv_composition_metrics = summarize_metrics_by_group(
        per_user_metrics,
        user_value_groups,
        group_column="high_clv_composition",
        group_order=HIGH_CLV_COMPOSITION_ORDER,
    )
    item_role_summary = summarize_segment_item_roles(occurrences)
    contrasts = miss_false_positive_contrasts(item_role_summary)
    nv_occurrences = occurrences[
        occurrences.nv_quadrant.isin(NV_QUADRANT_ORDER)
    ].copy()
    high_clv_occurrences = occurrences[
        occurrences.high_clv_composition.isin(HIGH_CLV_COMPOSITION_ORDER)
    ].copy()
    nv_quadrant_item_role_summary = summarize_item_roles_by(
        nv_occurrences, group_column="nv_quadrant"
    )
    high_clv_composition_item_role_summary = summarize_item_roles_by(
        high_clv_occurrences, group_column="high_clv_composition"
    )
    nv_quadrant_contrasts = contrasts_by_group(
        nv_quadrant_item_role_summary,
        group_column="nv_quadrant",
        group_order=NV_QUADRANT_ORDER,
    )
    high_clv_composition_contrasts = contrasts_by_group(
        high_clv_composition_item_role_summary,
        group_column="high_clv_composition",
        group_order=HIGH_CLV_COMPOSITION_ORDER,
    )
    category_summary = _category_summary(occurrences, cfg.top_examples)
    examples = _examples(occurrences, cfg.top_examples)
    nv_quadrant_category_summary = _category_summary(
        nv_occurrences, cfg.top_examples, group_column="nv_quadrant"
    )
    nv_quadrant_examples = _examples(
        nv_occurrences, cfg.top_examples, group_column="nv_quadrant"
    )
    high_clv_composition_category_summary = _category_summary(
        high_clv_occurrences,
        cfg.top_examples,
        group_column="high_clv_composition",
    )
    high_clv_composition_examples = _examples(
        high_clv_occurrences,
        cfg.top_examples,
        group_column="high_clv_composition",
    )
    report = {
        "segment_metrics": segment_metrics,
        "user_value_groups": user_value_groups,
        "group_thresholds": group_thresholds,
        "nv_quadrant_population": nv_quadrant_population,
        "nv_quadrant_metrics": nv_quadrant_metrics,
        "nv_quadrant_item_role_summary": nv_quadrant_item_role_summary,
        "nv_quadrant_contrasts": nv_quadrant_contrasts,
        "nv_quadrant_category_summary": nv_quadrant_category_summary,
        "nv_quadrant_examples": nv_quadrant_examples,
        "high_clv_composition_population": high_clv_composition_population,
        "high_clv_composition_metrics": high_clv_composition_metrics,
        "high_clv_composition_item_role_summary": high_clv_composition_item_role_summary,
        "high_clv_composition_contrasts": high_clv_composition_contrasts,
        "high_clv_composition_category_summary": high_clv_composition_category_summary,
        "high_clv_composition_examples": high_clv_composition_examples,
        "item_role_occurrences": occurrences,
        "item_role_summary": item_role_summary,
        "contrasts": contrasts,
        "category_summary": category_summary,
        "examples": examples,
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": file_sha256(checkpoint),
        },
    }
    report["paths"] = _persist(report, cfg)

    print("\n===== 1) 고정 CLV 구간별 M1 성과 =====")
    print(segment_metrics.to_string(index=False))
    print("\n===== 2) 전체 고객 N/V 4유형 구성과 M1 성과 =====")
    print(nv_quadrant_population.to_string(index=False))
    print(nv_quadrant_metrics.to_string(index=False))
    print("\n===== 3) 고CLV 내부 N/V 구성과 M1 성과 =====")
    print(high_clv_composition_population.to_string(index=False))
    print(high_clv_composition_metrics.to_string(index=False))
    print("\n===== 4) 정답·적중·오추천 상품 특성 =====")
    print(item_role_summary.to_string(index=False))
    print("\n===== 5) 전체 N/V 4유형 정답 누락 - Top-10 오추천 격차 =====")
    print(nv_quadrant_contrasts.to_string(index=False))
    print("\n===== 6) 고CLV 내부 정답 누락 - Top-10 오추천 격차 =====")
    print(high_clv_composition_contrasts.to_string(index=False))
    print("\n===== 7) 기존 저·중·고 CLV 정답 누락 - Top-10 오추천 격차 =====")
    print(contrasts.to_string(index=False))
    print("\n===== 8) 고CLV 내부 실제 상품 예시 =====")
    print(high_clv_composition_examples.to_string(index=False))
    print("\n===== 9) 기존 저·중·고 CLV 실제 상품 예시 =====")
    print(examples.to_string(index=False))
    print("\n결과 파일:", report["paths"])
    return report


if __name__ == "__main__":
    print(
        json.dumps(
            preflight_summary(configure_fixed_segment_error_diagnostic()),
            ensure_ascii=False,
            indent=2,
        )
    )
