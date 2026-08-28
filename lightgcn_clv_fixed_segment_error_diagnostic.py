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


CODE_VERSION = "m1-fixed-clv-segment-error-diagnostic-v1.3"
SEGMENT_ORDER = tuple(v3.SEG_NAMES)
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


def summarize_segment_item_roles(occurrences: pd.DataFrame) -> pd.DataFrame:
    available = [trait for trait in TRAITS if trait in occurrences]
    aggregations = {
        "item_occurrence_count": ("item_idx", "size"),
        "distinct_item_count": ("item_idx", "nunique"),
        **{f"mean_{trait}": (trait, "mean") for trait in available},
    }
    return (
        occurrences.groupby(["segment", "role"], sort=False, dropna=False)
        .agg(**aggregations)
        .reset_index()
    )


def miss_false_positive_contrasts(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for segment in SEGMENT_ORDER:
        selected = summary[summary.segment.eq(segment)].set_index("role")
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
                    "segment": segment,
                    "trait": trait,
                    "truth_miss_mean": missed,
                    "false_positive_mean": false_positive,
                    "miss_minus_false_positive": missed - false_positive,
                }
            )
    return pd.DataFrame(rows)


def _segment_metrics(
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
            "segment": segments_for_users(cache, users),
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


def _category_summary(occurrences: pd.DataFrame, top_n: int) -> pd.DataFrame:
    focused = occurrences[
        occurrences.role.isin(
            ["truth_hit_top10", "truth_miss_top10", "false_positive_top10"]
        )
    ]
    if focused.empty:
        return focused.copy()
    grouped = (
        focused.groupby(["segment", "role", "category"], dropna=False)
        .size()
        .rename("item_occurrence_count")
        .reset_index()
    )
    grouped["share_within_segment_role"] = grouped.groupby(
        ["segment", "role"]
    ).item_occurrence_count.transform(lambda values: values / values.sum())
    return (
        grouped.sort_values(
            ["segment", "role", "item_occurrence_count"],
            ascending=[True, True, False],
        )
        .groupby(["segment", "role"], sort=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def _examples(occurrences: pd.DataFrame, top_n: int) -> pd.DataFrame:
    focused = occurrences[
        occurrences.role.isin(["truth_miss_top10", "false_positive_top10"])
    ]
    if focused.empty:
        return focused.copy()
    grouped = (
        focused.groupby(["segment", "role", "item_idx"], dropna=False)
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
            ["segment", "role", "occurrence_count", "mean_evaluation_purchase_amount"],
            ascending=[True, True, False, False],
        )
        .groupby(["segment", "role"], sort=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def _persist(report: dict, cfg: FixedSegmentErrorDiagnosticConfig) -> dict[str, str]:
    root = Path(cfg.out_dir) / "checkpoint_diagnostics"
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "segment_metrics_csv": root / "m1_fixed_clv_segment_metrics.csv",
        "item_role_occurrences_csv": root / "m1_fixed_clv_item_role_occurrences.csv",
        "item_role_summary_csv": root / "m1_fixed_clv_item_role_summary.csv",
        "miss_false_positive_contrasts_csv": root / "m1_fixed_clv_miss_false_positive_contrasts.csv",
        "category_summary_csv": root / "m1_fixed_clv_category_summary.csv",
        "examples_csv": root / "m1_fixed_clv_error_examples.csv",
        "json": root / "m1_fixed_clv_segment_error_diagnostic.json",
    }
    frame_mapping = {
        "segment_metrics_csv": "segment_metrics",
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
        "segment_metrics": report["segment_metrics"].to_dict("records"),
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
    occurrences = item_role_occurrences(
        users=users,
        segments=segments_for_users(prepared["cache"], users),
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
    segment_metrics = _segment_metrics(
        users=users, top50=top50, prepared=prepared
    )
    item_role_summary = summarize_segment_item_roles(occurrences)
    contrasts = miss_false_positive_contrasts(item_role_summary)
    category_summary = _category_summary(occurrences, cfg.top_examples)
    examples = _examples(occurrences, cfg.top_examples)
    report = {
        "segment_metrics": segment_metrics,
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
    print("\n===== 2) 정답·적중·오추천 상품 특성 =====")
    print(item_role_summary.to_string(index=False))
    print("\\n===== 3) 정답 누락 - Top-10 오추천 격차 =====")
    print(contrasts.to_string(index=False))
    print("\\n===== 4) 실제 상품 예시 =====")
    print(examples.to_string(index=False))
    print("\\n결과 파일:", report["paths"])
    return report


if __name__ == "__main__":
    print(
        json.dumps(
            preflight_summary(configure_fixed_segment_error_diagnostic()),
            ensure_ascii=False,
            indent=2,
        )
    )
