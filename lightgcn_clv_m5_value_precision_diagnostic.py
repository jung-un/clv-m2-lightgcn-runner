"""Checkpoint-only value-precision diagnostic for the next M5 design.

No training, checkpoint selection, reranking, test or holdout.  The existing
seed-42 M1 checkpoint answers three train-only questions that decide three
open M5 design choices:

1. shrinking the user's mean transaction value toward the population mean with
   the observed transaction count, as the CLV spend literature does, versus the
   raw observed mean: which one sits closer to the price position of the new
   items M1 missed?
2. does that price fit change across the fixed historical CLV segments, and is
   the high-minus-low slope outside a user-level bootstrap interval in both
   datasets?
3. how many BPR positive rows are first purchases of that item, overall and by
   CLV segment, since evaluation removes every train pair?

Pairs, groups, price percentiles and the M1 checkpoint loader are reused from
``lightgcn_clv_candidate_relation_diagnostic`` so the two diagnostics read the
same population.
"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from clv_run_state import file_sha256
import lightgcn_clv_axis_specific_test10 as test10
import lightgcn_clv_candidate_relation_diagnostic as relation
import lightgcn_clv_fixed_segment_error_diagnostic as fixed
import lightgcn_clv_history_item_fit_diagnostic as item_fit
import lightgcn_clv_v3 as v3


CODE_VERSION = "m5-value-precision-diagnostic-v1"
SHRINKAGE_STRENGTH = 5.0
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 42
RAW_SIGNAL = "value_mean_transaction_raw"
SHRUNKEN_SIGNAL = "value_mean_transaction_shrunken"
REFERENCE_SIGNAL = "value_spend_weighted_price_position"
SIGNAL_ORDER = (RAW_SIGNAL, SHRUNKEN_SIGNAL, REFERENCE_SIGNAL)


def configure_value_precision_diagnostic(
    dataset: str = "dunnhumby", **overrides
) -> relation.CandidateRelationDiagnosticConfig:
    dataset = dataset.lower()
    defaults = {
        "out_dir": (
            f"{v3.default_out_dir(dataset)}_m5_value_precision_diagnostic_v1"
        )
    }
    return relation.configure_candidate_relation_diagnostic(
        dataset, **(defaults | overrides)
    )


def preflight_summary(cfg) -> dict:
    return {
        "code_version": CODE_VERSION,
        "dataset": cfg.dataset,
        "training": False,
        "checkpoint_selection": False,
        "final_test_executed": False,
        "holdout_executed": False,
        "model": "existing seed-42 M1 checkpoint",
        "new_item_task": True,
        "signals": {
            RAW_SIGNAL: (
                "negative absolute distance between the item price percentile "
                "and the percentile of the observed mean transaction value"
            ),
            SHRUNKEN_SIGNAL: (
                "the same distance after shrinking the mean transaction value "
                "toward the transaction-weighted population mean with weight "
                f"{SHRINKAGE_STRENGTH} transactions"
            ),
            REFERENCE_SIGNAL: (
                "negative absolute distance from the user's train purchase-"
                "amount-weighted overall item-price percentile; the relation "
                "already supported in both datasets"
            ),
        },
        "reading_rule": {
            "shrinkage_adopted": (
                "keep the shrunken value only if its pair-balanced win rate is "
                "at least the raw win rate in both datasets and the user-level "
                "bootstrap interval of the paired difference excludes zero in "
                "at least one dataset"
            ),
            "clv_slope": (
                "call the CLV slope of the price fit real only if the "
                "high-minus-low bootstrap interval excludes zero; opposite "
                "signs across datasets support a learned gate direction"
            ),
            "first_purchase_rows": "reported only; no pass or fail",
        },
        "statistical_note": (
            "single-checkpoint descriptive development diagnostic; bootstrap "
            "intervals describe this evaluation population only and no "
            "generalization or causal claim is made"
        ),
        "out_dir": cfg.out_dir,
    }


def transaction_counts(axes: dict) -> np.ndarray:
    """Observed transactions per user in the CLV snapshot window."""

    if "repeat_transaction_count" in axes:
        counts = np.asarray(axes["repeat_transaction_count"], dtype=np.float64)
        return counts + 1.0
    return np.asarray(axes["n_behavior_score"], dtype=np.float64)


def shrunken_mean_value(
    value: np.ndarray,
    counts: np.ndarray,
    valid: np.ndarray,
    *,
    strength: float = SHRINKAGE_STRENGTH,
) -> tuple[np.ndarray, float]:
    """Shrink each mean transaction value toward the population mean."""

    if strength <= 0:
        raise ValueError("축소강도는 양수여야 합니다")
    value = np.asarray(value, dtype=np.float64)
    counts = np.maximum(np.asarray(counts, dtype=np.float64), 0.0)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(value) & (counts > 0)
    if not valid.any():
        raise ValueError("축소추정에 사용할 유효 사용자가 없습니다")
    population = float(
        np.average(value[valid], weights=counts[valid])
    )
    shrunken = np.full(len(value), np.nan, dtype=np.float64)
    shrunken[valid] = (
        counts[valid] * value[valid] + strength * population
    ) / (counts[valid] + strength)
    return shrunken, population


def user_value_positions(
    axes: dict,
    price: dict[str, np.ndarray],
    *,
    strength: float = SHRINKAGE_STRENGTH,
) -> tuple[dict[str, np.ndarray], dict]:
    """Map each user to a [0,1] value position for every signal."""

    value = np.asarray(axes["v_behavior_score"], dtype=np.float64)
    counts = transaction_counts(axes)
    valid = (
        np.asarray(axes["valid_user"], dtype=bool)
        & np.asarray(axes["value_valid"], dtype=bool)
        & np.isfinite(value)
    )
    raw = relation._percentile(value, valid)
    shrunken_value, population = shrunken_mean_value(
        value, counts, valid, strength=strength
    )
    shrunken = relation._percentile(
        shrunken_value, valid & np.isfinite(shrunken_value)
    )
    positions = {
        RAW_SIGNAL: raw,
        SHRUNKEN_SIGNAL: shrunken,
        REFERENCE_SIGNAL: np.asarray(price["user_overall"], dtype=np.float64),
    }
    both = valid & np.isfinite(raw) & np.isfinite(shrunken)
    moved = np.abs(raw[both] - shrunken[both])
    diagnostics = {
        "shrinkage_strength": float(strength),
        "population_mean_transaction_value": population,
        "valid_user_share": float(valid.mean()),
        "median_transaction_count": float(np.median(counts[valid])),
        "mean_absolute_percentile_move": float(moved.mean()) if len(moved) else 0.0,
        "max_absolute_percentile_move": float(moved.max()) if len(moved) else 0.0,
    }
    return positions, diagnostics


def _pair_rows(
    *,
    users: np.ndarray,
    top10: np.ndarray,
    truth: dict[int, np.ndarray],
    membership: pd.DataFrame,
    item_price: np.ndarray,
    positions: dict[str, np.ndarray],
) -> pd.DataFrame:
    by_user = membership.set_index("user_idx")
    rows = []
    for position, user in enumerate(np.asarray(users, dtype=np.int64)):
        truth_items = np.asarray(truth[int(user)], dtype=np.int64)
        truth_set = set(map(int, truth_items))
        ranked = np.asarray(top10[position], dtype=np.int64)
        ranked_set = set(map(int, ranked))
        missed = np.asarray(
            [item for item in truth_items if int(item) not in ranked_set],
            dtype=np.int64,
        )
        false_positive = np.asarray(
            [item for item in ranked if int(item) not in truth_set],
            dtype=np.int64,
        )
        if not len(missed) or not len(false_positive):
            continue
        group = by_user.loc[int(user)]
        for signal in SIGNAL_ORDER:
            user_position = positions[signal][int(user)]
            if not np.isfinite(user_position):
                continue
            missed_value = -np.abs(item_price[missed] - user_position)
            false_value = -np.abs(item_price[false_positive] - user_position)
            difference = (
                missed_value[:, None] - false_value[None, :]
            ).reshape(-1)
            difference = difference[np.isfinite(difference)]
            if not len(difference):
                continue
            tolerance = 1e-12
            wins = int(np.sum(difference > tolerance))
            losses = int(np.sum(difference < -tolerance))
            ties = int(len(difference) - wins - losses)
            rows.append(
                {
                    "user_idx": int(user),
                    "signal": signal,
                    "fixed_clv_segment": group["fixed_clv_segment"],
                    "q_n": float(group["q_n"]),
                    "q_v": float(group["q_v"]),
                    "historical_clv_proxy": float(group["historical_clv_proxy"]),
                    "candidate_pair_count": int(len(difference)),
                    "truth_wins": wins,
                    "ties": ties,
                    "false_positive_wins": losses,
                    "balanced_win_rate": (wins + 0.5 * ties) / len(difference),
                }
            )
    return pd.DataFrame(rows)


def summarize_win_rates(per_user: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for signal in SIGNAL_ORDER:
        selected = per_user[per_user.signal.eq(signal)]
        if selected.empty:
            continue
        groups = [("overall", "전체", selected)]
        for name in fixed.SEGMENT_ORDER:
            groups.append(
                (
                    "fixed_clv_segment",
                    name,
                    selected[selected.fixed_clv_segment.eq(name)],
                )
            )
        for group_type, group, frame in groups:
            if frame.empty:
                continue
            pairs = int(frame.candidate_pair_count.sum())
            rows.append(
                {
                    "signal": signal,
                    "group_type": group_type,
                    "group": group,
                    "n_users": int(len(frame)),
                    "candidate_pair_count": pairs,
                    "pair_balanced_win_rate": float(
                        (frame.truth_wins.sum() + 0.5 * frame.ties.sum()) / pairs
                    ),
                    "user_macro_win_rate": float(frame.balanced_win_rate.mean()),
                }
            )
    return pd.DataFrame(rows)


def _percentile_interval(samples: np.ndarray) -> tuple[float, float]:
    low, high = np.percentile(samples, [2.5, 97.5])
    return float(low), float(high)


def bootstrap_contrasts(
    per_user: pd.DataFrame,
    *,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """User-level bootstrap for the two decisions this diagnostic makes."""

    wide = per_user.pivot_table(
        index="user_idx",
        columns="signal",
        values="balanced_win_rate",
        aggfunc="mean",
    )
    segment = (
        per_user.drop_duplicates("user_idx")
        .set_index("user_idx")["fixed_clv_segment"]
        .reindex(wide.index)
    )
    available = [name for name in SIGNAL_ORDER if name in wide.columns]
    paired = wide.dropna(subset=available)
    segment = segment.reindex(paired.index)
    if paired.empty:
        raise RuntimeError("bootstrap에 사용할 사용자 행이 없습니다")

    low_name, high_name = fixed.SEGMENT_ORDER[0], fixed.SEGMENT_ORDER[2]
    rng = np.random.default_rng(seed)
    index = np.arange(len(paired))
    values = {name: paired[name].to_numpy(np.float64) for name in available}
    is_low = (segment.to_numpy() == low_name)
    is_high = (segment.to_numpy() == high_name)

    def contrasts(rows: np.ndarray) -> dict[str, float]:
        result = {}
        if RAW_SIGNAL in values and SHRUNKEN_SIGNAL in values:
            result["shrunken_minus_raw"] = float(
                (values[SHRUNKEN_SIGNAL][rows] - values[RAW_SIGNAL][rows]).mean()
            )
        low_rows, high_rows = rows[is_low[rows]], rows[is_high[rows]]
        for name in available:
            if len(low_rows) and len(high_rows):
                result[f"{name}__high_minus_low_clv"] = float(
                    values[name][high_rows].mean() - values[name][low_rows].mean()
                )
        return result

    observed = contrasts(index)
    draws: dict[str, list[float]] = {key: [] for key in observed}
    for _ in range(int(samples)):
        rows = rng.integers(0, len(index), size=len(index))
        for key, value in contrasts(rows).items():
            draws[key].append(value)

    report = {}
    for key, value in observed.items():
        collected = np.asarray(draws[key], dtype=np.float64)
        low, high = _percentile_interval(collected)
        report[key] = {
            "observed": value,
            "ci_low": low,
            "ci_high": high,
            "excludes_zero": bool(low > 0.0 or high < 0.0),
        }
    return {
        "bootstrap_samples": int(samples),
        "bootstrap_seed": int(seed),
        "paired_users": int(len(paired)),
        "low_clv_users": int(is_low.sum()),
        "high_clv_users": int(is_high.sum()),
        "contrasts": report,
    }


def first_purchase_row_stats(
    train: pd.DataFrame, membership: pd.DataFrame
) -> dict:
    """Share of BPR positive rows that are the first purchase of that item."""

    order = ["t"] + (["b_raw"] if "b_raw" in train.columns else [])
    frame = train[["u_idx", "i_idx", *order]].sort_values(
        ["u_idx", *order], kind="stable"
    )
    first = ~frame.duplicated(subset=["u_idx", "i_idx"], keep="first")
    per_user = (
        pd.DataFrame({"u_idx": frame.u_idx.to_numpy(), "first": first.to_numpy()})
        .groupby("u_idx", sort=False)["first"]
        .agg(["size", "sum"])
        .rename(columns={"size": "rows", "sum": "first_rows"})
    )
    per_user["first_share"] = per_user.first_rows / per_user.rows

    def block(frame: pd.DataFrame) -> dict:
        rows = float(frame.rows.sum())
        return {
            "users": int(len(frame)),
            "train_rows": int(rows),
            "first_purchase_rows": int(frame.first_rows.sum()),
            "first_purchase_row_share": float(frame.first_rows.sum() / rows),
            "user_mean_first_purchase_share": float(frame.first_share.mean()),
        }

    segments = {}
    joined = membership.join(per_user, on="user_idx", how="inner").dropna(
        subset=["rows"]
    )
    for name in fixed.SEGMENT_ORDER:
        selected = joined[joined.fixed_clv_segment.eq(name)]
        if not selected.empty:
            segments[name] = block(selected)
    return {
        "overall": block(per_user),
        "rows_per_unique_pair": float(
            per_user.rows.sum() / max(per_user.first_rows.sum(), 1)
        ),
        "evaluation_user_segments": segments,
    }


def _paths(cfg) -> dict[str, Path]:
    root = Path(cfg.out_dir)
    root.mkdir(parents=True, exist_ok=True)
    stem = f"m5_value_precision_{cfg.dataset}"
    return {
        "summary_csv": root / f"{stem}_summary.csv",
        "per_user_csv": root / f"{stem}_per_user.csv",
        "json": root / f"{stem}_diagnostic.json",
    }


@torch.no_grad()
def run_value_precision_diagnostic(cfg=None) -> dict[str, str]:
    cfg = cfg or configure_value_precision_diagnostic()
    summary_preflight = preflight_summary(cfg)
    print(json.dumps(summary_preflight, ensure_ascii=False, indent=2))
    prepared, model, checkpoint, record, axes = relation._prepare_and_load(cfg)
    model.eval()
    users = np.asarray(prepared["cache"].users, dtype=np.int64)
    user_embedding, item_embedding = model.propagate_pref()
    ranked_users, top10 = item_fit._masked_topk(
        user_embedding,
        item_embedding,
        prepared,
        max_k=10,
        batch_size=cfg.eval_batch_size,
    )
    if not np.array_equal(ranked_users, users):
        raise RuntimeError("M1 평가 사용자 순서가 달라졌습니다")
    del user_embedding, item_embedding

    membership, thresholds = relation._membership(prepared, axes)
    train = prepared["data"]["train"]
    n_items = int(prepared["data"]["n_items"])
    item_category, _ = relation._item_categories(train, n_items)
    price, price_diagnostics = relation._price_inputs(
        train,
        n_users=int(prepared["data"]["n_users"]),
        n_items=n_items,
        item_category=item_category,
    )
    positions, position_diagnostics = user_value_positions(axes, price)
    per_user = _pair_rows(
        users=users,
        top10=top10,
        truth=prepared["cache"].gt,
        membership=membership,
        item_price=np.asarray(price["item_overall"], dtype=np.float64),
        positions=positions,
    )
    if per_user.empty:
        raise RuntimeError("비교 가능한 M1 누락정답–오추천 쌍이 없습니다")
    summary = summarize_win_rates(per_user)
    bootstrap = bootstrap_contrasts(per_user)
    first_purchase = first_purchase_row_stats(train, membership)

    paths = _paths(cfg)
    test10._atomic_csv(paths["summary_csv"], summary)
    test10._atomic_csv(paths["per_user_csv"], per_user)
    test10._atomic_json(
        paths["json"],
        {
            "code_version": CODE_VERSION,
            "config": asdict(cfg),
            "preflight": summary_preflight,
            "checkpoint": {
                "path": str(checkpoint),
                "sha256": file_sha256(checkpoint),
                "record": record,
            },
            "group_thresholds": thresholds,
            "price_diagnostics": price_diagnostics,
            "value_position_diagnostics": position_diagnostics,
            "summary_rows": summary.to_dict("records"),
            "bootstrap": bootstrap,
            "first_purchase_rows": first_purchase,
            "result_paths": {key: str(value) for key, value in paths.items()},
        },
    )
    print("\n1) 누락 정답이 M1 Top-10 오추천을 이기는 비율")
    print(summary.to_string(index=False))
    print("\n2) 사용자 단위 bootstrap 대비")
    print(json.dumps(bootstrap["contrasts"], ensure_ascii=False, indent=2))
    print("\n3) 첫 구매 학습행 비중")
    print(json.dumps(first_purchase, ensure_ascii=False, indent=2))
    print("\n결과 파일:", {key: str(value) for key, value in paths.items()})
    return {key: str(value) for key, value in paths.items()}


if __name__ == "__main__":
    print(
        json.dumps(
            preflight_summary(configure_value_precision_diagnostic()),
            ensure_ascii=False,
            indent=2,
        )
    )
