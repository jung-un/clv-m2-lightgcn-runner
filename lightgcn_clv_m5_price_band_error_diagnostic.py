"""Checkpoint-only price-band error diagnostic for the M2/M4 division of labor.

No training, checkpoint selection, reranking, test or holdout. The proposed M5
routes pairwise comparisons: a negative from another price band can be
separated by the fixed M2 value block, while a negative from the user's own
value band cannot, so its gradient must reshape the collaborative ID
embeddings. Before building that model, the existing seed-42 M1 checkpoint
answers three questions:

1. how many of M1's (missed truth, Top-10 false positive) pairs already sit in
   the same price band, i.e. how much within-band error is left for ID+M4;
2. whether the q_V price-position signal separates cross-band pairs but not
   same-band pairs, i.e. whether the two roles really split;
3. how much more often an unbought item in the user's own value band is a
   future truth than a uniformly drawn unbought item (false-negative risk of
   band-matched negatives).

Pairs, CLV groups, item price percentiles and the q_V value position are the
ones already validated by ``lightgcn_clv_m5_value_precision_diagnostic``.
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
import lightgcn_clv_m5_value_precision_diagnostic as precision
import lightgcn_clv_v3 as v3


CODE_VERSION = "m5-price-band-error-diagnostic-v1"
N_BANDS = 4
MIN_HIGH_CLV_SAME_BAND_SHARE = 0.30


def configure_price_band_error_diagnostic(
    dataset: str = "dunnhumby", **overrides
) -> relation.CandidateRelationDiagnosticConfig:
    dataset = dataset.lower()
    defaults = {
        "out_dir": f"{v3.default_out_dir(dataset)}_m5_price_band_error_diagnostic_v1"
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
        "model": "existing seed-42 M1 checkpoint (one uniform negative BPR)",
        "new_item_task": True,
        "price_band": (
            f"{N_BANDS} equal-count bands of the train mean item-price "
            "percentile; user value band = the band of the user's q_V"
        ),
        "questions": {
            "same_band_error_share": (
                "share of (missed truth, Top-10 false positive) pairs whose "
                "two items fall in the same price band, with the share expected "
                "if the two items' bands were independent"
            ),
            "division_of_labor": (
                "q_V price-position balanced win rate separately for cross-band "
                "and same-band pairs"
            ),
            "false_negative_lift": (
                "future-truth rate among unbought items in the user's value band "
                "divided by the future-truth rate among all unbought items"
            ),
        },
        "reading_rule": {
            "enough_within_band_error": (
                "fixed high-CLV same-band pair share >= "
                f"{MIN_HIGH_CLV_SAME_BAND_SHARE} in both datasets"
            ),
            "roles_split": (
                "fixed high-CLV cross-band q_V win rate > 0.5 and "
                "|same-band win rate - 0.5| < |cross-band win rate - 0.5| "
                "in both datasets"
            ),
            "design_supported": "enough_within_band_error and roles_split",
            "false_negative_lift": "reported only; informs the band-negative cap",
        },
        "statistical_note": (
            "single-checkpoint descriptive development diagnostic; no bootstrap, "
            "significance, generalization or causal claim"
        ),
        "out_dir": cfg.out_dir,
    }


def price_bands(percentile: np.ndarray, n_bands: int = N_BANDS) -> np.ndarray:
    """Map a (0,1] percentile to equal-width bands; missing values get -1."""

    percentile = np.asarray(percentile, dtype=np.float64)
    bands = np.full(len(percentile), -1, dtype=np.int64)
    finite = np.isfinite(percentile)
    bands[finite] = np.minimum(
        (percentile[finite] * n_bands).astype(np.int64), n_bands - 1
    )
    return bands


def train_band_counts(
    train: pd.DataFrame,
    users: np.ndarray,
    item_band: np.ndarray,
    n_bands: int = N_BANDS,
) -> np.ndarray:
    """Distinct purchased items per (evaluation user, price band)."""

    users = np.asarray(users, dtype=np.int64)
    row_of = {int(user): row for row, user in enumerate(users)}
    pairs = train[["u_idx", "i_idx"]].drop_duplicates()
    pairs = pairs[pairs.u_idx.isin(row_of)]
    rows = pairs.u_idx.map(row_of).to_numpy(np.int64)
    bands = item_band[pairs.i_idx.to_numpy(np.int64)]
    keep = bands >= 0
    counts = np.zeros((len(users), n_bands), dtype=np.int64)
    np.add.at(counts, (rows[keep], bands[keep]), 1)
    return counts


def _pair_rows(
    *,
    users: np.ndarray,
    top10: np.ndarray,
    truth: dict[int, np.ndarray],
    membership: pd.DataFrame,
    item_price: np.ndarray,
    item_band: np.ndarray,
    user_position: np.ndarray,
    user_band: np.ndarray,
    purchased_by_band: np.ndarray,
    band_sizes: np.ndarray,
) -> pd.DataFrame:
    by_user = membership.set_index("user_idx")
    rows = []
    tolerance = 1e-12
    for position, user in enumerate(np.asarray(users, dtype=np.int64)):
        truth_items = np.asarray(truth[int(user)], dtype=np.int64)
        truth_items = truth_items[item_band[truth_items] >= 0]
        ranked = np.asarray(top10[position], dtype=np.int64)
        ranked_set = set(map(int, ranked))
        truth_set = set(map(int, truth_items))
        missed = np.asarray(
            [item for item in truth_items if int(item) not in ranked_set],
            dtype=np.int64,
        )
        false_positive = np.asarray(
            [
                item
                for item in ranked
                if int(item) not in truth_set and item_band[int(item)] >= 0
            ],
            dtype=np.int64,
        )
        value_position = user_position[int(user)]
        if not len(missed) or not len(false_positive) or not np.isfinite(value_position):
            continue

        missed_band = item_band[missed]
        false_band = item_band[false_positive]
        same = (missed_band[:, None] == false_band[None, :]).reshape(-1)
        difference = (
            -np.abs(item_price[missed] - value_position)[:, None]
            + np.abs(item_price[false_positive] - value_position)[None, :]
        ).reshape(-1)
        wins = difference > tolerance
        ties = np.abs(difference) <= tolerance
        chance = float(
            np.sum(
                np.bincount(missed_band, minlength=N_BANDS) / len(missed)
                * np.bincount(false_band, minlength=N_BANDS) / len(false_positive)
            )
        )

        band = int(user_band[int(user)])
        unbought = band_sizes - purchased_by_band[position]
        truth_in_band = int(np.sum(item_band[truth_items] == band)) if band >= 0 else 0
        group = by_user.loc[int(user)]
        rows.append(
            {
                "user_idx": int(user),
                "fixed_clv_segment": group["fixed_clv_segment"],
                "q_v_position": float(value_position),
                "user_value_band": band,
                "candidate_pair_count": int(len(same)),
                "same_band_pairs": int(same.sum()),
                "chance_same_band_share": chance,
                "same_band_wins": int((wins & same).sum()),
                "same_band_ties": int((ties & same).sum()),
                "cross_band_wins": int((wins & ~same).sum()),
                "cross_band_ties": int((ties & ~same).sum()),
                "truth_count": int(len(truth_items)),
                "truth_in_value_band": truth_in_band,
                "unbought_items": int(unbought.sum()),
                "unbought_in_value_band": int(unbought[band]) if band >= 0 else 0,
            }
        )
    return pd.DataFrame(rows)


def _balanced(wins: float, ties: float, pairs: float) -> float:
    return float((wins + 0.5 * ties) / pairs) if pairs > 0 else float("nan")


def summarize(per_user: pd.DataFrame) -> pd.DataFrame:
    groups = [("overall", "전체", per_user)]
    for name in fixed.SEGMENT_ORDER:
        groups.append(
            (
                "fixed_clv_segment",
                name,
                per_user[per_user.fixed_clv_segment.eq(name)],
            )
        )
    rows = []
    for group_type, group, frame in groups:
        if frame.empty:
            continue
        pairs = float(frame.candidate_pair_count.sum())
        same = float(frame.same_band_pairs.sum())
        cross = pairs - same
        banded = frame[frame.user_value_band >= 0]
        in_band_rate = (
            banded.truth_in_value_band.sum() / banded.unbought_in_value_band.sum()
            if banded.unbought_in_value_band.sum() > 0
            else float("nan")
        )
        overall_rate = (
            banded.truth_count.sum() / banded.unbought_items.sum()
            if banded.unbought_items.sum() > 0
            else float("nan")
        )
        rows.append(
            {
                "group_type": group_type,
                "group": group,
                "n_users": int(len(frame)),
                "candidate_pair_count": int(pairs),
                "same_band_pair_share": same / pairs,
                "chance_same_band_share": float(
                    np.average(
                        frame.chance_same_band_share,
                        weights=frame.candidate_pair_count,
                    )
                ),
                "same_band_q_v_win_rate": _balanced(
                    frame.same_band_wins.sum(), frame.same_band_ties.sum(), same
                ),
                "cross_band_q_v_win_rate": _balanced(
                    frame.cross_band_wins.sum(), frame.cross_band_ties.sum(), cross
                ),
                "truth_rate_unbought_in_value_band": float(in_band_rate),
                "truth_rate_all_unbought": float(overall_rate),
                "false_negative_lift": float(in_band_rate / overall_rate)
                if overall_rate and np.isfinite(overall_rate)
                else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def dataset_reading(summary: pd.DataFrame) -> dict:
    high = summary[
        summary.group_type.eq("fixed_clv_segment")
        & summary.group.eq(fixed.SEGMENT_ORDER[2])
    ]
    if high.empty:
        raise RuntimeError("고CLV 구간 요약이 없습니다")
    row = high.iloc[0]
    same_gap = abs(float(row.same_band_q_v_win_rate) - 0.5)
    cross_gap = abs(float(row.cross_band_q_v_win_rate) - 0.5)
    enough = bool(row.same_band_pair_share >= MIN_HIGH_CLV_SAME_BAND_SHARE)
    split = bool(row.cross_band_q_v_win_rate > 0.5 and same_gap < cross_gap)
    return {
        "high_clv_same_band_pair_share": float(row.same_band_pair_share),
        "high_clv_chance_same_band_share": float(row.chance_same_band_share),
        "high_clv_same_band_q_v_win_rate": float(row.same_band_q_v_win_rate),
        "high_clv_cross_band_q_v_win_rate": float(row.cross_band_q_v_win_rate),
        "high_clv_false_negative_lift": float(row.false_negative_lift),
        "enough_within_band_error": enough,
        "roles_split": split,
        "design_supported_in_this_dataset": bool(enough and split),
        "cross_dataset_decision_pending": True,
    }


def _paths(cfg) -> dict[str, Path]:
    root = Path(cfg.out_dir)
    root.mkdir(parents=True, exist_ok=True)
    stem = f"m5_price_band_error_{cfg.dataset}"
    return {
        "summary_csv": root / f"{stem}_summary.csv",
        "per_user_csv": root / f"{stem}_per_user.csv",
        "json": root / f"{stem}_diagnostic.json",
    }


@torch.no_grad()
def run_price_band_error_diagnostic(cfg=None) -> dict[str, str]:
    cfg = cfg or configure_price_band_error_diagnostic()
    preflight = preflight_summary(cfg)
    print(json.dumps(preflight, ensure_ascii=False, indent=2))
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
    positions, position_diagnostics = precision.user_value_positions(axes, price)
    item_price = np.asarray(price["item_overall"], dtype=np.float64)
    item_band = price_bands(item_price)
    user_position = positions[precision.RAW_SIGNAL]
    user_band = price_bands(user_position)
    band_sizes = np.bincount(item_band[item_band >= 0], minlength=N_BANDS)

    per_user = _pair_rows(
        users=users,
        top10=top10,
        truth=prepared["cache"].gt,
        membership=membership,
        item_price=item_price,
        item_band=item_band,
        user_position=user_position,
        user_band=user_band,
        purchased_by_band=train_band_counts(train, users, item_band),
        band_sizes=band_sizes,
    )
    if per_user.empty:
        raise RuntimeError("비교 가능한 M1 누락정답–오추천 쌍이 없습니다")
    summary = summarize(per_user)
    reading = dataset_reading(summary)

    paths = _paths(cfg)
    test10._atomic_csv(paths["summary_csv"], summary)
    test10._atomic_csv(paths["per_user_csv"], per_user)
    test10._atomic_json(
        paths["json"],
        {
            "code_version": CODE_VERSION,
            "config": asdict(cfg),
            "preflight": preflight,
            "checkpoint": {
                "path": str(checkpoint),
                "sha256": file_sha256(checkpoint),
                "record": record,
            },
            "group_thresholds": thresholds,
            "price_diagnostics": price_diagnostics,
            "value_position_diagnostics": position_diagnostics,
            "band_sizes": band_sizes.tolist(),
            "summary_rows": summary.to_dict("records"),
            "reading": reading,
            "result_paths": {key: str(value) for key, value in paths.items()},
        },
    )
    print("\n1) 가격구간 기준 누락정답–오추천 쌍")
    print(summary.to_string(index=False))
    print("\n2) 이 데이터의 판독")
    print(json.dumps(reading, ensure_ascii=False, indent=2))
    print("\n결과 파일:", {key: str(value) for key, value in paths.items()})
    return {key: str(value) for key, value in paths.items()}


if __name__ == "__main__":
    print(
        json.dumps(
            preflight_summary(configure_price_band_error_diagnostic()),
            ensure_ascii=False,
            indent=2,
        )
    )
