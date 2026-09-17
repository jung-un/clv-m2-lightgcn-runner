"""Checkpoint-only diagnostic of CLV information beyond M1 and popularity.

For each held-out new-item truth, controls are unseen non-truth items from the
same train distinct-buyer popularity decile whose existing M1 score is nearest
to the truth score.  The diagnostic then asks whether three train-only,
candidate-specific relations rank truths above those matched controls:

* N: proximity between user q_N and the item's distinct-purchaser mean q_N;
* V: proximity between user q_V and the item's train price percentile;
* N/V: the arithmetic mean of the two relations.

There is no training, checkpoint selection, reranking, final test, or holdout.
The result is descriptive mechanism evidence, not a model-performance result.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
import torch

from clv_run_state import file_sha256
import lightgcn_clv_axis_specific_test10 as test10
import lightgcn_clv_candidate_relation_diagnostic as relation
import lightgcn_clv_fixed_segment_error_diagnostic as fixed
import lightgcn_clv_v3 as v3


CODE_VERSION = "clv-incremental-candidate-signal-diagnostic-v3"
N_SIGNAL = "n_purchaser_activity_fit"
V_SIGNAL = "v_price_position_fit"
NV_SIGNAL = "mean_n_v_candidate_fit"
QC_LEVEL = "q_c_user_level_constant"
SIGNAL_ORDER = (N_SIGNAL, V_SIGNAL, NV_SIGNAL)
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 42
DEFAULT_SCORE_GAP_CAPS = (0.25, 0.10)
UPPER_TAIL_CUT = 0.8
UPPER_TAIL_GROUPS = ("high_q_c", "high_q_n", "high_q_v")


@dataclass(frozen=True)
class IncrementalCandidateSignalConfig:
    dataset: str
    out_dir: str = ""
    baseline_result_dir: str = ""
    m1_checkpoint_dir: str = ""
    m1_checkpoint: str = ""
    eval_batch_size: int = 64
    cross_fit_folds: int = 5
    min_transition_support_users: int = 5
    transition_kappa: float = 20.0
    transition_log_lift_cap: float = float(np.log(3.0))
    popularity_bins: int = 10
    candidate_pool_per_bin: int = 512
    controls_per_truth: int = 5
    candidate_pool_seed: int = 20260917
    bootstrap_samples: int = BOOTSTRAP_SAMPLES
    bootstrap_seed: int = BOOTSTRAP_SEED
    score_gap_caps_in_user_sd: tuple[float, ...] = DEFAULT_SCORE_GAP_CAPS


def configure_incremental_candidate_signal_diagnostic(
    dataset: str = "dunnhumby", **overrides
) -> IncrementalCandidateSignalConfig:
    base = relation.configure_candidate_relation_diagnostic(dataset)
    values = asdict(base) | {
        "out_dir": (
            f"{v3.default_out_dir(dataset.lower())}"
            "_clv_incremental_candidate_signal_diagnostic_v2"
        )
    }
    values.update(overrides)
    cfg = IncrementalCandidateSignalConfig(**values)
    if cfg.dataset not in {"dunnhumby", "hm"}:
        raise ValueError("dataset은 dunnhumby 또는 hm이어야 합니다")
    if not cfg.out_dir:
        raise ValueError("out_dir가 필요합니다")
    numeric = (
        cfg.eval_batch_size,
        cfg.popularity_bins,
        cfg.candidate_pool_per_bin,
        cfg.controls_per_truth,
        cfg.bootstrap_samples,
    )
    if any(int(value) <= 0 for value in numeric):
        raise ValueError("배치·분위·후보·대조수·bootstrap 수는 양수여야 합니다")
    caps = tuple(float(value) for value in cfg.score_gap_caps_in_user_sd)
    if not caps or any(not np.isfinite(value) or value <= 0 for value in caps):
        raise ValueError("M1 점수차 상한은 하나 이상의 양수 유한값이어야 합니다")
    if len(set(caps)) != len(caps):
        raise ValueError("M1 점수차 상한은 중복될 수 없습니다")
    return cfg


def preflight_summary(cfg: IncrementalCandidateSignalConfig) -> dict:
    split = (
        "historical_development_days_684_690"
        if cfg.dataset == "dunnhumby"
        else "existing_hm2y_validation"
    )
    return {
        "code_version": CODE_VERSION,
        "dataset": cfg.dataset,
        "split": split,
        "training": False,
        "checkpoint_selection": False,
        "final_test_executed": False,
        "holdout_executed": False,
        "model": "existing seed-42 M1 checkpoint",
        "new_item_task": True,
        "matching": {
            "population": "held-out new-item truths and unseen non-truth controls",
            "popularity": "same train distinct-buyer popularity bin",
            "m1_score": "nearest existing M1 score within the same popularity bin",
            "popularity_bins": int(cfg.popularity_bins),
            "candidate_pool_per_bin": int(cfg.candidate_pool_per_bin),
            "controls_per_truth": int(cfg.controls_per_truth),
            "candidate_pool_seed": int(cfg.candidate_pool_seed),
        },
        "signals": {
            N_SIGNAL: "1-|user q_N-item distinct-purchaser mean q_N|",
            V_SIGNAL: "1-|user q_V-item train price percentile|",
            NV_SIGNAL: "arithmetic mean of the N and V candidate relations",
            QC_LEVEL: (
                "q_C is a user-level constant and therefore cannot distinguish "
                "candidates without an item interaction"
            ),
        },
        "reading_rule": (
            "a signal is descriptive incremental candidate information only "
            "when its user-macro balanced win-rate interval is above 0.5 in "
            "both datasets; this does not establish that M2, M3, or M4 can "
            "convert the signal into recommendation improvement"
        ),
        "robustness_rule": {
            "score_gap_caps_in_user_sd": [
                float(value) for value in cfg.score_gap_caps_in_user_sd
            ],
            "overall_n_signal": (
                "record the N signal as score-match robust only when its "
                "user-macro interval remains above 0.5 under every score-gap "
                "cap in both datasets"
            ),
            "high_clv_n_signal": (
                "report the fixed high-CLV segment separately; an interval "
                "above 0.5 in both datasets is descriptive segment robustness, "
                "not an independent confirmatory test or a tuned hard gate"
            ),
            "upper_tail_comparison": (
                "compare the N signal for train-only high-q_C, high-q_N and "
                "high-q_V users using one fixed >=0.8 percentile rule"
            ),
        },
        "statistical_note": (
            "single-checkpoint development diagnostic; bootstrap describes "
            "the current evaluation users only, with no causal, significance, "
            "generalization, or model-performance claim"
        ),
        "out_dir": cfg.out_dir,
    }


def _percentile(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    result = np.full(len(values), np.nan, dtype=np.float64)
    if valid.any():
        result[valid] = rankdata(values[valid], method="average") / int(valid.sum())
    return result


def build_item_context(
    train: pd.DataFrame,
    *,
    n_items: int,
    q_n: np.ndarray,
    clv_valid: np.ndarray,
    popularity_bins: int = 10,
) -> dict[str, np.ndarray]:
    """Create train-only item purchaser context, price, and popularity bins."""

    required = {"u_idx", "i_idx", "up"}
    missing = required.difference(train.columns)
    if missing:
        raise KeyError(f"train 필수열이 없습니다: {sorted(missing)}")
    q_n = np.asarray(q_n, dtype=np.float64)
    clv_valid = np.asarray(clv_valid, dtype=bool)
    pairs = train[["u_idx", "i_idx"]].drop_duplicates()
    users = pairs.u_idx.to_numpy(np.int64, copy=False)
    items = pairs.i_idx.to_numpy(np.int64, copy=False)
    valid_pair = clv_valid[users] & np.isfinite(q_n[users])
    buyer_sum = np.bincount(
        items[valid_pair], weights=q_n[users[valid_pair]], minlength=n_items
    )
    buyer_count = np.bincount(items[valid_pair], minlength=n_items).astype(np.int64)
    buyer_qn_mean = np.divide(
        buyer_sum,
        buyer_count,
        out=np.full(n_items, np.nan),
        where=buyer_count > 0,
    )
    distinct_buyer_count = np.bincount(items, minlength=n_items).astype(np.int64)
    price = (
        train.groupby("i_idx", sort=True)["up"]
        .mean()
        .reindex(np.arange(n_items))
        .to_numpy(np.float64)
    )
    valid_price = np.isfinite(price)
    price_percentile = _percentile(price, valid_price)
    popularity_percentile = _percentile(
        distinct_buyer_count.astype(np.float64), distinct_buyer_count > 0
    )
    popularity_bin = np.floor(popularity_percentile * popularity_bins).astype(
        np.int64
    )
    popularity_bin = np.clip(popularity_bin, 0, popularity_bins - 1)
    return {
        "buyer_qn_mean": buyer_qn_mean,
        "buyer_qn_valid": buyer_count > 0,
        "distinct_buyer_count": distinct_buyer_count,
        "price_percentile": price_percentile,
        "price_valid": valid_price,
        "popularity_percentile": popularity_percentile,
        "popularity_bin": popularity_bin,
    }


def candidate_signal_values(
    *,
    user_qn: float,
    user_qv: float,
    user_qc: float,
    items: np.ndarray,
    buyer_qn_mean: np.ndarray,
    price_percentile: np.ndarray,
) -> dict[str, np.ndarray]:
    items = np.asarray(items, dtype=np.int64)
    n_fit = 1.0 - np.abs(buyer_qn_mean[items] - float(user_qn))
    v_fit = 1.0 - np.abs(price_percentile[items] - float(user_qv))
    return {
        N_SIGNAL: n_fit,
        V_SIGNAL: v_fit,
        NV_SIGNAL: (n_fit + v_fit) / 2.0,
        QC_LEVEL: np.full(len(items), float(user_qc), dtype=np.float64),
    }


def build_candidate_reservoir(
    popularity_bin: np.ndarray,
    *,
    bins: int,
    per_bin: int,
    seed: int,
) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    counts = {}
    for group in range(bins):
        members = np.flatnonzero(popularity_bin == group).astype(np.int64)
        original = len(members)
        if len(members) > per_bin:
            members = np.sort(rng.choice(members, size=per_bin, replace=False))
        selected.append(members)
        counts[str(group)] = {"available": original, "sampled": int(len(members))}
    reservoir = np.concatenate(selected) if selected else np.empty(0, np.int64)
    return reservoir, counts


def match_controls_for_truth(
    *,
    truth_item: int,
    truth_score: float,
    candidate_items: np.ndarray,
    candidate_scores: np.ndarray,
    popularity_bin: np.ndarray,
    forbidden_items: set[int],
    controls_per_truth: int,
) -> np.ndarray:
    candidate_items = np.asarray(candidate_items, dtype=np.int64)
    candidate_scores = np.asarray(candidate_scores, dtype=np.float64)
    same_bin = popularity_bin[candidate_items] == popularity_bin[int(truth_item)]
    allowed = np.fromiter(
        (int(item) not in forbidden_items for item in candidate_items),
        dtype=bool,
        count=len(candidate_items),
    )
    valid = same_bin & allowed & np.isfinite(candidate_scores)
    items = candidate_items[valid]
    scores = candidate_scores[valid]
    if not len(items):
        return np.empty(0, dtype=np.int64)
    order = np.lexsort((items, np.abs(scores - float(truth_score))))
    return items[order[:controls_per_truth]]


def _q_inputs(axes: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    q_n = np.asarray(axes["q_n"], dtype=np.float64)
    q_v = np.asarray(axes["q_v"], dtype=np.float64)
    raw_clv = np.asarray(axes["clv_proxy"], dtype=np.float64)
    valid = (
        np.asarray(axes["valid_user"], dtype=bool)
        & np.asarray(axes["activity_valid"], dtype=bool)
        & np.asarray(axes["value_valid"], dtype=bool)
        & np.isfinite(raw_clv)
    )
    q_c = _percentile(raw_clv, valid)
    return q_n, q_v, q_c, valid


def select_current_axes(dataset: str, prepared: dict, loader_axes: dict) -> dict:
    """Use the axes used by the current H&M model, not its legacy diagnostic."""

    if dataset == "hm":
        if "axes" not in prepared:
            raise KeyError("H&M prepared 결과에 현재 CLV axes가 없습니다")
        return prepared["axes"]
    return loader_axes


def _gap_suffix(cap: float) -> str:
    return f"gap_le_{float(cap):g}".replace(".", "p")


def _stat_columns(cap: float | None) -> dict[str, str]:
    if cap is None:
        return {
            "pairs": "candidate_pair_count",
            "wins": "truth_wins",
            "ties": "ties",
            "losses": "control_wins",
            "gap_sum": "score_gap_sum",
            "gap_max": "score_gap_max",
            "normalized_gap_sum": "normalized_score_gap_sum",
            "normalized_gap_max": "normalized_score_gap_max",
        }
    prefix = _gap_suffix(cap)
    return {
        "pairs": f"{prefix}_candidate_pair_count",
        "wins": f"{prefix}_truth_wins",
        "ties": f"{prefix}_ties",
        "losses": f"{prefix}_control_wins",
        "gap_sum": f"{prefix}_score_gap_sum",
        "gap_max": f"{prefix}_score_gap_max",
        "normalized_gap_sum": f"{prefix}_normalized_score_gap_sum",
        "normalized_gap_max": f"{prefix}_normalized_score_gap_max",
    }


@torch.no_grad()
def collect_matched_pair_rows(
    *,
    user_embedding: torch.Tensor,
    item_embedding: torch.Tensor,
    prepared: dict,
    membership: pd.DataFrame,
    q_n: np.ndarray,
    q_v: np.ndarray,
    q_c: np.ndarray,
    context: dict[str, np.ndarray],
    reservoir: np.ndarray,
    controls_per_truth: int,
    batch_size: int,
    score_gap_caps_in_user_sd: tuple[float, ...] = DEFAULT_SCORE_GAP_CAPS,
) -> tuple[pd.DataFrame, dict]:
    users = np.asarray(prepared["cache"].users, dtype=np.int64)
    truth = prepared["cache"].gt
    by_user = membership.set_index("user_idx")
    csr_ptr = prepared["data"]["csr_ptr"]
    csr_items = prepared["data"]["csr_items"]
    reservoir_tensor = torch.as_tensor(
        reservoir, dtype=torch.long, device=item_embedding.device
    )
    reservoir_embedding = item_embedding.index_select(0, reservoir_tensor)
    reservoir_bins = context["popularity_bin"][reservoir]
    reservoir_group_positions = {
        group: np.flatnonzero(reservoir_bins == group)
        for group in range(int(context["popularity_bin"].max(initial=0)) + 1)
    }
    rows = []
    truth_total = 0
    matched_truths = 0
    unmatched_no_control = 0
    for start in range(0, len(users), batch_size):
        batch_users = users[start : start + batch_size]
        tensor_users = torch.as_tensor(
            batch_users, dtype=torch.long, device=user_embedding.device
        )
        batch_user_embedding = user_embedding.index_select(0, tensor_users)
        reservoir_scores = (batch_user_embedding @ reservoir_embedding.T).cpu().numpy()
        truth_lists = [np.asarray(truth[int(user)], dtype=np.int64) for user in batch_users]
        truth_counts = np.asarray([len(items) for items in truth_lists], dtype=np.int64)
        if truth_counts.sum():
            all_truth_items = np.concatenate(truth_lists)
            local_users = np.repeat(np.arange(len(batch_users)), truth_counts)
            all_truth_scores = (
                batch_user_embedding.index_select(
                    0,
                    torch.as_tensor(
                        local_users, dtype=torch.long, device=user_embedding.device
                    ),
                )
                * item_embedding.index_select(
                    0,
                    torch.as_tensor(
                        all_truth_items, dtype=torch.long, device=item_embedding.device
                    ),
                )
            ).sum(dim=1).cpu().numpy()
        else:
            all_truth_scores = np.empty(0, dtype=np.float32)
        truth_offset = 0
        for local, user in enumerate(batch_users):
            user = int(user)
            truth_items = truth_lists[local]
            truth_total += len(truth_items)
            if not len(truth_items) or user not in by_user.index:
                truth_offset += len(truth_items)
                continue
            truth_scores = all_truth_scores[
                truth_offset : truth_offset + len(truth_items)
            ]
            truth_offset += len(truth_items)
            left, right = int(csr_ptr[user]), int(csr_ptr[user + 1])
            forbidden = np.concatenate(
                [np.asarray(csr_items[left:right], dtype=np.int64), truth_items]
            )
            reservoir_allowed = ~np.isin(reservoir, forbidden, assume_unique=False)
            score_scale = max(float(np.std(reservoir_scores[local])), 1e-12)
            group = by_user.loc[user]
            for truth_item, truth_score in zip(
                truth_items.tolist(), truth_scores.tolist(), strict=True
            ):
                popularity_group = int(context["popularity_bin"][truth_item])
                group_positions = reservoir_group_positions.get(
                    popularity_group, np.empty(0, dtype=np.int64)
                )
                control_positions = group_positions[
                    reservoir_allowed[group_positions]
                ]
                if not len(control_positions):
                    unmatched_no_control += 1
                    continue
                order = np.lexsort(
                    (
                        reservoir[control_positions],
                        np.abs(
                            reservoir_scores[local, control_positions]
                            - float(truth_score)
                        ),
                    )
                )
                control_positions = control_positions[order[:controls_per_truth]]
                controls = reservoir[control_positions]
                matched_truths += 1
                control_scores = reservoir_scores[local, control_positions]
                item_set = np.concatenate(
                    [np.asarray([truth_item], dtype=np.int64), controls]
                )
                signal = candidate_signal_values(
                    user_qn=q_n[user],
                    user_qv=q_v[user],
                    user_qc=q_c[user],
                    items=item_set,
                    buyer_qn_mean=context["buyer_qn_mean"],
                    price_percentile=context["price_percentile"],
                )
                score_gap = np.abs(control_scores - float(truth_score))
                for name in SIGNAL_ORDER:
                    difference = signal[name][0] - signal[name][1:]
                    finite = np.isfinite(difference)
                    difference = difference[finite]
                    gaps = score_gap[finite]
                    normalized_gaps = gaps / score_scale
                    if not len(difference):
                        continue
                    tolerance = 1e-12
                    wins = int(np.sum(difference > tolerance))
                    losses = int(np.sum(difference < -tolerance))
                    ties = int(len(difference) - wins - losses)
                    row = {
                            "user_idx": user,
                            "fixed_clv_segment": group["fixed_clv_segment"],
                            "signal": name,
                            "candidate_pair_count": int(len(difference)),
                            "truth_wins": wins,
                            "ties": ties,
                            "control_wins": losses,
                            "score_gap_sum": float(gaps.sum()),
                            "score_gap_max": float(gaps.max()),
                            "normalized_score_gap_sum": float(
                                normalized_gaps.sum()
                            ),
                            "normalized_score_gap_max": float(
                                normalized_gaps.max()
                            ),
                        }
                    for cap in score_gap_caps_in_user_sd:
                        columns = _stat_columns(float(cap))
                        keep = normalized_gaps <= float(cap)
                        kept_difference = difference[keep]
                        kept_gaps = gaps[keep]
                        kept_normalized_gaps = normalized_gaps[keep]
                        kept_wins = int(np.sum(kept_difference > tolerance))
                        kept_losses = int(np.sum(kept_difference < -tolerance))
                        kept_ties = int(
                            len(kept_difference) - kept_wins - kept_losses
                        )
                        row.update(
                            {
                                columns["pairs"]: int(len(kept_difference)),
                                columns["wins"]: kept_wins,
                                columns["ties"]: kept_ties,
                                columns["losses"]: kept_losses,
                                columns["gap_sum"]: float(kept_gaps.sum()),
                                columns["gap_max"]: float(
                                    kept_gaps.max(initial=0.0)
                                ),
                                columns["normalized_gap_sum"]: float(
                                    kept_normalized_gaps.sum()
                                ),
                                columns["normalized_gap_max"]: float(
                                    kept_normalized_gaps.max(initial=0.0)
                                ),
                            }
                        )
                    rows.append(row)
    return pd.DataFrame(rows), {
        "evaluation_users": int(len(users)),
        "truth_items": int(truth_total),
        "matched_truth_items": int(matched_truths),
        "unmatched_truth_items_no_control": int(unmatched_no_control),
        "matched_truth_share": float(matched_truths / max(truth_total, 1)),
    }


def summarize_pairs(
    rows: pd.DataFrame,
    *,
    score_gap_caps_in_user_sd: tuple[float, ...] = DEFAULT_SCORE_GAP_CAPS,
) -> pd.DataFrame:
    output = []
    for cap in (None, *score_gap_caps_in_user_sd):
        columns = _stat_columns(cap)
        cap_label = "all" if cap is None else f"{float(cap):g}"
        for signal in SIGNAL_ORDER:
            selected_signal = rows[rows.signal.eq(signal)]
            groups = [("overall", "전체", selected_signal)]
            for segment in fixed.SEGMENT_ORDER:
                groups.append(
                    (
                        "fixed_clv_segment",
                        segment,
                        selected_signal[
                            selected_signal.fixed_clv_segment.eq(segment)
                        ],
                    )
                )
            for group_type, group, frame in groups:
                if frame.empty:
                    continue
                pairs = int(frame[columns["pairs"]].sum())
                if pairs <= 0:
                    continue
                output.append({
                    "signal": signal,
                    "m1_score_gap_cap_in_user_sd": cap_label,
                    "group_type": group_type,
                    "group": group,
                    "n_users": int(
                        frame.loc[frame[columns["pairs"]] > 0, "user_idx"].nunique()
                    ),
                    "candidate_pair_count": pairs,
                    "truth_wins": int(frame[columns["wins"]].sum()),
                    "ties": int(frame[columns["ties"]].sum()),
                    "control_wins": int(frame[columns["losses"]].sum()),
                    "pair_balanced_win_rate": float(
                        (frame[columns["wins"]].sum()
                         + 0.5 * frame[columns["ties"]].sum()) / pairs
                    ),
                    "mean_absolute_m1_score_gap": float(
                        frame[columns["gap_sum"]].sum() / pairs
                    ),
                    "max_absolute_m1_score_gap": float(
                        frame[columns["gap_max"]].max()
                    ),
                    "mean_m1_score_gap_in_user_sd": float(
                        frame[columns["normalized_gap_sum"]].sum() / pairs
                    ),
                    "max_m1_score_gap_in_user_sd": float(
                        frame[columns["normalized_gap_max"]].max()
                    ),
                    "same_popularity_bin_share": 1.0,
                })
    return pd.DataFrame(output)


def _per_user_rates(rows: pd.DataFrame, cap: float | None = None) -> pd.DataFrame:
    columns = _stat_columns(cap)
    grouped = rows.groupby(["user_idx", "fixed_clv_segment", "signal"], sort=False)
    frame = grouped[
        [columns["pairs"], columns["wins"], columns["ties"]]
    ].sum()
    frame = frame[frame[columns["pairs"]] > 0].copy()
    frame["balanced_win_rate"] = (
        frame[columns["wins"]] + 0.5 * frame[columns["ties"]]
    ) / frame[columns["pairs"]]
    return frame.reset_index()


def _bootstrap_group(
    per_user: pd.DataFrame,
    *,
    samples: int,
    seed: int,
) -> dict:
    pivot = per_user.pivot_table(
        index="user_idx", columns="signal", values="balanced_win_rate"
    ).dropna(subset=list(SIGNAL_ORDER))
    if pivot.empty:
        return {"paired_users": 0, "signals": {}}
    values = pivot.loc[:, SIGNAL_ORDER].to_numpy(np.float64)
    rng = np.random.default_rng(seed)
    draws = np.empty((samples, len(SIGNAL_ORDER)), dtype=np.float64)
    for draw in range(samples):
        positions = rng.integers(0, len(values), size=len(values))
        draws[draw] = values[positions].mean(axis=0)
    report = {}
    for column, signal in enumerate(SIGNAL_ORDER):
        low, high = np.percentile(draws[:, column], [2.5, 97.5])
        observed = float(values[:, column].mean())
        report[signal] = {
            "observed": observed,
            "ci_low": float(low),
            "ci_high": float(high),
            "interval_above_0_5": bool(low > 0.5),
        }
    return {"paired_users": int(len(pivot)), "signals": report}


def bootstrap_signal_rates(
    rows: pd.DataFrame,
    *,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
    score_gap_caps_in_user_sd: tuple[float, ...] = DEFAULT_SCORE_GAP_CAPS,
) -> dict:
    groups = {}
    for cap_index, cap in enumerate((None, *score_gap_caps_in_user_sd)):
        cap_label = "all" if cap is None else f"{float(cap):g}"
        per_user = _per_user_rates(rows, cap)
        cap_groups = {
            "overall": _bootstrap_group(
                per_user, samples=samples, seed=seed + cap_index * 100
            )
        }
        for segment_index, segment in enumerate(fixed.SEGMENT_ORDER, start=1):
            cap_groups[segment] = _bootstrap_group(
                per_user[per_user.fixed_clv_segment.eq(segment)],
                samples=samples,
                seed=seed + cap_index * 100 + segment_index,
            )
        groups[cap_label] = cap_groups
    overall = groups["all"]["overall"]
    if not overall["signals"]:
        raise RuntimeError("bootstrap에 사용할 공통 사용자 행이 없습니다")
    all_per_user = _per_user_rates(rows, None)
    return {
        "bootstrap_samples": int(samples),
        "bootstrap_seed": int(seed),
        "paired_users": int(overall["paired_users"]),
        "segment_counts": {
            name: int(
                all_per_user.loc[
                    all_per_user.fixed_clv_segment.eq(name), "user_idx"
                ].nunique()
            )
            for name in fixed.SEGMENT_ORDER
        },
        "signals": overall["signals"],
        "groups": groups,
    }


def upper_tail_membership(
    *,
    q_n: np.ndarray,
    q_v: np.ndarray,
    q_c: np.ndarray,
    valid: np.ndarray,
    percentile_cut: float = UPPER_TAIL_CUT,
) -> dict[str, np.ndarray]:
    """Return comparable upper-tail groups from train-only percentile inputs."""

    valid = np.asarray(valid, dtype=bool)
    values = {
        "high_q_c": np.asarray(q_c, dtype=np.float64),
        "high_q_n": np.asarray(q_n, dtype=np.float64),
        "high_q_v": np.asarray(q_v, dtype=np.float64),
    }
    if not 0.0 < float(percentile_cut) < 1.0:
        raise ValueError("상위집단 percentile_cut은 0과 1 사이여야 합니다")
    if any(value.shape != valid.shape for value in values.values()):
        raise ValueError("q_C·q_N·q_V와 valid shape이 같아야 합니다")
    output = {}
    for name, value in values.items():
        output[name] = valid & np.isfinite(value) & (value >= percentile_cut)
    return output


def upper_tail_n_signal_report(
    rows: pd.DataFrame,
    membership: dict[str, np.ndarray],
    *,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
    score_gap_caps_in_user_sd: tuple[float, ...] = DEFAULT_SCORE_GAP_CAPS,
) -> pd.DataFrame:
    """Compare the N signal in fixed high-q_C, high-q_N and high-q_V users."""

    required = set(UPPER_TAIL_GROUPS)
    if set(membership) != required:
        raise ValueError(f"상위집단은 {sorted(required)}를 정확히 포함해야 합니다")
    n_rows = len(next(iter(membership.values())))
    if any(np.asarray(value).shape != (n_rows,) for value in membership.values()):
        raise ValueError("상위집단 membership shape이 서로 다릅니다")
    if samples <= 0:
        raise ValueError("bootstrap samples는 양수여야 합니다")

    output = []
    n_rows_only = rows[rows.signal.eq(N_SIGNAL)]
    for cap_index, cap in enumerate((None, *score_gap_caps_in_user_sd)):
        cap_label = "all" if cap is None else f"{float(cap):g}"
        per_user = _per_user_rates(n_rows_only, cap)
        for group_index, group in enumerate(UPPER_TAIL_GROUPS):
            members = np.asarray(membership[group], dtype=bool)
            selected = per_user[
                per_user.user_idx.map(
                    lambda user: 0 <= int(user) < len(members) and members[int(user)]
                )
            ]
            values = selected.balanced_win_rate.to_numpy(np.float64)
            if not len(values):
                continue
            rng = np.random.default_rng(seed + cap_index * 100 + group_index)
            draws = np.empty(samples, dtype=np.float64)
            for draw in range(samples):
                positions = rng.integers(0, len(values), size=len(values))
                draws[draw] = values[positions].mean()
            low, high = np.percentile(draws, [2.5, 97.5])
            output.append(
                {
                    "m1_score_gap_cap_in_user_sd": cap_label,
                    "group": group,
                    "n_users": int(len(values)),
                    "observed": float(values.mean()),
                    "ci_low": float(low),
                    "ci_high": float(high),
                    "interval_above_0_5": bool(low > 0.5),
                }
            )
    return pd.DataFrame(output)


def bootstrap_report_frame(report: dict) -> pd.DataFrame:
    rows = []
    for cap, groups in report["groups"].items():
        for group, payload in groups.items():
            for signal, values in payload["signals"].items():
                rows.append(
                    {
                        "m1_score_gap_cap_in_user_sd": cap,
                        "group": group,
                        "signal": signal,
                        "paired_users": int(payload["paired_users"]),
                        **values,
                    }
                )
    return pd.DataFrame(rows)


def _paths(cfg: IncrementalCandidateSignalConfig) -> dict[str, Path]:
    root = Path(cfg.out_dir)
    root.mkdir(parents=True, exist_ok=True)
    stem = f"clv_incremental_candidate_signal_{cfg.dataset}"
    return {
        "summary_csv": root / f"{stem}_summary.csv",
        "per_user_csv": root / f"{stem}_per_user.csv",
        "bootstrap_csv": root / f"{stem}_bootstrap.csv",
        "upper_tail_csv": root / f"{stem}_upper_tail_n_comparison.csv",
        "json": root / f"{stem}_diagnostic.json",
    }


@torch.no_grad()
def run_incremental_candidate_signal_diagnostic(cfg=None) -> dict[str, str]:
    cfg = cfg or configure_incremental_candidate_signal_diagnostic()
    preflight = preflight_summary(cfg)
    print(json.dumps(preflight, ensure_ascii=False, indent=2))
    prepared, model, checkpoint, record, axes = relation._prepare_and_load(cfg)
    axes = select_current_axes(cfg.dataset, prepared, axes)
    model.eval()
    user_embedding, item_embedding = model.propagate_pref()
    q_n, q_v, q_c, clv_valid = _q_inputs(axes)
    membership, thresholds = relation._membership(prepared, axes)
    train = prepared["data"]["train"]
    context = build_item_context(
        train,
        n_items=int(prepared["data"]["n_items"]),
        q_n=q_n,
        clv_valid=clv_valid,
        popularity_bins=cfg.popularity_bins,
    )
    reservoir, reservoir_counts = build_candidate_reservoir(
        context["popularity_bin"],
        bins=cfg.popularity_bins,
        per_bin=cfg.candidate_pool_per_bin,
        seed=cfg.candidate_pool_seed,
    )
    if not len(reservoir):
        raise RuntimeError("대조상품 후보 reservoir가 비었습니다")
    pair_rows, matching = collect_matched_pair_rows(
        user_embedding=user_embedding,
        item_embedding=item_embedding,
        prepared=prepared,
        membership=membership,
        q_n=q_n,
        q_v=q_v,
        q_c=q_c,
        context=context,
        reservoir=reservoir,
        controls_per_truth=cfg.controls_per_truth,
        batch_size=cfg.eval_batch_size,
        score_gap_caps_in_user_sd=cfg.score_gap_caps_in_user_sd,
    )
    if pair_rows.empty:
        raise RuntimeError("M1 점수·인기도 조건부 비교쌍이 없습니다")
    summary = summarize_pairs(
        pair_rows, score_gap_caps_in_user_sd=cfg.score_gap_caps_in_user_sd
    )
    per_user = _per_user_rates(pair_rows)
    bootstrap = bootstrap_signal_rates(
        pair_rows,
        samples=cfg.bootstrap_samples,
        seed=cfg.bootstrap_seed,
        score_gap_caps_in_user_sd=cfg.score_gap_caps_in_user_sd,
    )
    bootstrap_frame = bootstrap_report_frame(bootstrap)
    upper_membership = upper_tail_membership(
        q_n=q_n, q_v=q_v, q_c=q_c, valid=clv_valid
    )
    upper_tail = upper_tail_n_signal_report(
        pair_rows,
        upper_membership,
        samples=cfg.bootstrap_samples,
        seed=cfg.bootstrap_seed,
        score_gap_caps_in_user_sd=cfg.score_gap_caps_in_user_sd,
    )
    paths = _paths(cfg)
    test10._atomic_csv(paths["summary_csv"], summary)
    test10._atomic_csv(paths["per_user_csv"], per_user)
    test10._atomic_csv(paths["bootstrap_csv"], bootstrap_frame)
    test10._atomic_csv(paths["upper_tail_csv"], upper_tail)
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
            "item_context": {
                "items": int(len(context["popularity_bin"])),
                "buyer_qn_valid_share": float(
                    np.asarray(context["buyer_qn_valid"]).mean()
                ),
                "price_valid_share": float(np.asarray(context["price_valid"]).mean()),
                "reservoir_items": int(len(reservoir)),
                "reservoir_by_popularity_bin": reservoir_counts,
            },
            "matching_diagnostics": matching,
            "q_c_candidate_diagnostic": {
                "within_user_candidate_std": 0.0,
                "interpretation": (
                    "q_C alone is constant across a user's candidates; it can "
                    "condition an item interaction or a training weight but "
                    "cannot rank candidates by itself"
                ),
            },
            "summary_rows": summary.to_dict("records"),
            "bootstrap": bootstrap,
            "upper_tail_n_comparison": {
                "percentile_cut": UPPER_TAIL_CUT,
                "groups": list(UPPER_TAIL_GROUPS),
                "membership_counts_all_train_users": {
                    name: int(values.sum())
                    for name, values in upper_membership.items()
                },
                "rows": upper_tail.to_dict("records"),
                "interpretation_rule": (
                    "q_C targeting is uniquely supported only if high_q_c is "
                    "above 0.5 and descriptively stronger than high_q_n and "
                    "high_q_v under every fixed score-gap cap in both datasets"
                ),
            },
            "result_paths": {key: str(value) for key, value in paths.items()},
        },
    )
    print("\n1) M1 점수·상품 인기도 조건부 후보 신호와 점수차 민감도")
    print(summary.to_string(index=False))
    print("\n2) 전체·CLV 구간·점수차 상한별 사용자 단위 bootstrap")
    print(json.dumps(bootstrap, ensure_ascii=False, indent=2))
    print("\n3) 매칭 진단")
    print(json.dumps(matching, ensure_ascii=False, indent=2))
    print("\n4) q_C·q_N·q_V 상위 20% N 후보 구별력 비교")
    print(upper_tail.to_string(index=False))
    print("\n결과 파일:", {key: str(value) for key, value in paths.items()})
    return {key: str(value) for key, value in paths.items()}


if __name__ == "__main__":
    print(
        json.dumps(
            preflight_summary(configure_incremental_candidate_signal_diagnostic()),
            ensure_ascii=False,
            indent=2,
        )
    )
