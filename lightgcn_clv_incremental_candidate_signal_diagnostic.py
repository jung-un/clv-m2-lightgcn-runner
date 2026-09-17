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


CODE_VERSION = "clv-incremental-candidate-signal-diagnostic-v1"
N_SIGNAL = "n_purchaser_activity_fit"
V_SIGNAL = "v_price_position_fit"
NV_SIGNAL = "mean_n_v_candidate_fit"
QC_LEVEL = "q_c_user_level_constant"
SIGNAL_ORDER = (N_SIGNAL, V_SIGNAL, NV_SIGNAL)
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 42


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


def configure_incremental_candidate_signal_diagnostic(
    dataset: str = "dunnhumby", **overrides
) -> IncrementalCandidateSignalConfig:
    base = relation.configure_candidate_relation_diagnostic(dataset)
    values = asdict(base) | {
        "out_dir": (
            f"{v3.default_out_dir(dataset.lower())}"
            "_clv_incremental_candidate_signal_diagnostic_v1"
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
    reservoir_position = {int(item): pos for pos, item in enumerate(reservoir)}
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
        for local, user in enumerate(batch_users):
            user = int(user)
            truth_items = np.asarray(truth[user], dtype=np.int64)
            truth_total += len(truth_items)
            if not len(truth_items) or user not in by_user.index:
                continue
            truth_tensor = torch.as_tensor(
                truth_items, dtype=torch.long, device=item_embedding.device
            )
            truth_scores = (
                batch_user_embedding[local] * item_embedding.index_select(0, truth_tensor)
            ).sum(dim=1).cpu().numpy()
            left, right = int(csr_ptr[user]), int(csr_ptr[user + 1])
            forbidden = set(map(int, csr_items[left:right]))
            forbidden.update(map(int, truth_items))
            group = by_user.loc[user]
            for truth_item, truth_score in zip(
                truth_items.tolist(), truth_scores.tolist(), strict=True
            ):
                controls = match_controls_for_truth(
                    truth_item=int(truth_item),
                    truth_score=float(truth_score),
                    candidate_items=reservoir,
                    candidate_scores=reservoir_scores[local],
                    popularity_bin=context["popularity_bin"],
                    forbidden_items=forbidden,
                    controls_per_truth=controls_per_truth,
                )
                if not len(controls):
                    unmatched_no_control += 1
                    continue
                matched_truths += 1
                control_positions = np.asarray(
                    [reservoir_position[int(item)] for item in controls], dtype=np.int64
                )
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
                    if not len(difference):
                        continue
                    tolerance = 1e-12
                    wins = int(np.sum(difference > tolerance))
                    losses = int(np.sum(difference < -tolerance))
                    ties = int(len(difference) - wins - losses)
                    rows.append(
                        {
                            "user_idx": user,
                            "fixed_clv_segment": group["fixed_clv_segment"],
                            "signal": name,
                            "candidate_pair_count": int(len(difference)),
                            "truth_wins": wins,
                            "ties": ties,
                            "control_wins": losses,
                            "score_gap_sum": float(gaps.sum()),
                            "score_gap_max": float(gaps.max()),
                        }
                    )
    return pd.DataFrame(rows), {
        "evaluation_users": int(len(users)),
        "truth_items": int(truth_total),
        "matched_truth_items": int(matched_truths),
        "unmatched_truth_items_no_control": int(unmatched_no_control),
        "matched_truth_share": float(matched_truths / max(truth_total, 1)),
    }


def summarize_pairs(rows: pd.DataFrame) -> pd.DataFrame:
    output = []
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
            pairs = int(frame.candidate_pair_count.sum())
            output.append(
                {
                    "signal": signal,
                    "group_type": group_type,
                    "group": group,
                    "n_users": int(frame.user_idx.nunique()),
                    "candidate_pair_count": pairs,
                    "truth_wins": int(frame.truth_wins.sum()),
                    "ties": int(frame.ties.sum()),
                    "control_wins": int(frame.control_wins.sum()),
                    "pair_balanced_win_rate": float(
                        (frame.truth_wins.sum() + 0.5 * frame.ties.sum()) / pairs
                    ),
                    "mean_absolute_m1_score_gap": float(
                        frame.score_gap_sum.sum() / pairs
                    ),
                    "max_absolute_m1_score_gap": float(frame.score_gap_max.max()),
                }
            )
    return pd.DataFrame(output)


def _per_user_rates(rows: pd.DataFrame) -> pd.DataFrame:
    grouped = rows.groupby(["user_idx", "fixed_clv_segment", "signal"], sort=False)
    frame = grouped[["candidate_pair_count", "truth_wins", "ties"]].sum()
    frame["balanced_win_rate"] = (
        frame.truth_wins + 0.5 * frame.ties
    ) / frame.candidate_pair_count
    return frame.reset_index()


def bootstrap_signal_rates(
    rows: pd.DataFrame,
    *,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    per_user = _per_user_rates(rows)
    pivot = per_user.pivot_table(
        index="user_idx", columns="signal", values="balanced_win_rate"
    ).dropna(subset=list(SIGNAL_ORDER))
    if pivot.empty:
        raise RuntimeError("bootstrap에 사용할 공통 사용자 행이 없습니다")
    segment = (
        per_user.drop_duplicates("user_idx")
        .set_index("user_idx")["fixed_clv_segment"]
        .reindex(pivot.index)
    )
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
    return {
        "bootstrap_samples": int(samples),
        "bootstrap_seed": int(seed),
        "paired_users": int(len(pivot)),
        "segment_counts": {
            name: int((segment == name).sum()) for name in fixed.SEGMENT_ORDER
        },
        "signals": report,
    }


def _paths(cfg: IncrementalCandidateSignalConfig) -> dict[str, Path]:
    root = Path(cfg.out_dir)
    root.mkdir(parents=True, exist_ok=True)
    stem = f"clv_incremental_candidate_signal_{cfg.dataset}"
    return {
        "summary_csv": root / f"{stem}_summary.csv",
        "per_user_csv": root / f"{stem}_per_user.csv",
        "json": root / f"{stem}_diagnostic.json",
    }


@torch.no_grad()
def run_incremental_candidate_signal_diagnostic(cfg=None) -> dict[str, str]:
    cfg = cfg or configure_incremental_candidate_signal_diagnostic()
    preflight = preflight_summary(cfg)
    print(json.dumps(preflight, ensure_ascii=False, indent=2))
    prepared, model, checkpoint, record, axes = relation._prepare_and_load(cfg)
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
    )
    if pair_rows.empty:
        raise RuntimeError("M1 점수·인기도 조건부 비교쌍이 없습니다")
    summary = summarize_pairs(pair_rows)
    per_user = _per_user_rates(pair_rows)
    bootstrap = bootstrap_signal_rates(
        pair_rows, samples=cfg.bootstrap_samples, seed=cfg.bootstrap_seed
    )
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
            "result_paths": {key: str(value) for key, value in paths.items()},
        },
    )
    print("\n1) M1 점수·상품 인기도 조건부 후보 신호")
    print(summary.to_string(index=False))
    print("\n2) 사용자 단위 bootstrap")
    print(json.dumps(bootstrap, ensure_ascii=False, indent=2))
    print("\n3) 매칭 진단")
    print(json.dumps(matching, ensure_ascii=False, indent=2))
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
