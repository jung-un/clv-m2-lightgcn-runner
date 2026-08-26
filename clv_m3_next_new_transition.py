"""Train-only CLV-weighted next-new-item transition relations for M3.

The purchase graph is not changed here.  This module constructs a separate
item-to-item relation from consecutive baskets and contains no model training.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import rankdata


REQUIRED_TRANSACTION_COLUMNS = {"u_idx", "i_idx", "t", "basket_id", "v"}


@dataclass(frozen=True)
class TransitionEvents:
    user_idx: np.ndarray
    source_item_idx: np.ndarray
    target_item_idx: np.ndarray
    contribution: np.ndarray
    eligible_pair_count_by_user: np.ndarray


@dataclass(frozen=True)
class HistoricalCLV:
    n_hat: np.ndarray
    v_hat: np.ndarray
    clv_proxy: np.ndarray
    percentile: np.ndarray
    coefficient: np.ndarray
    activity_decile: np.ndarray


@dataclass(frozen=True)
class TransitionGraphs:
    global_relation: sparse.csr_matrix
    clv_relation: sparse.csr_matrix
    shuffled_clv_relation: sparse.csr_matrix
    edge_support: sparse.csr_matrix


MODEL_GLOBAL = "transition_global"
MODEL_CLV = "transition_clv"
MODEL_SHUFFLE = "transition_clv_shuffle"


def _require_transaction_columns(transactions: pd.DataFrame) -> None:
    missing = REQUIRED_TRANSACTION_COLUMNS.difference(transactions.columns)
    if missing:
        raise ValueError(f"missing transaction columns: {sorted(missing)}")


def build_user_transition_events(
    transactions: pd.DataFrame,
    *,
    n_users: int,
) -> TransitionEvents:
    """Build user-normalized current-basket -> next-new-item contributions."""
    _require_transaction_columns(transactions)
    if n_users <= 0:
        raise ValueError("n_users must be positive")

    event_users: list[int] = []
    event_sources: list[int] = []
    event_targets: list[int] = []
    event_weights: list[float] = []
    pair_counts = np.zeros(n_users, dtype=np.int32)

    basket_items = (
        transactions[["u_idx", "t", "basket_id", "i_idx"]]
        .drop_duplicates()
        .sort_values(["u_idx", "t", "basket_id", "i_idx"], kind="mergesort")
        .groupby(["u_idx", "t", "basket_id"], sort=False)["i_idx"]
        .agg(lambda values: tuple(sorted(set(values))))
        .reset_index()
    )

    for user, user_baskets in basket_items.groupby("u_idx", sort=False):
        user = int(user)
        if not 0 <= user < n_users:
            raise ValueError(f"u_idx outside [0, n_users): {user}")
        baskets = [np.asarray(items, dtype=np.int32) for items in user_baskets["i_idx"]]
        history: set[int] = set()
        retained: list[tuple[np.ndarray, np.ndarray]] = []
        for position in range(len(baskets) - 1):
            current = np.asarray(baskets[position], dtype=np.int32)
            history.update(current.tolist())
            following = np.asarray(baskets[position + 1], dtype=np.int32)
            new_targets = np.asarray(
                [item for item in following if int(item) not in history],
                dtype=np.int32,
            )
            if current.size and new_targets.size:
                retained.append((current, new_targets))

        pair_counts[user] = len(retained)
        if not retained:
            continue
        user_scale = 1.0 / len(retained)
        for current, new_targets in retained:
            weight = user_scale / (current.size * new_targets.size)
            for source in current:
                for target in new_targets:
                    event_users.append(user)
                    event_sources.append(int(source))
                    event_targets.append(int(target))
                    event_weights.append(weight)

    return TransitionEvents(
        user_idx=np.asarray(event_users, dtype=np.int32),
        source_item_idx=np.asarray(event_sources, dtype=np.int32),
        target_item_idx=np.asarray(event_targets, dtype=np.int32),
        contribution=np.asarray(event_weights, dtype=np.float64),
        eligible_pair_count_by_user=pair_counts,
    )


def build_historical_clv(
    transactions: pd.DataFrame,
    *,
    n_users: int,
    shuffle_seed: int = 20260826,
) -> tuple[HistoricalCLV, np.ndarray]:
    """Calculate historical CLV coefficients and a within-N-decile shuffle."""
    _require_transaction_columns(transactions)
    if n_users <= 0:
        raise ValueError("n_users must be positive")

    baskets = (
        transactions.groupby(["u_idx", "basket_id"], sort=False)["v"]
        .sum()
        .rename("basket_value")
        .reset_index()
    )
    grouped = baskets.groupby("u_idx", sort=False)["basket_value"]
    n_series = grouped.size()
    v_series = grouped.mean()
    n_hat = np.zeros(n_users, dtype=np.float64)
    v_hat = np.zeros(n_users, dtype=np.float64)
    users = n_series.index.to_numpy(dtype=np.int64)
    if np.any((users < 0) | (users >= n_users)):
        raise ValueError("u_idx outside [0, n_users)")
    n_hat[users] = n_series.to_numpy(dtype=np.float64)
    v_hat[users] = v_series.reindex(n_series.index).to_numpy(dtype=np.float64)
    if np.any(n_hat == 0):
        raise ValueError("historical CLV requires at least one basket per indexed user")

    clv_proxy = n_hat * v_hat
    percentile = (rankdata(clv_proxy, method="average") - 0.5) / n_users
    raw_coefficient = 0.5 + percentile
    coefficient = raw_coefficient / raw_coefficient.mean()

    activity_percentile = (rankdata(n_hat, method="average") - 0.5) / n_users
    activity_decile = np.minimum(
        np.floor(activity_percentile * 10).astype(np.int8), 9
    )
    shuffled = coefficient.copy()
    rng = np.random.default_rng(shuffle_seed)
    for decile in np.unique(activity_decile):
        indices = np.flatnonzero(activity_decile == decile)
        shuffled[indices] = coefficient[rng.permutation(indices)]

    return (
        HistoricalCLV(
            n_hat=n_hat,
            v_hat=v_hat,
            clv_proxy=clv_proxy,
            percentile=percentile,
            coefficient=coefficient,
            activity_decile=activity_decile,
        ),
        shuffled,
    )


def _row_normalize(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    matrix = matrix.tocsr().astype(np.float64)
    row_mass = np.asarray(matrix.sum(axis=1)).ravel()
    inverse = np.divide(
        1.0,
        row_mass,
        out=np.zeros_like(row_mass, dtype=np.float64),
        where=row_mass > 0,
    )
    return (sparse.diags(inverse) @ matrix).tocsr()


def build_transition_graphs(
    events: TransitionEvents,
    *,
    clv_coefficient: np.ndarray,
    shuffled_coefficient: np.ndarray,
    n_items: int,
) -> TransitionGraphs:
    """Aggregate unweighted, CLV-weighted, and shuffled-CLV relations."""
    if n_items <= 0:
        raise ValueError("n_items must be positive")
    arrays = (
        events.user_idx,
        events.source_item_idx,
        events.target_item_idx,
        events.contribution,
    )
    if len({len(values) for values in arrays}) != 1:
        raise ValueError("transition event arrays must have equal length")
    if events.user_idx.size:
        if events.user_idx.max() >= len(clv_coefficient):
            raise ValueError("missing CLV coefficient for transition user")
        if events.source_item_idx.max() >= n_items or events.target_item_idx.max() >= n_items:
            raise ValueError("transition item outside graph shape")

    shape = (n_items, n_items)

    def aggregate(multiplier: np.ndarray) -> sparse.csr_matrix:
        values = events.contribution * multiplier[events.user_idx]
        return sparse.coo_matrix(
            (values, (events.source_item_idx, events.target_item_idx)),
            shape=shape,
        ).tocsr()

    ones = np.ones_like(clv_coefficient, dtype=np.float64)
    global_relation = _row_normalize(aggregate(ones))
    clv_relation = _row_normalize(aggregate(np.asarray(clv_coefficient, dtype=float)))
    shuffled_relation = _row_normalize(
        aggregate(np.asarray(shuffled_coefficient, dtype=float))
    )

    if events.user_idx.size:
        triples = np.unique(
            np.column_stack(
                [events.user_idx, events.source_item_idx, events.target_item_idx]
            ),
            axis=0,
        )
        support = sparse.coo_matrix(
            (
                np.ones(len(triples), dtype=np.int32),
                (triples[:, 1], triples[:, 2]),
            ),
            shape=shape,
        ).tocsr()
    else:
        support = sparse.csr_matrix(shape, dtype=np.int32)

    return TransitionGraphs(
        global_relation=global_relation,
        clv_relation=clv_relation,
        shuffled_clv_relation=shuffled_relation,
        edge_support=support,
    )


def rank_transition_candidates(
    relation: sparse.csr_matrix,
    *,
    last_basket_items: dict[int, np.ndarray],
    seen_items: dict[int, np.ndarray],
    eval_users: np.ndarray,
    top_k: int = 50,
) -> dict[int, np.ndarray]:
    """Rank positive transition candidates without any fallback."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    relation = relation.tocsr()
    rankings: dict[int, np.ndarray] = {}
    for raw_user in np.asarray(eval_users):
        user = int(raw_user)
        sources = np.unique(
            np.asarray(last_basket_items.get(user, []), dtype=np.int64)
        )
        if sources.size == 0:
            rankings[user] = np.empty(0, dtype=np.int32)
            continue
        if np.any((sources < 0) | (sources >= relation.shape[0])):
            raise ValueError(f"last-basket source outside relation: user={user}")
        scores = np.asarray(relation[sources].sum(axis=0)).ravel() / sources.size
        seen = np.asarray(seen_items.get(user, []), dtype=np.int64)
        seen = seen[(seen >= 0) & (seen < relation.shape[1])]
        scores[seen] = 0.0
        candidates = np.flatnonzero(scores > 0)
        if candidates.size:
            order = np.lexsort((candidates, -scores[candidates]))
            candidates = candidates[order[:top_k]]
        rankings[user] = candidates.astype(np.int32, copy=False)
    return rankings


def _discounted_gain(ranked: np.ndarray, truth: set[int], k: int) -> float:
    if not truth:
        return 0.0
    hits = np.fromiter(
        (int(item) in truth for item in ranked[:k]), dtype=np.float64
    )
    if hits.size == 0:
        return 0.0
    dcg = float((hits / np.log2(np.arange(2, hits.size + 2))).sum())
    ideal_size = min(len(truth), k)
    idcg = float((1.0 / np.log2(np.arange(2, ideal_size + 2))).sum())
    return dcg / idcg if idcg else 0.0


def _exposure_metrics(
    rankings: dict[int, np.ndarray], *, n_items: int, k: int
) -> dict[str, float]:
    exposure = np.zeros(n_items, dtype=np.int64)
    for ranked in rankings.values():
        selected = np.asarray(ranked[:k], dtype=np.int64)
        if selected.size:
            np.add.at(exposure, selected, 1)
    positive = exposure[exposure > 0]
    total = int(positive.sum())
    if total:
        probabilities = positive / total
        entropy = float(-(probabilities * np.log(probabilities)).sum())
        descending = np.sort(positive)[::-1]
        top10_share = float(descending[:10].sum() / total)
        top100_share = float(descending[:100].sum() / total)
    else:
        entropy = 0.0
        top10_share = 0.0
        top100_share = 0.0
    return {
        f"coverage@{k}": float(len(positive) / n_items),
        f"n_distinct@{k}": int(len(positive)),
        f"exposure_entropy@{k}": entropy,
        f"eff_catalog@{k}": float(np.exp(entropy)),
        f"top10_share@{k}": top10_share,
        f"top100_share@{k}": top100_share,
    }


def evaluate_transition_ranking(
    rankings: dict[int, np.ndarray],
    *,
    truth: dict[int, np.ndarray],
    n_items: int,
    ks: tuple[int, ...] = (10, 20, 50),
) -> tuple[dict[str, float], pd.DataFrame]:
    """Evaluate binary new-item truths and return aggregate and per-user rows."""
    if not truth:
        raise ValueError("truth must contain at least one evaluation user")
    rows: list[dict[str, float | int]] = []
    for user in sorted(truth):
        truth_set = set(np.asarray(truth[user], dtype=np.int64).tolist())
        if not truth_set:
            continue
        ranked = np.asarray(rankings.get(user, []), dtype=np.int64)
        row: dict[str, float | int] = {
            "user_idx": int(user),
            "n_truth": len(truth_set),
            "n_positive_candidates": len(ranked),
        }
        for k in ks:
            hits = sum(int(item) in truth_set for item in ranked[:k])
            row[f"recall@{k}"] = hits / len(truth_set)
            row[f"ndcg@{k}"] = _discounted_gain(ranked, truth_set, k)
        rows.append(row)
    per_user = pd.DataFrame(rows)
    if per_user.empty:
        raise ValueError("no nonempty evaluation truths")
    metrics: dict[str, float] = {
        "n_eval_users": int(len(per_user)),
        "mean_positive_candidates": float(per_user.n_positive_candidates.mean()),
    }
    for k in ks:
        metrics[f"recall@{k}"] = float(per_user[f"recall@{k}"].mean())
        metrics[f"ndcg@{k}"] = float(per_user[f"ndcg@{k}"].mean())
        metrics.update(_exposure_metrics(rankings, n_items=n_items, k=k))
    return metrics, per_user


def reachable_truth_share(
    relation: sparse.csr_matrix,
    *,
    last_basket_items: dict[int, np.ndarray],
    seen_items: dict[int, np.ndarray],
    truth: dict[int, np.ndarray],
) -> float:
    """Share of evaluation truths reachable by a positive relation edge."""
    relation = relation.tocsr()
    reached = 0
    total = 0
    for user, raw_truth in truth.items():
        truth_items = set(np.asarray(raw_truth, dtype=np.int64).tolist())
        total += len(truth_items)
        sources = np.unique(
            np.asarray(last_basket_items.get(int(user), []), dtype=np.int64)
        )
        if sources.size == 0:
            continue
        candidates = set(relation[sources].indices.tolist())
        candidates.difference_update(
            np.asarray(seen_items.get(int(user), []), dtype=np.int64).tolist()
        )
        reached += len(truth_items.intersection(candidates))
    return reached / total if total else 0.0


def decide_pilot(metric_table: pd.DataFrame) -> dict[str, object]:
    """Apply the six fixed Phase-1 feasibility conditions."""
    required_models = {MODEL_GLOBAL, MODEL_CLV, MODEL_SHUFFLE}
    if set(metric_table["model_id"]) != required_models:
        raise ValueError(f"metric table must contain exactly {sorted(required_models)}")
    rows = metric_table.set_index("model_id")
    global_row = rows.loc[MODEL_GLOBAL]
    clv_row = rows.loc[MODEL_CLV]
    shuffle_row = rows.loc[MODEL_SHUFFLE]

    accuracy_metrics = [
        f"{metric}@{k}"
        for metric in ("recall", "ndcg")
        for k in (10, 20, 50)
    ]
    accuracy_ratios = {
        metric: (
            float(clv_row[metric] / global_row[metric])
            if float(global_row[metric]) != 0
            else (1.0 if float(clv_row[metric]) == 0 else float("inf"))
        )
        for metric in accuracy_metrics
    }
    accuracy_guard = all(value >= 0.99 for value in accuracy_ratios.values())

    recall10_cmp = float(clv_row["recall@10"] - global_row["recall@10"])
    ndcg10_cmp = float(clv_row["ndcg@10"] - global_row["ndcg@10"])
    shallow_rank_improved = (
        (recall10_cmp > 0 and ndcg10_cmp > 0)
        or (recall10_cmp > 0 and np.isclose(ndcg10_cmp, 0.0))
        or (ndcg10_cmp > 0 and np.isclose(recall10_cmp, 0.0))
    )

    shuffle_recall_delta = float(
        clv_row["recall@10"] - shuffle_row["recall@10"]
    )
    shuffle_ndcg_delta = float(clv_row["ndcg@10"] - shuffle_row["ndcg@10"])
    assignment_guard = (
        shuffle_recall_delta >= 0
        and shuffle_ndcg_delta >= 0
        and (shuffle_recall_delta > 0 or shuffle_ndcg_delta > 0)
    )
    reachable_guard = float(clv_row["reachable_truth_share"]) >= 0.99 * float(
        global_row["reachable_truth_share"]
    )
    catalog_guard = float(clv_row["n_distinct@10"]) >= 0.95 * float(
        global_row["n_distinct@10"]
    )
    exposure_delta = float(
        clv_row["top10_share@10"] - global_row["top10_share@10"]
    )
    exposure_guard = exposure_delta <= 0.01

    checks = {
        "accuracy_guard": bool(accuracy_guard),
        "shallow_rank_improved": bool(shallow_rank_improved),
        "correct_assignment_guard": bool(assignment_guard),
        "reachable_truth_guard": bool(reachable_guard),
        "catalog_guard": bool(catalog_guard),
        "exposure_guard": bool(exposure_guard),
    }
    return {
        "passes_pilot": all(checks.values()),
        "checks": checks,
        "accuracy_ratios_vs_global": accuracy_ratios,
        "recall@10_delta_vs_global": recall10_cmp,
        "ndcg@10_delta_vs_global": ndcg10_cmp,
        "recall@10_delta_vs_shuffle": shuffle_recall_delta,
        "ndcg@10_delta_vs_shuffle": shuffle_ndcg_delta,
        "reachable_truth_ratio_vs_global": float(
            clv_row["reachable_truth_share"]
            / max(float(global_row["reachable_truth_share"]), np.finfo(float).eps)
        ),
        "catalog_ratio_vs_global": float(
            clv_row["n_distinct@10"]
            / max(float(global_row["n_distinct@10"]), np.finfo(float).eps)
        ),
        "top10_share_absolute_delta": exposure_delta,
        "interpretation": (
            "exploratory train-only feasibility diagnostic; no significance, "
            "generalization, or final-test claim"
        ),
    }
