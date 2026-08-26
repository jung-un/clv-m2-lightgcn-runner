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
