"""Train-only CLV-conditioned candidate-item relations for M3.

The binary M1 user-item graph remains unchanged.  This module adds a second
user-to-item relation whose rows describe *new-to-user* candidate items.  A
customer's historical CLV percentile changes both the next-category
distribution and the item distribution inside that category.  All relation
statistics are cross-fitted by user.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from clv_m3_clv_conditioned_category_transition_graph import (
    ACTIVE_ARMS,
    ARM_ACTUAL,
    ARM_GENERAL,
    ARM_SHUFFLE,
    DEFAULT_CROSS_FIT_FOLDS,
    DEFAULT_KAPPA,
    DEFAULT_MIN_SUPPORT_USERS,
    DEFAULT_SHUFFLE_DEGREE_BINS,
    DEFAULT_SHUFFLE_SEED,
    _basket_order,
    _degree_stratified_shuffle,
    _historical_clv,
    _midrank_percentile,
    _probabilities,
    _safe_spearman,
    _stats,
    _transition_evidence,
)


DEFAULT_ITEM_KAPPA = 20.0
DEFAULT_ITEM_MIN_SUPPORT_USERS = 5
DEFAULT_MAX_TARGET_CATEGORIES = 20
DEFAULT_MAX_CANDIDATE_ITEMS = 100
DEFAULT_BASE_CANDIDATE_ITEMS = 100
DEFAULT_SUPPLEMENTAL_CANDIDATE_ITEMS = 20
RELATION_MODE_POSITIVE_EXCESS = "positive_excess_own_support"
RELATION_MODE_COMMON_SUPPORT = "pooled_common_support_conditional_weights"
RELATION_MODE_SUPPLEMENTAL = "pooled_base_plus_clv_positive_excess"


@dataclass(frozen=True)
class CLVCandidateItemGraph:
    n_hat: np.ndarray
    v_hat: np.ndarray
    clv_proxy: np.ndarray
    clv_percentile: np.ndarray
    clv_shuffle_percentile: np.ndarray
    clv_shuffle_stratum: np.ndarray
    user_item_operators: dict[str, torch.Tensor]
    diagnostics: dict
    candidate_blocks: dict[str, torch.Tensor] | None = None


def _item_category_mapping(
    train: pd.DataFrame,
    n_items: int,
    n_cat: int,
) -> np.ndarray:
    mapping = train.groupby("i_idx", sort=True)["cat_idx"].agg(
        lambda values: values.iloc[0] if values.nunique() == 1 else -1
    )
    mapping = mapping.reindex(np.arange(n_items))
    if mapping.isna().any() or (mapping < 0).any():
        raise ValueError("every train item must map to exactly one category")
    result = mapping.to_numpy(np.int64)
    if result.min(initial=0) < 0 or result.max(initial=-1) >= n_cat:
        raise ValueError("item category index is outside n_cat")
    return result


def _first_acquisition_item_evidence(
    train: pd.DataFrame,
    n_users: int,
) -> tuple[pd.DataFrame, dict]:
    baskets = _basket_order(train)
    basket_column = "b_raw" if "b_raw" in train else "t"
    lines = (
        train[["u_idx", basket_column, "i_idx", "cat_idx"]]
        .drop_duplicates()
        .rename(columns={basket_column: "_basket_id"})
        .merge(
            baskets[["u_idx", "_basket_id", "basket_order"]],
            on=["u_idx", "_basket_id"],
            how="left",
            validate="many_to_one",
        )
    )
    if lines["basket_order"].isna().any():
        raise RuntimeError("basket ordering failed for first-acquisition evidence")
    lines["basket_order"] = lines["basket_order"].astype(np.int64)
    first_order = lines.groupby(["u_idx", "i_idx"], sort=False)[
        "basket_order"
    ].transform("min")
    first = lines.loc[
        lines["basket_order"].eq(first_order) & lines["basket_order"].gt(0),
        ["u_idx", "i_idx", "cat_idx"],
    ].drop_duplicates()
    first = first.rename(columns={"cat_idx": "d_idx"})
    group_size = first.groupby(["u_idx", "d_idx"], sort=False)[
        "i_idx"
    ].transform("size")
    first["mass"] = 1.0 / group_size.to_numpy(np.float64)
    user_category_mass = first.groupby(
        ["u_idx", "d_idx"], sort=False
    )["mass"].sum()
    if len(user_category_mass) and not np.allclose(user_category_mass, 1.0):
        raise RuntimeError("each user-category acquisition row must have unit mass")
    return first, {
        "n_users_with_first_acquisitions": int(first["u_idx"].nunique()),
        "n_first_acquisition_user_item_rows": int(len(first)),
        "n_supported_train_users": int(n_users),
        "max_user_category_mass_error": float(
            np.abs(user_category_mass.to_numpy(np.float64) - 1.0).max(
                initial=0.0
            )
        ),
    }


def _item_probabilities(
    evidence: pd.DataFrame,
    q_values: np.ndarray,
    item_category: np.ndarray,
    n_items: int,
    n_cat: int,
    *,
    min_support_users: int,
    kappa: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    support = (
        evidence.groupby("i_idx", sort=True)["u_idx"]
        .nunique()
        .rename("n_users")
    )
    supported_items = support.index[
        support.to_numpy(np.int64) >= min_support_users
    ].to_numpy(np.int64)
    supported_mask = np.zeros(n_items, dtype=bool)
    supported_mask[supported_items] = True
    kept = evidence.loc[
        supported_mask[evidence["i_idx"].to_numpy(np.int64)]
    ]

    users = kept["u_idx"].to_numpy(np.int64)
    items = kept["i_idx"].to_numpy(np.int64)
    mass = kept["mass"].to_numpy(np.float64)
    x0 = np.zeros(n_items, dtype=np.float64)
    xh = np.zeros(n_items, dtype=np.float64)
    xl = np.zeros(n_items, dtype=np.float64)
    np.add.at(x0, items, mass)
    np.add.at(xh, items, mass * q_values[users])
    np.add.at(xl, items, mass * (1.0 - q_values[users]))

    n0 = np.bincount(item_category, weights=x0, minlength=n_cat)
    nh = np.bincount(item_category, weights=xh, minlength=n_cat)
    nl = np.bincount(item_category, weights=xl, minlength=n_cat)
    p0 = np.divide(
        x0,
        n0[item_category],
        out=np.zeros_like(x0),
        where=n0[item_category] > 0,
    )
    ph = np.divide(
        xh + kappa * p0,
        nh[item_category] + kappa,
        out=np.zeros_like(xh),
        where=(nh[item_category] + kappa) > 0,
    )
    pl = np.divide(
        xl + kappa * p0,
        nl[item_category] + kappa,
        out=np.zeros_like(xl),
        where=(nl[item_category] + kappa) > 0,
    )
    for probability in (p0, pl, ph):
        if not np.isfinite(probability).all() or np.any(probability < 0):
            raise RuntimeError("candidate-item probability is invalid")
    return p0, pl, ph, {
        "raw_items": int(len(support)),
        "supported_items": int(len(supported_items)),
        "support_users": _stats(support.to_numpy(np.float64)),
        "categories_with_supported_items": int(np.sum(n0 > 0)),
    }


def _top_indices(values: np.ndarray, limit: int) -> np.ndarray:
    positive = np.flatnonzero(values > 1e-12)
    if len(positive) <= limit:
        return positive
    candidate = positive[np.argpartition(-values[positive], limit - 1)[:limit]]
    order = np.lexsort((candidate, -values[candidate]))
    return candidate[order]


def _candidate_rows(
    users: np.ndarray,
    train: pd.DataFrame,
    recent: pd.DataFrame,
    transition_reference: pd.DataFrame,
    item_reference: pd.DataFrame,
    q_relation: np.ndarray,
    q_user: np.ndarray,
    item_category: np.ndarray,
    n_items: int,
    n_cat: int,
    *,
    arm: str,
    category_min_support_users: int,
    category_kappa: float,
    item_min_support_users: int,
    item_kappa: float,
    max_target_categories: int,
    max_candidate_items: int,
) -> tuple[list[int], list[int], list[float], dict]:
    p0_category, pl_category, ph_category, category_diagnostics = _probabilities(
        transition_reference,
        q_relation,
        n_cat,
        min_support_users=category_min_support_users,
        kappa=category_kappa,
    )
    p0_item, pl_item, ph_item, item_diagnostics = _item_probabilities(
        item_reference,
        q_relation,
        item_category,
        n_items,
        n_cat,
        min_support_users=item_min_support_users,
        kappa=item_kappa,
    )
    items_by_category = {
        category: np.flatnonzero(
            (item_category == category) & (p0_item > 0)
        )
        for category in range(n_cat)
    }
    recent_by_user = {
        int(user): group[["c_idx", "recent_share"]].to_numpy(np.float64)
        for user, group in recent.loc[recent["u_idx"].isin(users)].groupby(
            "u_idx", sort=False
        )
    }
    seen_by_user = {
        int(user): set(group.to_numpy(np.int64).tolist())
        for user, group in train.loc[train["u_idx"].isin(users)].groupby(
            "u_idx", sort=False
        )["i_idx"]
    }

    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    zero_rows = 0
    for user in users:
        profile = recent_by_user.get(int(user))
        if profile is None:
            zero_rows += 1
            continue
        pooled_category = np.zeros(n_cat, dtype=np.float64)
        conditional_category = np.zeros(n_cat, dtype=np.float64)
        for category_value, share in profile:
            category = int(category_value)
            pooled_category += float(share) * p0_category[category]
            if arm == ARM_GENERAL:
                conditional_category += float(share) * p0_category[category]
            else:
                conditional_category += float(share) * (
                    (1.0 - q_user[user]) * pl_category[category]
                    + q_user[user] * ph_category[category]
                )
        target_categories = _top_indices(
            conditional_category,
            max_target_categories,
        )
        candidate_items: list[np.ndarray] = []
        candidate_scores: list[np.ndarray] = []
        for target in target_categories:
            item_ids = items_by_category[int(target)]
            if not len(item_ids):
                continue
            if arm == ARM_GENERAL:
                score = conditional_category[target] * p0_item[item_ids]
            else:
                conditional_item = (
                    (1.0 - q_user[user]) * pl_item[item_ids]
                    + q_user[user] * ph_item[item_ids]
                )
                conditional = conditional_category[target] * conditional_item
                pooled = pooled_category[target] * p0_item[item_ids]
                score = np.maximum(conditional - pooled, 0.0)
            candidate_items.append(item_ids)
            candidate_scores.append(score)
        if not candidate_items:
            zero_rows += 1
            continue
        item_ids = np.concatenate(candidate_items)
        score = np.concatenate(candidate_scores)
        unseen = np.fromiter(
            (int(item) not in seen_by_user.get(int(user), set()) for item in item_ids),
            dtype=bool,
            count=len(item_ids),
        )
        item_ids, score = item_ids[unseen], score[unseen]
        selected_local = _top_indices(score, max_candidate_items)
        if not len(selected_local):
            zero_rows += 1
            continue
        selected_items = item_ids[selected_local]
        selected_score = score[selected_local]
        selected_score = selected_score / selected_score.sum()
        rows.extend([int(user)] * len(selected_items))
        cols.extend(selected_items.tolist())
        values.extend(selected_score.tolist())
    return rows, cols, values, {
        "users": int(len(users)),
        "zero_relation_users": int(zero_rows),
        "category_probability": category_diagnostics,
        "candidate_item_probability": item_diagnostics,
    }


def _build_operator(
    train: pd.DataFrame,
    transition_evidence: pd.DataFrame,
    item_evidence: pd.DataFrame,
    recent: pd.DataFrame,
    q_values: np.ndarray,
    item_category: np.ndarray,
    n_users: int,
    n_items: int,
    n_cat: int,
    *,
    arm: str,
    category_min_support_users: int,
    category_kappa: float,
    item_min_support_users: int,
    item_kappa: float,
    cross_fit_folds: int,
    max_target_categories: int,
    max_candidate_items: int,
) -> tuple[torch.Tensor, dict]:
    all_rows: list[int] = []
    all_cols: list[int] = []
    all_values: list[float] = []
    folds = np.arange(n_users, dtype=np.int64) % cross_fit_folds
    fold_diagnostics = []
    for fold in range(cross_fit_folds):
        consumers = np.flatnonzero(folds == fold)
        transition_reference = transition_evidence.loc[
            ~transition_evidence["u_idx"].isin(consumers)
        ]
        item_reference = item_evidence.loc[
            ~item_evidence["u_idx"].isin(consumers)
        ]
        rows, cols, values, diagnostics = _candidate_rows(
            consumers,
            train,
            recent,
            transition_reference,
            item_reference,
            q_values,
            q_values,
            item_category,
            n_items,
            n_cat,
            arm=arm,
            category_min_support_users=category_min_support_users,
            category_kappa=category_kappa,
            item_min_support_users=item_min_support_users,
            item_kappa=item_kappa,
            max_target_categories=max_target_categories,
            max_candidate_items=max_candidate_items,
        )
        all_rows.extend(rows)
        all_cols.extend(cols)
        all_values.extend(values)
        fold_diagnostics.append({"fold": fold, **diagnostics})
    row_array = np.asarray(all_rows, dtype=np.int64)
    col_array = np.asarray(all_cols, dtype=np.int64)
    value_array = np.asarray(all_values, dtype=np.float64)
    with torch.sparse.check_sparse_tensor_invariants():
        operator = torch.sparse_coo_tensor(
            torch.from_numpy(np.stack([row_array, col_array])).long()
            if len(row_array)
            else torch.empty((2, 0), dtype=torch.long),
            torch.from_numpy(value_array.astype(np.float32)),
            size=(n_users, n_items),
        ).coalesce()
    row_mass = np.bincount(row_array, weights=value_array, minlength=n_users)
    active = row_mass > 0
    return operator, {
        "n_edges": int(operator._nnz()),
        "n_active_users": int(active.sum()),
        "n_zero_relation_users": int((~active).sum()),
        "active_user_share": float(active.mean()),
        "mean_edges_per_active_user": float(
            operator._nnz() / max(int(active.sum()), 1)
        ),
        "max_active_row_mass_error": float(
            np.abs(row_mass[active] - 1.0).max(initial=0.0)
        ),
        "folds": fold_diagnostics,
    }


def _normalize_or_fallback(
    values: np.ndarray,
    fallback: np.ndarray,
) -> tuple[np.ndarray, bool]:
    total = float(values.sum())
    if np.isfinite(total) and total > 0:
        return values / total, False
    fallback_total = float(fallback.sum())
    if not np.isfinite(fallback_total) or fallback_total <= 0:
        raise RuntimeError("common candidate support must have positive pooled mass")
    return fallback / fallback_total, True


def _common_support_rows(
    users: np.ndarray,
    train: pd.DataFrame,
    recent: pd.DataFrame,
    transition_reference: pd.DataFrame,
    item_reference: pd.DataFrame,
    q_actual: np.ndarray,
    q_shuffle: np.ndarray,
    item_category: np.ndarray,
    n_items: int,
    n_cat: int,
    *,
    category_min_support_users: int,
    category_kappa: float,
    item_min_support_users: int,
    item_kappa: float,
    max_target_categories: int,
    max_candidate_items: int,
) -> tuple[dict[str, tuple[list[int], list[int], list[float]]], dict]:
    (
        p0_category,
        pl_category_actual,
        ph_category_actual,
        category_diagnostics_actual,
    ) = _probabilities(
        transition_reference,
        q_actual,
        n_cat,
        min_support_users=category_min_support_users,
        kappa=category_kappa,
    )
    (
        p0_category_shuffle,
        pl_category_shuffle,
        ph_category_shuffle,
        category_diagnostics_shuffle,
    ) = _probabilities(
        transition_reference,
        q_shuffle,
        n_cat,
        min_support_users=category_min_support_users,
        kappa=category_kappa,
    )
    (
        p0_item,
        pl_item_actual,
        ph_item_actual,
        item_diagnostics_actual,
    ) = _item_probabilities(
        item_reference,
        q_actual,
        item_category,
        n_items,
        n_cat,
        min_support_users=item_min_support_users,
        kappa=item_kappa,
    )
    (
        p0_item_shuffle,
        pl_item_shuffle,
        ph_item_shuffle,
        item_diagnostics_shuffle,
    ) = _item_probabilities(
        item_reference,
        q_shuffle,
        item_category,
        n_items,
        n_cat,
        min_support_users=item_min_support_users,
        kappa=item_kappa,
    )
    if not np.allclose(p0_category, p0_category_shuffle):
        raise RuntimeError("pooled category probabilities must be arm invariant")
    if not np.allclose(p0_item, p0_item_shuffle):
        raise RuntimeError("pooled item probabilities must be arm invariant")

    items_by_category = {
        category: np.flatnonzero(
            (item_category == category) & (p0_item > 0)
        )
        for category in range(n_cat)
    }
    recent_by_user = {
        int(user): group[["c_idx", "recent_share"]].to_numpy(np.float64)
        for user, group in recent.loc[recent["u_idx"].isin(users)].groupby(
            "u_idx", sort=False
        )
    }
    seen_by_user = {
        int(user): set(group.to_numpy(np.int64).tolist())
        for user, group in train.loc[train["u_idx"].isin(users)].groupby(
            "u_idx", sort=False
        )["i_idx"]
    }
    arm_rows = {
        arm: ([], [], [])
        for arm in (ARM_GENERAL, ARM_ACTUAL, ARM_SHUFFLE)
    }
    zero_rows = 0
    fallback_counts = {ARM_ACTUAL: 0, ARM_SHUFFLE: 0}
    for user in users:
        profile = recent_by_user.get(int(user))
        if profile is None:
            zero_rows += 1
            continue

        pooled_category = np.zeros(n_cat, dtype=np.float64)
        actual_category = np.zeros(n_cat, dtype=np.float64)
        shuffle_category = np.zeros(n_cat, dtype=np.float64)
        for category_value, share in profile:
            category = int(category_value)
            pooled_category += float(share) * p0_category[category]
            actual_category += float(share) * (
                (1.0 - q_actual[user]) * pl_category_actual[category]
                + q_actual[user] * ph_category_actual[category]
            )
            shuffle_category += float(share) * (
                (1.0 - q_shuffle[user]) * pl_category_shuffle[category]
                + q_shuffle[user] * ph_category_shuffle[category]
            )

        target_categories = _top_indices(
            pooled_category,
            max_target_categories,
        )
        candidate_items: list[np.ndarray] = []
        pooled_scores: list[np.ndarray] = []
        for target in target_categories:
            item_ids = items_by_category[int(target)]
            if not len(item_ids):
                continue
            candidate_items.append(item_ids)
            pooled_scores.append(
                pooled_category[target] * p0_item[item_ids]
            )
        if not candidate_items:
            zero_rows += 1
            continue

        item_ids = np.concatenate(candidate_items)
        pooled = np.concatenate(pooled_scores)
        seen = seen_by_user.get(int(user), set())
        unseen = np.fromiter(
            (int(item) not in seen for item in item_ids),
            dtype=bool,
            count=len(item_ids),
        )
        item_ids, pooled = item_ids[unseen], pooled[unseen]
        selected_local = _top_indices(pooled, max_candidate_items)
        if not len(selected_local):
            zero_rows += 1
            continue
        selected_items = item_ids[selected_local]
        pooled = pooled[selected_local]
        selected_categories = item_category[selected_items]

        actual_item = (
            (1.0 - q_actual[user]) * pl_item_actual[selected_items]
            + q_actual[user] * ph_item_actual[selected_items]
        )
        shuffle_item = (
            (1.0 - q_shuffle[user]) * pl_item_shuffle[selected_items]
            + q_shuffle[user] * ph_item_shuffle[selected_items]
        )
        raw_weights = {
            ARM_GENERAL: pooled,
            ARM_ACTUAL: actual_category[selected_categories] * actual_item,
            ARM_SHUFFLE: (
                shuffle_category[selected_categories] * shuffle_item
            ),
        }
        normalized = {ARM_GENERAL: pooled / pooled.sum()}
        for arm in (ARM_ACTUAL, ARM_SHUFFLE):
            normalized[arm], used_fallback = _normalize_or_fallback(
                raw_weights[arm], pooled
            )
            fallback_counts[arm] += int(used_fallback)

        for arm, weights in normalized.items():
            rows, cols, values = arm_rows[arm]
            rows.extend([int(user)] * len(selected_items))
            cols.extend(selected_items.tolist())
            values.extend(weights.tolist())

    return arm_rows, {
        "users": int(len(users)),
        "zero_relation_users": int(zero_rows),
        "conditional_mass_fallback_users": fallback_counts,
        "category_probability_actual": category_diagnostics_actual,
        "category_probability_shuffle": category_diagnostics_shuffle,
        "candidate_item_probability_actual": item_diagnostics_actual,
        "candidate_item_probability_shuffle": item_diagnostics_shuffle,
    }


def _build_common_support_operators(
    train: pd.DataFrame,
    transition_evidence: pd.DataFrame,
    item_evidence: pd.DataFrame,
    recent: pd.DataFrame,
    q_actual: np.ndarray,
    q_shuffle: np.ndarray,
    item_category: np.ndarray,
    n_users: int,
    n_items: int,
    n_cat: int,
    *,
    category_min_support_users: int,
    category_kappa: float,
    item_min_support_users: int,
    item_kappa: float,
    cross_fit_folds: int,
    max_target_categories: int,
    max_candidate_items: int,
) -> tuple[dict[str, torch.Tensor], dict[str, dict], dict]:
    collected = {
        arm: ([], [], [])
        for arm in (ARM_GENERAL, ARM_ACTUAL, ARM_SHUFFLE)
    }
    folds = np.arange(n_users, dtype=np.int64) % cross_fit_folds
    fold_diagnostics = []
    for fold in range(cross_fit_folds):
        consumers = np.flatnonzero(folds == fold)
        transition_reference = transition_evidence.loc[
            ~transition_evidence["u_idx"].isin(consumers)
        ]
        item_reference = item_evidence.loc[
            ~item_evidence["u_idx"].isin(consumers)
        ]
        fold_rows, diagnostics = _common_support_rows(
            consumers,
            train,
            recent,
            transition_reference,
            item_reference,
            q_actual,
            q_shuffle,
            item_category,
            n_items,
            n_cat,
            category_min_support_users=category_min_support_users,
            category_kappa=category_kappa,
            item_min_support_users=item_min_support_users,
            item_kappa=item_kappa,
            max_target_categories=max_target_categories,
            max_candidate_items=max_candidate_items,
        )
        for arm in collected:
            for target, values in zip(collected[arm], fold_rows[arm], strict=True):
                target.extend(values)
        fold_diagnostics.append({"fold": fold, **diagnostics})

    operators = {}
    arm_diagnostics = {}
    reference_indices = None
    for arm, (rows, cols, values) in collected.items():
        row_array = np.asarray(rows, dtype=np.int64)
        col_array = np.asarray(cols, dtype=np.int64)
        value_array = np.asarray(values, dtype=np.float64)
        with torch.sparse.check_sparse_tensor_invariants():
            operator = torch.sparse_coo_tensor(
                torch.from_numpy(np.stack([row_array, col_array])).long()
                if len(row_array)
                else torch.empty((2, 0), dtype=torch.long),
                torch.from_numpy(value_array.astype(np.float32)),
                size=(n_users, n_items),
            ).coalesce()
        if reference_indices is None:
            reference_indices = operator.indices()
        elif not torch.equal(reference_indices, operator.indices()):
            raise RuntimeError("all common-support arms must share exact edges")
        row_mass = np.bincount(
            row_array, weights=value_array, minlength=n_users
        )
        active = row_mass > 0
        operators[arm] = operator
        arm_diagnostics[arm] = {
            "n_edges": int(operator._nnz()),
            "n_active_users": int(active.sum()),
            "n_zero_relation_users": int((~active).sum()),
            "active_user_share": float(active.mean()),
            "mean_edges_per_active_user": float(
                operator._nnz() / max(int(active.sum()), 1)
            ),
            "max_active_row_mass_error": float(
                np.abs(row_mass[active] - 1.0).max(initial=0.0)
            ),
            "folds": fold_diagnostics,
        }
    return (
        operators,
        arm_diagnostics,
        {
            "exact_common_edge_support": True,
            "common_edge_count": int(next(iter(operators.values()))._nnz()),
            "folds": fold_diagnostics,
        },
    )


def _ranked_positive_local_indices(
    item_ids: np.ndarray,
    scores: np.ndarray,
    limit: int,
    *,
    excluded_items: set[int] | None = None,
) -> np.ndarray:
    """Return deterministic score-descending local indices."""
    if item_ids.shape != scores.shape:
        raise ValueError("candidate item ids and scores must be aligned")
    eligible = np.isfinite(scores) & (scores > 1e-12)
    if excluded_items:
        eligible &= np.fromiter(
            (int(item) not in excluded_items for item in item_ids),
            dtype=bool,
            count=len(item_ids),
        )
    local = np.flatnonzero(eligible)
    if not len(local):
        return local
    order = np.lexsort((item_ids[local], -scores[local]))
    return local[order[:limit]]


def _supplemental_rows(
    users: np.ndarray,
    train: pd.DataFrame,
    recent: pd.DataFrame,
    transition_reference: pd.DataFrame,
    item_reference: pd.DataFrame,
    q_actual: np.ndarray,
    q_shuffle: np.ndarray,
    item_category: np.ndarray,
    n_items: int,
    n_cat: int,
    *,
    category_min_support_users: int,
    category_kappa: float,
    item_min_support_users: int,
    item_kappa: float,
    max_target_categories: int,
    base_candidate_items: int,
    supplemental_candidate_items: int,
) -> tuple[
    dict[str, tuple[list[int], list[int], list[float]]],
    dict[str, tuple[list[int], list[int], list[float]]],
    dict,
]:
    (
        p0_category,
        pl_category_actual,
        ph_category_actual,
        category_diagnostics_actual,
    ) = _probabilities(
        transition_reference,
        q_actual,
        n_cat,
        min_support_users=category_min_support_users,
        kappa=category_kappa,
    )
    (
        p0_category_shuffle,
        pl_category_shuffle,
        ph_category_shuffle,
        category_diagnostics_shuffle,
    ) = _probabilities(
        transition_reference,
        q_shuffle,
        n_cat,
        min_support_users=category_min_support_users,
        kappa=category_kappa,
    )
    (
        p0_item,
        pl_item_actual,
        ph_item_actual,
        item_diagnostics_actual,
    ) = _item_probabilities(
        item_reference,
        q_actual,
        item_category,
        n_items,
        n_cat,
        min_support_users=item_min_support_users,
        kappa=item_kappa,
    )
    (
        p0_item_shuffle,
        pl_item_shuffle,
        ph_item_shuffle,
        item_diagnostics_shuffle,
    ) = _item_probabilities(
        item_reference,
        q_shuffle,
        item_category,
        n_items,
        n_cat,
        min_support_users=item_min_support_users,
        kappa=item_kappa,
    )
    if not np.allclose(p0_category, p0_category_shuffle):
        raise RuntimeError("pooled category probabilities must be arm invariant")
    if not np.allclose(p0_item, p0_item_shuffle):
        raise RuntimeError("pooled item probabilities must be arm invariant")

    items_by_category = {
        category: np.flatnonzero(
            (item_category == category) & (p0_item > 0)
        )
        for category in range(n_cat)
    }
    recent_by_user = {
        int(user): group[["c_idx", "recent_share"]].to_numpy(np.float64)
        for user, group in recent.loc[recent["u_idx"].isin(users)].groupby(
            "u_idx", sort=False
        )
    }
    seen_by_user = {
        int(user): set(group.to_numpy(np.int64).tolist())
        for user, group in train.loc[train["u_idx"].isin(users)].groupby(
            "u_idx", sort=False
        )["i_idx"]
    }

    arm_rows = {
        arm: ([], [], [])
        for arm in (ARM_GENERAL, ARM_ACTUAL, ARM_SHUFFLE)
    }
    block_rows = {
        "base": ([], [], []),
        **{
            arm: ([], [], [])
            for arm in (ARM_GENERAL, ARM_ACTUAL, ARM_SHUFFLE)
        },
    }
    base_mass = base_candidate_items / (
        base_candidate_items + supplemental_candidate_items
    )
    extra_mass = 1.0 - base_mass
    extra_weight = extra_mass / supplemental_candidate_items
    insufficient: list[dict[str, int]] = []
    jaccards: list[float] = []
    identical_extra_rows = 0
    base_extra_overlap = 0
    train_pair_edges = 0
    base_mass_errors: list[float] = []
    extra_mass_errors: list[float] = []

    for user in users:
        profile = recent_by_user.get(int(user))
        if profile is None:
            insufficient.append(
                {"user": int(user), "general": 0, "actual": 0, "shuffle": 0}
            )
            continue

        pooled_category = np.zeros(n_cat, dtype=np.float64)
        actual_category = np.zeros(n_cat, dtype=np.float64)
        shuffle_category = np.zeros(n_cat, dtype=np.float64)
        for category_value, share in profile:
            category = int(category_value)
            pooled_category += float(share) * p0_category[category]
            actual_category += float(share) * (
                (1.0 - q_actual[user]) * pl_category_actual[category]
                + q_actual[user] * ph_category_actual[category]
            )
            shuffle_category += float(share) * (
                (1.0 - q_shuffle[user]) * pl_category_shuffle[category]
                + q_shuffle[user] * ph_category_shuffle[category]
            )

        category_sets = {
            ARM_GENERAL: _top_indices(
                pooled_category, max_target_categories
            ),
            ARM_ACTUAL: _top_indices(
                actual_category, max_target_categories
            ),
            ARM_SHUFFLE: _top_indices(
                shuffle_category, max_target_categories
            ),
        }
        seen = seen_by_user.get(int(user), set())

        def scored_items(
            categories: np.ndarray,
            category_probability: np.ndarray,
            low_item: np.ndarray,
            high_item: np.ndarray,
            q_value: float,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            item_chunks: list[np.ndarray] = []
            general_chunks: list[np.ndarray] = []
            conditional_chunks: list[np.ndarray] = []
            for target in categories:
                item_ids = items_by_category[int(target)]
                if not len(item_ids):
                    continue
                unseen = np.fromiter(
                    (int(item) not in seen for item in item_ids),
                    dtype=bool,
                    count=len(item_ids),
                )
                item_ids = item_ids[unseen]
                if not len(item_ids):
                    continue
                item_chunks.append(item_ids)
                general_chunks.append(
                    pooled_category[int(target)] * p0_item[item_ids]
                )
                conditional_item = (
                    (1.0 - q_value) * low_item[item_ids]
                    + q_value * high_item[item_ids]
                )
                conditional_chunks.append(
                    category_probability[int(target)] * conditional_item
                )
            if not item_chunks:
                empty_items = np.empty(0, dtype=np.int64)
                empty_scores = np.empty(0, dtype=np.float64)
                return empty_items, empty_scores, empty_scores
            return (
                np.concatenate(item_chunks),
                np.concatenate(general_chunks),
                np.concatenate(conditional_chunks),
            )

        general_items, general_scores, _ = scored_items(
            category_sets[ARM_GENERAL],
            pooled_category,
            p0_item,
            p0_item,
            0.0,
        )
        needed_general = base_candidate_items + supplemental_candidate_items
        general_order = _ranked_positive_local_indices(
            general_items, general_scores, needed_general
        )
        base_local = general_order[:base_candidate_items]
        general_extra_local = general_order[
            base_candidate_items:needed_general
        ]
        base_items = general_items[base_local]
        base_set = set(base_items.tolist())

        actual_items, actual_general, actual_conditional = scored_items(
            category_sets[ARM_ACTUAL],
            actual_category,
            pl_item_actual,
            ph_item_actual,
            float(q_actual[user]),
        )
        actual_excess = np.maximum(
            actual_conditional - actual_general, 0.0
        )
        actual_local = _ranked_positive_local_indices(
            actual_items,
            actual_excess,
            supplemental_candidate_items,
            excluded_items=base_set,
        )

        shuffle_items, shuffle_general, shuffle_conditional = scored_items(
            category_sets[ARM_SHUFFLE],
            shuffle_category,
            pl_item_shuffle,
            ph_item_shuffle,
            float(q_shuffle[user]),
        )
        shuffle_excess = np.maximum(
            shuffle_conditional - shuffle_general, 0.0
        )
        shuffle_local = _ranked_positive_local_indices(
            shuffle_items,
            shuffle_excess,
            supplemental_candidate_items,
            excluded_items=base_set,
        )

        counts = {
            "user": int(user),
            "general": int(len(general_extra_local)),
            "actual": int(len(actual_local)),
            "shuffle": int(len(shuffle_local)),
        }
        if (
            len(base_local) != base_candidate_items
            or any(
                counts[arm] != supplemental_candidate_items
                for arm in ("general", "actual", "shuffle")
            )
        ):
            insufficient.append(counts)
            continue

        base_scores = general_scores[base_local]
        base_weights = base_mass * base_scores / base_scores.sum()
        extras = {
            ARM_GENERAL: general_items[general_extra_local],
            ARM_ACTUAL: actual_items[actual_local],
            ARM_SHUFFLE: shuffle_items[shuffle_local],
        }
        actual_set = set(extras[ARM_ACTUAL].tolist())
        shuffle_set = set(extras[ARM_SHUFFLE].tolist())
        union = actual_set | shuffle_set
        jaccards.append(
            len(actual_set & shuffle_set) / len(union) if union else 1.0
        )
        identical_extra_rows += int(actual_set == shuffle_set)

        base_rows, base_cols, base_values = block_rows["base"]
        base_rows.extend([int(user)] * base_candidate_items)
        base_cols.extend(base_items.tolist())
        base_values.extend(base_weights.tolist())

        for arm, extra_items in extras.items():
            overlap = base_set & set(extra_items.tolist())
            base_extra_overlap += len(overlap)
            train_pair_edges += sum(
                int(item in seen)
                for item in np.concatenate([base_items, extra_items])
            )
            rows, cols, values = arm_rows[arm]
            rows.extend([int(user)] * base_candidate_items)
            cols.extend(base_items.tolist())
            values.extend(base_weights.tolist())
            rows.extend([int(user)] * supplemental_candidate_items)
            cols.extend(extra_items.tolist())
            values.extend(
                [float(extra_weight)] * supplemental_candidate_items
            )
            extra_rows, extra_cols, extra_values = block_rows[arm]
            extra_rows.extend([int(user)] * supplemental_candidate_items)
            extra_cols.extend(extra_items.tolist())
            extra_values.extend(
                [float(extra_weight)] * supplemental_candidate_items
            )
            base_mass_errors.append(
                abs(float(base_weights.sum()) - base_mass)
            )
            extra_mass_errors.append(
                abs(extra_weight * supplemental_candidate_items - extra_mass)
            )

    if insufficient:
        sample = insufficient[:5]
        raise RuntimeError(
            "insufficient positive excess candidates or pooled candidates; "
            f"affected_users={len(insufficient)}, sample={sample}"
        )

    return arm_rows, block_rows, {
        "users": int(len(users)),
        "base_candidate_items": int(base_candidate_items),
        "supplemental_candidate_items": int(supplemental_candidate_items),
        "base_mass": float(base_mass),
        "extra_mass": float(extra_mass),
        "base_edges_identical": True,
        "base_extra_overlap": int(base_extra_overlap),
        "train_pair_edges": int(train_pair_edges),
        "mean_actual_shuffle_extra_jaccard": float(
            np.mean(jaccards) if jaccards else 0.0
        ),
        "identical_actual_shuffle_extra_row_share": float(
            identical_extra_rows / len(users) if len(users) else 0.0
        ),
        "max_base_mass_error": float(max(base_mass_errors, default=0.0)),
        "max_extra_mass_error": float(max(extra_mass_errors, default=0.0)),
        "category_probability_actual": category_diagnostics_actual,
        "category_probability_shuffle": category_diagnostics_shuffle,
        "candidate_item_probability_actual": item_diagnostics_actual,
        "candidate_item_probability_shuffle": item_diagnostics_shuffle,
    }


def _build_supplemental_operators(
    train: pd.DataFrame,
    transition_evidence: pd.DataFrame,
    item_evidence: pd.DataFrame,
    recent: pd.DataFrame,
    q_actual: np.ndarray,
    q_shuffle: np.ndarray,
    item_category: np.ndarray,
    n_users: int,
    n_items: int,
    n_cat: int,
    *,
    category_min_support_users: int,
    category_kappa: float,
    item_min_support_users: int,
    item_kappa: float,
    cross_fit_folds: int,
    max_target_categories: int,
    base_candidate_items: int,
    supplemental_candidate_items: int,
) -> tuple[
    dict[str, torch.Tensor],
    dict[str, dict],
    dict,
    dict[str, torch.Tensor],
]:
    collected = {
        arm: ([], [], [])
        for arm in (ARM_GENERAL, ARM_ACTUAL, ARM_SHUFFLE)
    }
    collected_blocks = {
        "base": ([], [], []),
        **{
            arm: ([], [], [])
            for arm in (ARM_GENERAL, ARM_ACTUAL, ARM_SHUFFLE)
        },
    }
    folds = np.arange(n_users, dtype=np.int64) % cross_fit_folds
    fold_diagnostics = []
    for fold in range(cross_fit_folds):
        consumers = np.flatnonzero(folds == fold)
        transition_reference = transition_evidence.loc[
            ~transition_evidence["u_idx"].isin(consumers)
        ]
        item_reference = item_evidence.loc[
            ~item_evidence["u_idx"].isin(consumers)
        ]
        fold_rows, fold_blocks, diagnostics = _supplemental_rows(
            consumers,
            train,
            recent,
            transition_reference,
            item_reference,
            q_actual,
            q_shuffle,
            item_category,
            n_items,
            n_cat,
            category_min_support_users=category_min_support_users,
            category_kappa=category_kappa,
            item_min_support_users=item_min_support_users,
            item_kappa=item_kappa,
            max_target_categories=max_target_categories,
            base_candidate_items=base_candidate_items,
            supplemental_candidate_items=supplemental_candidate_items,
        )
        for arm in collected:
            for target, values in zip(
                collected[arm], fold_rows[arm], strict=True
            ):
                target.extend(values)
        for block in collected_blocks:
            for target, values in zip(
                collected_blocks[block], fold_blocks[block], strict=True
            ):
                target.extend(values)
        fold_diagnostics.append({"fold": fold, **diagnostics})

    operators = {}
    arm_diagnostics = {}
    expected_edges_per_user = base_candidate_items + supplemental_candidate_items
    for arm, (rows, cols, values) in collected.items():
        row_array = np.asarray(rows, dtype=np.int64)
        col_array = np.asarray(cols, dtype=np.int64)
        value_array = np.asarray(values, dtype=np.float64)
        with torch.sparse.check_sparse_tensor_invariants():
            operator = torch.sparse_coo_tensor(
                torch.from_numpy(np.stack([row_array, col_array])).long(),
                torch.from_numpy(value_array.astype(np.float32)),
                size=(n_users, n_items),
            ).coalesce()
        row_counts = np.bincount(row_array, minlength=n_users)
        row_mass = np.bincount(
            row_array, weights=value_array, minlength=n_users
        )
        if not np.all(row_counts == expected_edges_per_user):
            raise RuntimeError("supplemental relation row edge count mismatch")
        if not np.allclose(row_mass, 1.0, atol=1e-7):
            raise RuntimeError("supplemental relation row mass mismatch")
        operators[arm] = operator
        arm_diagnostics[arm] = {
            "n_edges": int(operator._nnz()),
            "n_active_users": int(n_users),
            "n_zero_relation_users": 0,
            "active_user_share": 1.0,
            "mean_edges_per_active_user": float(expected_edges_per_user),
            "max_active_row_mass_error": float(
                np.abs(row_mass - 1.0).max(initial=0.0)
            ),
        }

    block_operators = {}
    for block, (rows, cols, values) in collected_blocks.items():
        indices = np.stack(
            [np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)]
        )
        with torch.sparse.check_sparse_tensor_invariants():
            block_operators[block] = torch.sparse_coo_tensor(
                torch.from_numpy(indices).long(),
                torch.from_numpy(np.asarray(values, dtype=np.float32)),
                size=(n_users, n_items),
            ).coalesce()

    return operators, arm_diagnostics, {
        "base_candidate_items": int(base_candidate_items),
        "supplemental_candidate_items": int(supplemental_candidate_items),
        "edges_per_active_user": int(expected_edges_per_user),
        "base_mass": float(
            base_candidate_items / expected_edges_per_user
        ),
        "extra_mass": float(
            supplemental_candidate_items / expected_edges_per_user
        ),
        "base_edges_identical": bool(
            all(fold["base_edges_identical"] for fold in fold_diagnostics)
        ),
        "base_extra_overlap": int(
            sum(fold["base_extra_overlap"] for fold in fold_diagnostics)
        ),
        "train_pair_edges": int(
            sum(fold["train_pair_edges"] for fold in fold_diagnostics)
        ),
        "mean_actual_shuffle_extra_jaccard": float(
            np.average(
                [fold["mean_actual_shuffle_extra_jaccard"] for fold in fold_diagnostics],
                weights=[fold["users"] for fold in fold_diagnostics],
            )
        ),
        "identical_actual_shuffle_extra_row_share": float(
            np.average(
                [fold["identical_actual_shuffle_extra_row_share"] for fold in fold_diagnostics],
                weights=[fold["users"] for fold in fold_diagnostics],
            )
        ),
        "max_base_mass_error": float(
            max(fold["max_base_mass_error"] for fold in fold_diagnostics)
        ),
        "max_extra_mass_error": float(
            max(fold["max_extra_mass_error"] for fold in fold_diagnostics)
        ),
        "folds": fold_diagnostics,
    }, block_operators


def build_clv_conditioned_candidate_item_graph(
    train: pd.DataFrame,
    n_users: int,
    n_items: int,
    n_cat: int,
    *,
    category_kappa: float = DEFAULT_KAPPA,
    category_min_support_users: int = DEFAULT_MIN_SUPPORT_USERS,
    item_kappa: float = DEFAULT_ITEM_KAPPA,
    item_min_support_users: int = DEFAULT_ITEM_MIN_SUPPORT_USERS,
    shuffle_seed: int = DEFAULT_SHUFFLE_SEED,
    shuffle_degree_bins: int = DEFAULT_SHUFFLE_DEGREE_BINS,
    cross_fit_folds: int = DEFAULT_CROSS_FIT_FOLDS,
    max_target_categories: int = DEFAULT_MAX_TARGET_CATEGORIES,
    max_candidate_items: int = DEFAULT_MAX_CANDIDATE_ITEMS,
) -> CLVCandidateItemGraph:
    required = {"u_idx", "i_idx", "cat_idx", "b_raw", "t", "v"}
    missing = required - set(train.columns)
    if missing:
        raise ValueError(f"candidate-item graph requires {sorted(missing)}")
    if train.empty or min(n_users, n_items, n_cat) <= 0:
        raise ValueError("non-empty train and positive graph sizes are required")
    if category_kappa < 0 or item_kappa < 0:
        raise ValueError("shrinkage constants must be non-negative")
    if min(category_min_support_users, item_min_support_users) <= 0:
        raise ValueError("minimum support must be positive")
    if cross_fit_folds < 2 or cross_fit_folds > n_users:
        raise ValueError("cross_fit_folds must be between 2 and n_users")
    if min(max_target_categories, max_candidate_items) <= 0:
        raise ValueError("candidate limits must be positive")

    n_hat, v_hat, clv_proxy, valid = _historical_clv(train, n_users)
    q_clv = _midrank_percentile(clv_proxy, valid)
    unique_pairs = train[["u_idx", "i_idx"]].drop_duplicates()
    user_degree = np.bincount(
        unique_pairs["u_idx"].to_numpy(np.int64), minlength=n_users
    ).astype(np.float64)
    q_shuffle, shuffle_stratum = _degree_stratified_shuffle(
        q_clv,
        user_degree,
        n_bins=shuffle_degree_bins,
        seed=shuffle_seed,
    )
    transition, recent, transition_diagnostics = _transition_evidence(
        train, n_users
    )
    item_evidence, item_evidence_diagnostics = _first_acquisition_item_evidence(
        train, n_users
    )
    item_category = _item_category_mapping(train, n_items, n_cat)

    operators: dict[str, torch.Tensor] = {}
    arm_diagnostics = {}
    for arm, q_values in (
        (ARM_GENERAL, q_clv),
        (ARM_ACTUAL, q_clv),
        (ARM_SHUFFLE, q_shuffle),
    ):
        operator, diagnostics = _build_operator(
            train,
            transition,
            item_evidence,
            recent,
            q_values,
            item_category,
            n_users,
            n_items,
            n_cat,
            arm=arm,
            category_min_support_users=category_min_support_users,
            category_kappa=category_kappa,
            item_min_support_users=item_min_support_users,
            item_kappa=item_kappa,
            cross_fit_folds=cross_fit_folds,
            max_target_categories=max_target_categories,
            max_candidate_items=max_candidate_items,
        )
        operators[arm] = operator
        arm_diagnostics[arm] = diagnostics

    diagnostics = {
        "definition": {
            "historical_clv_proxy": "N_hat * V_hat",
            "n_hat": "number of distinct train baskets",
            "v_hat": "mean train basket value",
            "category_direction": (
                "CLV-conditioned next-category probability from the final train basket"
            ),
            "within_category_allocation": (
                "CLV-conditioned first-acquisition item probability within target category"
            ),
            "actual_candidate_relation": (
                "positive absolute probability excess over the pooled candidate-item distribution"
            ),
            "general_control": "pooled candidate-item probability",
            "candidate_items_exclude_user_train_pairs": True,
            "self_history_exclusion": "five-fold user cross-fitting",
            "item_price_used": False,
        },
        "settings": {
            "category_kappa": float(category_kappa),
            "category_min_distinct_user_support": int(
                category_min_support_users
            ),
            "item_kappa": float(item_kappa),
            "item_min_distinct_user_support": int(item_min_support_users),
            "shuffle_seed": int(shuffle_seed),
            "shuffle_degree_bins": int(shuffle_degree_bins),
            "cross_fit_folds": int(cross_fit_folds),
            "max_target_categories_per_user": int(max_target_categories),
            "max_candidate_items_per_user": int(max_candidate_items),
        },
        "transition_evidence": transition_diagnostics,
        "item_evidence": item_evidence_diagnostics,
        "arms": arm_diagnostics,
        "historical_clv": {
            "n_hat": _stats(n_hat[valid]),
            "v_hat": _stats(v_hat[valid]),
            "clv_proxy": _stats(clv_proxy[valid]),
            "q_clv": _stats(q_clv[valid]),
            "shuffle_preserves_values": bool(
                np.allclose(np.sort(q_clv[valid]), np.sort(q_shuffle[valid]))
            ),
            "actual_shuffle_spearman": _safe_spearman(
                q_clv[valid], q_shuffle[valid]
            ),
        },
        "m1_catalog_items_preserved": int(n_items),
    }
    return CLVCandidateItemGraph(
        n_hat=n_hat,
        v_hat=v_hat,
        clv_proxy=clv_proxy,
        clv_percentile=q_clv,
        clv_shuffle_percentile=q_shuffle,
        clv_shuffle_stratum=shuffle_stratum,
        user_item_operators=operators,
        diagnostics=diagnostics,
    )


def build_clv_conditioned_common_support_candidate_item_graph(
    train: pd.DataFrame,
    n_users: int,
    n_items: int,
    n_cat: int,
    *,
    category_kappa: float = DEFAULT_KAPPA,
    category_min_support_users: int = DEFAULT_MIN_SUPPORT_USERS,
    item_kappa: float = DEFAULT_ITEM_KAPPA,
    item_min_support_users: int = DEFAULT_ITEM_MIN_SUPPORT_USERS,
    shuffle_seed: int = DEFAULT_SHUFFLE_SEED,
    shuffle_degree_bins: int = DEFAULT_SHUFFLE_DEGREE_BINS,
    cross_fit_folds: int = DEFAULT_CROSS_FIT_FOLDS,
    max_target_categories: int = DEFAULT_MAX_TARGET_CATEGORIES,
    max_candidate_items: int = DEFAULT_MAX_CANDIDATE_ITEMS,
) -> CLVCandidateItemGraph:
    """Build arm-invariant pooled candidates with arm-specific row weights."""
    required = {"u_idx", "i_idx", "cat_idx", "b_raw", "t", "v"}
    missing = required - set(train.columns)
    if missing:
        raise ValueError(f"candidate-item graph requires {sorted(missing)}")
    if train.empty or min(n_users, n_items, n_cat) <= 0:
        raise ValueError("non-empty train and positive graph sizes are required")
    if category_kappa < 0 or item_kappa < 0:
        raise ValueError("shrinkage constants must be non-negative")
    if min(category_min_support_users, item_min_support_users) <= 0:
        raise ValueError("minimum support must be positive")
    if cross_fit_folds < 2 or cross_fit_folds > n_users:
        raise ValueError("cross_fit_folds must be between 2 and n_users")
    if min(max_target_categories, max_candidate_items) <= 0:
        raise ValueError("candidate limits must be positive")

    n_hat, v_hat, clv_proxy, valid = _historical_clv(train, n_users)
    q_clv = _midrank_percentile(clv_proxy, valid)
    unique_pairs = train[["u_idx", "i_idx"]].drop_duplicates()
    user_degree = np.bincount(
        unique_pairs["u_idx"].to_numpy(np.int64), minlength=n_users
    ).astype(np.float64)
    q_shuffle, shuffle_stratum = _degree_stratified_shuffle(
        q_clv,
        user_degree,
        n_bins=shuffle_degree_bins,
        seed=shuffle_seed,
    )
    transition, recent, transition_diagnostics = _transition_evidence(
        train, n_users
    )
    item_evidence, item_evidence_diagnostics = _first_acquisition_item_evidence(
        train, n_users
    )
    item_category = _item_category_mapping(train, n_items, n_cat)
    (
        operators,
        arm_diagnostics,
        common_support_diagnostics,
    ) = _build_common_support_operators(
        train,
        transition,
        item_evidence,
        recent,
        q_clv,
        q_shuffle,
        item_category,
        n_users,
        n_items,
        n_cat,
        category_min_support_users=category_min_support_users,
        category_kappa=category_kappa,
        item_min_support_users=item_min_support_users,
        item_kappa=item_kappa,
        cross_fit_folds=cross_fit_folds,
        max_target_categories=max_target_categories,
        max_candidate_items=max_candidate_items,
    )
    diagnostics = {
        "definition": {
            "historical_clv_proxy": "N_hat * V_hat",
            "n_hat": "number of distinct train baskets",
            "v_hat": "mean train basket value",
            "category_direction": (
                "CLV-conditioned next-category probability from the final train basket"
            ),
            "within_category_allocation": (
                "CLV-conditioned first-acquisition item probability within target category"
            ),
            "common_candidate_support": (
                "top pooled candidate-item probabilities, shared by every arm"
            ),
            "general_row_weight": "pooled probability normalized on common support",
            "actual_row_weight": (
                "actual-CLV conditional probability normalized on common support"
            ),
            "shuffle_row_weight": (
                "degree-matched shuffled-CLV conditional probability normalized on common support"
            ),
            "positive_excess_clipping": False,
            "candidate_items_exclude_user_train_pairs": True,
            "self_history_exclusion": "five-fold user cross-fitting",
            "item_price_used": False,
        },
        "settings": {
            "relation_mode": RELATION_MODE_COMMON_SUPPORT,
            "category_kappa": float(category_kappa),
            "category_min_distinct_user_support": int(
                category_min_support_users
            ),
            "item_kappa": float(item_kappa),
            "item_min_distinct_user_support": int(item_min_support_users),
            "shuffle_seed": int(shuffle_seed),
            "shuffle_degree_bins": int(shuffle_degree_bins),
            "cross_fit_folds": int(cross_fit_folds),
            "max_target_categories_per_user": int(max_target_categories),
            "max_candidate_items_per_user": int(max_candidate_items),
        },
        "transition_evidence": transition_diagnostics,
        "item_evidence": item_evidence_diagnostics,
        "common_support": common_support_diagnostics,
        "arms": arm_diagnostics,
        "historical_clv": {
            "n_hat": _stats(n_hat[valid]),
            "v_hat": _stats(v_hat[valid]),
            "clv_proxy": _stats(clv_proxy[valid]),
            "q_clv": _stats(q_clv[valid]),
            "shuffle_preserves_values": bool(
                np.allclose(np.sort(q_clv[valid]), np.sort(q_shuffle[valid]))
            ),
            "actual_shuffle_spearman": _safe_spearman(
                q_clv[valid], q_shuffle[valid]
            ),
        },
        "m1_catalog_items_preserved": int(n_items),
    }
    return CLVCandidateItemGraph(
        n_hat=n_hat,
        v_hat=v_hat,
        clv_proxy=clv_proxy,
        clv_percentile=q_clv,
        clv_shuffle_percentile=q_shuffle,
        clv_shuffle_stratum=shuffle_stratum,
        user_item_operators=operators,
        diagnostics=diagnostics,
    )


def build_clv_conditioned_supplemental_candidate_item_graph(
    train: pd.DataFrame,
    n_users: int,
    n_items: int,
    n_cat: int,
    *,
    category_kappa: float = DEFAULT_KAPPA,
    category_min_support_users: int = DEFAULT_MIN_SUPPORT_USERS,
    item_kappa: float = DEFAULT_ITEM_KAPPA,
    item_min_support_users: int = DEFAULT_ITEM_MIN_SUPPORT_USERS,
    shuffle_seed: int = DEFAULT_SHUFFLE_SEED,
    shuffle_degree_bins: int = DEFAULT_SHUFFLE_DEGREE_BINS,
    cross_fit_folds: int = DEFAULT_CROSS_FIT_FOLDS,
    max_target_categories: int = DEFAULT_MAX_TARGET_CATEGORIES,
    base_candidate_items: int = DEFAULT_BASE_CANDIDATE_ITEMS,
    supplemental_candidate_items: int = DEFAULT_SUPPLEMENTAL_CANDIDATE_ITEMS,
) -> CLVCandidateItemGraph:
    """Keep pooled candidates and add a fixed-mass CLV excess block."""
    required = {"u_idx", "i_idx", "cat_idx", "b_raw", "t", "v"}
    missing = required - set(train.columns)
    if missing:
        raise ValueError(f"candidate-item graph requires {sorted(missing)}")
    if train.empty or min(n_users, n_items, n_cat) <= 0:
        raise ValueError("non-empty train and positive graph sizes are required")
    if category_kappa < 0 or item_kappa < 0:
        raise ValueError("shrinkage constants must be non-negative")
    if min(category_min_support_users, item_min_support_users) <= 0:
        raise ValueError("minimum support must be positive")
    if cross_fit_folds < 2 or cross_fit_folds > n_users:
        raise ValueError("cross_fit_folds must be between 2 and n_users")
    if min(
        max_target_categories,
        base_candidate_items,
        supplemental_candidate_items,
    ) <= 0:
        raise ValueError("candidate limits must be positive")

    n_hat, v_hat, clv_proxy, valid = _historical_clv(train, n_users)
    q_clv = _midrank_percentile(clv_proxy, valid)
    unique_pairs = train[["u_idx", "i_idx"]].drop_duplicates()
    user_degree = np.bincount(
        unique_pairs["u_idx"].to_numpy(np.int64), minlength=n_users
    ).astype(np.float64)
    q_shuffle, shuffle_stratum = _degree_stratified_shuffle(
        q_clv,
        user_degree,
        n_bins=shuffle_degree_bins,
        seed=shuffle_seed,
    )
    transition, recent, transition_diagnostics = _transition_evidence(
        train, n_users
    )
    item_evidence, item_evidence_diagnostics = (
        _first_acquisition_item_evidence(train, n_users)
    )
    item_category = _item_category_mapping(train, n_items, n_cat)
    operators, arm_diagnostics, support_diagnostics, candidate_blocks = (
        _build_supplemental_operators(
            train,
            transition,
            item_evidence,
            recent,
            q_clv,
            q_shuffle,
            item_category,
            n_users,
            n_items,
            n_cat,
            category_min_support_users=category_min_support_users,
            category_kappa=category_kappa,
            item_min_support_users=item_min_support_users,
            item_kappa=item_kappa,
            cross_fit_folds=cross_fit_folds,
            max_target_categories=max_target_categories,
            base_candidate_items=base_candidate_items,
            supplemental_candidate_items=supplemental_candidate_items,
        )
    )

    within_stratum_spearman = {}
    for stratum in np.unique(shuffle_stratum[valid]):
        members = valid & (shuffle_stratum == stratum)
        within_stratum_spearman[str(int(stratum))] = _safe_spearman(
            q_clv[members], q_shuffle[members]
        )
    diagnostics = {
        "definition": {
            "historical_clv_proxy": "N_hat * V_hat",
            "n_hat": "number of distinct train baskets",
            "v_hat": "mean train basket value",
            "general_base_relation": (
                "pooled first-acquisition category-to-item probability"
            ),
            "actual_supplemental_relation": (
                "positive actual-CLV conditional probability excess outside the pooled Top-K"
            ),
            "shuffle_supplemental_relation": (
                "positive degree-matched shuffled-CLV conditional probability excess outside the pooled Top-K"
            ),
            "supplemental_weight": (
                "uniform within a fixed supplemental block mass"
            ),
            "candidate_items_exclude_user_train_pairs": True,
            "self_history_exclusion": "five-fold user cross-fitting",
            "item_price_used": False,
        },
        "settings": {
            "relation_mode": RELATION_MODE_SUPPLEMENTAL,
            "category_kappa": float(category_kappa),
            "category_min_distinct_user_support": int(
                category_min_support_users
            ),
            "item_kappa": float(item_kappa),
            "item_min_distinct_user_support": int(item_min_support_users),
            "shuffle_seed": int(shuffle_seed),
            "shuffle_degree_bins": int(shuffle_degree_bins),
            "cross_fit_folds": int(cross_fit_folds),
            "max_target_categories_per_user": int(max_target_categories),
            "base_candidate_items_per_user": int(base_candidate_items),
            "supplemental_candidate_items_per_user": int(
                supplemental_candidate_items
            ),
        },
        "transition_evidence": transition_diagnostics,
        "item_evidence": item_evidence_diagnostics,
        "supplemental_support": support_diagnostics,
        "arms": arm_diagnostics,
        "historical_clv": {
            "n_hat": _stats(n_hat[valid]),
            "v_hat": _stats(v_hat[valid]),
            "clv_proxy": _stats(clv_proxy[valid]),
            "q_clv": _stats(q_clv[valid]),
            "shuffle_preserves_values": bool(
                np.allclose(np.sort(q_clv[valid]), np.sort(q_shuffle[valid]))
            ),
            "actual_shuffle_spearman": _safe_spearman(
                q_clv[valid], q_shuffle[valid]
            ),
            "actual_shuffle_mean_absolute_change": float(
                np.mean(np.abs(q_clv[valid] - q_shuffle[valid]))
            ),
            "actual_shuffle_identical_user_share": float(
                np.mean(np.isclose(q_clv[valid], q_shuffle[valid]))
            ),
            "within_degree_stratum_spearman": within_stratum_spearman,
        },
        "m1_catalog_items_preserved": int(n_items),
    }
    return CLVCandidateItemGraph(
        n_hat=n_hat,
        v_hat=v_hat,
        clv_proxy=clv_proxy,
        clv_percentile=q_clv,
        clv_shuffle_percentile=q_shuffle,
        clv_shuffle_stratum=shuffle_stratum,
        user_item_operators=operators,
        diagnostics=diagnostics,
        candidate_blocks=candidate_blocks,
    )


__all__ = [
    "ACTIVE_ARMS",
    "ARM_ACTUAL",
    "ARM_GENERAL",
    "ARM_SHUFFLE",
    "CLVCandidateItemGraph",
    "DEFAULT_BASE_CANDIDATE_ITEMS",
    "DEFAULT_SUPPLEMENTAL_CANDIDATE_ITEMS",
    "RELATION_MODE_COMMON_SUPPORT",
    "RELATION_MODE_POSITIVE_EXCESS",
    "RELATION_MODE_SUPPLEMENTAL",
    "build_clv_conditioned_candidate_item_graph",
    "build_clv_conditioned_common_support_candidate_item_graph",
    "build_clv_conditioned_supplemental_candidate_item_graph",
]
