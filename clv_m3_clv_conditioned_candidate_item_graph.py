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


__all__ = [
    "ACTIVE_ARMS",
    "ARM_ACTUAL",
    "ARM_GENERAL",
    "ARM_SHUFFLE",
    "CLVCandidateItemGraph",
    "build_clv_conditioned_candidate_item_graph",
]
