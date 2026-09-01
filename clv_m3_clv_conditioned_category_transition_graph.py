"""Train-only CLV-conditioned first-acquisition category transitions.

The module keeps the M1 binary user-item graph outside this file.  It builds
one additional sparse user-to-target-category relation for each experimental
arm.  Transition statistics are cross-fitted by user so a user's own history
cannot create the relation later consumed by that same user.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
import torch


ARM_GENERAL = "general_transition"
ARM_ACTUAL = "actual_clv"
ARM_SHUFFLE = "clv_shuffle"
ACTIVE_ARMS = (ARM_GENERAL, ARM_ACTUAL, ARM_SHUFFLE)

DEFAULT_KAPPA = 20.0
DEFAULT_MIN_SUPPORT_USERS = 5
DEFAULT_LOG_LIFT_CAP = float(np.log(3.0))
DEFAULT_SHUFFLE_SEED = 42
DEFAULT_SHUFFLE_DEGREE_BINS = 10
DEFAULT_CROSS_FIT_FOLDS = 5
DEFAULT_MAX_TARGET_CATEGORIES = 20


@dataclass(frozen=True)
class CLVCategoryTransitionGraph:
    n_hat: np.ndarray
    v_hat: np.ndarray
    clv_proxy: np.ndarray
    clv_percentile: np.ndarray
    clv_shuffle_percentile: np.ndarray
    clv_shuffle_stratum: np.ndarray
    user_category_operators: dict[str, torch.Tensor]
    category_item_operator: torch.Tensor
    diagnostics: dict


def _stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "median": float("nan"),
            "max": float("nan"),
        }
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "median": float(np.median(values)),
        "max": float(values.max()),
    }


def _safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    finite = np.isfinite(left) & np.isfinite(right)
    if finite.sum() < 2 or left[finite].std() <= 1e-12:
        return 0.0
    if right[finite].std() <= 1e-12:
        return 0.0
    value = spearmanr(left[finite], right[finite]).statistic
    return float(value) if np.isfinite(value) else 0.0


def _midrank_percentile(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if values.ndim != 1 or valid.shape != values.shape:
        raise ValueError("values and valid must be aligned one-dimensional arrays")
    if not valid.any() or not np.isfinite(values[valid]).all():
        raise ValueError("at least one finite historical CLV value is required")
    result = np.zeros(len(values), dtype=np.float64)
    result[valid] = (
        rankdata(values[valid], method="average") - 0.5
    ) / int(valid.sum())
    return result


def _historical_clv(
    train: pd.DataFrame, n_users: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    basket = (
        train.groupby(["u_idx", "b_raw"], sort=False)["v"]
        .sum()
        .rename("basket_value")
    )
    by_user = basket.groupby(level="u_idx", sort=True)
    summary = pd.DataFrame(
        {"n_hat": by_user.size(), "v_hat": by_user.mean()}
    )
    summary["clv_proxy"] = summary["n_hat"] * summary["v_hat"]

    n_hat = np.full(n_users, np.nan, dtype=np.float64)
    v_hat = np.full(n_users, np.nan, dtype=np.float64)
    clv_proxy = np.full(n_users, np.nan, dtype=np.float64)
    user_ids = summary.index.to_numpy(np.int64)
    if user_ids.min(initial=0) < 0 or user_ids.max(initial=-1) >= n_users:
        raise ValueError("train user index is outside n_users")
    n_hat[user_ids] = summary["n_hat"].to_numpy(np.float64)
    v_hat[user_ids] = summary["v_hat"].to_numpy(np.float64)
    clv_proxy[user_ids] = summary["clv_proxy"].to_numpy(np.float64)
    valid = np.isfinite(clv_proxy)
    return n_hat, v_hat, clv_proxy, valid


def _degree_stratified_shuffle(
    values: np.ndarray,
    user_degree: np.ndarray,
    *,
    n_bins: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if n_bins <= 0:
        raise ValueError("shuffle_degree_bins must be positive")
    values = np.asarray(values, dtype=np.float64)
    user_degree = np.asarray(user_degree, dtype=np.float64)
    if values.shape != user_degree.shape:
        raise ValueError("CLV percentile and user degree must have the same shape")
    shuffled = values.copy()
    strata = np.full(len(values), -1, dtype=np.int16)
    active = np.flatnonzero((user_degree > 0) & np.isfinite(values))
    if not len(active):
        return shuffled, strata
    ranks = rankdata(user_degree[active], method="average")
    assigned = np.floor((ranks - 0.5) * n_bins / len(active)).astype(np.int16)
    assigned = np.minimum(assigned, n_bins - 1)
    strata[active] = assigned
    rng = np.random.default_rng(seed)
    for stratum in np.unique(assigned):
        index = active[assigned == stratum]
        if len(index) < 2:
            continue
        source = rng.permutation(index)
        if np.array_equal(source, index):
            source = np.roll(source, 1)
        shuffled[index] = values[source]
    return shuffled, strata


def _basket_order(train: pd.DataFrame) -> pd.DataFrame:
    baskets = train[["u_idx", "b_raw", "t"]].drop_duplicates()
    duplicate_time = baskets.duplicated(["u_idx", "b_raw"], keep=False)
    if duplicate_time.any():
        raise ValueError("one user-basket pair is associated with multiple times")
    baskets = baskets.assign(_basket_key=baskets["b_raw"].astype(str))
    baskets = baskets.sort_values(
        ["u_idx", "t", "_basket_key"], kind="stable"
    ).reset_index(drop=True)
    baskets["basket_order"] = baskets.groupby("u_idx", sort=False).cumcount()
    return baskets.drop(columns="_basket_key")


def _transition_evidence(
    train: pd.DataFrame,
    n_users: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    baskets = _basket_order(train)
    lines = (
        train[["u_idx", "b_raw", "i_idx", "cat_idx"]]
        .drop_duplicates()
        .merge(
            baskets[["u_idx", "b_raw", "basket_order"]],
            on=["u_idx", "b_raw"],
            how="left",
            validate="many_to_one",
        )
    )
    if lines["basket_order"].isna().any():
        raise RuntimeError("basket ordering failed for a train line")
    lines["basket_order"] = lines["basket_order"].astype(np.int64)

    basket_category = (
        lines.groupby(["u_idx", "basket_order", "cat_idx"], sort=False)
        .size()
        .rename("item_count")
        .reset_index()
    )
    basket_total = basket_category.groupby(
        ["u_idx", "basket_order"], sort=False
    )["item_count"].transform("sum")
    basket_category["category_share"] = (
        basket_category["item_count"] / basket_total
    )

    first_order = lines.groupby(["u_idx", "i_idx"], sort=False)[
        "basket_order"
    ].transform("min")
    first_item = lines.loc[lines["basket_order"].eq(first_order)].copy()
    first_item = first_item.loc[first_item["basket_order"].gt(0)]
    target_category = (
        first_item.groupby(
            ["u_idx", "basket_order", "cat_idx"], sort=False
        )["i_idx"]
        .nunique()
        .rename("new_item_count")
        .reset_index()
        .rename(columns={"basket_order": "target_order", "cat_idx": "d_idx"})
    )
    target_total = target_category.groupby(
        ["u_idx", "target_order"], sort=False
    )["new_item_count"].transform("sum")
    target_category["target_share"] = (
        target_category["new_item_count"] / target_total
    )

    source = basket_category.rename(
        columns={"basket_order": "source_order", "cat_idx": "c_idx"}
    )
    source["target_order"] = source["source_order"] + 1
    crossed = source.merge(
        target_category,
        on=["u_idx", "target_order"],
        how="inner",
        validate="many_to_many",
    )
    crossed["mass"] = (
        crossed["category_share"] * crossed["target_share"]
    )
    pair = (
        crossed.groupby(["u_idx", "c_idx", "d_idx"], sort=False)["mass"]
        .sum()
        .reset_index()
    )
    user_mass = pair.groupby("u_idx", sort=False)["mass"].transform("sum")
    if len(pair) and np.any(user_mass.to_numpy(np.float64) <= 0):
        raise RuntimeError("transition evidence must have positive user mass")
    pair["mass"] = pair["mass"] / user_mass

    last_order = baskets.groupby("u_idx", sort=False)["basket_order"].max()
    recent = basket_category.merge(
        last_order.rename("last_order"),
        left_on="u_idx",
        right_index=True,
        how="left",
        validate="many_to_one",
    )
    recent = recent.loc[recent["basket_order"].eq(recent["last_order"]), [
        "u_idx",
        "cat_idx",
        "category_share",
    ]].rename(columns={"cat_idx": "c_idx", "category_share": "recent_share"})
    recent_mass = recent.groupby("u_idx", sort=False)["recent_share"].sum()
    active_recent = recent_mass.index.to_numpy(np.int64)
    if len(active_recent) != n_users or not np.allclose(
        recent_mass.reindex(np.arange(n_users)).to_numpy(np.float64), 1.0
    ):
        raise RuntimeError("every train user must have one normalized recent basket")

    diagnostics = {
        "n_baskets": int(len(baskets)),
        "n_users_with_transition_evidence": int(pair["u_idx"].nunique()),
        "n_user_category_transition_rows": int(len(pair)),
        "n_consecutive_baskets_with_new_items": int(
            target_category[["u_idx", "target_order"]].drop_duplicates().shape[0]
        ),
        "max_user_transition_mass_error": float(
            np.abs(
                pair.groupby("u_idx", sort=False)["mass"].sum().to_numpy()
                - 1.0
            ).max(initial=0.0)
        ),
    }
    return pair, recent, diagnostics


def _category_item_operator(
    train: pd.DataFrame, n_items: int, n_cat: int
) -> torch.Tensor:
    mapping = train.groupby("i_idx", sort=True)["cat_idx"].agg(
        lambda values: values.iloc[0] if values.nunique() == 1 else -1
    )
    mapping = mapping.reindex(np.arange(n_items))
    if mapping.isna().any() or (mapping < 0).any():
        raise ValueError("every train item must map to exactly one category")
    item = np.arange(n_items, dtype=np.int64)
    category = mapping.to_numpy(np.int64)
    if category.min(initial=0) < 0 or category.max(initial=-1) >= n_cat:
        raise ValueError("item category index is outside n_cat")
    category_size = np.bincount(category, minlength=n_cat).astype(np.float64)
    values = 1.0 / category_size[category]
    with torch.sparse.check_sparse_tensor_invariants():
        return torch.sparse_coo_tensor(
            torch.from_numpy(np.stack([category, item])).long(),
            torch.from_numpy(values.astype(np.float32)),
            size=(n_cat, n_items),
        ).coalesce()


def _probabilities(
    pair: pd.DataFrame,
    q_values: np.ndarray,
    n_cat: int,
    *,
    min_support_users: int,
    kappa: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    if pair.empty:
        zeros = np.zeros((n_cat, n_cat), dtype=np.float64)
        return zeros, zeros.copy(), zeros.copy(), {
            "supported_edges": 0,
            "raw_edges": 0,
            "support_users": _stats(np.asarray([], dtype=np.float64)),
        }
    users = pair["u_idx"].to_numpy(np.int64)
    source = pair["c_idx"].to_numpy(np.int64)
    target = pair["d_idx"].to_numpy(np.int64)
    mass = pair["mass"].to_numpy(np.float64)
    support = (
        pair.groupby(["c_idx", "d_idx"], sort=True)["u_idx"]
        .nunique()
        .rename("n_users")
        .reset_index()
    )
    supported = support.loc[support["n_users"].ge(min_support_users)]
    support_mask = np.zeros((n_cat, n_cat), dtype=bool)
    support_mask[
        supported["c_idx"].to_numpy(np.int64),
        supported["d_idx"].to_numpy(np.int64),
    ] = True
    keep = support_mask[source, target]
    source, target, users, mass = (
        values[keep] for values in (source, target, users, mass)
    )

    x0 = np.zeros((n_cat, n_cat), dtype=np.float64)
    xh = np.zeros_like(x0)
    xl = np.zeros_like(x0)
    np.add.at(x0, (source, target), mass)
    np.add.at(xh, (source, target), mass * q_values[users])
    np.add.at(xl, (source, target), mass * (1.0 - q_values[users]))
    n0 = x0.sum(axis=1)
    p0 = np.divide(
        x0,
        n0[:, None],
        out=np.zeros_like(x0),
        where=n0[:, None] > 0,
    )
    nh = xh.sum(axis=1)
    nl = xl.sum(axis=1)
    ph = np.divide(
        xh + kappa * p0,
        nh[:, None] + kappa,
        out=np.zeros_like(xh),
        where=(nh[:, None] + kappa) > 0,
    )
    pl = np.divide(
        xl + kappa * p0,
        nl[:, None] + kappa,
        out=np.zeros_like(xl),
        where=(nl[:, None] + kappa) > 0,
    )
    for probability in (p0, ph, pl):
        if not np.isfinite(probability).all() or np.any(probability < 0):
            raise RuntimeError("transition probability is invalid")
    return p0, pl, ph, {
        "supported_edges": int(support_mask.sum()),
        "raw_edges": int(len(support)),
        "support_users": _stats(support["n_users"].to_numpy(np.float64)),
        "source_categories_with_supported_edges": int(np.sum(n0 > 0)),
    }


def _relation_rows(
    users: np.ndarray,
    recent: pd.DataFrame,
    pair_reference: pd.DataFrame,
    q_relation: np.ndarray,
    q_user: np.ndarray,
    n_cat: int,
    *,
    arm: str,
    min_support_users: int,
    kappa: float,
    log_lift_cap: float,
    max_target_categories: int,
) -> tuple[list[int], list[int], list[float], dict]:
    p0, pl, ph, probability_diagnostics = _probabilities(
        pair_reference,
        q_relation,
        n_cat,
        min_support_users=min_support_users,
        kappa=kappa,
    )
    recent_by_user = {
        int(user): group[["c_idx", "recent_share"]].to_numpy(np.float64)
        for user, group in recent.loc[recent["u_idx"].isin(users)].groupby(
            "u_idx", sort=False
        )
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
        score = np.zeros(n_cat, dtype=np.float64)
        for category_value, share in profile:
            category = int(category_value)
            if arm == ARM_GENERAL:
                contribution = p0[category]
            else:
                conditional = (
                    (1.0 - q_user[user]) * pl[category]
                    + q_user[user] * ph[category]
                )
                positive = (p0[category] > 0) & (conditional > p0[category])
                contribution = np.zeros(n_cat, dtype=np.float64)
                contribution[positive] = np.minimum(
                    np.log(conditional[positive] / p0[category, positive]),
                    log_lift_cap,
                )
            score += float(share) * contribution
        positive_target = np.flatnonzero(score > 1e-12)
        if not len(positive_target):
            zero_rows += 1
            continue
        if len(positive_target) > max_target_categories:
            order = np.lexsort(
                (positive_target, -score[positive_target])
            )
            positive_target = positive_target[order[:max_target_categories]]
        normalized = score[positive_target] / score[positive_target].sum()
        rows.extend([int(user)] * len(positive_target))
        cols.extend(positive_target.tolist())
        values.extend(normalized.tolist())
    probability_diagnostics = {
        **probability_diagnostics,
        "users": int(len(users)),
        "zero_relation_users": int(zero_rows),
    }
    return rows, cols, values, probability_diagnostics


def _build_user_category_operator(
    pair: pd.DataFrame,
    recent: pd.DataFrame,
    q_values: np.ndarray,
    n_users: int,
    n_cat: int,
    *,
    arm: str,
    min_support_users: int,
    kappa: float,
    log_lift_cap: float,
    cross_fit_folds: int,
    max_target_categories: int,
) -> tuple[torch.Tensor, dict]:
    all_rows: list[int] = []
    all_cols: list[int] = []
    all_values: list[float] = []
    folds = np.arange(n_users, dtype=np.int64) % cross_fit_folds
    fold_diagnostics = []
    for fold in range(cross_fit_folds):
        consumers = np.flatnonzero(folds == fold)
        reference = pair.loc[~pair["u_idx"].isin(consumers)]
        rows, cols, values, diagnostics = _relation_rows(
            consumers,
            recent,
            reference,
            q_values,
            q_values,
            n_cat,
            arm=arm,
            min_support_users=min_support_users,
            kappa=kappa,
            log_lift_cap=log_lift_cap,
            max_target_categories=max_target_categories,
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
            size=(n_users, n_cat),
        ).coalesce()
    row_mass = np.bincount(
        row_array, weights=value_array, minlength=n_users
    )
    active = row_mass > 0
    diagnostics = {
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
    return operator, diagnostics


def build_clv_conditioned_category_transition_graph(
    train: pd.DataFrame,
    n_users: int,
    n_items: int,
    n_cat: int,
    *,
    kappa: float = DEFAULT_KAPPA,
    min_support_users: int = DEFAULT_MIN_SUPPORT_USERS,
    log_lift_cap: float = DEFAULT_LOG_LIFT_CAP,
    shuffle_seed: int = DEFAULT_SHUFFLE_SEED,
    shuffle_degree_bins: int = DEFAULT_SHUFFLE_DEGREE_BINS,
    cross_fit_folds: int = DEFAULT_CROSS_FIT_FOLDS,
    max_target_categories: int = DEFAULT_MAX_TARGET_CATEGORIES,
) -> CLVCategoryTransitionGraph:
    required = {"u_idx", "i_idx", "cat_idx", "b_raw", "t", "v"}
    missing = required - set(train.columns)
    if missing:
        raise ValueError(f"category transition graph requires {sorted(missing)}")
    if train.empty or min(n_users, n_items, n_cat) <= 0:
        raise ValueError("non-empty train and positive graph sizes are required")
    if kappa < 0 or not np.isfinite(kappa):
        raise ValueError("kappa must be finite and non-negative")
    if min_support_users <= 0:
        raise ValueError("min_support_users must be positive")
    if log_lift_cap <= 0 or not np.isfinite(log_lift_cap):
        raise ValueError("log_lift_cap must be finite and positive")
    if cross_fit_folds < 2 or cross_fit_folds > n_users:
        raise ValueError("cross_fit_folds must be between 2 and n_users")
    if max_target_categories <= 0:
        raise ValueError("max_target_categories must be positive")

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
    pair, recent, transition_diagnostics = _transition_evidence(train, n_users)
    category_item = _category_item_operator(train, n_items, n_cat)
    operators: dict[str, torch.Tensor] = {}
    arm_diagnostics = {}
    for arm, q_values in (
        (ARM_GENERAL, q_clv),
        (ARM_ACTUAL, q_clv),
        (ARM_SHUFFLE, q_shuffle),
    ):
        operator, diagnostics = _build_user_category_operator(
            pair,
            recent,
            q_values,
            n_users,
            n_cat,
            arm=arm,
            min_support_users=min_support_users,
            kappa=kappa,
            log_lift_cap=log_lift_cap,
            cross_fit_folds=cross_fit_folds,
            max_target_categories=max_target_categories,
        )
        operators[arm] = operator
        arm_diagnostics[arm] = diagnostics

    diagnostics = {
        "definition": {
            "historical_clv_proxy": "N_hat * V_hat",
            "n_hat": "number of distinct train baskets",
            "v_hat": "mean train basket value",
            "transition": (
                "consecutive-basket source category to category of a "
                "first-acquired target item"
            ),
            "actual_relation": (
                "positive clipped log ratio of CLV-conditioned transition "
                "probability to pooled transition probability"
            ),
            "self_history_exclusion": (
                "five-fold user cross-fitting for transition estimation"
            ),
            "item_price_used": False,
        },
        "settings": {
            "kappa": float(kappa),
            "min_distinct_user_support": int(min_support_users),
            "log_lift_cap": float(log_lift_cap),
            "shuffle_seed": int(shuffle_seed),
            "shuffle_degree_bins": int(shuffle_degree_bins),
            "cross_fit_folds": int(cross_fit_folds),
            "max_target_categories_per_user": int(max_target_categories),
        },
        "transition_evidence": transition_diagnostics,
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
        "category_item_edges": int(category_item._nnz()),
        "category_item_row_mass_max_error": float(
            np.abs(
                np.bincount(
                    category_item.indices()[0].numpy(),
                    weights=category_item.values().numpy(),
                    minlength=n_cat,
                )
                - 1.0
            ).max(initial=0.0)
        ),
    }
    return CLVCategoryTransitionGraph(
        n_hat=n_hat,
        v_hat=v_hat,
        clv_proxy=clv_proxy,
        clv_percentile=q_clv,
        clv_shuffle_percentile=q_shuffle,
        clv_shuffle_stratum=shuffle_stratum,
        user_category_operators=operators,
        category_item_operator=category_item,
        diagnostics=diagnostics,
    )
