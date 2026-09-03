"""Train-only user-neighbor relations for the CLV-conditioned M3 graph."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import rankdata


DEFAULT_TOP_K = 20
DEFAULT_DEGREE_BINS = 10
DEFAULT_SHUFFLE_SEED = 42


@dataclass(frozen=True)
class HistoricalCLVGates:
    n_hat: np.ndarray
    v_hat: np.ndarray
    clv_proxy: np.ndarray
    clv_percentile: np.ndarray
    clv_shuffle_percentile: np.ndarray
    constant_gate: np.ndarray
    degree_percentile: np.ndarray
    degree_stratum: np.ndarray
    diagnostics: dict


def _stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {key: float("nan") for key in ("mean", "std", "min", "median", "max")}
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "median": float(np.median(values)),
        "max": float(values.max()),
    }


def _midrank_percentile(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    result = np.zeros(len(values), dtype=np.float64)
    count = int(valid.sum())
    if count:
        result[valid] = (rankdata(values[valid], method="average") - 0.5) / count
    return result


def _degree_strata(
    degree: np.ndarray, *, n_bins: int
) -> tuple[np.ndarray, np.ndarray]:
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    degree = np.asarray(degree, dtype=np.float64)
    active = np.flatnonzero(degree > 0)
    strata = np.full(len(degree), -1, dtype=np.int16)
    percentile = np.zeros(len(degree), dtype=np.float64)
    if not len(active):
        return strata, percentile
    ranks = rankdata(degree[active], method="average")
    percentile[active] = (ranks - 0.5) / len(active)
    assigned = np.floor(percentile[active] * n_bins).astype(np.int16)
    assigned = np.minimum(assigned, n_bins - 1)
    strata[active] = assigned
    return strata, percentile


def _binary_matrix(
    train: pd.DataFrame, n_users: int, n_items: int
) -> sparse.csr_matrix:
    pairs = train[["u_idx", "i_idx"]].drop_duplicates()
    users = pairs["u_idx"].to_numpy(np.int64)
    items = pairs["i_idx"].to_numpy(np.int64)
    if len(users) and (
        users.min() < 0
        or users.max() >= n_users
        or items.min() < 0
        or items.max() >= n_items
    ):
        raise ValueError("train indices exceed the declared graph shape")
    return sparse.csr_matrix(
        (np.ones(len(pairs)), (users, items)),
        shape=(n_users, n_items),
        dtype=np.float64,
    )


def _topk_row_normalized(
    matrix: sparse.spmatrix, *, top_k: int
) -> sparse.csr_matrix:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    matrix = matrix.tocsr(copy=True)
    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    for user in range(matrix.shape[0]):
        start, end = matrix.indptr[user], matrix.indptr[user + 1]
        index = matrix.indices[start:end]
        score = matrix.data[start:end]
        valid = (index != user) & np.isfinite(score) & (score > 0)
        index, score = index[valid], score[valid]
        if not len(index):
            continue
        order = np.lexsort((index, -score))[:top_k]
        index, score = index[order], score[order]
        total = float(score.sum())
        if total <= 0:
            continue
        rows.extend([user] * len(index))
        cols.extend(index.tolist())
        values.extend((score / total).tolist())
    return sparse.csr_matrix(
        (values, (rows, cols)), shape=matrix.shape, dtype=np.float64
    )


def _relation_diagnostics(
    operator: sparse.csr_matrix, *, normalization: str
) -> dict:
    counts = np.diff(operator.indptr)
    eligible = counts > 0
    row_mass = np.asarray(operator.sum(axis=1)).ravel()
    return {
        "normalization": normalization,
        "eligible_user_count": int(eligible.sum()),
        "eligible_user_share": float(eligible.mean()) if len(eligible) else 0.0,
        "neighbor_count_mean": float(counts[eligible].mean()) if eligible.any() else 0.0,
        "neighbor_count_max": int(counts.max(initial=0)),
        "row_mass_max_error": (
            float(np.abs(row_mass[eligible] - 1.0).max()) if eligible.any() else 0.0
        ),
        "self_edge_count": int(operator.diagonal().astype(bool).sum()),
    }


def build_tfidf_neighbor_operator(
    train: pd.DataFrame,
    n_users: int,
    n_items: int,
    *,
    top_k: int = DEFAULT_TOP_K,
) -> tuple[sparse.csr_matrix, dict]:
    binary = _binary_matrix(train, n_users, n_items)
    item_degree = np.asarray(binary.sum(axis=0)).ravel()
    idf = np.log((n_users + 1.0) / (item_degree + 1.0))
    profile = binary.multiply(idf).tocsr()
    norm = np.sqrt(np.asarray(profile.multiply(profile).sum(axis=1)).ravel())
    inv = np.zeros_like(norm)
    inv[norm > 0] = 1.0 / norm[norm > 0]
    normalized = sparse.diags(inv) @ profile
    similarity = (normalized @ normalized.T).tocsr()
    similarity.setdiag(0.0)
    similarity.eliminate_zeros()
    operator = _topk_row_normalized(similarity, top_k=top_k)
    return operator, _relation_diagnostics(
        operator, normalization="binary_tfidf_cosine_then_topk_row"
    )


def build_ordinary_copurchase_operator(
    train: pd.DataFrame,
    n_users: int,
    n_items: int,
    *,
    top_k: int = DEFAULT_TOP_K,
) -> tuple[sparse.csr_matrix, dict]:
    binary = _binary_matrix(train, n_users, n_items)
    user_degree = np.asarray(binary.sum(axis=1)).ravel()
    item_degree = np.asarray(binary.sum(axis=0)).ravel()
    user_inv = np.zeros_like(user_degree)
    item_inv = np.zeros_like(item_degree)
    user_inv[user_degree > 0] = 1.0 / np.sqrt(user_degree[user_degree > 0])
    item_inv[item_degree > 0] = 1.0 / np.sqrt(item_degree[item_degree > 0])
    normalized = sparse.diags(user_inv) @ binary @ sparse.diags(item_inv)
    similarity = (normalized @ normalized.T).tocsr()
    similarity.setdiag(0.0)
    similarity.eliminate_zeros()
    operator = _topk_row_normalized(similarity, top_k=top_k)
    return operator, _relation_diagnostics(
        operator, normalization="m1_symmetric_two_hop_then_topk_row"
    )


def build_degree_matched_random_neighbor_operator(
    user_degree: np.ndarray,
    *,
    top_k: int = DEFAULT_TOP_K,
    n_bins: int = DEFAULT_DEGREE_BINS,
    seed: int = DEFAULT_SHUFFLE_SEED,
) -> tuple[sparse.csr_matrix, dict]:
    degree = np.asarray(user_degree, dtype=np.float64)
    strata, _ = _degree_strata(degree, n_bins=n_bins)
    rng = np.random.default_rng(seed)
    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    for user in np.flatnonzero(strata >= 0):
        pool = np.flatnonzero(strata == strata[user])
        pool = pool[pool != user]
        if not len(pool):
            continue
        chosen = rng.choice(pool, size=min(top_k, len(pool)), replace=False)
        rows.extend([int(user)] * len(chosen))
        cols.extend(chosen.tolist())
        values.extend([1.0 / len(chosen)] * len(chosen))
    operator = sparse.csr_matrix(
        (values, (rows, cols)), shape=(len(degree), len(degree)), dtype=np.float64
    )
    diagnostics = _relation_diagnostics(
        operator, normalization="degree_stratified_random_then_uniform_row"
    )
    diagnostics["degree_bins"] = int(n_bins)
    diagnostics["seed"] = int(seed)
    return operator, diagnostics


def build_historical_clv_gates(
    train: pd.DataFrame,
    n_users: int,
    *,
    shuffle_degree_bins: int = DEFAULT_DEGREE_BINS,
    shuffle_seed: int = DEFAULT_SHUFFLE_SEED,
) -> HistoricalCLVGates:
    basket_column = "b_raw" if "b_raw" in train.columns else "t"
    baskets = (
        train.groupby(["u_idx", basket_column], sort=False)["v"]
        .sum()
        .rename("basket_value")
    )
    by_user = baskets.groupby(level="u_idx", sort=True)
    summary = pd.DataFrame({"n_hat": by_user.size(), "v_hat": by_user.mean()})
    summary["clv_proxy"] = summary["n_hat"] * summary["v_hat"]
    n_hat = np.full(n_users, np.nan)
    v_hat = np.full(n_users, np.nan)
    clv_proxy = np.full(n_users, np.nan)
    users = summary.index.to_numpy(np.int64)
    n_hat[users] = summary["n_hat"].to_numpy(np.float64)
    v_hat[users] = summary["v_hat"].to_numpy(np.float64)
    clv_proxy[users] = summary["clv_proxy"].to_numpy(np.float64)
    valid = np.isfinite(clv_proxy)
    clv_percentile = _midrank_percentile(clv_proxy, valid)

    binary = _binary_matrix(train, n_users, int(train["i_idx"].max()) + 1)
    user_degree = np.asarray(binary.sum(axis=1)).ravel()
    strata, degree_percentile = _degree_strata(
        user_degree, n_bins=shuffle_degree_bins
    )
    shuffled = clv_percentile.copy()
    rng = np.random.default_rng(shuffle_seed)
    for stratum in np.unique(strata[strata >= 0]):
        index = np.flatnonzero(strata == stratum)
        if len(index) < 2:
            continue
        source = rng.permutation(index)
        if np.array_equal(source, index):
            source = np.roll(source, 1)
        shuffled[index] = clv_percentile[source]
    mean_gate = float(clv_percentile[valid].mean()) if valid.any() else 0.0
    constant = np.zeros(n_users, dtype=np.float64)
    constant[valid] = mean_gate
    return HistoricalCLVGates(
        n_hat=n_hat,
        v_hat=v_hat,
        clv_proxy=clv_proxy,
        clv_percentile=clv_percentile,
        clv_shuffle_percentile=shuffled,
        constant_gate=constant,
        degree_percentile=degree_percentile,
        degree_stratum=strata,
        diagnostics={
            "n_hat": _stats(n_hat[valid]),
            "v_hat": _stats(v_hat[valid]),
            "clv_proxy": _stats(clv_proxy[valid]),
            "clv_percentile": _stats(clv_percentile[valid]),
            "shuffle_preserves_gate_multiset": bool(
                np.allclose(np.sort(shuffled[valid]), np.sort(clv_percentile[valid]))
            ),
            "shuffle_changed_user_share": float(
                np.not_equal(shuffled[valid], clv_percentile[valid]).mean()
            ),
            "degree_bins": int(shuffle_degree_bins),
        },
    )


def top_candidate_items(
    user_neighbor: sparse.spmatrix,
    train: pd.DataFrame,
    n_users: int,
    n_items: int,
    *,
    candidate_count: int,
) -> list[np.ndarray]:
    if candidate_count <= 0:
        raise ValueError("candidate_count must be positive")
    binary = _binary_matrix(train, n_users, n_items)
    score = (user_neighbor.tocsr() @ binary).tocsr()
    result: list[np.ndarray] = []
    for user in range(n_users):
        start, end = score.indptr[user], score.indptr[user + 1]
        items = score.indices[start:end]
        values = score.data[start:end]
        own_start, own_end = binary.indptr[user], binary.indptr[user + 1]
        own = binary.indices[own_start:own_end]
        keep = (values > 0) & ~np.isin(items, own, assume_unique=False)
        items, values = items[keep], values[keep]
        if len(items):
            order = np.lexsort((items, -values))[:candidate_count]
            items = items[order]
        result.append(items.astype(np.int32, copy=False))
    return result

