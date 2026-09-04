"""Train-only CLV-conditioned neighbor selection inside a TF-IDF taste shortlist."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse

from clv_m3_tfidf_neighbor_graph import (
    _binary_matrix,
    _degree_strata,
    _midrank_percentile,
    _stats,
)


PREFERENCE_RELATION = "preference_relation"
ACTUAL_CLV = "actual_clv"
CLV_SHUFFLE = "clv_shuffle"
DEGREE_RELATION = "degree_relation"
ARM_ORDER = (PREFERENCE_RELATION, ACTUAL_CLV, CLV_SHUFFLE, DEGREE_RELATION)


@dataclass(frozen=True)
class HistoricalCLVTasteFeatures:
    n_hat: np.ndarray
    v_hat: np.ndarray
    clv_proxy: np.ndarray
    q_clv: np.ndarray
    q_n: np.ndarray
    q_v: np.ndarray
    composition_coordinate: np.ndarray
    shuffled_q_clv: np.ndarray
    shuffled_q_n: np.ndarray
    shuffled_q_v: np.ndarray
    shuffled_composition_coordinate: np.ndarray
    degree_percentile: np.ndarray
    degree_stratum: np.ndarray
    reliability: np.ndarray
    valid: np.ndarray
    shuffle_source_user: np.ndarray
    diagnostics: dict


@dataclass(frozen=True)
class CLVTasteNeighborGraph:
    binary_user_item: sparse.csr_matrix
    preference_candidates: sparse.csr_matrix
    operators: dict[str, sparse.csr_matrix]
    features: HistoricalCLVTasteFeatures
    diagnostics: dict


def _row_topk(matrix: sparse.spmatrix, *, top_k: int) -> sparse.csr_matrix:
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
        rows.extend([user] * len(index))
        cols.extend(index.tolist())
        values.extend(score.tolist())
    return sparse.csr_matrix(
        (values, (rows, cols)), shape=matrix.shape, dtype=np.float64
    )


def build_preference_candidates(
    train: pd.DataFrame,
    n_users: int,
    n_items: int,
    *,
    candidate_neighbors: int = 100,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, dict]:
    """Return binary purchases and positive TF-IDF cosine Top-K user neighbors."""
    binary = _binary_matrix(train, n_users, n_items)
    item_degree = np.asarray(binary.sum(axis=0)).ravel()
    idf = np.log((n_users + 1.0) / (item_degree + 1.0))
    profile = binary.multiply(idf).tocsr()
    norms = np.sqrt(np.asarray(profile.multiply(profile).sum(axis=1)).ravel())
    inverse = np.zeros_like(norms)
    inverse[norms > 0] = 1.0 / norms[norms > 0]
    normalized = sparse.diags(inverse) @ profile
    similarity = (normalized @ normalized.T).tocsr()
    similarity.setdiag(0.0)
    similarity.eliminate_zeros()
    candidates = _row_topk(similarity, top_k=candidate_neighbors)
    counts = np.diff(candidates.indptr)
    return binary, candidates, {
        "definition": "binary-purchase TF-IDF cosine Top-K taste shortlist",
        "candidate_neighbors": int(candidate_neighbors),
        "eligible_user_count": int((counts > 0).sum()),
        "eligible_user_share": float((counts > 0).mean()) if len(counts) else 0.0,
        "candidate_count_mean_eligible": (
            float(counts[counts > 0].mean()) if np.any(counts > 0) else 0.0
        ),
        "candidate_count_max": int(counts.max(initial=0)),
        "self_edge_count": int(np.count_nonzero(candidates.diagonal())),
    }


def _shuffle_source_by_stratum(
    strata: np.ndarray, *, seed: int
) -> np.ndarray:
    source = np.arange(len(strata), dtype=np.int64)
    rng = np.random.default_rng(seed)
    for stratum in np.unique(strata[strata >= 0]):
        members = np.flatnonzero(strata == stratum)
        if len(members) < 2:
            continue
        target_order = rng.permutation(members)
        source[target_order] = np.roll(target_order, 1)
    return source


def _tuple_multiset_preserved(
    strata: np.ndarray,
    actual: np.ndarray,
    shuffled: np.ndarray,
) -> bool:
    for stratum in np.unique(strata[strata >= 0]):
        index = np.flatnonzero(strata == stratum)
        left = sorted(map(tuple, actual[index].tolist()))
        right = sorted(map(tuple, shuffled[index].tolist()))
        if left != right:
            return False
    return True


def build_historical_clv_features(
    train: pd.DataFrame,
    n_users: int,
    n_items: int,
    *,
    reliability_kappa: float = 5.0,
    degree_bins: int = 10,
    shuffle_seed: int = 42,
) -> HistoricalCLVTasteFeatures:
    if reliability_kappa <= 0:
        raise ValueError("reliability_kappa must be positive")
    if "v" not in train:
        raise ValueError("train must contain transaction amount column 'v'")
    basket_column = "b_raw" if "b_raw" in train.columns else "t"
    baskets = (
        train.groupby(["u_idx", basket_column], sort=False)["v"]
        .sum()
        .rename("basket_value")
    )
    by_user = baskets.groupby(level="u_idx", sort=True)
    summary = pd.DataFrame({"n_hat": by_user.size(), "v_hat": by_user.mean()})
    summary["clv_proxy"] = summary["n_hat"] * summary["v_hat"]

    n_hat = np.full(n_users, np.nan, dtype=np.float64)
    v_hat = np.full(n_users, np.nan, dtype=np.float64)
    clv_proxy = np.full(n_users, np.nan, dtype=np.float64)
    users = summary.index.to_numpy(np.int64)
    if len(users) and (users.min() < 0 or users.max() >= n_users):
        raise ValueError("train user indices exceed n_users")
    n_hat[users] = summary["n_hat"].to_numpy(np.float64)
    v_hat[users] = summary["v_hat"].to_numpy(np.float64)
    clv_proxy[users] = summary["clv_proxy"].to_numpy(np.float64)
    valid = np.isfinite(clv_proxy)
    q_clv = _midrank_percentile(clv_proxy, valid)
    q_n = _midrank_percentile(n_hat, valid)
    q_v = _midrank_percentile(v_hat, valid)
    composition = np.zeros(n_users, dtype=np.float64)
    composition[valid] = (q_n[valid] - q_v[valid] + 1.0) / 2.0

    binary = _binary_matrix(train, n_users, n_items)
    degree = np.asarray(binary.sum(axis=1)).ravel()
    strata, degree_percentile = _degree_strata(degree, n_bins=degree_bins)
    reliability = degree / (degree + float(reliability_kappa))

    shuffle_source = _shuffle_source_by_stratum(strata, seed=shuffle_seed)
    shuffled_q_clv = q_clv[shuffle_source]
    shuffled_q_n = q_n[shuffle_source]
    shuffled_q_v = q_v[shuffle_source]
    shuffled_composition = composition[shuffle_source]
    actual_tuple = np.column_stack([q_clv, q_n, q_v, composition])
    shuffled_tuple = np.column_stack(
        [shuffled_q_clv, shuffled_q_n, shuffled_q_v, shuffled_composition]
    )
    active = strata >= 0
    diagnostics = {
        "historical_clv_proxy": "N_hat * V_hat",
        "n_hat": _stats(n_hat[valid]),
        "v_hat": _stats(v_hat[valid]),
        "clv_proxy": _stats(clv_proxy[valid]),
        "q_clv": _stats(q_clv[valid]),
        "q_n": _stats(q_n[valid]),
        "q_v": _stats(q_v[valid]),
        "composition_coordinate": _stats(composition[valid]),
        "reliability": _stats(reliability[active]),
        "reliability_kappa": float(reliability_kappa),
        "degree_bins": int(degree_bins),
        "shuffle_seed": int(shuffle_seed),
        "shuffle_preserves_tuple_multiset_by_degree_stratum": (
            _tuple_multiset_preserved(strata, actual_tuple, shuffled_tuple)
        ),
        "shuffle_source_changed_user_share": (
            float(np.not_equal(shuffle_source[active], np.flatnonzero(active)).mean())
            if active.any()
            else 0.0
        ),
    }
    return HistoricalCLVTasteFeatures(
        n_hat=n_hat,
        v_hat=v_hat,
        clv_proxy=clv_proxy,
        q_clv=q_clv,
        q_n=q_n,
        q_v=q_v,
        composition_coordinate=composition,
        shuffled_q_clv=shuffled_q_clv,
        shuffled_q_n=shuffled_q_n,
        shuffled_q_v=shuffled_q_v,
        shuffled_composition_coordinate=shuffled_composition,
        degree_percentile=degree_percentile,
        degree_stratum=strata,
        reliability=reliability,
        valid=valid,
        shuffle_source_user=shuffle_source,
        diagnostics=diagnostics,
    )


def _gower_similarity(
    left_level: float,
    right_level: np.ndarray,
    left_composition: float,
    right_composition: np.ndarray,
) -> np.ndarray:
    return np.clip(
        1.0
        - (
            np.abs(left_level - right_level)
            + np.abs(left_composition - right_composition)
        )
        / 2.0,
        0.0,
        1.0,
    )


def _operator_from_affinity_rows(
    preference_candidates: sparse.csr_matrix,
    affinity,
    *,
    final_neighbors: int,
) -> sparse.csr_matrix:
    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    for user in range(preference_candidates.shape[0]):
        start = preference_candidates.indptr[user]
        end = preference_candidates.indptr[user + 1]
        neighbors = preference_candidates.indices[start:end]
        preference = preference_candidates.data[start:end]
        if not len(neighbors):
            continue
        scores = np.asarray(affinity(user, neighbors, preference), dtype=np.float64)
        valid = np.isfinite(scores) & (scores > 0)
        neighbors, scores = neighbors[valid], scores[valid]
        if not len(neighbors):
            continue
        order = np.lexsort((neighbors, -scores))[:final_neighbors]
        neighbors, scores = neighbors[order], scores[order]
        total = float(scores.sum())
        if total <= 0:
            continue
        rows.extend([user] * len(neighbors))
        cols.extend(neighbors.tolist())
        values.extend((scores / total).tolist())
    return sparse.csr_matrix(
        (values, (rows, cols)),
        shape=preference_candidates.shape,
        dtype=np.float64,
    )


def _operator_diagnostics(operator: sparse.csr_matrix) -> dict:
    counts = np.diff(operator.indptr)
    row_mass = np.asarray(operator.sum(axis=1)).ravel()
    eligible = counts > 0
    return {
        "eligible_user_count": int(eligible.sum()),
        "eligible_user_share": float(eligible.mean()) if len(eligible) else 0.0,
        "neighbor_count_mean_eligible": (
            float(counts[eligible].mean()) if eligible.any() else 0.0
        ),
        "neighbor_count_max": int(counts.max(initial=0)),
        "row_mass_max_error": (
            float(np.abs(row_mass[eligible] - 1.0).max())
            if eligible.any()
            else 0.0
        ),
        "self_edge_count": int(np.count_nonzero(operator.diagonal())),
    }


def neighbor_set_change(
    left: sparse.csr_matrix, right: sparse.csr_matrix
) -> dict[str, float]:
    if left.shape != right.shape:
        raise ValueError("neighbor operators must have the same shape")
    changed: list[bool] = []
    jaccard: list[float] = []
    for user in range(left.shape[0]):
        left_set = set(left[user].indices.tolist())
        right_set = set(right[user].indices.tolist())
        if not left_set and not right_set:
            continue
        changed.append(left_set != right_set)
        union = left_set | right_set
        jaccard.append(len(left_set & right_set) / len(union) if union else 1.0)
    return {
        "eligible_user_count": len(changed),
        "set_changed_user_share": float(np.mean(changed)) if changed else 0.0,
        "mean_jaccard": float(np.mean(jaccard)) if jaccard else 1.0,
    }


def _support_subset(
    operator: sparse.csr_matrix, candidates: sparse.csr_matrix
) -> bool:
    op_row, op_col = operator.nonzero()
    if not len(op_row):
        return True
    return bool(np.all(np.asarray(candidates[op_row, op_col]).ravel() > 0))


def build_neighbor_operators(
    preference_candidates: sparse.spmatrix,
    *,
    q_clv: np.ndarray,
    composition_coordinate: np.ndarray,
    shuffled_q_clv: np.ndarray,
    shuffled_composition_coordinate: np.ndarray,
    degree_percentile: np.ndarray,
    reliability: np.ndarray,
    final_neighbors: int = 20,
) -> tuple[dict[str, sparse.csr_matrix], dict]:
    if final_neighbors <= 0:
        raise ValueError("final_neighbors must be positive")
    candidates = preference_candidates.tocsr(copy=True)
    n_users = candidates.shape[0]
    arrays = {
        "q_clv": q_clv,
        "composition_coordinate": composition_coordinate,
        "shuffled_q_clv": shuffled_q_clv,
        "shuffled_composition_coordinate": shuffled_composition_coordinate,
        "degree_percentile": degree_percentile,
        "reliability": reliability,
    }
    arrays = {name: np.asarray(value, dtype=np.float64) for name, value in arrays.items()}
    for name, value in arrays.items():
        if value.shape != (n_users,):
            raise ValueError(f"{name} must have shape ({n_users},)")
    if np.any((arrays["reliability"] < 0) | (arrays["reliability"] >= 1)):
        raise ValueError("reliability must be in [0, 1)")

    def preference_affinity(user, neighbors, preference):
        return preference

    def conditioned_affinity(user, neighbors, preference, *, shuffled: bool):
        level = arrays["shuffled_q_clv"] if shuffled else arrays["q_clv"]
        composition = (
            arrays["shuffled_composition_coordinate"]
            if shuffled
            else arrays["composition_coordinate"]
        )
        similarity = _gower_similarity(
            level[user], level[neighbors], composition[user], composition[neighbors]
        )
        reliability_product = arrays["reliability"][user] * arrays["reliability"][neighbors]
        return preference * (
            (1.0 - reliability_product) + reliability_product * similarity
        )

    def degree_affinity(user, neighbors, preference):
        similarity = 1.0 - np.abs(
            arrays["degree_percentile"][user]
            - arrays["degree_percentile"][neighbors]
        )
        reliability_product = arrays["reliability"][user] * arrays["reliability"][neighbors]
        return preference * (
            (1.0 - reliability_product) + reliability_product * similarity
        )

    operators = {
        PREFERENCE_RELATION: _operator_from_affinity_rows(
            candidates, preference_affinity, final_neighbors=final_neighbors
        ),
        ACTUAL_CLV: _operator_from_affinity_rows(
            candidates,
            lambda user, neighbors, preference: conditioned_affinity(
                user, neighbors, preference, shuffled=False
            ),
            final_neighbors=final_neighbors,
        ),
        CLV_SHUFFLE: _operator_from_affinity_rows(
            candidates,
            lambda user, neighbors, preference: conditioned_affinity(
                user, neighbors, preference, shuffled=True
            ),
            final_neighbors=final_neighbors,
        ),
        DEGREE_RELATION: _operator_from_affinity_rows(
            candidates, degree_affinity, final_neighbors=final_neighbors
        ),
    }
    counts = {arm: np.diff(operator.indptr) for arm, operator in operators.items()}
    masses = {
        arm: np.asarray(operator.sum(axis=1)).ravel()
        for arm, operator in operators.items()
    }
    reference_count = counts[PREFERENCE_RELATION]
    reference_mass = masses[PREFERENCE_RELATION]
    same_count = all(
        np.array_equal(value, reference_count) for value in counts.values()
    )
    same_mass = all(
        np.allclose(value, reference_mass, atol=1e-12, rtol=0)
        for value in masses.values()
    )
    support_ok = all(
        _support_subset(operator, candidates) for operator in operators.values()
    )
    arm_diagnostics = {
        arm: _operator_diagnostics(operator) for arm, operator in operators.items()
    }
    diagnostics = {
        "candidate_neighbors": int(
            np.diff(candidates.indptr).max(initial=0)
        ),
        "final_neighbors": int(final_neighbors),
        "same_neighbor_count_all_arms": bool(same_count),
        "same_row_mass_all_arms": bool(same_mass),
        "all_arms_use_preference_candidate_support": bool(support_ok),
        "actual_vs_preference": neighbor_set_change(
            operators[ACTUAL_CLV], operators[PREFERENCE_RELATION]
        ),
        "actual_vs_shuffle": neighbor_set_change(
            operators[ACTUAL_CLV], operators[CLV_SHUFFLE]
        ),
        "actual_vs_degree": neighbor_set_change(
            operators[ACTUAL_CLV], operators[DEGREE_RELATION]
        ),
        "arms": arm_diagnostics,
    }
    diagnostics["quality_passed"] = bool(
        same_count
        and same_mass
        and support_ok
        and all(value["self_edge_count"] == 0 for value in arm_diagnostics.values())
        and all(value["row_mass_max_error"] <= 1e-10 for value in arm_diagnostics.values())
    )
    return operators, diagnostics


def build_clv_taste_neighbor_graph(
    train: pd.DataFrame,
    n_users: int,
    n_items: int,
    *,
    candidate_neighbors: int = 100,
    final_neighbors: int = 20,
    reliability_kappa: float = 5.0,
    degree_bins: int = 10,
    shuffle_seed: int = 42,
) -> CLVTasteNeighborGraph:
    if final_neighbors > candidate_neighbors:
        raise ValueError("final_neighbors cannot exceed candidate_neighbors")
    binary, candidates, candidate_diagnostics = build_preference_candidates(
        train,
        n_users,
        n_items,
        candidate_neighbors=candidate_neighbors,
    )
    features = build_historical_clv_features(
        train,
        n_users,
        n_items,
        reliability_kappa=reliability_kappa,
        degree_bins=degree_bins,
        shuffle_seed=shuffle_seed,
    )
    operators, relation_diagnostics = build_neighbor_operators(
        candidates,
        q_clv=features.q_clv,
        composition_coordinate=features.composition_coordinate,
        shuffled_q_clv=features.shuffled_q_clv,
        shuffled_composition_coordinate=features.shuffled_composition_coordinate,
        degree_percentile=features.degree_percentile,
        reliability=features.reliability,
        final_neighbors=final_neighbors,
    )
    diagnostics = {
        "preference_candidates": candidate_diagnostics,
        "historical_clv": features.diagnostics,
        "neighbor_relations": relation_diagnostics,
    }
    diagnostics["quality_passed"] = bool(
        candidate_diagnostics["self_edge_count"] == 0
        and features.diagnostics[
            "shuffle_preserves_tuple_multiset_by_degree_stratum"
        ]
        and relation_diagnostics["quality_passed"]
    )
    return CLVTasteNeighborGraph(
        binary_user_item=binary,
        preference_candidates=candidates,
        operators=operators,
        features=features,
        diagnostics=diagnostics,
    )


__all__ = [
    "PREFERENCE_RELATION",
    "ACTUAL_CLV",
    "CLV_SHUFFLE",
    "DEGREE_RELATION",
    "ARM_ORDER",
    "HistoricalCLVTasteFeatures",
    "CLVTasteNeighborGraph",
    "build_preference_candidates",
    "build_historical_clv_features",
    "build_neighbor_operators",
    "build_clv_taste_neighbor_graph",
    "neighbor_set_change",
]
