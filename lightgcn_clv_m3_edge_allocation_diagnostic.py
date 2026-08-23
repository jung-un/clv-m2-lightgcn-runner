"""Descriptive diagnosis for the failed historical M3 edge-allocation pilot.

This module reloads the already evaluated seed-42 M1 and M3 checkpoints.  It
does not train, select a checkpoint, or construct the final test interval.
Its purpose is to distinguish weak intervention, misdirected edge signal,
insufficient user-item reachability, and shallow-rank ordering failure.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import rankdata, spearmanr
import torch

from clv_run_state import file_sha256
import lightgcn_clv_axis_specific_behavior_diagnostic as common
import lightgcn_clv_m3_edge_allocation_backtest as pilot
import lightgcn_clv_v3 as v3


CODE_VERSION = "m3-clv-edge-allocation-historical-diagnostic-v1"
M1_ID, M3_ID = pilot.MODEL_IDS
MODELS = pilot.MODEL_IDS
RANK_LIMIT = 100
CONNECTIVITY_COLUMNS = (
    "shared_buyer_reach",
    "co_basket_reach",
    "forward_transition_reach",
    "history_category_share",
)


@dataclass(frozen=True)
class M3EdgeAllocationDiagnosticConfig:
    out_dir: str = ""
    run_json: str = ""
    score_batch_size: int = 64
    rank_limit: int = RANK_LIMIT
    max_representative_users: int = 24


def configure_m3_edge_allocation_diagnostic(
    **overrides,
) -> M3EdgeAllocationDiagnosticConfig:
    defaults = {"out_dir": pilot.configure_m3_edge_allocation_backtest().out_dir}
    return validate_config(
        M3EdgeAllocationDiagnosticConfig(**(defaults | overrides))
    )


def validate_config(
    cfg: M3EdgeAllocationDiagnosticConfig,
) -> M3EdgeAllocationDiagnosticConfig:
    if not cfg.out_dir or "m3_clv_edge_allocation_historical" not in cfg.out_dir:
        raise ValueError("historical M3 edge-allocation result directory is required")
    if cfg.rank_limit != 100:
        raise ValueError("diagnostic rank limit is fixed at 100")
    if cfg.score_batch_size <= 0 or cfg.max_representative_users <= 0:
        raise ValueError("diagnostic sizes must be positive")
    return cfg


def preflight_summary(cfg: M3EdgeAllocationDiagnosticConfig) -> dict:
    cfg = validate_config(cfg)
    return {
        "code_version": CODE_VERSION,
        "analysis_type": "descriptive post-hoc historical diagnostic",
        "source_models": list(MODELS),
        "source_split": "DAY 1--683 train; DAY 684--690 evaluation",
        "training": False,
        "checkpoint_selection": False,
        "final_test_constructed": False,
        "holdout_constructed": False,
        "rank_limit": cfg.rank_limit,
        "questions": [
            "was the edge intervention materially active?",
            "which truths entered or left Top-10/20/50/100?",
            "did M3 promote items matching its CLV-allocation signal?",
            "are truths better connected through buyers, baskets, transitions, or category?",
        ],
        "interpretation": (
            "hypothesis generation only; a changed model must be assessed on a new "
            "predeclared interval or independent data"
        ),
        "out_dir": cfg.out_dir,
    }


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(np.asarray(numerator, dtype=float)),
        where=np.asarray(denominator) != 0,
    )


def _edge_intervention_table(
    *,
    edge_users: np.ndarray,
    edge_items: np.ndarray,
    base: np.ndarray,
    adjusted: np.ndarray,
    relationship_share: np.ndarray,
    allocation: np.ndarray,
) -> pd.DataFrame:
    base = np.asarray(base, dtype=float)
    adjusted = np.asarray(adjusted, dtype=float)
    return pd.DataFrame(
        {
            "user_idx": np.asarray(edge_users, dtype=int),
            "item_idx": np.asarray(edge_items, dtype=int),
            "base_coefficient": base,
            "adjusted_coefficient": adjusted,
            "coefficient_ratio": _safe_divide(adjusted, base),
            "absolute_coefficient_change": np.abs(adjusted - base),
            "relationship_share": np.asarray(relationship_share, dtype=float),
            "edge_clv_allocation": np.asarray(allocation, dtype=float),
        }
    )


def _intervention_summary(edges: pd.DataFrame) -> pd.DataFrame:
    rows = []
    changed = ~np.isclose(edges.coefficient_ratio.to_numpy(float), 1.0)
    rows.append(
        {
            "grain": "edge",
            "group_id": "all_edges",
            "n_entities": len(edges),
            "share_edges_changed": float(changed.mean()),
            "median_absolute_ratio_shift": float(
                np.median(np.abs(edges.coefficient_ratio - 1.0))
            ),
            "median_within_entity_ratio_std": np.nan,
            "median_kish_effective_ratio": np.nan,
        }
    )
    for grain, key in (("user", "user_idx"), ("item", "item_idx")):
        grouped = edges.groupby(key, sort=False)
        ratio_std = grouped.coefficient_ratio.std(ddof=0)
        kish = grouped.adjusted_coefficient.agg(
            lambda values: (
                float(values.sum() ** 2 / (len(values) * np.square(values).sum()))
                if np.square(values).sum() > 0
                else np.nan
            )
        )
        rows.append(
            {
                "grain": grain,
                "group_id": f"all_{grain}s",
                "n_entities": int(grouped.ngroups),
                "share_edges_changed": float(changed.mean()),
                "median_absolute_ratio_shift": float(
                    np.median(np.abs(edges.coefficient_ratio - 1.0))
                ),
                "median_within_entity_ratio_std": float(ratio_std.median()),
                "median_kish_effective_ratio": float(kish.median()),
            }
        )
    return pd.DataFrame(rows)


def _basket_codes(train: pd.DataFrame) -> np.ndarray:
    basket_column = "b_raw" if "b_raw" in train else "t"
    keys = pd.MultiIndex.from_frame(train[["u_idx", basket_column]])
    return pd.factorize(keys, sort=False)[0].astype(np.int64)


def _binary_matrix(rows: np.ndarray, cols: np.ndarray, shape: tuple[int, int]):
    matrix = sparse.coo_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=shape
    ).tocsr()
    matrix.data[:] = 1.0
    matrix.eliminate_zeros()
    return matrix


def _transition_matrix(
    train: pd.DataFrame,
    n_items: int,
    target_items: np.ndarray | None = None,
) -> sparse.csr_matrix:
    basket_column = "b_raw" if "b_raw" in train else "t"
    basket_items = (
        train.groupby(["u_idx", basket_column, "t"], sort=False)["i_idx"]
        .agg(lambda values: tuple(np.unique(values.to_numpy(np.int64))))
        .reset_index()
        .sort_values(["u_idx", "t", basket_column], kind="stable")
    )
    target_set = (
        None if target_items is None else set(map(int, np.asarray(target_items)))
    )
    row_parts, col_parts = [], []
    for _, user_baskets in basket_items.groupby("u_idx", sort=False):
        arrays = [np.asarray(values, dtype=np.int64) for values in user_baskets.i_idx]
        for source, target in zip(arrays[:-1], arrays[1:]):
            if target_set is not None:
                target = np.asarray(
                    [item for item in target if int(item) in target_set],
                    dtype=np.int64,
                )
            if len(source) and len(target):
                row_parts.append(np.repeat(source, len(target)))
                col_parts.append(np.tile(target, len(source)))
    if not row_parts:
        return sparse.csr_matrix((n_items, n_items), dtype=np.float32)
    rows = np.concatenate(row_parts)
    cols = np.concatenate(col_parts)
    matrix = sparse.coo_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(n_items, n_items),
    ).tocsr()
    matrix.sum_duplicates()
    return matrix


def _candidate_connectivity(
    *,
    train: pd.DataFrame,
    candidates: pd.DataFrame,
    item_categories: np.ndarray,
    n_users: int,
    n_items: int,
) -> pd.DataFrame:
    required = {"user_idx", "item_idx", "role"}
    if not required.issubset(candidates.columns):
        raise ValueError(f"candidates require {sorted(required)}")
    unique_edges = train[["u_idx", "i_idx"]].drop_duplicates()
    user_item = _binary_matrix(
        unique_edges.u_idx.to_numpy(np.int64),
        unique_edges.i_idx.to_numpy(np.int64),
        (n_users, n_items),
    )
    basket_codes = _basket_codes(train)
    basket_item = _binary_matrix(
        basket_codes,
        train.i_idx.to_numpy(np.int64),
        (int(basket_codes.max()) + 1, n_items),
    )
    transitions = _transition_matrix(
        train, n_items, candidates.item_idx.unique()
    )
    item_buyer_count = np.asarray(user_item.sum(axis=0)).ravel()
    item_basket_count = np.asarray(basket_item.sum(axis=0)).ravel()
    transition_in = np.asarray(transitions.sum(axis=0)).ravel()
    categories = np.asarray(item_categories, dtype=object)

    records = []
    for user_idx, group in candidates.groupby("user_idx", sort=False):
        history = user_item.getrow(int(user_idx)).indices
        history_set = set(map(int, history))
        other_buyers = np.asarray(user_item[:, history].sum(axis=1)).ravel() > 0
        buyer_overlap = np.asarray(user_item[other_buyers].sum(axis=0)).ravel()
        related_baskets = np.asarray(basket_item[:, history].sum(axis=1)).ravel() > 0
        basket_overlap = np.asarray(basket_item[related_baskets].sum(axis=0)).ravel()
        transition_counts = np.asarray(transitions[history].sum(axis=0)).ravel()
        history_categories = categories[history]
        category_counts = pd.Series(history_categories).value_counts()
        history_size = max(len(history), 1)
        for row in group.itertuples(index=False):
            item = int(row.item_idx)
            record = row._asdict()
            record.update(
                {
                    "history_item_count": len(history),
                    "candidate_seen_in_train": item in history_set,
                    "shared_buyer_reach": float(
                        buyer_overlap[item] / item_buyer_count[item]
                        if item_buyer_count[item] > 0 else 0.0
                    ),
                    "co_basket_reach": float(
                        basket_overlap[item] / item_basket_count[item]
                        if item_basket_count[item] > 0 else 0.0
                    ),
                    "forward_transition_reach": float(
                        transition_counts[item] / transition_in[item]
                        if transition_in[item] > 0 else 0.0
                    ),
                    "history_category_share": float(
                        category_counts.get(categories[item], 0) / history_size
                    ),
                }
            )
            records.append(record)
    return pd.DataFrame(records)


def _structure_evidence(connectivity: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for signal in CONNECTIVITY_COLUMNS:
        means = connectivity.groupby("role", sort=False)[signal].mean()
        truth = float(means.get("truth", np.nan))
        m1_only = float(means.get("m1_only", np.nan))
        m3_only = float(means.get("m3_only", np.nan))
        gap = truth - m3_only
        rows.append(
            {
                "signal": signal,
                "truth_mean": truth,
                "m1_only_mean": m1_only,
                "m3_only_mean": m3_only,
                "truth_minus_m3_only": gap,
                "direction": (
                    "truth_stronger" if np.isfinite(gap) and gap > 0
                    else "m3_only_stronger" if np.isfinite(gap) and gap < 0
                    else "indeterminate"
                ),
            }
        )
    return pd.DataFrame(rows)


def _rank_movement_summary(truth: pd.DataFrame) -> pd.DataFrame:
    rows = []
    m1 = truth.m1_rank_capped_101.to_numpy(int)
    m3 = truth.m3_rank_capped_101.to_numpy(int)
    for cutoff in (10, 20, 50, 100):
        rows.append(
            {
                "cutoff": cutoff,
                "entered": int(((m1 > cutoff) & (m3 <= cutoff)).sum()),
                "left": int(((m1 <= cutoff) & (m3 > cutoff)).sum()),
                "net_entries": int(((m1 > cutoff) & (m3 <= cutoff)).sum() - ((m1 <= cutoff) & (m3 > cutoff)).sum()),
                "improved_within_cutoff": int(((m1 <= cutoff) & (m3 < m1)).sum()),
                "degraded_within_cutoff": int(((m3 <= cutoff) & (m3 > m1)).sum()),
            }
        )
    return pd.DataFrame(rows)


def _candidate_rows(
    truth: pd.DataFrame,
    recommendations: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    truth_base = truth[["user_idx", "item_idx", "test_purchase_amount"]].copy()
    for cutoff in (10, 50):
        rows.append(truth_base.assign(role="truth", cutoff=cutoff))
        selected = recommendations[recommendations["rank"] <= cutoff]
        for user_idx, group in selected.groupby("user_idx", sort=False):
            m1 = group[group.model_id.eq(M1_ID)].sort_values("rank")
            m3 = group[group.model_id.eq(M3_ID)].sort_values("rank")
            m1_set = set(map(int, m1.item_idx))
            m3_set = set(map(int, m3.item_idx))
            for role, frame, excluded in (
                ("m1_only", m1, m3_set),
                ("m3_only", m3, m1_set),
            ):
                chosen = frame[~frame.item_idx.isin(excluded)][
                    ["user_idx", "item_idx", "rank"]
                ].copy()
                if len(chosen):
                    chosen["role"] = role
                    chosen["cutoff"] = cutoff
                    chosen["test_purchase_amount"] = 0.0
                    rows.append(chosen)
    return pd.concat(rows, ignore_index=True, sort=False)


def _find_run_json(
    cfg: M3EdgeAllocationDiagnosticConfig,
) -> tuple[Path, dict, str]:
    root = Path(cfg.out_dir)
    candidates = (
        [Path(cfg.run_json)]
        if cfg.run_json
        else sorted(
            root.glob("m3_clv_edge_allocation_backtest_????????????.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    )
    if not candidates or not candidates[0].exists():
        raise FileNotFoundError("historical M3 pilot result JSON was not found")
    path = candidates[0]
    match = re.fullmatch(
        r"m3_clv_edge_allocation_backtest_([0-9a-f]{12})", path.stem
    )
    if match is None:
        raise ValueError(f"unexpected historical M3 result name: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("code_version") != pilot.CODE_VERSION:
        raise RuntimeError("source result and historical runner versions differ")
    preflight = payload.get("preflight", {})
    split = preflight.get("historical_development_split", {})
    if split.get("final_test_constructed") is not False:
        raise RuntimeError("source must not construct the final test")
    if split.get("holdout_constructed") is not False:
        raise RuntimeError("source must not construct holdout")
    return path, payload, match.group(1)


def _prepare_source(
    cfg: M3EdgeAllocationDiagnosticConfig,
    payload: dict,
) -> tuple[dict, pilot.M3EdgeAllocationBacktestConfig]:
    stored = dict(payload["config"])
    stored["out_dir"] = cfg.out_dir
    run_cfg = pilot.validate_config(pilot.M3EdgeAllocationBacktestConfig(**stored))
    prepared = pilot._prepare(run_cfg)
    expected_manifest = pilot.moe.manifest_hash(payload["input_manifest"])
    if prepared["input_hash"] != expected_manifest:
        raise RuntimeError("source data manifest differs from the completed pilot")
    source_revision = payload["source_revision"]
    prepared["revision"] = source_revision
    prepared["config_hash"] = pilot._config_hash(
        run_cfg, prepared["input_hash"], source_revision
    )
    return prepared, run_cfg


def _load_model(
    prepared: dict,
    run_cfg: pilot.M3EdgeAllocationBacktestConfig,
    model_id: str,
):
    paths = pilot._arm_paths(prepared, run_cfg, model_id)
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(f"completed pilot arm is missing: {path}")
    result = json.loads(paths["result"].read_text(encoding="utf-8"))
    if result.get("evaluation_count") != 1:
        raise RuntimeError(f"historical evaluation count is not one: {paths['result']}")
    if result.get("checkpoint_sha256") != file_sha256(paths["checkpoint"]):
        raise RuntimeError(f"checkpoint hash mismatch: {paths['checkpoint']}")
    model, _ = pilot._build_model(prepared, run_cfg, model_id)
    try:
        blob = torch.load(
            paths["checkpoint"], map_location=v3.DEVICE, weights_only=False
        )
    except TypeError:
        blob = torch.load(paths["checkpoint"], map_location=v3.DEVICE)
    if blob.get("input_hash") != prepared["input_hash"]:
        raise RuntimeError(f"checkpoint input hash mismatch: {paths['checkpoint']}")
    if blob.get("model_id") != model_id or int(blob.get("seed")) != run_cfg.seed:
        raise RuntimeError(f"checkpoint identity mismatch: {paths['checkpoint']}")
    model.load_state_dict(blob["state"], strict=True)
    model.eval()
    return model, result, paths


def _percentile(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    output = np.full(len(values), np.nan)
    valid = np.isfinite(values)
    if valid.any():
        output[valid] = (rankdata(values[valid], method="average") - 0.5) / valid.sum()
    return output


def _user_cohorts(prepared: dict) -> pd.DataFrame:
    graph = prepared["graph"]
    q_n = _percentile(graph.n_hat)
    q_v = _percentile(graph.v_hat)
    q_clv = _percentile(graph.clv_proxy)
    valid = np.isfinite(q_n) & np.isfinite(q_v) & np.isfinite(q_clv)
    quadrant = np.full(len(q_n), "invalid", dtype=object)
    quadrant[valid & (q_n < 0.5) & (q_v < 0.5)] = "low_n_low_v"
    quadrant[valid & (q_n >= 0.5) & (q_v < 0.5)] = "high_n_low_v"
    quadrant[valid & (q_n < 0.5) & (q_v >= 0.5)] = "low_n_high_v"
    quadrant[valid & (q_n >= 0.5) & (q_v >= 0.5)] = "high_n_high_v"
    quintile = np.full(len(q_n), "invalid", dtype=object)
    quintile[valid] = [f"Q{min(int(value * 5), 4) + 1}" for value in q_clv[valid]]
    raw = prepared["data"]["train"].groupby("u_idx", sort=False).u_raw.first()
    return pd.DataFrame(
        {
            "user_idx": np.arange(len(q_n), dtype=int),
            "user_id": raw.reindex(np.arange(len(q_n))).astype(object).to_numpy(),
            "n_hat": graph.n_hat,
            "v_hat": graph.v_hat,
            "clv_proxy": graph.clv_proxy,
            "q_n": q_n,
            "q_v": q_v,
            "q_clv": q_clv,
            "nv_quadrant": quadrant.astype(str),
            "clv_quintile": quintile.astype(str),
        }
    )


def _per_user_metrics(
    users: np.ndarray,
    topk: np.ndarray,
    prepared: dict,
    cohorts: pd.DataFrame,
    model_id: str,
) -> pd.DataFrame:
    cache, meta, data = prepared["cache"], prepared["meta"], prepared["data"]
    novelty = -np.log2(meta["pop_prob"] + 1e-12)
    scored = v3.score_topk(
        topk,
        users,
        [10, 20, 50],
        cache.pos_key,
        cache.pos_rev,
        data["n_items"],
        cache.P_arr,
        meta["price_pct"],
        novelty,
        meta["cat"],
        cache.ideal,
    )
    frame = cohorts.set_index("user_idx").loc[users].reset_index()
    frame.insert(0, "seed", 42)
    frame.insert(1, "model_id", model_id)
    frame["truth_item_count"] = cache.P_arr[users]
    for cutoff, values in scored.items():
        for name, array in values.items():
            public = {
                "revenue": "price_purchase_amount_weighted_hit",
                "arp": "mean_recommended_price_percentile",
            }.get(name, name)
            frame[f"{public}@{cutoff}"] = array
    return frame


def _recommendation_rows(
    users: np.ndarray,
    topk: np.ndarray,
    model_id: str,
    cohorts: pd.DataFrame,
    item_traits: pd.DataFrame,
) -> pd.DataFrame:
    user_info = cohorts.set_index("user_idx")
    rows = []
    for position, user in enumerate(users):
        info = user_info.loc[int(user)]
        for rank, item in enumerate(topk[position], start=1):
            rows.append(
                {
                    "seed": 42,
                    "model_id": model_id,
                    "user_idx": int(user),
                    "user_id": info.user_id,
                    "nv_quadrant": info.nv_quadrant,
                    "clv_quintile": info.clv_quintile,
                    "rank": rank,
                    "item_idx": int(item),
                }
            )
    return pd.DataFrame(rows).merge(item_traits, on="item_idx", how="left")


def _truth_rows(
    users: np.ndarray,
    topk_by_model: dict[str, np.ndarray],
    prepared: dict,
    cohorts: pd.DataFrame,
    item_traits: pd.DataFrame,
) -> pd.DataFrame:
    cache = prepared["cache"]
    user_info = cohorts.set_index("user_idx")
    rows = []
    for position, user in enumerate(users):
        ranks = {
            model_id: {int(item): rank + 1 for rank, item in enumerate(topk[position])}
            for model_id, topk in topk_by_model.items()
        }
        info = user_info.loc[int(user)]
        for item, amount in zip(cache.gt[int(user)], cache.rev[int(user)]):
            item = int(item)
            m1_rank = ranks[M1_ID].get(item, 101)
            m3_rank = ranks[M3_ID].get(item, 101)
            rows.append(
                {
                    "seed": 42,
                    "user_idx": int(user),
                    "user_id": info.user_id,
                    "nv_quadrant": info.nv_quadrant,
                    "clv_quintile": info.clv_quintile,
                    "q_n": info.q_n,
                    "q_v": info.q_v,
                    "q_clv": info.q_clv,
                    "item_idx": item,
                    "test_purchase_amount": float(amount),
                    "m1_rank_capped_101": m1_rank,
                    "m3_rank_capped_101": m3_rank,
                    "rank_improvement": m1_rank - m3_rank,
                }
            )
    return pd.DataFrame(rows).merge(item_traits, on="item_idx", how="left")


def _segment_summary(per_user: pd.DataFrame) -> pd.DataFrame:
    metrics = [column for column in per_user if "@" in column]
    rows = []
    for segment in ("nv_quadrant", "clv_quintile"):
        for segment_id, group in per_user.groupby(segment, sort=False):
            m1 = group[group.model_id.eq(M1_ID)].set_index("user_idx")
            m3 = group[group.model_id.eq(M3_ID)].set_index("user_idx")
            users = m1.index.intersection(m3.index)
            for metric in metrics:
                baseline = m1.loc[users, metric].to_numpy(float)
                changed = m3.loc[users, metric].to_numpy(float)
                delta = changed - baseline
                rows.append(
                    {
                        "segment_type": segment,
                        "segment_id": segment_id,
                        "n_users": len(users),
                        "metric": metric,
                        "m1_mean": float(baseline.mean()),
                        "m3_mean": float(changed.mean()),
                        "mean_delta": float(delta.mean()),
                        "relative_change_pct": (
                            float(delta.mean() / baseline.mean() * 100)
                            if abs(baseline.mean()) > 1e-12 else np.nan
                        ),
                        "improved_user_share": float((delta > 0).mean()),
                        "degraded_user_share": float((delta < 0).mean()),
                    }
                )
    return pd.DataFrame(rows)


def _item_mechanism(
    edges: pd.DataFrame,
    recommendations: pd.DataFrame,
    truth: pd.DataFrame,
    item_traits: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    item = edges.groupby("item_idx", sort=False).agg(
        edge_count=("user_idx", "size"),
        mean_coefficient_ratio=("coefficient_ratio", "mean"),
        std_coefficient_ratio=("coefficient_ratio", "std"),
        mean_absolute_coefficient_change=("absolute_coefficient_change", "mean"),
        mean_relationship_share=("relationship_share", "mean"),
        mean_edge_clv_allocation=("edge_clv_allocation", "mean"),
        max_edge_clv_allocation=("edge_clv_allocation", "max"),
    ).reset_index()
    for cutoff in (10, 50):
        selected = recommendations[recommendations["rank"] <= cutoff]
        promotions = np.zeros(len(item_traits), dtype=int)
        demotions = np.zeros(len(item_traits), dtype=int)
        promoted_hits = np.zeros(len(item_traits), dtype=int)
        truth_sets = truth.groupby("user_idx").item_idx.agg(lambda x: set(map(int, x)))
        for user, group in selected.groupby("user_idx", sort=False):
            m1 = set(map(int, group[group.model_id.eq(M1_ID)].item_idx))
            m3 = set(map(int, group[group.model_id.eq(M3_ID)].item_idx))
            user_truth = truth_sets.get(user, set())
            for candidate in m3 - m1:
                promotions[candidate] += 1
                promoted_hits[candidate] += int(candidate in user_truth)
            for candidate in m1 - m3:
                demotions[candidate] += 1
        item[f"top{cutoff}_promotion_count"] = promotions[item.item_idx]
        item[f"top{cutoff}_demotion_count"] = demotions[item.item_idx]
        item[f"top{cutoff}_promoted_hit_count"] = promoted_hits[item.item_idx]
    item = item_traits.merge(item, on="item_idx", how="left")
    numeric = (
        "std_coefficient_ratio",
        "mean_absolute_coefficient_change",
        "mean_relationship_share",
        "mean_edge_clv_allocation",
        "train_user_count",
        "repeat_purchase_share",
        "price_percentile",
    )
    rows = []
    for target in (
        "top10_promotion_count",
        "top10_demotion_count",
        "top10_promoted_hit_count",
        "top50_promotion_count",
    ):
        for feature in numeric:
            valid = item[[target, feature]].dropna()
            value = (
                spearmanr(valid[target], valid[feature]).correlation
                if len(valid) >= 3 and valid[target].nunique() > 1
                and valid[feature].nunique() > 1 else np.nan
            )
            rows.append(
                {
                    "target": target,
                    "feature": feature,
                    "n_items": len(valid),
                    "spearman": float(value) if np.isfinite(value) else np.nan,
                }
            )
    return item, pd.DataFrame(rows)


def _quality_checks(
    users: np.ndarray,
    topk_by_model: dict[str, np.ndarray],
    per_user: pd.DataFrame,
    truth: pd.DataFrame,
    prepared: dict,
    results: dict[str, dict],
) -> pd.DataFrame:
    rows = []
    n_items = prepared["data"]["n_items"]
    train_keys = prepared["data"]["pos_key"]
    metric_names = (
        "recall@10", "ndcg@10", "recall@20", "ndcg@20", "recall@50", "ndcg@50",
        "price_purchase_amount_weighted_hit@10",
        "mean_recommended_price_percentile@10",
    )
    for model_id, topk in topk_by_model.items():
        keys = users[:, None].astype(np.int64) * n_items + topk.astype(np.int64)
        locations = np.clip(np.searchsorted(train_keys, keys), 0, len(train_keys) - 1)
        overlap = int((train_keys[locations] == keys).sum())
        duplicates = int(sum(len(np.unique(row)) != len(row) for row in topk))
        rows.extend(
            [
                {"model_id": model_id, "check": "top100_excludes_train_pairs", "value": overlap, "passed": overlap == 0},
                {"model_id": model_id, "check": "top100_has_no_duplicates", "value": duplicates, "passed": duplicates == 0},
            ]
        )
        observed = per_user[per_user.model_id.eq(model_id)]
        for metric in metric_names:
            error = abs(float(observed[metric].mean()) - float(results[model_id]["metrics"][metric]))
            rows.append(
                {"model_id": model_id, "check": f"recomputed_{metric}", "value": error, "passed": error < 1e-7}
            )
    truth_keys = truth.user_idx.to_numpy(np.int64) * n_items + truth.item_idx.to_numpy(np.int64)
    locations = np.clip(np.searchsorted(train_keys, truth_keys), 0, len(train_keys) - 1)
    overlap = int((train_keys[locations] == truth_keys).sum())
    rows.append({"model_id": "all", "check": "truth_excludes_train_pairs", "value": overlap, "passed": overlap == 0})
    rows.append({"model_id": "all", "check": "historical_train_end_day", "value": float(prepared["data"]["train"].t.max()), "passed": float(prepared["data"]["train"].t.max()) == 683.0})
    return pd.DataFrame(rows)


def _representative_users(
    per_user: pd.DataFrame,
    truth: pd.DataFrame,
    recommendations: pd.DataFrame,
    limit: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    m1 = per_user[per_user.model_id.eq(M1_ID)].set_index("user_idx")
    m3 = per_user[per_user.model_id.eq(M3_ID)].set_index("user_idx")
    users = m1.index.intersection(m3.index)
    summary = m1.loc[users, ["user_id", "nv_quadrant", "clv_quintile", "q_n", "q_v", "q_clv"]].copy()
    for metric in ("recall@10", "ndcg@10", "recall@50"):
        summary[f"m1_{metric}"] = m1.loc[users, metric]
        summary[f"m3_{metric}"] = m3.loc[users, metric]
        summary[f"delta_{metric}"] = m3.loc[users, metric] - m1.loc[users, metric]
    summary = summary.reset_index()
    gains = summary.sort_values(["delta_ndcg@10", "user_idx"], ascending=[False, True]).head(limit // 2)
    losses = summary.sort_values(["delta_ndcg@10", "user_idx"], ascending=[True, True]).head(limit // 2)
    selected = pd.concat([gains.assign(selection="largest_gain"), losses.assign(selection="largest_loss")]).drop_duplicates("user_idx")
    detail = pd.concat(
        [
            selected[["user_idx", "selection"]].merge(truth, on="user_idx", how="left").assign(detail_role="truth"),
            selected[["user_idx", "selection"]].merge(recommendations[recommendations["rank"] <= 10], on="user_idx", how="left").assign(detail_role=lambda x: x.model_id + "_top10"),
        ],
        ignore_index=True,
        sort=False,
    )
    return selected, detail


def _persist(
    cfg: M3EdgeAllocationDiagnosticConfig,
    source_json: Path,
    source_hash: str,
    source_payload: dict,
    frames: dict[str, pd.DataFrame],
    checkpoints: list[dict],
) -> dict[str, str]:
    diagnostic_hash = hashlib.sha256(
        json.dumps(
            {"version": CODE_VERSION, "source": source_hash, "config": asdict(cfg)},
            sort_keys=True,
        ).encode()
    ).hexdigest()[:12]
    root = Path(cfg.out_dir) / "edge_allocation_diagnostic" / diagnostic_hash
    paths = {name: root / f"{name}.csv" for name in frames}
    paths["json"] = root / "diagnostic.json"
    for name, frame in frames.items():
        common._atomic_csv(paths[name], frame)
    quality = frames["quality_checks"]
    payload = {
        "code_version": CODE_VERSION,
        "source_run_json": str(source_json),
        "source_run_hash": source_hash,
        "source_revision": source_payload.get("source_revision"),
        "input_manifest": source_payload.get("input_manifest"),
        "config": asdict(cfg),
        "checkpoints": checkpoints,
        "quality_passed": bool(len(quality) and quality.passed.all()),
        "quality_failed": quality.loc[~quality.passed].to_dict("records"),
        "result_paths": {name: str(path) for name, path in paths.items()},
        "limits": [
            "one seed has no variance, interval, significance, or generalization claim",
            "this historical interval was already inspected and supports diagnosis only",
            "connectivity associations are descriptive and do not establish causal lift",
            "a changed M3 must use a new predeclared interval or independent data",
        ],
    }
    common._atomic_json(paths["json"], payload)
    return {name: str(path) for name, path in paths.items()}


def run_m3_edge_allocation_diagnostic(
    cfg: M3EdgeAllocationDiagnosticConfig | None = None,
) -> pd.DataFrame:
    cfg = validate_config(cfg or configure_m3_edge_allocation_diagnostic())
    print(json.dumps(preflight_summary(cfg), ensure_ascii=False, indent=2))
    source_json, source_payload, source_hash = _find_run_json(cfg)
    prepared, run_cfg = _prepare_source(cfg, source_payload)
    cohorts = _user_cohorts(prepared)
    item_traits = common._raw_item_traits(
        prepared["data"]["train"], prepared["data"]["n_items"]
    )
    loaded, topk_by_model, user_frames, rec_frames, checkpoints = {}, {}, [], [], []
    for model_id in MODELS:
        print(f"\n===== diagnostic | {model_id} | seed 42 =====")
        model, result, paths = _load_model(prepared, run_cfg, model_id)
        users, topk, _, _ = common._masked_topk(
            model, prepared, cfg.rank_limit, cfg.score_batch_size
        )
        topk_by_model[model_id] = topk
        loaded[model_id] = result
        user_frames.append(_per_user_metrics(users, topk, prepared, cohorts, model_id))
        rec_frames.append(_recommendation_rows(users, topk, model_id, cohorts, item_traits))
        checkpoints.append(
            {
                "model_id": model_id,
                "checkpoint": str(paths["checkpoint"]),
                "checkpoint_sha256": file_sha256(paths["checkpoint"]),
                "result": str(paths["result"]),
            }
        )
    if not np.array_equal(users, prepared["cache"].users):
        raise RuntimeError("evaluation user order differs from the source cache")
    per_user = pd.concat(user_frames, ignore_index=True)
    recommendations = pd.concat(rec_frames, ignore_index=True)
    truth = _truth_rows(users, topk_by_model, prepared, cohorts, item_traits)
    segments = _segment_summary(per_user)
    rank_movement = _rank_movement_summary(truth)
    graph = prepared["graph"]
    edges = _edge_intervention_table(
        edge_users=graph.edge_users,
        edge_items=graph.edge_items,
        base=graph.base_coefficients,
        adjusted=graph.item_user_coefficients,
        relationship_share=graph.relationship_share,
        allocation=graph.edge_clv_allocation,
    )
    intervention = _intervention_summary(edges)
    item_mechanism, correlations = _item_mechanism(
        edges, recommendations, truth, item_traits
    )
    candidate_rows = _candidate_rows(truth, recommendations)
    categories = item_traits.set_index("item_idx").category.reindex(
        np.arange(prepared["data"]["n_items"])
    ).fillna("UNKNOWN").to_numpy(object)
    connectivity = _candidate_connectivity(
        train=prepared["data"]["train"],
        candidates=candidate_rows,
        item_categories=categories,
        n_users=prepared["data"]["n_users"],
        n_items=prepared["data"]["n_items"],
    )
    evidence_parts = []
    for cutoff, group in connectivity.groupby("cutoff", sort=False):
        evidence_parts.append(_structure_evidence(group).assign(cutoff=int(cutoff)))
    evidence = pd.concat(evidence_parts, ignore_index=True)
    representatives, representative_detail = _representative_users(
        per_user, truth, recommendations, cfg.max_representative_users
    )
    quality = _quality_checks(
        users, topk_by_model, per_user, truth, prepared, loaded
    )
    frames = {
        "quality_checks": quality,
        "intervention_summary": intervention,
        "rank_movement_summary": rank_movement,
        "truth_rank_transition": truth,
        "user_metrics": per_user,
        "segment_metric_summary": segments,
        "recommendation_top100": recommendations,
        "item_mechanism": item_mechanism,
        "item_mechanism_correlations": correlations,
        "candidate_connectivity": connectivity,
        "next_structure_evidence": evidence,
        "representative_users": representatives,
        "representative_user_details": representative_detail,
    }
    paths = _persist(
        cfg, source_json, source_hash, source_payload, frames, checkpoints
    )
    evidence.attrs["result_paths"] = paths
    evidence.attrs["quality_passed"] = bool(quality.passed.all())
    print("\n===== intervention strength =====")
    print(intervention.to_string(index=False))
    print("\n===== truth rank movement =====")
    print(rank_movement.to_string(index=False))
    print("\n===== train-only connectivity evidence =====")
    print(evidence.to_string(index=False))
    print("\n===== quality checks =====")
    print(quality.to_string(index=False))
    print("\nResult files:", paths)
    if not quality.passed.all():
        raise RuntimeError("quality checks failed; diagnostic interpretation stopped")
    return evidence


if __name__ == "__main__":
    print(
        json.dumps(
            preflight_summary(configure_m3_edge_allocation_diagnostic()),
            ensure_ascii=False,
            indent=2,
        )
    )
