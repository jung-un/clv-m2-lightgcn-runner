"""Locked train-only feasibility diagnostic for an M3 transition relation.

Relations and historical CLV are constructed on Dunnhumby DAY 1--662.  The
only evaluation interval is the earlier pseudo-future DAY 663--669.  No model
is trained and no final test or holdout is constructed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from clv_m3_next_new_transition import (
    MODEL_CLV,
    MODEL_GLOBAL,
    MODEL_SHUFFLE,
    build_historical_clv,
    build_transition_graphs,
    build_user_transition_events,
    count_transition_candidates,
    decide_pilot,
    evaluate_transition_ranking,
    rank_transition_candidates,
    reachable_truth_share,
)
import lightgcn_clv_moe as moe
import lightgcn_clv_v3 as v3


CODE_VERSION = "m3-clv-weighted-next-new-transition-diagnostic-v1"
MODEL_IDS = (MODEL_GLOBAL, MODEL_CLV, MODEL_SHUFFLE)


@dataclass(frozen=True)
class M3NextNewTransitionDiagnosticConfig:
    dataset: str = "dunnhumby"
    construction_end_day: int = 662
    evaluation_start_day: int = 663
    evaluation_end_day: int = 669
    shuffle_seed: int = 20260826
    rank_limit: int = 50
    min_user_interactions: int = 1
    min_item_interactions: int = 1
    out_dir: str = ""


def _default_out_dir() -> str:
    if v3.IN_COLAB:
        return (
            "/content/drive/MyDrive/논문/data/"
            "results_m3_next_new_transition_diagnostic_dunnhumby"
        )
    return f"{v3.default_out_dir('dunnhumby')}_m3_next_new_transition_diagnostic"


def configure_m3_next_new_transition_diagnostic(
    **overrides,
) -> M3NextNewTransitionDiagnosticConfig:
    return validate_config(
        M3NextNewTransitionDiagnosticConfig(
            **({"out_dir": _default_out_dir()} | overrides)
        )
    )


def validate_config(
    cfg: M3NextNewTransitionDiagnosticConfig,
) -> M3NextNewTransitionDiagnosticConfig:
    fixed = {
        "dataset": "dunnhumby",
        "construction_end_day": 662,
        "evaluation_start_day": 663,
        "evaluation_end_day": 669,
        "shuffle_seed": 20260826,
        "rank_limit": 50,
        "min_user_interactions": 1,
        "min_item_interactions": 1,
    }
    for key, expected in fixed.items():
        if getattr(cfg, key) != expected:
            raise ValueError(f"locked transition diagnostic requires {key}={expected!r}")
    if not cfg.out_dir:
        raise ValueError("out_dir is required")
    return cfg


def preflight_summary(cfg: M3NextNewTransitionDiagnosticConfig) -> dict:
    cfg = validate_config(cfg)
    return {
        "code_version": CODE_VERSION,
        "analysis_type": "exploratory train-only relation feasibility diagnostic",
        "dataset": cfg.dataset,
        "models": list(MODEL_IDS),
        "construction_interval": {"start_inclusive": 1, "end_inclusive": 662},
        "pseudo_future_interval": {
            "start_inclusive": 663,
            "end_inclusive": 669,
        },
        "final_test_constructed": False,
        "holdout_constructed": False,
        "training": False,
        "checkpoint_selection": False,
        "new_item_task": True,
        "min_user_interactions": cfg.min_user_interactions,
        "min_item_interactions": cfg.min_item_interactions,
        "historical_clv_proxy": "N_hat * V_hat",
        "relation": (
            "consecutive current basket -> next basket first-purchase items; "
            "basket-size and user-mass normalized; source-row normalized"
        ),
        "controls": {
            MODEL_GLOBAL: "same transition relation without CLV weighting",
            MODEL_SHUFFLE: (
                "same CLV coefficient multiset shuffled within N_hat midrank deciles"
            ),
        },
        "no_price_input": True,
        "no_ppmi_or_pruning": True,
        "no_popularity_backfill": True,
        "interpretation": (
            "hypothesis-generation only; no significance, generalization, neural-model, "
            "or final-test claim"
        ),
        "out_dir": cfg.out_dir,
    }


def _prepare_transactions_frame(
    transactions: pd.DataFrame,
    cfg: M3NextNewTransitionDiagnosticConfig,
) -> dict:
    """Cap future rows first, apply train-universe filtering, and index entities."""
    cfg = validate_config(cfg)
    required = {"u_raw", "i_raw", "t", "v", "up", "b_raw"}
    missing = required.difference(transactions.columns)
    if missing:
        raise ValueError(f"missing raw transaction columns: {sorted(missing)}")
    source_rows = len(transactions)
    tx = transactions[transactions["t"] <= cfg.evaluation_end_day].copy()
    if tx.empty:
        raise ValueError("no transactions on or before DAY 669")
    if tx["t"].max() > cfg.evaluation_end_day:
        raise RuntimeError("future cap failed")

    construction_raw = tx[tx["t"] <= cfg.construction_end_day]
    if construction_raw.empty:
        raise ValueError("construction interval is empty")
    keep_users, keep_items, n_edges, kcore_iterations = v3.kcore_filter(
        construction_raw,
        cfg.min_user_interactions,
        cfg.min_item_interactions,
    )
    tx = tx[
        tx["u_raw"].isin(keep_users) & tx["i_raw"].isin(keep_items)
    ].copy()
    user_ids = np.sort(tx["u_raw"].unique())
    item_ids = np.sort(tx["i_raw"].unique())
    user_index = {raw: idx for idx, raw in enumerate(user_ids)}
    item_index = {raw: idx for idx, raw in enumerate(item_ids)}
    tx["u_idx"] = tx["u_raw"].map(user_index).astype(np.int32)
    tx["i_idx"] = tx["i_raw"].map(item_index).astype(np.int32)
    tx["basket_id"] = tx["b_raw"]
    construction = tx[tx["t"] <= cfg.construction_end_day].copy()
    evaluation = tx[
        (tx["t"] >= cfg.evaluation_start_day)
        & (tx["t"] <= cfg.evaluation_end_day)
    ].copy()
    return {
        "transactions": tx,
        "construction": construction,
        "evaluation": evaluation,
        "user_ids": user_ids,
        "item_ids": item_ids,
        "n_users": len(user_ids),
        "n_items": len(item_ids),
        "source_rows": source_rows,
        "capped_rows": len(tx),
        "discarded_future_rows": int(
            (transactions["t"] > cfg.evaluation_end_day).sum()
        ),
        "train_edges": n_edges,
        "kcore_iterations": kcore_iterations,
    }


def _build_truth(prepared: dict) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    construction = prepared["construction"]
    evaluation = prepared["evaluation"]
    n_items = prepared["n_items"]
    seen_keys = np.unique(
        construction.u_idx.to_numpy(np.int64) * n_items
        + construction.i_idx.to_numpy(np.int64)
    )
    eval_pairs = (
        evaluation.groupby(["u_idx", "i_idx"], sort=False)["v"]
        .sum()
        .reset_index()
    )
    if len(eval_pairs):
        keys = (
            eval_pairs.u_idx.to_numpy(np.int64) * n_items
            + eval_pairs.i_idx.to_numpy(np.int64)
        )
        positions = np.searchsorted(seen_keys, keys)
        present = (positions < len(seen_keys)) & (
            seen_keys[np.minimum(positions, max(len(seen_keys) - 1, 0))] == keys
        )
        eval_pairs = eval_pairs[~present]
    truth: dict[int, np.ndarray] = {}
    truth_value: dict[int, np.ndarray] = {}
    for user, rows in eval_pairs.groupby("u_idx", sort=False):
        order = np.argsort(rows.i_idx.to_numpy())
        truth[int(user)] = rows.i_idx.to_numpy(np.int32)[order]
        truth_value[int(user)] = rows.v.to_numpy(np.float64)[order]
    return truth, truth_value


def _history_maps(prepared: dict) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    construction = prepared["construction"]
    seen = {
        int(user): np.sort(rows.i_idx.unique()).astype(np.int32)
        for user, rows in construction.groupby("u_idx", sort=False)
    }
    last_basket: dict[int, np.ndarray] = {}
    ordered = construction.sort_values(
        ["u_idx", "t", "basket_id"], kind="mergesort"
    )
    for user, rows in ordered.groupby("u_idx", sort=False):
        last = rows.iloc[-1]
        mask = (rows["t"] == last["t"]) & (rows["basket_id"] == last["basket_id"])
        last_basket[int(user)] = np.sort(rows.loc[mask, "i_idx"].unique()).astype(
            np.int32
        )
    return seen, last_basket


def _safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    if np.unique(left).size < 2 or np.unique(right).size < 2:
        return float("nan")
    return float(spearmanr(left, right).statistic)


def _item_diagnostics(
    rankings: dict[int, np.ndarray], prepared: dict, *, k: int = 10
) -> dict[str, float]:
    construction = prepared["construction"]
    n_items = prepared["n_items"]
    popularity = np.zeros(n_items, dtype=np.float64)
    buyers = construction[["u_idx", "i_idx"]].drop_duplicates().i_idx.value_counts()
    popularity[buyers.index.to_numpy(int)] = buyers.to_numpy(float)
    price = np.full(n_items, float(construction.up.median()), dtype=np.float64)
    medians = construction.groupby("i_idx")["up"].median()
    price[medians.index.to_numpy(int)] = medians.to_numpy(float)
    price_percentile = (rankdata(price, method="average") - 0.5) / n_items
    exposure = np.zeros(n_items, dtype=np.float64)
    selected_prices: list[float] = []
    for ranked in rankings.values():
        selected = np.asarray(ranked[:k], dtype=np.int64)
        np.add.at(exposure, selected, 1)
        selected_prices.extend(price_percentile[selected].tolist())
    return {
        "mean_recommended_price_percentile@10": (
            float(np.mean(selected_prices)) if selected_prices else float("nan")
        ),
        "exposure_popularity_spearman@10": _safe_spearman(exposure, popularity),
        "exposure_price_percentile_spearman@10": _safe_spearman(
            exposure, price_percentile
        ),
    }


def _weighted_hit(
    rankings: dict[int, np.ndarray],
    truth: dict[int, np.ndarray],
    truth_value: dict[int, np.ndarray],
    *,
    k: int,
) -> float:
    values = []
    for user, items in truth.items():
        value_by_item = dict(zip(items.tolist(), truth_value[user].tolist()))
        values.append(sum(value_by_item.get(int(item), 0.0) for item in rankings[user][:k]))
    return float(np.mean(values)) if values else 0.0


def _truth_support_table(
    *,
    truth: dict[int, np.ndarray],
    rankings_by_model: dict[str, dict[int, np.ndarray]],
    last_basket: dict[int, np.ndarray],
    edge_support,
) -> pd.DataFrame:
    rows = []
    for user, items in truth.items():
        sources = last_basket.get(user, np.empty(0, dtype=np.int32))
        for item in items:
            support = (
                int(edge_support[sources, int(item)].max()) if len(sources) else 0
            )
            if support == 0:
                stratum = "0"
            elif support == 1:
                stratum = "1"
            elif support <= 4:
                stratum = "2-4"
            else:
                stratum = "5+"
            row = {
                "user_idx": user,
                "truth_item_idx": int(item),
                "max_user_support": support,
                "support_stratum": stratum,
            }
            for model_id, rankings in rankings_by_model.items():
                ranked = rankings[user]
                row[f"{model_id}_hit@10"] = int(int(item) in ranked[:10])
                row[f"{model_id}_hit@50"] = int(int(item) in ranked[:50])
            rows.append(row)
    return pd.DataFrame(rows)


def _segment_table(
    per_user_by_model: dict[str, pd.DataFrame], clv_percentile: np.ndarray
) -> pd.DataFrame:
    rows = []
    for model_id, frame in per_user_by_model.items():
        local = frame.copy()
        local["clv_quintile"] = np.minimum(
            (clv_percentile[local.user_idx.to_numpy(int)] * 5).astype(int) + 1, 5
        )
        for quintile, group in local.groupby("clv_quintile"):
            for metric in ["recall@10", "ndcg@10", "recall@20", "ndcg@20", "recall@50", "ndcg@50"]:
                rows.append(
                    {
                        "model_id": model_id,
                        "segment_type": "historical_clv_quintile",
                        "segment_id": f"Q{int(quintile)}",
                        "n_users": len(group),
                        "metric": metric,
                        "value": float(group[metric].mean()),
                    }
                )
    return pd.DataFrame(rows)


def _json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value)!r}")


def _run_hash(cfg, input_manifest, source_revision) -> str:
    payload = {
        "code_version": CODE_VERSION,
        "config": asdict(cfg),
        "input_manifest": input_manifest,
        "source_revision": source_revision,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=_json_default).encode()
    ).hexdigest()[:12]


def run_m3_next_new_transition_diagnostic_from_frames(
    transactions: pd.DataFrame,
    cfg: M3NextNewTransitionDiagnosticConfig,
    *,
    input_manifest: dict,
    source_revision: str,
) -> pd.DataFrame:
    cfg = validate_config(cfg)
    prepared = _prepare_transactions_frame(transactions, cfg)
    truth, truth_value = _build_truth(prepared)
    if not truth:
        raise ValueError("pseudo-future contains no eligible new-item truths")
    seen, last_basket = _history_maps(prepared)
    construction = prepared["construction"]
    clv, shuffled_coefficient = build_historical_clv(
        construction, n_users=prepared["n_users"], shuffle_seed=cfg.shuffle_seed
    )
    events = build_user_transition_events(
        construction, n_users=prepared["n_users"]
    )
    graphs = build_transition_graphs(
        events,
        clv_coefficient=clv.coefficient,
        shuffled_coefficient=shuffled_coefficient,
        n_items=prepared["n_items"],
    )
    relations = {
        MODEL_GLOBAL: graphs.global_relation,
        MODEL_CLV: graphs.clv_relation,
        MODEL_SHUFFLE: graphs.shuffled_clv_relation,
    }
    eval_users = np.asarray(sorted(truth), dtype=np.int32)
    rankings_by_model = {
        model_id: rank_transition_candidates(
            relation,
            last_basket_items=last_basket,
            seen_items=seen,
            eval_users=eval_users,
            top_k=cfg.rank_limit,
        )
        for model_id, relation in relations.items()
    }
    candidate_counts_by_model = {
        model_id: count_transition_candidates(
            relation,
            last_basket_items=last_basket,
            seen_items=seen,
            eval_users=eval_users,
        )
        for model_id, relation in relations.items()
    }

    absolute_rows = []
    per_user_by_model = {}
    for model_id, relation in relations.items():
        metrics, per_user = evaluate_transition_ranking(
            rankings_by_model[model_id],
            truth=truth,
            n_items=prepared["n_items"],
        )
        per_user["n_positive_candidates"] = per_user.user_idx.map(
            candidate_counts_by_model[model_id]
        )
        metrics["mean_positive_candidates"] = float(
            per_user.n_positive_candidates.mean()
        )
        metrics.update(
            _item_diagnostics(rankings_by_model[model_id], prepared, k=10)
        )
        metrics["reachable_truth_share"] = reachable_truth_share(
            relation,
            last_basket_items=last_basket,
            seen_items=seen,
            truth=truth,
        )
        metrics["price_purchase_amount_weighted_hit@10"] = _weighted_hit(
            rankings_by_model[model_id], truth, truth_value, k=10
        )
        absolute_rows.append({"model_id": model_id, "role": "model" if model_id == MODEL_CLV else "control", **metrics})
        per_user_by_model[model_id] = per_user
    absolute = pd.DataFrame(absolute_rows)
    pilot_decision = decide_pilot(absolute)

    reference = absolute.set_index("model_id").loc[MODEL_GLOBAL]
    comparison_rows = []
    numeric_columns = absolute.select_dtypes(include=[np.number]).columns
    for model_id in (MODEL_CLV, MODEL_SHUFFLE):
        model = absolute.set_index("model_id").loc[model_id]
        for metric in numeric_columns:
            comparison_rows.append(
                {
                    "model_id": model_id,
                    "reference": MODEL_GLOBAL,
                    "metric": metric,
                    "reference_value": float(reference[metric]),
                    "model_value": float(model[metric]),
                    "absolute_delta": float(model[metric] - reference[metric]),
                }
            )
    comparison = pd.DataFrame(comparison_rows)
    segments = _segment_table(per_user_by_model, clv.percentile)
    support = _truth_support_table(
        truth=truth,
        rankings_by_model=rankings_by_model,
        last_basket=last_basket,
        edge_support=graphs.edge_support,
    )

    quality_rows = []
    for model_id, relation in relations.items():
        row_mass = np.asarray(relation.sum(axis=1)).ravel()
        nonempty = np.diff(relation.indptr) > 0
        quality_rows.extend(
            [
                {
                    "model_id": model_id,
                    "check": "nonempty_rows_sum_to_one",
                    "value": float(np.max(np.abs(row_mass[nonempty] - 1.0))) if nonempty.any() else 0.0,
                    "passed": bool((np.abs(row_mass[nonempty] - 1.0) <= 1e-10).all()),
                },
                {
                    "model_id": model_id,
                    "check": "empty_rows_sum_to_zero",
                    "value": float(np.max(np.abs(row_mass[~nonempty]))) if (~nonempty).any() else 0.0,
                    "passed": bool((np.abs(row_mass[~nonempty]) <= 1e-12).all()),
                },
            ]
        )
        seen_violations = sum(
            len(set(rankings_by_model[model_id][user]).intersection(seen[user]))
            for user in truth
        )
        quality_rows.append(
            {
                "model_id": model_id,
                "check": "recommendations_exclude_construction_pairs",
                "value": seen_violations,
                "passed": seen_violations == 0,
            }
        )
        recomputed, _ = evaluate_transition_ranking(
            rankings_by_model[model_id],
            truth=truth,
            n_items=prepared["n_items"],
        )
        recompute_error = max(
            abs(
                float(recomputed[metric])
                - float(absolute.set_index("model_id").loc[model_id, metric])
            )
            for metric in (
                "recall@10",
                "ndcg@10",
                "recall@20",
                "ndcg@20",
                "recall@50",
                "ndcg@50",
            )
        )
        quality_rows.append(
            {
                "model_id": model_id,
                "check": "ranking_metrics_recompute",
                "value": recompute_error,
                "passed": recompute_error <= 1e-12,
            }
        )
    truth_seen_violations = sum(
        len(set(items).intersection(seen[user])) for user, items in truth.items()
    )
    quality_rows.extend(
        [
            {
                "model_id": "all",
                "check": "truth_excludes_construction_pairs",
                "value": truth_seen_violations,
                "passed": truth_seen_violations == 0,
            },
            {
                "model_id": "all",
                "check": "transactions_after_day_669",
                "value": int((prepared["transactions"].t > cfg.evaluation_end_day).sum()),
                "passed": bool((prepared["transactions"].t <= cfg.evaluation_end_day).all()),
            },
            {
                "model_id": "all",
                "check": "clv_coefficient_mean_one",
                "value": float(abs(clv.coefficient.mean() - 1.0)),
                "passed": bool(np.isclose(clv.coefficient.mean(), 1.0)),
            },
            {
                "model_id": "all",
                "check": "eligible_user_transition_mass_one",
                "value": float(
                    max(
                        (
                            abs(events.contribution[events.user_idx == user].sum() - 1.0)
                            for user in np.flatnonzero(
                                events.eligible_pair_count_by_user > 0
                            )
                        ),
                        default=0.0,
                    )
                ),
                "passed": bool(
                    all(
                        np.isclose(
                            events.contribution[events.user_idx == user].sum(), 1.0
                        )
                        for user in np.flatnonzero(
                            events.eligible_pair_count_by_user > 0
                        )
                    )
                ),
            },
            {
                "model_id": "all",
                "check": "shuffle_preserves_coefficients_within_activity_decile",
                "value": int(
                    any(
                        not np.allclose(
                            np.sort(shuffled_coefficient[clv.activity_decile == decile]),
                            np.sort(clv.coefficient[clv.activity_decile == decile]),
                        )
                        for decile in np.unique(clv.activity_decile)
                    )
                ),
                "passed": bool(
                    all(
                        np.allclose(
                            np.sort(shuffled_coefficient[clv.activity_decile == decile]),
                            np.sort(clv.coefficient[clv.activity_decile == decile]),
                        )
                        for decile in np.unique(clv.activity_decile)
                    )
                ),
            },
        ]
    )
    quality = pd.DataFrame(quality_rows)
    quality_passed = bool(quality.passed.all())

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_hash = _run_hash(cfg, input_manifest, source_revision)
    prefix = out_dir / f"m3_next_new_transition_diagnostic_{run_hash}"
    paths = {
        "absolute_csv": str(prefix.with_suffix(".csv")),
        "comparison_csv": str(prefix.parent / f"{prefix.name}_comparison.csv"),
        "segment_csv": str(prefix.parent / f"{prefix.name}_segments.csv"),
        "support_csv": str(prefix.parent / f"{prefix.name}_truth_support.csv"),
        "quality_csv": str(prefix.parent / f"{prefix.name}_quality.csv"),
        "json": str(prefix.with_suffix(".json")),
    }
    absolute.to_csv(paths["absolute_csv"], index=False)
    comparison.to_csv(paths["comparison_csv"], index=False)
    segments.to_csv(paths["segment_csv"], index=False)
    support.to_csv(paths["support_csv"], index=False)
    quality.to_csv(paths["quality_csv"], index=False)
    payload = {
        "code_version": CODE_VERSION,
        "config": asdict(cfg),
        "preflight": preflight_summary(cfg),
        "input_manifest": input_manifest,
        "source_revision": source_revision,
        "run_hash": run_hash,
        "split": {
            "construction_end_day": cfg.construction_end_day,
            "evaluation_start_day": cfg.evaluation_start_day,
            "evaluation_end_day": cfg.evaluation_end_day,
            "transactions_after_day_669": int(
                (prepared["transactions"].t > cfg.evaluation_end_day).sum()
            ),
            "discarded_future_rows_before_preparation": prepared["discarded_future_rows"],
            "construction_rows": len(construction),
            "evaluation_rows": len(prepared["evaluation"]),
            "evaluation_users": len(truth),
            "truth_pairs": int(sum(map(len, truth.values()))),
        },
        "relation_diagnostic": {
            "n_transition_events": len(events.user_idx),
            "n_users_with_eligible_basket_pairs": int(
                (events.eligible_pair_count_by_user > 0).sum()
            ),
            "n_global_relation_edges": int(graphs.global_relation.nnz),
            "coefficient_mean": float(clv.coefficient.mean()),
            "coefficient_std": float(clv.coefficient.std()),
        },
        "absolute_metrics": absolute.to_dict(orient="records"),
        "pilot_decision": pilot_decision,
        "quality_passed": quality_passed,
        "quality_checks": quality.to_dict(orient="records"),
        "result_paths": paths,
    }
    Path(paths["json"]).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    absolute.attrs["quality_passed"] = quality_passed
    absolute.attrs["pilot_decision"] = pilot_decision
    absolute.attrs["result_paths"] = paths
    return absolute


def run_m3_next_new_transition_diagnostic(
    cfg: M3NextNewTransitionDiagnosticConfig,
) -> pd.DataFrame:
    cfg = validate_config(cfg)
    schema = v3.SCHEMA[cfg.dataset]
    transactions = v3.load_transactions(schema)
    return run_m3_next_new_transition_diagnostic_from_frames(
        transactions,
        cfg,
        input_manifest=moe.build_input_manifest(schema),
        source_revision=moe.source_revision(),
    )
