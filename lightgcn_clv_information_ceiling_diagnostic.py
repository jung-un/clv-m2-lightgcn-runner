"""Train-only information-ceiling diagnostics for historical CLV.

This module does not train a recommender.  It asks whether binary purchase
history already proxies historical CLV and whether CLV-based user segments
add held-out new-item information beyond global and activity-matched item
popularity distributions.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import rankdata, spearmanr
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from clv_m3_next_new_transition import (
    HistoricalCLV,
    build_historical_clv,
    evaluate_transition_ranking,
)
import lightgcn_clv_moe as moe
from lightgcn_clv_m3_next_new_transition_diagnostic import (
    M3NextNewTransitionDiagnosticConfig,
    _build_truth,
    _prepare_transactions_frame,
)
import lightgcn_clv_v3 as v3


CODE_VERSION = "clv-information-ceiling-diagnostic-v1"
MODEL_IDS = (
    "global_popularity",
    "n_decile_popularity",
    "clv_decile_popularity",
    "n_matched_clv_popularity",
)
ACCURACY_METRICS = tuple(
    f"{metric}@{k}"
    for metric in ("recall", "ndcg")
    for k in (10, 20, 50)
)


@dataclass(frozen=True)
class CLVInformationCeilingConfig:
    dataset: str = "dunnhumby"
    construction_end_day: int = 662
    evaluation_start_day: int = 663
    evaluation_end_day: int = 669
    segment_seed: int = 20260826
    rank_limit: int = 50
    min_user_interactions: int = 1
    min_item_interactions: int = 1
    out_dir: str = ""

    def base_config(self) -> M3NextNewTransitionDiagnosticConfig:
        return M3NextNewTransitionDiagnosticConfig(
            dataset=self.dataset,
            construction_end_day=self.construction_end_day,
            evaluation_start_day=self.evaluation_start_day,
            evaluation_end_day=self.evaluation_end_day,
            shuffle_seed=20260826,
            rank_limit=self.rank_limit,
            min_user_interactions=self.min_user_interactions,
            min_item_interactions=self.min_item_interactions,
            out_dir=self.out_dir,
        )


def _default_out_dir() -> str:
    if v3.IN_COLAB:
        return (
            "/content/drive/MyDrive/논문/data/"
            "results_clv_information_ceiling_dunnhumby"
        )
    return f"{v3.default_out_dir('dunnhumby')}_clv_information_ceiling"


def configure_clv_information_ceiling_diagnostic(
    **overrides,
) -> CLVInformationCeilingConfig:
    cfg = CLVInformationCeilingConfig(
        **({"out_dir": _default_out_dir()} | overrides)
    )
    fixed = {
        "dataset": "dunnhumby",
        "construction_end_day": 662,
        "evaluation_start_day": 663,
        "evaluation_end_day": 669,
        "segment_seed": 20260826,
        "rank_limit": 50,
        "min_user_interactions": 1,
        "min_item_interactions": 1,
    }
    for key, expected in fixed.items():
        if getattr(cfg, key) != expected:
            raise ValueError(f"locked information diagnostic requires {key}={expected!r}")
    if not cfg.out_dir:
        raise ValueError("out_dir is required")
    return cfg


def metric_contract(cfg: CLVInformationCeilingConfig) -> dict:
    return {
        "population": (
            "users and items present in DAY 1--662 after train-unique-edge "
            "MIN_USER_INTER=1, MIN_ITEM_INTER=1 filtering"
        ),
        "construction": "Dunnhumby DAY 1--662 inclusive",
        "pseudo_future": "Dunnhumby DAY 663--669 inclusive",
        "truth": "pseudo-future items absent from each user's construction history",
        "deduplication": {
            "source": "exact duplicate transaction rows removed by shared loader",
            "basket_item": "unique item within user basket for history graph features",
            "popularity": "unique construction user-item pair",
            "truth": "unique pseudo-future user-item pair; purchase amount summed",
        },
        "historical_clv_proxy": (
            "N_hat distinct construction baskets * V_hat mean construction basket value"
        ),
        "history_proxy_features": (
            "binary user-item degree plus buyer-count and item-price-percentile summaries; "
            "not an M1 embedding and not proof of learned representation content"
        ),
        "oracle_comparisons": list(MODEL_IDS),
        "accuracy": list(ACCURACY_METRICS),
        "balanced_accuracy_index": (
            "geometric mean of the six model/reference Recall/NDCG ratios"
        ),
        "final_test_constructed": False,
        "holdout_constructed": False,
        "causal_claim": False,
        "out_dir": cfg.out_dir,
    }


def _binary_history_features(
    prepared: dict, clv: HistoricalCLV
) -> pd.DataFrame:
    construction = prepared["construction"]
    n_items = prepared["n_items"]
    pairs = construction[["u_idx", "i_idx"]].drop_duplicates()
    buyer_count = np.zeros(n_items, dtype=np.float64)
    counts = pairs.i_idx.value_counts()
    buyer_count[counts.index.to_numpy(int)] = counts.to_numpy(float)
    price = np.full(n_items, float(construction.up.median()), dtype=np.float64)
    medians = construction.groupby("i_idx")["up"].median()
    price[medians.index.to_numpy(int)] = medians.to_numpy(float)
    price_percentile = (rankdata(price, method="average") - 0.5) / n_items

    rows = []
    for user, user_pairs in pairs.groupby("u_idx", sort=False):
        user = int(user)
        items = user_pairs.i_idx.to_numpy(int)
        popularity = buyer_count[items]
        item_price = price_percentile[items]
        rows.append(
            {
                "user_idx": user,
                "unique_items": len(items),
                "mean_item_buyer_count": float(popularity.mean()),
                "std_item_buyer_count": float(popularity.std()),
                "max_item_buyer_count": float(popularity.max()),
                "mean_item_price_percentile": float(item_price.mean()),
                "std_item_price_percentile": float(item_price.std()),
                "min_item_price_percentile": float(item_price.min()),
                "max_item_price_percentile": float(item_price.max()),
                "clv_percentile": float(clv.percentile[user]),
            }
        )
    return pd.DataFrame(rows).sort_values("user_idx").reset_index(drop=True)


def _midrank_bins(values: np.ndarray, n_bins: int) -> np.ndarray:
    percentile = (rankdata(values, method="average") - 0.5) / len(values)
    return np.minimum(np.floor(percentile * n_bins).astype(np.int16), n_bins - 1)


def _assign_segments(clv: HistoricalCLV) -> dict[str, np.ndarray]:
    n_decile = _midrank_bins(clv.n_hat, 10)
    clv_decile = _midrank_bins(clv.clv_proxy, 10)
    within_n_quintile = np.zeros(len(clv.n_hat), dtype=np.int16)
    for activity_segment in np.unique(n_decile):
        indices = np.flatnonzero(n_decile == activity_segment)
        within_n_quintile[indices] = _midrank_bins(clv.clv_proxy[indices], 5)
    return {
        "global": np.zeros(len(clv.n_hat), dtype=np.int16),
        "n_decile": n_decile,
        "clv_decile": clv_decile,
        "n_matched_clv": (n_decile * 5 + within_n_quintile).astype(np.int16),
    }


def _rank_segment_popularity(
    user_item_pairs: pd.DataFrame,
    *,
    segment_by_user: np.ndarray,
    eval_users: np.ndarray,
    seen_items: dict[int, np.ndarray],
    n_items: int,
    top_k: int,
) -> dict[int, np.ndarray]:
    user_column = "user_idx" if "user_idx" in user_item_pairs else "u_idx"
    pairs = (
        user_item_pairs[[user_column, "i_idx"]]
        .rename(columns={user_column: "user_idx"})
        .drop_duplicates()
        .copy()
    )
    pairs["segment_id"] = segment_by_user[pairs.user_idx.to_numpy(int)]
    counts = (
        pairs.groupby(["segment_id", "i_idx"], sort=False)
        .size()
        .rename("buyer_count")
        .reset_index()
    )
    ordered_by_segment: dict[int, np.ndarray] = {}
    for segment, rows in counts.groupby("segment_id", sort=False):
        ordered = rows.sort_values(
            ["buyer_count", "i_idx"], ascending=[False, True], kind="mergesort"
        )
        ordered_by_segment[int(segment)] = ordered.i_idx.to_numpy(np.int32)

    rankings = {}
    for raw_user in eval_users:
        user = int(raw_user)
        candidates = ordered_by_segment.get(
            int(segment_by_user[user]), np.empty(0, dtype=np.int32)
        )
        seen = set(np.asarray(seen_items.get(user, []), dtype=np.int64).tolist())
        selected = [int(item) for item in candidates if int(item) not in seen]
        rankings[user] = np.asarray(selected[:top_k], dtype=np.int32)
    return rankings


def _balanced_accuracy_index(model: dict, reference: dict) -> float:
    ratios = []
    for metric in ACCURACY_METRICS:
        denominator = float(reference[metric])
        numerator = float(model[metric])
        if denominator <= 0:
            raise ValueError(f"balanced index requires positive reference {metric}")
        ratios.append(numerator / denominator)
    ratios_array = np.asarray(ratios)
    if np.any(ratios_array < 0):
        raise ValueError("balanced index requires nonnegative model metrics")
    if np.any(ratios_array == 0):
        return 0.0
    return float(np.exp(np.mean(np.log(ratios_array))))


def _paired_bootstrap_balanced_index(
    model_per_user: pd.DataFrame,
    reference_per_user: pd.DataFrame,
    *,
    comparison_id: str,
    seed: int,
    n_bootstrap: int = 2000,
) -> dict:
    model = model_per_user.set_index("user_idx").sort_index()
    reference = reference_per_user.set_index("user_idx").sort_index()
    if not model.index.equals(reference.index):
        raise ValueError("paired bootstrap requires identical evaluation users")
    model_values = model[list(ACCURACY_METRICS)].to_numpy(float)
    reference_values = reference[list(ACCURACY_METRICS)].to_numpy(float)
    point_estimate = _balanced_accuracy_index(
        dict(zip(ACCURACY_METRICS, model_values.mean(axis=0))),
        dict(zip(ACCURACY_METRICS, reference_values.mean(axis=0))),
    )
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_bootstrap, dtype=np.float64)
    for iteration in range(n_bootstrap):
        sampled = rng.integers(0, len(model_values), size=len(model_values))
        estimates[iteration] = _balanced_accuracy_index(
            dict(zip(ACCURACY_METRICS, model_values[sampled].mean(axis=0))),
            dict(zip(ACCURACY_METRICS, reference_values[sampled].mean(axis=0))),
        )
    return {
        "comparison_id": comparison_id,
        "n_users": len(model_values),
        "n_bootstrap": n_bootstrap,
        "point_estimate": point_estimate,
        "lo": float(np.quantile(estimates, 0.025)),
        "hi": float(np.quantile(estimates, 0.975)),
        "positive_bootstrap_share": float((estimates > 1.0).mean()),
        "interval_unit": "paired user bootstrap of six-metric geometric ratio",
    }


def _redundancy_diagnostic(features: pd.DataFrame, seed: int) -> pd.DataFrame:
    feature_columns = [
        column
        for column in features.columns
        if column not in {"user_idx", "clv_percentile"}
    ]
    x = features[feature_columns].to_numpy(float)
    y = features.clv_percentile.to_numpy(float)
    folds = min(5, len(features))
    if folds < 2:
        raise ValueError("at least two users are required for redundancy diagnosis")
    splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)

    rows = []
    definitions = {
        "binary_degree_only": [feature_columns.index("unique_items")],
        "binary_history_summary": list(range(len(feature_columns))),
    }
    for model_id, columns in definitions.items():
        estimator = TransformedTargetRegressor(
            regressor=make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
            transformer=StandardScaler(),
        )
        prediction = cross_val_predict(estimator, x[:, columns], y, cv=splitter)
        rows.append(
            {
                "model_id": model_id,
                "n_users": len(y),
                "target": "historical_clv_midrank_percentile",
                "cv_folds": folds,
                "r2": float(r2_score(y, prediction)),
                "spearman": float(spearmanr(y, prediction).statistic),
                "mae_percentile": float(mean_absolute_error(y, prediction)),
            }
        )
    return pd.DataFrame(rows)


def _weighted_segment_js(
    truth: dict[int, np.ndarray],
    *,
    segment_by_user: np.ndarray,
    parent_by_user: np.ndarray,
    n_items: int,
    comparison_id: str,
) -> dict:
    truth_rows = [
        (int(user), int(item))
        for user, items in truth.items()
        for item in np.asarray(items, dtype=np.int64)
    ]
    frame = pd.DataFrame(truth_rows, columns=["user_idx", "i_idx"])
    frame["segment_id"] = segment_by_user[frame.user_idx.to_numpy(int)]
    frame["parent_id"] = parent_by_user[frame.user_idx.to_numpy(int)]
    divergences = []
    weights = []
    for (_, parent), group in frame.groupby(["segment_id", "parent_id"], sort=False):
        parent_rows = frame[frame.parent_id == parent]
        group_counts = np.bincount(group.i_idx, minlength=n_items).astype(float)
        parent_counts = np.bincount(parent_rows.i_idx, minlength=n_items).astype(float)
        divergence = float(jensenshannon(group_counts, parent_counts, base=2) ** 2)
        divergences.append(divergence)
        weights.append(len(group))
    return {
        "comparison_id": comparison_id,
        "n_truth_pairs": len(frame),
        "n_segments": int(frame.segment_id.nunique()),
        "weighted_mean_js_divergence_bits": float(
            np.average(divergences, weights=weights)
        ),
    }


def _json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
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


def run_clv_information_ceiling_diagnostic_from_frames(
    transactions: pd.DataFrame,
    cfg: CLVInformationCeilingConfig,
    *,
    input_manifest: dict,
    source_revision: str,
) -> pd.DataFrame:
    cfg = configure_clv_information_ceiling_diagnostic(**asdict(cfg))
    prepared = _prepare_transactions_frame(transactions, cfg.base_config())
    truth, _ = _build_truth(prepared)
    if not truth:
        raise ValueError("pseudo-future contains no eligible new-item truths")
    construction = prepared["construction"]
    clv, _ = build_historical_clv(
        construction, n_users=prepared["n_users"], shuffle_seed=cfg.segment_seed
    )
    features = _binary_history_features(prepared, clv)
    redundancy = _redundancy_diagnostic(features, cfg.segment_seed)
    segments = _assign_segments(clv)
    construction_pairs = construction[["u_idx", "i_idx"]].drop_duplicates()
    seen = {
        int(user): np.sort(rows.i_idx.unique()).astype(np.int32)
        for user, rows in construction.groupby("u_idx", sort=False)
    }
    eval_users = np.asarray(sorted(truth), dtype=np.int32)
    segment_key_by_model = {
        "global_popularity": "global",
        "n_decile_popularity": "n_decile",
        "clv_decile_popularity": "clv_decile",
        "n_matched_clv_popularity": "n_matched_clv",
    }
    rankings_by_model = {
        model_id: _rank_segment_popularity(
            construction_pairs,
            segment_by_user=segments[segment_key],
            eval_users=eval_users,
            seen_items=seen,
            n_items=prepared["n_items"],
            top_k=cfg.rank_limit,
        )
        for model_id, segment_key in segment_key_by_model.items()
    }
    rows = []
    per_user = {}
    for model_id, rankings in rankings_by_model.items():
        metrics, user_metrics = evaluate_transition_ranking(
            rankings, truth=truth, n_items=prepared["n_items"]
        )
        rows.append({"model_id": model_id, **metrics})
        per_user[model_id] = user_metrics
    results = pd.DataFrame(rows)
    indexed = results.set_index("model_id")
    global_reference = indexed.loc["global_popularity"].to_dict()
    n_reference = indexed.loc["n_decile_popularity"].to_dict()
    results["balanced_index_vs_global"] = [
        1.0
        if row["model_id"] == "global_popularity"
        else _balanced_accuracy_index(row.to_dict(), global_reference)
        for _, row in results.iterrows()
    ]
    results["balanced_index_vs_n_decile"] = [
        _balanced_accuracy_index(row.to_dict(), n_reference)
        for _, row in results.iterrows()
    ]
    bootstrap = pd.DataFrame(
        [
            _paired_bootstrap_balanced_index(
                per_user["clv_decile_popularity"],
                per_user["global_popularity"],
                comparison_id="clv_decile_popularity_vs_global_popularity",
                seed=cfg.segment_seed,
            ),
            _paired_bootstrap_balanced_index(
                per_user["n_matched_clv_popularity"],
                per_user["n_decile_popularity"],
                comparison_id="n_matched_clv_popularity_vs_n_decile_popularity",
                seed=cfg.segment_seed,
            ),
        ]
    )

    distribution = pd.DataFrame(
        [
            _weighted_segment_js(
                truth,
                segment_by_user=segments["clv_decile"],
                parent_by_user=segments["global"],
                n_items=prepared["n_items"],
                comparison_id="clv_decile_vs_global",
            ),
            _weighted_segment_js(
                truth,
                segment_by_user=segments["n_decile"],
                parent_by_user=segments["global"],
                n_items=prepared["n_items"],
                comparison_id="n_decile_vs_global",
            ),
            _weighted_segment_js(
                truth,
                segment_by_user=segments["n_matched_clv"],
                parent_by_user=segments["n_decile"],
                n_items=prepared["n_items"],
                comparison_id="clv_within_n_decile_vs_n_decile",
            ),
        ]
    )

    construction_keys = set(
        zip(construction.u_idx.astype(int), construction.i_idx.astype(int))
    )
    truth_violations = sum(
        (int(user), int(item)) in construction_keys
        for user, items in truth.items()
        for item in items
    )
    recommendation_violations = {
        model_id: sum(
            (int(user), int(item)) in construction_keys
            for user, ranked in rankings.items()
            for item in ranked
        )
        for model_id, rankings in rankings_by_model.items()
    }
    quality = pd.DataFrame(
        [
            {
                "check": "transactions_after_day_669",
                "value": int((prepared["transactions"].t > cfg.evaluation_end_day).sum()),
                "passed": bool((prepared["transactions"].t <= cfg.evaluation_end_day).all()),
            },
            {
                "check": "truth_excludes_construction_pairs",
                "value": truth_violations,
                "passed": truth_violations == 0,
            },
            *[
                {
                    "check": f"{model_id}_recommendations_exclude_construction_pairs",
                    "value": violations,
                    "passed": violations == 0,
                }
                for model_id, violations in recommendation_violations.items()
            ],
            {
                "check": "feature_users_reconcile",
                "value": abs(len(features) - prepared["n_users"]),
                "passed": len(features) == prepared["n_users"],
            },
            {
                "check": "historical_clv_coefficient_mean_one",
                "value": abs(float(clv.coefficient.mean()) - 1.0),
                "passed": bool(np.isclose(clv.coefficient.mean(), 1.0)),
            },
        ]
    )
    quality_passed = bool(quality.passed.all())

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_hash = _run_hash(cfg, input_manifest, source_revision)
    prefix = out_dir / f"clv_information_ceiling_{run_hash}"
    paths = {
        "oracle_csv": str(prefix.with_suffix(".csv")),
        "redundancy_csv": str(prefix.parent / f"{prefix.name}_redundancy.csv"),
        "distribution_csv": str(prefix.parent / f"{prefix.name}_distribution.csv"),
        "bootstrap_csv": str(prefix.parent / f"{prefix.name}_bootstrap.csv"),
        "quality_csv": str(prefix.parent / f"{prefix.name}_quality.csv"),
        "json": str(prefix.with_suffix(".json")),
    }
    results.to_csv(paths["oracle_csv"], index=False)
    redundancy.to_csv(paths["redundancy_csv"], index=False)
    distribution.to_csv(paths["distribution_csv"], index=False)
    bootstrap.to_csv(paths["bootstrap_csv"], index=False)
    quality.to_csv(paths["quality_csv"], index=False)
    payload = {
        "code_version": CODE_VERSION,
        "config": asdict(cfg),
        "metric_contract": metric_contract(cfg),
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
        "redundancy_diagnostic": redundancy.to_dict(orient="records"),
        "oracle_results": results.to_dict(orient="records"),
        "distribution_diagnostic": distribution.to_dict(orient="records"),
        "paired_user_bootstrap": bootstrap.to_dict(orient="records"),
        "quality_passed": quality_passed,
        "quality_checks": quality.to_dict(orient="records"),
        "result_paths": paths,
        "limitations": [
            "binary-history summaries are proxies, not learned M1 embeddings",
            "segment popularity is an information-ceiling diagnostic, not a proposed recommender",
            "one historical pseudo-future interval; no confidence interval or causal claim",
        ],
    }
    Path(paths["json"]).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    results.attrs["quality_passed"] = quality_passed
    results.attrs["result_paths"] = paths
    results.attrs["redundancy"] = redundancy
    results.attrs["distribution"] = distribution
    results.attrs["bootstrap"] = bootstrap
    return results


def run_clv_information_ceiling_diagnostic(
    cfg: CLVInformationCeilingConfig,
) -> pd.DataFrame:
    cfg = configure_clv_information_ceiling_diagnostic(**asdict(cfg))
    schema = v3.SCHEMA[cfg.dataset]
    transactions = v3.load_transactions(schema)
    return run_clv_information_ceiling_diagnostic_from_frames(
        transactions,
        cfg,
        input_manifest=moe.build_input_manifest(schema),
        source_revision=moe.source_revision(),
    )
