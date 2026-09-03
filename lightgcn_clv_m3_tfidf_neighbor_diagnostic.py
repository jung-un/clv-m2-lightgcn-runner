"""Train-only multi-anchor mechanism check for the TF-IDF neighbor M3."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from clv_m3_tfidf_neighbor_graph import (
    build_degree_matched_random_neighbor_operator,
    build_historical_clv_gates,
    build_ordinary_copurchase_operator,
    build_tfidf_neighbor_operator,
    top_candidate_items,
)
import lightgcn_clv_v3 as v3


CODE_VERSION = "m3-clv-tfidf-neighbor-train-only-diagnostic-v1"
TFIDF = "tfidf_topk_neighbor"
ORDINARY = "ordinary_copurchase_propagation"
RANDOM = "degree_matched_random_neighbor"
COMPARATORS = (ORDINARY, RANDOM)


@dataclass(frozen=True)
class TFIDFNeighborDiagnosticConfig:
    dataset: str = "dunnhumby"
    seed: int = 42
    time_cutoff: int = 690
    train_end: int = 683
    horizon_days: int = 7
    n_anchors: int = 5
    top_k_neighbors: int = 20
    candidate_count: int = 100
    degree_bins: int = 10
    out_dir: str = ""


def configure_tfidf_neighbor_diagnostic(
    **overrides,
) -> TFIDFNeighborDiagnosticConfig:
    defaults = {
        "out_dir": (
            f"{v3.default_out_dir('dunnhumby')}"
            "_m3_clv_tfidf_neighbor_residual_historical_screen_v1"
            "/train_only_mechanism_diagnostic"
        )
    }
    return validate_diagnostic_config(
        TFIDFNeighborDiagnosticConfig(**(defaults | overrides))
    )


def validate_diagnostic_config(
    cfg: TFIDFNeighborDiagnosticConfig,
) -> TFIDFNeighborDiagnosticConfig:
    fixed = {
        "dataset": "dunnhumby",
        "seed": 42,
        "time_cutoff": 690,
        "train_end": 683,
        "horizon_days": 7,
        "n_anchors": 5,
        "top_k_neighbors": 20,
        "candidate_count": 100,
        "degree_bins": 10,
    }
    for name, expected in fixed.items():
        if getattr(cfg, name) != expected:
            raise ValueError(f"M3 관계 진단은 {name}={expected!r}이어야 합니다")
    if not cfg.out_dir:
        raise ValueError("out_dir is required")
    return cfg


def historical_anchor_ends(
    train_end: int, *, horizon_days: int, n_anchors: int
) -> list[int]:
    if min(horizon_days, n_anchors) <= 0:
        raise ValueError("horizon_days and n_anchors must be positive")
    return [
        int(train_end - horizon_days * remaining)
        for remaining in range(n_anchors, 0, -1)
    ]


def build_anchor_truth(
    history: pd.DataFrame,
    *,
    anchor_end: int,
    horizon_days: int,
    n_users: int,
    n_items: int,
) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    past = history.loc[history["t"].le(anchor_end)].copy()
    future = history.loc[
        history["t"].gt(anchor_end)
        & history["t"].le(anchor_end + horizon_days)
    ].copy()
    if past.empty:
        return past, {}
    past_users = set(past["u_idx"].unique())
    past_items = set(past["i_idx"].unique())
    future = future.loc[
        future["u_idx"].isin(past_users) & future["i_idx"].isin(past_items)
    ]
    past_key = np.unique(
        past["u_idx"].to_numpy(np.int64) * n_items
        + past["i_idx"].to_numpy(np.int64)
    )
    future = future[["u_idx", "i_idx"]].drop_duplicates()
    if len(future):
        future_key = (
            future["u_idx"].to_numpy(np.int64) * n_items
            + future["i_idx"].to_numpy(np.int64)
        )
        future = future.loc[~np.isin(future_key, past_key)]
    truth = {
        int(user): group["i_idx"].to_numpy(np.int32)
        for user, group in future.groupby("u_idx", sort=False)
    }
    return past, truth


def _candidate_recall(candidates: np.ndarray, truth: np.ndarray) -> float:
    if not len(truth):
        return 0.0
    return float(np.isin(truth, candidates).sum() / len(truth))


def _safe_spearman(left: pd.Series, right: pd.Series) -> float:
    left = left.to_numpy(np.float64)
    right = right.to_numpy(np.float64)
    if len(left) < 2 or left.std() <= 1e-12 or right.std() <= 1e-12:
        return 0.0
    value = spearmanr(left, right).statistic
    return float(value) if np.isfinite(value) else 0.0


def mechanism_reading(user_rows: pd.DataFrame) -> dict:
    required = {
        "anchor_end",
        "u_idx",
        "clv_group",
        "degree_stratum",
        "eligible_tfidf",
        "q_clv",
        TFIDF,
        ORDINARY,
        RANDOM,
    }
    missing = required - set(user_rows)
    if missing:
        raise ValueError(f"diagnostic user rows are missing columns: {sorted(missing)}")
    if user_rows.empty:
        raise ValueError("diagnostic needs at least one evaluation user")
    comparisons = {}
    passed = True
    for comparator in COMPARATORS:
        delta = user_rows[TFIDF] - user_rows[comparator]
        high = user_rows["clv_group"].eq("high")
        anchor_high = (
            user_rows.assign(_delta=delta)
            .loc[high]
            .groupby("anchor_end", sort=True)["_delta"]
            .mean()
        )
        degree_mean = (
            user_rows.assign(_delta=delta)
            .groupby(["anchor_end", "degree_stratum"], sort=True)["_delta"]
            .mean()
            .mean()
        )
        overall_mean = float(delta.mean())
        high_mean = float(delta[high].mean()) if high.any() else float("nan")
        positive_anchors = int((anchor_high > 0).sum())
        checks = {
            "overall_positive": bool(overall_mean > 0),
            "high_clv_positive": bool(np.isfinite(high_mean) and high_mean > 0),
            "high_clv_not_below_overall": bool(
                np.isfinite(high_mean) and high_mean >= overall_mean
            ),
            "high_clv_positive_in_anchor_majority": bool(
                positive_anchors > len(anchor_high) / 2
            ),
            "degree_stratified_positive": bool(degree_mean > 0),
        }
        passed = passed and all(checks.values())
        comparisons[comparator] = {
            "overall_mean_delta": overall_mean,
            "high_clv_mean_delta": high_mean,
            "degree_stratified_mean_delta": float(degree_mean),
            "positive_high_clv_anchor_count": positive_anchors,
            "n_anchor_high_clv_rows": int(len(anchor_high)),
            "spearman_clv_delta": _safe_spearman(
                user_rows["q_clv"], delta
            ),
            "checks": checks,
        }
    return {
        "automatic_model_selection": False,
        "analysis_type": "train-only multi-anchor relationship diagnostic",
        "precheck_passed": bool(passed),
        "comparisons": comparisons,
        "eligible_tfidf_user_share": float(user_rows["eligible_tfidf"].mean()),
        "n_user_anchor_rows": int(len(user_rows)),
        "n_anchors": int(user_rows["anchor_end"].nunique()),
        "routing": (
            "Run the fixed seed-42 M3 screen."
            if passed
            else "Stop before high-cost M3 training; the taste relation did not pass."
        ),
        "limitation": (
            "train-only mechanism evidence; no performance, significance, or "
            "generalization claim"
        ),
    }


def _base_config(cfg: TFIDFNeighborDiagnosticConfig) -> dict:
    return dict(
        v3.configure_run(
            cfg.dataset,
            out_dir=cfg.out_dir,
            ARCH="pref_only",
            SEED_LIST=[cfg.seed],
            WINDOW_DAYS=None,
            TIME_CUTOFF=cfg.time_cutoff,
            TRAIN_ON_VAL=True,
            VAL_DAYS=7,
            TEST_DAYS=7,
            HOLDOUT_DAYS=0,
            EVAL_TEST=False,
            EVAL_HOLDOUT=False,
            GRAPH_MODE="binary",
            LOSS_MODE="plain",
            NEG_MODE="uniform",
            GATE_MODE="none",
            MIN_USER_INTER=1,
            MIN_ITEM_INTER=1,
            DIM=64,
            N_LAYERS=2,
            EPOCHS=100,
            EARLY_STOP=100,
            REPORT_LEGACY_VALUE_FEATURES=False,
        )
    )


def run_tfidf_neighbor_mechanism_diagnostic(
    cfg: TFIDFNeighborDiagnosticConfig | None = None,
) -> pd.DataFrame:
    cfg = validate_diagnostic_config(
        cfg or configure_tfidf_neighbor_diagnostic()
    )
    preflight = {
        "code_version": CODE_VERSION,
        "config": asdict(cfg),
        "official_test_read": False,
        "holdout_constructed": False,
        "training": False,
        "checkpoint_selection": False,
        "truth_definition": (
            "new-to-user items in each next 7-day train-only window; "
            "item must already exist at the anchor"
        ),
    }
    print(json.dumps(preflight, ensure_ascii=False, indent=2))
    data = v3.prepare_data(_base_config(cfg), v3.DCFG)
    history = data["train"].copy()
    if float(history["t"].max()) != cfg.train_end:
        raise RuntimeError("train-only diagnostic received the wrong train boundary")

    user_records: list[dict] = []
    relation_records: list[dict] = []
    for anchor in historical_anchor_ends(
        cfg.train_end,
        horizon_days=cfg.horizon_days,
        n_anchors=cfg.n_anchors,
    ):
        past, truth = build_anchor_truth(
            history,
            anchor_end=anchor,
            horizon_days=cfg.horizon_days,
            n_users=data["n_users"],
            n_items=data["n_items"],
        )
        gates = build_historical_clv_gates(
            past,
            data["n_users"],
            shuffle_degree_bins=cfg.degree_bins,
            shuffle_seed=cfg.seed,
        )
        tfidf, tfidf_diag = build_tfidf_neighbor_operator(
            past,
            data["n_users"],
            data["n_items"],
            top_k=cfg.top_k_neighbors,
        )
        ordinary, ordinary_diag = build_ordinary_copurchase_operator(
            past,
            data["n_users"],
            data["n_items"],
            top_k=cfg.top_k_neighbors,
        )
        degree = (
            past[["u_idx", "i_idx"]]
            .drop_duplicates()
            .groupby("u_idx")
            .size()
            .reindex(np.arange(data["n_users"]), fill_value=0)
            .to_numpy(np.int64)
        )
        random, random_diag = build_degree_matched_random_neighbor_operator(
            degree,
            top_k=cfg.top_k_neighbors,
            n_bins=cfg.degree_bins,
            seed=cfg.seed + anchor,
        )
        operators = {TFIDF: tfidf, ORDINARY: ordinary, RANDOM: random}
        candidates = {
            name: top_candidate_items(
                operator,
                past,
                data["n_users"],
                data["n_items"],
                candidate_count=cfg.candidate_count,
            )
            for name, operator in operators.items()
        }
        eligible = np.diff(tfidf.indptr) > 0
        for user, user_truth in truth.items():
            q_clv = float(gates.clv_percentile[user])
            clv_group = "low" if q_clv <= 1 / 3 else "high" if q_clv >= 2 / 3 else "middle"
            user_records.append(
                {
                    "anchor_end": anchor,
                    "truth_end": anchor + cfg.horizon_days,
                    "u_idx": user,
                    "truth_count": int(len(user_truth)),
                    "q_clv": q_clv,
                    "clv_group": clv_group,
                    "degree": int(degree[user]),
                    "degree_stratum": int(gates.degree_stratum[user]),
                    "eligible_tfidf": bool(eligible[user]),
                    **{
                        name: _candidate_recall(candidate[user], user_truth)
                        for name, candidate in candidates.items()
                    },
                }
            )
        for name, diagnostics in (
            (TFIDF, tfidf_diag),
            (ORDINARY, ordinary_diag),
            (RANDOM, random_diag),
        ):
            relation_records.append(
                {"anchor_end": anchor, "relation": name, **diagnostics}
            )

    user_rows = pd.DataFrame(user_records)
    relation_rows = pd.DataFrame(relation_records)
    reading = mechanism_reading(user_rows)
    summary = (
        user_rows.groupby(["anchor_end", "clv_group"], sort=True)[
            [TFIDF, ORDINARY, RANDOM]
        ]
        .mean()
        .reset_index()
    )
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = "m3_clv_tfidf_neighbor_train_only_diagnostic"
    paths = {
        "user_csv": out_dir / f"{stem}_user.csv",
        "summary_csv": out_dir / f"{stem}_summary.csv",
        "relation_csv": out_dir / f"{stem}_relation.csv",
        "json": out_dir / f"{stem}.json",
    }
    user_rows.to_csv(paths["user_csv"], index=False)
    summary.to_csv(paths["summary_csv"], index=False)
    relation_rows.to_csv(paths["relation_csv"], index=False)
    payload = {
        "code_version": CODE_VERSION,
        "preflight": preflight,
        "reading": reading,
        "relation_rows": relation_rows.to_dict("records"),
        "summary_rows": summary.to_dict("records"),
        "result_paths": {key: str(value) for key, value in paths.items()},
    }
    paths["json"].write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    user_rows.attrs["reading"] = reading
    user_rows.attrs["summary"] = summary.to_dict("records")
    user_rows.attrs["result_paths"] = {
        key: str(value) for key, value in paths.items()
    }
    print(json.dumps(reading, ensure_ascii=False, indent=2))
    print("결과 파일:", user_rows.attrs["result_paths"])
    return user_rows


if __name__ == "__main__":
    run_tfidf_neighbor_mechanism_diagnostic()

