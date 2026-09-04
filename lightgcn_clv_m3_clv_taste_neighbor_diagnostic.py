"""Train-only multi-anchor precheck for CLV-conditioned taste neighbors."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from clv_m3_clv_taste_neighbor_graph import (
    ACTUAL_CLV,
    CLV_SHUFFLE,
    DEGREE_RELATION,
    PREFERENCE_RELATION,
    build_clv_taste_neighbor_graph,
)
from clv_m3_tfidf_neighbor_graph import top_candidate_items
from lightgcn_clv_m3_tfidf_neighbor_diagnostic import (
    build_anchor_truth,
    historical_anchor_ends,
)
import lightgcn_clv_v3 as v3


CODE_VERSION = "m3-clv-conditioned-taste-neighbor-train-only-diagnostic-v1"
COMPARATORS = (PREFERENCE_RELATION, CLV_SHUFFLE, DEGREE_RELATION)


@dataclass(frozen=True)
class CLVTasteNeighborDiagnosticConfig:
    dataset: str = "dunnhumby"
    seed: int = 42
    time_cutoff: int = 690
    train_end: int = 683
    horizon_days: int = 7
    n_anchors: int = 5
    preference_candidate_neighbors: int = 100
    final_neighbors: int = 20
    candidate_items: int = 100
    reliability_kappa: float = 5.0
    degree_bins: int = 10
    out_dir: str = ""


def configure_clv_taste_neighbor_diagnostic(
    **overrides,
) -> CLVTasteNeighborDiagnosticConfig:
    defaults = {
        "out_dir": (
            f"{v3.default_out_dir('dunnhumby')}"
            "_m3_clv_taste_neighbor_historical_screen_v1"
            "/train_only_mechanism_diagnostic"
        )
    }
    return validate_diagnostic_config(
        CLVTasteNeighborDiagnosticConfig(**(defaults | overrides))
    )


def validate_diagnostic_config(
    cfg: CLVTasteNeighborDiagnosticConfig,
) -> CLVTasteNeighborDiagnosticConfig:
    fixed = {
        "dataset": "dunnhumby",
        "seed": 42,
        "time_cutoff": 690,
        "train_end": 683,
        "horizon_days": 7,
        "n_anchors": 5,
        "preference_candidate_neighbors": 100,
        "final_neighbors": 20,
        "candidate_items": 100,
        "reliability_kappa": 5.0,
        "degree_bins": 10,
    }
    for name, expected in fixed.items():
        if getattr(cfg, name) != expected:
            raise ValueError(f"M3 관계 진단은 {name}={expected!r}이어야 합니다")
    if not cfg.out_dir:
        raise ValueError("out_dir is required")
    return cfg


def _candidate_recall(candidates: np.ndarray, truth: np.ndarray) -> float:
    if not len(truth):
        return 0.0
    return float(np.isin(truth, candidates).sum() / len(truth))


def _safe_spearman(left: pd.Series, right: pd.Series) -> float:
    left_values = left.to_numpy(np.float64)
    right_values = right.to_numpy(np.float64)
    if (
        len(left_values) < 2
        or left_values.std() <= 1e-12
        or right_values.std() <= 1e-12
    ):
        return 0.0
    value = spearmanr(left_values, right_values).statistic
    return float(value) if np.isfinite(value) else 0.0


def mechanism_reading(
    user_rows: pd.DataFrame, relation_rows: pd.DataFrame
) -> dict:
    required_user = {
        "anchor_end",
        "u_idx",
        "q_clv",
        "clv_group",
        "degree_stratum",
        ACTUAL_CLV,
        *COMPARATORS,
    }
    required_relation = {
        "anchor_end",
        "is_full_train",
        "quality_passed",
        "same_neighbor_count_all_arms",
        "same_row_mass_all_arms",
        "actual_shuffle_changed_user_share",
    }
    missing_user = required_user - set(user_rows)
    missing_relation = required_relation - set(relation_rows)
    if missing_user or missing_relation:
        raise ValueError(
            "diagnostic rows are missing columns: "
            f"user={sorted(missing_user)}, relation={sorted(missing_relation)}"
        )
    if user_rows.empty:
        raise ValueError("diagnostic needs at least one evaluation user")
    full_train = relation_rows.loc[relation_rows["is_full_train"].astype(bool)]
    if len(full_train) != 1:
        raise ValueError("exactly one full-train relation diagnostic is required")

    comparisons: dict[str, dict] = {}
    segment_diagnostics: dict[str, dict[str, float]] = {}
    comparison_checks: dict[str, bool] = {}
    for comparator in COMPARATORS:
        delta = user_rows[ACTUAL_CLV] - user_rows[comparator]
        anchor_delta = (
            user_rows.assign(_delta=delta)
            .groupby("anchor_end", sort=True)["_delta"]
            .mean()
        )
        degree_stratified = (
            user_rows.assign(_delta=delta)
            .groupby(["anchor_end", "degree_stratum"], sort=True)["_delta"]
            .mean()
            .mean()
        )
        segment = (
            user_rows.assign(_delta=delta)
            .groupby("clv_group", sort=True)["_delta"]
            .mean()
            .to_dict()
        )
        segment = {str(key): float(value) for key, value in segment.items()}
        segment_diagnostics[comparator] = segment
        overall = float(delta.mean())
        positive_anchors = int((anchor_delta > 0).sum())
        checks = {
            "overall_positive": bool(overall > 0),
            "positive_in_at_least_three_of_five_anchors": bool(
                positive_anchors >= 3
            ),
        }
        comparison_checks[comparator] = all(checks.values())
        comparisons[comparator] = {
            "overall_mean_delta": overall,
            "degree_stratified_mean_delta": float(degree_stratified),
            "positive_anchor_count": positive_anchors,
            "anchor_mean_deltas": {
                str(int(key)): float(value) for key, value in anchor_delta.items()
            },
            "spearman_clv_delta": _safe_spearman(user_rows["q_clv"], delta),
            "checks": checks,
        }

    full_changed = float(
        full_train.iloc[0]["actual_shuffle_changed_user_share"]
    )
    structural_quality = bool(
        relation_rows["quality_passed"].astype(bool).all()
        and relation_rows["same_neighbor_count_all_arms"].astype(bool).all()
        and relation_rows["same_row_mass_all_arms"].astype(bool).all()
    )
    checks = {
        **{
            f"actual_beats_{comparator}": value
            for comparator, value in comparison_checks.items()
        },
        "actual_shuffle_neighbor_change_at_least_10pct": bool(
            full_changed >= 0.10
        ),
        "structural_controls_passed": structural_quality,
    }
    passed = all(checks.values())
    return {
        "automatic_model_selection": False,
        "analysis_type": "train-only multi-anchor relationship diagnostic",
        "precheck_passed": bool(passed),
        "checks": checks,
        "comparisons": comparisons,
        "segment_diagnostics": segment_diagnostics,
        "full_train_actual_shuffle_changed_user_share": full_changed,
        "n_user_anchor_rows": int(len(user_rows)),
        "n_anchors": int(user_rows["anchor_end"].nunique()),
        "routing": (
            "Run the fixed seed-42 M3 historical screen."
            if passed
            else "Stop before high-cost M3 training; the CLV-conditioned neighbor relation did not pass."
        ),
        "high_clv_not_used_as_success_guard": True,
        "limitation": (
            "train-only mechanism evidence; no performance, significance, or "
            "generalization claim"
        ),
    }


def _base_config(cfg: CLVTasteNeighborDiagnosticConfig) -> dict:
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


def _relation_record(graph, *, anchor_end: int, is_full_train: bool) -> dict:
    relation = graph.diagnostics["neighbor_relations"]
    return {
        "anchor_end": int(anchor_end),
        "is_full_train": bool(is_full_train),
        "quality_passed": bool(graph.diagnostics["quality_passed"]),
        "same_neighbor_count_all_arms": bool(
            relation["same_neighbor_count_all_arms"]
        ),
        "same_row_mass_all_arms": bool(relation["same_row_mass_all_arms"]),
        "actual_shuffle_changed_user_share": float(
            relation["actual_vs_shuffle"]["set_changed_user_share"]
        ),
        "actual_preference_changed_user_share": float(
            relation["actual_vs_preference"]["set_changed_user_share"]
        ),
        "actual_degree_changed_user_share": float(
            relation["actual_vs_degree"]["set_changed_user_share"]
        ),
        "eligible_user_share": float(
            graph.diagnostics["preference_candidates"]["eligible_user_share"]
        ),
    }


def run_clv_taste_neighbor_mechanism_diagnostic(
    cfg: CLVTasteNeighborDiagnosticConfig | None = None,
) -> pd.DataFrame:
    cfg = validate_diagnostic_config(
        cfg or configure_clv_taste_neighbor_diagnostic()
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
        graph = build_clv_taste_neighbor_graph(
            past,
            data["n_users"],
            data["n_items"],
            candidate_neighbors=cfg.preference_candidate_neighbors,
            final_neighbors=cfg.final_neighbors,
            reliability_kappa=cfg.reliability_kappa,
            degree_bins=cfg.degree_bins,
            shuffle_seed=cfg.seed,
        )
        candidates = {
            arm: top_candidate_items(
                operator,
                past,
                data["n_users"],
                data["n_items"],
                candidate_count=cfg.candidate_items,
            )
            for arm, operator in graph.operators.items()
        }
        for user, user_truth in truth.items():
            q_clv = float(graph.features.q_clv[user])
            group = (
                "low"
                if q_clv <= 1 / 3
                else "high"
                if q_clv >= 2 / 3
                else "middle"
            )
            user_records.append(
                {
                    "anchor_end": int(anchor),
                    "truth_end": int(anchor + cfg.horizon_days),
                    "u_idx": int(user),
                    "truth_count": int(len(user_truth)),
                    "q_clv": q_clv,
                    "clv_group": group,
                    "degree_stratum": int(graph.features.degree_stratum[user]),
                    **{
                        arm: _candidate_recall(candidate[user], user_truth)
                        for arm, candidate in candidates.items()
                    },
                }
            )
        relation_records.append(
            _relation_record(graph, anchor_end=anchor, is_full_train=False)
        )

    full_graph = build_clv_taste_neighbor_graph(
        history,
        data["n_users"],
        data["n_items"],
        candidate_neighbors=cfg.preference_candidate_neighbors,
        final_neighbors=cfg.final_neighbors,
        reliability_kappa=cfg.reliability_kappa,
        degree_bins=cfg.degree_bins,
        shuffle_seed=cfg.seed,
    )
    relation_records.append(
        _relation_record(full_graph, anchor_end=cfg.train_end, is_full_train=True)
    )
    user_rows = pd.DataFrame(user_records)
    relation_rows = pd.DataFrame(relation_records)
    reading = mechanism_reading(user_rows, relation_rows)
    summary = (
        user_rows.groupby(["anchor_end", "clv_group"], sort=True)[
            [PREFERENCE_RELATION, ACTUAL_CLV, CLV_SHUFFLE, DEGREE_RELATION]
        ]
        .mean()
        .reset_index()
    )
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = "m3_clv_taste_neighbor_train_only_diagnostic"
    paths = {
        "user_csv": out_dir / f"{stem}_user.csv",
        "summary_csv": out_dir / f"{stem}_summary.csv",
        "relation_csv": out_dir / f"{stem}_relation.csv",
        "json": out_dir / f"{stem}.json",
    }
    user_rows.to_csv(paths["user_csv"], index=False)
    summary.to_csv(paths["summary_csv"], index=False)
    relation_rows.to_csv(paths["relation_csv"], index=False)
    paths["json"].write_text(
        json.dumps(
            {
                "code_version": CODE_VERSION,
                "preflight": preflight,
                "reading": reading,
                "relation_rows": relation_rows.to_dict("records"),
                "summary_rows": summary.to_dict("records"),
                "result_paths": {key: str(value) for key, value in paths.items()},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    user_rows.attrs["reading"] = reading
    user_rows.attrs["summary"] = summary.to_dict("records")
    user_rows.attrs["relation_rows"] = relation_rows.to_dict("records")
    user_rows.attrs["result_paths"] = {
        key: str(value) for key, value in paths.items()
    }
    print(json.dumps(reading, ensure_ascii=False, indent=2))
    print("결과 파일:", user_rows.attrs["result_paths"])
    return user_rows


if __name__ == "__main__":
    run_clv_taste_neighbor_mechanism_diagnostic()
