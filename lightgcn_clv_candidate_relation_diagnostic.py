"""Checkpoint-only candidate-relation diagnostic for Dunnhumby and H&M.

For each user, pair every M1 Top-10 false positive with every held-out
new-item truth missed by M1.  Two train-only candidate relations are checked:

* N relation: user-cross-fitted first-acquisition category-transition lift
  from the user's final training basket to the candidate category;
* V relation: proximity between the candidate's price percentile and the
  user's purchase-amount-weighted historical price position.

There is no training, checkpoint selection, reranking, test, or holdout.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
import torch

from clv_run_state import file_sha256
import clv_m3_clv_conditioned_category_transition_graph as transition
import lightgcn_clv_axis_specific_gate_hm2y as hm2y
import lightgcn_clv_axis_specific_test10 as test10
import lightgcn_clv_fixed_segment_error_diagnostic as fixed
import lightgcn_clv_fixed_segment_error_diagnostic_hm2y as fixed_hm2y
import lightgcn_clv_gatefree_lowdim as gatefree
import lightgcn_clv_gatefree_lowdim_diagnostic as checkpoint_common
import lightgcn_clv_history_item_fit_diagnostic as item_fit
import lightgcn_clv_joint_nv as joint
import lightgcn_clv_v3 as v3


CODE_VERSION = "m1-clv-candidate-relation-diagnostic-v1"
SIGNAL_ORDER = (
    "activity_first_acquisition_category_lift",
    "value_overall_price_fit",
    "value_within_category_price_fit",
)
GROUP_ORDERS = {
    "fixed_clv_segment": fixed.SEGMENT_ORDER,
    "nv_quadrant": fixed.NV_QUADRANT_ORDER,
    "high_clv_composition": fixed.HIGH_CLV_COMPOSITION_ORDER,
}


@dataclass(frozen=True)
class CandidateRelationDiagnosticConfig:
    dataset: str
    out_dir: str = ""
    baseline_result_dir: str = ""
    m1_checkpoint_dir: str = ""
    m1_checkpoint: str = ""
    eval_batch_size: int = 64
    cross_fit_folds: int = 5
    min_transition_support_users: int = 5
    transition_kappa: float = 20.0
    transition_log_lift_cap: float = float(np.log(3.0))


def configure_candidate_relation_diagnostic(
    dataset: str = "dunnhumby", **overrides
) -> CandidateRelationDiagnosticConfig:
    dataset = dataset.lower()
    if dataset not in {"dunnhumby", "hm"}:
        raise ValueError("dataset은 dunnhumby 또는 hm이어야 합니다")
    if dataset == "dunnhumby":
        defaults = gatefree.configure_gatefree_lowdim_run()
        values = {
            "dataset": dataset,
            "out_dir": (
                f"{v3.default_out_dir('dunnhumby')}"
                "_m1_clv_candidate_relation_diagnostic_v1"
            ),
            "baseline_result_dir": defaults.baseline_result_dir,
            "eval_batch_size": 32,
        }
    else:
        values = {
            "dataset": dataset,
            "out_dir": (
                f"{v3.default_out_dir('hm')}"
                "_m1_clv_candidate_relation_diagnostic_hm2y_v1"
            ),
            "m1_checkpoint_dir": v3.default_out_dir("hm"),
            "eval_batch_size": 256,
        }
    values.update(overrides)
    cfg = CandidateRelationDiagnosticConfig(**values)
    if not cfg.out_dir:
        raise ValueError("out_dir가 필요합니다")
    if dataset == "dunnhumby" and not (
        cfg.baseline_result_dir or cfg.m1_checkpoint
    ):
        raise ValueError("Dunnhumby M1 checkpoint 위치가 필요합니다")
    if dataset == "hm" and not (cfg.m1_checkpoint_dir or cfg.m1_checkpoint):
        raise ValueError("H&M M1 checkpoint 위치가 필요합니다")
    if cfg.eval_batch_size <= 0 or cfg.cross_fit_folds < 2:
        raise ValueError("배치크기는 양수, 교차추정 fold는 2 이상이어야 합니다")
    if cfg.min_transition_support_users <= 0 or cfg.transition_kappa < 0:
        raise ValueError("전이 지지도와 kappa는 음수일 수 없습니다")
    if cfg.transition_kappa == 0:
        raise ValueError("빈 전이행도 안정적으로 처리하도록 kappa는 양수여야 합니다")
    return cfg


def preflight_summary(cfg: CandidateRelationDiagnosticConfig) -> dict:
    split = (
        "historical_development_days_684_690"
        if cfg.dataset == "dunnhumby"
        else "existing_hm2y_validation"
    )
    return {
        "code_version": CODE_VERSION,
        "dataset": cfg.dataset,
        "training": False,
        "checkpoint_selection": False,
        "model": "existing seed-42 M1@64 checkpoint",
        "split": split,
        "new_item_task": True,
        "final_test_executed": False,
        "holdout_executed": False,
        "fixed_clv_source": "training-window historical N×V proxy",
        "pair_definition": (
            "within each user, every M1 Top-10 false positive is paired with "
            "every held-out new-item truth missed by M1 Top-10"
        ),
        "signals": {
            SIGNAL_ORDER[0]: (
                "five-fold user-cross-fitted first-acquisition category "
                "transition log-lift from the user's final train basket"
            ),
            SIGNAL_ORDER[1]: (
                "negative absolute distance from the user's train purchase-"
                "amount-weighted overall item-price percentile"
            ),
            SIGNAL_ORDER[2]: (
                "negative absolute distance from the user's train purchase-"
                "amount-weighted within-category item-price percentile"
            ),
        },
        "repeatshare_used": False,
        "raw_item_degree_used": False,
        "reading_rule": (
            "a relation is a common M2 candidate only if its pair-balanced "
            "win rate is above 0.5 overall and for fixed high-CLV users in "
            "both Dunnhumby and H&M"
        ),
        "statistical_note": (
            "single-checkpoint descriptive development diagnostic; no "
            "statistical significance or generalization claim"
        ),
        "out_dir": cfg.out_dir,
    }


def _prepare_and_load(cfg: CandidateRelationDiagnosticConfig):
    if cfg.dataset == "dunnhumby":
        runner_cfg = gatefree.configure_gatefree_lowdim_run(
            out_dir=cfg.out_dir,
            baseline_result_dir=cfg.baseline_result_dir,
        )
        prepared = gatefree._prepare(runner_cfg)
        model, checkpoint, record = checkpoint_common._load_m1(prepared, cfg)
        axes = prepared["axes"]
    else:
        runner_cfg = hm2y.configure_axis_specific_gate_hm2y_run(
            out_dir=cfg.out_dir,
            m1_checkpoint_dir=cfg.m1_checkpoint_dir,
        )
        prepared = joint._prepare(runner_cfg)
        model, checkpoint = fixed_hm2y._load_existing_m1(prepared, cfg)
        record = {}
        axes = fixed_hm2y.build_purchase_occasion_axes(
            prepared["data"]["train"], prepared["data"]["n_users"]
        )
    return prepared, model, checkpoint, record, axes


def _membership(prepared: dict, axes: dict) -> tuple[pd.DataFrame, dict]:
    clv = np.asarray(axes["clv_proxy"], dtype=np.float64)
    valid = np.asarray(axes["valid_user"], dtype=bool) & np.isfinite(clv)
    low_edge, high_edge = prepared["base_cfg"]["SEG_EDGES"]
    thresholds = tuple(
        float(value) for value in np.quantile(clv[valid], [low_edge, high_edge])
    )
    return fixed.build_user_value_groups(
        axes,
        clv_thresholds=thresholds,
        evaluation_users=prepared["cache"].users,
    )


def _item_categories(train: pd.DataFrame, n_items: int) -> tuple[np.ndarray, int]:
    mapping = train.groupby("i_idx", sort=True)["cat_idx"].agg(
        lambda values: values.iloc[0] if values.nunique() == 1 else -1
    )
    mapping = mapping.reindex(np.arange(n_items))
    if mapping.isna().any() or (mapping < 0).any():
        raise ValueError("모든 train 상품은 하나의 카테고리에 속해야 합니다")
    categories = mapping.to_numpy(np.int64)
    return categories, int(categories.max(initial=-1) + 1)


def _cross_fitted_category_lift_scores(
    *,
    pair: pd.DataFrame,
    recent: pd.DataFrame,
    evaluation_users: np.ndarray,
    n_users: int,
    n_categories: int,
    folds: int,
    min_support_users: int,
    kappa: float,
    log_lift_cap: float,
) -> tuple[np.ndarray, dict]:
    """Return one candidate-category score row per evaluation user."""
    evaluation_users = np.asarray(evaluation_users, dtype=np.int64)
    if pair.empty:
        return np.zeros((len(evaluation_users), n_categories)), {
            "transition_evidence_users": 0,
            "transition_rows": 0,
            "active_score_user_share": 0.0,
        }
    user = pair["u_idx"].to_numpy(np.int64)
    source = pair["c_idx"].to_numpy(np.int64)
    target = pair["d_idx"].to_numpy(np.int64)
    mass = pair["mass"].to_numpy(np.float64)
    flat = source * n_categories + target
    size = n_categories * n_categories
    total_mass = np.bincount(flat, weights=mass, minlength=size)
    total_support = np.bincount(flat, minlength=size).astype(np.int64)
    fold_mass = np.zeros((folds, size), dtype=np.float64)
    fold_support = np.zeros((folds, size), dtype=np.int64)
    user_fold = user % folds
    for fold in range(folds):
        selected = user_fold == fold
        fold_mass[fold] = np.bincount(
            flat[selected], weights=mass[selected], minlength=size
        )
        fold_support[fold] = np.bincount(flat[selected], minlength=size)

    position = np.full(n_users, -1, dtype=np.int64)
    position[evaluation_users] = np.arange(len(evaluation_users), dtype=np.int64)
    recent_user = recent["u_idx"].to_numpy(np.int64)
    recent_position = position[recent_user]
    use_recent = recent_position >= 0
    recent_position = recent_position[use_recent]
    recent_source = recent.loc[use_recent, "c_idx"].to_numpy(np.int64)
    recent_share = recent.loc[use_recent, "recent_share"].to_numpy(np.float64)

    result = np.zeros((len(evaluation_users), n_categories), dtype=np.float64)
    supported_edges = []
    for fold in range(folds):
        reference_mass = (total_mass - fold_mass[fold]).reshape(
            n_categories, n_categories
        )
        reference_support = (total_support - fold_support[fold]).reshape(
            n_categories, n_categories
        )
        supported = reference_support >= min_support_users
        reference_mass = np.where(supported, reference_mass, 0.0)
        target_mass = reference_mass.sum(axis=0)
        target_total = float(target_mass.sum())
        target_prior = (
            target_mass / target_total
            if target_total > 0
            else np.full(n_categories, 1.0 / n_categories)
        )
        row_mass = reference_mass.sum(axis=1)
        smoothed = (reference_mass + kappa * target_prior[None, :]) / (
            row_mass[:, None] + kappa
        )
        lift = np.log(
            np.maximum(smoothed, 1e-12) / np.maximum(target_prior[None, :], 1e-12)
        )
        lift = np.clip(lift, -log_lift_cap, log_lift_cap)
        lift[~supported] = 0.0
        supported_edges.append(int(supported.sum()))

        selected_rows = np.flatnonzero(evaluation_users % folds == fold)
        if not len(selected_rows):
            continue
        profile = np.zeros((len(selected_rows), n_categories), dtype=np.float64)
        local = np.full(len(evaluation_users), -1, dtype=np.int64)
        local[selected_rows] = np.arange(len(selected_rows), dtype=np.int64)
        recent_local = local[recent_position]
        active = recent_local >= 0
        np.add.at(
            profile,
            (recent_local[active], recent_source[active]),
            recent_share[active],
        )
        result[selected_rows] = profile @ lift

    return result, {
        "transition_evidence_users": int(pair["u_idx"].nunique()),
        "transition_rows": int(len(pair)),
        "mean_supported_edges_per_fold": float(np.mean(supported_edges)),
        "active_score_user_share": float(np.any(np.abs(result) > 1e-12, axis=1).mean()),
        "score_std": float(result.std()),
    }


def _percentile(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    result = np.full(len(values), np.nan, dtype=np.float64)
    if valid.any():
        result[valid] = rankdata(values[valid], method="average") / int(valid.sum())
    return result


def _price_inputs(
    train: pd.DataFrame,
    *,
    n_users: int,
    n_items: int,
    item_category: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict]:
    mean_price = (
        train.groupby("i_idx", sort=True)["up"]
        .mean()
        .reindex(np.arange(n_items))
        .to_numpy(np.float64)
    )
    valid_price = np.isfinite(mean_price)
    overall = _percentile(mean_price, valid_price)
    within = np.full(n_items, np.nan, dtype=np.float64)
    for category in np.unique(item_category):
        selected = (item_category == category) & valid_price
        within[selected] = _percentile(
            mean_price[selected], np.ones(selected.sum(), dtype=bool)
        )

    user = train["u_idx"].to_numpy(np.int64, copy=False)
    item = train["i_idx"].to_numpy(np.int64, copy=False)
    value = np.maximum(train["v"].to_numpy(np.float64, copy=True), 0.0)
    valid_row = np.isfinite(value) & np.isfinite(overall[item])
    denominator = np.bincount(
        user[valid_row], weights=value[valid_row], minlength=n_users
    )
    overall_numerator = np.bincount(
        user[valid_row],
        weights=value[valid_row] * overall[item[valid_row]],
        minlength=n_users,
    )
    within_numerator = np.bincount(
        user[valid_row],
        weights=value[valid_row] * within[item[valid_row]],
        minlength=n_users,
    )
    user_overall = np.divide(
        overall_numerator,
        denominator,
        out=np.full(n_users, np.nan),
        where=denominator > 0,
    )
    user_within = np.divide(
        within_numerator,
        denominator,
        out=np.full(n_users, np.nan),
        where=denominator > 0,
    )
    del value
    return {
        "item_overall": overall,
        "item_within": within,
        "user_overall": user_overall,
        "user_within": user_within,
    }, {
        "item_price_valid_share": float(valid_price.mean()),
        "user_value_weighted_price_valid_share": float(np.isfinite(user_overall).mean()),
    }


def _candidate_signal_values(
    *,
    user_position: int,
    user_id: int,
    items: np.ndarray,
    item_category: np.ndarray,
    activity_scores: np.ndarray,
    price: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    items = np.asarray(items, dtype=np.int64)
    return {
        SIGNAL_ORDER[0]: activity_scores[user_position, item_category[items]],
        SIGNAL_ORDER[1]: -np.abs(
            price["item_overall"][items] - price["user_overall"][user_id]
        ),
        SIGNAL_ORDER[2]: -np.abs(
            price["item_within"][items] - price["user_within"][user_id]
        ),
    }


def _user_pair_rows(
    *,
    users: np.ndarray,
    top10: np.ndarray,
    truth: dict[int, np.ndarray],
    membership: pd.DataFrame,
    item_category: np.ndarray,
    activity_scores: np.ndarray,
    price: dict[str, np.ndarray],
) -> pd.DataFrame:
    by_user = membership.set_index("user_idx")
    rows = []
    for position, user in enumerate(np.asarray(users, dtype=np.int64)):
        truth_items = np.asarray(truth[int(user)], dtype=np.int64)
        truth_set = set(map(int, truth_items))
        ranked = np.asarray(top10[position], dtype=np.int64)
        ranked_set = set(map(int, ranked))
        missed = np.asarray(
            [item for item in truth_items if int(item) not in ranked_set],
            dtype=np.int64,
        )
        false_positive = np.asarray(
            [item for item in ranked if int(item) not in truth_set],
            dtype=np.int64,
        )
        if not len(missed) or not len(false_positive):
            continue
        missed_values = _candidate_signal_values(
            user_position=position,
            user_id=int(user),
            items=missed,
            item_category=item_category,
            activity_scores=activity_scores,
            price=price,
        )
        false_values = _candidate_signal_values(
            user_position=position,
            user_id=int(user),
            items=false_positive,
            item_category=item_category,
            activity_scores=activity_scores,
            price=price,
        )
        group = by_user.loc[int(user)]
        for signal in SIGNAL_ORDER:
            difference = (
                missed_values[signal][:, None] - false_values[signal][None, :]
            ).reshape(-1)
            difference = difference[np.isfinite(difference)]
            if not len(difference):
                continue
            tolerance = 1e-12
            wins = int(np.sum(difference > tolerance))
            losses = int(np.sum(difference < -tolerance))
            ties = int(len(difference) - wins - losses)
            rows.append(
                {
                    "user_idx": int(user),
                    "signal": signal,
                    "fixed_clv_segment": group["fixed_clv_segment"],
                    "nv_quadrant": group["nv_quadrant"],
                    "high_clv_composition": group["high_clv_composition"],
                    "q_n": float(group["q_n"]),
                    "q_v": float(group["q_v"]),
                    "historical_clv_proxy": float(group["historical_clv_proxy"]),
                    "missed_truth_count": int(len(missed)),
                    "false_positive_count": int(len(false_positive)),
                    "candidate_pair_count": int(len(difference)),
                    "truth_wins": wins,
                    "ties": ties,
                    "false_positive_wins": losses,
                    "balanced_win_rate": (wins + 0.5 * ties) / len(difference),
                    "strict_win_rate": wins / len(difference),
                    "mean_pair_score_difference": float(difference.mean()),
                }
            )
    return pd.DataFrame(rows)


def summarize_pair_directions(per_user: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_specs = [("overall", "전체", np.ones(len(per_user), dtype=bool))]
    for column, order in GROUP_ORDERS.items():
        values = per_user[column].to_numpy()
        for group in order:
            mask = values == group
            if mask.any():
                group_specs.append((column, group, mask))
    for signal in SIGNAL_ORDER:
        signal_mask = per_user.signal.eq(signal).to_numpy()
        for group_type, group, group_mask in group_specs:
            selected = per_user.loc[signal_mask & group_mask]
            if selected.empty:
                continue
            pairs = int(selected.candidate_pair_count.sum())
            wins = int(selected.truth_wins.sum())
            ties = int(selected.ties.sum())
            losses = int(selected.false_positive_wins.sum())
            rows.append(
                {
                    "signal": signal,
                    "group_type": group_type,
                    "group": group,
                    "n_users": int(selected.user_idx.nunique()),
                    "candidate_pair_count": pairs,
                    "truth_wins": wins,
                    "ties": ties,
                    "false_positive_wins": losses,
                    "pair_balanced_win_rate": (wins + 0.5 * ties) / pairs,
                    "pair_strict_win_rate": wins / pairs,
                    "pair_tie_rate": ties / pairs,
                    "macro_user_balanced_win_rate": float(
                        selected.balanced_win_rate.mean()
                    ),
                    "mean_pair_score_difference": float(
                        np.average(
                            selected.mean_pair_score_difference,
                            weights=selected.candidate_pair_count,
                        )
                    ),
                }
            )
    return pd.DataFrame(rows)


def _screen_reading(summary: pd.DataFrame) -> dict:
    readings = {}
    for signal in SIGNAL_ORDER:
        selected = summary[summary.signal.eq(signal)].set_index(
            ["group_type", "group"]
        )
        overall = float(selected.at[("overall", "전체"), "pair_balanced_win_rate"])
        high = float(
            selected.at[
                ("fixed_clv_segment", fixed.SEGMENT_ORDER[2]),
                "pair_balanced_win_rate",
            ]
        )
        readings[signal] = {
            "overall_pair_balanced_win_rate": overall,
            "high_clv_pair_balanced_win_rate": high,
            "direction_supported_in_this_dataset": bool(overall > 0.5 and high > 0.5),
        }
    return {
        "signals": readings,
        "cross_dataset_decision_pending": True,
        "rule": (
            "use a relation in M2 only after the same signal is above 0.5 "
            "overall and in fixed high-CLV users in both datasets"
        ),
    }


def _paths(cfg: CandidateRelationDiagnosticConfig) -> dict[str, Path]:
    root = Path(cfg.out_dir)
    root.mkdir(parents=True, exist_ok=True)
    stem = f"m1_candidate_relation_{cfg.dataset}"
    return {
        "relation_summary_csv": root / f"{stem}_summary.csv",
        "per_user_csv": root / f"{stem}_per_user.csv",
        "json": root / f"{stem}_diagnostic.json",
    }


@torch.no_grad()
def run_candidate_relation_diagnostic(
    cfg: CandidateRelationDiagnosticConfig | None = None,
) -> dict[str, str]:
    cfg = cfg or configure_candidate_relation_diagnostic()
    print(json.dumps(preflight_summary(cfg), ensure_ascii=False, indent=2))
    prepared, model, checkpoint, record, axes = _prepare_and_load(cfg)
    model.eval()
    users = np.asarray(prepared["cache"].users, dtype=np.int64)
    user_embedding, item_embedding = model.propagate_pref()
    ranked_users, top10 = item_fit._masked_topk(
        user_embedding,
        item_embedding,
        prepared,
        max_k=10,
        batch_size=cfg.eval_batch_size,
    )
    if not np.array_equal(ranked_users, users):
        raise RuntimeError("M1 평가 사용자 순서가 달라졌습니다")
    del user_embedding, item_embedding

    membership, thresholds = _membership(prepared, axes)
    train = prepared["data"]["train"]
    n_users = int(prepared["data"]["n_users"])
    n_items = int(prepared["data"]["n_items"])
    item_category, n_categories = _item_categories(train, n_items)
    pair, recent, transition_diagnostics = transition._transition_evidence(
        train, n_users
    )
    if recent.columns.duplicated().any():
        # The shared graph builder historically emitted u_idx twice.  Keep the
        # first copy locally without changing the already frozen M3 runner.
        recent = recent.loc[:, ~recent.columns.duplicated()].copy()
    activity_scores, activity_diagnostics = _cross_fitted_category_lift_scores(
        pair=pair,
        recent=recent,
        evaluation_users=users,
        n_users=n_users,
        n_categories=n_categories,
        folds=cfg.cross_fit_folds,
        min_support_users=cfg.min_transition_support_users,
        kappa=cfg.transition_kappa,
        log_lift_cap=cfg.transition_log_lift_cap,
    )
    price, price_diagnostics = _price_inputs(
        train,
        n_users=n_users,
        n_items=n_items,
        item_category=item_category,
    )
    per_user = _user_pair_rows(
        users=users,
        top10=top10,
        truth=prepared["cache"].gt,
        membership=membership,
        item_category=item_category,
        activity_scores=activity_scores,
        price=price,
    )
    if per_user.empty:
        raise RuntimeError("비교 가능한 M1 누락정답–오추천 쌍이 없습니다")
    summary = summarize_pair_directions(per_user)
    paths = _paths(cfg)
    test10._atomic_csv(paths["relation_summary_csv"], summary)
    test10._atomic_csv(paths["per_user_csv"], per_user)
    payload = {
        "code_version": CODE_VERSION,
        "config": asdict(cfg),
        "preflight": preflight_summary(cfg),
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": file_sha256(checkpoint),
            "record": record,
        },
        "group_thresholds": thresholds,
        "transition_diagnostics": {
            **transition_diagnostics,
            **activity_diagnostics,
        },
        "price_diagnostics": price_diagnostics,
        "screen_reading": _screen_reading(summary),
        "relation_summary": summary.to_dict("records"),
        "result_paths": {name: str(path) for name, path in paths.items()},
        "interpretation_limits": [
            "checkpoint-only descriptive development diagnostic",
            "no model training, checkpoint selection, test, or holdout",
            "historical CLV proxy is not future or incremental CLV",
            "pair win rates are not recommendation accuracy metrics",
            "no statistical significance or cross-dataset generalization claim",
        ],
    }
    test10._atomic_json(paths["json"], payload)
    return {name: str(path) for name, path in paths.items()}
