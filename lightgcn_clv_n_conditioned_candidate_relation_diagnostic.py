"""Checkpoint-only test of whether train-only N conditions category transition.

Compares pooled, actual-N-conditioned, and degree-stratified shuffled-N
transition scores on M1 Top-10 false positives versus missed new-item truths.
There is no training, reranking, checkpoint selection, final test, or holdout.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from clv_run_state import file_sha256
import clv_m3_clv_conditioned_category_transition_graph as transition
import lightgcn_clv_axis_specific_test10 as test10
import lightgcn_clv_candidate_relation_diagnostic as base
import lightgcn_clv_fixed_segment_error_diagnostic as fixed
import lightgcn_clv_history_item_fit_diagnostic as item_fit


CODE_VERSION = "m1-n-conditioned-candidate-relation-diagnostic-v1"
ARM_ORDER = ("pooled", "actual_n_conditioned", "shuffled_n_conditioned")
GROUP_ORDERS = base.GROUP_ORDERS


@dataclass(frozen=True)
class NConditionedCandidateRelationConfig:
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
    shuffle_degree_bins: int = 10
    shuffle_seed: int = 42


def configure_n_conditioned_candidate_relation_diagnostic(
    dataset: str = "dunnhumby", **overrides
) -> NConditionedCandidateRelationConfig:
    source = base.configure_candidate_relation_diagnostic(dataset)
    values = asdict(source)
    suffix = (
        "_m1_n_conditioned_candidate_relation_diagnostic_v1"
        if source.dataset == "dunnhumby"
        else "_m1_n_conditioned_candidate_relation_diagnostic_hm2y_v1"
    )
    values["out_dir"] = source.out_dir.split("_m1_clv_candidate_relation")[0] + suffix
    values.update(
        {"shuffle_degree_bins": 10, "shuffle_seed": 42, **overrides}
    )
    cfg = NConditionedCandidateRelationConfig(**values)
    if cfg.cross_fit_folds < 2 or cfg.eval_batch_size <= 0:
        raise ValueError("교차추정 fold는 2 이상이고 배치크기는 양수여야 합니다")
    if cfg.min_transition_support_users <= 0 or cfg.transition_kappa <= 0:
        raise ValueError("전이 지지도와 kappa는 양수여야 합니다")
    if cfg.shuffle_degree_bins <= 0:
        raise ValueError("degree 순열 구간 수는 양수여야 합니다")
    return cfg


def _base_config(cfg: NConditionedCandidateRelationConfig):
    values = asdict(cfg)
    values.pop("shuffle_degree_bins")
    values.pop("shuffle_seed")
    return base.CandidateRelationDiagnosticConfig(**values)


def preflight_summary(cfg: NConditionedCandidateRelationConfig) -> dict:
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
        "n_definition": "train-only percentile of the pre-registered N behavior score",
        "arms": {
            "pooled": "five-fold user-cross-fitted pooled category transition",
            "actual_n_conditioned": (
                "low/high transition endpoints estimated and interpolated with actual q_N"
            ),
            "shuffled_n_conditioned": (
                "same estimator after shuffling q_N within binary user-degree deciles"
            ),
        },
        "pair_definition": (
            "within each user, every M1 Top-10 false positive is paired with "
            "every held-out new-item truth missed by M1 Top-10"
        ),
        "cross_fitting": (
            "five user folds; a user's own transition evidence never estimates "
            "the transition applied to that user"
        ),
        "repeatshare_used": False,
        "raw_item_degree_used_as_signal": False,
        "degree_used_only_for_shuffle_control": True,
        "decision_metric": "macro user balanced win rate",
        "reading_rule": (
            "N attribution is supported in one dataset only if actual-N has macro "
            "win rate >0.5 and pair mean score difference >0 overall and for fixed "
            "high-CLV users, and its macro win rate exceeds pooled and shuffled-N "
            "in both groups. A common M2 N relation requires this in both datasets."
        ),
        "statistical_note": (
            "single-checkpoint descriptive development diagnostic; no statistical "
            "significance or generalization claim"
        ),
        "out_dir": cfg.out_dir,
    }


def _degree_matched_q_n_shuffle(
    q_n: np.ndarray,
    valid_n: np.ndarray,
    user_degree: np.ndarray,
    *,
    n_bins: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    q_n = np.asarray(q_n, dtype=np.float64)
    valid_n = np.asarray(valid_n, dtype=bool) & np.isfinite(q_n)
    shuffled, strata = transition._degree_stratified_shuffle(
        np.where(valid_n, q_n, np.nan),
        user_degree,
        n_bins=n_bins,
        seed=seed,
    )
    result = q_n.copy()
    result[valid_n] = shuffled[valid_n]
    changed = valid_n & ~np.isclose(result, q_n, rtol=0.0, atol=0.0)
    if valid_n.sum() > 1 and not changed.any():
        raise RuntimeError("degree-matched N 순열이 유효 고객의 N 배정을 바꾸지 못했습니다")
    return result, strata, {
        "valid_n_user_count": int(valid_n.sum()),
        "changed_valid_n_user_share": float(changed.sum() / max(valid_n.sum(), 1)),
        "shuffle_degree_bins": int(n_bins),
        "shuffle_seed": int(seed),
    }


def _lift_from_mass(
    weighted_mass: np.ndarray,
    supported: np.ndarray,
    target_prior: np.ndarray,
    *,
    kappa: float,
    log_lift_cap: float,
) -> np.ndarray:
    weighted_mass = np.where(supported, weighted_mass, 0.0)
    row_mass = weighted_mass.sum(axis=1)
    probability = (weighted_mass + kappa * target_prior[None, :]) / (
        row_mass[:, None] + kappa
    )
    lift = np.log(
        np.maximum(probability, 1e-12)
        / np.maximum(target_prior[None, :], 1e-12)
    )
    lift = np.clip(lift, -log_lift_cap, log_lift_cap)
    lift[~supported] = 0.0
    return lift


def _fold_lifts(
    reference: dict[str, np.ndarray],
    *,
    n_categories: int,
    min_support_users: int,
    kappa: float,
    log_lift_cap: float,
) -> tuple[dict[str, np.ndarray], int]:
    mass = reference["mass"].reshape(n_categories, n_categories)
    support = reference["support"].reshape(n_categories, n_categories)
    supported = support >= min_support_users
    mass = np.where(supported, mass, 0.0)
    target_mass = mass.sum(axis=0)
    target_total = float(target_mass.sum())
    target_prior = (
        target_mass / target_total
        if target_total > 0
        else np.full(n_categories, 1.0 / n_categories)
    )
    lifts = {
        name: _lift_from_mass(
            values.reshape(n_categories, n_categories),
            supported,
            target_prior,
            kappa=kappa,
            log_lift_cap=log_lift_cap,
        )
        for name, values in reference.items()
        if name != "support"
    }
    return lifts, int(supported.sum())


def _cross_fitted_transition_score_arms(
    *,
    pair: pd.DataFrame,
    recent: pd.DataFrame,
    evaluation_users: np.ndarray,
    q_n: np.ndarray,
    q_n_shuffled: np.ndarray,
    valid_n: np.ndarray,
    n_users: int,
    n_categories: int,
    folds: int,
    min_support_users: int,
    kappa: float,
    log_lift_cap: float,
) -> tuple[dict[str, np.ndarray], dict]:
    evaluation_users = np.asarray(evaluation_users, dtype=np.int64)
    q_n = np.asarray(q_n, dtype=np.float64)
    q_n_shuffled = np.asarray(q_n_shuffled, dtype=np.float64)
    valid_n = np.asarray(valid_n, dtype=bool) & np.isfinite(q_n)
    valid_n &= np.isfinite(q_n_shuffled)
    if not (len(q_n) == len(q_n_shuffled) == len(valid_n) == n_users):
        raise ValueError("N 조건과 사용자 수가 일치하지 않습니다")
    score_arms = {
        arm: np.zeros((len(evaluation_users), n_categories), dtype=np.float64)
        for arm in ARM_ORDER
    }
    if pair.empty:
        return score_arms, {"transition_rows_used": 0, "active_score_user_share": 0.0}

    pair_user_all = pair["u_idx"].to_numpy(np.int64)
    keep = valid_n[pair_user_all]
    pair_user = pair_user_all[keep]
    source = pair.loc[keep, "c_idx"].to_numpy(np.int64)
    target = pair.loc[keep, "d_idx"].to_numpy(np.int64)
    mass = pair.loc[keep, "mass"].to_numpy(np.float64)
    if not len(mass):
        return score_arms, {"transition_rows_used": 0, "active_score_user_share": 0.0}
    flat = source * n_categories + target
    size = n_categories * n_categories
    fold_index = pair_user % folds
    row_weights = {
        "mass": mass,
        "support": None,
        "actual_low": mass * (1.0 - q_n[pair_user]),
        "actual_high": mass * q_n[pair_user],
        "shuffle_low": mass * (1.0 - q_n_shuffled[pair_user]),
        "shuffle_high": mass * q_n_shuffled[pair_user],
    }
    total = {
        name: np.bincount(flat, weights=weights, minlength=size)
        for name, weights in row_weights.items()
    }
    total["support"] = total["support"].astype(np.int64)
    by_fold = {
        name: np.zeros((folds, size), dtype=values.dtype)
        for name, values in total.items()
    }
    for fold in range(folds):
        selected = fold_index == fold
        for name, weights in row_weights.items():
            by_fold[name][fold] = np.bincount(
                flat[selected],
                weights=None if weights is None else weights[selected],
                minlength=size,
            )

    position = np.full(n_users, -1, dtype=np.int64)
    position[evaluation_users] = np.arange(len(evaluation_users), dtype=np.int64)
    recent_user = recent["u_idx"].to_numpy(np.int64)
    recent_position = position[recent_user]
    use_recent = recent_position >= 0
    recent_position = recent_position[use_recent]
    recent_source = recent.loc[use_recent, "c_idx"].to_numpy(np.int64)
    recent_share = recent.loc[use_recent, "recent_share"].to_numpy(np.float64)

    supported_edges = []
    for fold in range(folds):
        reference = {name: values - by_fold[name][fold] for name, values in total.items()}
        lifts, n_supported = _fold_lifts(
            reference,
            n_categories=n_categories,
            min_support_users=min_support_users,
            kappa=kappa,
            log_lift_cap=log_lift_cap,
        )
        supported_edges.append(n_supported)
        selected_rows = np.flatnonzero(
            (evaluation_users % folds == fold) & valid_n[evaluation_users]
        )
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
        score_arms["pooled"][selected_rows] = profile @ lifts["mass"]
        q_actual = q_n[evaluation_users[selected_rows]][:, None]
        score_arms["actual_n_conditioned"][selected_rows] = (
            (1.0 - q_actual) * (profile @ lifts["actual_low"])
            + q_actual * (profile @ lifts["actual_high"])
        )
        q_control = q_n_shuffled[evaluation_users[selected_rows]][:, None]
        score_arms["shuffled_n_conditioned"][selected_rows] = (
            (1.0 - q_control) * (profile @ lifts["shuffle_low"])
            + q_control * (profile @ lifts["shuffle_high"])
        )

    active_eval = valid_n[evaluation_users]
    diagnostics = {
        "transition_rows_used": int(len(mass)),
        "transition_evidence_users_used": int(np.unique(pair_user).size),
        "valid_n_evaluation_user_count": int(active_eval.sum()),
        "valid_n_evaluation_user_share": float(active_eval.mean()),
        "mean_supported_edges_per_fold": float(np.mean(supported_edges)),
    }
    for arm, scores in score_arms.items():
        selected = scores[active_eval]
        diagnostics[f"{arm}_score_std"] = float(selected.std())
        diagnostics[f"{arm}_active_score_user_share"] = float(
            np.any(np.abs(selected) > 1e-12, axis=1).mean()
        )
    return score_arms, diagnostics


def _user_pair_rows(
    *,
    users: np.ndarray,
    top10: np.ndarray,
    truth: dict[int, np.ndarray],
    membership: pd.DataFrame,
    item_category: np.ndarray,
    score_arms: dict[str, np.ndarray],
    q_n_shuffled: np.ndarray,
    valid_n: np.ndarray,
) -> pd.DataFrame:
    by_user = membership.set_index("user_idx")
    rows = []
    for position, user in enumerate(np.asarray(users, dtype=np.int64)):
        if not valid_n[user]:
            continue
        truth_items = np.asarray(truth[int(user)], dtype=np.int64)
        truth_set = set(map(int, truth_items))
        ranked = np.asarray(top10[position], dtype=np.int64)
        ranked_set = set(map(int, ranked))
        missed = np.asarray(
            [item for item in truth_items if int(item) not in ranked_set], dtype=np.int64
        )
        false_positive = np.asarray(
            [item for item in ranked if int(item) not in truth_set], dtype=np.int64
        )
        if not len(missed) or not len(false_positive):
            continue
        group = by_user.loc[int(user)]
        for arm in ARM_ORDER:
            scores = score_arms[arm][position]
            difference = (
                scores[item_category[missed]][:, None]
                - scores[item_category[false_positive]][None, :]
            ).reshape(-1)
            difference = difference[np.isfinite(difference)]
            if not len(difference):
                continue
            tolerance = 1e-12
            wins = int(np.sum(difference > tolerance))
            losses = int(np.sum(difference < -tolerance))
            ties = int(len(difference) - wins - losses)
            assigned = np.nan
            if arm == "actual_n_conditioned":
                assigned = float(group["q_n"])
            elif arm == "shuffled_n_conditioned":
                assigned = float(q_n_shuffled[user])
            rows.append(
                {
                    "user_idx": int(user),
                    "arm": arm,
                    "fixed_clv_segment": group["fixed_clv_segment"],
                    "nv_quadrant": group["nv_quadrant"],
                    "high_clv_composition": group["high_clv_composition"],
                    "q_n": float(group["q_n"]),
                    "q_n_assigned": assigned,
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
    for arm in ARM_ORDER:
        arm_rows = per_user.loc[per_user.arm.eq(arm)].copy()
        group_specs = [("overall", "전체", np.ones(len(arm_rows), dtype=bool))]
        for column, order in GROUP_ORDERS.items():
            values = arm_rows[column].to_numpy()
            for group in order:
                mask = values == group
                if mask.any():
                    group_specs.append((column, group, mask))
        for group_type, group, mask in group_specs:
            selected = arm_rows.loc[mask]
            pairs = int(selected.candidate_pair_count.sum())
            wins = int(selected.truth_wins.sum())
            ties = int(selected.ties.sum())
            losses = int(selected.false_positive_wins.sum())
            rows.append(
                {
                    "arm": arm,
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
                    "macro_user_balanced_win_rate": float(selected.balanced_win_rate.mean()),
                    "mean_pair_score_difference": float(
                        np.average(
                            selected.mean_pair_score_difference,
                            weights=selected.candidate_pair_count,
                        )
                    ),
                }
            )
    return pd.DataFrame(rows)


def compare_arms(summary: pd.DataFrame) -> pd.DataFrame:
    keys = ["group_type", "group"]
    metrics = (
        "pair_balanced_win_rate",
        "macro_user_balanced_win_rate",
        "mean_pair_score_difference",
    )
    actual = summary.loc[summary.arm.eq("actual_n_conditioned")].set_index(keys)
    rows = []
    for reference_arm in ("pooled", "shuffled_n_conditioned"):
        reference = summary.loc[summary.arm.eq(reference_arm)].set_index(keys)
        for key in actual.index.intersection(reference.index):
            group_type, group = key
            row = {
                "model_arm": "actual_n_conditioned",
                "reference_arm": reference_arm,
                "group_type": group_type,
                "group": group,
            }
            for metric in metrics:
                model_value = float(actual.at[key, metric])
                reference_value = float(reference.at[key, metric])
                row[f"{metric}_actual"] = model_value
                row[f"{metric}_reference"] = reference_value
                row[f"{metric}_delta"] = model_value - reference_value
            rows.append(row)
    return pd.DataFrame(rows)


def _screen_reading(summary: pd.DataFrame) -> dict:
    indexed = summary.set_index(["arm", "group_type", "group"])
    groups = {
        "overall": ("overall", "전체"),
        "high_clv": ("fixed_clv_segment", fixed.SEGMENT_ORDER[2]),
    }
    readings = {}
    passes = []
    for label, (group_type, group) in groups.items():
        actual = indexed.loc[("actual_n_conditioned", group_type, group)]
        pooled = indexed.loc[("pooled", group_type, group)]
        shuffled = indexed.loc[("shuffled_n_conditioned", group_type, group)]
        checks = {
            "actual_macro_above_half": bool(actual.macro_user_balanced_win_rate > 0.5),
            "actual_mean_difference_positive": bool(actual.mean_pair_score_difference > 0),
            "actual_macro_above_pooled": bool(
                actual.macro_user_balanced_win_rate > pooled.macro_user_balanced_win_rate
            ),
            "actual_macro_above_shuffled": bool(
                actual.macro_user_balanced_win_rate
                > shuffled.macro_user_balanced_win_rate
            ),
        }
        readings[label] = {
            "actual_macro_user_balanced_win_rate": float(
                actual.macro_user_balanced_win_rate
            ),
            "actual_pair_balanced_win_rate": float(actual.pair_balanced_win_rate),
            "actual_mean_pair_score_difference": float(actual.mean_pair_score_difference),
            "macro_delta_vs_pooled": float(
                actual.macro_user_balanced_win_rate
                - pooled.macro_user_balanced_win_rate
            ),
            "macro_delta_vs_shuffled": float(
                actual.macro_user_balanced_win_rate
                - shuffled.macro_user_balanced_win_rate
            ),
            "checks": checks,
            "passes": bool(all(checks.values())),
        }
        passes.append(all(checks.values()))
    return {
        "n_attribution_supported_in_this_dataset": bool(all(passes)),
        "groups": readings,
        "cross_dataset_decision_pending": True,
    }


def _paths(cfg: NConditionedCandidateRelationConfig) -> dict[str, Path]:
    root = Path(cfg.out_dir)
    root.mkdir(parents=True, exist_ok=True)
    stem = f"m1_n_conditioned_candidate_relation_{cfg.dataset}"
    return {
        "relation_summary_csv": root / f"{stem}_summary.csv",
        "arm_comparison_csv": root / f"{stem}_comparison.csv",
        "per_user_csv": root / f"{stem}_per_user.csv",
        "json": root / f"{stem}_diagnostic.json",
    }


@torch.no_grad()
def run_n_conditioned_candidate_relation_diagnostic(
    cfg: NConditionedCandidateRelationConfig | None = None,
) -> dict[str, str]:
    cfg = cfg or configure_n_conditioned_candidate_relation_diagnostic()
    print(json.dumps(preflight_summary(cfg), ensure_ascii=False, indent=2))
    prepared, model, checkpoint, record, axes = base._prepare_and_load(_base_config(cfg))
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

    membership, thresholds = base._membership(prepared, axes)
    train = prepared["data"]["train"]
    n_users = int(prepared["data"]["n_users"])
    n_items = int(prepared["data"]["n_items"])
    item_category, n_categories = base._item_categories(train, n_items)
    pair, recent, transition_diagnostics = transition._transition_evidence(train, n_users)
    if recent.columns.duplicated().any():
        recent = recent.loc[:, ~recent.columns.duplicated()].copy()

    q_n = np.asarray(axes["q_n"], dtype=np.float64)
    valid_n = np.asarray(axes["activity_valid"], dtype=bool) & np.isfinite(q_n)
    train_edges = train[["u_idx", "i_idx"]].drop_duplicates()
    user_degree = np.bincount(
        train_edges["u_idx"].to_numpy(np.int64), minlength=n_users
    )
    q_n_shuffled, shuffle_strata, shuffle_diagnostics = _degree_matched_q_n_shuffle(
        q_n,
        valid_n,
        user_degree,
        n_bins=cfg.shuffle_degree_bins,
        seed=cfg.shuffle_seed,
    )
    score_arms, score_diagnostics = _cross_fitted_transition_score_arms(
        pair=pair,
        recent=recent,
        evaluation_users=users,
        q_n=q_n,
        q_n_shuffled=q_n_shuffled,
        valid_n=valid_n,
        n_users=n_users,
        n_categories=n_categories,
        folds=cfg.cross_fit_folds,
        min_support_users=cfg.min_transition_support_users,
        kappa=cfg.transition_kappa,
        log_lift_cap=cfg.transition_log_lift_cap,
    )
    per_user = _user_pair_rows(
        users=users,
        top10=top10,
        truth=prepared["cache"].gt,
        membership=membership,
        item_category=item_category,
        score_arms=score_arms,
        q_n_shuffled=q_n_shuffled,
        valid_n=valid_n,
    )
    if per_user.empty:
        raise RuntimeError("비교 가능한 M1 누락정답–오추천 쌍이 없습니다")
    summary = summarize_pair_directions(per_user)
    comparison = compare_arms(summary)
    paths = _paths(cfg)
    test10._atomic_csv(paths["relation_summary_csv"], summary)
    test10._atomic_csv(paths["arm_comparison_csv"], comparison)
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
        "transition_diagnostics": {**transition_diagnostics, **score_diagnostics},
        "shuffle_diagnostics": {
            **shuffle_diagnostics,
            "active_strata": int(np.unique(shuffle_strata[shuffle_strata >= 0]).size),
        },
        "screen_reading": _screen_reading(summary),
        "relation_summary": summary.to_dict("records"),
        "arm_comparison": comparison.to_dict("records"),
        "result_paths": {name: str(path) for name, path in paths.items()},
        "interpretation_limits": [
            "checkpoint-only descriptive development diagnostic",
            "category transition T(u,i) is not defined as N itself",
            "this run tests whether train-only N conditions that transition",
            "no model training, checkpoint selection, final test, or holdout",
            "historical CLV proxy is not future or incremental CLV",
            "pair win rates are not recommendation accuracy metrics",
            "no statistical significance or cross-dataset generalization claim",
        ],
    }
    test10._atomic_json(paths["json"], payload)
    return {name: str(path) for name, path in paths.items()}
