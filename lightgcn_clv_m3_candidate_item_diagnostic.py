"""Checkpoint-only failure diagnostic for the candidate-item M3.

The source M3 already trained three arms (pooled relation, correctly assigned
historical CLV, and degree-matched shuffled CLV).  This module does not train or
select a model.  It traces the existing intervention through three stages:

1. candidate relation rows,
2. auxiliary score contribution, and
3. final Top-K recommendations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from clv_m3_clv_conditioned_candidate_item_graph import (
    ARM_ACTUAL,
    ARM_GENERAL,
    ARM_SHUFFLE,
    build_clv_conditioned_candidate_item_graph,
)
from clv_run_state import file_sha256
import lightgcn_clv_axis_specific_test10 as fixed_train
import lightgcn_clv_history_item_fit_diagnostic as item_fit
import lightgcn_clv_m3_category_transition_diagnostic as shared
import lightgcn_clv_m3_clv_conditioned_candidate_item as source_runner
import lightgcn_clv_moe as moe
import lightgcn_clv_v3 as v3


CODE_VERSION = "m3-clv-candidate-item-failure-diagnostic-v1"
ARM_MODEL_IDS = {
    ARM_GENERAL: source_runner.GENERAL_ID,
    ARM_ACTUAL: source_runner.ACTUAL_ID,
    ARM_SHUFFLE: source_runner.SHUFFLE_ID,
}


@dataclass(frozen=True)
class M3CandidateItemDiagnosticConfig:
    dataset: str = "dunnhumby"
    seed: int = 42
    rank_limit: int = 50
    score_batch_size: int = 64
    source_result_id: str = "d5c0423bfd90"
    source_out_dir: str = ""
    diagnostic_out_dir: str = ""


def configure_m3_candidate_item_diagnostic(
    **overrides,
) -> M3CandidateItemDiagnosticConfig:
    source = (
        f"{v3.default_out_dir('dunnhumby')}"
        "_m3_clv_candidate_item_historical_screen_v1"
    )
    defaults = {
        "source_out_dir": source,
        "diagnostic_out_dir": f"{source}/failure_mechanism_diagnostic_v1",
    }
    cfg = M3CandidateItemDiagnosticConfig(**(defaults | overrides))
    if cfg.dataset != "dunnhumby" or cfg.seed != 42 or cfg.rank_limit != 50:
        raise ValueError("진단은 Dunnhumby seed 42, Top-50으로 고정합니다")
    if cfg.score_batch_size <= 0:
        raise ValueError("score_batch_size must be positive")
    if not cfg.source_result_id or not cfg.source_out_dir or not cfg.diagnostic_out_dir:
        raise ValueError("source result and output directories are required")
    return cfg


def preflight_summary(cfg: M3CandidateItemDiagnosticConfig) -> dict:
    cfg = configure_m3_candidate_item_diagnostic(**cfg.__dict__)
    return {
        "code_version": CODE_VERSION,
        "analysis_type": "descriptive post-hoc checkpoint diagnostic",
        "source_models": list(ARM_MODEL_IDS.values()),
        "source_split": "DAY 1--683 train; DAY 684--690 evaluation",
        "training": False,
        "checkpoint_selection": False,
        "final_test_constructed": False,
        "holdout_constructed": False,
        "rank_limit": cfg.rank_limit,
        "source_result_id": cfg.source_result_id,
        "questions": [
            "do correctly assigned CLV rows cover more held-out truths than pooled and shuffled rows?",
            "does the auxiliary message favor held-out truths over competitive Top-50 items?",
            "does the score contribution change Top-10/20/50 recommendations?",
            "which single stage should a future M3 change?",
        ],
        "routing_rule": {
            "candidate_relation_construction": (
                "actual candidate truth coverage does not beat shuffled CLV"
            ),
            "relation_to_score_transfer": (
                "actual candidate truth coverage beats shuffle but its truth-minus-Top50 auxiliary score contrast does not"
            ),
            "score_to_rank_boundary": (
                "candidate and score signals beat shuffle but actual and shuffle recommendation sets never differ"
            ),
            "ranking_alignment": (
                "CLV-specific signal reaches Top-K but the already observed accuracy attribution still fails"
            ),
        },
        "routing_note": (
            "direction-only descriptive routing; no tuned magnitude threshold and no automatic model selection"
        ),
        "interpretation": (
            "hypothesis generation only; no retraining, model selection, significance, or generalization claim"
        ),
        "source_out_dir": cfg.source_out_dir,
        "diagnostic_out_dir": cfg.diagnostic_out_dir,
    }


def _source_result(
    cfg: M3CandidateItemDiagnosticConfig,
) -> tuple[Path, dict]:
    path = Path(cfg.source_out_dir) / (
        f"m3_clv_candidate_item_{cfg.source_result_id}.json"
    )
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("code_version") != source_runner.CODE_VERSION:
        raise RuntimeError("source result code_version does not match the M3 runner")
    if payload.get("config", {}).get("seed") != cfg.seed:
        raise RuntimeError("source result seed does not match the diagnostic")
    source_split = payload.get("preflight", {}).get(
        "historical_development_split", {}
    )
    if source_split.get("final_test_constructed") is not False:
        raise RuntimeError("source result is not the historical development screen")
    if source_split.get("holdout_constructed") is not False:
        raise RuntimeError("source result unexpectedly constructed a holdout")
    return path, payload


def _prepare_source(
    cfg: M3CandidateItemDiagnosticConfig,
    source_payload: dict,
) -> tuple[dict, source_runner.CLVCandidateItemConfig, str]:
    source_cfg = source_runner.validate_clv_candidate_item_config(
        source_runner.CLVCandidateItemConfig(**source_payload["config"])
    )
    manifest = moe.build_input_manifest(v3.SCHEMA[cfg.dataset])
    input_hash = moe.manifest_hash(manifest)
    source_manifest_hash = moe.manifest_hash(source_payload["input_manifest"])
    if input_hash != source_manifest_hash:
        raise RuntimeError("current input files differ from the source M3 result")
    base_cfg = source_runner._base_config(source_cfg)
    data = v3.prepare_data(base_cfg, v3.DCFG)
    if set(data["splits"]) != {"test"} or float(data["train"].t.max()) != 683.0:
        raise RuntimeError("source historical split was not reconstructed exactly")
    graph = build_clv_conditioned_candidate_item_graph(
        data["train"],
        data["n_users"],
        data["n_items"],
        data["n_cat"],
        category_kappa=source_cfg.category_kappa,
        category_min_support_users=source_cfg.category_min_support_users,
        item_kappa=source_cfg.item_kappa,
        item_min_support_users=source_cfg.item_min_support_users,
        shuffle_seed=source_cfg.shuffle_seed,
        shuffle_degree_bins=source_cfg.shuffle_degree_bins,
        cross_fit_folds=source_cfg.cross_fit_folds,
        max_target_categories=source_cfg.max_target_categories,
        max_candidate_items=source_cfg.max_candidate_items,
    )
    thresholds = v3.segment_thresholds(graph.clv_proxy, base_cfg["SEG_EDGES"])
    cache = v3.EvalCache(
        *data["splits"]["test"],
        graph.clv_proxy,
        thresholds,
        data["n_items"],
    )
    prepared = {
        "base_cfg": base_cfg,
        "data": data,
        "graph": graph,
        "cache": cache,
        "input_hash": input_hash,
        "source_manifest_hash": source_manifest_hash,
    }
    return prepared, source_cfg, input_hash


def _load_arm_model(
    cfg: M3CandidateItemDiagnosticConfig,
    prepared: dict,
    source_cfg: source_runner.CLVCandidateItemConfig,
    *,
    arm: str,
    model_id: str,
) -> tuple[torch.nn.Module, dict, Path]:
    record_path = (
        Path(cfg.source_out_dir)
        / "arms"
        / cfg.source_result_id
        / f"{model_id}_s{cfg.seed}.json"
    )
    if not record_path.exists():
        raise FileNotFoundError(record_path)
    record = json.loads(record_path.read_text(encoding="utf-8"))
    checkpoint = Path(record["checkpoint"])
    if not checkpoint.exists():
        checkpoint = record_path.with_suffix(".pt")
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    expected_sha = record.get("checkpoint_sha256")
    if expected_sha and file_sha256(checkpoint) != expected_sha:
        raise RuntimeError(f"checkpoint hash mismatch: {model_id}")
    blob = torch.load(checkpoint, map_location=v3.DEVICE, weights_only=False)
    expected = {
        "record.model_id": (model_id, record.get("model_id")),
        "record.graph_arm": (arm, record.get("graph_arm")),
        "record.seed": (cfg.seed, record.get("seed")),
        "record.input_hash": (prepared["input_hash"], record.get("input_hash")),
        "checkpoint.model_id": (model_id, blob.get("model_id")),
        "checkpoint.graph_arm": (arm, blob.get("graph_arm")),
        "checkpoint.input_hash": (prepared["input_hash"], blob.get("input_hash")),
        "checkpoint.config.seed": (cfg.seed, blob.get("config", {}).get("seed")),
    }
    mismatches = {
        name: {"expected": pair[0], "actual": pair[1]}
        for name, pair in expected.items()
        if pair[0] != pair[1]
    }
    if mismatches:
        raise RuntimeError(f"checkpoint identity mismatch: {mismatches}")
    model = source_runner._build_model(prepared, source_cfg, arm)
    model.load_state_dict(blob["state"], strict=True)
    model.eval()
    return model, record, checkpoint


def _relation_rows(operator: torch.Tensor) -> list[dict[int, float]]:
    return shared._sparse_rows(operator)


def candidate_truth_coverage(
    operators: dict[str, torch.Tensor],
    *,
    evaluation_users: np.ndarray,
    truths: dict[int, set[int]],
    q_actual: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Measure whether each relation contains each user's held-out truths."""
    evaluation_users = np.asarray(evaluation_users, dtype=np.int64)
    q_actual = np.asarray(q_actual, dtype=np.float64)
    if not set(ARM_MODEL_IDS).issubset(operators):
        raise ValueError("general, actual, and shuffle operators are required")
    if evaluation_users.min(initial=0) < 0 or evaluation_users.max(initial=-1) >= len(
        q_actual
    ):
        raise ValueError("evaluation user index is outside q_actual")
    quintile = shared._clv_quintile(q_actual)
    rows = []
    for arm, model_id in ARM_MODEL_IDS.items():
        sparse_rows = _relation_rows(operators[arm])
        for user in evaluation_users:
            truth = set(map(int, truths[int(user)]))
            relation = sparse_rows[int(user)]
            ranking = sorted(relation, key=lambda item: (-relation[item], item))
            rank_by_item = {item: rank + 1 for rank, item in enumerate(ranking)}
            hits = truth & relation.keys()
            hit_ranks = [rank_by_item[item] for item in hits]
            truth_mass = float(sum(relation[item] for item in hits))
            rows.append(
                {
                    "model_id": model_id,
                    "graph_arm": arm,
                    "user_idx": int(user),
                    "clv_group": str(quintile[int(user)]),
                    "n_truth": int(len(truth)),
                    "n_candidates": int(len(relation)),
                    "truth_hits": int(len(hits)),
                    "truth_recall_in_candidates": (
                        len(hits) / len(truth) if truth else np.nan
                    ),
                    "any_truth_in_candidates": bool(hits),
                    "truth_edge_weight_mass": truth_mass,
                    "mean_truth_edge_weight_all_truth": (
                        truth_mass / len(truth) if truth else np.nan
                    ),
                    "mean_hit_rank": (
                        float(np.mean(hit_ranks)) if hit_ranks else np.nan
                    ),
                    "best_hit_rank": (
                        float(min(hit_ranks)) if hit_ranks else np.nan
                    ),
                }
            )
    per_user = pd.DataFrame(rows)
    summaries = []
    for (model_id, arm), model_rows in per_user.groupby(
        ["model_id", "graph_arm"], sort=False
    ):
        for group in ("전체", "Q1", "Q2", "Q3", "Q4", "Q5"):
            selected = (
                model_rows
                if group == "전체"
                else model_rows.loc[model_rows["clv_group"].eq(group)]
            )
            if selected.empty:
                continue
            truth_count = int(selected["n_truth"].sum())
            hit_count = int(selected["truth_hits"].sum())
            hit_users = selected.loc[selected["any_truth_in_candidates"]]
            summaries.append(
                {
                    "model_id": model_id,
                    "graph_arm": arm,
                    "clv_group": group,
                    "n_users": int(len(selected)),
                    "n_truth_pairs": truth_count,
                    "candidate_truth_hits": hit_count,
                    "candidate_truth_pair_coverage": (
                        hit_count / truth_count if truth_count else np.nan
                    ),
                    "macro_candidate_truth_recall": float(
                        selected["truth_recall_in_candidates"].mean()
                    ),
                    "user_any_truth_share": float(
                        selected["any_truth_in_candidates"].mean()
                    ),
                    "mean_truth_edge_weight_all_truth": (
                        float(selected["truth_edge_weight_mass"].sum())
                        / truth_count
                        if truth_count
                        else np.nan
                    ),
                    "mean_hit_rank": float(hit_users["mean_hit_rank"].mean()),
                    "mean_candidate_count": float(selected["n_candidates"].mean()),
                }
            )
    return per_user, pd.DataFrame(summaries)


def candidate_truth_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    metrics = (
        "candidate_truth_pair_coverage",
        "macro_candidate_truth_recall",
        "user_any_truth_share",
        "mean_truth_edge_weight_all_truth",
        "mean_hit_rank",
    )
    rows = []
    for group in summary["clv_group"].drop_duplicates():
        selected = summary.loc[summary["clv_group"].eq(group)].set_index("graph_arm")
        actual = selected.loc[ARM_ACTUAL]
        for reference in (ARM_GENERAL, ARM_SHUFFLE):
            other = selected.loc[reference]
            for metric in metrics:
                direction = -1.0 if metric == "mean_hit_rank" else 1.0
                raw_delta = float(actual[metric]) - float(other[metric])
                rows.append(
                    {
                        "clv_group": group,
                        "reference": reference,
                        "metric": metric,
                        "reference_value": float(other[metric]),
                        "actual_value": float(actual[metric]),
                        "absolute_delta": raw_delta,
                        "actual_is_better": bool(direction * raw_delta > 0),
                    }
                )
    return pd.DataFrame(rows)


@torch.no_grad()
def _model_views(
    model: torch.nn.Module,
    prepared: dict,
    cfg: M3CandidateItemDiagnosticConfig,
) -> dict:
    base_user, base_item, message = model.representation_parts()
    full_user = base_user + model.gamma * message
    users, full_top50 = item_fit._masked_topk(
        full_user,
        base_item,
        prepared,
        max_k=cfg.rank_limit,
        batch_size=cfg.score_batch_size,
    )
    base_users, base_top50 = item_fit._masked_topk(
        base_user,
        base_item,
        prepared,
        max_k=cfg.rank_limit,
        batch_size=cfg.score_batch_size,
    )
    if not np.array_equal(users, base_users):
        raise RuntimeError("base and full recommendation users are misaligned")
    return {
        "users": users,
        "full_top50": full_top50,
        "base_top50": base_top50,
        "base_user": base_user,
        "base_item": base_item,
        "message": message,
        "gamma": float(model.gamma),
    }


def _score_pair_populations(
    views: dict[str, dict],
    cache: v3.EvalCache,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    users = views[ARM_ACTUAL]["users"]
    union_users = []
    union_items = []
    for row, user in enumerate(users):
        union = set()
        for arm in ARM_MODEL_IDS:
            union.update(map(int, views[arm]["full_top50"][row]))
        for item in sorted(union):
            union_users.append(int(user))
            union_items.append(item)
    truth_users = []
    truth_items = []
    for user in users:
        for item in cache.gt[int(user)]:
            truth_users.append(int(user))
            truth_items.append(int(item))
    return {
        "three_arm_top50_union": (
            np.asarray(union_users, dtype=np.int64),
            np.asarray(union_items, dtype=np.int64),
        ),
        "heldout_truth": (
            np.asarray(truth_users, dtype=np.int64),
            np.asarray(truth_items, dtype=np.int64),
        ),
    }


def _score_tables(
    views: dict[str, dict],
    populations: dict[str, tuple[np.ndarray, np.ndarray]],
    q_actual: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tables = []
    for arm, model_id in ARM_MODEL_IDS.items():
        view = views[arm]
        for role, (users, items) in populations.items():
            base, auxiliary = shared.score_pairs_from_parts(
                view["base_user"],
                view["base_item"],
                view["message"],
                users=users,
                items=items,
                gamma=view["gamma"],
            )
            tables.append(
                shared.score_component_long_summary(
                    base_scores=base,
                    auxiliary_scores=auxiliary,
                    score_users=users,
                    q_by_user=q_actual,
                    model_id=model_id,
                    candidate_role=role,
                )
            )
    components = pd.concat(tables, ignore_index=True)
    values = components.pivot_table(
        index=["model_id", "clv_group"],
        columns="candidate_role",
        values=["auxiliary_score_mean", "mean_abs_auxiliary_score"],
        aggfunc="first",
    )
    values.columns = [f"{metric}__{role}" for metric, role in values.columns]
    contrast = values.reset_index()
    contrast["truth_minus_top50_auxiliary_mean"] = (
        contrast["auxiliary_score_mean__heldout_truth"]
        - contrast["auxiliary_score_mean__three_arm_top50_union"]
    )
    return components, contrast


def _recommendation_overlap(
    views: dict[str, dict],
    q_eval: np.ndarray,
) -> pd.DataFrame:
    comparisons = [
        (
            "actual_full_vs_shuffle_full",
            views[ARM_ACTUAL]["full_top50"],
            views[ARM_SHUFFLE]["full_top50"],
        ),
        (
            "actual_full_vs_general_full",
            views[ARM_ACTUAL]["full_top50"],
            views[ARM_GENERAL]["full_top50"],
        ),
    ]
    for arm in ARM_MODEL_IDS:
        comparisons.append(
            (
                f"{arm}_base_vs_{arm}_full",
                views[arm]["base_top50"],
                views[arm]["full_top50"],
            )
        )
    return pd.concat(
        [
            shared.recommendation_overlap_summary(
                reference,
                model,
                q_actual=q_eval,
                comparison=name,
            )
            for name, reference, model in comparisons
        ],
        ignore_index=True,
    )


def diagnostic_route(
    candidate_summary: pd.DataFrame,
    score_contrast: pd.DataFrame,
    recommendation_overlap: pd.DataFrame,
    *,
    source_attribution_supported: bool,
) -> dict:
    overall_candidates = candidate_summary.loc[
        candidate_summary["clv_group"].eq("전체")
    ].set_index("graph_arm")
    actual_candidates = overall_candidates.loc[ARM_ACTUAL]
    shuffled_candidates = overall_candidates.loc[ARM_SHUFFLE]
    candidate_deltas = {
        metric: float(actual_candidates[metric])
        - float(shuffled_candidates[metric])
        for metric in (
            "candidate_truth_pair_coverage",
            "macro_candidate_truth_recall",
            "mean_truth_edge_weight_all_truth",
        )
    }
    candidate_signal = bool(
        candidate_deltas["candidate_truth_pair_coverage"] > 0
        and candidate_deltas["macro_candidate_truth_recall"] > 0
    )

    overall_scores = score_contrast.loc[
        score_contrast["clv_group"].eq("전체")
    ].set_index("model_id")
    score_delta = float(
        overall_scores.loc[
            source_runner.ACTUAL_ID, "truth_minus_top50_auxiliary_mean"
        ]
        - overall_scores.loc[
            source_runner.SHUFFLE_ID, "truth_minus_top50_auxiliary_mean"
        ]
    )
    score_signal = bool(score_delta > 0)

    overlap = recommendation_overlap.loc[
        recommendation_overlap["comparison"].eq(
            "actual_full_vs_shuffle_full"
        )
        & recommendation_overlap["clv_group"].eq("전체")
    ].set_index("k")
    set_changed = {
        f"top{k}": float(overlap.loc[k, "set_changed_user_share"])
        for k in (10, 20, 50)
    }
    rank_signal = bool(any(value > 0 for value in set_changed.values()))

    if not candidate_signal:
        bottleneck = "candidate_relation_construction"
        next_change = (
            "Change only the CLV-to-item relation statistic or evidence aggregation; "
            "keep gamma, the optimizer, and BPR fixed. Do not amplify the current rows."
        )
    elif not score_signal:
        bottleneck = "relation_to_score_transfer"
        next_change = (
            "Keep the candidate rows; change only how the candidate message is centered "
            "against the pooled relation and injected inside the forward graph."
        )
    elif not rank_signal:
        bottleneck = "score_to_rank_boundary"
        next_change = (
            "Keep graph targets and direction; change only bounded message normalization "
            "or integration strength in a newly predeclared run."
        )
    else:
        bottleneck = "ranking_alignment"
        next_change = (
            "The CLV-specific path reaches Top-K but changes the wrong items; inspect entered "
            "and lost truths by CLV group before changing relation semantics. Do not increase gamma."
        )
    return {
        "automatic_model_selection": False,
        "source_clv_attribution_supported": bool(source_attribution_supported),
        "candidate_actual_minus_shuffle": candidate_deltas,
        "candidate_signal_directionally_positive": candidate_signal,
        "truth_minus_top50_auxiliary_contrast_actual_minus_shuffle": score_delta,
        "score_transfer_directionally_positive": score_signal,
        "actual_shuffle_set_changed_user_share": set_changed,
        "rank_change_observed": rank_signal,
        "descriptive_bottleneck": bottleneck,
        "next_change_scope": next_change,
        "rule": (
            "direction-only stage routing with no tuned magnitude threshold; future changes "
            "must be evaluated on a newly predeclared interval or independent data"
        ),
        "limitation": (
            "post-hoc on the seen development interval; no significance, causal, or generalization claim"
        ),
    }


def _relation_quality(
    name: str,
    operator: torch.Tensor,
    prepared: dict,
) -> list[dict]:
    rows = _relation_rows(operator)
    ptr = prepared["data"]["csr_ptr"]
    history = prepared["data"]["csr_items"]
    train_pairs = 0
    duplicate_edges = 0
    for user, relation in enumerate(rows):
        duplicate_edges += int(len(relation) != len(set(relation)))
        seen = set(map(int, history[ptr[user] : ptr[user + 1]]))
        train_pairs += len(seen & relation.keys())
    coalesced = operator.coalesce()
    return [
        {
            "check": f"{name}_candidate_train_pairs",
            "value": int(train_pairs),
            "passed": train_pairs == 0,
        },
        {
            "check": f"{name}_candidate_duplicate_rows",
            "value": int(duplicate_edges),
            "passed": duplicate_edges == 0,
        },
        {
            "check": f"{name}_candidate_edge_count",
            "value": int(coalesced._nnz()),
            "passed": coalesced._nnz() == 250000,
        },
    ]


def run_m3_candidate_item_diagnostic(
    cfg: M3CandidateItemDiagnosticConfig | None = None,
) -> pd.DataFrame:
    cfg = configure_m3_candidate_item_diagnostic(
        **(cfg.__dict__ if cfg is not None else {})
    )
    preflight = preflight_summary(cfg)
    print(json.dumps(preflight, ensure_ascii=False, indent=2))
    source_path, source_payload = _source_result(cfg)
    prepared, source_cfg, input_hash = _prepare_source(cfg, source_payload)
    graph = prepared["graph"]

    models = {}
    records = {}
    checkpoints = {}
    for arm, model_id in ARM_MODEL_IDS.items():
        model, record, checkpoint = _load_arm_model(
            cfg,
            prepared,
            source_cfg,
            arm=arm,
            model_id=model_id,
        )
        models[arm] = model
        records[arm] = record
        checkpoints[arm] = checkpoint

    graph_users = shared.sparse_row_similarity(
        graph.user_item_operators[ARM_ACTUAL],
        graph.user_item_operators[ARM_SHUFFLE],
        q_actual=graph.clv_percentile,
        q_shuffle=graph.clv_shuffle_percentile,
        strata=graph.clv_shuffle_stratum,
    )
    eval_users = prepared["cache"].users.astype(np.int64)
    graph_users["is_evaluation_user"] = graph_users["user_idx"].isin(eval_users)
    graph_summaries = []
    for population, selected in (
        ("all_train_users", graph_users),
        ("evaluation_users", graph_users.loc[graph_users["is_evaluation_user"]]),
    ):
        summary = shared.graph_similarity_summary(selected)
        summary.insert(0, "population", population)
        graph_summaries.append(summary)
    graph_summary = pd.concat(graph_summaries, ignore_index=True)
    assignment = shared.clv_assignment_correlation(graph_users)

    coverage_users, coverage_summary = candidate_truth_coverage(
        graph.user_item_operators,
        evaluation_users=eval_users,
        truths=prepared["cache"].gt,
        q_actual=graph.clv_percentile,
    )
    coverage_comparison = candidate_truth_comparison(coverage_summary)

    views = {
        arm: _model_views(model, prepared, cfg)
        for arm, model in models.items()
    }
    for arm in ARM_MODEL_IDS:
        if not np.array_equal(views[arm]["users"], eval_users):
            raise RuntimeError(f"{arm} recommendations and evaluation users are misaligned")
    populations = _score_pair_populations(views, prepared["cache"])
    score_components, score_contrast = _score_tables(
        views,
        populations,
        graph.clv_percentile,
    )
    recommendation_overlap = _recommendation_overlap(
        views,
        graph.clv_percentile[eval_users],
    )

    quality_rows = [
        {
            "check": "source_manifest_hash_matches_current_files",
            "value": int(input_hash == prepared["source_manifest_hash"]),
            "passed": input_hash == prepared["source_manifest_hash"],
        },
        {
            "check": "evaluation_user_count",
            "value": int(len(eval_users)),
            "passed": len(eval_users) == 1230,
        },
        {
            "check": "heldout_truth_pair_count",
            "value": int(len(populations["heldout_truth"][0])),
            "passed": len(populations["heldout_truth"][0]) == 13587,
        },
    ]
    for arm in ARM_MODEL_IDS:
        quality_rows.extend(
            _relation_quality(arm, graph.user_item_operators[arm], prepared)
        )
        quality_rows.extend(
            shared._topk_quality(
                arm,
                views[arm]["users"],
                views[arm]["full_top50"],
                prepared,
            )
        )
    quality = pd.DataFrame(quality_rows)
    if not bool(quality["passed"].all()):
        raise RuntimeError(
            "diagnostic quality checks failed:\n"
            + quality.loc[~quality["passed"]].to_string(index=False)
        )

    source_attribution = source_payload["attribution_reading"]
    reading = diagnostic_route(
        coverage_summary,
        score_contrast,
        recommendation_overlap,
        source_attribution_supported=source_attribution[
            "clv_attribution_supported"
        ],
    )

    root = Path(cfg.diagnostic_out_dir)
    root.mkdir(parents=True, exist_ok=True)
    stem = f"m3_clv_candidate_item_failure_diagnostic_{cfg.source_result_id}"
    paths = {
        "graph_user_similarity_csv": root / f"{stem}_graph_users.csv",
        "graph_similarity_summary_csv": root / f"{stem}_graph_summary.csv",
        "clv_assignment_correlation_csv": root / f"{stem}_assignment.csv",
        "candidate_truth_users_csv": root / f"{stem}_candidate_truth_users.csv",
        "candidate_truth_summary_csv": root / f"{stem}_candidate_truth_summary.csv",
        "candidate_truth_comparison_csv": root / f"{stem}_candidate_truth_comparison.csv",
        "score_components_csv": root / f"{stem}_scores.csv",
        "score_truth_candidate_contrast_csv": root / f"{stem}_score_contrast.csv",
        "recommendation_overlap_csv": root / f"{stem}_recommendations.csv",
        "quality_checks_csv": root / f"{stem}_quality.csv",
        "json": root / f"{stem}.json",
    }
    frames = {
        "graph_user_similarity_csv": graph_users,
        "graph_similarity_summary_csv": graph_summary,
        "clv_assignment_correlation_csv": assignment,
        "candidate_truth_users_csv": coverage_users,
        "candidate_truth_summary_csv": coverage_summary,
        "candidate_truth_comparison_csv": coverage_comparison,
        "score_components_csv": score_components,
        "score_truth_candidate_contrast_csv": score_contrast,
        "recommendation_overlap_csv": recommendation_overlap,
        "quality_checks_csv": quality,
    }
    for key, frame in frames.items():
        fixed_train._atomic_csv(paths[key], frame)
    ledger = {
        "raw_sources": {
            "source_result": str(source_path),
            "checkpoints": {
                arm: str(path) for arm, path in checkpoints.items()
            },
            "input_hash": input_hash,
        },
        "population": {
            "graph": "all 2,500 train users",
            "candidate_truth_and_ranking": (
                "1,230 DAY 684--690 evaluation users; train user-item pairs masked"
            ),
            "heldout_truth_pairs": int(len(populations["heldout_truth"][0])),
        },
        "transformations": [
            "reconstruct train-only pooled, actual, and degree-matched shuffled candidate-item rows",
            "load immutable seed-42 checkpoints with identity and SHA checks",
            "measure held-out truth inclusion among each user's 100 candidate edges",
            "split each checkpoint score into base dot product and gamma-weighted candidate-message dot product",
            "compare graph rows, scores, and Top-K recommendations overall and by train-only CLV quintile",
        ],
        "formulas": {
            "candidate_truth_pair_coverage": "sum_u |B_u intersect truth_u| / sum_u |truth_u|",
            "macro_candidate_truth_recall": "mean_u |B_u intersect truth_u| / |truth_u|",
            "edge_jaccard": "|actual targets intersect shuffled targets| / |union|",
            "total_variation": "0.5 * sum_i |w_actual(u,i)-w_shuffle(u,i)|",
            "base_score": "z_u_M1 dot z_i_M1 within each jointly trained arm",
            "auxiliary_score": "gamma * m_u_candidate dot z_i_M1",
            "truth_score_contrast": "mean auxiliary score on truths minus mean on the three-arm Top-50 union",
        },
        "validation": quality.to_dict("records"),
        "limitations": [
            "descriptive post-hoc diagnostic on the already seen development interval",
            "base embeddings differ across jointly trained arms and are not a frozen M1 checkpoint",
            "no causal, significance, generalization, or model-selection claim",
        ],
    }
    payload = {
        "code_version": CODE_VERSION,
        "config": asdict(cfg),
        "preflight": preflight,
        "source_result": str(source_path),
        "source_attribution_reading": source_attribution,
        "checkpoint_records": records,
        "analysis_ledger": ledger,
        "graph_similarity_summary": graph_summary.to_dict("records"),
        "clv_assignment_correlation": assignment.to_dict("records"),
        "candidate_truth_summary": coverage_summary.to_dict("records"),
        "candidate_truth_comparison": coverage_comparison.to_dict("records"),
        "score_components": score_components.to_dict("records"),
        "score_truth_candidate_contrast": score_contrast.to_dict("records"),
        "recommendation_overlap": recommendation_overlap.to_dict("records"),
        "quality_checks": quality.to_dict("records"),
        "diagnostic_reading": reading,
        "result_paths": {key: str(path) for key, path in paths.items()},
    }
    fixed_train._atomic_json(paths["json"], payload)
    graph_summary.attrs["diagnostic_reading"] = reading
    graph_summary.attrs["result_paths"] = {
        key: str(path) for key, path in paths.items()
    }
    graph_summary.attrs["candidate_truth_summary"] = coverage_summary
    graph_summary.attrs["candidate_truth_comparison"] = coverage_comparison
    graph_summary.attrs["score_components"] = score_components
    graph_summary.attrs["score_truth_candidate_contrast"] = score_contrast
    graph_summary.attrs["recommendation_overlap"] = recommendation_overlap
    graph_summary.attrs["clv_assignment_correlation"] = assignment
    graph_summary.attrs["quality_checks"] = quality

    print("\n1. 실제 CLV와 shuffle 후보행 차이")
    print(graph_summary.to_string(index=False))
    print("\n2. 보조후보 100개 안의 개발정답 포함률")
    print(coverage_summary.to_string(index=False))
    print("\n3. 정답과 Top-50 경쟁상품의 점수 분해")
    print(score_contrast.to_string(index=False))
    print("\n4. 추천목록 변경")
    print(recommendation_overlap.to_string(index=False))
    print("\n5. 결과에 따른 다음 수정 위치")
    print(json.dumps(reading, ensure_ascii=False, indent=2))
    print("결과 파일:", graph_summary.attrs["result_paths"])
    return graph_summary


if __name__ == "__main__":
    run_m3_candidate_item_diagnostic()
